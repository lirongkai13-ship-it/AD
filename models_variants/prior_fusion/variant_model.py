"""Baseline + 自适应先验知识图融合"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block, TCNBlock


class AdaptivePriorGraph(nn.Module):
    """
    自适应先验-数据图融合：
    A_combined = σ(α) * A_prior + (1-σ(α)) * A_data
    其中 α 是可学习参数，σ 是 sigmoid 函数，确保权重在 [0,1]
    """

    def __init__(self, n_vars, prior_edge_index, prior_weights, data_threshold=0.3):
        super().__init__()
        # 可学习融合权重 α（初始化为 0.5 = 先验和数据各一半）
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 0 → sigmoid(0)=0.5
        self.threshold = data_threshold
        self.n_vars = n_vars

        # 注册先验边为 buffer（不参与梯度）
        self.register_buffer("prior_edge_index", prior_edge_index)
        self.register_buffer("prior_weights", prior_weights)

    def forward(self, x):
        """
        x: [B, W, N]
        返回: combined_edge_index [2, E]
        """
        b, w, n = x.shape

        # ── 数据驱动图：批量 Pearson 相关 ──
        x_c = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_c.transpose(1, 2), x_c) / (w - 1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
        std_prod = std.unsqueeze(1) * std.unsqueeze(2)
        data_corr = cov / (std_prod + 1e-8)
        data_corr = torch.nan_to_num(data_corr, nan=0.0, posinf=0.0, neginf=0.0)
        avg_corr = data_corr.mean(dim=0).abs()  # [N, N]

        # 阈值筛选动态边（排除自环）
        mask = (avg_corr >= self.threshold) & (~torch.eye(n, dtype=torch.bool, device=x.device))
        if mask.any():
            src, dst = torch.where(mask)
            dyn_edges = torch.stack([src, dst], dim=0)
        else:
            dyn_edges = torch.empty(2, 0, dtype=torch.long, device=x.device)

        # ── 融合：sigmoid(α) 控制先验权重 ──
        prior_w = torch.sigmoid(self.alpha)  # [0,1]

        # 合并两边集合（去重），然后通过注意力权重体现差异
        all_edges = torch.cat([self.prior_edge_index, dyn_edges], dim=1)  # [2, E_all]
        hashed = all_edges[0] * n + all_edges[1]
        unique_idx = torch.unique(hashed)
        combined = torch.stack([unique_idx // n, unique_idx % n], dim=0)

        return combined

    def get_fusion_weight(self):
        """返回当前先验权重，用于日志"""
        return torch.sigmoid(self.alpha).item()


class GATv2_PriorFusion_TCN_GRU(nn.Module):
    """Baseline + 自适应先验-数据图融合"""

    def __init__(self, num_variables, window_size, prior_edge_index, prior_weights,
                 hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()
        self.num_variables = num_variables
        self.window_size = window_size

        # ── 自适应先验图融合 ──
        self.graph_fusion = AdaptivePriorGraph(num_variables, prior_edge_index, prior_weights)

        # ── 以下与 baseline 一致 ──
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        tcn_in = hidden_dim
        self.tcn_layers = nn.ModuleList()
        for i in range(tcn_blocks):
            self.tcn_layers.append(
                TCNBlock(tcn_in, tcn_channels, kernel_size=3, dilation=2**i, dropout=dropout))
            tcn_in = tcn_channels

        self.gru = nn.GRU(tcn_channels, gru_hidden, num_layers=1, batch_first=True)
        self.pred_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_variables))
        self.recon_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gru_hidden, window_size * num_variables))

    def forward(self, x, edge_index):
        b, w, n = x.shape

        # ── 自适应先验图融合 ──
        combined_edge = self.graph_fusion(x)

        node_feat = x.transpose(1, 2)               # [B, N, W]
        h = self.gat(node_feat, combined_edge)      # 使用融合后的边

        z = h.transpose(1, 2)
        for tcn in self.tcn_layers:
            z = tcn(z)

        z = z.transpose(1, 2)
        _, h_last = self.gru(z)
        h_last = h_last[-1]

        pred = self.pred_head(h_last)
        recon = self.recon_head(h_last).view(b, w, n)
        return pred, recon
