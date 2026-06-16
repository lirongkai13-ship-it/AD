"""Baseline + 动态 Pearson 图模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block, TCNBlock


class DynamicPearsonGraph(nn.Module):
    """
    动态 Pearson 相关图：每批根据当前数据计算变量间相关性，
    与静态边合并，实现自适应图结构。
    """

    def __init__(self, static_edge_index, threshold=0.3, dynamic_ratio=0.5):
        """
        static_edge_index: [2, E_static] 静态 Pearson 边
        dynamic_ratio: 动态边在最终图中的权重（0.5 = 与静态等权）
        """
        super().__init__()
        self.register_buffer("static_edge_index", static_edge_index)
        self.threshold = threshold
        self.dynamic_ratio = dynamic_ratio

    def forward(self, x):
        """
        x: [B, W, N] 当前批次输入
        返回: combined_edge_index [2, E_combined]
        使用向量化批量 Pearson 计算，速度远快于逐样本 for-loop
        """
        b, w, n = x.shape

        # ─── 向量化批量 Pearson 相关 ───
        x_c = x - x.mean(dim=1, keepdim=True)           # [B, W, N] 中心化
        cov = torch.bmm(x_c.transpose(1, 2), x_c) / (w - 1)  # [B, N, N] 协方差矩阵
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)  # [B, N]
        # 批量相关矩阵 = cov / (std_i * std_j)
        std_prod = std.unsqueeze(1) * std.unsqueeze(2)  # [B, N, N]
        corr = cov / (std_prod + 1e-8)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

        # ─── 取批次平均相关，阈值筛选 ───
        avg_corr = corr.mean(dim=0).abs()                # [N, N] 平均绝对值相关
        mask = (avg_corr >= self.threshold) & (~torch.eye(n, dtype=torch.bool, device=x.device))

        if not mask.any():
            return self.static_edge_index

        src, dst = torch.where(mask)
        dynamic_edges = torch.stack([src, dst], dim=0)   # [2, E_dyn]

        # ─── 合并静态边 ───
        static = self.static_edge_index
        all_edges = torch.cat([static, dynamic_edges], dim=1)
        hashed = all_edges[0] * n + all_edges[1]
        unique_idx = torch.unique(hashed)
        combined = torch.stack([unique_idx // n, unique_idx % n], dim=0)

        # ─── 确保自环 ───
        self_loops = torch.stack([torch.arange(n, device=x.device),
                                  torch.arange(n, device=x.device)], dim=0)
        all_with_self = torch.cat([combined, self_loops], dim=1)
        hashed_final = all_with_self[0] * n + all_with_self[1]
        unique_final = torch.unique(hashed_final)
        return torch.stack([unique_final // n, unique_final % n], dim=0)


class GATv2_DG_TCN_GRU(nn.Module):
    """Baseline + 动态 Pearson 图：每批自适应构建新边"""

    def __init__(self, num_variables, window_size, static_edge_index,
                 hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2,
                 dynamic_threshold=0.3):
        super().__init__()
        self.num_variables = num_variables
        self.window_size = window_size

        # ── 新增：动态图 ──
        self.dynamic_graph = DynamicPearsonGraph(static_edge_index,
                                                  threshold=dynamic_threshold)

        # ── 以下与 baseline 一致 ──
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim,
                              heads=gat_heads, dropout=dropout)
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

        # ── 动态图：合并静态 + 动态边 ──
        dyn_edges = self.dynamic_graph(x)  # 动态计算当前批次的相关图
        # 确保 edge_index 在正确设备上
        if dyn_edges.device != x.device:
            dyn_edges = dyn_edges.to(x.device)
        combined_edge = dyn_edges

        node_feat = x.transpose(1, 2)               # [B, N, W]
        h = self.gat(node_feat, combined_edge)      # 使用合并后的边

        z = h.transpose(1, 2)
        for tcn in self.tcn_layers:
            z = tcn(z)

        z = z.transpose(1, 2)
        _, h_last = self.gru(z)
        h_last = h_last[-1]

        pred = self.pred_head(h_last)
        recon = self.recon_head(h_last).view(b, w, n)
        return pred, recon
