"""Baseline + 逐边自适应融合: e_ij = σ(w1·C_ij + w2·P_ij)"""
import torch
import torch.nn as nn
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block, TCNBlock


class EdgeWiseFusion(nn.Module):
    """
    逐边自适应融合:
      e_ij = σ(w1 · C_ij + w2 · P_ij)
    C_ij: 动态 Pearson 相关系数 (每批计算)
    P_ij: 先验知识边权重 (Excel 构建, 无边则为0)
    w1, w2: 可训练标量参数
    """

    def __init__(self, n_vars, prior_edge_index, prior_weights, threshold=0.3):
        super().__init__()
        self.n_vars = n_vars
        self.threshold = threshold

        # ── 可训练权重 w1(数据), w2(先验) + 偏置 b ──
        # b=-2 → sigmoid(-2)≈0.12，只有 C 或 P 足够强才过 threshold=0.3
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.b  = nn.Parameter(torch.tensor(-1.0))

        # ── 构建先验矩阵 P [N, N] ──
        P = torch.zeros(n_vars, n_vars)
        for i in range(prior_edge_index.shape[1]):
            src, dst = prior_edge_index[0, i].item(), prior_edge_index[1, i].item()
            P[src, dst] = max(P[src, dst], prior_weights[i].item())  # 重复边取最大
        # 自环强置
        for i in range(n_vars):
            P[i, i] = 1.0
        self.register_buffer("P", P)  # 先验矩阵不参与梯度

    def forward(self, x):
        """
        x: [B, W, N]
        返回: combined_edge_index [2, E]
        """
        b, w, n = x.shape

        # ── 1. 每张图分别计算动态 Pearson C [B, N, N] ──
        x_c = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_c.transpose(1, 2), x_c) / (w - 1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
        C_batch = cov / (std.unsqueeze(1) * std.unsqueeze(2) + 1e-8)
        C_batch = torch.nan_to_num(C_batch, nan=0.0, posinf=0.0, neginf=0.0)
        C_batch = C_batch.abs()  # [B, N, N] 每张图的相关矩阵

        # ── 2. 每张图分别与先验融合: e = σ(w1·C + w2·P + b) ──
        e_batch = torch.sigmoid(self.w1 * C_batch + self.w2 * self.P + self.b)

        # ── 3. 批次平均后阈值选边 ──
        C_avg = C_batch.mean(dim=0)
        e = torch.sigmoid(self.w1 * C_avg + self.w2 * self.P + self.b)
        mask = e >= self.threshold
        mask[torch.arange(n, device=x.device), torch.arange(n, device=x.device)] = True
        src, dst = torch.where(mask)
        combined = torch.stack([src, dst], dim=0)  # [2, E]
        return combined

    def get_weights(self):
        """返回当前 w1, w2 的值，用于监控"""
        return self.w1.item(), self.w2.item()


class GATv2_PD_TCN_GRU(nn.Module):
    """Baseline + 逐边自适应先验-数据融合图"""

    def __init__(self, num_variables, window_size, prior_edge_index, prior_weights,
                 hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()
        self.graph_fusion = EdgeWiseFusion(num_variables, prior_edge_index, prior_weights)
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
        combined_edge = self.graph_fusion(x)
        node_feat = x.transpose(1, 2)
        h = self.gat(node_feat, combined_edge)
        z = h.transpose(1, 2)
        for tcn in self.tcn_layers:
            z = tcn(z)
        z = z.transpose(1, 2)
        _, h_last = self.gru(z)
        h_last = h_last[-1]
        pred = self.pred_head(h_last)
        recon = self.recon_head(h_last).view(b, w, n)
        return pred, recon
