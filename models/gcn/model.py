"""GCN + TCN + GRU 异常检测（替换 GATv2 为图卷积）"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualGCNLayer(nn.Module):
    """简化 GCN 层：聚合邻居特征，取均值"""

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        b, n, _ = x.shape
        src, dst = edge_index[0], edge_index[1]

        # 投影
        h = self.proj(x)  # [B, N, out_dim]

        # 聚合邻居
        out = torch.zeros(b, n, h.shape[-1], device=x.device)
        # 统计每个节点入度
        degree = torch.zeros(n, device=x.device)
        for e in range(src.numel()):
            out[:, dst[e], :] += h[:, src[e], :]
            degree[dst[e]] += 1
        # 均值（避免除零）
        degree = degree.clamp(min=1)
        out = out / degree.view(1, n, 1)
        return F.relu(self.dropout(out))


class GCNBlock(nn.Module):
    """2 层 GCN + 残差"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.gcn1 = ManualGCNLayer(in_dim, hidden_dim, dropout)
        self.gcn2 = ManualGCNLayer(hidden_dim, out_dim, dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        residual = self.res(x)
        h = self.gcn1(x, edge_index)
        h = self.gcn2(h, edge_index)
        return self.norm(h + residual)


# Reuse TCN + GRU from existing model
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import TCNBlock


class GCN_TCN_GRU(nn.Module):
    """GCN + TCN + GRU 双分支检测器"""

    def __init__(self, num_variables=51, window_size=60, hidden_dim=32,
                 gru_hidden=32, tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()
        self.gcn = GCNBlock(window_size, hidden_dim, hidden_dim, dropout=dropout)
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
        h = self.gcn(x.transpose(1, 2), edge_index)  # [B, N, hidden]
        z = h.transpose(1, 2)
        for tcn in self.tcn_layers: z = tcn(z)
        z = z.transpose(1, 2)
        _, hl = self.gru(z); hl = hl[-1]
        pred = self.pred_head(hl)
        recon = self.recon_head(hl).view(b, w, n)
        return pred, recon  # 兼容 BaseTrainer._collect_errors
