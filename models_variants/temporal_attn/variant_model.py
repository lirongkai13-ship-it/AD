"""Baseline + 时间注意力模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block, TCNBlock


class TemporalAttention(nn.Module):
    """沿时间维 (W) 的多头自注意力，增强窗口内时序表示"""

    def __init__(self, n_vars, window, n_heads=None, dropout=0.1):
        super().__init__()
        # 自动选能被 n_vars 整除的头数: 优先 4 → 3 → 1
        if n_heads is None:
            for h in [4, 3, 1]:
                if n_vars % h == 0:
                    n_heads = h; break
        self.n_heads = n_heads
        self.d_k = n_vars // n_heads
        self.w_q = nn.Linear(n_vars, n_vars)
        self.w_k = nn.Linear(n_vars, n_vars)
        self.w_v = nn.Linear(n_vars, n_vars)
        self.w_o = nn.Linear(n_vars, n_vars)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_vars)

    def forward(self, x):
        # x: [B, W, N] → attention along W
        b, w, n = x.shape
        q = self.w_q(x).view(b, w, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, W, D]
        k = self.w_k(x).view(b, w, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, w, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5), dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, w, n)
        return self.norm(x + self.w_o(out))  # 残差连接


class GATv2_TA_TCN_GRU(nn.Module):
    """Baseline + 时间注意力：在 GAT 之前沿 W 维加 Multi-Head Attention"""

    def __init__(self, num_variables, window_size, hidden_dim=32, gat_heads=2,
                 gru_hidden=32, tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()
        self.num_variables = num_variables
        self.window_size = window_size

        # ── 新增：时间注意力 (自动选头数，如 51 → 3 heads) ──
        self.time_attn = TemporalAttention(num_variables, window_size, dropout=dropout)

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

        # ── 时间注意力 (沿 W 维) ──
        x = self.time_attn(x)                       # [B, W, N]

        node_feat = x.transpose(1, 2)               # [B, N, W]
        h = self.gat(node_feat, edge_index)         # [B, N, hidden]

        z = h.transpose(1, 2)                       # [B, hidden, N]
        for tcn in self.tcn_layers:
            z = tcn(z)

        z = z.transpose(1, 2)                       # [B, N, tcn_ch]
        _, h_last = self.gru(z)
        h_last = h_last[-1]

        pred = self.pred_head(h_last)
        recon = self.recon_head(h_last).view(b, w, n)
        return pred, recon
