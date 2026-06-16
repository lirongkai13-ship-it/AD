"""USAD 双解码器 + GATv2+TCN+GRU 主干"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block, TCNBlock


class USAD_DualDecoder(nn.Module):
    """
    USAD 风格：共享 GAT+TCN+GRU 编码器 + 双解码器 + 二次重构
    """

    def __init__(self, num_variables, window_size, hidden_dim=32, gat_heads=2,
                 gru_hidden=32, tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()
        self.num_variables = num_variables
        self.window_size = window_size

        # ── 共享编码器: GAT + TCN + GRU ──
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        tcn_in = hidden_dim
        self.tcn_layers = nn.ModuleList()
        for i in range(tcn_blocks):
            self.tcn_layers.append(
                TCNBlock(tcn_in, tcn_channels, kernel_size=3, dilation=2**i, dropout=dropout))
            tcn_in = tcn_channels
        self.gru = nn.GRU(tcn_channels, gru_hidden, num_layers=1, batch_first=True)

        # ── 双解码器（USAD 风格，2层与基线pred/recon_head一致） ──
        dec_feat = window_size * num_variables
        self.decoder1 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, dec_feat),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, dec_feat),
        )

    def forward(self, x, edge_index):
        b, w, n = x.shape

        # 编码
        node_feat = x.transpose(1, 2)
        h = self.gat(node_feat, edge_index)
        z = h.transpose(1, 2)
        for tcn in self.tcn_layers:
            z = tcn(z)
        z = z.transpose(1, 2)
        _, h_last = self.gru(z)
        h_last = h_last[-1]  # [B, gru_hidden]

        # 双解码
        r1 = self.decoder1(h_last).view(b, w, n)  # Decoder 1 重构
        r2 = self.decoder2(h_last).view(b, w, n)  # Decoder 2 重构

        # 二次重构: r1 → 编码器 → Decoder2
        with torch.no_grad():
            node_feat2 = r1.transpose(1, 2)
            h2 = self.gat(node_feat2, edge_index)
            z2 = h2.transpose(1, 2)
            for tcn in self.tcn_layers:
                z2 = tcn(z2)
            z2 = z2.transpose(1, 2)
            _, h_last2 = self.gru(z2)
            h_last2 = h_last2[-1].detach()
        r12 = self.decoder2(h_last2).view(b, w, n)

        return r1, r2, r12

    def forward_eval(self, x, edge_index):
        """评估时只返回 Decoder1 的输出"""
        b, w, n = x.shape
        node_feat = x.transpose(1, 2)
        h = self.gat(node_feat, edge_index)
        z = h.transpose(1, 2)
        for tcn in self.tcn_layers:
            z = tcn(z)
        z = z.transpose(1, 2)
        _, h_last = self.gru(z)
        return self.decoder1(h_last[-1]).view(b, w, n)
