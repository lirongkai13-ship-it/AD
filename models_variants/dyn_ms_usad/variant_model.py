"""动态 Pearson + Multi-Scale TCN + USAD 双解码器"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block


class DynamicPearsonGraph(nn.Module):
    """每批计算动态 Pearson 相关图，叠加在静态边上"""
    def __init__(self, n_vars, static_edge_index, threshold=0.3):
        super().__init__()
        self.n_vars = n_vars
        self.threshold = threshold
        self.register_buffer("static_ei", static_edge_index)

    def forward(self, x):
        b, w, n = x.shape
        x_c = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_c.transpose(1,2), x_c) / (w-1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
        C = cov / (std.unsqueeze(1)*std.unsqueeze(2) + 1e-8)
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        avg = C.abs().mean(dim=0)
        mask = (avg >= self.threshold) & (~torch.eye(n, dtype=torch.bool, device=x.device))
        if mask.any():
            src, dst = torch.where(mask)
            dyn = torch.stack([src, dst], dim=0)
            comb = torch.cat([self.static_ei.to(x.device), dyn], dim=1)
        else:
            comb = self.static_ei.to(x.device)
        h = comb[0] * n + comb[1]
        uniq = torch.unique(h)
        return torch.stack([uniq // n, uniq % n], dim=0)


class MultiScaleTCN(nn.Module):
    """多尺度 TCN：kernel=3,5,7 三个并行分支，输出拼接 + 线性融合"""
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        hidden = out_channels // 3
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 3, padding=1), nn.ReLU(), nn.Dropout(dropout))
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 5, padding=2), nn.ReLU(), nn.Dropout(dropout))
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 7, padding=3), nn.ReLU(), nn.Dropout(dropout))
        self.fuse = nn.Conv1d(hidden*3, out_channels, 1)
        self.res = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: [B, C, L]
        h3 = self.branch3(x)
        h5 = self.branch5(x)
        h7 = self.branch7(x)
        h = torch.cat([h3, h5, h7], dim=1)
        return self.fuse(h) + self.res(x)


class DynMS_USAD(nn.Module):
    """Dynamic Pearson + Multi-Scale TCN + USAD"""

    def __init__(self, num_variables, window_size, static_edge_index,
                 hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()
        dec_feat = window_size * num_variables

        self.dyn_graph = DynamicPearsonGraph(num_variables, static_edge_index)
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)

        # Multi-Scale TCN 替代标准 TCN
        self.ms_tcn = MultiScaleTCN(hidden_dim, tcn_channels, dropout=dropout)

        self.gru = nn.GRU(tcn_channels, gru_hidden, num_layers=1, batch_first=True)

        self.decoder1 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden*2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gru_hidden*2, dec_feat))
        self.decoder2 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden*2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gru_hidden*2, dec_feat))

    def forward(self, x, edge_index):
        b, w, n = x.shape
        edges = self.dyn_graph(x)
        h = self.gat(x.transpose(1,2), edges)
        z = self.ms_tcn(h.transpose(1,2))     # [B, tcn_ch, N]
        z = z.transpose(1,2)                  # [B, N, tcn_ch]
        _, hl = self.gru(z); hl = hl[-1]
        r1 = self.decoder1(hl).view(b, w, n)
        r2 = self.decoder2(hl).view(b, w, n)
        with torch.no_grad():
            h2 = self.gat(r1.transpose(1,2), edges)
            z2 = self.ms_tcn(h2.transpose(1,2))
            z2 = z2.transpose(1,2)
            _, hl2 = self.gru(z2); hl2 = hl2[-1].detach()
        r12 = self.decoder2(hl2).view(b, w, n)
        return r1, r2, r12

    def forward_eval(self, x, edge_index):
        b, w, n = x.shape
        edges = self.dyn_graph(x)
        h = self.gat(x.transpose(1,2), edges)
        z = self.ms_tcn(h.transpose(1,2))
        z = z.transpose(1,2)
        _, hl = self.gru(z); hl = hl[-1]
        return self.decoder1(hl).view(b, w, n)
