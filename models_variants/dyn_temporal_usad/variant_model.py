"""动态 Pearson + Temporal Attention + USAD 双解码器"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block, TCNBlock


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


class TemporalAttention(nn.Module):
    """沿时间维 (W) 的多头自注意力: 每个时间步注意其他时间步"""
    def __init__(self, n_vars, window, n_heads=None, dropout=0.1):
        super().__init__()
        # attend over time: [B, W, N] where W=60 steps, each with N=51 features
        if n_heads is None:
            n_heads = 3  # 51 % 3 == 0
        self.attn = nn.MultiheadAttention(n_vars, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        # x: [B, N, W] -> [B, W, N] for temporal attention
        x_t = x.transpose(1, 2)   # [B, W, N]
        out, _ = self.attn(x_t, x_t, x_t)
        return out.transpose(1, 2)  # back to [B, N, W]


class DynTA_USAD(nn.Module):
    """Dynamic Pearson + Temporal Attention + USAD"""

    def __init__(self, num_variables, window_size, static_edge_index,
                 hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()
        dec_feat = window_size * num_variables

        self.dyn_graph = DynamicPearsonGraph(num_variables, static_edge_index)
        self.time_attn = TemporalAttention(num_variables, window_size, dropout=dropout)

        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        tcn_in = hidden_dim
        self.tcn_layers = nn.ModuleList()
        for i in range(tcn_blocks):
            self.tcn_layers.append(TCNBlock(tcn_in, tcn_channels, kernel_size=3, dilation=2**i, dropout=dropout))
            tcn_in = tcn_channels
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
        # 时间注意力: 在 GAT 之前增强时序特征
        h = self.time_attn(x.transpose(1,2))  # [B,N,W]
        h = self.gat(h, edges)                 # [B,N,H]
        z = h.transpose(1,2)
        for tcn in self.tcn_layers: z = tcn(z)
        z = z.transpose(1,2)
        _, hl = self.gru(z); hl = hl[-1]
        r1 = self.decoder1(hl).view(b, w, n)
        r2 = self.decoder2(hl).view(b, w, n)
        with torch.no_grad():
            h2 = self.time_attn(r1.transpose(1,2))
            h2 = self.gat(h2, edges)
            z2 = h2.transpose(1,2)
            for tcn in self.tcn_layers: z2 = tcn(z2)
            z2 = z2.transpose(1,2)
            _, hl2 = self.gru(z2); hl2 = hl2[-1].detach()
        r12 = self.decoder2(hl2).view(b, w, n)
        return r1, r2, r12

    def forward_eval(self, x, edge_index):
        b, w, n = x.shape
        edges = self.dyn_graph(x)
        h = self.time_attn(x.transpose(1,2))
        h = self.gat(h, edges)
        z = h.transpose(1,2)
        for tcn in self.tcn_layers: z = tcn(z)
        z = z.transpose(1,2)
        _, hl = self.gru(z); hl = hl[-1]
        return self.decoder1(hl).view(b, w, n)
