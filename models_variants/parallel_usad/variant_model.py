"""
节点级时空并行融合 + USAD 双解码器 (优化版)

Shape 流程:
  Input:            X [B, 60, 51]
  Spatial Branch:   X → GATv2 → H_space [B, 51, 32]
  Temporal Branch:  X → per-var Conv1d(轻量) → H_time [B, 51, 32]
  Node Fusion:      concat [B,51,64] → MLP → H_fuse [B, 51, 32]
  Latent (flatten): [B, 1632] → Linear → z [B, 64]
  USAD Decoder:     z → r1, r2, r12 [B, 60, 51]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block


# ═══════════════════════════════════════════════════════════════
# 1. Dynamic Pearson Graph (复用)
# ═══════════════════════════════════════════════════════════════
class DynamicPearsonGraph(nn.Module):
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


# ═══════════════════════════════════════════════════════════════
# 2. 轻量 TemporalVariableEncoder (reshape 方案)
#    [B*N, 1, W] → Conv1d(1→16→32) → pool → [B*N, 32]
# ═══════════════════════════════════════════════════════════════
class TemporalVariableEncoder(nn.Module):
    """轻量逐变量时间编码: Conv1d(1→16→32) + pool"""
    def __init__(self, window_size=60, out_dim=32, hidden_channels=16, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, out_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """x: [B*N, 1, W] → [B*N, out_dim]"""
        h = F.relu(self.conv1(x))      # [B*N, 16, W]
        h = self.dropout(h)
        h = F.relu(self.conv2(h))      # [B*N, 32, W]
        h = self.pool(h)               # [B*N, 32, 1]
        return h.squeeze(-1)           # [B*N, 32]


# ═══════════════════════════════════════════════════════════════
# 3. GroupedConv TemporalEncoder (可选, 更快)
#    [B, N, W] → grouped Conv1d(groups=N) → [B, N, d]
#    无需 reshape, 一次前向处理所有变量
# ═══════════════════════════════════════════════════════════════
class GroupedTemporalEncoder(nn.Module):
    """Grouped Conv1D: groups=N, 每个变量独立时间卷积"""
    def __init__(self, num_variables=51, window_size=60, out_dim=32, hidden=16, dropout=0.1):
        super().__init__()
        self.N = num_variables
        self.out_dim = out_dim
        self.conv1 = nn.Conv1d(num_variables, num_variables * hidden, kernel_size=3,
                               padding=1, groups=num_variables)
        self.conv2 = nn.Conv1d(num_variables * hidden, num_variables * out_dim, kernel_size=3,
                               padding=1, groups=num_variables)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """x: [B, N, W] → [B, N, out_dim]"""
        b, n, w = x.shape
        h = F.relu(self.conv1(x))       # [B, N*16, W]
        h = self.dropout(h)
        h = F.relu(self.conv2(h))       # [B, N*32, W]
        h = self.pool(h)                # [B, N*32, 1]
        h = h.view(b, n, self.out_dim)  # [B, N, 32]
        return h


# ═══════════════════════════════════════════════════════════════
# 4. Node-Level Fusion
# ═══════════════════════════════════════════════════════════════
class NodeLevelFusion(nn.Module):
    """节点级 concat + MLP: [B, N, 2d] → [B, N, d_fuse]"""
    def __init__(self, d_space=32, d_time=32, d_fuse=32, dropout=0.2):
        super().__init__()
        in_dim = d_space + d_time
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_fuse),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_fuse, d_fuse),
        )

    def forward(self, h_space, h_time):
        return self.mlp(torch.cat([h_space, h_time], dim=-1))


# ═══════════════════════════════════════════════════════════════
# 5. ParallelSpatioTemporalEncoder
# ═══════════════════════════════════════════════════════════════
class ParallelSpatioTemporalEncoder(nn.Module):
    def __init__(self, num_variables=51, window_size=60, static_edge_index=None,
                 hidden_dim=32, gat_heads=2, dropout=0.2, latent_dim=64, use_flatten=True,
                 temporal_mode="lightweight"):
        super().__init__()
        self.num_variables = num_variables
        self.window_size = window_size
        self.use_flatten = use_flatten
        self.temporal_mode = temporal_mode

        # ── 动态图 ──
        self.dyn_graph = DynamicPearsonGraph(num_variables, static_edge_index)

        # ── 空间: GATv2 → [B, N, hidden_dim] ──
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim,
                              heads=gat_heads, dropout=dropout)

        # ── 时间编码 ──
        if temporal_mode == "grouped_conv":
            self.temporal_enc = GroupedTemporalEncoder(
                num_variables=num_variables, window_size=window_size,
                out_dim=hidden_dim, hidden=hidden_dim//2, dropout=dropout)
        else:
            self.temporal_enc = TemporalVariableEncoder(
                window_size=window_size, out_dim=hidden_dim,
                hidden_channels=hidden_dim//2, dropout=dropout)

        # ── 融合: [B,N,2*hidden] → [B,N,hidden] ──
        self.fusion = NodeLevelFusion(d_space=hidden_dim, d_time=hidden_dim,
                                      d_fuse=hidden_dim, dropout=dropout)

        # ── Latent 投影 — 参数减半 ──
        if use_flatten:
            flat_dim = num_variables * hidden_dim  # 51*32 = 1632
            mid_dim = flat_dim // 16                # 1632/16=102
            self.latent_proj = nn.Sequential(
                nn.Linear(flat_dim, mid_dim),       # 1632→102 (167K)
                nn.ReLU(),
                nn.Linear(mid_dim, latent_dim),     # 102→64  (6.5K)
            )
        else:
            self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        b, w, n = x.shape

        # ── 动态图 ──
        edges = self.dyn_graph(x)

        # ── 空间: [B,W,N] → [B,N,W] → GATv2 → [B,N,hidden] ──
        h_space = self.gat(x.permute(0, 2, 1), edges)

        # ── 时间 ──
        if self.temporal_mode == "grouped_conv":
            # [B, N, W] → grouped conv → [B, N, hidden]
            h_time = self.temporal_enc(x.permute(0, 2, 1))
        else:
            # [B,W,N] → [B,N,W] → [B*N,1,W] → per-var → [B,N,hidden]
            x_var = x.permute(0, 2, 1).reshape(b * n, 1, w)
            h_time = self.temporal_enc(x_var).reshape(b, n, -1)

        # ── 融合: [B,N,2*hidden] → [B,N,hidden] ──
        h_fuse = self.fusion(h_space, h_time)

        # ── Latent ──
        if self.use_flatten:
            z = h_fuse.reshape(b, n * h_fuse.shape[-1])
        else:
            z = h_fuse.mean(dim=1)
        z = self.latent_proj(z)

        return z, edges


# ═══════════════════════════════════════════════════════════════
# 6. Parallel_USAD (完整模型)
# ═══════════════════════════════════════════════════════════════
class Parallel_USAD(nn.Module):
    def __init__(self, num_variables, window_size, static_edge_index,
                 hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2,
                 latent_dim=64, use_flatten=True, temporal_mode="lightweight"):
        super().__init__()
        dec_feat = window_size * num_variables  # 3060

        self.encoder = ParallelSpatioTemporalEncoder(
            num_variables=num_variables, window_size=window_size,
            static_edge_index=static_edge_index,
            hidden_dim=hidden_dim, gat_heads=gat_heads,
            dropout=dropout, latent_dim=latent_dim,
            use_flatten=use_flatten, temporal_mode=temporal_mode,
        )

        # Decoder: 缩小中间层减少参数
        dec_hidden = latent_dim * 2
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dec_hidden, dec_feat))
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, dec_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dec_hidden, dec_feat))

    def forward(self, x, edge_index):
        b, w, n = x.shape
        z, edges = self.encoder(x)
        r1 = self.decoder1(z).view(b, w, n)
        r2 = self.decoder2(z).view(b, w, n)
        with torch.no_grad():
            z2, _ = self.encoder(r1)
            z2 = z2.detach()
        r12 = self.decoder2(z2).view(b, w, n)
        return r1, r2, r12

    def forward_eval(self, x, edge_index):
        b, w, n = x.shape
        z, _ = self.encoder(x)
        return self.decoder1(z).view(b, w, n)
