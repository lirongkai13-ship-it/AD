"""
方案E: 并行时空 USAD + 多尺度 TCN + GRU 时间分支 (dilation=1,2,4 + GRU)

Shape 流程:
  Input:            X [B, 60, 51]
  Spatial Branch:   X → dyn_prior_graph → GATv2 → H_space [B, 51, 32]
  Temporal Branch:  X → per-var MultiScale TCN(d1,2,4) → GRU → H_time [B, 51, 32]
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
# 1. Dynamic Pearson + 先验图 boost 融合 (不变)
# ═══════════════════════════════════════════════════════════════
class DynamicPearsonPriorGraph(nn.Module):
    def __init__(self, n_vars, prior_edge_index, prior_weights, boost=0.3, threshold=0.3):
        super().__init__()
        self.n_vars = n_vars; self.boost = boost; self.threshold = threshold
        A_prior = torch.zeros(n_vars, n_vars)
        for i in range(prior_edge_index.shape[1]):
            src, dst = prior_edge_index[0, i].item(), prior_edge_index[1, i].item()
            w = prior_weights[i].item()
            if src < n_vars and dst < n_vars:
                A_prior[src, dst] = max(A_prior[src, dst], w)
                A_prior[dst, src] = max(A_prior[dst, src], w)
        self.register_buffer("A_prior", A_prior)
        self.register_buffer("prior_mask", A_prior > 0)

    def forward(self, x):
        b, w, n = x.shape
        x_c = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_c.transpose(1,2), x_c) / (w-1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
        C = cov / (std.unsqueeze(1)*std.unsqueeze(2) + 1e-8)
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        A_dyn = C.abs().mean(dim=0)
        A_fused = A_dyn.clone()
        overlap = self.prior_mask.to(x.device) & (A_dyn >= self.threshold)
        only_prior = self.prior_mask.to(x.device) & (A_dyn < self.threshold)
        boost_val = torch.tensor(self.boost, dtype=A_fused.dtype, device=A_fused.device)
        A_fused[overlap] = A_fused[overlap] + boost_val
        A_fused[only_prior] = self.A_prior.to(device=x.device, dtype=A_fused.dtype)[only_prior]
        diag_mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        edge_mask = (A_fused.abs() >= self.threshold) & diag_mask
        if edge_mask.any():
            src, dst = torch.where(edge_mask)
            edges = torch.stack([src, dst], dim=0)
        else:
            edges = torch.zeros(2, 0, dtype=torch.long, device=x.device)
        return edges


# ═══════════════════════════════════════════════════════════════
# 2. TCN Block (带残差, 保持时间长度)
# ═══════════════════════════════════════════════════════════════
class TCNResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_channels, hidden_channels, 1) if in_channels != hidden_channels else nn.Identity()

    def forward(self, x):
        residual = self.res(x)
        h = F.relu(self.conv1(x))
        h = self.dropout(h)
        h = F.relu(self.conv2(h))
        return h + residual


# ═══════════════════════════════════════════════════════════════
# 3. 方案E: 多尺度 TCN + GRU 时间编码器
# ═══════════════════════════════════════════════════════════════
class MultiScaleTCNGRUTemporalEncoder(nn.Module):
    """3个 TCN 分支 (dil=1,2,4) → channel concat → GRU → h_last → [B*N, d]"""
    def __init__(self, window_size=60, out_dim=32, branch_channels=8,
                 gru_hidden=32, dropout=0.1):
        super().__init__()
        self.branch1 = TCNResBlock(1, branch_channels, kernel_size=3, dilation=1, dropout=dropout)
        self.branch2 = TCNResBlock(1, branch_channels, kernel_size=3, dilation=2, dropout=dropout)
        self.branch3 = TCNResBlock(1, branch_channels, kernel_size=3, dilation=4, dropout=dropout)
        concat_dim = branch_channels * 3
        self.gru = nn.GRU(concat_dim, gru_hidden, num_layers=1, batch_first=True)
        self.proj = nn.Linear(gru_hidden, out_dim)

    def forward(self, x):
        """x: [B*N, 1, W] → [B*N, out_dim]"""
        h1 = F.relu(self.branch1(x))        # [B*N, bc, W]
        h2 = F.relu(self.branch2(x))
        h3 = F.relu(self.branch3(x))
        h = torch.cat([h1, h2, h3], dim=1)   # [B*N, 3*bc, W]
        h = h.transpose(1, 2)                # [B*N, W, 3*bc]
        _, h_last = self.gru(h)              # [1, B*N, gru_h]
        return self.proj(h_last[-1])          # [B*N, out_dim]


# ═══════════════════════════════════════════════════════════════
# 4. 先验节点嵌入 (不变)
# ═══════════════════════════════════════════════════════════════
class PriorNodeEmbedding(nn.Module):
    def __init__(self, n_vars, prior_edge_index, prior_weights, hidden_dim):
        super().__init__()
        P = torch.zeros(n_vars, n_vars)
        for i in range(prior_edge_index.shape[1]):
            src, dst = prior_edge_index[0, i].item(), prior_edge_index[1, i].item()
            P[src, dst] = max(P[src, dst], prior_weights[i].item())
        deg = P.sum(dim=1, keepdim=True).clamp(min=1)
        P_norm = P / deg
        self.register_buffer("P_norm", P_norm)
        self.node_embed = nn.Parameter(torch.randn(n_vars, hidden_dim) * 0.1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self):
        return self.proj(torch.matmul(self.P_norm, self.node_embed))


# ═══════════════════════════════════════════════════════════════
# 5. Node-Level Fusion (不变)
# ═══════════════════════════════════════════════════════════════
class NodeLevelFusion(nn.Module):
    def __init__(self, d_space=32, d_time=32, d_fuse=32, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_space + d_time, d_fuse),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_fuse, d_fuse))

    def forward(self, h_space, h_time):
        return self.mlp(torch.cat([h_space, h_time], dim=-1))


# ═══════════════════════════════════════════════════════════════
# 6. ParallelSpatioTemporalEncoder (方案E版)
# ═══════════════════════════════════════════════════════════════
class ParallelSpatioTemporalEncoder(nn.Module):
    def __init__(self, num_variables=51, window_size=60,
                 prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, dropout=0.2,
                 latent_dim=64, use_flatten=True, boost=0.3):
        super().__init__()
        self.num_variables = num_variables; self.window_size = window_size
        self.use_flatten = use_flatten

        self.dyn_graph = DynamicPearsonPriorGraph(num_variables, prior_edge_index, prior_weights, boost=boost)
        self.prior_embed = PriorNodeEmbedding(num_variables, prior_edge_index, prior_weights, hidden_dim)
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())

        # ── 方案E: 多尺度 TCN + GRU ──
        self.temporal_enc = MultiScaleTCNGRUTemporalEncoder(
            window_size=window_size, out_dim=hidden_dim,
            branch_channels=hidden_dim//4, gru_hidden=hidden_dim, dropout=dropout)

        self.fusion = NodeLevelFusion(d_space=hidden_dim, d_time=hidden_dim,
                                      d_fuse=hidden_dim, dropout=dropout)

        if use_flatten:
            flat_dim = num_variables * hidden_dim
            mid_dim = flat_dim // 16
            self.latent_proj = nn.Sequential(
                nn.Linear(flat_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, latent_dim))
        else:
            self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        b, w, n = x.shape
        edges = self.dyn_graph(x)
        h_gat = self.gat(x.permute(0,2,1), edges)
        h_prior = self.prior_embed().unsqueeze(0).expand(b, -1, -1)
        g = self.gate(torch.cat([h_gat, h_prior], dim=-1))
        h_space = g * h_gat + (1-g) * h_prior
        x_var = x.permute(0,2,1).reshape(b*n, 1, w)
        h_time = self.temporal_enc(x_var).reshape(b, n, -1)
        h_fuse = self.fusion(h_space, h_time)
        if self.use_flatten:
            z = h_fuse.reshape(b, n * h_fuse.shape[-1])
        else:
            z = h_fuse.mean(dim=1)
        z = self.latent_proj(z)
        return z, edges


# ═══════════════════════════════════════════════════════════════
# 7. ParallelE_USAD (方案E完整模型)
# ═══════════════════════════════════════════════════════════════
class ParallelE_USAD(nn.Module):
    def __init__(self, num_variables, window_size, static_edge_index,
                 prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2,
                 latent_dim=64, use_flatten=True):
        super().__init__()
        dec_feat = window_size * num_variables
        self.encoder = ParallelSpatioTemporalEncoder(
            num_variables=num_variables, window_size=window_size,
            prior_edge_index=prior_edge_index, prior_weights=prior_weights,
            hidden_dim=hidden_dim, gat_heads=gat_heads, dropout=dropout,
            latent_dim=latent_dim, use_flatten=use_flatten)
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
            z2, _ = self.encoder(r1); z2 = z2.detach()
        r12 = self.decoder2(z2).view(b, w, n)
        return r1, r2, r12

    def forward_eval(self, x, edge_index):
        b, w, n = x.shape
        z, _ = self.encoder(x)
        return self.decoder1(z).view(b, w, n)
