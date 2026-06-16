"""
节点级时空并行 + 多尺度时间分支 + USAD

与 parallel_usad_prior 唯一区别:
  时间分支从「单尺度 Conv1d(k=3)」改为「多尺度 Conv1d(k=3,5,7)」

Shape 流程:
  Input:            X [B, 60, 51]
  Spatial Branch:   X → dyn_prior_graph → GATv2 → H_space [B, 51, 32]
  Temporal Branch:  X → per-var MultiScale Conv1d(k=3,5,7) → H_time [B, 51, 32]
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
# 1. Dynamic Pearson + 先验图 boost融合 (不变)
# ═══════════════════════════════════════════════════════════════
class DynamicPearsonPriorGraph(nn.Module):
    def __init__(self, n_vars, prior_edge_index, prior_weights, boost=0.3, threshold=0.3):
        super().__init__()
        self.n_vars = n_vars
        self.boost = boost
        self.threshold = threshold
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
# 2. 多尺度时间编码器 [NEW] — 替代原 TemporalVariableEncoder
# ═══════════════════════════════════════════════════════════════
class MultiScaleTemporalEncoder(nn.Module):
    """轻量多尺度逐变量时间编码: 3个并行Conv1d分支(k=3,5,7) → concat → Linear"""
    def __init__(self, window_size=60, out_dim=32, hidden_per_scale=8, dropout=0.2):
        super().__init__()
        # 3个独立分支
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, hidden_per_scale, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.branch5 = nn.Sequential(
            nn.Conv1d(1, hidden_per_scale, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.branch7 = nn.Sequential(
            nn.Conv1d(1, hidden_per_scale, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        # 融合层: 3 * hidden_per_scale → out_dim
        concat_dim = hidden_per_scale * 3  # 24
        self.fuse = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(concat_dim, out_dim))

    def forward(self, x):
        """x: [B*N, 1, W] → [B*N, out_dim]"""
        h3 = self.branch3(x).squeeze(-1)   # [B*N, hp]
        h5 = self.branch5(x).squeeze(-1)   # [B*N, hp]
        h7 = self.branch7(x).squeeze(-1)   # [B*N, hp]
        h = torch.cat([h3, h5, h7], dim=-1) # [B*N, 3*hp]
        return self.fuse(h)                 # [B*N, out_dim]


# ═══════════════════════════════════════════════════════════════
# 3. 先验节点嵌入 (不变)
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
# 4. Node-Level Fusion (不变)
# ═══════════════════════════════════════════════════════════════
class NodeLevelFusion(nn.Module):
    def __init__(self, d_space=32, d_time=32, d_fuse=32, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_space + d_time, d_fuse),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_fuse, d_fuse),
        )

    def forward(self, h_space, h_time):
        return self.mlp(torch.cat([h_space, h_time], dim=-1))


# ═══════════════════════════════════════════════════════════════
# 5. ParallelSpatioTemporalEncoder (时间分支改为多尺度)
# ═══════════════════════════════════════════════════════════════
class ParallelSpatioTemporalEncoder(nn.Module):
    def __init__(self, num_variables=51, window_size=60,
                 prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, dropout=0.2,
                 latent_dim=64, use_flatten=True, boost=0.3):
        super().__init__()
        self.num_variables = num_variables
        self.window_size = window_size
        self.use_flatten = use_flatten

        self.dyn_graph = DynamicPearsonPriorGraph(
            num_variables, prior_edge_index, prior_weights, boost=boost)

        self.prior_embed = PriorNodeEmbedding(
            num_variables, prior_edge_index, prior_weights, hidden_dim)

        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim,
                              heads=gat_heads, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        # ── 时间: 多尺度卷积 [NEW] ──
        self.temporal_enc = MultiScaleTemporalEncoder(
            window_size=window_size, out_dim=hidden_dim,
            hidden_per_scale=hidden_dim//4, dropout=dropout)  # 32/4=8

        self.fusion = NodeLevelFusion(d_space=hidden_dim, d_time=hidden_dim,
                                      d_fuse=hidden_dim, dropout=dropout)

        if use_flatten:
            flat_dim = num_variables * hidden_dim
            mid_dim = flat_dim // 16
            self.latent_proj = nn.Sequential(
                nn.Linear(flat_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, latent_dim),
            )
        else:
            self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        b, w, n = x.shape

        edges = self.dyn_graph(x)

        h_gat = self.gat(x.permute(0, 2, 1), edges)              # [B, N, hidden]
        h_prior = self.prior_embed().unsqueeze(0).expand(b, -1, -1)
        g = self.gate(torch.cat([h_gat, h_prior], dim=-1))
        h_space = g * h_gat + (1 - g) * h_prior                   # [B, N, hidden]

        # ── 时间: per-var 多尺度卷积 ──
        x_var = x.permute(0, 2, 1).reshape(b * n, 1, w)          # [B*51, 1, 60]
        h_time = self.temporal_enc(x_var).reshape(b, n, -1)       # [B, 51, hidden]

        h_fuse = self.fusion(h_space, h_time)

        if self.use_flatten:
            z = h_fuse.reshape(b, n * h_fuse.shape[-1])
        else:
            z = h_fuse.mean(dim=1)
        z = self.latent_proj(z)
        return z, edges


# ═══════════════════════════════════════════════════════════════
# 6. ParallelMS_USAD (完整模型)
# ═══════════════════════════════════════════════════════════════
class ParallelMS_USAD(nn.Module):
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
            latent_dim=latent_dim, use_flatten=use_flatten,
        )
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
