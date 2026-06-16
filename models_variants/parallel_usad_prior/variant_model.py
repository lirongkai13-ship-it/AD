"""
节点级时空并行融合 + 动态Pearson+先验图融合 + USAD 双解码器

与 parallel_usad 唯一区别:
  动态图融合对象从「静态Pearson边」改为「先验知识图」

Shape 流程:
  Input:            X [B, 60, 51]
  Spatial Branch:   X → dyn_prior_graph → GATv2 → H_space [B, 51, 32]
  Temporal Branch:  X → per-var Conv1d(轻量) → H_time [B, 51, 32]
  Node Fusion:      concat → MLP → H_fuse [B, 51, 32]
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
# 1. Dynamic Pearson + 先验图 逐边加权融合
# ═══════════════════════════════════════════════════════════════
class DynamicPearsonPriorGraph(nn.Module):
    """动态Pearson + 先验图: 拼接去重 + 重复边权重增强"""
    def __init__(self, n_vars, prior_edge_index, prior_weights, boost=0.3, threshold=0.3):
        super().__init__()
        self.n_vars = n_vars
        self.boost = boost          # 重合边额外boost
        self.threshold = threshold

        # 构建先验邻接矩阵 A_prior [N, N]
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

        # ── 动态 Pearson 矩阵 [N,N] ──
        x_c = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_c.transpose(1,2), x_c) / (w-1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
        C = cov / (std.unsqueeze(1)*std.unsqueeze(2) + 1e-8)
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        A_dyn = C.abs().mean(dim=0)

        # ── 融合: 动态 + 重合边boost ──
        A_fused = A_dyn.clone()
        overlap = self.prior_mask.to(x.device) & (A_dyn >= self.threshold)    # 两边都有
        only_prior = self.prior_mask.to(x.device) & (A_dyn < self.threshold)  # 仅先验有
        # 重合边: 动态权重 + 先验boost
        boost_val = torch.tensor(self.boost, dtype=A_fused.dtype, device=A_fused.device)
        A_fused[overlap] = A_fused[overlap] + boost_val
        # 仅先验边: 直接用先验权重
        A_fused[only_prior] = self.A_prior.to(device=x.device, dtype=A_fused.dtype)[only_prior]

        # ── 阈值提取边 ──
        diag_mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        edge_mask = (A_fused.abs() >= self.threshold) & diag_mask
        if edge_mask.any():
            src, dst = torch.where(edge_mask)
            edges = torch.stack([src, dst], dim=0)
        else:
            edges = torch.zeros(2, 0, dtype=torch.long, device=x.device)
        return edges


# ═══════════════════════════════════════════════════════════════
# 2. 轻量 TemporalVariableEncoder (复用)
# ═══════════════════════════════════════════════════════════════
class TemporalVariableEncoder(nn.Module):
    def __init__(self, window_size=60, out_dim=32, hidden_channels=16, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, out_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.dropout(h)
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        return h.squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# 3. 先验节点嵌入 (复用 prior_fusion 的逻辑)
# ═══════════════════════════════════════════════════════════════
class PriorNodeEmbedding(nn.Module):
    """从先验图构建节点嵌入"""
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
        return self.proj(torch.matmul(self.P_norm, self.node_embed))  # [N, hidden_dim]


# ═══════════════════════════════════════════════════════════════
# 4. Node-Level Fusion
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
# 5. ParallelSpatioTemporalEncoder (先验图版)
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

        # ── 动态 Pearson + 先验图: 拼接去重 + 重复边boost ──
        self.dyn_graph = DynamicPearsonPriorGraph(
            num_variables, prior_edge_index, prior_weights, boost=boost)

        # ── 先验节点嵌入 (提供节点先验特征) ──
        self.prior_embed = PriorNodeEmbedding(
            num_variables, prior_edge_index, prior_weights, hidden_dim)

        # ── 空间: GATv2 + prior embedding gate → [B, N, hidden_dim] ──
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim,
                              heads=gat_heads, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        # ── 时间: 轻量逐变量编码 ──
        self.temporal_enc = TemporalVariableEncoder(
            window_size=window_size, out_dim=hidden_dim,
            hidden_channels=hidden_dim//2, dropout=dropout)

        # ── 融合 ──
        self.fusion = NodeLevelFusion(d_space=hidden_dim, d_time=hidden_dim,
                                      d_fuse=hidden_dim, dropout=dropout)

        # ── Latent ──
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

        # ── 动态+先验图 ──
        edges = self.dyn_graph(x)

        # ── 空间: GATv2 + 先验特征 gate 融合 ──
        h_gat = self.gat(x.permute(0, 2, 1), edges)           # [B, N, hidden]
        h_prior = self.prior_embed().unsqueeze(0).expand(b, -1, -1)  # [B, N, hidden]
        g = self.gate(torch.cat([h_gat, h_prior], dim=-1))     # [B, N, hidden]
        h_space = g * h_gat + (1 - g) * h_prior                # [B, N, hidden]

        # ── 时间: 轻量逐变量编码 ──
        x_var = x.permute(0, 2, 1).reshape(b * n, 1, w)
        h_time = self.temporal_enc(x_var).reshape(b, n, -1)    # [B, N, hidden]

        # ── 节点融合 ──
        h_fuse = self.fusion(h_space, h_time)                  # [B, N, hidden]

        # ── Latent ──
        if self.use_flatten:
            z = h_fuse.reshape(b, n * h_fuse.shape[-1])
        else:
            z = h_fuse.mean(dim=1)
        z = self.latent_proj(z)
        return z, edges


# ═══════════════════════════════════════════════════════════════
# 6. ParallelPrior_USAD
# ═══════════════════════════════════════════════════════════════
class ParallelPrior_USAD(nn.Module):
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
            hidden_dim=hidden_dim, gat_heads=gat_heads,
            dropout=dropout, latent_dim=latent_dim, use_flatten=use_flatten,
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
