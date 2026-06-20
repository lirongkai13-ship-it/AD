"""
Tri-Branch USAD v4 — attention pooling + expand + vector gate + learnable gamma.

Hybrid of v1 (expand + vector gate) and v2 (attention pooling).
No node-conditioned fusion. No position embedding. No decay bias.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block


# ============================================================
# Dynamic Pearson + Prior Boost (UNCHANGED)
# ============================================================
class DynamicPearsonPriorGraph(nn.Module):
    def __init__(self, n_vars, prior_edge_index, prior_weights, boost=0.3, threshold=0.3):
        super().__init__()
        self.n_vars = n_vars; self.boost = boost; self.threshold = threshold
        A = torch.zeros(n_vars, n_vars)
        for i in range(prior_edge_index.shape[1]):
            s, d = prior_edge_index[0, i].item(), prior_edge_index[1, i].item()
            w = prior_weights[i].item()
            if s < n_vars and d < n_vars:
                A[s, d] = max(A[s, d], w); A[d, s] = max(A[d, s], w)
        self.register_buffer("A_prior", A); self.register_buffer("prior_mask", A > 0)

    def forward(self, x):
        b, w, n = x.shape
        xc = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(xc.transpose(1, 2), xc) / (w - 1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
        C = torch.nan_to_num(cov / (std.unsqueeze(1) * std.unsqueeze(2) + 1e-8), nan=0.0)
        Ad = C.abs().mean(0); Af = Ad.clone()
        ov = self.prior_mask.to(x.device) & (Ad >= self.threshold)
        op = self.prior_mask.to(x.device) & (Ad < self.threshold)
        Af[ov] = Af[ov] + self.boost
        Af[op] = self.A_prior.to(device=x.device, dtype=Af.dtype)[op]
        dm = ~torch.eye(n, dtype=torch.bool, device=x.device)
        em = (Af.abs() >= self.threshold) & dm
        if em.any(): s, d = torch.where(em); return torch.stack([s, d], 0)
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)


class PriorNodeEmbedding(nn.Module):
    def __init__(self, nv, pei, pw, hd):
        super().__init__()
        P = torch.zeros(nv, nv)
        for i in range(pei.shape[1]):
            s, d = pei[0, i].item(), pei[1, i].item()
            P[s, d] = max(P[s, d], pw[i].item())
        P_n = P / P.sum(1, keepdim=True).clamp(1)
        self.register_buffer("P_norm", P_n)
        self.node_embed = nn.Parameter(torch.randn(nv, hd) * 0.1)
        self.proj = nn.Linear(hd, hd)

    def forward(self):
        return self.proj(torch.matmul(self.P_norm, self.node_embed))


# ============================================================
# Branch 1: Per-variable temporal Conv1d (UNCHANGED)
# ============================================================
class PerVariableConv(nn.Module):
    def __init__(self, ws=60, out_dim=32, hidden=16, dropout=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(1, hidden, 3, padding=1)
        self.c2 = nn.Conv1d(hidden, out_dim, 3, padding=1)
        self.drop = nn.Dropout(dropout); self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        h = F.relu(self.c1(x)); h = self.drop(h)
        h = F.relu(self.c2(h))
        return self.pool(h).squeeze(-1)


# ============================================================
# Branch 3 v4: attention pooling + expand + vector gate (v1-style)
# ============================================================
class GlobalTemporalAttentionBranch(nn.Module):
    """
    v4 Branch 3: attention pooling (v2) + expand + vector gate (v1).
    No node-conditioned fusion.
    """

    def __init__(self, num_variables=51, window_size=60, d=32,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.d = d; self.n_vars = num_variables; self.window_size = window_size

        # 1. Temporal Projection
        self.proj_in = nn.Linear(num_variables, d)

        # 2. Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(d); self.drop_attn = nn.Dropout(dropout)

        # 3. Temporal Attention Pooling
        self.temporal_score = nn.Sequential(
            nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

    def forward(self, x):
        """
        x: [B, W, N]
        Returns: H_global [B, N, d] (expanded from g_global)
        """
        B, W, N = x.shape; d = self.d

        # 1. Temporal Projection
        T_emb = self.proj_in(x)  # [B, W, d]

        # 2. Multi-head Self-Attention
        T_attn_raw, _ = self.attn(T_emb, T_emb, T_emb)
        T_attn = self.norm_attn(T_emb + self.drop_attn(T_attn_raw))  # [B, W, d]

        # 3. Temporal Attention Pooling (learnable, NOT mean)
        scores = self.temporal_score(T_attn).squeeze(-1)   # [B, W]
        pool_weights = torch.softmax(scores, dim=1)          # [B, W]
        g_global = torch.sum(pool_weights.unsqueeze(-1) * T_attn, dim=1)  # [B, d]

        # 4. Expand to nodes (v1-style, no node-conditioned fusion)
        return g_global.unsqueeze(1).expand(B, N, d), pool_weights  # [B,N,d], [B,W]


# ============================================================
# Residual Gated Fusion — v1-style: gate computed from H_global
# ============================================================
class ResidualGatedFusion(nn.Module):
    """
    v1-style: gate = sigmoid(MLP(H_global)), vector gate [B,N,d].
    gamma and gate_scale are configurable.
    """

    def __init__(self, d=32, gamma_init=0.05, dropout=0.1,
                 gamma_mode='fixed', gate_scale=1.0):
        super().__init__()
        self.gamma_mode = gamma_mode; self.gate_scale = gate_scale
        if gamma_mode == 'learnable':
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        else:
            self.register_buffer('gamma', torch.tensor(float(gamma_init)))
        self.gate_mlp = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, d))

    def forward(self, H_base, H_global):
        """
        H_base:   [B, N, d]
        H_global: [B, N, d]  (expanded from g_global)
        Returns:  H_fuse [B, N, d], gate [B, N, d]
        """
        gate = torch.sigmoid(self.gate_mlp(H_global))  # [B, N, d] vector gate
        return H_base + self.gamma * self.gate_scale * gate * H_global, gate


# ============================================================
# Main Encoder
# ============================================================
class TriBranchEncoder(nn.Module):
    def __init__(self, nv=51, ws=60, prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, dropout=0.2,
                 latent_dim=64, use_flatten=True, boost=0.3,
                 temporal_mode="per_variable_conv",
                 encoder_mode="tri_branch_residual_gate",
                 gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0):
        super().__init__()
        self.nv = nv; self.ws = ws; self.use_flatten = use_flatten
        self.encoder_mode = encoder_mode

        # Branch 1
        self.temporal_enc = PerVariableConv(ws, hidden_dim, hidden_dim // 2, dropout)

        # Branch 2
        self.dyn_graph = DynamicPearsonPriorGraph(nv, prior_edge_index, prior_weights, boost)
        self.prior_embed = PriorNodeEmbedding(nv, prior_edge_index, prior_weights, hidden_dim)
        self.gat = GATv2Block(ws, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        # Branch 3 v4
        if encoder_mode == "tri_branch_residual_gate":
            self.global_temp = GlobalTemporalAttentionBranch(
                num_variables=nv, window_size=ws, d=hidden_dim, n_heads=4, dropout=dropout)
            self.gated_fusion = ResidualGatedFusion(
                d=hidden_dim, gamma_init=gamma_value, gamma_mode=gamma_mode,
                gate_scale=gate_scale)
        else:
            self.global_temp = None; self.gated_fusion = None

        # Base fusion
        self.base_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

        # Latent
        if use_flatten:
            fd = nv * hidden_dim; md = fd // 16
            self.latent_proj = nn.Sequential(
                nn.Linear(fd, md), nn.ReLU(), nn.Linear(md, latent_dim))
        else:
            self.latent_proj = nn.Linear(hidden_dim, latent_dim)

        self._debug_done = False

    def forward(self, x):
        b, w, n = x.shape

        # Branch 1
        xv = x.permute(0, 2, 1).reshape(b * n, 1, w)
        h_node = self.temporal_enc(xv).reshape(b, n, -1)

        # Branch 2
        edges = self.dyn_graph(x)
        hg = self.gat(x.permute(0, 2, 1), edges)
        hp = self.prior_embed().unsqueeze(0).expand(b, -1, -1)
        g = self.gate(torch.cat([hg, hp], -1))
        h_space = g * hg + (1 - g) * hp

        # Base fusion
        h_base = self.base_fusion(torch.cat([h_node, h_space], -1))

        # Branch 3 v4: attention pooling + expand + vector gate
        if self.encoder_mode == "tri_branch_residual_gate" and self.global_temp is not None:
            h_global, attn_w = self.global_temp(x)  # [B,N,d], [B,W]
            h_fuse, gate_val = self.gated_fusion(h_base, h_global)
            g_global = h_global.mean(dim=1)  # average for logging

            if not self._debug_done and self.training:
                self._debug_done = True
                fr = (h_fuse - h_base).norm().item() / max(h_base.norm().item(), 1e-8)
                gm = gate_val.mean().item()
                print(f"\n[v4 DEBUG] gate_type=vector gate_shape={list(gate_val.shape)}")
                print(f"  gate_mean={gate_val.mean().item():.4f} gate_min={gate_val.min().item():.4f} gate_max={gate_val.max().item():.4f}")
                print(f"  H_base_norm={h_base.norm().item():.2f} H_global_norm={h_global.norm().item():.2f}")
                print(f"  fusion_ratio={fr:.4f} gamma={self.gated_fusion.gamma.item():.4f}")
        else:
            h_fuse = h_base; g_global = None; attn_w = None; gate_val = None

        # Latent
        z = h_fuse.reshape(b, n * h_fuse.shape[-1]) if self.use_flatten else h_fuse.mean(1)
        z = self.latent_proj(z)

        return z, edges, {
            'h_node': h_node, 'h_space': h_space, 'h_base': h_base,
            'h_fuse': h_fuse, 'g_global': g_global,
            'attn_weights': attn_w, 'gate': gate_val}


# ============================================================
# Full Model (UNCHANGED decoder)
# ============================================================
class TriBranch_USAD_v4(nn.Module):
    def __init__(self, nv, ws, static_edge_index,
                 prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2,
                 gru_hidden=32, tcn_channels=32, tcn_blocks=1,
                 dropout=0.2, latent_dim=64, use_flatten=True,
                 temporal_mode="per_variable_conv",
                 encoder_mode="tri_branch_residual_gate",
                 gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0):
        super().__init__()
        df = nv * ws
        self.encoder = TriBranchEncoder(
            nv, ws, prior_edge_index, prior_weights,
            hidden_dim, gat_heads, dropout,
            latent_dim, use_flatten,
            temporal_mode=temporal_mode,
            encoder_mode=encoder_mode,
            gamma_mode=gamma_mode, gamma_value=gamma_value,
            gate_scale=gate_scale)

        dh = latent_dim * 2
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, dh), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dh, df))
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, dh), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dh, df))

    def forward(self, x, edge_index):
        b, w, n = x.shape
        z, edges, extras = self.encoder(x)
        r1 = self.decoder1(z).view(b, w, n)
        r2 = self.decoder2(z).view(b, w, n)
        with torch.no_grad():
            z2, _, _ = self.encoder(r1); z2 = z2.detach()
        r12 = self.decoder2(z2).view(b, w, n)
        return r1, r2, r12, extras

    def forward_eval(self, x, edge_index):
        b, w, n = x.shape
        z, _, _ = self.encoder(x)
        return self.decoder1(z).view(b, w, n)
