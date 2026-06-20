"""
Tri-Branch USAD v5 — v1 learnable + Temporal Position Embedding.

Minimal change: only adds learnable pos_emb to v1's Branch 3.
Everything else (mean pooling, expand, vector gate, learnable gamma) unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block


# === Dynamic Pearson + Prior Boost (UNCHANGED) ===
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


# === Branch 1: Per-variable temporal Conv1d (UNCHANGED) ===
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


# === Branch 3 v5: v1 + Position Embedding ===
class GlobalTemporalAttentionBranch(nn.Module):
    """
    v5: v1 structure + learnable temporal position embedding.
    Pipeline: proj_in → +pos_emb → MHA+LN → mean_pool → output [B,N,d]
    """

    def __init__(self, num_variables=51, window_size=60, d=32,
                 n_heads=4, dropout=0.1, max_window_size=128):
        super().__init__()
        self.d = d; self.window_size = window_size

        # 1. Temporal Projection
        self.proj_in = nn.Linear(num_variables, d)

        # 2. Position Embedding (ONLY CHANGE vs v1)
        self.pos_emb = nn.Parameter(torch.randn(1, max_window_size, d) * 0.02)

        # 3. Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(d); self.drop_attn = nn.Dropout(dropout)

    def forward(self, x, debug=False):
        """
        x: [B, W, N]
        Returns: H_global [B, N, d] (expanded from mean pooling)
        """
        B, W, N = x.shape; d = self.d

        # 1. Temporal Projection
        T_emb = self.proj_in(x)  # [B, W, d]

        # 2. Add position embedding (v5 NEW)
        pos = self.pos_emb[:, :W, :]  # [1, W, d]
        T_emb = T_emb + pos

        # 3. Multi-head Self-Attention
        T_attn_raw, _ = self.attn(T_emb, T_emb, T_emb)
        T_attn = self.norm_attn(T_emb + self.drop_attn(T_attn_raw))  # [B, W, d]

        # 4. Mean pooling (v1-style, NOT attention pooling)
        g_global = T_attn.mean(dim=1)  # [B, d]

        # 5. Expand to nodes
        return g_global.unsqueeze(1).expand(B, N, d), T_emb, T_attn  # H_global, T_emb, T_attn


# === Residual Gated Fusion (v1-style vector gate) ===
class ResidualGatedFusion(nn.Module):
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
        gate = torch.sigmoid(self.gate_mlp(H_global))  # [B, N, d] vector gate
        return H_base + self.gamma * self.gate_scale * gate * H_global, gate


# === Main Encoder ===
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

        self.temporal_enc = PerVariableConv(ws, hidden_dim, hidden_dim // 2, dropout)
        self.dyn_graph = DynamicPearsonPriorGraph(nv, prior_edge_index, prior_weights, boost)
        self.prior_embed = PriorNodeEmbedding(nv, prior_edge_index, prior_weights, hidden_dim)
        self.gat = GATv2Block(ws, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        if encoder_mode == "tri_branch_residual_gate":
            self.global_temp = GlobalTemporalAttentionBranch(
                num_variables=nv, window_size=ws, d=hidden_dim, n_heads=4, dropout=dropout)
            self.gated_fusion = ResidualGatedFusion(
                d=hidden_dim, gamma_init=gamma_value, gamma_mode=gamma_mode, gate_scale=gate_scale)
        else:
            self.global_temp = None; self.gated_fusion = None

        self.base_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

        if use_flatten:
            fd = nv * hidden_dim; md = fd // 16
            self.latent_proj = nn.Sequential(nn.Linear(fd, md), nn.ReLU(), nn.Linear(md, latent_dim))
        else:
            self.latent_proj = nn.Linear(hidden_dim, latent_dim)

        self._debug_done = False

    def forward(self, x):
        b, w, n = x.shape

        xv = x.permute(0, 2, 1).reshape(b * n, 1, w)
        h_node = self.temporal_enc(xv).reshape(b, n, -1)
        edges = self.dyn_graph(x)
        hg = self.gat(x.permute(0, 2, 1), edges)
        hp = self.prior_embed().unsqueeze(0).expand(b, -1, -1)
        g = self.gate(torch.cat([hg, hp], -1))
        h_space = g * hg + (1 - g) * hp
        h_base = self.base_fusion(torch.cat([h_node, h_space], -1))

        if self.encoder_mode == "tri_branch_residual_gate" and self.global_temp is not None:
            h_global, T_emb, T_attn = self.global_temp(x)
            h_fuse, gate_val = self.gated_fusion(h_base, h_global)

            if not self._debug_done and self.training:
                self._debug_done = True
                fr = (h_fuse - h_base).norm().item() / max(h_base.norm().item(), 1e-8)
                pe = self.global_temp.pos_emb
                print(f"\n[v5 DEBUG] pos_emb shape={list(pe.shape)} pos_mean={pe.mean().item():.4f} pos_std={pe.std().item():.4f}")
                print(f"  T_emb shape={list(T_emb.shape)} T_emb_norm={T_emb.norm().item():.2f}")
                print(f"  T_attn shape={list(T_attn.shape)}")
                print(f"  H_base_norm={h_base.norm().item():.2f} H_global_norm={h_global.norm().item():.2f}")
                print(f"  gate: shape={list(gate_val.shape)} mean={gate_val.mean().item():.4f} min={gate_val.min().item():.4f} max={gate_val.max().item():.4f}")
                print(f"  fusion_ratio={fr:.4f} gamma={self.gated_fusion.gamma.item():.4f}")
        else:
            h_fuse = h_base; gate_val = None

        z = h_fuse.reshape(b, n * h_fuse.shape[-1]) if self.use_flatten else h_fuse.mean(1)
        z = self.latent_proj(z)
        return z, edges, {'h_node': h_node, 'h_base': h_base, 'h_fuse': h_fuse, 'gate': gate_val}


# === Full Model (UNCHANGED decoder) ===
class TriBranch_USAD_v5(nn.Module):
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
            temporal_mode=temporal_mode, encoder_mode=encoder_mode,
            gamma_mode=gamma_mode, gamma_value=gamma_value, gate_scale=gate_scale)
        dh = latent_dim * 2
        self.decoder1 = nn.Sequential(nn.Linear(latent_dim, dh), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dh, df))
        self.decoder2 = nn.Sequential(nn.Linear(latent_dim, dh), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dh, df))

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
