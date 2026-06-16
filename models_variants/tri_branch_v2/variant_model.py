"""
Tri-Branch USAD v2 — upgraded Branch 3 (global temporal attention).

Upgrades over v1 (original):
  - Temporal attention pooling (learnable) instead of mean pooling
  - Node-conditioned global fusion instead of expand-broadcast
  - Node-conditioned gate (scalar or vector) instead of global gate
  - Ablation switches for controlled experiments

Branch 1 (per-variable Conv1d) and Branch 2 (GATv2 + prior) — UNCHANGED.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block


# ============================================================
# Dynamic Pearson + Prior Boost (UNCHANGED from v1)
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
                A[s, d] = max(A[s, d], w)
                A[d, s] = max(A[d, s], w)
        self.register_buffer("A_prior", A)
        self.register_buffer("prior_mask", A > 0)

    def forward(self, x):
        b, w, n = x.shape
        xc = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(xc.transpose(1, 2), xc) / (w - 1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
        C = torch.nan_to_num(cov / (std.unsqueeze(1) * std.unsqueeze(2) + 1e-8), nan=0.0)
        Ad = C.abs().mean(0)
        Af = Ad.clone()
        ov = self.prior_mask.to(x.device) & (Ad >= self.threshold)
        op = self.prior_mask.to(x.device) & (Ad < self.threshold)
        bv = torch.tensor(self.boost, dtype=Af.dtype, device=Af.device)
        Af[ov] = Af[ov] + bv
        Af[op] = self.A_prior.to(device=x.device, dtype=Af.dtype)[op]
        dm = ~torch.eye(n, dtype=torch.bool, device=x.device)
        em = (Af.abs() >= self.threshold) & dm
        if em.any():
            s, d = torch.where(em)
            return torch.stack([s, d], 0)
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
        self.drop = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B*N, 1, W]
        h = F.relu(self.c1(x))
        h = self.drop(h)
        h = F.relu(self.c2(h))
        return self.pool(h).squeeze(-1)  # [B*N, out_dim]


# ============================================================
# Branch 3 (UPGRADED): Global Temporal Attention with
#   temporal attention pooling + node-conditioned fusion
# ============================================================
class GlobalTemporalAttentionBranch(nn.Module):
    """
    Upgraded global temporal attention branch.

    Pipeline:
      1. Temporal Projection:  Linear(N, d)  -> T_emb [B, W, d]
      2. Multi-head Self-Attn:  MHA(T_emb) + residual + LN  -> T_attn [B, W, d]
      3. Temporal Attn Pooling: learnable scores -> softmax -> weighted sum
         -> g_global [B, d]
      4. Node-conditioned Fusion:
           C_i = concat(H_base_i, g_global, H_base_i * g_global)
           H_global_i = MLP(C_i)
         -> H_global [B, N, d]
      5. Node-conditioned Gate:
           gate_i = sigmoid(MLP_gate(C_i))
         -> gate [B, N, 1] or [B, N, d]
      6. Return H_global, gate, g_global, attn_weights
    """

    def __init__(self, num_variables=51, window_size=60, d=32,
                 n_heads=4, dropout=0.1,
                 use_temporal_attn_pooling=True,
                 use_node_conditioned_fusion=True,
                 gate_type="scalar"):
        """
        Args:
            num_variables: N — number of sensor nodes
            window_size:   W — time window length
            d:             hidden dimension (matches H_base dim)
            n_heads:       MHA heads
            dropout:       dropout rate
            use_temporal_attn_pooling:  False -> mean pooling fallback
            use_node_conditioned_fusion: False -> expand-broadcast fallback
            gate_type:     "scalar" -> gate [B,N,1], "vector" -> gate [B,N,d]
        """
        super().__init__()
        self.d = d
        self.n_vars = num_variables
        self.window_size = window_size
        self.use_temporal_attn_pooling = use_temporal_attn_pooling
        self.use_node_conditioned_fusion = use_node_conditioned_fusion
        self.gate_type = gate_type

        # 1. Temporal Projection
        self.proj_in = nn.Linear(num_variables, d)  # N -> d

        # 2. Multi-head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(d)
        self.drop_attn = nn.Dropout(dropout)

        # 3. Temporal Attention Pooling (learnable)
        if use_temporal_attn_pooling:
            self.temporal_score = nn.Sequential(
                nn.Linear(d, d // 2), nn.Tanh(),
                nn.Linear(d // 2, 1))
        # (else: fallback to mean pooling — no params needed)

        # 4. Node-conditioned Fusion MLP
        if use_node_conditioned_fusion:
            # C_i = [H_base_i (d), g_global (d), H_base_i*g_global (d)] = 3d
            self.fusion_mlp = nn.Sequential(
                nn.Linear(3 * d, d), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d, d))
        # (else: no fusion MLP, just expand g_global)

        # 5. Node-conditioned Gate MLP
        gate_out_dim = 1 if gate_type == "scalar" else d
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * d, d // 2), nn.ReLU(),
            nn.Linear(d // 2, gate_out_dim))

    def forward(self, x, H_base):
        """
        Args:
            x:      [B, W, N] — raw input window
            H_base: [B, N, d] — fused output from Branch 1 + Branch 2

        Returns:
            H_global:       [B, N, d] — node-conditioned global context
            gate:           [B, N, 1] or [B, N, d] — node-conditioned gate
            g_global:       [B, d] — pooled global vector
            attn_weights:   [B, W] — temporal attention weights
        """
        B, W, N = x.shape
        d = self.d

        # 1. Temporal Projection: [B,W,N] -> [B,W,d]
        T_emb = self.proj_in(x)  # [B, W, d]

        # 2. Multi-head Self-Attention
        T_attn_raw, _ = self.attn(T_emb, T_emb, T_emb)  # [B, W, d]
        T_attn = self.norm_attn(T_emb + self.drop_attn(T_attn_raw))  # [B, W, d]

        # 3. Temporal Pooling
        if self.use_temporal_attn_pooling:
            # Learnable attention scores
            scores = self.temporal_score(T_attn).squeeze(-1)  # [B, W]
            attn_weights = torch.softmax(scores, dim=1)        # [B, W]
            g_global = torch.sum(
                attn_weights.unsqueeze(-1) * T_attn, dim=1)    # [B, d]
        else:
            # Fallback: mean pooling
            attn_weights = torch.ones(B, W, device=x.device) / W
            g_global = T_attn.mean(dim=1)  # [B, d]

        # 4. Node-conditioned Fusion or Expand
        if self.use_node_conditioned_fusion:
            # Build C_i = [H_base_i, g_global, H_base_i * g_global]
            g_expand = g_global.unsqueeze(1).expand(B, N, d)  # [B, N, d]
            C = torch.cat([
                H_base,           # [B, N, d]
                g_expand,         # [B, N, d]
                H_base * g_expand # [B, N, d]
            ], dim=-1)             # [B, N, 3d]

            H_global = self.fusion_mlp(C)  # [B, N, d]

            # 5. Node-conditioned Gate
            gate_raw = self.gate_mlp(C)    # [B, N, gate_out_dim]
            gate = torch.sigmoid(gate_raw)
        else:
            # Fallback: expand g_global, no node conditioning
            H_global = g_global.unsqueeze(1).expand(B, N, d)  # [B, N, d]
            # Fallback gate: global scalar gate
            gate_raw = self.gate_mlp(
                torch.cat([g_global, g_global, g_global], dim=-1)
                .unsqueeze(1).expand(B, N, 3 * d))
            gate = torch.sigmoid(gate_raw)
            if self.gate_type == "scalar":
                gate = gate[..., :1]  # force [B,N,1]

        return H_global, gate, g_global, attn_weights


# ============================================================
# Residual Gated Fusion (simplified — gate is now per-node)
# ============================================================
class ResidualGatedFusion(nn.Module):
    """
    H_fuse = H_base + gamma * gate_scale * gate * H_global

    gate is now per-node from Branch 3 (not computed here).
    gamma and gate_scale remain configurable.
    """

    def __init__(self, gamma_init=0.05, gamma_mode='fixed', gate_scale=1.0):
        super().__init__()
        self.gamma_mode = gamma_mode
        self.gate_scale = gate_scale
        if gamma_mode == 'learnable':
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        else:
            self.register_buffer('gamma', torch.tensor(float(gamma_init)))

    def forward(self, H_base, H_global, gate):
        """
        Args:
            H_base:   [B, N, d]
            H_global: [B, N, d]
            gate:     [B, N, 1] or [B, N, d]
        Returns:
            H_fuse:   [B, N, d]
        """
        return H_base + self.gamma * self.gate_scale * gate * H_global


# ============================================================
# Main Encoder (Branch 1 + Branch 2 unchanged, Branch 3 upgraded)
# ============================================================
class TriBranchEncoder(nn.Module):
    def __init__(self, nv=51, ws=60, prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, dropout=0.2,
                 latent_dim=64, use_flatten=True, boost=0.3,
                 temporal_mode="per_variable_conv",
                 encoder_mode="tri_branch_residual_gate",
                 gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0,
                 # --- Branch 3 v2 options ---
                 use_temporal_attn_pooling=True,
                 use_node_conditioned_fusion=True,
                 gate_type="scalar"):
        super().__init__()
        self.nv = nv; self.ws = ws; self.use_flatten = use_flatten
        self.temporal_mode = temporal_mode
        self.encoder_mode = encoder_mode
        self.gate_scale = gate_scale

        # Branch 1: Per-variable temporal (UNCHANGED)
        if temporal_mode == "per_variable_conv":
            self.temporal_enc = PerVariableConv(ws, hidden_dim, hidden_dim // 2, dropout)
        elif temporal_mode == "per_variable_dilated_conv":
            from .variant_model import PerVariableDilatedConv
            self.temporal_enc = PerVariableDilatedConv(ws, hidden_dim, hidden_dim // 2, dropout)
        elif temporal_mode == "per_variable_residual_ms_tcn":
            from .variant_model import PerVariableResidualMSTCN
            self.temporal_enc = PerVariableResidualMSTCN(ws, hidden_dim, hidden_dim // 2, dropout)
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        # Branch 2: Spatial GATv2 + prior (UNCHANGED)
        self.dyn_graph = DynamicPearsonPriorGraph(nv, prior_edge_index, prior_weights, boost)
        self.prior_embed = PriorNodeEmbedding(nv, prior_edge_index, prior_weights, hidden_dim)
        self.gat = GATv2Block(ws, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        # Branch 3: Global temporal attention (UPGRADED v2)
        if encoder_mode == "tri_branch_residual_gate":
            self.global_temp = GlobalTemporalAttentionBranch(
                num_variables=nv, window_size=ws, d=hidden_dim,
                n_heads=4, dropout=dropout,
                use_temporal_attn_pooling=use_temporal_attn_pooling,
                use_node_conditioned_fusion=use_node_conditioned_fusion,
                gate_type=gate_type)
            self.gated_fusion = ResidualGatedFusion(
                gamma_init=gamma_value, gamma_mode=gamma_mode,
                gate_scale=gate_scale)
        else:
            self.global_temp = None
            self.gated_fusion = None

        # Base fusion: H_node + H_space (UNCHANGED)
        self.base_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim))

        # Latent projection (UNCHANGED)
        if use_flatten:
            fd = nv * hidden_dim
            md = fd // 16
            self.latent_proj = nn.Sequential(
                nn.Linear(fd, md), nn.ReLU(),
                nn.Linear(md, latent_dim))
        else:
            self.latent_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        b, w, n = x.shape

        # Branch 1: H_node [B,N,d]
        xv = x.permute(0, 2, 1).reshape(b * n, 1, w)
        h_node = self.temporal_enc(xv).reshape(b, n, -1)  # [B,N,d]

        # Branch 2: H_space [B,N,d]
        edges = self.dyn_graph(x)
        hg = self.gat(x.permute(0, 2, 1), edges)          # [B,N,d]
        hp = self.prior_embed().unsqueeze(0).expand(b, -1, -1)  # [B,N,d]
        g = self.gate(torch.cat([hg, hp], -1))            # [B,N,d]
        h_space = g * hg + (1 - g) * hp                    # [B,N,d]

        # Base fusion: H_node + H_space
        h_base = self.base_fusion(torch.cat([h_node, h_space], -1))  # [B,N,d]

        # Branch 3: Global temporal attention (UPGRADED v2)
        if self.encoder_mode == "tri_branch_residual_gate" and self.global_temp is not None:
            # New Branch 3 returns: H_global, gate, g_global, attn_weights
            h_global, node_gate, g_global, attn_w = self.global_temp(
                x, h_base)  # x is [B,W,N], h_base is [B,N,d]
            h_fuse = self.gated_fusion(h_base, h_global, node_gate)
        else:
            h_fuse = h_base
            g_global = None
            attn_w = None
            node_gate = None

        # Latent
        z = h_fuse.reshape(b, n * h_fuse.shape[-1]) if self.use_flatten else h_fuse.mean(1)
        z = self.latent_proj(z)

        # Return extras for visualization/analysis
        return z, edges, {
            'h_node': h_node, 'h_space': h_space, 'h_base': h_base,
            'h_fuse': h_fuse, 'g_global': g_global,
            'temporal_attn_weights': attn_w, 'node_gate': node_gate}


# ============================================================
# Full Model (UNCHANGED decoder and training flow)
# ============================================================
class TriBranch_USAD_v2(nn.Module):
    def __init__(self, nv, ws, static_edge_index,
                 prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2,
                 gru_hidden=32, tcn_channels=32, tcn_blocks=1,
                 dropout=0.2, latent_dim=64, use_flatten=True,
                 temporal_mode="per_variable_conv",
                 encoder_mode="tri_branch_residual_gate",
                 gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0,
                 # --- Branch 3 v2 options ---
                 use_temporal_attn_pooling=True,
                 use_node_conditioned_fusion=True,
                 gate_type="scalar"):
        super().__init__()
        df = nv * ws
        self.encoder = TriBranchEncoder(
            nv, ws, prior_edge_index, prior_weights,
            hidden_dim, gat_heads, dropout,
            latent_dim, use_flatten,
            temporal_mode=temporal_mode,
            encoder_mode=encoder_mode,
            gamma_mode=gamma_mode, gamma_value=gamma_value,
            gate_scale=gate_scale,
            use_temporal_attn_pooling=use_temporal_attn_pooling,
            use_node_conditioned_fusion=use_node_conditioned_fusion,
            gate_type=gate_type)

        dh = latent_dim * 2
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, dh), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dh, df))
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, dh), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dh, df))

    def forward(self, x, edge_index):
        b, w, n = x.shape
        z, edges, extras = self.encoder(x)
        r1 = self.decoder1(z).view(b, w, n)
        r2 = self.decoder2(z).view(b, w, n)
        with torch.no_grad():
            z2, _, _ = self.encoder(r1)
            z2 = z2.detach()
        r12 = self.decoder2(z2).view(b, w, n)
        return r1, r2, r12, extras

    def forward_eval(self, x, edge_index):
        b, w, n = x.shape
        z, _, _ = self.encoder(x)
        return self.decoder1(z).view(b, w, n)
