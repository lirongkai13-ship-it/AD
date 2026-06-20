"""
Tri-Branch USAD v3 — Branch 3 with Temporal Position Embedding + Relative Decay Bias.

Upgrades over v2:
  - Temporal Position Embedding (learnable)
  - Learnable Relative Temporal Decay Bias in MHA attention_mask
  - All v2 features preserved (attn pooling, node-cond fusion, per-node gate)

Branch 1 (per-variable Conv1d) and Branch 2 (GATv2 + prior) — UNCHANGED.
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
# Branch 3 (V3 UPGRADE): Position Embedding + Decay Bias
# ============================================================
class GlobalTemporalAttentionBranch(nn.Module):
    """
    v3 Branch 3: v2 base + temporal position embedding + relative decay bias.

    Pipeline:
      1. Temporal Projection: Linear(N, d) -> T_emb [B, W, d]
      2. + Position Embedding (learnable) -> T_emb = T_emb + pos
      3. Multi-head Self-Attn with temporal decay bias
      4. Temporal Attn Pooling
      5. Node-conditioned Fusion
      6. Node-conditioned Gate
    """

    def __init__(self, num_variables=51, window_size=60, d=32,
                 n_heads=4, dropout=0.1,
                 use_temporal_attn_pooling=True,
                 use_node_conditioned_fusion=True,
                 gate_type="scalar",
                 # --- v3 new options ---
                 use_pos_emb=True,
                 use_relative_decay_bias=True,
                 learnable_decay=True,
                 init_raw_lambda=-4.0,
                 fixed_lambda_decay=0.02,
                 max_window_size=128):
        super().__init__()
        self.d = d; self.n_vars = num_variables; self.window_size = window_size
        self.use_temporal_attn_pooling = use_temporal_attn_pooling
        self.use_node_conditioned_fusion = use_node_conditioned_fusion
        self.gate_type = gate_type
        self.use_pos_emb = use_pos_emb
        self.use_relative_decay_bias = use_relative_decay_bias
        self.learnable_decay = learnable_decay
        self.ep = -1  # debug epoch counter

        # 1. Temporal Projection
        self.proj_in = nn.Linear(num_variables, d)

        # 2. Position Embedding (v3 NEW)
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, max_window_size, d) * 0.02)

        # 3. Relative Decay Bias (v3 NEW)
        if use_relative_decay_bias:
            if learnable_decay:
                self.raw_lambda = nn.Parameter(torch.tensor(float(init_raw_lambda)))
            else:
                self.register_buffer('raw_lambda', torch.tensor(float(init_raw_lambda)))
        else:
            self.register_buffer('raw_lambda', torch.tensor(float(init_raw_lambda)))

        # 4. Multi-head Self-Attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(d); self.drop_attn = nn.Dropout(dropout)

        # 5. Temporal Attention Pooling
        if use_temporal_attn_pooling:
            self.temporal_score = nn.Sequential(
                nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))

        # 6. Node-conditioned Fusion
        if use_node_conditioned_fusion:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))

        # 7. Node-conditioned Gate
        gate_out_dim = 1 if gate_type == "scalar" else d
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * d, d // 2), nn.ReLU(), nn.Linear(d // 2, gate_out_dim))

        # Debug info store
        self.debug_info = {}

    def forward(self, x, H_base, debug=False):
        """
        Args:
            x:      [B, W, N]
            H_base: [B, N, d]

        Returns:
            H_global:   [B, N, d]
            gate:       [B, N, 1] or [B, N, d]
            g_global:   [B, d]
            attn_w:     [B, W]  temporal pooling weights
        """
        B, W, N = x.shape; d = self.d

        # 1. Temporal Projection
        T_emb = self.proj_in(x)  # [B, W, d]

        # 2. Position Embedding (v3 NEW)
        if self.use_pos_emb:
            pos = self.pos_emb[:, :W, :]  # [1, W, d]
            T_emb = T_emb + pos

        # 3. Build temporal decay bias (v3 NEW)
        lambda_decay = F.softplus(self.raw_lambda)  # always > 0
        temporal_bias = None
        if self.use_relative_decay_bias:
            idx = torch.arange(W, device=x.device)
            dist = (idx[None, :] - idx[:, None]).abs().float()  # [W, W]
            temporal_bias = -lambda_decay * dist  # [W, W]

        # 4. Multi-head Self-Attention with decay bias
        # nn.MultiheadAttention attn_mask: add to scores BEFORE softmax
        # Shape: [W, W] broadcast to [B*heads, W, W]
        attn_out, attn_weights_raw = self.temporal_attn(
            T_emb, T_emb, T_emb,
            attn_mask=temporal_bias,
            need_weights=True,
            average_attn_weights=False)
        # attn_out: [B, W, d]; attn_weights_raw: [B, heads, W, W]
        T_attn = self.norm_attn(T_emb + self.drop_attn(attn_out))  # [B, W, d]

        # 5. Temporal Pooling
        if self.use_temporal_attn_pooling:
            scores = self.temporal_score(T_attn).squeeze(-1)   # [B, W]
            pool_weights = torch.softmax(scores, dim=1)          # [B, W]
            g_global = torch.sum(pool_weights.unsqueeze(-1) * T_attn, dim=1)  # [B, d]
        else:
            pool_weights = torch.ones(B, W, device=x.device) / W
            g_global = T_attn.mean(dim=1)

        # 6. Node-conditioned Fusion
        if self.use_node_conditioned_fusion:
            g_expand = g_global.unsqueeze(1).expand(B, N, d)
            C = torch.cat([H_base, g_expand, H_base * g_expand], dim=-1)  # [B, N, 3d]
            H_global = self.fusion_mlp(C)  # [B, N, d]
            gate_raw = self.gate_mlp(C)
            gate = torch.sigmoid(gate_raw)
        else:
            H_global = g_global.unsqueeze(1).expand(B, N, d)
            gate_raw = self.gate_mlp(
                torch.cat([g_global, g_global, g_global], dim=-1)
                .unsqueeze(1).expand(B, N, 3 * d))
            gate = torch.sigmoid(gate_raw)
            if self.gate_type == "scalar":
                gate = gate[..., :1]

        # Debug output (first batch only)
        if debug:
            fusion_ratio = 0.0  # computed outside
            self.debug_info = {
                'use_pos_emb': self.use_pos_emb,
                'use_relative_decay_bias': self.use_relative_decay_bias,
                'lambda_decay': lambda_decay.item(),
                'temporal_bias_min': temporal_bias.min().item() if temporal_bias is not None else None,
                'temporal_bias_max': temporal_bias.max().item() if temporal_bias is not None else None,
                'temporal_bias_mean': temporal_bias.mean().item() if temporal_bias is not None else None,
                'attn_weights_shape': list(attn_weights_raw.shape),
                'pool_weights_mean': pool_weights.mean().item(),
                'pool_weights_min': pool_weights.min().item(),
                'pool_weights_max': pool_weights.max().item(),
                'H_base_norm': H_base.norm().item(),
                'H_global_norm': H_global.norm().item(),
                'gate_mean': gate.mean().item(),
                'gate_min': gate.min().item(),
                'gate_max': gate.max().item(),
            }

        return H_global, gate, g_global, pool_weights


# ============================================================
# Residual Gated Fusion (UNCHANGED)
# ============================================================
class ResidualGatedFusion(nn.Module):
    def __init__(self, gamma_init=0.05, gamma_mode='fixed', gate_scale=1.0):
        super().__init__()
        self.gamma_mode = gamma_mode; self.gate_scale = gate_scale
        if gamma_mode == 'learnable':
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        else:
            self.register_buffer('gamma', torch.tensor(float(gamma_init)))

    def forward(self, H_base, H_global, gate):
        return H_base + self.gamma * self.gate_scale * gate * H_global


# ============================================================
# Main Encoder (same as v2, using v3 GlobalTemporalAttentionBranch)
# ============================================================
class TriBranchEncoder(nn.Module):
    def __init__(self, nv=51, ws=60, prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2, dropout=0.2,
                 latent_dim=64, use_flatten=True, boost=0.3,
                 temporal_mode="per_variable_conv",
                 encoder_mode="tri_branch_residual_gate",
                 gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0,
                 # --- v2 options ---
                 use_temporal_attn_pooling=True,
                 use_node_conditioned_fusion=True,
                 gate_type="scalar",
                 # --- v3 options ---
                 use_pos_emb=True,
                 use_relative_decay_bias=True,
                 learnable_decay=True,
                 init_raw_lambda=-4.0):
        super().__init__()
        self.nv = nv; self.ws = ws; self.use_flatten = use_flatten
        self.encoder_mode = encoder_mode

        # Branch 1
        if temporal_mode == "per_variable_conv":
            self.temporal_enc = PerVariableConv(ws, hidden_dim, hidden_dim // 2, dropout)
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        # Branch 2
        self.dyn_graph = DynamicPearsonPriorGraph(nv, prior_edge_index, prior_weights, boost)
        self.prior_embed = PriorNodeEmbedding(nv, prior_edge_index, prior_weights, hidden_dim)
        self.gat = GATv2Block(ws, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        # Branch 3 (v3)
        if encoder_mode == "tri_branch_residual_gate":
            self.global_temp = GlobalTemporalAttentionBranch(
                num_variables=nv, window_size=ws, d=hidden_dim,
                n_heads=4, dropout=dropout,
                use_temporal_attn_pooling=use_temporal_attn_pooling,
                use_node_conditioned_fusion=use_node_conditioned_fusion,
                gate_type=gate_type,
                use_pos_emb=use_pos_emb,
                use_relative_decay_bias=use_relative_decay_bias,
                learnable_decay=learnable_decay,
                init_raw_lambda=init_raw_lambda)
            self.gated_fusion = ResidualGatedFusion(
                gamma_init=gamma_value, gamma_mode=gamma_mode, gate_scale=gate_scale)
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

        # Branch 3 (v3)
        if self.encoder_mode == "tri_branch_residual_gate" and self.global_temp is not None:
            do_debug = (not self._debug_done) and self.training
            h_global, node_gate, g_global, attn_w = self.global_temp(x, h_base, debug=do_debug)
            h_fuse = self.gated_fusion(h_base, h_global, node_gate)

            if do_debug:
                self._debug_done = True
                fusion_ratio = (h_fuse - h_base).norm().item() / max(h_base.norm().item(), 1e-8)
                self.global_temp.debug_info['fusion_ratio'] = fusion_ratio
                self.global_temp.debug_info['gamma'] = self.gated_fusion.gamma.item()
                print("\n[v3 Branch3 DEBUG]")
                for k, v in self.global_temp.debug_info.items():
                    print(f"  {k}: {v}")
        else:
            h_fuse = h_base
            g_global = None; attn_w = None; node_gate = None

        # Latent
        z = h_fuse.reshape(b, n * h_fuse.shape[-1]) if self.use_flatten else h_fuse.mean(1)
        z = self.latent_proj(z)

        return z, edges, {
            'h_node': h_node, 'h_space': h_space, 'h_base': h_base,
            'h_fuse': h_fuse, 'g_global': g_global,
            'temporal_attn_weights': attn_w, 'node_gate': node_gate}


# ============================================================
# Full Model (UNCHANGED decoder)
# ============================================================
class TriBranch_USAD_v3(nn.Module):
    def __init__(self, nv, ws, static_edge_index,
                 prior_edge_index=None, prior_weights=None,
                 hidden_dim=32, gat_heads=2,
                 gru_hidden=32, tcn_channels=32, tcn_blocks=1,
                 dropout=0.2, latent_dim=64, use_flatten=True,
                 temporal_mode="per_variable_conv",
                 encoder_mode="tri_branch_residual_gate",
                 gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0,
                 # --- v2 options ---
                 use_temporal_attn_pooling=True,
                 use_node_conditioned_fusion=True,
                 gate_type="scalar",
                 # --- v3 options ---
                 use_pos_emb=True,
                 use_relative_decay_bias=True,
                 learnable_decay=True,
                 init_raw_lambda=-4.0):
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
            gate_type=gate_type,
            use_pos_emb=use_pos_emb,
            use_relative_decay_bias=use_relative_decay_bias,
            learnable_decay=learnable_decay,
            init_raw_lambda=init_raw_lambda)

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
