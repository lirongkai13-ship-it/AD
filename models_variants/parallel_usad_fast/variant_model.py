"""
parallel_usad_prior + 向量化 GATv2 (独立文件, 不改 model.py)

唯一区别: ManualGATv2Layer 用 scatter_add 替代 Python for-loop
其余完全同 parallel_usad_prior
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# 注意: 不用 model.GATv2Block, 自己实现向量化版本


# ═══════════════════════════════════════════════════════════════
# 向量化 ManualGATv2Layer (独立实现)
# ═══════════════════════════════════════════════════════════════
class VectorizedGATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2, concat=True):
        super().__init__()
        self.in_dim = in_dim; self.out_dim = out_dim
        self.heads = heads; self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.lin_l = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_r = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        b, n, _ = x.shape
        src, dst = edge_index[0], edge_index[1]

        h_l = self.lin_l(x).view(b, n, self.heads, self.out_dim)
        h_r = self.lin_r(x).view(b, n, self.heads, self.out_dim)
        h_src = h_l[:, src, :, :]
        h_dst = h_r[:, dst, :, :]

        e = F.leaky_relu(h_src + h_dst, negative_slope=0.2)
        e = (e * self.att.view(1, 1, self.heads, self.out_dim)).sum(dim=-1)

        # 向量化 per-node softmax
        e_max = torch.zeros(b, n, self.heads, device=x.device, dtype=e.dtype)
        e_max.scatter_reduce_(1, dst.view(1, -1, 1).expand(b, -1, self.heads),
                              e, reduce='amax', include_self=False)
        e_exp = torch.exp(e - e_max[:, dst, :])
        e_sum = torch.zeros(b, n, self.heads, device=x.device, dtype=e.dtype)
        e_sum.scatter_add_(1, dst.view(1, -1, 1).expand(b, -1, self.heads), e_exp)
        alpha = e_exp / (e_sum[:, dst, :] + 1e-8)
        alpha = self.dropout(alpha)

        # 向量化消息聚合
        msg = h_src * alpha.unsqueeze(-1)
        out = torch.zeros(b, n, self.heads, self.out_dim, device=x.device, dtype=msg.dtype)
        out.scatter_add_(1, dst.view(1, -1, 1, 1).expand(b, -1, self.heads, self.out_dim), msg)

        if self.concat:
            out = out.reshape(b, n, self.heads * self.out_dim)
        else:
            out = out.mean(dim=2)
        return out + self.bias


class VectorizedGATv2Block(nn.Module):
    """等同于 model.GATv2Block, 但使用向量化层"""
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = VectorizedGATv2Layer(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = VectorizedGATv2Layer(hidden_dim * heads, out_dim, heads=1, dropout=dropout, concat=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        residual = self.res_proj(x)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.gat2(h, edge_index)
        return self.norm(h + residual)


# ═══════════════════════════════════════════════════════════════
# Dynamic Pearson + 先验图 boost 融合 (不变)
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
# TemporalVariableEncoder (单尺度, 不变)
# ═══════════════════════════════════════════════════════════════
class TemporalVariableEncoder(nn.Module):
    def __init__(self, window_size=60, out_dim=32, hidden_channels=16, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, out_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        h = F.relu(self.conv1(x)); h = self.dropout(h)
        h = F.relu(self.conv2(h)); h = self.pool(h)
        return h.squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# PriorNodeEmbedding, NodeLevelFusion (不变)
# ═══════════════════════════════════════════════════════════════
class PriorNodeEmbedding(nn.Module):
    def __init__(self, n_vars, prior_edge_index, prior_weights, hidden_dim):
        super().__init__()
        P = torch.zeros(n_vars, n_vars)
        for i in range(prior_edge_index.shape[1]):
            src, dst = prior_edge_index[0, i].item(), prior_edge_index[1, i].item()
            P[src, dst] = max(P[src, dst], prior_weights[i].item())
        deg = P.sum(dim=1, keepdim=True).clamp(min=1); P_norm = P / deg
        self.register_buffer("P_norm", P_norm)
        self.node_embed = nn.Parameter(torch.randn(n_vars, hidden_dim) * 0.1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self):
        return self.proj(torch.matmul(self.P_norm, self.node_embed))


class NodeLevelFusion(nn.Module):
    def __init__(self, d_space=32, d_time=32, d_fuse=32, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_space + d_time, d_fuse),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_fuse, d_fuse))

    def forward(self, h_space, h_time):
        return self.mlp(torch.cat([h_space, h_time], dim=-1))


# ═══════════════════════════════════════════════════════════════
# ParallelSpatioTemporalEncoder
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
        self.gat = VectorizedGATv2Block(window_size, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())
        self.temporal_enc = TemporalVariableEncoder(window_size=window_size, out_dim=hidden_dim,
                                                     hidden_channels=hidden_dim//2, dropout=dropout)
        self.fusion = NodeLevelFusion(d_space=hidden_dim, d_time=hidden_dim, d_fuse=hidden_dim, dropout=dropout)

        if use_flatten:
            flat_dim = num_variables * hidden_dim; mid_dim = flat_dim // 16
            self.latent_proj = nn.Sequential(nn.Linear(flat_dim, mid_dim), nn.ReLU(),
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
# ParallelFast_USAD
# ═══════════════════════════════════════════════════════════════
class ParallelFast_USAD(nn.Module):
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
        self.decoder1 = nn.Sequential(nn.Linear(latent_dim, dec_hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(dec_hidden, dec_feat))
        self.decoder2 = nn.Sequential(nn.Linear(latent_dim, dec_hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(dec_hidden, dec_feat))

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
