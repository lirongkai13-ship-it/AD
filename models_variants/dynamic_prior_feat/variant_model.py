"""动态 Pearson 图 + 特征级先验融合 = 边动态 + 特征静态先验"""
import torch
import torch.nn as nn
import sys, os, math
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
        cov = torch.bmm(x_c.transpose(1, 2), x_c) / (w - 1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
        corr = cov / (std.unsqueeze(1) * std.unsqueeze(2) + 1e-8)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        avg_corr = corr.abs().mean(dim=0)
        mask = (avg_corr >= self.threshold) & (~torch.eye(n, dtype=torch.bool, device=x.device))
        if mask.any():
            src, dst = torch.where(mask)
            dyn_edges = torch.stack([src, dst], dim=0)
            combined = torch.cat([self.static_ei.to(x.device), dyn_edges], dim=1)
        else:
            combined = self.static_ei.to(x.device)
        # 去重
        hashed = combined[0] * n + combined[1]
        unique_idx = torch.unique(hashed)
        return torch.stack([unique_idx // n, unique_idx % n], dim=0)


class PriorNodeEmbedding(nn.Module):
    """从先验邻接矩阵 P 学习每个变量的固定嵌入，注入 GAT 输出"""

    def __init__(self, n_vars, prior_edge_index, prior_weights, hidden_dim):
        super().__init__()
        # 构建先验邻接矩阵 P [N, N]
        P = torch.zeros(n_vars, n_vars)
        for i in range(prior_edge_index.shape[1]):
            src, dst = prior_edge_index[0, i].item(), prior_edge_index[1, i].item()
            P[src, dst] = max(P[src, dst], prior_weights[i].item())
        # 归一化（度矩阵^-1 * P）
        deg = P.sum(dim=1, keepdim=True).clamp(min=1)
        P_norm = P / deg  # [N, N]
        self.register_buffer("P_norm", P_norm)

        # 学习每个节点的先验嵌入
        self.node_embed = nn.Parameter(torch.randn(n_vars, hidden_dim) * 0.1)
        # 图卷积：用 P_norm 聚合邻居嵌入 → 先验特征
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self):
        # 先验图聚合: h_prior = P_norm × embed × proj
        N = self.P_norm.shape[0]
        h = torch.matmul(self.P_norm, self.node_embed)  # [N, H] 邻居聚合
        return self.proj(h)  # [N, H]


class GATv2_DynPrior_TCN_GRU(nn.Module):
    """
    动态 Pearson 图 + 特征级先验融合:
      GAT(动态边) → h_spatial
                         ↓
                     gate(h_spatial + h_prior) → TCN → GRU
    """

    def __init__(self, num_variables, window_size, prior_edge_index, prior_weights,
                 static_edge_index=None, hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()

        # 动态图（静态边 + 动态边）
        self.dyn_graph = DynamicPearsonGraph(num_variables, static_edge_index)

        # 先验节点嵌入
        self.prior_embed = PriorNodeEmbedding(num_variables, prior_edge_index,
                                               prior_weights, hidden_dim)

        # GAT（使用动态边）
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim,
                              heads=gat_heads, dropout=dropout)

        # 融合门控: gate = σ(W[h_spatial; h_prior])
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        # TCN + GRU
        tcn_in = hidden_dim
        self.tcn_layers = nn.ModuleList()
        for i in range(tcn_blocks):
            self.tcn_layers.append(
                TCNBlock(tcn_in, tcn_channels, kernel_size=3, dilation=2**i, dropout=dropout))
            tcn_in = tcn_channels
        self.gru = nn.GRU(tcn_channels, gru_hidden, num_layers=1, batch_first=True)
        self.pred_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_variables))
        self.recon_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gru_hidden, window_size * num_variables))

    def forward(self, x, edge_index):
        b, w, n = x.shape

        # 1. 动态图
        dyn_edges = self.dyn_graph(x)

        # 2. GAT 空间编码
        node_feat = x.transpose(1, 2)       # [B, N, W]
        h = self.gat(node_feat, dyn_edges)  # [B, N, hidden]

        # 3. 先验特征融合
        h_prior = self.prior_embed()        # [N, hidden]
        h_prior = h_prior.unsqueeze(0).expand(b, -1, -1)  # [B, N, hidden]

        # gate 融合: h_fused = gate * h + (1-gate) * h_prior
        gate_in = torch.cat([h, h_prior], dim=-1)  # [B, N, 2H]
        gate = self.gate(gate_in)                   # [B, N, H]
        h_fused = gate * h + (1 - gate) * h_prior  # [B, N, H]

        # 4. TCN + GRU
        z = h_fused.transpose(1, 2)
        for tcn in self.tcn_layers:
            z = tcn(z)
        z = z.transpose(1, 2)
        _, h_last = self.gru(z)
        h_last = h_last[-1]

        pred = self.pred_head(h_last)
        recon = self.recon_head(h_last).view(b, w, n)
        return pred, recon
