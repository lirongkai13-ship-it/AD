"""动态 Pearson + 先验特征融合 + USAD 双解码器"""
import torch, math
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block, TCNBlock


# ─── 动态 Pearson 图（复用） ───
class DynamicPearsonGraph(nn.Module):
    def __init__(self, n_vars, static_edge_index, threshold=0.3):
        super().__init__()
        self.n_vars = n_vars; self.threshold = threshold
        self.register_buffer("static_ei", static_edge_index)
    def forward(self, x):
        b,w,n = x.shape
        x_c = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_c.transpose(1,2), x_c)/(w-1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True)+1e-8)
        C = cov/(std.unsqueeze(1)*std.unsqueeze(2)+1e-8)
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        avg = C.abs().mean(dim=0)
        mask = (avg >= self.threshold) & (~torch.eye(n, dtype=torch.bool, device=x.device))
        if mask.any():
            src,dst = torch.where(mask); dyn = torch.stack([src,dst],dim=0)
            comb = torch.cat([self.static_ei.to(x.device), dyn], dim=1)
        else: comb = self.static_ei.to(x.device)
        h = comb[0]*n+comb[1]; u = torch.unique(h)
        return torch.stack([u//n, u%n], dim=0)


# ─── 先验节点嵌入（复用） ───
class PriorNodeEmbedding(nn.Module):
    def __init__(self, n_vars, prior_edge_index, prior_weights, hidden_dim):
        super().__init__()
        P = torch.zeros(n_vars, n_vars)
        for i in range(prior_edge_index.shape[1]):
            src,dst = prior_edge_index[0,i].item(), prior_edge_index[1,i].item()
            P[src,dst] = max(P[src,dst], prior_weights[i].item())
        deg = P.sum(dim=1, keepdim=True).clamp(min=1); P_norm = P/deg
        self.register_buffer("P_norm", P_norm)
        self.node_embed = nn.Parameter(torch.randn(n_vars, hidden_dim)*0.1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self):
        return self.proj(torch.matmul(self.P_norm, self.node_embed))


# ─── 完整模型 ───
class DynPrior_USAD(nn.Module):
    def __init__(self, num_variables, window_size, static_edge_index,
                 prior_edge_index, prior_weights,
                 hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()
        dec_feat = window_size * num_variables
        self.dyn_graph = DynamicPearsonGraph(num_variables, static_edge_index)
        self.prior_embed = PriorNodeEmbedding(num_variables, prior_edge_index, prior_weights, hidden_dim)
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        # gate: σ(W[h;h_prior])
        self.gate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())
        tcn_in = hidden_dim
        self.tcn_layers = nn.ModuleList()
        for i in range(tcn_blocks):
            self.tcn_layers.append(TCNBlock(tcn_in, tcn_channels, kernel_size=3, dilation=2**i, dropout=dropout))
            tcn_in = tcn_channels
        self.gru = nn.GRU(tcn_channels, gru_hidden, num_layers=1, batch_first=True)
        self.decoder1 = nn.Sequential(nn.Linear(gru_hidden, gru_hidden*2), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(gru_hidden*2, dec_feat))
        self.decoder2 = nn.Sequential(nn.Linear(gru_hidden, gru_hidden*2), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(gru_hidden*2, dec_feat))

    def forward(self, x, edge_index):
        b,w,n = x.shape
        edges = self.dyn_graph(x)
        h = self.gat(x.transpose(1,2), edges)           # [B,N,H]
        h_prior = self.prior_embed().unsqueeze(0).expand(b,-1,-1)  # [B,N,H]
        g = self.gate(torch.cat([h, h_prior], dim=-1))   # [B,N,H]
        h = g*h + (1-g)*h_prior                           # 融合
        z = h.transpose(1,2)
        for tcn in self.tcn_layers: z = tcn(z)
        z = z.transpose(1,2)
        _, hl = self.gru(z); hl = hl[-1]
        r1 = self.decoder1(hl).view(b,w,n)
        r2 = self.decoder2(hl).view(b,w,n)
        with torch.no_grad():
            h2 = self.gat(r1.transpose(1,2), edges)
            h2 = g*h2 + (1-g)*h_prior  # 二次编碼也用同样的gate
            z2 = h2.transpose(1,2)
            for tcn in self.tcn_layers: z2 = tcn(z2)
            z2 = z2.transpose(1,2)
            _, hl2 = self.gru(z2); hl2 = hl2[-1].detach()
        r12 = self.decoder2(hl2).view(b,w,n)
        return r1, r2, r12

    def forward_eval(self, x, edge_index):
        b,w,n = x.shape
        edges = self.dyn_graph(x)
        h = self.gat(x.transpose(1,2), edges)
        h_prior = self.prior_embed().unsqueeze(0).expand(b,-1,-1)
        g = self.gate(torch.cat([h, h_prior], dim=-1))
        h = g*h + (1-g)*h_prior
        z = h.transpose(1,2)
        for tcn in self.tcn_layers: z = tcn(z)
        z = z.transpose(1,2)
        _, hl = self.gru(z); hl = hl[-1]
        return self.decoder1(hl).view(b,w,n)
