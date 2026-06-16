"""串行 Dynamic USAD: 动态Pearson + 先验图融合（与 dynamic_usad 唯一区别：图来源）"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block, TCNBlock


# ─── 动态 Pearson + 先验图: 拼接去重 + 重复边boost ───
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


class DynPriorGraph_USAD(nn.Module):
    """与 dynamic_usad 完全相同，仅图来源从静态Pearson改为先验图"""

    def __init__(self, num_variables, window_size, prior_edge_index, prior_weights,
                 hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()
        dec_feat = window_size * num_variables

        self.dyn_graph = DynamicPearsonPriorGraph(num_variables, prior_edge_index, prior_weights)
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        tcn_in = hidden_dim
        self.tcn_layers = nn.ModuleList()
        for i in range(tcn_blocks):
            self.tcn_layers.append(TCNBlock(tcn_in, tcn_channels, kernel_size=3, dilation=2**i, dropout=dropout))
            tcn_in = tcn_channels
        self.gru = nn.GRU(tcn_channels, gru_hidden, num_layers=1, batch_first=True)

        self.decoder1 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden*2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gru_hidden*2, dec_feat))
        self.decoder2 = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden*2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gru_hidden*2, dec_feat))

    def forward(self, x, edge_index):
        b, w, n = x.shape
        edges = self.dyn_graph(x)
        h = self.gat(x.transpose(1,2), edges)
        z = h.transpose(1,2)
        for tcn in self.tcn_layers: z = tcn(z)
        z = z.transpose(1,2)
        _, hl = self.gru(z); hl = hl[-1]
        r1 = self.decoder1(hl).view(b, w, n)
        r2 = self.decoder2(hl).view(b, w, n)
        with torch.no_grad():
            h2 = self.gat(r1.transpose(1,2), edges)
            z2 = h2.transpose(1,2)
            for tcn in self.tcn_layers: z2 = tcn(z2)
            z2 = z2.transpose(1,2)
            _, hl2 = self.gru(z2); hl2 = hl2[-1].detach()
        r12 = self.decoder2(hl2).view(b, w, n)
        return r1, r2, r12

    def forward_eval(self, x, edge_index):
        b, w, n = x.shape
        edges = self.dyn_graph(x)
        h = self.gat(x.transpose(1,2), edges)
        z = h.transpose(1,2)
        for tcn in self.tcn_layers: z = tcn(z)
        z = z.transpose(1,2)
        _, hl = self.gru(z); hl = hl[-1]
        return self.decoder1(hl).view(b, w, n)
