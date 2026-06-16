"""Dynamic USAD + Top-K Edge: each node keeps only K strongest edges"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block, TCNBlock


class DynamicPearsonTopK(nn.Module):
    """动态 Pearson + Top-K 保边"""
    def __init__(self, n_vars, static_edge_index, top_k=10, threshold=0.3):
        super().__init__()
        self.n_vars = n_vars
        self.top_k = top_k
        self.threshold = threshold
        self.register_buffer("static_ei", static_edge_index)

    def forward(self, x):
        b, w, n = x.shape
        x_c = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_c.transpose(1,2), x_c) / (w-1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
        C = cov / (std.unsqueeze(1)*std.unsqueeze(2) + 1e-8)
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        C_avg = C.abs().mean(dim=0)

        # Top-K: 每个节点只保留最强的 K 条边
        topk_edges = []
        for node in range(n):
            scores = C_avg[node].clone()
            scores[node] = -1  # 排除自环
            if scores.max() <= 0:
                continue
            # 选 top-k 最大的邻居
            _, top_indices = torch.topk(scores, min(self.top_k, n-1))
            for dst in top_indices:
                if scores[dst] >= self.threshold:
                    topk_edges.append([node, dst.item()])

        dyn = torch.tensor(topk_edges, dtype=torch.long, device=x.device).T if topk_edges else \
              torch.zeros(2, 0, dtype=torch.long, device=x.device)

        if dyn.shape[1] > 0:
            comb = torch.cat([self.static_ei.to(x.device), dyn], dim=1)
        else:
            comb = self.static_ei.to(x.device)
        h = comb[0] * n + comb[1]
        uniq = torch.unique(h)
        return torch.stack([uniq // n, uniq % n], dim=0)


class DynTopK_USAD(nn.Module):
    def __init__(self, num_variables, window_size, static_edge_index,
                 top_k=10, hidden_dim=32, gat_heads=2, gru_hidden=32,
                 tcn_channels=32, tcn_blocks=1, dropout=0.2):
        super().__init__()
        dec_feat = window_size * num_variables
        self.dyn_graph = DynamicPearsonTopK(num_variables, static_edge_index, top_k)
        self.gat = GATv2Block(window_size, hidden_dim, hidden_dim, heads=gat_heads, dropout=dropout)
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
