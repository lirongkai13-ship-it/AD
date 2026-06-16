"""MTAD-GAT: Multivariate Time-series Anomaly Detection with GAT [Zhao 2020]"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model import GATv2Block

class MTADGAT(nn.Module):
    def __init__(self, n_vars=51, window=60, hidden=48, heads=2, latent=32, dropout=0.1):
        super().__init__()
        # Feature-oriented GAT
        self.gat_feat = GATv2Block(window, hidden, hidden, heads=heads, dropout=dropout)
        # Temporal GAT (沿时间维)
        self.gat_temp = GATv2Block(n_vars, hidden, hidden, heads=heads, dropout=dropout)
        # VAE
        self.enc_fc = nn.Linear(hidden*2, latent*2)  # μ + logσ
        self.dec_fc = nn.Linear(latent, hidden*2)
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_vars),
        )

    def forward(self, x, edge_index=None):
        b, w, n = x.shape
        # Feature GAT
        h_f = self.gat_feat(x.transpose(1, 2), edge_index)  # [B, N, H]
        h_f = h_f.mean(dim=1)  # [B, H]
        # Temporal GAT
        h_t = self.gat_temp(x, edge_index)  # [B, W, H]
        h_t = h_t.mean(dim=1)  # [B, H]
        # Combine
        h = torch.cat([h_f, h_t], dim=1)  # [B, 2H]
        mu_logvar = self.enc_fc(h)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * torch.exp(logvar * 0.5)
        h_dec = self.dec_fc(z)
        # Broadcast to window
        out = self.output(h_dec).unsqueeze(1).repeat(1, w, 1)  # [B, W, N]
        return out
