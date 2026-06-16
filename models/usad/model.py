"""USAD: UnSupervised Anomaly Detection [Audibert 2020]"""
import torch.nn as nn
import torch

class USAD(nn.Module):
    def __init__(self, n_vars=51, window=60, hidden=64, latent=32, dropout=0.1):
        super().__init__()
        feat = window * n_vars
        self.encoder = nn.Sequential(
            nn.Linear(feat, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, latent),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, feat),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, feat),
        )

    def forward(self, x):
        b, w, n = x.shape
        x_flat = x.reshape(b, -1)
        z = self.encoder(x_flat)
        recon1 = self.decoder1(z).reshape(b, w, n)
        recon2 = self.decoder2(z).reshape(b, w, n)
        return recon1, recon2  # 训练时用两个输出做对抗
