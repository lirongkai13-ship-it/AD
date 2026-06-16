"""DAGMM: Deep Autoencoding Gaussian Mixture Model [Zong 2018]"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DAGMM(nn.Module):
    def __init__(self, n_vars=51, window=60, hidden=32, latent=8, n_gmm=4, dropout=0.1):
        super().__init__()
        self.window, self.n_vars, self.latent, self.n_gmm = window, n_vars, latent, n_gmm
        feat_dim = window * n_vars
        # Autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden*2), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden*2, hidden), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden*2), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden*2, feat_dim),
        )
        # Estimation network: [latent + 2] → soft assignment
        est_in = latent + 2  # latent + cosine_sim + euclidean
        self.estimation = nn.Sequential(
            nn.Linear(est_in, 16), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(16, n_gmm), nn.Softmax(dim=1),
        )

    def forward(self, x):
        b, w, n = x.shape
        x_flat = x.reshape(b, -1)               # [B, W*N]
        z = self.encoder(x_flat)                 # [B, latent]
        recon_flat = self.decoder(z)              # [B, W*N]
        recon = recon_flat.reshape(b, w, n)       # [B, W, N]
        return recon
