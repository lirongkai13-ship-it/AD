"""LSTM-AE: 双层 LSTM 自编码器 [Malhotra 2016]"""
import torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, n_vars=51, window=60, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.LSTM(n_vars, hidden, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(hidden, hidden, num_layers, batch_first=True, dropout=dropout)
        self.output = nn.Linear(hidden, n_vars)

    def forward(self, x):
        # x: [B, W, N] → encode
        _, (h_n, _) = self.encoder(x)
        # h_n: [L, B, H] → repeat as decoder input
        b, w, n = x.shape
        z = h_n[-1].unsqueeze(1).repeat(1, w, 1)  # [B, W, H]
        out, _ = self.decoder(z)
        recon = self.output(out)  # [B, W, N]
        return recon
