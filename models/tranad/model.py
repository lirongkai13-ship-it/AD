"""TranAD [Tuli 2022 VLDB]: Transformer + 对抗训练 双解码器"""
import torch, math
import torch.nn as nn

class TranAD(nn.Module):
    def __init__(self, n_vars=51, window=60, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_vars, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        # 共享 Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4,
                                                dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        # 两个解码器（结构相同）
        dec_layer1 = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4,
                                                 dropout=dropout, batch_first=True)
        self.decoder1 = nn.TransformerEncoder(dec_layer1, n_layers)
        dec_layer2 = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4,
                                                 dropout=dropout, batch_first=True)
        self.decoder2 = nn.TransformerEncoder(dec_layer2, n_layers)
        self.output = nn.Linear(d_model, n_vars)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        b, w, n = x.shape
        h = self.norm(self.input_proj(x) + self.pos_enc)  # [B, W, D]
        z = self.encoder(h)
        d1 = self.decoder1(z)
        d2 = self.decoder2(z)
        r1 = self.output(d1)  # [B, W, N]
        r2 = self.output(d2)
        return r1, r2  # 评估时取平均
