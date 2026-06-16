"""Anomaly Transformer [Xu 2022] - 简化版"""
import torch
import torch.nn as nn
import math

class AnomalyAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Prior sigma (learnable)
        self.sigma = nn.Parameter(torch.ones(n_heads, 1, 1))

    def forward(self, x):
        b, l, d = x.shape
        q = self.w_q(x).view(b, l, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L, D]
        k = self.w_k(x).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
        # Self-attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, L, L]
        # Prior association (Gaussian kernel over distance)
        dist = torch.arange(l, device=x.device).float()
        dist = (dist.unsqueeze(0) - dist.unsqueeze(1)).abs()  # [L, L]
        prior = torch.exp(-dist.unsqueeze(0).unsqueeze(0) / (2 * self.sigma ** 2 + 1e-8))
        attn = torch.softmax(attn, dim=-1)
        self.attn = attn  # 保存用于异常评分
        self.prior = prior
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, l, d)
        return self.w_o(out)

class AnomalyTransformer(nn.Module):
    def __init__(self, n_vars=51, window=60, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_vars, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        self.attn_layers = nn.ModuleList([
            AnomalyAttention(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, n_vars)

    def forward(self, x):
        b, w, n = x.shape
        h = self.input_proj(x) + self.pos_enc  # [B, W, D]
        for attn in self.attn_layers:
            h = h + attn(self.norm1(h))
            h = h + self.ffn(self.norm2(h))
        recon = self.output(h)  # [B, W, N]
        return recon
