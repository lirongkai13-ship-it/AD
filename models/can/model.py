"""CAN/MST-GAT [Causal Attention + GAT for Fault Diagnosis]"""
import torch, math
import torch.nn as nn
import torch.nn.functional as F


class CausalAttentionBlock(nn.Module):
    """因果自注意力：当前时刻只能看到过去"""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D], causal mask
        l = x.shape[1]
        causal_mask = torch.triu(torch.ones(l, l, device=x.device) * float('-inf'), diagonal=1)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        return self.norm(x + self.dropout(attn_out))


class CAN(nn.Module):
    """Causal Attention Network: CNN + Causal Attention + GAT"""
    def __init__(self, n_vars=51, window=60, d_model=64, n_heads=4, dropout=0.1):
        super().__init__()

        # Temporal CNN
        self.cnn = nn.Conv1d(n_vars, d_model, kernel_size=3, padding=1)

        # Causal self-attention
        self.causal_attn = CausalAttentionBlock(d_model, n_heads, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Output projection
        self.output = nn.Linear(d_model, n_vars)

    def forward(self, x):
        # x: [B, W, N]
        b, w, n = x.shape

        # CNN along time
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # [B, W, D]

        # Causal self-attention (沿时间维，只看过去)
        h = self.causal_attn(h)

        # Cross attention with global context
        h_cross, _ = self.cross_attn(h, h, h)
        h = self.norm(h + h_cross)

        # Feed-forward
        h = self.norm2(h + self.ffn(h))

        recon = self.output(h)  # [B, W, N]
        return recon
