"""DCdetector [Yang 2023 KDD]: 双注意力对比学习"""
import torch, math
import torch.nn as nn
import torch.nn.functional as F


class PatchAttention(nn.Module):
    """Patch-level 注意力"""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, D]
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class DCdetector(nn.Module):
    """简化版 DCdetector: patch + dual attention + contrastive"""
    def __init__(self, n_vars=51, window=60, d_model=64, patch_len=10, n_heads=4, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        n_patches = window // patch_len
        self.input_proj = nn.Linear(patch_len * n_vars, d_model)

        self.perm_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.time_attn = PatchAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.output = nn.Linear(d_model, patch_len * n_vars)

    def forward(self, x):
        b, w, n = x.shape
        p = self.patch_len
        n_patches = w // p
        # Split into patches: [B, N_patches, P*N]
        x_patches = x[:, :n_patches * p, :].reshape(b, n_patches, p * n)
        h = self.input_proj(x_patches)  # [B, N_patches, D]

        # Permutation-invariant attention
        h_perm, _ = self.perm_attn(h, h, h)
        h = self.norm1(h + h_perm)

        # Temporal attention
        h = self.time_attn(h)

        recon = self.output(h)  # [B, N_patches, P*N]
        recon = recon.reshape(b, n_patches, p, n).reshape(b, n_patches * p, n)

        # 补齐到原始长度
        if recon.shape[1] < w:
            recon = F.pad(recon, (0, 0, 0, w - recon.shape[1]))
        return recon[:, :w, :]
