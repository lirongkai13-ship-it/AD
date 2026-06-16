"""TimesNet [Wu 2023 ICLR]: FFT 周期检测 + 2D Inception 卷积"""
import torch, math
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """2D Inception 多尺度卷积"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b3 = nn.Conv2d(in_ch, out_ch//4, 3, padding=1)
        self.b5 = nn.Conv2d(in_ch, out_ch//4, 5, padding=2)
        self.b7 = nn.Conv2d(in_ch, out_ch//4, 7, padding=3)
        self.b1 = nn.Conv2d(in_ch, out_ch//4, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(torch.cat([self.b3(x), self.b5(x), self.b7(x), self.b1(x)], dim=1))


class TimesBlock(nn.Module):
    def __init__(self, d_model, top_k=5, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlock(1, 32),
            InceptionBlock(32, 16),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        b, l, d = x.shape
        # FFT 找周期 — 必须用 float32，cuFFT 半精度要求 2 的幂
        x_float = x.float()
        l_fft = 2 ** int(math.ceil(math.log2(l)))
        pad_fft = l_fft - l
        x_padded = F.pad(x_float, (0, 0, 0, pad_fft)) if pad_fft > 0 else x_float
        xf = torch.fft.rfft(x_padded, dim=1)
        amp = xf.abs().mean(dim=-1)       # [B, L_fft//2+1]
        amp = amp[:, :l // 2 + 1]         # 截回原始长度对应频率
        top_k = min(self.top_k, amp.shape[1] - 2)
        _, top_idx = torch.topk(amp[:, 1:], top_k, dim=1)
        periods = l // (top_idx + 1).float()

        out_list = []
        for k in range(top_k):
            p = max(2, int(periods[0, k].item()))
            pad_len = (p - l % p) % p
            x_pad = F.pad(x, (0, 0, 0, pad_len)) if pad_len > 0 else x
            lp = x_pad.shape[1]
            n_rows = lp // p
            x_2d = x_pad.reshape(b, p, n_rows, d).permute(0, 3, 1, 2)  # [B, D, P, R]
            x_2d = x_2d.mean(dim=1, keepdim=True)  # [B, 1, P, R] 压缩通道
            x_2d = self.conv(x_2d)  # [B, 1, P, R]
            x_rec = x_2d.squeeze(1).transpose(1, 2).contiguous()  # [B, R, P]
            x_rec = x_rec.unsqueeze(-1).expand(-1, -1, -1, d).permute(0, 2, 1, 3)
            x_rec = x_rec.reshape(b, lp, d)[:, :l, :]
            out_list.append(x_rec)

        if out_list:
            out = sum(out_list) / len(out_list)
        else:
            out = torch.zeros_like(x)
        return self.norm(x + out)


class TimesNet(nn.Module):
    """TimesNet: TimesBlock × 3 + 输入/输出投影"""
    def __init__(self, n_vars=51, window=60, d_model=128, n_blocks=3, top_k=5, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_vars, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(d_model, top_k, dropout) for _ in range(n_blocks)
        ])
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, n_vars),
        )

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output(h)
