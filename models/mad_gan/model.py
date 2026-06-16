"""MAD-GAN: Multivariate Anomaly Detection with GAN [Li 2019]"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, n_vars=51, noise_dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden*2), nn.ReLU(),
            nn.Linear(hidden*2, n_vars),
        )

    def forward(self, z):
        return self.net(z)  # [B, N]

class Discriminator(nn.Module):
    def __init__(self, n_vars=51, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_vars, hidden*2), nn.ReLU(),
            nn.Linear(hidden*2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)  # [B, 1]

class MADGAN(nn.Module):
    """使用 GAN 做重建: 噪声→生成→用生成值作为重构"""
    def __init__(self, n_vars=51, window=60, noise_dim=32, hidden=64):
        super().__init__()
        self.window, self.n_vars = window, n_vars
        self.gen = Generator(n_vars, noise_dim, hidden)
        self.disc = Discriminator(n_vars, hidden)

    def forward(self, x):
        b, w, n = x.shape
        # 对每个时间步独立生成重构值
        recon_list = []
        for t in range(w):
            z = torch.randn(b, 32, device=x.device)
            gen_out = self.gen(z)  # [B, N]
            recon_list.append(gen_out)
        recon = torch.stack(recon_list, dim=1)  # [B, W, N]
        return recon
