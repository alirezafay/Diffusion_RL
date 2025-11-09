import torch
import torch.nn as nn
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0,1))
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv1d(dim, dim, 3, padding=1)
        )
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim))
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv1d(dim, dim, 3, padding=1)
        )

    def forward(self, x, cond):
        h = self.block1(x)
        c = self.cond_proj(cond)[:, :, None]
        h = h + c
        h = self.block2(h)
        return x + h

class UNet1D(nn.Module):
    def __init__(self, seq_dim, model_dim=256, n_layers=4, cond_dim=128, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Conv1d(seq_dim, model_dim, 1)
        self.time_emb = nn.Sequential(SinusoidalPosEmb(cond_dim), nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.cond_proj = nn.Linear(cond_dim, cond_dim)
        self.layers = nn.ModuleList([ResidualBlock(model_dim, cond_dim) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.GroupNorm(8, model_dim), nn.SiLU(), nn.Conv1d(model_dim, seq_dim, 1))

    def forward(self, x, t, cond_vec):
        # x: [B, C(seq_dim), H], cond_vec: [B, cond_dim]
        h = self.in_proj(x)
        temb = self.time_emb(t)
        cond = self.cond_proj(cond_vec + temb)
        for layer in self.layers:
            h = layer(h, cond)
        return self.out(h)
