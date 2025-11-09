import torch
import math

class CosineScheduler:
    def __init__(self, T):
        self.T = T
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.alphas_cumprod = alphas_cumprod.float()  # [T+1]
        self.alphas = self.alphas_cumprod[1:] / self.alphas_cumprod[:-1]
        self.betas = 1 - self.alphas

    def q_sample(self, x0, t, noise):
        # x_t = sqrt(a_bar_t) * x0 + sqrt(1 - a_bar_t) * noise
        a_bar = self.alphas_cumprod[t]
        while len(a_bar.shape) < len(x0.shape):
            a_bar = a_bar.unsqueeze(-1)
        return (a_bar.sqrt() * x0) + ((1 - a_bar).sqrt() * noise)

    def posterior_mean_variance(self, x0_pred, x_t, t):
        # DDPM posterior for p(x_{t-1} | x_t, x0)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        a_bar_t = self.alphas_cumprod[t]
        a_bar_tm1 = self.alphas_cumprod[t-1] if (t>0).all() else torch.ones_like(a_bar_t)
        while len(beta_t.shape) < len(x_t.shape):
            beta_t = beta_t.unsqueeze(-1)
            alpha_t = alpha_t.unsqueeze(-1)
            a_bar_t = a_bar_t.unsqueeze(-1)
            a_bar_tm1 = a_bar_tm1.unsqueeze(-1)
        mean = (torch.sqrt(a_bar_tm1) * beta_t / (1 - a_bar_t)) * x0_pred + (torch.sqrt(alpha_t) * (1 - a_bar_tm1) / (1 - a_bar_t)) * x_t
        var = ((1 - a_bar_tm1) / (1 - a_bar_t)) * beta_t
        return mean, var
