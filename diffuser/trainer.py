import torch, os
from torch.utils.data import DataLoader
from tqdm import tqdm
from .unet1d import UNet1D
from .scheduler import CosineScheduler
from utils.ema import EMA
from torch.utils.tensorboard import SummaryWriter

def build_condition(states0, cond_dim):
    # Simple linear projection into cond space
    B = states0.shape[0]
    flat = states0.reshape(B, -1)
    # layer-free normalization
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True) + 1e-6
    normed = (flat - mean) / std
    # pad/truncate to cond_dim
    if normed.shape[1] >= cond_dim:
        return normed[:, :cond_dim]
    else:
        pad = torch.zeros(B, cond_dim - normed.shape[1], device=flat.device)
        return torch.cat([normed, pad], dim=1)

class DiffusionModel(torch.nn.Module):
    def __init__(self, seq_channels, horizon, model_dim=256, n_layers=4, T=1000, cond_dim=128, dropout=0.0):
        super().__init__()
        self.horizon = horizon
        self.T = T
        self.scheduler = CosineScheduler(T=T)
        self.unet = UNet1D(seq_dim=seq_channels, model_dim=model_dim, n_layers=n_layers, cond_dim=cond_dim, dropout=dropout)
        self.cond_dim = cond_dim

    def loss(self, x0, states0, t):
        noise = torch.randn_like(x0)
        xt = self.scheduler.q_sample(x0, t, noise)
        cond = build_condition(states0, self.cond_dim)
        eps_hat = self.unet(xt, t.float(), cond)
        return torch.nn.functional.mse_loss(eps_hat, noise)

    @torch.no_grad()
    def sample(self, B, cond_states0, device, guidance_grad=None, guidance_scale=0.0):
        cond = build_condition(cond_states0, self.cond_dim)
        C = self.unet.in_proj.in_channels
        H = self.horizon
        x = torch.randn(B, C, H, device=device)
        for t in range(self.T-1, -1, -1):
            tt = torch.full((B,), t, device=device, dtype=torch.long)
            eps = self.unet(x, tt.float(), cond)
            # Predict x0
            a_bar = self.scheduler.alphas_cumprod[tt]
            while len(a_bar.shape) < len(x.shape):
                a_bar = a_bar.unsqueeze(-1)
            x0_pred = (x - (1 - a_bar).sqrt() * eps) / a_bar.sqrt().clamp(min=1e-6)

            # Guidance on action channels (assume last dims correspond to actions)
            if guidance_grad is not None and guidance_scale > 0:
                g = guidance_grad(x0_pred)  # same shape as x
                x0_pred = x0_pred + guidance_scale * g

            if t > 0:
                mean, var = self.scheduler.posterior_mean_variance(x0_pred, x, tt)
                x = mean + torch.randn_like(x) * var.sqrt()
            else:
                x = x0_pred
        return x

def train_diffusion(dataset, state_dim, action_dim, horizon, cfg, device, logdir="runs/diffusion"):
    model = DiffusionModel(seq_channels=action_dim, horizon=horizon, model_dim=cfg["model_dim"], n_layers=cfg["n_layers"], T=cfg["T"])
    model.to(device)
    ema = EMA(model.unet, decay=cfg.get("ema", 0.999))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    dl = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, drop_last=True)

    writer = SummaryWriter(logdir)
    step = 0
    model.train()
    for _ in tqdm(range(cfg["train_steps"]), desc="Diffusion"):
        for states, actions in dl:
            states = states.to(device)  # [B, H+1, sdim]
            actions = actions.to(device)  # [B, H, adim]
            B, H, A = actions.shape
            # Arrange actions to [B, C, H]
            x0 = actions.permute(0,2,1).contiguous()
            t = torch.randint(0, model.T, (B,), device=device, dtype=torch.long)
            loss = model.loss(x0, states[:,0], t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update()
            if step % 100 == 0:
                writer.add_scalar("loss", loss.item(), step)
            step += 1
            if step >= cfg["train_steps"]:
                break

    # Save checkpoint (with EMA weights applied for inference)
    ema.apply_to(model.unet)
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/diffusion.pt"
    torch.save({
        "model": model.state_dict(),
        "state_dim": state_dim, "action_dim": action_dim, "horizon": horizon
    }, ckpt_path)
    return ckpt_path
