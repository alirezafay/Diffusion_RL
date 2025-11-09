import hydra, torch, numpy as np, os
from omegaconf import OmegaConf
from envs.gym_loader import make_env
from envs.dataset_loader import load_dims
from diffuser.trainer import DiffusionModel, build_condition
from offline_rl.iql import IQLAgent
from tqdm import tqdm

def load_diffusion(path, device):
    ckpt = torch.load(path, map_location=device)
    model = DiffusionModel(seq_channels=ckpt['action_dim'], horizon=ckpt['horizon'])
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model, ckpt['horizon'], ckpt['action_dim']

def load_iql(path, device):
    ckpt = torch.load(path, map_location=device)
    from types import SimpleNamespace
    cfg = {"lr":3e-4, "gamma":0.99, "expectile":0.7, "temperature":3.0, "hidden_dim":256}
    agent = IQLAgent(ckpt['sdim'], ckpt['adim'], cfg, device)
    agent.critic.load_state_dict(ckpt['critic'])
    agent.value.load_state_dict(ckpt['value'])
    agent.actor.load_state_dict(ckpt['actor'])
    return agent

@hydra.main(config_path=None, config_name=None)
def main(cfg):
    if "config" in cfg:
        import yaml
        with open(cfg.config, "r") as f:
            base = yaml.safe_load(f)
        for k,v in base.items():
            if k not in cfg:
                cfg[k]=v
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu")

    env = make_env(cfg.env.id, cfg.env.max_episode_steps, seed=cfg.seed)

    # Infer dims from dataset and checkpoints
    sdim, adim = load_dims(cfg.dataset.path)
    diff, H, A = load_diffusion("checkpoints/diffusion.pt", device)
    agent = load_iql("checkpoints/iql.pt", device)

    returns = []
    for ep in range(cfg.mpc.eval_episodes):
        s,_ = env.reset(seed=cfg.seed + ep)
        done = False
        ep_ret = 0.0
        t = 0
        while not done:
            s0 = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)  # [1, sdim]
            cond = s0  # only needs s0 for cond builder
            B = cfg.mpc.samples
            cond_batch = cond.repeat(B, 1)  # [B, sdim]

            def guidance_grad(x0_pred):
                # x0_pred: [B, A, H] actions sequence
                # Build batch of states repeated across horizon, use only first step for Q
                a0 = x0_pred[:,:,0].T.permute(1,0)  # wrong shape guard
                # Simplify: evaluate Q on first action only
                a0 = x0_pred[:,:,0].permute(0,2,1) if x0_pred.dim()==4 else x0_pred[:,:,0]
                # Actually x0_pred is [B, A, H]; we need [B, A] for a0
                a0 = x0_pred[:,:,0]  # [B, A]
                a0.requires_grad_(True)
                s_rep = cond_batch  # [B, sdim]
                q = agent.q_value(s_rep, a0).sum()
                g = torch.autograd.grad(q, a0, retain_graph=False)[0]  # [B, A]
                # Expand grad across horizon, only guide step 0
                g_expand = torch.zeros_like(x0_pred)
                g_expand[:,:,0] = g
                return g_expand

            with torch.no_grad():
                pass
            # We need gradients inside sampling; wrap in closure called by sampler (no grad guard)

            # Run sampling with guidance
            x = diff.sample(B, cond_batch, device, guidance_grad=guidance_grad, guidance_scale=cfg.mpc.guidance_scale)
            # Select best by Q
            a0 = x[:,:,0]  # [B, A]
            with torch.no_grad():
                q = agent.q_value(cond_batch, a0).squeeze(-1)
                best = torch.argmax(q).item()
                action = a0[best].cpu().numpy()

            # Add small exploration noise
            action = np.clip(action + cfg.mpc.action_noise * np.random.randn(*action.shape), -1.0, 1.0)

            s, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += r
            t += 1

        returns.append(ep_ret)
        print(f"Episode {ep+1}: return={ep_ret:.2f}")

    print(f"Avg return over {len(returns)} episodes: {np.mean(returns):.2f} ± {np.std(returns):.2f}")

if __name__ == "__main__":
    main()
