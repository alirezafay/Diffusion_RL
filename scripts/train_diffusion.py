import hydra, torch, os
from omegaconf import OmegaConf
from utils.replay import TrajectoryDataset
from envs.dataset_loader import load_dims
from diffuser.trainer import train_diffusion

@hydra.main(config_path=None, config_name=None)
def main(cfg):
    # cfg.config points to yaml
    if "config" in cfg:
        import yaml
        with open(cfg.config, "r") as f:
            base = yaml.safe_load(f)
        for k,v in base.items():
            if k not in cfg:
                cfg[k]=v

    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu")
    sdim, adim = load_dims(cfg.dataset.path)
    ds = TrajectoryDataset(cfg.dataset.path, horizon=cfg.dataset.horizon)
    ckpt = train_diffusion(ds, sdim, adim, cfg.dataset.horizon, cfg.diffusion, device, logdir="runs/diffusion")
    print("Saved diffusion checkpoint:", ckpt)

if __name__ == "__main__":
    main()
