import hydra, torch, os
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from utils.replay import TransitionReplay
from envs.dataset_loader import load_dims
from offline_rl.iql import IQLAgent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
    sdim, adim = load_dims(cfg.dataset.path)
    ds = TransitionReplay(cfg.dataset.path)
    dl = DataLoader(ds, batch_size=cfg.critic.batch_size, shuffle=True, drop_last=True, num_workers=2)

    agent = IQLAgent(sdim, adim, cfg.critic, device)
    writer = SummaryWriter("runs/iql")

    step = 0
    for _ in tqdm(range(cfg.critic.train_steps), desc="IQL"):
        for batch in dl:
            logs = agent.iql_step(batch)
            if step % 100 == 0:
                for k,v in logs.items():
                    writer.add_scalar(k, v, step)
            step += 1
            if step >= cfg.critic.train_steps: break

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"critic": agent.critic.state_dict(), "value": agent.value.state_dict(), "actor": agent.actor.state_dict(),
                "sdim": sdim, "adim": adim}, "checkpoints/iql.pt")
    print("Saved IQL checkpoint to checkpoints/iql.pt")

if __name__ == "__main__":
    main()
