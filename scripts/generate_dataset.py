import hydra, os, numpy as np, torch
from omegaconf import OmegaConf
from envs.gym_loader import make_env
from envs.dataset_loader import save_npz

@hydra.main(config_path=None, config_name=None)
def main(cfg):
    # Read from CLI: env=HalfCheetah-v4 episodes=50 out=data/file.npz
    env_id = cfg.get("env", "HalfCheetah-v4")
    episodes = int(cfg.get("episodes", 50))
    out = cfg.get("out", f"data/{env_id}_random.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env = make_env(env_id)
    rng = np.random.default_rng(seed=cfg.get("seed", 42))

    eps = []
    for _ in range(episodes):
        s,_ = env.reset()
        done = False
        ep = {"states":[s], "actions":[], "rewards":[], "next_states":[], "dones":[]}
        while not done:
            # random policy
            a = env.action_space.sample()
            s2, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            ep["actions"].append(a)
            ep["rewards"].append(r)
            ep["next_states"].append(s2)
            ep["dones"].append(float(done))
            ep["states"].append(s2)
            s = s2
        for k in ep:
            ep[k] = np.array(ep[k], dtype=np.float32)
        eps.append(ep)

    save_npz(out, eps)
    print(f"Saved dataset to {out} with {len(eps)} episodes.")

if __name__ == "__main__":
    main()
