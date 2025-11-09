# Diffuser-RL: Q-Guided Diffusion for Offline-to-Online Control

This repository implements a minimal, end-to-end project:
1) Train a **trajectory diffusion model** on offline data.
2) Train an **offline RL critic** (IQL).
3) **Plan** with **Q-guided diffusion sampling** in an MPC loop.
4) (Optional) do short **online refinement**.

It runs on **Gymnasium + MuJoCo** (HalfCheetah-v4, Hopper-v4, Walker2d-v4).
If you want D4RL datasets, enable the optional dependency in `requirements.txt` and point configs to D4RL loaders.

> ⚠️ D4RL depends on the legacy `gym`. The default codepath here uses **Gymnasium** to avoid conflicts.

## Quickstart

```bash
# Python >=3.10 is recommended
python -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

# 1) Make a small offline dataset by rolling a random policy
python scripts/generate_dataset.py env=HalfCheetah-v4 episodes=50 out=data/halfcheetah_random.npz

# 2) Train diffusion model on offline trajectories
python scripts/train_diffusion.py config=configs/halfcheetah_random.yaml

# 3) Train IQL critic on the same dataset
python scripts/train_q.py config=configs/halfcheetah_random.yaml

# 4) Evaluate MPC with Q-guided diffusion
python scripts/evaluate_mpc.py config=configs/halfcheetah_random.yaml
```

TensorBoard logs are written to `runs/`.

## Configs

Edit the YAMLs in `configs/`. Example (`halfcheetah_random.yaml`) controls horizon, diffusion steps, guidance scale, etc.

## Notes

- This is a **clean, minimal** educational codebase. It trades some performance for simplicity.
- For **D4RL**, install it manually and modify `envs/dataset_loader.py` to point to `d4rl` datasets.
- The MPC planner uses a short horizon (H≈10–16) and Q-guidance to bias diffusion samples.

## Citation-like pointers

- Diffuser-style control: trajectory diffusion + planning.
- Q-guidance akin to energy/classifier-free guidance.
- Offline critic: IQL (Implicit Q-Learning).
