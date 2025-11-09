import gymnasium as gym

def make_env(env_id, max_episode_steps=None, seed=42):
    env = gym.make(env_id)
    env.reset(seed=seed)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
