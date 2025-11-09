import numpy as np

def save_npz(path, episodes):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for ep in episodes:
        s = np.array(ep['states'], dtype=np.float32)
        a = np.array(ep['actions'], dtype=np.float32)
        r = np.array(ep['rewards'], dtype=np.float32)
        s2 = np.array(ep['next_states'], dtype=np.float32)
        d = np.array(ep['dones'], dtype=np.float32)
        states.append(s); actions.append(a); rewards.append(r); next_states.append(s2); dones.append(d)
    np.savez_compressed(path, states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)

def load_dims(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    sdim = data['states'][0].shape[-1]
    adim = data['actions'][0].shape[-1]
    return sdim, adim
