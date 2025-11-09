import numpy as np
import torch

class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, horizon):
        data = np.load(npz_path, allow_pickle=True)
        self.states = data['states']  # list of arrays (episodes) or flat?
        self.actions = data['actions']
        # flatten into windows
        S = []
        A = []
        H = horizon
        for s_ep, a_ep in zip(self.states, self.actions):
            L = min(len(a_ep), len(s_ep)-1)
            for t in range(0, L - H):
                S.append(s_ep[t:t+H+1])  # H+1 states to define H transitions
                A.append(a_ep[t:t+H])
        self.S = np.array(S, dtype=np.float32)  # [N, H+1, sdim]
        self.A = np.array(A, dtype=np.float32)  # [N, H, adim]

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        return self.S[idx], self.A[idx]

class TransitionReplay(torch.utils.data.Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        states = np.concatenate(data['states'], axis=0)
        actions = np.concatenate(data['actions'], axis=0)
        rewards = np.concatenate(data['rewards'], axis=0)
        next_states = np.concatenate(data['next_states'], axis=0)
        dones = np.concatenate(data['dones'], axis=0)
        self.s = states.astype(np.float32)
        self.a = actions.astype(np.float32)
        self.r = rewards.astype(np.float32).reshape(-1,1)
        self.s2 = next_states.astype(np.float32)
        self.d = dones.astype(np.float32).reshape(-1,1)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]
