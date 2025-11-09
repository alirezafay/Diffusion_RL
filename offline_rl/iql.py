import torch, torch.nn as nn, torch.nn.functional as F
from utils.mlp import MLP

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(state_dim+action_dim, 1, hidden=hidden, layers=2)
        self.q2 = MLP(state_dim+action_dim, 1, hidden=hidden, layers=2)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.v = MLP(state_dim, 1, hidden=hidden, layers=2)

    def forward(self, s):
        return self.v(s)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = MLP(state_dim, hidden, hidden=hidden, layers=2)
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, s):
        h = self.net(s)
        mu = torch.tanh(self.mu(h))
        std = torch.exp(self.log_std).clamp(1e-3, 1.0)
        return mu, std

class IQLAgent:
    def __init__(self, state_dim, action_dim, cfg, device):
        self.gamma = cfg.get("gamma", 0.99)
        self.expectile = cfg.get("expectile", 0.7)
        self.temperature = cfg.get("temperature", 3.0)
        hidden = cfg.get("hidden_dim", 256)
        self.device = device

        self.critic = Critic(state_dim, action_dim, hidden).to(device)
        self.value = ValueNet(state_dim, hidden).to(device)
        self.actor = Actor(state_dim, action_dim, hidden).to(device)

        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr"])
        self.opt_v = torch.optim.Adam(self.value.parameters(), lr=cfg["lr"])
        self.opt_a = torch.optim.Adam(list(self.actor.parameters()), lr=cfg["lr"])

    @staticmethod
    def expectile_loss(diff, expectile):
        w = torch.where(diff > 0, expectile, 1 - expectile)
        return (w * (diff ** 2)).mean()

    def iql_step(self, batch):
        s, a, r, s2, d = [x.to(self.device) for x in batch]
        with torch.no_grad():
            q1_t, q2_t = self.critic(s2, self.actor(s2)[0])
            target_q = torch.min(q1_t, q2_t)
            target_v = r + (1 - d) * self.gamma * target_q

        # V update
        v = self.value(s)
        diff = (target_v - v)
        v_loss = self.expectile_loss(diff, self.expectile)

        self.opt_v.zero_grad(); v_loss.backward(); self.opt_v.step()

        # Critic update (TD on Q towards r + γ V(s'))
        with torch.no_grad():
            v2 = self.value(s2)
            q_target = r + (1 - d) * self.gamma * v2
        q1, q2 = self.critic(s, a)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_c.zero_grad(); q_loss.backward(); self.opt_c.step()

        # Advantage-weighted behavior cloning for actor
        with torch.no_grad():
            q1_pi, q2_pi = self.critic(s, a)
            v_s = self.value(s)
            adv = torch.min(q1_pi, q2_pi) - v_s
            weights = torch.exp(adv * self.temperature).clamp(max=100.0)

        mu, std = self.actor(s)
        logp = -0.5 * (((a - mu) / std) ** 2 + 2 * torch.log(std)).sum(-1, keepdim=True)
        act_loss = -(weights * logp).mean()

        self.opt_a.zero_grad(); act_loss.backward(); self.opt_a.step()

        return {
            "q_loss": q_loss.item(),
            "v_loss": v_loss.item(),
            "actor_loss": act_loss.item()
        }

    def q_value(self, s, a):
        q1, q2 = self.critic(s, a)
        return torch.min(q1, q2)

    def grad_wrt_action(self, s, a):
        a.requires_grad_(True)
        q = self.q_value(s, a).sum()
        g = torch.autograd.grad(q, a, retain_graph=False)[0]
        return g
