import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Shared feature extractor
        self.base = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # Actor
        self.actor_mean = nn.Linear(128, action_dim)

        # Global log std (stable for PPO)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -1.0)

        # Critic
        self.critic = nn.Linear(128, 1)

    def get_value(self, state):
        return self.critic(self.base(state)).squeeze(-1)

    def get_action_and_value(self, state, action=None):
        latent = self.base(state)

        mean = self.actor_mean(latent)
        log_std = torch.clamp(self.actor_logstd, -3.0, 0.0)
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        if action is None:
            action = dist.rsample()

        # Clamp to env bounds (NO tanh)
        action_clamped = torch.clamp(action, -1.0, 1.0)

        log_prob = dist.log_prob(action_clamped).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(latent).squeeze(-1)

        return action_clamped, log_prob, entropy, value, action_clamped
