import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor (The "Base")
        # Forces the Actor and Critic to learn the underlying physics together
        self.base = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        
        # Actor head: Outputs the mean of the Gaussian distribution
        self.actor_mean = nn.Linear(256, action_dim)
        
        # Standard Deviation Parameter: Learned, but clamped later for stability
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head: Outputs the "State Value" (V)
        self.critic = nn.Linear(256, 1)

    def get_value(self, state):
        return self.critic(self.base(state))

    def get_action_and_value(self, state, action=None):
        latent = self.base(state)
        mean = self.actor_mean(latent)
        
        # Clamp log_std to prevent "Standard Deviation Collapse"
        # Prevents the agent from becoming too certain (no exploration)
        log_std = torch.clamp(self.actor_logstd, -2, 1)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            raw_action = dist.sample()
        else:
            raw_action = action 

        squashed_action = torch.tanh(raw_action)
        
        # Log-prob calculation with Tanh correction for squashed actions
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6).sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(latent).squeeze(-1)

        return squashed_action, log_prob, entropy, value, raw_action