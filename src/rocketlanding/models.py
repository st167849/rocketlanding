import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # --- The Critic (The Value Network) ---
        # It estimates: "How good is the current state?" (Score: -infinity to +infinity)
        # We use Tanh activations because they are centered around 0, 
        # which often helps with physics-based stability.
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Outputs a single "Value" score
        )

        # --- The Actor (The Policy Network) ---
        # It decides: "What should I do?"
        # 1. Action Mean: The target action (e.g., 0.5 throttle)
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Output range [-1, 1] to match Gymnasium's input format
        )
        
        # 2. Action Std: The exploration noise (Learning confidence)
        # We learn this as a parameter. Initially high (lots of random exploration),
        # then it shrinks as the agent gets confident.
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self):
        raise NotImplementedError("Use get_action or get_value instead.")

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
    # Actor: mean of Gaussian
        mean = self.actor_mean(state)

    # Log std is state-independent (standard PPO)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)

        if action is None:
        # Reparameterization trick for better gradients
            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
        else:
        # During training, the action is already squashed
        # We need to invert tanh to recover raw_action
            eps = 1e-6
            clipped_action = torch.clamp(action, -1 + eps, 1 - eps)
            raw_action = 0.5 * torch.log((1 + clipped_action) / (1 - clipped_action))

    # Log-probability with tanh correction
        log_prob = dist.log_prob(raw_action)
            
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

    # Entropy (use unsquashed entropy — standard PPO practice)
        entropy = dist.entropy().sum(dim=-1)

    # Critic value
        value = self.critic(state).squeeze(-1)

        return action, log_prob, entropy, value
