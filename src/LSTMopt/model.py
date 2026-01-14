import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SymmetricActorCriticLSTM(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_size=256):
        super().__init__()
        
        # 1. Feature Extractor (Blind Input -> Embedding)
        self.embedding = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_size)),
            nn.Tanh()
        )
        
        # 2. Shared LSTM Memory
        # The LSTM must do the heavy lifting of inferring velocity
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, batch_first=False)
        
        # 3. Actor Head
        self.actor_head = layer_init(nn.Linear(hidden_size // 2, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # 4. Critic Head
        self.critic_head = layer_init(nn.Linear(hidden_size // 2, 1), std=1)

    def get_states(self, x, lstm_state, done):
        # x shape: [Seq_Len, Batch_Size, Input_Dim]
        seq_len, batch_size, _ = x.shape
        
        # Embed inputs
        x_emb = self.embedding(x.reshape(-1, x.shape[-1]))
        x_emb = x_emb.view(seq_len, batch_size, -1)
        
        # Pass through LSTM
        # We don't mask the state here manually; the training loop handles the logic
        # by resetting states when 'done' is True.
        hidden_out, (h_n, c_n) = self.lstm(x_emb, lstm_state)
        
        return hidden_out, (h_n, c_n)

    def get_action_and_value(self, x, lstm_state, done=None, action=None):
        # 1. Get Latent Physics State from LSTM
        hidden_out, next_lstm_state = self.get_states(x, lstm_state, done)
        
        # 2. Compute Policy and Value
        # We use the output at every step (for training) or just the last one (inference)
        mean = self.actor_head(hidden_out)
        std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic_head(hidden_out).squeeze(-1)
        
        return action, log_prob, entropy, value, next_lstm_state

    def save_model(self, path):
        torch.save(self.state_dict(), path)