import torch

import torch.nn as nn


class ActorCriticLSTM(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(ActorCriticLSTM, self).__init__()

       

        self.fc1 = nn.Linear(state_dim, 128)

       

        # LSTM Layer

        # Returns (output, (h_n, c_n))

        self.lstm = nn.LSTM(128, 128, batch_first=True)

       

        self.actor_mean = nn.Linear(128, action_dim)

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Linear(128, 1)


    def get_value(self, state, lstm_states):

        # lstm_states is a tuple: (h_t, c_t)

        x = torch.tanh(self.fc1(state))

        x, next_lstm_states = self.lstm(x, lstm_states)

        return self.critic(x).squeeze(-1), next_lstm_states


    def get_action_and_value(self, state, lstm_states, action=None):

        if state.dim() == 2:

            state = state.unsqueeze(1) # Add sequence dimension

           

        x = torch.tanh(self.fc1(state))

       

        # LSTM returns output sequence and a tuple of (h_n, c_n)

        x, next_lstm_states = self.lstm(x, lstm_states)

       

        latent = x[:, -1, :]

       

        mean = self.actor_mean(latent)

        log_std = torch.clamp(self.actor_logstd, -2, 1)

        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)

       

        if action is None:

            raw_action = dist.sample()

        else:

            raw_action = action


        squashed_action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6).sum(dim=-1)

       

        entropy = dist.entropy().sum(dim=-1)

        value = self.critic(latent).squeeze(-1)


        return squashed_action, log_prob, entropy, value, raw_action, next_lstm_states 