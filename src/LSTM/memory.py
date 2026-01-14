import numpy as np

import torch


class LSTMRolloutBuffer:

    def __init__(self):

        self.states, self.actions, self.logprobs = [], [], []

        self.rewards, self.is_terminals, self.values = [], [], []

        self.h_states, self.c_states = [], [] # Two states for LSTM


    def add(self, s, a, lp, r, d, v, h, c):

        self.states.append(s); self.actions.append(a); self.logprobs.append(lp)

        self.rewards.append(r); self.is_terminals.append(d); self.values.append(v)

        self.h_states.append(h); self.c_states.append(c)


    def clear(self):

        self.states.clear(); self.actions.clear(); self.logprobs.clear()

        self.rewards.clear(); self.is_terminals.clear(); self.values.clear()

        self.h_states.clear(); self.c_states.clear()


    def get_tensors(self, device):

        return (

            torch.FloatTensor(np.array(self.states)).to(device),

            torch.FloatTensor(np.array(self.actions)).to(device),

            torch.FloatTensor(np.array(self.logprobs)).to(device),

            torch.FloatTensor(np.array(self.rewards)).to(device),

            torch.FloatTensor(np.array(self.is_terminals)).to(device),

            torch.FloatTensor(np.array(self.values)).to(device),

            torch.FloatTensor(np.array(self.h_states)).to(device),

            torch.FloatTensor(np.array(self.c_states)).to(device)

        ) 