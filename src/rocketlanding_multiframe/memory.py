import numpy as np
import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
    def add(self, state, action, logprob, reward, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def get_tensors(self, device):
        # Convert lists to PyTorch Tensors
        return (
            torch.tensor(np.array(self.states), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.actions), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.logprobs), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.rewards), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.is_terminals), dtype=torch.float32).to(device)
        )