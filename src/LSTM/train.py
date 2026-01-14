import torch
import torch.optim as optim
import gymnasium as gym
import numpy as np
from torch.optim.lr_scheduler import LinearLR
import torch.nn as nn 

# Import your custom modules
from LSTM.model import ActorCriticLSTM
from LSTM.memory import LSTMRolloutBuffer

def train():
    # Observation Indices: 0:x, 1:y, 4:angle, 6:legL, 7:legR
    MASK_INDICES = [0, 1, 4, 6, 7]
    ENV_NAME = "LunarLanderContinuous-v3"
    total_timesteps = 2_000_000 # Increased for state inference complexity
    max_steps = 4096
    batch_size = 256
    K_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    eps_clip = 0.2
    entropy_coef = 0.005

    env = gym.make(ENV_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    policy = ActorCriticLSTM(len(MASK_INDICES), env.action_space.shape[0]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    buffer = LSTMRolloutBuffer()

    current_step = 0
    while current_step < total_timesteps:
        state, _ = env.reset()
        # Initialize LSTM hidden and cell states
        h_state = torch.zeros(1, 1, 128).to(device)
        c_state = torch.zeros(1, 1, 128).to(device)
        
        # --- 1. Collection ---
        for _ in range(max_steps):
            current_step += 1
            masked_state = state[MASK_INDICES]
            st_tensor = torch.FloatTensor(masked_state).to(device).view(1, 1, -1)
            
            with torch.no_grad():
                # We save the states that were used to generate the action
                cur_h = h_state.cpu().numpy().squeeze(0)
                cur_c = c_state.cpu().numpy().squeeze(0)
                
                action, lp, _, val, raw_act, (h_state, c_state) = \
                    policy.get_action_and_value(st_tensor, (h_state, c_state))

            next_state, reward, term, trunc, _ = env.step(action.cpu().numpy().flatten())
            
            buffer.add(masked_state, raw_act.cpu().numpy().flatten(), 
                       lp.cpu().item(), reward, term, val.cpu().item(), cur_h, cur_c)
            
            state = next_state
            if term or trunc:
                state, _ = env.reset()
                h_state = torch.zeros(1, 1, 128).to(device)
                c_state = torch.zeros(1, 1, 128).to(device)

        # --- 2. GAE & Advantages ---
        s_ts, a_ts, lp_ts, r_ts, d_ts, v_ts, h_ts, c_ts = buffer.get_tensors(device)
        
        returns, advantages = [], []
        gae = 0
        with torch.no_grad():
            last_masked = state[MASK_INDICES]
            next_val, _ = policy.get_value(torch.FloatTensor(last_masked).to(device).view(1,1,-1), 
                                           (h_state, c_state))
            next_val = next_val.item()
        
        for i in reversed(range(len(r_ts))):
            mask = 1.0 - d_ts[i]
            delta = r_ts[i] + gamma * next_val * mask - v_ts[i]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_val = v_ts[i]
            returns.insert(0, gae + v_ts[i])

        adv_ts = torch.FloatTensor(advantages).to(device)
        ret_ts = torch.FloatTensor(returns).to(device)
        adv_ts = (adv_ts - adv_ts.mean()) / (adv_ts.std() + 1e-8)

        # --- 3. Optimization ---
        indices = np.arange(max_steps)
        for _ in range(K_epochs):
            np.random.shuffle(indices)
            for start in range(0, max_steps, batch_size):
                idx = indices[start:start+batch_size]
                
                # Reconstruct batch states: (1, Batch, 128)
                batch_h = h_ts[idx].transpose(0, 1).contiguous()
                batch_c = c_ts[idx].transpose(0, 1).contiguous()
                
                _, new_lp, ent, new_v, _, _ = policy.get_action_and_value(
                    s_ts[idx].unsqueeze(1), (batch_h, batch_c), a_ts[idx]
                )
                
                ratio = torch.exp(new_lp - lp_ts[idx])
                surr1 = ratio * adv_ts[idx]
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv_ts[idx]
                
                loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(new_v, ret_ts[idx]) - (entropy_coef * ent.mean())
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()


        current_std = torch.exp(torch.clamp(policy.actor_logstd, -2, 1)).detach().cpu().numpy()
        mean_reward = r_ts.mean().item()
        print(f"Step: {current_step} | Mean Reward: {mean_reward:.2f} | Std: {current_std[0]}")

        buffer.clear()
        
    
    torch.save(policy.state_dict(), "ppo_lstm_moonlander.pth")

if __name__ == "__main__":
    train()