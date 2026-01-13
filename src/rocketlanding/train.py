import torch
import torch.optim as optim
import gymnasium as gym
import numpy as np
from torch.optim.lr_scheduler import LinearLR
import torch.nn as nn 

# Import your custom modules
from rocketlanding.models import ActorCritic
from rocketlanding.memory import RolloutBuffer

def train():
    # --- Configuration ---
    ENV_NAME = "LunarLanderContinuous-v3"
    total_timesteps = 1_000_000
    max_steps = 2048
    batch_size = 64
    K_epochs = 10
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    eps_clip = 0.2
    entropy_coef = 0.0005

    # Initialize environment (Naked, no observation normalization for now)
    env = gym.make(ENV_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    policy = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    
    current_step = 0
    while current_step < total_timesteps:
        # Data storage for current rollout
        states, raw_actions, logprobs, rewards, dones, values = [], [], [], [], [], []
        state, _ = env.reset()
        
        # --- Data Collection (Rollout) ---
        for _ in range(max_steps):
            current_step += 1
            st_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            
            with torch.no_grad():
                action, lp, _, val, raw_act = policy.get_action_and_value(st_tensor)

            next_state, reward, term, trunc, _ = env.step(action.cpu().numpy().flatten())
            
            states.append(state)
            raw_actions.append(raw_act.cpu().numpy().flatten())
            logprobs.append(lp.cpu().item())
            rewards.append(reward)
            dones.append(term)
            values.append(val.cpu().item())
            
            state = next_state
            if term or trunc:
                state, _ = env.reset()

        # --- GAE Advantage Calculation ---
        returns, advantages = [], []
        gae = 0
        with torch.no_grad():
            # Bootstrap value of the last state
            next_val = policy.get_value(torch.FloatTensor(state).to(device)).item()
        
        for i in reversed(range(len(rewards))):
            mask = 1.0 - dones[i]
            delta = rewards[i] + gamma * next_val * mask - values[i]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_val = values[i]
            returns.insert(0, gae + values[i])

        # Convert to Tensors
        s_ts = torch.FloatTensor(np.array(states)).to(device)
        ra_ts = torch.FloatTensor(np.array(raw_actions)).to(device)
        lp_ts = torch.FloatTensor(np.array(logprobs)).to(device)
        adv_ts = torch.FloatTensor(advantages).to(device)
        ret_ts = torch.FloatTensor(returns).to(device)
        
        # Advantage Normalization (Essential for PPO stability)
        adv_ts = (adv_ts - adv_ts.mean()) / (adv_ts.std() + 1e-8)

        # --- Optimization (Mini-batch SGD) ---
        indices = np.arange(max_steps)
        for _ in range(K_epochs):
            np.random.shuffle(indices)
            for start in range(0, max_steps, batch_size):
                idx = indices[start:start+batch_size]
                
                _, new_lp, ent, new_v, _ = policy.get_action_and_value(s_ts[idx], ra_ts[idx])
                
                ratio = torch.exp(new_lp - lp_ts[idx])
                surr1 = ratio * adv_ts[idx]
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv_ts[idx]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(new_v, ret_ts[idx])
                
                loss = actor_loss + 0.5 * critic_loss - entropy_coef * ent.mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5) # Gradient clipping
                optimizer.step()

        # --- Progress Monitoring ---
        current_std = torch.exp(torch.clamp(policy.actor_logstd, -2, 1)).detach().cpu().numpy()
        print(f"Step: {current_step} | Mean Reward: {np.mean(rewards):.2f} | Std: {current_std[0]}")

        # Save Checkpoint
        if current_step % 50_000 < max_steps:
            torch.save(policy.state_dict(), "ppo_moonlander_checkpoint.pth")
    torch.save(policy.state_dict(), "ppo_moonlander_final.pth")
if __name__ == "__main__":
    train()