import torch
import torch.optim as optim
import torch.nn as nn
import gymnasium as gym
import numpy as np
from collections import deque

# UPDATED IMPORT: FrameStack is now FrameStackObservation
from gymnasium.wrappers import FrameStackObservation, FlattenObservation

# Import your custom modules
from rocketlanding.models import ActorCritic


# --- Custom Wrapper to Remove Velocities ---
class NoVelocityWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Original LunarLander obs: [x, y, vx, vy, angle, v_angle, leg1, leg2]
        # We keep: [0, 1, 4, 6, 7]
        self.keep_indices = [0, 1, 4, 6, 7]
        
        low = self.observation_space.low[self.keep_indices]
        high = self.observation_space.high[self.keep_indices]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return observation[self.keep_indices]

def train():
    # --- Configuration ---
    ENV_NAME = "LunarLanderContinuous-v3"
    total_timesteps = 1_000_000
    max_steps = 2048
    batch_size = 64
    K_epochs = 10
    lr = 2e-4
    gamma = 0.99
    gae_lambda = 0.95
    eps_clip = 0.2
    entropy_coef = 0.001

    # --- 1. Environment Setup ---
    env = gym.make(ENV_NAME)
    
    # Step A: Remove velocities
    env = NoVelocityWrapper(env)
    
    # Step B: Stack frames
    env = FrameStackObservation(env, stack_size=8)
    
    # Step C: Flatten to 1D vector
    env = FlattenObservation(env)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    policy = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    
    current_step = 0
    print(f"Training started on {device}. Observation Space: {env.observation_space.shape[0]}")

    # --- NEW: Episode Reward Tracking ---
    # Stores the last 100 finished episode rewards
    episode_rewards = deque(maxlen=100) 
    current_ep_reward = 0.0

    state, _ = env.reset()

    while current_step < total_timesteps:
        states, raw_actions, logprobs, rewards, dones, values = [], [], [], [], [], []
        
        for _ in range(max_steps):
            current_step += 1
            
            state_np = np.array(state, dtype=np.float32)
            st_tensor = torch.FloatTensor(state_np).to(device).unsqueeze(0)
            
            with torch.no_grad():
                action, lp, _, val, raw_act = policy.get_action_and_value(st_tensor)

            next_state, reward, term, trunc, _ = env.step(action.cpu().numpy().flatten())
            
            # --- NEW: Accumulate episode reward ---
            current_ep_reward += reward

            states.append(state_np)
            raw_actions.append(raw_act.cpu().numpy().flatten())
            logprobs.append(lp.cpu().item())
            rewards.append(reward)
            dones.append(term)
            values.append(val.cpu().item())
            
            state = next_state
            if term or trunc:
                # --- NEW: Log finished flight and reset ---
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                state, _ = env.reset()

        # --- GAE Advantage Calculation ---
        returns, advantages = [], []
        gae = 0
        
        with torch.no_grad():
            last_state_np = np.array(state, dtype=np.float32)
            if dones[-1]:
                next_val = 0.0
            else:
                # --- FIX 1: Added .unsqueeze(0) to match network input shape ---
                next_val = policy.get_value(
                    torch.FloatTensor(last_state_np).to(device).unsqueeze(0)
                ).item()

        for i in reversed(range(len(rewards))):
            mask = 1.0 - dones[i]
            delta = rewards[i] + gamma * next_val * mask - values[i]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_val = values[i]
            returns.insert(0, gae + values[i])

        s_ts = torch.FloatTensor(np.array(states)).to(device)
        ra_ts = torch.FloatTensor(np.array(raw_actions)).to(device)
        lp_ts = torch.FloatTensor(np.array(logprobs)).to(device)
        adv_ts = torch.FloatTensor(advantages).to(device)
        ret_ts = torch.FloatTensor(returns).to(device)
        
        adv_ts = (adv_ts - adv_ts.mean()) / (adv_ts.std() + 1e-8)

        num_samples = len(states)
        indices = np.arange(num_samples)

        for _ in range(K_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                idx = indices[start:start + batch_size]
                if len(idx) == 0:
                    continue

                _, new_lp, ent, new_v, _ = policy.get_action_and_value(
                    s_ts[idx], ra_ts[idx]
                )

                ratio = torch.exp(new_lp - lp_ts[idx])
                surr1 = ratio * adv_ts[idx]
                surr2 = torch.clamp(
                    ratio, 1 - eps_clip, 1 + eps_clip
                ) * adv_ts[idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(new_v.squeeze(-1), ret_ts[idx])

                loss = actor_loss + 0.5 * critic_loss - entropy_coef * ent.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()


        current_std = torch.exp(torch.clamp(policy.actor_logstd, -2, 1)).detach().cpu().numpy()
        
        # --- NEW: Print Mean Episode Reward instead of Mean Step Reward ---
        mean_ep_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
        print(f"Step: {current_step} | Mean Flight Reward (last 100): {mean_ep_reward:.2f} | Std: {current_std[0]}")

    torch.save(policy.state_dict(), "ppo_moonlander_final.pth")
    print("model saved")

if __name__ == "__main__":
    train()