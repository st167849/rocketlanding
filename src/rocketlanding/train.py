import torch
import torch.optim as optim
import gymnasium as gym
import numpy as np
from torch.optim.lr_scheduler import LinearLR
import torch.nn as nn 

# Import your custom modules
from rocketlanding.models import ActorCritic
from rocketlanding.memory import RolloutBuffer

# --- Hyperparameters ---
ENV_NAME = "LunarLanderContinuous-v3"  # The environment
lr = 1e-4                    # Learning Rate (Standard for PPO)
gamma = 0.99                 # Discount Factor (How much we care about future rewards)
K_epochs = 10                 # How many times we update on the same batch of data
eps_clip = 0.2               # PPO Clipping (Prevents the agent from changing too fast)
max_timesteps = 5000         # How many steps to collect before updating
total_timesteps = 1500_000    # Total training duration

def train():
    # 1. Setup Environment
    # continuous=True is CRITICAL for rocket throttling (0.0 to 1.0) rather than On/Off
    env = gym.make(ENV_NAME)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. Setup Agent & Memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer = RolloutBuffer()
    # ... after optimizer = optim.Adam(...) ...
    
    # Calculate how many update steps we will do in total
    num_updates = total_timesteps // max_timesteps

    # Decay LR linearly from 100% to 1% over the course of training
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_updates)
    print(f"Training on {device}...")

    # 3. Main Training Loop
    time_step = 0
    i_episode = 0
    
    while time_step < total_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0

        # --- Phase 1: Data Collection (Rollout) ---
        for t in range(max_timesteps):
            time_step += 1
            
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).to(device)
            
            # Select action with no gradient (we are just playing, not training yet)
            with torch.no_grad():
                action, log_prob, _, value = policy.get_action_and_value(state_tensor)

            # Interact with env
            action_np = action.cpu().numpy().flatten()
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            
            # Save data to buffer
            buffer.add(state, action_np, log_prob.cpu().numpy(), reward, terminated)
            
            state = next_state
            current_ep_reward += reward

            # Handle end of episode
            if terminated or truncated:
                break
        
        # --- Phase 2: Update (Training) ---
        # We update the network after every "batch" of experience
        
        # Calculate Discounted Rewards (Rewards-to-Go)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing rewards helps training stability
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert buffer to tensors
        old_states = torch.tensor(np.array(buffer.states), dtype=torch.float32).to(device)
        old_actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(np.array(buffer.logprobs), dtype=torch.float32).to(device)

        # Optimize policy for K epochs
        for _ in range(K_epochs):
            # Evaluate old actions and values
            _, logprobs, entropy, state_values = policy.get_action_and_value(old_states, old_actions)
            
            # Calculate ratios (How much did the policy change?)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Calculate Advantages
            advantages = rewards - state_values.detach().squeeze()

            # PPO Loss Formula (The "Surrogate" Loss)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            
            # Final Loss: Maximize Reward - Minimize Value Error + Bonus for Entropy (Exploration)
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values.squeeze(), rewards) - 0.01 * entropy
            
            # Gradient Step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        # Clear buffer for next batch
        buffer.clear()
        
        scheduler.step()
        # Logging
        i_episode += 1
        if i_episode % 10 == 0:
            print(f"Episode: {i_episode} \t Timestep: {time_step} \t Reward: {current_ep_reward:.2f}")

        # Save Checkpoint
        if i_episode % 1000 == 0:
            torch.save(policy.state_dict(), f"ppo_rocket_{i_episode}.pth")
     # ... [Existing while loop code above] ...

    # --- SAVE FINAL MODEL ---
    final_path = "ppo_rocket_final.pth"
    torch.save(policy.state_dict(), final_path)
    print("-------------------------------------------------")
    print(f"Training finished! Final model saved to: {final_path}")
    print("-------------------------------------------------")

    
if __name__ == '__main__':
    # Fix for 'nn' is not defined error in loss calculation
    train()
   