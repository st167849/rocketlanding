import torch
import gymnasium as gym
import numpy as np
from models import ActorCritic
import time

def evaluate(model_path="ppo_moonlander_checkpoint.pth"):
    # 1. Setup Environment with rendering
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Load Model
    policy = ActorCritic(state_dim, action_dim).to(device)
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded {model_path}")
    except FileNotFoundError:
        print("Checkpoint not found. Running with random initialization.")

    policy.eval() # Set to evaluation mode

    # 3. Play Episodes
    for episode in range(5):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            
            with torch.no_grad():
                # During evaluation, we often just take the mean (no noise)
                # To see the "perfect" version of what it learned:
                latent = policy.base(state_tensor)
                action = torch.tanh(policy.actor_mean(latent))
            
            state, reward, terminated, truncated, _ = env.step(action.cpu().numpy().flatten())
            total_reward += reward
            env.render()
            
        print(f"Episode {episode + 1} | Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate()