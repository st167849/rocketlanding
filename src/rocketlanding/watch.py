import gymnasium as gym
import torch
from rocketlanding.models import ActorCritic

def watch():
    # 1. Load Environment
    env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    
    # 2. Load Model
    device = torch.device('cpu')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = ActorCritic(state_dim, action_dim).to(device)
    
    # LOAD THE WEIGHTS (Change this filename to your latest .pth file)
    try:
        policy.load_state_dict(torch.load("/root/projects/ppo_rocket_checkpoi.pth", map_location=device))
        print("Model loaded!")
    except:
        print("No model found. Flying with random weights.")

    # 3. Fly
    state, _ = env.reset()
    while True:
        state_tensor = torch.FloatTensor(state).to(device)
        
        with torch.no_grad():
            # For watching, we use the MEAN action (deterministic), not sampling
            # But our current get_action_and_value samples. 
            # For simplicity, let's just sample. It works fine.
            action, log_prob, _, value, raw_act = policy.get_action_and_value(state_tensor)
            
        action = action.cpu().numpy().flatten()
        state, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            state, _ = env.reset()

if __name__ == '__main__':
    watch()