import torch
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStackObservation, FlattenObservation

# Adjust this import based on where your ActorCritic class is located
# from rocketlanding.models import ActorCritic
from rocketlanding_multiframe.models import ActorCritic 

# --- Must define the same Wrapper class as training ---
class NoVelocityWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Keep: [x, y, angle, leg1, leg2]
        self.keep_indices = [0, 1, 4, 6, 7]
        
        low = self.observation_space.low[self.keep_indices]
        high = self.observation_space.high[self.keep_indices]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return observation[self.keep_indices]

def evaluate(model_path="ppo_moonlander_final.pth"):
    # 1. Setup Environment
    # We create the raw env, then apply the EXACT same wrappers as training
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    
    # A: Remove velocities
    env = NoVelocityWrapper(env)
    
    # B: Stack frames (size 4)
    env = FrameStackObservation(env, stack_size=6)
    
    # C: Flatten (4x5 -> 20)
    env = FlattenObservation(env)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Observation Dimension: {state_dim} (Should be 20)")

    # 2. Load Model
    policy = ActorCritic(state_dim, action_dim).to(device)
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Make sure you trained the model first!")
        return

    policy.eval()

    # 3. Play Episodes
    for episode in range(5):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            # Important: Handle LazyFrames just like in training
            state_np = np.array(state)
            
            state_tensor = torch.FloatTensor(state_np).to(device).unsqueeze(0)
            
            with torch.no_grad():
                # For evaluation, we want DETERMINISTIC actions (no random noise).
                # We calculate the mean directly and squash it with tanh.
                latent = policy.base(state_tensor)
                mean = policy.actor_mean(latent)
                action = torch.tanh(mean)
            
            # Take step
            state, reward, terminated, truncated, _ = env.step(action.cpu().numpy().flatten())
            total_reward += reward
            
            # Rendering is handled automatically by render_mode="human" in gym.make
            
        print(f"Episode {episode + 1} | Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate()