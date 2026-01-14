import torch
import gymnasium as gym
import numpy as np
from model import SymmetricActorCriticLSTM

def evaluate(model_path="ppo_moonlander_lstm_opt.pth"):
    # 1. Setup Environment
    # Mask must match training exactly: x, y, angle, leg1, leg2
    MASK_INDICES = [0, 1, 4, 6, 7] 
    env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Load Model
    # Input Dim = 5 (Masked), Action Dim = 2
    policy = SymmetricActorCriticLSTM(len(MASK_INDICES), 2, hidden_size=256).to(device)
    
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded {model_path}")
    except FileNotFoundError:
        print(f"Checkpoint {model_path} not found. Running with random weights.")

    policy.eval()

    # 3. Play Episodes
    for episode in range(5):
        state, _ = env.reset()
        
        # --- LSTM INITIALIZATION ---
        # FIX 1: Your model defines LSTM size as hidden_size // 2
        # So if hidden_size=256, the LSTM state is 128.
        h_state = torch.zeros(1, 1, 256).to(device)
        c_state = torch.zeros(1, 1, 256).to(device)
        lstm_state = (h_state, c_state)
        
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            # Manually mask the observation since we don't have the wrapper here
            masked_state = state[MASK_INDICES]
            
            # Prepare tensor: [Seq=1, Batch=1, Dim=5]
            state_tensor = torch.FloatTensor(masked_state).to(device).view(1, 1, -1)
            
            with torch.no_grad():
                # FIX 2: Your model returns 5 values, not 6
                # Returns: action, log_prob, entropy, value, next_lstm_state
                action, _, _, _, lstm_state = policy.get_action_and_value(
                    state_tensor, 
                    lstm_state
                )
            
            # Step the environment
            real_action = action.cpu().numpy().flatten()
            state, reward, terminated, truncated, _ = env.step(real_action)
            total_reward += reward
            
        print(f"Episode {episode + 1} | Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate()