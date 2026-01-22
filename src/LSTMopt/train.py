import torch
import torch.optim as optim
import gymnasium as gym
import numpy as np
import time
from collections import deque
from model import SymmetricActorCriticLSTM
from gymnasium.spaces import Box
import torch.nn as nn

def make_env(env_id, mask_indices):
    def thunk():
        env = gym.make(env_id)
        
        # --- 1. Add Statistics Wrapper (CRITICAL FOR LOGGING) ---
        # This wrapper tracks the cumulative reward and adds it to info["episode"]["r"]
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        new_shape = (len(mask_indices),)
        new_observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=new_shape, 
            dtype=np.float32
        )
        
        # --- 3. Mask the Observation ---
        env = gym.wrappers.TransformObservation(
            env, 
            func=lambda obs: obs[mask_indices], 
            observation_space=new_observation_space
        )
        
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk
def train():
    # --- Hyperparameters ---
    MASK_INDICES = [0, 1, 4, 6, 7] # x, y, angle, leg1, leg2
    ENV_NAME = "LunarLanderContinuous-v3"
    
    NUM_ENVS = 256        
    NUM_STEPS = 256       
    TOTAL_TIMESTEPS = 20_000_000
    
    # LSTM Training Params
    MINIBATCH_ENVS = 16
    K_EPOCHS = 4
    
    # PPO Params
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_COEF = 0.2
    ENT_COEF = 0.005 
    VF_COEF = 0.75
    MAX_GRAD_NORM = 0.5
    BPTT_STEPS = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(ENV_NAME, MASK_INDICES) for _ in range(NUM_ENVS)]
    )
    
    policy = SymmetricActorCriticLSTM(len(MASK_INDICES), 2).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Storage
    obs_buffer = torch.zeros((NUM_STEPS, NUM_ENVS, len(MASK_INDICES))).to(DEVICE)
    actions_buffer = torch.zeros((NUM_STEPS, NUM_ENVS, 2)).to(DEVICE)
    logprobs_buffer = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
    rewards_buffer = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
    dones_buffer = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)
    values_buffer = torch.zeros((NUM_STEPS, NUM_ENVS)).to(DEVICE)

    # LSTM States (Current running state)
    h_rollout = torch.zeros(1, NUM_ENVS, 256).to(DEVICE)
    c_rollout = torch.zeros(1, NUM_ENVS, 256).to(DEVICE)

    h_initial_buffer = torch.zeros(1, NUM_ENVS, 256).to(DEVICE)
    c_initial_buffer = torch.zeros(1, NUM_ENVS, 256).to(DEVICE)


    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_done = torch.zeros(NUM_ENVS).to(DEVICE)
    ep_reward_deque = deque(maxlen=50)

    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)

    for update in range(1, num_updates + 1):
        
        # === 1. ROLLOUT PHASE ===
        policy.eval()

        # ### FIX 2: Save the state before we start collecting data ###
        # We need to remember exactly what the LSTM was thinking at step 0
        h_initial_buffer = h_rollout.clone()
        c_initial_buffer = c_rollout.clone()

        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            
            obs_buffer[step] = torch.Tensor(next_obs).to(DEVICE)
            dones_buffer[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, (h_next, c_next) = policy.get_action_and_value(
                    obs_buffer[step].unsqueeze(0), 
                    (h_rollout, c_rollout), 
                    done=next_done
                )

            values_buffer[step] = value.flatten()
            actions_buffer[step] = action.flatten(0, 1)
            logprobs_buffer[step] = logprob.flatten()

            real_action = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminations, truncations, infos = envs.step(real_action)
            
            next_done = torch.Tensor(terminations | truncations).to(DEVICE)
            rewards_buffer[step] = torch.Tensor(reward).to(DEVICE)

            h_rollout = h_next * (1.0 - next_done).view(1, -1, 1)
            c_rollout = c_next * (1.0 - next_done).view(1, -1, 1)

            if "episode" in infos:
                for r, done in zip(infos["episode"]["r"], next_done.cpu().numpy()):
                    if done:
                        ep_reward_deque.append(r)

        # === 2. GAE CALCULATION ===
        with torch.no_grad():
            _, _, _, next_value, _ = policy.get_action_and_value(
                torch.Tensor(next_obs).to(DEVICE).unsqueeze(0),
                (h_rollout, c_rollout)
            )
            next_value = next_value.reshape(1, -1)
            
            advantages = torch.zeros_like(rewards_buffer).to(DEVICE)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value.squeeze(0)
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                
                delta = rewards_buffer[t] + GAMMA * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            
            returns = advantages + values_buffer

        # === 3. OPTIMIZATION ===
        policy.train()

        env_indices = np.arange(NUM_ENVS)

        for epoch in range(K_EPOCHS):
            np.random.shuffle(env_indices)

            for start in range(0, NUM_ENVS, MINIBATCH_ENVS):
                end = start + MINIBATCH_ENVS
                mb_env_inds = env_indices[start:end]

                # Initial LSTM state for these environments
                h0 = h_initial_buffer[:, mb_env_inds]
                c0 = c_initial_buffer[:, mb_env_inds]

                # Truncated BPTT over time
                for t_start in range(0, NUM_STEPS, BPTT_STEPS):
                    t_end = t_start + BPTT_STEPS

                    mb_obs = obs_buffer[t_start:t_end, mb_env_inds]
                    mb_actions = actions_buffer[t_start:t_end, mb_env_inds]
                    mb_logprobs = logprobs_buffer[t_start:t_end, mb_env_inds]
                    mb_advantages = advantages[t_start:t_end, mb_env_inds]
                    mb_returns = returns[t_start:t_end, mb_env_inds]
                    mb_values = values_buffer[t_start:t_end, mb_env_inds]
                    mb_dones = dones_buffer[t_start:t_end, mb_env_inds]

                    # Truncate gradient flow
                    h0 = h0.detach()
                    c0 = c0.detach()

                    new_action, new_logprob, entropy, new_value, (h0, c0) = policy.get_action_and_value(
                        mb_obs,
                        (h0, c0),
                        action=mb_actions
                    )

                    # Flatten everything
                    b_logprobs = mb_logprobs.reshape(-1)
                    b_advantages = mb_advantages.reshape(-1)
                    b_returns = mb_returns.reshape(-1)
                    b_values = mb_values.reshape(-1)

                    new_logprob = new_logprob.reshape(-1)
                    new_value = new_value.reshape(-1)
                    entropy = entropy.reshape(-1)

                    # --- Advantage normalization (CRITICAL) ---
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                    # --- PPO policy loss ---
                    logratio = new_logprob - b_logprobs
                    ratio = logratio.exp()

                    pg_loss1 = -b_advantages * ratio
                    pg_loss2 = -b_advantages * torch.clamp(
                        ratio, 1 - CLIP_COEF, 1 + CLIP_COEF
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # --- Value loss ---
                    v_loss_unclipped = (new_value - b_returns) ** 2
                    v_clipped = b_values + torch.clamp(
                        new_value - b_values, -CLIP_COEF, CLIP_COEF
                    )
                    v_loss_clipped = (v_clipped - b_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # --- Total loss ---
                    loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

        avg_rew = np.mean(ep_reward_deque) if ep_reward_deque else 0
        if update % 10 == 0:
            
            sps = int(global_step / (time.time() - start_time))
            print(f"Update: {update} | Step: {global_step} | SPS: {sps} | Mean Reward: {avg_rew:.2f}")
        if avg_rew >= 250:
            torch.save(policy.state_dict(), "ppo_moonlander_lstm_success.pth")
            envs.close()
            print(f"training successfull at Update {update} with reward {avg_rew}") 
            break

    envs.close()
    torch.save(policy.state_dict(), "ppo_moonlander_lstm_opt.pth")

if __name__ == "__main__":
    train()