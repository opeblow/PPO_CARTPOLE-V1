import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from multiple_env.config import Config
from multiple_env.ppo_agent import PPOAgent

def plot_results(rewards, lengths):
    """Optional: Still generates the plot for a quick visual check."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot Rewards
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw Reward')
    avg_r = [np.mean(rewards[max(0, i-10):i+1]) for i in range(len(rewards))]
    ax1.plot(avg_r, color='blue', linewidth=2, label='Avg Reward')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward", color='blue')
    
    # Plot Lengths on a second y-axis
    ax2 = ax1.twinx()
    avg_l = [np.mean(lengths[max(0, i-10):i+1]) for i in range(len(lengths))]
    ax2.plot(avg_l, color='green', linestyle='--', alpha=0.6, label='Avg Length')
    ax2.set_ylabel("Episode Length", color='green')
    
    plt.title("Pytorch PPO: 4-Env Training Performance")
    fig.tight_layout()
    plt.savefig("logs/final_performance.png")
    plt.show()

def train():
    cfg = Config
    # Create directories for logs and checkpoints
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Initialize Vectorized Env with required wrappers
    env = gym.make_vec(
        cfg.env_name, 
        num_envs=cfg.num_envs, 
        vectorization_mode="sync", 
        disable_env_checker=True
    )
    # This wrapper is CRITICAL to see Average Reward and Length
    env = gym.wrappers.vector.RecordEpisodeStatistics(env)

    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n
    agent = PPOAgent(state_dim, action_dim, cfg)
    
    state, _ = env.reset()
    all_episode_rewards = []
    all_episode_lengths = []
    
    # Main training loop
    pbar = tqdm(range(cfg.total_updates), desc="PPO Training")
    for update in pbar:
        for _ in range(cfg.n_steps):
            # Agent selects action
            action, log_prob, value = agent.select_action(state)
            
            # Flatten to 1D array for Gymnasium VecEnv
            action = action.astype(np.int64).flatten()
            
            # Step the environment
            next_state, reward, term, trunc, info = env.step(action)
            done = np.logical_or(term, trunc)
            
            # Store transition in buffer
            agent.buffer.add(state, action, reward, done, log_prob, value)
            state = next_state
            
            # ===== FIXED SECTION: Extract episode statistics =====
            # Check for completed episodes using the correct structure
            if "_episode" in info:
                # info["_episode"] is a boolean array indicating which envs finished
                episode_mask = info["_episode"]
                for idx in range(cfg.num_envs):
                    if episode_mask[idx]:
                        # Extract the reward and length for this completed episode
                        r = float(info["episode"]["r"][idx])
                        l = int(info["episode"]["l"][idx])
                        all_episode_rewards.append(r)
                        all_episode_lengths.append(l)
                        
                        # Use tqdm.write to avoid breaking the progress bar
                        tqdm.write(f"Update {update} | Ep Reward: {r:.2f} | Ep Length: {l}")
            
            # Fallback: Also check final_info (some Gymnasium versions use this)
            elif "final_info" in info:
                for final_info_item in info["final_info"]:
                    if final_info_item is not None and "episode" in final_info_item:
                        r = float(final_info_item["episode"]["r"])
                        l = int(final_info_item["episode"]["l"])
                        all_episode_rewards.append(r)
                        all_episode_lengths.append(l)
                        tqdm.write(f"Update {update} | Ep Reward: {r:.2f} | Ep Length: {l}")

        # Update the PPO policy
        agent.update(state)
        
        # Display performance in the progress bar
        if all_episode_rewards:
            avg_r = np.mean(all_episode_rewards[-10:])
            avg_l = np.mean(all_episode_lengths[-10:])
            pbar.set_postfix({
                'Avg R': f'{avg_r:.2f}', 
                'Avg L': f'{avg_l:.1f}'
            })
            
            # Early stopping if target reward is reached
            if avg_r > 495:
                tqdm.write("Target reward reached! Saving final model...")
                agent.save("checkpoints/ppo_final_solved.pth")
                break
        
        # Periodic Checkpoints
        if update % 20 == 0:
            agent.save(f"checkpoints/ppo_checkpoint_{update}.pth")

    # Cleanup and Logging
    env.close()
    
    # Save training data to JSON
    logs = {
        "rewards": all_episode_rewards,
        "lengths": all_episode_lengths
    }
    with open("logs/training_logs.json", "w") as f:
        json.dump(logs, f)
    
    tqdm.write("Training complete. Logs saved to logs/training_logs.json")
    
    # Generate final plot from the logged lists
    if all_episode_rewards:
        plot_results(all_episode_rewards, all_episode_lengths)
    else:
        tqdm.write("Warning: No episodes completed during training!")

if __name__ == "__main__":
    train()

    