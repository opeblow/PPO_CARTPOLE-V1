import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time
from single_env.config import Config
from single_env.ppo_agent import PPOAgent

def train():
    config=Config()
    start_time=time.time()
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"Environment: {config.env_name}, State Dim: {state_dim}, Action Dim: {action_dim}")
    agent = PPOAgent(state_dim, action_dim, config)
    episode_rewards = []
    episode_lengths = []
    timesteps = []
    state,_= env.reset()
    episode_reward=0
    episode_length=0
    total_updates=config.total_timesteps // config.n_steps
    pbar = tqdm(range(total_updates), desc="Training Progress")
    for update in range(total_updates):
        for step in range(config.n_steps):
            action, log_prob, state_value = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.buffer.add(state, action, reward, done or truncated, log_prob, state_value)
            state = next_state
            episode_reward += reward
            episode_length += 1
            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                timesteps.append(update * config.n_steps + step)
                state,_ = env.reset()
                episode_reward = 0
                episode_length = 0
        agent.update()
        pbar.update(1)
        if (update + 1) % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.log_interval:])
            pbar.set_postfix({'Avg Reward': f'{avg_reward:.2f}'})
        if (update + 1) % config.save_interval == 0:
           agent.save(f"{config.save_dir}/ppo_single_env_{update + 1}.pth")
    pbar.close()
    env.close()

    total_time = time.time() - start_time
    hours=int(total_time // 3600)
    minutes=int((total_time % 3600) // 60)
    seconds=int(total_time % 60)

    print(f"Training completed in {hours}h {minutes}m {seconds}s")
    agent.save(f"{config.save_dir}/ppo_{config.env_name}_final.pth")
    save_training_stats(episode_rewards,episode_lengths,timesteps,total_time,config)
    plot_results(episode_rewards,episode_lengths,timesteps,total_time,config)
def save_training_stats(rewards,lengths,timesteps,total_time,config):
    with open(os.path.join(config.log_dir, f"training_stats_{config.env_name}.txt"), "w") as f:
        f.write(f"Total Time: {total_time}\n")
        f.write(f"Average Reward: {np.mean(rewards)}\n")
        f.write(f"Average Length: {np.mean(lengths)}\n")
        f.write(f"Total Timesteps: {timesteps[-1]}\n")

def plot_results(rewards,lengths,timesteps,total_time,config):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(timesteps, rewards)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward over Timesteps')

    plt.subplot(1,2,2)
    plt.plot(timesteps, lengths)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Length')
    plt.title('Episode Length over Timesteps')

    plt.tight_layout()
    plt.savefig(os.path.join(config.log_dir, f"training_plots_{config.env_name}.png"))
    plt.show()

if __name__ == "__main__":
    train()
