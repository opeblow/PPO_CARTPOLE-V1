import gymnasium as gym
import numpy as np
import torch
from single_env.ppo_agent import PPOAgent
from single_env.config import config

def evaluate(model_path, num_episodes=10,render=True):
    config=config()
    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim, config)
    agent.policy.load_state_dict(torch.load(model_path, map_location=config.device))
    agent.policy.eval()
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state,_ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if render:
                env.render()
            action, _, _ = agent.policy.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            done = done or truncated
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    env.close()
if __name__ == "__main__":
    evaluate(model_path='checkpoints/ppo_single_env_final.pt', num_episodes=10, render=True)
        