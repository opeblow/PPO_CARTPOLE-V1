import gymnasium as gym
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiple_env.config import Config
from multiple_env.ppo_agent import PPOAgent

def plot_results(rewards):
    plt.figure(figsize=(10,5))
    plt.plot(rewards,alpha=0.3,color='blue',label='Raw Reward')
    avg=[np.mean(rewards[max(0,i-10):i+1])for i in range(len(rewards))]
    plt.plot(avg,color='red',linewidth=2,label='Moving Average')
    plt.title("Pytorch PPO:4-Env Parallel Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("results.png")
    plt.show()

def train():
    cfg=Config
    env=gym.make_vec(cfg.env_name,num_envs=cfg.num_envs,vectorization_mode="sync",disable_env_checker=True)
    env=gym.wrappers.vector.RecordEpisodeStatistics(env)
    state_dim=env.single_observation_space.shape[0]
    action_dim=env.single_action_space.n
    agent=PPOAgent(state_dim,action_dim,cfg)
    state,_=env.reset()
    all_episode_rewards=[]
    pbar=tqdm(range(cfg.total_updates))
    for update in pbar:
        for _ in range(cfg.n_steps):
            action,log_prob,value=agent.select_action(state)
            action=action.astype(np.int64).flatten()
            next_state,reward,term,trunc,info=env.step(action)
            done=np.logical_or(term,trunc)
            agent.buffer.add(state,action,reward,done,log_prob,value)
            state=next_state
            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        res=item["episode"]["r"]
                        all_episode_rewards.append(res)
                        print(f"Episode Reward:{res}")
        if all_episode_rewards:
            avg_r=np.mean(all_episode_rewards[-10:])
            pbar.set_postfix(avg_reward=f"{avg_r:.2f}")
                        
        pbar.update(1)                
        agent.update(state)
        if all_episode_rewards:
            avg_r=np.mean(all_episode_rewards[-10:])
            pbar.set_postfix({'Avg Reward':f'{avg_r:.2f}'})
            if avg_r > 495:
                break
    env.close()
    plot_results(all_episode_rewards)

if __name__=="__main__":
    train()
