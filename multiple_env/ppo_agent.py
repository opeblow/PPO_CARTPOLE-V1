import torch
import torch.optim as optim
from multiple_env.network import ActorCritic
from multiple_env.buffer import RolloutBuffer
import numpy as np
import torch.nn as nn
from multiple_env.config import Config

class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.device = Config.device
        self.gamma = Config.gamma
        self.gae_lambda = Config.gae_lambda
        self.clipson_epsilon = Config.clipson_epsilon
        self.entropy_coef = Config.entropy_coef
        self.max_grad_norm = Config.max_grad_norm
        self.n_epochs = Config.n_epochs
        self.batch_size = Config.batch_size
        self.num_envs=Config.num_envs
        
        self.policy = ActorCritic(state_dim, action_dim, Config.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Config.learning_rate)
        
        self.buffer = RolloutBuffer(config.n_steps,Config.num_envs, state_dim, action_dim, self.device)
    
    def select_action(self, state):
        state_tensor=torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action, log_prob, _ = self.policy.act(state_tensor)
            value=self.policy.critic(state_tensor)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy().flatten()
    
    def update(self,next_state):
        next_state_tensor=torch.from_numpy(next_state).float().to(self.device)
        with torch.no_grad():
            next_value=self.policy.critic(next_state_tensor).cpu().numpy().flatten()
        self.buffer.compute_advantages_and_returns(next_value, self.gamma, self.gae_lambda)
        for _ in range(self.n_epochs):
            for states,actions,old_log_probs,advantages,returns in self.buffer.get_batches(self.batch_size):
                advantages=(advantages-advantages.mean())/(advantages.std() + 1e-8)
                new_log_probs,state_values,dist_entropy=self.policy.evaluate(states,actions)
                ratios=torch.exp(new_log_probs - old_log_probs)
                surr1=ratios * advantages
                surr2=torch.clamp(ratios,1-self.clipson_epsilon,1 + self.clipson_epsilon) * advantages
                actor_loss=-torch.min(surr1,surr2).mean()
                critic_loss=nn.MSELoss()(state_values,returns)
                loss=actor_loss + 0.5 * critic_loss - self.entropy_coef * dist_entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm)
                self.optimizer.step()
            self.buffer.clear()
    def save(self,path):
        torch.save(self.policy.state_dict(),path)      
