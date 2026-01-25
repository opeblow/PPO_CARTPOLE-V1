import torch
import torch.optim as optim
from single_env.network import ActorCritic
from single_env.buffer import RolloutBuffer
import numpy as np
import torch.nn as nn
from single_env.config import Config

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
        
        self.policy = ActorCritic(state_dim, action_dim, Config.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Config.learning_rate)
        
        self.buffer = RolloutBuffer(config.n_steps, state_dim, action_dim, self.device)

    def select_action(self, state):
        with torch.no_grad():
            action, log_prob, _ = self.policy.act(state)
            state_tensor=torch.as_tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
            state_value = self.policy.critic(state_tensor).item()
        return action, log_prob.item(), state_value
    
    def update(self):
        with torch.no_grad():
            last_state = self.buffer.states[self.buffer.size - 1]
            last_state_tensor = torch.as_tensor(last_state,dtype=torch.float32).unsqueeze(0).to(self.device)
            last_value = self.policy.critic(last_state_tensor).item()
            self.buffer.compute_advantages_and_returns(last_value, self.gamma, self.gae_lambda)
        for _ in range(self.n_epochs):
            for states, actions, rewards, dones, old_log_probs, state_values, advantages, returns in self.buffer.get_batches(self.batch_size):
                log_probs, values, dist_entropy = self.policy.evaluate(states, actions)
                
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clipson_epsilon, 1.0 + self.clipson_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = nn.MSELoss()(values, returns)
                
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * dist_entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        self.buffer.clear()
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        