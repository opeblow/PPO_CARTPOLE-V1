import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, buffer_size,num_envs, state_dim,action_dim , device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_envs=num_envs
        self.states = np.zeros((buffer_size,num_envs, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size,num_envs,action_dim),dtype=np.float32)
        self.rewards = np.zeros((buffer_size,num_envs),dtype=np.float32)
        self.dones = np.zeros((buffer_size,num_envs),dtype=np.float32)
        self.log_probs = np.zeros((buffer_size,num_envs),dtype=np.float32)
        self.state_values = np.zeros((buffer_size,num_envs),dtype=np.float32)
        
        self.advantages = np.zeros((buffer_size,num_envs),dtype=np.float32)
        self.returns = np.zeros((buffer_size,num_envs),dtype=np.float32)
        
        self.ptr = 0
     

    def add(self, state, action, reward, done, log_prob, state_value):
        if self.ptr >= self.buffer_size:
           return
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.state_values[self.ptr] = state_value
        
        self.ptr = (self.ptr + 1)% self.buffer_size
      

    def compute_advantages_and_returns(self, last_value, gamma, gae_lambda):
        last_gae_lambda = 0
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - 0.0
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.state_values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.state_values[step]
            last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
            self.advantages[step] = last_gae_lambda
        
        self.returns = self.advantages + self.state_values
    

    def get_batches(self,batch_size):
        flat_states=self.states.reshape(-1,self.states.shape[-1])
        flat_actions=self.actions.reshape(-1)
        flat_long_probs=self.log_probs.reshape(-1)
        flat_advantages=self.advantages.reshape(-1)
        flat_returns=self.returns.reshape(-1)

        total_samples=self.buffer_size * self.num_envs
        indices=np.arange(total_samples)
        np.random.shuffle(indices)

        for i in range(0, total_samples, batch_size):
            idx = indices[i:i+batch_size]
            yield (
                torch.tensor(flat_states[idx]).to(self.device),
                torch.tensor(flat_actions[idx]).to(self.device),
                torch.tensor(flat_long_probs[idx]).to(self.device),
                torch.tensor(flat_advantages[idx]).to(self.device),
                torch.tensor(flat_returns[idx]).to(self.device)
            )

    def clear(self):
        self.ptr=0
