import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device):
        self.buffer_size = buffer_size
        self.device = device
        
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size,dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.state_values = np.zeros(buffer_size, dtype=np.float32)
        
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, done, log_prob, state_value):
        if self.ptr >= self.buffer_size:
           return
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.state_values[self.ptr] = state_value
        
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)

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

    def get_batches(self, batch_size):
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        for i in range(0, self.size, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield (
                torch.tensor(self.states[batch_indices]).to(self.device),
                torch.tensor(self.actions[batch_indices]).to(self.device),
                torch.tensor(self.rewards[batch_indices]).to(self.device),
                torch.tensor(self.dones[batch_indices]).to(self.device),
                torch.tensor(self.log_probs[batch_indices]).to(self.device),
                torch.tensor(self.state_values[batch_indices]).to(self.device),
                torch.tensor(self.advantages[batch_indices]).to(self.device),
                torch.tensor(self.returns[batch_indices]).to(self.device)
            )

    def clear(self):
        self.ptr = 0
        self.size = 0
