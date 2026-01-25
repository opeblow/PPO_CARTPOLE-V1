import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
    
    def evaluate(self, state, action):
        state = torch.as_tensor(state,dtype=torch.float32)
        action = torch.as_tensor(action).long()
        
        probs = self.actor(state)
        dist = Categorical(probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy
        