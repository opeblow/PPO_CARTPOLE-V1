"""Neural network architecture for Actor-Critic PPO agent."""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class ActorCriticNetwork(nn.Module):
    """Actor-Critic neural network for PPO.

    The network consists of a shared feature extractor and separate
    actor (policy) and critic (value) heads.

    Attributes:
        actor: Policy network that outputs action probabilities.
        critic: Value network that estimates state values.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64) -> None:
        """Initialize the Actor-Critic network.

        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            hidden_size: Size of hidden layers. Defaults to 64.
        """
        super().__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self) -> None:
        """Forward pass not implemented for policy network."""
        raise NotImplementedError("Use act() or evaluate() for policy operations")

    def act(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action given a state.

        Args:
            state: Current state tensor.

        Returns:
            Tuple of (action, log_prob, entropy).
        """
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions given states.

        Args:
            state: Batch of states.
            action: Batch of actions.

        Returns:
            Tuple of (log_probs, state_values, entropy).
        """
        probs = self.actor(state)
        dist = Categorical(probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, torch.squeeze(state_values), dist_entropy
