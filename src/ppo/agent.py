"""PPO Agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from ppo.network import ActorCriticNetwork
from ppo.buffer import RolloutBuffer
from ppo.config import TrainConfig


class PPOAgent:
    """Proximal Policy Optimization Agent.

    Implements the PPO algorithm with clipped surrogate objective,
    GAE advantage estimation, and entropy regularization.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TrainConfig,
    ) -> None:
        """Initialize the PPO agent.

        Args:
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            config: Training configuration.
        """
        self._device = config.device
        self._gamma = config.gamma
        self._gae_lambda = config.gae_lambda
        self._clip_epsilon = config.clip_epsilon
        self._entropy_coef = config.entropy_coef
        self._max_grad_norm = config.max_grad_norm
        self._n_epochs = config.n_epochs
        self._batch_size = config.batch_size
        self._num_envs = config.num_envs

        self._policy = ActorCriticNetwork(
            state_dim, action_dim, config.hidden_size
        ).to(self._device)
        self._optimizer = optim.Adam(
            self._policy.parameters(), lr=config.learning_rate
        )

        self._buffer = RolloutBuffer(
            config.n_steps,
            state_dim,
            action_dim,
            self._device,
            config.num_envs,
        )

    @property
    def policy(self) -> ActorCriticNetwork:
        """Get the policy network."""
        return self._policy

    def select_action(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action given a state.

        Args:
            state: Current state.

        Returns:
            Tuple of (action, log_prob, value).
        """
        with torch.no_grad():
            action, log_prob, _ = self._policy.act(state)
            value = self._policy.critic(state)

        if self._num_envs == 1:
            return action, log_prob, value
        return action, log_prob, value

    def update(self, next_state: torch.Tensor) -> None:
        """Update the policy using PPO.

        Args:
            next_state: The state after the last step in the rollout.
        """
        with torch.no_grad():
            next_value = self._policy.critic(next_state).cpu().numpy().flatten()

        self._buffer.compute_advantages_and_returns(
            next_value, self._gamma, self._gae_lambda
        )

        for _ in range(self._n_epochs):
            for (
                states,
                actions,
                old_log_probs,
                advantages,
                returns,
            ) in self._buffer.get_batches(self._batch_size):
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                new_log_probs, state_values, dist_entropy = self._policy.evaluate(
                    states, actions
                )

                ratios = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(
                        ratios, 1 - self._clip_epsilon, 1 + self._clip_epsilon
                    )
                    * advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(state_values, returns)

                loss = (
                    actor_loss
                    + 0.5 * critic_loss
                    - self._entropy_coef * dist_entropy.mean()
                )

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._policy.parameters(), self._max_grad_norm
                )
                self._optimizer.step()

        self._buffer.clear()

    def save(self, path: str) -> None:
        """Save the policy network.

        Args:
            path: Path to save the model.
        """
        torch.save(self._policy.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the policy network.

        Args:
            path: Path to load the model from.
        """
        self._policy.load_state_dict(
            torch.load(path, map_location=self._device)
        )
