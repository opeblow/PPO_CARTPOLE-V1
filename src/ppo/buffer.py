"""Rollout buffer for storing trajectories during PPO training."""

import numpy as np
import torch
from typing import Iterator


class RolloutBuffer:
    """Buffer for storing rollout experience with GAE computation.

    Stores states, actions, rewards, dones, log probabilities, and state values.
    Computes advantages and returns using Generalized Advantage Estimation (GAE).
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        num_envs: int = 1,
    ) -> None:
        """Initialize the rollout buffer.

        Args:
            buffer_size: Maximum number of steps to store.
            state_dim: Dimension of the state space.
            action_dim: Dimension of the action space.
            device: Device to store tensors on. Defaults to "cpu".
            num_envs: Number of parallel environments. Defaults to 1.
        """
        self._buffer_size = buffer_size
        self._device = device
        self._num_envs = num_envs

        self.states = np.zeros((buffer_size, num_envs, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.state_values = np.zeros((buffer_size, num_envs), dtype=np.float32)

        self.advantages = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_envs), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    @property
    def size(self) -> int:
        """Current size of the buffer."""
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        state_value: np.ndarray,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            done: Done flag.
            log_prob: Log probability of action.
            state_value: Estimated state value.
        """
        if self._ptr >= self._buffer_size:
            return

        self.states[self._ptr] = state
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = done
        self.log_probs[self._ptr] = log_prob
        self.state_values[self._ptr] = state_value

        self._ptr = (self._ptr + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def compute_advantages_and_returns(
        self, last_value: np.ndarray, gamma: float, gae_lambda: float
    ) -> None:
        """Compute advantages and returns using GAE.

        Args:
            last_value: Value of the last state.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
        """
        last_gae_lambda = 0
        for step in reversed(range(self._buffer_size)):
            if step == self._buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.state_values[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal
                - self.state_values[step]
            )
            last_gae_lambda = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
            )
            self.advantages[step] = last_gae_lambda

        self.returns = self.advantages + self.state_values

    def get_batches(self, batch_size: int) -> Iterator[tuple]:
        """Yield shuffled batches for training.

        Args:
            batch_size: Size of each batch.

        Yields:
            Tuples of (states, actions, old_log_probs, advantages, returns) tensors.
        """
        flat_states = self.states.reshape(-1, self.states.shape[-1])
        flat_actions = self.actions.reshape(-1)
        flat_log_probs = self.log_probs.reshape(-1)
        flat_advantages = self.advantages.reshape(-1)
        flat_returns = self.returns.reshape(-1)

        total_samples = self._buffer_size * self._num_envs
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        for i in range(0, total_samples, batch_size):
            idx = indices[i : i + batch_size]
            yield (
                torch.tensor(flat_states[idx]).to(self._device),
                torch.tensor(flat_actions[idx]).to(self._device),
                torch.tensor(flat_log_probs[idx]).to(self._device),
                torch.tensor(flat_advantages[idx]).to(self._device),
                torch.tensor(flat_returns[idx]).to(self._device),
            )

    def clear(self) -> None:
        """Reset the buffer."""
        self._ptr = 0
        self._size = 0
