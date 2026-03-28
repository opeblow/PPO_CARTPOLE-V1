"""Tests for PPO buffer."""

import numpy as np
import pytest
import torch

from ppo.buffer import RolloutBuffer


class TestRolloutBuffer:
    """Tests for RolloutBuffer class."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = RolloutBuffer(
            buffer_size=100,
            state_dim=4,
            action_dim=2,
            device="cpu",
        )

        assert buffer.size == 0
        assert buffer.states.shape == (100, 1, 4)
        assert buffer.actions.shape == (100, 1)

    def test_initialization_multi_env(self):
        """Test buffer initialization with multiple envs."""
        buffer = RolloutBuffer(
            buffer_size=100,
            state_dim=4,
            action_dim=2,
            device="cpu",
            num_envs=4,
        )

        assert buffer.states.shape == (100, 4, 4)
        assert buffer.actions.shape == (100, 4)

    def test_add_single_env(self):
        """Test adding transitions (single env)."""
        buffer = RolloutBuffer(
            buffer_size=10,
            state_dim=4,
            action_dim=2,
            device="cpu",
            num_envs=1,
        )

        state = np.random.randn(4).astype(np.float32)
        action = np.array(1)
        reward = np.array(1.0)
        done = np.array(0.0)
        log_prob = np.array(0.5)
        state_value = np.array(0.5)

        buffer.add(state, action, reward, done, log_prob, state_value)

        assert buffer.size == 1
        np.testing.assert_array_almost_equal(buffer.states[0, 0], state)

    def test_add_multi_env(self):
        """Test adding transitions (multi env)."""
        buffer = RolloutBuffer(
            buffer_size=10,
            state_dim=4,
            action_dim=2,
            device="cpu",
            num_envs=4,
        )

        state = np.random.randn(4, 4).astype(np.float32)
        action = np.array([0, 1, 0, 1])
        reward = np.array([1.0, 0.0, 1.0, 0.0])
        done = np.array([0, 0, 1, 0])
        log_prob = np.array([0.5, 0.3, 0.7, 0.2])
        state_value = np.array([0.5, 0.4, 0.6, 0.3])

        buffer.add(state, action, reward, done, log_prob, state_value)

        assert buffer.size == 1

    def test_compute_advantages(self):
        """Test GAE advantage computation."""
        buffer_size = 5
        buffer = RolloutBuffer(
            buffer_size=buffer_size,
            state_dim=4,
            action_dim=2,
            device="cpu",
            num_envs=1,
        )

        buffer.rewards = np.random.randn(buffer_size, 1).astype(np.float32)
        buffer.state_values = np.random.randn(buffer_size, 1).astype(np.float32)
        buffer.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        buffer._size = buffer_size

        last_value = np.array([0.0])
        gamma = 0.99
        gae_lambda = 0.95

        buffer.compute_advantages_and_returns(last_value, gamma, gae_lambda)

        assert buffer.advantages.shape == (buffer_size, 1)
        assert buffer.returns.shape == (buffer_size, 1)

    def test_clear(self):
        """Test buffer clearing."""
        buffer = RolloutBuffer(
            buffer_size=10,
            state_dim=4,
            action_dim=2,
            device="cpu",
        )

        for _ in range(5):
            buffer.add(
                np.random.randn(4),
                np.array(1),
                np.array(1.0),
                np.array(0.0),
                np.array(0.5),
                np.array(0.5),
            )

        assert buffer.size == 5
        buffer.clear()
        assert buffer.size == 0
        assert buffer._ptr == 0
