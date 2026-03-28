"""Tests for PPO config."""

import pytest

from ppo.config import Config, TrainConfig


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainConfig()

        assert config.env_name == "CartPole-v1"
        assert config.num_envs == 1
        assert config.gamma == 0.99
        assert config.learning_rate == 3e-4

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainConfig(
            env_name="LunarLander-v2",
            num_envs=4,
            learning_rate=1e-3,
            n_epochs=5,
        )

        assert config.env_name == "LunarLander-v2"
        assert config.num_envs == 4
        assert config.learning_rate == 1e-3
        assert config.n_epochs == 5


class TestConfig:
    """Tests for legacy Config class."""

    def test_from_train_config(self):
        """Test Config creation from TrainConfig."""
        train_config = TrainConfig(
            env_name="CartPole-v1",
            learning_rate=5e-4,
            hidden_size=128,
        )

        config = Config.from_train_config(train_config)

        assert config.env_name == train_config.env_name
        assert config.learning_rate == train_config.learning_rate
        assert config.hidden_size == train_config.hidden_size
