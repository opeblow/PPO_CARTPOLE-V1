"""Configuration management for PPO training."""

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class TrainConfig:
    """Configuration for PPO training with sensible defaults."""

    env_name: str = "CartPole-v1"
    num_envs: int = 1
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    hidden_size: int = 64
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    log_interval: int = 10
    save_interval: int = 50
    target_reward: float = 495.0
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed: int = 42


@dataclass
class Config:
    """Legacy configuration class for backward compatibility."""

    env_name: str = "CartPole-v1"
    num_envs: int = 1
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    hidden_size: int = 64
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    log_interval: int = 10
    save_interval: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "checkpoints"
    log_dir: str = "logs"

    @classmethod
    def from_train_config(cls, config: TrainConfig) -> "Config":
        """Create Config from TrainConfig.

        Args:
            config: TrainConfig instance.

        Returns:
            Config instance.
        """
        return cls(
            env_name=config.env_name,
            num_envs=config.num_envs,
            total_timesteps=config.total_timesteps,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_epsilon=config.clip_epsilon,
            hidden_size=config.hidden_size,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            log_interval=config.log_interval,
            save_interval=config.save_interval,
            device=config.device,
        )
