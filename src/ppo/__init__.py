"""PPO: Proximal Policy Optimization for Reinforcement Learning.

A production-ready PPO implementation for CartPole-v1 and other Gymnasium environments.
"""

import sys
from pathlib import Path

if Path(__file__).parent.parent.parent not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from ppo.agent import PPOAgent
from ppo.config import Config, TrainConfig
from ppo.network import ActorCriticNetwork
from ppo.buffer import RolloutBuffer

__all__ = [
    "PPOAgent",
    "Config",
    "TrainConfig",
    "ActorCriticNetwork",
    "RolloutBuffer",
]
