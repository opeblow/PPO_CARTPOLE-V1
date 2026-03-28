# PPO CartPole-v1

![Logo](docs/logo.svg)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-purple)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Production-ready Proximal Policy Optimization (PPO) implementation for reinforcement learning research on CartPole-v1.

## Overview

A clean, well-documented PPO implementation featuring both single-environment and vectorized multi-environment training modes. Designed for research, education, and production use with comprehensive type hints and Google-style documentation.

### Key Features

- **Dual Training Modes**: Single-environment (debugging) and multi-environment (production)
- **Vectorized Environments**: 4x faster data collection with parallel environments
- **GAE Advantage Estimation**: Variance-reduced advantage computation
- **Real-time Monitoring**: Live training metrics with progress bars
- **Automatic Checkpointing**: Periodic saves with early stopping
- **Type-Safe**: Full type hints and Google-style docstrings
- **Tested**: Unit tests with pytest

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single environment training (fast, for learning)
python -m ppo.train --num_envs 1 --total_timesteps 500000

# Multi-environment training (4x faster)
python -m ppo.train --num_envs 4 --total_timesteps 500000

# Evaluate a trained model
python -m ppo.evaluate --model_path checkpoints/ppo_solved.pth --num_episodes 100
```

## Installation

```bash
# Clone repository
git clone https://github.com/username/PPO_CARTPOLE-V1.git
cd ppo-cartpole

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Project Structure

```
ppo-cartpole/
├── src/ppo/              # Main package
│   ├── __init__.py       # Package exports
│   ├── config.py         # Configuration dataclasses
│   ├── network.py        # Actor-Critic neural network
│   ├── buffer.py         # Rollout buffer with GAE
│   ├── agent.py          # PPO agent implementation
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── tests/                # Unit tests
├── docs/                 # Documentation assets
├── configs/              # YAML configs
├── README.md             # This file
├── requirements.txt      # Dependencies
└── pyproject.toml        # Project metadata
```

## Algorithm

This implementation follows the [PPO paper](https://arxiv.org/abs/1707.06347) with:

- **Clipped Surrogate Objective**: Prevents destructive policy updates
- **Generalized Advantage Estimation (GAE)**: Variance-reduced advantage computation
- **Actor-Critic Architecture**: Combined policy and value function learning
- **Entropy Regularization**: Encourages exploration

### Network Architecture

```
State (4) → [Linear 64] → [ReLU] → [Linear 64] → [ReLU] → Actor Head (2) + Critic Head (1)
```

## Configuration

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_envs` | 1 | Number of parallel environments |
| `--total_timesteps` | 500000 | Total training steps |
| `--learning_rate` | 3e-4 | Optimizer learning rate |
| `--n_steps` | 2048 | Steps per update |
| `--batch_size` | 64 | Batch size for updates |
| `--n_epochs` | 10 | Epochs per update |
| `--target_reward` | 495.0 | Early stopping threshold |
| `--seed` | 42 | Random seed |

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_epsilon` | 0.2 | PPO clipping range |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |

## Usage Examples

### Single Environment Training

```python
from ppo.train import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    args = parser.parse_args()
    
    main(args)
```

### Programmatic Training

```python
import gymnasium as gym
from ppo.agent import PPOAgent
from ppo.config import TrainConfig

config = TrainConfig(
    env_name="CartPole-v1",
    num_envs=4,
    total_timesteps=500_000,
)
env = gym.make_vec(config.env_name, num_envs=config.num_envs)
# ... training loop
```

### Evaluation

```python
from ppo.evaluate import evaluate

stats = evaluate(
    model_path="checkpoints/ppo_solved.pth",
    env_name="CartPole-v1",
    num_episodes=100,
)
print(f"Mean Reward: {stats['mean_reward']:.2f}")
```

## Performance

- **Sample Efficiency**: ~100K steps to solve CartPole-v1
- **Training Time**: ~3-5 minutes (multi-env) / ~10-15 minutes (single-env)
- **Success Rate**: Consistently achieves 495+ average reward

```
Training Progress: 100%|██████████| 200/200 [02:30<00:00, Avg R: 497.23]
Target reward (495.0) reached!
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/ppo --cov-report=term-missing

# Run specific test file
pytest tests/test_network.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

```bibtex
@software{ppo_cartpole_2026,
  author = {Mobolaji Opeyemi Bolatito },
  title = {PPO CartPole-v1: Production-Ready PPO Implementation},
  year = {2026},
  url = {https://github.com/opeblow/PPO_CARTPOLE-V1.git
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with PyTorch** | **Battle-Tested** | **Production-Ready**
