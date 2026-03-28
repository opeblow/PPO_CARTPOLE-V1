r"""Evaluation script for trained PPO agent.

Usage:
    python -m ppo.evaluate --model_path checkpoints/ppo_solved.pth --num_episodes 100
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from ppo.agent import PPOAgent
from ppo.config import TrainConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment name",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def evaluate(
    model_path: str,
    env_name: str,
    num_episodes: int,
    render: bool = False,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate a trained PPO agent.

    Args:
        model_path: Path to the saved model.
        env_name: Name of the environment.
        num_episodes: Number of episodes to evaluate.
        render: Whether to render the environment.
        seed: Random seed.

    Returns:
        Dictionary containing evaluation statistics.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    config = TrainConfig(env_name=env_name, device="cpu")
    env = gym.make(env_name, render_mode="human" if render else None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim, config)
    agent.load(model_path)
    agent._policy.eval()

    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action, _, _ = agent.select_action(state_tensor)
            action_np = action.item() if agent._num_envs == 1 else action[0].item()

            next_state, reward, term, trunc, _ = env.step(action_np)
            episode_reward += reward
            episode_length += 1
            state = next_state
            done = term or trunc

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "solved": np.mean(episode_rewards) >= 195.0,
    }


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print(f"Evaluating: {args.model_path}")
    print(f"Environment: {args.env_name}")
    print(f"Episodes: {args.num_episodes}")

    stats = evaluate(
        args.model_path,
        args.env_name,
        args.num_episodes,
        args.render,
        args.seed,
    )

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Min/Max Reward: {stats['min_reward']:.0f} / {stats['max_reward']:.0f}")
    print(f"Mean Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"Solved (≥195): {'✓ Yes' if stats['solved'] else '✗ No'}")


if __name__ == "__main__":
    main()
