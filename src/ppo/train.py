r"""Training script for PPO agent on CartPole-v1.

Usage:
    python -m ppo.train --num_envs 4 --total_timesteps 500000

    python -m ppo.train  # Single environment mode
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ppo.agent import PPOAgent
from ppo.config import TrainConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on CartPole-v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment name",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments (1 for single-env mode)",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Number of steps per update",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for updates",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs per update",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Hidden layer size",
    )
    parser.add_argument(
        "--target_reward",
        type=float,
        default=495.0,
        help="Target average reward for early stopping",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save models",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def plot_results(
    rewards: list[float],
    lengths: list[int],
    save_path: str,
) -> None:
    """Plot training results.

    Args:
        rewards: List of episode rewards.
        lengths: List of episode lengths.
        save_path: Path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rewards, alpha=0.3, color="blue", label="Raw Reward")
    avg_r = [
        np.mean(rewards[max(0, i - 10) : i + 1]) for i in range(len(rewards))
    ]
    ax1.plot(avg_r, color="blue", linewidth=2, label="Moving Avg (10)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(lengths, alpha=0.3, color="green", label="Raw Length")
    avg_l = [
        np.mean(lengths[max(0, i - 10) : i + 1]) for i in range(len(lengths))
    ]
    ax2.plot(avg_l, color="green", linewidth=2, label="Moving Avg (10)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")
    ax2.set_title("Episode Length")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(config: TrainConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Train the PPO agent.

    Args:
        config: Training configuration.
        args: Command line arguments.

    Returns:
        Dictionary containing training statistics.
    """
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    if args.num_envs > 1:
        env = gym.make_vec(
            config.env_name,
            num_envs=config.num_envs,
            vectorization_mode="sync",
            disable_env_checker=True,
        )
        env = gym.wrappers.vector.RecordEpisodeStatistics(env)
    else:
        env = gym.make(config.env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)

    state_dim = (
        env.single_observation_space.shape[0]
        if args.num_envs > 1
        else env.observation_space.shape[0]
    )
    action_dim = (
        env.single_action_space.n
        if args.num_envs > 1
        else env.action_space.n
    )

    print(
        f"Environment: {config.env_name}, "
        f"State Dim: {state_dim}, Action Dim: {action_dim}, "
        f"Num Envs: {args.num_envs}"
    )

    agent = PPOAgent(state_dim, action_dim, config)

    state, _ = env.reset(seed=args.seed)
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    total_updates = args.total_timesteps // (config.n_steps * max(1, args.num_envs))
    pbar = tqdm(range(total_updates), desc="PPO Training")

    start_time = time.time()

    for update in pbar:
        for _ in range(config.n_steps):
            state_tensor = (
                torch.from_numpy(state).float().to(config.device)
            )
            action, log_prob, value = agent.select_action(state_tensor)

            if args.num_envs > 1:
                action_np = action.cpu().numpy().astype(np.int64).flatten()
                next_state, reward, term, trunc, info = env.step(action_np)
                done = np.logical_or(term, trunc)
                value_np = value.cpu().numpy().flatten()
                log_prob_np = log_prob.cpu().numpy()
            else:
                action_np = int(action.item())
                next_state, reward, term, trunc, info = env.step(action_np)
                done = bool(term or trunc)
                value_np = float(value.item())
                log_prob_np = float(log_prob.item())

            agent._buffer.add(
                state, action_np, reward, done, log_prob_np, value_np
            )
            state = next_state

            if "_episode" in info:
                episode_mask = info["_episode"]
                for idx in range(args.num_envs):
                    if episode_mask[idx]:
                        r = float(info["episode"]["r"][idx])
                        l = int(info["episode"]["l"][idx])
                        episode_rewards.append(r)
                        episode_lengths.append(l)
                        pbar.write(
                            f"Update {update} | Ep {len(episode_rewards)}: "
                            f"R={r:.1f}, L={l}"
                        )

        next_state_tensor = (
            torch.from_numpy(next_state).float().to(config.device)
        )
        agent.update(next_state_tensor)

        if episode_rewards:
            avg_r = np.mean(episode_rewards[-10:])
            avg_l = np.mean(episode_lengths[-10:])
            pbar.set_postfix({"Avg R": f"{avg_r:.1f}", "Avg L": f"{avg_l:.0f}"})

            if avg_r >= args.target_reward:
                pbar.write(
                    f"Target reward ({args.target_reward}) reached! "
                    f"Saving model..."
                )
                agent.save(f"{args.save_dir}/ppo_solved.pth")
                break

        if update % 20 == 0:
            agent.save(f"{args.save_dir}/ppo_checkpoint_{update}.pth")

    env.close()

    total_time = time.time() - start_time

    results = {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "total_time": total_time,
        "final_avg_reward": (
            np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
        ),
    }

    with open(f"{args.log_dir}/training_logs.json", "w") as f:
        json.dump(
            {
                "rewards": episode_rewards,
                "lengths": episode_lengths,
                "total_time_seconds": total_time,
            },
            f,
            indent=2,
        )

    if episode_rewards:
        plot_results(
            episode_rewards,
            episode_lengths,
            f"{args.log_dir}/training_curves.png",
        )

    agent.save(f"{args.save_dir}/ppo_final.pth")

    return results


def main() -> None:
    """Main entry point."""
    args = parse_args()

    config = TrainConfig(
        env_name=args.env_name,
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        hidden_size=args.hidden_size,
        target_reward=args.target_reward,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
    )

    set_seed(args.seed)

    print(f"Training PPO on {args.env_name}")
    print(f"Device: {config.device}")
    print(f"Num Envs: {args.num_envs}")

    results = train(config, args)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Total Episodes: {len(results['rewards'])}")
    print(f"Final Avg Reward (last 10): {results['final_avg_reward']:.2f}")
    print(f"Total Time: {results['total_time']:.1f}s")
    print(f"\nModels saved to: {args.save_dir}/")
    print(f"Logs saved to: {args.log_dir}/")


if __name__ == "__main__":
    main()
