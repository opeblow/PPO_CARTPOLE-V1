# Multi-Environment PPO Training

A high-performance Proximal Policy Optimization (PPO) implementation with vectorized environments for accelerated reinforcement learning training.

## Features

- *Vectorized Training*: Parallel environment execution for 4x faster data collection
- *Modular Architecture*: Clean separation of concerns across components
- *Real-time Monitoring*: Live episode statistics and training metrics
- *Automatic Checkpointing*: Periodic model saves with early stopping
- *Performance Visualization*: Automated plotting of training curves
- *Built-in Evaluation*: Dedicated evaluation module for model testing

## Quick Start
bash
# Install dependencies
pip install gymnasium numpy matplotlib tqdm torch

# Run training
python - multiple_env.train 



## Architecture

multiple_env/
├── config.py           # Hyperparameters and training configuration
├── network.py          # Actor-Critic neural network architecture
├── buffer.py           # Rollout buffer for experience storage
├── ppo_agent.py        # PPO algorithm implementation
├── train.py            # Vectorized training loop
└── evaluate.py         # Model evaluation and testing


## Core Components

*network.py* - Policy and value function approximators  
*buffer.py* - Experience replay with GAE computation  
*ppo_agent.py* - PPO update logic and optimization  
*train.py* - Parallel environment training pipeline  
*evaluate.py* - Performance benchmarking utilities  

## Configuration

Examples of Key parameters in config.py:
python
env_name = "CartPole-v1"    # Gymnasium environment
num_envs = 4                 # Parallel environments
total_updates = 1000         # Training iterations
n_steps = 128                # Steps per update
learning_rate = 3e-4         # Optimizer learning rate
gamma = 0.99                 # Discount factor
gae_lambda = 0.95            # GAE parameter
clip_epsilon = 0.2           # PPO clipping range


## Training Output

*Real-time metrics:*

PPO Training: 35%|████████| 350/1000 [01:45<03:15, Avg R: 497.23, Avg L: 497.2]
Update 334 | Ep Reward: 498.00 | Ep Length: 498
Target reward reached! Saving final model...


*Generated artifacts:*
- logs/training_logs.json - Complete training history
- logs/final_performance.png - Training curves visualization
- checkpoints/ppo_checkpoint_*.pth - Periodic saves
- checkpoints/ppo_final_solved.pth - Best model

## Performance

- *Data Efficiency*: 4x sample throughput via vectorization
- *Training Time*: ~5 minutes to solve CartPole-v1
- *Success Criteria*: 495+ average reward over 10 episodes
- *Sample Complexity*: ~100K environment steps

## Results Visualization

Training curves show reward progression and episode length stability:

![Training Performance](logs/final_performance.png)

## Technical Details

*Algorithm*: Proximal Policy Optimization (PPO)
- Clipped surrogate objective for stable policy updates
- Generalized Advantage Estimation (GAE) for variance reduction
- Value function loss with gradient clipping
- Entropy bonus for exploration

*Vectorization*: Gymnasium synchronous vector environments
- RecordEpisodeStatistics wrapper for metric tracking
- Efficient batch processing across parallel instances

*Network Architecture*: Shared feature extraction with separate heads
- Actor: Policy network (state → action probabilities)
- Critic: Value network (state → state value estimate)



## Requirements

python>=3.8
torch>=1.10
gymnasium>=0.29
numpy
matplotlib
tqdm


## Project Structure

.
├── multiple_env/
│   ├── __init__.py
│   ├── config.py
│   ├── network.py
│   ├── buffer.py
│   ├── ppo_agent.py
│   ├── train.py
│   └── evaluate.py
├── logs/
├── checkpoints/
└── README.md






## Citation
  author = {Your Name},
  title = {Multi-Environment PPO Training},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/opeblow/PPO_CARTPOLE-V1.git}





*Built with PyTorch • Optimized for Production • Ready to Scale*
