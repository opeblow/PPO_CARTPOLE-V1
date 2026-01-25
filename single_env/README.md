# PPO ON CARTPOLE-V1-SINGLE ENVIRONMENT
A focus implementation of Proximal Policy Optimization (PPO) to study reinforcement learning fundamentals and training dynamics in a single-environment setting.

## OVERVIEW
This repository contains a from scratch implementation of Proximal Policy Optimization(PPO)applied to the cartpole-v1 environment using a single environment setup.

### THE GOAL OF THIS PROJECT :
    - Validate a correct PPO implementation.
    - Analyze learning dynamics and stability.
    - Establish a clean baseline before scaling to vectorized environments.
    This single-environment version serves as the foundation for further scalability experiments.

### WHY SINGLE-ENVIRONMENT PPO?
Training with a single environment allows  precise inspection of :
    - Policy improvement behavior
    - Reward variance and exploration effects
    - Update stability under one policy learning
    By removing parallelism,the learning signal becomes easier to interpret,making this setup ideal for validating algorithmic correctness.


## IMPLEMENTATION DETAILS
Algorithm:Proximal Policy Optimization (PPO)
Environment:Cartpole-v1
Networks:
    - Shared feature extractor
    - Separate policy and value heads
Advantage Estimation:Generalized Advantage Estimation (GAE)
Optimization:PPO clipped surrogate objective
Training Loop:
  - Rollout Collection
  - Advantage Normalization
  - Mini-batch policy updates
  - Periodic Evaluation
All components are implemented explicitly without reliance on high-level RL libraries.

## EXPERIMENTAL SETUP
- Environment Instances:1 (single-env)
- Total training timestamps:1e6
- Evaluation performed separately from training
This configuration ensures clear attribution between algorithim behavior and observed results.


## RESULTS
- Average Episode Reward: 428
- Average Episode Length:428
Training curves demonstrate:
- Rapid convergence from random behavior.
- Sustained high reward performance
- Occasional variance due to stochastic exploration (expected in PPO)
These characteristics indicate a stable and correctly functioning PPO agent.


