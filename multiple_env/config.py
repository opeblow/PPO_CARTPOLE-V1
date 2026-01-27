import torch
class Config:
    env_name="CartPole-v1"
    num_envs=4
    total_timesteps=100000
    learning_rate=3e-4
    gamma=0.99
    gae_lambda=0.95
    clipson_epsilon=0.2
    hidden_size=64
    n_steps=2048
    batch_size=64
    total_updates=100
    n_steps=2048
    n_epochs=10
    entropy_coef=0.01
    max_grad_norm=0.5
    log_interval=10
    save_interval=50
    device="cuda" if torch.cuda.is_available() else "cpu"
    save_dir="checkpoints/"
    log_dir="logs"
    