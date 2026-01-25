import torch
class Config:
    env_name="CartPole-v1"
    num_env=1
    total_timesteps=1000000
    learning_rate=3e-4
    gamma=0.99
    gae_lambda=0.95
    clipson_epsilon=0.2
    hidden_size=64
    n_steps=2048
    batch_size=128
    n_epochs=10
    entropy_coef=0.005
    max_grad_norm=0.5
    log_interval=10
    save_interval=50
    device="cuda" if torch.cuda.is_available() else "cpu"
    save_dir="checkpoints/"
    log_dir="logs"
