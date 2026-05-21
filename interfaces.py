STATE_DIM  = 8
EMBED_DIM  = 64
NUM_PHASES = 4
EDGE_DIM   = 2
 
DEFAULT_CONFIG = {
    "lr": 3e-4, "gamma": 0.99, "batch_size": 64,
    "total_steps": 300_000, "warmup_steps": 1_000, "checkpoint_freq": 5_000,
    "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 10_000,
    "target_update_freq": 200, "buffer_capacity": 50_000,
    "ppo_epochs": 4, "ppo_clip": 0.2, "gae_lambda": 0.95,
    "entropy_coef": 0.01, "value_coef": 0.5, "rollout_steps": 2048,
    "delta_time": 10, "episode_seconds": 3600, "num_intersections": 16,
    "embed_dim": EMBED_DIM, "gat_heads": 4, "gat_layers": 2,
    "hidden_dim": 128, "dropout": 0.1,
}
