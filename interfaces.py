STATE_DIM  = 8
EMBED_DIM  = 64
NUM_PHASES = 4
EDGE_DIM   = 2
 
DEFAULT_CONFIG = {
    # Environment
    "num_intersections": 16,
    "episode_seconds":   3600,
    "delta_time":        10,
    "cityflow_threads":  4,

    # Training
    "lr":                3e-4,
    "gamma":             0.99,
    "batch_size":        64,
    "total_steps":       300_000,
    "warmup_steps":      1_000,

    # DQN  
    "replay_capacity":   50_000,   
    "target_update_freq":200,
    "train_freq":        4,

    "epsilon_start":     1.0,
    "epsilon_end":       0.05,
    "epsilon_decay":     10_000,

    # PPO
    "rollout_steps":     2048,
    "ppo_epochs":        10,
    "clip_eps":          0.2,
    "vf_coef":           0.5,
    "ent_coef":          0.01,
    "gae_lambda":        0.95,
    "max_grad_norm":     0.5,

    # GNN
    "embed_dim":         64,
    "gat_heads":         4,
    "gat_layers":        2,
    "hidden_dim":        128,
    "dropout":           0.1,

    # Checkpointing
    "save_freq":         5_000,
    "eval_freq":         10_000,
    "eval_episodes":     3,
}