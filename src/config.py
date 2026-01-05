config = {
    "num_episodes": 2000,
    "max_steps_per_episode": 200,
    "gamma": 0.99,

    # Epsilon-greedy decay
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_steps": 1000,

    # Optimizer and learning rate
    "learning_rate": 1e-3,
    "use_dueling": True,

    # Experience replay
    "batch_size": 64,
    "warmup_steps": 1000,
    "replay_buffer_size": 10000,

    # Target network update
    "target_update_freq": 10,   # used only if using hard update (optional)
    "tau": 0.005,               # for soft updates (Polyak averaging)

    # Misc
    "grad_clip_norm": 1.0,
    "moving_avg_window": 50,

    # Reward shaping
    "target_threshold": 0.1,
    "target_bonus": 10.0
}
