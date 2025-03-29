CONFIG = {
    "in_channels": 3,
    "base_channels": 64,
    "time_dim": 256,
    "num_classes": 14,
    "beta_schedule": "cosine",
    "diffusion_steps": 1000,
    "ema_decay": 0.9999,
    "amp": True,
    "image_size": 32,
    "use_conditional_unet": True,
    "guidance_method": "classifier-free",  # "classifier-free" or "classifier"
    "data_dir": "./data"
}