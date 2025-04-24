#!/usr/bin/env python3
"""
Script to train an Ambient Diffusion model on the CheXpert dataset.
Based on "Does Generation Require Memorization? Creative Diffusion Models using Ambient Diffusion".
"""

import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch

from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from src.data.chexpert_datamodule import CheXpertDataModule  # citeturn0file4
from src.models.ambient_diffusion import AmbientDiffusion   # citeturn1file10
from src.utils.constants import CHEXPERT_CLASSES
from src.utils.progress_visualization_callback import ProgressVisualizationCallback  # citeturn1file3

load_dotenv()

def train_ambient_diffusion(args):
    """
    Train an AmbientDiffusion model for CheXpert with rigorous monitoring.
    """
    print("=== Training Ambient Diffusion Model ===")

    # Reproducibility
    pl.seed_everything(args.seed)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Precision configuration
    precision = "32"
    if torch.cuda.is_available():
        # choose bf16 if supported, else fp16
        if torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"
        torch.set_float32_matmul_precision("medium")

    # Data module
    datamodule = CheXpertDataModule(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        debug_mode=args.debug_mode,
        pin_memory=True,
    )

    # Model
    model = AmbientDiffusion(
        pretrained_model_name_or_path=args.pretrained_model_path,
        img_size=args.img_size,
        in_channels=1,
        out_channels=1,
        num_classes=len(CHEXPERT_CLASSES),
        conditioning_dropout_prob=args.dropout_prob,
        ambient_t_nature=args.ambient_t_nature,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        optimizer_type=args.optimizer_type,
        noise_scheduler_beta_schedule=args.beta_schedule,
        noise_scheduler_num_train_timesteps=args.timesteps,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_update_every=args.ema_update_every,
    )

    # Loggers
    loggers = []
    neptune_api_key = os.environ.get("NEPTUNE_API_KEY")
    neptune_project = os.environ.get("NEPTUNE_PROJECT")
    if neptune_api_key and neptune_project:
        neptune_logger = NeptuneLogger(
            project=neptune_project,
            api_key=neptune_api_key,
            tags=[f"image_size_{args.img_size}", "ambient_diffusion", f"t_nature_{args.ambient_t_nature}"]
        )
        loggers.append(neptune_logger)

    # Callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", "ambient_diffusion"),
        filename="ambient_diffusion-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=True
    )
    callbacks.append(early_stop)

    vis_callback = ProgressVisualizationCallback(
        every_n_epochs=args.vis_every_n_epochs,
        num_samples=args.vis_num_samples,
        guidance_scale=args.guidance_scale,
        inference_steps=args.inference_steps,
        fixed_noise=True
    )
    callbacks.append(vis_callback)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Fit
    trainer.fit(model, datamodule=datamodule)

    # Final checkpoint
    final_ckpt = os.path.join("checkpoints", "ambient_diffusion", "ambient_diffusion_final.ckpt")
    trainer.save_checkpoint(final_ckpt)
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Ambient Diffusion model on CheXpert")

    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/dtu/blackhole/1d/214141/CheXpert-v1.0-small",
        help="Path to CheXpert dataset",
    )

    # Model parameters
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Input image resolution",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Optional path to pretrained diffusion model",
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Warmup steps for LR scheduler",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug mode with smaller dataset",
    )

    # Diffusion parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Diffusion timesteps",
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Noise beta schedule",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability",
    )

    # Ambient diffusion parameter
    parser.add_argument(
        "--ambient_t_nature",
        type=float,
        default=0.5,
        help="Ambient noise threshold factor",
    )

    # EMA parameters
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Enable EMA",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="Exponential moving average decay",
    )
    parser.add_argument(
        "--ema_update_every",
        type=int,
        default=1,
        help="EMA update frequency (steps)",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience epochs",
    )

    # Visualization parameters
    parser.add_argument(
        "--vis_every_n_epochs",
        type=int,
        default=1,
        help="Visualization interval (epochs)",
    )
    parser.add_argument(
        "--vis_num_samples",
        type=int,
        default=1,
        help="Number of samples for visualization",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
        help="Inference steps for sampling",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="CFG guidance scale for visualization",
    )

    # Logging and accumulation
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="Logging frequency in steps",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation over n batches",
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs("checkpoints/ambient_diffusion", exist_ok=True)
    os.makedirs("logs/ambient_diffusion", exist_ok=True)

    # Launch training
    train_ambient_diffusion(args)
    print("Training completed!")


if __name__ == "__main__":
    main()