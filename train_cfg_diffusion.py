#!/usr/bin/env python3
"""
Script to train a Classifier-Free Guided Diffusion model on the CheXpert dataset.
Updated to work with the simplified ClassifierFreeGuidedDiffusion implementation.
"""

import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch

# Load .env file
from dotenv import load_dotenv
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

from src.data.chexpert_datamodule import CheXpertDataModule
from src.models.cfg_diffusion import ClassifierFreeGuidedDiffusion
from src.utils.constants import CHEXPERT_CLASSES
from src.utils.progress_visualization_callback import ProgressVisualizationCallback

load_dotenv()


def train_cfg_diffusion(args):
    """
    Train a Classifier-Free Guided Diffusion model for CheXpert.

    Args:
        args: Command line arguments
    """
    print("=== Training Classifier-Free Guided Diffusion Model ===")

    # Configure tensor computation precision
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Create data module
    datamodule = CheXpertDataModule(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        debug_mode=args.debug_mode,
        pin_memory=True,
    )

    # Create simplified diffusion model
    model = ClassifierFreeGuidedDiffusion(
        pretrained_model_name_or_path=args.pretrained_model_path
        if args.pretrained_model_path
        else None,
        img_size=args.img_size,
        num_classes=len(CHEXPERT_CLASSES),
        conditioning_dropout_prob=args.dropout_prob,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        optimizer_type=args.optimizer_type,
        
        # Scheduler parameters
        lr_scheduler_type=args.lr_scheduler_type,
        min_lr=args.min_lr,
        lr_num_cycles=args.lr_num_cycles,
        
        # Noise scheduler parameters
        noise_scheduler_beta_schedule=args.beta_schedule,
        noise_scheduler_num_train_timesteps=args.timesteps,
    )

    # Set up logging
    loggers = []
    
    # Neptune Logger (if credentials are available)
    neptune_api_key = os.environ.get("NEPTUNE_API_KEY")
    neptune_project = os.environ.get("NEPTUNE_PROJECT")
    
    if neptune_api_key and neptune_project:
        neptune_logger = NeptuneLogger(
            project=neptune_project,
            api_key=neptune_api_key,
            tags=[f"image_size_{args.img_size}", "cfg_diffusion", args.lr_scheduler_type],
        )
        loggers.append(neptune_logger)
    
    # TensorBoard Logger (always available)
    tensorboard_logger = TensorBoardLogger(
        save_dir="logs",
        name="cfg_diffusion",
        version=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    loggers.append(tensorboard_logger)

    # Set up callbacks
    callbacks = []
    
    # Custom checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", "cfg_diffusion"),
        filename="cfg_diffusion-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,  # Only save the best model to reduce storage usage
        monitor="val_loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # LR Monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=args.patience, mode="min", verbose=True
    )
    callbacks.append(early_stop_callback)

    # Progress Visualization
    vis_callback = ProgressVisualizationCallback(
        every_n_epochs=args.vis_every_n_epochs,
        num_samples=args.vis_num_samples,
        guidance_scale=args.guidance_scale,
        inference_steps=args.inference_steps,
        fixed_noise=True  # Use the same initial noise to compare evolution
    )
    callbacks.append(vis_callback)

    # Use appropriate precision based on hardware
    precision = "32"
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            precision = "bf16-mixed"
        else:
            precision = "16-mixed"

    # Set up trainer with optimized parameters
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",  # This will automatically detect the available hardware
        devices=1,
        precision=precision,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)

    print(f"Best model path: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train a Classifier-Free Guided Diffusion model on CheXpert"
    )

    # Data parameters
    parser.add_argument(
        "--data_dir",
        default="/dtu/blackhole/1d/214141/CheXpert-v1.0-small",
        type=str,
        help="Path to CheXpert-v1.0-small folder with train.csv and valid.csv",
    )

    # Model parameters
    parser.add_argument("--img_size", type=int, default=192, help="Image size")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to pretrained diffusion model (optional)",
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=500, 
        help="Number of warmup steps for learning rate scheduler"
    )
    
    # Scheduler parameters
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine_with_warmup",
        choices=["constant_with_warmup", "cosine_with_warmup", "one_cycle", "reduce_on_plateau"],
        help="Type of learning rate scheduler to use",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for schedulers that support it",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles for cosine scheduler with restarts",
    )
    
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamW",
        choices=["adam", "adamw"],
        help="Optimizer type to use",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Use a tiny subset of data for debugging",
    )

    # Diffusion parameters
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Beta schedule",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.1,
        help="Probability of dropping class conditioning (for CFG training)",
    )

    # Early stopping
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )

    # Visualization parameters
    parser.add_argument(
        "--vis_every_n_epochs",
        type=int,
        default=1,
        help="Generate progress images every N epochs",
    )
    parser.add_argument(
        "--vis_num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per condition",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=25,
        help="Inference steps for generating samples",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Guidance strength for generation",
    )

    # Gradient accumulation
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=5,
        help="Number of batches to accumulate gradients for",
    )

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("checkpoints/cfg_diffusion", exist_ok=True)
    os.makedirs("logs/cfg_diffusion", exist_ok=True)

    # Set seed for reproducibility
    pl.seed_everything(args.seed)

    # Train classifier-free guided diffusion model
    train_cfg_diffusion(args)

    print("Training completed!")


if __name__ == "__main__":
    main()
