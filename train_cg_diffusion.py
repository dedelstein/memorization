#!/usr/bin/env python3
"""
Script to train a Classifier-Free Guided Diffusion model on the CheXpert dataset.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import NeptuneLogger

from src.data.chexpert_datamodule import CheXpertDataModule
from src.models.cfg_diffusion import ClassifierFreeGuidedDiffusion
from src.utils.constants import CHEXPERT_CLASSES


def train_cfg_diffusion(args):
    """
    Train a Classifier-Free Guided Diffusion model for CheXpert.

    Args:
        args: Command line arguments
    """
    print("=== Training Classifier-Free Guided Diffusion Model ===")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data module
    datamodule = CheXpertDataModule(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_frontal_only=True,
        seed=args.seed,
        debug_mode=args.debug_mode,
    )

    # Create diffusion model
    model = ClassifierFreeGuidedDiffusion(
        pretrained_model_name_or_path=args.pretrained_model_path
        if args.pretrained_model_path
        else None,
        img_size=args.img_size,
        in_channels=3,
        out_channels=3,
        num_classes=len(CHEXPERT_CLASSES),
        conditioning_dropout_prob=args.dropout_prob,
        lr=args.lr,
        lr_warmup_steps=args.warmup_steps,
        noise_scheduler_beta_schedule=args.beta_schedule,
        noise_scheduler_num_train_timesteps=args.timesteps,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_update_every=10,  # Valor predeterminado adecuado
    )

    # Set up logging with Neptune instead of TensorBoard
    logger = NeptuneLogger(
        project=os.environ.get("NEPTUNE_PROJECT"),
        api_key=os.environ.get("NEPTUNE_API_KEY"),
        log_model_checkpoints=os.environ.get(
            "NEPTUNE_LOG_MODEL_CHECKPOINTS", "False"
        ).lower()
        == "true",
        tags=[f"img_size={args.img_size}", "cfg_diffusion"],
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", "cfg_diffusion"),
        filename="cfg_diffusion-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")

    # Sample generation callback - custom implementation to periodically generate and log samples
    class SampleCallback(pl.Callback):
        def __init__(self, every_n_epochs=args.vis_every_n_epochs, num_samples=args.vis_num_samples):
            super().__init__()
            self.every_n_epochs = every_n_epochs
            self.num_samples = num_samples

        def on_epoch_end(self, trainer, pl_module):
            if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
                # Create labels for "No Finding" and "Pneumonia"
                normal_label = torch.zeros(
                    (self.num_samples, len(CHEXPERT_CLASSES)), device=pl_module.device
                )
                normal_label[:, CHEXPERT_CLASSES.index("No Finding")] = 1.0

                pneumonia_label = torch.zeros(
                    (self.num_samples, len(CHEXPERT_CLASSES)), device=pl_module.device
                )
                pneumonia_label[:, CHEXPERT_CLASSES.index("Pneumonia")] = 1.0

                # Generate samples
                normal_samples = pl_module.generate_samples(
                    batch_size=self.num_samples,
                    labels=normal_label,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.inference_steps,
                )

                pneumonia_samples = pl_module.generate_samples(
                    batch_size=self.num_samples,
                    labels=pneumonia_label,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.inference_steps,
                )

                # Convert to numpy and normalize to [0, 1]
                normal_grid = torch.stack(
                    [img.float() / 255.0 for img in normal_samples]
                )
                pneumonia_grid = torch.stack(
                    [img.float() / 255.0 for img in pneumonia_samples]
                )

                # Log to Neptune
                if trainer.logger:
                    # Neptune requires a different logging approach than TensorBoard
                    for i in range(len(normal_samples)):
                        trainer.logger.experiment[
                            f"samples/normal/{trainer.current_epoch}/sample_{i}"
                        ].upload(normal_grid[i].permute(1, 2, 0).cpu().numpy())
                        trainer.logger.experiment[
                            f"samples/pneumonia/{trainer.current_epoch}/sample_{i}"
                        ].upload(pneumonia_grid[i].permute(1, 2, 0).cpu().numpy())

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        precision=16
        if torch.cuda.is_available()
        else 32,  # Use mixed precision if available
        logger=logger,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            early_stop_callback,
            SampleCallback(every_n_epochs=args.vis_every_n_epochs),
        ],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,  # Acumulación de gradiente para simular lotes más grandes
    )

    # Train model
    trainer.fit(model, datamodule=datamodule)

    # Save final model
    trainer.save_checkpoint(
        os.path.join("checkpoints", "cfg_diffusion", "cfg_diffusion_final.ckpt")
    )

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
    parser.add_argument("--img_size", type=int, default=64, help="Image size")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to pretrained diffusion model (optional)",
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Learning rate warmup steps"
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
    
    # EMA parameters
    parser.add_argument("--use_ema", type=bool, default=True, help="Whether to use EMA")
    parser.add_argument(
        "--ema_decay", type=float, default=0.9999, help="EMA decay rate"
    )
    
    # Early stopping
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    
    # Visualization parameters
    parser.add_argument('--vis_every_n_epochs', type=int, default=5, 
                        help='Generate progress images every N epochs')
    parser.add_argument('--vis_num_samples', type=int, default=4, 
                        help='Number of samples to generate per condition')
    parser.add_argument('--inference_steps', type=int, default=20, 
                        help='Inference steps for generating samples')
    parser.add_argument('--guidance_scale', type=float, default=3.0, 
                        help='Guidance strength for generation')

    # Gradient accumulation
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients for",
    )

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("checkpoints/cfg_diffusion", exist_ok=True)
    os.makedirs("logs/cfg_diffusion", exist_ok=True)

    # Train classifier-free guided diffusion model
    train_cfg_diffusion(args)

    print("Training completed!")


if __name__ == "__main__":
    main()
