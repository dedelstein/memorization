#!/usr/bin/env python3
"""
Script to train a classifier for CheXpert data.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

from src.data.chexpert_datamodule import CheXpertDataModule
from src.models.classifier import ClassifierModule
from src.utils.constants import CHEXPERT_CLASSES


def train_classifier(args):
    """
    Train a classifier for CheXpert.
    
    Args:
        args: Command line arguments
    """
    print("=== Training Classifier ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data module
    datamodule = CheXpertDataModule(
        data_dir=args.data_dir, 
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_frontal_only=True,
        seed=args.seed,
        debug_mode=args.debug_mode
    )
    
    # Create classifier model
    model = ClassifierModule(
        img_size=args.img_size,
        c_in=1,  # Changed from 3 to 1
        num_labels=len(CHEXPERT_CLASSES),
        lr=args.lr,
        channels=args.channels,
        time_dim=args.time_dim
    )
    
    # Set up logging with Neptune instead of TensorBoard
    logger = NeptuneLogger(
        project=os.environ.get("NEPTUNE_PROJECT"),
        api_key=os.environ.get("NEPTUNE_API_KEY"),
        log_model_checkpoints=os.environ.get("NEPTUNE_LOG_MODEL_CHECKPOINTS", "False").lower() == "true",
        tags=[f"img_size={args.img_size}"]
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('checkpoints', 'classifier'),
        filename='classifier-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",  # This will automatically detect the available hardware
        devices=1 if torch.cuda.is_available() else 1,  # Use 1 device regardless of hardware type
        precision="16-mixed" if torch.cuda.is_available() else 32,  # Use mixed precision only on GPU
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(model, datamodule=datamodule)
    
    # Save final model
    trainer.save_checkpoint(os.path.join('checkpoints', 'classifier', 'classifier_final.ckpt'))
    
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train a classifier on CheXpert")
    
    # Data parameters
    parser.add_argument(
        "--data_dir",
        default="/dtu/blackhole/1d/214141/CheXpert-v1.0-small",
        type=str,
        help="Path to CheXpert-v1.0-small folder with train.csv and valid.csv",
    )
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--channels', type=int, default=64, help='Base channel multiplier')
    parser.add_argument('--time_dim', type=int, default=256, help='Time embedding dimension')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug_mode', action='store_true', help='Use a tiny subset of data for debugging')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('checkpoints/classifier', exist_ok=True)
    os.makedirs('logs/classifier', exist_ok=True)
    
    # Train classifier
    train_classifier(args)
    
    print("Training completed!")


if __name__ == "__main__":
    main()