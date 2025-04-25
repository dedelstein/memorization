import argparse
import os
from datetime import datetime
from dotenv import load_dotenv
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.chexpert_datamodule import CheXpertDataModule
from src.models.cfg_diffusion import ClassifierFreeGuidedDiffusion
from src.utils.constants import CHEXPERT_CLASSES
from src.utils.progress_visualization_callback import ProgressVisualizationCallback


torch.set_float32_matmul_precision("medium")
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a classifier-free guided diffusion model using PyTorch Lightning"
    )

    # Model parameters
    parser.add_argument(
        "--sample_size", type=int, default=224, help="The size of the generated images"
    )
    
    # Optimizer parameters
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="The learning rate scheduler type",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.95,
        help="The beta1 parameter for the Adam optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay to use"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )

    # EMA parameters
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the model",
    )
    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="The inverse gamma value for the EMA decay",
    )
    parser.add_argument(
        "--ema_power",
        type=float,
        default=0.75,
        help="The power value for the EMA decay",
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.9999,
        help="The maximum decay value for EMA",
    )

    # Diffusion parameters
    parser.add_argument(
        "--ddpm_num_steps", type=int, default=1000, help="Number of steps for DDPM"
    )
    parser.add_argument(
        "--ddpm_beta_schedule",
        type=str,
        default="linear",
        help="The beta schedule type for DDPM",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="The prediction type for the model",
    )

    # Classifier-free guidance parameters
    parser.add_argument(
        "--class_cond", action="store_true", help="Whether to use class conditioning"
    )
    parser.add_argument(
        "--unconditional_probability",
        type=float,
        default=0.1,
        help="The probability of training on unconditional samples",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="The scale for classifier-free guidance",
    )

    # Performance optimization
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether to enable memory efficient attention",
    )

    # Training parameters
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )

    # Data parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default="CheXpert-v1.0-small",# /dtu/blackhole/1d/214141/CheXpert-v1.0-small
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="The size of the input images"
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Use debug mode with a small subset of data",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data splitting"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Use pin memory for data loading",
    )

    # Logging parameters
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/cfg_diffusion",
        help="Directory to store logs",
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=5,
        help="Generate samples every N epochs",
    )

    # Checkpointing parameters
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/cfg_diffusion",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_top_k", type=int, default=1, help="Number of best checkpoints to keep"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )

    # Visualization parameters
    parser.add_argument(
        "--visualize_progress",
        action="store_true",
        help="Whether to visualize training progress with samples",
    )
    parser.add_argument(
        "--viz_every_n_epochs",
        type=int,
        default=10,
        help="Generate visualization samples every N epochs",
    )
    parser.add_argument(
        "--viz_num_samples",
        type=int,
        default=6,
        help="Number of samples per condition to generate for visualization",
    )
    parser.add_argument(
        "--viz_inference_steps",
        type=int,
        default=25,
        help="Number of inference steps for visualization",
    )
    parser.add_argument(
        "--viz_fixed_noise",
        action="store_true",
        help="Use fixed noise for visualization samples",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create a unique run name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{current_time}"

    # Set up logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=run_name,
        version=current_time,
    )
    logger.log_hyperparams(args)
    

    # Set up the DataModule
    datamodule = CheXpertDataModule(
        data_dir=args.data_path,
        img_size=args.image_size,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        seed=args.seed,
        debug_mode=args.debug_mode,
        pin_memory=args.pin_memory,
    )

    # Initialize model
    model = ClassifierFreeGuidedDiffusion(
        sample_size=args.sample_size,
        in_channels=1,
        out_channels=1,
        
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        use_ema=args.use_ema,
        ema_inv_gamma=args.ema_inv_gamma,
        ema_power=args.ema_power,
        ema_max_decay=args.ema_max_decay,
        ddpm_num_steps=args.ddpm_num_steps,
        ddpm_beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        class_cond=args.class_cond,
        num_classes=len(CHEXPERT_CLASSES),
        unconditional_probability=args.unconditional_probability,
        guidance_scale=args.guidance_scale,
    )

    # Set up callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, run_name),
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=args.save_top_k,
        monitor="val_loss",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Add the progress visualization callback if requested
    if args.visualize_progress:
        progress_viz_callback = ProgressVisualizationCallback(
            every_n_epochs=args.viz_every_n_epochs,
            num_samples=args.viz_num_samples,
            output_dir="progress_samples",
            guidance_scale=args.guidance_scale,
            inference_steps=args.viz_inference_steps,
            fixed_noise=args.viz_fixed_noise,
        )
        callbacks.append(progress_viz_callback)

    # Set up trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=args.gradient_clip_val,
        precision="bf16-mixed",
        log_every_n_steps=10,
        default_root_dir=args.log_dir,
    )

    # Train model using the DataModule
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)

    print(f"Training completed. Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
