"""
Simplified callback to visualize training progress through generated images.
"""

import os
from datetime import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import NeptuneLogger
from neptune.types import File

from src.utils.constants import CHEXPERT_CLASSES


class ProgressVisualizationCallback(Callback):
    """
    Callback that generates and saves images every 10 epochs during training
    to visualize improvements in generation quality.
    """

    def __init__(
        self,
        every_n_epochs=10,
        num_samples=6,
        conditions=None,
        output_dir="progress_samples",
        guidance_scale=3.0,
        inference_steps=20,
        fixed_noise=True,
    ):
        """
        Initialize the callback.

        Args:
            every_n_epochs: How often to generate images (in epochs)
            num_samples: Number of images to generate per condition
            conditions: List of conditions to generate (indices or names from CHEXPERT_CLASSES).
                        If None, all CHEXPERT_CLASSES will be used
            output_dir: Directory to save images
            guidance_scale: Guidance strength for generation
            inference_steps: Number of inference steps
            fixed_noise: If True, use the same initial noise for comparison across epochs
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.guidance_scale = guidance_scale
        self.inference_steps = inference_steps
        self.fixed_noise = fixed_noise

        # Determine conditions to use
        if conditions is None:
            self.conditions = CHEXPERT_CLASSES
        else:
            self.conditions = conditions

        # Convert names to indices if needed
        self.condition_indices = []
        for cond in self.conditions:
            if isinstance(cond, str):
                self.condition_indices.append(CHEXPERT_CLASSES.index(cond))
            else:
                self.condition_indices.append(cond)

        # Create output directory with timestamp to avoid overwrites
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{output_dir}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        # To store fixed noise for consistent comparisons
        self.fixed_noises = {}

    def on_train_start(self, trainer, pl_module):
        """Save configuration at the start of training."""
        # Save configuration to a file
        with open(os.path.join(self.output_dir, "config.txt"), "w") as f:
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Conditions: {self.conditions}\n")
            f.write(f"Inference steps: {self.inference_steps}\n")
            f.write(f"Guidance scale: {self.guidance_scale}\n")
            f.write(f"Samples per condition: {self.num_samples}\n")
            f.write(f"Fixed noise: {self.fixed_noise}\n\n")
            f.write("Model hyperparameters:\n")
            for k, v in pl_module.hparams.items():
                f.write(f"  {k}: {v}\n")

    def _prepare_image_for_display(self, img):
        """Convert tensor image to numpy array ready for display."""
        img_np = img.cpu().numpy() / 255.0

        # Handle different tensor shapes properly
        if img_np.shape[0] == 1:  # Single channel (grayscale)
            # Remove the channel dimension to get a 2D array for grayscale
            img_np = np.squeeze(img_np, axis=0)
        else:  # RGB
            img_np = np.transpose(img_np, (1, 2, 0))

        return img_np

    def on_epoch_end(self, trainer, pl_module):
        """
        Generate and save images at the end of specified epochs.
        """
        epoch = trainer.current_epoch
        
        # Skip epoch 0 and only generate every N epochs or on the last epoch
        if epoch == 0:
            return
            
        is_last_epoch = epoch == trainer.max_epochs - 1
        if (epoch % self.every_n_epochs != 0) and not is_last_epoch:
            return

        # Ensure we have a directory for this epoch
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        print(f"\nGenerating progress images for epoch {epoch}...")

        # Generate samples for all conditions
        all_samples = []
        all_condition_names = []

        try:
            # For each condition, generate images
            for i, cond_idx in enumerate(self.condition_indices):
                cond_name = self.conditions[i]
                print(f"  Generating for condition: {cond_name}")

                # Create label tensor for this condition
                labels = torch.zeros(
                    (self.num_samples, len(CHEXPERT_CLASSES)), device=pl_module.device
                )
                labels[:, cond_idx] = 1.0

                # Use fixed noise if enabled
                if self.fixed_noise:
                    # Create or retrieve fixed noise for this condition
                    if cond_name not in self.fixed_noises:
                        # Generate and save fixed noise
                        noise = pl_module.prepare_latents(
                            batch_size=self.num_samples,
                            channels=pl_module.unet.config.in_channels,
                            height=pl_module.unet.config.sample_size,
                            width=pl_module.unet.config.sample_size,
                        )
                        self.fixed_noises[cond_name] = noise

                    # Generate samples with fixed noise
                    samples = pl_module.generate_samples(
                        batch_size=self.num_samples,
                        labels=labels,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.inference_steps,
                    )
                else:
                    # Generate samples with random noise
                    samples = pl_module.generate_samples(
                        batch_size=self.num_samples,
                        labels=labels,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.inference_steps,
                    )

                # Add samples to the collection
                all_samples.extend(samples)
                all_condition_names.extend([cond_name] * len(samples))

            # Create a grid with all samples (3 columns)
            self._create_grid(all_samples, all_condition_names, epoch, epoch_dir)

            # Log to Neptune if available
            if any(isinstance(logger, NeptuneLogger) for logger in trainer.loggers):
                for logger in trainer.loggers:
                    if isinstance(logger, NeptuneLogger):
                        try:
                            grid_path = os.path.join(epoch_dir, f"grid_epoch_{epoch:03d}.png")
                            logger.experiment[f"images/grid/epoch_{epoch}"].upload(File(grid_path))
                        except Exception as e:
                            print(f"Error sending images to Neptune: {e}")
                        break

            print(f"Progress images saved to: {epoch_dir}\n")

        except Exception as e:
            print(f"Error generating progress images: {e}")

    def _create_grid(self, samples, condition_names, epoch, save_dir):
        """
        Create a grid with all generated samples using 3 columns.
        
        Args:
            samples: List of generated image tensors
            condition_names: List of condition names corresponding to each sample
            epoch: Current epoch number
            save_dir: Directory to save the grid image
        """
        # Calculate grid dimensions
        total_samples = len(samples)
        n_cols = 3  # Fixed at 3 columns as requested
        n_rows = math.ceil(total_samples / n_cols)
        
        # Create figure
        plt.figure(figsize=(n_cols * 4, n_rows * 4))
        plt.suptitle(f"Generated Samples - Epoch {epoch}", fontsize=16)
        
        # Add each image to the grid
        for idx, (img, condition) in enumerate(zip(samples, condition_names)):
            plt.subplot(n_rows, n_cols, idx + 1)
            img_np = self._prepare_image_for_display(img)
            plt.imshow(img_np, cmap="gray")
            plt.axis("off")
            plt.title(condition, fontsize=12)  # Add condition name as title
        
        # Fill empty subplots if needed
        for idx in range(total_samples, n_rows * n_cols):
            plt.subplot(n_rows, n_cols, idx + 1)
            plt.axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust space for title
        grid_path = os.path.join(save_dir, f"grid_epoch_{epoch:03d}.png")
        plt.savefig(grid_path, dpi=200)
        plt.close()
        
        return grid_path