"""
Callback to visualize training progress through generated images.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from neptune.types import File
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import NeptuneLogger

from src.utils.constants import CHEXPERT_CLASSES


class ProgressVisualizationCallback(Callback):
    """
    Callback that generates and saves images at regular intervals during training
    to visualize improvements in generation quality.
    """

    def __init__(
        self,
        every_n_epochs=5,
        num_samples=4,
        conditions=None,
        output_dir="progress_samples",
        guidance_scale=3.0,
        inference_steps=20,
        fixed_noise=True,
        save_on_validation=True,  # New parameter to control validation image saving
    ):
        """
        Initialize the callback.

        Args:
            every_n_epochs: How often to generate images (in epochs)
            num_samples: Number of images to generate per condition
            conditions: List of conditions to generate (indices or names from CHEXPERT_CLASSES).
                        If None, default conditions will be used
            output_dir: Directory to save images
            guidance_scale: Guidance strength for generation
            inference_steps: Number of inference steps
            fixed_noise: If True, use the same initial noise for comparison across epochs
            save_on_validation: If True, save images after each validation phase
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.guidance_scale = guidance_scale
        self.inference_steps = inference_steps
        self.fixed_noise = fixed_noise
        self.save_on_validation = save_on_validation

        # Determine conditions to use
        if conditions is None:
            # Default conditions: normal, pneumonia, cardiomegaly, pleural effusion
            self.conditions = [
                "No Finding",
                "Pneumonia",
                "Cardiomegaly",
                "Pleural Effusion",
            ]
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
            f.write(f"Model: {pl_module.__class__.__name__}\n")
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
        Generate and save images at the end of certain epochs.
        """
        epoch = trainer.current_epoch

        # Only generate images every N epochs or on the last epoch
        is_last_epoch = epoch == trainer.max_epochs - 1
        if (epoch + 1) % self.every_n_epochs != 0 and not is_last_epoch:
            return

        # Ensure we have a directory for this epoch
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        print(f"\nGenerating progress images for epoch {epoch + 1}...")

        # Save all generated images for the final grid
        all_samples = []
        all_conditions = []

        try:
            # For each condition, generate images
            for i, cond_idx in enumerate(self.condition_indices):
                cond_name = self.conditions[i]
                print(f"  Generating for condition: {cond_name}")

                # Create directory for this condition
                cond_dir = os.path.join(epoch_dir, cond_name.replace(" ", "_"))
                os.makedirs(cond_dir, exist_ok=True)

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
                            device=pl_module.device,
                        )
                        self.fixed_noises[cond_name] = noise

                    # Generate samples with fixed noise
                    samples = pl_module.generate_samples(
                        batch_size=self.num_samples,
                        labels=labels,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.inference_steps,
                        initial_noise=self.fixed_noises[
                            cond_name
                        ].clone(),  # Use copy to not alter the original
                    )
                else:
                    # Generate samples with random noise
                    samples = pl_module.generate_samples(
                        batch_size=self.num_samples,
                        labels=labels,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.inference_steps,
                    )

                # Save individual samples
                for j, img in enumerate(samples):
                    img_np = self._prepare_image_for_display(img)
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img_np, cmap="gray")
                    plt.axis("off")
                    plt.title(f"{cond_name} - Epoch {epoch + 1}")
                    plt.tight_layout()
                    sample_path = os.path.join(cond_dir, f"sample_{j + 1:02d}.png")
                    plt.savefig(sample_path, dpi=150)
                    plt.close()

                # Add samples to the final grid
                all_samples.append(samples)
                all_conditions.extend([cond_name] * self.num_samples)

            # Create a grid with all samples
            grid_path = self._create_grid(
                all_samples, all_conditions, epoch + 1, epoch_dir
            )

            # Save temporal evolution if there's more than one epoch
            if epoch > 0:
                timeline_paths = self._create_timeline(trainer, pl_module)

            # Send to Neptune if available
            if trainer.logger and isinstance(trainer.logger, NeptuneLogger):
                try:
                    # Log the grid image - using unique path instead of appending
                    trainer.logger.experiment[f"images/grid/epoch_{epoch + 1}"].upload(
                        File(grid_path)
                    )

                    # Log individual condition images - use unique path structure to avoid conflicts
                    for i, cond_idx in enumerate(self.condition_indices):
                        cond_name = self.conditions[i]
                        cond_name_slug = cond_name.replace(" ", "_")
                        sample_path = os.path.join(
                            epoch_dir, cond_name_slug, "sample_01.png"
                        )

                        # Only log the first sample for each condition to avoid clutter
                        if os.path.exists(sample_path):
                            # Create unique field for each condition+epoch combination
                            # Instead of appending to same series with same step value
                            trainer.logger.experiment[
                                f"images/epoch_{epoch + 1}/{cond_name_slug}"
                            ].upload(File(sample_path))
                except Exception as e:
                    print(f"Error sending images to Neptune: {e}")

            print(f"Progress images saved to: {epoch_dir}\n")

        except Exception as e:
            print(f"Error generating progress images: {e}")

    def _create_grid(self, all_samples, all_conditions, epoch, save_dir):
        """Create a grid with all generated samples."""
        # Calculate grid dimensions
        total_samples = sum(len(samples) for samples in all_samples)
        n_cols = self.num_samples
        n_rows = len(self.conditions)

        # Create figure
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        plt.suptitle(f"Generated Samples - Epoch {epoch}", fontsize=16)

        # Add each image to the grid
        sample_idx = 0
        for i, samples in enumerate(all_samples):
            for j, img in enumerate(samples):
                plt.subplot(n_rows, n_cols, sample_idx + 1)
                img_np = self._prepare_image_for_display(img)
                plt.imshow(img_np, cmap="gray")
                plt.axis("off")
                if j == 0:  # Only show label in the first column
                    plt.ylabel(self.conditions[i], fontsize=12)
                sample_idx += 1

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust space for title
        grid_path = os.path.join(save_dir, f"grid_epoch_{epoch:03d}.png")
        plt.savefig(grid_path, dpi=200)
        plt.close()

        return grid_path

    def _create_timeline(self, trainer, pl_module):
        """
        Create a visualization of the temporal evolution for each condition.
        """
        # Calculate available epochs
        available_epochs = []
        epoch_dirs = []

        for epoch in range(trainer.current_epoch + 1):
            if epoch == 0 or (epoch + 1) % self.every_n_epochs == 0:
                epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}")
                if os.path.exists(epoch_dir):
                    available_epochs.append(epoch + 1)
                    epoch_dirs.append(epoch_dir)

        if len(available_epochs) <= 1:
            return []  # Not enough epochs to show evolution

        # Create directory for temporal evolution
        timeline_dir = os.path.join(self.output_dir, "timeline")
        os.makedirs(timeline_dir, exist_ok=True)

        # Store paths of all timeline images for logging to Neptune
        timeline_paths = []

        # For each condition, create a visualization of the evolution
        for i, cond_name in enumerate(self.conditions):
            fig, axes = plt.subplots(
                1, len(available_epochs), figsize=(len(available_epochs) * 3, 3)
            )
            fig.suptitle(f"Evolution of {cond_name} throughout training", fontsize=14)

            for j, (epoch, epoch_dir) in enumerate(zip(available_epochs, epoch_dirs)):
                # Load first image of this condition and epoch
                cond_dir = os.path.join(epoch_dir, cond_name.replace(" ", "_"))
                if os.path.exists(cond_dir):
                    sample_path = os.path.join(cond_dir, "sample_01.png")
                    if os.path.exists(sample_path):
                        img = plt.imread(sample_path)
                        if len(available_epochs) > 1:
                            ax = axes[j]
                        else:
                            ax = axes  # Only one axis
                        ax.imshow(img, cmap="gray")
                        ax.set_title(f"Epoch {epoch}")
                        ax.axis("off")

            plt.tight_layout()
            timeline_path = os.path.join(
                timeline_dir, f"evolution_{cond_name.replace(' ', '_')}.png"
            )
            plt.savefig(timeline_path, dpi=200)
            plt.close()
            timeline_paths.append(timeline_path)

        # Create a gif for each condition if possible
        try:
            from PIL import Image

            for cond_name in self.conditions:
                images = []
                for epoch, epoch_dir in zip(available_epochs, epoch_dirs):
                    cond_dir = os.path.join(epoch_dir, cond_name.replace(" ", "_"))
                    sample_path = os.path.join(cond_dir, "sample_01.png")
                    if os.path.exists(sample_path):
                        images.append(Image.open(sample_path))

                if images:
                    # Save the gif with a duration of 1 second per image and infinite loop
                    gif_path = os.path.join(
                        timeline_dir, f"evolution_{cond_name.replace(' ', '_')}.gif"
                    )
                    images[0].save(
                        gif_path,
                        save_all=True,
                        append_images=images[1:],
                        duration=1000,
                        loop=0,
                    )
                    timeline_paths.append(gif_path)

                    # Log the GIF to Neptune if logger exists
                    if trainer.logger and isinstance(trainer.logger, NeptuneLogger):
                        try:
                            trainer.logger.experiment[
                                f"images/evolution/{cond_name.replace(' ', '_')}"
                            ].upload(File(gif_path))
                        except Exception as e:
                            print(f"Error uploading GIF to Neptune: {e}")
        except Exception as e:
            print(f"Could not create evolution GIFs: {e}")

        print(f"Temporal evolution visualizations saved to: {timeline_dir}")
        return timeline_paths

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Generate and save images at the end of each validation epoch.
        """
        if not self.save_on_validation:
            return

        epoch = trainer.current_epoch

        # Create directory for validation samples
        val_dir = os.path.join(self.output_dir, f"val_epoch_{epoch + 1:03d}")
        os.makedirs(val_dir, exist_ok=True)

        print(f"\nGenerating validation images for epoch {epoch + 1}...")

        # Save all generated images for the final grid
        all_samples = []
        all_conditions = []

        try:
            # For each condition, generate images
            for i, cond_idx in enumerate(self.condition_indices):
                cond_name = self.conditions[i]
                print(f"  Generating validation samples for condition: {cond_name}")

                # Create directory for this condition
                cond_dir = os.path.join(val_dir, cond_name.replace(" ", "_"))
                os.makedirs(cond_dir, exist_ok=True)

                # Create label tensor for this condition
                labels = torch.zeros(
                    (self.num_samples, len(CHEXPERT_CLASSES)), device=pl_module.device
                )
                labels[:, cond_idx] = 1.0

                # Generate validation samples (use fixed noise if available)
                if self.fixed_noise and cond_name in self.fixed_noises:
                    samples = pl_module.generate_samples(
                        batch_size=self.num_samples,
                        labels=labels,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.inference_steps,
                        initial_noise=self.fixed_noises[cond_name].clone(),
                    )
                else:
                    samples = pl_module.generate_samples(
                        batch_size=self.num_samples,
                        labels=labels,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.inference_steps,
                    )

                # Save individual samples
                for j, img in enumerate(samples):
                    img_np = self._prepare_image_for_display(img)
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img_np, cmap="gray")
                    plt.axis("off")
                    plt.title(f"{cond_name} - Val Epoch {epoch + 1}")
                    plt.tight_layout()
                    sample_path = os.path.join(cond_dir, f"val_sample_{j + 1:02d}.png")
                    plt.savefig(sample_path, dpi=150)
                    plt.close()

                # Add samples to the final grid
                all_samples.append(samples)
                all_conditions.extend([cond_name] * self.num_samples)

            # Create a grid with all samples
            grid_path = self._create_validation_grid(
                all_samples, all_conditions, epoch + 1, val_dir
            )

            # Log to Neptune if available
            if trainer.logger and isinstance(trainer.logger, NeptuneLogger):
                try:
                    # Log the validation grid - using unique paths instead of appending
                    trainer.logger.experiment[
                        f"images/validation/grid/epoch_{epoch + 1}"
                    ].upload(File(grid_path))

                    # Log individual condition images
                    for i, cond_idx in enumerate(self.condition_indices):
                        cond_name = self.conditions[i]
                        cond_name_slug = cond_name.replace(" ", "_")
                        sample_path = os.path.join(
                            val_dir, cond_name_slug, f"val_sample_01.png"
                        )

                        # Only log the first sample for each condition
                        if os.path.exists(sample_path):
                            trainer.logger.experiment[
                                f"images/validation/epoch_{epoch + 1}/{cond_name_slug}"
                            ].upload(File(sample_path))
                except Exception as e:
                    print(f"Error sending validation images to Neptune: {e}")

            print(f"Validation images saved to: {val_dir}\n")

        except Exception as e:
            print(f"Error generating validation images: {e}")

    def _create_validation_grid(self, all_samples, all_conditions, epoch, save_dir):
        """Create a grid of validation samples."""
        # Calculate grid dimensions
        n_cols = self.num_samples
        n_rows = len(self.conditions)

        # Create figure
        plt.figure(figsize=(n_cols * 3, n_rows * 3))
        plt.suptitle(f"Validation Samples - Epoch {epoch}", fontsize=16)

        # Add each image to the grid
        sample_idx = 0
        for i, samples in enumerate(all_samples):
            for j, img in enumerate(samples):
                plt.subplot(n_rows, n_cols, sample_idx + 1)
                img_np = self._prepare_image_for_display(img)
                plt.imshow(img_np, cmap="gray")
                plt.axis("off")
                if j == 0:  # Only show label in the first column
                    plt.ylabel(self.conditions[i], fontsize=12)
                sample_idx += 1

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust space for title
        grid_path = os.path.join(save_dir, f"val_grid_epoch_{epoch:03d}.png")
        plt.savefig(grid_path, dpi=200)
        plt.close()

        return grid_path
