"""
Classifier Guidance implementation using the diffusers library.
"""

import os
from typing import Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from src.utils.constants import CHEXPERT_CLASSES


class ClassifierGuidedDiffusion(pl.LightningModule):
    """
    PyTorch Lightning module for Classifier-Guided Diffusion using diffusers.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        img_size: int = 64,
        in_channels: int = 1,  # Changed from 3 to 1
        out_channels: int = 1,  # Changed from 3 to 1
        center_crop_size: Optional[int] = None,
        random_flip: bool = True,
        lr: float = 1e-4,
        lr_warmup_steps: int = 500,
        optimizer_type: str = "AdamW",
        noise_scheduler_beta_schedule: str = "linear",
        noise_scheduler_num_train_timesteps: int = 1000,
        use_ema: bool = True,
        ema_inv_gamma: float = 1.0,
        ema_power: float = 0.75,
        ema_max_decay: float = 0.9999,
        classifier_path: Optional[str] = None,
    ):
        """
        Initialize the Classifier-Guided Diffusion model.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models
            img_size: Size of the input image (assumed square)
            in_channels: Number of input channels
            out_channels: Number of output channels
            center_crop_size: Size for center cropping input images
            random_flip: Whether to randomly flip input images
            lr: Learning rate
            lr_warmup_steps: Number of warmup steps for learning rate scheduler
            optimizer_type: Type of optimizer to use
            noise_scheduler_beta_schedule: Schedule for noise variance
            noise_scheduler_num_train_timesteps: Number of diffusion steps
            use_ema: Whether to use EMA model
            ema_inv_gamma: EMA inverse gamma parameter
            ema_power: EMA power parameter
            ema_max_decay: EMA maximum decay parameter
            classifier_path: Path to pretrained classifier for guidance
        """
        super().__init__()
        self.save_hyperparameters()

        # Create the UNet model - unconditional for classifier guidance
        if pretrained_model_name_or_path:
            self.unet = UNet2DModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="unet"
            )
        else:
            self.unet = UNet2DModel(
                sample_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                layers_per_block=2,
                block_out_channels=(64, 128, 256, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                ),
                up_block_types=(
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )

        # Create noise scheduler
        self.noise_scheduler = DDPMScheduler(
            beta_schedule=noise_scheduler_beta_schedule,
            num_train_timesteps=noise_scheduler_num_train_timesteps,
        )

        # EMA model
        self.use_ema = use_ema
        if use_ema:
            # Inicializamos el modelo EMA durante el entrenamiento
            self.ema_model_instance = None  # Se inicializará en on_fit_start
            self.ema_inv_gamma = ema_inv_gamma
            self.ema_power = ema_power
            self.ema_max_decay = ema_max_decay
        else:
            self.ema_model = None

        # Classifier
        self.classifier = None
        if classifier_path is not None:
            self.load_classifier(classifier_path)

    def load_classifier(self, classifier_path):
        """
        Load a pretrained classifier for guidance.

        Args:
            classifier_path: Path to the pretrained classifier
        """
        from src.models.classifier import ClassifierModule

        # Load the classifier
        self.classifier_module = ClassifierModule.load_from_checkpoint(
            classifier_path, map_location=self.device
        )

        # Extract the classifier model
        self.classifier = self.classifier_module.classifier

        # Freeze the classifier parameters
        for param in self.classifier.parameters():
            param.requires_grad = False

        print(f"Loaded classifier from {classifier_path}")

    def prepare_latents(self, batch_size, channels, height, width, device=None):
        """
        Prepare random noise as input to the diffusion process.
        """
        if device is None:
            device = self.device

        # Generate random noise
        latents = torch.randn(
            (batch_size, channels, height, width),
            device=device,
        )

        return latents

    def training_step(self, batch, batch_idx):
        """
        Training step for unconditional diffusion models.
        For classifier guidance, we train the diffusion model unconditionally.
        """
        images, _ = batch  # We don't use labels during training for classifier guidance

        # Move data to the correct device
        images = images.to(self.device)

        # Convert images to latent space if needed
        latents = images

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual (unconditional)
        noise_pred = self.unet(noisy_latents, timesteps).sample

        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for unconditional diffusion models.
        """
        images, _ = (
            batch  # We don't use labels during validation for classifier guidance
        )

        # Move data to the correct device
        images = images.to(self.device)

        # Convert images to latent space if needed
        latents = images

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual (unconditional)
        model = self.ema_model.averaged_model if self.use_ema else self.unet
        noise_pred = model(noisy_latents, timesteps).sample

        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def on_fit_start(self):
        """
        Called at the beginning of training after model has been moved to the correct device.
        This is the perfect place to properly initialize the EMA model with compiled UNet.
        """
        if self.use_ema and self.ema_model_instance is None:
            # Create a new EMA model instance now that the model is on the correct device
            self.ema_model = EMAModel(
                model=self.unet,
                inv_gamma=self.ema_inv_gamma,
                power=self.ema_power,
                max_decay=self.ema_max_decay,
            )
            print(f"EMA model initialized on device: {self.device}")
            
    def on_train_batch_end(self, *args, **kwargs):
        """
        Update EMA model after each training batch.
        """
        if self.use_ema and hasattr(self, 'ema_model'):
            self.ema_model.step(self.unet.parameters())

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.
        """
        if self.hparams.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.unet.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                weight_decay=1e-6,
                eps=1e-8,
            )
        else:
            optimizer = torch.optim.Adam(
                self.unet.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        scheduler = get_scheduler(
            name="constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=self.hparams.lr_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def classifier_gradient(self, x, t, labels, classifier_scale=1.0):
        """
        Compute gradient of classifier log-probability with respect to input.

        Args:
            x: Input image
            t: Timestep
            labels: Target labels
            classifier_scale: Strength of the classifier guidance

        Returns:
            Gradient of log probability
        """
        # Ensure classifier is loaded
        if self.classifier is None:
            raise ValueError("Classifier not loaded. Use load_classifier() first.")

        # Enable gradient tracking
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)

            # For multi-label classification
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                pos_mask = (labels > 0.5).float()
                neg_mask = (labels < 0.5).float()

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                # Calculate log probs for positive and negative labels
                log_pos_probs = torch.log(probs + 1e-10) * pos_mask
                log_neg_probs = torch.log(1 - probs + 1e-10) * neg_mask

                # Sum all log probs
                log_probs_sum = (log_pos_probs + log_neg_probs).sum(dim=1)

                # Get gradient of log probability with respect to input
                gradient = torch.autograd.grad(log_probs_sum.sum(), x_in)[0]
            else:
                # Single-label case
                log_probs = F.log_softmax(logits, dim=-1)
                y_log_probs = log_probs[range(len(logits)), labels.view(-1)]
                gradient = torch.autograd.grad(y_log_probs.sum(), x_in)[0]

        return gradient * classifier_scale

    def generate_samples(
        self,
        batch_size=4,
        labels=None,
        classifier_scale=1.0,
        num_inference_steps=50,
        return_all_timesteps=False,
    ):
        """
        Generate samples using the classifier-guided diffusion model.

        Args:
            batch_size: Number of images to generate
            labels: Labels for conditional generation
            classifier_scale: Strength of classifier guidance (higher = stronger conditioning)
            num_inference_steps: Number of inference steps
            return_all_timesteps: Whether to return intermediate results

        Returns:
            Generated images (and optionally intermediate results)
        """
        # Ensure classifier is loaded
        if self.classifier is None:
            raise ValueError("Classifier not loaded. Use load_classifier() first.")

        # Set models to evaluation mode
        self.unet.eval()
        self.classifier.eval()

        # Use DDIM scheduler for faster sampling
        ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        ddim_scheduler.set_timesteps(num_inference_steps)

        # Create random noise
        latents = self.prepare_latents(
            batch_size=batch_size,
            channels=self.unet.config.in_channels,
            height=self.unet.config.sample_size,
            width=self.unet.config.sample_size,
        )

        # Move labels to the correct device if provided
        if labels is not None:
            labels = labels.to(self.device)
        else:
            # Default to "No Finding" if no labels provided
            labels = torch.zeros(
                (batch_size, len(CHEXPERT_CLASSES)), device=self.device
            )
            labels[:, CHEXPERT_CLASSES.index("No Finding")] = 1.0

        # Store intermediate results if requested
        timestep_latents = []

        # Use EMA model if available
        model = self.ema_model.averaged_model if self.use_ema else self.unet

        # Generate samples using DDIM with classifier guidance
        with torch.no_grad():
            for i, t in enumerate(ddim_scheduler.timesteps):
                # Create timestep tensor
                timestep = torch.full(
                    (batch_size,), t, device=self.device, dtype=torch.long
                )

                # Get classifier gradient for guidance (enables grad temporarily)
                if classifier_scale > 0:
                    with torch.enable_grad():
                        gradient = self.classifier_gradient(
                            latents, timestep, labels, classifier_scale
                        )
                else:
                    gradient = 0

                # Predict noise residual without conditioning (unconditional model)
                noise_pred = model(latents, timestep).sample

                # Step function should incorporate gradient, but we'll do it manually for clarity:
                # 1. Get the predicted x_0
                alpha_prod_t = ddim_scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t

                # Predict x_0 from current noisy sample and predicted noise
                pred_original_sample = (
                    latents - beta_prod_t**0.5 * noise_pred
                ) / alpha_prod_t**0.5

                # 2. Get the coefficient for gradient step
                gradient_step_coef = ddim_scheduler.betas[t] / (beta_prod_t**0.5)

                # 3. Apply classifier guidance
                # The gradient is calculated w.r.t the current noisy image,
                # so we add it directly to the predicted noise
                guided_noise_pred = noise_pred - gradient_step_coef * gradient

                # 4. Get the previous timestep using the scheduler
                prev_timestep = ddim_scheduler.previous_timestep(t)

                # 5. Compute coefficients for the ddim scheduler
                alpha_prod_t_prev = (
                    ddim_scheduler.alphas_cumprod[prev_timestep]
                    if prev_timestep >= 0
                    else torch.tensor(1.0)
                )

                # 6. Compute predicted previous sample with guidance
                latents = (
                    alpha_prod_t_prev**0.5 * pred_original_sample
                    + (1 - alpha_prod_t_prev) ** 0.5 * guided_noise_pred
                )

                # Save intermediate result if requested
                if return_all_timesteps:
                    timestep_latents.append(latents.clone())

        # Denormalize latents if needed (for UNet output)
        images = self.denormalize_latents(latents)

        # Set models back to training mode
        self.unet.train()
        self.classifier.train()

        # Return results
        if return_all_timesteps:
            # Denormalize all timestep latents
            timestep_images = [self.denormalize_latents(l) for l in timestep_latents]
            return images, timestep_images
        else:
            return images

    def denormalize_latents(self, latents):
        """
        Denormalize latents to image space. For direct UNet output, this is a simple
        scaling and shifting from [-1, 1] to [0, 255].
        """
        # Scale from [-1, 1] to [0, 1]
        images = (latents * 0.5 + 0.5).clamp(0, 1)
        # Scale to [0, 255] and convert to uint8
        images = (images * 255).type(torch.uint8)
        return images
