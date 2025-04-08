"""
Classifier-Free Guidance implementation using the diffusers library.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, List, Union, Dict

from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler

# Importamos ExponentialMovingAverage de pytorch_ema para una implementación más robusta
from torch_ema import ExponentialMovingAverage

from src.utils.constants import CHEXPERT_CLASSES


class ClassifierFreeGuidedDiffusion(pl.LightningModule):
    """
    PyTorch Lightning module for Classifier-Free Guided Diffusion using diffusers.
    """
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        img_size: int = 64,
        in_channels: int = 1,
        out_channels: int = 3,
        num_classes: int = len(CHEXPERT_CLASSES),
        center_crop_size: Optional[int] = None,
        random_flip: bool = True,
        conditioning_dropout_prob: float = 0.1,
        lr: float = 1e-4,
        lr_warmup_steps: int = 500,
        optimizer_type: str = "AdamW",
        noise_scheduler_beta_schedule: str = "linear",
        noise_scheduler_num_train_timesteps: int = 1000,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_update_every: int = 1,
    ):
        """
        Initialize the Classifier-Free Guided Diffusion model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier from huggingface.co/models
            img_size: Size of the input image (assumed square)
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_classes: Number of classes for conditional generation
            center_crop_size: Size for center cropping input images
            random_flip: Whether to randomly flip input images
            conditioning_dropout_prob: Probability of dropping class conditioning for CFG training
            lr: Learning rate
            lr_warmup_steps: Number of warmup steps for learning rate scheduler
            optimizer_type: Type of optimizer to use
            noise_scheduler_beta_schedule: Schedule for noise variance
            noise_scheduler_num_train_timesteps: Number of diffusion steps
            use_ema: Whether to use EMA model
            ema_decay: EMA decay rate (higher = slower updating)
            ema_update_every: Update EMA every N steps
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create the UNet model
        if pretrained_model_name_or_path:
            self.unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="unet"
            )
        else:
            self.unet = UNet2DConditionModel(
                sample_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                layers_per_block=2,
                block_out_channels=(64, 128, 256, 512),
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ),
                cross_attention_dim=num_classes,
            )
            
        # Create noise scheduler
        self.noise_scheduler = DDPMScheduler(
            beta_schedule=noise_scheduler_beta_schedule,
            num_train_timesteps=noise_scheduler_num_train_timesteps,
        )
        
        # EMA settings - We'll initialize the actual EMA model in on_fit_start
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.ema = None
        self.ema_unet = None  # Para almacenar una copia del modelo con pesos EMA
        self._step_counter = 0  # Contador privado para EMA updates
        
        # Conditioning dropout probability
        self.conditioning_dropout_prob = conditioning_dropout_prob

    def on_fit_start(self):
        """
        Called at the beginning of training after model has been moved to the correct device.
        This is the perfect place to initialize the EMA model since we know the device.
        """
        if self.use_ema:
            # Inicializar EMA usando pytorch_ema, que es más robusto
            self.ema = ExponentialMovingAverage(
                self.unet.parameters(),
                decay=self.ema_decay
            )
            # Crear una copia del modelo para poder hacer evaluaciones con EMA
            self.ema_unet = UNet2DConditionModel(**self.unet.config)
            self.ema_unet.load_state_dict(self.unet.state_dict())
            self.ema_unet.to(self.device)
            
            print(f"EMA initialized on device: {self.device} with decay: {self.ema_decay}")

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
    
    def encode_condition(self, labels):
        """
        Encode class labels for conditioning.
        
        Args:
            labels: One-hot encoded class labels [batch_size, num_classes]
            
        Returns:
            Encoded condition for UNet in the format [batch_size, sequence_length=1, hidden_dim]
        """
        # UNet2DConditionModel expects encoder_hidden_states to be of shape [batch_size, sequence_length, hidden_dim]
        # For our case, we'll use sequence_length=1 and treat num_classes as hidden_dim
        if labels is None:
            return None
            
        # Reshape to add sequence_length dimension [batch_size, num_classes] -> [batch_size, 1, num_classes]
        return labels.unsqueeze(1)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for conditional diffusion models.
        """
        images, labels = batch
        
        # Move data to the correct device
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Convert images to latent space if needed
        latents = images
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        ).long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get the condition embedding for the labels
        encoder_hidden_states = self.encode_condition(labels)
        
        # Classifier-free guidance: randomly drop conditioning with probability conditioning_dropout_prob
        if self.training and self.conditioning_dropout_prob > 0:
            mask = torch.bernoulli(
                torch.ones(bsz, device=encoder_hidden_states.device) * 
                (1 - self.conditioning_dropout_prob)
            ).view(bsz, 1, 1)
            
            encoder_hidden_states = encoder_hidden_states * mask
        
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        # Incrementar contador de pasos interno
        self._step_counter += 1
        
        return loss
    
    def on_after_backward(self):
        """
        Llamado después de calcular gradientes y antes de actualizarlos.
        Es un buen lugar para actualizar EMA porque los parámetros aún no han cambiado.
        """
        if self.use_ema and self.ema is not None and self._step_counter % self.ema_update_every == 0:
            self.ema.update(self.unet.parameters())
    
    def _get_model_for_evaluation(self):
        """
        Método privado para obtener el modelo adecuado para evaluación/generación.
        """
        if not self.use_ema or self.ema is None:
            return self.unet
            
        # Para evaluación, aplicamos los pesos de EMA a una copia del modelo
        self.ema.copy_to(self.ema_unet.parameters())
        return self.ema_unet
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for conditional diffusion models.
        """
        images, labels = batch
        
        # Move data to the correct device
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Convert images to latent space if needed
        latents = images
        
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        ).long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get the condition embedding for the labels
        encoder_hidden_states = self.encode_condition(labels)
        
        # Get model for prediction (EMA or regular)
        model = self._get_model_for_evaluation()
            
        # Predict the noise residual
        noise_pred = model(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        
        return loss
    
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
    
    def generate_samples(
        self,
        batch_size=4,
        labels=None,
        guidance_scale=3.0,
        num_inference_steps=50,
        return_all_timesteps=False,
    ):
        """
        Generate samples using the classifier-free guided diffusion model.
        
        Args:
            batch_size: Number of images to generate
            labels: Optional labels for conditional generation
            guidance_scale: Strength of classifier-free guidance (higher = stronger conditioning)
            num_inference_steps: Number of inference steps
            return_all_timesteps: Whether to return intermediate results
            
        Returns:
            Generated images (and optionally intermediate results)
        """
        # Set model to evaluation mode
        model = self._get_model_for_evaluation()
        model.eval()
        
        # Use DDIM scheduler for faster sampling
        ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        ddim_scheduler.set_timesteps(num_inference_steps)
        
        # Create random noise
        latents = self.prepare_latents(
            batch_size=batch_size,
            channels=model.config.in_channels,
            height=model.config.sample_size,
            width=model.config.sample_size,
        )
        
        # Move labels to the correct device if provided
        if labels is not None:
            labels = labels.to(self.device)
        else:
            # Default to "No Finding" if no labels provided
            labels = torch.zeros((batch_size, len(CHEXPERT_CLASSES)), device=self.device)
            labels[:, CHEXPERT_CLASSES.index('No Finding')] = 1.0
        
        # Encode condition
        encoder_hidden_states = self.encode_condition(labels)
        
        # Prepare unconditional embedding for classifier-free guidance
        if guidance_scale > 1.0:
            uncond_embedding = torch.zeros_like(encoder_hidden_states)
        
        # Store intermediate results if requested
        timestep_latents = []
        
        # Generate samples using DDIM
        with torch.no_grad():
            for i, t in enumerate(ddim_scheduler.timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                
                # predict the noise residual
                if guidance_scale > 1.0:
                    # Concatenate condition and unconditional embeddings for batch processing
                    combined_embeddings = torch.cat([uncond_embedding, encoder_hidden_states])
                    
                    # Get the noise predictions
                    noise_pred = model(
                        latent_model_input,
                        t,
                        encoder_hidden_states=combined_embeddings,
                    ).sample
                    
                    # Perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = model(
                        latents,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = ddim_scheduler.step(noise_pred, t, latents).prev_sample
                
                # Save intermediate result if requested
                if return_all_timesteps:
                    timestep_latents.append(latents.clone())
        
        # Denormalize latents if needed (for UNet output)
        images = self.denormalize_latents(latents)
        
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
    
    