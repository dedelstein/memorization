"""
Classifier-Free Guidance implementation using the diffusers library.
"""

import inspect

import torch
import torch.nn.functional as F


from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available



import pytorch_lightning as pl
from diffusers import UNet2DConditionModel

from src.utils.constants import CHEXPERT_CLASSES


class ClassifierFreeGuidedDiffusion(pl.LightningModule):
    """
    PyTorch Lightning module for Classifier-Free Guided Diffusion using diffusers.
    """

    def __init__(
        self,
        sample_size=64,
        in_channels=1,
        out_channels=1,
        
        learning_rate=1e-4,
        lr_scheduler="cosine",
        lr_warmup_steps=500,
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_weight_decay=1e-6,
        adam_epsilon=1e-08,
        use_ema=True,
        ema_inv_gamma=1.0,
        ema_power=3/4,
        ema_max_decay=0.9999,
        ddpm_num_steps=1000,
        ddpm_beta_schedule="linear",
        prediction_type="epsilon",
        enable_xformers_memory_efficient_attention=False,
        class_cond=True,
        num_classes=len(CHEXPERT_CLASSES),
        unconditional_probability=0.1,
        guidance_scale=5.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        
        # Optimizer parameters
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        
        # EMA parameters
        self.use_ema = use_ema
        self.ema_inv_gamma = ema_inv_gamma
        self.ema_power = ema_power
        self.ema_max_decay = ema_max_decay
        self.ema_model = None  # Will be initialized in setup()
        
        # Diffusion parameters
        self.ddpm_num_steps = ddpm_num_steps
        self.ddpm_beta_schedule = ddpm_beta_schedule
        self.prediction_type = prediction_type
        
        # Classifier-free guidance parameters
        self.class_cond = class_cond
        self.num_classes = num_classes
        self.unconditional_probability = unconditional_probability
        self.guidance_scale = guidance_scale
        
        # Performance optimizations
        self.enable_xformers_memory_efficient_attention = enable_xformers_memory_efficient_attention
        
        # Initialize model
        if self.class_cond:
            self.model = UNet2DConditionModel(
                sample_size=self.sample_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                # Use text embeddings for conditioning
                cross_attention_dim=self.num_classes,
            )
            
            # Create a simple class embedder
            self.class_embedder = torch.nn.Linear(
                self.num_classes, self.num_classes
            )
        else:
            self.model = UNet2DModel(
                sample_size=self.sample_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
            
        # Enable xformers if requested
        if self.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.model.enable_xformers_memory_efficient_attention()
            else:
                print("Warning: xformers is not available. Make sure it is installed correctly.")
        
        # Initialize noise scheduler
        accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
        if accepts_prediction_type:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.ddpm_num_steps,
                beta_schedule=self.ddpm_beta_schedule,
                prediction_type=self.prediction_type,
            )
        else:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.ddpm_num_steps,
                beta_schedule=self.ddpm_beta_schedule
            )
    
    def setup(self, stage=None):
        """
        Called when the trainer is initializing, ensure EMA model is on the correct device.
        """
        if self.use_ema and self.ema_model is None:
            # Initialize EMA model with parameters on the same device
            self.ema_model = EMAModel(
                self.model.parameters(),
                decay=self.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=self.ema_inv_gamma,
                power=self.ema_power,
                model_cls=UNet2DConditionModel if self.class_cond else UNet2DModel,
                model_config=self.model.config,
                device=self.device
            )
    
    def forward(self, noisy_images, timesteps, class_labels=None):
        """
        Forward pass through the model
        """
        if self.class_cond and class_labels is not None:
            # Convert class labels to embeddings for cross-attention conditioning
            encoder_hidden_states = self.class_embedder(class_labels)
            
            # Reshape to [batch_size, seq_length, hidden_dim]
            # For cross-attention, we need a sequence dimension
            batch_size = encoder_hidden_states.shape[0]
            # Add a sequence length dimension of 1 (one token per image)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            
            return self.model(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        else:
            return self.model(noisy_images, timesteps).sample
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.
        
        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(arr)
        res = arr[timesteps].float().to(timesteps.device)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        clean_images = batch["image"]
        
        # For classifier-free guidance, we need to handle class labels
        if self.class_cond:
            class_labels = batch.get("labels", None)
            
            # Randomly set some labels to zeros for classifier-free guidance training
            if class_labels is not None and self.unconditional_probability > 0:
                mask = torch.rand(class_labels.shape[0], device=class_labels.device) < self.unconditional_probability
                class_labels = torch.where(mask[:, None], torch.zeros_like(class_labels), class_labels)
        else:
            class_labels = None
        
        # Sample noise
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bsz = clean_images.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
        ).long()
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # Predict the noise residual
        model_output = self(noisy_images, timesteps, class_labels)
        
        # Calculate loss based on prediction type
        if self.prediction_type == "epsilon":
            loss = F.mse_loss(model_output, noise)
        elif self.prediction_type == "sample":
            alpha_t = self._extract_into_tensor(
                self.noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
            )
            snr_weights = alpha_t / (1 - alpha_t)
            # use SNR weighting from distillation paper
            loss = snr_weights * F.mse_loss(model_output, clean_images, reduction="none")
            loss = loss.mean()
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")
        
        # Update EMA model as part of the training step
        if self.use_ema and self.trainer.global_step > 0:
            if self.ema_model is None:
                self.setup()  # Ensure EMA is initialized
            self.ema_model.to(self.device)
            self.ema_model.step(self.model.parameters())
        
        # Log loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - calculate loss on validation data
        """
        clean_images = batch["image"]
        
        if self.class_cond:
            class_labels = batch.get("labels", None)
        else:
            class_labels = None
        
        # Sample noise
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bsz = clean_images.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
        ).long()
        
        # Add noise to the clean images
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # Predict the noise residual
        model_output = self(noisy_images, timesteps, class_labels)
        
        # Calculate loss based on prediction type
        if self.prediction_type == "epsilon":
            loss = F.mse_loss(model_output, noise)
        elif self.prediction_type == "sample":
            alpha_t = self._extract_into_tensor(
                self.noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
            )
            snr_weights = alpha_t / (1 - alpha_t)
            loss = snr_weights * F.mse_loss(model_output, clean_images, reduction="none")
            loss = loss.mean()
        
        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """
        Generate and log sample images at the end of validation
        """
        # Skip if not on main process or not time to generate samples
        if not self.trainer.is_global_zero:
            return
        
        # Use EMA model if available
        if self.use_ema and self.ema_model is not None:
            # Store current model parameters
            stored_params = [param.clone() for param in self.model.parameters()]
            # Copy EMA parameters to model
            self.ema_model.copy_to(self.model.parameters())
        
        # Sample images with the model
        if self.class_cond:
            # Create condition labels for sampling (one per class)
            class_labels = torch.eye(self.num_classes, device=self.device)
            # Also add an unconditional sample
            class_labels = torch.cat([torch.zeros(1, self.num_classes, device=self.device), class_labels], dim=0)
            
            # Sample with classifier-free guidance
            with torch.no_grad():
                # Initialize noise
                image_shape = (class_labels.shape[0], self.in_channels, self.sample_size, self.sample_size)
                noise = torch.randn(image_shape, device=self.device)
                
                # Progressive denoising
                images = noise
                for t in self.noise_scheduler.timesteps:
                    timesteps = torch.full((class_labels.shape[0],), t, device=self.device, dtype=torch.long)
                    
                    # Predict noise for both conditional and unconditional
                    with torch.no_grad():
                        # Create embeddings for unconditional generation
                        unconditional_embeddings = self.class_embedder(torch.zeros_like(class_labels))
                        # Add sequence dimension
                        unconditional_embeddings = unconditional_embeddings.unsqueeze(1)
                        
                        unconditional_output = self.model(
                            images, 
                            timesteps, 
                            encoder_hidden_states=unconditional_embeddings
                        ).sample
                        
                        # Conditional prediction with class embeddings
                        conditional_embeddings = self.class_embedder(class_labels)
                        # Add sequence dimension
                        conditional_embeddings = conditional_embeddings.unsqueeze(1)
                        
                        conditional_output = self.model(
                            images, 
                            timesteps, 
                            encoder_hidden_states=conditional_embeddings
                        ).sample
                        
                        # Apply classifier-free guidance
                        model_output = unconditional_output + self.guidance_scale * (conditional_output - unconditional_output)
                    
                    # Denoise one step
                    images = self.noise_scheduler.step(model_output, t, images).prev_sample
        else:
            # Use diffusers pipeline for non-conditional sampling
            pipeline = DDPMPipeline(
                unet=self.model,
                scheduler=self.noise_scheduler,
            )
            
            with torch.no_grad():
                images = pipeline(
                    batch_size=4,
                    generator=torch.manual_seed(self.trainer.current_epoch),
                    output_type="tensor",
                ).images
        
        # Restore original model parameters if EMA was used
        if self.use_ema and self.ema_model is not None:
            for param, stored in zip(self.model.parameters(), stored_params):
                param.data.copy_(stored.data)
        
        # Log images
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.add_images(
                f"generated_samples/epoch_{self.trainer.current_epoch}", 
                (images + 1) / 2,  # Normalize from [-1, 1] to [0, 1]
                self.trainer.global_step
            )
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers
        """
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )
        
        # Create learning rate scheduler
        scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def on_save_checkpoint(self, checkpoint):
        """
        Include EMA model state in checkpoint
        """
        if self.use_ema:
            checkpoint["ema_model"] = self.ema_model.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """
        Load EMA model state from checkpoint
        """
        if self.use_ema and "ema_model" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_model"])

    def generate_samples(self, batch_size=4, labels=None, guidance_scale=3.0, num_inference_steps=50, noise=None):
        """
        Generate samples with the model.
        
        Args:
            batch_size: Number of samples to generate
            labels: Conditioning labels for generation (if None, use unconditional generation)
            guidance_scale: Strength of classifier-free guidance (higher = stronger conditioning)
            num_inference_steps: Number of denoising steps
            noise: Optional initial noise. If None, random noise will be used.
            
        Returns:
            Generated image tensors
        """
        # Store device for use throughout the method
        device = self.device
        
        # Initialize with random noise if not provided
        if noise is None:
            shape = (batch_size, self.in_channels, self.sample_size, self.sample_size)
            noise = torch.randn(shape, device=device)
            self.last_used_noise = noise  # Store for potential reuse
        else:
            self.last_used_noise = noise  # Store the provided noise
            
        # Configure scheduler for inference
        self.noise_scheduler.set_timesteps(num_inference_steps)
            
        # Start from pure noise
        images = noise
        
        # For conditional model, apply classifier-free guidance
        if self.class_cond and labels is not None:
            with torch.no_grad():
                # Progressive denoising with classifier-free guidance
                for t in self.noise_scheduler.timesteps:
                    # Expand the timesteps tensor for batch dimension
                    timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
                    
                    # Create unconditional embeddings (zero labels)
                    unconditional_embeddings = self.class_embedder(torch.zeros_like(labels))
                    # Add sequence dimension (batch_size, 1, hidden_dim)
                    unconditional_embeddings = unconditional_embeddings.unsqueeze(1)
                    
                    # Predict unconditional noise residual
                    unconditional_output = self.model(
                        images, 
                        timesteps, 
                        encoder_hidden_states=unconditional_embeddings
                    ).sample
                    
                    # Predict conditional noise residual
                    conditional_embeddings = self.class_embedder(labels)
                    # Add sequence dimension (batch_size, 1, hidden_dim)
                    conditional_embeddings = conditional_embeddings.unsqueeze(1)
                    
                    conditional_output = self.model(
                        images, 
                        timesteps, 
                        encoder_hidden_states=conditional_embeddings
                    ).sample
                    
                    # Apply classifier-free guidance formula
                    model_output = unconditional_output + guidance_scale * (conditional_output - unconditional_output)
                    
                    # Denoise one step
                    images = self.noise_scheduler.step(model_output, t, images).prev_sample
        else:
            # For unconditional generation, use standard diffusers pipeline
            pipeline = DDPMPipeline(
                unet=self.model,
                scheduler=self.noise_scheduler,
            )
            
            with torch.no_grad():
                images = pipeline(
                    batch_size=batch_size,
                    generator=torch.manual_seed(0),
                    output_type="tensor",
                    num_inference_steps=num_inference_steps
                ).images
                
        return images