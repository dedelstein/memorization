import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from torch_ema import ExponentialMovingAverage

from src.utils.constants import CHEXPERT_CLASSES
from src.models.unet_guidedddpm import UNetModel  # Changed: use custom UNetModel instead of Diffusers' UNet2DConditionModel

class ClassifierFreeGuidedDiffusion(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        img_size: int = 64,
        in_channels: int = 1,
        out_channels: int = 1,
        num_classes: int = len(CHEXPERT_CLASSES),
        center_crop_size: Optional[int] = None,
        random_flip: bool = True,
        conditioning_dropout_prob: float = 0.1,
        lr: float = 1e-4,
        lr_warmup_steps: int = 500,
        optimizer_type: str = "AdamW",
        lr_scheduler_type: str = "cosine_with_warmup",
        min_lr: float = 1e-6,
        lr_num_cycles: int = 1,
        noise_scheduler_beta_schedule: str = "linear",
        noise_scheduler_num_train_timesteps: int = 1000,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_update_every: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Changed: instantiate custom UNetModel with multi-label support
        self.unet = UNetModel(
            image_size=img_size,
            in_channels=in_channels,
            model_channels=64,
            out_channels=out_channels,
            num_res_blocks=2,
            attention_resolutions=(img_size // 8, img_size // 16, img_size // 32),
            dropout=conditioning_dropout_prob,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=num_classes,
            use_checkpoint=False,
            use_scale_shift_norm=False,
        )

        # optionally load pretrained UNet weights
        if pretrained_model_name_or_path:
            sd = torch.load(pretrained_model_name_or_path, map_location="cpu")
            # if you saved just the unet.state_dict():
            if "state_dict" in sd and "unet" in sd["state_dict"]:
                # Lightning checkpoint -> strip "unet." prefix
                unet_sd = {k.replace("unet.", ""):v
                           for k,v in sd["state_dict"].items()
                           if k.startswith("unet.")}
                self.unet.load_state_dict(unet_sd, strict=False)
            else:
                # pure UNet state-dict
                self.unet.load_state_dict(sd, strict=False)
            print(f"Loaded UNet weights from {pretrained_model_name_or_path}")

        self.noise_scheduler = DDPMScheduler(
            beta_schedule=noise_scheduler_beta_schedule,
            num_train_timesteps=noise_scheduler_num_train_timesteps,
        )

        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.ema = None
        self.ema_unet = None
        self._step_counter = 0

    def on_fit_start(self):
        if self.use_ema:
            self.ema_unet = UNetModel(
                image_size=self.hparams.img_size,
                in_channels=self.hparams.in_channels,
                model_channels=64,
                out_channels=self.hparams.out_channels,
                num_res_blocks=2,
                attention_resolutions=(self.hparams.img_size // 8, self.hparams.img_size // 16, self.hparams.img_size // 32),
                dropout=self.hparams.conditioning_dropout_prob,
                channel_mult=(1, 2, 4, 8),
                conv_resample=True,
                dims=2,
                num_classes=self.hparams.num_classes,
                use_checkpoint=False,
                use_scale_shift_norm=False,
            )
            with torch.no_grad():
                for p_ema, p in zip(self.ema_unet.parameters(), self.unet.parameters()):
                    p_ema.data.copy_(p.data)
            self.ema_unet.to(self.device)
            self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=self.ema_decay)

    def prepare_latents(self, batch_size, channels, height, width, device=None):
        if device is None:
            device = self.device
        return torch.randn((batch_size, channels, height, width), device=device)

    def training_step(self, batch, batch_idx):
        """
        Compute the training loss with conditional guidance and sample filtering by a threshold tau.

        The threshold tau is taken from hyperparameters (self.hparams.tau). Samples whose guidance signal magnitude
        exceeds tau are excluded from the loss to mitigate overly confident conditional predictions.

        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch
        Returns:
            loss tensor
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Initialize latent and noise
        latents = images
        noise = torch.randn_like(latents)

        # Sample random timesteps
        bsz = latents.size(0)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
        ).long()

        # Add noise at sampled timesteps
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Classifier-free guidance dropout
        if self.training and self.hparams.conditioning_dropout_prob > 0:
            keep = torch.bernoulli(
                torch.full((bsz,), 1 - self.hparams.conditioning_dropout_prob, device=self.device)
            )
            cond_labels = labels * keep.unsqueeze(1)
        else:
            cond_labels = labels

        # Predict noise with and without conditioning
        noise_pred_cond = self.unet(noisy_latents, timesteps, y=cond_labels)
        noise_pred_uncond = self.unet(noisy_latents, timesteps, y=None)

        # Compute guidance magnitude per sample
        # Flatten spatial dims and compute L2 norm
        magnitude_diff = torch.norm(
            (noise_pred_cond - noise_pred_uncond).view(bsz, -1),
            dim=1
        )

        # Retrieve tau threshold from hyperparameters
        tau = getattr(self.hparams, 'tau', 1.0)

        # Select samples where magnitude is below threshold
        valid_mask = magnitude_diff <= tau

        if not valid_mask.any():
            # If no valid samples, return a zero loss to maintain graph
            zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log("train_loss", zero_loss, prog_bar=True)
            return zero_loss

        # Compute MSE loss only over valid samples
        loss = F.mse_loss(
            noise_pred_cond[valid_mask],
            noise[valid_mask]
        )

        # Log training loss
        self.log("train_loss", loss, prog_bar=True)

        # Increment internal step counter
        self._step_counter += 1

        return loss

    def on_after_backward(self):
        if (
            self.use_ema
            and self.ema is not None
            and self._step_counter % self.ema_update_every == 0
        ):
            self.ema.update(self.unet.parameters())

    def _get_model_for_evaluation(self):
        if not self.use_ema or self.ema is None:
            return self.unet

        self.ema.copy_to(self.ema_unet.parameters())
        return self.ema_unet

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        latents = images
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        model = self._get_model_for_evaluation()
        noise_pred = model(noisy_latents, timesteps, y=labels)
        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
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

        total_steps = self.trainer.estimated_stepping_batches
        st = self.hparams.lr_scheduler_type.lower()

        if st == "constant_with_warmup":
            scheduler = get_scheduler(
                name="constant_with_warmup",
                optimizer=optimizer,
                num_warmup_steps=self.hparams.lr_warmup_steps,
                num_training_steps=total_steps,
            )
            sched_config = {"scheduler": scheduler, "interval": "step"}
        elif st == "cosine_with_warmup":
            scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.hparams.lr_warmup_steps,
            num_training_steps=total_steps,
            num_cycles=self.hparams.lr_num_cycles,
            )
            sched_config = {"scheduler": scheduler, "interval": "step"}
        
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",   # call .step() each training step
                    # "frequency": 1,     # uncomment to change frequency
                    # "reduce_on_plateau": False,  # only if you use ReduceLROnPlateau
                }
            }
        else:
            return optimizer

    @torch.no_grad()
    def generate_samples(
        self,
        labels: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_samples: int = 1,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 100,
    ) -> torch.Tensor:
        """
        Returns a tensor of shape (N, C, H, W):
          – If `labels` is given, we generate B * num_samples conditioned samples.
          – Otherwise, we generate batch_size * num_samples unconditional samples.
        Sampling is done entirely in float32 to avoid dtype mismatches.
        """
        device = self.device

        # 0) Force float32 precision for UNet and latents
        param_dtype = torch.float32
        self.unet.to(device=device, dtype=param_dtype)
        self.unet.dtype = param_dtype

        # 1) Build initial latents + label‐tensor
        if labels is not None:
            B, K = labels.shape
            labels = labels.to(device=device, dtype=param_dtype)
            labels = labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1, K)
            N = B * num_samples
            latents = torch.randn(
                (N, self.hparams.out_channels, self.hparams.img_size, self.hparams.img_size),
                device=device,
                dtype=param_dtype,
            )
        elif batch_size is not None:
            N = batch_size * num_samples
            labels = None
            latents = torch.randn(
                (N, self.hparams.out_channels, self.hparams.img_size, self.hparams.img_size),
                device=device,
                dtype=param_dtype,
            )
        else:
            raise ValueError("generate_samples requires either `labels` or `batch_size`")

        # 2) Prepare DDIM sampler
        sampler = DDIMScheduler.from_config(self.noise_scheduler.config)
        sampler.set_timesteps(num_inference_steps)

        # 3) Denoising loop with (optional) classifier-free guidance
        for t in sampler.timesteps:
            ts = torch.full((N,), t, device=device, dtype=torch.long)

            if labels is not None:
                e_uncond = self.unet(latents, ts, y=None)
                e_cond   = self.unet(latents, ts, y=labels)
                eps = e_uncond + guidance_scale * (e_cond - e_uncond)
            else:
                eps = self.unet(latents, ts, y=None)

            # All tensors here are float32, matching the scheduler’s buffers
            latents = sampler.step(eps, t, latents).prev_sample

        return latents

