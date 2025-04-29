import os
import math
import random
from collections import deque
from typing import Optional, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from torch_ema import ExponentialMovingAverage

from src.utils.constants import CHEXPERT_CLASSES
from src.models.unet_guidedddpm import UNetModel


class AmbientDiffusion(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        img_size: int = 64,
        in_channels: int = 1,
        out_channels: int = 1,
        num_classes: int = len(CHEXPERT_CLASSES),
        conditioning_dropout_prob: float = 0.1,
        ambient_t_nature: float = 0.5,
        lr: float = 1e-4,
        lr_warmup_steps: int = 500,
        optimizer_type: str = "AdamW",
        noise_scheduler_beta_schedule: str = "linear",
        noise_scheduler_num_train_timesteps: int = 1000,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_update_every: int = 1,
        cache_size: int = 10000,
    ):
        super().__init__()
        # save hyperparameters (including cache_size)
        self.save_hyperparameters()

        # create custom UNet
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
        self.noise_scheduler = DDPMScheduler(
            beta_schedule=noise_scheduler_beta_schedule,
            num_train_timesteps=noise_scheduler_num_train_timesteps,
        )
        self.register_buffer('betas', self.noise_scheduler.betas.clone())

        # EMA settings
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.ema = None
        self.ema_unet = None
        self._step_counter = 0

        # ambient threshold settings
        self.ambient_t_nature = ambient_t_nature
        total = self.noise_scheduler.config.num_train_timesteps
        self.t_nature_step = int(ambient_t_nature * total)
        self.sigma_t_nature = self.noise_scheduler.betas[self.t_nature_step].sqrt().item()

        # FIFO cache for noisy-at-t_n samples (on GPU)
        self.cache: deque[torch.Tensor] = deque(maxlen=cache_size)

    def on_fit_start(self):
        if self.use_ema:
            self.ema_unet = UNetModel(**self.unet.config)
            with torch.no_grad():
                for ema_p, p in zip(self.ema_unet.parameters(), self.unet.parameters()):
                    ema_p.data.copy_(p.data)
            self.ema_unet.to(self.device)
            self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=self.ema_decay)

    def prepare_latents(self, batch_size, channels, height, width, device=None):
        if device is None:
            device = self.device
        return torch.randn((batch_size, channels, height, width), device=device)

    def _get_ambient_score_matching_loss(self, x_tn_samples, t, noise, cond):
        # compute coefficients
        sigma_t = self.betas[t].sqrt().view(-1,1,1,1)
        sigma_n = self.sigma_t_nature
        alpha_t = (1 - sigma_t**2).sqrt()
        alpha_n = math.sqrt(1 - sigma_n**2)
        coef1 = (alpha_t / alpha_n)
        coef2 = ((sigma_t**2 - sigma_n**2) / (1 - sigma_n**2)).sqrt()
        # build the noisy input
        noisy = coef1 * x_tn_samples + coef2 * noise
        # target is scaled prediction of x_tn
        target = (sigma_n**2 / sigma_t**2).view(-1,1,1,1) * coef1 * noisy - x_tn_samples
        # predict with conditioning
        pred = self.unet(noisy, t, y=cond)
        return F.mse_loss(pred, target)

    def _get_regular_diffusion_loss(self, images, labels, t, noise):
        noisy = self.noise_scheduler.add_noise(images, noise, t)
        # classifier-free guidance dropout
        cond = labels
        if self.training and self.hparams.conditioning_dropout_prob > 0:
            keep = torch.bernoulli(
                torch.full((images.size(0),), 1 - self.hparams.conditioning_dropout_prob, device=self.device)
            )
            cond = cond * keep.unsqueeze(1)
        pred = self.unet(noisy, t, y=cond)
        return F.mse_loss(pred, noise)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        bsz = images.size(0)
        noise = torch.randn_like(images)

        # 1) Fill/update cache with new noisy-at-t_n samples
        t_n = torch.full((bsz,), self.t_nature_step, device=self.device, dtype=torch.long)
        with torch.no_grad():
            x_tn = self.noise_scheduler.add_noise(images, torch.randn_like(images), t_n)
        for xtn in x_tn:
            # cache on GPU as float32
            self.cache.append(xtn.detach())

        # 2) Randomly choose phase
        if torch.rand(1).item() > 0.5 and len(self.cache) >= bsz:
            # --- ambient phase with classifier-free conditioning ---
            # sample bsz entries from the cache
            sampled_xtn = random.sample(self.cache, k=bsz)
            x_tn_batch = torch.stack(sampled_xtn, dim=0)
            # sample high-noise timesteps
            t = torch.randint(
                self.t_nature_step + 1,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=self.device,
            )
            # conditioning dropout
            cond = labels
            if self.training and self.hparams.conditioning_dropout_prob > 0:
                keep = torch.bernoulli(
                    torch.full((bsz,), 1 - self.hparams.conditioning_dropout_prob, device=self.device)
                )
                cond = cond * keep.unsqueeze(1)
            loss = self._get_ambient_score_matching_loss(x_tn_batch, t, noise, cond)
            self.log('ambient_loss', loss, prog_bar=True)
        else:
            # --- regular DDPM phase ---
            t = torch.randint(0, self.t_nature_step + 1, (bsz,), device=self.device)
            loss = self._get_regular_diffusion_loss(images, labels, t, noise)
            self.log('standard_loss', loss, prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)
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
        images, labels = images.to(self.device), labels.to(self.device)
        bsz = images.size(0)
        noise = torch.randn_like(images)
        t = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=self.device,
        )
        noisy = self.noise_scheduler.add_noise(images, noise, t)
        model = self._get_model_for_evaluation()
        pred = model(noisy, t, y=labels)
        loss = F.mse_loss(pred, noise)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt_cls = torch.optim.AdamW if self.hparams.optimizer_type.lower() == 'adamw' else torch.optim.Adam
        optimizer = opt_cls(self.unet.parameters(), lr=self.hparams.lr)
        scheduler = get_scheduler(
            name='constant_with_warmup',
            optimizer=optimizer,
            num_warmup_steps=self.hparams.lr_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def generate_samples(
        self,
        batch_size: int = 4,
        labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 50,
        return_all_timesteps: bool = False,
        initial_noise: Optional[torch.Tensor] = None,
    ):
        """Generate samples with detailed debugging."""
        device = self.device
        model = self._get_model_for_evaluation().eval()
        
        # Build scheduler
        scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(num_inference_steps)

        # Handle labels
        if labels is not None:
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, device=device)
            elif labels.dim() == 0:
                idx = int(labels)
                labels = torch.zeros(batch_size, len(CHEXPERT_CLASSES), device=device)
                labels[:, idx] = 1.0
        else:
            labels = torch.zeros(batch_size, len(CHEXPERT_CLASSES), device=device)
            labels[:, CHEXPERT_CLASSES.index("No Finding")] = 1.0

        # Handle latents
        if initial_noise is not None:
            latents = initial_noise.to(device)
        else:
            latents = self.prepare_latents(
                batch_size, model.in_channels, model.image_size, model.image_size, device
            )

        # Handle classifier-free guidance
        if guidance_scale > 1.0:
            uncond = torch.zeros_like(labels)
            cat_labels = torch.cat([uncond, labels], dim=0)
            latents = torch.cat([latents, latents], dim=0)
        else:
            cat_labels = labels

        # Denoising loop
        all_latents = []
        with torch.no_grad():
            for t in scheduler.timesteps:
                
                # Convert timestep to tensor with correct shape
                timesteps = torch.full(
                    (latents.shape[0],), t, device=device, dtype=torch.long
                )
                
                try:
                    # Model forward pass
                    model_output = model(latents, timesteps, y=cat_labels)
                    
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = model_output
                    
                    # Scheduler step
                    scheduler_out = scheduler.step(noise_pred, t, latents)
                    latents = scheduler_out.prev_sample
                    
                except Exception as e:
                    print(f"[DEBUG] Error during model forward pass or scheduler step:")
                    print(f"Error type: {type(e)}")
                    print(f"Error message: {str(e)}")
                    import traceback
                    print(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
                    raise

                if return_all_timesteps:
                    all_latents.append(latents[:batch_size].clone())

        # Final output processing
        final_latents = latents[:batch_size]
        images = self.denormalize_latents(final_latents)

        if return_all_timesteps:
            all_images = [self.denormalize_latents(x) for x in all_latents]
            return images, all_images

        return images

    def denormalize_latents(self, latents):
        imgs = (latents * 0.5 + 0.5).clamp(0,1)
        return (imgs * 255).type(torch.uint8)

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema and self.ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
            checkpoint['ema_unet_state_dict'] = self.ema_unet.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.use_ema:
            if self.ema_unet is None:
                self.ema_unet = UNetModel(**self.unet.config)
                self.ema_unet.load_state_dict(checkpoint.get('ema_unet_state_dict', {}))
                self.ema_unet.to(self.device)
            if self.ema is None:
                self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=self.ema_decay)
            if 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
