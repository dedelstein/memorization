"""
All-in-one utilities for Ambient Diffusion (random-inpainting variant).

Implements
----------
sample_inpainting_mask   - A  (Eq. 3.1)
further_corrupt          - ~A (Eq. 3.2)
make_ambient_batch       - build (~A x_t , A) for training
ambient_loss             - J_corr  (Eq. 3.2)
AmbientDDPMPipeline      - fixed-mask sampler (Eq. 3.3)
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Tuple, Optional
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import DDPMPipeline

# ------------------------------------------------------------------
# Corruptions
# ------------------------------------------------------------------

def sample_inpainting_mask(
    shape: Tuple[int, int, int, int],
    p: float = 0.9,
    device: torch.device | str | None = None,
) -> Tensor:
    """Diagonal Bernoulli mask A: 1 = keep, 0 = erase."""
    B, C, H, W = shape
    return torch.bernoulli(torch.full((B, 1, H, W), p, device=device)).expand(-1, C, -1, -1)


def further_corrupt(A: Tensor, delta: float = 0.05) -> Tensor:
    """Sample ~A = B A by turning surviving pixels off with prob delta."""
    B_mat = torch.bernoulli(torch.full_like(A, 1.0 - delta))
    return B_mat * A

# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def make_ambient_batch(
    clean: Tensor,
    noise_scheduler: DDPMScheduler,
    timesteps: Tensor,
    p: float = 0.9,
    delta: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """
    Returns:
        y_t_tilde : ~A x_t   (to feed the network)
        A_mask    : A        (needed only for the loss)
    """
    B, C, H, W = clean.shape
    device      = clean.device

    # 1.  A x₀
    A = sample_inpainting_mask((B, C, H, W), p=p, device=device)

    # 2.  add diffusion noise → x_t , then A x_t
    noise   = torch.randn_like(clean)

    if timesteps.ndim == 0:
        timesteps = torch.full((B,), timesteps, device=device, dtype=torch.long)
    elif timesteps.shape[0] != B:
        raise ValueError(f"Timesteps batch size {timesteps.shape[0]} does not match input batch size {B}")
    
    x_t = noise_scheduler.add_noise(clean, noise, timesteps)

    # 3.  ~A x_t
    A_tilde     = further_corrupt(A, delta=delta)
    mask_ch     = A_tilde[:, :1]                       # (B,1,H,W)
    net_input   = torch.cat([A_tilde * x_t, mask_ch], dim=1)
    return net_input, A

def ambient_loss(pred, clean, A_mask, snr_weights=None):
    """
    Masked L2 loss with optional SNR weights (Eq. 2 in the paper).
    """
    diff = A_mask * (pred - clean)
    loss = 0.5 * diff.pow(2)
    if snr_weights is not None:
        loss = snr_weights * loss
    return loss.mean()
# ------------------------------------------------------------------
# Sampler
# ------------------------------------------------------------------

class AmbientDDPMPipeline(DDPMPipeline):
    """
    Fixed-mask sampler (Eq. 3.3).  Drop-in replacement for DDPMPipeline.
    """

    def __init__(self, *, unet, scheduler, p_mask: float = 0.9):
        # 1) Let base class register the trainable modules
        super().__init__(unet=unet, scheduler=scheduler)

        # 2) Store hyper‑parameter in the *config* **and** as an attribute
        self.register_to_config(p_mask=p_mask)   # guarantees serialisation
        self.p_mask = p_mask                     # convenient runtime access
        
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 250,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        output_type: str = "pt",
        return_dict: bool = True,
    ):
        device = self.device
        h = w = self.unet.config.sample_size

        mask = sample_inpainting_mask(
            (batch_size, 1, h, w), p=self.p_mask, device=device
        )

        x = torch.randn(
            (batch_size, 1, h, w),
            generator=generator,
            device=device,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)


        if class_labels is None:
                cond_lbls = torch.zeros(
                    (batch_size,
                    getattr(self.unet.config, "multihot_dim", 1)),
                    dtype=torch.long if not hasattr(self.unet.config, "multihot_dim") else torch.float,
                    device=device,
                )
        else:
            cond_lbls = class_labels.to(device)

        uncond_lbls = torch.zeros_like(cond_lbls)

        for t in self.scheduler.timesteps:

            if guidance_scale == 1.0 or class_labels is None:
                # single unconditional (or conditional) pass
                net_inp = torch.cat([mask * x, mask[:, :1]], dim=1)
                eps = self.unet(net_inp, t, class_labels=cond_lbls).sample
            else:
                # duplicate batch: cond | uncond
                net_inp = torch.cat([mask * x, mask[:, :1]], dim=1)  # (B,C+1,H,W)
                inp     = torch.cat([net_inp, net_inp], dim=0)       # (2B,C+1,H,W)
                lbls = torch.cat([cond_lbls, uncond_lbls], dim=0)
                tids = t.expand(2 * batch_size)

                eps_cond, eps_uncond = (
                    self.unet(inp, tids, class_labels=lbls).sample.chunk(2)
                )
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # ---------- Eq. 3.3 update (must be INSIDE the loop) ----------
            x0_hat = eps
            sigma = self.scheduler._get_variance(t).sqrt()
            gamma = sigma ** 2 / (sigma ** 2 + 1)
            x = gamma * x + (1 - gamma) * x0_hat
            x = self.scheduler.step(eps, t, x).prev_sample

        x = (x / 2 + 0.5).clamp(0, 1)
        img = x.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            img = self.numpy_to_pil(img)
        return dict(images=img) if return_dict else (img,)