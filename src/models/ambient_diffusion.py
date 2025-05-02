"""
All-in-one utilities for Ambient Diffusion (random-inpainting variant).

Implements
----------
sample_inpainting_mask   – A  (Eq. 3.1)
further_corrupt          – ˜A (Eq. 3.2)
make_ambient_batch       – build (˜A x_t , A) for training
ambient_loss             – J_corr  (Eq. 3.2)
AmbientDDPMPipeline      – fixed-mask sampler (Eq. 3.3)
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import Tuple
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
    B, _, H, W = shape
    return torch.bernoulli(
        torch.full((B, 1, H, W), 1.0 - p, device=device)
    )


def further_corrupt(A: Tensor, δ: float = 0.05) -> Tensor:
    """Sample ˜A = B A by turning surviving pixels off with prob δ."""
    B_mat = torch.bernoulli(torch.full_like(A, 1.0 - δ))
    return B_mat * A

# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def make_ambient_batch(
    clean: Tensor,
    noise_scheduler: DDPMScheduler,
    timesteps: Tensor,
    p: float = 0.9,
    δ: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """
    Returns:
        y_t_tilde : ˜A x_t   (to feed the network)
        A_mask    : A        (needed only for the loss)
    """
    B, C, H, W = clean.shape
    device      = clean.device

    # 1.  A x₀
    A = sample_inpainting_mask((B, C, H, W), p=p, device=device)

    # 2.  add diffusion noise → x_t , then A x_t
    noise   = torch.randn_like(clean)
    x_t     = noise_scheduler.add_noise(clean, noise, timesteps)

    # 3.  ˜A x_t
    A_tilde = further_corrupt(A, δ=δ)
    y_t_tilde = A_tilde * x_t

    return y_t_tilde, A


def ambient_loss(pred: Tensor, clean: Tensor, A_mask: Tensor) -> Tensor:
    """MSE on observable pixels only."""
    diff = A_mask * (pred - clean)
    return 0.5 * diff.pow(2).mean()

# ------------------------------------------------------------------
# Sampler
# ------------------------------------------------------------------

class AmbientDDPMPipeline(DDPMPipeline):
    """
    Fixed-mask sampler (Eq. 3.3).  Drop-in replacement for DDPMPipeline.
    """

    def __init__(self, *args, p_mask: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_mask = p_mask

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        generator=None,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        device = self.device
        h = w = self.unet.config.sample_size

        mask = sample_inpainting_mask(
            (batch_size, 1, h, w), p=self.p_mask, device=device
        )

        x = torch.randn(
            (batch_size, self.unet.config.in_channels, h, w),
            generator=generator,
            device=device,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for t in self.scheduler.timesteps:
            eps = self.unet(mask * x, t).sample         # hθ(˜A, ˜A x_t , t)
            x0_hat = eps                                # network ≈ x₀
            γ = self.scheduler.sigmas[t] ** 2 / (
                self.scheduler.sigmas[t] ** 2 + 1
            )
            x = γ * x + (1 - γ) * x0_hat                # Eq. 3.3
            x = self.scheduler.step(eps, t, x).prev_sample

        x = (x / 2 + 0.5).clamp(0, 1)
        img = x.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            img = self.numpy_to_pil(img)
        return dict(images=img) if return_dict else (img,)
