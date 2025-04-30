"""
Helper functions for the project.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils


def _extract_into_tensor(arr, timesteps, broadcast_shape):
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




def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_image(img):
    """Normalize image to range [0, 1]."""
    return (img - img.min()) / max((img.max() - img.min()), 1e-8)


def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a numpy array for visualization."""
    img = tensor.detach().cpu().numpy()
    img = np.squeeze(img)
    if img.ndim == 2:
        return img
    else:
        return img.transpose(1, 2, 0)  # CHW -> HWC


def save_images(images, path, nrow=4, title=None, normalize=True):
    """Save a batch of images to a file."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert to [0, 1] range if needed
    if images.dtype == torch.uint8:
        images = images.float() / 255.0

    # Create grid
    grid = vutils.make_grid(images, nrow=nrow, normalize=normalize, padding=2)

    # Convert to numpy and transpose
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    # Plot and save
    plt.figure(figsize=(10, 10))
    if title:
        plt.title(title)
    plt.imshow(grid_np)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()
