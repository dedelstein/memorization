#!/usr/bin/env python3
"""
Script to generate samples from trained diffusion models with either
classifier guidance or classifier-free guidance.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Import both types of diffusion models
from src.models.cfg_diffusion import ClassifierFreeGuidedDiffusion
from src.models.cg_diffusion import ClassifierGuidedDiffusion
from src.utils.constants import CHEXPERT_CLASSES


def save_images(images, output_dir, prefix="sample", title=None):
    """
    Save individual images and a grid of images.

    Args:
        images: Tensor of images [batch_size, channels, height, width]
        output_dir: Directory to save images
        prefix: Prefix for filenames
        title: Optional title for grid image
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save individual images
    for i, img in enumerate(images):
        # Convert to numpy and normalize to [0, 1]
        if torch.is_tensor(img):
            img = img.permute(1, 2, 0).cpu().numpy()
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0

        # Plot and save
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap="gray" if img.shape[-1] == 1 else None)
        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_{i:04d}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    # Create and save grid
    n_cols = min(4, len(images))
    n_rows = (len(images) + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    if title:
        plt.suptitle(title, fontsize=16)

    for i, img in enumerate(images):
        if i >= n_rows * n_cols:
            break

        plt.subplot(n_rows, n_cols, i + 1)

        # Convert to numpy and normalize to [0, 1]
        if torch.is_tensor(img):
            img = img.permute(1, 2, 0).cpu().numpy()
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0

        plt.imshow(img, cmap="gray" if img.shape[-1] == 1 else None)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_grid.png"), bbox_inches="tight")
    plt.close()


def generate_samples(args):
    """
    Generate samples from a trained diffusion model.

    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine model type from checkpoint path
    if "cfg" in args.model_path.lower():
        print("Loading Classifier-Free Guided Diffusion model")
        model = ClassifierFreeGuidedDiffusion.load_from_checkpoint(
            args.model_path, map_location=device
        )
        guidance_type = "cfg"
    else:
        print("Loading Classifier-Guided Diffusion model")
        model = ClassifierGuidedDiffusion.load_from_checkpoint(
            args.model_path, map_location=device
        )

        # For classifier guidance, we also need to load the classifier
        if args.classifier_path:
            print(f"Loading classifier from {args.classifier_path}")
            model.load_classifier(args.classifier_path)
        else:
            raise ValueError("Classifier path must be provided for classifier guidance")

        guidance_type = "cg"

    # Move model to device
    model.to(device)
    model.eval()

    # Generate samples for each condition
    if args.conditions:
        condition_indices = args.conditions
    else:
        # Default to interesting conditions
        condition_indices = [
            CHEXPERT_CLASSES.index("No Finding"),
            CHEXPERT_CLASSES.index("Pneumonia"),
            CHEXPERT_CLASSES.index("Cardiomegaly"),
            CHEXPERT_CLASSES.index("Pleural Effusion"),
        ]

    for condition_idx in condition_indices:
        condition_name = CHEXPERT_CLASSES[condition_idx]
        print(f"Generating samples for condition: {condition_name}")

        # Create condition label
        label = torch.zeros((args.num_samples, len(CHEXPERT_CLASSES)), device=device)
        label[:, condition_idx] = 1.0

        # Generate samples
        if guidance_type == "cfg":
            samples = model.generate_samples(
                batch_size=args.num_samples,
                labels=label,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.inference_steps,
            )
        else:
            samples = model.generate_samples(
                batch_size=args.num_samples,
                labels=label,
                classifier_scale=args.guidance_scale,
                num_inference_steps=args.inference_steps,
            )

        # Save samples
        condition_dir = os.path.join(args.output_dir, condition_name.replace(" ", "_"))
        save_images(
            samples,
            condition_dir,
            prefix=f"{guidance_type}_sample",
            title=f"{condition_name} - {guidance_type.upper()} Scale: {args.guidance_scale}",
        )

    print(f"Generated samples saved to {args.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate samples from a trained diffusion model"
    )

    # Model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained diffusion model",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default=None,
        help="Path to the classifier (for CG)",
    )

    # Generation parameters
    parser.add_argument(
        "--output_dir", type=str, default="generated_samples", help="Output directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate per condition",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Guidance scale (higher = stronger conditioning)",
    )
    parser.add_argument(
        "--inference_steps", type=int, default=20, help="Number of inference steps"
    )

    # Condition parameters
    parser.add_argument(
        "--conditions",
        type=int,
        nargs="+",
        default=None,
        help="Label indices to condition on",
    )
    parser.add_argument(
        "--list_classes",
        action="store_true",
        help="List available class indices for conditioning",
    )

    args = parser.parse_args()

    # List class indices if requested
    if args.list_classes:
        print("Available classes for conditioning:")
        for i, class_name in enumerate(CHEXPERT_CLASSES):
            print(f"{i}: {class_name}")
        return

    # Generate samples
    generate_samples(args)


if __name__ == "__main__":
    main()
