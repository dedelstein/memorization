"""
Classifier-Free Guidance implementation using the diffusers library.
"""

import torch.nn as nn
from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DModel

class CustomClassConditionedUnet(UNet2DModel):
    """UNet2DModel adapted for multi-hot classification vectors"""

    def __init__(
        self,
        sample_size,
        in_channels=1,
        out_channels=1,
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
        multihot_dim=14,
        **kwargs,
    ):
        # Remove conflicting parameters if they exist
        kwargs.pop("class_embed_type", None)
        kwargs.pop("num_class_embeds", None)

        # Initialize the base model without class conditioning
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            **kwargs,
        )

        # Compute the time embedding dimension
        time_embed_dim = block_out_channels[0] * 4

        # Replace the class embedding with a linear layer for multihot vectors
        self.class_embedding = nn.Linear(multihot_dim, time_embed_dim)

        # Save the multihot dimension
        self.config.multihot_dim = multihot_dim

    def forward(self, sample, timestep, class_labels=None, return_dict=True):
        """
        Forward pass that accepts multi-hot vectors for class_labels

        Args:
            sample: Image tensor [batch_size, channels, height, width]
            timestep: Time steps [batch_size] or scalar
            class_labels: Multi-hot vector [batch_size, multihot_dim]
            return_dict: Whether to return a dictionary or just the sample

        Returns:
            Model prediction (noise or clean image depending on configuration)
        """
        return super().forward(sample, timestep, class_labels, return_dict)
