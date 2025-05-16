import os
import torch
from torch import Tensor
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
from typing import Tuple, Optional
from diffusers import DDPMScheduler, DDPMPipeline
import torch.nn as nn
from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DModel
import sys

CHEXPERT_CLASSES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
# from conditional_ddpm_inference import load_model, save_images, generate_images

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

class ConditionalDDPMPipeline(DDPMPipeline):
    """DDPM Pipeline with class conditioning support"""

    def __call__(
        self,
        batch_size=1,
        generator=None,
        num_inference_steps=1000,
        output_type="pil",
        class_labels=None,
        guidance_scale=3.0,
        return_dict=True,
        mem=False,
    ):
        # Initialize with random noise (exactly like original)
        image = torch.randn(
            (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            ),
            generator=generator,
            device=self.device,
        )

        # Setup the scheduler (exactly like original)
        self.scheduler.set_timesteps(num_inference_steps)

        if mem:
            TCNP = torch.empty((batch_size, num_inference_steps), device=self.device)

        # Denoising process
        for i, t in enumerate(self.scheduler.timesteps):
            # Only difference is we pass class_labels to the model
            with torch.no_grad():
                if guidance_scale > 1.0 and class_labels is not None:
                    # Conditional pass
                    cond_output = self.unet(
                        image, t, class_labels=class_labels
                    ).sample

                    # Unconditional pass
                    uncond_labels = torch.zeros_like(class_labels)
                    uncond_output = self.unet(
                        image, t, class_labels=uncond_labels
                    ).sample

                    # Combine with guidance scale
                    model_output = uncond_output + guidance_scale * (
                        cond_output - uncond_output
                    )

                    if mem:
                        # print(cond_output.squeeze().shape, uncond_output.shape)
                        TCNP[:, i] = torch.linalg.norm(cond_output.squeeze() - uncond_output.squeeze(), dim=[0,1])
                else:
                    # Standard pass with conditioning
                    model_output = self.unet(
                        image, t, class_labels=class_labels
                    ).sample

            # Scheduler step (exactly like original)
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

        # Final processing (exactly like original)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)
        
        if mem:
            return dict(images=image, TCNP=TCNP)

        return dict(images=image)

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

def load_model(model_path, ambient=False):
    """
    Load a trained DDPM model from the specified path manually
    """
    # Importamos aquí para evitar problemas de importación circular
    # from src.models.cfg_diffusion import CustomClassConditionedUnet
    # from src.models.conditional_ddpm_pipeline import ConditionalDDPMPipeline
    # from src.models.ambient_diffusion import AmbientDDPMPipeline
    
    print(f"Loading model components from {model_path}")
    
    # Cargar el modelo manualmente en lugar de usar from_pretrained
    try:
        # Cargar UNet
        unet_path = os.path.join(model_path, "custom_unet")
        if os.path.exists(unet_path):
            print(f"Loading UNet from {unet_path}")
            unet = CustomClassConditionedUnet.from_pretrained(unet_path)
        else:
            # Intentar cargar directamente si la estructura es diferente
            print(f"Trying to load UNet directly from {model_path}")
            unet = CustomClassConditionedUnet.from_pretrained(model_path)
        
        # Cargar scheduler
        scheduler_path = os.path.join(model_path, "scheduler")
        if os.path.exists(scheduler_path):
            print(f"Loading scheduler from {scheduler_path}")
            scheduler = DDPMScheduler.from_pretrained(scheduler_path)
        else:
            # Crear un scheduler por defecto si no está guardado
            print("Creating default scheduler")
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule="linear",
                prediction_type="epsilon",
            )
        
        # Crear pipeline manualmente
        if ambient:
            pipeline = AmbientDDPMPipeline(
                unet=unet,
                scheduler=scheduler
            )
        else:
            pipeline = ConditionalDDPMPipeline(
                unet=unet,
                scheduler=scheduler
            )
        
        # Mover a GPU si está disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = pipeline.to(device)
        
        print("Model loaded successfully")
        return pipeline
    
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Intentar un método alternativo si falla el primero
        try:
            print("Attempting alternative loading method...")
            
            # Cargar el estado del modelo directamente
            unet = CustomClassConditionedUnet(sample_size=128)
            
            # Intentar cargar los pesos manualmente
            unet_state_dict_path = os.path.join(model_path, "custom_unet", "diffusion_pytorch_model.bin")
            if os.path.exists(unet_state_dict_path):
                print(f"Loading UNet state dict from {unet_state_dict_path}")
                unet.load_state_dict(torch.load(unet_state_dict_path, map_location="cpu"))
            else:
                # Buscar el archivo .bin en otras ubicaciones
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        if file.endswith(".bin"):
                            print(f"Found model weights at {os.path.join(root, file)}")
                            unet.load_state_dict(torch.load(os.path.join(root, file), map_location="cpu"))
                            break
            
            # Crear scheduler
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule="linear",
                prediction_type="epsilon",
            )
            
            # Crear pipeline
            pipeline = ConditionalDDPMPipeline(
                unet=unet,
                scheduler=scheduler
            )
            
            # Mover a GPU si está disponible
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline = pipeline.to(device)
            
            print("Model loaded successfully using alternative method")
            return pipeline
            
        except Exception as e2:
            print(f"Alternative loading method also failed: {e2}")
            raise ValueError(f"Could not load model from {model_path}. Original error: {e}. Second error: {e2}")

def generate_images(pipeline, class_labels, condition_names, num_images, num_inference_steps, guidance_scale, seed, mem=False):
    """
    Generate images for each condition
    """
    generated_images = {}
    if mem:
        metrics = torch.empty(len(condition_names), num_inference_steps)
    
    # Set the seed for reproducibility
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    for i, condition_name in enumerate(condition_names):
        print(f"Generating images for condition: {condition_name}")
        condition_images = []
        
        # Generate multiple images for this condition
        for j in range(num_images):
            # Update seed for variety
            generator = torch.Generator(device=pipeline.device).manual_seed(seed + i + j)
            
            # Use a single condition for generation
            condition = class_labels[i].unsqueeze(0)
            
            # Generate the image
            result = pipeline(
                generator=generator,
                batch_size=1,
                num_inference_steps=num_inference_steps,
                output_type="np",
                class_labels=condition,
                guidance_scale=guidance_scale,
                mem=mem
            )
            
            # Get the generated image
            image = result["images"][0]
            
            # Convert to 3-channel for visualization
            image = np.repeat(image, 3, axis=2)
            
            # Denormalize the image
            image = (image * 255).round().astype("uint8")
            
            condition_images.append(image)
            if mem:
                # print(result["TCNP"])
                metrics[i, :] = result["TCNP"]
        
        generated_images[condition_name] = condition_images

    if mem:
        return generated_images, metrics
    
    return generated_images

def save_images(generated_images, output_dir):
    """
    Save generated images to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for condition_name, images in generated_images.items():
        # Create a subdirectory for this condition
        condition_dir = output_dir # os.path.join(output_dir, condition_name.replace(" ", "_"))
        # os.makedirs(condition_dir, exist_ok=True)
        
        # Save each image
        for i, image in enumerate(images):
            image_path = os.path.join(condition_dir, f'{condition_name.replace(" ","_")}.png')
            Image.fromarray(image).save(image_path)
            
    print(f"Images saved to {output_dir}")



#################################################################################################################################

# model_path = "checkpoint-31000"
# model_path = "model_epoch_99"
model_path = sys.argv[1]
print(model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")



pipe = load_model(model_path, ambient=False)
pipe.unet.eval()

GUIDE   = 1.5        # or sweep multiple values
STEPS   = 1000
device  = pipe.device
print(device)

output_dir = 'cfg_samples'

seed = 42

# condition_names = ["No Finding",
#                    "Pneumonia",
#                    "Cardiomegaly",
#                    "Pleural Effusion"]
# condition_names = ["Common 1", "Common 2", "Common 3", "Common 4", "Common 5",
#                     "Rare 1", "Rare 2", "Rare 3", "Rare 4", "Rare 5", "Rare 6", "Rare 7", "Rare 8", "Rare 9"]

# condition_indices = [
#     [3, 10, 13], [3, 13], [3], [3, 5, 10, 13], [0],
#     [1, 2, 3, 5, 7], [1, 2, 3, 5, 8, 10, 13], [1, 3, 9, 13], [8, 9, 10, 13], [5, 6], [11, 12, 13], [7, 8], [4, 7], [4, 6]
# ]
labels = np.load('labels.npy')
counts = np.load('labels_counts.npy')
labels1 = labels[:10]
labels2 = labels[-90:]
labelst = np.vstack((labels1, labels2))

condition_names = []
condition_indices = []
labels = []

print(labelst)
for i in range(labelst.shape[0]):
    idcs = (np.where(labelst[i])[0]).tolist()
    condition_indices.append(idcs)
    name = f'{(np.where(labelst[i])[0]+1).tolist()}'
    condition_names.append(name)
    label = torch.zeros(len(CHEXPERT_CLASSES), device=device)
    label[idcs] = 1.0
    labels.append(label)
# print(labels)
# print(condition_indices)
print(condition_names)
# condition_names = ["Rare 6", "Rare 7", "Rare 8", "Rare 9"]
# condition_indices = [[11, 12, 13], [7, 8], [4, 7], [4, 6]]
# labels = torch.zeros((len(condition_indices), len(CHEXPERT_CLASSES)), device=device)
# labels = []
# for i, condition_idx in enumerate(condition_indices):
#     label = torch.zeros(len(CHEXPERT_CLASSES), device=device)
#     label[condition_idx] = 1.0
#     print(label)
#     labels.append(label)


#################################################################################
    
# Set the seed for reproducibility
generator = torch.Generator(device=pipe.device).manual_seed(seed)

images, metrics = generate_images(pipe, labels, condition_names, 1, STEPS, GUIDE, seed, mem=True)

# Generate the image
# result = pipe(
#     generator=generator,
#     batch_size=len(condition_indices),
#     num_inference_steps=STEPS,
#     output_type="np",
#     class_labels=labels,
#     guidance_scale=GUIDE,
#     mem=True
# )

# # Get the generated image
# images = result["images"][0]

# # Convert to 3-channel for visualization
# # image = np.repeat(image, 3, axis=2)

# # Denormalize the image
# # image = (image * 255).round().astype("uint8")

# metrics = result["TCNP"]

print(f'Mean metric over diffusion time: {metrics.mean(dim=1)}, max: {metrics.max(dim=1)}')

np.save(f'metrics_{STEPS}_{seed}_finetuned', metrics.cpu().numpy())

condition_dir = os.path.join(output_dir, 'finetuned')
os.makedirs(condition_dir, exist_ok=True)

# # Save images
# # save_images(generated_images, condition_dir)
# for i, image in enumerate(images):
#     image = np.repeat(image, 3, axis=2)
    
#     # Denormalize the image
#     image = (image * 255).round().astype("uint8")
#     print(image)
#     print(type(image))
#     print(image.shape)
#     image_path = os.path.join(condition_dir, f"{i+1}.png")
#     Image.fromarray(image).save(image_path)

save_images(images, output_dir=condition_dir)

# Create grid visualization
# create_grid_visualization(generated_images, condition_dir)

print(f"Generated samples saved to {condition_dir}")