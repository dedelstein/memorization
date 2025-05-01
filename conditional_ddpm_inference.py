import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from src.utils.constants import CHEXPERT_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="CheXpert DDPM Inference Script")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate per condition",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Resolution of generated images",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Scale for classifier-free guidance (higher values strengthen the condition)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[],
        help="Specific conditions to generate (from CHEXPERT_CLASSES). If empty, generates for all classes."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--ambient",
        action="store_true",
        help="Use ambient diffusion model",
    )
    
    return parser.parse_args()

def load_model(model_path, ambient=False):
    """
    Load a trained DDPM model from the specified path manually
    """
    # Importamos aquí para evitar problemas de importación circular
    from src.models.cfg_diffusion import CustomClassConditionedUnet
    from src.models.conditional_ddpm_pipeline import ConditionalDDPMPipeline
    from src.models.ambient_diffusion import AmbientDDPMPipeline
    
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

def create_class_conditions(conditions, device):
    """
    Create condition vectors based on specified classes
    """
    # If no specific conditions are given, use all classes
    if not conditions:
        conditions = CHEXPERT_CLASSES
    
    # Create a condition vector for each specified condition
    condition_vectors = []
    condition_names = []
    
    for condition in conditions:
        if condition not in CHEXPERT_CLASSES:
            print(f"Warning: {condition} is not in CHEXPERT_CLASSES, skipping.")
            continue
            
        # Create a one-hot encoded vector
        condition_vector = torch.zeros(len(CHEXPERT_CLASSES), device=device)
        condition_index = CHEXPERT_CLASSES.index(condition)
        condition_vector[condition_index] = 1.0
        
        condition_vectors.append(condition_vector)
        condition_names.append(condition)
    
    return torch.stack(condition_vectors), condition_names

def generate_images(pipeline, class_labels, condition_names, num_images, num_inference_steps, guidance_scale, seed):
    """
    Generate images for each condition
    """
    generated_images = {}
    
    # Set the seed for reproducibility
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    for i, condition_name in enumerate(condition_names):
        print(f"Generating images for condition: {condition_name}")
        condition_images = []
        
        # Generate multiple images for this condition
        for j in range(num_images):
            # Update seed for variety
            generator = torch.Generator(device=pipeline.device).manual_seed(seed + j)
            
            # Use a single condition for generation
            condition = class_labels[i].unsqueeze(0)
            
            # Generate the image
            result = pipeline(
                generator=generator,
                batch_size=1,
                num_inference_steps=num_inference_steps,
                output_type="np",
                class_labels=condition,
                guidance_scale=guidance_scale
            )
            
            # Get the generated image
            image = result["images"][0]
            
            # Convert to 3-channel for visualization
            image = np.repeat(image, 3, axis=2)
            
            # Denormalize the image
            image = (image * 255).round().astype("uint8")
            
            condition_images.append(image)
        
        generated_images[condition_name] = condition_images
    
    return generated_images

def save_images(generated_images, output_dir):
    """
    Save generated images to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for condition_name, images in generated_images.items():
        # Create a subdirectory for this condition
        condition_dir = os.path.join(output_dir, condition_name.replace(" ", "_"))
        os.makedirs(condition_dir, exist_ok=True)
        
        # Save each image
        for i, image in enumerate(images):
            image_path = os.path.join(condition_dir, f"{i+1}.png")
            Image.fromarray(image).save(image_path)
            
    print(f"Images saved to {output_dir}")

def create_grid_visualization(generated_images, output_dir):
    """
    Create a grid visualization of the generated images
    """
    conditions = list(generated_images.keys())
    num_conditions = len(conditions)
    num_images = len(generated_images[conditions[0]])
    
    # Create a grid figure
    fig, axes = plt.subplots(num_conditions, num_images, figsize=(num_images * 3, num_conditions * 3))
    
    # If there's only one condition, make sure axes is 2D
    if num_conditions == 1:
        axes = axes.reshape(1, -1)
    
    # Add each image to the grid
    for i, condition in enumerate(conditions):
        for j in range(num_images):
            ax = axes[i, j]
            ax.imshow(generated_images[condition][j])
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add title only for the first column
            if j == 0:
                ax.set_ylabel(condition, fontsize=10, rotation=0, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_visualization.png"), dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    # Print the available CHEXPERT classes if no specific conditions are provided
    if not args.conditions:
        print("No specific conditions provided. Available classes:")
        for i, cls in enumerate(CHEXPERT_CLASSES):
            print(f"  {i}: {cls}")
        print(f"Will generate images for all {len(CHEXPERT_CLASSES)} conditions")
    
    # Load the model
    pipeline = load_model(args.model_path, ambient=args.ambient)
    
    # Create class conditions
    class_labels, condition_names = create_class_conditions(args.conditions, pipeline.device)
    
    # Generate images
    generated_images = generate_images(
        pipeline=pipeline,
        class_labels=class_labels,
        condition_names=condition_names,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )
    
    # Save images
    save_images(generated_images, args.output_dir)
    
    # Create grid visualization
    create_grid_visualization(generated_images, args.output_dir)
    
    print("Image generation complete!")

if __name__ == "__main__":
    main()