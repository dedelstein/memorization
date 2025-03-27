import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
from diffusers import UNet2DModel, UNet2DConditionModel
import torch.nn as nn
from config import CONFIG

class ClassEmbedding(nn.Module):
    """Class embedding for conditional UNet."""
    
    def __init__(self, num_classes, embed_dim):
        """Initialize class embedding.
        
        Args:
            num_classes: Number of classes
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, class_labels):
        """Generate embeddings for class labels.
        
        Args:
            class_labels: Class indices tensor
            
        Returns:
            Tensor of class embeddings
        """
        embeddings = self.embedding(class_labels)
        return self.projection(embeddings).unsqueeze(1)

class Diffusion:
    """Diffusion model process."""
    
    def __init__(self, model, steps, beta_schedule='cosine', num_classes=None, class_embed=None, 
                 classifier=None, guidance_method="classifier-free"):
        """Initialize diffusion process.
        
        Args:
            model: UNet model for noise prediction
            steps: Number of diffusion steps
            beta_schedule: Type of beta schedule ('cosine' or 'linear')
            num_classes: Number of classes for conditional generation
            class_embed: Class embedding module for conditional UNets
            classifier: External classifier for guidance
            guidance_method: Type of guidance to use
        """
        self.model = model
        self.steps = steps
        self.num_classes = num_classes
        self.class_embed = class_embed
        self.is_conditional = isinstance(model, UNet2DConditionModel)
        self.classifier = classifier
        self.guidance_method = guidance_method
        
        if beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(steps)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, steps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Create a cosine beta schedule.
        
        Args:
            timesteps: Number of diffusion steps
            s: Small offset for the cosine schedule
            
        Returns:
            Tensor of beta values
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * (math.pi * 0.5)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)
    
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0).
        
        Args:
            x_0: Clean input image
            t: Timestep
            noise: Optional pre-generated noise
            
        Returns:
            Tuple of noisy image and added noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise, noise
    
    def _run_model(self, x_t, t, y=None):
        """Run model with appropriate format for conditional/unconditional.
        
        Args:
            x_t: Input tensor
            t: Timestep tensor
            y: Optional class labels
            
        Returns:
            Tensor of predicted noise
        """
        t_float = t.float()
    
        if self.is_conditional:
            if y is not None and self.class_embed is not None:
                y_emb = self.class_embed(y)
                return self.model(x_t, t_float, encoder_hidden_states=y_emb).sample
            else:
                batch_size = x_t.shape[0]
                device = x_t.device
                # Create empty embeddings with proper shape: [batch_size, seq_len=1, hidden_dim]
                empty_emb = torch.zeros((batch_size, 1, CONFIG["time_dim"]), device=device)
                return self.model(x_t, t_float, encoder_hidden_states=empty_emb).sample
        else:
            return self.model(x_t, t_float).sample
        
    def _get_classifier_gradient(self, x_t, t, y, scale=1.0):
        """Get gradient from classifier for guidance.
        
        Args:
            x_t: Current noisy image
            t: Current timestep
            y: Target class labels
            scale: Gradient scale factor
            
        Returns:
            Tensor of gradients for guidance
        """
        if self.classifier is None:
            return torch.zeros_like(x_t)
        
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t.requires_grad_(True)
        
        logits = self.classifier(x_t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, y.view(-1, 1))
        
        grad = torch.autograd.grad(selected_log_probs.sum(), x_t)[0]
        grad = grad * scale * alpha_t.sqrt()
        
        x_t.requires_grad_(False)
        
        return grad
    
    def p_mean_variance(self, x_t, t, y=None, guidance_scale=3.0):
        """Compute parameters for p(x_{t-1} | x_t).
        
        Args:
            x_t: Current noisy image
            t: Current timestep
            y: Optional class labels
            guidance_scale: Scale for guidance strength
            
        Returns:
            Tuple of (mean, variance, predicted x_0, predicted noise)
        """
        if self.guidance_method == "classifier-free":
            cond_out = self._run_model(x_t, t, y)
            uncond_out = self._run_model(x_t, t, None)
            pred_noise = uncond_out + guidance_scale * (cond_out - uncond_out)
        else:  # classifier guidance
            pred_noise = self._run_model(x_t, t, None)
            grad = self._get_classifier_gradient(x_t, t, y, scale=guidance_scale)
            pred_noise = pred_noise - grad
        
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        
        pred_x0 = (x_t - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * pred_noise) / \
                 self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        model_mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t * pred_noise / torch.sqrt(1 - alpha_cumprod_t)))
        model_variance = torch.zeros_like(model_mean) if t[0] == 0 else self.posterior_variance[t].view(-1, 1, 1, 1)
        
        return model_mean, model_variance, pred_x0, pred_noise
    
    def p_sample(self, x_t, t, y=None, guidance_scale=3.0):
        """Sample from p(x_{t-1} | x_t).
        
        Args:
            x_t: Current noisy image
            t: Current timestep
            y: Optional class labels
            guidance_scale: Scale for guidance strength
            
        Returns:
            Tensor sampled from p(x_{t-1} | x_t)
        """
        model_mean, model_variance, _, _ = self.p_mean_variance(x_t, t, y, guidance_scale)
        if t[0] == 0:
            return model_mean
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(model_variance) * noise
    
    def p_sample_loop(self, shape, device, y=None, guidance_scale=3.0):
        """Sample a batch of images using the reverse diffusion process.
        
        Args:
            shape: Shape of the output tensor
            device: Device to run on
            y: Optional class labels
            guidance_scale: Scale for guidance strength
            
        Returns:
            Tensor of generated samples
        """
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(self.steps)), desc="Sampling"):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, y, guidance_scale)
        return img
    
    def train_step(self, x_0, t, y=None, p_uncond=0.1):
        """Compute the training loss for a single step.
        
        Args:
            x_0: Clean input image
            t: Timestep
            y: Optional class labels
            p_uncond: Probability of dropping conditioning
            
        Returns:
            Loss tensor
        """
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)
        
        use_y = y
        if y is not None and self.num_classes is not None and torch.rand(1).item() < p_uncond:
            use_y = None
        
        noise_pred = self._run_model(x_t, t, use_y)
        loss = F.mse_loss(noise_pred, noise)
        return loss

def generate_samples(diffusion, device, epoch=0, num_samples=4, classes=None, guidance_scale=3.0):
    """Generate samples from the diffusion model.
    
    Args:
        diffusion: Diffusion process
        device: Device to run on
        epoch: Current epoch (for naming saved files)
        num_samples: Number of samples to generate
        classes: Optional class indices for conditional generation
        guidance_scale: Scale for guidance strength
        
    Returns:
        Tensor of generated samples
    """
    diffusion.model.eval()
    
    y = None
    if classes is not None and diffusion.num_classes is not None:
        y = torch.tensor(classes, device=device)
    
    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            shape=(num_samples, CONFIG["in_channels"], CONFIG["image_size"], CONFIG["image_size"]),
            device=device, 
            y=y,
            guidance_scale=guidance_scale
        )
    
    samples = (samples.clamp(-1, 1) + 1) / 2
    diffusion.model.train()
    return samples

def create_model(device, use_conditional=CONFIG["use_conditional_unet"], num_classes=CONFIG["num_classes"]):
    """Create either conditional or unconditional UNet model.
    
    Args:
        device: Device to create model on
        use_conditional: Whether to use conditional UNet
        num_classes: Number of classes for conditional generation
        
    Returns:
        Tuple of (model, class_embedding)
    """
    channels = (
        CONFIG["base_channels"], 
        CONFIG["base_channels"] * 2, 
        CONFIG["base_channels"] * 4, 
        CONFIG["base_channels"] * 4
    )
    
    if use_conditional and num_classes is not None:
        model = UNet2DConditionModel(
            sample_size=CONFIG["image_size"],
            in_channels=CONFIG["in_channels"],
            out_channels=CONFIG["in_channels"],
            layers_per_block=2,
            block_out_channels=channels,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D"
            ),
            cross_attention_dim=CONFIG["time_dim"],
        ).to(device)
        
        class_embed = ClassEmbedding(
            num_classes=num_classes,
            embed_dim=CONFIG["time_dim"]
        ).to(device)
        
        return model, class_embed
    else:
        model = UNet2DModel(
            sample_size=CONFIG["image_size"],
            in_channels=CONFIG["in_channels"],
            out_channels=CONFIG["in_channels"],
            layers_per_block=2,
            block_out_channels=channels,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D", 
                "AttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", 
                "AttnUpBlock2D",
                "UpBlock2D", 
                "UpBlock2D"
            ),
        ).to(device)
        
        return model, None