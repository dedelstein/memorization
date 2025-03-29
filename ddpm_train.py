import os
import torch
from tqdm import tqdm

from config import CONFIG
from ddpm import Diffusion, create_model, generate_samples
from util import get_chexpert_dataloaders

class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model, decay=0.9999):
        """Initialize EMA.
        
        Args:
            model: PyTorch model
            decay: EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in model.parameters()]
    
    def update(self):
        """Update EMA parameters."""
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, self.model.parameters()):
                s_param.copy_(self.decay * s_param + (1 - self.decay) * param.detach())
    
    def copy_to(self, target_model):
        """Copy EMA parameters to target model.
        
        Args:
            target_model: PyTorch model to copy parameters to
        """
        with torch.no_grad():
            for s_param, param in zip(self.shadow_params, target_model.parameters()):
                param.copy_(s_param)

def train(model, diffusion, train_dataloader, val_dataloader, optimizer, device, num_epochs=100, ema_decay=CONFIG["ema_decay"], 
         amp_enabled=CONFIG["amp"], scheduler=None, save_dir="checkpoints", class_embed=None):
    """Train the diffusion model.
    
    Args:
        model: UNet model
        diffusion: Diffusion process
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to run on
        num_epochs: Number of training epochs
        ema_decay: EMA decay rate
        amp_enabled: Whether to use mixed precision training
        scheduler: Learning rate scheduler
        
    Returns:
        Trained model
    """
    model.train()
    ema = EMA(model, decay=ema_decay)
    scaler = torch.GradScaler() if amp_enabled else None

    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                t = torch.randint(0, diffusion.steps, (x.shape[0],), device=device)
                optimizer.zero_grad()
                
                if amp_enabled:
                    with torch.autocast(device_type=device.type):
                        loss = diffusion.train_step(x, t, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = diffusion.train_step(x, t, y)
                    loss.backward()
                    optimizer.step()
                
                ema.update()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        if scheduler is not None:
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        val_loss = validate(model, diffusion, val_dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
        }
        
        if class_embed is not None:
            checkpoint['class_embed_state_dict'] = class_embed.state_dict()
        
        torch.save(checkpoint, os.path.join(save_dir, f'model_epoch_{epoch+1}.pt'))
 
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
    
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
    }
    if class_embed is not None:
        final_checkpoint['class_embed_state_dict'] = class_embed.state_dict()
    
    torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pt'))
    
    return model

def validate(model, diffusion, dataloader, device):
    """Evaluate the diffusion model on validation data."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            t = torch.randint(0, diffusion.steps, (x.shape[0],), device=device)
            loss = diffusion.train_step(x, t, y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    """Main function to set up and train the diffusion model.
    
    Returns:
        Tuple of (model, diffusion, class_embed)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, class_embed = create_model(
        device=device,
        use_conditional=CONFIG["use_conditional_unet"],
        num_classes=CONFIG["num_classes"]
    )
    
    # Example of loading a pre-trained classifier
    # Replace this with your own classifier
    # classifier = YourClassifierModel().to(device)
    # classifier.load_state_dict(torch.load("path_to_classifier_weights.pth"))
    # classifier.eval()  # Set to evaluation mode
    # Replace with your classifier
    classifier = None
    
    diffusion = Diffusion(
        model=model,
        steps=CONFIG["diffusion_steps"],
        beta_schedule=CONFIG["beta_schedule"],
        num_classes=CONFIG["num_classes"],
        class_embed=class_embed,
        classifier=classifier,
        guidance_method=CONFIG["guidance_method"]
    )

    params_to_optimize = list(model.parameters())
    if class_embed is not None:
        params_to_optimize += list(class_embed.parameters())
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=1e-4,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=100
    )
    
    # Create a dataloader, this is a toy example
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Prepare dataloaders
    train_loader, val_loader = get_chexpert_dataloaders(CONFIG["data_dir"])

    model = train(
        model=model,
        diffusion=diffusion,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=100,
        scheduler=scheduler,
        save_dir="checkpoints",
        class_embed=class_embed
    )
    
    return model, diffusion, class_embed

if __name__ == "__main__":
    model, diffusion, class_embed = main()