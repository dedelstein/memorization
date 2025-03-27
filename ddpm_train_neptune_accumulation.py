import torch
import os
import neptune
from tqdm import tqdm
from dotenv import load_dotenv
from accelerate import Accelerator

from config import CONFIG
from ddpm import Diffusion, create_model, generate_samples
from util import prepare_dataloaders

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
         amp_enabled=CONFIG["amp"], scheduler=None, save_dir="checkpoints", class_embed=None, run=None, gradient_accumulation_steps=4):
    """Train the diffusion model."""
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    if class_embed is not None:
        class_embed = accelerator.prepare(class_embed)
    
    ema = EMA(model, decay=ema_decay)
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                with accelerator.accumulate(model):
                    x, y = batch
                    
                    t = torch.randint(0, diffusion.steps, (x.shape[0],), device=accelerator.device)
                    loss = diffusion.train_step(x, t, y)
                    
                    accelerator.backward(loss)
                    
                    if ((batch_idx + 1) % gradient_accumulation_steps == 0 or 
                        batch_idx == len(train_dataloader) - 1):
                        optimizer.step()
                        optimizer.zero_grad()
                        ema.update()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                    
                    if run is not None:
                        run["train/batch_loss"].log(loss.item(), step=epoch * len(train_dataloader) + batch_idx)
        
        if scheduler is not None:
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        val_loss = validate(model, diffusion, val_dataloader, accelerator.device)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if run is not None:
            run["train/epoch"].log(epoch + 1)
            run["train/avg_loss"].log(avg_train_loss)
            run["validation/loss"].log(val_loss)
            run["train/learning_rate"].log(optimizer.param_groups[0]['lr'])
        
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
        }
        
        if class_embed is not None:
            unwrapped_class_embed = accelerator.unwrap_model(class_embed)
            checkpoint['class_embed_state_dict'] = unwrapped_class_embed.state_dict()
        
        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            
            if run is not None:
                run["validation/best_loss"] = best_loss
                run["models/best_model"].upload(best_model_path)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            samples = generate_samples(diffusion, accelerator.device, epoch+1)
            
            if run is not None:
                for i, sample in enumerate(samples):
                    sample_path = os.path.join(save_dir, f'sample_{epoch+1}_{i}.png')
                    sample_img = sample.permute(1, 2, 0).cpu().numpy()
                    import matplotlib.pyplot as plt
                    plt.imsave(sample_path, sample_img)
                    run[f"samples/epoch_{epoch+1}"].upload(sample_path)
    
    unwrapped_model = accelerator.unwrap_model(model)
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
    }
    if class_embed is not None:
        unwrapped_class_embed = accelerator.unwrap_model(class_embed)
        final_checkpoint['class_embed_state_dict'] = unwrapped_class_embed.state_dict()
    
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save(final_checkpoint, final_model_path)
    
    if run is not None:
        run["models/final_model"].upload(final_model_path)
    
    return accelerator.unwrap_model(model)

def validate(model, diffusion, dataloader, device):
    """Evaluate the diffusion model on validation data."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            
            t = torch.randint(0, diffusion.steps, (x.shape[0],), device=device)
            loss = diffusion.train_step(x, t, y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    """Main function to set up and train the diffusion model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dotenv()

    NEPTUNE_API_TOKEN = os.getenv('NEPTUNE_API_TOKEN')
    
    run = neptune.init_run(
        project="dedelstein/Memorization",
        api_token=NEPTUNE_API_TOKEN,
        tags=["diffusion-model"],
        capture_hardware_metrics=True,
    )
    
    for key, value in CONFIG.items():
        run["parameters/config"][key] = value
    
    run["parameters/optimizer"] = "AdamW"
    run["parameters/lr"] = 1e-4
    run["parameters/weight_decay"] = 1e-5
    run["parameters/scheduler"] = "CosineAnnealingLR"
    run["parameters/gradient_accumulation_steps"] = 4
    
    model, class_embed = create_model(
        device=device,
        use_conditional=CONFIG["use_conditional_unet"],
        num_classes=CONFIG["num_classes"]
    )
    
    # Example of loading a pre-trained classifier
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
    
    # Create a dataloader - this is a toy example
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
    # train_loader, val_loader, _ = prepare_dataloaders(batch_size=64, val_batch_size=32)

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
        class_embed=class_embed,
        run=run,
        gradient_accumulation_steps=4
    )
    
    run.stop()
    
    return model, diffusion, class_embed

if __name__ == "__main__":
    model, diffusion, class_embed = main()