"""
Simplified classifier model for guiding the diffusion process.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC

from src.utils.constants import CHEXPERT_CLASSES


class Classifier(nn.Module):
    """
    Classifier model for chest X-ray pathology prediction.
    Includes time embedding to be compatible with the diffusion process.
    """
    def __init__(
        self, 
        img_size=64, 
        c_in=1,  # Changed from 3 to 1
        num_labels=len(CHEXPERT_CLASSES), 
        time_dim=256, 
        channels=64
    ):
        """
        Initialize classifier model.
        
        Args:
            img_size: Size of the input image (assumed square)
            c_in: Number of input channels
            num_labels: Number of output labels for classification
            time_dim: Dimension of time embedding
            channels: Base channel count
        """
        super().__init__()
        self.time_dim = time_dim
        
        # Create ResNet-style backbone
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling layers
        self.layer1 = self._make_layer(channels, channels, stride=1)
        self.layer2 = self._make_layer(channels, channels*2, stride=2)
        self.layer3 = self._make_layer(channels*2, channels*4, stride=2)
        self.layer4 = self._make_layer(channels*4, channels*8, stride=2)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Time embedding projection for each layer
        self.time_emb1 = nn.Linear(time_dim, channels)
        self.time_emb2 = nn.Linear(time_dim, channels*2)
        self.time_emb3 = nn.Linear(time_dim, channels*4)
        self.time_emb4 = nn.Linear(time_dim, channels*8)
        
        # Global average pooling and final classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        fc_dim = channels*8
        
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, fc_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_dim // 2, num_labels)
        )

    def _make_layer(self, in_channels, out_channels, stride):
        """Create a ResNet-style layer with residual connections."""
        layers = []
        # Downsample if needed
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            downsample = None
            
        # Add ResNet block
        layers.append(ResNetBlock(in_channels, out_channels, stride, downsample))
        layers.append(ResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x, t):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            t: Time steps [batch_size]
            
        Returns:
            Classification logits [batch_size, num_labels]
        """
        # Time embedding
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Initial features
        x = self.conv1(x)
        
        # Down 1
        x = self.layer1(x)
        x = x + self.time_emb1(t_emb)[:, :, None, None]
        
        # Down 2
        x = self.layer2(x)
        x = x + self.time_emb2(t_emb)[:, :, None, None]
        
        # Down 3
        x = self.layer3(x)
        x = x + self.time_emb3(t_emb)[:, :, None, None]
        
        # Down 4
        x = self.layer4(x)
        x = x + self.time_emb4(t_emb)[:, :, None, None]
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ResNetBlock(nn.Module):
    """Basic ResNet block with time embedding."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


def sinusoidal_embedding(n, d):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        n: Time step (or batch of time steps)
        d: Embedding dimension
        
    Returns:
        Embedding vectors with shape [n, d]
    """
    # Ensure n is a batch of time steps
    if not isinstance(n, torch.Tensor):
        n = torch.tensor(n)
    n = n.unsqueeze(-1)
    
    # Calculate frequencies
    half_d = d // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_d - 1)
    emb = torch.exp(torch.arange(half_d, device=n.device) * -emb)
    
    # Calculate embeddings
    emb = n * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    # Pad if dimension is odd
    if d % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    
    return emb


class ClassifierModule(pl.LightningModule):
    """
    PyTorch Lightning module for training the classifier.
    """
    def __init__(
        self,
        img_size=64,
        c_in=3,
        num_labels=len(CHEXPERT_CLASSES),
        lr=1e-4,
        channels=64,
        time_dim=256
    ):
        """
        Initialize classifier module.
        
        Args:
            img_size: Size of the input image
            c_in: Number of input channels
            num_labels: Number of output labels
            lr: Learning rate
            channels: Base channel count
            time_dim: Dimension of time embedding
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.classifier = Classifier(
            img_size=img_size,
            c_in=c_in,
            num_labels=num_labels,
            time_dim=time_dim,
            channels=channels
        )
        
        # BCEWithLogitsLoss for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Maximum diffusion steps
        self.max_timestep = 1000
        
        # Metrics
        self.train_f1 = MultilabelF1Score(num_labels=num_labels)
        self.val_f1 = MultilabelF1Score(num_labels=num_labels)
        self.val_auroc = MultilabelAUROC(num_labels=num_labels)

    def forward(self, x, t):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            t: Time steps
            
        Returns:
            Classification logits
        """
        return self.classifier(x, t)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        
        # Sample random timesteps
        t = torch.randint(0, self.max_timestep, (x.shape[0],), device=self.device)
        
        # Forward pass
        logits = self(x, t)
        
        # Calculate loss
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.sigmoid(logits)
        f1 = self.train_f1(preds, y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1', f1, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Batch index
        """
        x, y = batch
        
        # Sample random timesteps
        t = torch.randint(0, self.max_timestep, (x.shape[0],), device=self.device)
        
        # Forward pass
        logits = self(x, t)
        
        # Calculate loss
        loss = self.criterion(logits, y)
        
        # Calculate predictions
        preds = torch.sigmoid(logits)
        
        # Update metrics
        self.val_f1.update(preds, y)
        self.val_auroc.update(preds, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        
        return loss

    def on_validation_epoch_end(self):
        """
        Process validation epoch results.
        """
        # Compute metrics
        val_f1 = self.val_f1.compute()
        val_auroc = self.val_auroc.compute()
        
        # Log metrics
        self.log('val_f1', val_f1)
        self.log('val_auroc', val_auroc)
        
        # Reset metrics
        self.val_f1.reset()
        self.val_auroc.reset()

    def configure_optimizers(self):
        """
        Configure optimizers.
        
        Returns:
            Optimizer
        """
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)