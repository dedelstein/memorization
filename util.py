import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SEED = 1

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

class CheXpertDataset(Dataset):
    """
    CheXpert dataset class for PyTorch
    """
    def __init__(self, csv_file, root_dir, transform=None, policy='ones', 
                 class_index=None, use_frontal=True, use_metadata=True):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Root directory containing the images.
            transform (callable, optional): Optional transform to be applied to samples.
            policy (string): How to handle uncertain labels: 'ones', 'zeros', 'ignore'.
            class_index (list, optional): Which classes to use (defaults to all 14).
            use_frontal (bool): Whether to use only frontal X-rays.
        """
        self.metadata = use_metadata

        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame.drop("Path", axis=1)

        # -------------------
        # Filter and clean df

        # Age - Grouped, in range (0-4)
        self.data_frame = self.data_frame[self.data_frame["Age"] > 1]
        age_bins = [18, 44, 64, 79, np.inf]
        self.data_frame["Age"] = np.digitize(self.data_frame["Age"], bins=age_bins, right=True).astype(np.int64)

        # Sex (0 - Female, 1 - Male)
        self.data_frame["Sex"] = self.data_frame["Sex"].apply(lambda x: 1 if x == "Male" else 0)

        # Frontal, AP/PA - AP/PA in (0 - None, 1 - AP, 2 - PA)
        self.data_frame["AP/PA"] = self.data_frame["AP/PA"].apply(lambda x: 1 if x == "AP" else 2 if x == "PA" else 0)

        if use_frontal:
            self.data_frame = self.data_frame[self.data_frame["Frontal/Lateral"] == "Frontal"]
        else:
            self.data_frame["Frontal/Lateral"] = self.data_frame["Frontal/Lateral"].apply(lambda x: 1 if x == "Frontal" else 0)

        # Replace all NaN values in findings with 0 - we have 0 as negative here
        # Not sure if this is ideal, some discussion in the slack
        self.data_frame= self.data_frame.fillna(0)
        # -------------------

        self.root_dir = root_dir
        self.transform = transform
        self.policy = policy
        
        self.findings = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", 
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", 
            "Pleural Other", "Fracture", "Support Devices"
        ]

        self.metadata = [
            "Sex", "Age", "Frontal/Lateral", "AP/PA"
        ]

        if class_index is None:
            self.class_index = list(range(len(self.findings)))
        else:
            self.class_index = class_index
            
        self.classes = [self.findings[i] for i in self.class_index]
        
        self._apply_policy()
        
    def _apply_policy(self):
        """Apply the selected policy for uncertain labels (-1)"""
        if self.policy == 'ones':
            self.data_frame.loc[:, self.classes] = self.data_frame.loc[:, self.classes].replace(-1, 1)
        elif self.policy == 'zeros':
            self.data_frame.loc[:, self.classes] = self.data_frame.loc[:, self.classes].replace(-1, 0)
        elif self.policy == 'ignore':
            pass
        else:
            raise ValueError(f"Unknown policy: {self.policy}")
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.root_dir, self.data_frame.loc[idx, "Path"])
        
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            placeholder = np.zeros((224, 224), dtype=np.uint8)
            image = Image.fromarray(placeholder)
            
        if self.transform:
            image = self.transform(image)

        # Feature only labels    
        labels = torch.tensor(
            self.data_frame.loc[idx, self.classes].values.astype(np.float32)
        )

        # If you want Age, Sex, F/L, AP/PA
        if self.metadata:
            metadata =  torch.tensor(
                self.data_frame.loc[idx, self.metadata].values.astype(np.float32)
            )
            labels = torch.cat([labels, metadata], dim=0)

        return image, labels


def get_chexpert_dataloaders(root_dir, batch_size=32, policy='ones', class_index=None,
                             use_frontal=True, num_workers=4, use_metadata=True):
    """
    Create CheXpert dataloaders for training, validation and test using predefined splits
    
    Args:
        root_dir (string): Root directory containing the dataset.
        batch_size (int): Batch size for the dataloaders.
        policy (string): How to handle uncertain labels: 'ones', 'zeros', 'ignore'.
        class_index (list, optional): Which classes to use (defaults to all 14).
        use_frontal (bool): Whether to use only frontal X-rays.
        num_workers (int): Number of workers for data loading.
        include_test (bool): Whether to include test set loader.
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation and test.
        Note: test_loader will be None if include_test is False or test.csv doesn't exist.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
    #   transforms.RandomHorizontalFlip(),
    #   transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_csv = os.path.join(root_dir, 'train.csv')
    valid_csv = os.path.join(root_dir, 'valid.csv')
    
    train_dataset = CheXpertDataset(
        csv_file=train_csv,
        root_dir=root_dir,
        transform=train_transform,
        policy=policy,
        class_index=class_index,
        use_frontal=use_frontal,
        use_metadata=use_metadata
    )
    
    val_dataset = CheXpertDataset(
        csv_file=valid_csv,
        root_dir=root_dir,
        transform=val_transform,
        policy=policy,
        class_index=class_index,
        use_frontal=use_frontal,
        use_metadata=use_metadata
    )
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def show(imgs, title=None, fig_titles=None, save_path=None): 

    if fig_titles is not None:
        assert len(imgs) == len(fig_titles)

    fig, axs = plt.subplots(1, ncols=len(imgs), figsize=(15, 5))
    for i, img in enumerate(imgs):
        axs[i].imshow(img)
        axs[i].axis('off')
        if fig_titles is not None:
            axs[i].set_title(fig_titles[i], fontweight='bold')

    if title is not None:
        plt.suptitle(title, fontweight='bold')
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()
