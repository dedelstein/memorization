"""
PyTorch Lightning DataModule for CheXpert dataset.
Modified to incorporate preprocessing from original implementation.
"""
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheXpertDataset(Dataset):
    """
    CheXpert dataset for multi-label classification of chest X-rays.
    Incorporates preprocessing from original implementation.
    """
    def __init__(self, 
                 csv_file,
                 base_dir,
                 transform=None,
                 use_frontal_only=True,
                 debug_mode=False,
                 policy='ones',
                 class_index=None,
                 use_metadata=True):
        """
        Args:
            csv_file: Path to the csv file with annotations.
            base_dir: Base directory containing train/valid folders.
            transform: Optional transform to be applied on a sample.
            use_frontal_only: If True, only use frontal views.
            debug_mode: If True, only use a small subset of data for debugging.
            policy: How to handle uncertain labels: 'ones', 'zeros', 'ignore'.
            class_index: Which classes to use (defaults to all 14).
            use_metadata: Whether to include metadata in the labels.
        """
        # Read the CSV file
        self.data_frame = pd.read_csv(csv_file)
        self.use_metadata = use_metadata
        
        # -------------------
        # Filter and clean df - Using preprocessing from original implementation
        
        # Age - Grouped, in range (0-4)
        self.data_frame = self.data_frame[self.data_frame["Age"] > 1]
        age_bins = [18, 44, 64, 79, np.inf]
        self.data_frame["Age"] = np.digitize(self.data_frame["Age"], bins=age_bins, right=True).astype(np.int64)

        # Sex (0 - Female, 1 - Male)
        self.data_frame["Sex"] = self.data_frame["Sex"].apply(lambda x: 1 if x == "Male" else 0)

        # Frontal, AP/PA - AP/PA in (0 - None, 1 - AP, 2 - PA)
        self.data_frame["AP/PA"] = self.data_frame["AP/PA"].apply(lambda x: 1 if x == "AP" else 2 if x == "PA" else 0)
        self.data_frame["Frontal/Lateral"] = self.data_frame["Frontal/Lateral"].apply(lambda x: 1 if x == "Frontal" else 0)

        # Apply frontal filtering if requested
        if use_frontal_only:
            logger.info("Filtering for frontal views only")
            self.data_frame = self.data_frame[self.data_frame["Frontal/Lateral"] == 1]

        # Replace all NaN values in findings with 0
        self.data_frame = self.data_frame.fillna(0)
        self.data_frame = self.data_frame.reset_index(drop=True)
        # -------------------
        
        # Use a small subset for debugging if requested
        if debug_mode:
            logger.info("Debug mode enabled - using only 5000 samples")
            self.data_frame = self.data_frame.iloc[:5000]
        
        # Store directory and transform
        self.base_dir = base_dir
        self.transform = transform
        self.policy = policy
        
        # Define findings (labels) list
        self.findings = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", 
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", 
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", 
            "Pleural Other", "Fracture", "Support Devices"
        ]
        
        # Define metadata fields
        self.metadata = [
            "Sex", "Age", "Frontal/Lateral", "AP/PA"
        ]
        
        # Allow filtering for specific classes
        if class_index is None:
            self.class_index = list(range(len(self.findings)))
        else:
            self.class_index = class_index
            
        self.classes = [self.findings[i] for i in self.class_index]
        
        # Apply the policy for uncertain labels
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
        
        # Get the image path from the Path column
        path_in_csv = self.data_frame.iloc[idx]["Path"]
        
        # Handle the "CheXpert-v1.0-small" prefix in the CSV
        if path_in_csv.startswith("CheXpert-v1.0-small/"):
            # Remove the prefix
            path_in_csv = path_in_csv[len("CheXpert-v1.0-small/"):]
        
        # Construct the full path
        img_path = os.path.join(self.base_dir, path_in_csv)
        
        # Log the path being used (only for the first few images)
        if idx < 3:
            logger.info(f"Loading image at path: {img_path}")
        
        # Load the image as grayscale (L mode) as in the first implementation
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Create a gray image as fallback
            image = Image.new('L', (224, 224), color=128)
        
        # Transform the image if needed
        if self.transform:
            image = self.transform(image)
        
        # Feature only labels using the class index
        labels = torch.tensor(
            self.data_frame.loc[idx, self.classes].values.astype(np.float32)
        )

        # If you want Age, Sex, F/L, AP/PA
        if self.use_metadata:
            metadata = torch.tensor(
                self.data_frame.loc[idx, self.metadata].values.astype(np.float32)
            )
            labels = torch.cat([labels, metadata], dim=0)
        
        return image, labels


class CheXpertDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the CheXpert dataset.
    """
    def __init__(
        self,
        data_dir,
        img_size=224,
        batch_size=16,
        num_workers=4,
        use_frontal_only=True,
        seed=42,
        debug_mode=False,
        pin_memory=True,
        policy='ones',
        class_index=None,
        use_metadata=True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_frontal_only = use_frontal_only
        self.seed = seed
        self.debug_mode = debug_mode
        self.pin_memory = pin_memory
        self.policy = policy
        self.class_index = class_index
        self.use_metadata = False # use_metadata
        
        # Define transforms using the normalization parameters from the first implementation
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.3])  # Original implementation values
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.3])  # Original implementation values
        ])
    
    def prepare_data(self):
        """Check if data directory exists."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Check for train.csv and valid.csv
        train_csv = os.path.join(self.data_dir, 'train.csv')
        valid_csv = os.path.join(self.data_dir, 'valid.csv')
        
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Train CSV file not found: {train_csv}")
        if not os.path.exists(valid_csv):
            raise FileNotFoundError(f"Valid CSV file not found: {valid_csv}")
        
        # Log the data directory
        logger.info(f"Using data directory: {self.data_dir}")
        logger.info(f"Train CSV exists: {os.path.exists(train_csv)}")
        logger.info(f"Valid CSV exists: {os.path.exists(valid_csv)}")
    
    def setup(self, stage=None):
        """Set up the dataset splits."""
        # Find the CSV files
        train_csv = os.path.join(self.data_dir, 'train.csv')
        valid_csv = os.path.join(self.data_dir, 'valid.csv')
        
        logger.info(f"Using train CSV: {train_csv}")
        logger.info(f"Using valid CSV: {valid_csv}")
        
        if stage == 'fit' or stage is None:
            logger.info("Setting up training dataset...")
            # Set up training dataset with our updated CheXpertDataset class
            self.train_dataset = CheXpertDataset(
                csv_file=train_csv,
                base_dir=self.data_dir,
                transform=self.train_transform,
                use_frontal_only=self.use_frontal_only,
                debug_mode=self.debug_mode,
                policy=self.policy,
                class_index=self.class_index,
                use_metadata=self.use_metadata
            )
            
            logger.info("Setting up validation dataset...")
            # Set up validation dataset
            self.val_dataset = CheXpertDataset(
                csv_file=valid_csv,
                base_dir=self.data_dir,
                transform=self.val_transform,
                use_frontal_only=self.use_frontal_only,
                debug_mode=self.debug_mode,
                policy=self.policy,
                class_index=self.class_index,
                use_metadata=self.use_metadata
            )
            
            # Split validation into val and test
            logger.info("Splitting validation dataset into val and test sets...")
            val_size = int(0.9 * len(self.val_dataset))
            test_size = len(self.val_dataset) - val_size
            
            self.val_dataset, self.test_dataset = random_split(
                self.val_dataset, 
                [val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed)
            )
            
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")
            logger.info(f"Test dataset size: {len(self.test_dataset)}")
        
        if stage == 'test' or stage is None:
            # If we're just testing and haven't set up datasets yet
            if not hasattr(self, 'test_dataset'):
                logger.info("Setting up test dataset...")
                val_dataset = CheXpertDataset(
                    csv_file=valid_csv,
                    base_dir=self.data_dir,
                    transform=self.val_transform,
                    use_frontal_only=self.use_frontal_only,
                    debug_mode=self.debug_mode,
                    policy=self.policy,
                    class_index=self.class_index,
                    use_metadata=self.use_metadata
                )
                
                # Split validation into val and test
                val_size = int(0.9 * len(val_dataset))
                test_size = len(val_dataset) - val_size
                
                _, self.test_dataset = random_split(
                    val_dataset, 
                    [val_size, test_size],
                    generator=torch.Generator().manual_seed(self.seed)
                )
                
                logger.info(f"Test dataset size: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )