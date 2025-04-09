"""
PyTorch Lightning DataModule for CheXpert dataset.
Modified to incorporate preprocessing from original implementation.
"""
<<<<<<< HEAD
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import logging
import pyarrow.parquet as pq  # Add import for Parquet handling
from datasets import load_dataset

=======
>>>>>>> ef7ab8a (new version)

import logging
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # Add import for Parquet handling
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CheXpertDataset(Dataset):
    """
    CheXpert dataset for multi-label classification of chest X-rays.
    Incorporates preprocessing from original implementation.
    """

    def __init__(
        self,
        parquet_file,
        base_dir,
        transform=None,
        use_frontal_only=True,
        debug_mode=False,
        policy="ones",
        class_index=None,
        use_metadata=True,
    ):
        """
        Args:
            parquet_file: Path to the Parquet file with annotations.
            base_dir: Base directory containing train/valid folders.
            transform: Optional transform to be applied on a sample.
            use_frontal_only: If True, only use frontal views.
            debug_mode: If True, only use a small subset of data for debugging.
            policy: How to handle uncertain labels: 'ones', 'zeros', 'ignore'.
            class_index: Which classes to use (defaults to all 14).
            use_metadata: Whether to include metadata in the labels.
        """
        self.parquet_file = parquet_file  # Store the Parquet file path
        self.base_dir = base_dir
        self.transform = transform
        self.policy = policy
        self.use_metadata = use_metadata

        # Define findings (labels) list
        self.findings = [
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

        # Define metadata fields (only if use_metadata is True)
        self.metadata = (
            ["Sex", "Age", "Frontal/Lateral", "AP/PA"] if use_metadata else []
        )

        # Load the entire Parquet file into memory as a Pandas DataFrame
        self.data_frame = pd.read_parquet(parquet_file)  # Load all columns
        self.num_samples = len(self.data_frame)

        # Filter for frontal views if requested
        if use_frontal_only:
            self.data_frame = self.data_frame[self.data_frame["Frontal/Lateral"] == 1]
            self.num_samples = len(self.data_frame)

        # Allow filtering for specific classes
        if class_index is None:
            self.class_index = list(range(len(self.findings)))
        else:
            self.class_index = class_index

        self.classes = [self.findings[i] for i in self.class_index]

        # Debug mode: Use a small subset of data
        if debug_mode:
            logger.info("Debug mode enabled - using only 100 samples")
            self.data_frame = self.data_frame.iloc[:100]
            self.num_samples = len(self.data_frame)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Access the row directly from the preloaded DataFrame
        row = self.data_frame.iloc[idx]

        # Extract the image data directly from the "image" column
        image_data = row["image"]  # Assuming the "image" column contains raw image data

        # Convert the image data to a PIL Image
        try:
            image = Image.fromarray(np.array(image_data, dtype=np.uint8)).convert("L")
        except Exception as e:
            logger.error(f"Error processing image data at index {idx}: {e}")
            image = Image.new("L", (224, 224), color=128)  # Fallback to a blank image

        # Transform the image if needed
        if self.transform:
            image = self.transform(image)

        # Extract labels
        labels = torch.tensor(row[self.classes].values.astype(np.float32))

        # Add metadata to labels only if use_metadata is True
        if self.use_metadata:
            metadata = torch.tensor(row[self.metadata].values.astype(np.float32))
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
        policy="ones",
        class_index=None,
        use_metadata=True,
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
        self.use_metadata = use_metadata

        # Define transforms using the normalization parameters from the first implementation
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5], std=[0.3]
                ),  # Original implementation values
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5], std=[0.3]
                ),  # Original implementation values
            ]
        )

    def prepare_data(self):
        """Check if data directory exists."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")

        # Check for train.parquet and valid.parquet
        train_parquet = os.path.join(
            self.data_dir, "train.parquet"
        )  # Changed to Parquet
        valid_parquet = os.path.join(
            self.data_dir, "valid.parquet"
        )  # Changed to Parquet

        if not os.path.exists(train_parquet):
            raise FileNotFoundError(f"Train Parquet file not found: {train_parquet}")
        if not os.path.exists(valid_parquet):
            raise FileNotFoundError(f"Valid Parquet file not found: {valid_parquet}")

        # Log the data directory
        logger.info(f"Using data directory: {self.data_dir}")
        logger.info(f"Train Parquet exists: {os.path.exists(train_parquet)}")
        logger.info(f"Valid Parquet exists: {os.path.exists(valid_parquet)}")

    def setup(self, stage=None):
        """Set up the dataset splits."""
        # Find the Parquet files
        train_parquet = os.path.join(
            self.data_dir, "train.parquet"
        )  # Changed to Parquet
        valid_parquet = os.path.join(
            self.data_dir, "valid.parquet"
        )  # Changed to Parquet

        logger.info(f"Using train Parquet: {train_parquet}")
        logger.info(f"Using valid Parquet: {valid_parquet}")

        if stage == "fit" or stage is None:
            logger.info("Setting up training dataset...")
            # Set up training dataset with our updated CheXpertDataset class
            self.train_dataset = CheXpertDataset(
                parquet_file=train_parquet,  # Changed to Parquet
                base_dir=self.data_dir,
                transform=self.train_transform,
                use_frontal_only=self.use_frontal_only,
                debug_mode=self.debug_mode,
                policy=self.policy,
                class_index=self.class_index,
                use_metadata=self.use_metadata,
            )

            logger.info("Setting up validation dataset...")
            # Set up validation dataset
            self.val_dataset = CheXpertDataset(
                parquet_file=valid_parquet,  # Changed to Parquet
                base_dir=self.data_dir,
                transform=self.val_transform,
                use_frontal_only=self.use_frontal_only,
                debug_mode=self.debug_mode,
                policy=self.policy,
                class_index=self.class_index,
                use_metadata=self.use_metadata,
            )

            # Split validation into val and test
            logger.info("Splitting validation dataset into val and test sets...")
            val_size = int(0.9 * len(self.val_dataset))
            test_size = len(self.val_dataset) - val_size

            self.val_dataset, self.test_dataset = random_split(
                self.val_dataset,
                [val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")
            logger.info(f"Test dataset size: {len(self.test_dataset)}")

        if stage == "test" or stage is None:
            # If we're just testing and haven't set up datasets yet
            if not hasattr(self, "test_dataset"):
                logger.info("Setting up test dataset...")
                val_dataset = CheXpertDataset(
                    parquet_file=valid_parquet,  # Changed to Parquet
                    base_dir=self.data_dir,
                    transform=self.val_transform,
                    use_frontal_only=self.use_frontal_only,
                    debug_mode=self.debug_mode,
                    policy=self.policy,
                    class_index=self.class_index,
                    use_metadata=self.use_metadata,
                )

                # Split validation into val and test
                val_size = int(0.9 * len(val_dataset))
                test_size = len(val_dataset) - val_size

                _, self.test_dataset = random_split(
                    val_dataset,
                    [val_size, test_size],
                    generator=torch.Generator().manual_seed(self.seed),
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
            persistent_workers=True,  # Enable persistent workers for efficiency
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
            persistent_workers=True,  # Enable persistent workers for efficiency
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
            persistent_workers=True,  # Enable persistent workers for efficiency
        )


# Add this to test DataLoader efficiency
if __name__ == "__main__":
    data_dir = "/work3/s243891/CheXpert-v1.0-small"
    from time import time

    data_module = CheXpertDataModule(data_dir=data_dir, batch_size=16, num_workers=4)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    train_loader = data_module.train_dataloader()

    start_time = time()
    for i, (images, labels) in enumerate(train_loader):
        if i == 10:  # Test the first 10 batches
            break
    end_time = time()

    print(f"Time to load 10 batches: {end_time - start_time:.2f} seconds")

    # df = pd.read_parquet("/work3/s243891/CheXpert-v1.0-small/valid.parquet")
    # print(df.head())
    # print(df.columns)
