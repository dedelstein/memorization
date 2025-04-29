"""
PyTorch Lightning DataModule for CheXpert dataset.
"""

import os

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.io as io
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from src.utils.constants import CHEXPERT_CLASSES


class CheXpertDataset(Dataset):
    """
    CheXpert dataset for multi-label classification of chest X-rays.
    """

    def __init__(
        self,
        csv_file: str,
        base_dir: str,
        transform: transforms.Compose = None,
        debug_mode: bool = False,
        class_index: list = None,
        img_size=224,
    ):
        """
        Args:
            csv_file: Path to the csv file with annotations.
            base_dir: Base directory containing images.
            transform: Optional transform to be applied on a sample.
            debug_mode: If True, only use a small subset of data for debugging.
            class_index: Which classes to use (defaults to all 14).
            img_size: Size for image resizing.
        """
        # Configure class indices
        if class_index is None:
            self.class_index = list(range(len(CHEXPERT_CLASSES)))
        else:
            self.class_index = class_index

        self.classes = [CHEXPERT_CLASSES[i] for i in self.class_index]

        # Load and filter data
        df = pd.read_csv(csv_file)

        # Preprocess dataset
        self.data_frame = self._preprocess_dataset(
            df,
            debug_mode=debug_mode,
        )

        # Set up class variables
        self.base_dir = base_dir[: -len("CheXpert-v1.0-small")]
        self.transform = transform
        self.img_size = img_size

    def _preprocess_dataset(
        self, df, filter_frontal_ap=True, fill_na=True, debug_mode=False
    ):
        """
        Preprocess the CheXpert dataset.

        Args:
            df: Input DataFrame
            filter_frontal_ap: Whether to filter for frontal AP images
            fill_na: Whether to fill NA values with 0
            debug_mode: Whether to use a small subset for debugging

        Returns:
            Processed DataFrame
        """
        if filter_frontal_ap:
            # Filter for frontal images with AP projection
            df = df[(df["Frontal/Lateral"] == "Frontal") & (df["AP/PA"] == "AP")]

        if fill_na:
            # Fill NA values and convert -1.0 to 0.0
            df = df.fillna(0.0).replace(-1.0, 0.0)

        # Select only necessary columns
        filtered_df = df[["Path"] + self.classes]

        # Debug mode to use a small subset
        if debug_mode:
            filtered_df = filtered_df.iloc[: min(1000, len(filtered_df))]

        # Return the processed dataframe
        return filtered_df

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset, optimized for performance.
        """
        # Get image path
        img_path = os.path.join(self.base_dir, self.data_frame.iloc[idx]["Path"])

        try:
            # Read the image using torchvision
            image = io.read_image(img_path, mode=io.ImageReadMode.GRAY)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            # Return a default black image if reading fails
            image = torch.zeros(1, self.img_size, self.img_size)

        # Convert to float and normalize
        image = image.float() / 255.0

        # Resize if needed
        if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0

        # Apply transforms if available
        if self.transform:
            image = self.transform(image)

        # Get labels as tensor from npy array
        labels = torch.tensor(
            self.data_frame.iloc[idx][self.classes].values.astype(float),
            dtype=torch.float32,
        )

        return {"image": image, "labels": labels}  # Return as dict for compatibility


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
        seed=42,
        debug_mode=False,
        pin_memory=True,
        class_index=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.debug_mode = debug_mode
        self.pin_memory = pin_memory
        self.class_index = class_index

        # No transform as preprocessing happens in the dataset class
        self.transform = None

    def prepare_data(self):
        """Check if data directory and required files exist."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")

        train_csv = os.path.join(self.data_dir, "train.csv")
        valid_csv = os.path.join(self.data_dir, "valid.csv")

        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Train CSV file not found: {train_csv}")
        if not os.path.exists(valid_csv):
            raise FileNotFoundError(f"Valid CSV file not found: {valid_csv}")

    def setup(self, stage=None):
        """Set up train, validation, and test datasets."""
        train_csv = os.path.join(self.data_dir, "train.csv")
        valid_csv = os.path.join(self.data_dir, "valid.csv")

        if stage == "fit" or stage is None:
            self.train_dataset = CheXpertDataset(
                csv_file=train_csv,
                base_dir=self.data_dir,
                transform=self.transform,
                debug_mode=self.debug_mode,
                class_index=self.class_index,
                img_size=self.img_size,
            )

            val_dataset = CheXpertDataset(
                csv_file=valid_csv,
                base_dir=self.data_dir,
                transform=self.transform,
                debug_mode=self.debug_mode,
                class_index=self.class_index,
                img_size=self.img_size,
            )

            # Split validation into val and test
            val_size = int(0.9 * len(val_dataset))
            test_size = len(val_dataset) - val_size

            self.val_dataset, self.test_dataset = random_split(
                val_dataset,
                [val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
