"""
PyTorch Lightning DataModule for CheXpert dataset.
"""

import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.io as io
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


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

        # Load and filter data
        df = pd.read_csv(csv_file)

        # Filter for frontal images with AP projection
        filtered_df = df[(df["Frontal/Lateral"] == "Frontal") & (df["AP/PA"] == "AP")]

        # Fill NA values and convert -1.0 to 0.0
        filtered_df = filtered_df.fillna(0.0).replace(-1.0, 0.0)

        # Select only necessary columns
        filtered_df = filtered_df[["Path"] + self.findings]

        self.data_frame = filtered_df

        # Debug mode to use a small subset
        if debug_mode:
            self.data_frame = self.data_frame.iloc[: min(10000, len(self.data_frame))]

        # Balance the dataset
        self.data_frame = self._balance_dataset(
            self.data_frame,
            min_ratio=0.05,
            max_samples_per_class=1000,
        )

        # Set up class variables
        self.base_dir = base_dir
        self.transform = transform
        self.img_size = img_size

        # Configure class indices (class indices are
        if class_index is None:
            self.class_index = list(range(len(self.findings)))
        else:
            self.class_index = class_index

        self.classes = [self.findings[i] for i in self.class_index]

    def _balance_dataset(self, df, min_ratio=0.05, max_samples_per_class=None):
        """
        Balances the dataset to improve the representation of minority classes.

        Args:
            df: Original DataFrame
            min_ratio: Minimum desired ratio for each class
            max_samples_per_class: Maximum number of samples per class

        Returns:
            Balanced DataFrame
        """
        # Initialize a set for selected indices
        selected_indices = set()

        # For each medical finding, select samples in a balanced manner
        for finding in self.findings:
            positive_samples = df[df[finding] == 1]
            negative_samples = df[df[finding] == 0]

            # Determine the number of samples to select
            positive_count = len(positive_samples)

            # If there are very few positive samples, take all of them
            if positive_count < 100:
                sample_size = positive_count
            else:
                # Define a sample size based on the minimum desired ratio
                # and limited by max_samples_per_class if specified
                sample_size = min(
                    positive_count,
                    int(len(df) * min_ratio),
                    max_samples_per_class if max_samples_per_class else float("inf"),
                )

            # Take a random balanced sample
            if sample_size < positive_count:
                pos_indices = positive_samples.sample(sample_size).index.tolist()
            else:
                pos_indices = positive_samples.index.tolist()

            neg_indices = negative_samples.sample(
                min(sample_size * 2, len(negative_samples))
            ).index.tolist()

            # Add the selected indices to the set
            selected_indices.update(pos_indices)
            selected_indices.update(neg_indices)

        # Create a new DataFrame with the selected indices
        balanced_df = df.loc[list(selected_indices)]

        # Show the size of the original and balanced datasets
        print(f"Original dataset size: {len(df)}")
        print(f"Balanced dataset size: {len(balanced_df)}")

        return balanced_df

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset, optimized for performance.
        """

        # Get image path
        img_path = self.data_frame.iloc[idx]["Path"]

        # Read the image using torchvision
        image = io.read_image(img_path, mode=io.ImageReadMode.GRAY)

        # Convert to float and normalize
        image = image.float() / 255.0

        # Redimensionar
        if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Normalize(mean=0.5, std=0.5)
        image = image * 2.0 - 1.0

        labels = torch.tensor(
            self.data_frame.iloc[idx][self.classes].values.astype(np.float32)
        )

        if self.transform:
            image = self.transform(image)

        return image, labels  # (1, img_size, img_size), (num_classes)


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

        self.transform = (
            None  # No transform is applied in the dataset (all done in dataset class)
        )

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
