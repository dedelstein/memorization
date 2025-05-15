"""
PyTorch Lightning DataModule for CheXpert dataset.
"""

import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from src.data.chexpert_dataset import CheXpertDataset


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
        data_size: int = 5000,
        overfit: bool = False,
        seed=42,
        debug_mode=False,
        pin_memory=True,
        class_index=None
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
        self.data_size = data_size
        self.overfit = overfit

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
                overfit=self.overfit,
                data_size=self.data_size,
                class_index=self.class_index,
                img_size=self.img_size,
            )

            val_dataset = CheXpertDataset(
                csv_file=valid_csv,
                base_dir=self.data_dir,
                transform=self.transform,
                debug_mode=self.debug_mode,
                overfit=self.overfit,
                data_size=self.data_size,
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
