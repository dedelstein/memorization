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
from sklearn.model_selection import GroupShuffleSplit
from torchvision import transforms
from pathlib import Path
from typing import Optional

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
        data_frame: pd.DataFrame = None,   # ← new
        balance: bool = False
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
        # If a DataFrame is passed, use it directly; otherwise read csv_file
        if data_frame is not None:
            df = data_frame.copy()
        else:
            df = pd.read_csv(csv_file)

        # Filter for frontal images with AP projection
        filtered_df = df[(df["Frontal/Lateral"] == "Frontal") & (df["AP/PA"] == "AP")]

        # Fill NA values and convert -1.0 to 0.0
        filtered_df = filtered_df.fillna(0.0).replace(-1.0, 0.0)

        # Select only necessary columns
        filtered_df = filtered_df[["Path"] + self.findings]

        filtered_df = filtered_df.copy()
        filtered_df["patient_id"] = filtered_df["Path"].apply(
             lambda p: os.path.basename(os.path.dirname(p))
        )
        # Now store for indexing AND for your sampler to see:
        self.data_frame = filtered_df

        # Debug mode to use a small subset
        if debug_mode:
            self.data_frame = self.data_frame.iloc[: min(10000, len(self.data_frame))]

        # Balance the dataset
        if balance:
             filtered_df = self._balance_dataset(
                 filtered_df,
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

    def _balance_dataset(
        self,
        df: pd.DataFrame,
        min_ratio: float = 0.05,
        max_samples_per_class: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Light balancing by finding only; patient weighting is done externally.

        Args:
            df: Original DataFrame of frontal-AP images.
            min_ratio: Minimum desired positive ratio per finding.
            max_samples_per_class: Upper cap on positives sampled per finding.
        Returns:
            df_balanced: Subset DataFrame with a bit more minority-class support.
        """
        selected = set()
        findings = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax",
            "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
        ]
        N = len(df)

        for finding in findings:
            pos_df = df[df[finding] == 1]
            neg_df = df[df[finding] == 0]
            n_pos = len(pos_df)
            if n_pos == 0:
                continue

            # determine how many positives to sample
            target = int(N * min_ratio)
            take_pos = min(n_pos, target, max_samples_per_class or n_pos)

            pos_idx = (
                pos_df.sample(take_pos).index.tolist()
                if take_pos < n_pos
                else pos_df.index.tolist()
            )
            # sample up to twice as many negatives
            take_neg = min(len(neg_df), take_pos * 2)
            neg_idx = neg_df.sample(take_neg).index.tolist()

            selected.update(pos_idx + neg_idx)

        # build the balanced DataFrame
        df_balanced = df.loc[list(selected)].reset_index(drop=True)
        print(f"Balanced dataset: {len(df_balanced)} / {N} samples")

        return df_balanced


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset, optimized for performance.
        """

        # Get image path
        raw = self.data_frame.iloc[idx]["Path"]
        p = Path(raw)

        if not p.is_absolute():
            base = Path(self.base_dir)
            try:
                p = base / p.relative_to(base.name)
            except ValueError:
                p = base / p
        img_path = str(p)

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
        """Set up train and validation datasets (no test split)."""
        # 1) Load & filter CSVs
        train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        valid_df = pd.read_csv(os.path.join(self.data_dir, "valid.csv"))

        train_df = train_df[
            (train_df["Frontal/Lateral"] == "Frontal")
            & (train_df["AP/PA"] == "AP")
        ].reset_index(drop=True)

        valid_df = valid_df[
            (valid_df["Frontal/Lateral"] == "Frontal")
            & (valid_df["AP/PA"] == "AP")
        ].reset_index(drop=True)

        # 2) Extract patient_id
        def extract_pid(path):
            return os.path.basename(path).split("_")[0]

        train_df["patient_id"] = train_df["Path"].apply(extract_pid)
        valid_df["patient_id"] = valid_df["Path"].apply(extract_pid)

        # 3) Remove any patients from valid that appear in train
        valid_df = valid_df[
            ~valid_df["patient_id"].isin(train_df["patient_id"])
        ].reset_index(drop=True)

        # 4) (Optional) further split train_df internally if you want an 80/20 train/val—
        #    otherwise you can train on all of train_df and validate on valid_df.
        #    Here we'll train on *all* of train_df:
        df_train = train_df.reset_index(drop=True)
        df_val   = valid_df.reset_index(drop=True)

        # 5) Instantiate datasets
        if stage in (None, "fit"):
            self.train_dataset = CheXpertDataset(
                csv_file=None,               # ignore csv_file when data_frame is provided
                base_dir=self.data_dir,
                transform=self.transform,
                debug_mode=self.debug_mode,
                class_index=self.class_index,
                img_size=self.img_size,
                data_frame=df_train,        # use our DataFrame directly
                balance=True
            )
            self.val_dataset = CheXpertDataset(
                csv_file=None,
                base_dir=self.data_dir,
                transform=self.transform,
                debug_mode=self.debug_mode,
                class_index=self.class_index,
                img_size=self.img_size,
                data_frame=df_val,
                balance=False
            )
    def train_dataloader(self):

        patient_counts = self.train_dataset.data_frame["patient_id"].value_counts().to_dict()
        weights = [
            1.0 / patient_counts[pid]
            for pid in self.train_dataset.data_frame["patient_id"]
        ]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            sampler=sampler
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
