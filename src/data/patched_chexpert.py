# ---------------------------------------------------------------- patched_chexpert.py
from pathlib import Path
import os
import numpy as np
import torch
import torchvision.io as io
import torch.nn.functional as F

# re-use all constructor / helpers from the original dataset
from .chexpert_dataset import CheXpertDataset as _CheXpert


class PatchedCheXpert(_CheXpert):
    """
    A CheXpertDataset that transparently fixes broken absolute paths
    by re-rooting them under `base_dir` when the file is missing.
    Only __getitem__ is overridden; everything else is inherited.
    """

    def __getitem__(self, idx):
        raw = self.data_frame.iloc[idx]["Path"]
        p = Path(raw)

        if p.is_absolute():
            # If /dtu/CheXpert-v1.0-small/... really exists, great.
            # Otherwise fall back to <base_dir>/CheXpert-v1.0-small/...
            if not p.exists():
                try:
                    rel = p.relative_to(p.anchor)        # drop the leading '/'
                except ValueError:
                    rel = p.name                         # safety fallback
                p = Path(self.base_dir) / rel
        else:                                            # relative → original logic
            base = Path(self.base_dir)
            try:
                p = base / p.relative_to(base.name)
            except ValueError:
                p = base / p

        img_path = str(p)

        # ---- read and preprocess ------------------------------------------------
        image = io.read_image(img_path, mode=io.ImageReadMode.GRAY).float() / 255.0

        if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        image = image * 2.0 - 1.0                # map [0,1] → [-1,1]

        labels = torch.tensor(
            self.data_frame.iloc[idx][self.classes].values.astype(np.float32)
        )

        return image, labels
