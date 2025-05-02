#!/usr/bin/env python3
"""
fid_test.py -- Evaluate a Classifier‑Free Guided Diffusion (.ckpt) checkpoint on
CheXpert using the XRAY‑FID metric.

Example
-------
python fid_test.py \
       --ckpt checkpoints/cfg_diffusion/best.ckpt \
       --data_dir /path/to/CheXpert-v1.0-small \
       --csv /path/to/CheXpert-v1.0-small/valid.csv
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision as tv
import torchxrayvision as xrv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

# project‑local imports
from src.models.old_models.cfg_diffusion import ClassifierFreeGuidedDiffusion
from src.data.patched_chexpert import PatchedCheXpert as CheXpertDataset
from torchmetrics import Metric
from torchmetrics.image.fid import _compute_fid

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_grid(tensor, fname, title):
    # tensor in [-1,1] → [0,1]
    grid = make_grid((tensor + 1) * 0.5, nrow=4, padding=2)
    npimg = grid.cpu().numpy().transpose(1, 2, 0).squeeze()
    plt.figure(figsize=(6,6))
    plt.imshow(npimg, cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


###############################################################################
# Helper: load a Lightning checkpoint                                          
###############################################################################

def load_ckpt(path: str, device: torch.device | None = None) -> ClassifierFreeGuidedDiffusion:
    """Return a *ClassifierFreeGuidedDiffusion* model in *eval()* mode.

    If EMA weights were tracked during training we automatically swap the
    forward‑facing ``model.unet`` to the EMA copy for lower‑variance sampling.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    model = ClassifierFreeGuidedDiffusion.load_from_checkpoint(path, map_location="cpu", strict=False)
    model.eval()

    if device is not None:
        model.to(device)

    # Switch to EMA weights if available
    if getattr(model, "use_ema", False) and hasattr(model, "ema_unet") and model.ema_unet is not None:
        model.ema.copy_to(model.ema_unet.parameters())
        model.unet = model.ema_unet  # type: ignore[attr-defined]

    return model

###############################################################################
# Metric: XRAY‑FID                                                            
###############################################################################

class XRAYFID(Metric):
    """FID variant that uses TorchXRayVision DenseNet‑121 as feature extractor."""

    full_state_update = False  # stream updates instead of storing everything

    def __init__(self, device: torch.device, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.device_feat = device

        # Feature extractor (global‑avg‑pooled CNN trunk)
        self.encoder = xrv.models.DenseNet(weights="densenet121-res224-chex")
        self.encoder.classifier = torch.nn.Identity()
        self.encoder.eval().to(device)

        # Streaming buffers
        self.add_state("preds_real", default=[], dist_reduce_fx=None)
        self.add_state("preds_fake", default=[], dist_reduce_fx=None)

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """Convert *[-1,1]* grayscale images → 1024‑D DenseNet features."""
        img = ((img + 1) * 0.5 * 255).squeeze(1).cpu().numpy()  # B×H×W
        img = xrv.datasets.normalize(img, 255)
        img = torch.from_numpy(img).float().unsqueeze(1).to(self.device_feat)
        img = tv.transforms.Resize(224)(img)

        feats = self.encoder.features(img)          # B×1024×7×7
        feats = F.relu(feats, inplace=False)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
        return feats.cpu()

    def update(self, imgs: torch.Tensor, *, real: bool):
        feats = self._encode(imgs)
        (self.preds_real if real else self.preds_fake).append(feats)

    def _stats(self, lst):
        feats = torch.cat(lst, dim=0)
        return feats.mean(0), torch.cov(feats.T)

    def compute(self):
        mu_r, sig_r = self._stats(self.preds_real)
        mu_f, sig_f = self._stats(self.preds_fake)
        return _compute_fid(mu_r, sig_r, mu_f, sig_f)

###############################################################################
# Main                                                                        
###############################################################################

@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Compute XRAY‑FID for a CFG‑Diffusion checkpoint")
    p.add_argument("--ckpt", required=True, help="Path to *.ckpt* file")
    p.add_argument("--data_dir", required=True, help="Root folder of CheXpert images")
    p.add_argument("--csv", required=True, help="Validation CSV (frontal‑AP subset)")
    p.add_argument("--total", type=int, default=223_000, help="Number of fake samples to draw")
    p.add_argument("--batch", type=int, default=32, help="Batch size (real & fake)")
    p.add_argument("--steps", type=int, default=50, help="DDIM inference steps")
    p.add_argument("--guidance", type=float, default=3.0, help="Classifier‑free guidance scale")
    p.add_argument("--baseline",  choices=["none", "noise", "real"],default="none",
                   help="Skip the model and generate baseline fake images.")
    p.add_argument("--save_real_stats",
                    type=str, default=None,
                    help="If set, write μ_real and Σ_real to this .npz file "
                         "and quit (no generator needed).")
    
    args = p.parse_args()

    # ------------------------------------------------------------------ devices
    device_gen = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_metric = torch.device("cpu")

    # ------------------------------------------------------------------ generator
    model = load_ckpt(args.ckpt, device_gen)
    img_size = model.hparams.img_size  # size used during training
    n_classes = model.hparams.num_classes
    n_batches = math.ceil(args.total / args.batch)

    def gen_loader():
        if args.baseline == "noise":
            for _ in range(n_batches):
                imgs = torch.rand(args.batch, 1, img_size, img_size,
                                device=device_gen) * 2.0 - 1.0   # [-1,1]
                yield imgs

        elif args.baseline == "real":
            for imgs, _ in real_loader:           # reuse real batches
                yield imgs.to(device_gen)
                if (yielded := imgs.size(0)) >= args.total:
                    break

        else:  # normal diffusion model
            
            for _ in range(n_batches):
                labels = torch.bernoulli(
                    torch.full((args.batch, n_classes), 0.15, device=device_gen)
                )
                imgs = model.generate_samples(
                    labels=labels,
                    guidance_scale=args.guidance,
                    num_inference_steps=args.steps,
                )
                yield imgs  # (B,1,H,W) in [-1,1]

    # ------------------------------------------------------------------ real data
    import csv
    from pathlib import Path

    # first image path as written in the CSV
    with open(args.csv) as fh:
        rel_path = next(csv.DictReader(fh))["Path"]

    data_dir = Path(args.data_dir).resolve()
    
    # Ensure data_dir points to the CheXpert-v1.0-small directory
    if not data_dir.name == "CheXpert-v1.0-small":
        data_dir = data_dir / "CheXpert-v1.0-small"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"CheXpert directory not found at: {data_dir}")
    
    real_ds = CheXpertDataset(
        csv_file=args.csv,
        base_dir=str(data_dir),
        img_size=img_size,
        transform=None,
    )
    real_loader = DataLoader(
        real_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    if args.save_real_stats is not None:

        encoder = xrv.models.DenseNet(weights="densenet121-res224-chex")
        encoder.classifier = torch.nn.Identity()
        encoder.eval().to(device_metric)

        feats = []
        with torch.no_grad():
            for imgs, _ in tqdm(real_loader, desc="Real ↦ features"):
                imgs = imgs.to(device_metric, non_blocking=True)
                f = encoder.features(imgs)                # B×1024×7×7
                f = torch.relu(f)
                f = torch.nn.functional.adaptive_avg_pool2d(f, 1)
                feats.append(f.flatten(1).cpu().numpy())  # B×1024

        feats = np.concatenate(feats, axis=0)
        mu    = feats.mean(axis=0)
        sigma = np.cov(feats, rowvar=False)

        np.savez(args.save_real_stats, mu=mu, sigma=sigma)
        print(f"[saved] μ shape {mu.shape}, Σ shape {sigma.shape} → {args.save_real_stats}")
        raise SystemExit  # finished; skip the rest of the script


    # ------------------------------------------------------------------ metric
    fid = XRAYFID(device_metric)
    n_real = len(real_loader)
    n_fake = math.ceil(args.total / args.batch)

    preview_real  = None
    preview_fake  = None

    i = 0
    with tqdm(total=n_real + n_fake, unit="batch") as bar:
        # stream real batches
        for imgs, _ in tqdm(real_loader, desc="Real", unit="batch"):
            fid.update(imgs.to(device_metric), real=True)
            bar.update()
            if i == 0:                      # keep only the very first batch
                preview_real = imgs[:16]    # 4×4 grid; adjust if batch < 16
                save_grid(preview_real, "preview_real.png", "Real CheXpert (val)")
                i +=1

        # stream fake batches
        for imgs in tqdm(gen_loader(), total=n_fake, desc="Fake", unit="batch"):
            fid.update(imgs.to(device_metric), real=False)
            bar.update()
            if i == 1:                      # keep only the very first batch
                preview_fake = imgs[:16]    # 4×4 grid; adjust if batch < 16
                save_grid(preview_fake, "preview_fake.png", "Generated (guidance=%.2f)" % args.guidance)
                print("Saved comparison grids → preview_real.png  /  preview_fake.png")
                i += 1

    print(f"XRAY‑FID = {fid.compute().item():.4f}")

if __name__ == "__main__":
    main()