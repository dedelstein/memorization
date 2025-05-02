import math
import torch, torchvision as tv
import torchxrayvision as xrv
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.chexpert_dataset import CheXpertDataset
from src.models.conditional_ddpm_pipeline import ConditionalDDPMPipeline
from torchmetrics.image.fid import _compute_fid
from torchmetrics import Metric
from conditional_ddpm_inference import load_model
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

BATCH = 32                 # any value that fits on GPU
IMG_SIZE = 224             # must match the generator

model_path = "checkpoint-31000"
metric_device = torch.device("cpu")

real_ds = CheXpertDataset(
    csv_file="/dtu/blackhole/1d/214141/CheXpert-v1.0-small/valid.csv",
    base_dir="/dtu/blackhole/1d/214141/CheXpert-v1.0-small",
    img_size=IMG_SIZE,
    transform=None          # dataset already gives [-1,1]
)

real_loader = DataLoader(
    real_ds,
    batch_size=BATCH,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

pipe = load_model(model_path, ambient=False)
pipe.unet.eval()

GUIDE   = 1.5        # or sweep multiple values
STEPS   = 20
TOTAL   = 5_000      # number of images you want in FID
device  = pipe.device

def gen_loader():
    n_batches = math.ceil(TOTAL / BATCH)
    for _ in range(n_batches):
        # sample random multi-hot label vectors with the same dimension as CheXpert
        labels = torch.bernoulli(torch.full((BATCH, 14), 0.15, device=device))
        out = pipe(
            batch_size=BATCH,
            num_inference_steps=STEPS,
            class_labels=labels,
            guidance_scale=GUIDE,
            output_type="np",
            return_dict=True
        )
        imgs = torch.from_numpy(out["images"]).permute(0,3,1,2).float()  # → CHW
        imgs = imgs * 2.0 - 1.0                       # back to [-1,1] like real set
        yield imgs.to(device)

# ---------------- feature extractor ----------------
encoder = xrv.models.DenseNet(weights="densenet121-res224-chex")
encoder.classifier = torch.nn.Identity()
encoder.eval().cuda()

class XRAYFID(Metric):
    full_state_update = False

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.add_state("preds_real", default=[], dist_reduce_fx=None)
        self.add_state("preds_fake", default=[], dist_reduce_fx=None)

    # ---------------- feature extractor + preprocess ------------------
    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        img  : float tensor in [-1, 1], shape [B, 1, H, W]
        return: float tensor [B, 1024]
        """
        # ---------- TorchXRayVision normalisation ------------------
        img = ((img + 1) * 0.5 * 255).squeeze(1).cpu().numpy()   # [B,H,W]
        img = xrv.datasets.normalize(img, 255)
        img = torch.from_numpy(img).float().unsqueeze(1).to(device)

        # (CheXpertDataset already gives 224×224, but resize is harmless)
        img = tv.transforms.Resize(224)(img)

        # ---------- DenseNet feature stack -------------------------
        with torch.no_grad():
            feats = encoder.features(img)             # [B,1024,7,7]
            feats = F.relu(feats, inplace=False)
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
            # feats is now [B, 1024]

        return feats.cpu()

    # ---------------- metric bookkeeping ----------------------------
    def update(self, imgs: torch.Tensor, *, real: bool) -> None:
        feats = self._encode(imgs)
        (self.preds_real if real else self.preds_fake).append(feats)

    def _stats(self, feats):
        feats = torch.cat(feats, dim=0)
        return feats.mean(0), torch.cov(feats.T)

    def compute(self):
        mu1, sig1 = self._stats(self.preds_real)
        mu2, sig2 = self._stats(self.preds_fake)
        return _compute_fid(mu1, sig1, mu2, sig2)

#chex_fid = XRAYFID().to(device if torch.cuda.is_available() else "cpu")
chex_fid = XRAYFID().to(metric_device)

preview_real  = None
preview_fake  = None

n_real  = len(real_loader)
n_fake  = math.ceil(TOTAL / BATCH)

with tqdm(total=n_real + n_fake, unit="batch") as pbar:
    for i, batch in enumerate(tqdm(real_loader, desc="Real", unit="batch")):
        imgs = batch["image"].to(device)
        chex_fid.update(imgs, real=True)
        if i == 0:                      # keep only the very first batch
            preview_real = imgs[:16]    # 4×4 grid; adjust if batch < 16

    for i, imgs in enumerate(tqdm(gen_loader(),
            total=n_fake,
            desc="Fake", unit="batch")):
        chex_fid.update(imgs, real=False)
        if i == 0:
            preview_fake = imgs[:16]

print("CheX-FID =", chex_fid.compute().item())

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

save_grid(preview_real, "preview_real.png", "Real CheXpert (val)")
save_grid(preview_fake, "preview_fake.png", "Generated (guidance=%.2f)" % GUIDE)
print("Saved comparison grids → preview_real.png  /  preview_fake.png")