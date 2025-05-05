import cv2
import math
import numpy as np
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
GUIDE   = 1.5        # or sweep multiple values
STEPS   = 20
TOTAL   = 5_000      # number of images you want in FID

model_path = "checkpoint-31000"
#metric_device = torch.device("cpu")
metric_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class XRAYFID(Metric):
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds_real", default=[], dist_reduce_fx=None)
        self.add_state("preds_fake", default=[], dist_reduce_fx=None)

    # ---------------- feature extractor + preprocess ------------------
    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        img  : float tensor in [-1, 1], shape [B, 1, H, W]
        return: float tensor [B, 1024]
        """
        # ---------- TorchXRayVision normalisation ------------------
        if metric_device.type == "cuda":
            img = ((img + 1) * 0.5 * 255).squeeze(1).cpu().numpy()
        else:
            img = ((img + 1) * 0.5 * 255).squeeze(1).numpy()

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

        return feats.cpu() if metric_device == "cpu" else feats

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

class AnatomicalFID(Metric):
    """
    Anatomical‐accuracy FID that needs *no extra training*.

    It feeds each image through the frozen TorchXRayVision PSPNet,
    keeps the per‑class probability maps, spatially pools them to a
    fixed grid (8 × 8 by default) and runs standard FID on the resulting
    vectors.  The pooled maps encode organ size, position *and* coarse
    shape, giving a far richer signal than the old
    (area + centroid) triple.
    """

    full_state_update = False          # required by torchmetrics ≥0.11

    def __init__(self, pool_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size      # spatial grid size for pooling
        self.add_state("real_feats",  default=[], dist_reduce_fx=None)
        self.add_state("fake_feats",  default=[], dist_reduce_fx=None)
        self.add_state("fake_imgs",   default=[], dist_reduce_fx=None)


    # ------------------------------------------------------------------
    #  Feature extractor
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        imgs : float tensor in **[-1, 1]**, shape *(B, 1, H, W)*

        Returns
        -------
        feats : float tensor, shape *(B, C × pool_size²)*
        """
        # PSPNet expects HU‑like range [‑1024, 1024]
        imgs = imgs[:, :1].to(device)
        imgs = ((imgs + 1.0) * 0.5) * 2048.0 - 1024.0
        imgs = torch.nn.functional.interpolate(
            imgs, 512, mode="bilinear", align_corners=False
        )

        # Forward through the frozen segmenter
        logits = SEG_MODEL(imgs)                      # [B, C, H, W]
        probs  = torch.softmax(logits, dim=1)

        # Pool to a fixed grid and flatten
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            probs, self.pool_size                    # [B, C, p, p]
        )
        feats = pooled.flatten(1)                    # [B, C·p²]

        return feats if metric_device.type == device.type else feats.to(metric_device)

    # ------------------------------------------------------------------
    #  Book‑keeping
    # ------------------------------------------------------------------
    def update(self, imgs: torch.Tensor, *, real: bool) -> None:
        feats = self._encode(imgs)
        (self.real_feats if real else self.fake_feats).append(feats)
        if not real:                         # keep a lightweight copy for vis
            self.fake_imgs.extend(imgs.cpu())


    def compute(self) -> torch.Tensor:
        real = torch.cat(self.real_feats).double()
        fake = torch.cat(self.fake_feats).double()

        mu_r,  mu_f  = real.mean(0),  fake.mean(0)
        sigma_r, sigma_f = torch.cov(real.T), torch.cov(fake.T)

        return _compute_fid(mu_r, sigma_r, mu_f, sigma_f)

encoder = xrv.models.DenseNet(weights="densenet121-res224-chex")
encoder.classifier = torch.nn.Identity()
encoder.eval().cuda()

SEG_MODEL = xrv.baseline_models.chestx_det.PSPNet().eval().to(device)
anat_fid = AnatomicalFID().to(metric_device)

t        = SEG_MODEL.targets
L_LUNG   = t.index("Left Lung")
R_LUNG   = t.index("Right Lung")
HEART    = t.index("Heart")

#chex_fid = XRAYFID().to(device if torch.cuda.is_available() else "cpu")
chex_fid = XRAYFID().to(metric_device)

preview_real  = None
preview_fake  = None

n_real  = len(real_loader)
n_fake  = math.ceil(TOTAL / BATCH)

with tqdm(total=n_real + n_fake, unit="batch") as pbar:
    for i, batch in enumerate(real_loader):
        imgs = batch["image"].to(device)
        chex_fid.update(imgs, real=True)
        anat_fid.update(imgs, real=True) 
        if i == 0:                      # keep only the very first batch
            preview_real = imgs[:16]    # 4×4 grid; adjust if batch < 16

    for i, imgs in enumerate(gen_loader()):
        chex_fid.update(imgs, real=False)
        anat_fid.update(imgs, real=False) 
        if i == 0:
            preview_fake = imgs[:16]

print("Anatomical FID =", anat_fid.compute().item())
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

K = 12                     # how many outliers to display
OUT_PATH = "anat_outliers.png"
COLORS = [(0,255,0), (255,0,0), (0,0,255), (255,255,0),
          (0,255,255), (255,0,255), (192,128,0), (0,128,192),
          (128,0,192), (192,0,128), (128,192,0), (192,0,64),
          (64,192,0), (0,64,192)]   # up to 14 classes

# --- 1. Gather features and Mahalanobis distance ------------------
real_feats = torch.cat(anat_fid.real_feats).double()
fake_feats = torch.cat(anat_fid.fake_feats).double()
mu_r   = real_feats.mean(0, keepdim=True)             # (1,C)
Sigma  = (torch.cov(real_feats.T) + 1e-6 * torch.eye(real_feats.shape[1], device=real_feats.device))
# pre‑compute Σ⁻¹ once
Sigma_inv = torch.linalg.inv(Sigma)

# d^2 = (z‑μ)^T Σ⁻¹ (z‑μ)
diff     = fake_feats - mu_r
maha_sq  = (diff @ Sigma_inv * diff).sum(1)           # (N,)

topk_idx = torch.topk(maha_sq, k=min(K, maha_sq.numel()))[1]

# --- 2. Build overlay images -------------------------------------
grid_imgs = []
SEG_MODEL.eval()                                      # ensure inference mode
CLASS_NAMES = SEG_MODEL.targets

with torch.no_grad():
    for idx in topk_idx:
        img = anat_fid.fake_imgs[idx].to(device)      # keep a copy earlier
        img_gray = ((img.cpu()*0.5+0.5)*255).clamp_(0,255)[0].to(torch.uint8).numpy()
        vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # ── segment ───────────────────────────────────────────────
        logits = SEG_MODEL(torch.nn.functional.interpolate(
                ((img+1)*0.5*2048-1024).unsqueeze(0), 512,
                mode="bilinear", align_corners=False))[0] # (C,H,W)

        probs = logits.softmax(0)                       # (C,H,W)
        conf, cls = probs.max(0)                        # (H,W)
        mask  = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(0, cls.unsqueeze(0), (conf > 0.6).unsqueeze(0))  # high‑confidence pixels

        # keep only largest blob per class
        mask_np = mask.cpu().numpy()
        clean = np.zeros_like(mask_np)
        for c in range(mask_np.shape[0]):
            cnt, lab = cv2.connectedComponents(mask_np[c].astype('uint8'))
            if cnt > 1:
                sizes = np.bincount(lab.flat)[1:]
                clean[c] = lab == (1 + sizes.argmax())
        mask = torch.from_numpy(clean)
        h, w = vis.shape[:2]                                           # 224,224
        mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).float(), size=(h, w),
                    mode="nearest")[0].bool()                          # (C,H,W)
        mask_np = mask.cpu().numpy().astype(np.uint8)                  # NumPy!

        # --------  overlay: filled alpha + 2‑px contour  ---------------
        alpha   = 0.1
        overlay = vis.copy()
        for c, col in enumerate(COLORS):

            cls_name = SEG_MODEL.targets[c]
            color    = tuple(int(x) for x in col)             # BGR 0‑255
            m        = mask_np[c]                             # (H,W) uint8

            pix = int(m.sum())                                # how many pixels?
            if pix:                                           # skip empty
                overlay[m.astype(bool)] = color               # filled label
                
        vis = cv2.addWeighted(overlay, alpha, vis, 1-alpha, 0)

        grid_imgs.append(torch.from_numpy(vis).permute(2,0,1))

def build_legend(class_names, colors, height=24, pad=6):
    """Return a HxWx3 uint8 legend image."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1

    # estimate width
    widths = [cv2.getTextSize(n, font, scale, thick)[0][0] for n in class_names]
    W = sum(widths) + pad * (len(class_names) + 1)
    legend = np.zeros((height, W, 3), dtype=np.uint8) + 255   # white bg

    x = pad
    for name, col, w in zip(class_names, colors, widths):
        overlay = legend.copy()
        cv2.rectangle(legend, (x-3, 5), (x+15, 17), col, -1)
        cv2.addWeighted(overlay, alpha, legend,
                    1-alpha, 0, legend)
        cv2.putText(legend, name, (x+18, 17), font, scale, (0,0,0), thick,
                    cv2.LINE_AA)
        x += w + pad
    return legend

# --- 3. Save grid -------------------------------------------------
if grid_imgs:
    # ---------- 3×4 grid of worst‑K images ----------
    grid = make_grid(grid_imgs, nrow=4, padding=2, value_range=(0, 255))
    grid_img = (grid / 255.0).permute(1, 2, 0).cpu().numpy() * 255   # H×W×3 uint8

    # ---------- build legend strip (title + coloured rows) ----------
    square   = 14           # coloured box size
    gap      = 8            # vertical gap
    font_t   = cv2.FONT_HERSHEY_SIMPLEX
    scale_t  = 0.48         # title font
    thick_t  = 1
    title    = ["High-confidence organ regions",
                "(p > 0.6, PSPNet)"]

    (tw1, th1), _ = cv2.getTextSize(title[0], font_t, scale_t, thick_t)
    (tw2, th2), _ = cv2.getTextSize(title[1], font_t, scale_t, thick_t)
    title_h = th1 + th2 + gap

    square_font = cv2.FONT_HERSHEY_SIMPLEX
    scale_f     = 0.45
    thick_f     = 1

    row_h = square + gap
    H_leg = title_h + gap + row_h * len(CLASS_NAMES) + gap
    W_leg = max(max(tw1, tw2) + gap * 2, 150)
    legend = np.ones((H_leg, W_leg, 3), dtype=np.uint8) * 255   # white bg

    # ---- title lines ----
    y0 = gap + th1
    cv2.putText(legend, title[0],
                ((W_leg - tw1) // 2, y0),
                font_t, scale_t, (0, 0, 0), thick_t, cv2.LINE_AA)
    cv2.putText(legend, title[1],
                ((W_leg - tw2) // 2, y0 + th2 + 2),
                font_t, scale_t, (0, 0, 0), thick_t, cv2.LINE_AA)

    # ---- coloured rows ----
    y = title_h + gap
    for name, col in zip(CLASS_NAMES, COLORS):
        col_bgr = tuple(int(v) for v in col)  # ensure BGR order for OpenCV
        cv2.rectangle(legend, (6, y), (6 + square, y + square), col_bgr, -1)
        cv2.putText(legend, f" {name}",
                    (6 + square + 4, y + square - 2),
                    square_font, scale_f, (0, 0, 0), thick_f, cv2.LINE_AA)
        y += row_h

    # ---------- compose final image (grid left, legend right) ----------
    pad  = 4
    H_out = max(grid_img.shape[0], legend.shape[0])
    W_out = grid_img.shape[1] + legend.shape[1] + pad
    out   = np.ones((H_out, W_out, 3), dtype=np.uint8) * 255

    # place grid
    out[:grid_img.shape[0], :grid_img.shape[1]] = grid_img

    # place legend, centred vertically
    y_leg = (H_out - legend.shape[0]) // 2
    out[y_leg : y_leg + legend.shape[0],
        grid_img.shape[1] + pad : grid_img.shape[1] + pad + legend.shape[1]] = legend

    cv2.imwrite(OUT_PATH, out)          # 'out' already in BGR order
    print(f"Saved anatomical outlier grid →  {OUT_PATH}")
else:
    print("No fake images were collected for outlier vis.")