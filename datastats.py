import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Paths & constants – change these if your environment differs
# -----------------------------------------------------------------------------
DATA_DIR   = Path("/dtu/blackhole/1d/214141/CheXpert-v1.0-small")
TRAIN_CSV  = DATA_DIR / "train.csv"
OUTPUT_DIR = Path("./")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOP_N = 5

# -----------------------------------------------------------------------------
# 2. Load datasets (uses your existing CheXpertDataset wrapper)
# -----------------------------------------------------------------------------
from src.data.chexpert_dataset import CheXpertDataset  # keep local import after path setup

full_ds  = CheXpertDataset(csv_file=str(TRAIN_CSV), base_dir=str(DATA_DIR), debug_mode=False)
debug_ds = CheXpertDataset(csv_file=str(TRAIN_CSV), base_dir=str(DATA_DIR), debug_mode=True)
label_names = full_ds.classes  # «14 canonical CheXpert labels»

# -----------------------------------------------------------------------------
# 3. Per‑label frequencies – **index = label name** so order is preserved after sort
# -----------------------------------------------------------------------------
full_counts  = full_ds.data_frame[label_names].sum().astype(int)
debug_counts = debug_ds.data_frame[label_names].sum().astype(int)

full_df  = pd.DataFrame({
    "count_full": full_counts,
    "pct_full"  : (full_counts  / len(full_ds)  * 100).round(2),
})

debug_df = pd.DataFrame({
    "count_5k" : debug_counts,
    "pct_5k"   : (debug_counts / len(debug_ds) * 100).round(2),
})

# concat keeps the *label names* as the shared index
comp_df = pd.concat([debug_df, full_df], axis=1)

# -----------------------------------------------------------------------------
# 4. Sort by (debug) prevalence — index carries the labels so they stay aligned
# -----------------------------------------------------------------------------
comp_df = comp_df.sort_values("count_5k", ascending=True)

# -----------------------------------------------------------------------------
# 5. Comparative prevalence plot (debug subset vs full set)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 3))
comp_df[["pct_5k", "pct_full"]].plot.barh(ax=ax, legend=False)

# Data‑series legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title="Dataset", fontsize=6, title_fontsize="x-small", loc="lower center")

# Class‑index→name legend (optional, retains your numeric mapping)
legend_handles = [
    Patch(facecolor="white", edgecolor="none", label=f"{i}: {name}")
    for i, name in enumerate(label_names, 1)
]

ser_handles, ser_labels = ax.get_legend_handles_labels()
series_leg = ax.legend(
    ser_handles,
    ser_labels,
    title="Dataset",
    loc="lower center",
    #bbox_to_anchor=(.5, 0.5),
    title_fontsize='x-small',
    fontsize=5
)

ax.legend(handles=legend_handles, title="Class indices", fontsize=5, title_fontsize="x-small", loc="lower right", frameon=True)
ax.add_artist(series_leg)

orig_number = {name: i + 1              # 1, 2, …, 14
               for i, name in enumerate(label_names)}

y_labels = [orig_number[label] for label in comp_df.index]

ax.set_xlabel("Prevalence (%)")
ax.set_title("Label Prevalence: Debug Subset vs Full Training Set")
ax.set_xlim(right=62)
ax.set_yticks(np.arange(len(comp_df)))
ax.set_yticklabels(y_labels, fontsize=6)  # ← label names now match sorted bars
plt.setp(ax.patches, height=0.5)
ax.margins(y=-.45)
ax.tick_params(axis="y", pad=0)
fig.tight_layout()
fig.subplots_adjust(right=0.7)
fig.savefig(OUTPUT_DIR / "comparative_prevalence.png", dpi=300)

# -----------------------------------------------------------------------------
# 6. Top‑5 full 14‑label multihot patterns (incl. all‑zeros)
def tuple_to_pos_labels(t):
    """Return a list of class names whose bit is 1 in the 14-bit tuple."""
    return [lbl for lbl, bit in zip(label_names, t) if bit == 1]

zero_tuple = tuple([0] * len(label_names))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pattern frequency tables for full set and debug subset
# ─────────────────────────────────────────────────────────────────────────────
full_patterns = full_ds.data_frame[label_names].apply(tuple, axis=1).value_counts()
sub_patterns  = debug_ds.data_frame[label_names].apply(tuple, axis=1).value_counts()

for patterns in (full_patterns, sub_patterns):
    if zero_tuple not in patterns.index:
        patterns.loc[zero_tuple] = 0  # guarantee all-zero row exists

# ─────────────────────────────────────────────────────────────────────────────
# 5. Select the top-N patterns (by full-set prevalence) – ensure all-zero in list
# ─────────────────────────────────────────────────────────────────────────────
sel_keys = list(full_patterns.sort_values(ascending=False).head(TOP_N).index)
if zero_tuple not in sel_keys:
    sel_keys[-1] = zero_tuple  # overwrite last entry so all-zero is included

# ─────────────────────────────────────────────────────────────────────────────
# 6. Comparison DataFrame (flat Index via dtype="object" to avoid MultiIndex)
# ─────────────────────────────────────────────────────────────────────────────
idx = pd.Index(sel_keys, dtype="object")
# Then, set the name of the Index
idx.name = "pattern"
# Now, create the DataFrame with this correctly formed Index
pat_df = pd.DataFrame(index=idx)
pat_df["count_full"] = pat_df.index.map(full_patterns.get).astype(int)
pat_df["pct_full"]   = (pat_df["count_full"] / len(full_ds)   * 100).round(2)
pat_df["count_5k"]   = pat_df.index.map(sub_patterns.get).fillna(0).astype(int)
pat_df["pct_5k"]     = (pat_df["count_5k"]   / len(debug_ds) * 100).round(2)
pat_df["pos_labels"] = pat_df.index.map(tuple_to_pos_labels)
pat_df["label_str"]  = pat_df["pos_labels"].apply(lambda lst: ", ".join(lst) if lst else "[ ]")
pat_df["label_idx_str"] = pat_df["pos_labels"].apply(
    lambda lst: str([orig_number[name] for name in lst]) if lst else "[]"
)

# Sort for nicer bar order (smallest prevalence at bottom)
pat_df = pat_df.sort_values("pct_full", ascending=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plot: comparative prevalence (debug subset vs full)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3, 3))
pat_df[["pct_5k", "pct_full"]].plot.barh(ax=ax, legend=False)

ax.set_xlabel("Prevalence (%)")
ax.set_yticks(np.arange(len(pat_df)))
ax.set_yticklabels(pat_df["label_idx_str"], fontsize=6)
plt.setp(ax.patches, height=0.5)
ax.margins(y=-.45)
ax.tick_params(axis="y", pad=0)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "comparative_topN_multihot.png", dpi=300)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Console summary table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== Top-{TOP_N} multihot patterns (incl. all-zero) ===")
print(pat_df.sort_values("pct_full", ascending=False)[[
    "count_5k", "pct_5k", "count_full", "pct_full", "label_str"
]].to_string(index=False))