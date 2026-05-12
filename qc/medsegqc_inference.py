#!/usr/bin/env python
"""
Standalone segmentation QC scorer.

Given a CT image and a binary segmentation mask, predicts:
  1. A Dice-like quality score in [0, 1]  (higher = better segmentation)
  2. Which of 12 abdominal organs the mask most likely depicts

No other files from this repository are required.  ``tests/test_inference.py`` is
optional: it only regression-tests against this repo's harvested CSV/H5 data; you
do not ship or import it when using this script elsewhere.

**Preprocessing (built into this file):** load CT and mask, take a bounding-box
crop around the mask (margin 8 voxels), resample both to 160³ (trilinear CT,
nearest mask), z-score the CT crop, stack ``[CT, mask]`` as two channels.  CT and
mask must be the **same shape** and aligned in voxel space (typical: full-volume
NIfTIs in the same reference).

**Harvest HDF5 caveat:** our internal ``error_zoo`` stores **pre-cropped** 160³
predictions.  Do not pass those tiny volumes as ``--mask`` together with a
full-resolution CT; spatial grids will not match.  For plug-in use, always pass
full-volume (or same-grid) CT + segmentation.

Dependencies:
    pip install torch monai nibabel numpy

Usage examples:
    # NIfTI inputs
    python inference.py --image ct.nii.gz --mask seg.nii.gz

    # NumPy .npz inputs (first array in each file is used)
    python inference.py --image ct.npz --mask seg.npz

    # Specify a local checkpoint instead of downloading
    python inference.py --image ct.nii.gz --mask seg.nii.gz --checkpoint /path/to/qc_regressor.ckpt

    # Use CPU
    python inference.py --image ct.nii.gz --mask seg.nii.gz --device cpu

The checkpoint is downloaded automatically on first run to ~/.cache/norefqca/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet18

# ──────────────────────────────────────────────
#  Organ label map (matches training config)
# ──────────────────────────────────────────────
ORGAN_NAMES = [
    "liver",
    "spleen",
    "kidney_left",
    "kidney_right",
    "pancreas",
    "stomach",
    "aorta",
    "adrenal_gland_left",
    "adrenal_gland_right",
    "prostate",
    "urinary_bladder",
    "gallbladder",
]

CHECKPOINT_URL = (
    "https://github.com/ucsdbiag/NoRefSegQCA/releases/download/v1.0.0/"
    "qc_regressor_epch33.ckpt"
)
CACHE_DIR = Path.home() / ".cache" / "norefqca"
SPATIAL_SIZE = (160, 160, 160)


# ──────────────────────────────────────────────
#  Model definition (self-contained)
# ──────────────────────────────────────────────
class DiceQANet(nn.Module):
    def __init__(self, in_channels=2, num_organs=12):
        super().__init__()
        self.backbone = resnet18(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=512,
            feed_forward=True,
            pretrained=False,
        )
        self.fc_dice = nn.Linear(512, 1)
        self.fc_organ = nn.Linear(512, num_organs)

    def forward(self, x):
        h = self.backbone(x)
        dice = torch.sigmoid(self.fc_dice(h)).squeeze(-1)
        organ_logits = self.fc_organ(h)
        return dice, organ_logits


# ──────────────────────────────────────────────
#  Preprocessing helpers
# ──────────────────────────────────────────────
def load_volume(path: str) -> np.ndarray:
    """Load a 3D volume from .nii/.nii.gz or .npz."""
    if path.endswith(".npz"):
        data = np.load(path)
        key = list(data.keys())[0]
        return data[key].astype(np.float32)
    else:
        return nib.load(path).get_fdata().astype(np.float32)


def bbox_crop(mask: np.ndarray, margin: int = 8):
    """Tight bounding-box crop around non-zero region."""
    nz = np.where(mask > 0.5)
    if len(nz[0]) == 0:
        return slice(None), slice(None), slice(None)
    slices = []
    for ax in range(3):
        lo = max(0, int(nz[ax].min()) - margin)
        hi = int(nz[ax].max()) + margin + 1
        slices.append(slice(lo, hi))
    return tuple(slices)


def resize3d(vol: np.ndarray, shape, is_label: bool) -> np.ndarray:
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    mode = "nearest" if is_label else "trilinear"
    kw = {} if is_label else {"align_corners": False}
    y = F.interpolate(t, size=shape, mode=mode, **kw)
    return y[0, 0].cpu().numpy()


def preprocess(image: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """Crop around mask, resize to 160^3, normalise CT, stack channels."""
    mask_bin = (mask > 0.5).astype(np.float32)
    slc = bbox_crop(mask_bin)
    ct_crop = image[slc]
    mk_crop = mask_bin[slc]

    ct_r = resize3d(ct_crop, SPATIAL_SIZE, is_label=False)
    mk_r = resize3d(mk_crop, SPATIAL_SIZE, is_label=True)
    mk_r = (mk_r > 0.5).astype(np.float32)

    mu, sig = float(ct_r.mean()), float(ct_r.std()) + 1e-6
    ct_r = (ct_r - mu) / sig

    x = np.stack([ct_r, mk_r], axis=0)  # (2, D, H, W)
    return torch.from_numpy(x).unsqueeze(0)  # (1, 2, D, H, W)


# ──────────────────────────────────────────────
#  Checkpoint loading
# ──────────────────────────────────────────────
def download_checkpoint(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint to {dest} ...")
    torch.hub.download_url_to_file(url, str(dest))
    return dest


def load_model(checkpoint_path: str | None, device: torch.device) -> DiceQANet:
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        local = CACHE_DIR / "qc_regressor_epch33.ckpt"
        if not local.is_file():
            download_checkpoint(CHECKPOINT_URL, local)
        checkpoint_path = str(local)

    model = DiceQANet(in_channels=2, num_organs=len(ORGAN_NAMES))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # Lightning checkpoints prefix keys with "net."
    clean = {}
    for k, v in sd.items():
        if k.startswith("net."):
            clean[k[4:]] = v
        else:
            clean[k] = v
    model.load_state_dict(clean, strict=True)
    model.eval().to(device)
    return model


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def predict(image_path: str, mask_path: str,
            checkpoint: str | None = None,
            device: str = "cuda") -> dict:
    """Run QC prediction. Returns dict with score, organ name, and organ index."""
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = load_model(checkpoint, dev)

    image = load_volume(image_path)
    mask = load_volume(mask_path)
    x = preprocess(image, mask).to(dev)

    with torch.no_grad():
        dice_pred, organ_logits = model(x)

    score = float(dice_pred.cpu())
    organ_idx = int(organ_logits.argmax(dim=-1).cpu())
    organ_name = ORGAN_NAMES[organ_idx]

    return {"qc_score": score, "organ": organ_name, "organ_idx": organ_idx}


def main():
    ap = argparse.ArgumentParser(
        description="Reference-free segmentation QC: predict Dice quality score and organ."
    )
    ap.add_argument("--image", required=True, help="CT image (.nii.gz or .npz)")
    ap.add_argument("--mask", required=True, help="Binary segmentation mask (.nii.gz or .npz)")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to .ckpt file (auto-downloaded if omitted)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="Inference device (default: cuda)")
    args = ap.parse_args()

    result = predict(args.image, args.mask, args.checkpoint, args.device)

    print(f"QC score (predicted Dice): {result['qc_score']:.4f}")
    print(f"Predicted organ:           {result['organ']}")


if __name__ == "__main__":
    main()