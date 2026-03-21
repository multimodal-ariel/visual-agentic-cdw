"""
BiomedParse3D Runner Script
============================
Thin CLI wrapper around the BiomedParse v2 Python API.
Called by biomedparse3d.py via conda run (inside cdw_biomedparse3d env).

Usage:
    python run_biomedparse3d.py
        --input   /path/to/image_nifti.nii.gz
        --output  /path/to/segmentations_biomedparse3d/
        --prompts liver spleen kidney
        --weights /path/to/biomedparse_3D_AllData_MultiView_edge.ckpt
        --biomedparse_dir /path/to/external/BiomedParse3D

The script:
  1. Loads the NIfTI volume as a numpy array (float32, windowed for CT).
  2. Builds the text prompt string: "liver[SEP]spleen[SEP]kidney".
  3. Runs the BiomedParse3D model.
  4. Thresholds the sigmoid output per-organ.
  5. Saves per-organ binary uint8 NIfTI files to output/.
"""

import argparse
import os
import sys
import gc

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F


def window_ct(vol: np.ndarray, wl: float = 60.0, ww: float = 400.0) -> np.ndarray:
    """Window/level a CT HU array to [0, 255]."""
    lo = wl - ww / 2
    hi = wl + ww / 2
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo) * 255.0
    return vol.astype(np.float32)


def load_nifti_as_array(image_path: str) -> tuple:
    """Return (data_float32, affine, header)."""
    img = nib.load(image_path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def main():
    parser = argparse.ArgumentParser(description="BiomedParse3D single-case inference")
    parser.add_argument("--input", required=True, help="Path to input NIfTI image")
    parser.add_argument("--output", required=True, help="Output directory for per-organ masks")
    parser.add_argument("--prompts", required=True, nargs="+", help="Organ text prompts")
    parser.add_argument("--weights", required=True, help="Path to model weights .ckpt file")
    parser.add_argument(
        "--biomedparse_dir",
        default=None,
        help="Path to BiomedParse3D repo root (defaults to BIOMEDPARSE3D_DIR env var)",
    )
    parser.add_argument("--slice_batch_size", type=int, default=4)
    args = parser.parse_args()

    biomedparse_dir = args.biomedparse_dir or os.environ.get("BIOMEDPARSE3D_DIR", "")
    if not biomedparse_dir or not os.path.isdir(biomedparse_dir):
        print(f"[ERROR] BiomedParse3D directory not found: {biomedparse_dir}", flush=True)
        sys.exit(1)

    # Add BiomedParse3D to sys.path so its imports work
    sys.path.insert(0, biomedparse_dir)

    import hydra
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra


    os.makedirs(args.output, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BiomedParse3D] Using device: {device}", flush=True)

    GlobalHydra.instance().clear()
    # initialize_config_dir accepts absolute paths (unlike initialize which needs relative)
    configs_model_dir = os.path.join(biomedparse_dir, "configs", "model")
    hydra.initialize_config_dir(config_dir=configs_model_dir, job_name="cdw_inference")
    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")
    model.load_pretrained(args.weights)
    model.to(device)
    model.eval()

    # ── Load image ──────────────────────────────────────────────────────────
    vol, affine, header = load_nifti_as_array(args.input)

    # Intensity windowing (CT: HU → [0,255]; MRI: percentile normalize → [0,255])
    p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
    if p1 < -500:  # likely CT (HU range)
        vol = window_ct(vol)
    else:
        vol = np.clip((vol - p1) / (p99 - p1 + 1e-8) * 255.0, 0, 255).astype(np.float32)

    # BiomedParse3D utils: process_input expects (D, H, W) or similar 3D ndarray
    from utils import process_input, process_output

    vol_processed, pad_width, padded_size, valid_axis = process_input(vol, size=512)

    # ── Run inference ────────────────────────────────────────────────────────
    text = "[SEP]".join(args.prompts)
    print(f"[BiomedParse3D] Prompts: {text}", flush=True)

    input_tensor = {
        "image": vol_processed.to(device).int().unsqueeze(0),
        "text": [text],
    }

    with torch.no_grad():
        output = model(input_tensor, mode="eval", slice_batch_size=args.slice_batch_size)

    mask_preds = output["predictions"]["pred_gmasks"]
    mask_preds = F.interpolate(
        mask_preds,
        size=(512, 512),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    # Threshold: sigmoid > 0.5 per organ channel
    object_existence = output["predictions"]["object_existence"]
    threshold = 0.5
    masks = (mask_preds.sigmoid()) * (
        object_existence.sigmoid() > threshold
    ).int().unsqueeze(-1).unsqueeze(-1)  # shape: (N_prompts, D, H, W)

    masks = process_output(masks, pad_width, padded_size, valid_axis)

    del input_tensor, output, mask_preds
    gc.collect()
    torch.cuda.empty_cache()

    # ── Save per-organ NIfTI files ───────────────────────────────────────────
    masks_np = masks.cpu().numpy()  # (N_prompts, D, H, W)
    shape_3d = vol.shape

    for i, organ in enumerate(args.prompts):
        if i >= masks_np.shape[0]:
            break
        mask = masks_np[i]  # (D, H, W) or (H, W, D) depending on axis handling
        # Restore original orientation: valid_axis was moved to dim 0 in process_input
        mask = np.moveaxis(mask, 0, valid_axis)
        # Resize back to original shape if needed
        if mask.shape != shape_3d:
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_t = F.interpolate(mask_t, size=shape_3d, mode="nearest")
            mask = mask_t.squeeze().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8)
        out_path = os.path.join(args.output, f"{organ}.nii.gz")
        nib.save(nib.Nifti1Image(binary_mask, affine, header), out_path)
        print(f"[BiomedParse3D] Saved {organ}: {binary_mask.sum()} voxels → {out_path}", flush=True)

    print("[BiomedParse3D] Done.", flush=True)


if __name__ == "__main__":
    main()
