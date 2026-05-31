#!/usr/bin/env python3
"""
Create quick PNG overlays for SynthSeg smoke-test outputs.

For each case in the smoke filelist, this script loads:
  <case_path>/image_nifti.nii.gz
  <case_path>/segmentations_synthseg/brain.nii.gz
  <case_path>/segmentations_synthseg/synthseg_labels_resampled.nii.gz (optional)

and writes one 3-panel montage PNG with the brain mask and label contours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.constants import DST_PREFIX, IMAGE_FILENAME, SRC_PREFIX, TOOL_OUTPUT_DIRS


DEFAULT_FILELIST = REPO_ROOT / "new_data_paths" / "synthseg_smoke_head_ct_mri.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "synthseg_smoke"
SYNTHSEG_DIRNAME = TOOL_OUTPUT_DIRS["SynthSeg"]


def remap_case_path(path: str) -> Path:
    if path.startswith(SRC_PREFIX):
        path = path.replace(SRC_PREFIX, DST_PREFIX, 1)
    return Path(path)


def robust_window(volume: np.ndarray) -> tuple[float, float]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def center_of_mask(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return tuple(int(s // 2) for s in mask.shape[:3])
    return tuple(int(x) for x in np.median(coords, axis=0))


def slice_triplet(volume: np.ndarray, mask: np.ndarray, labels: np.ndarray | None) -> list[dict]:
    z, y, x = center_of_mask(mask)
    panels = [
        {
            "title": f"axial z={z}",
            "image": volume[:, :, z].T,
            "mask": mask[:, :, z].T,
            "labels": labels[:, :, z].T if labels is not None else None,
        },
        {
            "title": f"coronal y={y}",
            "image": volume[:, y, :].T,
            "mask": mask[:, y, :].T,
            "labels": labels[:, y, :].T if labels is not None else None,
        },
        {
            "title": f"sagittal x={x}",
            "image": volume[x, :, :].T,
            "mask": mask[x, :, :].T,
            "labels": labels[x, :, :].T if labels is not None else None,
        },
    ]
    return panels


def make_case_title(case_path: Path) -> str:
    parts = case_path.parts
    try:
        idx = parts.index("segmentations_3d")
        return "/".join(parts[idx + 1 :])
    except ValueError:
        return str(case_path)


def visualize_case(case_path: Path, output_dir: Path) -> dict:
    image_path = case_path / IMAGE_FILENAME
    synthseg_dir = case_path / SYNTHSEG_DIRNAME
    brain_path = synthseg_dir / "brain.nii.gz"
    labels_path = synthseg_dir / "synthseg_labels_resampled.nii.gz"

    record = {
        "case_path": str(case_path),
        "image_path": str(image_path),
        "brain_path": str(brain_path),
        "labels_path": str(labels_path),
        "status": "pending",
        "png": "",
        "error": "",
    }

    if not image_path.is_file():
        record["status"] = "missing_image"
        record["error"] = f"Missing image: {image_path}"
        return record
    if not brain_path.is_file():
        record["status"] = "missing_brain_mask"
        record["error"] = f"Missing SynthSeg brain mask: {brain_path}"
        return record

    image_img = nib.load(str(image_path))
    brain_img = nib.load(str(brain_path))
    volume = np.asanyarray(image_img.dataobj).astype(np.float32)
    brain = np.asanyarray(brain_img.dataobj) > 0

    labels = None
    if labels_path.is_file():
        labels = np.asanyarray(nib.load(str(labels_path)).dataobj)

    if volume.shape[:3] != brain.shape[:3]:
        record["status"] = "shape_mismatch"
        record["error"] = f"image shape {volume.shape[:3]} != brain shape {brain.shape[:3]}"
        return record

    vmin, vmax = robust_window(volume)
    panels = slice_triplet(volume, brain, labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_name = case_path.name.replace("/", "_") + "_synthseg_overlay.png"
    png_path = output_dir / png_name

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, panel in zip(axes, panels):
        ax.imshow(panel["image"], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.imshow(np.ma.masked_where(~panel["mask"], panel["mask"]), cmap="autumn", alpha=0.25, origin="lower")
        if panel["labels"] is not None:
            ax.contour(panel["labels"] > 0, levels=[0.5], colors="cyan", linewidths=0.6, origin="lower")
        ax.set_title(panel["title"])
        ax.axis("off")

    fig.suptitle(make_case_title(case_path), fontsize=9)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    record["status"] = "visualized"
    record["png"] = str(png_path)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SynthSeg smoke-test outputs.")
    parser.add_argument("--filelist", default=str(DEFAULT_FILELIST), help="Smoke-test JSON filelist")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for PNG overlays")
    parser.add_argument("--summary", default="", help="Optional JSON summary path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.filelist).open("r", encoding="utf-8") as f:
        paths = json.load(f)

    output_dir = Path(args.output_dir)
    records = [visualize_case(remap_case_path(path), output_dir) for path in paths]

    summary = {
        "filelist": args.filelist,
        "output_dir": str(output_dir),
        "counts": {
            status: sum(1 for record in records if record["status"] == status)
            for status in sorted({record["status"] for record in records})
        },
        "records": records,
    }

    summary_path = Path(args.summary) if args.summary else output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0 if summary["counts"].get("visualized", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
