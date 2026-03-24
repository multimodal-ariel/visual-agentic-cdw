"""
Shared utilities: NIfTI loading, mask discovery, color assignment, slice selection.
"""

from __future__ import annotations

import colorsys
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Tool directory ↔ display name ─────────────────────────────────────────────

TOOL_DIR_MAP: dict[str, str] = {
    "segmentations_totalseg_ct":   "TotalSeg_CT",
    "segmentations_totalseg_mr":   "TotalSeg_MR",
    "segmentations_mrseg":         "MRSegmentator",
    "segmentations_mrisegmenter":  "MRISegmenter",
    "segmentations_vista3d":       "VISTA3D",
    "segmentations_voxtell":       "VoxTell",
    "segmentations_vibeseg":       "VIBESeg",
    "segmentations_textmedseg3d":  "TextMedSeg3D",
}

SEVERITY_RANK = {"FAIL": 0, "WARN": 1, "PASS": 2, "UNKNOWN": 3}

SEVERITY_COLORS: dict[str, tuple[float, float, float]] = {
    "FAIL":    (0.90, 0.15, 0.10),
    "WARN":    (1.00, 0.60, 0.00),
    "PASS":    (0.10, 0.75, 0.20),
    "UNKNOWN": (0.55, 0.55, 0.55),
}

SEVERITY_HEX = {
    "FAIL":    "#e82619",
    "WARN":    "#ff9900",
    "PASS":    "#19bf33",
    "UNKNOWN": "#888888",
}


# ── NIfTI I/O ─────────────────────────────────────────────────────────────────

def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NIfTI, reorient to RAS canonical, return (float32 data, affine)."""
    import nibabel as nib
    img = nib.load(str(path))
    canonical = nib.as_closest_canonical(img)
    return canonical.get_fdata(dtype=np.float32), canonical.affine


def load_image(case_path: str | Path) -> Optional[np.ndarray]:
    """Load image_nifti.nii.gz if present, else None."""
    p = Path(case_path) / "image_nifti.nii.gz"
    if not p.exists():
        return None
    data, _ = load_nifti(p)
    return data


# ── Mask discovery ────────────────────────────────────────────────────────────

def list_tool_dirs(case_path: str | Path) -> dict[str, Path]:
    """Return {display_name: dir_path} for all non-empty seg dirs in a case."""
    case_path = Path(case_path)
    result = {}
    for d in sorted(case_path.iterdir()):
        if d.is_dir() and d.name.startswith("segmentations_") and any(d.glob("*.nii.gz")):
            result[TOOL_DIR_MAP.get(d.name, d.name)] = d
    return result


def list_masks(case_path: str | Path, tool_dir: Optional[str] = None) -> dict:
    """
    Discover per-organ NIfTI masks.

    Returns:
        tool_dir given  → {organ_name: Path}
        tool_dir None   → {display_name: {organ_name: Path}}
    """
    case_path = Path(case_path)

    def _in(d: Path) -> dict[str, Path]:
        return {
            p.name.removesuffix(".nii.gz").removesuffix(".nii"): p
            for p in sorted(d.glob("*.nii.gz"))
        }

    if tool_dir is not None:
        seg_dir = case_path / tool_dir
        return _in(seg_dir) if seg_dir.exists() else {}

    return {disp: _in(d) for disp, d in list_tool_dirs(case_path).items()}


# ── Colors ────────────────────────────────────────────────────────────────────

def organ_color(name: str) -> tuple[float, float, float]:
    """Deterministic RGB for an organ name — same name → same color always."""
    hue = int(hashlib.md5(name.lower().encode()).hexdigest()[:6], 16) / 0xFFFFFF
    return colorsys.hsv_to_rgb(hue, 0.85, 0.95)


def organ_color_map(names: list[str]) -> dict[str, tuple[float, float, float]]:
    return {n: organ_color(n) for n in names}


# ── Array helpers ─────────────────────────────────────────────────────────────

def get_slice(arr: np.ndarray, idx: int, axis: int) -> np.ndarray:
    """Extract 2D slice from a 3D array along the given axis."""
    if axis == 0:
        return arr[idx, :, :]
    if axis == 1:
        return arr[:, idx, :]
    return arr[:, :, idx]


def normalize_for_display(slc: np.ndarray) -> np.ndarray:
    """Clip to [1st, 99th] percentile and scale to [0, 1]."""
    lo, hi = np.percentile(slc, [1, 99])
    if hi <= lo:
        return np.zeros_like(slc, dtype=np.float32)
    return np.clip((slc - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def nonzero_bbox(arr: np.ndarray) -> tuple[int, int, int, int, int, int]:
    """(z0, z1, y0, y1, x0, x1) bounding box of nonzero voxels."""
    nz = np.nonzero(arr)
    if len(nz[0]) == 0:
        s = arr.shape
        return 0, s[0] - 1, 0, s[1] - 1, 0, s[2] - 1
    return (int(nz[0].min()), int(nz[0].max()),
            int(nz[1].min()), int(nz[1].max()),
            int(nz[2].min()), int(nz[2].max()))


# ── Pathology-aware slice selection ──────────────────────────────────────────

def find_pathology_slice(mask_arr: np.ndarray, axis: int = 0) -> int:
    """
    Return the slice index that best reveals a segmentation quality problem.

    Logic:
    - >1 CC:  pick the slice where non-LCC fragment voxels are densest.
              This shows fragmentation directly rather than the clean midslice.
    - 1 CC:   midslice of the nonzero bounding box.
    - Empty:  volume midpoint.
    """
    from scipy import ndimage

    mask = (mask_arr > 0).astype(np.uint8)
    if not mask.any():
        return mask_arr.shape[axis] // 2

    labeled, n_cc = ndimage.label(mask)
    if n_cc > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        lcc = int(counts.argmax())
        non_lcc = mask & (labeled != lcc)
        if non_lcc.any():
            sum_axes = tuple(i for i in range(3) if i != axis)
            return int(non_lcc.sum(axis=sum_axes).argmax())

    bb = nonzero_bbox(mask)
    return int((bb[axis * 2] + bb[axis * 2 + 1]) // 2)


def find_overlap_slice(a: np.ndarray, b: np.ndarray, axis: int = 0) -> int:
    """
    Slice with the most voxel overlap between two masks.

    If shapes differ (tools in different voxel spaces), falls back to the
    pathology slice of mask_a.
    """
    if a.shape != b.shape:
        return find_pathology_slice(a, axis)
    overlap = (a > 0) & (b > 0)
    if not overlap.any():
        return find_pathology_slice(a, axis)
    sum_axes = tuple(i for i in range(3) if i != axis)
    return int(overlap.sum(axis=sum_axes).argmax())


# ── RGBA overlay builder ──────────────────────────────────────────────────────

def build_rgba_overlay(
    mask_paths: dict[str, "Path | np.ndarray"],
    slice_idx: int,
    axis: int = 0,
    qc_severities: Optional[dict[str, str]] = None,
    alpha: float = 0.55,
) -> tuple[Optional[np.ndarray], dict[str, tuple[float, float, float]]]:
    """
    Build an [H, W, 4] RGBA float32 overlay for one 2D slice.

    Organs are colored by name (deterministic); if qc_severities is given,
    color reflects severity (green/orange/red) instead of organ color.

    Returns:
        (rgba, legend_colors) — legend_colors maps organ_name → (r, g, b).
    """
    loaded: dict[str, np.ndarray] = {}
    for name, src in mask_paths.items():
        try:
            arr = src if isinstance(src, np.ndarray) else load_nifti(src)[0]
            slc = get_slice(arr > 0, slice_idx, axis)
            if slc.any():
                loaded[name] = slc.astype(bool)
        except Exception as e:
            logger.debug("Skipping mask %s: %s", name, e)

    if not loaded:
        return None, {}

    H, W = next(iter(loaded.values())).shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    legend: dict[str, tuple[float, float, float]] = {}

    for name, slc in loaded.items():
        if qc_severities and name in qc_severities:
            r, g, b = SEVERITY_COLORS.get(qc_severities[name], SEVERITY_COLORS["UNKNOWN"])
        else:
            r, g, b = organ_color(name)
        legend[name] = (r, g, b)
        rgba[slc, :3] = (r, g, b)
        rgba[slc, 3] = alpha

    return rgba, legend
