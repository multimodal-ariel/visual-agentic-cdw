"""
Matplotlib-based CDW segmentation slice visualizations (headless / Agg backend).

Key functions
-------------
plot_case_overview(case_path, tool_dir, output_path, ...)
    3-panel axial/coronal/sagittal PNG, pathology-aware slice, organs colored
    by QC severity when available.

plot_disagreement_overlay(case_path, organ, tool_a_dir, tool_b_dir, output_path, ...)
    Single-panel disagreement figure: Tool A = red, Tool B = blue, overlap = green.
    Slice selected at maximum overlap region — shows where tools diverge, not
    where they agree.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from visualizations.utils import (
    SEVERITY_COLORS, TOOL_DIR_MAP,
    build_rgba_overlay, find_overlap_slice, find_pathology_slice,
    get_slice, list_masks, load_image, load_nifti,
    nonzero_bbox, normalize_for_display, organ_color,  # noqa: F401
)

logger = logging.getLogger(__name__)

_BG = "#111111"
_FIG_BG = "#1a1a1a"
_AXIS_LABELS = {0: "Axial (Z)", 1: "Coronal (Y)", 2: "Sagittal (X)"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _worst_severity(qc_severities: Optional[dict[str, str]]) -> str:
    if not qc_severities:
        return "UNKNOWN"
    for sev in ("FAIL", "WARN", "PASS"):
        if sev in qc_severities.values():
            return sev
    return "UNKNOWN"


def _show_panel(ax, bg_slc, rgba, slice_idx: int, label: str) -> None:
    """Render one panel: grayscale background + RGBA overlay."""
    ax.set_facecolor(_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, color="#cccccc", fontsize=9, pad=3)

    if bg_slc is not None:
        ax.imshow(bg_slc, cmap="gray", vmin=0, vmax=1, aspect="auto", origin="lower")

    if rgba is not None:
        if bg_slc is None:
            # Mask-only: black background sized to match mask
            ax.imshow(np.zeros(rgba.shape[:2]), cmap="gray",
                      aspect="auto", origin="lower", vmin=0, vmax=1)
        ax.imshow(rgba, aspect="auto", origin="lower", interpolation="nearest")

    ax.set_xlabel(f"slice {slice_idx}", color="#666666", fontsize=6, labelpad=1)


# ── Public API ────────────────────────────────────────────────────────────────

def plot_case_overview(
    case_path: str | Path,
    tool_dir: str,
    output_path: str | Path,
    qc_severities: Optional[dict[str, str]] = None,
    title_prefix: str = "",
    dpi: int = 150,
) -> Optional[Path]:
    """
    Generate a 3-panel (axial / coronal / sagittal) overview PNG.

    Slice selection strategy:
    - If a FAIL organ is present, the slice is chosen to maximally show its
      fragmentation (non-LCC voxel concentration).
    - If only WARN organs are present, same strategy for the worst WARN organ.
    - Otherwise: midslice of the combined nonzero bounding box.

    Organ colors:
    - qc_severities provided: green=PASS, orange=WARN, red=FAIL
    - qc_severities absent: deterministic per-organ color

    Args:
        case_path:       Case directory (contains image_nifti.nii.gz + seg subdir).
        tool_dir:        Segmentation subdir, e.g. "segmentations_totalseg_ct".
        output_path:     PNG save path.
        qc_severities:   {organ_name: "PASS"/"WARN"/"FAIL"}.
        title_prefix:    Prepended to the figure title.
        dpi:             Output resolution (150 = good thumbnail, 300 = publication).

    Returns:
        Path to saved PNG, or None if no masks found.
    """
    case_path = Path(case_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_paths = list_masks(case_path, tool_dir)
    if not mask_paths:
        logger.warning("No masks in %s / %s", case_path.name, tool_dir)
        return None

    image_vol = load_image(case_path)

    # Load all masks, build combined volume for fallback slice selection
    loaded_masks: dict[str, np.ndarray] = {}
    for name, path in mask_paths.items():
        try:
            arr, _ = load_nifti(path)
            loaded_masks[name] = (arr > 0).astype(np.uint8)
        except Exception as e:
            logger.debug("Could not load %s: %s", name, e)

    if not loaded_masks:
        return None

    combined = np.max(np.stack(list(loaded_masks.values()), axis=0), axis=0)

    # Pick the "worst" organ to drive slice selection
    reference_organ = None
    if qc_severities:
        for sev in ("FAIL", "WARN"):
            candidates = [k for k, v in qc_severities.items() if v == sev and k in loaded_masks]
            if candidates:
                reference_organ = candidates[0]
                break

    slices: dict[int, int] = {}
    for axis in (0, 1, 2):
        ref = loaded_masks.get(reference_organ, combined) if reference_organ else combined
        slices[axis] = find_pathology_slice(ref, axis)

    # Build figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(_FIG_BG)

    overall_sev = _worst_severity(qc_severities)
    sev_color = SEVERITY_COLORS.get(overall_sev, SEVERITY_COLORS["UNKNOWN"])
    tool_label = tool_dir.replace("segmentations_", "").replace("_", " ")
    fig.suptitle(
        f"{title_prefix}{case_path.name}  ·  {tool_label}  ·  {overall_sev}",
        color=(*sev_color, 1.0), fontsize=13, y=1.01,
    )

    all_legend: dict[str, tuple[float, float, float]] = {}

    for col, axis in enumerate((0, 1, 2)):
        s_idx = slices[axis]

        bg_slc = None
        if image_vol is not None:
            raw = get_slice(image_vol, s_idx, axis)
            bg_slc = normalize_for_display(raw)

        rgba, legend = build_rgba_overlay(
            mask_paths, s_idx, axis, qc_severities=qc_severities,
        )
        all_legend.update(legend)
        _show_panel(axes[col], bg_slc, rgba, s_idx, _AXIS_LABELS[axis])

    # Legend (up to 24 organs; truncate long names)
    if all_legend:
        handles = [
            mpatches.Patch(color=c, label=n[:22])
            for n, c in sorted(all_legend.items())[:24]
        ]
        fig.legend(
            handles=handles, loc="lower center",
            ncol=min(6, len(handles)), fontsize=6,
            framealpha=0.3, facecolor="#333333", labelcolor="white",
            bbox_to_anchor=(0.5, -0.06),
        )

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    return output_path


def plot_disagreement_overlay(
    case_path: str | Path,
    organ: str,
    tool_a_dir: str,
    tool_b_dir: str,
    output_path: str | Path,
    dice: Optional[float] = None,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Single-panel disagreement figure for one organ across two tools.

    Color scheme:
    - Tool A only (no overlap):  red,   alpha 0.5
    - Tool B only (no overlap):  blue,  alpha 0.5
    - Overlap (both agree):      green, alpha 0.7
    - Contour outlines drawn for both tools for clean boundary visibility

    Slice: selected at the axial plane with maximum overlap — shows the
    region of disagreement rather than a clean "both agree" midslice.

    Args:
        case_path:    Case directory.
        organ:        Organ name (must exist in both tool dirs).
        tool_a_dir:   First segmentation subdir name.
        tool_b_dir:   Second segmentation subdir name.
        output_path:  PNG save path.
        dice:         Dice score (shown in title if provided).
        dpi:          Output resolution.

    Returns:
        Path to saved PNG, or None if either mask is missing.
    """
    case_path = Path(case_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path_a = case_path / tool_a_dir / f"{organ}.nii.gz"
    path_b = case_path / tool_b_dir / f"{organ}.nii.gz"

    if not path_a.exists() or not path_b.exists():
        logger.warning("Missing mask for %s in %s or %s", organ, tool_a_dir, tool_b_dir)
        return None

    arr_a, _ = load_nifti(path_a)
    arr_b, _ = load_nifti(path_b)
    mask_a = (arr_a > 0).astype(np.uint8)
    mask_b = (arr_b > 0).astype(np.uint8)

    image_vol = load_image(case_path)

    # Find the axial slice — if shapes differ, align on the larger volume's slice
    axis = 0
    shapes_match = (mask_a.shape == mask_b.shape)

    if shapes_match:
        s_idx = find_overlap_slice(mask_a, mask_b, axis)
        slc_a = get_slice(mask_a, s_idx, axis).astype(bool)
        slc_b = get_slice(mask_b, s_idx, axis).astype(bool)
    else:
        # Different voxel spaces — show each independently, aligned by nonzero centroid
        s_idx = find_pathology_slice(mask_a, axis)
        slc_a = get_slice(mask_a, s_idx, axis).astype(bool)
        # For mask_b, find the corresponding slice by centroid fraction
        bb_a = nonzero_bbox(mask_a)
        bb_b = nonzero_bbox(mask_b)
        frac = (s_idx - bb_a[0]) / max(bb_a[1] - bb_a[0], 1)
        s_idx_b = int(bb_b[0] + frac * (bb_b[1] - bb_b[0]))
        s_idx_b = max(0, min(s_idx_b, mask_b.shape[axis] - 1))
        slc_b_raw = get_slice(mask_b, s_idx_b, axis).astype(bool)
        # Pad/crop slc_b to match slc_a shape for overlay
        H_a, W_a = slc_a.shape
        H_b, W_b = slc_b_raw.shape
        slc_b = np.zeros((H_a, W_a), dtype=bool)
        h_min, w_min = min(H_a, H_b), min(W_a, W_b)
        slc_b[:h_min, :w_min] = slc_b_raw[:h_min, :w_min]

    overlap  = slc_a & slc_b
    only_a   = slc_a & ~slc_b
    only_b   = slc_b & ~slc_a

    # Build RGBA
    H, W = slc_a.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[only_a]  = (0.90, 0.10, 0.10, 0.50)   # red   — Tool A only
    rgba[only_b]  = (0.10, 0.20, 0.90, 0.50)   # blue  — Tool B only
    rgba[overlap] = (0.10, 0.85, 0.20, 0.70)   # green — shared

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.patch.set_facecolor(_FIG_BG)
    ax.set_facecolor(_BG)
    ax.set_xticks([])
    ax.set_yticks([])

    if image_vol is not None:
        img_s = min(s_idx, image_vol.shape[axis] - 1)
        bg = normalize_for_display(get_slice(image_vol, img_s, axis))
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, aspect="auto", origin="lower")
    else:
        ax.imshow(np.zeros((H, W)), cmap="gray", aspect="auto", origin="lower", vmin=0, vmax=1)

    ax.imshow(rgba, aspect="auto", origin="lower", interpolation="nearest")

    # Contour outlines for clean boundary display
    if slc_a.any():
        ax.contour(slc_a.astype(float), levels=[0.5], colors=["#ff4444"], linewidths=[1.0])
    if slc_b.any():
        ax.contour(slc_b.astype(float), levels=[0.5], colors=["#4466ff"], linewidths=[1.0])

    tool_a_label = TOOL_DIR_MAP.get(tool_a_dir, tool_a_dir.replace("segmentations_", ""))
    tool_b_label = TOOL_DIR_MAP.get(tool_b_dir, tool_b_dir.replace("segmentations_", ""))
    dice_str = f"  Dice={dice:.3f}" if dice is not None else ""
    ax.set_title(
        f"{case_path.name}  ·  {organ}{dice_str}",
        color="#dddddd", fontsize=10,
    )
    ax.set_xlabel(f"axial slice {s_idx}", color="#666666", fontsize=8)

    handles = [
        mpatches.Patch(color="#ff4444", label=f"{tool_a_label} only", alpha=0.7),
        mpatches.Patch(color="#19bf33", label="Overlap", alpha=0.7),
        mpatches.Patch(color="#4466ff", label=f"{tool_b_label} only", alpha=0.7),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.4, facecolor="#333333", labelcolor="white")

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=_FIG_BG)
    plt.close(fig)
    return output_path
