#!/usr/bin/env python3
"""
Segmentation feature distribution plots across patients, per modality.

Parses the pipeline output directory structure:
    <root>/<patient_dir>/
        image_nifti.nii.gz
        segmentations_<tool>/
            <organ>.nii.gz
            manifest.json

Computes per-organ features (volume in mL, mean intensity) from each
segmentation tool, then generates box + strip plots faceted by modality
(CT vs MRI) to reveal distributions and outliers.

Usage:
    python stats/plot_segmentation_distributions.py /path/to/pipeline_outputs/
    python stats/plot_segmentation_distributions.py /path/to/pipeline_outputs/ --output-dir stats/figures/
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    sys.exit("ERROR: nibabel is required. Install with: pip install nibabel")

# ── Configuration ─────────────────────────────────────────────────────────────

# Primary organs of interest (canonical names).
# We match against these using normalized versions of the filenames.
PRIMARY_ORGANS = ["kidney_left", "kidney_right", "liver", "pancreas", "gallbladder", "spleen"]

# Aliases: map common filename variants → canonical organ name
ORGAN_ALIASES = {
    # kidneys
    "left_kidney":        "kidney_left",
    "kidney_left":        "kidney_left",
    "right_kidney":       "kidney_right",
    "kidney_right":       "kidney_right",
    # liver
    "liver":              "liver",
    # pancreas
    "pancreas":           "pancreas",
    # gallbladder
    "gallbladder":        "gallbladder",
    "gall_bladder":       "gallbladder",
    # spleen
    "spleen":             "spleen",
}

# Display-friendly organ labels
ORGAN_DISPLAY = {
    "kidney_left":  "Left Kidney",
    "kidney_right": "Right Kidney",
    "liver":        "Liver",
    "pancreas":     "Pancreas",
    "gallbladder":  "Gallbladder",
    "spleen":       "Spleen",
}

# Tool color palette (distinguishable, colorblind-friendly)
TOOL_COLORS = {
    "vista3d":       "#1f77b4",
    "totalseg_ct":   "#ff7f0e",
    "totalseg_mr":   "#ffbb78",
    "mrseg":         "#2ca02c",
    "voxtell":       "#d62728",
    "consensus":     "#9467bd",
}
DEFAULT_TOOL_COLOR = "#8c564b"

# Tools that are invalid for a given modality — filter these out
TOOL_MODALITY_EXCLUDE = {
    "CT":  {"totalseg_mr", "mrseg"},       # MR-only tools don't belong in CT
    "MRI": {"totalseg_ct"},                 # CT-only tool doesn't belong in MRI
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ── Parsing ───────────────────────────────────────────────────────────────────

def extract_modality_from_dirname(dirname: str) -> str:
    """Extract modality (CT or MRI) from patient directory name.

    Expected pattern: ..._{COHORT}_{ID}_{DATE}_{MODALITY}_...
    e.g. data_soumitri_segmentations_3d_LUPUS_<hash>_20221103_CT_ABDOMEN_...
    """
    # Look for _CT_ or _MRI_ or _MR_ in the directory name
    upper = dirname.upper()
    if "_MRI_" in upper or "_MR_" in upper:
        return "MRI"
    if "_CT_" in upper:
        return "CT"
    return "UNKNOWN"


def extract_patient_id_from_dirname(dirname: str) -> str:
    """Extract a short patient identifier from the directory name."""
    # Try to grab the hash-like ID portion
    # Pattern: ..._LUPUS_<HASH>_<DATE>_...  or  ..._CONTR_<HASH>_<DATE>_...
    match = re.search(r'_(LUPUS|RHEUM|CONTR)_([A-F0-9]{8,})', dirname, re.IGNORECASE)
    if match:
        return match.group(2)[:12]  # truncate for readability
    # Fallback: use last 12 chars
    return dirname[-12:]


def normalize_organ_name(filename: str) -> str | None:
    """Map a segmentation filename to a canonical organ name, or None if not primary."""
    stem = Path(filename).stem
    # Remove .nii if double-extension (.nii.gz)
    if stem.endswith(".nii"):
        stem = stem[:-4]
    stem = stem.lower().strip()
    return ORGAN_ALIASES.get(stem)


def extract_tool_name(seg_dirname: str) -> str:
    """Extract tool name from segmentation subdirectory name like 'segmentations_vista3d'."""
    prefix = "segmentations_"
    if seg_dirname.startswith(prefix):
        return seg_dirname[len(prefix):]
    return seg_dirname


def compute_volume_ml(mask_data: np.ndarray, voxel_spacing_mm: tuple) -> float:
    """Compute volume in mL from a binary mask and voxel spacing (mm)."""
    voxel_vol_mm3 = float(np.prod(voxel_spacing_mm))
    n_voxels = int(np.count_nonzero(mask_data))
    return n_voxels * voxel_vol_mm3 / 1000.0  # mm³ → mL


def compute_mean_intensity(image_data: np.ndarray, mask_data: np.ndarray) -> float | None:
    """Compute mean intensity of the image within the mask region."""
    mask_bool = mask_data > 0
    if not np.any(mask_bool):
        return None
    return float(np.mean(image_data[mask_bool]))


def scan_patient_dir(patient_dir: str, compute_intensity: bool = True) -> list[dict]:
    """Scan one patient directory and return feature rows for primary organs."""
    patient_path = Path(patient_dir)
    dirname = patient_path.name

    modality = extract_modality_from_dirname(dirname)
    patient_id = extract_patient_id_from_dirname(dirname)

    # Load the original image (for intensity stats)
    image_path = patient_path / "image_nifti.nii.gz"
    image_data = None
    if compute_intensity and image_path.exists():
        try:
            img_nii = nib.load(str(image_path))
            image_data = img_nii.get_fdata()
        except Exception as e:
            print(f"  WARN: Could not load image {image_path}: {e}")

    rows = []

    # Iterate over segmentation tool subdirectories
    for seg_subdir in sorted(patient_path.iterdir()):
        if not seg_subdir.is_dir():
            continue
        if not seg_subdir.name.startswith("segmentations_"):
            continue

        tool = extract_tool_name(seg_subdir.name)

        # Iterate over organ masks
        for mask_file in sorted(seg_subdir.glob("*.nii.gz")):
            organ = normalize_organ_name(mask_file.name)
            if organ is None:
                continue  # not a primary organ

            try:
                mask_nii = nib.load(str(mask_file))
                mask_data = mask_nii.get_fdata()
                spacing = mask_nii.header.get_zooms()[:3]
            except Exception as e:
                print(f"  WARN: Could not load mask {mask_file}: {e}")
                continue

            volume_ml = compute_volume_ml(mask_data, spacing)

            mean_intensity = None
            if image_data is not None:
                mean_intensity = compute_mean_intensity(image_data, mask_data)

            rows.append({
                "patient_id": patient_id,
                "modality": modality,
                "tool": tool,
                "organ": organ,
                "organ_display": ORGAN_DISPLAY.get(organ, organ),
                "volume_ml": volume_ml,
                "mean_intensity": mean_intensity,
                "patient_dir": dirname,
            })

    return rows


def scan_all_patients(root_dir: str, compute_intensity: bool = True) -> pd.DataFrame:
    """Scan all patient directories under root_dir and return a feature DataFrame."""
    root = Path(root_dir)
    patient_dirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("data_") and "segmentations_3d" in d.name
    ])

    if not patient_dirs:
        sys.exit(f"ERROR: No patient directories found under {root_dir}")

    print(f"Found {len(patient_dirs)} patient directories")

    all_rows = []
    for i, pdir in enumerate(patient_dirs):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(patient_dirs)}: {pdir.name[:80]}...")
        rows = scan_patient_dir(str(pdir), compute_intensity=compute_intensity)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"  Total feature rows: {len(df)}")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def _filter_tools_for_modality(df: pd.DataFrame) -> pd.DataFrame:
    """Remove modality-inappropriate tools (e.g. totalseg_mr from CT rows)."""
    masks = []
    for modality, excluded in TOOL_MODALITY_EXCLUDE.items():
        masks.append((df["modality"] == modality) & (df["tool"].isin(excluded)))
    if masks:
        drop_mask = masks[0]
        for m in masks[1:]:
            drop_mask = drop_mask | m
        df = df[~drop_mask].copy()
    return df


def _add_jitter(values: np.ndarray, jitter_amount: float = 0.15) -> np.ndarray:
    """Add horizontal jitter to values for strip plot overlay."""
    rng = np.random.default_rng(42)
    return values + rng.uniform(-jitter_amount, jitter_amount, size=len(values))


def _draw_box_strip_on_ax(ax, df_subset: pd.DataFrame, tools: list, value_col: str):
    """Draw grouped box + strip plot for one organ on a single Axes.

    X-axis positions = tools (one box per tool).
    """
    n_tools = len(tools)
    for t_idx, tool in enumerate(tools):
        df_tool = df_subset[df_subset["tool"] == tool]
        color = TOOL_COLORS.get(tool, DEFAULT_TOOL_COLOR)
        vals = df_tool[value_col].dropna().values

        # Box
        if len(vals) > 0:
            bp = ax.boxplot(
                [vals],
                positions=[t_idx],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color=color, linewidth=1),
                capprops=dict(color=color, linewidth=1),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
                patch.set_edgecolor(color)

        # Strip
        if len(vals) > 0:
            x_jittered = _add_jitter(np.full(len(vals), t_idx), jitter_amount=0.15)
            ax.scatter(
                x_jittered, vals,
                c=color, s=22, alpha=0.7, edgecolors="white",
                linewidths=0.3, zorder=3,
            )

    ax.set_xticks(range(n_tools))
    ax.set_xticklabels(tools, rotation=40, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_violin_strip_on_ax(ax, df_subset: pd.DataFrame, tools: list, value_col: str):
    """Draw grouped violin + strip plot for one organ on a single Axes."""
    n_tools = len(tools)
    for t_idx, tool in enumerate(tools):
        df_tool = df_subset[df_subset["tool"] == tool]
        color = TOOL_COLORS.get(tool, DEFAULT_TOOL_COLOR)
        vals = df_tool[value_col].dropna().values

        # Violin (needs >=2 points)
        if len(vals) >= 2:
            vp = ax.violinplot(
                [vals],
                positions=[t_idx],
                widths=0.65,
                showmedians=True,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.3)
                body.set_edgecolor(color)
            vp["cmedians"].set_color(color)
            vp["cmedians"].set_linewidth(2)

        # Strip
        if len(vals) > 0:
            x_jittered = _add_jitter(np.full(len(vals), t_idx), jitter_amount=0.12)
            ax.scatter(
                x_jittered, vals,
                c=color, s=22, alpha=0.7, edgecolors="white",
                linewidths=0.3, zorder=3,
            )

    ax.set_xticks(range(n_tools))
    ax.set_xticklabels(tools, rotation=40, ha="right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _organ_grid_plot(df: pd.DataFrame, output_dir: str, value_col: str,
                     ylabel: str, suptitle: str, filename_stem: str,
                     draw_fn=_draw_box_strip_on_ax):
    """
    Generic grid plot: rows = modalities (CT, MRI), cols = organs.
    Each subplot has its own y-scale so small organs aren't dwarfed by large ones.
    """
    modalities = sorted(df["modality"].unique())
    modalities = [m for m in modalities if m != "UNKNOWN"]
    if not modalities:
        print(f"  SKIP ({filename_stem}): No CT/MRI data found")
        return

    organs = [o for o in PRIMARY_ORGANS if o in df["organ"].unique()]
    n_rows = len(modalities)
    n_cols = len(organs)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 5 * n_rows),
                             squeeze=False)

    for row, modality in enumerate(modalities):
        df_mod = df[df["modality"] == modality]
        # Get tools present for this modality (after filtering)
        tools = sorted(df_mod["tool"].unique())

        for col, organ in enumerate(organs):
            ax = axes[row, col]
            df_cell = df_mod[df_mod["organ"] == organ]

            if df_cell.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#999")
                ax.set_xticks([])
            else:
                draw_fn(ax, df_cell, tools, value_col)

            # Title: organ name on top row only
            if row == 0:
                ax.set_title(ORGAN_DISPLAY.get(organ, organ), fontsize=13, fontweight="bold")

            # Y-label on leftmost column only
            if col == 0:
                ax.set_ylabel(f"{modality}\n{ylabel}", fontsize=11)
            else:
                ax.set_ylabel("")

            # Format y-axis
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Add a shared legend using tool colors
    all_tools = sorted(df["tool"].unique())
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=TOOL_COLORS.get(t, DEFAULT_TOOL_COLOR),
                    markersize=8, label=t)
        for t in all_tools
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(all_tools), frameon=False, fontsize=10,
               title="Segmentation Tool", title_fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{filename_stem}.png")
    fig.savefig(path)
    fig.savefig(os.path.join(output_dir, f"{filename_stem}.pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_volume_distributions(df: pd.DataFrame, output_dir: str):
    """Box + strip grid: rows=modality, cols=organ, y=volume."""
    _organ_grid_plot(
        df, output_dir,
        value_col="volume_ml",
        ylabel="Volume (mL)",
        suptitle="Organ Volume Distributions by Modality and Tool",
        filename_stem="seg_volume_distribution",
        draw_fn=_draw_box_strip_on_ax,
    )


def plot_intensity_distributions(df: pd.DataFrame, output_dir: str):
    """Box + strip grid: rows=modality, cols=organ, y=mean intensity."""
    df_int = df.dropna(subset=["mean_intensity"])
    if df_int.empty:
        print("  SKIP: No intensity data available")
        return
    _organ_grid_plot(
        df_int, output_dir,
        value_col="mean_intensity",
        ylabel="Mean Intensity (HU / a.u.)",
        suptitle="Mean Intensity Distributions Within Organ Masks",
        filename_stem="seg_intensity_distribution",
        draw_fn=_draw_box_strip_on_ax,
    )


def plot_volume_violin(df: pd.DataFrame, output_dir: str):
    """Violin + strip grid: rows=modality, cols=organ, y=volume."""
    _organ_grid_plot(
        df, output_dir,
        value_col="volume_ml",
        ylabel="Volume (mL)",
        suptitle="Organ Volume Distributions (Violin) by Modality and Tool",
        filename_stem="seg_volume_violin",
        draw_fn=_draw_violin_strip_on_ax,
    )


def save_feature_csv(df: pd.DataFrame, output_dir: str):
    """Save the raw feature table for further analysis."""
    path = os.path.join(output_dir, "seg_organ_features.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def print_outlier_summary(df: pd.DataFrame):
    """Print potential outliers (beyond 1.5×IQR) per modality/organ/tool."""
    print("\n── Potential Outliers (beyond 1.5×IQR) ──")
    for modality in sorted(df["modality"].unique()):
        df_mod = df[df["modality"] == modality]
        for organ in sorted(df_mod["organ"].unique()):
            df_org = df_mod[df_mod["organ"] == organ]
            for tool in sorted(df_org["tool"].unique()):
                vals = df_org[df_org["tool"] == tool]["volume_ml"]
                if len(vals) < 4:
                    continue
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = vals[(vals < lo) | (vals > hi)]
                if len(outliers) > 0:
                    print(f"  {modality} | {ORGAN_DISPLAY.get(organ, organ):15s} | {tool:15s} | "
                          f"{len(outliers)} outlier(s): "
                          f"range [{vals.min():.1f}, {vals.max():.1f}] mL, "
                          f"IQR [{q1:.1f}, {q3:.1f}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot segmentation feature distributions across patients"
    )
    parser.add_argument("root_dir",
                        help="Root directory containing patient output folders")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output plots (default: <root_dir>/stats_figures/)")
    parser.add_argument("--no-intensity", action="store_true",
                        help="Skip intensity computation (faster, volume-only)")
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        sys.exit(f"ERROR: Not a directory: {args.root_dir}")

    output_dir = args.output_dir or os.path.join(args.root_dir, "stats_figures")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning: {args.root_dir}")
    df = scan_all_patients(args.root_dir, compute_intensity=not args.no_intensity)

    if df.empty:
        sys.exit("ERROR: No primary organ segmentations found")

    # Filter out modality-inappropriate tools (e.g. totalseg_mr from CT)
    n_before = len(df)
    df = _filter_tools_for_modality(df)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows from modality-mismatched tools")

    print(f"\nModalities: {sorted(df['modality'].unique())}")
    print(f"Tools:      {sorted(df['tool'].unique())}")
    print(f"Organs:     {sorted(df['organ'].unique())}")
    print(f"Patients:   {df['patient_id'].nunique()}")
    print()

    save_feature_csv(df, output_dir)
    print("\nGenerating plots...")
    plot_volume_distributions(df, output_dir)
    plot_intensity_distributions(df, output_dir)
    plot_volume_violin(df, output_dir)
    print_outlier_summary(df)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
