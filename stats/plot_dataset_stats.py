#!/usr/bin/env python3
"""
Publication-quality dataset statistics plots for CDW imaging pipeline.

Generates four plot types:
  (a) Frequency of images per modality
  (b) Frequency of each anatomy covered
  (c) Distribution of modalities and anatomies across cohorts (RHEUM, CONTR, LUPUS)
  (d) XR bone anatomy breakdown for LUPUS only

Usage:
    python stats/plot_dataset_stats.py data_paths/3d_scans_list.json data_paths/2d_scans_list.json data_paths/dicom_dirs_list.json
    python stats/plot_dataset_stats.py data_paths/*.json --output-dir stats/figures/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Add repo root for imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stats.parse_dataset import parse_multiple_jsons


# ── Style ────────────────────────────────────────────────────────────────────

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

# Cohort-consistent colors
COHORT_COLORS = {
    "RHEUM": "#2196F3",   # blue
    "CONTR": "#4CAF50",   # green
    "LUPUS": "#F44336",   # red
}

# Modality color palette (categorical, colorblind-safe)
MODALITY_COLORS = {
    "CT":     "#1f77b4",
    "MRI":    "#ff7f0e",
    "XR":     "#2ca02c",
    "MAMMO":  "#d62728",
    "US":     "#9467bd",
    "NM":     "#8c564b",
    "PET_CT": "#e377c2",
    "CTA":    "#7f7f7f",
    "MRA":    "#bcbd22",
    "FL":     "#17becf",
    "IR":     "#aec7e8",
    "DEXA":   "#ffbb78",
    "OTHER":  "#c7c7c7",
}

ANATOMY_COLORS = {
    "chest":                 "#1f77b4",
    "abdomen":               "#ff7f0e",
    "abdomen_pelvis":        "#2ca02c",
    "chest_abdomen_pelvis":  "#d62728",
    "head":                  "#9467bd",
    "extremity":             "#8c564b",
    "spine":                 "#e377c2",
    "pelvis":                "#7f7f7f",
    "breast":                "#bcbd22",
    "cardiac":               "#17becf",
    "whole_body":            "#aec7e8",
    "neck":                  "#ffbb78",
    "vascular":              "#98df8a",
    "OTHER":                 "#c7c7c7",
}


def _bar_value_labels(ax, fontsize=8, fmt="{:,}"):
    """Add count labels on top of bars."""
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                fmt.format(int(h)),
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold",
            )


def _clean_label(s: str) -> str:
    return s.replace("_", " ").title()


# ── Plot A: Modality frequency ──────────────────────────────────────────────

def plot_modality_frequency(df: pd.DataFrame, output_dir: str):
    """Bar chart of image count per modality."""
    counts = df["modality"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [MODALITY_COLORS.get(m, "#c7c7c7") for m in counts.index]
    bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_title("Image Frequency by Modality")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add percentage labels
    total = counts.sum()
    for i, (bar, val) in enumerate(zip(bars, counts.values)):
        pct = val / total * 100
        label = f"{val:,}\n({pct:.1f}%)"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, val),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()

    path = os.path.join(output_dir, "a_modality_frequency.png")
    fig.savefig(path)
    path_pdf = os.path.join(output_dir, "a_modality_frequency.pdf")
    fig.savefig(path_pdf)
    plt.close(fig)
    print(f"  [A] Saved: {path}")


# ── Plot B: Anatomy frequency ───────────────────────────────────────────────

def plot_anatomy_frequency(df: pd.DataFrame, output_dir: str):
    """Pie chart of anatomy categories."""
    counts = df["anatomy"].value_counts().sort_values(ascending=False)
    total = counts.sum()

    # Group tiny slices (<2%) into "Other" to keep pie readable
    threshold = 0.02
    main = counts[counts / total >= threshold]
    small = counts[counts / total < threshold]
    if len(small) > 0:
        # Merge small slices into existing OTHER or create one
        other_val = small.sum()
        if "OTHER" in main.index:
            main = main.copy()
            main["OTHER"] += other_val
        else:
            main = pd.concat([main, pd.Series({"OTHER": other_val})])

    labels = [_clean_label(a) for a in main.index]
    colors = [ANATOMY_COLORS.get(a, "#c7c7c7") for a in main.index]

    fig, ax = plt.subplots(figsize=(9, 7))

    wedges, texts, autotexts = ax.pie(
        main.values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * total)):,})",
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )

    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(10)

    ax.set_title("Image Frequency by Anatomy", pad=20)

    fig.tight_layout()

    path = os.path.join(output_dir, "b_anatomy_frequency.png")
    fig.savefig(path)
    fig.savefig(os.path.join(output_dir, "b_anatomy_frequency.pdf"))
    plt.close(fig)
    print(f"  [B] Saved: {path}")


# ── Plot C: Cohort distribution (modality + anatomy) ────────────────────────

def plot_cohort_distribution(df: pd.DataFrame, output_dir: str):
    """Side-by-side grouped bar charts: modality and anatomy by cohort."""
    cohorts = sorted(df["cohort"].unique())
    n_cohorts = len(cohorts)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # C1: Modality by cohort
    ax = axes[0]
    modality_order = df["modality"].value_counts().index.tolist()
    pivot_mod = df.groupby(["modality", "cohort"]).size().unstack(fill_value=0)
    pivot_mod = pivot_mod.reindex(modality_order)
    # Ensure all cohorts present
    for c in cohorts:
        if c not in pivot_mod.columns:
            pivot_mod[c] = 0
    pivot_mod = pivot_mod[sorted(pivot_mod.columns)]

    x = np.arange(len(modality_order))
    width = 0.8 / n_cohorts
    for i, cohort in enumerate(sorted(pivot_mod.columns)):
        color = COHORT_COLORS.get(cohort, "#999999")
        ax.bar(x + i * width - 0.4 + width / 2, pivot_mod[cohort].values,
               width, label=cohort, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(modality_order, rotation=45, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_title("Modality Distribution by Cohort")
    ax.legend(title="Cohort", frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # C2: Anatomy by cohort
    ax = axes[1]
    anatomy_order = df["anatomy"].value_counts().index.tolist()
    pivot_anat = df.groupby(["anatomy", "cohort"]).size().unstack(fill_value=0)
    pivot_anat = pivot_anat.reindex(anatomy_order)
    for c in cohorts:
        if c not in pivot_anat.columns:
            pivot_anat[c] = 0
    pivot_anat = pivot_anat[sorted(pivot_anat.columns)]

    x = np.arange(len(anatomy_order))
    width = 0.8 / n_cohorts
    for i, cohort in enumerate(sorted(pivot_anat.columns)):
        color = COHORT_COLORS.get(cohort, "#999999")
        ax.bar(x + i * width - 0.4 + width / 2, pivot_anat[cohort].values,
               width, label=cohort, color=color, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([_clean_label(a) for a in anatomy_order], rotation=45, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_title("Anatomy Distribution by Cohort")
    ax.legend(title="Cohort", frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    path = os.path.join(output_dir, "c_cohort_distribution.png")
    fig.savefig(path)
    fig.savefig(os.path.join(output_dir, "c_cohort_distribution.pdf"))
    plt.close(fig)
    print(f"  [C] Saved: {path}")


# ── Plot D: LUPUS XR bone anatomies ─────────────────────────────────────────

def plot_lupus_xr_bone(df: pd.DataFrame, output_dir: str):
    """Bar chart of XR bone anatomy detail for LUPUS cohort only."""
    lupus_xr = df[(df["cohort"] == "LUPUS") & (df["modality"] == "XR")]

    if lupus_xr.empty:
        print("  [D] SKIPPED: No LUPUS XR data found")
        return

    # Filter to musculoskeletal anatomies (exclude chest, abdomen, etc.)
    bone_anatomies = {"extremity", "spine", "pelvis"}
    lupus_bone = lupus_xr[lupus_xr["anatomy"].isin(bone_anatomies)]

    if lupus_bone.empty:
        # Fall back to all XR anatomies
        lupus_bone = lupus_xr

    counts = lupus_bone["anatomy_detail"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.35 + 1)))

    cmap = plt.cm.Set2
    colors = [cmap(i / max(len(counts) - 1, 1)) for i in range(len(counts))]

    bars = ax.barh(range(len(counts)), counts.values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([_clean_label(a) for a in counts.index])
    ax.set_xlabel("Number of Images")
    ax.set_title("LUPUS Cohort: XR Bone Anatomy Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.annotate(f" {val:,} ({pct:.1f}%)",
                    (val, bar.get_y() + bar.get_height() / 2),
                    va="center", fontsize=9)

    ax.set_xlim(0, counts.max() * 1.3)

    # Subtitle with totals
    ax.text(0.98, 0.02,
            f"Total LUPUS XR bone images: {total:,}\nTotal LUPUS XR images: {len(lupus_xr):,}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, style="italic", color="#555555")

    fig.tight_layout()

    path = os.path.join(output_dir, "d_lupus_xr_bone_anatomy.png")
    fig.savefig(path)
    fig.savefig(os.path.join(output_dir, "d_lupus_xr_bone_anatomy.pdf"))
    plt.close(fig)
    print(f"  [D] Saved: {path}")


# ── Bonus: Summary table ────────────────────────────────────────────────────

def save_summary_table(df: pd.DataFrame, output_dir: str):
    """Save a cross-tabulation of cohort x modality as CSV and a formatted PNG."""
    # Cohort x Modality
    cross = pd.crosstab(df["modality"], df["cohort"], margins=True, margins_name="Total")
    cross = cross.sort_values("Total", ascending=False)
    csv_path = os.path.join(output_dir, "summary_cohort_modality.csv")
    cross.to_csv(csv_path)

    # Cohort x Anatomy
    cross_anat = pd.crosstab(df["anatomy"], df["cohort"], margins=True, margins_name="Total")
    cross_anat = cross_anat.sort_values("Total", ascending=False)
    csv_path_anat = os.path.join(output_dir, "summary_cohort_anatomy.csv")
    cross_anat.to_csv(csv_path_anat)

    print(f"  [S] Summary tables: {csv_path}, {csv_path_anat}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality dataset statistics plots"
    )
    parser.add_argument("inputs", nargs="+", help="JSON files with DICOM paths")
    parser.add_argument(
        "--output-dir", default=os.path.join(REPO_ROOT, "stats", "figures"),
        help="Directory for output plots (default: stats/figures/)",
    )
    args = parser.parse_args()

    for inp in args.inputs:
        if not os.path.isfile(inp):
            print(f"ERROR: File not found: {inp}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Parsing dataset paths...")
    df = parse_multiple_jsons(args.inputs)
    print(f"  Total entries: {len(df):,}")
    print(f"  Unique patients: {df['patient_id'].nunique():,}")
    print(f"  Cohorts: {sorted(df['cohort'].unique())}")
    print(f"  Modalities: {sorted(df['modality'].unique())}")
    print()

    print("Generating plots...")
    plot_modality_frequency(df, args.output_dir)
    plot_anatomy_frequency(df, args.output_dir)
    plot_cohort_distribution(df, args.output_dir)
    plot_lupus_xr_bone(df, args.output_dir)
    save_summary_table(df, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
