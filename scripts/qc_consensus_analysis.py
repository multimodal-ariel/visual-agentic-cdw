from __future__ import annotations

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import binary_erosion

# Optional radiomics
try:
    import SimpleITK as sitk
    from radiomics import featureextractor
    PYRADIOMICS_AVAILABLE = True
except Exception:
    PYRADIOMICS_AVAILABLE = False


# =============================================================================
# Config
# =============================================================================

KNOWN_MODEL_DIRS = {
    "segmentations_mrseg": "mrseg",
    "segmentations_totalseg_ct": "totalseg_ct",
    "segmentations_totalseg_mr": "totalseg_mr",
    "segmentations_voxtell": "voxtell",
    "segmentations_mrisegmenter": "mrisegmenter",
    "segmentations_vibeseg": "vibeseg",
    "segmentations_vista3d": "vista3d",
}

DEFAULT_ORGANS = [
    "liver",
    "spleen",
    "kidney_left",
    "kidney_right",
    "pancreas",
]

DEFAULT_RADIOMICS_FEATURES = [
    "firstorder_Mean",
    "firstorder_Entropy",
    "firstorder_StandardDeviation",
    "glcm_Contrast",
    "glcm_Homogeneity1",
]


# =============================================================================
# General utilities
# =============================================================================

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def list_case_dirs(population_root: str | Path) -> List[Path]:
    root = Path(population_root)
    return sorted([p for p in root.iterdir() if p.is_dir()])


def find_volume_path(case_dir: Path) -> Optional[Path]:
    for candidate in ["volume.nii.gz", "image_nifti.nii.gz"]:
        p = case_dir / candidate
        if p.exists():
            return p
    return None


def load_nifti(path: str | Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    nii = nib.load(str(path))
    data = np.asarray(nii.dataobj)
    return data, nii


def load_mask(path: str | Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    data, nii = load_nifti(path)
    mask = data > 0
    return mask.astype(np.uint8), nii


def get_spacing_hwd(nii: nib.Nifti1Image) -> Tuple[float, float, float]:
    return tuple(float(x) for x in nii.header.get_zooms()[:3])


def compute_volume_ml(mask: np.ndarray, spacing_hwd: Tuple[float, float, float]) -> float:
    voxel_vol_mm3 = float(np.prod(spacing_hwd))
    return float(np.sum(mask > 0) * voxel_vol_mm3 / 1000.0)


def compute_entropy_from_probability_map(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(prob.astype(np.float32), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def compute_soft_uncertainty_stats(prob: np.ndarray) -> Dict[str, float]:
    prob = prob.astype(np.float32)

    core = prob >= 0.75
    shell = (prob >= 0.25) & (prob < 0.75)
    support = prob > 0.0
    uncertain = (prob > 0.25) & (prob < 0.75)

    entropy_map = compute_entropy_from_probability_map(prob)
    entropy_support = entropy_map[support] if np.any(support) else np.array([], dtype=np.float32)

    core_vol = int(np.sum(core))
    shell_vol = int(np.sum(shell))
    support_vol = int(np.sum(support))
    uncertain_vol = int(np.sum(uncertain))

    return {
        "support_voxels": support_vol,
        "core_voxels": core_vol,
        "uncertain_shell_voxels": shell_vol,
        "uncertainty_fraction": float(uncertain_vol / support_vol) if support_vol > 0 else 0.0,
        "shell_core_ratio": float(shell_vol / max(core_vol, 1)),
        "mean_probability_in_support": float(prob[support].mean()) if np.any(support) else 0.0,
        "mean_entropy_in_support": float(entropy_support.mean()) if entropy_support.size > 0 else 0.0,
        "max_entropy_in_support": float(entropy_support.max()) if entropy_support.size > 0 else 0.0,
    }


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.sum(a & b)
    denom = np.sum(a) + np.sum(b)
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask > 0
    if not np.any(mask):
        return mask.astype(np.uint8)
    eroded = binary_erosion(mask)
    boundary = mask ^ eroded
    return boundary.astype(np.uint8)


# =============================================================================
# Directory parsing
# =============================================================================

def get_model_dirs(case_dir: Path) -> Dict[str, Path]:
    out = {}
    for dirname, model_name in KNOWN_MODEL_DIRS.items():
        p = case_dir / dirname
        if p.is_dir():
            out[model_name] = p
    return out


def get_qc_model_dirs(case_dir: Path) -> Dict[str, Path]:
    qc_root = case_dir / "qc_tier1_filtered"
    out = {}
    if not qc_root.is_dir():
        return out

    for dirname, model_name in KNOWN_MODEL_DIRS.items():
        p = qc_root / dirname
        if p.is_dir():
            out[model_name] = p
    return out


def get_consensus_dirs(case_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    hard_dir = case_dir / "segmentations_consensus"
    soft_dir = case_dir / "segmentations_consensus_soft"
    return (hard_dir if hard_dir.is_dir() else None,
            soft_dir if soft_dir.is_dir() else None)


def organ_mask_path(seg_dir: Path, organ: str) -> Optional[Path]:
    p = seg_dir / f"{organ}.nii.gz"
    return p if p.exists() else None


# =============================================================================
# Radiomics
# =============================================================================

def make_radiomics_extractor() -> Optional[Any]:
    if not PYRADIOMICS_AVAILABLE:
        return None

    params = {
        "binWidth": 25,
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        "enableCExtensions": True,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")

    return extractor


def sitk_from_nib(nii: nib.Nifti1Image, data: np.ndarray) -> "sitk.Image":
    img = sitk.GetImageFromArray(np.asarray(data))
    spacing = nii.header.get_zooms()[:3]
    img.SetSpacing(tuple(float(x) for x in spacing))
    return img


def compute_selected_radiomics(
    image_path: Path,
    mask_path: Path,
    selected_features: List[str],
    extractor: Optional[Any] = None,
) -> Dict[str, float]:
    if not PYRADIOMICS_AVAILABLE or extractor is None:
        return {}

    image_nib = nib.load(str(image_path))
    image_arr = np.asarray(image_nib.dataobj).astype(np.float32)
    mask_arr = (np.asarray(nib.load(str(mask_path)).dataobj) > 0).astype(np.uint8)

    if np.sum(mask_arr) == 0:
        return {}

    image_sitk = sitk_from_nib(image_nib, image_arr)
    mask_sitk = sitk_from_nib(image_nib, mask_arr)

    try:
        res = extractor.execute(image_sitk, mask_sitk)
    except Exception:
        return {}

    out = {}
    for feat in selected_features:
        key = f"original_{feat}"
        if key in res:
            try:
                out[feat] = float(res[key])
            except Exception:
                pass
    return out


# =============================================================================
# Per-case extraction
# =============================================================================

def summarize_case(
    case_dir: Path,
    organs: List[str],
    selected_features: List[str],
    extractor: Optional[Any] = None,
) -> Dict[str, Any]:
    volume_path = find_volume_path(case_dir)

    raw_model_dirs = get_model_dirs(case_dir)
    qc_model_dirs = get_qc_model_dirs(case_dir)
    consensus_hard_dir, consensus_soft_dir = get_consensus_dirs(case_dir)

    case_summary: Dict[str, Any] = {
        "case_id": case_dir.name,
        "case_path": str(case_dir),
        "volume_path": str(volume_path) if volume_path else None,
        "organs": {},
    }

    for organ in organs:
        organ_info: Dict[str, Any] = {
            "raw_models": {},
            "qc_models": {},
            "consensus_binary": None,
            "consensus_soft": None,
        }

        # Raw model masks
        for model_name, seg_dir in raw_model_dirs.items():
            p = organ_mask_path(seg_dir, organ)
            if p is None:
                organ_info["raw_models"][model_name] = {"present": False}
                continue

            mask, nii = load_mask(p)
            spacing = get_spacing_hwd(nii)

            entry = {
                "present": True,
                "mask_path": str(p),
                "volume_ml": compute_volume_ml(mask, spacing),
                "num_voxels": int(np.sum(mask)),
            }

            if volume_path is not None:
                entry["radiomics"] = compute_selected_radiomics(
                    image_path=volume_path,
                    mask_path=p,
                    selected_features=selected_features,
                    extractor=extractor,
                )
            else:
                entry["radiomics"] = {}

            organ_info["raw_models"][model_name] = entry

        # QC-kept model masks
        for model_name, seg_dir in qc_model_dirs.items():
            p = organ_mask_path(seg_dir, organ)
            if p is None:
                organ_info["qc_models"][model_name] = {"present": False}
                continue

            mask, nii = load_mask(p)
            spacing = get_spacing_hwd(nii)

            entry = {
                "present": True,
                "mask_path": str(p),
                "volume_ml": compute_volume_ml(mask, spacing),
                "num_voxels": int(np.sum(mask)),
            }

            if volume_path is not None:
                entry["radiomics"] = compute_selected_radiomics(
                    image_path=volume_path,
                    mask_path=p,
                    selected_features=selected_features,
                    extractor=extractor,
                )
            else:
                entry["radiomics"] = {}

            organ_info["qc_models"][model_name] = entry

        # Consensus hard
        if consensus_hard_dir is not None:
            p = organ_mask_path(consensus_hard_dir, organ)
            if p is not None:
                mask, nii = load_mask(p)
                spacing = get_spacing_hwd(nii)

                entry = {
                    "present": True,
                    "mask_path": str(p),
                    "volume_ml": compute_volume_ml(mask, spacing),
                    "num_voxels": int(np.sum(mask)),
                }

                if volume_path is not None:
                    entry["radiomics"] = compute_selected_radiomics(
                        image_path=volume_path,
                        mask_path=p,
                        selected_features=selected_features,
                        extractor=extractor,
                    )
                else:
                    entry["radiomics"] = {}

                organ_info["consensus_binary"] = entry

        # Consensus soft
        if consensus_soft_dir is not None:
            p = organ_mask_path(consensus_soft_dir, organ)
            if p is not None:
                soft_arr, nii = load_nifti(p)
                soft_arr = np.asarray(soft_arr).astype(np.float32)
                organ_info["consensus_soft"] = {
                    "present": True,
                    "mask_path": str(p),
                    **compute_soft_uncertainty_stats(soft_arr),
                }

        case_summary["organs"][organ] = organ_info

    return case_summary


def extract_population_case_summaries(
    population_root: str | Path,
    out_dir: str | Path,
    organs: List[str],
    selected_features: List[str],
) -> None:
    ensure_dir(out_dir)
    extractor = make_radiomics_extractor()

    case_dirs = list_case_dirs(population_root)
    print(f"[INFO] Found {len(case_dirs)} case directories")

    for case_dir in case_dirs:
        try:
            summary = summarize_case(
                case_dir=case_dir,
                organs=organs,
                selected_features=selected_features,
                extractor=extractor,
            )
            out_path = Path(out_dir) / f"{case_dir.name}.json"
            save_json(summary, out_path)
            print(f"[OK] Saved: {out_path}")
        except Exception as e:
            print(f"[FAIL] {case_dir.name}: {e}")


# =============================================================================
# Population aggregation
# =============================================================================

def load_all_case_summaries(summary_dir: str | Path) -> List[Dict[str, Any]]:
    summary_dir = Path(summary_dir)
    files = sorted(summary_dir.glob("*.json"))
    return [load_json(p) for p in files]


def aggregate_population(summaries: List[Dict[str, Any]], organs: List[str]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {
        "organs": {},
        "cases": len(summaries),
    }

    for organ in organs:
        organ_agg = {
            "raw_presence": {},
            "qc_presence": {},
            "raw_volumes": {},
            "qc_volumes": {},
            "consensus_volumes": [],
            "soft_uncertainty_fraction": [],
            "soft_shell_core_ratio": [],
            "soft_mean_entropy": [],
        }

        all_models = set()
        for case in summaries:
            all_models.update(case["organs"][organ]["raw_models"].keys())
            all_models.update(case["organs"][organ]["qc_models"].keys())

        for model in sorted(all_models):
            organ_agg["raw_presence"][model] = 0
            organ_agg["qc_presence"][model] = 0
            organ_agg["raw_volumes"][model] = []
            organ_agg["qc_volumes"][model] = []

        for case in summaries:
            c = case["organs"][organ]

            for model, info in c["raw_models"].items():
                if info.get("present", False):
                    organ_agg["raw_presence"][model] += 1
                    organ_agg["raw_volumes"][model].append(info["volume_ml"])

            for model, info in c["qc_models"].items():
                if info.get("present", False):
                    organ_agg["qc_presence"][model] += 1
                    organ_agg["qc_volumes"][model].append(info["volume_ml"])

            if c["consensus_binary"] is not None and c["consensus_binary"].get("present", False):
                organ_agg["consensus_volumes"].append(c["consensus_binary"]["volume_ml"])

            if c["consensus_soft"] is not None and c["consensus_soft"].get("present", False):
                organ_agg["soft_uncertainty_fraction"].append(c["consensus_soft"]["uncertainty_fraction"])
                organ_agg["soft_shell_core_ratio"].append(c["consensus_soft"]["shell_core_ratio"])
                organ_agg["soft_mean_entropy"].append(c["consensus_soft"]["mean_entropy_in_support"])

        agg["organs"][organ] = organ_agg

    return agg


def save_population_aggregate(summary_dir: str | Path, out_path: str | Path, organs: List[str]) -> None:
    summaries = load_all_case_summaries(summary_dir)
    agg = aggregate_population(summaries, organs)
    save_json(agg, out_path)
    print(f"[OK] Saved population aggregate: {out_path}")


# =============================================================================
# Plotting
# =============================================================================

def plot_retention_heatmap(pop_agg: Dict[str, Any], out_dir: str | Path) -> None:
    ensure_dir(out_dir)

    organs = list(pop_agg["organs"].keys())
    all_models = sorted({
        m
        for organ in organs
        for m in pop_agg["organs"][organ]["raw_presence"].keys()
    })

    mat = np.zeros((len(organs), len(all_models)), dtype=np.float32)

    for i, organ in enumerate(organs):
        raw_p = pop_agg["organs"][organ]["raw_presence"]
        qc_p = pop_agg["organs"][organ]["qc_presence"]
        for j, model in enumerate(all_models):
            raw_n = raw_p.get(model, 0)
            qc_n = qc_p.get(model, 0)
            mat[i, j] = float(qc_n / raw_n) if raw_n > 0 else np.nan

    fig, ax = plt.subplots(figsize=(1.2 * len(all_models) + 3, 0.7 * len(organs) + 2))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(all_models)))
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(organs)))
    ax.set_yticklabels(organs)
    ax.set_title("QC retention rate by organ and model")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Retention rate")

    for i in range(len(organs)):
        for j in range(len(all_models)):
            val = mat[i, j]
            txt = "NA" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "retention_heatmap.png", dpi=200)
    plt.close(fig)


def plot_volume_distributions(pop_agg: Dict[str, Any], organs: List[str], out_dir: str | Path) -> None:
    ensure_dir(out_dir)

    for organ in organs:
        organ_agg = pop_agg["organs"][organ]
        models = sorted(organ_agg["raw_volumes"].keys())

        raw_data = [organ_agg["raw_volumes"][m] for m in models]
        qc_data = [organ_agg["qc_volumes"][m] for m in models]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].boxplot(raw_data, tick_labels=models, vert=True)
        axes[0].set_title(f"{organ} - Raw volumes")
        axes[0].set_ylabel("Volume (mL)")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].boxplot(qc_data, tick_labels=models, vert=True)
        axes[1].set_title(f"{organ} - QC-kept volumes")
        axes[1].set_ylabel("Volume (mL)")
        axes[1].tick_params(axis="x", rotation=45)

        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"{organ}_volume_boxplots.png", dpi=200)
        plt.close(fig)


def plot_consensus_uncertainty(pop_agg: Dict[str, Any], organs: List[str], out_dir: str | Path) -> None:
    ensure_dir(out_dir)

    metrics = [
        ("soft_uncertainty_fraction", "Uncertainty fraction"),
        ("soft_shell_core_ratio", "Shell / core ratio"),
        ("soft_mean_entropy", "Mean entropy in support"),
    ]

    for metric_key, title in metrics:
        data = [pop_agg["organs"][organ][metric_key] for organ in organs]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data, tick_labels=organs, vert=True)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"{metric_key}.png", dpi=200)
        plt.close(fig)


# =============================================================================
# Qualitative visualization
# =============================================================================

def choose_slice_from_mask(mask: np.ndarray, axis: int = 2) -> int:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return mask.shape[axis] // 2
    return int(np.median(coords[:, axis]))


def extract_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return arr[idx, :, :]
    elif axis == 1:
        return arr[:, idx, :]
    else:
        return arr[:, :, idx]


def overlay_case_organ(
    case_dir: str | Path,
    organ: str,
    axis: int,
    out_path: str | Path,
) -> None:
    case_dir = Path(case_dir)
    volume_path = find_volume_path(case_dir)
    if volume_path is None:
        raise FileNotFoundError(f"No volume found in {case_dir}")

    image, _ = load_nifti(volume_path)
    raw_model_dirs = get_model_dirs(case_dir)
    qc_model_dirs = get_qc_model_dirs(case_dir)
    consensus_hard_dir, consensus_soft_dir = get_consensus_dirs(case_dir)

    model_masks = {}
    for model_name, seg_dir in qc_model_dirs.items():
        p = organ_mask_path(seg_dir, organ)
        if p is not None:
            model_masks[model_name], _ = load_mask(p)

    hard_mask = None
    if consensus_hard_dir is not None:
        p = organ_mask_path(consensus_hard_dir, organ)
        if p is not None:
            hard_mask, _ = load_mask(p)

    soft_mask = None
    if consensus_soft_dir is not None:
        p = organ_mask_path(consensus_soft_dir, organ)
        if p is not None:
            soft_mask, _ = load_nifti(p)
            soft_mask = np.asarray(soft_mask).astype(np.float32)

    ref_mask = hard_mask
    if ref_mask is None and model_masks:
        ref_mask = next(iter(model_masks.values()))
    if ref_mask is None:
        raise ValueError(f"No masks found for organ={organ} in case={case_dir.name}")

    slice_idx = choose_slice_from_mask(ref_mask, axis=axis)

    img_sl = extract_slice(image, axis, slice_idx)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_sl.T, cmap="gray", origin="lower")

    # soft overlay
    if soft_mask is not None:
        soft_sl = extract_slice(soft_mask, axis, slice_idx)
        ax.imshow(soft_sl.T, alpha=0.35, origin="lower")

    # model contours
    for model_name, mask in model_masks.items():
        bnd = boundary_mask(mask)
        bnd_sl = extract_slice(bnd, axis, slice_idx)
        ys, xs = np.where(bnd_sl > 0)
        if len(xs) > 0:
            ax.scatter(xs, ys, s=1, label=model_name)

    # hard consensus contour
    if hard_mask is not None:
        hard_bnd = boundary_mask(hard_mask)
        hard_bnd_sl = extract_slice(hard_bnd, axis, slice_idx)
        ys, xs = np.where(hard_bnd_sl > 0)
        if len(xs) > 0:
            ax.scatter(xs, ys, s=2, label="consensus_binary")

    ax.set_title(f"{case_dir.name}\n{organ} | axis={axis} | slice={slice_idx}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--population_root", required=True)
    p_extract.add_argument("--out_dir", required=True)
    p_extract.add_argument("--organs", nargs="+", default=DEFAULT_ORGANS)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--summary_dir", required=True)
    p_agg.add_argument("--out_json", required=True)
    p_agg.add_argument("--organs", nargs="+", default=DEFAULT_ORGANS)

    p_plot = sub.add_parser("plot")
    p_plot.add_argument("--population_json", required=True)
    p_plot.add_argument("--out_dir", required=True)
    p_plot.add_argument("--organs", nargs="+", default=DEFAULT_ORGANS)

    p_vis = sub.add_parser("visualize")
    p_vis.add_argument("--case_dir", required=True)
    p_vis.add_argument("--organ", required=True)
    p_vis.add_argument("--axis", type=int, default=2)
    p_vis.add_argument("--out_path", required=True)

    args = parser.parse_args()

    if args.cmd == "extract":
        extract_population_case_summaries(
            population_root=args.population_root,
            out_dir=args.out_dir,
            organs=args.organs,
            selected_features=DEFAULT_RADIOMICS_FEATURES,
        )

    elif args.cmd == "aggregate":
        save_population_aggregate(
            summary_dir=args.summary_dir,
            out_path=args.out_json,
            organs=args.organs,
        )

    elif args.cmd == "plot":
        pop_agg = load_json(args.population_json)
        ensure_dir(args.out_dir)
        plot_retention_heatmap(pop_agg, args.out_dir)
        plot_volume_distributions(pop_agg, args.organs, args.out_dir)
        plot_consensus_uncertainty(pop_agg, args.organs, args.out_dir)

    elif args.cmd == "visualize":
        overlay_case_organ(
            case_dir=args.case_dir,
            organ=args.organ,
            axis=args.axis,
            out_path=args.out_path,
        )


if __name__ == "__main__":
    main()

""" 
python scripts/qc_consensus_analysis.py extract \
  --population_root /data/soumitri/test_pipeline_new_2_llm \
  --out_dir /data/soumitri/test_pipeline_new_2_llm/analysis_case_jsons \
  --organs liver spleen kidney_left kidney_right pancreas

python scripts/qc_consensus_analysis.py aggregate \
  --summary_dir /data/soumitri/test_pipeline_new_2_llm/analysis_case_jsons \
  --out_json /data/soumitri/test_pipeline_new_2_llm/population_summary.json \
  --organs liver spleen kidney_left kidney_right pancreas

python scripts/qc_consensus_analysis.py plot \
  --population_json /data/soumitri/test_pipeline_new_2_llm/population_summary.json \
  --out_dir /data/soumitri/test_pipeline_new_2_llm/analysis_plots \
  --organs liver spleen kidney_left kidney_right pancreas

python scripts/qc_consensus_analysis.py visualize \
  --case_dir /data/soumitri/test_pipeline_new_2_llm/<CASE_DIR> \
  --organ liver \
  --axis 2 \
  --out_path /data/soumitri/test_pipeline_new_2_llm/liver_overlay.png
"""