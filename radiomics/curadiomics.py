"""
================================================================================
CDW Agentic Pipeline — cuRadiomics Wrapper
================================================================================
GPU-accelerated radiomic feature extraction via CUDA TensorFlow custom op.

Source: external/cuRadiomics  (cloned from github.com/shengfly/cuRadiomics)
Built:  external/cuRadiomics/build/libRadiomics.so

Compatibility notes
───────────────────
cuRadiomics was built against TF 1.12 / CUDA 9.2 / Python 3.6.
The wrapper handles TF 1→2 differences:
  - In TF 2.x, eager execution is ON by default; skip tfe.enable_eager_execution().
  - `tf.compat.v1` is used for ops that changed namespace.
  - The compiled .so must match the TensorFlow C ABI used at build time.
    If the ABI mismatches, a clear error is raised with build instructions.

Output shape
────────────
cuRadiomics computes features per 2D slice (axis 0 of the input volume).
  raw output shape: [num_features, num_slices]

The wrapper aggregates across slices to produce per-volume statistics:
  mean, std, median, min, max  → stored as separate feature keys.

API
───
    result = extract_curadiomics(arr_img, arr_msk, organ, ...)
    result = extract_curadiomics_nifti(image_path, mask_path, organ, ...)
    result.features          # dict[str, float]
    result.backend           # "curadiomics"

Environment: cdw_radiomics  (conda run -n cdw_radiomics python ...)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CURADIOMICS_ROOT = _REPO_ROOT / "external" / "cuRadiomics"
_SO_PATH = str(_CURADIOMICS_ROOT / "build" / "libRadiomics.so")
_DEFAULT_YAML = str(_CURADIOMICS_ROOT / "python" / "params.yaml")


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass — same schema as pyradiomics.py
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RadiomicsResult:
    """Unified output schema for both pyradiomics and curadiomics backends."""

    organ: str
    image_path: str
    mask_path: str
    backend: str
    features: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return (
            f"RadiomicsResult(organ={self.organ!r}, backend={self.backend!r}, "
            f"features={self.feature_count}, warnings={len(self.warnings)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Feature name lists (mirror of func_cuRadiomics.py)
# ──────────────────────────────────────────────────────────────────────────────

_FEATURE_NAMES_GLCM = [
    "Autocorrelation", "JointAverage", "ClusterProminence", "ClusterShade",
    "ClusterTendency", "Contrast", "Correlation", "DifferenceAverage",
    "DifferenceEntropy", "DifferenceVariance", "JointEnergy", "JointEntropy",
    "Imc1", "Imc2", "Idm", "Idmn", "Id", "Idn", "InverseVariance",
    "MaximumProbability", "SumAverage", "SumEntropy", "SumSquares",
]  # 23 features

_FEATURE_NAMES_FIRSTORDER = [
    "Energy", "Entropy", "Minimum", "TenthPercentile", "NintiethPercentile",
    "Maximum", "Mean", "Median", "InterquartileRange", "Range", "MAD",
    "rMAD", "RMS", "StandardDeviation", "Skewness", "Kurtosis", "Varianc",
    "Uniformity",
]  # 18 features


# ──────────────────────────────────────────────────────────────────────────────
# TensorFlow loader — handles TF1 and TF2
# ──────────────────────────────────────────────────────────────────────────────

def _load_tf_op(so_path: str):
    """
    Load the cuRadiomics CUDA custom op and return the callable.

    Handles both:
      - TF 1.x: tf.load_op_library(so_path).radiomics
      - TF 2.x: tf.load_op_library(so_path).radiomics  (same API, eager by default)

    Raises
    ------
    ImportError  — TensorFlow not installed
    OSError      — .so not found or ABI mismatch (with build instructions)
    """
    if not os.path.isfile(so_path):
        raise OSError(
            f"cuRadiomics shared library not found: {so_path}\n"
            "Build it inside the cdw_radiomics environment:\n"
            "  cd external/cuRadiomics && mkdir -p build && cd build\n"
            "  cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4\n"
            "The library requires: CUDA ≥9.2, TF 1.12 or TF 2.x headers, Python 3.6+."
        )

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TensorFlow is not installed in the current environment.\n"
            f"Original error: {e}"
        ) from e

    import tensorflow as tf

    # TF 2.x: disable eager globally only if TF 1.x API is used
    tf_major = int(tf.__version__.split(".")[0])
    if tf_major == 1:
        # TF 1.x: need to enable eager explicitly
        try:
            import tensorflow.contrib.eager as tfe
            tfe.enable_eager_execution()
        except ImportError:
            pass  # already enabled or contrib not present

    try:
        lib = tf.load_op_library(so_path)
    except Exception as exc:
        raise OSError(
            f"Failed to load cuRadiomics .so from {so_path}.\n"
            "Most likely cause: TensorFlow C ABI mismatch between build-time and runtime.\n"
            "Solution: rebuild the .so against the installed TensorFlow:\n"
            "  conda activate cdw_radiomics\n"
            "  cd external/cuRadiomics && rm -rf build && mkdir build && cd build\n"
            "  cmake .. && make -j4\n"
            f"Original error: {exc}"
        ) from exc

    return lib.radiomics


# ──────────────────────────────────────────────────────────────────────────────
# Per-slice → per-volume aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_slices(
    arr_features: np.ndarray,
    names: List[str],
) -> Dict[str, float]:
    """
    Convert per-slice feature matrix [num_features, num_slices] to per-volume
    summary statistics (mean, std, median, min, max) per feature.

    Parameters
    ----------
    arr_features : np.ndarray, shape [num_features, num_slices]
    names : list of feature names (length == num_features)

    Returns
    -------
    dict mapping "<feature>_mean" / "_std" / "_median" / "_min" / "_max" → float
    """
    out: Dict[str, float] = {}
    for i, name in enumerate(names):
        vals = arr_features[i, :]
        valid = vals[np.isfinite(vals)]
        if valid.size == 0:
            valid = np.array([0.0])
        out[f"{name}_mean"]   = float(np.mean(valid))
        out[f"{name}_std"]    = float(np.std(valid))
        out[f"{name}_median"] = float(np.median(valid))
        out[f"{name}_min"]    = float(np.min(valid))
        out[f"{name}_max"]    = float(np.max(valid))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction (array input)
# ──────────────────────────────────────────────────────────────────────────────

def extract_curadiomics(
    arr_img: np.ndarray,
    arr_msk: np.ndarray,
    organ: str,
    *,
    image_path: str = "",
    mask_path: str = "",
    yaml_addr: Optional[str] = None,
    label: int = 1,
    extract_glcm: bool = True,
    extract_firstorder: bool = True,
) -> RadiomicsResult:
    """
    Extract GPU-accelerated radiomic features from in-memory arrays.

    Parameters
    ----------
    arr_img : np.ndarray, dtype int, shape [slices, H, W]
        Image intensity array (e.g. CT HU values).
    arr_msk : np.ndarray, shape [slices, H, W]
        Binary mask array. Voxels with value == `label` are included.
    organ : str
        Anatomical structure label (stored in result, not used for compute).
    image_path / mask_path : str
        Original file paths (optional; stored in result for traceability).
    yaml_addr : str | None
        Path to cuRadiomics params.yaml. Defaults to external/cuRadiomics/python/params.yaml.
    label : int
        Mask foreground label. Default 1.
    extract_glcm : bool
        Whether to extract GLCM features (23 features). Default True.
    extract_firstorder : bool
        Whether to extract first-order features (18 features). Default True.

    Returns
    -------
    RadiomicsResult
        .features: per-volume aggregated statistics (mean/std/median/min/max per feature).
    """
    import tensorflow as tf

    collected_warnings: List[str] = []

    # Load CUDA op (raises on failure)
    _radiomics_op = _load_tf_op(_SO_PATH)

    # Resolve params yaml
    yaml_path = yaml_addr or _DEFAULT_YAML
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"cuRadiomics params yaml not found: {yaml_path}")

    import yaml
    with open(yaml_path) as f:
        parameters = yaml.safe_load(f)

    # Build SETTING vector: [range_min, range_max, FirstOrder, GLCM, label]
    arr_img_masked = arr_img.copy()
    arr_img_masked[arr_msk != label] = -1

    rng_min = int(np.min(arr_img_masked))
    rng_max = int(np.max(arr_img_masked))

    first_order_flag = int(extract_firstorder and parameters.get("FirstOrder", 1))
    glcm_flag        = int(extract_glcm and parameters.get("GLCM", 1))
    lbl              = parameters.get("label", label)

    SETTING = np.array([rng_min, rng_max, first_order_flag, glcm_flag, lbl])

    # Cast to int32 (required by the custom op)
    arr_img_int = arr_img_masked.astype(np.int32)

    # Run CUDA op
    try:
        arr_features_tf = _radiomics_op(arr_img_int, SETTING)
        arr_features = arr_features_tf.numpy()  # TF2 eager → numpy directly
    except AttributeError:
        # TF1: tensor.eval() or sess.run()
        try:
            import tensorflow as tf
            with tf.compat.v1.Session() as sess:
                arr_features = sess.run(arr_features_tf)
        except Exception as exc:
            collected_warnings.append(f"TF session fallback failed: {exc}")
            arr_features = np.array(arr_features_tf)

    # Determine feature names based on flags
    names: List[str] = []
    num_features = 0
    if glcm_flag:
        names += _FEATURE_NAMES_GLCM
        num_features += 23
    if first_order_flag:
        names += _FEATURE_NAMES_FIRSTORDER
        num_features += 18

    num_slices = arr_img.shape[0]
    try:
        arr_features = np.reshape(arr_features, (num_features, num_slices))
    except ValueError as e:
        raise RuntimeError(
            f"cuRadiomics output shape mismatch. "
            f"Expected [{num_features}, {num_slices}], got {arr_features.shape}. "
            f"Error: {e}"
        ) from e

    # Aggregate per-slice → per-volume
    features = _aggregate_slices(arr_features, names)

    logger.info(
        "cuRadiomics: organ=%s  features=%d  slices=%d  warnings=%d",
        organ, len(features), num_slices, len(collected_warnings),
    )

    return RadiomicsResult(
        organ=organ,
        image_path=image_path,
        mask_path=mask_path,
        backend="curadiomics",
        features=features,
        warnings=collected_warnings,
    )


# ──────────────────────────────────────────────────────────────────────────────
# NIfTI-file convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def extract_curadiomics_nifti(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    organ: str,
    *,
    yaml_addr: Optional[str] = None,
    label: int = 1,
    extract_glcm: bool = True,
    extract_firstorder: bool = True,
) -> RadiomicsResult:
    """
    Load NIfTI files and run GPU feature extraction.

    Parameters
    ----------
    image_path : str | Path
        NIfTI image (.nii or .nii.gz).
    mask_path : str | Path
        NIfTI binary mask (.nii or .nii.gz).
    organ : str
        Anatomical structure label.
    yaml_addr : str | None
        cuRadiomics params.yaml path.
    label : int
        Mask foreground label value.

    Returns
    -------
    RadiomicsResult
        Same schema as extract_curadiomics().
    """
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError(
            "SimpleITK is not installed. "
            "conda run -n cdw_radiomics python -m pip install SimpleITK"
        ) from e

    image_path = str(image_path)
    mask_path = str(mask_path)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    img_itk = sitk.ReadImage(image_path)
    msk_itk = sitk.ReadImage(mask_path)

    arr_img = sitk.GetArrayFromImage(img_itk).astype(np.int32)  # [Z, H, W]
    arr_msk = sitk.GetArrayFromImage(msk_itk).astype(np.int32)

    return extract_curadiomics(
        arr_img=arr_img,
        arr_msk=arr_msk,
        organ=organ,
        image_path=image_path,
        mask_path=mask_path,
        yaml_addr=yaml_addr,
        label=label,
        extract_glcm=extract_glcm,
        extract_firstorder=extract_firstorder,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="cuRadiomics GPU feature extraction")
    parser.add_argument("--image",   required=True, help="NIfTI image path")
    parser.add_argument("--mask",    required=True, help="NIfTI mask path")
    parser.add_argument("--organ",   default="unknown")
    parser.add_argument("--yaml",    default=None, help="cuRadiomics params.yaml")
    parser.add_argument("--label",   type=int, default=1)
    parser.add_argument("--no-glcm",        dest="glcm",        action="store_false")
    parser.add_argument("--no-firstorder",  dest="firstorder",  action="store_false")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = extract_curadiomics_nifti(
        image_path=args.image,
        mask_path=args.mask,
        organ=args.organ,
        yaml_addr=args.yaml,
        label=args.label,
        extract_glcm=args.glcm,
        extract_firstorder=args.firstorder,
    )
    print(json.dumps({"organ": result.organ, "feature_count": result.feature_count,
                      "features": result.features, "warnings": result.warnings}, indent=2))
    sys.exit(0)
