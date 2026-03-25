"""
================================================================================
Format Utilities — Canonical Output Normalization
================================================================================
All segmentation tool wrappers must produce per-organ binary uint8 NIfTI files
with the same affine and shape as the input image_nifti.nii.gz.

Tools that natively output multilabel NIfTIs (MRSegmentator, MRISegmenter,
VISTA3D, etc.) use split_multilabel_nifti() to convert before writing their
manifest. Tools like TotalSegmentator that already output per-organ NIfTIs
can use save_organ_mask() directly if they need to re-save with a forced dtype.

Scope: 3D volumes only.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Organ Name Normalization
# ──────────────────────────────────────────────────────────────────────────────
# Tools use inconsistent naming for the same organ:
#   VISTA3D:    left_kidney, left_adrenal_gland, left_lung_lower_lobe
#   TotalSeg:   kidney_left, adrenal_gland_left, lung_lower_lobe_left
#   VIBESeg:    intestine (vs small_bowel), IVD (vs intervertebral_discs)
#
# Canonical form: {structure}_{side} (TotalSegmentator convention).
# All names lowercased, underscores for spaces/hyphens.

# Exact synonym mapping (applied after lowercase + directional swap)
_ORGAN_SYNONYMS: Dict[str, str] = {
    "intestine": "small_bowel",
    "ivd": "intervertebral_discs",
    "bladder": "urinary_bladder",
    "spinal_channel": "spinal_canal",
    "vertebra_body": "vertebrae_body",
    "vertebra_posterior_elements": "vertebrae_posterior_elements",
}

# Regex: match VISTA3D-style "left_<organ>" or "right_<organ>" prefixes
_DIRECTIONAL_PREFIX_RE = re.compile(
    r"^(left|right)_rib_(\d+)$"   # Special case: left_rib_10 → rib_left_10
)
_DIRECTIONAL_GENERIC_RE = re.compile(
    r"^(left|right)_(.+)$"        # Generic: left_kidney → kidney_left
)


def normalize_organ_name(name: str) -> str:
    """
    Normalize an organ name to canonical form.

    Canonical conventions (matching TotalSegmentator / majority of tools):
      - All lowercase
      - Underscores for separators (no spaces, no hyphens)
      - Laterality as suffix: kidney_left, adrenal_gland_right
      - Known synonyms resolved: intestine → small_bowel, IVD → intervertebral_discs

    Examples:
        >>> normalize_organ_name("left_kidney")
        'kidney_left'
        >>> normalize_organ_name("Left_Adrenal_Gland")
        'adrenal_gland_left'
        >>> normalize_organ_name("left_rib_10")
        'rib_left_10'
        >>> normalize_organ_name("IVD")
        'intervertebral_discs'
        >>> normalize_organ_name("intestine")
        'small_bowel'
        >>> normalize_organ_name("kidney_left")  # already canonical
        'kidney_left'
    """
    # Step 1: lowercase, normalize separators
    name = name.lower().strip().replace(" ", "_").replace("-", "_")

    # Step 2: directional prefix → suffix (VISTA3D convention → TotalSeg convention)
    # Special case for ribs: left_rib_10 → rib_left_10 (not rib_10_left)
    m = _DIRECTIONAL_PREFIX_RE.match(name)
    if m:
        side, num = m.group(1), m.group(2)
        name = f"rib_{side}_{num}"
    else:
        m = _DIRECTIONAL_GENERIC_RE.match(name)
        if m:
            side, rest = m.group(1), m.group(2)
            name = f"{rest}_{side}"

    # Step 3: exact synonyms
    name = _ORGAN_SYNONYMS.get(name, name)

    return name


def build_normalized_mask_index(
    seg_dir: str,
) -> Dict[str, str]:
    """
    Build a mapping of canonical_organ_name → file_path for all masks in a seg_dir.

    Skips non-organ files (manifest.json, statistics.json, multilabel_seg, image_nifti_seg).

    Returns:
        Dict mapping normalized organ name to the absolute path of the .nii.gz file.
    """
    _SKIP_STEMS = {
        "manifest", "statistics", "multilabel_seg", "image_nifti_seg",
        "combined", "segmentation", "multilabel",
    }
    index: Dict[str, str] = {}
    if not os.path.isdir(seg_dir):
        return index
    for fname in os.listdir(seg_dir):
        if not fname.endswith(".nii.gz"):
            continue
        stem = fname.replace(".nii.gz", "")
        if stem in _SKIP_STEMS:
            continue
        canonical = normalize_organ_name(stem)
        index[canonical] = os.path.join(seg_dir, fname)
    return index


def load_reference(
    reference_nifti_path: str,
) -> Tuple[np.ndarray, nib.Nifti1Header, Tuple[int, ...]]:
    """
    Load affine, header, and shape from a reference NIfTI.

    Args:
        reference_nifti_path: Path to image_nifti.nii.gz (or any same-space NIfTI).

    Returns:
        (affine, header, shape) — shape is the full nibabel shape tuple.
    """
    ref = nib.load(reference_nifti_path)
    return ref.affine, ref.header, ref.shape


def save_organ_mask(
    mask: np.ndarray,
    organ: str,
    seg_dir: str,
    affine: np.ndarray,
    header: Optional[nib.Nifti1Header] = None,
) -> str:
    """
    Save a binary mask array as <seg_dir>/<organ>.nii.gz.

    The mask is cast to uint8 regardless of input dtype. The affine and header
    come from the reference image so spatial metadata is preserved.

    Args:
        mask:   Binary array of any numeric dtype.
        organ:  Organ name used as the filename stem (e.g. "liver" → liver.nii.gz).
        seg_dir: Output directory; created if absent.
        affine: 4×4 affine matrix from the reference image.
        header: NIfTI header from the reference image (optional but recommended).

    Returns:
        Absolute path to the saved file.
    """
    os.makedirs(seg_dir, exist_ok=True)
    out_path = os.path.join(seg_dir, f"{organ}.nii.gz")
    img = nib.Nifti1Image(mask.astype(np.uint8), affine, header)
    nib.save(img, out_path)
    return out_path


def split_multilabel_nifti(
    multilabel_path: str,
    label_map: Dict[int, str],
    seg_dir: str,
) -> List[str]:
    """
    Split a multilabel NIfTI into per-organ binary NIfTIs.

    The multilabel NIfTI is assumed to share the same affine/space as the
    input image (guaranteed when the model produced it from that image).
    Labels with zero non-zero voxels are silently skipped.

    Args:
        multilabel_path: Path to multilabel NIfTI (integer-valued).
        label_map:       {label_index: organ_name} mapping.
        seg_dir:         Output directory for per-organ files.

    Returns:
        List of organ names that had non-zero voxels and were saved.
    """
    os.makedirs(seg_dir, exist_ok=True)
    img = nib.load(multilabel_path)
    data = np.asarray(img.dataobj, dtype=np.int32)

    saved: List[str] = []
    for label_idx, organ in label_map.items():
        binary = (data == label_idx).astype(np.uint8)
        if binary.sum() == 0:
            continue
        save_organ_mask(binary, organ, seg_dir, img.affine, img.header)
        saved.append(organ)

    return saved
