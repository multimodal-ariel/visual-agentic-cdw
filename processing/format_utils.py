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
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


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
