"""
CDW Agentic Pipeline — Tier 2 Multi-Tool Agreement QC
=====================================================
Cross-tool pairwise agreement checks for segmentation masks.
No ground truth required — tools are compared against each other.

Checks:
  1. Pairwise Dice coefficient between each tool pair for shared organs
  2. Volume agreement between tool pairs
  3. STAPLE / majority-vote consensus generation (optional)

Usage
-----
from qc.multi_tool_qc import MultiToolQC

qc = MultiToolQC()
result = qc.run(
    case_path="/data/soumitri/segmentations_3d/RHEUM/PT123/CT_ABD",
    seg_dirs={
        "TotalSegmentator_CT": "/path/to/segmentations_totalseg_ct",
        "MRSegmentator": "/path/to/segmentations_mrseg",
    },
)
print(result.mean_dice, result.organs_with_disagreement)
"""

from __future__ import annotations

import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from qc.dataclasses import MultiToolQCResult, OrganAgreement

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dice coefficient
# ──────────────────────────────────────────────────────────────────────────────

def _dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Dice similarity coefficient between two binary masks."""
    a_bin = mask_a > 0
    b_bin = mask_b > 0
    intersection = int(np.sum(a_bin & b_bin))
    total = int(np.sum(a_bin)) + int(np.sum(b_bin))
    if total == 0:
        return 1.0  # both empty → perfect agreement
    return 2.0 * intersection / total


def _compute_volume_ml(mask_data: np.ndarray, voxel_dims_mm: tuple) -> float:
    """Compute volume of non-zero voxels in mL."""
    voxel_vol_mm3 = float(np.prod(voxel_dims_mm))
    return int(np.sum(mask_data > 0)) * voxel_vol_mm3 / 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Tool QC engine
# ──────────────────────────────────────────────────────────────────────────────

class MultiToolQC:
    """
    Tier 2 cross-tool agreement checks.

    Compares segmentation outputs from multiple tools pairwise.
    Flags organs where tools disagree significantly.

    Args:
        dice_threshold: Minimum Dice for agreement (default 0.70).
        volume_ratio_threshold: Maximum allowed volume ratio divergence (default 0.50 = 50%).
    """

    def __init__(
        self,
        dice_threshold: float = 0.70,
        volume_ratio_threshold: float = 0.50,
    ):
        self.dice_threshold = dice_threshold
        self.volume_ratio_threshold = volume_ratio_threshold

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        case_path: str,
        seg_dirs: Dict[str, str],
    ) -> MultiToolQCResult:
        """
        Run Tier 2 cross-tool agreement checks for one case.

        Args:
            case_path: Root directory for this case.
            seg_dirs: Dict of tool_name → seg_dir path (at least 2 tools).

        Returns:
            MultiToolQCResult with pairwise organ agreements.
        """
        case_id = Path(case_path).name
        result = MultiToolQCResult(case_id=case_id, case_path=case_path)

        # Need at least 2 tools
        valid_dirs = {t: d for t, d in seg_dirs.items() if os.path.isdir(d)}
        if len(valid_dirs) < 2:
            return result

        # Load all tool masks: tool_name → {organ_name → (mask_data, voxel_dims)}
        tool_masks: Dict[str, Dict[str, Tuple[np.ndarray, tuple]]] = {}
        for tool_name, seg_dir in valid_dirs.items():
            masks = self._load_tool_masks(seg_dir)
            if masks:
                tool_masks[tool_name] = masks

        if len(tool_masks) < 2:
            return result

        # Pairwise comparison
        tool_names = list(tool_masks.keys())
        all_agreements: List[OrganAgreement] = []
        dice_values: List[float] = []

        for tool_a, tool_b in combinations(tool_names, 2):
            result.tool_pairs_checked += 1
            pair_agreements = self._compare_tool_pair(
                tool_a, tool_masks[tool_a],
                tool_b, tool_masks[tool_b],
                case_id,
            )
            all_agreements.extend(pair_agreements)
            for ag in pair_agreements:
                dice_values.append(ag.dice)

        result.agreements = all_agreements
        if dice_values:
            result.mean_dice = float(np.mean(dice_values))

        result.organs_with_agreement = sum(1 for a in all_agreements if a.agrees)
        result.organs_with_disagreement = sum(1 for a in all_agreements if not a.agrees)

        return result

    def generate_consensus(
        self,
        seg_dirs: Dict[str, str],
        organ: str,
        method: str = "staple",
    ) -> Optional[np.ndarray]:
        """
        Generate consensus mask for a single organ across multiple tools.

        Args:
            seg_dirs: Dict of tool_name → seg_dir path.
            organ: Organ name (filename stem without .nii.gz).
            method: "staple" or "majority_vote".

        Returns:
            Consensus binary mask as numpy array, or None if insufficient data.
        """
        from processing.postprocessing import majority_vote, staple_fusion

        masks = []
        for seg_dir in seg_dirs.values():
            fpath = os.path.join(seg_dir, f"{organ}.nii.gz")
            if os.path.isfile(fpath):
                try:
                    mask_data = np.asarray(nib.load(fpath).dataobj, dtype=np.float32)
                    if np.sum(mask_data > 0) > 0:
                        masks.append((mask_data > 0).astype(np.uint8))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", fpath, e)

        if len(masks) < 2:
            return masks[0] if masks else None

        # Ensure all masks have the same shape
        ref_shape = masks[0].shape
        masks = [m for m in masks if m.shape == ref_shape]
        if len(masks) < 2:
            return masks[0] if masks else None

        if method == "staple":
            return staple_fusion(masks)
        else:
            return majority_vote(masks)

    # ── Internal ────────────────────────────────────────────────────────────

    def _load_tool_masks(
        self, seg_dir: str
    ) -> Dict[str, Tuple[np.ndarray, tuple]]:
        """Load all organ masks from a segmentation directory.

        Organ names are normalized to canonical form (e.g., left_kidney → kidney_left)
        so cross-tool matching works regardless of tool-specific naming conventions.
        """
        from processing.format_utils import normalize_organ_name

        _SKIP_STEMS = {
            "statistics", "combined", "multilabel", "multilabel_seg",
            "image_nifti_seg", "segmentation", "manifest",
        }
        masks = {}
        for fname in os.listdir(seg_dir):
            if not fname.endswith(".nii.gz"):
                continue
            raw_name = fname.replace(".nii.gz", "")
            if raw_name in _SKIP_STEMS:
                continue
            organ_name = normalize_organ_name(raw_name)
            fpath = os.path.join(seg_dir, fname)
            try:
                nii = nib.load(fpath)
                data = np.asarray(nii.dataobj, dtype=np.float32)
                voxel_dims = nii.header.get_zooms()[:3]
                masks[organ_name] = (data, voxel_dims)
            except Exception as e:
                logger.warning("Failed to load %s: %s", fpath, e)
        return masks

    def _compare_tool_pair(
        self,
        tool_a: str,
        masks_a: Dict[str, Tuple[np.ndarray, tuple]],
        tool_b: str,
        masks_b: Dict[str, Tuple[np.ndarray, tuple]],
        case_id: str,
    ) -> List[OrganAgreement]:
        """Compare all shared organs between two tools."""
        shared_organs = set(masks_a.keys()) & set(masks_b.keys())
        agreements = []

        for organ in sorted(shared_organs):
            data_a, dims_a = masks_a[organ]
            data_b, dims_b = masks_b[organ]

            # Skip if shapes don't match (different image spaces)
            if data_a.shape != data_b.shape:
                logger.warning(
                    "[%s] Shape mismatch for %s: %s vs %s (%s vs %s)",
                    case_id, organ, tool_a, tool_b, data_a.shape, data_b.shape,
                )
                continue

            vol_a = _compute_volume_ml(data_a, dims_a)
            vol_b = _compute_volume_ml(data_b, dims_b)

            # Skip organs not in FOV for either tool
            if vol_a < 1e-3 and vol_b < 1e-3:
                continue

            ag = OrganAgreement(
                organ=organ,
                tool_a=tool_a,
                tool_b=tool_b,
                volume_a_ml=vol_a,
                volume_b_ml=vol_b,
            )

            # Dice
            ag.dice = _dice_coefficient(data_a, data_b)

            # Agreement check
            flags = []
            if ag.dice < self.dice_threshold:
                flags.append(
                    f"LOW_DICE: {organ} {tool_a}/{tool_b} = {ag.dice:.3f} "
                    f"(threshold {self.dice_threshold:.2f})"
                )

            # Volume divergence check
            if vol_a > 1e-3 and vol_b > 1e-3:
                vol_ratio = abs(vol_a - vol_b) / max(vol_a, vol_b)
                if vol_ratio > self.volume_ratio_threshold:
                    flags.append(
                        f"VOLUME_DIVERGENCE: {organ} {tool_a}={vol_a:.1f}mL "
                        f"vs {tool_b}={vol_b:.1f}mL ({vol_ratio*100:.0f}% diff)"
                    )
            elif vol_a > 1e-3 or vol_b > 1e-3:
                # One tool found it, the other didn't
                flags.append(
                    f"PRESENCE_MISMATCH: {organ} found by "
                    f"{'both' if vol_a > 1e-3 and vol_b > 1e-3 else (tool_a if vol_a > 1e-3 else tool_b)} "
                    f"but not {tool_b if vol_a > 1e-3 else tool_a}"
                )

            if flags:
                ag.agrees = False
                ag.flag = "; ".join(flags)

            agreements.append(ag)

        return agreements
