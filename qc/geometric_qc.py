"""
CDW Agentic Pipeline — Tier 1 Geometric QC
===========================================
Reference-free quality checks for segmentation masks.
No ground truth required. No image loading — masks only.

Ported from scripts/qc_segs_old.py and generalized to:
  - Any modality (CT, MRI, PET) — no hardcoded CT filter
  - Any tool's output dir (segmentations_totalseg_ct/, segmentations_mrseg/, etc.)
  - Reference data from config/organ_reference.json (not hardcoded)
  - Optional anatomy-aware expected organ list from OrganListGenerator
  - Importable module with clean API (not a standalone CLI)

Checks:
  1. Volume plausibility      — organ volume in mL vs anatomical reference ranges
  2. Paired volume ratios      — bilateral symmetry (kidneys, adrenals, iliacs, lungs)
  3. Connected components      — fragment count + largest-CC fraction
  4. Mask overlap              — pairwise voxel overlap between organ masks

Usage
-----
from qc.geometric_qc import GeometricQC

qc = GeometricQC()
result = qc.run(
    case_path="/data/soumitri/segmentations_3d/RHEUM/PT123/CT_ABD",
    seg_dir="/data/soumitri/segmentations_3d/RHEUM/PT123/CT_ABD/segmentations_totalseg_ct",
    tool_name="TotalSegmentator_CT",
)
print(result.worst_severity, result.num_organs_flagged)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

from qc.dataclasses import OrganQCResult, ToolQCResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Reference data loader
# ──────────────────────────────────────────────────────────────────────────────

_REF_PATH = Path(__file__).resolve().parent.parent / "config" / "organ_reference.json"


def _load_reference(path: Path = _REF_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Geometric QC engine
# ──────────────────────────────────────────────────────────────────────────────

class GeometricQC:
    """
    Tier 1 reference-free geometric quality checks.

    Args:
        organ_ref_path: Path to organ_reference.json. Defaults to config/organ_reference.json.
        lcc_min_fraction: Minimum fraction for largest connected component (default 0.85).
        warn_max_flags: Maximum flags before severity escalates from WARN to FAIL (default 2).
    """

    def __init__(
        self,
        organ_ref_path: str | Path = _REF_PATH,
        lcc_min_fraction: float = 0.85,
        warn_max_flags: int = 2,
    ):
        ref = _load_reference(Path(organ_ref_path))
        self.volume_ranges: Dict[str, Tuple[float, float]] = {
            k: tuple(v) for k, v in ref.get("volume_ranges_ml", {}).items()
        }
        self.paired_ratios: List[dict] = ref.get("paired_organ_ratios", [])
        self.lung_ratios: List[dict] = ref.get("lung_composite_ratios", [])
        self.expected_max_cc: Dict[str, int] = ref.get("expected_max_components", {})
        self.qc_organs: set = set(self.volume_ranges.keys()) | set(self.expected_max_cc.keys())
        self.lcc_min_fraction = lcc_min_fraction
        self.warn_max_flags = warn_max_flags

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        case_path: str,
        seg_dir: str,
        tool_name: str = "",
        expected_organs: Optional[List[str]] = None,
        modality: str = "CT",
    ) -> ToolQCResult:
        """
        Run all Tier 1 checks on one tool's output for one case.

        Args:
            case_path: Root directory for this case.
            seg_dir: Directory containing per-organ NIfTI masks.
            tool_name: Name of the segmentation tool (for tracking).
            expected_organs: Optional list of organ names expected in FOV.
                             If provided, organs NOT in this list are still checked
                             but their volume flags use relaxed thresholds.
            modality: Imaging modality ("CT", "MRI", etc.). MRI skips volume
                      plausibility checks (reference ranges are CT-calibrated)
                      and relies on CC + overlap checks only.

        Returns:
            ToolQCResult with per-organ results and aggregate severity.
        """
        case_id = Path(case_path).name
        result = ToolQCResult(
            case_id=case_id,
            case_path=case_path,
            tool_name=tool_name,
            seg_dir=seg_dir,
        )

        if not os.path.isdir(seg_dir):
            result.error = f"Segmentation directory not found: {seg_dir}"
            result.worst_severity = "ERROR"
            return result

        # Discover organ masks
        organ_files = self._discover_masks(seg_dir)
        if not organ_files:
            result.error = f"No organ mask NIfTIs found in {seg_dir}"
            result.worst_severity = "ERROR"
            return result

        result.num_organs = len(organ_files)

        # Run per-organ checks
        organ_masks: Dict[str, np.ndarray] = {}
        organ_volumes: Dict[str, float] = {}
        organ_results: Dict[str, OrganQCResult] = {}

        for organ, fpath in organ_files.items():
            oqc = OrganQCResult(
                case_id=case_id, organ=organ, tool_name=tool_name,
                case_path=case_path, mask_path=fpath,
            )
            try:
                mask_nii = nib.load(fpath)
                mask_data = np.asarray(mask_nii.dataobj, dtype=np.float32)
                voxel_dims = mask_nii.header.get_zooms()[:3]
                vol_ml = _compute_volume_ml(mask_data, voxel_dims)
                organ_volumes[organ] = vol_ml

                if vol_ml < 1e-3:
                    oqc.volume_ml = 0.0
                    oqc.volume_flag = f"NOT_IN_FOV: {organ}"
                    oqc.severity = "PASS"
                    organ_results[organ] = oqc
                    continue

                organ_masks[organ] = mask_data
                # Volume plausibility: skip for MRI (ranges are CT-calibrated)
                if modality != "MRI":
                    self._check_volume(organ, vol_ml, oqc)
                else:
                    oqc.volume_ml = vol_ml
                    oqc.volume_in_range = True  # no reference for MRI
                self._check_connected_components(organ, mask_data, oqc)

            except Exception as e:
                oqc.volume_flag = f"LOAD_ERROR: {e}"
                oqc.num_flags = 1
                oqc.severity = "FAIL"
                logger.warning("[%s/%s] Error processing %s: %s", case_id, tool_name, organ, e)

            organ_results[organ] = oqc

        # Cross-organ checks (skip paired ratios for MRI — CT-calibrated)
        if modality != "MRI":
            self._check_paired_ratios(organ_volumes, organ_results)
        overlap_pairs = self._check_overlap(organ_masks, organ_results)
        result.overlap_pairs = overlap_pairs

        # Compute per-organ severities and aggregate
        for organ, oqc in organ_results.items():
            oqc.compute_severity(warn_max=self.warn_max_flags)
            result.organ_results.append(oqc)
            result.total_flags += oqc.num_flags
            if oqc.num_flags > 0:
                result.num_organs_flagged += 1

        result.num_organs_in_fov = sum(1 for oqc in result.organ_results if oqc.volume_ml > 1e-3)

        severities = [oqc.severity for oqc in result.organ_results]
        if "FAIL" in severities:
            result.worst_severity = "FAIL"
        elif "WARN" in severities:
            result.worst_severity = "WARN"
        else:
            result.worst_severity = "PASS"

        return result

    def run_all_tools(
        self,
        case_path: str,
        seg_dirs: Dict[str, str],
        expected_organs: Optional[List[str]] = None,
    ) -> List[ToolQCResult]:
        """
        Run Tier 1 QC on multiple tools' outputs for one case.

        Args:
            case_path: Root directory for this case.
            seg_dirs: Dict of tool_name → seg_dir path.
            expected_organs: Optional expected organ list.

        Returns:
            List of ToolQCResult, one per tool.
        """
        results = []
        for tool_name, seg_dir in seg_dirs.items():
            if os.path.isdir(seg_dir):
                results.append(self.run(case_path, seg_dir, tool_name, expected_organs))
        return results

    # ── Mask discovery ──────────────────────────────────────────────────────

    # Non-organ filenames that can appear in segmentation directories
    _SKIP_NAMES = {
        "statistics", "combined", "multilabel",
        "image_nifti_seg", "image_nifti", "image",
        "plan", "metadata", "summary",
    }

    def _discover_masks(self, seg_dir: str) -> Dict[str, str]:
        """Find all per-organ .nii.gz mask files in a segmentation directory."""
        organ_files = {}
        for fname in os.listdir(seg_dir):
            if not fname.endswith(".nii.gz"):
                continue
            organ_name = fname.replace(".nii.gz", "")
            # Skip non-organ files (combined segs, images, metadata)
            if organ_name in self._SKIP_NAMES:
                continue
            # Skip names that look like images rather than masks
            if organ_name.startswith(("image", "img_", "vol_")):
                continue
            organ_files[organ_name] = os.path.join(seg_dir, fname)
        return organ_files

    # ── Check 1: Volume plausibility ────────────────────────────────────────

    def _check_volume(self, organ: str, volume_ml: float, result: OrganQCResult) -> None:
        result.volume_ml = volume_ml
        if organ not in self.volume_ranges:
            result.volume_in_range = True
            return
        vmin, vmax = self.volume_ranges[organ]
        if volume_ml < vmin:
            result.volume_in_range = False
            pct_below = (1 - volume_ml / vmin) * 100 if vmin > 0 else 0
            result.volume_flag = (
                f"UNDER_VOLUME: {organ} = {volume_ml:.1f} mL, "
                f"expected >= {vmin:.0f} mL ({pct_below:.0f}% below)"
            )
        elif volume_ml > vmax:
            result.volume_in_range = False
            pct_above = (volume_ml / vmax - 1) * 100 if vmax > 0 else 0
            result.volume_flag = (
                f"OVER_VOLUME: {organ} = {volume_ml:.1f} mL, "
                f"expected <= {vmax:.0f} mL ({pct_above:.0f}% above)"
            )

    # ── Check 2: Connected components ───────────────────────────────────────

    def _check_connected_components(
        self, organ: str, mask_data: np.ndarray, result: OrganQCResult
    ) -> None:
        if organ not in self.expected_max_cc:
            return
        binary = (mask_data > 0).astype(np.uint8)
        total_voxels = int(np.sum(binary))
        if total_voxels == 0:
            return
        structure = ndimage.generate_binary_structure(3, 3)  # 26-connected
        labeled, num_cc = ndimage.label(binary, structure=structure)
        result.num_components = num_cc
        component_sizes = ndimage.sum(binary, labeled, range(1, num_cc + 1))
        largest_size = float(np.max(component_sizes))
        result.largest_component_fraction = largest_size / total_voxels

        expected_max = self.expected_max_cc[organ]
        flags = []
        if num_cc > expected_max:
            flags.append(f"EXCESS_COMPONENTS: {organ} has {num_cc} CC (expected <= {expected_max})")
        if result.largest_component_fraction < self.lcc_min_fraction and num_cc > 1:
            flags.append(f"FRAGMENTED: {organ} largest CC = {result.largest_component_fraction * 100:.1f}%")
        result.cc_flag = "; ".join(flags)

    # ── Check 3: Paired volume ratios ───────────────────────────────────────

    def _check_paired_ratios(
        self,
        organ_volumes: Dict[str, float],
        organ_results: Dict[str, OrganQCResult],
    ) -> None:
        # Standard paired organs
        for pair in self.paired_ratios:
            org_a, org_b = pair["organ_a"], pair["organ_b"]
            rmin, rmax = pair["min_ratio"], pair["max_ratio"]
            if org_a in organ_volumes and org_b in organ_volumes:
                va, vb = organ_volumes[org_a], organ_volumes[org_b]
                if va < 1e-3 or vb < 1e-3:
                    continue
                ratio = va / vb
                for org, partner in [(org_a, org_b), (org_b, org_a)]:
                    if org in organ_results:
                        organ_results[org].ratio_partner = partner
                        organ_results[org].ratio_value = ratio if org == org_a else 1.0 / ratio
                if ratio < rmin or ratio > rmax:
                    flag = f"RATIO_OUTLIER: {org_a}/{org_b} = {ratio:.2f}, expected [{rmin:.1f}, {rmax:.1f}]"
                    for org in (org_a, org_b):
                        if org in organ_results:
                            organ_results[org].ratio_in_range = False
                            organ_results[org].ratio_flag = flag

        # Lung composite ratios
        for entry in self.lung_ratios:
            left_lobes = entry["left_lobes"]
            right_lobes = entry["right_lobes"]
            rmin, rmax = entry["min_ratio"], entry["max_ratio"]
            left_vol = sum(organ_volumes.get(l, 0.0) for l in left_lobes)
            right_vol = sum(organ_volumes.get(r, 0.0) for r in right_lobes)
            if left_vol < 1e-3 or right_vol < 1e-3:
                continue
            ratio = left_vol / right_vol
            if ratio < rmin or ratio > rmax:
                flag = f"LUNG_RATIO_OUTLIER: left/right = {ratio:.2f}, expected [{rmin:.1f}, {rmax:.1f}]"
                for lobe in left_lobes + right_lobes:
                    if lobe in organ_results:
                        organ_results[lobe].ratio_in_range = False
                        organ_results[lobe].ratio_flag = flag

    # ── Check 4: Mask overlap ───────────────────────────────────────────────

    def _check_overlap(
        self,
        organ_masks: Dict[str, np.ndarray],
        organ_results: Dict[str, OrganQCResult],
    ) -> List[str]:
        overlap_pairs = []
        if len(organ_masks) < 2:
            return overlap_pairs

        organ_names = list(organ_masks.keys())
        ref_shape = organ_masks[organ_names[0]].shape

        # Build count map (how many organs claim each voxel)
        count_map = np.zeros(ref_shape, dtype=np.int16)
        valid = []
        for name in organ_names:
            if organ_masks[name].shape == ref_shape:
                count_map += (organ_masks[name] > 0).astype(np.int16)
                valid.append(name)

        overlap_voxels = count_map > 1
        if int(np.sum(overlap_voxels)) == 0:
            return overlap_pairs

        # Find which organs overlap
        overlap_organs = [n for n in valid if np.any((organ_masks[n] > 0) & overlap_voxels)]
        for i in range(len(overlap_organs)):
            for j in range(i + 1, len(overlap_organs)):
                a, b = overlap_organs[i], overlap_organs[j]
                pairwise = int(np.sum((organ_masks[a] > 0) & (organ_masks[b] > 0)))
                if pairwise > 0:
                    pair_str = f"{a} ∩ {b} ({pairwise} vox)"
                    overlap_pairs.append(pair_str)
                    flag = f"OVERLAP: {pair_str}"
                    for org in (a, b):
                        if org in organ_results:
                            organ_results[org].has_overlap = True
                            organ_results[org].overlap_flag = flag

        return overlap_pairs


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _compute_volume_ml(mask_data: np.ndarray, voxel_dims_mm: tuple) -> float:
    """Compute volume of non-zero voxels in mL."""
    voxel_vol_mm3 = float(np.prod(voxel_dims_mm))
    return int(np.sum(mask_data > 0)) * voxel_vol_mm3 / 1000.0
