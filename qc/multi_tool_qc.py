from __future__ import annotations

import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

from qc.dataclasses import MultiToolQCResult, OrganAgreement

logger = logging.getLogger(__name__)


def _binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return mask > threshold


def _dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a_bin = _binarize(mask_a)
    b_bin = _binarize(mask_b)
    intersection = int(np.sum(a_bin & b_bin))
    total = int(np.sum(a_bin)) + int(np.sum(b_bin))
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def _compute_volume_ml(mask_data: np.ndarray, voxel_dims_mm: tuple) -> float:
    voxel_vol_mm3 = float(np.prod(voxel_dims_mm))
    return float(np.sum(_binarize(mask_data)) * voxel_vol_mm3 / 1000.0)


def _compute_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Centroid in voxel coordinates as [H, W, D].
    """
    coords = np.argwhere(_binarize(mask))
    if coords.size == 0:
        return None
    return coords.mean(axis=0)


def _centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray, spacing_hwd: Optional[tuple] = None) -> float:
    ca = _compute_centroid(mask_a)
    cb = _compute_centroid(mask_b)

    if ca is None or cb is None:
        return float("inf")

    diff = ca - cb
    if spacing_hwd is not None:
        diff = diff * np.asarray(spacing_hwd, dtype=float)

    return float(np.linalg.norm(diff))


def _compute_tight_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Tight-fit 3D bbox with inclusive bounds:
    (hmin, hmax, wmin, wmax, dmin, dmax)
    """
    coords = np.argwhere(_binarize(mask))
    if coords.size == 0:
        return None

    hmin, wmin, dmin = coords.min(axis=0)
    hmax, wmax, dmax = coords.max(axis=0)
    return int(hmin), int(hmax), int(wmin), int(wmax), int(dmin), int(dmax)


def _bbox_volume(bbox: Tuple[int, int, int, int, int, int]) -> int:
    hmin, hmax, wmin, wmax, dmin, dmax = bbox
    return (hmax - hmin + 1) * (wmax - wmin + 1) * (dmax - dmin + 1)


def _bbox_intersection_volume(
    bbox_a: Tuple[int, int, int, int, int, int],
    bbox_b: Tuple[int, int, int, int, int, int],
) -> int:
    ah0, ah1, aw0, aw1, ad0, ad1 = bbox_a
    bh0, bh1, bw0, bw1, bd0, bd1 = bbox_b

    ih0 = max(ah0, bh0)
    ih1 = min(ah1, bh1)
    iw0 = max(aw0, bw0)
    iw1 = min(aw1, bw1)
    id0 = max(ad0, bd0)
    id1 = min(ad1, bd1)

    if ih0 > ih1 or iw0 > iw1 or id0 > id1:
        return 0

    return (ih1 - ih0 + 1) * (iw1 - iw0 + 1) * (id1 - id0 + 1)


def _bbox_overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    bbox_a = _compute_tight_bbox(mask_a)
    bbox_b = _compute_tight_bbox(mask_b)

    if bbox_a is None or bbox_b is None:
        return 0.0

    inter = _bbox_intersection_volume(bbox_a, bbox_b)
    vol_a = _bbox_volume(bbox_a)
    vol_b = _bbox_volume(bbox_b)

    denom = min(vol_a, vol_b)
    if denom == 0:
        return 0.0

    return float(inter / denom)


def _signed_distance_map(mask: np.ndarray, spacing_hwd: Optional[tuple] = None) -> np.ndarray:
    bin_mask = _binarize(mask)

    if spacing_hwd is None:
        inside = distance_transform_edt(bin_mask)
        outside = distance_transform_edt(~bin_mask)
    else:
        inside = distance_transform_edt(bin_mask, sampling=spacing_hwd)
        outside = distance_transform_edt(~bin_mask, sampling=spacing_hwd)

    return inside - outside


class MultiToolQC:
    """
    Tier 2 cross-tool agreement checks + organ-level fusion metadata.
    """

    def __init__(
        self,
        dice_threshold: float = 0.70,
        bbox_overlap_threshold: float = 0.20,
        centroid_dist_mm_threshold: Optional[float] = None,
        min_present_volume_ml: float = 1e-3,
    ):
        self.dice_threshold = dice_threshold
        self.bbox_overlap_threshold = bbox_overlap_threshold
        self.centroid_dist_mm_threshold = centroid_dist_mm_threshold
        self.min_present_volume_ml = min_present_volume_ml

    def run(
        self,
        case_path: str,
        seg_dirs: Dict[str, str],
    ) -> MultiToolQCResult:
        case_id = Path(case_path).name
        result = MultiToolQCResult(case_id=case_id, case_path=case_path)

        valid_dirs = {t: d for t, d in seg_dirs.items() if os.path.isdir(d)}
        if len(valid_dirs) < 2:
            return result

        tool_masks: Dict[str, Dict[str, Tuple[np.ndarray, tuple]]] = {}
        for tool_name, seg_dir in valid_dirs.items():
            masks = self._load_tool_masks(seg_dir)
            if masks:
                tool_masks[tool_name] = masks

        if len(tool_masks) < 2:
            return result

        tool_names = list(tool_masks.keys())
        all_agreements: List[OrganAgreement] = []
        dice_values: List[float] = []
        bbox_values: List[float] = []
        centroid_values: List[float] = []

        for tool_a, tool_b in combinations(tool_names, 2):
            result.tool_pairs_checked += 1
            pair_agreements = self._compare_tool_pair(
                tool_a, tool_masks[tool_a],
                tool_b, tool_masks[tool_b],
                case_id,
            )
            all_agreements.extend(pair_agreements)

            for ag in pair_agreements:
                dice_values.append(float(ag.dice))
                bbox_values.append(float(ag.bbox_overlap))
                if np.isfinite(ag.centroid_distance_mm):
                    centroid_values.append(float(ag.centroid_distance_mm))

        result.agreements = all_agreements
        result.mean_dice = float(np.mean(dice_values)) if dice_values else 0.0
        result.mean_bbox_overlap = float(np.mean(bbox_values)) if bbox_values else 0.0
        result.mean_centroid_distance_mm = float(np.mean(centroid_values)) if centroid_values else 0.0

        result.organs_with_agreement = sum(1 for a in all_agreements if a.agrees)
        result.organs_with_disagreement = sum(1 for a in all_agreements if not a.agrees)

        fused_soft_by_organ: Dict[str, np.ndarray] = {}
        fused_binary_by_organ: Dict[str, np.ndarray] = {}
        kept_tools_by_organ: Dict[str, List[str]] = {}
        rejected_tools_by_organ: Dict[str, List[str]] = {}

        all_organs = sorted(set().union(*[set(m.keys()) for m in tool_masks.values()]))

        for organ in all_organs:
            organ_entries = []
            for tool_name, organ_map in tool_masks.items():
                if organ not in organ_map:
                    continue
                data, voxel_dims = organ_map[organ]
                vol_ml = _compute_volume_ml(data, voxel_dims)
                if vol_ml < self.min_present_volume_ml:
                    continue
                organ_entries.append((tool_name, data, voxel_dims))

            if len(organ_entries) < 2:
                continue

            kept_entries, rejected_entries = self._filter_organ_outliers(organ_entries)

            if len(kept_entries) < 2:
                continue

            kept_tools_by_organ[organ] = [t for t, _, _ in kept_entries]
            rejected_tools_by_organ[organ] = [t for t, _, _ in rejected_entries]

            spacing_hwd = self._to_spacing_hwd(kept_entries[0][2])
            soft_map, binary_map = self._fuse_organ_masks(
                [m for _, m, _ in kept_entries],
                spacing_hwd=spacing_hwd,
            )

            fused_soft_by_organ[organ] = soft_map.astype(np.float32)
            fused_binary_by_organ[organ] = binary_map.astype(np.uint8)

        result.fused_soft_by_organ = fused_soft_by_organ
        result.fused_binary_by_organ = fused_binary_by_organ
        result.kept_tools_by_organ = kept_tools_by_organ
        result.rejected_tools_by_organ = rejected_tools_by_organ

        return result

    def _load_tool_masks(self, seg_dir: str) -> Dict[str, Tuple[np.ndarray, tuple]]:
        from processing.format_utils import normalize_organ_name

        _SKIP_STEMS = {
            "statistics", "combined", "multilabel", "multilabel_seg",
            "image_nifti_seg", "segmentation", "manifest",
        }

        masks: Dict[str, Tuple[np.ndarray, tuple]] = {}
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
        shared_organs = set(masks_a.keys()) & set(masks_b.keys())
        agreements: List[OrganAgreement] = []

        for organ in sorted(shared_organs):
            data_a, dims_a = masks_a[organ]
            data_b, dims_b = masks_b[organ]

            if data_a.shape != data_b.shape:
                logger.warning(
                    "[%s] Shape mismatch for %s: %s vs %s (%s vs %s)",
                    case_id, organ, tool_a, tool_b, data_a.shape, data_b.shape,
                )
                continue

            vol_a = _compute_volume_ml(data_a, dims_a)
            vol_b = _compute_volume_ml(data_b, dims_b)

            if vol_a < self.min_present_volume_ml and vol_b < self.min_present_volume_ml:
                continue

            ag = OrganAgreement(
                organ=organ,
                tool_a=tool_a,
                tool_b=tool_b,
                volume_a_ml=vol_a,
                volume_b_ml=vol_b,
            )

            ag.dice = _dice_coefficient(data_a, data_b)
            spacing_hwd = self._to_spacing_hwd(dims_a)
            ag.bbox_overlap = _bbox_overlap_ratio(data_a, data_b)
            ag.centroid_distance_mm = _centroid_distance(data_a, data_b, spacing_hwd=spacing_hwd)

            flags = []

            if ag.dice < self.dice_threshold:
                flags.append(
                    f"LOW_DICE: {organ} {tool_a}/{tool_b} = {ag.dice:.3f} "
                    f"(threshold {self.dice_threshold:.2f})"
                )

            if ag.bbox_overlap < self.bbox_overlap_threshold:
                flags.append(
                    f"LOW_BBOX_OVERLAP: {organ} {tool_a}/{tool_b} = {ag.bbox_overlap:.3f} "
                    f"(threshold {self.bbox_overlap_threshold:.2f})"
                )

            if (
                self.centroid_dist_mm_threshold is not None
                and np.isfinite(ag.centroid_distance_mm)
                and ag.centroid_distance_mm > self.centroid_dist_mm_threshold
            ):
                flags.append(
                    f"HIGH_CENTROID_DISTANCE: {organ} {tool_a}/{tool_b} = "
                    f"{ag.centroid_distance_mm:.2f} mm "
                    f"(threshold {self.centroid_dist_mm_threshold:.2f} mm)"
                )

            if (vol_a < self.min_present_volume_ml) != (vol_b < self.min_present_volume_ml):
                flags.append(
                    f"PRESENCE_MISMATCH: {organ} found by "
                    f"{tool_a if vol_a >= self.min_present_volume_ml else tool_b} "
                    f"but not "
                    f"{tool_b if vol_a >= self.min_present_volume_ml else tool_a}"
                )

            if flags:
                ag.agrees = False
                ag.flag = "; ".join(flags)

            agreements.append(ag)

        return agreements

    def _filter_organ_outliers(
        self,
        organ_entries: List[Tuple[str, np.ndarray, tuple]],
    ) -> Tuple[List[Tuple[str, np.ndarray, tuple]], List[Tuple[str, np.ndarray, tuple]]]:
        if len(organ_entries) < 2:
            return organ_entries, []

        n = len(organ_entries)
        mean_centroid_dist = np.zeros(n, dtype=float)
        mean_bbox_overlap = np.zeros(n, dtype=float)

        spacing_hwd = self._to_spacing_hwd(organ_entries[0][2])

        for i in range(n):
            cdist_vals = []
            bbox_vals = []

            for j in range(n):
                if i == j:
                    continue

                _, mask_i, _ = organ_entries[i]
                _, mask_j, _ = organ_entries[j]

                cdist_vals.append(_centroid_distance(mask_i, mask_j, spacing_hwd=spacing_hwd))
                bbox_vals.append(_bbox_overlap_ratio(mask_i, mask_j))

            mean_centroid_dist[i] = float(np.mean(cdist_vals)) if cdist_vals else 0.0
            mean_bbox_overlap[i] = float(np.mean(bbox_vals)) if bbox_vals else 1.0

        if self.centroid_dist_mm_threshold is None:
            q1 = float(np.percentile(mean_centroid_dist, 25))
            q3 = float(np.percentile(mean_centroid_dist, 75))
            iqr = q3 - q1
            centroid_thresh = q3 + 1.5 * iqr
        else:
            centroid_thresh = self.centroid_dist_mm_threshold

        kept, rejected = [], []
        for idx, entry in enumerate(organ_entries):
            low_bbox = mean_bbox_overlap[idx] < self.bbox_overlap_threshold
            high_centroid = mean_centroid_dist[idx] > centroid_thresh if np.isfinite(centroid_thresh) else False

            reject = low_bbox and high_centroid

            if reject:
                rejected.append(entry)
            else:
                kept.append(entry)

        if len(kept) < 2 and len(organ_entries) >= 2:
            ranked_idx = np.argsort(-mean_bbox_overlap)
            keep_idx = set(ranked_idx[:2].tolist())
            kept = [organ_entries[i] for i in range(n) if i in keep_idx]
            rejected = [organ_entries[i] for i in range(n) if i not in keep_idx]

        return kept, rejected

    def _fuse_organ_masks(
        self,
        masks: List[np.ndarray],
        spacing_hwd: Optional[tuple] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bin_masks = np.stack([_binarize(m).astype(np.float32) for m in masks], axis=0)
        soft_map = np.mean(bin_masks, axis=0)

        sdms = np.stack(
            [_signed_distance_map(m, spacing_hwd=spacing_hwd).astype(np.float32) for m in masks],
            axis=0,
        )
        mean_sdm = np.mean(sdms, axis=0)
        binary_map = mean_sdm > 0

        return soft_map, binary_map

    @staticmethod
    def _to_spacing_hwd(voxel_dims: tuple) -> tuple:
        if len(voxel_dims) < 3:
            return (1.0, 1.0, 1.0)
        return tuple(float(x) for x in voxel_dims[:3])