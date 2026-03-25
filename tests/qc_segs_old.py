"""
================================================================================
Tier 1 Reference-Free QC for CT Segmentation Pipelines
================================================================================
Geometric quality checks for TotalSegmentator outputs on CT data.
No ground truth required. No image loading — masks only.

Active checks:
  1. Volume Plausibility  — against anatomical reference ranges
  2. Paired Volume Ratios — symmetry check for bilateral organs + lung L/R
  3. Connected Components  — fragment count & largest-component fraction (soft-tissue only)
  4. Mask Overlap          — mutual exclusivity between organ masks

Usage:
    python qc_tier1.py --filelist cases.json --output qc_report.csv --workers 16
    python qc_tier1.py --filelist raw.json --src-prefix /data/RAD/ --dst-prefix /data/seg/ --output qc.csv

Expected layout per case path:
    <case_path>/
        segmentations_separate/
            liver.nii.gz
            spleen.nii.gz
            ...
"""

import json
import os
import sys
import logging
import argparse
import traceback
import numpy as np
import nibabel as nib
from scipy import ndimage
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("qc_tier1")

# ──────────────────────────────────────────────────────────────────────────────
# ANATOMICAL REFERENCE DATA
# ──────────────────────────────────────────────────────────────────────────────
# Volume ranges in mL. Wide adult ranges to minimize false positives.
# TODO: replace with data-driven percentile bounds from your own corpus.

VOLUME_RANGES_ML: Dict[str, Tuple[float, float]] = {
    "liver":                  (800.0,  2500.0),
    "spleen":                 (40.0,   600.0),
    "kidney_left":            (80.0,   350.0),
    "kidney_right":           (80.0,   350.0),
    "pancreas":               (30.0,   180.0),
    "gallbladder":            (5.0,    100.0),
    "stomach":                (50.0,   1500.0),
    "duodenum":               (10.0,   200.0),
    "small_bowel":            (100.0,  3000.0),
    "colon":                  (100.0,  3000.0),
    "urinary_bladder":        (20.0,   800.0),
    "adrenal_gland_left":     (1.0,    20.0),
    "adrenal_gland_right":    (1.0,    20.0),
    "lung_upper_lobe_left":   (200.0,  1500.0),
    "lung_lower_lobe_left":   (300.0,  2000.0),
    "lung_upper_lobe_right":  (200.0,  1500.0),
    "lung_middle_lobe_right": (80.0,   700.0),
    "lung_lower_lobe_right":  (300.0,  2000.0),
    "heart":                  (400.0,  1200.0),
    "aorta":                  (50.0,   400.0),
    "pulmonary_artery":       (10.0,   150.0),
    "trachea":                (5.0,    60.0),
    "esophagus":              (5.0,    80.0),
    "inferior_vena_cava":     (15.0,   150.0),
    "portal_vein_and_splenic_vein": (5.0, 80.0),
    "iliac_artery_left":      (3.0,    50.0),
    "iliac_artery_right":     (3.0,    50.0),
    "iliac_vena_left":        (3.0,    50.0),
    "iliac_vena_right":       (3.0,    50.0),
}

PAIRED_ORGAN_RATIOS: List[Tuple[str, str, float, float]] = [
    ("kidney_left",  "kidney_right", 0.5, 2.0),
    ("adrenal_gland_left", "adrenal_gland_right", 0.3, 3.0),
    ("iliac_artery_left", "iliac_artery_right", 0.4, 2.5),
    ("iliac_vena_left", "iliac_vena_right", 0.4, 2.5),
]

LUNG_COMPOSITE_RATIOS = [
    (
        ["lung_upper_lobe_left", "lung_lower_lobe_left"],
        ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        0.6, 1.2,
    ),
]

# CC expectations — soft-tissue organs only. Not in this dict = no CC check.
EXPECTED_MAX_COMPONENTS: Dict[str, int] = {
    "liver": 1, "spleen": 1, "pancreas": 1, "gallbladder": 1,
    "stomach": 1, "duodenum": 1, "urinary_bladder": 1,
    "kidney_left": 1, "kidney_right": 1,
    "adrenal_gland_left": 1, "adrenal_gland_right": 1,
    "heart": 1, "aorta": 1, "esophagus": 1, "trachea": 1,
    "inferior_vena_cava": 1,
    "lung_upper_lobe_left": 1, "lung_lower_lobe_left": 1,
    "lung_upper_lobe_right": 1, "lung_middle_lobe_right": 1,
    "lung_lower_lobe_right": 1,
}

LARGEST_COMPONENT_MIN_FRACTION = 0.85

# Organs we load — union of all reference dicts. Everything else skipped.
QC_ORGANS: set = set(VOLUME_RANGES_ML.keys()) | set(EXPECTED_MAX_COMPONENTS.keys())


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class OrganQCResult:
    case_id: str
    organ: str
    case_path: str = ""
    mask_path: str = ""
    volume_ml: float = 0.0
    volume_in_range: bool = True
    volume_flag: str = ""
    ratio_partner: str = ""
    ratio_value: float = -1.0
    ratio_in_range: bool = True
    ratio_flag: str = ""
    num_components: int = 0
    largest_component_fraction: float = 1.0
    cc_flag: str = ""
    has_overlap: bool = False
    overlap_flag: str = ""
    num_flags: int = 0
    severity: str = "PASS"

    def compute_severity(self):
        flags = []
        if not self.volume_in_range and self.volume_flag:
            flags.append(self.volume_flag)
        if not self.ratio_in_range and self.ratio_flag:
            flags.append(self.ratio_flag)
        if self.cc_flag:
            flags.append(self.cc_flag)
        if self.has_overlap:
            flags.append(self.overlap_flag)
        self.num_flags = len(flags)
        if self.num_flags == 0:
            self.severity = "PASS"
        elif self.num_flags <= 2:
            self.severity = "WARN"
        else:
            self.severity = "FAIL"
        return flags


@dataclass
class CaseQCResult:
    case_id: str
    case_path: str
    image_path: str = ""
    num_organs: int = 0
    num_organs_flagged: int = 0
    total_flags: int = 0
    worst_severity: str = "PASS"
    overlap_pairs: List[str] = field(default_factory=list)
    organ_results: List[OrganQCResult] = field(default_factory=list)
    error: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def load_filelist(filelist_path: str, src_prefix: str = "", dst_prefix: str = "") -> List[str]:
    with open(filelist_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw_list = None
        for key in ["files", "paths", "cases", "data"]:
            if key in data:
                raw_list = data[key]
                break
        if raw_list is None:
            raise ValueError(f"No recognized key in filelist dict. Keys: {list(data.keys())}")
    elif isinstance(data, list):
        raw_list = data
    else:
        raise ValueError(f"Unexpected filelist format: {type(data)}")

    paths = []
    for entry in raw_list:
        if isinstance(entry, str):
            paths.append(entry)
        elif isinstance(entry, list) and len(entry) >= 1 and isinstance(entry[0], str):
            paths.append(entry[0])
        else:
            raise ValueError(f"Unrecognized entry: {entry}")

    if src_prefix and dst_prefix:
        paths = [p.replace(src_prefix, dst_prefix, 1) if p.startswith(src_prefix) else p for p in paths]

    return paths


def compute_volume_ml(mask_data: np.ndarray, voxel_dims_mm: Tuple[float, ...]) -> float:
    voxel_vol_mm3 = float(np.prod(voxel_dims_mm))
    return int(np.sum(mask_data > 0)) * voxel_vol_mm3 / 1000.0


def get_case_id(case_path: str) -> str:
    return Path(case_path).name


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 1: VOLUME PLAUSIBILITY
# ──────────────────────────────────────────────────────────────────────────────

def check_volume_plausibility(organ: str, volume_ml: float, result: OrganQCResult) -> None:
    result.volume_ml = volume_ml
    if organ not in VOLUME_RANGES_ML:
        result.volume_in_range = True
        return
    vmin, vmax = VOLUME_RANGES_ML[organ]
    if volume_ml < vmin:
        result.volume_in_range = False
        result.volume_flag = (
            f"UNDER_VOLUME: {organ} = {volume_ml:.1f} mL, "
            f"expected >= {vmin:.0f} mL ({(1 - volume_ml/vmin)*100:.0f}% below)"
        )
    elif volume_ml > vmax:
        result.volume_in_range = False
        result.volume_flag = (
            f"OVER_VOLUME: {organ} = {volume_ml:.1f} mL, "
            f"expected <= {vmax:.0f} mL ({(volume_ml/vmax - 1)*100:.0f}% above)"
        )
    else:
        result.volume_in_range = True


def check_paired_ratios(organ_volumes: Dict[str, float], organ_results: Dict[str, OrganQCResult]) -> None:
    for org_a, org_b, rmin, rmax in PAIRED_ORGAN_RATIOS:
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

    for left_lobes, right_lobes, rmin, rmax in LUNG_COMPOSITE_RATIOS:
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


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 2: CONNECTED COMPONENTS (soft-tissue organs only)
# ──────────────────────────────────────────────────────────────────────────────

def check_connected_components(organ: str, mask_data: np.ndarray, result: OrganQCResult) -> None:
    if organ not in EXPECTED_MAX_COMPONENTS:
        return
    binary = (mask_data > 0).astype(np.uint8)
    total_voxels = int(np.sum(binary))
    if total_voxels == 0:
        return
    structure = ndimage.generate_binary_structure(3, 3)
    labeled, num_cc = ndimage.label(binary, structure=structure)
    result.num_components = num_cc
    component_sizes = ndimage.sum(binary, labeled, range(1, num_cc + 1))
    largest_size = float(np.max(component_sizes))
    result.largest_component_fraction = largest_size / total_voxels
    expected_max = EXPECTED_MAX_COMPONENTS[organ]
    flags = []
    if num_cc > expected_max:
        flags.append(f"EXCESS_COMPONENTS: {organ} has {num_cc} CC (expected <= {expected_max})")
    if result.largest_component_fraction < LARGEST_COMPONENT_MIN_FRACTION and num_cc > 1:
        flags.append(f"FRAGMENTED: {organ} largest CC = {result.largest_component_fraction*100:.1f}%")
    result.cc_flag = "; ".join(flags)


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 3: MASK OVERLAP
# ──────────────────────────────────────────────────────────────────────────────

def check_mask_overlap(organ_masks: Dict[str, np.ndarray], organ_results: Dict[str, OrganQCResult]) -> List[str]:
    overlap_pairs = []
    if len(organ_masks) < 2:
        return overlap_pairs
    organ_names = list(organ_masks.keys())
    ref_shape = organ_masks[organ_names[0]].shape
    count_map = np.zeros(ref_shape, dtype=np.int16)
    valid = []
    for name in organ_names:
        if organ_masks[name].shape == ref_shape:
            count_map += (organ_masks[name] > 0).astype(np.int16)
            valid.append(name)
    overlap_voxels = count_map > 1
    if int(np.sum(overlap_voxels)) == 0:
        return overlap_pairs
    overlap_organs = [n for n in valid if np.any((organ_masks[n] > 0) & overlap_voxels)]
    for i in range(len(overlap_organs)):
        for j in range(i + 1, len(overlap_organs)):
            a, b = overlap_organs[i], overlap_organs[j]
            pairwise = int(np.sum((organ_masks[a] > 0) & (organ_masks[b] > 0)))
            if pairwise > 0:
                pair_str = f"{a} n {b} ({pairwise} vox)"
                overlap_pairs.append(pair_str)
                flag = f"OVERLAP: {pair_str}"
                for org in (a, b):
                    if org in organ_results:
                        organ_results[org].has_overlap = True
                        organ_results[org].overlap_flag = flag
    return overlap_pairs


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PER-CASE QC
# ──────────────────────────────────────────────────────────────────────────────

def run_qc_for_case(case_path: str) -> CaseQCResult:
    case_id = get_case_id(case_path)
    case_result = CaseQCResult(case_id=case_id, case_path=case_path)

    try:
        if "/CT_" not in case_path:
            case_result.error = "SKIPPED: not CT"
            return case_result

        seg_dir = os.path.join(case_path, "segmentations_separate")
        image_path = os.path.join(case_path, "image_nifti.nii.gz")

        if not os.path.isdir(seg_dir):
            case_result.error = f"No segmentations_separate/ at {case_path}"
            return case_result

        organ_files = {}
        for fname in os.listdir(seg_dir):
            if fname.endswith(".nii.gz"):
                organ_name = fname.replace(".nii.gz", "")
                if organ_name in QC_ORGANS:
                    organ_files[organ_name] = os.path.join(seg_dir, fname)

        if not organ_files:
            case_result.error = "No QC-relevant organ masks found"
            return case_result

        case_result.num_organs = len(organ_files)
        case_result.image_path = image_path

        organ_masks: Dict[str, np.ndarray] = {}
        organ_volumes: Dict[str, float] = {}
        organ_results: Dict[str, OrganQCResult] = {}

        for organ, fpath in organ_files.items():
            result = OrganQCResult(case_id=case_id, organ=organ, case_path=case_path, mask_path=fpath)
            try:
                mask_nii = nib.load(fpath)
                mask_data = mask_nii.get_fdata(dtype=np.float32)
                voxel_dims = mask_nii.header.get_zooms()[:3]
                vol_ml = compute_volume_ml(mask_data, voxel_dims)
                organ_volumes[organ] = vol_ml

                if vol_ml < 1e-3:
                    result.volume_ml = 0.0
                    result.volume_in_range = True
                    result.volume_flag = f"NOT_IN_FOV: {organ}"
                    result.severity = "PASS"
                    organ_results[organ] = result
                    continue

                organ_masks[organ] = mask_data
                check_volume_plausibility(organ, vol_ml, result)
                check_connected_components(organ, mask_data, result)

            except Exception as e:
                result.volume_flag = f"LOAD_ERROR: {str(e)}"
                result.num_flags = 1
                result.severity = "FAIL"
                logger.warning(f"[{case_id}] Error processing {organ}: {e}")

            organ_results[organ] = result

        check_paired_ratios(organ_volumes, organ_results)
        overlap_pairs = check_mask_overlap(organ_masks, organ_results)
        case_result.overlap_pairs = overlap_pairs

        for organ, result in organ_results.items():
            result.compute_severity()
            case_result.organ_results.append(result)
            case_result.total_flags += result.num_flags
            if result.num_flags > 0:
                case_result.num_organs_flagged += 1

        severities = [r.severity for r in case_result.organ_results]
        if "FAIL" in severities:
            case_result.worst_severity = "FAIL"
        elif "WARN" in severities:
            case_result.worst_severity = "WARN"
        else:
            case_result.worst_severity = "PASS"

    except Exception as e:
        case_result.error = f"Unexpected error: {traceback.format_exc()}"
        case_result.worst_severity = "ERROR"
        logger.error(f"[{case_id}] Fatal error: {e}")

    return case_result


# ──────────────────────────────────────────────────────────────────────────────
# CSV — one row per case
# ──────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "case_id", "case_path", "image_path",
    "severity", "total_flags", "num_organs_in_fov", "num_organs_flagged",
    "flagged_organs", "overlap_pairs", "error",
]


def _format_flagged_organs(case: CaseQCResult) -> str:
    parts = []
    for r in case.organ_results:
        if r.num_flags == 0:
            continue
        ftypes = []
        if r.volume_flag and not r.volume_flag.startswith("NOT_IN_FOV"):
            ftypes.append(r.volume_flag.split(":")[0] if ":" in r.volume_flag else "VOLUME")
        if r.ratio_flag:
            ftypes.append("RATIO_OUTLIER")
        if r.cc_flag:
            if "EXCESS_COMPONENTS" in r.cc_flag:
                ftypes.append(f"CC={r.num_components}")
            if "FRAGMENTED" in r.cc_flag:
                ftypes.append(f"FRAG={r.largest_component_fraction:.0%}")
        if r.overlap_flag:
            ftypes.append("OVERLAP")
        if ftypes:
            parts.append(f"{r.organ}({','.join(ftypes)})")
    return "; ".join(parts)


def _write_case_to_csv(writer: csv.DictWriter, case: CaseQCResult) -> None:
    if case.error:
        writer.writerow({
            "case_id": case.case_id, "case_path": case.case_path,
            "image_path": case.image_path, "severity": "ERROR",
            "total_flags": 0, "num_organs_in_fov": 0, "num_organs_flagged": 0,
            "flagged_organs": "", "overlap_pairs": "", "error": case.error,
        })
        return
    writer.writerow({
        "case_id": case.case_id, "case_path": case.case_path,
        "image_path": case.image_path, "severity": case.worst_severity,
        "total_flags": case.total_flags,
        "num_organs_in_fov": sum(1 for r in case.organ_results if r.volume_ml > 1e-3),
        "num_organs_flagged": case.num_organs_flagged,
        "flagged_organs": _format_flagged_organs(case),
        "overlap_pairs": "; ".join(case.overlap_pairs) if case.overlap_pairs else "",
        "error": "",
    })


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY JSON
# ──────────────────────────────────────────────────────────────────────────────

def write_summary_json(all_results: List[CaseQCResult], output_path: str) -> None:
    total = len(all_results)
    err = sum(1 for r in all_results if r.error)
    pas = sum(1 for r in all_results if r.worst_severity == "PASS")
    wrn = sum(1 for r in all_results if r.worst_severity == "WARN")
    fal = sum(1 for r in all_results if r.worst_severity == "FAIL")

    organ_flags: Dict[str, int] = {}
    type_flags: Dict[str, int] = {}
    for case in all_results:
        for r in case.organ_results:
            if r.num_flags > 0:
                organ_flags[r.organ] = organ_flags.get(r.organ, 0) + 1
            for attr in ["volume_flag", "ratio_flag", "cc_flag", "overlap_flag"]:
                val = getattr(r, attr, "")
                if val and not val.startswith("NOT_IN_FOV"):
                    ftype = val.split(":")[0] if ":" in val else val[:30]
                    type_flags[ftype] = type_flags.get(ftype, 0) + 1

    summary = {
        "total_cases": total, "pass": pas, "warn": wrn, "fail": fal, "error": err,
        "pass_rate": f"{pas/total*100:.1f}%" if total > 0 else "N/A",
        "top_flagged_organs": [{"organ": o, "count": c} for o, c in sorted(organ_flags.items(), key=lambda x: -x[1])[:15]],
        "flag_type_distribution": type_flags,
        "cases_needing_review": [
            {
                "case_id": r.case_id, "case_path": r.case_path, "image_path": r.image_path,
                "severity": r.worst_severity, "total_flags": r.total_flags,
                "flagged_organs": [
                    {"organ": o.organ, "mask_path": o.mask_path, "severity": o.severity,
                     "flags": [f for f in [o.volume_flag, o.ratio_flag, o.cc_flag, o.overlap_flag] if f and not f.startswith("NOT_IN_FOV")]}
                    for o in r.organ_results if o.num_flags > 0
                ],
            }
            for r in sorted(all_results, key=lambda x: x.total_flags, reverse=True)
            if r.worst_severity in ("WARN", "FAIL")
        ][:100],
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tier 1 QC for CT Segmentation Pipelines")
    parser.add_argument("--filelist", required=True)
    parser.add_argument("--output", default="qc_report.csv")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--src-prefix", default="")
    parser.add_argument("--dst-prefix", default="")
    args = parser.parse_args()

    if args.summary is None:
        args.summary = str(Path(args.output).parent / f"{Path(args.output).stem}_summary.json")

    logger.info(f"Loading: {args.filelist}")
    case_paths = load_filelist(args.filelist, args.src_prefix, args.dst_prefix)
    logger.info(f"{len(case_paths)} cases")

    all_results: List[CaseQCResult] = []
    t0 = time.time()
    FLUSH_EVERY = 50

    csv_file = open(args.output, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    csv_writer.writeheader()
    csv_file.flush()

    try:
        if args.workers > 1:
            logger.info(f"{args.workers} workers")
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(run_qc_for_case, cp): cp for cp in case_paths}
                it = as_completed(futures)
                if tqdm:
                    it = tqdm(it, total=len(case_paths), desc="QC", unit="case", dynamic_ncols=True)
                for i, future in enumerate(it, 1):
                    result = future.result()
                    all_results.append(result)
                    _write_case_to_csv(csv_writer, result)
                    if i % FLUSH_EVERY == 0:
                        csv_file.flush()
                    if tqdm and hasattr(it, 'set_postfix'):
                        n_err = sum(1 for r in all_results if r.error)
                        n_flag = sum(1 for r in all_results if r.worst_severity in ("WARN", "FAIL"))
                        it.set_postfix(errors=n_err, flagged=n_flag, refresh=False)
        else:
            pbar = tqdm(case_paths, desc="QC", unit="case", dynamic_ncols=True) if tqdm else None
            for i, cp in enumerate(case_paths, 1):
                result = run_qc_for_case(cp)
                all_results.append(result)
                _write_case_to_csv(csv_writer, result)
                if i % FLUSH_EVERY == 0:
                    csv_file.flush()
                if pbar:
                    n_err = sum(1 for r in all_results if r.error)
                    n_flag = sum(1 for r in all_results if r.worst_severity in ("WARN", "FAIL"))
                    pbar.set_postfix(errors=n_err, flagged=n_flag, refresh=False)
                    pbar.update(1)
            if pbar:
                pbar.close()
    finally:
        csv_file.flush()
        csv_file.close()
        logger.info(f"CSV: {args.output} ({len(all_results)} cases)")

    elapsed = time.time() - t0
    logger.info(f"Summary: {args.summary}")
    write_summary_json(all_results, args.summary)

    p = sum(1 for r in all_results if r.worst_severity == "PASS")
    w = sum(1 for r in all_results if r.worst_severity == "WARN")
    f_ = sum(1 for r in all_results if r.worst_severity == "FAIL")
    e = sum(1 for r in all_results if r.error)
    n = len(all_results)

    logger.info("=" * 50)
    logger.info(f"DONE — {n} cases in {elapsed:.0f}s")
    logger.info(f"  PASS:  {p:>5d}  ({p/n*100:.1f}%)")
    logger.info(f"  WARN:  {w:>5d}  ({w/n*100:.1f}%)")
    logger.info(f"  FAIL:  {f_:>5d}  ({f_/n*100:.1f}%)")
    if e:
        logger.info(f"  ERROR: {e:>5d}  ({e/n*100:.1f}%)")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()