#!/usr/bin/env python3
"""
Test MRISegmenter on 2 MR dummy volumes (MRI-only tool).
Validates: tool runs, output shape/affine matches input, organs are non-empty.
Generates visualization PNGs.
"""

import json
import os
import shutil
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

from processing.format_utils import split_multilabel_nifti
from tools.mrisegmenter import MRISegmenterTool

DUMMY_DIR = os.path.join(PIPELINE_ROOT, "dummy_data")
OUTPUT_ROOT = os.path.join(PIPELINE_ROOT, "test_mrisegmenter_output")
TOOL_DIR_NAME = "segmentations_mrisegmenter"

# MRI-only tool — only MR cases
CASES = [
    {"name": "mr_001", "file": "001.nii.gz"},
    {"name": "mr_002", "file": "002.nii.gz"},
]

# Key organs to check (subset of MRISegmenter's 62 classes)
CHECK_ORGANS = ["liver", "spleen", "kidney_right", "kidney_left", "stomach", "pancreas"]


def run_single_case(case: dict) -> dict:
    image_path = os.path.join(DUMMY_DIR, case["file"])
    case_dir = os.path.join(OUTPUT_ROOT, case["name"])
    seg_dir = os.path.join(case_dir, TOOL_DIR_NAME)
    os.makedirs(seg_dir, exist_ok=True)

    # Copy image for visualization (grayscale background)
    shutil.copy2(image_path, os.path.join(case_dir, "image_nifti.nii.gz"))

    orig_nii = nib.load(image_path)
    orig_shape = orig_nii.shape
    orig_affine = orig_nii.affine
    print(f"\n{'='*60}")
    print(f"Case: {case['name']}")
    print(f"  Image: {image_path}")
    print(f"  Shape: {orig_shape}, voxel size: {np.abs(np.diag(orig_affine)[:3]).round(2)}")

    # Run MRISegmenter
    multilabel_path = os.path.join(seg_dir, "segmentation.nii.gz")

    cmd = [
        "conda", "run", "-n", "cdw_mriseg",
        "MRISegmentator",
        "-i", image_path,
        "-o", multilabel_path,
        "-d", "gpu",
    ]

    print(f"  Running MRISegmenter...")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - t0
        print(f"  Finished in {elapsed:.1f}s (exit code {proc.returncode})")

        if proc.returncode != 0:
            print(f"  STDERR (last 500 chars):\n{proc.stderr[-500:]}")
            return {"case": case["name"], "success": False, "error": proc.stderr[-200:]}
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return {"case": case["name"], "success": False, "error": "timeout"}

    if not os.path.isfile(multilabel_path):
        print(f"  ERROR: multilabel output not found")
        return {"case": case["name"], "success": False, "error": "no multilabel output"}

    # Split multilabel into per-organ masks
    organs_saved = split_multilabel_nifti(multilabel_path, MRISegmenterTool.LABEL_MAP, seg_dir)
    print(f"  Organs segmented: {len(organs_saved)}")

    # Validate shape/affine
    results = {
        "case": case["name"],
        "success": True,
        "original_shape": list(orig_shape),
        "num_organs": len(organs_saved),
        "organs": {},
    }

    ml_nii = nib.load(multilabel_path)
    ml_shape_match = (ml_nii.shape == orig_shape)
    ml_affine_match = np.allclose(ml_nii.affine, orig_affine, atol=1e-3)
    print(f"  Multilabel shape match: {ml_shape_match}, affine match: {ml_affine_match}")

    if not ml_shape_match or not ml_affine_match:
        print(f"  WARNING: multilabel shape={ml_nii.shape} vs orig={orig_shape}")
        results["success"] = False
        results["error"] = "shape/affine mismatch on multilabel"

    for organ in CHECK_ORGANS:
        mask_path = os.path.join(seg_dir, f"{organ}.nii.gz")
        if os.path.isfile(mask_path):
            mask_nii = nib.load(mask_path)
            voxel_count = int(np.sum(np.asarray(mask_nii.dataobj) > 0))
            voxel_size = np.abs(np.prod(np.diag(mask_nii.affine)[:3]))
            volume_ml = voxel_count * voxel_size / 1000.0
            results["organs"][organ] = {
                "voxel_count": voxel_count,
                "volume_ml": round(volume_ml, 1),
                "status": "OK",
            }
            print(f"  [+] {organ}: {voxel_count} voxels, {volume_ml:.1f} mL")
        else:
            results["organs"][organ] = {"status": "missing"}
            print(f"  [-] {organ}: NOT PRODUCED")

    results["all_organs"] = organs_saved
    return results


def visualize(case_name: str):
    from visualizations.slice_viewer import plot_case_overview
    case_dir = os.path.join(OUTPUT_ROOT, case_name)
    png_path = os.path.join(OUTPUT_ROOT, f"{case_name}_overview.png")
    result = plot_case_overview(
        case_path=case_dir,
        tool_dir=TOOL_DIR_NAME,
        output_path=png_path,
        title_prefix="MRISegmenter | ",
    )
    if result:
        print(f"  Visualization: {png_path}")
    else:
        print(f"  Visualization: no masks found")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print("MRISegmenter tool validation test")
    print(f"Output: {OUTPUT_ROOT}")

    all_results = []
    for case in CASES:
        result = run_single_case(case)
        all_results.append(result)

    # Visualize
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    for case in CASES:
        visualize(case["name"])

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_ok = True
    for r in all_results:
        if not r["success"]:
            print(f"  {r['case']}: FAILED — {r.get('error', 'unknown')}")
            all_ok = False
        else:
            ok = [o for o, info in r.get("organs", {}).items() if info.get("status") == "OK"]
            missing = [o for o, info in r.get("organs", {}).items() if info.get("status") == "missing"]
            print(f"  {r['case']}: {len(ok)}/{len(CHECK_ORGANS)} key organs, "
                  f"{r['num_organs']} total organs segmented")

    results_path = os.path.join(OUTPUT_ROOT, "mrisegmenter_test_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results: {results_path}")

    if all_ok:
        print("\nMRISEGMENTER TEST PASSED")
    else:
        print("\nMRISEGMENTER TEST FAILED — see details above")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
