#!/usr/bin/env python3
"""
Quick test: run TextMedSeg3D on 2 CT + 2 MR dummy volumes,
then verify that output masks match the original image shape/affine.
"""

import json
import os
import sys
import tempfile
import subprocess
import time

import nibabel as nib
import numpy as np

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

DUMMY_DIR = os.path.join(PIPELINE_ROOT, "dummy_data")
OUTPUT_ROOT = os.path.join(PIPELINE_ROOT, "test_textmedseg3d_output")

# 2 CT + 2 MR from dummy_data
CASES = [
    {"name": "ct_0002", "file": "0002.nii.gz", "modality": "ct"},
    {"name": "ct_0003", "file": "0003.nii.gz", "modality": "ct"},
    {"name": "mr_001",  "file": "001.nii.gz",  "modality": "mri"},
    {"name": "mr_002",  "file": "002.nii.gz",  "modality": "mri"},
]

# Common abdominal organs to segment
TARGET_ORGANS = ["liver", "spleen", "left kidney", "right kidney", "stomach", "pancreas"]

TEXTMEDSEG_DIR = os.path.join(PIPELINE_ROOT, "external", "TextMedSeg3D")
CKPT_ROOT = os.path.join(PIPELINE_ROOT, "checkpoints", "TextMedSeg3D")

# Resolve checkpoint — Pro if available, else Nano
if os.path.isfile(os.path.join(CKPT_ROOT, "Pro", "SAT_Pro.pth")):
    CHECKPOINT = os.path.join(CKPT_ROOT, "Pro", "SAT_Pro.pth")
    TEXT_ENCODER_CKPT = os.path.join(CKPT_ROOT, "Pro", "text_encoder.pth")
    VISION_BACKBONE = "UNET-L"
    print(f"Using SAT-Pro checkpoint")
else:
    CHECKPOINT = os.path.join(CKPT_ROOT, "Nano", "nano_cvpr25_v0.pth")
    TEXT_ENCODER_CKPT = os.path.join(CKPT_ROOT, "Nano", "text_encoder_cvpr25_v0.pth")
    VISION_BACKBONE = "UNET"
    print(f"Using SAT-Nano checkpoint")


def run_single_case(case: dict, tmp_dir: str) -> dict:
    """Run TextMedSeg3D on a single case and return results."""
    image_path = os.path.join(DUMMY_DIR, case["file"])
    case_output = os.path.join(OUTPUT_ROOT, case["name"])
    os.makedirs(case_output, exist_ok=True)

    # Load original image info
    orig_nii = nib.load(image_path)
    orig_shape = orig_nii.shape
    orig_affine = orig_nii.affine
    print(f"\n{'='*60}")
    print(f"Case: {case['name']} ({case['modality'].upper()})")
    print(f"  Image: {image_path}")
    print(f"  Original shape: {orig_shape}")
    print(f"  Original affine diagonal: {np.diag(orig_affine)[:3]}")

    # Write JSONL
    jsonl_path = os.path.join(tmp_dir, f"{case['name']}.jsonl")
    sample = {
        "image": image_path,
        "label": TARGET_ORGANS,
        "modality": case["modality"],
        "dataset": "TestRun",
    }
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(sample) + "\n")

    rcd_dir = os.path.join(tmp_dir, f"rcd_{case['name']}")
    os.makedirs(rcd_dir, exist_ok=True)

    cmd = [
        "conda", "run", "-n", "cdw_textmedseg",
        "torchrun", "--nproc_per_node=1",
        "inference.py",
        "--datasets_jsonl", jsonl_path,
        "--vision_backbone", VISION_BACKBONE,
        "--checkpoint", CHECKPOINT,
        "--text_encoder", "ours",
        "--text_encoder_checkpoint", TEXT_ENCODER_CKPT,
        "--rcd_dir", rcd_dir,
    ]

    print(f"  Running TextMedSeg3D...")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=TEXTMEDSEG_DIR,
            capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - t0
        print(f"  Finished in {elapsed:.1f}s (exit code {proc.returncode})")

        if proc.returncode != 0:
            print(f"  STDERR (last 500 chars):\n{proc.stderr[-500:]}")
            return {"case": case["name"], "success": False, "error": proc.stderr[-200:]}
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return {"case": case["name"], "success": False, "error": "timeout"}

    # Collect outputs
    image_stem = case["file"].replace(".nii.gz", "")
    seg_dir = os.path.join(rcd_dir, "TestRun", f"seg_{image_stem}")

    results = {
        "case": case["name"],
        "modality": case["modality"],
        "success": True,
        "original_shape": list(orig_shape),
        "organs": {},
    }

    if not os.path.isdir(seg_dir):
        print(f"  WARNING: No output directory at {seg_dir}")
        results["success"] = False
        results["error"] = f"No output dir: {seg_dir}"
        return results

    for organ in TARGET_ORGANS:
        mask_path = os.path.join(seg_dir, f"{organ}.nii.gz")
        if os.path.isfile(mask_path):
            mask_nii = nib.load(mask_path)
            mask_shape = mask_nii.shape
            mask_affine = mask_nii.affine
            shape_match = (mask_shape == orig_shape)
            affine_close = np.allclose(mask_affine, orig_affine, atol=1e-3)
            voxel_count = int(np.sum(mask_nii.get_fdata() > 0.5))
            volume_ml = voxel_count * np.abs(np.prod(np.diag(mask_affine)[:3])) / 1000.0

            # Copy to output dir for visualization
            import shutil
            shutil.copy2(mask_path, os.path.join(case_output, f"{organ}.nii.gz"))

            status = "OK" if (shape_match and affine_close) else "MISMATCH"
            results["organs"][organ] = {
                "mask_shape": list(mask_shape),
                "shape_match": shape_match,
                "affine_match": affine_close,
                "voxel_count": voxel_count,
                "volume_ml": round(volume_ml, 1),
                "status": status,
            }
            sym = "+" if status == "OK" else "X"
            print(f"  [{sym}] {organ}: shape={mask_shape} (match={shape_match}), "
                  f"affine_match={affine_close}, vol={volume_ml:.1f}mL, voxels={voxel_count}")
        else:
            print(f"  [-] {organ}: NOT PRODUCED")
            results["organs"][organ] = {"status": "missing"}

    # Also copy the original image for visualization
    import shutil
    shutil.copy2(image_path, os.path.join(case_output, "image_nifti.nii.gz"))

    return results


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"TextMedSeg3D resampling validation test")
    print(f"Output: {OUTPUT_ROOT}")

    all_results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for case in CASES:
            result = run_single_case(case, tmp_dir)
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_ok = True
    for r in all_results:
        if not r["success"]:
            print(f"  {r['case']}: FAILED — {r.get('error', 'unknown')}")
            all_ok = False
            continue

        mismatches = [o for o, info in r.get("organs", {}).items()
                      if info.get("status") == "MISMATCH"]
        missing = [o for o, info in r.get("organs", {}).items()
                   if info.get("status") == "missing"]
        ok = [o for o, info in r.get("organs", {}).items()
              if info.get("status") == "OK"]

        if mismatches:
            print(f"  {r['case']}: SHAPE/AFFINE MISMATCH on {mismatches}")
            all_ok = False
        elif missing and not ok:
            print(f"  {r['case']}: No organs produced")
            all_ok = False
        else:
            print(f"  {r['case']}: OK — {len(ok)} organs matched, {len(missing)} missing")

    # Save results JSON
    results_path = os.path.join(OUTPUT_ROOT, "resampling_test_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results: {results_path}")

    if all_ok:
        print("\nRESAMPLING TEST PASSED — all output masks match input image space")
    else:
        print("\nRESAMPLING TEST FAILED — see details above")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
