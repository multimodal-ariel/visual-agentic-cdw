"""
CDW Agentic Pipeline — STAPLE Consensus Mask Generation
========================================================
Fuses per-organ segmentation masks from multiple tools into a single
consensus mask using STAPLE (Simultaneous Truth and Performance Level
Estimation) or majority-vote fallback.

STAPLE (Warfield, Zou & Wells 2004) is an EM algorithm that:
  1. Estimates a latent "true" segmentation from multiple raters
  2. Simultaneously estimates each rater's sensitivity and specificity
  3. Weights raters by their estimated reliability (unlike majority vote)

Usage — standalone
------------------
    python -m processing.staple_consensus \\
        --case dummy_outputs/0004 \\
        --method staple \\
        --min-tools 2

Usage — from pipeline
---------------------
    from processing.staple_consensus import generate_case_consensus

    result = generate_case_consensus(
        case_path="dummy_outputs/0004",
        method="staple",
        min_tools=2,
    )
    print(result["consensus_dir"])    # path to segmentations_consensus/
    print(result["organs_fused"])     # list of organs with consensus
    print(result["per_organ"])        # per-organ stats (tools, method, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from processing.format_utils import build_normalized_mask_index, normalize_organ_name
from processing.postprocessing import majority_vote, staple_fusion

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Core: generate consensus for one case
# ──────────────────────────────────────────────────────────────────────────────

def generate_case_consensus(
    case_path: str,
    seg_dirs: Optional[Dict[str, str]] = None,
    method: str = "staple",
    min_tools: int = 2,
    output_dirname: str = "segmentations_consensus",
) -> Dict[str, Any]:
    """
    Generate STAPLE (or majority-vote) consensus masks for all organs
    present in ≥ min_tools segmentation directories.

    Args:
        case_path:      Root directory for this case (contains image_nifti.nii.gz
                        and segmentations_* subdirectories).
        seg_dirs:       Optional explicit {tool_name: seg_dir_path} mapping.
                        If None, auto-discovers all segmentations_* directories.
        method:         "staple" (default) or "majority_vote".
        min_tools:      Minimum number of tools with a mask to generate consensus
                        (default 2). Organs with fewer masks are skipped.
        output_dirname: Name of the consensus output directory (default
                        "segmentations_consensus").

    Returns:
        Dict with keys:
            consensus_dir:  Path to the output directory with consensus masks.
            organs_fused:   List of organ names that received consensus masks.
            organs_skipped: List of (organ, reason) tuples for organs not fused.
            per_organ:      List of dicts with per-organ stats:
                            {organ, num_tools, tools, method, voxel_count}
            total_time_s:   Wall time for the entire operation.
    """
    t0 = time.time()
    consensus_dir = os.path.join(case_path, output_dirname)

    # Step 1: discover or use provided seg_dirs
    if seg_dirs is None:
        seg_dirs = _discover_seg_dirs(case_path)

    if len(seg_dirs) < min_tools:
        return {
            "consensus_dir": consensus_dir,
            "organs_fused": [],
            "organs_skipped": [("*", f"only {len(seg_dirs)} tool(s), need ≥{min_tools}")],
            "per_organ": [],
            "total_time_s": round(time.time() - t0, 3),
        }

    # Step 2: build normalized organ index per tool
    # canonical_name → {tool_name: filepath}
    organ_to_tools: Dict[str, Dict[str, str]] = {}
    for tool_name, seg_dir in seg_dirs.items():
        index = build_normalized_mask_index(seg_dir)
        for canonical, fpath in index.items():
            organ_to_tools.setdefault(canonical, {})[tool_name] = fpath

    # Step 3: get reference affine from image_nifti.nii.gz
    image_path = os.path.join(case_path, "image_nifti.nii.gz")
    if os.path.isfile(image_path):
        ref_nii = nib.load(image_path)
        ref_affine = ref_nii.affine
        ref_header = ref_nii.header
    else:
        # Fall back to first mask's affine
        ref_affine = None
        ref_header = None

    # Step 4: fuse each organ
    os.makedirs(consensus_dir, exist_ok=True)
    organs_fused: List[str] = []
    organs_skipped: List[Tuple[str, str]] = []
    per_organ: List[Dict[str, Any]] = []

    for organ in sorted(organ_to_tools.keys()):
        tool_paths = organ_to_tools[organ]

        if len(tool_paths) < min_tools:
            organs_skipped.append((organ, f"only {len(tool_paths)} tool(s)"))
            continue

        # Load all masks
        masks: List[np.ndarray] = []
        tool_names: List[str] = []
        first_affine = None
        first_header = None
        ref_shape = None

        for tname, fpath in sorted(tool_paths.items()):
            try:
                nii = nib.load(fpath)
                data = np.asarray(nii.dataobj, dtype=np.float32)
                binary = (data > 0).astype(np.uint8)

                if ref_shape is None:
                    ref_shape = binary.shape
                    first_affine = nii.affine
                    first_header = nii.header
                elif binary.shape != ref_shape:
                    logger.warning(
                        "Shape mismatch for %s from %s: %s vs %s — skipping",
                        organ, tname, binary.shape, ref_shape,
                    )
                    continue

                if np.sum(binary) == 0:
                    continue  # empty mask — don't count

                masks.append(binary)
                tool_names.append(tname)
            except Exception as e:
                logger.warning("Failed to load %s from %s: %s", organ, tname, e)

        if len(masks) < min_tools:
            reason = f"only {len(masks)} non-empty mask(s) after loading"
            organs_skipped.append((organ, reason))
            continue

        # Fuse
        try:
            if method == "staple":
                consensus = staple_fusion(masks)
            else:
                consensus = majority_vote(masks)
        except Exception as e:
            logger.warning("Fusion failed for %s: %s", organ, e)
            organs_skipped.append((organ, f"fusion error: {e}"))
            continue

        # Save consensus mask
        affine = ref_affine if ref_affine is not None else first_affine
        header = ref_header if ref_header is not None else first_header
        out_path = os.path.join(consensus_dir, f"{organ}.nii.gz")
        out_nii = nib.Nifti1Image(consensus.astype(np.uint8), affine, header)
        nib.save(out_nii, out_path)

        voxel_count = int(np.sum(consensus > 0))
        organs_fused.append(organ)
        per_organ.append({
            "organ": organ,
            "num_tools": len(masks),
            "tools": tool_names,
            "method": method,
            "voxel_count": voxel_count,
        })

        logger.info(
            "Consensus %s: %d tools (%s), %d voxels",
            organ, len(masks), ", ".join(tool_names), voxel_count,
        )

    elapsed = round(time.time() - t0, 3)

    # Save summary manifest
    manifest = {
        "case_path": case_path,
        "method": method,
        "min_tools": min_tools,
        "num_tools_available": len(seg_dirs),
        "tools": list(seg_dirs.keys()),
        "organs_fused": len(organs_fused),
        "organs_skipped": len(organs_skipped),
        "total_time_s": elapsed,
        "per_organ": per_organ,
    }
    manifest_path = os.path.join(consensus_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "consensus_dir": consensus_dir,
        "organs_fused": organs_fused,
        "organs_skipped": organs_skipped,
        "per_organ": per_organ,
        "total_time_s": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helper: auto-discover segmentation directories
# ──────────────────────────────────────────────────────────────────────────────

_SKIP_DIRS = {"segmentations_consensus", "radiomics_test"}


def _discover_seg_dirs(case_path: str) -> Dict[str, str]:
    """
    Find all segmentations_* directories in a case, excluding consensus itself.
    Returns {tool_name: seg_dir_path}.
    """
    seg_dirs = {}
    for entry in sorted(os.listdir(case_path)):
        full = os.path.join(case_path, entry)
        if not os.path.isdir(full):
            continue
        if not entry.startswith("segmentations_"):
            continue
        if entry in _SKIP_DIRS:
            continue
        tool_name = entry.replace("segmentations_", "")
        seg_dirs[tool_name] = full
    return seg_dirs


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate STAPLE consensus masks from multi-tool segmentations",
    )
    parser.add_argument(
        "--case", required=True,
        help="Path to case directory (contains image_nifti.nii.gz + segmentations_*/)",
    )
    parser.add_argument(
        "--method", default="staple", choices=["staple", "majority_vote"],
        help="Fusion method (default: staple)",
    )
    parser.add_argument(
        "--min-tools", type=int, default=2,
        help="Minimum number of tools for consensus (default: 2)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    result = generate_case_consensus(
        case_path=args.case,
        method=args.method,
        min_tools=args.min_tools,
    )

    print(f"\nConsensus directory: {result['consensus_dir']}")
    print(f"Organs fused: {len(result['organs_fused'])}")
    print(f"Organs skipped: {len(result['organs_skipped'])}")
    print(f"Time: {result['total_time_s']:.1f}s")

    if result["per_organ"]:
        print(f"\nPer-organ summary:")
        for entry in result["per_organ"]:
            print(f"  {entry['organ']}: {entry['num_tools']} tools, "
                  f"{entry['voxel_count']} voxels ({', '.join(entry['tools'])})")

    if result["organs_skipped"]:
        print(f"\nSkipped:")
        for organ, reason in result["organs_skipped"]:
            print(f"  {organ}: {reason}")


if __name__ == "__main__":
    main()
