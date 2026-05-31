#!/usr/bin/env python3
"""
Backfill MRSegmentator lung_left/lung_right masks into GeometricQC outputs.

This covers cases that were migrated from legacy segmentations_best/ before the
main QC/radiomics runner learned to prefer MRSegmentator full-lung masks for
CT chest/abdomen anatomy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.constants import IMAGE_FILENAME, SEGMENTATION_ROOT
from orchestrator.batch_qc_radiomics_runner import (
    GEOMETRIC_QC_DIRNAME,
    LUNG_ELIGIBLE_CT_ANATOMIES,
    MRSEG_DIRNAME,
    candidate_filenames_for_organ,
    is_ct_like_modality,
    load_completed_work,
    mask_volume,
    metadata_from_case_path,
    write_json,
)


LUNG_ORGANS = ("lung_left", "lung_right")


def first_existing_mask(directory: str, organ: str) -> str:
    for filename in candidate_filenames_for_organ(organ):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return path
    return ""


def load_json_if_exists(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def case_is_lung_eligible_ct(case_path: str) -> tuple[bool, dict[str, Any]]:
    metadata = metadata_from_case_path(case_path)
    modality = str(metadata.get("modality", "UNKNOWN")).upper()
    anatomy = str(metadata.get("anatomy", "unknown")).lower()
    return bool(is_ct_like_modality(modality) and anatomy in LUNG_ELIGIBLE_CT_ANATOMIES), metadata


def update_selection_summary(
    *,
    summary_path: str,
    case_id: str,
    case_path: str,
    metadata: dict[str, Any],
    organ: str,
    source_path: str,
    dest_path: str,
) -> None:
    summary = load_json_if_exists(summary_path)
    if not summary:
        summary = {
            "case_id": case_id,
            "case_path": case_path,
            "metadata": metadata,
            "target_organs": [],
            "strategy": "GeometricQC",
            "selected_organs": {},
            "notes": [],
        }

    selected_organs = summary.setdefault("selected_organs", {})
    image_path = os.path.join(case_path, IMAGE_FILENAME)
    voxel_count, volume_ml, geometry = mask_volume(source_path, image_path)
    selected_organs[organ] = {
        "selected_by": "largest_volume",
        "source_tool": "MRSegmentator",
        "source_mask_path": source_path,
        "selected_mask_path": dest_path,
        "volume_ml": round(volume_ml, 6),
        "voxel_count": voxel_count,
        "image_spacing": geometry.get("image_spacing"),
        "mask_spacing": geometry.get("mask_spacing"),
        "shape_match": geometry.get("shape_match"),
        "spacing_match": geometry.get("spacing_match"),
        "affine_match": geometry.get("affine_match"),
        "geometry_match": geometry.get("geometry_match"),
        "geometry_warning": geometry.get("geometry_warning", ""),
        "qc_score": "",
        "qc_predicted_organ": "",
        "qc_error": "MedSegQC not run: organ outside supported set",
        "selection_reason": "mrsegmentator_full_lung_default_backfill",
    }

    target_organs = summary.setdefault("target_organs", [])
    if organ not in target_organs:
        target_organs.append(organ)

    notes = summary.setdefault("notes", [])
    note = "Backfilled MRSegmentator full-lung masks for CT chest/abdomen anatomy."
    if note not in notes:
        notes.append(note)

    write_json(summary_path, summary)


def backfill_case(case_id: str, case_path: str, *, dry_run: bool, overwrite: bool) -> list[str]:
    actions: list[str] = []
    ok, metadata = case_is_lung_eligible_ct(case_path)
    if not ok:
        return ["not_lung_eligible_ct"]

    geometric_dir = os.path.join(case_path, GEOMETRIC_QC_DIRNAME)
    if not os.path.isdir(geometric_dir):
        return ["missing_geometric_qc"]

    mrseg_dir = os.path.join(case_path, MRSEG_DIRNAME)
    if not os.path.isdir(mrseg_dir):
        return ["missing_mrseg_dir"]

    summary_path = os.path.join(geometric_dir, "selection_summary.json")
    for organ in LUNG_ORGANS:
        source_path = first_existing_mask(mrseg_dir, organ)
        if not source_path:
            actions.append(f"missing_mrseg_{organ}")
            continue

        dest_path = os.path.join(geometric_dir, f"{organ}.nii.gz")
        if os.path.isfile(dest_path) and not overwrite:
            actions.append(f"{organ}_target_exists")
            continue

        if dry_run:
            actions.append(f"would_copy_{organ}")
            continue

        os.makedirs(geometric_dir, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        update_selection_summary(
            summary_path=summary_path,
            case_id=case_id,
            case_path=case_path,
            metadata=metadata,
            organ=organ,
            source_path=source_path,
            dest_path=dest_path,
        )
        actions.append(f"copied_{organ}")

    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-state", required=True, help="pipeline_state_v2.json")
    parser.add_argument("--segmentation-root", default=SEGMENTATION_ROOT)
    parser.add_argument("--execute", action="store_true", help="Actually copy masks. Default is dry-run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing lung masks in GeometricQC.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    work = load_completed_work(args.segmentation_state, args.segmentation_root)
    if args.limit is not None:
        work = work[: args.limit]

    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for item in work:
        actions = backfill_case(
            item.case_id,
            item.case_path,
            dry_run=not args.execute,
            overwrite=args.overwrite,
        )
        for action in actions:
            counts[action] = counts.get(action, 0) + 1
            examples.setdefault(action, item.case_path)

    print(
        json.dumps(
            {
                "dry_run": not args.execute,
                "overwrite": args.overwrite,
                "counts": counts,
                "examples": examples,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
