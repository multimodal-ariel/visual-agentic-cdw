#!/usr/bin/env python3
"""
Migrate legacy segmentations_best/ outputs to segmentations_filtered_GeometricQC/.

This is intentionally a standalone, dry-run-first filesystem utility. The main
batch runner never performs one-time folder migrations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.constants import SEGMENTATION_ROOT
from orchestrator.batch_qc_radiomics_runner import (
    GEOMETRIC_QC_DIRNAME,
    LEGACY_BEST_DIRNAME,
    MEDSEGQC_DIRNAME,
    MEDSEGQC_ORGANS,
    load_completed_work,
    write_json,
    write_rows_csv,
)


def dir_nonempty(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def best_medsegqc_rows(qc_csv: str) -> dict[str, dict]:
    if not os.path.isfile(qc_csv):
        return {}
    best: dict[str, dict] = {}
    with open(qc_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            organ = row.get("organ", "")
            if organ not in MEDSEGQC_ORGANS:
                continue
            mask_path = row.get("mask_path", "")
            if not mask_path or not os.path.isfile(mask_path):
                continue
            try:
                score = float(row.get("qc_score", "nan"))
            except ValueError:
                continue
            if score != score:
                continue
            prev = best.get(organ)
            if prev is None or score > float(prev.get("qc_score", "-inf")):
                copied = dict(row)
                copied["qc_score"] = score
                best[organ] = copied
    return best


def write_medsegqc_from_rows(case_path: str, rows_by_organ: dict[str, dict], *, dry_run: bool) -> str:
    medsegqc_dir = os.path.join(case_path, MEDSEGQC_DIRNAME)
    if os.path.exists(medsegqc_dir):
        return "medsegqc_target_exists"
    if not rows_by_organ:
        return "medsegqc_no_winner"
    if dry_run:
        return "would_create_medsegqc"

    os.makedirs(medsegqc_dir, exist_ok=True)
    selected = {}
    output_rows = []
    for organ, row in rows_by_organ.items():
        src = row["mask_path"]
        dest = os.path.join(medsegqc_dir, f"{organ}.nii.gz")
        shutil.copy2(src, dest)
        out = dict(row)
        out["selected_by_highest_qc"] = True
        out["medsegqc_selected_mask_path"] = dest
        out["selection_reason"] = "highest_medsegqc_score_from_legacy_qc_scores"
        output_rows.append(out)
        selected[organ] = {
            "selected_by": "highest_medsegqc_score",
            "source_tool": row.get("source_tool", ""),
            "source_mask_path": src,
            "selected_mask_path": dest,
            "qc_score": row.get("qc_score"),
            "qc_predicted_organ": row.get("qc_predicted_organ", ""),
            "selection_reason": "highest_medsegqc_score_from_legacy_qc_scores",
        }

    write_rows_csv(os.path.join(medsegqc_dir, "qc_scores.csv"), output_rows)
    write_json(
        os.path.join(medsegqc_dir, "selection_summary.json"),
        {
            "case_path": case_path,
            "strategy": "MedSegQC",
            "selected_organs": selected,
            "notes": [
                "Migrated from legacy qc_scores.csv by selecting highest numeric qc_score.",
                "No MedSegQC inference was rerun by this migration script.",
            ],
        },
    )
    return "created_medsegqc"


def migrate_case(case_path: str, *, mode: str, dry_run: bool) -> list[str]:
    legacy = os.path.join(case_path, LEGACY_BEST_DIRNAME)
    target = os.path.join(case_path, GEOMETRIC_QC_DIRNAME)
    actions: list[str] = []
    if not dir_nonempty(legacy):
        actions.append("missing_legacy")
    else:
        if os.path.exists(target):
            actions.append("geometric_target_exists")
        elif dry_run:
            actions.append(f"would_{mode}_geometric")
        elif mode == "copy":
            shutil.copytree(legacy, target)
            actions.append("copied_geometric")
        elif mode == "move":
            shutil.move(legacy, target)
            actions.append("moved_geometric")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    qc_csv_candidates = [
        os.path.join(legacy, "qc_scores.csv"),
        os.path.join(target, "qc_scores.csv"),
    ]
    rows_by_organ = {}
    for qc_csv in qc_csv_candidates:
        rows_by_organ = best_medsegqc_rows(qc_csv)
        if rows_by_organ:
            break
    actions.append(write_medsegqc_from_rows(case_path, rows_by_organ, dry_run=dry_run))
    return actions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segmentation-state", required=True, help="pipeline_state_v2.json")
    ap.add_argument("--segmentation-root", default=SEGMENTATION_ROOT)
    ap.add_argument("--mode", choices=["copy", "move"], default="copy")
    ap.add_argument("--execute", action="store_true", help="Actually copy/move. Default is dry-run.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    work = load_completed_work(args.segmentation_state, args.segmentation_root)
    if args.limit is not None:
        work = work[: args.limit]

    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for item in work:
        actions = migrate_case(item.case_path, mode=args.mode, dry_run=not args.execute)
        for action in actions:
            counts[action] = counts.get(action, 0) + 1
            examples.setdefault(action, item.case_path)

    print(json.dumps({"dry_run": not args.execute, "mode": args.mode, "counts": counts, "examples": examples}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
