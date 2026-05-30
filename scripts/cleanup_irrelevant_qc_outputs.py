#!/usr/bin/env python3
"""
Report or remove QC output folders for unsupported anatomy/modality cases.

By default this is a dry-run report. Pass --execute to remove folders.
"""

from __future__ import annotations

import argparse
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
    filter_case_targets,
    load_completed_work,
)


QC_DIRNAMES = (LEGACY_BEST_DIRNAME, GEOMETRIC_QC_DIRNAME, MEDSEGQC_DIRNAME)


def existing_qc_dirs(case_path: str) -> list[str]:
    return [os.path.join(case_path, d) for d in QC_DIRNAMES if os.path.isdir(os.path.join(case_path, d))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segmentation-state", required=True, help="pipeline_state_v2.json")
    ap.add_argument("--segmentation-root", default=SEGMENTATION_ROOT)
    ap.add_argument("--execute", action="store_true", help="Actually remove irrelevant QC folders. Default is dry-run.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    work = load_completed_work(args.segmentation_state, args.segmentation_root)
    if args.limit is not None:
        work = work[: args.limit]

    counts = {"kept": 0, "would_remove": 0, "removed": 0, "no_qc_dirs": 0}
    removals = []
    for item in work:
        dirs = existing_qc_dirs(item.case_path)
        if not dirs:
            counts["no_qc_dirs"] += 1
            continue
        metadata, targets, skip_reason = filter_case_targets(item.case_path)
        if not skip_reason and targets:
            counts["kept"] += 1
            continue
        action = "removed" if args.execute else "would_remove"
        counts[action] += len(dirs)
        removals.append(
            {
                "case_id": item.case_id,
                "case_path": item.case_path,
                "modality": metadata.get("modality", ""),
                "anatomy": metadata.get("anatomy", ""),
                "reason": skip_reason or "no targets",
                "dirs": dirs,
            }
        )
        if args.execute:
            for path in dirs:
                shutil.rmtree(path)

    print(json.dumps({"dry_run": not args.execute, "counts": counts, "removals": removals[:50]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
