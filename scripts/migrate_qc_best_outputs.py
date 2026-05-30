#!/usr/bin/env python3
"""
Migrate legacy segmentations_best/ outputs to segmentations_filtered_GeometricQC/.

This is intentionally a standalone, dry-run-first filesystem utility. The main
batch runner never performs one-time folder migrations.
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
    load_completed_work,
)


def dir_nonempty(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def migrate_case(case_path: str, *, mode: str, dry_run: bool) -> str:
    legacy = os.path.join(case_path, LEGACY_BEST_DIRNAME)
    target = os.path.join(case_path, GEOMETRIC_QC_DIRNAME)
    if not dir_nonempty(legacy):
        return "missing_legacy"
    if os.path.exists(target):
        return "target_exists"
    if dry_run:
        return f"would_{mode}"
    if mode == "copy":
        shutil.copytree(legacy, target)
    elif mode == "move":
        shutil.move(legacy, target)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return mode


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
        action = migrate_case(item.case_path, mode=args.mode, dry_run=not args.execute)
        counts[action] = counts.get(action, 0) + 1
        examples.setdefault(action, item.case_path)

    print(json.dumps({"dry_run": not args.execute, "mode": args.mode, "counts": counts, "examples": examples}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
