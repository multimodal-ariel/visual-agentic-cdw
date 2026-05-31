#!/usr/bin/env python3
"""
Build a tiny SynthSeg smoke-test filelist from new_data_paths/3d_scans_list.json.

The output is intentionally a DICOM-root filelist (/data/RAD/...). batch_runner.py
already remaps these paths to SEGMENTATION_ROOT, so the same JSON can be used on
the clinical server without hand-editing paths.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "new_data_paths" / "3d_scans_list.json"
DEFAULT_OUTPUT = REPO_ROOT / "new_data_paths" / "synthseg_smoke_head_ct_mri.json"
DEFAULT_SUMMARY = REPO_ROOT / "new_data_paths" / "synthseg_smoke_head_ct_mri_summary.json"

DATE_RE = re.compile(r"^\d{8}$")
COHORT_ORDER = ("CONTR", "LUPUS", "RHEUM")
MODALITY_ORDER = ("CT", "MRI")
NON_DIAGNOSTIC_CONTAINS = (
    "LOCALIZER",
    "SCOUT",
    "SCREEN",
    "REFORMAT",
    "SECONDARY",
    "TOPOGRAM",
    "VRT",
    "VOLUMERENDER",
    "PROJECTION",
    "STRAIGHTEN",
)
MRI_SERIES_HINTS = (
    "MPRAGE",
    "FLAIR",
    "T1",
    "T2",
    "DWI",
    "ADC",
    "SWI",
    "SPC",
    "SPACE",
)
CT_SERIES_HINTS = (
    "HEAD",
    "BRAIN",
    "STANDARD",
    "SOFT",
)
CT_SERIES_EXCLUDE_CONTAINS = ("BONE", "Hr", "J70", "H70")


def is_volume_shape(shape: Any) -> bool:
    if not isinstance(shape, (list, tuple)) or len(shape) < 3:
        return False
    try:
        return min(int(x) for x in shape[:3]) >= 10
    except Exception:
        return False


def split_path(path: str) -> tuple[str, str, str, str, str] | None:
    parts = path.split("/")
    if len(parts) < 7 or parts[1:3] != ["data", "RAD"]:
        return None
    cohort = parts[3]
    date_idx = next((i for i in range(4, len(parts)) if DATE_RE.match(parts[i])), -1)
    if date_idx < 0 or date_idx + 2 >= len(parts):
        return None
    patient = "/".join(parts[4:date_idx])
    study = parts[date_idx + 1]
    series = "/".join(parts[date_idx + 2:])
    return cohort, patient, parts[date_idx], study, series


def classify_head_modality(study: str) -> str | None:
    study_u = study.upper()
    is_head = any(token in study_u for token in ("HEAD", "BRAIN"))
    if not is_head:
        return None
    if any(token in study_u for token in ("MRI", "MR_", "MRA", "MRV")):
        return "MRI"
    if any(token in study_u for token in ("CT", "CTA", "NCCT")):
        return "CT"
    return None


def is_non_diagnostic_series(series: str) -> bool:
    series_u = series.upper()
    tokens = [tok for tok in re.split(r"[^A-Z0-9]+", series_u) if tok]
    if any(bad in tok for bad in NON_DIAGNOSTIC_CONTAINS for tok in tokens):
        return True
    if any(re.search(r"MIP(?:S|\d|$)", tok) for tok in tokens):
        return True
    # MPRAGE is a primary 3D MRI acquisition; other MPR labels are usually reformats.
    if any(tok != "MPRAGE" and (tok.startswith("MPR") or tok.endswith("MPR")) for tok in tokens):
        return True
    return False


def is_unsuitable_head_ct_series(series: str) -> bool:
    series_u = series.upper()
    return any(token.upper() in series_u for token in CT_SERIES_EXCLUDE_CONTAINS)


def series_priority(modality: str, series: str) -> tuple[int, str]:
    series_u = series.upper()
    hints = MRI_SERIES_HINTS if modality == "MRI" else CT_SERIES_HINTS
    for idx, hint in enumerate(hints):
        if hint in series_u:
            return idx, series_u
    return len(hints), series_u


def build_smoke_list(entries: list[Any], per_modality: int) -> tuple[list[str], dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()

    for item in entries:
        if not isinstance(item, list) or len(item) != 2:
            rejection_counts["bad_entry"] += 1
            continue
        path, shape = item
        if not isinstance(path, str) or not is_volume_shape(shape):
            rejection_counts["not_volume"] += 1
            continue
        parsed = split_path(path)
        if parsed is None:
            rejection_counts["bad_path"] += 1
            continue
        cohort, patient, date, study, series = parsed
        modality = classify_head_modality(study)
        if modality is None:
            rejection_counts["not_head_ct_mri"] += 1
            continue
        if cohort not in COHORT_ORDER:
            rejection_counts["unknown_cohort"] += 1
            continue
        if is_non_diagnostic_series(series):
            rejection_counts["non_diagnostic_series"] += 1
            continue
        if modality == "CT" and is_unsuitable_head_ct_series(series):
            rejection_counts["unsuitable_ct_head_kernel"] += 1
            continue
        candidates[modality].append(
            {
                "path": path,
                "shape": shape,
                "modality": modality,
                "cohort": cohort,
                "patient": patient,
                "date": date,
                "study": study,
                "series": series,
                "priority": series_priority(modality, series),
            }
        )

    selected: list[dict[str, Any]] = []
    for modality in MODALITY_ORDER:
        seen_patients: set[tuple[str, str]] = set()
        by_cohort = {
            cohort: sorted(
                [row for row in candidates.get(modality, []) if row["cohort"] == cohort],
                key=lambda row: (row["patient"], row["priority"], row["date"], row["path"]),
            )
            for cohort in COHORT_ORDER
        }
        cohort_offsets = {cohort: 0 for cohort in COHORT_ORDER}
        while sum(1 for x in selected if x["modality"] == modality) < per_modality:
            added_this_round = False
            for cohort in COHORT_ORDER:
                rows = by_cohort[cohort]
                while cohort_offsets[cohort] < len(rows):
                    row = rows[cohort_offsets[cohort]]
                    cohort_offsets[cohort] += 1
                    patient_key = (row["cohort"], row["patient"])
                    if patient_key in seen_patients:
                        continue
                    selected.append(row)
                    seen_patients.add(patient_key)
                    added_this_round = True
                    break
                if sum(1 for x in selected if x["modality"] == modality) >= per_modality:
                    break
            if not added_this_round:
                break

    # Interleave CT/MRI for quick visual feedback in logs.
    by_modality = {mod: [row for row in selected if row["modality"] == mod] for mod in MODALITY_ORDER}
    ordered: list[dict[str, Any]] = []
    for idx in range(per_modality):
        for modality in MODALITY_ORDER:
            rows = by_modality.get(modality, [])
            if idx < len(rows):
                ordered.append(rows[idx])

    summary = {
        "input_entries": len(entries),
        "output_paths": len(ordered),
        "per_modality_requested": per_modality,
        "candidate_counts": {mod: len(candidates.get(mod, [])) for mod in MODALITY_ORDER},
        "selected_counts": dict(Counter(row["modality"] for row in ordered)),
        "selected_by_cohort": dict(Counter(row["cohort"] for row in ordered)),
        "rejection_counts": dict(rejection_counts.most_common()),
        "selected": [
            {
                "modality": row["modality"],
                "cohort": row["cohort"],
                "patient": row["patient"],
                "shape": row["shape"],
                "study": row["study"],
                "series": row["series"],
                "path": row["path"],
            }
            for row in ordered
        ],
    }
    return [row["path"] for row in ordered], summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SynthSeg head CT/MRI smoke-test filelist.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to 3d_scans_list.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON filelist path")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Output summary JSON path")
    parser.add_argument("--per-modality", type=int, default=3, help="Number of CT and MRI cases to select")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.input).open("r", encoding="utf-8") as f:
        entries = json.load(f)

    paths, summary = build_smoke_list(entries, per_modality=args.per_modality)
    if not paths:
        raise SystemExit("No SynthSeg smoke-test candidates found.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    try:
        run_filelist = output_path.relative_to(REPO_ROOT)
    except ValueError:
        run_filelist = output_path
    print("\nRun on the server:")
    print(
        "conda run -n cdw_radiomics python -m orchestrator.batch_runner "
        f"--filelist {run_filelist} "
        "--state-file logs/synthseg_smoke_state.json "
        "--output-csv logs/synthseg_smoke_results.csv "
        "--gpus 0 --workers 1 --no-llm --tools SynthSeg --tool-timeout 900"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
