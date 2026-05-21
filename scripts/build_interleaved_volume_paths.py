#!/usr/bin/env python3
"""
From 3d_scans_list.json ([path, [D,H,W]], ...):
  - Keep paths where min(D,H,W) >= 10 (heuristic volumetric series).
  - Group by cohort + patient (patient = path segments between /RAD/COHORT/ and the
    first YYYYMMDD study date; supports LUPUS .../org_hash/patient_hash/date/...).
  - Emit a flat JSON list of paths (verbatim strings), ordered by round-robin over
    cohorts LUPUS -> CONTR -> RHEUM, one full patient block at a time. Patients are
    taken in sorted order within each cohort.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

DATE_RE = re.compile(r"^\d{8}$")
COHORT_ORDER = ("LUPUS", "CONTR", "RHEUM")


def patient_cohort_from_path(path: str) -> tuple[str, str] | None:
    parts = path.split("/")
    if len(parts) < 7 or parts[1] != "data" or parts[2] != "RAD":
        return None
    cohort = parts[3]
    date_idx: int | None = None
    for i in range(4, len(parts)):
        if DATE_RE.match(parts[i]):
            date_idx = i
            break
    if date_idx is None or date_idx == 4:
        return None
    patient = "/".join(parts[4:date_idx])
    return cohort, patient


def modality_folder_from_path(path: str) -> str:
    parts = path.split("/")
    date_idx = None
    for i in range(4, len(parts)):
        if DATE_RE.match(parts[i]):
            date_idx = i
            break
    if date_idx is not None and date_idx + 1 < len(parts):
        return parts[date_idx + 1]
    return "(unknown)"


def is_volume_entry(shape: list) -> bool:
    return min(int(x) for x in shape) >= 10


def build_interleaved_paths(
    entries: list[list],
) -> tuple[list[str], dict[str, int]]:
    """Returns (ordered_paths, stats)."""
    stats = {
        "total_input": len(entries),
        "kept_volume": 0,
        "dropped_small_shape": 0,
        "skipped_bad_path": 0,
        "skipped_unknown_cohort": 0,
    }
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)

    for item in entries:
        if len(item) != 2:
            stats["skipped_bad_path"] += 1
            continue
        path, shape = item[0], item[1]
        if not isinstance(path, str) or not isinstance(shape, (list, tuple)):
            stats["skipped_bad_path"] += 1
            continue
        if not is_volume_entry(list(shape)):
            stats["dropped_small_shape"] += 1
            continue
        pc = patient_cohort_from_path(path)
        if pc is None:
            stats["skipped_bad_path"] += 1
            continue
        cohort, _patient = pc
        if cohort not in COHORT_ORDER:
            stats["skipped_unknown_cohort"] += 1
            continue
        groups[pc].append(path)
        stats["kept_volume"] += 1

    queues: dict[str, deque[list[str]]] = {}
    for cohort in COHORT_ORDER:
        items = [(patient, paths) for (c, patient), paths in groups.items() if c == cohort]
        items.sort(key=lambda t: t[0])
        block_lists = [sorted(paths) for _, paths in items]
        queues[cohort] = deque(block_lists)

    merged: list[str] = []
    while any(queues[c] for c in COHORT_ORDER):
        for cohort in COHORT_ORDER:
            if queues[cohort]:
                merged.extend(queues[cohort].popleft())

    stats["patient_blocks"] = {
        c: sum(1 for (ch, _), _paths in groups.items() if ch == c) for c in COHORT_ORDER
    }
    stats["output_paths"] = len(merged)
    return merged, stats


def prefix_coverage(paths: list[str], prefix_len: int) -> dict:
    subset = paths[:prefix_len]
    path_counts = Counter()
    patient_sets: dict[str, set[str]] = {c: set() for c in COHORT_ORDER}
    modality_by_cohort: dict[str, Counter[str]] = {c: Counter() for c in COHORT_ORDER}

    for p in subset:
        pc = patient_cohort_from_path(p)
        if pc is None:
            continue
        cohort, patient = pc
        if cohort not in COHORT_ORDER:
            continue
        path_counts[cohort] += 1
        patient_sets[cohort].add(patient)
        modality_by_cohort[cohort][modality_folder_from_path(p)] += 1

    return {
        "paths": dict(path_counts),
        "unique_patients": {c: len(patient_sets[c]) for c in COHORT_ORDER},
        "top_modalities": {
            c: modality_by_cohort[c].most_common(5) for c in COHORT_ORDER if path_counts[c]
        },
    }


def imbalance_ratio(counts: dict[str, int]) -> float | None:
    vals = [counts.get(c, 0) for c in COHORT_ORDER]
    if not any(vals):
        return None
    mn, mx = min(vals), max(vals)
    return mx / mn if mn else float("inf")


def print_simulation(paths: list[str], checkpoints: list[int]) -> None:
    n = len(paths)
    print("\n=== Coverage simulation (prefix of ordered list) ===\n")
    print(
        "Round-robin is over patient blocks (all series for one patient, then next cohort).\n"
        "While all three cohorts still have patients in the queue, unique-patient counts stay\n"
        "matched; path counts only stay roughly balanced when average volumes/patient are similar.\n"
        "After the smallest cohort runs out of patients, remaining paths are only LUPUS/RHEUM.\n"
    )
    print(f"Total paths: {n}\n")
    header = (
        f"{'stop_idx':>9} {'pct':>6} "
        f"{'L_paths':>8} {'C_paths':>8} {'R_paths':>8} "
        f"{'L_pts':>7} {'C_pts':>7} {'R_pts':>7} "
        f"{'path_ratio':>10} {'pt_ratio':>10}"
    )
    print(header)
    print("-" * len(header))

    for k in checkpoints:
        k = min(k, n)
        cov = prefix_coverage(paths, k)
        pc = cov["paths"]
        pt = cov["unique_patients"]
        pr = imbalance_ratio(pc)
        ptr = imbalance_ratio(pt)
        pct = 100.0 * k / n if n else 0.0
        print(
            f"{k:9d} {pct:5.1f}% "
            f"{pc.get('LUPUS', 0):8d} {pc.get('CONTR', 0):8d} {pc.get('RHEUM', 0):8d} "
            f"{pt['LUPUS']:7d} {pt['CONTR']:7d} {pt['RHEUM']:7d} "
            f"{pr if pr is not None else float('nan'):10.3f} "
            f"{ptr if ptr is not None else float('nan'):10.3f}"
        )

    print("\n--- Top modalities per cohort at 50% prefix ---\n")
    half = n // 2
    cov = prefix_coverage(paths, half)
    for c in COHORT_ORDER:
        print(f"{c} (first {half} paths):")
        for name, cnt in cov["top_modalities"].get(c, []):
            print(f"  {cnt:5d}  {name}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "new_data_paths" / "3d_scans_list.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "new_data_paths"
        / "3d_volume_paths_interleaved.json",
    )
    args = parser.parse_args()

    with args.input.open() as f:
        entries = json.load(f)

    paths, stats = build_interleaved_paths(entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(paths, f, indent=2)
        f.write("\n")

    print("Build summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    n = len(paths)
    checkpoints = sorted(
        {
            0,
            max(1, n // 100),
            n // 20,
            n // 10,
            n // 4,
            n // 2,
            (3 * n) // 4,
            (9 * n) // 10,
            (19 * n) // 20,
            n - 1,
            n,
        }
    )
    print_simulation(paths, checkpoints)

    return 0


if __name__ == "__main__":
    sys.exit(main())
