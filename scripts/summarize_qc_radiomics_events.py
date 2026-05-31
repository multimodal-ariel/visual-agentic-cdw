#!/usr/bin/env python3
"""
Summarize the append-only QC/radiomics event log.

The batch runner may write multiple events for the same case across reruns.
This script treats the last valid event for each case_id as the current state
and reports a compact snapshot of completed, skipped, failed, and reprocessable
cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from config.constants import LOG_DIR
except Exception:
    LOG_DIR = str(REPO_ROOT / "logs")


DEFAULT_EVENTS_PATH = Path(LOG_DIR) / "qc_radiomics_events.jsonl"


def load_latest_events(events_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "events_path": str(events_path),
        "raw_events": 0,
        "valid_events": 0,
        "invalid_json_lines": 0,
        "missing_case_id_lines": 0,
    }

    with events_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["raw_events"] += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json_lines"] += 1
                continue

            case_id = str(event.get("case_id", "")).strip()
            if not case_id:
                stats["missing_case_id_lines"] += 1
                continue

            event["_line_number"] = line_number
            latest[case_id] = event
            stats["valid_events"] += 1

    return latest, stats


def load_segmentation_completed_cases(segmentation_state_path: Path) -> set[str]:
    with segmentation_state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    cases = state.get("cases", {})
    return {
        str(case_id)
        for case_id, entry in cases.items()
        if isinstance(entry, dict) and entry.get("status") == "completed"
    }


def cohort_from_case_id(case_id: str) -> str:
    return case_id.split("__", 1)[0] if "__" in case_id else "UNKNOWN"


def reason_from_event(event: dict[str, Any]) -> str:
    reason = str(event.get("error") or event.get("skip_reason") or "").strip()
    return reason or "(no reason recorded)"


def should_reprocess(event: dict[str, Any]) -> bool:
    status = str(event.get("status", "")).lower()
    if status in {"failed", "running"}:
        return True
    if status == "completed":
        return not (bool(event.get("geometric_done")) and bool(event.get("medsegqc_done")))
    return False


def build_summary(
    latest: dict[str, dict[str, Any]],
    log_stats: dict[str, Any],
    *,
    segmentation_completed_cases: set[str] | None = None,
    example_limit: int = 10,
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    cohort_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    anatomy_counts: Counter[str] = Counter()
    status_by_modality: dict[str, Counter[str]] = defaultdict(Counter)
    status_by_anatomy: dict[str, Counter[str]] = defaultdict(Counter)
    skipped_by_reason: Counter[str] = Counter()
    failed_by_reason: Counter[str] = Counter()
    reprocess_examples: list[dict[str, Any]] = []

    completed_geometric_done = 0
    completed_medsegqc_done = 0
    completed_both_done = 0
    completed_selected_organs = 0
    completed_candidate_rows = 0

    for case_id, event in latest.items():
        status = str(event.get("status", "unknown")).lower() or "unknown"
        modality = str(event.get("modality", "") or "unknown").upper()
        anatomy = str(event.get("anatomy", "") or "unknown").lower()

        status_counts[status] += 1
        cohort_counts[cohort_from_case_id(case_id)] += 1
        modality_counts[modality] += 1
        anatomy_counts[anatomy] += 1
        status_by_modality[modality][status] += 1
        status_by_anatomy[anatomy][status] += 1

        if status == "skipped":
            skipped_by_reason[reason_from_event(event)] += 1
        elif status == "failed":
            failed_by_reason[reason_from_event(event)] += 1

        if status == "completed":
            geometric_done = bool(event.get("geometric_done"))
            medsegqc_done = bool(event.get("medsegqc_done"))
            completed_geometric_done += int(geometric_done)
            completed_medsegqc_done += int(medsegqc_done)
            completed_both_done += int(geometric_done and medsegqc_done)
            completed_selected_organs += int(event.get("geometric_selected_count") or 0)
            completed_candidate_rows += int(event.get("candidate_rows") or 0)

        if should_reprocess(event) and len(reprocess_examples) < example_limit:
            reprocess_examples.append(
                {
                    "case_id": case_id,
                    "status": status,
                    "modality": modality,
                    "anatomy": anatomy,
                    "geometric_done": bool(event.get("geometric_done")),
                    "medsegqc_done": bool(event.get("medsegqc_done")),
                    "reason": reason_from_event(event),
                    "case_path": event.get("case_path", ""),
                }
            )

    segmentation_completed_total = None
    segmentation_completed_not_seen = None
    if segmentation_completed_cases is not None:
        segmentation_completed_total = len(segmentation_completed_cases)
        segmentation_completed_not_seen = len(segmentation_completed_cases - set(latest))

    return {
        "log": log_stats,
        "latest_cases": len(latest),
        "status_counts": dict(status_counts.most_common()),
        "cohort_counts": dict(cohort_counts.most_common()),
        "modality_counts": dict(modality_counts.most_common()),
        "anatomy_counts": dict(anatomy_counts.most_common()),
        "status_by_modality": {
            modality: dict(counter.most_common())
            for modality, counter in sorted(status_by_modality.items())
        },
        "status_by_anatomy": {
            anatomy: dict(counter.most_common())
            for anatomy, counter in sorted(status_by_anatomy.items())
        },
        "skipped_by_reason": dict(skipped_by_reason.most_common()),
        "failed_by_reason": dict(failed_by_reason.most_common()),
        "completed_outputs": {
            "geometric_done": completed_geometric_done,
            "medsegqc_done": completed_medsegqc_done,
            "both_done": completed_both_done,
            "selected_organs_total": completed_selected_organs,
            "candidate_rows_total": completed_candidate_rows,
        },
        "reprocess": {
            "count": sum(1 for event in latest.values() if should_reprocess(event)),
            "examples": reprocess_examples,
        },
        "segmentation_state_completed_total": segmentation_completed_total,
        "segmentation_completed_not_seen_in_events": segmentation_completed_not_seen,
    }


def print_counter(title: str, values: dict[str, int], *, top_n: int | None = None) -> None:
    print(f"\n{title}")
    if not values:
        print("  (none)")
        return
    items = list(values.items())
    if top_n is not None:
        items = items[:top_n]
    width = max(len(str(key)) for key, _ in items)
    for key, value in items:
        print(f"  {str(key):<{width}}  {value}")


def print_summary(summary: dict[str, Any], *, top_n: int, show_examples: bool) -> None:
    log = summary["log"]
    completed_outputs = summary["completed_outputs"]
    reprocess = summary["reprocess"]

    print("QC/Radiomics Event Snapshot")
    print(f"Events file: {log['events_path']}")
    print(f"Raw events: {log['raw_events']}")
    print(f"Valid events: {log['valid_events']}")
    print(f"Latest unique cases: {summary['latest_cases']}")
    if log["invalid_json_lines"] or log["missing_case_id_lines"]:
        print(
            "Ignored lines: "
            f"{log['invalid_json_lines']} invalid JSON, "
            f"{log['missing_case_id_lines']} missing case_id"
        )

    if summary["segmentation_state_completed_total"] is not None:
        print(f"Segmentation completed cases: {summary['segmentation_state_completed_total']}")
        print(
            "Segmentation completed but unseen in QC events: "
            f"{summary['segmentation_completed_not_seen_in_events']}"
        )

    print_counter("Latest status counts", summary["status_counts"])
    print_counter("Cohort counts", summary["cohort_counts"])
    print_counter("Modality counts", summary["modality_counts"])

    print("\nCompleted output checks")
    print(f"  GeometricQC done: {completed_outputs['geometric_done']}")
    print(f"  MedSegQC done:    {completed_outputs['medsegqc_done']}")
    print(f"  Both done:        {completed_outputs['both_done']}")
    print(f"  Selected organs:  {completed_outputs['selected_organs_total']}")
    print(f"  Candidate rows:   {completed_outputs['candidate_rows_total']}")

    print(f"\nCases likely needing reprocessing: {reprocess['count']}")
    if show_examples:
        for example in reprocess["examples"]:
            print(
                "  "
                f"{example['case_id']} | {example['status']} | "
                f"{example['modality']}/{example['anatomy']} | "
                f"geo={example['geometric_done']} med={example['medsegqc_done']} | "
                f"{example['reason']}"
            )

    print_counter("Top skipped reasons", summary["skipped_by_reason"], top_n=top_n)
    print_counter("Top failed reasons", summary["failed_by_reason"], top_n=top_n)
    print_counter("Anatomy counts", summary["anatomy_counts"], top_n=top_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize latest QC/radiomics status from qc_radiomics_events.jsonl."
    )
    parser.add_argument(
        "--events",
        default=str(DEFAULT_EVENTS_PATH),
        help=f"Path to qc_radiomics_events.jsonl (default: {DEFAULT_EVENTS_PATH})",
    )
    parser.add_argument(
        "--segmentation-state",
        default="",
        help=(
            "Optional pipeline_state_v2.json. When provided, report how many "
            "segmentation-completed cases have not appeared in the QC event log yet."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print the full summary as JSON.")
    parser.add_argument("--write-json", default="", help="Write the full summary to this JSON path.")
    parser.add_argument("--top-n", type=int, default=20, help="Rows to show for grouped counters.")
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Print example case IDs likely needing reprocessing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events_path = Path(args.events)
    if not events_path.is_file():
        print(f"ERROR: event log not found: {events_path}", file=sys.stderr)
        return 2

    latest, log_stats = load_latest_events(events_path)

    segmentation_completed_cases = None
    if args.segmentation_state:
        segmentation_state_path = Path(args.segmentation_state)
        if not segmentation_state_path.is_file():
            print(f"ERROR: segmentation state not found: {segmentation_state_path}", file=sys.stderr)
            return 2
        segmentation_completed_cases = load_segmentation_completed_cases(segmentation_state_path)

    summary = build_summary(
        latest,
        log_stats,
        segmentation_completed_cases=segmentation_completed_cases,
        example_limit=max(args.top_n, 1),
    )

    if args.write_json:
        out_path = Path(args.write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary, top_n=args.top_n, show_examples=args.show_examples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
