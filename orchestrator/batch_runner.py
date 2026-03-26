"""
CDW Agentic Pipeline — Batch Runner
====================================
Runs the single-case pipeline across a filelist with:
  - GPU pool allocation (prevents two tools from fighting over the same GPU)
  - Persistent case tracking with automatic resume
  - Parallel execution via ThreadPoolExecutor
  - Aggregate CSV output for downstream analysis

Usage
-----
# CLI
python -m orchestrator.batch_runner \\
    --filelist /data/soumitri/new_data_paths/ct_axial_cases.json \\
    --workers 2 --gpus 0,1 --dry-run

# Python
from orchestrator.batch_runner import BatchRunner
runner = BatchRunner(filelist="cases.json", gpus=[0, 1], workers=2)
runner.run()
runner.run(retry_failed=True)   # re-process failed cases
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import List, Optional

from config.constants import (
    IMAGE_FILENAME,
    LOG_DIR,
    SEGMENTATION_ROOT,
    SRC_PREFIX,
    DST_PREFIX,
)
from orchestrator.case_tracker import CaseTracker
from orchestrator.pipeline import CasePipeline, CaseResult

logger = logging.getLogger(__name__)


class BatchRunner:
    """
    Batch pipeline runner with GPU pooling and persistent state.

    Two-model architecture:
      - planner_llm (Qwen3-8B): metadata, tool selection — fast, 1 GPU
      - clinical_llm (MedGemma-27B): QC interpretation, radiomics gating — 2 GPUs, loaded when needed

    Args:
        filelist: Path to JSON file with list of case paths.
        state_file: Path to JSON state file (default: logs/pipeline_state.json).
        output_csv: Path to aggregate CSV output (default: logs/pipeline_results.csv).
        gpus: List of GPU indices to use (e.g. [0, 1]). Default [0].
        workers: Number of parallel workers. Should not exceed len(gpus).
        dry_run: If True, tools produce mock output.
        no_llm: If True, skip all LLM calls.
        skip_radiomics: If True, skip radiomics extraction step.
        planner_llm: LLM for metadata/tool selection (Qwen3-8B). None for no_llm mode.
        clinical_llm: LLM for QC interpretation/radiomics gating (MedGemma-27B). None for rule-based fallback.
        llm: Single LLM for all tasks (legacy). Overridden by planner_llm/clinical_llm.
    """

    def __init__(
        self,
        filelist: str,
        state_file: str = "",
        output_csv: str = "",
        gpus: Optional[List[int]] = None,
        workers: int = 1,
        dry_run: bool = False,
        no_llm: bool = False,
        skip_radiomics: bool = False,
        planner_llm=None,
        clinical_llm=None,
        llm=None,
        consensus: bool = False,
    ):
        self.filelist = filelist
        self.gpus = gpus or [0]
        self.workers = min(workers, len(self.gpus))
        self.dry_run = dry_run
        self.no_llm = no_llm 
        self.skip_radiomics = skip_radiomics
        self.planner_llm = planner_llm
        self.clinical_llm = clinical_llm
        self.consensus = consensus
        self.llm = llm

        if not self.no_llm and self.planner_llm is None:
            logger.warning("LLM enabled but planner_llm is None; metadata/tool-selection will fall back unless provided.")
        if not self.no_llm and self.clinical_llm is None:
            logger.warning("LLM enabled but clinical_llm is None; QC interpretation will use fallback.")

        # Default paths
        os.makedirs(LOG_DIR, exist_ok=True)
        self.state_file = state_file or os.path.join(LOG_DIR, "pipeline_state.json")
        self.output_csv = output_csv or os.path.join(LOG_DIR, "pipeline_results.csv")

        # GPU pool: thread-safe queue of device strings
        self._gpu_pool: Queue = Queue()
        for gpu_id in self.gpus:
            self._gpu_pool.put(f"gpu:{gpu_id}")

        # Case tracker
        self.tracker = CaseTracker(self.state_file)

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, retry_failed: bool = False) -> dict:
        """
        Run the batch pipeline.

        Args:
            retry_failed: If True, also re-process previously failed cases.

        Returns:
            Summary dict with counts and timing.
        """
        t0 = time.time()

        # Load case paths
        case_paths = self._load_filelist()
        if not case_paths:
            logger.error("No cases found in %s", self.filelist)
            return {"error": "No cases found"}

        # Register all cases (idempotent — completed cases stay completed)
        case_ids = [Path(p).name for p in case_paths]
        self.tracker.register_cases(case_ids)

        # Build work queue: pending + optionally failed
        id_to_path = {Path(p).name: p for p in case_paths}
        work = []
        for cid in case_ids:
            status = self.tracker.get_status(cid)
            if status == "pending":
                work.append(cid)
            elif status == "failed" and retry_failed:
                work.append(cid)
            # completed / running / skipped → skip

        logger.info(
            "Batch: %d total cases, %d to process (%d completed, %d failed, retry=%s)",
            len(case_ids), len(work),
            self.tracker.count("completed"),
            self.tracker.count("failed"),
            retry_failed,
        )

        if not work:
            logger.info("Nothing to do — all cases already processed")
            return self.tracker.summary()

        # Initialize CSV
        self._init_csv()

        # Run with thread pool
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {}
            for cid in work:
                case_path = id_to_path.get(cid)
                if not case_path:
                    continue
                future = executor.submit(self._run_one_case, cid, case_path)
                futures[future] = cid

            for future in as_completed(futures):
                cid = futures[future]
                try:
                    result = future.result()
                    if result.status == "completed":
                        completed += 1
                    elif result.status == "failed":
                        failed += 1
                except Exception as e:
                    logger.error("[%s] Unhandled error: %s", cid, e)
                    self.tracker.set_status(cid, "failed", error=str(e))
                    failed += 1

        elapsed = time.time() - t0
        summary = self.tracker.summary()
        summary["batch_time_s"] = round(elapsed, 1)

        logger.info(
            "Batch complete: %d completed, %d failed, %d skipped in %.1fs",
            completed, failed,
            summary.get("skipped", 0), elapsed,
        )

        return summary

    # ── Single case worker ──────────────────────────────────────────────────

    def _run_one_case(self, case_id: str, case_path: str) -> CaseResult:
        """Process one case, managing GPU allocation."""
        # Acquire GPU from pool
        device = self._gpu_pool.get()
        try:
            self.tracker.set_status(case_id, "running")
            logger.info("[%s] Starting on %s", case_id, device)

            pipeline = CasePipeline(
                planner_llm=self.planner_llm,
                clinical_llm=self.clinical_llm,
                device=device,
                dry_run=self.dry_run,
                no_llm=self.no_llm,
                skip_radiomics=self.skip_radiomics,
                consensus=self.consensus,
            )

            result = pipeline.run(case_path)

            # Update tracker
            tools_run = [t["tool_name"] for t in result.tool_outputs if t.get("success")]
            self.tracker.set_status(
                case_id,
                result.status,
                error=result.error,
                tools_run=tools_run,
            )

            # Append to CSV
            self._append_csv(result)

            logger.info(
                "[%s] %s in %.1fs (tools: %s, QC: %s)",
                case_id, result.status, result.total_time_s,
                ", ".join(result.selected_tools),
                result.qc_report.get("overall_severity", "N/A"),
            )

            return result

        finally:
            # Always return GPU to pool
            self._gpu_pool.put(device)

    # ── Filelist loading ────────────────────────────────────────────────────

    def _load_filelist(self) -> List[str]:
        """Load case paths from a JSON filelist."""
        if not os.path.isfile(self.filelist):
            raise FileNotFoundError(f"Filelist not found: {self.filelist}")

        with open(self.filelist) as f:
            raw = json.load(f)

        # Support both flat list and {"cases": [...]} format
        if isinstance(raw, list):
            paths = raw
        elif isinstance(raw, dict) and "cases" in raw:
            paths = raw["cases"]
        else:
            raise ValueError(f"Unexpected filelist format in {self.filelist}")

        # Remap DICOM paths to segmentation paths if needed
        remapped = []
        for p in paths:
            if p.startswith(SRC_PREFIX):
                p = p.replace(SRC_PREFIX, DST_PREFIX, 1)
            remapped.append(p)

        return remapped

    # ── CSV aggregation ─────────────────────────────────────────────────────

    _CSV_FIELDS = [
        "case_id", "case_path", "status", "modality", "anatomy",
        "tools_run", "qc_severity", "qc_organs_flagged",
        "agreement_dice", "interpretation_quality",
        "radiomics_organs", "total_time_s", "error",
    ]

    def _init_csv(self) -> None:
        """Write CSV header if file doesn't exist."""
        if not os.path.isfile(self.output_csv):
            with open(self.output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
                writer.writeheader()

    def _append_csv(self, result: CaseResult) -> None:
        """Append one row to the aggregate CSV."""
        try:
            qc = result.qc_report
            agreement = qc.get("agreement", {})
            interp = qc.get("interpretation", {})

            row = {
                "case_id": result.case_id,
                "case_path": result.case_path,
                "status": result.status,
                "modality": result.metadata.get("modality", ""),
                "anatomy": result.metadata.get("anatomy", ""),
                "tools_run": ";".join(result.selected_tools),
                "qc_severity": qc.get("overall_severity", ""),
                "qc_organs_flagged": sum(
                    t.get("num_flagged", 0) for t in qc.get("per_tool", [])
                ),
                "agreement_dice": agreement.get("mean_dice", ""),
                "interpretation_quality": interp.get("overall_quality", ""),
                "radiomics_organs": len(result.radiomics),
                "total_time_s": result.total_time_s,
                "error": result.error,
            }

            with open(self.output_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
                writer.writerow(row)

        except Exception as e:
            logger.warning("[%s] Failed to write CSV row: %s", result.case_id, e)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CDW Agentic Pipeline — Batch Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--filelist", required=True, help="JSON filelist of case paths")
    parser.add_argument("--state-file", default="", help="Pipeline state JSON (for resume)")
    parser.add_argument("--output-csv", default="", help="Aggregate results CSV")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU indices (e.g. 0,1)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (≤ num GPUs)")
    parser.add_argument("--dry-run", action="store_true", help="Mock segmentation, real QC")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls, use defaults")
    parser.add_argument("--skip-radiomics", action="store_true", help="Skip radiomics extraction")
    parser.add_argument("--consensus", action="store_true", help="Generate STAPLE consensus masks from multi-tool segmentations")
    parser.add_argument("--retry-failed", action="store_true", help="Re-process failed cases")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    gpus = [int(g) for g in args.gpus.split(",")]

    planner_llm = None
    clinical_llm = None
    if not args.no_llm:
        from planner import PlannerLLM
        planner_llm = PlannerLLM.from_local("checkpoints/qwen3-8b")
        planner_llm.load()
        clinical_llm = PlannerLLM.from_local("checkpoints/medgemma-27b-text-it")
        clinical_llm.load()

    runner = BatchRunner(
        filelist=args.filelist,
        state_file=args.state_file,
        output_csv=args.output_csv,
        gpus=gpus,
        workers=args.workers,
        dry_run=args.dry_run,
        skip_radiomics=args.skip_radiomics,
        no_llm=args.no_llm,
        consensus=args.consensus,
        planner_llm=planner_llm,
        clinical_llm=clinical_llm,
    )

    summary = runner.run(retry_failed=args.retry_failed)
    print(json.dumps(summary, indent=2))
