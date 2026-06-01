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
from queue import Queue
from typing import List, Optional

from config.constants import (
    DICOM_ROOT,
    IMAGE_FILENAME,
    LOG_DIR,
    SEGMENTATION_ROOT,
    SRC_PREFIX,
    DST_PREFIX,
)
from orchestrator.case_id import assert_unique_ids
from orchestrator.case_tracker import CaseTracker
from orchestrator.pipeline import CasePipeline, CaseResult
from processing.dicom_to_nifti_3d import ensure_nifti_for_case

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
        tools_whitelist: Optional[List[str]] = None,
        tool_timeout_s: Optional[float] = None,
    ):
        self.filelist = filelist
        self.gpus = gpus or [0]
        requested_workers = workers
        self.workers = min(workers, len(self.gpus))
        if requested_workers < len(self.gpus):
            logger.warning(
                "Workers (%d) < GPUs (%d): %d GPU(s) will sit idle. "
                "Pass --workers %d to use them all.",
                requested_workers, len(self.gpus),
                len(self.gpus) - requested_workers, len(self.gpus),
            )
        elif requested_workers > len(self.gpus):
            logger.warning(
                "Workers (%d) > GPUs (%d): capped to %d. Add more GPUs or lower --workers.",
                requested_workers, len(self.gpus), self.workers,
            )
        self.dry_run = dry_run
        self.no_llm = no_llm 
        self.skip_radiomics = skip_radiomics
        self.planner_llm = planner_llm
        self.clinical_llm = clinical_llm
        self.consensus = consensus
        self.llm = llm
        self.tools_whitelist = tools_whitelist
        self.tool_timeout_s = tool_timeout_s

        # Validate --tools whitelist against the registry up front. A typo
        # like 'TotalSegmentatorCT' (missing underscore) would otherwise
        # silently make every diagnostic case complete with zero masks.
        if self.tools_whitelist:
            self._validate_tools_whitelist(self.tools_whitelist)

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

        # Derive stable, unique case IDs. Raises if filelist paths collide
        # under the configured roots — better to fail fast than silently
        # overwrite tracker entries (basenames collide ~3:1 in clinical data).
        id_to_path = assert_unique_ids(case_paths)
        case_ids = list(id_to_path.keys())

        # Register all cases (idempotent — completed cases stay completed)
        self.tracker.register_cases(case_ids)
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

            materialize = self._ensure_case_image(case_id, case_path)
            if materialize.status == "failed":
                self.tracker.set_status(case_id, "failed", error=materialize.error)
                self._append_csv(materialize)
                logger.error("[%s] %s", case_id, materialize.error)
                return materialize

            pipeline = CasePipeline(
                planner_llm=self.planner_llm,
                clinical_llm=self.clinical_llm,
                device=device,
                dry_run=self.dry_run,
                no_llm=self.no_llm,
                skip_radiomics=self.skip_radiomics,
                consensus=self.consensus,
                tools_whitelist=self.tools_whitelist,
                tool_timeout_s=self.tool_timeout_s,
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
                "[%s] %s in %.1fs (tools: %s, QC: %s)%s",
                case_id, result.status, result.total_time_s,
                ", ".join(result.selected_tools),
                result.qc_report.get("overall_severity", "N/A"),
                f" error={result.error}" if result.error else "",
            )

            return result

        finally:
            # Always return GPU to pool
            self._gpu_pool.put(device)

    def _ensure_case_image(self, case_id: str, case_path: str) -> CaseResult:
        """Create image_nifti.nii.gz from the matching DICOM series if missing."""

        image_path = os.path.join(case_path, IMAGE_FILENAME)
        if os.path.isfile(image_path):
            return CaseResult(case_id=case_id, case_path=case_path, status="completed")

        dicom_path = self._dicom_path_for_case(case_path)
        status = ensure_nifti_for_case(
            case_path,
            dicom_path,
            image_filename=IMAGE_FILENAME,
        )
        if status["status"] in {"exists", "converted"}:
            logger.info("[%s] NIfTI %s: %s", case_id, status["status"], status["image_path"])
            return CaseResult(case_id=case_id, case_path=case_path, status="completed")

        result = CaseResult(case_id=case_id, case_path=case_path, status="failed")
        result.error = status.get("error", f"NIfTI materialization failed: {status}")
        result.warnings.append(f"DICOM source: {dicom_path}")
        return result

    @staticmethod
    def _dicom_path_for_case(case_path: str) -> str:
        """Map a segmentation case directory back to its raw DICOM directory."""

        if case_path.startswith(DST_PREFIX):
            return case_path.replace(DST_PREFIX, SRC_PREFIX, 1)
        if case_path.startswith(f"{SEGMENTATION_ROOT}/"):
            rel = case_path[len(f"{SEGMENTATION_ROOT}/"):]
            return os.path.join(DICOM_ROOT, rel)
        return case_path

    # ── Filelist loading ────────────────────────────────────────────────────

    def _validate_tools_whitelist(self, names: List[str]) -> None:
        """Cross-check the --tools list against tool_registry.json.

        Raises ValueError on the first unknown name. Fails fast at startup
        rather than silently letting every case complete with zero masks
        (a typo like 'TotalSegmentatorCT' for 'TotalSegmentator_CT' would
        otherwise pass through unnoticed in a 24K-volume run).
        """
        reg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "tool_registry.json",
        )
        with open(reg_path) as f:
            reg = json.load(f)
        known = set()
        for category in ("fixed_class_tools", "text_promptable_tools", "label_prompted_tools"):
            for entry in reg.get(category, []):
                if entry.get("deferred"):
                    continue
                if entry.get("name"):
                    known.add(entry["name"])
        unknown = [n for n in names if n not in known]
        if unknown:
            raise ValueError(
                f"--tools contains unknown tool name(s): {unknown}. "
                f"Available active tools: {sorted(known)}"
            )

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
    parser.add_argument(
        "--tools",
        default="",
        help="Comma-separated tool names to restrict to (e.g. 'TotalSegmentator_CT,VISTA3D'). "
             "Empty = use all registry-compatible tools.",
    )
    parser.add_argument(
        "--tool-timeout",
        type=float,
        default=None,
        help="Per-tool subprocess timeout in seconds (e.g. 600 = 10 min). "
             "Default: no timeout. Tool will be marked failed on timeout.",
    )
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
        # clinical_llm = PlannerLLM.from_local("checkpoints/medgemma-27b-text-it")
        # clinical_llm.load()

    tools_whitelist = (
        [t.strip() for t in args.tools.split(",") if t.strip()] if args.tools else None
    )

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
        tools_whitelist=tools_whitelist,
        tool_timeout_s=args.tool_timeout,
    )

    summary = runner.run(retry_failed=args.retry_failed)
    print(json.dumps(summary, indent=2))
