#!/usr/bin/env python
"""
CDW Agentic Pipeline — End-to-End Test Harness
===============================================
Validates the full pipeline on 3 CT + 3 MRI dummy cases.

Two modes:
  --mode string   Test planning-only stages (no images loaded):
                  path → metadata → is_diagnostic → tool_selector → organ_list
                  Validates the LLM/heuristic decision chain on real-shaped paths.

  --mode image    Test full pipeline on dummy_outputs (real images + masks):
                  metadata → tool selection → skip seg (masks exist) →
                  postprocess → Tier 1+2 QC → QC interpretation → radiomics
                  Validates QC on real masks, radiomics on real images.

  --mode full     Run both sequentially.

All decisions are tracked in a structured JSON report:
  tests/results/e2e_report.json

Usage:
  conda run -n cdw_totalseg python tests/run_e2e.py --mode string
  conda run -n cdw_radiomics python tests/run_e2e.py --mode image
  conda run -n cdw_totalseg python tests/run_e2e.py --mode full
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root on path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e")

# ──────────────────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────────────────

# 3 CT (4-digit) + 3 MRI (001, 002 are 3-digit MRI; 0002 is MRI despite 4-digit)
TEST_CASES = {
    # CT cases
    "0003": {"modality_expected": "CT", "anatomy": "abdomen_pelvis", "dummy_path": "dummy_outputs/0003"},
    "0004": {"modality_expected": "CT", "anatomy": "chest_abdomen_pelvis", "dummy_path": "dummy_outputs/0004"},
    "0005": {"modality_expected": "CT", "anatomy": "abdomen", "dummy_path": "dummy_outputs/0005"},
    # MRI cases
    "0002": {"modality_expected": "MRI", "anatomy": "abdomen", "dummy_path": "dummy_outputs/0002"},
    "001":  {"modality_expected": "MRI", "anatomy": "abdomen_pelvis", "dummy_path": "dummy_outputs/001"},
    "002":  {"modality_expected": "MRI", "anatomy": "chest", "dummy_path": "dummy_outputs/002"},
}

# Simulated production paths for string pipeline testing
# (mimics what the real filelist would contain)
SIMULATED_PATHS = {
    "0003": "/data/soumitri/segmentations_3d/RHEUM/PT0003/CT_ABD_PELVIS",
    "0004": "/data/soumitri/segmentations_3d/LUPUS/PT0004/CT_CHEST_ABD",
    "0005": "/data/soumitri/segmentations_3d/CONTR/PT0005/CT_ABD",
    "0002": "/data/soumitri/segmentations_3d/RHEUM/PT0002/MRI_ABD",
    "001":  "/data/soumitri/segmentations_3d/LUPUS/PT001/MRI_ABD_PELVIS",
    "002":  "/data/soumitri/segmentations_3d/CONTR/PT002/MRI_CHEST",
}

RESULTS_DIR = os.path.join(REPO_ROOT, "tests", "results")


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: String Pipeline (planning only, no images)
# ──────────────────────────────────────────────────────────────────────────────

def run_string_pipeline() -> List[Dict[str, Any]]:
    """
    Test the planning stages on simulated production paths.
    No images are loaded — tests metadata extraction, tool selection,
    and organ list generation using path heuristics (no_llm mode).
    """
    logger.info("=" * 70)
    logger.info("MODE 1: STRING PIPELINE (planning only, no images)")
    logger.info("=" * 70)

    from orchestrator.pipeline import CasePipeline

    # Use path heuristics (no LLM) — validates the default decision chain
    pipeline = CasePipeline(no_llm=True, dry_run=True, skip_radiomics=True)

    results = []
    for case_id, info in TEST_CASES.items():
        sim_path = SIMULATED_PATHS[case_id]
        logger.info("[%s] Processing simulated path: %s", case_id, sim_path)

        t0 = time.time()
        # Step 1: Metadata from path
        metadata = pipeline._metadata_from_path(sim_path)

        # Step 2: Tool selection
        tools = pipeline._default_tools(metadata.get("modality", "CT"))

        # Step 3: Organ list
        from planner.organ_list_generator import OrganListGenerator
        gen = OrganListGenerator(llm=None)
        organs = gen.organs_for_qc(metadata.get("anatomy", "UNKNOWN"))

        elapsed = time.time() - t0

        entry = {
            "case_id": case_id,
            "simulated_path": sim_path,
            "modality_expected": info["modality_expected"],
            "metadata_extracted": metadata,
            "modality_correct": metadata.get("modality") == info["modality_expected"],
            "is_diagnostic": metadata.get("is_diagnostic", True),
            "tools_selected": tools,
            "expected_organs": organs,
            "num_expected_organs": len(organs),
            "elapsed_s": round(elapsed, 3),
        }
        results.append(entry)

        status = "PASS" if entry["modality_correct"] else "FAIL"
        logger.info(
            "  [%s] modality=%s (expected %s) | tools=%s | organs=%d | %.3fs",
            status, metadata.get("modality"), info["modality_expected"],
            tools, len(organs), elapsed,
        )

    # Summary
    correct = sum(1 for r in results if r["modality_correct"])
    logger.info("")
    logger.info("String pipeline: %d/%d modality correct", correct, len(results))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Image Pipeline (full pipeline on dummy data)
# ──────────────────────────────────────────────────────────────────────────────

def run_image_pipeline() -> List[Dict[str, Any]]:
    """
    Test the full pipeline on dummy_outputs with real images and masks.
    Segmentation is skipped (masks already exist) but QC and radiomics
    run on real data.
    """
    logger.info("=" * 70)
    logger.info("MODE 2: IMAGE PIPELINE (real images + masks)")
    logger.info("=" * 70)

    from orchestrator.pipeline import CasePipeline

    pipeline = CasePipeline(
        no_llm=True,
        dry_run=True,        # skip segmentation (use existing masks)
        skip_radiomics=False, # run radiomics on real images
        postprocess=False,    # don't modify existing masks
    )

    results = []
    for case_id, info in TEST_CASES.items():
        case_path = os.path.join(REPO_ROOT, info["dummy_path"])
        image_path = os.path.join(case_path, "image_nifti.nii.gz")

        if not os.path.isfile(image_path):
            logger.warning("[%s] Image not found: %s — skipping", case_id, image_path)
            continue

        logger.info("[%s] Running full pipeline on %s", case_id, case_path)
        t0 = time.time()

        # Pass metadata override so pipeline uses correct modality/anatomy
        # (dummy_outputs paths don't encode this info like production paths do)
        metadata_override = {
            "modality": info["modality_expected"],
            "anatomy": info["anatomy"],
            "is_diagnostic": True,
            "source": "e2e_test_override",
        }
        result = pipeline.run(case_path, metadata_override=metadata_override)
        elapsed = time.time() - t0

        # Discover which seg dirs actually have masks
        seg_dirs_found = {}
        for item in os.listdir(case_path):
            full = os.path.join(case_path, item)
            if os.path.isdir(full) and item.startswith("segmentations_"):
                mask_count = len([f for f in os.listdir(full) if f.endswith(".nii.gz")])
                if mask_count > 0:
                    seg_dirs_found[item] = mask_count

        # Build decision trace
        entry = {
            "case_id": case_id,
            "case_path": case_path,
            "modality_expected": info["modality_expected"],
            "status": result.status,
            "total_time_s": round(elapsed, 2),

            # Decision 1: Metadata
            "metadata": result.metadata,

            # Decision 2: Tool selection
            "tools_selected": result.selected_tools,

            # Decision 3: Segmentation (what's available)
            "seg_dirs_available": seg_dirs_found,
            "tool_outputs": result.tool_outputs,

            # Decision 4: QC assessment
            "qc_overall_severity": result.qc_report.get("overall_severity", "N/A"),
            "qc_per_tool": result.qc_report.get("per_tool", []),
            "qc_agreement": result.qc_report.get("agreement", {}),
            "qc_interpretation": result.qc_report.get("interpretation", {}),

            # Decision 5: Radiomics
            "radiomics_organs_extracted": len(result.radiomics),
            "radiomics_details": [
                {"organ": r["organ"], "features": r["feature_count"], "backend": r["backend"]}
                for r in result.radiomics
            ],

            # Timing breakdown
            "step_times": result.step_times,
            "warnings": result.warnings,
            "error": result.error,
        }
        results.append(entry)

        # Log summary
        qc_sev = result.qc_report.get("overall_severity", "N/A")
        n_tools = len([t for t in result.tool_outputs if t.get("success")])
        n_rad = len(result.radiomics)
        logger.info(
            "  status=%s | tools=%d | QC=%s | radiomics=%d organs | %.1fs",
            result.status, n_tools, qc_sev, n_rad, elapsed,
        )

        # Log per-tool QC detail
        for pt in result.qc_report.get("per_tool", []):
            logger.info(
                "    %s: severity=%s, flagged=%d/%d organs",
                pt["tool"], pt["severity"], pt["num_flagged"], pt["num_organs"],
            )

        # Log radiomics detail
        for r in result.radiomics:
            logger.info("    radiomics: %s → %d features", r["organ"], r["feature_count"])

    # Summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("IMAGE PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "%-6s | %-4s | %-6s | %-9s | %-5s | %-8s | %s",
        "Case", "Mod", "Status", "QC", "Tools", "Radiomics", "Time",
    )
    logger.info("-" * 70)
    for r in results:
        n_tools = len([t for t in r["tool_outputs"] if t.get("success")])
        logger.info(
            "%-6s | %-4s | %-6s | %-9s | %-5d | %-8d | %.1fs",
            r["case_id"],
            r["metadata"].get("modality", "?"),
            r["status"],
            r["qc_overall_severity"],
            n_tools,
            r["radiomics_organs_extracted"],
            r["total_time_s"],
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def save_report(string_results: list, image_results: list) -> str:
    """Save full decision-traced report to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "test_cases": list(TEST_CASES.keys()),
        "string_pipeline": {
            "num_cases": len(string_results),
            "modality_accuracy": (
                sum(1 for r in string_results if r["modality_correct"]) / len(string_results)
                if string_results else 0
            ),
            "results": string_results,
        },
        "image_pipeline": {
            "num_cases": len(image_results),
            "num_completed": sum(1 for r in image_results if r["status"] == "completed"),
            "num_failed": sum(1 for r in image_results if r["status"] == "failed"),
            "results": image_results,
        },
        "summary": _build_summary(string_results, image_results),
    }

    out_path = os.path.join(RESULTS_DIR, "e2e_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Full report saved: %s", out_path)
    return out_path


def _build_summary(string_results: list, image_results: list) -> dict:
    """Build a human-readable summary of all pipeline decisions."""
    summary = {
        "decisions_per_case": {},
    }

    for r in image_results:
        cid = r["case_id"]
        qc_interp = r.get("qc_interpretation", {})
        summary["decisions_per_case"][cid] = {
            "modality": r["metadata"].get("modality", "?"),
            "anatomy": r["metadata"].get("anatomy", "?"),
            "is_diagnostic": r["metadata"].get("is_diagnostic", True),
            "tools_selected": r["tools_selected"],
            "tools_succeeded": [
                t["tool_name"] for t in r["tool_outputs"] if t.get("success")
            ],
            "tools_failed": [
                t["tool_name"] for t in r["tool_outputs"] if not t.get("success")
            ],
            "qc_severity": r["qc_overall_severity"],
            "qc_quality_assessment": qc_interp.get("overall_quality", "N/A (rule-based)"),
            "organs_usable_for_radiomics": qc_interp.get("usable_for_radiomics", []),
            "organs_unusable": qc_interp.get("unusable_organs", []),
            "radiomics_tool_used": _get_radiomics_tool(r),
            "radiomics_organs": [
                d["organ"] for d in r.get("radiomics_details", [])
            ],
            "radiomics_total_features": sum(
                d["features"] for d in r.get("radiomics_details", [])
            ),
            "total_time_s": r["total_time_s"],
        }

    return summary


def _get_radiomics_tool(r: dict) -> str:
    """Determine which tool's masks were used for radiomics."""
    # Radiomics uses the first tool in selected_tools that has a valid seg_dir
    for t in r.get("tool_outputs", []):
        if t.get("success") and t.get("seg_dir"):
            return t["tool_name"]
    return "none"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CDW E2E Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_e2e.py --mode string    # planning only, no images
  python tests/run_e2e.py --mode image     # full pipeline on dummy data
  python tests/run_e2e.py --mode full      # both modes
        """,
    )
    parser.add_argument(
        "--mode", choices=["string", "image", "full"], default="full",
        help="Test mode: string (planning only), image (full pipeline), full (both)",
    )
    args = parser.parse_args()

    string_results = []
    image_results = []

    if args.mode in ("string", "full"):
        string_results = run_string_pipeline()

    if args.mode in ("image", "full"):
        image_results = run_image_pipeline()

    # Save report
    report_path = save_report(string_results, image_results)

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("E2E TEST COMPLETE")
    logger.info("=" * 70)
    if string_results:
        correct = sum(1 for r in string_results if r["modality_correct"])
        logger.info("String pipeline: %d/%d modality correct", correct, len(string_results))
    if image_results:
        completed = sum(1 for r in image_results if r["status"] == "completed")
        total_radiomics = sum(r["radiomics_organs_extracted"] for r in image_results)
        logger.info("Image pipeline: %d/%d completed, %d organs with radiomics",
                     completed, len(image_results), total_radiomics)
    logger.info("Report: %s", report_path)
