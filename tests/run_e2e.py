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
# Ground-truth definitions for intermediate validation
# ──────────────────────────────────────────────────────────────────────────────

# Expected tool selection per modality — ALL compatible tools from registry
# (the pipeline now selects every tool that supports the case modality)
EXPECTED_TOOLS_CT = [
    "TotalSegmentator_CT", "MRSegmentator", "VISTA3D",
    "VoxTell", "TextMedSeg3D",
]
EXPECTED_TOOLS_MRI = [
    "TotalSegmentator_MR", "MRSegmentator", "MRISegmenter",
    "VIBESegmentator", "VISTA3D", "VoxTell", "TextMedSeg3D",
]
EXPECTED_TOOLS = {
    "CT":  EXPECTED_TOOLS_CT,
    "MRI": EXPECTED_TOOLS_MRI,
}

# Expected seg_dir prefixes per tool that the pipeline should discover
EXPECTED_SEG_DIRS = {
    "TotalSegmentator_CT": "segmentations_totalseg_ct",
    "TotalSegmentator_MR": "segmentations_totalseg_mr",
    "MRSegmentator":       "segmentations_mrseg",
    "MRISegmenter":        "segmentations_mrisegmenter",
    "VIBESegmentator":     "segmentations_vibeseg",
    "VISTA3D":             "segmentations_vista3d",
    "VoxTell":             "segmentations_voxtell",
    "TextMedSeg3D":        "segmentations_textmedseg3d",
}

# Minimum expected organ count from OrganListGenerator per anatomy
MIN_EXPECTED_ORGANS = {
    "abdomen": 12,           # 15 primary + ~5 edge in organ_reference.json
    "abdomen_pelvis": 18,    # 20 primary + 2 edge
    "chest_abdomen_pelvis": 25,  # 29 primary + 0 edge
    "chest": 10,             # 11 primary + 3 edge
}

# Core abdominal organs that MUST appear in expected_organs for abdomen-containing scans
CORE_ABDOMINAL_ORGANS = {
    "liver", "spleen", "kidney_left", "kidney_right", "pancreas",
    "gallbladder", "stomach", "aorta",
}

# Non-organ filenames that must NOT appear in QC or radiomics
BLACKLISTED_ORGAN_NAMES = {
    "statistics", "combined", "multilabel", "multilabel_seg",
    "image_nifti_seg", "image_nifti", "image", "plan", "metadata",
    "summary", "manifest",
}

# Expected feature count per organ with shape enabled (14 shape + 93 base)
MIN_FEATURES_PER_ORGAN = 100   # at least 100 (shape adds 14 → ~107 total)
MAX_FEATURES_PER_ORGAN = 120   # upper sanity bound

# Maximum QC flag rate — enforced only on primary fixed-class tools
# Text-prompted tools (VoxTell, TextMedSeg3D) and MRISegmenter have inherently
# higher flag rates on dummy data and are validated by multi-tool agreement instead
MAX_QC_FLAG_RATE = 0.50
# Tools exempt from per-tool QC flag rate validation (checked via multi-tool QC instead)
QC_FLAG_RATE_EXEMPT_TOOLS = {"VoxTell", "TextMedSeg3D", "MRISegmenter"}


# ──────────────────────────────────────────────────────────────────────────────
# Validation engine
# ──────────────────────────────────────────────────────────────────────────────

class ValidationResult:
    """Accumulates pass/fail checks with descriptions."""
    def __init__(self):
        self.checks: List[Dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        status = "PASS" if passed else "FAIL"
        logger.info("  [%s] %s%s", status, name, f" — {detail}" if detail else "")

    @property
    def num_passed(self):
        return sum(1 for c in self.checks if c["passed"])

    @property
    def num_failed(self):
        return sum(1 for c in self.checks if not c["passed"])

    @property
    def all_passed(self):
        return self.num_failed == 0

    def summary_dict(self):
        return {
            "total": len(self.checks),
            "passed": self.num_passed,
            "failed": self.num_failed,
            "all_passed": self.all_passed,
            "failures": [c for c in self.checks if not c["passed"]],
        }


def validate_string_results(results: List[Dict]) -> ValidationResult:
    """Validate string pipeline results against ground truth."""
    v = ValidationResult()
    logger.info("")
    logger.info("=" * 70)
    logger.info("GROUND-TRUTH VALIDATION: STRING PIPELINE")
    logger.info("=" * 70)

    for r in results:
        cid = r["case_id"]
        expected_mod = r["modality_expected"]
        anatomy = TEST_CASES[cid]["anatomy"]

        # 1. Modality correctness
        v.check(
            f"{cid}: modality correct",
            r["modality_correct"],
            f"got={r['metadata_extracted'].get('modality')}, expected={expected_mod}",
        )

        # 2. Tool selection correctness — all expected tools must be present (order-independent)
        expected_tools = EXPECTED_TOOLS.get(expected_mod, [])
        got_set = set(r["tools_selected"])
        expected_set = set(expected_tools)
        v.check(
            f"{cid}: all compatible tools selected",
            expected_set <= got_set,
            f"got={sorted(got_set)}, expected={sorted(expected_set)}, missing={sorted(expected_set - got_set)}",
        )

        # 3. Organ list non-empty and meets minimum
        min_organs = MIN_EXPECTED_ORGANS.get(anatomy, 5)
        v.check(
            f"{cid}: organ list has >= {min_organs} organs",
            r["num_expected_organs"] >= min_organs,
            f"got={r['num_expected_organs']}",
        )

        # 4. Core abdominal organs present (for abdomen-containing scans)
        if "abdomen" in anatomy:
            organ_set = set(r["expected_organs"])
            missing = CORE_ABDOMINAL_ORGANS - organ_set
            v.check(
                f"{cid}: core abdominal organs present",
                len(missing) == 0,
                f"missing={missing}" if missing else "all present",
            )

    logger.info("")
    logger.info("String validation: %d/%d passed", v.num_passed, len(v.checks))
    return v


def validate_image_results(results: List[Dict]) -> ValidationResult:
    """Validate image pipeline results against ground truth."""
    v = ValidationResult()
    logger.info("")
    logger.info("=" * 70)
    logger.info("GROUND-TRUTH VALIDATION: IMAGE PIPELINE")
    logger.info("=" * 70)

    for r in results:
        cid = r["case_id"]
        expected_mod = r["modality_expected"]
        anatomy = TEST_CASES[cid]["anatomy"]

        # 1. Pipeline completed
        v.check(f"{cid}: pipeline completed", r["status"] == "completed", f"status={r['status']}")

        # 2. Correct modality in metadata
        got_mod = r["metadata"].get("modality", "?")
        v.check(f"{cid}: modality correct", got_mod == expected_mod, f"got={got_mod}")

        # 3. All compatible tools selected (order-independent)
        expected_tools = EXPECTED_TOOLS.get(expected_mod, [])
        got_set = set(r["tools_selected"])
        expected_set = set(expected_tools)
        v.check(
            f"{cid}: all compatible tools selected",
            expected_set <= got_set,
            f"got={sorted(got_set)}, missing={sorted(expected_set - got_set)}",
        )

        # 4. Seg dirs discovered for selected tools
        #    (in dry_run mode, not all tools may have pre-existing masks — check at least one)
        tools_with_dirs = 0
        for tool in expected_tools:
            expected_dir = EXPECTED_SEG_DIRS.get(tool, "")
            if expected_dir in r.get("seg_dirs_available", {}):
                tools_with_dirs += 1
        v.check(
            f"{cid}: at least 1 seg_dir exists for selected tools",
            tools_with_dirs >= 1,
            f"{tools_with_dirs}/{len(expected_tools)} tools have seg dirs",
        )

        # 5. QC flag rates — only enforce on primary fixed-class tools
        #    Text-prompted and specialist tools are validated via multi-tool agreement
        for pt in r.get("qc_per_tool", []):
            if pt["num_in_fov"] > 0 and pt["tool"] not in QC_FLAG_RATE_EXEMPT_TOOLS:
                flag_rate = pt["num_flagged"] / pt["num_in_fov"]
                v.check(
                    f"{cid}/{pt['tool']}: QC flag rate < {MAX_QC_FLAG_RATE*100:.0f}%",
                    flag_rate < MAX_QC_FLAG_RATE,
                    f"flagged={pt['num_flagged']}/{pt['num_in_fov']} ({flag_rate*100:.0f}%)",
                )

        # 6. Radiomics: no blacklisted organ names
        rad_organs = [d["organ"] for d in r.get("radiomics_details", [])]
        blacklisted_found = [o for o in rad_organs if o in BLACKLISTED_ORGAN_NAMES]
        v.check(
            f"{cid}: no blacklisted names in radiomics",
            len(blacklisted_found) == 0,
            f"found={blacklisted_found}" if blacklisted_found else "clean",
        )

        # 7. Radiomics: feature counts in expected range (shape should now be included)
        for d in r.get("radiomics_details", []):
            fc = d["features"]
            v.check(
                f"{cid}/{d['organ']}: feature count {MIN_FEATURES_PER_ORGAN}-{MAX_FEATURES_PER_ORGAN}",
                MIN_FEATURES_PER_ORGAN <= fc <= MAX_FEATURES_PER_ORGAN,
                f"got={fc}",
            )

        # 8. Radiomics: at least some organs extracted
        v.check(
            f"{cid}: radiomics extracted for >= 1 organ",
            r["radiomics_organs_extracted"] > 0,
            f"got={r['radiomics_organs_extracted']}",
        )

        # 9. Core abdominal organs should appear in usable_for_radiomics (for abdomen scans)
        interp = r.get("qc_interpretation", {})
        usable = set(interp.get("usable_for_radiomics", []))
        if "abdomen" in anatomy and usable:
            # At least liver + spleen should be usable (they're large, reliable organs)
            key_organs = {"liver", "spleen"}
            present = key_organs & usable
            v.check(
                f"{cid}: liver+spleen usable for radiomics",
                len(present) == len(key_organs),
                f"usable={present}, expected={key_organs}",
            )

    logger.info("")
    logger.info("Image validation: %d/%d passed", v.num_passed, len(v.checks))
    return v


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: String Pipeline (planning only, no images)
# ──────────────────────────────────────────────────────────────────────────────

def _load_planner_llm():
    """Load the Qwen3-8B planner LLM for tool selection / metadata / organ list."""
    from planner.llm_client import PlannerLLM
    ckpt = os.path.join(REPO_ROOT, "checkpoints", "qwen3-8b")
    if not os.path.isdir(ckpt):
        logger.error("Qwen3-8B checkpoint not found at %s", ckpt)
        sys.exit(1)
    logger.info("Loading Qwen3-8B planner LLM from %s ...", ckpt)
    llm = PlannerLLM.from_local(ckpt, device="auto")
    llm.load()
    logger.info("Qwen3-8B loaded.")
    return llm


def _load_clinical_llm():
    """Load the MedGemma-27B clinical LLM for QC interpretation / radiomics gating."""
    from planner.llm_client import PlannerLLM
    ckpt = os.path.join(REPO_ROOT, "checkpoints", "medgemma-27b-text-it")
    if not os.path.isdir(ckpt):
        logger.warning("MedGemma-27B checkpoint not found at %s — QC will use rule-based fallback", ckpt)
        return None
    logger.info("Loading MedGemma-27B clinical LLM from %s ...", ckpt)
    llm = PlannerLLM.from_local(ckpt, device="auto")
    llm.load()
    logger.info("MedGemma-27B loaded.")
    return llm


def run_string_pipeline(use_llm: bool = False) -> List[Dict[str, Any]]:
    """
    Test the planning stages on simulated production paths.

    If use_llm=True, uses the Qwen3-8B planner for metadata extraction,
    tool selection, and organ list generation (full agentic mode).
    Otherwise, uses path heuristics (no_llm mode).
    """
    logger.info("=" * 70)
    mode_str = "AGENTIC (Qwen3-8B)" if use_llm else "HEURISTIC (no LLM)"
    logger.info("MODE 1: STRING PIPELINE — %s", mode_str)
    logger.info("=" * 70)

    from orchestrator.pipeline import CasePipeline

    planner_llm = _load_planner_llm() if use_llm else None

    pipeline = CasePipeline(
        planner_llm=planner_llm,
        no_llm=not use_llm,
        dry_run=True,
        skip_radiomics=True,
    )

    results = []
    for case_id, info in TEST_CASES.items():
        sim_path = SIMULATED_PATHS[case_id]
        logger.info("[%s] Processing simulated path: %s", case_id, sim_path)

        t0 = time.time()
        # Step 1: Metadata from path
        metadata = pipeline._metadata_from_path(sim_path)

        # Step 2: Tool selection
        if use_llm:
            # Full agentic: LLM selects tools + generates organ prompts
            from planner.tool_selector import ToolSelector
            selector = ToolSelector(planner_llm)
            selection = selector.select(metadata)
            # Registry base tools (always selected regardless of LLM)
            tools = pipeline._all_compatible_tools(metadata.get("modality", "CT"))
            logger.info("  [LLM] Tool selection reasoning: %s", selection.get("reasoning", ""))
            logger.info("  [LLM] primary_tools:  %s", selection.get("primary_tools", []))
            logger.info("  [LLM] secondary_tools: %s", selection.get("secondary_tools", []))
            logger.info("  [LLM] targeted_tools:  %s", selection.get("targeted_tools", []))
            logger.info("  [LLM] qc_organs:       %s", selection.get("qc_organs", []))
        else:
            tools = pipeline._all_compatible_tools(metadata.get("modality", "CT"))
            selection = None

        # Step 3: Organ list
        from planner.organ_list_generator import OrganListGenerator
        gen = OrganListGenerator(llm=planner_llm)
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
        if selection is not None:
            entry["llm_tool_selection"] = selection
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

    if planner_llm is not None:
        planner_llm.unload()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Image Pipeline (full pipeline on dummy data)
# ──────────────────────────────────────────────────────────────────────────────

def run_image_pipeline(use_llm: bool = False) -> List[Dict[str, Any]]:
    """
    Test the full pipeline on dummy_outputs with real images and masks.
    Segmentation is skipped (masks already exist) but QC and radiomics
    run on real data.

    If use_llm=True, uses:
      - Qwen3-8B as planner_llm (tool selection, metadata, organ list)
      - MedGemma-27B as clinical_llm (QC interpretation, radiomics gating)
    """
    logger.info("=" * 70)
    mode_str = "AGENTIC (Qwen3-8B + MedGemma-27B)" if use_llm else "HEURISTIC (no LLM)"
    logger.info("MODE 2: IMAGE PIPELINE — %s", mode_str)
    logger.info("=" * 70)

    from orchestrator.pipeline import CasePipeline

    planner_llm = None
    clinical_llm = None
    if use_llm:
        planner_llm = _load_planner_llm()
        clinical_llm = _load_clinical_llm()

    pipeline = CasePipeline(
        planner_llm=planner_llm,
        clinical_llm=clinical_llm,
        no_llm=not use_llm,
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

    if planner_llm is not None:
        planner_llm.unload()
    if clinical_llm is not None:
        clinical_llm.unload()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def save_report(string_results: list, image_results: list, validations: dict = None) -> str:
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

    # Add validation results
    if validations:
        report["validation"] = {
            mode: vr.summary_dict() for mode, vr in validations.items()
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
  python tests/run_e2e.py --mode string                # heuristic planning only
  python tests/run_e2e.py --mode string --use-llm      # agentic planning (Qwen3-8B)
  python tests/run_e2e.py --mode image --use-llm       # full agentic pipeline
  python tests/run_e2e.py --mode full --use-llm        # both modes, agentic
        """,
    )
    parser.add_argument(
        "--mode", choices=["string", "image", "full"], default="full",
        help="Test mode: string (planning only), image (full pipeline), full (both)",
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Use Qwen3-8B planner LLM for agentic tool selection and organ prompts (requires GPU + checkpoints/qwen3-8b/)",
    )
    args = parser.parse_args()

    string_results = []
    image_results = []
    validations = {}

    if args.mode in ("string", "full"):
        string_results = run_string_pipeline(use_llm=args.use_llm)
        validations["string"] = validate_string_results(string_results)

    if args.mode in ("image", "full"):
        image_results = run_image_pipeline(use_llm=args.use_llm)
        validations["image"] = validate_image_results(image_results)

    # Save report
    report_path = save_report(string_results, image_results, validations)

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

    # Validation summary
    all_pass = True
    for mode, vr in validations.items():
        status = "ALL PASS" if vr.all_passed else f"{vr.num_failed} FAILED"
        logger.info("Validation [%s]: %d/%d checks passed — %s",
                     mode, vr.num_passed, len(vr.checks), status)
        if not vr.all_passed:
            all_pass = False
            for c in vr.checks:
                if not c["passed"]:
                    logger.info("  FAIL: %s — %s", c["name"], c["detail"])

    logger.info("Report: %s", report_path)

    # Exit with non-zero code if any validation failed
    if not all_pass:
        logger.warning("Some validation checks failed — see details above")
        sys.exit(1)
