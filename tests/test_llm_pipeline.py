#!/usr/bin/env python
"""
CDW Agentic Pipeline — Step-by-Step LLM Validation
====================================================
Tests each LLM decision point individually:
  Step 1: Metadata extraction from filepath (Qwen3-8B)
  Step 2: Tool selection/routing (Qwen3-8B)
  Step 3: Organ list prediction for anatomy (MedGemma-27B)
  Step 4: QC report generation from QC outputs (MedGemma-27B)
  Step 5: Radiomics gating from QC report (MedGemma-27B)

Usage:
  conda run -n cdw_llm python tests/test_llm_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
import time

REPO_ROOT = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# ──────────────────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "path": "/data/soumitri/segmentations_3d/RHEUM/PT0003/CT_ABD_PELVIS",
        "expected_modality": "CT",
        "expected_anatomy": "abdomen_pelvis",
        "shape": "[512, 512, 237]",
    },
    {
        "path": "/data/soumitri/segmentations_3d/LUPUS/PT0004/CT_CHEST_ABD_PELVIS_WITH_CONTRAST/Axial_ST_3mm",
        "expected_modality": "CT",
        "expected_anatomy": "chest_abdomen_pelvis",
        "shape": "[512, 512, 350]",
    },
    {
        "path": "/data/soumitri/segmentations_3d/RHEUM/PT0002/MRI_ABD/T1_VIBE_Dixon_Water",
        "expected_modality": "MRI",
        "expected_anatomy": "abdomen",
        "shape": "[320, 260, 128]",
    },
    {
        "path": "/data/soumitri/segmentations_3d/LUPUS/PT001/MRI_ABD_PELVIS/T2_HASTE_Axial",
        "expected_modality": "MRI",
        "expected_anatomy": "abdomen_pelvis",
        "shape": "[384, 384, 40]",
    },
]

# Expected tool sets per modality
EXPECTED_TOOLS = {
    "CT": {"TotalSegmentator_CT", "MRSegmentator", "VISTA3D", "VoxTell", "TextMedSeg3D"},
    "MRI": {"TotalSegmentator_MR", "MRSegmentator", "MRISegmenter", "VIBESegmentator", "VISTA3D", "VoxTell", "TextMedSeg3D"},
}

# Core organs that MUST be in the predicted organ list for abdominal anatomy
CORE_ABDOMEN_ORGANS = {"liver", "spleen", "kidney_left", "kidney_right", "pancreas", "gallbladder", "stomach", "aorta"}


def header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def check(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    detail_str = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{detail_str}")
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Metadata extraction (Qwen3-8B — string processing)
# ──────────────────────────────────────────────────────────────────────────────

def test_metadata_extraction(planner_llm):
    header("STEP 1: Metadata Extraction from Filepath (Qwen3-8B)")
    from config.prompts.example_prompts import METADATA_EXTRACTION_SINGLE

    all_pass = True
    for tc in TEST_CASES:
        prompt = METADATA_EXTRACTION_SINGLE.format(file_path=tc["path"], shape=tc["shape"])
        print(f"\n  Path: {tc['path']}")
        try:
            t0 = time.time()
            result = planner_llm.query_json(
                system_prompt="You are a medical imaging metadata parser. Return only valid JSON. Do not include reasoning.",
                user_prompt=prompt,
                max_new_tokens=4096,
                temperature=0.1,
            )
            elapsed = time.time() - t0

            modality = result.get("modality", "UNKNOWN")
            anatomy = result.get("anatomy", "UNKNOWN")
            is_diag = result.get("is_diagnostic", None)

            print(f"  LLM response ({elapsed:.1f}s): modality={modality}, anatomy={anatomy}, is_diagnostic={is_diag}")

            p1 = check("modality correct", modality == tc["expected_modality"],
                        f"got={modality}, expected={tc['expected_modality']}")
            p2 = check("anatomy correct", anatomy == tc["expected_anatomy"],
                        f"got={anatomy}, expected={tc['expected_anatomy']}")
            p3 = check("is_diagnostic set", is_diag is not None, f"got={is_diag}")

            if not (p1 and p2 and p3):
                all_pass = False
        except Exception as e:
            print(f"  ERROR: {e}")
            check("metadata extraction", False, str(e))
            all_pass = False

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Tool selection/routing (Qwen3-8B — structured routing)
# ──────────────────────────────────────────────────────────────────────────────

def test_tool_selection(planner_llm):
    header("STEP 2: Tool Selection/Routing (Qwen3-8B)")
    from planner.tool_selector import ToolSelector

    selector = ToolSelector(planner_llm)
    all_pass = True

    for tc in TEST_CASES:
        metadata = {
            "modality": tc["expected_modality"],
            "anatomy": tc["expected_anatomy"],
            "is_diagnostic": True,
            "shape": "3D",
        }
        print(f"\n  Case: {tc['expected_modality']} {tc['expected_anatomy']}")
        try:
            t0 = time.time()
            selection = selector.select(metadata)
            elapsed = time.time() - t0

            all_selected = set(selection.get("primary_tools", []) +
                              selection.get("secondary_tools", []))
            targeted = selection.get("targeted_tools", [])
            targeted_names = {t["tool"] if isinstance(t, dict) else t for t in targeted}
            all_selected |= targeted_names

            expected = EXPECTED_TOOLS.get(tc["expected_modality"], set())

            print(f"  ({elapsed:.1f}s) Reasoning: {selection.get('reasoning', 'N/A')}")
            print(f"  Primary: {selection.get('primary_tools', [])}")
            print(f"  Secondary: {selection.get('secondary_tools', [])}")
            print(f"  Targeted: {targeted}")
            print(f"  QC organs: {selection.get('qc_organs', [])}")

            p1 = check("all expected tools covered",
                        expected <= all_selected,
                        f"missing={sorted(expected - all_selected)}" if expected - all_selected else "all present")

            # Text-prompted tools should have organ lists
            for t in targeted:
                if isinstance(t, dict) and t.get("tool") in ("VoxTell", "TextMedSeg3D"):
                    organs = t.get("organs", [])
                    p2 = check(f"{t['tool']} has organ prompts", len(organs) > 0,
                               f"{len(organs)} organs: {organs[:5]}")
                    if not p2:
                        all_pass = False

            if not p1:
                all_pass = False
        except Exception as e:
            print(f"  ERROR: {e}")
            check("tool selection", False, str(e))
            all_pass = False

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Organ list prediction for anatomy (MedGemma-27B — clinical)
# ──────────────────────────────────────────────────────────────────────────────

def test_organ_prediction(clinical_llm):
    header("STEP 3: Organ List Prediction for Anatomy (MedGemma-27B)")
    from config.prompts.example_prompts import ORGAN_LIST_GENERATION

    all_pass = True
    for tc in TEST_CASES:
        prompt = ORGAN_LIST_GENERATION.format(
            anatomy=tc["expected_anatomy"],
            modality=tc["expected_modality"],
            series_type=f"standard {tc['expected_modality']} {tc['expected_anatomy']}",
        )
        t0 = time.time()
        result = clinical_llm.query_json(
            system_prompt="You are a medical imaging anatomy expert. Return only valid JSON.",
            user_prompt=prompt,
            max_new_tokens=2048,
            temperature=0.1,
        )
        elapsed = time.time() - t0

        primary = result.get("primary_organs", [])
        edge = result.get("edge_organs", [])
        exclude = result.get("exclude_organs", [])

        print(f"\n  Case: {tc['expected_modality']} {tc['expected_anatomy']} ({elapsed:.1f}s)")
        print(f"  Primary organs ({len(primary)}): {primary}")
        print(f"  Edge organs ({len(edge)}): {edge}")
        print(f"  Excluded ({len(exclude)}): {exclude[:5]}...")

        p1 = check("primary organs >= 8", len(primary) >= 8, f"got {len(primary)}")
        p2 = check("primary organs <= 40", len(primary) <= 40, f"got {len(primary)}")

        # For abdominal anatomy, check core organs are present
        if "abdomen" in tc["expected_anatomy"]:
            # Normalize organ names for fuzzy matching
            primary_normalized = set()
            for o in primary:
                norm = o.lower().replace(" ", "_").replace("-", "_")
                primary_normalized.add(norm)
            # Allow flexible matching: "kidneys_(both)" matches "kidney_left"/"kidney_right"
            core_found = set()
            for core in CORE_ABDOMEN_ORGANS:
                # Direct match
                if core in primary_normalized or any(core in p for p in primary_normalized):
                    core_found.add(core)
                    continue
                # "kidney_left"/"kidney_right" matched by "kidneys" or "kidneys_(both)"
                core_root = core.replace("_left", "").replace("_right", "")
                if any(core_root in p for p in primary_normalized):
                    core_found.add(core)
            missing = CORE_ABDOMEN_ORGANS - core_found
            p3 = check("core abdominal organs present",
                       len(missing) <= 2,  # allow some flexibility
                       f"found={sorted(core_found)}, missing={sorted(missing)}")
            if not p3:
                all_pass = False

        if not (p1 and p2):
            all_pass = False

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: QC report generation (MedGemma-27B — clinical interpretation)
# ──────────────────────────────────────────────────────────────────────────────

def test_qc_interpretation(clinical_llm):
    header("STEP 4: QC Report from QC Outputs (MedGemma-27B)")
    from config.prompts.example_prompts import QC_INTERPRETATION

    # Simulate QC flags for a CT abdomen case
    flagged = """- pancreas: UNDER_VOLUME (1.2ml vs expected 50-150ml); CC=3; FRAG=45%
- gallbladder: CC=2; FRAG=85%
- adrenal_gland_left: UNDER_VOLUME (0.3ml vs expected 3-6ml)"""

    passed = "liver, spleen, kidney_left, kidney_right, stomach, aorta, inferior_vena_cava, urinary_bladder, colon, small_bowel"

    prompt = QC_INTERPRETATION.format(
        study_description="CT Abdomen Pelvis with Contrast",
        anatomy="abdomen_pelvis",
        tool_name="TotalSegmentator_CT",
        num_organs_in_fov=13,
        flagged_organs_formatted=flagged,
        passed_organs_list=passed,
    )

    t0 = time.time()
    result = clinical_llm.query_json(
        system_prompt="",
        user_prompt=prompt,
        max_new_tokens=4096,
        temperature=0.1,
    )
    elapsed = time.time() - t0

    quality = result.get("overall_quality", "N/A")
    usable = result.get("usable_for_radiomics", [])
    unusable = result.get("unusable_organs", [])
    explanation = result.get("explanation", "")

    print(f"  Response time: {elapsed:.1f}s")
    print(f"  Overall quality: {quality}")
    print(f"  Usable for radiomics ({len(usable)}): {usable}")
    print(f"  Unusable ({len(unusable)}): {unusable}")
    print(f"  Explanation: {explanation}")

    all_pass = True
    all_pass &= check("overall_quality is valid", quality in ("GOOD", "ACCEPTABLE", "POOR", "UNUSABLE"), f"got={quality}")
    all_pass &= check("usable list non-empty", len(usable) > 0, f"got {len(usable)}")
    all_pass &= check("liver in usable", "liver" in usable, f"usable={usable}")
    all_pass &= check("spleen in usable", "spleen" in usable, f"usable={usable}")
    all_pass &= check("pancreas NOT in usable (flagged UNDER_VOLUME+CC=3)",
                       "pancreas" not in usable, f"usable={usable}")
    all_pass &= check("usable count reasonable (5-15)", 5 <= len(usable) <= 15, f"got {len(usable)}")

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Radiomics gating (MedGemma-27B — clinical decision)
# ──────────────────────────────────────────────────────────────────────────────

def test_radiomics_gating(clinical_llm):
    header("STEP 5: Radiomics Gating from QC Report (MedGemma-27B)")
    from config.prompts.example_prompts import RADIOMICS_GATING

    per_organ_qc = json.dumps([
        {"organ": "liver", "volume_ml": 1450.2, "severity": "PASS", "num_components": 1, "largest_component_fraction": 1.0},
        {"organ": "spleen", "volume_ml": 180.5, "severity": "PASS", "num_components": 1, "largest_component_fraction": 1.0},
        {"organ": "kidney_left", "volume_ml": 145.0, "severity": "PASS", "num_components": 1, "largest_component_fraction": 1.0},
        {"organ": "kidney_right", "volume_ml": 150.3, "severity": "PASS", "num_components": 1, "largest_component_fraction": 1.0},
        {"organ": "pancreas", "volume_ml": 1.2, "severity": "FAIL", "num_components": 3, "largest_component_fraction": 0.45, "volume_flag": "UNDER_VOLUME", "cc_flag": "CC=3; FRAG=45%"},
        {"organ": "gallbladder", "volume_ml": 25.0, "severity": "WARN", "num_components": 2, "largest_component_fraction": 0.85},
        {"organ": "stomach", "volume_ml": 280.0, "severity": "PASS", "num_components": 1, "largest_component_fraction": 1.0},
        {"organ": "aorta", "volume_ml": 95.0, "severity": "PASS", "num_components": 1, "largest_component_fraction": 1.0},
    ], indent=2)

    prompt = (
        RADIOMICS_GATING
        .replace("{case_id}", "PT0003")
        .replace("{study_description}", "CT Abdomen Pelvis with Contrast")
        .replace("{overall_quality}", "ACCEPTABLE")
        .replace("{per_organ_qc_json}", per_organ_qc)
    )

    t0 = time.time()
    result = clinical_llm.query_json(
        system_prompt="",
        user_prompt=prompt,
        max_new_tokens=4096,
        temperature=0.1,
    )
    elapsed = time.time() - t0

    extract = result.get("extract", [])
    skip = result.get("skip", [])
    notes = result.get("notes", "")

    print(f"  Response time: {elapsed:.1f}s")
    print(f"  Extract ({len(extract)}): {[e.get('organ') for e in extract]}")
    print(f"  Skip ({len(skip)}): {[s.get('organ') for s in skip]}")
    print(f"  Notes: {notes}")

    all_pass = True
    extract_organs = {e.get("organ") for e in extract}
    skip_organs = {s.get("organ") for s in skip}

    all_pass &= check("liver in extract", "liver" in extract_organs)
    all_pass &= check("spleen in extract", "spleen" in extract_organs)
    all_pass &= check("pancreas in skip (FAIL severity)", "pancreas" in skip_organs)
    all_pass &= check("gallbladder in extract (WARN, fixable with LCC)",
                       "gallbladder" in extract_organs)
    all_pass &= check("extract count >= 5", len(extract) >= 5, f"got {len(extract)}")

    # Check that extract entries have correct structure
    if extract:
        e = extract[0]
        all_pass &= check("extract has features_allowed", "features_allowed" in e, f"keys={list(e.keys())}")
        all_pass &= check("extract has postprocessing_needed", "postprocessing_needed" in e, f"keys={list(e.keys())}")

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from planner.llm_client import PlannerLLM

    results = {}

    # Load Qwen3-8B for string processing tasks
    print("Loading Qwen3-8B (planner — string processing)...")
    planner = PlannerLLM.from_local("checkpoints/qwen3-8b", device="auto")
    planner.load()

    results["Step 1: Metadata"] = test_metadata_extraction(planner)
    results["Step 2: Tool Selection"] = test_tool_selection(planner)

    planner.unload()

    # Load MedGemma-27B for clinical/anatomy tasks
    print("\nLoading MedGemma-27B (clinical — anatomy reasoning)...")
    clinical = PlannerLLM.from_local("checkpoints/medgemma-27b-text-it", device="auto")
    clinical.load()

    results["Step 3: Organ Prediction"] = test_organ_prediction(clinical)
    results["Step 4: QC Interpretation"] = test_qc_interpretation(clinical)
    results["Step 5: Radiomics Gating"] = test_radiomics_gating(clinical)

    clinical.unload()

    # Summary
    header("SUMMARY")
    all_pass = True
    for step, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {step}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ALL STEPS PASSED")
    else:
        print("  SOME STEPS FAILED — see details above")

    sys.exit(0 if all_pass else 1)
