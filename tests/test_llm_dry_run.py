"""
================================================================================
LLM Dry Run — CDW Pipeline Prompt Tests
================================================================================
Tests all three LLM clients (Qwen3-8B, Qwen3-32B, MedGemma-27B) against
real prompt templates from config/prompts/example_prompts.py using actual
file paths from data_paths/3d_scans_list.json.

Run from repo root:
    conda run -n cdw_llm python tests/test_llm_dry_run.py

Models tested:
    Qwen3-8B      → METADATA_EXTRACTION_SINGLE  (CT + MRI + NM cases)
    Qwen3-32B     → METADATA_EXTRACTION_BATCH   (all 3 cases)
    MedGemma-27B  → TOOL_SELECTION + QC_INTERPRETATION

Note: prompts are self-contained (include "You are..."), so system_prompt=""
to avoid double role assignment. CDW_SYSTEM_PROMPT is used only where it
adds context without conflicting.
================================================================================
"""

import json
import sys
import textwrap
from pathlib import Path

# ── Repo root on sys.path ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.prompts.example_prompts import (
    METADATA_EXTRACTION_SINGLE,
    METADATA_EXTRACTION_BATCH,
    TOOL_SELECTION,
    QC_INTERPRETATION,
    RADIOMICS_GATING,
)
from llms import MedGemma27bText, Qwen3

# ── Real file paths from data_paths/3d_scans_list.json ────────────────────────
# Selected: proper 3D volumes (depth > 30), diverse modalities, no scouts/MIPs
TEST_CASES = {
    "CT": {
        "path":  "/data/RAD/CONTR/60DA3064468E084A9C857477BF74B6E0/20051029/CT_AB_PELVIS_COMBO_WWO_CONTR/DELAYS",
        "shape": "[73, 512, 512]",
    },
    "MRI": {
        "path":  "/data/RAD/CONTR/60FAEC0A8BF20F78DB3AE91B08A5F78E/20210529/MRI_CERVICAL_SPINE_WO_GAD/MEDIC_AXIAL",
        "shape": "[32, 226, 320]",
    },
    "NM": {
        "path":  "/data/RAD/CONTR/60DA3064468E084A9C857477BF74B6E0/20250108/NM_MYOCARDIAL_PERFUSION_SPECT_MULTIPLE/AC_Rest_Cardiac_5_0__B08s",
        "shape": "[44, 512, 512]",
    },
}

# Synthetic QC flags for QC_INTERPRETATION test (realistic pass-4 output)
QC_TEST_DATA = {
    "study_description": "CT_AB_PELVIS_COMBO_WWO_CONTR",
    "anatomy": "abdomen_pelvis",
    "tool_name": "TotalSegmentator_CT",
    "num_organs_in_fov": 24,
    "flagged_organs_formatted": textwrap.dedent("""\
        aorta:         CC=3, FRAG=61%
        pancreas:      CC=4, FRAG=48%, UNDER_VOLUME (18 cm³, expected 60-120 cm³)
        liver:         OVER_VOLUME (2,841 cm³, expected 1,000-2,500 cm³)
        duodenum:      CC=5, FRAG=39%
        portal_vein:   CC=2, FRAG=88%"""),
    "passed_organs_list": (
        "spleen, kidney_right, kidney_left, gallbladder, stomach, colon, "
        "small_bowel, bladder, IVC, adrenal_gland_right, adrenal_gland_left"
    ),
}

# ── Helpers ────────────────────────────────────────────────────────────────────

DIVIDER = "=" * 72

def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def show_prompt(prompt: str, max_lines: int = 30) -> None:
    lines = prompt.strip().splitlines()
    if len(lines) > max_lines:
        shown = lines[:max_lines]
        print("\n".join(shown))
        print(f"  ... [{len(lines) - max_lines} more lines] ...")
    else:
        print(prompt.strip())

def show_response(resp) -> None:
    print(f"\n[Response — {resp.model_name}]")
    print(f"  tokens: {resp.prompt_tokens} prompt + {resp.completion_tokens} completion")
    print()
    print(resp.content)

# ── Test 1: Qwen3-8B — METADATA_EXTRACTION_SINGLE ────────────────────────────

def test_qwen3_8b_metadata():
    section("TEST 1 · Qwen3-8B · METADATA_EXTRACTION_SINGLE")
    print("Cases: CT (abd/pelvis, WWO contrast), MRI (cervical spine), NM (myocardial SPECT)")

    with Qwen3(size="8b") as llm:
        for label, case in TEST_CASES.items():
            prompt = METADATA_EXTRACTION_SINGLE.format(
                file_path=case["path"],
                shape=case["shape"],
            )
            print(f"\n{'─'*60}")
            print(f"  [{label}]  {case['path'].split('/')[-2]}/{case['path'].split('/')[-1]}")
            print(f"  Shape: {case['shape']}")
            print(f"\n[Prompt]\n")
            show_prompt(prompt)
            resp = llm.query(user_prompt=prompt, system_prompt="", max_new_tokens=512)
            show_response(resp)

# ── Test 1b: Qwen3-8B — QC_INTERPRETATION + RADIOMICS_GATING ─────────────────

def test_qwen3_8b_qc():
    section("TEST 1b · Qwen3-8B · QC_INTERPRETATION + RADIOMICS_GATING")
    print("Synthetic pass-4 QC output — same model instance for both prompts")

    # Radiomics gating uses the QC interpretation output as input
    QC_RESULT = {
        "case_id": "CT_AB_PELVIS_COMBO",
        "study_description": "CT_AB_PELVIS_COMBO_WWO_CONTR",
        "overall_quality": "ACCEPTABLE",
        "per_organ_qc_json": json.dumps([
            {"organ": "liver",       "volume_ml": 1420.0, "volume_in_range": True,
             "num_components": 1,    "largest_component_fraction": 1.0,
             "has_overlap": False,   "num_flags": 0, "severity": "PASS"},
            {"organ": "spleen",      "volume_ml": 188.0,  "volume_in_range": True,
             "num_components": 1,    "largest_component_fraction": 1.0,
             "has_overlap": False,   "num_flags": 0, "severity": "PASS"},
            {"organ": "aorta",       "volume_ml": 48.0,   "volume_in_range": True,
             "num_components": 3,    "largest_component_fraction": 0.61,
             "has_overlap": False,   "num_flags": 1, "severity": "WARN",
             "cc_flag": "CC=3"},
            {"organ": "pancreas",    "volume_ml": 18.0,   "volume_in_range": False,
             "num_components": 4,    "largest_component_fraction": 0.48,
             "has_overlap": False,   "num_flags": 2, "severity": "FAIL",
             "volume_flag": "UNDER_VOLUME", "cc_flag": "CC=4"},
            {"organ": "kidney_left", "volume_ml": 142.0,  "volume_in_range": True,
             "num_components": 1,    "largest_component_fraction": 1.0,
             "has_overlap": False,   "num_flags": 0, "severity": "PASS"},
        ], indent=2),
    }

    with Qwen3(size="8b") as llm:
        # QC interpretation
        qc_prompt = QC_INTERPRETATION.format(**QC_TEST_DATA)
        print(f"\n[Prompt — QC_INTERPRETATION]\n")
        show_prompt(qc_prompt)
        resp = llm.query(user_prompt=qc_prompt, system_prompt="", max_new_tokens=768)
        show_response(resp)

        # Radiomics gating (same model, no reload)
        section("TEST 1b (cont.) · Qwen3-8B · RADIOMICS_GATING")
        print("Same model instance — feeds QC output into radiomics gating prompt")
        # RADIOMICS_GATING contains literal {organ, features_allowed: ...} in the
        # return-schema description — use manual substitution to avoid KeyError.
        rad_prompt = (
            RADIOMICS_GATING
            .replace("{case_id}",          QC_RESULT["case_id"])
            .replace("{study_description}", QC_RESULT["study_description"])
            .replace("{overall_quality}",  QC_RESULT["overall_quality"])
            .replace("{per_organ_qc_json}", QC_RESULT["per_organ_qc_json"])
        )
        print(f"\n[Prompt — RADIOMICS_GATING]\n")
        show_prompt(rad_prompt)
        resp = llm.query(user_prompt=rad_prompt, system_prompt="", max_new_tokens=768)
        show_response(resp)


# ── Test 2: Qwen3-32B — METADATA_EXTRACTION_BATCH ────────────────────────────

def test_qwen3_32b_batch():
    section("TEST 2 · Qwen3-32B · METADATA_EXTRACTION_BATCH")
    print("Batch of 3 cases passed in a single prompt")

    numbered = "\n".join(
        f"{i+1}. Path: {case['path']}\n   Shape: {case['shape']}"
        for i, (_, case) in enumerate(TEST_CASES.items())
    )
    prompt = METADATA_EXTRACTION_BATCH.format(numbered_paths_with_shapes=numbered)

    print(f"\n[Prompt]\n")
    show_prompt(prompt)

    with Qwen3(size="32b") as llm:
        resp = llm.query(user_prompt=prompt, system_prompt="", max_new_tokens=1024)
        show_response(resp)

# ── Test 3: MedGemma-27B — TOOL_SELECTION ────────────────────────────────────

def test_medgemma_tool_selection():
    section("TEST 3 · MedGemma-27B · TOOL_SELECTION")
    print("Case: CT abd/pelvis with contrast (expected: TotalSeg_CT + MRSegmentator ensemble)")

    # Metadata as it would come out of METADATA_EXTRACTION_SINGLE for the CT case
    case_metadata = json.dumps({
        "modality": "CT",
        "anatomy": "abdomen_pelvis",
        "shape": "3D",
        "is_diagnostic": True,
        "series_type": "axial soft tissue CT delayed phase with contrast",
        "contrast_status": "post",
        "mri_sequence": None,
        "file_path": TEST_CASES["CT"]["path"],
    }, indent=2)

    prompt = TOOL_SELECTION.format(case_metadata_json=case_metadata)

    print(f"\n[Prompt]\n")
    show_prompt(prompt)

    with MedGemma27bText() as llm:
        resp = llm.query(user_prompt=prompt, system_prompt="", max_new_tokens=1024)
        show_response(resp)

        # ── Test 4: MedGemma-27B — QC_INTERPRETATION (reuse loaded model) ────
        section("TEST 4 · MedGemma-27B · QC_INTERPRETATION")
        print("Same model instance reused (no reload) — synthetic pass-4 QC flags")

        qc_prompt = QC_INTERPRETATION.format(**QC_TEST_DATA)

        print(f"\n[Prompt]\n")
        show_prompt(qc_prompt)

        resp = llm.query(user_prompt=qc_prompt, system_prompt="", max_new_tokens=1024)
        show_response(resp)

# ── Test 5: is_diagnostic edge cases — all 3 models ──────────────────────────

# 5 hard cases where naive LLMs often get is_diagnostic wrong.
# Each entry: series_name, shape, expected_is_diagnostic, reason
IS_DIAGNOSTIC_EDGE_CASES = [
    {
        "label":    "BW_neck",
        "path":     "/data/RAD/CONTR/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/20230415/CT_NECK_WWO_CONTR/Neck_3mm_Axial_BW",
        "shape":    "[112, 512, 512]",
        "expected": False,
        "reason":   "Bone window reconstruction (_BW suffix) — not diagnostic",
    },
    {
        "label":    "MIP_chest",
        "path":     "/data/RAD/CONTR/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/20221109/CT_CHEST_PE_PROTOCOL/SUP_INSP__8_0_AX_MIP",
        "shape":    "[1, 512, 512]",
        "expected": False,
        "reason":   "Maximum Intensity Projection (MIP) — 2D projection, not diagnostic",
    },
    {
        "label":    "SUB_mri",
        "path":     "/data/RAD/CONTR/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/20240301/MRI_BREAST_W_GAD/T1_STAR_VIBE_AX_FS_POST_II_SUB",
        "shape":    "[128, 256, 256]",
        "expected": False,
        "reason":   "Subtraction map (_SUB suffix) — post minus pre, not primary diagnostic",
    },
    {
        "label":    "iMAR_abdomen",
        "path":     "/data/RAD/CONTR/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/20231020/CT_AB_PELVIS_WWO_CONTR/Abdomen_Pelvis_3mm_Axial_ST_imar_iMAR",
        "shape":    "[98, 512, 512]",
        "expected": True,
        "reason":   "iMAR combined with diagnostic base name — still diagnostic",
    },
    {
        "label":    "PET_reformat",
        "path":     "/data/RAD/CONTR/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/20220718/PET_CT_WHOLE_BODY/PET_AC_Cor",
        "shape":    "[1, 512, 512]",
        "expected": False,
        "reason":   "PET coronal reformat (PET_AC_Cor) — not primary axial diagnostic volume",
    },
]

EXPECTED_LABEL = {True: "TRUE (diagnostic)", False: "FALSE (non-diagnostic)"}


def _run_is_diagnostic_edge_cases(llm, model_label: str) -> None:
    """Run the 5 edge cases against an already-loaded LLM and print results."""
    section(f"is_diagnostic EDGE CASES · {model_label}")
    print(f"  5 hard cases — model must correctly classify is_diagnostic\n")

    results = []
    for case in IS_DIAGNOSTIC_EDGE_CASES:
        prompt = METADATA_EXTRACTION_SINGLE.format(
            file_path=case["path"],
            shape=case["shape"],
        )
        print(f"\n{'─'*60}")
        print(f"  [{case['label']}]  .../{case['path'].split('/')[-1]}")
        print(f"  Shape: {case['shape']}")
        print(f"  Expected is_diagnostic: {EXPECTED_LABEL[case['expected']]}")
        print(f"  Reason: {case['reason']}")
        print(f"\n[Prompt]\n")
        show_prompt(prompt)

        resp = llm.query(user_prompt=prompt, system_prompt="", max_new_tokens=512)
        show_response(resp)

        # Parse is_diagnostic from response
        import re
        raw = resp.content.strip()
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        predicted = None
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                predicted = parsed.get("is_diagnostic")
            except json.JSONDecodeError:
                pass
        # Fallback: look for is_diagnostic: true/false in raw text
        if predicted is None:
            m = re.search(r'"is_diagnostic"\s*:\s*(true|false)', raw, re.IGNORECASE)
            if m:
                predicted = m.group(1).lower() == "true"

        correct = predicted == case["expected"] if predicted is not None else None
        status = "CORRECT" if correct else ("WRONG" if correct is False else "PARSE_FAIL")
        print(f"\n  >> is_diagnostic predicted: {predicted}  |  expected: {case['expected']}  |  [{status}]")
        results.append({
            "label": case["label"],
            "expected": case["expected"],
            "predicted": predicted,
            "status": status,
        })

    # Summary table
    print(f"\n{'─'*60}")
    print(f"  SUMMARY — {model_label}")
    print(f"  {'Case':<20} {'Expected':<10} {'Predicted':<10} {'Status'}")
    print(f"  {'─'*56}")
    correct_count = 0
    for r in results:
        exp_str = str(r["expected"])
        pred_str = str(r["predicted"]) if r["predicted"] is not None else "?"
        print(f"  {r['label']:<20} {exp_str:<10} {pred_str:<10} {r['status']}")
        if r["status"] == "CORRECT":
            correct_count += 1
    print(f"\n  Score: {correct_count}/{len(results)}")


def test_is_diagnostic_edge_cases_qwen8b():
    section("TEST 5a · Qwen3-8B · is_diagnostic EDGE CASES")
    with Qwen3(size="8b") as llm:
        _run_is_diagnostic_edge_cases(llm, "Qwen3-8B")


def test_is_diagnostic_edge_cases_medgemma():
    section("TEST 5b · MedGemma-27B · is_diagnostic EDGE CASES")
    with MedGemma27bText() as llm:
        _run_is_diagnostic_edge_cases(llm, "MedGemma-27B")


def test_is_diagnostic_edge_cases_qwen32b():
    section("TEST 5c · Qwen3-32B · is_diagnostic EDGE CASES")
    with Qwen3(size="32b") as llm:
        _run_is_diagnostic_edge_cases(llm, "Qwen3-32B")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CDW LLM dry run")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen8b", "qwen32b", "medgemma"],
        default=["qwen8b", "qwen32b", "medgemma"],
        help="Which model tests to run (default: all)",
    )
    parser.add_argument(
        "--edge-cases-only",
        action="store_true",
        help="Run only the is_diagnostic edge case tests",
    )
    args = parser.parse_args()

    print(DIVIDER)
    print("  CDW Pipeline — LLM Dry Run")
    print(f"  Repo: {REPO_ROOT}")
    print(DIVIDER)

    if args.edge_cases_only:
        if "qwen8b" in args.models:
            test_is_diagnostic_edge_cases_qwen8b()
        if "medgemma" in args.models:
            test_is_diagnostic_edge_cases_medgemma()
        if "qwen32b" in args.models:
            test_is_diagnostic_edge_cases_qwen32b()
    else:
        if "qwen8b" in args.models:
            test_qwen3_8b_metadata()
            test_qwen3_8b_qc()
            test_is_diagnostic_edge_cases_qwen8b()

        if "qwen32b" in args.models:
            test_qwen3_32b_batch()
            test_is_diagnostic_edge_cases_qwen32b()

        if "medgemma" in args.models:
            test_medgemma_tool_selection()   # also runs QC_INTERPRETATION internally
            test_is_diagnostic_edge_cases_medgemma()

    print(f"\n{DIVIDER}")
    print("  All tests complete.")
    print(DIVIDER)
