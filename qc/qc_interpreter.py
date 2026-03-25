"""
CDW Agentic Pipeline — Tier 3 LLM QC Interpretation
====================================================
Uses LLM to interpret geometric QC flags and generate:
  1. Clinical quality assessment (QC_INTERPRETATION)
  2. Per-organ radiomics gating decisions (RADIOMICS_GATING)

Requires a PlannerLLM instance (default: local Transformers; optional: vLLM).

Usage
-----
from planner import PlannerLLM
from qc.qc_interpreter import QCInterpreter

llm = PlannerLLM.from_local("checkpoints/qwen3-8b")
llm.load()
interpreter = QCInterpreter(llm)
interpretation = interpreter.interpret(tool_qc_result, study_description="CT ABD/PELVIS")
print(interpretation.overall_quality, interpretation.extract, interpretation.skip)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from config.prompts.example_prompts import QC_INTERPRETATION, RADIOMICS_GATING
from planner.llm_client import PlannerLLM, _parse_json_from_response
from qc.dataclasses import QCInterpretation, ToolQCResult

logger = logging.getLogger(__name__)


class QCInterpreter:
    """
    Tier 3 LLM-based QC interpretation and radiomics gating.

    Args:
        llm: A PlannerLLM instance (default: local transformers; optional: vLLM).
        max_tokens: Maximum tokens for LLM generation (default 8192).
    """

    def __init__(self, llm: PlannerLLM, max_tokens: int = 8192):
        self.llm = llm
        self.max_tokens = max_tokens

    def interpret(
        self,
        tool_result: ToolQCResult,
        study_description: str = "",
        anatomy: str = "",
    ) -> QCInterpretation:
        """
        Run both QC interpretation and radiomics gating for one tool's results.

        Args:
            tool_result: Tier 1 ToolQCResult from GeometricQC.
            study_description: Study description string (e.g. "CT ABD/PELVIS").
            anatomy: Anatomy region (e.g. "ABDOMEN").

        Returns:
            QCInterpretation with clinical assessment and radiomics decisions.
        """
        interp = QCInterpretation(
            case_id=tool_result.case_id,
            case_path=tool_result.case_path,
            tool_name=tool_result.tool_name,
        )

        # Step 1: QC Interpretation
        self._run_interpretation(tool_result, interp, study_description, anatomy)

        # Step 2: Radiomics Gating (uses interpretation output)
        self._run_radiomics_gating(tool_result, interp, study_description)

        return interp

    # ── Step 1: QC Interpretation ──────────────────────────────────────────

    def _run_interpretation(
        self,
        tool_result: ToolQCResult,
        interp: QCInterpretation,
        study_description: str,
        anatomy: str,
    ) -> None:
        """Populate clinical QC interpretation fields via LLM."""
        flagged, passed = self._format_organ_lists(tool_result)

        prompt = QC_INTERPRETATION.format(
            study_description=study_description or tool_result.case_id,
            anatomy=anatomy or "UNKNOWN",
            tool_name=tool_result.tool_name or "unknown",
            num_organs_in_fov=tool_result.num_organs_in_fov,
            flagged_organs_formatted=flagged,
            passed_organs_list=passed,
        )

        try:
            response = self.llm.query_json(system_prompt="", user_prompt=prompt, max_new_tokens=self.max_tokens)
            if response is None:
                logger.warning("[%s] LLM returned no valid JSON for QC interpretation", tool_result.case_id)
                # Derive quality from geometric severity instead of blindly defaulting
                sev = tool_result.worst_severity
                quality_map = {"PASS": "GOOD", "WARN": "ACCEPTABLE", "FAIL": "POOR"}
                interp.overall_quality = quality_map.get(sev, "ACCEPTABLE")
                interp.explanation = (
                    f"LLM unavailable — derived from geometric QC severity ({sev})"
                )
                return

            interp.overall_quality = response.get("overall_quality", "ACCEPTABLE")
            interp.usable_for_radiomics = response.get("usable_for_radiomics", [])
            interp.unusable_organs = response.get("unusable_organs", [])
            interp.first_order_safe = response.get("first_order_safe", [])
            interp.shape_safe = response.get("shape_safe", [])
            interp.texture_safe = response.get("texture_safe", [])
            interp.explanation = response.get("explanation", "")
            interp.raw_response = json.dumps(response, indent=2)

        except Exception as e:
            logger.error("[%s] QC interpretation failed: %s", tool_result.case_id, e)
            sev = tool_result.worst_severity
            quality_map = {"PASS": "GOOD", "WARN": "ACCEPTABLE", "FAIL": "POOR"}
            interp.overall_quality = quality_map.get(sev, "ACCEPTABLE")
            interp.explanation = f"LLM error: {e}. Derived from geometric severity ({sev})"

    # ── Step 2: Radiomics Gating ───────────────────────────────────────────

    def _run_radiomics_gating(
        self,
        tool_result: ToolQCResult,
        interp: QCInterpretation,
        study_description: str,
    ) -> None:
        """Populate radiomics gating decisions via LLM."""
        per_organ_qc = self._build_per_organ_json(tool_result)

        # RADIOMICS_GATING contains literal {organ, features_allowed: ...} in the
        # return-schema description — use manual substitution to avoid str.format() KeyError.
        prompt = (
            RADIOMICS_GATING
            .replace("{case_id}",           tool_result.case_id)
            .replace("{study_description}", study_description or tool_result.case_id)
            .replace("{overall_quality}",   interp.overall_quality)
            .replace("{per_organ_qc_json}", per_organ_qc)
        )

        try:
            response = self.llm.query_json(system_prompt="", user_prompt=prompt, max_new_tokens=self.max_tokens)
            if response is None:
                logger.warning("[%s] LLM returned no valid JSON for radiomics gating", tool_result.case_id)
                interp.notes = "LLM gating unavailable — using QC flags for fallback decisions"
                self._fallback_gating(tool_result, interp)
                return

            interp.extract = response.get("extract", [])
            interp.skip = response.get("skip", [])
            interp.notes = response.get("notes", "")

        except Exception as e:
            logger.error("[%s] Radiomics gating failed: %s", tool_result.case_id, e)
            interp.notes = f"LLM gating error: {e}. Using fallback."
            self._fallback_gating(tool_result, interp)

    # ── Formatting helpers ─────────────────────────────────────────────────

    def _format_organ_lists(self, tool_result: ToolQCResult) -> tuple[str, str]:
        """Format flagged and passed organ lists for prompt insertion."""
        flagged_lines = []
        passed_organs = []

        for oqc in tool_result.organ_results:
            if oqc.volume_ml < 1e-3:
                continue  # skip organs not in FOV

            flags = oqc.compute_severity()
            if flags:
                flag_str = "; ".join(flags)
                flagged_lines.append(f"- {oqc.organ}: {flag_str}")
            else:
                passed_organs.append(oqc.organ)

        flagged = "\n".join(flagged_lines) if flagged_lines else "(none)"
        passed = ", ".join(passed_organs) if passed_organs else "(none)"
        return flagged, passed

    def _build_per_organ_json(self, tool_result: ToolQCResult) -> str:
        """Build per-organ QC JSON for radiomics gating prompt."""
        organs = []
        for oqc in tool_result.organ_results:
            if oqc.volume_ml < 1e-3:
                continue
            entry = {
                "organ": oqc.organ,
                "volume_ml": round(oqc.volume_ml, 1),
                "volume_in_range": oqc.volume_in_range,
                "num_components": oqc.num_components,
                "largest_component_fraction": round(oqc.largest_component_fraction, 3),
                "has_overlap": oqc.has_overlap,
                "num_flags": oqc.num_flags,
                "severity": oqc.severity,
            }
            if oqc.volume_flag:
                entry["volume_flag"] = oqc.volume_flag
            if oqc.cc_flag:
                entry["cc_flag"] = oqc.cc_flag
            if oqc.ratio_flag:
                entry["ratio_flag"] = oqc.ratio_flag
            if oqc.overlap_flag:
                entry["overlap_flag"] = oqc.overlap_flag
            organs.append(entry)
        return json.dumps(organs, indent=2)

    def _fallback_gating(
        self, tool_result: ToolQCResult, interp: QCInterpretation
    ) -> None:
        """Rule-based fallback when LLM gating is unavailable."""
        for oqc in tool_result.organ_results:
            if oqc.volume_ml < 1e-3:
                continue

            if oqc.severity == "PASS":
                interp.extract.append({
                    "organ": oqc.organ,
                    "features_allowed": ["first_order", "shape", "texture"],
                    "postprocessing_needed": ["none"],
                })
                interp.usable_for_radiomics.append(oqc.organ)
                interp.first_order_safe.append(oqc.organ)
                interp.shape_safe.append(oqc.organ)
                interp.texture_safe.append(oqc.organ)
            elif oqc.severity == "WARN":
                # Allow extraction with caution — feature tiers depend on flag type
                features = ["first_order"]
                postprocessing = []
                interp.first_order_safe.append(oqc.organ)

                if oqc.largest_component_fraction >= 0.9:
                    features.extend(["shape", "texture"])
                    interp.shape_safe.append(oqc.organ)
                    interp.texture_safe.append(oqc.organ)
                elif oqc.largest_component_fraction >= 0.7:
                    # Fragmented but recoverable — shape OK after LCC, no texture
                    features.append("shape")
                    interp.shape_safe.append(oqc.organ)

                if oqc.num_components > 1:
                    postprocessing.append("lcc")
                if not postprocessing:
                    postprocessing.append("none")

                interp.extract.append({
                    "organ": oqc.organ,
                    "features_allowed": features,
                    "postprocessing_needed": postprocessing,
                })
                interp.usable_for_radiomics.append(oqc.organ)
            else:
                # FAIL — skip
                flags = []
                if oqc.volume_flag:
                    flags.append(oqc.volume_flag)
                if oqc.cc_flag:
                    flags.append(oqc.cc_flag)
                interp.skip.append({
                    "organ": oqc.organ,
                    "reason": "; ".join(flags) if flags else "Multiple QC failures",
                })
                interp.unusable_organs.append(oqc.organ)
