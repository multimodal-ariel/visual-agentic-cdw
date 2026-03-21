"""
CDW Agentic Pipeline — Organ List Generator
============================================
Generates the list of anatomical organs expected within the field of view
for a given scan, for use in QC checks and radiomics gating.

Two modes:
  1. Fast (default): deterministic lookup from config/organ_reference.json
     anatomy_to_expected_organs map. No LLM call needed. Covers all standard
     anatomies in the pipeline.

  2. LLM fallback: for ambiguous or UNKNOWN anatomies, call the LLM using the
     ORGAN_LIST_GENERATION prompt. Returns primary, edge, and exclude lists.

Output schema:
    {
        "primary_organs": [...],   # fully in FOV — always check in QC
        "edge_organs":    [...],   # partially in FOV — check with relaxed thresholds
        "exclude_organs": [...],   # not expected — skip in QC
    }

Usage
-----
# Fast lookup (no LLM):
gen = OrganListGenerator()
result = gen.generate("abdomen_pelvis", modality="CT")

# With LLM fallback for unusual cases:
gen = OrganListGenerator(llm=llm)
result = gen.generate("UNKNOWN", modality="CT", series_type="renal mass protocol CT")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from config.prompts.example_prompts import ORGAN_LIST_GENERATION

logger = logging.getLogger(__name__)

_ORGAN_REF_PATH = Path(__file__).resolve().parent.parent / "config" / "organ_reference.json"

# Anatomy regions where edge organs are expected (partially in FOV)
# Key: anatomy → edge organs likely at the boundary
_EDGE_ORGANS: dict[str, list[str]] = {
    "chest": ["liver", "stomach", "spleen"],
    "abdomen": [
        "lung_lower_lobe_left", "lung_lower_lobe_right",
        "urinary_bladder", "iliac_artery_left", "iliac_artery_right",
    ],
    "pelvis": [
        "kidney_left", "kidney_right", "colon", "aorta",
    ],
    "abdomen_pelvis": [
        "lung_lower_lobe_left", "lung_lower_lobe_right",
    ],
    "chest_abdomen_pelvis": [],
    "whole_body": [],
    "neck": [
        "lung_upper_lobe_left", "lung_upper_lobe_right", "heart",
    ],
    "head": ["neck"],
    "spine": [
        "aorta", "inferior_vena_cava",
    ],
}


class OrganListGenerator:
    """
    Generate expected organ lists for a given anatomy region.

    Args:
        llm: Optional PlannerLLM. If provided, used as fallback for unknown anatomies.
             Not required for standard anatomies covered by organ_reference.json.
        organ_ref_path: Path to organ_reference.json.
    """

    SYSTEM_PROMPT = (
        "You are a medical imaging anatomy expert. "
        "Return ONLY valid JSON. Do not include explanations or markdown."
    )

    def __init__(self, llm=None, organ_ref_path: str | Path = _ORGAN_REF_PATH):
        self.llm = llm
        self._ref = self._load_ref(organ_ref_path)
        self._anatomy_map: dict[str, list[str]] = self._ref.get("anatomy_to_expected_organs", {})

    # ── Public API ──────────────────────────────────────────────────────────

    def generate(
        self,
        anatomy: str,
        modality: str = "CT",
        series_type: str = "",
    ) -> dict:
        """
        Return expected organ lists for the given anatomy.

        Args:
            anatomy: Anatomy string from MetadataExtractor
                     (e.g. "abdomen_pelvis", "chest", "UNKNOWN").
            modality: Imaging modality (e.g. "CT", "MRI"). Used to filter
                      organs from MRI-only tools in QC.
            series_type: Series description for LLM fallback context.

        Returns:
            Dict with keys: primary_organs, edge_organs, exclude_organs.
        """
        anatomy_lower = anatomy.lower()

        # Fast path: known anatomy in organ reference
        if anatomy_lower in self._anatomy_map:
            return self._from_lookup(anatomy_lower)

        # LLM path: UNKNOWN or unusual anatomy
        if self.llm is not None:
            logger.info(
                "OrganListGenerator: anatomy '%s' not in lookup — querying LLM.", anatomy
            )
            return self._from_llm(anatomy, modality, series_type)

        # No LLM: return empty (QC will skip organ checks)
        logger.warning(
            "OrganListGenerator: anatomy '%s' unknown and no LLM provided — returning empty list.", anatomy
        )
        return {"primary_organs": [], "edge_organs": [], "exclude_organs": []}

    def organs_for_qc(self, anatomy: str, modality: str = "CT") -> list[str]:
        """
        Convenience method: return the flat list of organs to QC-check
        (primary + edge organs combined).
        """
        result = self.generate(anatomy, modality)
        return result["primary_organs"] + result["edge_organs"]

    # ── Internal ────────────────────────────────────────────────────────────

    def _from_lookup(self, anatomy: str) -> dict:
        primary = list(self._anatomy_map.get(anatomy, []))
        edge = list(_EDGE_ORGANS.get(anatomy, []))

        # Remove edge organs from primary (they're already in primary from the map)
        # and build exclude list as everything in the reference NOT in primary/edge
        all_known = set(self._ref.get("volume_ranges_ml", {}).keys())
        in_fov = set(primary) | set(edge)
        exclude = sorted(all_known - in_fov)

        return {
            "primary_organs": primary,
            "edge_organs":    edge,
            "exclude_organs": exclude,
        }

    def _from_llm(self, anatomy: str, modality: str, series_type: str) -> dict:
        prompt = ORGAN_LIST_GENERATION.format(
            anatomy=anatomy,
            modality=modality,
            series_type=series_type or "not specified",
        )
        try:
            result = self.llm.query_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                max_new_tokens=512,
                temperature=0.1,
            )
        except Exception as e:
            logger.warning("LLM organ list generation failed: %s", e)
            return {"primary_organs": [], "edge_organs": [], "exclude_organs": []}

        if not isinstance(result, dict):
            return {"primary_organs": [], "edge_organs": [], "exclude_organs": []}

        return {
            "primary_organs": [str(o) for o in result.get("primary_organs", [])],
            "edge_organs":    [str(o) for o in result.get("edge_organs",    [])],
            "exclude_organs": [str(o) for o in result.get("exclude_organs", [])],
        }

    def _load_ref(self, path) -> dict:
        with open(path) as f:
            return json.load(f)
