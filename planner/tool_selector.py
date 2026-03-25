"""
CDW Agentic Pipeline — Tool Selector
=====================================
LLM-guided segmentation tool selection.

Given case metadata (from MetadataExtractor), selects which tools to run:
  - primary_tools:   Tier 1 fixed-class tools (run always for coverage)
  - secondary_tools: Tier 2 tools for ensemble / second opinion
  - targeted_tools:  Text-promptable tools (VoxTell / TextMedSeg3D) with
                     explicit organ prompt lists provided by the LLM

Only tools registered in config/tool_registry.json and compatible with the
case modality are returned. Invalid tool names are silently dropped.

Usage
-----
from planner.llm_client import PlannerLLM
from planner.tool_selector import ToolSelector

with PlannerLLM.from_local("checkpoints/medgemma-27b-text-it") as llm:
    selector = ToolSelector(llm)
    plan = selector.select({
        "modality": "CT",
        "anatomy": "abdomen_pelvis",
        "series_type": "axial soft tissue CT with contrast",
        "is_diagnostic": True,
        "shape": "3D",
        "contrast_status": "post",
        "mri_sequence": None,
    })
    # plan: {
    #   "primary_tools": ["TotalSegmentator_CT"],
    #   "secondary_tools": ["MRSegmentator"],
    #   "targeted_tools": [
    #       {"tool": "TextMedSeg3D", "organs": ["adrenal cortex", "psoas muscle"]},
    #   ],
    #   "qc_organs": [...],
    #   "reasoning": "...",
    # }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from config.prompts.example_prompts import TOOL_SELECTION

logger = logging.getLogger(__name__)

# Path to the tool registry
_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "config" / "tool_registry.json"

# Text-promptable tools that require an organ list
_TEXT_PROMPTABLE = {"VoxTell", "TextMedSeg3D"}

# ── Organ name normalization ─────────────────────────────────────────────────
# Maps colloquial / plural LLM outputs → canonical organ_reference.json names.
# Single-value entries map 1:1; list entries expand a plural into its parts.
_ORGAN_NAME_MAP: dict[str, str | list[str]] = {
    # Paired organs — plurals
    "kidneys":        ["kidney_left", "kidney_right"],
    "kidney":         ["kidney_left", "kidney_right"],
    "adrenal glands": ["adrenal_gland_left", "adrenal_gland_right"],
    "adrenal gland":  ["adrenal_gland_left", "adrenal_gland_right"],
    "adrenals":       ["adrenal_gland_left", "adrenal_gland_right"],
    # Lungs — expand to 5 lobes
    "lungs":          ["lung_upper_lobe_left", "lung_lower_lobe_left",
                       "lung_upper_lobe_right", "lung_middle_lobe_right",
                       "lung_lower_lobe_right"],
    "lung":           ["lung_upper_lobe_left", "lung_lower_lobe_left",
                       "lung_upper_lobe_right", "lung_middle_lobe_right",
                       "lung_lower_lobe_right"],
    # Iliac vessels
    "iliac arteries": ["iliac_artery_left", "iliac_artery_right"],
    "iliac veins":    ["iliac_vena_left", "iliac_vena_right"],
    # Common aliases
    "bladder":         "urinary_bladder",
    "ivc":             "inferior_vena_cava",
    "portal vein":     "portal_vein_and_splenic_vein",
    "splenic vein":    "portal_vein_and_splenic_vein",
    "small intestine": "small_bowel",
    "intestines":      ["small_bowel", "colon"],
    "bowel":           ["small_bowel", "colon"],
    # Cardiac
    "atria":           ["atrium_left", "atrium_right"],
    "ventricles":      ["ventricle_left", "ventricle_right"],
    "left ventricle":  "ventricle_left",
    "right ventricle": "ventricle_right",
    "left atrium":     "atrium_left",
    "right atrium":    "atrium_right",
    # Vascular
    "thoracic aorta":  "aorta",
    "superior vena cava": "superior_vena_cava",
    "inferior vena cava": "inferior_vena_cava",
    "hepatic artery":  "hepatic_artery",
}


def _normalize_organ_names(organs: list[str]) -> list[str]:
    """Expand colloquial organ names to canonical organ_reference names.
    Deduplicates and preserves order."""
    seen: set[str] = set()
    result: list[str] = []
    for name in organs:
        # Try lookup with original, then lowercased, then with spaces→underscores
        key = name.strip()
        mapped = _ORGAN_NAME_MAP.get(key) or _ORGAN_NAME_MAP.get(key.lower())
        if mapped is None:
            # Convert spaces to underscores as a final normalization
            canonical = key.lower().replace(" ", "_")
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        elif isinstance(mapped, str):
            if mapped not in seen:
                seen.add(mapped)
                result.append(mapped)
        else:  # list
            for m in mapped:
                if m not in seen:
                    seen.add(m)
                    result.append(m)
    return result


# Prompt for organ-list generation for text-prompted tools
_ORGAN_LIST_PROMPT = """\
Given a medical imaging case with the following metadata:
{case_metadata_json}

And the selected text-prompted tool: {tool_name}
This tool supports open-set 3D segmentation via free-text organ prompts.

List the specific anatomical structures that should be segmented given:
1. The anatomy region and modality
2. Structures NOT already covered by the fixed-class tools also being run: {fixed_tools}
3. Structures of clinical interest for rheumatology/lupus research (focus on joints, muscles, kidneys, organs at risk)

Return a JSON object with:
- organs: list of organ/structure names using canonical names: kidney_left (not "kidneys"), lung_upper_lobe_left (not "lungs"), adrenal_gland_left (not "adrenal glands"), urinary_bladder (not "bladder"), snake_case, singular, with laterality suffix for paired organs
- reasoning: one sentence explaining why these structures were chosen"""


class ToolSelector:
    """
    LLM-guided tool selector.

    Args:
        llm: A loaded PlannerLLM instance.
        registry_path: Path to tool_registry.json. Defaults to config/tool_registry.json.
        generate_organ_prompts: If True (default), call LLM again to generate
                                organ lists for VoxTell / TextMedSeg3D.
    """

    SYSTEM_PROMPT = (
        "You are a medical imaging pipeline planner. "
        "Select tools based on the case metadata and return ONLY valid JSON. "
        "Do not include any reasoning or explanation outside the JSON."
    )

    def __init__(
        self,
        llm,
        registry_path: str | Path = _REGISTRY_PATH,
        generate_organ_prompts: bool = True,
    ):
        self.llm = llm
        self.generate_organ_prompts = generate_organ_prompts
        self._registry = self._load_registry(registry_path)
        self._valid_tool_names, self._modality_map = self._index_registry()

    # ── Public API ──────────────────────────────────────────────────────────

    def select(self, metadata: dict) -> dict:
        """
        Select tools for a single case.

        Args:
            metadata: Dict from MetadataExtractor.extract(). Required keys:
                      modality, anatomy, is_diagnostic, shape.

        Returns:
            Dict with keys: primary_tools, secondary_tools, targeted_tools,
            qc_organs, reasoning.
            targeted_tools is a list of {"tool": str, "organs": list[str]}.
        """
        if not metadata.get("is_diagnostic", True):
            logger.info("Case marked non-diagnostic — skipping tool selection.")
            return self._empty_plan()

        if metadata.get("shape", "3D") != "3D":
            logger.info("2D/4D case — pipeline handles 3D only. Skipping tool selection.")
            return self._empty_plan()

        modality = metadata.get("modality", "UNKNOWN")
        if modality == "UNKNOWN":
            logger.warning("Modality UNKNOWN — cannot select tools reliably.")
            return self._empty_plan()

        # ── LLM tool selection ──────────────────────────────────────────────
        meta_json = json.dumps(metadata, indent=2)
        prompt = TOOL_SELECTION.format(case_metadata_json=meta_json)

        logger.debug("ToolSelector.select(): querying LLM for %s %s", modality, metadata.get("anatomy"))
        try:
            raw = self.llm.query_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                max_new_tokens=4096,
                temperature=0.1,
            )
        except Exception as e:
            logger.warning("LLM tool selection failed: %s", e)
            return self._fallback(modality)

        plan = self._validate_plan(raw, modality)

        # ── Organ prompts for text-promptable tools ─────────────────────────
        if self.generate_organ_prompts:
            plan = self._fill_organ_prompts(plan, metadata)

        return plan

    # ── Internal helpers ────────────────────────────────────────────────────

    def _validate_plan(self, raw: dict, modality: str) -> dict:
        """
        Strip unknown tool names and filter by modality compatibility.
        """
        if not isinstance(raw, dict):
            logger.warning("Tool selection response is not a dict; using fallback.")
            return self._fallback(modality)

        primary   = self._filter_tools(raw.get("primary_tools",   []), modality)
        secondary = self._filter_tools(raw.get("secondary_tools", []), modality)
        targeted_raw = raw.get("targeted_tools", [])

        # targeted_tools may be plain names or dicts {"tool": ..., "organs": [...]}
        targeted = []
        for item in targeted_raw:
            if isinstance(item, str):
                name = item
                organs = []
            elif isinstance(item, dict):
                name = item.get("tool", "")
                organs = item.get("organs", [])
            else:
                continue
            if name in self._valid_tool_names and self._tool_supports_modality(name, modality):
                targeted.append({"tool": name, "organs": organs})

        qc_organs = raw.get("qc_organs", [])
        if not isinstance(qc_organs, list):
            qc_organs = []

        # Normalize organ names to canonical form (kidney_left, not "kidneys")
        for entry in targeted:
            entry["organs"] = _normalize_organ_names(entry["organs"])
        qc_organs = _normalize_organ_names([str(o) for o in qc_organs])

        return {
            "primary_tools":   primary,
            "secondary_tools": secondary,
            "targeted_tools":  targeted,
            "qc_organs":       qc_organs,
            "reasoning":       str(raw.get("reasoning", "")),
        }

    def _filter_tools(self, names: Any, modality: str) -> list[str]:
        """Return only valid tool names compatible with the given modality."""
        if not isinstance(names, list):
            return []
        out = []
        for n in names:
            name = str(n)
            if name in self._valid_tool_names and self._tool_supports_modality(name, modality):
                out.append(name)
            else:
                logger.debug("Dropping tool '%s' (unknown or incompatible with %s)", name, modality)
        return out

    def _tool_supports_modality(self, tool_name: str, modality: str) -> bool:
        """Check if a tool supports the given modality."""
        supported = self._modality_map.get(tool_name, [])
        if not supported:
            return True  # unknown → don't block
        modality_upper = modality.upper()
        return any(m.upper() == modality_upper for m in supported)

    def _fill_organ_prompts(self, plan: dict, metadata: dict) -> dict:
        """
        For each text-promptable tool in the plan with an empty organs list,
        ask the LLM to generate a targeted organ list.
        """
        fixed_tools = plan["primary_tools"] + plan["secondary_tools"]
        meta_json = json.dumps(metadata, indent=2)

        new_targeted = []
        for entry in plan["targeted_tools"]:
            tool_name = entry["tool"]
            if tool_name in _TEXT_PROMPTABLE and not entry["organs"]:
                logger.debug("Generating organ prompts for %s via LLM ...", tool_name)
                try:
                    prompt = _ORGAN_LIST_PROMPT.format(
                        case_metadata_json=meta_json,
                        tool_name=tool_name,
                        fixed_tools=", ".join(fixed_tools) if fixed_tools else "none",
                    )
                    result = self.llm.query_json(
                        system_prompt="You are a medical imaging anatomy expert. Return only valid JSON. Do not use <think> tags.",
                        user_prompt=prompt,
                        max_new_tokens=2048,
                        temperature=0.1,
                    )
                    organs = result.get("organs", []) if isinstance(result, dict) else []
                    entry = {"tool": tool_name, "organs": _normalize_organ_names([str(o) for o in organs])}
                except Exception as e:
                    logger.warning("Organ prompt generation failed for %s: %s", tool_name, e)
            new_targeted.append(entry)

        plan["targeted_tools"] = new_targeted
        return plan

    def _load_registry(self, registry_path) -> dict:
        with open(registry_path) as f:
            return json.load(f)

    def _index_registry(self) -> tuple[set[str], dict[str, list[str]]]:
        """Build a set of all valid tool names and a modality support map.
        Skips tools marked as deferred."""
        valid_names: set[str] = set()
        modality_map: dict[str, list[str]] = {}

        for section in ("fixed_class_tools", "text_promptable_tools", "label_prompted_tools"):
            for tool in self._registry.get(section, []):
                if tool.get("deferred"):
                    continue
                name = tool.get("name", "")
                if name:
                    valid_names.add(name)
                    modality_map[name] = tool.get("supported_modalities", [])

        return valid_names, modality_map

    def _empty_plan(self) -> dict:
        return {
            "primary_tools":   [],
            "secondary_tools": [],
            "targeted_tools":  [],
            "qc_organs":       [],
            "reasoning":       "No tools selected (non-diagnostic, 2D, or unknown modality).",
        }

    def _fallback(self, modality: str) -> dict:
        """
        Return ALL registry-compatible tools when LLM fails.
        Maximum coverage: every tool that supports the case modality.
        """
        primary = []
        targeted = []
        for name in sorted(self._valid_tool_names):
            if self._tool_supports_modality(name, modality):
                if name in _TEXT_PROMPTABLE:
                    targeted.append({"tool": name, "organs": []})
                else:
                    primary.append(name)

        return {
            "primary_tools":   primary,
            "secondary_tools": [],
            "targeted_tools":  targeted,
            "qc_organs":       [],
            "reasoning":       f"LLM failed — fallback: all {modality}-compatible tools selected.",
        }
