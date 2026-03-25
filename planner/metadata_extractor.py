"""
CDW Agentic Pipeline — Metadata Extractor
==========================================
Extracts structured metadata from clinical imaging file paths using the LLM.

Wraps the METADATA_EXTRACTION_SINGLE and METADATA_EXTRACTION_BATCH prompts
validated in config/prompts/example_prompts.py.

Output schema (per case):
    {
        "modality":        "CT" | "MRI" | "PET_CT" | "NM" | "CR" | "US" | "FL" | "UNKNOWN",
        "anatomy":         "head" | "neck" | "chest" | "abdomen" | "pelvis" |
                           "abdomen_pelvis" | "chest_abdomen_pelvis" |
                           "spine" | "extremity" | "whole_body" | "cardiac" | "UNKNOWN",
        "shape":           "2D" | "3D" | "4D",
        "is_diagnostic":   true | false,
        "series_type":     str,          # e.g. "axial soft tissue CT with contrast"
        "contrast_status": "pre" | "post" | "with" | "without" | "unknown",
        "mri_sequence":    str | null,   # e.g. "T1", "VIBE", "Dixon" or null for non-MRI
    }

Usage
-----
from planner.llm_client import PlannerLLM
from planner.metadata_extractor import MetadataExtractor

with PlannerLLM.from_local("checkpoints/medgemma-27b-text-it") as llm:
    extractor = MetadataExtractor(llm)
    meta = extractor.extract("/data/RAD/RHEUM/PT123/CT_ABD_PELVIS/image_nifti.nii.gz",
                              shape=[512, 512, 237])
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config.prompts.example_prompts import (
    METADATA_EXTRACTION_BATCH,
    METADATA_EXTRACTION_SINGLE,
)

logger = logging.getLogger(__name__)

# Valid field values — used for post-processing validation
_VALID_MODALITIES = {"CT", "MRI", "PET_CT", "NM", "CR", "US", "FL", "UNKNOWN"}
_VALID_ANATOMIES = {
    "head", "neck", "chest", "abdomen", "pelvis",
    "abdomen_pelvis", "chest_abdomen_pelvis", "spine",
    "extremity", "whole_body", "cardiac", "UNKNOWN",
}
_VALID_SHAPES = {"2D", "3D", "4D"}
_VALID_CONTRAST = {"pre", "post", "with", "without", "unknown"}

# Heuristic modality keywords (for fast pre-screening; NOT used as ground truth)
_PATH_MODALITY_HINTS = {
    "CT": ["CT", "_ct_", "/ct/", "computed", "tomography"],
    "MRI": ["MR", "MRI", "_mr_", "/mr/", "magnetic", "resonance", "VIBE", "HASTE", "T1", "T2", "FLAIR", "DWI"],
    "PET_CT": ["PET", "pet", "PET_CT", "PETCT"],
    "NM": ["NM", "nuclear", "scintigraphy", "SPECT"],
}


class MetadataExtractor:
    """
    Extract structured clinical metadata from NIfTI file paths using LLM.

    Args:
        llm: A loaded PlannerLLM (or TransformersLLM / VLLMClient) instance.
             For TransformersLLM, the model must be loaded before calling extract().
    """

    SYSTEM_PROMPT = (
        "You are a medical imaging metadata parser. "
        "Return ONLY valid JSON. Do not add explanations or markdown fences."
    )

    def __init__(self, llm):
        self.llm = llm

    # ── Public API ──────────────────────────────────────────────────────────

    def extract(self, file_path: str, shape: Any = None) -> dict:
        """
        Extract metadata for a single file.

        Args:
            file_path: Path to the NIfTI file (or its parent directory).
                       The series name is derived from the path components.
            shape: Volume shape as a list [X, Y, Z] or string "[512, 512, 237]".
                   Pass None if unknown.

        Returns:
            Metadata dict with keys: modality, anatomy, shape, is_diagnostic,
            series_type, contrast_status, mri_sequence.
        """
        shape_str = str(shape) if shape is not None else "unknown"
        prompt = METADATA_EXTRACTION_SINGLE.format(
            file_path=file_path,
            shape=shape_str,
        )

        logger.debug("MetadataExtractor.extract(): querying LLM for %s", file_path)
        try:
            result = self.llm.query_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                max_new_tokens=256,
                temperature=0.1,
            )
        except Exception as e:
            logger.warning("LLM metadata extraction failed for %s: %s", file_path, e)
            return self._fallback(file_path)

        return self._validate(result, file_path)

    def extract_batch(self, cases: list[dict]) -> list[dict]:
        """
        Extract metadata for a batch of cases in a single LLM call.

        Args:
            cases: List of dicts, each with keys:
                   - "file_path" (str, required)
                   - "shape" (any, optional)

        Returns:
            List of metadata dicts, one per case (same order as input).
            Falls back to per-case extraction if batch parse fails.
        """
        if not cases:
            return []

        lines = []
        for i, c in enumerate(cases, 1):
            path = c["file_path"]
            shape = c.get("shape", "unknown")
            lines.append(f"{i}. {path}  [shape: {shape}]")

        numbered = "\n".join(lines)
        prompt = METADATA_EXTRACTION_BATCH.format(numbered_paths_with_shapes=numbered)

        logger.debug("MetadataExtractor.extract_batch(): querying LLM for %d cases", len(cases))
        try:
            results = self.llm.query_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                max_new_tokens=256,
                temperature=0.1,
            )
            if not isinstance(results, list) or len(results) != len(cases):
                raise ValueError(
                    f"Batch response has {len(results) if isinstance(results, list) else 'non-list'} "
                    f"items; expected {len(cases)}."
                )
            return [self._validate(r, c["file_path"]) for r, c in zip(results, cases)]
        except Exception as e:
            logger.warning(
                "Batch metadata extraction failed (%s). Falling back to per-case extraction.", e
            )
            return [self.extract(c["file_path"], c.get("shape")) for c in cases]

    # ── Validation + fallback ───────────────────────────────────────────────

    def _validate(self, raw: dict, file_path: str) -> dict:
        """Coerce field values to the canonical vocabulary. Log warnings for unexpected values."""
        if not isinstance(raw, dict):
            logger.warning("Metadata for %s is not a dict; using fallback.", file_path)
            return self._fallback(file_path)

        modality = str(raw.get("modality", "UNKNOWN")).upper()
        if modality not in _VALID_MODALITIES:
            logger.warning("Unexpected modality '%s' for %s; setting UNKNOWN.", modality, file_path)
            modality = "UNKNOWN"

        anatomy = str(raw.get("anatomy", "UNKNOWN")).lower()
        if anatomy not in _VALID_ANATOMIES:
            logger.warning("Unexpected anatomy '%s' for %s; setting UNKNOWN.", anatomy, file_path)
            anatomy = "UNKNOWN"

        shape = str(raw.get("shape", "3D")).upper()
        if shape not in _VALID_SHAPES:
            shape = "3D"

        contrast = str(raw.get("contrast_status", "unknown")).lower()
        if contrast not in _VALID_CONTRAST:
            contrast = "unknown"

        is_diagnostic = bool(raw.get("is_diagnostic", True))

        # Post LLM heuristic correction for is_diagnostic edge cases
        # The LLM misses embedded markers (_cor_ mid-string, SUB_ prefix,
        # Sagittal/Coronal concatenated without separators). Code catches these
        # reliably in microseconds without burning tokens.
        if is_diagnostic:
            path_lower = file_path.lower()
            non_diag_markers = [
                "_cor_", "_cor.", "coronal", "sagittal",
                "_sag_", "_sag.",
                "_bw", "_bone",
            ]
            sub_markers = ["sub_", "_sub_", "_sub."]
            if any(m in path_lower for m in non_diag_markers):
                is_diagnostic = False
                logger.debug("Post-LLM correction: %s → non-diagnostic (reformat marker)", file_path)
            elif any(path_lower.endswith(m.rstrip(".")) or m in path_lower for m in sub_markers):
                is_diagnostic = False
                logger.debug("Post-LLM correction: %s → non-diagnostic (subtraction marker)", file_path)
        
        return {
            "file_path":      file_path,
            "modality":       modality,
            "anatomy":        anatomy,
            "shape":          shape,
            "is_diagnostic":  is_diagnostic,
            "series_type":    str(raw.get("series_type", "")),
            "contrast_status": contrast,
            "mri_sequence":   raw.get("mri_sequence"),  # None for non-MRI
        }
        
    def _fallback(self, file_path: str) -> dict:
        """Heuristic fallback when LLM fails - extracts modality and anatomy from a path."""
        from orchestrator.pipeline import CasePipeline
        meta = CasePipeline._metadata_from_path(None, file_path)
        # supplement with defaults for the fields the heuristic cannot produce
        meta.setdefault("file_path", file_path)
        meta.setdefault("shape", "3D")
        meta.setdefault("series_type", "")
        meta.setdefault("mri_sequence", None)
        logger.info("Fallback heuristic for %s: modality=%s anatompy=%s",
                        file_path, meta.get("modality"), meta.get("anatomy"))
        return meta