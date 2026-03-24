"""
CDW Agentic Pipeline — Single-Case Pipeline
============================================
End-to-end flow for one case:

  input path
    → metadata_extractor (LLM or dry-run)
    → is_diagnostic check (skip if false)
    → tool_selector (which models to run)
    → organ_list_generator (which organs to QC)
    → for each selected tool:
        → tool.run() (or tool._dry_run())
        → postprocess masks (LCC, closing, hole fill)
    → geometric_qc on postprocessed masks
    → if multi-tool: multi_tool_qc (pairwise Dice, STAPLE)
    → qc_interpreter (LLM generates clinical report)
    → radiomics_gating (which organs to extract)
    → radiomics extraction (PyRadiomics on passing organs)
    → save results (per-case JSON + append to aggregate CSV)

Two-model architecture:
  - planner_llm (Qwen3-8B): metadata, is_diagnostic, tool selection — fast, 1 GPU
  - clinical_llm (MedGemma-27B): QC interpretation, radiomics gating — 2 GPUs, only loaded when needed

Usage
-----
from orchestrator.pipeline import CasePipeline
from planner import PlannerLLM

# Two-model production flow
planner = PlannerLLM.from_local("checkpoints/qwen3-8b")
clinical = PlannerLLM.from_local("checkpoints/medgemma-27b-text-it")
pipeline = CasePipeline(planner_llm=planner, clinical_llm=clinical, device="gpu:0")
result = pipeline.run("/data/soumitri/segmentations_3d/RHEUM/PT123/CT_ABD")

# Dry-run (no LLM, mock segmentation, real QC)
pipeline = CasePipeline(dry_run=True, no_llm=True)
result = pipeline.run(case_path)
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.constants import (
    IMAGE_FILENAME,
    LOG_DIR,
    TOOL_OUTPUT_DIRS,
)
from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput

logger = logging.getLogger(__name__)

# Module-level cache for extract_pyradiomics to survive sys.modules swap
_EXTRACT_PYRADIOMICS = None

def _get_extract_pyradiomics():
    """Import and cache extract_pyradiomics once to avoid naming collision."""
    global _EXTRACT_PYRADIOMICS
    if _EXTRACT_PYRADIOMICS is None:
        from radiomics.pyradiomics import extract_pyradiomics
        _EXTRACT_PYRADIOMICS = extract_pyradiomics
    return _EXTRACT_PYRADIOMICS


# ──────────────────────────────────────────────────────────────────────────────
# Case result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    """Full pipeline result for one case."""
    case_id: str
    case_path: str
    status: str = "pending"             # completed / failed / skipped

    # Step outputs
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_diagnostic: bool = True
    selected_tools: List[str] = field(default_factory=list)
    expected_organs: List[str] = field(default_factory=list)

    tool_outputs: List[Dict] = field(default_factory=list)    # serialized ToolOutputs
    qc_report: Dict[str, Any] = field(default_factory=dict)   # serialized CaseQCReport
    radiomics: List[Dict] = field(default_factory=list)        # per-organ feature summaries

    # Timing
    total_time_s: float = 0.0
    step_times: Dict[str, float] = field(default_factory=dict)

    # Errors
    error: str = ""
    warnings: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Tool loader (dynamic import from tool_registry.json class path)
# ──────────────────────────────────────────────────────────────────────────────

_tool_cache: Dict[str, BaseSegmentationTool] = {}


def _load_tool(class_path: str) -> BaseSegmentationTool:
    """
    Dynamically import and instantiate a tool from its dotted class path.
    e.g. "tools.totalsegmentator_ct.TotalSegmentatorCTTool"
    """
    if class_path in _tool_cache:
        return _tool_cache[class_path]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls()
    _tool_cache[class_path] = instance
    return instance


def _load_tool_registry() -> dict:
    """Load tool_registry.json."""
    reg_path = Path(__file__).resolve().parent.parent / "config" / "tool_registry.json"
    with open(reg_path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Single-case pipeline
# ──────────────────────────────────────────────────────────────────────────────

class CasePipeline:
    """
    End-to-end pipeline for a single case.

    Two-model architecture (validated 2026-03-20):
      - planner_llm (Qwen3-8B): metadata extraction, tool selection — fast, rule-following
      - clinical_llm (MedGemma-27B): QC interpretation, radiomics gating — clinical domain knowledge

    Args:
        planner_llm: LLM for metadata/tool selection (Qwen3-8B recommended). Default: None (no_llm).
        clinical_llm: LLM for QC interpretation/radiomics gating (MedGemma-27B). Default: None (fallback rules).
        llm: Single LLM for all tasks (legacy, used if planner_llm/clinical_llm not set).
        device: GPU device string (e.g. "gpu:0").
        dry_run: If True, tools produce mock output instead of real segmentation.
        no_llm: If True, skip all LLM calls (use default metadata/tool selection).
        postprocess: If True, apply LCC/closing/hole-fill to masks after segmentation.
        skip_radiomics: If True, skip radiomics extraction.
    """

    def __init__(
        self,
        planner_llm=None,
        clinical_llm=None,
        llm=None,
        device: str = "gpu:0",
        dry_run: bool = False,
        no_llm: bool = False,
        postprocess: bool = True,
        skip_radiomics: bool = False,
    ):
        # Two-model routing: planner (Qwen3-8B) + clinical (MedGemma-27B)
        # Falls back to single `llm` if specific models not provided
        self.planner_llm = planner_llm or llm
        self.clinical_llm = clinical_llm or llm
        self.llm = llm  # legacy compat
        self.device = device
        self.dry_run = dry_run
        self.no_llm = no_llm if (self.planner_llm is not None or llm is not None) else True
        self.postprocess = postprocess
        self.skip_radiomics = skip_radiomics

        # Lazy-loaded components (avoid heavy imports at init)
        self._tool_registry = None
        self._gqc = None
        self._mtqc = None

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, case_path: str, metadata_override: Optional[Dict[str, Any]] = None) -> CaseResult:
        """
        Execute the full pipeline on one case.

        Args:
            case_path: Directory containing image_nifti.nii.gz.
            metadata_override: If provided, skip metadata extraction and use these values.
                Useful for testing when case_path doesn't encode modality/anatomy.

        Returns:
            CaseResult with all outputs, timing, and errors.
        """
        t0 = time.time()
        case_id = Path(case_path).name
        result = CaseResult(case_id=case_id, case_path=case_path)

        try:
            # Validate input
            image_path = os.path.join(case_path, IMAGE_FILENAME)
            if not os.path.isfile(image_path):
                result.status = "failed"
                result.error = f"Image not found: {image_path}"
                return result

            # Step 1: Metadata extraction (or use override)
            if metadata_override:
                result.metadata = metadata_override
                result.is_diagnostic = metadata_override.get("is_diagnostic", True)
                result.step_times["metadata"] = 0.0
                logger.info("[%s] Metadata (override): modality=%s anatomy=%s",
                            result.case_id, metadata_override.get("modality"),
                            metadata_override.get("anatomy"))
            else:
                result.metadata = self._step_metadata(case_path, image_path, result)
            if not result.is_diagnostic:
                result.status = "skipped"
                result.warnings.append("Non-diagnostic scan — skipped")
                return result

            # Step 2: Tool selection
            result.selected_tools = self._step_tool_selection(result.metadata, result)

            # Step 3: Organ list generation
            result.expected_organs = self._step_organ_list(result.metadata, result)

            # Step 4: Segmentation (per tool)
            seg_dirs = self._step_segmentation(
                case_path, image_path, result.metadata, result
            )

            # Step 5: QC (Tier 1 + Tier 2)
            self._step_qc(case_path, seg_dirs, result)

            # Step 6: LLM QC interpretation (Tier 3)
            self._step_qc_interpretation(result)

            # Step 7: Radiomics extraction
            if not self.skip_radiomics:
                self._step_radiomics(case_path, image_path, seg_dirs, result)

            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            logger.exception("[%s] Pipeline failed: %s", case_id, e)

        result.total_time_s = round(time.time() - t0, 2)

        # Save per-case log
        self._save_case_log(result)

        return result

    # ── Step 1: Metadata extraction ─────────────────────────────────────────

    def _step_metadata(
        self, case_path: str, image_path: str, result: CaseResult
    ) -> Dict[str, Any]:
        t0 = time.time()

        if self.no_llm:
            # Default metadata from path heuristics
            meta = self._metadata_from_path(case_path)
        else:
            from planner.metadata_extractor import MetadataExtractor
            extractor = MetadataExtractor(self.planner_llm)
            meta = extractor.extract(image_path)

        result.is_diagnostic = meta.get("is_diagnostic", True)
        result.step_times["metadata"] = round(time.time() - t0, 2)
        logger.info("[%s] Metadata: modality=%s anatomy=%s diagnostic=%s",
                     result.case_id, meta.get("modality"), meta.get("anatomy"),
                     result.is_diagnostic)
        return meta

    def _metadata_from_path(self, case_path: str) -> dict:
        """Heuristic metadata from directory path (no LLM)."""
        path_upper = case_path.upper()
        modality = "CT"
        if "/MR_" in path_upper or "/MRI_" in path_upper:
            modality = "MRI"
        elif "/PT_" in path_upper or "/PET_" in path_upper:
            modality = "PET_CT"

        anatomy = "UNKNOWN"
        for region in ["ABD", "CHEST", "HEAD", "PELVIS", "SPINE", "NECK"]:
            if region in path_upper:
                anatomy = region
                break
        if "ABD" in path_upper and "PELVIS" in path_upper:
            anatomy = "ABDOMEN_PELVIS"
        elif "CHEST" in path_upper and "ABD" in path_upper:
            anatomy = "CHEST_ABDOMEN_PELVIS"

        # Normalize to organ_reference.json keys
        _ANATOMY_NORM = {
            "ABD": "abdomen", "CHEST": "chest", "HEAD": "head",
            "PELVIS": "pelvis", "SPINE": "spine", "NECK": "neck",
            "ABDOMEN_PELVIS": "abdomen_pelvis",
            "CHEST_ABDOMEN_PELVIS": "chest_abdomen_pelvis",
        }
        anatomy = _ANATOMY_NORM.get(anatomy, anatomy)

        return {
            "modality": modality,
            "anatomy": anatomy,
            "is_diagnostic": True,
            "source": "path_heuristic",
        }

    # ── Step 2: Tool selection ──────────────────────────────────────────────

    # Text-promptable tools that need organ lists from the planner
    _TEXT_PROMPTABLE = {"VoxTell", "TextMedSeg3D"}

    def _step_tool_selection(
        self, metadata: dict, result: CaseResult
    ) -> List[str]:
        """
        Select tools for this case.

        Always starts from the registry: ALL tools compatible with the case
        modality are selected (maximum coverage, the core design principle).

        If an LLM planner is available, it can additionally provide targeted
        organ prompts for text-promptable tools (VoxTell, TextMedSeg3D).
        """
        t0 = time.time()

        # Base: all registry-compatible tools (always)
        tools = self._all_compatible_tools(metadata.get("modality", "CT"))

        # LLM refinement: get targeted organ prompts for text-promptable tools
        targeted_organs: Dict[str, List[str]] = {}
        if self.planner_llm is not None and not self.no_llm:
            try:
                from planner.tool_selector import ToolSelector
                selector = ToolSelector(self.planner_llm)
                selection = selector.select(metadata)
                for entry in selection.get("targeted_tools", []):
                    tool_name = entry.get("tool", "") if isinstance(entry, dict) else str(entry)
                    organs = entry.get("organs", []) if isinstance(entry, dict) else []
                    if tool_name:
                        targeted_organs[tool_name] = organs
            except Exception as e:
                logger.warning("[%s] LLM tool selection refinement failed: %s", result.case_id, e)

        result._targeted_tool_organs = targeted_organs
        result.step_times["tool_selection"] = round(time.time() - t0, 2)
        logger.info("[%s] Selected tools (%d): %s", result.case_id, len(tools), tools)
        return tools

    def _all_compatible_tools(self, modality: str) -> List[str]:
        """
        Return ALL tools from the registry that support the given modality.
        Maximum coverage — the original design principle.
        """
        if self._tool_registry is None:
            self._tool_registry = _load_tool_registry()

        tools = []
        for category in ("fixed_class_tools", "text_promptable_tools", "label_prompted_tools"):
            for entry in self._tool_registry.get(category, []):
                name = entry.get("name", "")
                if entry.get("deferred"):
                    continue
                supported = [m.upper() for m in entry.get("supported_modalities", [])]
                if not supported or modality.upper() in supported:
                    tools.append(name)
        return tools

    # ── Step 3: Organ list generation ───────────────────────────────────────

    def _step_organ_list(
        self, metadata: dict, result: CaseResult
    ) -> List[str]:
        t0 = time.time()
        anatomy = metadata.get("anatomy", "UNKNOWN")

        if self.no_llm:
            from planner.organ_list_generator import OrganListGenerator
            gen = OrganListGenerator(llm=None)
            organs = gen.organs_for_qc(anatomy)
        else:
            from planner.organ_list_generator import OrganListGenerator
            gen = OrganListGenerator(llm=self.planner_llm)
            organs = gen.organs_for_qc(anatomy)

        result.step_times["organ_list"] = round(time.time() - t0, 2)
        logger.info("[%s] Expected organs: %d", result.case_id, len(organs))
        return organs

    # ── Step 4: Segmentation ────────────────────────────────────────────────

    def _step_segmentation(
        self,
        case_path: str,
        image_path: str,
        metadata: dict,
        result: CaseResult,
    ) -> Dict[str, str]:
        """Run each selected tool. Returns {tool_name: seg_dir}."""
        if self._tool_registry is None:
            self._tool_registry = _load_tool_registry()

        # Build tool name → class path mapping
        tool_classes = {}
        for category in self._tool_registry.values():
            if isinstance(category, list):
                for entry in category:
                    tool_classes[entry["name"]] = entry.get("class", "")

        seg_dirs: Dict[str, str] = {}

        for tool_name in result.selected_tools:
            t0 = time.time()
            output_dirname = TOOL_OUTPUT_DIRS.get(tool_name, f"segmentations_{tool_name.lower()}")
            seg_dir = os.path.join(case_path, output_dirname)

            # Skip if already segmented (resume-friendly)
            if os.path.isdir(seg_dir) and _has_masks(seg_dir):
                logger.info("[%s] %s: already segmented, skipping", result.case_id, tool_name)
                seg_dirs[tool_name] = seg_dir
                result.tool_outputs.append({
                    "tool_name": tool_name,
                    "seg_dir": seg_dir,
                    "success": True,
                    "skipped": True,
                    "runtime_seconds": 0.0,
                })
                continue

            class_path = tool_classes.get(tool_name, "")
            if not class_path:
                result.warnings.append(f"No class path for tool {tool_name}")
                continue

            # For text-promptable tools, use LLM-generated organ prompts if available,
            # otherwise fall back to expected_organs from the organ list generator
            targeted_organs = getattr(result, "_targeted_tool_organs", {})
            if tool_name in self._TEXT_PROMPTABLE and targeted_organs.get(tool_name):
                tool_organs = targeted_organs[tool_name]
            else:
                tool_organs = result.expected_organs

            inp = ToolInput(
                image_path=image_path,
                case_path=case_path,
                modality=metadata.get("modality", "CT"),
                anatomy=metadata.get("anatomy", "UNKNOWN"),
                target_organs=tool_organs,
                output_dir=seg_dir,
                device=self.device,
            )

            try:
                tool = _load_tool(class_path)
                if self.dry_run and hasattr(tool, "_dry_run"):
                    output = tool._dry_run(inp, seg_dir, t0)
                else:
                    output = tool._timed_run(inp)

                seg_dirs[tool_name] = output.seg_dir
                result.tool_outputs.append({
                    "tool_name": tool_name,
                    "seg_dir": output.seg_dir,
                    "success": output.success,
                    "error": output.error,
                    "organs_segmented": output.organs_segmented,
                    "runtime_seconds": output.runtime_seconds,
                })

                if not output.success:
                    result.warnings.append(f"{tool_name} failed: {output.error}")

            except Exception as e:
                logger.error("[%s] %s failed: %s", result.case_id, tool_name, e)
                result.warnings.append(f"{tool_name} error: {e}")
                result.tool_outputs.append({
                    "tool_name": tool_name,
                    "seg_dir": seg_dir,
                    "success": False,
                    "error": str(e),
                    "runtime_seconds": round(time.time() - t0, 2),
                })

            result.step_times[f"seg_{tool_name}"] = round(time.time() - t0, 2)

        # Postprocess masks
        if self.postprocess and not self.dry_run:
            self._postprocess_masks(seg_dirs, result)

        return seg_dirs

    def _postprocess_masks(
        self, seg_dirs: Dict[str, str], result: CaseResult
    ) -> None:
        """Apply LCC + hole fill to all masks in all seg_dirs."""
        t0 = time.time()
        try:
            from processing.postprocessing import keep_largest_component, fill_holes
            import nibabel as nib
            import numpy as np

            for tool_name, seg_dir in seg_dirs.items():
                if not os.path.isdir(seg_dir):
                    continue
                for fname in os.listdir(seg_dir):
                    if not fname.endswith(".nii.gz"):
                        continue
                    fpath = os.path.join(seg_dir, fname)
                    try:
                        nii = nib.load(fpath)
                        data = np.asarray(nii.dataobj, dtype=np.uint8)
                        if np.sum(data > 0) < 10:
                            continue
                        cleaned = keep_largest_component(data)
                        cleaned = fill_holes(cleaned)
                        if not np.array_equal(data, cleaned):
                            out_nii = nib.Nifti1Image(cleaned, nii.affine, nii.header)
                            nib.save(out_nii, fpath)
                    except Exception:
                        pass  # don't fail the pipeline on postprocessing errors
        except ImportError:
            result.warnings.append("Postprocessing skipped: nibabel/scipy not available")
        result.step_times["postprocess"] = round(time.time() - t0, 2)

    # ── Step 5: QC (Tier 1 + Tier 2) ───────────────────────────────────────

    def _step_qc(
        self,
        case_path: str,
        seg_dirs: Dict[str, str],
        result: CaseResult,
    ) -> None:
        t0 = time.time()
        try:
            from qc.geometric_qc import GeometricQC
            from qc.multi_tool_qc import MultiToolQC
            from qc.dataclasses import CaseQCReport

            if self._gqc is None:
                self._gqc = GeometricQC()
            if self._mtqc is None:
                self._mtqc = MultiToolQC()

            report = CaseQCReport(case_id=result.case_id, case_path=case_path)

            # Tier 1: per-tool geometric QC
            modality = result.metadata.get("modality", "CT")
            for tool_name, seg_dir in seg_dirs.items():
                if os.path.isdir(seg_dir):
                    t1 = self._gqc.run(
                        case_path, seg_dir, tool_name,
                        modality=modality,
                    )
                    report.tool_results.append(t1)

            # Tier 2: multi-tool agreement (if ≥2 tools)
            if len(seg_dirs) >= 2:
                report.agreement = self._mtqc.run(case_path, seg_dirs)

            report.compute_overall()

            # Store serializable summary
            result.qc_report = {
                "overall_severity": report.overall_severity,
                "num_tools": len(report.tool_results),
                "per_tool": [
                    {
                        "tool": tr.tool_name,
                        "severity": tr.worst_severity,
                        "num_organs": tr.num_organs,
                        "num_in_fov": tr.num_organs_in_fov,
                        "num_flagged": tr.num_organs_flagged,
                        "total_flags": tr.total_flags,
                    }
                    for tr in report.tool_results
                ],
            }
            if report.agreement:
                result.qc_report["agreement"] = {
                    "mean_dice": round(report.agreement.mean_dice, 3),
                    "organs_agree": report.agreement.organs_with_agreement,
                    "organs_disagree": report.agreement.organs_with_disagreement,
                }

            # Stash the full report object for Tier 3
            result._qc_report_obj = report

        except Exception as e:
            logger.error("[%s] QC failed: %s", result.case_id, e)
            result.warnings.append(f"QC error: {e}")

        result.step_times["qc"] = round(time.time() - t0, 2)

    # ── Step 6: LLM QC interpretation (Tier 3) ─────────────────────────────

    def _step_qc_interpretation(self, result: CaseResult) -> None:
        report = getattr(result, "_qc_report_obj", None)
        if report is None or not report.tool_results:
            return

        t0 = time.time()

        from qc.qc_interpreter import QCInterpreter

        if self.no_llm or self.clinical_llm is None:
            # Use rule-based fallback (mock LLM returns None → triggers fallback gating)
            class _MockLLM:
                def query_json(self, *a, **kw):
                    return None
            interp_engine = QCInterpreter(_MockLLM())
        else:
            # MedGemma-27B: clinical domain knowledge for QC interpretation
            interp_engine = QCInterpreter(self.clinical_llm)

        try:
            # Interpret the first (primary) tool's results
            primary_tr = report.tool_results[0]
            interp = interp_engine.interpret(
                primary_tr,
                study_description=result.metadata.get("study_description", result.case_id),
                anatomy=result.metadata.get("anatomy", "UNKNOWN"),
            )
            report.interpretation = interp
            report.compute_overall()

            # Update QC report with interpretation
            result.qc_report["interpretation"] = {
                "overall_quality": interp.overall_quality,
                "usable_for_radiomics": interp.usable_for_radiomics,
                "unusable_organs": interp.unusable_organs,
                "extract": interp.extract,
                "skip": interp.skip,
                "explanation": interp.explanation,
            }
            result.qc_report["overall_severity"] = report.overall_severity

        except Exception as e:
            logger.error("[%s] QC interpretation failed: %s", result.case_id, e)
            result.warnings.append(f"QC interpretation error: {e}")

        result.step_times["qc_interpretation"] = round(time.time() - t0, 2)

    # ── Step 7: Radiomics extraction ────────────────────────────────────────

    def _step_radiomics(
        self,
        case_path: str,
        image_path: str,
        seg_dirs: Dict[str, str],
        result: CaseResult,
    ) -> None:
        t0 = time.time()

        # Determine which organs to extract from QC gating
        extract_list = result.qc_report.get("interpretation", {}).get("extract", [])
        if not extract_list:
            # Fallback: extract from primary tool with severity-aware feature gating
            report = getattr(result, "_qc_report_obj", None)
            if report and report.tool_results:
                for oqc in report.tool_results[0].organ_results:
                    if oqc.volume_ml < 1e-3:
                        continue
                    if oqc.severity == "PASS":
                        extract_list.append({
                            "organ": oqc.organ,
                            "features_allowed": ["first_order", "shape", "texture"],
                            "postprocessing_needed": ["none"],
                        })
                    elif oqc.severity == "WARN":
                        features = ["first_order"]
                        postproc = []
                        if oqc.largest_component_fraction >= 0.9:
                            features.extend(["shape", "texture"])
                        elif oqc.largest_component_fraction >= 0.7:
                            features.append("shape")
                        if oqc.num_components > 1:
                            postproc.append("lcc")
                        extract_list.append({
                            "organ": oqc.organ,
                            "features_allowed": features,
                            "postprocessing_needed": postproc or ["none"],
                        })
                    # FAIL → skip (not added to extract_list)

        if not extract_list:
            result.step_times["radiomics"] = round(time.time() - t0, 2)
            return

        # Filter extract list against expected_organs whitelist
        # (prevents extracting radiomics from edge-of-FOV or incidental structures)
        if result.expected_organs:
            allowed = set(result.expected_organs)
            before = len(extract_list)
            extract_list = [e for e in extract_list if e["organ"] in allowed]
            if len(extract_list) < before:
                logger.info(
                    "[%s] Radiomics organ whitelist: %d → %d organs (filtered %d out-of-FOV)",
                    result.case_id, before, len(extract_list), before - len(extract_list),
                )

        if not extract_list:
            logger.info("[%s] No organs in extract_list after filtering — skipping radiomics", result.case_id)
            result.step_times["radiomics"] = round(time.time() - t0, 2)
            return

        logger.info("[%s] Radiomics extract_list: %s", result.case_id,
                     [e["organ"] for e in extract_list])

        # Use first available seg_dir
        seg_dir = None
        for tool_name in result.selected_tools:
            candidate = seg_dirs.get(tool_name)
            if candidate and os.path.isdir(candidate):
                seg_dir = candidate
                break

        if not seg_dir:
            result.warnings.append("No valid seg_dir for radiomics extraction")
            result.step_times["radiomics"] = round(time.time() - t0, 2)
            return

        try:
            # Cache at module level to survive sys.modules["radiomics"] swap
            # (pyradiomics import replaces our local radiomics package in sys.modules)
            extract_fn = _get_extract_pyradiomics()

            # Build map of available masks in seg_dir for fuzzy matching
            available_masks = {}
            for f in os.listdir(seg_dir):
                if f.endswith(".nii.gz"):
                    name = f.replace(".nii.gz", "")
                    available_masks[name] = os.path.join(seg_dir, f)

            for entry in extract_list:
                organ = entry["organ"]
                mask_path = available_masks.get(organ)
                if mask_path is None:
                    # Try common name normalizations
                    normalized = organ.replace(" ", "_").replace("-", "_").lower()
                    mask_path = available_masks.get(normalized)
                if mask_path is None:
                    logger.debug("[%s] No mask found for organ '%s' in %s", result.case_id, organ, seg_dir)
                    continue

                try:
                    rad_result = extract_fn(
                        image_path=image_path,
                        mask_path=mask_path,
                        organ=organ,
                    )
                    result.radiomics.append({
                        "organ": organ,
                        "feature_count": rad_result.feature_count,
                        "features_allowed": entry.get("features_allowed", []),
                        "backend": rad_result.backend,
                    })
                except Exception as e:
                    result.warnings.append(f"Radiomics failed for {organ}: {e}")

        except ImportError:
            result.warnings.append("PyRadiomics not available in this environment")

        result.step_times["radiomics"] = round(time.time() - t0, 2)

    # ── Per-case log ────────────────────────────────────────────────────────

    def _save_case_log(self, result: CaseResult) -> None:
        """Save per-case JSON log to logs directory."""
        try:
            log_dir = os.path.join(LOG_DIR, "cases")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{result.case_id}_log.json")

            log_data = {
                "case_id": result.case_id,
                "case_path": result.case_path,
                "status": result.status,
                "total_time_s": result.total_time_s,
                "metadata": result.metadata,
                "is_diagnostic": result.is_diagnostic,
                "selected_tools": result.selected_tools,
                "expected_organs": result.expected_organs,
                "tool_outputs": result.tool_outputs,
                "qc_report": result.qc_report,
                "radiomics_summary": [
                    {"organ": r["organ"], "feature_count": r["feature_count"]}
                    for r in result.radiomics
                ],
                "step_times": result.step_times,
                "error": result.error,
                "warnings": result.warnings,
            }

            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.warning("[%s] Failed to save case log: %s", result.case_id, e)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _has_masks(seg_dir: str) -> bool:
    """Check if a segmentation directory has at least one .nii.gz mask."""
    for f in os.listdir(seg_dir):
        if f.endswith(".nii.gz") and f not in ("statistics.json",):
            return True
    return False
