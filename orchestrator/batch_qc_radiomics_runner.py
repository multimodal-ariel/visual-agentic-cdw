"""
Batch QC + radiomics runner for already-segmented CDW cases.

This runner is intentionally separate from ``batch_runner.py``. It consumes the
live segmentation state file, looks only at cases whose segmentation status is
``completed``, keeps only anatomy-relevant masks, writes one filtered output by
largest-volume geometric selection and one by MedSegQC score selection, and
extracts fast shape + first-order PyRadiomics features from both outputs.

Default production invocation:

    conda run -n cdw_radiomics python orchestrator/batch_qc_radiomics_runner.py \
        --segmentation-state /data/soumitri/visual-agentic-cdw/logs/pipeline_state_v2.json \
        --gpus 0,1,2,3,4,5,6,7 --workers 8 --watch

The QC model checkpoint paths default to the private-server locations provided
by the deployment notes. Override them from the CLI if needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Iterable, Optional

import nibabel as nib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.constants import (
    IMAGE_FILENAME,
    LOG_DIR,
    SEGMENTATION_ROOT,
    TOOL_OUTPUT_DIRS,
)
from orchestrator.case_id import case_id_to_abspath
from radiomics.pyradiomics import (
    _get_pyradiomics_featureextractor as _get_local_pyradiomics_featureextractor,
)

logger = logging.getLogger(__name__)
_METADATA_PIPELINE = None


MEDSEGQC_ORGANS = ("liver", "kidney_left", "kidney_right", "spleen")
GEOMETRIC_QC_DIRNAME = "segmentations_filtered_GeometricQC"
MEDSEGQC_DIRNAME = "segmentations_filtered_MedSegQC"
LEGACY_BEST_DIRNAME = "segmentations_best"
MRSEG_DIRNAME = TOOL_OUTPUT_DIRS["MRSegmentator"]
SUPPORTED_CT_ANATOMIES = {
    "chest",
    "abdomen",
    "pelvis",
    "abdomen_pelvis",
    "chest_abdomen_pelvis",
    "whole_body",
    "cardiac",
}
SUPPORTED_MRI_ANATOMIES = {"abdomen", "abdomen_pelvis", "cardiac", "spine", "pelvis"}
MRI_TARGET_ORGAN_OVERRIDES = {
    "cardiac": ["heart"],
    "spine": ["spine"],
    "pelvis": ["prostate"],
}
CHEST_ANATOMIES = {"chest", "chest_abdomen_pelvis", "whole_body"}
DEFAULT_SEGMENTATION_STATE = (
    "/data/soumitri/visual-agentic-cdw/logs/pipeline_state_v2.json"
)
DEFAULT_CT_QC_CHECKPOINT = (
    "/data/soumitri/visual-agentic-cdw/checkpoints/MedSegQC/qc_ct_epch33.ckpt"
)
DEFAULT_MRI_QC_CHECKPOINT = (
    "/data/soumitri/visual-agentic-cdw/checkpoints/MedSegQC/qc_mri_epch08.ckpt"
)


@dataclass
class CandidateMask:
    organ: str
    tool_name: str
    seg_dir: str
    mask_path: str
    voxel_count: int
    volume_ml: float
    image_spacing: tuple[float, float, float]
    mask_spacing: tuple[float, float, float]
    shape_match: bool
    spacing_match: bool
    affine_match: bool
    geometry_match: bool
    volume_spacing_source: str
    geometry_warning: str = ""


@dataclass
class QCResult:
    qc_score: Optional[float] = None
    qc_predicted_organ: str = ""
    qc_predicted_organ_idx: Optional[int] = None
    error: str = ""


@dataclass
class CaseWorkItem:
    case_id: str
    case_path: str
    tools_run: list[str] = field(default_factory=list)


@dataclass
class CaseOutcome:
    case_id: str
    case_path: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    target_organs: list[str] = field(default_factory=list)
    selected_organs: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidate_rows: list[dict[str, Any]] = field(default_factory=list)
    geometric_radiomics_rows: list[dict[str, Any]] = field(default_factory=list)
    medsegqc_radiomics_rows: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    total_time_s: float = 0.0


class JsonStateTracker:
    """Thread-safe, atomically-written compact JSON state for this pass."""

    def __init__(self, state_file: str):
        self.state_file = state_file
        self._lock = Lock()
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if os.path.isfile(self.state_file):
            try:
                with open(self.state_file) as f:
                    loaded = json.load(f)
                # Older state files stored very large selected_organs payloads.
                # Keep status/resume information but do not carry detailed
                # per-organ data forward into the compact state.
                cases = {}
                for cid, entry in loaded.get("cases", {}).items():
                    cases[cid] = {
                        "status": entry.get("status", "pending"),
                        "updated_at": entry.get("updated_at", ""),
                        "case_path": entry.get("case_path", ""),
                        "error": entry.get("error", ""),
                        "skip_reason": entry.get("skip_reason", entry.get("error", "")),
                        "warnings": entry.get("warnings", []),
                        "total_time_s": entry.get("total_time_s"),
                        "geometric_done": bool(entry.get("geometric_done", False)),
                        "medsegqc_done": bool(entry.get("medsegqc_done", False)),
                    }
                loaded["cases"] = cases
                loaded.setdefault("schema_version", 2)
                return loaded
            except json.JSONDecodeError:
                logger.warning("State file is not valid JSON yet: %s", self.state_file)
        now = datetime.now().isoformat()
        return {
            "schema_version": 2,
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": now,
            "updated_at": now,
            "cases": {},
        }

    def get_case(self, case_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self.state.get("cases", {}).get(case_id, {}))

    def set_status(
        self,
        case_id: str,
        status: str,
        *,
        case_path: str = "",
        error: str = "",
        skip_reason: str = "",
        warnings: Optional[list[str]] = None,
        total_time_s: Optional[float] = None,
        geometric_done: Optional[bool] = None,
        medsegqc_done: Optional[bool] = None,
    ) -> None:
        with self._lock:
            cases = self.state.setdefault("cases", {})
            entry = cases.setdefault(case_id, {})
            entry["status"] = status
            entry["updated_at"] = datetime.now().isoformat()
            if case_path:
                entry["case_path"] = case_path
            if error:
                entry["error"] = error
            elif status in {"running", "completed", "skipped"}:
                entry["error"] = ""
            if skip_reason:
                entry["skip_reason"] = skip_reason
            if warnings is not None:
                entry["warnings"] = warnings
            if total_time_s is not None:
                entry["total_time_s"] = round(total_time_s, 3)
            if geometric_done is not None:
                entry["geometric_done"] = geometric_done
            if medsegqc_done is not None:
                entry["medsegqc_done"] = medsegqc_done
            self.state["updated_at"] = datetime.now().isoformat()
            self._write_locked()

    def _write_locked(self) -> None:
        directory = os.path.dirname(self.state_file)
        fd, tmp = tempfile.mkstemp(prefix=".tmp_qc_state_", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.state, f, indent=2)
                f.write("\n")
            os.replace(tmp, self.state_file)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


class CSVAppender:
    """Thread-safe CSV writer that can accept varying feature columns."""

    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        self._lock = Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writeheader()

    def append_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self._lock:
            with open(self.path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writerows(rows)


class JSONLAppender:
    """Thread-safe append-only event log."""

    def __init__(self, path: str):
        self.path = path
        self._lock = Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, payload: dict[str, Any]) -> None:
        row = dict(payload)
        row.setdefault("timestamp", datetime.now().isoformat())
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")


class QCModelBundle:
    """One per GPU worker; keeps CT/MRI QC models resident on that device."""

    def __init__(
        self,
        gpu_id: Optional[int],
        ct_checkpoint: str,
        mri_checkpoint: str,
        *,
        dry_run_qc: bool = False,
    ):
        self.gpu_id = gpu_id
        self.ct_checkpoint = ct_checkpoint
        self.mri_checkpoint = mri_checkpoint
        self.dry_run_qc = dry_run_qc
        if gpu_id is None or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{gpu_id}")
        self._models: dict[str, Any] = {}

    def _model_for_modality(self, modality: str):
        from qc.medsegqc_inference import load_model

        key = modality.upper()
        if key not in {"CT", "MRI"}:
            raise ValueError(f"Unsupported QC modality: {modality}")
        if key not in self._models:
            ckpt = self.ct_checkpoint if key == "CT" else self.mri_checkpoint
            logger.info("Loading %s QC model on %s from %s", key, self.device, ckpt)
            self._models[key] = load_model(ckpt, self.device)
        return self._models[key]

    def predict(
        self,
        image_array: np.ndarray,
        mask_path: str,
        modality: str,
        *,
        fallback_volume_ml: float = 0.0,
    ) -> QCResult:
        if self.dry_run_qc:
            # Deterministic fake score for smoke tests. Real runs should not use it.
            score = max(0.0, min(1.0, math.log1p(fallback_volume_ml) / 10.0))
            return QCResult(qc_score=score, qc_predicted_organ="dry_run", qc_predicted_organ_idx=-1)

        try:
            from qc.medsegqc_inference import ORGAN_NAMES, load_volume, preprocess

            mask_array = load_volume(mask_path)
            x = preprocess(image_array, mask_array).to(self.device)
            model = self._model_for_modality(modality)
            with torch.no_grad():
                dice_pred, organ_logits = model(x)
            score = float(dice_pred.detach().cpu())
            organ_idx = int(organ_logits.argmax(dim=-1).detach().cpu())
            organ_name = ORGAN_NAMES[organ_idx]
            return QCResult(
                qc_score=score,
                qc_predicted_organ=organ_name,
                qc_predicted_organ_idx=organ_idx,
            )
        except Exception as exc:
            return QCResult(error=str(exc))


def output_dir_complete(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        required = ("selection_summary.json", "qc_scores.csv")
        return any(os.scandir(path)) and all(os.path.isfile(os.path.join(path, f)) for f in required)
    except OSError:
        return False


def case_outputs_done(case_path: str) -> bool:
    geometric_done = output_dir_complete(os.path.join(case_path, GEOMETRIC_QC_DIRNAME))
    # Require an explicit MedSegQC summary/marker too. Chest-only cases write an
    # empty marker, so absence means the case still needs the new runner/migration.
    medsegqc_done = medsegqc_output_done(case_path)
    return geometric_done and medsegqc_done


def medsegqc_output_done(case_path: str) -> bool:
    medsegqc_dir = os.path.join(case_path, MEDSEGQC_DIRNAME)
    if not os.path.isdir(medsegqc_dir):
        return False
    return output_dir_complete(medsegqc_dir)


def metadata_from_case_path(case_path: str) -> dict[str, Any]:
    """Reuse the main pipeline path heuristic without constructing a pipeline."""
    global _METADATA_PIPELINE
    from orchestrator.pipeline import CasePipeline

    if _METADATA_PIPELINE is None:
        _METADATA_PIPELINE = CasePipeline(no_llm=True)
    return _METADATA_PIPELINE._metadata_from_path(case_path)


def expected_organs_from_metadata(metadata: dict[str, Any]) -> list[str]:
    from planner.organ_list_generator import OrganListGenerator

    generator = OrganListGenerator()
    return generator.organs_for_qc(
        metadata.get("anatomy", "unknown"),
        modality=metadata.get("modality", "UNKNOWN"),
    )


def is_ct_like_modality(modality: str) -> bool:
    return modality.upper() in {"CT", "PET_CT"}


def filter_case_targets(case_path: str) -> tuple[dict[str, Any], list[str], str]:
    metadata = metadata_from_case_path(case_path)
    modality = str(metadata.get("modality", "UNKNOWN")).upper()
    anatomy = str(metadata.get("anatomy", "unknown")).lower()
    if not metadata.get("is_diagnostic", True):
        return metadata, [], metadata.get("skip_reason", "non-diagnostic case")

    if is_ct_like_modality(modality):
        if anatomy not in SUPPORTED_CT_ANATOMIES:
            return metadata, [], f"unsupported CT anatomy: {anatomy}"
    elif modality == "MRI":
        if anatomy not in SUPPORTED_MRI_ANATOMIES:
            return metadata, [], f"unsupported MRI anatomy: {anatomy} (allowed: abdomen, abdomen_pelvis, cardiac, spine, pelvis)"
    else:
        return metadata, [], f"unsupported modality: {modality}"

    if modality == "MRI" and anatomy in MRI_TARGET_ORGAN_OVERRIDES:
        organs = list(MRI_TARGET_ORGAN_OVERRIDES[anatomy])
    else:
        organs = expected_organs_from_metadata(metadata)
    if not organs:
        return metadata, [], f"no expected organs for anatomy: {anatomy}"
    return metadata, organs, ""


def normalize_organ_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def candidate_filenames_for_organ(organ: str) -> list[str]:
    normalized = normalize_organ_name(organ)
    aliases = {
        "kidney_left": ["kidney_left", "left_kidney", "kidney_l"],
        "kidney_right": ["kidney_right", "right_kidney", "kidney_r"],
        "lung_left": ["lung_left", "left_lung"],
        "lung_right": ["lung_right", "right_lung"],
    }
    names = aliases.get(normalized, [normalized])
    return [f"{name}.nii.gz" for name in names] + [f"{name}.nii" for name in names]


def discover_tool_dirs(case_path: str, tools_run: list[str]) -> list[tuple[str, str]]:
    """Return existing segmentation dirs from the completed segmentation tools."""
    pairs: list[tuple[str, str]] = []
    seen_dirs: set[str] = set()
    for tool_name in tools_run:
        dirname = TOOL_OUTPUT_DIRS.get(tool_name)
        if not dirname:
            continue
        seg_dir = os.path.join(case_path, dirname)
        if os.path.isdir(seg_dir) and seg_dir not in seen_dirs:
            pairs.append((tool_name, seg_dir))
            seen_dirs.add(seg_dir)

    # Be forgiving if the upstream state omitted tools_run or a new tool was added.
    if not pairs:
        for tool_name, dirname in TOOL_OUTPUT_DIRS.items():
            seg_dir = os.path.join(case_path, dirname)
            if os.path.isdir(seg_dir) and seg_dir not in seen_dirs:
                pairs.append((tool_name, seg_dir))
                seen_dirs.add(seg_dir)
    return pairs


def mask_volume(mask_path: str, image_path: str) -> tuple[int, float, dict[str, Any]]:
    mask_img = nib.load(mask_path)
    data = np.asanyarray(mask_img.dataobj)
    voxel_count = int(np.count_nonzero(data > 0))

    # Prefer the reference image spacing. Masks are expected to be on the same
    # grid as image_nifti.nii.gz, but tool-written mask headers are not always
    # the safest source of truth. If the grids do not match, fall back to the
    # mask header because the count came from the mask array.
    image_img = nib.load(image_path)
    image_spacing = tuple(float(x) for x in image_img.header.get_zooms()[:3])
    mask_spacing = tuple(float(x) for x in mask_img.header.get_zooms()[:3])
    shape_match = tuple(mask_img.shape[:3]) == tuple(image_img.shape[:3])
    spacing_match = np.allclose(mask_spacing, image_spacing, rtol=1e-4, atol=1e-4)
    affine_match = np.allclose(mask_img.affine, image_img.affine, rtol=1e-4, atol=1e-4)
    geometry_match = bool(shape_match and spacing_match and affine_match)

    if shape_match:
        zooms = image_spacing
        spacing_source = "image"
    else:
        logger.warning(
            "Mask/image shape mismatch for volume calculation: mask=%s image=%s "
            "(using mask spacing)",
            mask_img.shape[:3],
            image_img.shape[:3],
        )
        zooms = mask_spacing
        spacing_source = "mask"

    voxel_volume_ml = float(np.prod(zooms)) / 1000.0 if len(zooms) >= 3 else 0.001
    warning_parts = []
    if not shape_match:
        warning_parts.append(f"shape mask={tuple(mask_img.shape[:3])} image={tuple(image_img.shape[:3])}")
    if not spacing_match:
        warning_parts.append(f"spacing mask={mask_spacing} image={image_spacing}")
    if not affine_match:
        warning_parts.append("affine mismatch")

    return (
        voxel_count,
        voxel_count * voxel_volume_ml,
        {
            "image_spacing": image_spacing,
            "mask_spacing": mask_spacing,
            "shape_match": shape_match,
            "spacing_match": bool(spacing_match),
            "affine_match": bool(affine_match),
            "geometry_match": geometry_match,
            "volume_spacing_source": spacing_source,
            "geometry_warning": "; ".join(warning_parts),
        },
    )


def load_volume_array(path: str) -> np.ndarray:
    """Load .nii/.nii.gz or .npz arrays without importing MedSegQC/MONAI."""
    if path.endswith(".npz"):
        data = np.load(path)
        key = list(data.keys())[0]
        return data[key].astype(np.float32)
    return nib.load(path).get_fdata().astype(np.float32)


def discover_candidates(
    case_path: str,
    tools_run: list[str],
    target_organs: Iterable[str],
) -> dict[str, list[CandidateMask]]:
    image_path = os.path.join(case_path, IMAGE_FILENAME)
    target_organs = list(dict.fromkeys(normalize_organ_name(o) for o in target_organs))
    out: dict[str, list[CandidateMask]] = {organ: [] for organ in target_organs}
    for tool_name, seg_dir in discover_tool_dirs(case_path, tools_run):
        existing = {name.lower(): name for name in os.listdir(seg_dir)}
        for organ in target_organs:
            mask_path = ""
            for fname in candidate_filenames_for_organ(organ):
                real_name = existing.get(fname.lower())
                if real_name:
                    mask_path = os.path.join(seg_dir, real_name)
                    break
            if not mask_path:
                continue
            try:
                voxel_count, volume_ml, geometry = mask_volume(mask_path, image_path)
            except Exception as exc:
                logger.warning("Could not read mask volume (%s): %s", mask_path, exc)
                continue
            if voxel_count <= 0:
                continue
            out[organ].append(
                CandidateMask(
                    organ=organ,
                    tool_name=tool_name,
                    seg_dir=seg_dir,
                    mask_path=mask_path,
                    voxel_count=voxel_count,
                    volume_ml=volume_ml,
                    image_spacing=geometry["image_spacing"],
                    mask_spacing=geometry["mask_spacing"],
                    shape_match=geometry["shape_match"],
                    spacing_match=geometry["spacing_match"],
                    affine_match=geometry["affine_match"],
                    geometry_match=geometry["geometry_match"],
                    volume_spacing_source=geometry["volume_spacing_source"],
                    geometry_warning=geometry["geometry_warning"],
                )
            )
    return out


def copy_selected_masks(
    selected_by_organ: dict[str, CandidateMask],
    output_dir: str,
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    copied: dict[str, str] = {}
    for organ, candidate in selected_by_organ.items():
        dest = os.path.join(output_dir, f"{organ}.nii.gz")
        shutil.copy2(candidate.mask_path, dest)
        copied[organ] = dest
    return copied


def write_rows_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta_cols = [
        "case_id",
        "case_path",
        "organ",
        "source_tool",
        "mask_path",
        "selected_mask_path",
        "selected_by",
        "backend",
    ]
    fieldnames = [c for c in meta_cols if any(c in row for row in rows)]
    rest = sorted({key for row in rows for key in row if key not in fieldnames})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + rest, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_json_with_retries(path: str, *, attempts: int = 5, delay_s: float = 0.25) -> dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay_s)
                continue
            raise
    raise RuntimeError(f"Could not load JSON from {path}: {last_error}")


def get_pyradiomics_extractor():
    """Build a fast shape + first-order extractor; never enable texture classes."""
    RadiomicsFeatureExtractor = _get_local_pyradiomics_featureextractor()
    extractor = RadiomicsFeatureExtractor(
        label=1,
        correctMask=True,
        geometryTolerance=1e-4,
        normalize=False,
        binWidth=25,
    )
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    return extractor


def extract_fast_radiomics(
    image_path: str,
    mask_path: str,
    organ: str,
    extractor: Any,
) -> dict[str, Any]:
    raw = extractor.execute(image_path, mask_path)
    row = {
        "organ": organ,
        "mask_path": mask_path,
        "backend": "pyradiomics_shape_firstorder",
    }
    for key, value in raw.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            row[key] = float(value)
        except (TypeError, ValueError):
            continue
    return row


def extract_manual_radiomics_fallback(
    image_path: str,
    mask_path: str,
    organ: str,
) -> dict[str, Any]:
    """Fallback only for environments where PyRadiomics import fails."""
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    image = np.asanyarray(image_nii.dataobj).astype(np.float32)
    mask = np.asanyarray(mask_nii.dataobj) > 0
    voxels = image[mask]
    zooms = mask_nii.header.get_zooms()[:3]
    voxel_volume_ml = float(np.prod(zooms)) / 1000.0 if len(zooms) >= 3 else 0.001
    row = {
        "organ": organ,
        "mask_path": mask_path,
        "backend": "manual_fallback",
        "manual_voxel_count": int(mask.sum()),
        "manual_volume_ml": float(mask.sum() * voxel_volume_ml),
    }
    if voxels.size:
        row.update(
            {
                "manual_intensity_mean": float(np.mean(voxels)),
                "manual_intensity_median": float(np.median(voxels)),
                "manual_intensity_std": float(np.std(voxels)),
                "manual_intensity_min": float(np.min(voxels)),
                "manual_intensity_max": float(np.max(voxels)),
            }
        )
    return row


def make_test_case(
    root: str,
    case_rel: str,
    masks_by_dir: dict[str, dict[str, int]],
    *,
    image_shape: tuple[int, int, int] = (24, 24, 24),
) -> tuple[str, str]:
    case_id = case_rel.replace("/", "__")
    case_path = os.path.join(root, case_rel)
    os.makedirs(case_path, exist_ok=True)
    image = np.zeros(image_shape, dtype=np.float32)
    image[4 : min(18, image_shape[0]), 4 : min(18, image_shape[1]), 4 : min(18, image_shape[2])] = 100.0
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(image, affine), os.path.join(case_path, IMAGE_FILENAME))
    z, y, x = np.ogrid[: image_shape[0], : image_shape[1], : image_shape[2]]
    center = tuple(s // 2 for s in image_shape)
    for dirname, organ_radii in masks_by_dir.items():
        seg_dir = os.path.join(case_path, dirname)
        os.makedirs(seg_dir, exist_ok=True)
        for organ, radius in organ_radii.items():
            mask = (
                (z - center[0]) ** 2
                + (y - center[1]) ** 2
                + (x - center[2]) ** 2
                <= radius**2
            ).astype(np.uint8)
            nib.save(nib.Nifti1Image(mask, affine), os.path.join(seg_dir, f"{organ}.nii.gz"))
    return case_id, case_path


def load_completed_work(segmentation_state_file: str, segmentation_root: str) -> list[CaseWorkItem]:
    state = load_json_with_retries(segmentation_state_file)
    cases = state.get("cases", {})
    work: list[CaseWorkItem] = []
    for case_id, entry in cases.items():
        if entry.get("status") != "completed":
            continue
        case_path = case_id_to_abspath(case_id, segmentation_root)
        work.append(
            CaseWorkItem(
                case_id=case_id,
                case_path=case_path,
                tools_run=list(entry.get("tools_run") or []),
            )
        )
    return work


class BatchQCRadiomicsRunner:
    def __init__(
        self,
        *,
        segmentation_state: str = DEFAULT_SEGMENTATION_STATE,
        segmentation_root: str = SEGMENTATION_ROOT,
        qc_state_file: str = "",
        candidate_csv: str = "",
        event_log: str = "",
        gpus: Optional[list[int]] = None,
        workers: int = 1,
        ct_checkpoint: str = DEFAULT_CT_QC_CHECKPOINT,
        mri_checkpoint: str = DEFAULT_MRI_QC_CHECKPOINT,
        dry_run_qc: bool = False,
        skip_radiomics: bool = False,
        allow_manual_radiomics_fallback: bool = True,
        force: bool = False,
        poll_seconds: float = 60.0,
        max_cases: Optional[int] = None,
    ):
        self.segmentation_state = segmentation_state
        self.segmentation_root = segmentation_root
        self.gpus: list[Optional[int]] = [0] if gpus is None else list(gpus)
        if not self.gpus:
            self.gpus = [None]
        self.workers = min(max(1, workers), len(self.gpus))
        self.ct_checkpoint = ct_checkpoint
        self.mri_checkpoint = mri_checkpoint
        self.dry_run_qc = dry_run_qc
        self.skip_radiomics = skip_radiomics
        self.allow_manual_radiomics_fallback = allow_manual_radiomics_fallback
        self.force = force
        self.poll_seconds = poll_seconds
        self.max_cases = max_cases

        os.makedirs(LOG_DIR, exist_ok=True)
        self.qc_state_file = qc_state_file or os.path.join(LOG_DIR, "qc_radiomics_state.json")
        self.candidate_csv = candidate_csv or os.path.join(LOG_DIR, "qc_radiomics_candidates.csv")
        self.event_log = event_log or os.path.join(LOG_DIR, "qc_radiomics_events.jsonl")
        self.tracker = JsonStateTracker(self.qc_state_file)
        self.events = JSONLAppender(self.event_log)
        self.csv = CSVAppender(
            self.candidate_csv,
            [
                "case_id",
                "case_path",
                "modality",
                "anatomy",
                "organ",
                "source_tool",
                "mask_path",
                "voxel_count",
                "volume_ml",
                "image_spacing",
                "mask_spacing",
                "shape_match",
                "spacing_match",
                "affine_match",
                "geometry_match",
                "volume_spacing_source",
                "geometry_warning",
                "qc_score",
                "qc_predicted_organ",
                "qc_predicted_organ_idx",
                "qc_error",
                "selected_by_largest_volume",
                "selected_by_highest_qc",
                "geometric_selected_mask_path",
                "medsegqc_selected_mask_path",
                "selection_reason",
            ],
        )

        self._gpu_pool: Queue[Optional[int]] = Queue()
        for gpu_id in self.gpus:
            self._gpu_pool.put(gpu_id)
        self._model_bundles: dict[Optional[int], QCModelBundle] = {}
        self._bundle_lock = Lock()

    def _bundle_for_gpu(self, gpu_id: Optional[int]) -> QCModelBundle:
        with self._bundle_lock:
            if gpu_id not in self._model_bundles:
                self._model_bundles[gpu_id] = QCModelBundle(
                    gpu_id,
                    self.ct_checkpoint,
                    self.mri_checkpoint,
                    dry_run_qc=self.dry_run_qc,
                )
            return self._model_bundles[gpu_id]

    def run(self, *, watch: bool = False) -> dict[str, Any]:
        started = time.time()
        total_completed = 0
        total_failed = 0
        total_skipped = 0
        seen_in_this_process: set[str] = set()

        while True:
            work = self._collect_work(seen_in_this_process)
            if self.max_cases is not None:
                work = work[: max(0, self.max_cases - total_completed - total_failed)]
            if work:
                logger.info("QC/radiomics batch: %d case(s) to process", len(work))
                completed, failed, skipped = self._process_work(work, seen_in_this_process)
                total_completed += completed
                total_failed += failed
                total_skipped += skipped
            elif not watch:
                break

            if self.max_cases is not None and (total_completed + total_failed) >= self.max_cases:
                break
            if not watch:
                break
            logger.info("No eligible new cases. Sleeping %.1fs before next state snapshot.", self.poll_seconds)
            time.sleep(self.poll_seconds)

        elapsed = time.time() - started
        return {
            "completed": total_completed,
            "failed": total_failed,
            "skipped": total_skipped,
            "elapsed_s": round(elapsed, 1),
            "qc_state_file": self.qc_state_file,
            "candidate_csv": self.candidate_csv,
            "event_log": self.event_log,
        }

    def _collect_work(self, seen_in_this_process: set[str]) -> list[CaseWorkItem]:
        if not os.path.isfile(self.segmentation_state):
            raise FileNotFoundError(f"Segmentation state not found: {self.segmentation_state}")
        completed = load_completed_work(self.segmentation_state, self.segmentation_root)
        out: list[CaseWorkItem] = []
        for item in completed:
            if item.case_id in seen_in_this_process:
                continue
            if not os.path.isdir(item.case_path):
                continue
            if not self.force and case_outputs_done(item.case_path):
                self.tracker.set_status(
                    item.case_id,
                    "completed",
                    case_path=item.case_path,
                    geometric_done=True,
                    medsegqc_done=medsegqc_output_done(item.case_path),
                )
                continue
            local = self.tracker.get_case(item.case_id)
            if not self.force and local.get("status") == "skipped":
                continue
            if local.get("status") == "running":
                # The previous process may have died; re-run unless best dir says done.
                pass
            out.append(item)
        return out

    def _process_work(
        self,
        work: list[CaseWorkItem],
        seen_in_this_process: set[str],
    ) -> tuple[int, int, int]:
        completed = failed = skipped = 0
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {}
            for item in work:
                gpu_id = self._gpu_pool.get()
                future = executor.submit(self._run_one_case_with_gpu, item, gpu_id)
                futures[future] = (item, gpu_id)
            for future in as_completed(futures):
                item, _gpu_id = futures[future]
                seen_in_this_process.add(item.case_id)
                try:
                    outcome = future.result()
                except Exception as exc:
                    logger.exception("[%s] Unhandled worker error", item.case_id)
                    self.tracker.set_status(item.case_id, "failed", case_path=item.case_path, error=str(exc))
                    failed += 1
                    continue
                if outcome.status == "completed":
                    completed += 1
                elif outcome.status == "skipped":
                    skipped += 1
                else:
                    failed += 1
        return completed, failed, skipped

    def _run_one_case_with_gpu(self, item: CaseWorkItem, gpu_id: Optional[int]) -> CaseOutcome:
        try:
            bundle = self._bundle_for_gpu(gpu_id)
            return self._run_one_case(item, bundle)
        finally:
            self._gpu_pool.put(gpu_id)

    def _run_one_case(self, item: CaseWorkItem, bundle: QCModelBundle) -> CaseOutcome:
        t0 = time.time()
        self.tracker.set_status(item.case_id, "running", case_path=item.case_path)
        geometric_dir = os.path.join(item.case_path, GEOMETRIC_QC_DIRNAME)
        medsegqc_dir = os.path.join(item.case_path, MEDSEGQC_DIRNAME)
        image_path = os.path.join(item.case_path, IMAGE_FILENAME)
        outcome = CaseOutcome(case_id=item.case_id, case_path=item.case_path, status="failed")

        try:
            if not os.path.isfile(image_path):
                outcome.status = "failed"
                outcome.error = f"Image not found: {image_path}"
                return self._finalize_outcome(outcome, t0)

            metadata, target_organs, skip_reason = filter_case_targets(item.case_path)
            outcome.metadata = metadata
            outcome.target_organs = target_organs
            modality = str(metadata.get("modality", "UNKNOWN")).upper()
            anatomy = str(metadata.get("anatomy", "unknown")).lower()
            if skip_reason:
                outcome.status = "skipped"
                outcome.error = skip_reason
                return self._finalize_outcome(outcome, t0)

            candidates = discover_candidates(item.case_path, item.tools_run, target_organs)
            if not any(candidates.values()):
                outcome.status = "skipped"
                outcome.error = "No anatomy-relevant non-empty masks found"
                return self._finalize_outcome(outcome, t0)

            geometric_by_organ: dict[str, CandidateMask] = {}
            for organ, organ_candidates in candidates.items():
                if organ_candidates:
                    geometric_by_organ[organ] = max(organ_candidates, key=lambda c: c.volume_ml)

            self._add_mrseg_lung_shortcut(
                item.case_path,
                candidates,
                geometric_by_organ,
                modality=modality,
                anatomy=anatomy,
            )

            if not geometric_by_organ:
                outcome.status = "skipped"
                outcome.error = "No non-empty target-organ masks found"
                return self._finalize_outcome(outcome, t0)

            image_array = load_volume_array(image_path)
            geometric_paths = copy_selected_masks(geometric_by_organ, geometric_dir)
            qc_by_key: dict[tuple[str, str], QCResult] = {}
            medsegqc_by_organ: dict[str, CandidateMask] = {}

            for organ, organ_candidates in candidates.items():
                best_volume_candidate = geometric_by_organ.get(organ)
                best_qc_candidate: Optional[CandidateMask] = None
                best_qc_score = -math.inf

                rows_for_organ: list[dict[str, Any]] = []
                for candidate in organ_candidates:
                    if organ in MEDSEGQC_ORGANS:
                        qc = bundle.predict(
                            image_array,
                            candidate.mask_path,
                            modality,
                            fallback_volume_ml=candidate.volume_ml,
                        )
                    else:
                        qc = QCResult(error="MedSegQC not run: organ outside supported set")
                    qc_by_key[(organ, candidate.tool_name)] = qc
                    if qc.qc_score is not None and qc.qc_score > best_qc_score:
                        best_qc_score = qc.qc_score
                        best_qc_candidate = candidate
                    row = {
                        "case_id": item.case_id,
                        "case_path": item.case_path,
                        "modality": modality,
                        "anatomy": anatomy,
                        "organ": organ,
                        "source_tool": candidate.tool_name,
                        "mask_path": candidate.mask_path,
                        "voxel_count": candidate.voxel_count,
                        "volume_ml": round(candidate.volume_ml, 6),
                        "image_spacing": "x".join(f"{x:.6g}" for x in candidate.image_spacing),
                        "mask_spacing": "x".join(f"{x:.6g}" for x in candidate.mask_spacing),
                        "shape_match": candidate.shape_match,
                        "spacing_match": candidate.spacing_match,
                        "affine_match": candidate.affine_match,
                        "geometry_match": candidate.geometry_match,
                        "volume_spacing_source": candidate.volume_spacing_source,
                        "geometry_warning": candidate.geometry_warning,
                        "qc_score": "" if qc.qc_score is None else round(qc.qc_score, 6),
                        "qc_predicted_organ": qc.qc_predicted_organ,
                        "qc_predicted_organ_idx": qc.qc_predicted_organ_idx,
                        "qc_error": qc.error,
                        "selected_by_largest_volume": candidate is best_volume_candidate,
                        "geometric_selected_mask_path": geometric_paths.get(organ, ""),
                        "medsegqc_selected_mask_path": "",
                        "selection_reason": "largest_volume",
                    }
                    rows_for_organ.append(row)

                for row in rows_for_organ:
                    row["selected_by_highest_qc"] = (
                        best_qc_candidate is not None
                        and row["source_tool"] == best_qc_candidate.tool_name
                        and row["mask_path"] == best_qc_candidate.mask_path
                    )
                    if row["selected_by_highest_qc"]:
                        row["selection_reason"] = "largest_volume_and_highest_qc" if row["selected_by_largest_volume"] else "highest_medsegqc_score"
                if organ in MEDSEGQC_ORGANS and best_qc_candidate is not None:
                    medsegqc_by_organ[organ] = best_qc_candidate
                outcome.candidate_rows.extend(rows_for_organ)

            medsegqc_paths = copy_selected_masks(medsegqc_by_organ, medsegqc_dir) if medsegqc_by_organ else {}
            if medsegqc_paths:
                for row in outcome.candidate_rows:
                    organ = row.get("organ", "")
                    if organ in medsegqc_paths:
                        row["medsegqc_selected_mask_path"] = medsegqc_paths[organ]
            for row in outcome.candidate_rows:
                if row.get("organ") in {"lung_left", "lung_right"} and row.get("source_tool") == "MRSegmentator" and row.get("selected_by_largest_volume"):
                    row["selection_reason"] = "mrsegmentator_full_lung_default"

            outcome.geometric_radiomics_rows = self._extract_case_radiomics(
                image_path,
                geometric_paths,
                item.case_id,
                item.case_path,
                selected_by="largest_volume",
            )
            outcome.medsegqc_radiomics_rows = self._extract_case_radiomics(
                image_path,
                medsegqc_paths,
                item.case_id,
                item.case_path,
                selected_by="highest_medsegqc_score",
            )

            selected_organs = {}
            for organ, best in geometric_by_organ.items():
                qc = qc_by_key.get((organ, best.tool_name), QCResult())
                selected_organs[organ] = {
                    "selected_by": "largest_volume",
                    "source_tool": best.tool_name,
                    "source_mask_path": best.mask_path,
                    "selected_mask_path": geometric_paths.get(organ, ""),
                    "volume_ml": round(best.volume_ml, 6),
                    "voxel_count": best.voxel_count,
                    "image_spacing": best.image_spacing,
                    "mask_spacing": best.mask_spacing,
                    "shape_match": best.shape_match,
                    "spacing_match": best.spacing_match,
                    "affine_match": best.affine_match,
                    "geometry_match": best.geometry_match,
                    "volume_spacing_source": best.volume_spacing_source,
                    "geometry_warning": best.geometry_warning,
                    "qc_score": qc.qc_score,
                    "qc_predicted_organ": qc.qc_predicted_organ,
                    "qc_error": qc.error,
                }
                if organ in {"lung_left", "lung_right"} and best.tool_name == "MRSegmentator":
                    selected_organs[organ]["selection_reason"] = "mrsegmentator_full_lung_default"
                if not best.geometry_match:
                    outcome.warnings.append(
                        f"{organ}: selected {best.tool_name} mask geometry mismatch "
                        f"({best.geometry_warning or 'see spacing/affine flags'})"
                    )
            outcome.selected_organs = selected_organs

            self._write_selection_outputs(
                output_dir=geometric_dir,
                item=item,
                metadata=metadata,
                target_organs=target_organs,
                candidate_rows=outcome.candidate_rows,
                selected=selected_organs,
                radiomics_rows=outcome.geometric_radiomics_rows,
                strategy="GeometricQC",
                notes=[
                    "Masks are selected by largest volume among anatomy-relevant non-empty candidates.",
                    "MRSegmentator full-lung masks are preferred for lung_left/lung_right in relevant CT chest anatomy.",
                    "Radiomics features are shape + first-order only; texture classes are disabled.",
                ],
            )
            medsegqc_candidate_rows = [
                r for r in outcome.candidate_rows if r.get("organ") in MEDSEGQC_ORGANS
            ]
            if medsegqc_by_organ:
                medsegqc_selected = self._selected_summary(
                    medsegqc_by_organ,
                    medsegqc_paths,
                    qc_by_key,
                    selected_by="highest_medsegqc_score",
                )
                self._write_selection_outputs(
                    output_dir=medsegqc_dir,
                    item=item,
                    metadata=metadata,
                    target_organs=[o for o in target_organs if o in MEDSEGQC_ORGANS],
                    candidate_rows=medsegqc_candidate_rows,
                    selected=medsegqc_selected,
                    radiomics_rows=outcome.medsegqc_radiomics_rows,
                    strategy="MedSegQC",
                    notes=[
                        "Masks are selected by highest MedSegQC predicted Dice score.",
                        "MedSegQC is only applied to liver, spleen, kidney_left, and kidney_right.",
                        "Radiomics features are shape + first-order only; texture classes are disabled.",
                    ],
                )
            else:
                self._write_selection_outputs(
                    output_dir=medsegqc_dir,
                    item=item,
                    metadata=metadata,
                    target_organs=[o for o in target_organs if o in MEDSEGQC_ORGANS],
                    candidate_rows=medsegqc_candidate_rows,
                    selected={},
                    radiomics_rows=[],
                    strategy="MedSegQC",
                    notes=[
                        "No valid MedSegQC winner was available for this case.",
                        "MedSegQC is only applied to liver, spleen, kidney_left, and kidney_right.",
                        "If candidate rows exist, inspect qc_error for model/checkpoint/preprocessing failures.",
                    ],
                )

            outcome.status = "completed"
            return self._finalize_outcome(outcome, t0)

        except Exception as exc:
            logger.exception("[%s] Failed", item.case_id)
            outcome.status = "failed"
            outcome.error = str(exc)
            return self._finalize_outcome(outcome, t0)

    def _add_mrseg_lung_shortcut(
        self,
        case_path: str,
        candidates: dict[str, list[CandidateMask]],
        geometric_by_organ: dict[str, CandidateMask],
        *,
        modality: str,
        anatomy: str,
    ) -> None:
        if not is_ct_like_modality(modality) or anatomy not in CHEST_ANATOMIES:
            return
        for organ in ("lung_left", "lung_right"):
            organ_candidates = candidates.get(organ, [])
            mrseg_candidate = next(
                (c for c in organ_candidates if c.tool_name == "MRSegmentator"),
                None,
            )
            if mrseg_candidate is not None:
                geometric_by_organ[organ] = mrseg_candidate

    def _selected_summary(
        self,
        selected_by_organ: dict[str, CandidateMask],
        selected_paths: dict[str, str],
        qc_by_key: dict[tuple[str, str], QCResult],
        *,
        selected_by: str,
    ) -> dict[str, dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for organ, candidate in selected_by_organ.items():
            qc = qc_by_key.get((organ, candidate.tool_name), QCResult())
            selected[organ] = {
                "selected_by": selected_by,
                "source_tool": candidate.tool_name,
                "source_mask_path": candidate.mask_path,
                "selected_mask_path": selected_paths.get(organ, ""),
                "volume_ml": round(candidate.volume_ml, 6),
                "voxel_count": candidate.voxel_count,
                "image_spacing": candidate.image_spacing,
                "mask_spacing": candidate.mask_spacing,
                "shape_match": candidate.shape_match,
                "spacing_match": candidate.spacing_match,
                "affine_match": candidate.affine_match,
                "geometry_match": candidate.geometry_match,
                "volume_spacing_source": candidate.volume_spacing_source,
                "geometry_warning": candidate.geometry_warning,
                "qc_score": qc.qc_score,
                "qc_predicted_organ": qc.qc_predicted_organ,
                "qc_error": qc.error,
            }
            if selected_by == "largest_volume" and organ in {"lung_left", "lung_right"} and candidate.tool_name == "MRSegmentator":
                selected[organ]["selection_reason"] = "mrsegmentator_full_lung_default"
            else:
                selected[organ]["selection_reason"] = selected_by
        return selected

    def _write_selection_outputs(
        self,
        *,
        output_dir: str,
        item: CaseWorkItem,
        metadata: dict[str, Any],
        target_organs: list[str],
        candidate_rows: list[dict[str, Any]],
        selected: dict[str, dict[str, Any]],
        radiomics_rows: list[dict[str, Any]],
        strategy: str,
        notes: list[str],
    ) -> None:
        write_rows_csv(os.path.join(output_dir, "qc_scores.csv"), candidate_rows)
        if not candidate_rows:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "qc_scores.csv"), "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["case_id", "case_path", "strategy", "note"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": item.case_id,
                        "case_path": item.case_path,
                        "strategy": strategy,
                        "note": "No eligible candidate masks for this strategy.",
                    }
                )
        if radiomics_rows:
            write_rows_csv(os.path.join(output_dir, "radiomics_features.csv"), radiomics_rows)
        write_json(
            os.path.join(output_dir, "selection_summary.json"),
            {
                "case_id": item.case_id,
                "case_path": item.case_path,
                "metadata": metadata,
                "target_organs": target_organs,
                "strategy": strategy,
                "selected_organs": selected,
                "notes": notes,
            },
        )

    def _extract_case_radiomics(
        self,
        image_path: str,
        selected_paths: dict[str, str],
        case_id: str,
        case_path: str,
        *,
        selected_by: str,
    ) -> list[dict[str, Any]]:
        if self.skip_radiomics:
            return []

        extractor = None
        try:
            extractor = get_pyradiomics_extractor()
        except Exception as exc:
            if not self.allow_manual_radiomics_fallback:
                raise
            logger.warning("[%s] PyRadiomics unavailable; using manual fallback: %s", case_id, exc)

        rows: list[dict[str, Any]] = []
        for organ, mask_path in selected_paths.items():
            try:
                if extractor is None:
                    row = extract_manual_radiomics_fallback(image_path, mask_path, organ)
                else:
                    row = extract_fast_radiomics(image_path, mask_path, organ, extractor)
                row.update(
                    {
                        "case_id": case_id,
                        "case_path": case_path,
                        "selected_mask_path": mask_path,
                        "source_tool": "",
                        "selected_by": selected_by,
                    }
                )
                rows.append(row)
            except Exception as exc:
                logger.warning("[%s] Radiomics failed for %s: %s", case_id, organ, exc)
        return rows

    def _finalize_outcome(self, outcome: CaseOutcome, t0: float) -> CaseOutcome:
        outcome.total_time_s = time.time() - t0
        self.csv.append_rows(outcome.candidate_rows)
        geometric_done = output_dir_complete(os.path.join(outcome.case_path, GEOMETRIC_QC_DIRNAME))
        medsegqc_done = medsegqc_output_done(outcome.case_path)
        self.tracker.set_status(
            outcome.case_id,
            outcome.status,
            case_path=outcome.case_path,
            error=outcome.error,
            skip_reason=outcome.error if outcome.status == "skipped" else "",
            warnings=outcome.warnings,
            total_time_s=outcome.total_time_s,
            geometric_done=geometric_done,
            medsegqc_done=medsegqc_done,
        )
        self.events.append(
            {
                "case_id": outcome.case_id,
                "case_path": outcome.case_path,
                "status": outcome.status,
                "modality": outcome.metadata.get("modality", ""),
                "anatomy": outcome.metadata.get("anatomy", ""),
                "target_organs_count": len(outcome.target_organs),
                "geometric_selected_count": len(outcome.selected_organs),
                "candidate_rows": len(outcome.candidate_rows),
                "geometric_done": geometric_done,
                "medsegqc_done": medsegqc_done,
                "error": outcome.error,
                "warnings": outcome.warnings,
                "total_time_s": round(outcome.total_time_s, 3),
            }
        )
        logger.info(
            "[%s] %s in %.1fs (%d selected organs, %d QC candidate rows)",
            outcome.case_id,
            outcome.status,
            outcome.total_time_s,
            len(outcome.selected_organs),
            len(outcome.candidate_rows),
        )
        return outcome

def parse_gpus(raw: str) -> list[int]:
    if raw.strip().lower() in {"", "none", "cpu"}:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_self_test() -> None:
    """Dry-run integration tests for anatomy filtering and output contracts."""
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "segmentations_3d")
        abd_id, abd_path = make_test_case(
            root,
            "LUPUS/PT001/20240101/CT_ABDOMEN_PELVIS_W_CONTRAST/AXIAL",
            {
                "segmentations_totalseg_ct": {"liver": 3, "kidney_left": 3},
                "segmentations_vista3d": {"liver": 5, "kidney_left": 4},
            },
        )
        chest_id, chest_path = make_test_case(
            root,
            "LUPUS/PT002/20240102/CT_CHEST_W_CONTRAST/AXIAL",
            {
                "segmentations_mrseg": {"lung_left": 4, "lung_right": 4},
                "segmentations_vista3d": {"heart": 3},
            },
        )
        brain_id, brain_path = make_test_case(
            root,
            "LUPUS/PT003/20240103/MRI_BRAIN_W_WO_CONTRAST/T1_AX",
            {
                "segmentations_vista3d": {"kidney_left": 5, "liver": 5},
            },
        )
        cardiac_id, cardiac_path = make_test_case(
            root,
            "LUPUS/PT004/20240104/MRI_CARDIAC_MORPH_WO_CONTRAST/TRUE_FISP_AXIAL",
            {
                "segmentations_mrseg": {"heart": 4, "liver": 5},
            },
        )
        spine_id, spine_path = make_test_case(
            root,
            "LUPUS/PT005/20240105/MRI_CERVICAL_SPINE_WO_CONTRAST/t2_tse_sag",
            {
                "segmentations_mrseg": {"spine": 4, "kidney_left": 5},
            },
        )
        pelvis_id, pelvis_path = make_test_case(
            root,
            "LUPUS/PT006/20240106/MRI_PELVIS_W_WO_CONTRAST_MSK/T1_AXIAL_PELVIS",
            {
                "segmentations_vista3d": {"prostate": 4, "kidney_left": 5},
            },
        )
        seg_state = os.path.join(tmp, "pipeline_state_v2.json")
        with open(seg_state, "w") as f:
            json.dump(
                {
                    "run_id": "self_test",
                    "cases": {
                        abd_id: {
                            "status": "completed",
                            "tools_run": ["TotalSegmentator_CT", "VISTA3D"],
                            "error": "",
                        },
                        chest_id: {
                            "status": "completed",
                            "tools_run": ["MRSegmentator", "VISTA3D"],
                            "error": "",
                        },
                        brain_id: {
                            "status": "completed",
                            "tools_run": ["VISTA3D"],
                            "error": "",
                        },
                        cardiac_id: {
                            "status": "completed",
                            "tools_run": ["MRSegmentator"],
                            "error": "",
                        },
                        spine_id: {
                            "status": "completed",
                            "tools_run": ["MRSegmentator"],
                            "error": "",
                        },
                        pelvis_id: {
                            "status": "completed",
                            "tools_run": ["VISTA3D"],
                            "error": "",
                        },
                    },
                },
                f,
            )

        runner = BatchQCRadiomicsRunner(
            segmentation_state=seg_state,
            segmentation_root=root,
            qc_state_file=os.path.join(tmp, "qc_state.json"),
            candidate_csv=os.path.join(tmp, "candidates.csv"),
            event_log=os.path.join(tmp, "events.jsonl"),
            gpus=[],
            workers=1,
            dry_run_qc=True,
            skip_radiomics=True,
        )
        summary = runner.run()
        geometric_liver = os.path.join(abd_path, GEOMETRIC_QC_DIRNAME, "liver.nii.gz")
        medsegqc_liver = os.path.join(abd_path, MEDSEGQC_DIRNAME, "liver.nii.gz")
        if summary["completed"] != 5 or summary["skipped"] != 1:
            raise AssertionError(f"Self-test failed: {summary}")
        if not os.path.isfile(geometric_liver) or not os.path.isfile(medsegqc_liver):
            raise AssertionError("Expected abdomen CT outputs in both filtered folders")
        selected = json.load(open(os.path.join(abd_path, GEOMETRIC_QC_DIRNAME, "selection_summary.json")))
        tool = selected["selected_organs"]["liver"]["source_tool"]
        if tool != "VISTA3D":
            raise AssertionError(f"Expected largest VISTA3D mask, got {tool}")
        lung_left = os.path.join(chest_path, GEOMETRIC_QC_DIRNAME, "lung_left.nii.gz")
        if not os.path.isfile(lung_left):
            raise AssertionError("Expected MRSegmentator lung_left in GeometricQC output")
        chest_selected = json.load(open(os.path.join(chest_path, GEOMETRIC_QC_DIRNAME, "selection_summary.json")))
        if chest_selected["selected_organs"]["lung_left"].get("selection_reason") != "mrsegmentator_full_lung_default":
            raise AssertionError("Expected MRSegmentator lung shortcut reason")
        if os.path.isdir(os.path.join(brain_path, GEOMETRIC_QC_DIRNAME)):
            raise AssertionError("Brain MRI should not produce filtered outputs")
        if not os.path.isfile(os.path.join(cardiac_path, GEOMETRIC_QC_DIRNAME, "heart.nii.gz")):
            raise AssertionError("Cardiac MRI should keep heart")
        if os.path.isfile(os.path.join(cardiac_path, GEOMETRIC_QC_DIRNAME, "liver.nii.gz")):
            raise AssertionError("Cardiac MRI should not keep irrelevant liver")
        if not os.path.isfile(os.path.join(spine_path, GEOMETRIC_QC_DIRNAME, "spine.nii.gz")):
            raise AssertionError("Spine MRI should keep spine")
        if not os.path.isfile(os.path.join(pelvis_path, GEOMETRIC_QC_DIRNAME, "prostate.nii.gz")):
            raise AssertionError("Pelvis MRI should keep prostate")
        summary2 = runner.run()
        if summary2["completed"] != 0:
            raise AssertionError(f"Second self-test run should be idempotent: {summary2}")
        print("Self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MedSegQC + fast radiomics over completed segmentation cases."
    )
    parser.add_argument("--segmentation-state", default=DEFAULT_SEGMENTATION_STATE)
    parser.add_argument("--segmentation-root", default=SEGMENTATION_ROOT)
    parser.add_argument("--qc-state-file", default="")
    parser.add_argument("--candidate-csv", default="")
    parser.add_argument("--event-log", default="")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, or 'cpu'")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ct-checkpoint", default=DEFAULT_CT_QC_CHECKPOINT)
    parser.add_argument("--mri-checkpoint", default=DEFAULT_MRI_QC_CHECKPOINT)
    parser.add_argument("--watch", action="store_true", help="Keep polling segmentation state for new completed cases")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-run even if filtered QC outputs already exist")
    parser.add_argument("--dry-run-qc", action="store_true", help="Do not load QC checkpoints; use deterministic fake QC")
    parser.add_argument("--skip-radiomics", action="store_true")
    parser.add_argument("--no-manual-radiomics-fallback", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.self_test:
        run_self_test()
        return 0

    gpus = parse_gpus(args.gpus)
    runner = BatchQCRadiomicsRunner(
        segmentation_state=args.segmentation_state,
        segmentation_root=args.segmentation_root,
        qc_state_file=args.qc_state_file,
        candidate_csv=args.candidate_csv,
        event_log=args.event_log,
        gpus=gpus,
        workers=args.workers,
        ct_checkpoint=args.ct_checkpoint,
        mri_checkpoint=args.mri_checkpoint,
        dry_run_qc=args.dry_run_qc,
        skip_radiomics=args.skip_radiomics,
        allow_manual_radiomics_fallback=not args.no_manual_radiomics_fallback,
        force=args.force,
        poll_seconds=args.poll_seconds,
        max_cases=args.max_cases,
    )
    summary = runner.run(watch=args.watch)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
