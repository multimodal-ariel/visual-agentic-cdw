"""
Batch QC + radiomics runner for already-segmented CDW cases.

This runner is intentionally separate from ``batch_runner.py``. It consumes the
live segmentation state file, looks only at cases whose segmentation status is
``completed``, selects a best mask per target organ by largest volume, records
MedSegQC scores for every candidate mask, and extracts fast shape + first-order
PyRadiomics features from the selected masks.

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


TARGET_ORGANS = ("liver", "kidney_left", "kidney_right", "spleen")
BEST_DIRNAME = "segmentations_best"
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
    selected_organs: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidate_rows: list[dict[str, Any]] = field(default_factory=list)
    radiomics_rows: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    total_time_s: float = 0.0


class JsonStateTracker:
    """Thread-safe, atomically-written JSON state for this QC/radiomics pass."""

    def __init__(self, state_file: str):
        self.state_file = state_file
        self._lock = Lock()
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if os.path.isfile(self.state_file):
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("State file is not valid JSON yet: %s", self.state_file)
        now = datetime.now().isoformat()
        return {
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
        selected_organs: Optional[dict[str, dict[str, Any]]] = None,
        error: str = "",
        warnings: Optional[list[str]] = None,
        total_time_s: Optional[float] = None,
    ) -> None:
        with self._lock:
            cases = self.state.setdefault("cases", {})
            entry = cases.setdefault(case_id, {})
            entry["status"] = status
            entry["updated_at"] = datetime.now().isoformat()
            if case_path:
                entry["case_path"] = case_path
            if selected_organs is not None:
                entry["selected_organs"] = selected_organs
            if error:
                entry["error"] = error
            elif status in {"running", "completed", "skipped"}:
                entry["error"] = ""
            if warnings is not None:
                entry["warnings"] = warnings
            if total_time_s is not None:
                entry["total_time_s"] = round(total_time_s, 3)
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


def best_dir_done(case_path: str) -> bool:
    best_dir = os.path.join(case_path, BEST_DIRNAME)
    if not os.path.isdir(best_dir):
        return False
    try:
        return any(os.scandir(best_dir))
    except OSError:
        return False


def modality_from_case_path(case_path: str) -> Optional[str]:
    """Infer CT vs MRI from path segments, preferring the segment after YYYYMMDD."""
    parts = [p for p in Path(case_path).parts if p]
    study_desc = ""
    for idx, part in enumerate(parts):
        if len(part) == 8 and part.isdigit() and idx + 1 < len(parts):
            study_desc = parts[idx + 1].upper()
            break
    haystack = study_desc or " ".join(parts).upper()
    if haystack.startswith(("MRI", "MR_", "MRA")) or "MRI_" in haystack:
        return "MRI"
    if haystack.startswith(("CT", "CTA", "PET_CT")) or "_CT_" in haystack:
        return "CT"
    return None


def normalize_organ_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def candidate_filenames_for_organ(organ: str) -> list[str]:
    normalized = normalize_organ_name(organ)
    aliases = {
        "kidney_left": ["kidney_left", "left_kidney", "kidney_l"],
        "kidney_right": ["kidney_right", "right_kidney", "kidney_r"],
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


def mask_volume(mask_path: str) -> tuple[int, float]:
    img = nib.load(mask_path)
    data = np.asanyarray(img.dataobj)
    voxel_count = int(np.count_nonzero(data > 0))
    zooms = img.header.get_zooms()[:3]
    voxel_volume_ml = float(np.prod(zooms)) / 1000.0 if len(zooms) >= 3 else 0.001
    return voxel_count, voxel_count * voxel_volume_ml


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
    target_organs: tuple[str, ...] = TARGET_ORGANS,
) -> dict[str, list[CandidateMask]]:
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
                voxel_count, volume_ml = mask_volume(mask_path)
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
                )
            )
    return out


def copy_best_masks(
    best_by_organ: dict[str, CandidateMask],
    best_dir: str,
) -> dict[str, str]:
    os.makedirs(best_dir, exist_ok=True)
    copied: dict[str, str] = {}
    for organ, candidate in best_by_organ.items():
        dest = os.path.join(best_dir, f"{organ}.nii.gz")
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
        self.tracker = JsonStateTracker(self.qc_state_file)
        self.csv = CSVAppender(
            self.candidate_csv,
            [
                "case_id",
                "case_path",
                "modality",
                "organ",
                "source_tool",
                "mask_path",
                "voxel_count",
                "volume_ml",
                "qc_score",
                "qc_predicted_organ",
                "qc_predicted_organ_idx",
                "qc_error",
                "selected_by_largest_volume",
                "selected_by_highest_qc",
                "selected_mask_path",
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
                self.write_correlation_outputs(max_cases=100)
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
            if not self.force and best_dir_done(item.case_path):
                self.tracker.set_status(item.case_id, "completed", case_path=item.case_path)
                continue
            local = self.tracker.get_case(item.case_id)
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
        best_dir = os.path.join(item.case_path, BEST_DIRNAME)
        image_path = os.path.join(item.case_path, IMAGE_FILENAME)
        outcome = CaseOutcome(case_id=item.case_id, case_path=item.case_path, status="failed")

        try:
            if not os.path.isfile(image_path):
                outcome.status = "failed"
                outcome.error = f"Image not found: {image_path}"
                return self._finalize_outcome(outcome, t0)

            modality = modality_from_case_path(item.case_path)
            if modality not in {"CT", "MRI"}:
                outcome.status = "skipped"
                outcome.error = "Unsupported or unknown modality for MedSegQC"
                return self._finalize_outcome(outcome, t0)

            candidates = discover_candidates(item.case_path, item.tools_run)
            if not any(candidates.values()):
                outcome.status = "skipped"
                outcome.error = "No target-organ masks found"
                return self._finalize_outcome(outcome, t0)

            best_by_organ: dict[str, CandidateMask] = {}
            for organ, organ_candidates in candidates.items():
                if organ_candidates:
                    best_by_organ[organ] = max(organ_candidates, key=lambda c: c.volume_ml)

            if not best_by_organ:
                outcome.status = "skipped"
                outcome.error = "No non-empty target-organ masks found"
                return self._finalize_outcome(outcome, t0)

            image_array = load_volume_array(image_path)
            selected_paths = copy_best_masks(best_by_organ, best_dir)
            qc_by_key: dict[tuple[str, str], QCResult] = {}

            for organ, organ_candidates in candidates.items():
                best_volume_candidate = best_by_organ.get(organ)
                best_qc_candidate: Optional[CandidateMask] = None
                best_qc_score = -math.inf

                rows_for_organ: list[dict[str, Any]] = []
                for candidate in organ_candidates:
                    qc = bundle.predict(
                        image_array,
                        candidate.mask_path,
                        modality,
                        fallback_volume_ml=candidate.volume_ml,
                    )
                    qc_by_key[(organ, candidate.tool_name)] = qc
                    if qc.qc_score is not None and qc.qc_score > best_qc_score:
                        best_qc_score = qc.qc_score
                        best_qc_candidate = candidate
                    row = {
                        "case_id": item.case_id,
                        "case_path": item.case_path,
                        "modality": modality,
                        "organ": organ,
                        "source_tool": candidate.tool_name,
                        "mask_path": candidate.mask_path,
                        "voxel_count": candidate.voxel_count,
                        "volume_ml": round(candidate.volume_ml, 6),
                        "qc_score": "" if qc.qc_score is None else round(qc.qc_score, 6),
                        "qc_predicted_organ": qc.qc_predicted_organ,
                        "qc_predicted_organ_idx": qc.qc_predicted_organ_idx,
                        "qc_error": qc.error,
                        "selected_by_largest_volume": candidate is best_volume_candidate,
                        "selected_mask_path": selected_paths.get(organ, ""),
                    }
                    rows_for_organ.append(row)

                for row in rows_for_organ:
                    row["selected_by_highest_qc"] = (
                        best_qc_candidate is not None
                        and row["source_tool"] == best_qc_candidate.tool_name
                        and row["mask_path"] == best_qc_candidate.mask_path
                    )
                outcome.candidate_rows.extend(rows_for_organ)

            outcome.radiomics_rows = self._extract_case_radiomics(
                image_path,
                selected_paths,
                item.case_id,
                item.case_path,
            )

            selected_organs = {}
            for organ, best in best_by_organ.items():
                qc = qc_by_key.get((organ, best.tool_name), QCResult())
                selected_organs[organ] = {
                    "selected_by": "largest_volume",
                    "source_tool": best.tool_name,
                    "source_mask_path": best.mask_path,
                    "selected_mask_path": selected_paths.get(organ, ""),
                    "volume_ml": round(best.volume_ml, 6),
                    "voxel_count": best.voxel_count,
                    "qc_score": qc.qc_score,
                    "qc_predicted_organ": qc.qc_predicted_organ,
                    "qc_error": qc.error,
                }
            outcome.selected_organs = selected_organs

            write_rows_csv(os.path.join(best_dir, "qc_scores.csv"), outcome.candidate_rows)
            if outcome.radiomics_rows:
                write_rows_csv(os.path.join(best_dir, "radiomics_features.csv"), outcome.radiomics_rows)
            write_json(
                os.path.join(best_dir, "selection_summary.json"),
                {
                    "case_id": item.case_id,
                    "case_path": item.case_path,
                    "modality": modality,
                    "target_organs": list(TARGET_ORGANS),
                    "selected_organs": selected_organs,
                    "notes": [
                        "Best masks are selected by largest volume, not by QC score.",
                        "QC scores are recorded for posthoc comparison and possible correction.",
                        "Radiomics features are shape + first-order only; texture classes are disabled.",
                    ],
                },
            )

            outcome.status = "completed"
            return self._finalize_outcome(outcome, t0)

        except Exception as exc:
            logger.exception("[%s] Failed", item.case_id)
            outcome.status = "failed"
            outcome.error = str(exc)
            return self._finalize_outcome(outcome, t0)

    def _extract_case_radiomics(
        self,
        image_path: str,
        selected_paths: dict[str, str],
        case_id: str,
        case_path: str,
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
                        "selected_by": "largest_volume",
                    }
                )
                rows.append(row)
            except Exception as exc:
                logger.warning("[%s] Radiomics failed for %s: %s", case_id, organ, exc)
        return rows

    def _finalize_outcome(self, outcome: CaseOutcome, t0: float) -> CaseOutcome:
        outcome.total_time_s = time.time() - t0
        self.csv.append_rows(outcome.candidate_rows)
        self.tracker.set_status(
            outcome.case_id,
            outcome.status,
            case_path=outcome.case_path,
            selected_organs=outcome.selected_organs,
            error=outcome.error,
            warnings=outcome.warnings,
            total_time_s=outcome.total_time_s,
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

    def write_correlation_outputs(self, *, max_cases: int = 100) -> None:
        if not os.path.isfile(self.candidate_csv):
            return
        rows: list[dict[str, str]] = []
        seen_cases: set[str] = set()
        with open(self.candidate_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = row.get("case_id", "")
                if case_id not in seen_cases:
                    if len(seen_cases) >= max_cases:
                        break
                    seen_cases.add(case_id)
                rows.append(row)
        if not rows:
            return

        numeric = []
        for row in rows:
            try:
                volume = float(row.get("volume_ml") or "nan")
                score = float(row.get("qc_score") or "nan")
            except ValueError:
                continue
            if math.isfinite(volume) and math.isfinite(score):
                numeric.append((volume, score, row.get("organ", ""), row.get("selected_by_largest_volume") == "True"))
        if len(numeric) < 3:
            return

        volumes = np.array([x[0] for x in numeric], dtype=float)
        scores = np.array([x[1] for x in numeric], dtype=float)
        pearson = float(np.corrcoef(volumes, scores)[0, 1]) if len(numeric) >= 2 else float("nan")
        ranks_v = np.argsort(np.argsort(volumes))
        ranks_s = np.argsort(np.argsort(scores))
        spearman = float(np.corrcoef(ranks_v, ranks_s)[0, 1]) if len(numeric) >= 2 else float("nan")

        summary = {
            "max_cases": max_cases,
            "case_count": len(seen_cases),
            "candidate_mask_count": len(numeric),
            "pearson_volume_vs_qc_score": pearson,
            "spearman_volume_vs_qc_score": spearman,
            "note": "Computed over QC candidate masks from the first completed cases in qc_radiomics_candidates.csv.",
        }
        write_json(os.path.join(LOG_DIR, "qc_volume_correlation_first100.json"), summary)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            selected = np.array([x[3] for x in numeric], dtype=bool)
            plt.figure(figsize=(7, 5))
            plt.scatter(volumes[~selected], scores[~selected], s=18, alpha=0.45, label="candidate")
            plt.scatter(volumes[selected], scores[selected], s=28, alpha=0.8, label="selected by volume")
            plt.xlabel("Mask volume (mL)")
            plt.ylabel("MedSegQC predicted Dice")
            plt.title(f"QC score vs mask volume, first {len(seen_cases)} cases\nPearson={pearson:.3f}, Spearman={spearman:.3f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(LOG_DIR, "qc_volume_correlation_first100.png"), dpi=160)
            plt.close()
        except Exception as exc:
            logger.warning("Could not write QC-volume correlation plot: %s", exc)


def parse_gpus(raw: str) -> list[int]:
    if raw.strip().lower() in {"", "none", "cpu"}:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_self_test() -> None:
    """Tiny dry-run integration test for path discovery, best-mask choice, and state."""
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "segmentations_3d")
        case_rel = "LUPUS/PT001/20240101/CT_ABDOMEN_PELVIS_W_CONTRAST/AXIAL"
        case_id = case_rel.replace("/", "__")
        case_path = os.path.join(root, case_rel)
        os.makedirs(case_path, exist_ok=True)
        image = np.zeros((24, 24, 24), dtype=np.float32)
        image[4:18, 4:18, 4:18] = 100.0
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(image, affine), os.path.join(case_path, IMAGE_FILENAME))
        for dirname, radius in [("segmentations_totalseg_ct", 3), ("segmentations_vista3d", 5)]:
            seg_dir = os.path.join(case_path, dirname)
            os.makedirs(seg_dir, exist_ok=True)
            z, y, x = np.ogrid[:24, :24, :24]
            mask = ((z - 12) ** 2 + (y - 12) ** 2 + (x - 12) ** 2 <= radius**2).astype(np.uint8)
            nib.save(nib.Nifti1Image(mask, affine), os.path.join(seg_dir, "liver.nii.gz"))
        seg_state = os.path.join(tmp, "pipeline_state_v2.json")
        with open(seg_state, "w") as f:
            json.dump(
                {
                    "run_id": "self_test",
                    "cases": {
                        case_id: {
                            "status": "completed",
                            "tools_run": ["TotalSegmentator_CT", "VISTA3D"],
                            "error": "",
                        }
                    },
                },
                f,
            )

        runner = BatchQCRadiomicsRunner(
            segmentation_state=seg_state,
            segmentation_root=root,
            qc_state_file=os.path.join(tmp, "qc_state.json"),
            candidate_csv=os.path.join(tmp, "candidates.csv"),
            gpus=[],
            workers=1,
            dry_run_qc=True,
            skip_radiomics=True,
        )
        summary = runner.run()
        best_mask = os.path.join(case_path, BEST_DIRNAME, "liver.nii.gz")
        if summary["completed"] != 1 or not os.path.isfile(best_mask):
            raise AssertionError(f"Self-test failed: {summary}")
        selected = json.load(open(os.path.join(case_path, BEST_DIRNAME, "selection_summary.json")))
        tool = selected["selected_organs"]["liver"]["source_tool"]
        if tool != "VISTA3D":
            raise AssertionError(f"Expected largest VISTA3D mask, got {tool}")
        print("Self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MedSegQC + fast radiomics over completed segmentation cases."
    )
    parser.add_argument("--segmentation-state", default=DEFAULT_SEGMENTATION_STATE)
    parser.add_argument("--segmentation-root", default=SEGMENTATION_ROOT)
    parser.add_argument("--qc-state-file", default="")
    parser.add_argument("--candidate-csv", default="")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, or 'cpu'")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ct-checkpoint", default=DEFAULT_CT_QC_CHECKPOINT)
    parser.add_argument("--mri-checkpoint", default=DEFAULT_MRI_QC_CHECKPOINT)
    parser.add_argument("--watch", action="store_true", help="Keep polling segmentation state for new completed cases")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-run even if segmentations_best/ is non-empty")
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
