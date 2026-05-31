"""
SynthSeg Tool Wrapper
=====================
Contrast-agnostic brain MRI and CT segmentation from Billot et al.

SynthSeg writes a multilabel segmentation at 1 mm isotropic resolution. This
wrapper resamples that label map back to the original image_nifti.nii.gz grid
with nearest-neighbour interpolation, then writes per-structure binary masks
using the standard tool contract.

GitHub: https://github.com/BBillot/SynthSeg
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from typing import Dict

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from config.constants import (
    SYNTHSEG_PARC_CHECKPOINT,
    SYNTHSEG_QC_CHECKPOINT,
    SYNTHSEG_ROBUST_CHECKPOINT,
    SYNTHSEG_STANDARD_CHECKPOINT,
)
from processing.format_utils import save_organ_mask
from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput


class SynthSegTool(BaseSegmentationTool):
    name = "SynthSeg"
    conda_env = "cdw_synthseg"
    supported_modalities = ["MRI", "CT"]
    supported_anatomies = ["head"]
    supports_2d = False
    supports_3d = True

    # SynthSeg whole-brain labels, canonicalised to repo naming conventions.
    # The combined "brain" mask is also written from all non-zero labels.
    LABEL_MAP: Dict[int, str] = {
        2: "cerebral_white_matter_left",
        3: "cerebral_cortex_left",
        4: "lateral_ventricle_left",
        5: "inferior_lateral_ventricle_left",
        7: "cerebellum_white_matter_left",
        8: "cerebellum_cortex_left",
        10: "thalamus_left",
        11: "caudate_left",
        12: "putamen_left",
        13: "pallidum_left",
        14: "third_ventricle",
        15: "fourth_ventricle",
        16: "brainstem",
        17: "hippocampus_left",
        18: "amygdala_left",
        24: "cerebrospinal_fluid",
        26: "accumbens_left",
        28: "ventral_dc_left",
        41: "cerebral_white_matter_right",
        42: "cerebral_cortex_right",
        43: "lateral_ventricle_right",
        44: "inferior_lateral_ventricle_right",
        46: "cerebellum_white_matter_right",
        47: "cerebellum_cortex_right",
        49: "thalamus_right",
        50: "caudate_right",
        51: "putamen_right",
        52: "pallidum_right",
        53: "hippocampus_right",
        54: "amygdala_right",
        58: "accumbens_right",
        60: "ventral_dc_right",
    }

    def __init__(
        self,
        dry_run: bool = False,
        synthseg_dir: str | None = None,
        robust: bool | None = None,
        parc: bool = False,
        segmentation_checkpoint: str | None = None,
        qc_checkpoint: str | None = None,
        parc_checkpoint: str | None = None,
    ):
        self.dry_run = dry_run
        if synthseg_dir:
            self.synthseg_dir = synthseg_dir
        elif os.environ.get("SYNTHSEG_DIR"):
            self.synthseg_dir = os.environ["SYNTHSEG_DIR"]
        else:
            pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.synthseg_dir = os.path.join(pipeline_root, "external", "SynthSeg")

        if robust is None:
            robust = os.environ.get("SYNTHSEG_ROBUST", "1").strip().lower() not in {"0", "false", "no"}
        self.robust = robust
        self.parc = parc
        default_seg_ckpt = SYNTHSEG_ROBUST_CHECKPOINT if self.robust else SYNTHSEG_STANDARD_CHECKPOINT
        self.segmentation_checkpoint = (
            segmentation_checkpoint
            or os.environ.get("SYNTHSEG_CHECKPOINT")
            or default_seg_ckpt
        )
        self.qc_checkpoint = qc_checkpoint or os.environ.get("SYNTHSEG_QC_CHECKPOINT") or SYNTHSEG_QC_CHECKPOINT
        self.parc_checkpoint = (
            parc_checkpoint
            or os.environ.get("SYNTHSEG_PARC_CHECKPOINT")
            or SYNTHSEG_PARC_CHECKPOINT
        )

    def run(self, inp: ToolInput) -> ToolOutput:
        t0 = time.time()
        output_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_synthseg")

        if self.dry_run:
            return self._dry_run(inp, output_dir, t0)

        modality = inp.modality.upper()
        if modality not in {"MRI", "CT"} or inp.anatomy.lower() != "head":
            return ToolOutput(
                tool_name=self.name,
                case_path=inp.case_path,
                seg_dir=output_dir,
                success=False,
                error=f"SynthSeg is restricted to brain/head MRI or CT (got {inp.modality}/{inp.anatomy})",
                runtime_seconds=time.time() - t0,
            )

        if not os.path.isfile(inp.image_path):
            return ToolOutput(
                tool_name=self.name,
                case_path=inp.case_path,
                seg_dir=output_dir,
                success=False,
                error=f"Image not found: {inp.image_path}",
                runtime_seconds=time.time() - t0,
            )

        runner = os.path.join(self.synthseg_dir, "scripts", "commands", "SynthSeg_predict.py")
        if not os.path.isfile(runner):
            return ToolOutput(
                tool_name=self.name,
                case_path=inp.case_path,
                seg_dir=output_dir,
                success=False,
                error=f"SynthSeg runner not found: {runner}. Set SYNTHSEG_DIR or clone into external/SynthSeg.",
                runtime_seconds=time.time() - t0,
            )

        try:
            self._ensure_upstream_model_files()
        except FileNotFoundError as exc:
            return ToolOutput(
                tool_name=self.name,
                case_path=inp.case_path,
                seg_dir=output_dir,
                success=False,
                error=str(exc),
                runtime_seconds=time.time() - t0,
            )

        os.makedirs(output_dir, exist_ok=True)
        raw_seg = os.path.join(output_dir, "synthseg_raw_1mm.nii.gz")
        resampled_seg = os.path.join(output_dir, "synthseg_labels_resampled.nii.gz")
        volumes_csv = os.path.join(output_dir, "synthseg_volumes.csv")
        qc_csv = os.path.join(output_dir, "synthseg_qc.csv")

        cmd = [
            "python",
            runner,
            "--i",
            inp.image_path,
            "--o",
            raw_seg,
            "--vol",
            volumes_csv,
            "--qc",
            qc_csv,
        ]
        if self.robust:
            cmd.append("--robust")
        if self.parc:
            cmd.append("--parc")
        if modality == "CT":
            cmd.append("--ct")
        if inp.device == "cpu":
            cmd.append("--cpu")

        gpu_id = self._parse_gpu_id(inp.device)
        try:
            self._run_in_env(cmd, cwd=self.synthseg_dir, gpu_id=gpu_id, timeout=inp.timeout_s)
        except subprocess.TimeoutExpired:
            return ToolOutput(
                tool_name=self.name,
                case_path=inp.case_path,
                seg_dir=output_dir,
                success=False,
                error=f"SynthSeg timed out after {inp.timeout_s}s",
                runtime_seconds=time.time() - t0,
            )
        except subprocess.CalledProcessError as exc:
            return ToolOutput(
                tool_name=self.name,
                case_path=inp.case_path,
                seg_dir=output_dir,
                success=False,
                error=f"SynthSeg failed (exit {exc.returncode})",
                runtime_seconds=time.time() - t0,
            )

        if not os.path.isfile(raw_seg):
            return ToolOutput(
                tool_name=self.name,
                case_path=inp.case_path,
                seg_dir=output_dir,
                success=False,
                error=f"Expected SynthSeg output not found: {raw_seg}",
                runtime_seconds=time.time() - t0,
            )

        try:
            organs_segmented = self._normalise_and_split(
                raw_seg=raw_seg,
                reference_image=inp.image_path,
                output_dir=output_dir,
                resampled_seg=resampled_seg,
            )
        except Exception as exc:
            return ToolOutput(
                tool_name=self.name,
                case_path=inp.case_path,
                seg_dir=output_dir,
                success=False,
                error=f"SynthSeg output normalisation failed: {exc}",
                runtime_seconds=time.time() - t0,
            )

        return ToolOutput(
            tool_name=self.name,
            case_path=inp.case_path,
            seg_dir=output_dir,
            organs_segmented=organs_segmented,
            statistics=self._read_synthseg_stats(volumes_csv, qc_csv),
            success=len(organs_segmented) > 0,
            error="" if organs_segmented else "SynthSeg produced no non-empty labels.",
            runtime_seconds=time.time() - t0,
        )

    def _normalise_and_split(
        self,
        *,
        raw_seg: str,
        reference_image: str,
        output_dir: str,
        resampled_seg: str,
    ) -> list[str]:
        ref_img = nib.load(reference_image)
        label_img = nib.load(raw_seg)
        resampled = resample_from_to(label_img, ref_img, order=0)
        labels = np.rint(np.asanyarray(resampled.dataobj)).astype(np.int16)
        nib.save(nib.Nifti1Image(labels, ref_img.affine, ref_img.header), resampled_seg)

        saved: list[str] = []
        brain = labels > 0
        if np.any(brain):
            save_organ_mask(brain.astype(np.uint8), "brain", output_dir, ref_img.affine, ref_img.header)
            saved.append("brain")

        for label_idx, organ in self.LABEL_MAP.items():
            mask = labels == label_idx
            if not np.any(mask):
                continue
            save_organ_mask(mask.astype(np.uint8), organ, output_dir, ref_img.affine, ref_img.header)
            saved.append(organ)
        return saved

    def _read_synthseg_stats(self, volumes_csv: str, qc_csv: str) -> dict:
        stats = {"volumes_csv": volumes_csv, "qc_csv": qc_csv}
        for key, path in (("volumes", volumes_csv), ("qc", qc_csv)):
            if not os.path.isfile(path):
                continue
            try:
                with open(path, newline="") as f:
                    stats[key] = list(csv.DictReader(f))
            except Exception:
                stats[key] = []
        return stats

    def _ensure_upstream_model_files(self) -> None:
        models_dir = os.path.join(self.synthseg_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        required = {
            "synthseg_robust_2.0.h5" if self.robust else "synthseg_2.0.h5": self.segmentation_checkpoint,
            "synthseg_qc_2.0.h5": self.qc_checkpoint,
        }
        if self.parc:
            required["synthseg_parc_2.0.h5"] = self.parc_checkpoint

        for model_name, source in required.items():
            destination = os.path.join(models_dir, model_name)
            if os.path.isfile(destination):
                continue
            if not os.path.isfile(source):
                raise FileNotFoundError(
                    "SynthSeg checkpoint not found. Expected "
                    f"{source}. Download/place it under checkpoints/SynthSeg/ "
                    f"or set the appropriate SYNTHSEG_*_CHECKPOINT environment variable."
                )
            try:
                os.symlink(os.path.abspath(source), destination)
            except OSError:
                shutil.copy2(source, destination)

    def _dry_run(self, inp: ToolInput, output_dir: str, t0: float) -> ToolOutput:
        os.makedirs(output_dir, exist_ok=True)
        ref = nib.load(inp.image_path)
        shape = ref.shape[:3]
        z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
        center = tuple(s // 2 for s in shape)
        radius = max(2, min(shape) // 4)
        brain = (
            (z - center[0]) ** 2
            + (y - center[1]) ** 2
            + (x - center[2]) ** 2
            <= radius**2
        ).astype(np.uint8)
        save_organ_mask(brain, "brain", output_dir, ref.affine, ref.header)
        return ToolOutput(
            tool_name=self.name,
            case_path=inp.case_path,
            seg_dir=output_dir,
            organs_segmented=["brain"],
            statistics={},
            success=True,
            runtime_seconds=time.time() - t0,
        )
