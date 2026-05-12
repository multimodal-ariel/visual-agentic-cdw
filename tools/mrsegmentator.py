"""
MRSegmentator Tool Wrapper
============================
40-class multi-organ segmentation for CT and MRI (T1, T2, Dixon).
Outputs a single multilabel NIfTI; wrapper splits to per-organ binary masks.

CLI: mrsegmentator --input <img> --outdir <dir>
Output: <outdir>/<input_stem>_seg.nii.gz  (multilabel, 40 classes)

Häntze et al., arXiv 2405.06463, 2024
https://github.com/hhaentze/MRSegmentator
Version verified: 1.3.1
"""

import os
import subprocess
import time
from typing import List

from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput
from processing.format_utils import split_multilabel_nifti


class MRSegmentatorTool(BaseSegmentationTool):

    name = "MRSegmentator"
    conda_env = "cdw_mrseg"
    supported_modalities = ["CT", "MRI"]
    supported_anatomies = ["chest", "abdomen", "pelvis", "abdomen_pelvis",
                           "chest_abdomen_pelvis", "whole_body"]
    supports_2d = False
    supports_3d = True

    # Verified 40-class label map from MRSegmentator/README.md (labels 1-40)
    LABEL_MAP = {
        1:  "spleen",
        2:  "kidney_right",
        3:  "kidney_left",
        4:  "gallbladder",
        5:  "liver",
        6:  "stomach",
        7:  "pancreas",
        8:  "adrenal_gland_right",
        9:  "adrenal_gland_left",
        10: "lung_left",
        11: "lung_right",
        12: "heart",
        13: "aorta",
        14: "inferior_vena_cava",
        15: "portal_vein_and_splenic_vein",
        16: "iliac_artery_left",
        17: "iliac_artery_right",
        18: "iliac_vena_left",
        19: "iliac_vena_right",
        20: "esophagus",
        21: "small_bowel",
        22: "duodenum",
        23: "colon",
        24: "urinary_bladder",
        25: "spine",
        26: "sacrum",
        27: "hip_left",
        28: "hip_right",
        29: "femur_left",
        30: "femur_right",
        31: "autochthon_left",
        32: "autochthon_right",
        33: "iliopsoas_left",
        34: "iliopsoas_right",
        35: "gluteus_maximus_left",
        36: "gluteus_maximus_right",
        37: "gluteus_medius_left",
        38: "gluteus_medius_right",
        39: "gluteus_minimus_left",
        40: "gluteus_minimus_right",
    }

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(self, inp: ToolInput) -> ToolOutput:
        t0 = time.time()
        output_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_mrseg")

        if self.dry_run:
            return self._dry_run(inp, output_dir, t0)

        if not os.path.isfile(inp.image_path):
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"Image not found: {inp.image_path}",
                runtime_seconds=time.time() - t0,
            )

        os.makedirs(output_dir, exist_ok=True)

        # CLI outputs: <outdir>/<input_stem>_seg.nii.gz (postfix="seg" is default)
        input_stem = os.path.basename(inp.image_path).replace(".nii.gz", "")
        multilabel_path = os.path.join(output_dir, f"{input_stem}_seg.nii.gz")

        cmd = [
            "mrsegmentator",
            "--input", inp.image_path,
            "--outdir", output_dir,
        ]

        gpu_id = self._parse_gpu_id(inp.device)

        try:
            self._run_in_env(cmd, gpu_id=gpu_id, timeout=inp.timeout_s)
        except subprocess.TimeoutExpired:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"MRSegmentator timed out after {inp.timeout_s}s",
                runtime_seconds=time.time() - t0,
            )
        except subprocess.CalledProcessError as e:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"MRSegmentator failed (exit {e.returncode})",
                runtime_seconds=time.time() - t0,
            )

        if not os.path.isfile(multilabel_path):
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"Expected multilabel output not found: {multilabel_path}",
                runtime_seconds=time.time() - t0,
            )

        organs_segmented = split_multilabel_nifti(multilabel_path, self.LABEL_MAP, output_dir)

        return ToolOutput(
            tool_name=self.name,
            case_path=inp.case_path,
            seg_dir=output_dir,
            organs_segmented=organs_segmented,
            statistics={},
            success=True,
            runtime_seconds=time.time() - t0,
        )

    def _dry_run(self, inp: ToolInput, output_dir: str, t0: float) -> ToolOutput:
        import random
        simulated_organs = [
            "spleen", "kidney_left", "kidney_right", "liver", "stomach", "pancreas",
            "gallbladder", "adrenal_gland_left", "adrenal_gland_right",
            "lung_left", "lung_right", "heart", "aorta", "inferior_vena_cava",
            "esophagus", "small_bowel", "duodenum", "colon", "urinary_bladder",
        ]
        simulated_stats = {o: {"volume_mm3": random.uniform(1000, 500000)} for o in simulated_organs}
        return ToolOutput(
            tool_name=self.name,
            case_path=inp.case_path,
            seg_dir=output_dir,
            organs_segmented=simulated_organs,
            statistics=simulated_stats,
            success=True,
            runtime_seconds=time.time() - t0,
        )
