"""
MRISegmenter Tool Wrapper
===========================
62-structure segmentation for T1-weighted abdominal MRI.
15 organs, 7 vessels, 8 muscles, 32 bones/vertebrae.
Trained on pre-contrast and multi-phase contrast-enhanced T1w abdominal MRI.
Outputs a single multilabel NIfTI; wrapper splits to per-organ binary masks.

CLI: MRISegmentator -i <input> -o <output.nii.gz> -d <device>

Zhuang et al., Radiology, 2025
https://github.com/rsummers11/MRISegmenter
Version verified: 0.4.2
"""

import os
import subprocess
import time
from typing import List

from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput
from processing.format_utils import split_multilabel_nifti


class MRISegmenterTool(BaseSegmentationTool):

    name = "MRISegmenter"
    conda_env = "cdw_mriseg"
    supported_modalities = ["MRI"]
    supported_anatomies = ["abdomen", "pelvis", "abdomen_pelvis"]
    supports_2d = False
    supports_3d = True

    # Verified 62-class label map from MRISegmenter/README.md
    LABEL_MAP = {
        1:  "spleen",
        2:  "kidney_right",
        3:  "kidney_left",
        4:  "gallbladder",
        5:  "liver",
        6:  "esophagus",
        7:  "stomach",
        8:  "aorta",
        9:  "inferior_vena_cava",
        10: "portal_vein_and_splenic_vein",
        11: "pancreas",
        12: "adrenal_gland_right",
        13: "adrenal_gland_left",
        14: "lung_right",
        15: "lung_left",
        16: "small_bowel",
        17: "duodenum",
        18: "colon",
        19: "iliac_artery_left",
        20: "iliac_artery_right",
        21: "iliac_vena_left",
        22: "iliac_vena_right",
        23: "gluteus_maximus_left",
        24: "gluteus_maximus_right",
        25: "gluteus_medius_left",
        26: "gluteus_medius_right",
        27: "autochthon_left",
        28: "autochthon_right",
        29: "iliopsoas_left",
        30: "iliopsoas_right",
        31: "hip_left",
        32: "hip_right",
        33: "sacrum",
        34: "rib_left_4",
        35: "rib_left_5",
        36: "rib_left_6",
        37: "rib_left_7",
        38: "rib_left_8",
        39: "rib_left_9",
        40: "rib_left_10",
        41: "rib_left_11",
        42: "rib_left_12",
        43: "rib_right_4",
        44: "rib_right_5",
        45: "rib_right_6",
        46: "rib_right_7",
        47: "rib_right_8",
        48: "rib_right_9",
        49: "rib_right_10",
        50: "rib_right_11",
        51: "rib_right_12",
        52: "vertebrae_L5",
        53: "vertebrae_L4",
        54: "vertebrae_L3",
        55: "vertebrae_L2",
        56: "vertebrae_L1",
        57: "vertebrae_T12",
        58: "vertebrae_T11",
        59: "vertebrae_T10",
        60: "vertebrae_T9",
        61: "vertebrae_T8",
        62: "vertebrae_T7",
    }

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(self, inp: ToolInput) -> ToolOutput:
        t0 = time.time()
        output_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_mrisegmenter")

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
        multilabel_path = os.path.join(output_dir, "segmentation.nii.gz")

        # MRISegmenter device arg accepts {gpu, cpu, mps} only
        device_str = "gpu" if inp.device.startswith("gpu") else inp.device

        cmd = [
            "MRISegmentator",
            "-i", inp.image_path,
            "-o", multilabel_path,
            "-d", device_str,
        ]

        try:
            self._run_in_env(cmd)
        except subprocess.CalledProcessError as e:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"MRISegmenter failed (exit {e.returncode})",
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
            "spleen", "kidney_right", "kidney_left", "gallbladder", "liver",
            "esophagus", "stomach", "aorta", "inferior_vena_cava", "pancreas",
            "adrenal_gland_right", "adrenal_gland_left", "lung_right", "lung_left",
            "small_bowel", "duodenum", "colon",
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
