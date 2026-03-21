"""
TotalSegmentator MR Tool Wrapper
==================================
Segments 50 anatomical structures from MRI volumes (sequence-independent).
Invoked with the -ta total_mr flag.

D'Antonoli et al., Radiology, 2025
https://github.com/wasserth/TotalSegmentator
Version verified: 2.13.0
"""

import json
import os
import subprocess
import time
from typing import List

from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput


class TotalSegmentatorMRTool(BaseSegmentationTool):

    name = "TotalSegmentator_MR"
    conda_env = "cdw_totalseg"
    supported_modalities = ["MRI"]
    supported_anatomies = ["all"]
    supports_2d = False
    supports_3d = True

    # Verified 50 structures from TotalSegmentator/totalsegmentator/map_to_binary.py (class_map['total_mr'])
    ALL_ORGANS = [
        "spleen", "kidney_right", "kidney_left", "gallbladder", "liver",
        "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left",
        "lung_left", "lung_right", "esophagus", "small_bowel", "duodenum",
        "colon", "urinary_bladder", "prostate",
        "sacrum", "vertebrae", "intervertebral_discs", "spinal_cord",
        "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
        "iliac_artery_left", "iliac_artery_right",
        "iliac_vena_left", "iliac_vena_right",
        "humerus_left", "humerus_right",
        "scapula_left", "scapula_right",
        "clavicula_left", "clavicula_right",
        "femur_left", "femur_right",
        "hip_left", "hip_right",
        "gluteus_maximus_left", "gluteus_maximus_right",
        "gluteus_medius_left", "gluteus_medius_right",
        "gluteus_minimus_left", "gluteus_minimus_right",
        "autochthon_left", "autochthon_right",
        "iliopsoas_left", "iliopsoas_right",
        "brain",
    ]

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(self, inp: ToolInput) -> ToolOutput:
        t0 = time.time()
        output_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_totalseg_mr")

        if self.dry_run:
            return self._dry_run(inp, output_dir, t0)

        if not os.path.isfile(inp.image_path):
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"Image not found: {inp.image_path}",
                runtime_seconds=time.time() - t0,
            )

        device = "gpu" if inp.device.startswith("gpu") else inp.device

        cmd = [
            "TotalSegmentator",
            "-i", inp.image_path,
            "-o", output_dir,
            "-ot", "nifti",
            "-ta", "total_mr",
            "--statistics",
            "--device", device,
        ]

        try:
            self._run_in_env(cmd)
        except subprocess.CalledProcessError as e:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"TotalSegmentator MR failed (exit {e.returncode})",
                runtime_seconds=time.time() - t0,
            )

        organs_segmented = [
            organ for organ in self.ALL_ORGANS
            if os.path.isfile(os.path.join(output_dir, f"{organ}.nii.gz"))
        ]

        statistics = {}
        stats_file = os.path.join(output_dir, "statistics.json")
        if os.path.isfile(stats_file):
            with open(stats_file) as f:
                statistics = json.load(f)

        return ToolOutput(
            tool_name=self.name,
            case_path=inp.case_path,
            seg_dir=output_dir,
            organs_segmented=organs_segmented,
            statistics=statistics,
            success=True,
            runtime_seconds=time.time() - t0,
        )

    def _dry_run(self, inp: ToolInput, output_dir: str, t0: float) -> ToolOutput:
        import random
        simulated_organs = [
            "spleen", "kidney_right", "kidney_left", "gallbladder", "liver",
            "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left",
            "lung_left", "lung_right", "esophagus", "small_bowel", "duodenum",
            "colon", "urinary_bladder", "heart", "aorta", "inferior_vena_cava",
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
