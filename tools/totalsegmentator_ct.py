"""
TotalSegmentator CT Tool Wrapper
=================================
Segments 117 anatomical structures from CT images.
nnU-Net based. Per-organ NIfTI files + statistics.json written to output dir.

Wasserthal et al., Radiology: AI, 2023
https://github.com/wasserth/TotalSegmentator
Version verified: 2.13.0
"""

import json
import os
import subprocess
import time
from typing import List

from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput


class TotalSegmentatorCTTool(BaseSegmentationTool):

    name = "TotalSegmentator_CT"
    conda_env = "cdw_totalseg"
    supported_modalities = ["CT"]
    supported_anatomies = ["all"]
    supports_2d = False
    supports_3d = True

    # Verified 117 structures from TotalSegmentator/totalsegmentator/map_to_binary.py (class_map['total'])
    ALL_ORGANS = [
        "spleen", "kidney_right", "kidney_left", "gallbladder", "liver",
        "stomach", "pancreas", "adrenal_gland_right", "adrenal_gland_left",
        "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
        "lung_middle_lobe_right", "lung_lower_lobe_right", "esophagus", "trachea",
        "thyroid_gland", "small_bowel", "duodenum", "colon", "urinary_bladder",
        "prostate", "kidney_cyst_left", "kidney_cyst_right",
        "sacrum", "vertebrae_S1",
        "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
        "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
        "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
        "vertebrae_T2", "vertebrae_T1",
        "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4",
        "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
        "heart", "aorta", "pulmonary_vein", "brachiocephalic_trunk",
        "subclavian_artery_right", "subclavian_artery_left",
        "common_carotid_artery_right", "common_carotid_artery_left",
        "brachiocephalic_vein_left", "brachiocephalic_vein_right",
        "atrial_appendage_left", "superior_vena_cava", "inferior_vena_cava",
        "portal_vein_and_splenic_vein",
        "iliac_artery_left", "iliac_artery_right",
        "iliac_vena_left", "iliac_vena_right",
        "humerus_left", "humerus_right",
        "scapula_left", "scapula_right",
        "clavicula_left", "clavicula_right",
        "femur_left", "femur_right",
        "hip_left", "hip_right",
        "spinal_cord",
        "gluteus_maximus_left", "gluteus_maximus_right",
        "gluteus_medius_left", "gluteus_medius_right",
        "gluteus_minimus_left", "gluteus_minimus_right",
        "autochthon_left", "autochthon_right",
        "iliopsoas_left", "iliopsoas_right",
        "brain", "skull",
        "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5",
        "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10",
        "rib_left_11", "rib_left_12",
        "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5",
        "rib_right_6", "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10",
        "rib_right_11", "rib_right_12",
        "sternum", "costal_cartilages",
    ]

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(self, inp: ToolInput) -> ToolOutput:
        t0 = time.time()
        output_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_totalseg_ct")

        if self.dry_run:
            return self._dry_run(inp, output_dir, t0)

        if not os.path.isfile(inp.image_path):
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"Image not found: {inp.image_path}",
                runtime_seconds=time.time() - t0,
            )

        # TotalSegmentator device format: "gpu:0" → "gpu" (it selects GPU automatically).
        # The physical GPU is pinned via CUDA_VISIBLE_DEVICES on the subprocess so
        # parallel workers don't all default to cuda:0.
        device = "gpu" if inp.device.startswith("gpu") else inp.device
        gpu_id = self._parse_gpu_id(inp.device)

        cmd = [
            "TotalSegmentator",
            "-i", inp.image_path,
            "-o", output_dir,
            "-ot", "nifti",
            "--statistics",
            "--device", device,
        ]

        try:
            self._run_in_env(cmd, gpu_id=gpu_id, timeout=inp.timeout_s)
        except subprocess.TimeoutExpired:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"TotalSegmentator CT timed out after {inp.timeout_s}s",
                runtime_seconds=time.time() - t0,
            )
        except subprocess.CalledProcessError as e:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"TotalSegmentator CT failed (exit {e.returncode})",
                runtime_seconds=time.time() - t0,
            )

        # Collect present organ masks
        organs_segmented = [
            organ for organ in self.ALL_ORGANS
            if os.path.isfile(os.path.join(output_dir, f"{organ}.nii.gz"))
        ]

        # Load native statistics if available
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
            o for o in self.ALL_ORGANS
            if not o.startswith(("rib_", "vertebrae_", "skull", "costal_cartilages",
                                  "humerus", "scapula", "clavicula", "femur", "hip_",
                                  "gluteus", "autochthon", "iliopsoas", "sternum"))
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
