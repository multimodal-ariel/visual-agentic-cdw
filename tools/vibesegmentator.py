"""
VIBESegmentator Tool Wrapper
==============================
Full-body segmentation for MRI and CT. 72 anatomical structures.
Script-based; requires cloned repo at VIBESEG_DIR or vibeseg_dir init arg.
Outputs a single multilabel NIfTI to out_path; wrapper splits to per-organ.

CLI: python run_VIBESegmentator.py --img <input> --out_path <output.nii.gz>
                                   --ddevice cuda --gpu <id>

Graf et al., European Radiology, 2025
https://github.com/robert-graf/VIBESegmentator
Version verified: latest (poetry-based, TotalVibeSegmentator package)
"""

import os
import subprocess
import time
from typing import List

from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput
from processing.format_utils import split_multilabel_nifti


class VIBESegmentatorTool(BaseSegmentationTool):

    name = "VIBESegmentator"
    conda_env = "cdw_vibeseg"
    supported_modalities = ["MRI", "CT"]
    supported_anatomies = ["all"]
    supports_2d = False
    supports_3d = True

    # Verified 72-class label map from VIBESegmentator/run_VIBESegmentator_multi.py
    # Label 20 ("unused") is intentionally skipped.
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
        10: "lung_upper_lobe_left",
        11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
        15: "esophagus",
        16: "trachea",
        17: "thyroid_gland",
        18: "intestine",
        19: "duodenum",
        # 20: "unused" — skipped
        21: "urinary_bladder",
        22: "prostate",
        23: "sacrum",
        24: "heart",
        25: "aorta",
        26: "pulmonary_vein",
        27: "brachiocephalic_trunk",
        28: "subclavian_artery_right",
        29: "subclavian_artery_left",
        30: "common_carotid_artery_right",
        31: "common_carotid_artery_left",
        32: "brachiocephalic_vein_left",
        33: "brachiocephalic_vein_right",
        34: "atrial_appendage_left",
        35: "superior_vena_cava",
        36: "inferior_vena_cava",
        37: "portal_vein_and_splenic_vein",
        38: "iliac_artery_left",
        39: "iliac_artery_right",
        40: "iliac_vena_left",
        41: "iliac_vena_right",
        42: "humerus_left",
        43: "humerus_right",
        44: "scapula_left",
        45: "scapula_right",
        46: "clavicula_left",
        47: "clavicula_right",
        48: "femur_left",
        49: "femur_right",
        50: "hip_left",
        51: "hip_right",
        52: "spinal_cord",
        53: "gluteus_maximus_left",
        54: "gluteus_maximus_right",
        55: "gluteus_medius_left",
        56: "gluteus_medius_right",
        57: "gluteus_minimus_left",
        58: "gluteus_minimus_right",
        59: "autochthon_left",
        60: "autochthon_right",
        61: "iliopsoas_left",
        62: "iliopsoas_right",
        63: "sternum",
        64: "costal_cartilages",
        65: "subcutaneous_fat",
        66: "muscle",
        67: "inner_fat",
        68: "IVD",
        69: "vertebra_body",
        70: "vertebra_posterior_elements",
        71: "spinal_channel",
        72: "bone_other",
    }

    def __init__(self, dry_run: bool = False, vibeseg_dir: str = None):
        """
        Args:
            dry_run: If True, return simulated outputs.
            vibeseg_dir: Path to cloned VIBESegmentator repo. Falls back to
                         VIBESEG_DIR env var, then defaults to
                         external/VIBESegmentator relative to pipeline root.
        """
        self.dry_run = dry_run
        if vibeseg_dir:
            self.vibeseg_dir = vibeseg_dir
        elif os.environ.get("VIBESEG_DIR"):
            self.vibeseg_dir = os.environ["VIBESEG_DIR"]
        else:
            # Default: alongside this repo's external/ directory
            pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.vibeseg_dir = os.path.join(pipeline_root, "external", "VIBESegmentator")

    def run(self, inp: ToolInput) -> ToolOutput:
        t0 = time.time()
        output_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_vibeseg")

        if self.dry_run:
            return self._dry_run(inp, output_dir, t0)

        if not os.path.isfile(inp.image_path):
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"Image not found: {inp.image_path}",
                runtime_seconds=time.time() - t0,
            )

        script = os.path.join(self.vibeseg_dir, "run_VIBESegmentator.py")
        if not os.path.isfile(script):
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"VIBESegmentator script not found: {script}. "
                      f"Set vibeseg_dir or VIBESEG_DIR env var.",
                runtime_seconds=time.time() - t0,
            )

        os.makedirs(output_dir, exist_ok=True)

        # out_path is the multilabel output file (single NIfTI)
        multilabel_path = os.path.join(output_dir, "multilabel_seg.nii.gz")

        # Device: "gpu:0" → ddevice="cuda", gpu=0
        if "gpu:" in inp.device:
            ddevice = "cuda"
            gpu_id = inp.device.split(":")[1]
        elif inp.device == "cpu":
            ddevice = "cpu"
            gpu_id = "0"
        else:
            ddevice = "cuda"
            gpu_id = "0"

        cmd = [
            "python", script,
            "--img", inp.image_path,
            "--out_path", multilabel_path,
            "--ddevice", ddevice,
            "--gpu", gpu_id,
        ]

        try:
            self._run_in_env(cmd, cwd=self.vibeseg_dir)
        except subprocess.CalledProcessError as e:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"VIBESegmentator failed (exit {e.returncode})",
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
            "spleen", "kidney_right", "kidney_left", "liver", "stomach", "pancreas",
            "gallbladder", "heart", "aorta", "inferior_vena_cava",
            "lung_upper_lobe_left", "lung_lower_lobe_left",
            "lung_upper_lobe_right", "lung_lower_lobe_right",
            "urinary_bladder", "subcutaneous_fat", "muscle", "inner_fat",
            "spinal_cord", "spinal_channel",
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
