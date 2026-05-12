"""
VISTA3D Tool Wrapper (NV-Segment-CTMR)
=========================================
NVIDIA VISTA3D — 345+ class CT/MRI segmentation foundation model.
218M parameters. Trained on CT, finetuned on CT+MRI.

Two prompt modes:
  label_prompt — list of integer class indices (targeted segmentation)
  modality     — "CT_BODY" | "MRI_BODY" | "MRI_BRAIN" (whole-body preset)

The LLM planner provides organ names; wrapper translates to integer indices
using the verified label map from NVSegmentCTMR/metadata.json.

Python API only — invoked via tools/runners/run_vista3d.py
inside the cdw_nvseg conda environment.

NVIDIA VISTA3D, 2024
https://github.com/Project-MONAI/VISTA
Version verified: MONAI Core v1.5.0, PyTorch 2.4.0
"""

import os
import subprocess
import time
from typing import Dict, List, Optional

from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput


_RUNNER = os.path.join(os.path.dirname(__file__), "runners", "run_vista3d.py")


class VISTA3DTool(BaseSegmentationTool):

    name = "VISTA3D"
    conda_env = "cdw_nvseg"
    supported_modalities = ["CT", "MRI"]
    supported_anatomies = ["all"]
    supports_2d = False
    supports_3d = True

    # Verified label map from NVSegmentCTMR/metadata.json (CT_BODY, 132 labels)
    # Maps organ name → integer label index. Names normalised: spaces→underscore, lowercase.
    ORGAN_TO_INDEX: Dict[str, int] = {
        "liver": 1,
        "kidney": 2,
        "spleen": 3,
        "pancreas": 4,
        "aorta": 6,
        "inferior_vena_cava": 7,
        "adrenal_gland_right": 8,
        "adrenal_gland_left": 9,
        "gallbladder": 10,
        "esophagus": 11,
        "stomach": 12,
        "duodenum": 13,
        "kidney_left": 14,
        "kidney_right": 5,
        "bladder": 15,
        "urinary_bladder": 15,
        "prostate_or_uterus": 16,
        "prostate": 118,
        "portal_vein_and_splenic_vein": 17,
        "rectum": 18,
        "small_bowel": 19,
        "lung": 20,
        "bone": 21,
        "brain": 22,
        "lung_tumor": 23,
        "pancreatic_tumor": 24,
        "hepatic_vessel": 25,
        "hepatic_tumor": 26,
        "colon": 62,
        "lung_upper_lobe_left": 28,
        "lung_lower_lobe_left": 29,
        "lung_upper_lobe_right": 30,
        "lung_middle_lobe_right": 31,
        "lung_lower_lobe_right": 32,
        "vertebrae_l5": 33, "vertebrae_L5": 33,
        "vertebrae_l4": 34, "vertebrae_L4": 34,
        "vertebrae_l3": 35, "vertebrae_L3": 35,
        "vertebrae_l2": 36, "vertebrae_L2": 36,
        "vertebrae_l1": 37, "vertebrae_L1": 37,
        "vertebrae_t12": 38, "vertebrae_T12": 38,
        "vertebrae_t11": 39, "vertebrae_T11": 39,
        "vertebrae_t10": 40, "vertebrae_T10": 40,
        "vertebrae_t9": 41,  "vertebrae_T9": 41,
        "vertebrae_t8": 42,  "vertebrae_T8": 42,
        "vertebrae_t7": 43,  "vertebrae_T7": 43,
        "vertebrae_t6": 44,  "vertebrae_T6": 44,
        "vertebrae_t5": 45,  "vertebrae_T5": 45,
        "vertebrae_t4": 46,  "vertebrae_T4": 46,
        "vertebrae_t3": 47,  "vertebrae_T3": 47,
        "vertebrae_t2": 48,  "vertebrae_T2": 48,
        "vertebrae_t1": 49,  "vertebrae_T1": 49,
        "vertebrae_c7": 50,  "vertebrae_C7": 50,
        "vertebrae_c6": 51,  "vertebrae_C6": 51,
        "vertebrae_c5": 52,  "vertebrae_C5": 52,
        "vertebrae_c4": 53,  "vertebrae_C4": 53,
        "vertebrae_c3": 54,  "vertebrae_C3": 54,
        "vertebrae_c2": 55,  "vertebrae_C2": 55,
        "vertebrae_c1": 56,  "vertebrae_C1": 56,
        "vertebrae_s1": 127, "vertebrae_S1": 127,
        "trachea": 57,
        "iliac_artery_left": 58,
         "iliac_artery_right": 59,
        "iliac_vena_left": 60,
        "iliac_vena_right": 61,
        "rib_left_1": 63,
        "rib_left_2": 64,
        "rib_left_3": 65,
        "rib_left_4": 66,
        "rib_left_5": 67,
        "rib_left_6": 68,
        "rib_left_7": 69,
        "rib_left_8": 70,
        "rib_left_9": 71,
        "rib_left_10": 72,
        "rib_left_11": 73,
        "rib_left_12": 74,
        "rib_right_1": 75,
        "rib_right_2": 76,
        "rib_right_3": 77,
        "rib_right_4": 78,
        "rib_right_5": 79,
        "rib_right_6": 80,
        "rib_right_7": 81,
        "rib_right_8": 82,
        "rib_right_9": 83,
        "rib_right_10": 84,
        "rib_right_11": 85,
        "rib_right_12": 86,
        "humerus_left": 87,
        "humerus_right": 88,
        "scapula_left": 89,
        "scapula_right": 90,
        "clavicula_left": 91,
        "clavicula_right": 92,
        "femur_left": 93,
        "femur_right": 94,
        "hip_left": 95,
        "hip_right": 96,
        "sacrum": 97,
        "gluteus_maximus_left": 98,
        "gluteus_maximus_right": 99,
        "gluteus_medius_left": 100,
        "gluteus_medius_right": 101,
        "gluteus_minimus_left": 102,
        "gluteus_minimus_right": 103,
        "autochthon_left": 104,
        "autochthon_right": 105,
        "iliopsoas_left": 106,
        "iliopsoas_right": 107,
        "atrial_appendage_left": 108,
        "brachiocephalic_trunk": 109,
        "brachiocephalic_vein_left": 110,
        "brachiocephalic_vein_right": 111,
        "common_carotid_artery_left": 112,
        "common_carotid_artery_right": 113,
        "costal_cartilages": 114,
        "heart": 115,
        "kidney_cyst_left": 116,
        "kidney_cyst_right": 117,
        "pulmonary_vein": 119,
        "skull": 120,
        "spinal_cord": 121,
        "sternum": 122,
        "subclavian_artery_left": 123,
        "subclavian_artery_right": 124,
        "superior_vena_cava": 125,
        "thyroid_gland": 126,
        "bone_lesion": 128,
        "kidney_mass": 129,
        "liver_tumor": 130,
        "airway": 132,
    }

    # Modality preset → VISTA3D preset string
    _MODALITY_PRESET = {
        "CT": "CT_BODY",
        "MRI": "MRI_BODY",
    }

    def __init__(
        self,
        dry_run: bool = False,
        nvseg_dir: str = None,
        use_modality_preset: bool = False,
    ):
        """
        Args:
            dry_run: If True, return simulated outputs.
            nvseg_dir: Path to external/NVSegmentCTMR repo root.
                       Falls back to NVSEG_DIR env var, then
                       <pipeline_root>/external/NVSegmentCTMR.
            use_modality_preset: If True, use modality-preset mode (CT_BODY/MRI_BODY)
                                 instead of label-prompt mode. Useful for whole-body scans.
        """
        self.dry_run = dry_run
        self.use_modality_preset = use_modality_preset

        if nvseg_dir:
            self.nvseg_dir = nvseg_dir
        elif os.environ.get("NVSEG_DIR"):
            self.nvseg_dir = os.environ["NVSEG_DIR"]
        else:
            pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Prefer checkpoints/NVSegmentCTMR (has weights + metadata + Python code);
            # fall back to external/NVSegmentCTMR if checkpoints not present.
            ckpt_dir = os.path.join(pipeline_root, "checkpoints", "NVSegmentCTMR")
            ext_dir  = os.path.join(pipeline_root, "external",    "NVSegmentCTMR")
            self.nvseg_dir = ckpt_dir if os.path.isdir(ckpt_dir) else ext_dir

    def _organs_to_label_indices(self, organs: List[str]) -> List[int]:
        """Translate organ name list to VISTA3D integer indices. Unrecognised names are skipped."""
        indices = []
        for organ in organs:
            idx = self.ORGAN_TO_INDEX.get(organ) or self.ORGAN_TO_INDEX.get(organ.lower())
            if idx is not None and idx not in indices:
                indices.append(idx)
        return sorted(indices)

    def run(self, inp: ToolInput) -> ToolOutput:
        t0 = time.time()
        output_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_vista3d")

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

        # Physical GPU is pinned via CUDA_VISIBLE_DEVICES on the subprocess,
        # so the runner only sees one GPU and addresses it as cuda:0.
        # run_vista3d.py passes args.device into torch.device(), which only
        # accepts "cpu" / "cuda" / "cuda:N" — never the bare "gpu" string.
        physical_gpu_id = self._parse_gpu_id(inp.device)
        device_str = "cpu" if inp.device == "cpu" else "cuda:0"

        cmd = [
            "python", _RUNNER,
            "--input", inp.image_path,
            "--output", output_dir,
            "--model_path", self.nvseg_dir,
            "--device", device_str,
        ]

        if self.use_modality_preset:
            preset = self._MODALITY_PRESET.get(inp.modality.upper(), "CT_BODY")
            cmd += ["--modality", preset]
        else:
            label_indices = self._organs_to_label_indices(inp.target_organs or [])
            if not label_indices:
                return ToolOutput(
                    tool_name=self.name, case_path=inp.case_path,
                    seg_dir=output_dir, success=False,
                    error=f"None of the requested organs are in the VISTA3D label map: {inp.target_organs}",
                    runtime_seconds=time.time() - t0,
                )
            cmd += ["--label_prompt"] + [str(i) for i in label_indices]

        try:
            self._run_in_env(cmd, gpu_id=physical_gpu_id, timeout=inp.timeout_s)
        except subprocess.TimeoutExpired:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"VISTA3D timed out after {inp.timeout_s}s",
                runtime_seconds=time.time() - t0,
            )
        except subprocess.CalledProcessError as e:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"VISTA3D runner failed (exit {e.returncode})",
                runtime_seconds=time.time() - t0,
            )

        # Collect output NIfTI files written by the pipeline
        organs_segmented = [
            f.replace(".nii.gz", "")
            for f in os.listdir(output_dir)
            if f.endswith(".nii.gz")
        ]

        return ToolOutput(
            tool_name=self.name,
            case_path=inp.case_path,
            seg_dir=output_dir,
            organs_segmented=organs_segmented,
            statistics={},
            success=len(organs_segmented) > 0,
            error="" if organs_segmented else "No segmentation outputs produced.",
            runtime_seconds=time.time() - t0,
        )

    def _dry_run(self, inp: ToolInput, output_dir: str, t0: float) -> ToolOutput:
        import random
        simulated_organs = inp.target_organs or [
            "liver", "spleen", "kidney_right", "kidney_left", "pancreas",
            "gallbladder", "aorta", "heart",
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
