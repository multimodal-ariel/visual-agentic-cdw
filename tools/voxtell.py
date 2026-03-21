"""
VoxTell Tool Wrapper
======================
Free-text-prompted universal 3D segmentation.
Trained on 158 public datasets. Prompt encoder: Qwen3-Embedding-4B.
Images must be in RAS orientation. PyTorch < 2.9 required.

CLI: voxtell-predict -i <input> -o <output_dir> -m <model_dir>
                     -p <organ1> <organ2> ... --device cuda --gpu <id>

Output: per-organ NIfTI files named <organ>.nii.gz in output_dir.

MIC-DKFZ, 2024
https://github.com/MIC-DKFZ/VoxTell
Version verified: 0.1.0
"""

import os
import subprocess
import time
from typing import List

from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput


class VoxTellTool(BaseSegmentationTool):

    name = "VoxTell"
    conda_env = "cdw_voxtell"
    supported_modalities = ["CT", "MRI", "PET_CT"]
    supported_anatomies = ["all"]
    supports_2d = False
    supports_3d = True

    def __init__(self, dry_run: bool = False, model_dir: str = None):
        """
        Args:
            dry_run: If True, return simulated outputs without running VoxTell.
            model_dir: Path to VoxTell model directory (plans.json + fold_0/).
                       Falls back to VOXTELL_MODEL_DIR env var, then
                       <pipeline_root>/checkpoints/VoxTell/voxtell_v1.1/.
        """
        self.dry_run = dry_run
        if model_dir:
            self.model_dir = model_dir
        elif os.environ.get("VOXTELL_MODEL_DIR"):
            self.model_dir = os.environ["VOXTELL_MODEL_DIR"]
        else:
            pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_dir = os.path.join(pipeline_root, "checkpoints", "VoxTell", "voxtell_v1.1")

    def run(self, inp: ToolInput) -> ToolOutput:
        t0 = time.time()
        output_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_voxtell")

        if self.dry_run:
            return self._dry_run(inp, output_dir, t0)

        if not os.path.isfile(inp.image_path):
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"Image not found: {inp.image_path}",
                runtime_seconds=time.time() - t0,
            )

        if not inp.target_organs:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error="No target_organs provided. VoxTell requires text prompts from the LLM planner.",
                runtime_seconds=time.time() - t0,
            )

        if not self.model_dir:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error="VoxTell model_dir not set. Pass model_dir= or set VOXTELL_MODEL_DIR env var.",
                runtime_seconds=time.time() - t0,
            )

        os.makedirs(output_dir, exist_ok=True)

        # Device: "gpu:0" → device=cuda, gpu=0
        if "gpu:" in inp.device:
            device_str = "cuda"
            gpu_id = inp.device.split(":")[1]
        elif inp.device == "cpu":
            device_str = "cpu"
            gpu_id = "0"
        else:
            device_str = "cuda"
            gpu_id = "0"

        # voxtell-predict accepts all prompts in one call via -p arg (nargs='+')
        cmd = [
            "voxtell-predict",
            "-i", inp.image_path,
            "-o", output_dir,
            "-m", self.model_dir,
            "-p", *inp.target_organs,
            "--device", device_str,
            "--gpu", gpu_id,
        ]

        try:
            self._run_in_env(cmd)
        except subprocess.CalledProcessError as e:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"VoxTell failed (exit {e.returncode})",
                runtime_seconds=time.time() - t0,
            )

        # VoxTell names outputs {input_stem}_{organ}.nii.gz; rename to canonical <organ>.nii.gz
        input_stem = os.path.basename(inp.image_path).replace(".nii.gz", "")
        organs_segmented = []
        for organ in inp.target_organs:
            src = os.path.join(output_dir, f"{input_stem}_{organ}.nii.gz")
            dst = os.path.join(output_dir, f"{organ}.nii.gz")
            if os.path.isfile(src):
                os.rename(src, dst)
                organs_segmented.append(organ)
            elif os.path.isfile(dst):
                organs_segmented.append(organ)

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
        simulated_organs = inp.target_organs or ["liver", "spleen", "kidney_left"]
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
