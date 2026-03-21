"""
BiomedParse3D Tool Wrapper
============================
BiomedParse v2. Text-prompted 3D volumetric segmentation with BoltzFormer.
200+ anatomies across CT, MRI, PET, Ultrasound, and 3D Microscopy.
Processes volumes natively in 3D (not slice-by-slice).

Python API only — invoked via tools/runners/run_biomedparse3d.py
inside the cdw_biomedparse3d conda environment.

Microsoft Research, 2024
https://github.com/microsoft/BiomedParse
Version verified: v2 (Oct 2025)
"""

import os
import subprocess
import time
from typing import List

from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput


_RUNNER = os.path.join(os.path.dirname(__file__), "runners", "run_biomedparse3d.py")


class BiomedParse3DTool(BaseSegmentationTool):

    name = "BiomedParse3D"
    conda_env = "cdw_biomedparse3d"
    supported_modalities = ["CT", "MRI", "PET_CT", "US"]
    supported_anatomies = ["all"]
    supports_2d = False
    supports_3d = True

    def __init__(
        self,
        dry_run: bool = False,
        weights_path: str = None,
        biomedparse_dir: str = None,
    ):
        """
        Args:
            dry_run: If True, return simulated outputs.
            weights_path: Path to biomedparse_3D_AllData_MultiView_edge.ckpt.
                          Falls back to BIOMEDPARSE3D_WEIGHTS env var.
            biomedparse_dir: Path to external/BiomedParse3D repo root.
                             Falls back to BIOMEDPARSE3D_DIR env var, then
                             <pipeline_root>/external/BiomedParse3D.
        """
        self.dry_run = dry_run
        self.weights_path = weights_path or os.environ.get("BIOMEDPARSE3D_WEIGHTS", "")

        if biomedparse_dir:
            self.biomedparse_dir = biomedparse_dir
        elif os.environ.get("BIOMEDPARSE3D_DIR"):
            self.biomedparse_dir = os.environ["BIOMEDPARSE3D_DIR"]
        else:
            pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.biomedparse_dir = os.path.join(pipeline_root, "external", "BiomedParse3D")

    def run(self, inp: ToolInput) -> ToolOutput:
        t0 = time.time()
        output_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_biomedparse3d")

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
                error="No target_organs provided. BiomedParse3D requires text prompts from the LLM planner.",
                runtime_seconds=time.time() - t0,
            )

        if not self.weights_path:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error="BiomedParse3D weights not set. Pass weights_path= or set BIOMEDPARSE3D_WEIGHTS env var.",
                runtime_seconds=time.time() - t0,
            )

        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python", _RUNNER,
            "--input", inp.image_path,
            "--output", output_dir,
            "--weights", self.weights_path,
            "--biomedparse_dir", self.biomedparse_dir,
            "--prompts", *inp.target_organs,
        ]

        try:
            self._run_in_env(cmd)
        except subprocess.CalledProcessError as e:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error=f"BiomedParse3D runner failed (exit {e.returncode})",
                runtime_seconds=time.time() - t0,
            )

        organs_segmented = [
            organ for organ in inp.target_organs
            if os.path.isfile(os.path.join(output_dir, f"{organ}.nii.gz"))
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
        simulated_organs = inp.target_organs or ["liver", "tumor"]
        simulated_stats = {o: {"volume_mm3": random.uniform(100, 50000)} for o in simulated_organs}
        return ToolOutput(
            tool_name=self.name,
            case_path=inp.case_path,
            seg_dir=output_dir,
            organs_segmented=simulated_organs,
            statistics=simulated_stats,
            success=True,
            runtime_seconds=time.time() - t0,
        )
