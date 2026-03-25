"""
================================================================================
Base Segmentation Tool Interface
================================================================================
All segmentation tool wrappers inherit from BaseSegmentationTool.
Uniform ToolInput/ToolOutput ensures the orchestrator doesn't care
which tool it's calling — they all look the same.

Adding a new tool:
  1. Create tools/my_new_tool.py
  2. Subclass BaseSegmentationTool
  3. Implement run()
  4. Declare name, supported_modalities, supported_anatomies
  5. Add entry to config/tool_registry.json
  Done. Zero changes to orchestrator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
import subprocess
import time

import numpy as np


@dataclass
class ToolInput:
    """Standardized input to any segmentation tool."""
    image_path: str          # absolute path to image_nifti.nii.gz
    case_path: str           # parent directory of the image
    modality: str            # CT, MRI, PET_CT, etc.
    anatomy: str             # head, chest, abdomen, abdomen_pelvis, etc.
    target_organs: List[str] # organs expected in this scan (from LLM planner)
    output_dir: str = ""     # where to write segmentations (tool-specific subdir)
    device: str = "gpu:0"   # GPU device string


@dataclass
class ToolOutput:
    """
    Standardized output from any segmentation tool.

    On-disk contract (always holds after a successful run):
      <seg_dir>/
          <organ>.nii.gz   — binary uint8, same affine + shape as image_nifti.nii.gz
          manifest.json    — written automatically by _timed_run()
    """
    tool_name: str
    case_path: str
    seg_dir: str                          # directory where per-organ masks were saved
    organs_segmented: List[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)  # native tool stats (optional)
    success: bool = True
    error: str = ""
    runtime_seconds: float = 0.0
    timestamp: str = ""                   # ISO-8601, set by _timed_run()


class BaseSegmentationTool(ABC):
    """
    Abstract base class for all segmentation tools.

    Subclasses must:
      - Set class attributes: name, supported_modalities, supported_anatomies
      - Implement run(ToolInput) -> ToolOutput

    Subclasses do NOT need to:
      - Write manifest.json (handled by _timed_run)
      - Track runtime or timestamp (handled by _timed_run)
    """

    name: str = ""
    conda_env: str = ""          # conda environment name for subprocess isolation
    supported_modalities: List[str] = []
    supported_anatomies: List[str] = []  # ["all"] = anatomy-agnostic
    supports_2d: bool = False
    supports_3d: bool = True

    def is_applicable(self, modality: str, anatomy: str, shape: str) -> bool:
        """
        Return True if this tool can process the given case.
        Called by the planner to filter the active tool list.
        """
        mod_ok = modality.upper() in [m.upper() for m in self.supported_modalities]
        anat_ok = (
            "all" in [a.lower() for a in self.supported_anatomies]
            or anatomy.lower() in [a.lower() for a in self.supported_anatomies]
        )
        dim_ok = (shape == "2D" and self.supports_2d) or (
            shape in ("3D", "4D") and self.supports_3d
        )
        return mod_ok and anat_ok and dim_ok

    @abstractmethod
    def run(self, inp: ToolInput) -> ToolOutput:
        """
        Execute segmentation and write per-organ NIfTI masks to seg_dir.

        Contract:
          - On success: seg_dir contains <organ>.nii.gz for each segmented organ.
            Each mask is binary uint8 with the same affine/shape as inp.image_path.
          - On failure: return ToolOutput(success=False, error=<reason>).
          - Do NOT write manifest.json — _timed_run() handles that.
        """

    def _run_in_env(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run cmd inside this tool's conda environment via `conda run`.

        cmd is a flat list of strings, e.g.:
            ["TotalSegmentator", "-i", "/path/in.nii.gz", "-o", "/path/out/"]
            ["python", "/path/runner.py", "--input", "...", "--output", "..."]
            ["torchrun", "--nproc_per_node=1", "inference.py", ...]

        Raises subprocess.CalledProcessError on non-zero exit; callers should
        catch it and return a ToolOutput(success=False, error=...).
        """
        if not self.conda_env:
            raise RuntimeError(
                f"{self.__class__.__name__}: conda_env class attribute is not set."
            )
        full_cmd = ["/home/soumitri/env/miniconda3/bin/conda", "run", "-n", self.conda_env] + cmd
        return subprocess.run(
            full_cmd,
            check=True,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def postprocess(self, mask: np.ndarray, organ: str, config: Dict) -> np.ndarray:
        """Apply organ-specific postprocessing chain to a binary mask array."""
        from processing import postprocessing as postproc
        if config.get("remove_small", True):
            mask = postproc.remove_small_components(
                mask, min_volume_voxels=config.get("min_vol", 100)
            )
        if config.get("closing", False):
            mask = postproc.morphological_close(
                mask, radius=config.get("closing_radius", 2)
            )
        if config.get("fill_holes", True):
            mask = postproc.fill_holes(mask)
        if config.get("keep_largest", True):
            mask = postproc.keep_largest_component(mask)
        if config.get("smooth", False):
            mask = postproc.smooth_boundary(mask, sigma=config.get("smooth_sigma", 1.0))
        return mask

    def _write_manifest(self, output: ToolOutput) -> None:
        """
        Write manifest.json to output.seg_dir.

        Called automatically by _timed_run() — never call from run().
        Creates seg_dir if it doesn't exist (needed for failed runs).
        """
        if not output.seg_dir:
            return
        os.makedirs(output.seg_dir, exist_ok=True)
        manifest = {
            "tool_name": output.tool_name,
            "success": output.success,
            "organs_segmented": output.organs_segmented,
            "runtime_seconds": round(output.runtime_seconds, 2),
            "timestamp": output.timestamp,
            "error": output.error,
            "native_stats": output.statistics,
        }
        with open(os.path.join(output.seg_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    def _timed_run(self, inp: ToolInput) -> ToolOutput:
        """
        Time run(), stamp the output, write manifest. Use this instead of run().
        """
        t0 = time.time()
        output = self.run(inp)
        output.runtime_seconds = time.time() - t0
        output.timestamp = datetime.now().isoformat(timespec="seconds")
        self._write_manifest(output)
        return output

    def __repr__(self) -> str:
        return (
            f"{self.name}(modalities={self.supported_modalities}, "
            f"anatomies={self.supported_anatomies}, "
            f"2D={self.supports_2d}, 3D={self.supports_3d})"
        )
