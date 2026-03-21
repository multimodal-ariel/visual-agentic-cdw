"""
TextMedSeg3D Tool Wrapper (SAT — Segment Anything Textual)
===========================================================
Text-prompted 3D segmentation trained on 72 public datasets, 497 classes.
Covers 8 body regions. Uses torchrun (DDP) for inference.

Input: JSONL file per the SAT spec:
    {"image": "/path/img.nii.gz", "label": ["liver","kidney"],
     "modality": "ct", "dataset": "AbdomenCT1K"}

Output: per-organ NIfTI files at
    {rcd_dir}/{dataset_name}/seg_{image_stem}/{label}.nii.gz

Wrapper collects these files and moves them to the canonical seg_dir.

CLI (torchrun):
    torchrun --nproc_per_node=1 inference.py
        --datasets_jsonl <jsonl>
        --vision_backbone UNET-L
        --checkpoint <ckpt>
        --text_encoder ours
        --text_encoder_checkpoint <text_ckpt>
        --rcd_dir <rcd_dir>

SAT: Segment Anything in Medical Images with Text, 2024
https://github.com/zhaoziheng/SAT
Version verified: latest (no version pin in repo)
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import List

from tools.base_tool import BaseSegmentationTool, ToolInput, ToolOutput


class TextMedSeg3DTool(BaseSegmentationTool):

    name = "TextMedSeg3D"
    conda_env = "cdw_textmedseg"
    supported_modalities = ["CT", "MRI", "PET_CT"]
    supported_anatomies = ["all"]
    supports_2d = False
    supports_3d = True

    # SAT dataset name to use for single-case inference
    _DATASET_NAME = "AbdomenCT1K"

    def __init__(
        self,
        dry_run: bool = False,
        textmedseg_dir: str = None,
        checkpoint: str = None,
        text_encoder_checkpoint: str = None,
        vision_backbone: str = "UNET",
        text_encoder: str = "ours",
    ):
        """
        Args:
            dry_run: If True, return simulated outputs.
            textmedseg_dir: Path to external/TextMedSeg3D repo root.
                            Falls back to TEXTMEDSEG_DIR env var, then
                            <pipeline_root>/external/TextMedSeg3D.
            checkpoint: Path to vision backbone checkpoint (.pth).
                        Falls back to TEXTMEDSEG_CHECKPOINT env var, then
                        <pipeline_root>/checkpoints/TextMedSeg3D/Nano/nano_cvpr25_v0.pth.
            text_encoder_checkpoint: Path to text encoder checkpoint (.pth).
                        Falls back to TEXTMEDSEG_TEXT_ENCODER_CKPT env var, then
                        <pipeline_root>/checkpoints/TextMedSeg3D/Nano/text_encoder_cvpr25_v0.pth.
            vision_backbone: Model variant — "UNET" (SAT-Nano) or "UNET-L" (SAT-Pro).
                             Default: "UNET" (Nano — uses nano_cvpr25_v0.pth).
            text_encoder: Text encoder type — "ours" (default).
        """
        self.dry_run = dry_run
        self.vision_backbone = vision_backbone
        self.text_encoder = text_encoder

        if textmedseg_dir:
            self.textmedseg_dir = textmedseg_dir
        elif os.environ.get("TEXTMEDSEG_DIR"):
            self.textmedseg_dir = os.environ["TEXTMEDSEG_DIR"]
        else:
            pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.textmedseg_dir = os.path.join(pipeline_root, "external", "TextMedSeg3D")

        pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_root = os.path.join(pipeline_root, "checkpoints", "TextMedSeg3D")

        # Default checkpoint resolution: Pro > Nano
        # Pro uses UNET-L backbone; Nano uses UNET backbone
        if checkpoint:
            self.checkpoint = checkpoint
        elif os.environ.get("TEXTMEDSEG_CHECKPOINT"):
            self.checkpoint = os.environ["TEXTMEDSEG_CHECKPOINT"]
        elif os.path.isfile(os.path.join(ckpt_root, "Pro", "SAT_Pro.pth")):
            # SAT-Pro available — use it (better quality)
            self.checkpoint = os.path.join(ckpt_root, "Pro", "SAT_Pro.pth")
            if vision_backbone == "UNET":
                self.vision_backbone = "UNET-L"  # Pro uses UNET-L
        else:
            self.checkpoint = os.path.join(ckpt_root, "Nano", "nano.pth")

        if text_encoder_checkpoint:
            self.text_encoder_checkpoint = text_encoder_checkpoint
        elif os.environ.get("TEXTMEDSEG_TEXT_ENCODER_CKPT"):
            self.text_encoder_checkpoint = os.environ["TEXTMEDSEG_TEXT_ENCODER_CKPT"]
        elif os.path.isfile(os.path.join(ckpt_root, "Pro", "text_encoder.pth")) and \
                "SAT_Pro" in self.checkpoint:
            self.text_encoder_checkpoint = os.path.join(ckpt_root, "Pro", "text_encoder.pth")
        else:
            self.text_encoder_checkpoint = os.path.join(ckpt_root, "Nano", "nano_text_encoder.pth")

    def run(self, inp: ToolInput) -> ToolOutput:
        t0 = time.time()
        output_dir = inp.output_dir or os.path.join(inp.case_path, "segmentations_textmedseg3d")

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
                error="No target_organs provided. TextMedSeg3D requires text prompts from the LLM planner.",
                runtime_seconds=time.time() - t0,
            )

        if not self.checkpoint:
            return ToolOutput(
                tool_name=self.name, case_path=inp.case_path,
                seg_dir=output_dir, success=False,
                error="TextMedSeg3D checkpoint not set. Pass checkpoint= or set TEXTMEDSEG_CHECKPOINT.",
                runtime_seconds=time.time() - t0,
            )

        os.makedirs(output_dir, exist_ok=True)

        # Determine modality string (SAT uses lowercase)
        modality_map = {"CT": "ct", "MRI": "mri", "PET_CT": "pet"}
        modality_str = modality_map.get(inp.modality.upper(), inp.modality.lower())

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write JSONL spec for this single case
            jsonl_path = os.path.join(tmp_dir, "case.jsonl")
            sample = {
                "image": inp.image_path,
                "label": inp.target_organs,
                "modality": modality_str,
                "dataset": self._DATASET_NAME,
            }
            with open(jsonl_path, "w") as f:
                f.write(json.dumps(sample) + "\n")

            rcd_dir = os.path.join(tmp_dir, "rcd")
            os.makedirs(rcd_dir, exist_ok=True)

            cmd = [
                "torchrun", "--nproc_per_node=1",
                "inference.py",
                "--datasets_jsonl", jsonl_path,
                "--vision_backbone", self.vision_backbone,
                "--checkpoint", self.checkpoint,
                "--text_encoder", self.text_encoder,
                "--rcd_dir", rcd_dir,
            ]
            if self.text_encoder_checkpoint:
                cmd += ["--text_encoder_checkpoint", self.text_encoder_checkpoint]

            try:
                self._run_in_env(cmd, cwd=self.textmedseg_dir)
            except subprocess.CalledProcessError as e:
                return ToolOutput(
                    tool_name=self.name, case_path=inp.case_path,
                    seg_dir=output_dir, success=False,
                    error=f"TextMedSeg3D failed (exit {e.returncode})",
                    runtime_seconds=time.time() - t0,
                )

            # Collect outputs: {rcd_dir}/{dataset_name}/seg_{image_stem}/{label}.nii.gz
            # sample_id = basename without .nii.gz
            image_stem = os.path.basename(inp.image_path).replace(".nii.gz", "")
            organs_segmented = []
            for organ in inp.target_organs:
                src = os.path.join(rcd_dir, self._DATASET_NAME,
                                   f"seg_{image_stem}", f"{organ}.nii.gz")
                if os.path.isfile(src):
                    dst = os.path.join(output_dir, f"{organ}.nii.gz")
                    shutil.copy2(src, dst)
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
