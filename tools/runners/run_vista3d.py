"""
VISTA3D Runner Script
======================
Thin CLI wrapper around the NVSegmentCTMR / VISTA3D Python API.
Called by vista3d.py via conda run (inside cdw_nvseg env).

Two modes:
  label_prompt  — provide specific organ integer indices
  modality      — provide "CT_BODY", "MRI_BODY", or "MRI_BRAIN"

Usage:
    # Label-prompt mode
    python run_vista3d.py
        --input  /path/to/image_nifti.nii.gz
        --output /path/to/segmentations_vista3d/
        --model_path /path/to/checkpoints/NVSegmentCTMR/
        --label_prompt 1 3 5 10
        --device cuda:0

    # Modality-preset mode
    python run_vista3d.py
        --input  /path/to/image_nifti.nii.gz
        --output /path/to/segmentations_vista3d/
        --model_path /path/to/checkpoints/NVSegmentCTMR/
        --modality CT_BODY
        --device cuda:0

Output: per-organ NIfTI files named by label name (from metadata.json channel_def)
        saved to --output directory as <organ_name>.nii.gz.
"""

import argparse
import os
import sys
import json
import tempfile


def _normalize_name(label_name: str) -> str:
    """Convert 'left kidney' → 'left_kidney', strip deprecated notes."""
    name = label_name.split(" (deprecated)")[0].strip()
    return name.lower().replace(" ", "_").replace("-", "_")


def _build_index_to_name(model_path: str) -> dict:
    """
    Load label index → canonical name from metadata.json channel_def.
    Falls back to empty dict if metadata.json not found.
    """
    meta_path = os.path.join(model_path, "metadata.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    channel_def = (
        meta.get("network_data_format", {})
            .get("outputs", {})
            .get("pred", {})
            .get("channel_def", {})
    )
    return {int(k): _normalize_name(v) for k, v in channel_def.items() if int(k) > 0}


def main():
    parser = argparse.ArgumentParser(description="VISTA3D / NVSegmentCTMR inference")
    parser.add_argument("--input", required=True, help="Path to input NIfTI image")
    parser.add_argument("--output", required=True, help="Output directory for per-organ masks")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to NVSegmentCTMR repo root (contains metadata.json and vista3d_pretrained_model/)",
    )
    parser.add_argument(
        "--label_prompt",
        nargs="+",
        type=int,
        default=None,
        help="Integer label indices to segment (label-prompt mode)",
    )
    parser.add_argument(
        "--modality",
        default=None,
        choices=["CT_BODY", "MRI_BODY", "MRI_BRAIN"],
        help="Modality preset for whole-body segmentation (modality mode)",
    )
    parser.add_argument("--device", default="cuda:0", help="PyTorch device string")
    args = parser.parse_args()

    if args.label_prompt is None and args.modality is None:
        print("[ERROR] Provide either --label_prompt or --modality.", flush=True)
        sys.exit(1)

    if not os.path.isdir(args.model_path):
        print(f"[ERROR] model_path not found: {args.model_path}", flush=True)
        sys.exit(1)

    # NVSegmentCTMR Python modules live in model_path
    sys.path.insert(0, args.model_path)

    import numpy as np
    import nibabel as nib
    import torch
    from vista3d_config import VISTA3DConfig
    from vista3d_model import VISTA3DModel
    from vista3d_pipeline import VISTA3DPipeline

    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device)
    pretrained_path = os.path.join(args.model_path, "vista3d_pretrained_model")
    model_pt = os.path.join(pretrained_path, "model.pt")

    print(f"[VISTA3D] Loading model weights from {model_pt}", flush=True)
    config = VISTA3DConfig()
    model = VISTA3DModel(config)
    state_dict = torch.load(model_pt, map_location="cpu", weights_only=False)
    model.network.load_state_dict(state_dict)
    pipeline = VISTA3DPipeline(model, device=device)

    # Build input spec
    inp_spec = {"image": args.input}
    if args.label_prompt is not None:
        inp_spec["label_prompt"] = args.label_prompt
    else:
        inp_spec["modality"] = args.modality

    # Run inference into a temp directory; pipeline creates {tmp}/{stem}/{stem}_seg.nii.gz
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"[VISTA3D] Running inference: {inp_spec}", flush=True)
        pipeline([inp_spec], output_dir=tmp_dir)

        # Find the multilabel output
        input_stem = os.path.basename(args.input).replace(".nii.gz", "")
        multilabel_path = os.path.join(tmp_dir, input_stem, f"{input_stem}_seg.nii.gz")

        if not os.path.isfile(multilabel_path):
            print(f"[VISTA3D][ERROR] Multilabel output not found at {multilabel_path}", flush=True)
            sys.exit(1)

        # Load multilabel seg and split into per-organ binary NIfTIs
        seg_img = nib.load(multilabel_path)
        seg_data = np.round(seg_img.get_fdata()).astype(np.int32)
        affine = seg_img.affine
        header = seg_img.header

        # Build label index → name mapping from metadata.json
        index_to_name = _build_index_to_name(args.model_path)

        # Determine which label indices to write
        if args.label_prompt is not None:
            label_indices = args.label_prompt
        else:
            label_indices = [int(v) for v in np.unique(seg_data) if v > 0]

        written = []
        for idx in label_indices:
            organ_name = index_to_name.get(idx, f"label_{idx:03d}")
            mask = (seg_data == idx).astype(np.uint8)
            if mask.sum() == 0:
                continue
            out_nii = nib.Nifti1Image(mask, affine, header)
            out_nii.set_data_dtype(np.uint8)
            out_path = os.path.join(args.output, f"{organ_name}.nii.gz")
            nib.save(out_nii, out_path)
            written.append(organ_name)

    print(f"[VISTA3D] Saved {len(written)} organ masks to {args.output}", flush=True)


if __name__ == "__main__":
    main()
