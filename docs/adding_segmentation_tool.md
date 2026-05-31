# Adding a Segmentation Tool

This repo treats every model as a wrapper around a common on-disk contract:

```text
<case_path>/
  image_nifti.nii.gz
  segmentations_<tool>/
    <organ>.nii.gz
    manifest.json
```

Every mask must be binary `uint8` and must share the shape, affine, and voxel
grid of `image_nifti.nii.gz`. If an upstream model writes a different geometry,
the wrapper must resample or convert before exposing masks to the rest of the
pipeline.

## Where Code Goes

- Put third-party source checkouts under `external/<ToolName>/`.
- Put the CDW wrapper under `tools/<tool_name>.py`.
- Put helper subprocess runners under `tools/runners/` only when a Python API
  needs isolation from the main process.
- Add installation notes in `external/<ToolName>_SETUP.md` or
  `external/<ToolName>/README_CDW.md` when the tool has non-trivial dependencies
  or weights.

## Wrapper Checklist

1. Subclass `BaseSegmentationTool` from `tools/base_tool.py`.
2. Set class attributes:
   - `name`
   - `conda_env`
   - `supported_modalities`
   - `supported_anatomies`
   - `supports_2d`
   - `supports_3d`
3. Implement `run(self, inp: ToolInput) -> ToolOutput`.
4. Use `inp.output_dir` when supplied, otherwise write to the standard
   `segmentations_<tool>` directory.
5. Use `self._parse_gpu_id(inp.device)` and `self._run_in_env(...)` for
   subprocess tools so multi-GPU batch runs pin each worker correctly.
6. Catch `subprocess.TimeoutExpired` and `subprocess.CalledProcessError`, then
   return `ToolOutput(success=False, error=...)`.
7. Return `ToolOutput(success=True, organs_segmented=[...])` only when at least
   one valid mask was written.
8. Implement `_dry_run(...)` when practical; it makes smoke tests independent of
   private model weights.

## Format Normalization

Use `processing/format_utils.py` helpers when possible:

- `split_multilabel_nifti(...)` for same-grid multilabel model outputs.
- `save_organ_mask(...)` for binary per-organ writes.
- `build_normalized_mask_index(...)` for locating masks with inconsistent tool
  naming.

If the upstream model changes geometry, resample before splitting. For label
maps, always use nearest-neighbour interpolation. For example, SynthSeg outputs
1 mm isotropic labels, so `tools/synthseg.py` resamples the multilabel NIfTI back
to `image_nifti.nii.gz` before writing per-structure masks.

## Registry Checklist

Update `config/constants.py`:

```python
TOOL_OUTPUT_DIRS["ToolName"] = "segmentations_tool"
TOOL_CONDA_ENVS["ToolName"] = "cdw_tool"
```

Update `config/tool_registry.json` in the appropriate category:

- `fixed_class_tools`: predetermined label set.
- `text_promptable_tools`: free-text organ prompts.
- `label_prompted_tools`: integer-label or preset-prompt models.
- `deferred_tools`: known tools not active in batch selection.

Update `tools/__init__.py` so imports and quick smoke tests can discover the
wrapper.

## Testing

At minimum:

```bash
python3 -m py_compile tools/<tool_name>.py
python - <<'PY'
from tools.<tool_name> import ToolClass
print(ToolClass(dry_run=True))
PY
```

If the tool has `_dry_run`, create a tiny synthetic `image_nifti.nii.gz` and
verify that the wrapper writes masks in the expected output directory. Real model
execution should be tested on the clinical server environment where the conda
env, CUDA, and private paths exist.
