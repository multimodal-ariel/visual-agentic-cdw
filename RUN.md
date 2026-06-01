# Running the Pipeline

Operational notes for running *You See Voxels, I See Features* on a clinical GPU server. For full installation details, see `docs/installation.md` and `docs/deployment.md`.

All commands assume repo root:

```bash
cd /path/to/visual-agentic-cdw
```

## 1. One-Time Setup

Create tool environments and download checkpoints:

```bash
export PYTHONNOUSERSITE=1
git submodule update --init --recursive
bash scripts/setup_envs.sh
bash scripts/download_checkpoints.sh
```

Set deployment paths in `config/constants.py`:

```python
DICOM_ROOT = "/data/RAD"
SEGMENTATION_ROOT = "/data/soumitri/segmentations_3d"
```

Model weights, data, logs, and generated outputs should remain untracked by git.

## 2. Input Filelist

The batch runner accepts either a flat JSON list:

```json
[
  "/data/RAD/RHEUM/PT001/20240101/CT_ABDOMEN_PELVIS/AXIAL",
  "/data/RAD/LUPUS/PT002/20240201/MRI_BRAIN/T1_AX"
]
```

or:

```json
{
  "cases": [
    "/data/RAD/RHEUM/PT001/20240101/CT_ABDOMEN_PELVIS/AXIAL"
  ]
}
```

Raw DICOM paths under `DICOM_ROOT` are remapped to matching output paths under `SEGMENTATION_ROOT`. If `image_nifti.nii.gz` is missing, the runner attempts DICOM-to-NIfTI materialization before segmentation.

Recommended full-corpus filelist:

```text
new_data_paths/3d_volume_paths_interleaved.json
```

## 3. Segmentation Batch

Run without LLMs for deterministic, rule-based metadata/tool selection:

```bash
conda run -n cdw_radiomics python -m orchestrator.batch_runner \
  --filelist new_data_paths/3d_volume_paths_interleaved.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers 8 \
  --no-llm \
  --progress-every 50 \
  --log-level WARNING
```

For a lighter run:

```bash
conda run -n cdw_radiomics python -m orchestrator.batch_runner \
  --filelist new_data_paths/3d_volume_paths_interleaved.json \
  --gpus 0,1,2,3 \
  --workers 4 \
  --no-llm \
  --progress-every 25 \
  --log-level WARNING
```

Useful options:

- `--retry-failed`: re-run cases currently marked failed.
- `--tools TotalSegmentator_CT,VISTA3D`: restrict tool selection.
- `--tool-timeout 900`: mark a tool failed if it exceeds the timeout.
- `--progress-every 0`: disable progress snapshots.

State and results:

```text
logs/pipeline_state_v3.json
logs/pipeline_results_v3.csv
```

Completed cases are skipped automatically on rerun. Stale `running` cases are retried.

## 4. QC and Radiomics

Run after segmentation has completed:

```bash
conda run -n cdw_radiomics python orchestrator/batch_qc_radiomics_runner.py \
  --segmentation-state logs/pipeline_state_v3.json \
  --segmentation-root /data/soumitri/segmentations_3d \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers 8 \
  --progress-every 50 \
  --log-level WARNING
```

Run concurrently with an active segmentation batch only if GPUs are available:

```bash
conda run -n cdw_radiomics python orchestrator/batch_qc_radiomics_runner.py \
  --segmentation-state logs/pipeline_state_v3.json \
  --segmentation-root /data/soumitri/segmentations_3d \
  --gpus 5 \
  --workers 1 \
  --watch \
  --progress-every 25 \
  --log-level WARNING
```

Use `--watch` when segmentation is still producing new completed cases. Omit it for a one-time pass.

Per-case QC outputs:

```text
segmentations_filtered_GeometricQC/
segmentations_filtered_MedSegQC/
qc_scores.csv
radiomics_features.csv
selection_summary.json
```

GeometricQC selects the largest non-empty anatomy-relevant mask per organ. MedSegQC is only applied to supported CT/MRI abdominal organs: `liver`, `spleen`, `kidney_left`, and `kidney_right`. PET/CT still receives GeometricQC and radiomics, but MedSegQC is skipped.

## 5. Monitoring

Watch GPU memory:

```bash
watch -n 5 '
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
echo
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d " "); do
  ps -o user,pid,ppid,etime,cmd -p "$p" --no-headers
done
'
```

Map high-VRAM processes to parents:

```bash
pstree -aps <PID>
```

SynthSeg can consume most of an L40S per process. If many head cases are active, reduce segmentation concurrency with fewer `--workers` or fewer GPUs.

## 6. Validation

QC/radiomics self-test:

```bash
conda run -n cdw_radiomics python orchestrator/batch_qc_radiomics_runner.py --self-test
```

Run tests:

```bash
conda run -n cdw_radiomics python -m pytest
```

`pytest.ini` is intentionally kept so pytest only discovers `tests/` and avoids third-party test suites under `external/`.

## 7. Common Recovery

Retry failed segmentation cases:

```bash
conda run -n cdw_radiomics python -m orchestrator.batch_runner \
  --filelist new_data_paths/3d_volume_paths_interleaved.json \
  --gpus 0,1,2,3 \
  --workers 4 \
  --no-llm \
  --retry-failed
```

Re-run QC/radiomics even if outputs already exist:

```bash
conda run -n cdw_radiomics python orchestrator/batch_qc_radiomics_runner.py \
  --segmentation-state logs/pipeline_state_v3.json \
  --segmentation-root /data/soumitri/segmentations_3d \
  --gpus 0,1 \
  --workers 2 \
  --force
```

Stop tracking generated outputs while keeping local files:

```bash
git rm --cached -r .logs stats visualizations
```

Then commit the cleanup with `.gitignore`.
