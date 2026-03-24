# CDW Agentic Pipeline — Deployment (8×46 GB L40S)

Condensed checklist to bring up the full pipeline on a fresh multi-GPU node. Assumes Ubuntu 22.04+, CUDA driver ≥ 12.4 (tested on Blackwell with 12.9), and Miniconda on PATH.

## Quick checklist
- [ ] CUDA driver visible: `nvidia-smi` (8×L40S, 46 GB each)
- [ ] Repo cloned **with submodules**
- [ ] Conda envs created via `scripts/setup_envs.sh`
- [ ] Checkpoints pulled via `scripts/download_checkpoints.sh` (HF login for MedGemma)
- [ ] Paths set in `config/constants.py` (DICOM/SEG roots, filelists)
- [ ] Optional model dir env vars set (VoxTell, VISTA3D, TextMedSeg3D)
- [ ] Smoke tests run (`scripts/run_dummy_test.py`, `tests/run_e2e.py`)

## 1) Clone with submodules
```bash
git clone <repo_url> agentic-cdw
cd agentic-cdw
git submodule update --init --recursive
```
All third-party code lives in `external/`. Do not pip-install upstream packages; always install from the local sources referenced by the setup script.

## 2) Base system sanity
- NVIDIA driver ≥ 12.x (`nvidia-smi`)
- Disk: ~80 GB free for checkpoints; additional space for outputs
- Network: outbound HTTPS to HuggingFace for weights
- Optional: `conda install -c conda-forge git-lfs` if you plan to fetch any LFS assets (none required by default)

## 3) Create conda envs
Run from repo root:
```bash
bash scripts/setup_envs.sh        # all envs
# or a single one, e.g.:
bash scripts/setup_envs.sh --env cdw_totalseg
```
Notes:
- Uses PyTorch cu128 wheels and applies the Blackwell runtime fix (`nvidia-cusparselt-cu12`, `nvidia-nvjitlink-cu12`).
- nnU-Net and SAT are patched for `weights_only=False` (PyTorch ≥2.6 compatibility).
- BiomedParse3D env is intentionally skipped; VoxTell + TextMedSeg3D cover 3D text-prompting.

## 4) Download checkpoints
Install the CLI once:
```bash
pip install "huggingface_hub[cli]"
huggingface-cli login   # required for MedGemma-27B (gated)
```
Then pull all weights (LLMs + segmentation):
```bash
bash scripts/download_checkpoints.sh
```
Outputs land in `checkpoints/`. TotalSegmentator, MRSegmentator, MRISegmenter, and VIBESegmentator download their own weights on first run.

## 5) Configure paths and model dirs
Edit `config/constants.py` for your cluster paths:
- `DICOM_ROOT`, `SEGMENTATION_ROOT`: where inputs and outputs live
- `FILELIST_*`: JSON manifests pointing at your cases
- Optional defaults: `PLANNER_DEVICE`, `SEGMENTATION_DEVICE`, worker counts

Common env vars (set in shell or a `.env` you source before running):
```bash
export VOXTELL_MODEL_DIR="$REPO/checkpoints/VoxTell"
export NVSEG_DIR="$REPO/checkpoints/NVSegmentCTMR"
export TEXTMEDSEG3D_DIR="$REPO/checkpoints/TextMedSeg3D"
```
Adjust if checkpoints live elsewhere.

## 6) Validate GPUs per env (one-time spot check)
```bash
conda run -n cdw_totalseg python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```
Repeat for another env if desired (e.g., `cdw_nvseg`).

## 7) Smoke tests
From repo root:
```bash
# Tool wrappers on dummy data (writes to dummy_outputs/)
python scripts/run_dummy_test.py --tool totalseg_ct --device gpu:0
python scripts/run_dummy_test.py --tool voxtell --device gpu:0 --cases 0004

# Planner/string + image modes
dhp="conda run -n cdw_radiomics python"
$dhp tests/run_e2e.py --mode string
$dhp tests/run_e2e.py --mode image
```
Expect all dummy cases to pass; see `tests/results/e2e_report.json` for summary.

## 8) Running production batches (8×L40S)
- Use `orchestrator.batch_runner` and enumerate GPUs, e.g.:
```bash
python -m orchestrator.batch_runner \
  --filelist /data/.../ct_axial_cases.json \
  --gpus 0,1,2,3,4,5,6,7 --workers 2 \
  --segmentation_device gpu            # TotalSeg uses "gpu" not cuda:0
```
- The runner distributes cases round-robin across GPUs; set `--workers` to 2–3 per GPU for TotalSegmentator. NVSegment/VoxTell/SAT are heavier—start with 1–2.
- LLMs: Qwen3-8B fits on one L40S; Qwen3-32B or MedGemma-27B can use tensor parallel via vLLM if you prefer a server: `conda run -n cdw_llm vllm serve Qwen/Qwen3-32B --tensor-parallel-size 2` and point `VLLM_ENDPOINT` in `config/constants.py`.

## 9) Troubleshooting shortcuts
- Missing `libcusparseLt.so.0` or `libnvJitLink.so`: install the Blackwell packages (already in `setup_envs.sh`).
- nnU-Net `weights_only` errors: rerun `scripts/setup_envs.sh --env cdw_totalseg` (applies patch), or manually set `weights_only=False` in `nnunetv2/inference/predict_from_raw_data.py`.
- SAT torch.load errors: ensure `scripts/setup_envs.sh --env cdw_textmedseg` was run so the `weights_only=False` patch is applied.
- If dummy/e2e tests fail, check env activation (`conda env list | grep cdw`) and that checkpoints are present under `checkpoints/`.

## 10) What to back up / copy
- Code + submodules: git clone + `git submodule update --init --recursive`
- Checkpoints directory if you want to avoid re-downloading (~80 GB)
- `config/constants.py` and any custom filelists under `data_paths/`
- Logs/QC/radiomics outputs under `logs/` and your segmentation root
