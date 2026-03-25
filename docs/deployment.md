# CDW Agentic Pipeline — Full Deployment Guide (8×L40S)

This is the comprehensive guide to deploying the entire codebase onto a fresh multi-GPU node. Tested and optimized for 8×46 GB L40S, Ubuntu 22.04+, CUDA driver ≥ 12.4. All setup uses user-land `conda` — **zero sudo/root access required**.

**Estimated time**: ~2-3 hours (environment setup + checkpoint downloads).
**Estimated disk**: ~300 GB (checkpoints ~200 GB, conda envs ~80 GB, data varies).

---

## 0. Pre-Flight Checklist

Run these checks on the target node BEFORE starting setup:

```bash
# 1. GPU visibility — must show 8 GPUs
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 2. CUDA driver version — must be ≥ 12.4
nvidia-smi | head -3

# 3. Conda available on PATH
conda --version         # tested with conda 24.x+

# 4. Disk space — need ≥ 300 GB free
df -h /home/$USER       # or wherever checkpoints/envs will live

# 5. Network access (for HuggingFace downloads, pip packages)
curl -s https://huggingface.co > /dev/null && echo "HF reachable" || echo "HF blocked"

# 6. Confirm no PIP_USER_SITE interference
echo $PYTHONNOUSERSITE  # should be "1" or empty
```

If `PYTHONNOUSERSITE` is not set, add to your shell profile:
```bash
echo 'export PYTHONNOUSERSITE=1' >> ~/.bashrc
source ~/.bashrc
```

---

## 1. Cloning the Repository & Submodules (CRITICAL)

The pipeline depends on 11+ patched third-party repos as git submodules under `external/`. You **must** use recursive clone to get the exact pinned versions.

```bash
git clone https://github.com/multimodal-ariel/visual-agentic-cdw.git agentic-cdw
cd agentic-cdw
```

**Verify submodules populated** (this MUST pass before proceeding):
```bash
# Should list 11+ directories, each non-empty
for d in external/*/; do
    count=$(find "$d" -maxdepth 1 -type f | wc -l)
    echo "$d — $count files"
done
```

If any directory shows "0 files", the submodule init failed. Re-run `git submodule update --init --recursive`.

> **Do NOT pip-install upstream versions** of these tools. Our `setup_envs.sh` installs from the local patched source in `external/`.

---

## 2. Creating Isolated Conda Environments

Each segmentation tool runs in its own isolated conda env to prevent dependency conflicts. The pipeline has **10 environments** (9 tools + 1 radiomics).

```bash
# Prevent user-site packages from interfering (IMPORTANT on shared servers)
export PYTHONNOUSERSITE=1

# Create ALL environments (takes ~30-45 min with fast network)
bash scripts/setup_envs.sh

# Or create a single environment:
bash scripts/setup_envs.sh --env cdw_totalseg
```

### What `setup_envs.sh` does

1. Creates 10 isolated Python 3.10 conda environments (`cdw_totalseg`, `cdw_mrseg`, `cdw_mriseg`, `cdw_nvseg`, `cdw_voxtell`, `cdw_vibeseg`, `cdw_textmedseg`, `cdw_llm`, `cdw_radiomics`, plus a stub for `cdw_biomedparse3d`)
2. Installs PyTorch with CUDA 12.8 wheels (L40S/Blackwell compatible)
3. Installs Blackwell runtime fixes (`nvidia-cusparselt-cu12`, `nvidia-nvjitlink-cu12`) in all GPU envs
4. Builds the patched local submodules from `external/` into their respective environments
5. Applies nnU-Net `weights_only=False` patches for PyTorch 2.6+ compatibility
6. Applies SAT `torch.load` patch for the same reason

### Environment ↔ Tool mapping

| Env | Tool(s) | PyTorch | Notes |
|-----|---------|---------|-------|
| `cdw_totalseg` | TotalSegmentator CT + MR | latest cu128 | nnU-Net patched |
| `cdw_mrseg` | MRSegmentator | latest cu128 | nnU-Net patched |
| `cdw_mriseg` | MRISegmenter | latest cu128 | |
| `cdw_nvseg` | VISTA3D | 2.10.0 cu128 | MONAI 1.5.0, transformers 4.46.3 |
| `cdw_voxtell` | VoxTell | latest cu128 | huggingface-hub pinned <1.0 |
| `cdw_vibeseg` | VIBESegmentator | latest cu128 | nnU-Net patched, not pip-installed (sys.path) |
| `cdw_textmedseg` | TextMedSeg3D / SAT | 2.10.0 cu128 | Custom dynamic-network-architectures fork |
| `cdw_llm` | Qwen3-8B/32B, MedGemma-27B | via vLLM | transformers ≥4.50 |
| `cdw_radiomics` | PyRadiomics | CPU only | SimpleITK, no PyTorch |
| `cdw_biomedparse3d` | (SKIPPED) | — | Stub only; detectron2 build broken on Blackwell |

### Verify environments

```bash
# List all CDW environments
conda env list | grep cdw

# Quick CUDA check in a tool env
conda run -n cdw_totalseg python -c \
    "import torch; print(f'CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()} | GPU 0: {torch.cuda.get_device_name(0)}')"
```

---

## 3. Downloading Model Checkpoints

### 3a. HuggingFace authentication (required for gated MedGemma)

```bash
pip install -U "huggingface_hub[cli]"

# Login with your HF User Access Token
huggingface-cli login

# Verify authentication
huggingface-cli whoami

# IMPORTANT: Accept the MedGemma license at:
# https://huggingface.co/google/medgemma-27b-text-it
# The download will FAIL silently if the license is not accepted.
```

### 3b. Run the checkpoint downloader

```bash
bash scripts/download_checkpoints.sh
```

**What this fetches** (~200 GB total to `checkpoints/`):

| Checkpoint | Size (approx) | Notes |
|-----------|---------------|-------|
| `medgemma-27b-text-it/` | ~55 GB | Gated — requires HF login + license |
| `qwen3-8b/` | ~16 GB | Open-access |
| `qwen3-32b/` | ~64 GB | Open-access |
| `VoxTell/` | ~2 GB | |
| `NVSegmentCTMR/` | ~2 GB | VISTA3D weights |
| `TextMedSeg3D/Pro/` | ~2 GB | SAT-Pro (preferred) |
| `TextMedSeg3D/Nano/` | ~0.5 GB | SAT-Nano (fallback) |
| `BiomedParse2d3d/` | ~2 GB | Not used in pipeline (deferred) |

**Auto-downloading models** (NOT handled by this script — download on first inference):
- TotalSegmentator → `~/.totalsegmentator/` (~1 GB)
- MRSegmentator → `~/.mrsegmentator/` (~1 GB)
- MRISegmenter → `~/.mrisegmenter_weights/` (~1 GB)
- VIBESegmentator → downloaded by nnU-Net runtime (~1 GB)

> **Tip**: If the target server has no internet access after initial setup, run each tool once on a dummy volume to trigger auto-downloads, THEN go offline.

---

## 4. Environment & Path Configuration

### 4a. Edit `config/constants.py`

This is the **only file you must edit** for deployment. Update these paths:

```python
# === CHANGE THESE for your deployment ===
DICOM_ROOT = "/data/RAD"                                     # Raw DICOM/NIfTI source
SEGMENTATION_ROOT = "/data/soumitri/segmentations_3d"        # Output directory
FILELIST_ALL = "/data/soumitri/new_data_paths/3d_scans_list.json"
FILELIST_CT_AXIAL = "/data/soumitri/new_data_paths/ct_axial_cases.json"
```

Everything else (checkpoint paths, tool envs, QC params) is relative to the repo root and should work without changes.

### 4b. Model directory overrides (optional)

If checkpoints are on a different filesystem (e.g., `/mnt/scratch/models`):
```bash
export VOXTELL_MODEL_DIR="/mnt/scratch/models/VoxTell"
export NVSEG_DIR="/mnt/scratch/models/NVSegmentCTMR"
export VIBESEG_DIR="/path/to/VIBESegmentator"  # cloned repo, not checkpoints
```

---

## 5. Verification and Smoke Testing

Run these in order. Each must pass before proceeding.

### 5a. Per-tool sanity (individual tools on dummy data)

```bash
# Test one fast tool first
python scripts/run_dummy_test.py --tool totalseg_ct --device gpu:0

# Test all active tools
python scripts/run_dummy_test.py --tool all
```

### 5b. E2E pipeline validation (no LLM, no real data)

```bash
# String pipeline — tests planning chain only (metadata → tools → organs)
conda run -n cdw_totalseg python tests/run_e2e.py --mode string

# Image pipeline — tests full flow on dummy_outputs (QC + radiomics on real masks)
conda run -n cdw_radiomics python tests/run_e2e.py --mode image
```

**Success criteria**: `tests/results/e2e_report.json` should show 6/6 cases completed with no errors.

### 5c. LLM validation (optional — requires GPU)

```bash
# Test Qwen3-8B on metadata extraction + tool selection
conda run -n cdw_llm python tests/test_llm_dry_run.py --models qwen8b

# Test MedGemma-27B on QC interpretation (needs ~54 GB VRAM → 2 GPUs)
conda run -n cdw_llm python tests/test_llm_dry_run.py --models medgemma
```

---

## 6. Launching Production Batches (8×L40S)

### Basic usage

```bash
# CT cases on all 8 GPUs, 1 worker per GPU (safe for VRAM-heavy tools)
python -m orchestrator.batch_runner \
    --filelist /data/soumitri/new_data_paths/ct_axial_cases.json \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers 8

# With LLM (Qwen3-8B planner + MedGemma-27B clinical)
python -m orchestrator.batch_runner \
    --filelist /data/.../ct_axial_cases.json \
    --gpus 0,1,2,3,4,5,6,7 \
    --workers 6
```

### Resume after crash

The `CaseTracker` persists state to `logs/pipeline_state.json`. On restart, completed cases are skipped automatically:

```bash
# Just re-run the same command — completed cases are skipped
python -m orchestrator.batch_runner \
    --filelist /data/.../ct_axial_cases.json \
    --gpus 0,1,2,3,4,5,6,7 --workers 8

# Re-process only failed cases
python -m orchestrator.batch_runner \
    --filelist /data/.../ct_axial_cases.json --retry-failed
```

### GPU allocation strategy

| Scenario | GPUs | Workers | Notes |
|----------|------|---------|-------|
| Segmentation only (no LLM) | 0-7 | 8 | 1 worker/GPU, safest |
| Seg + Qwen3-8B planner | 0-5 (seg), 6 (Qwen3) | 6 | Reserve 1 GPU for LLM |
| Seg + Qwen3-8B + MedGemma-27B | 0-5 (seg), 6-7 (MedGemma) | 6 | MedGemma needs 2 GPUs |
| LLM via vLLM server | All 8 for seg | 8 | Run vLLM as separate process on reserved GPUs |

### vLLM server (optional, for heavy LLM use)

For high-throughput LLM calls, run a persistent vLLM server instead of loading models per-case:

```bash
# On reserved GPUs (e.g., 6-7), start vLLM server for MedGemma-27B
CUDA_VISIBLE_DEVICES=6,7 conda run -n cdw_llm \
    vllm serve google/medgemma-27b-text-it \
    --host 127.0.0.1 --port 41260 \
    --tensor-parallel-size 2

# Pipeline will auto-connect via VLLM_ENDPOINT in constants.py
```

### Output locations

| Output | Path | Description |
|--------|------|-------------|
| Per-case segmentations | `{SEGMENTATION_ROOT}/{study}/{case}/segmentations_*/` | Per-organ NIfTI masks |
| Pipeline state | `logs/pipeline_state.json` | Case statuses (resume support) |
| Results CSV | `logs/pipeline_results.csv` | One row/case: modality, tools, QC, radiomics |
| Per-case audit log | `logs/cases/{case_id}_log.json` | Full decision trace |

---

## 7. Common Troubleshooting

### `libcusparseLt.so.0` / `libnvJitLink.so` missing
L40S (Ada Lovelace) or Blackwell GPUs need explicit NVIDIA runtime packages:
```bash
conda run -n <env> pip install nvidia-cusparselt-cu12==0.7.1 nvidia-nvjitlink-cu12==12.8.93
```
`setup_envs.sh` handles this automatically. If you see this error, re-run `bash scripts/setup_envs.sh --env <env>`.

### `weights_only=True` / `EOFError: Ran out of input`
PyTorch 2.6+ changed `torch.load` default. Our patches add `weights_only=False`:
- nnU-Net: patched in `predict_from_raw_data.py` (via `setup_envs.sh`)
- SAT: patched in `external/TextMedSeg3D/model/build_model.py`

If the error returns after upgrading packages, re-run `bash scripts/setup_envs.sh --env <env>`.

### `No module named 'X'` / wrong version loaded
User site-packages (`~/.local/lib/python3.*/site-packages`) override conda env packages:
```bash
export PYTHONNOUSERSITE=1
```
Add this to `~/.bashrc` permanently on the deployment server.

### `dynamic_network_architectures` not found (VIBESegmentator)
This is a transitive dependency of nnU-Net that sometimes fails to install:
```bash
conda run -n cdw_vibeseg pip install dynamic_network_architectures scikit-image
```

### VISTA3D `from_pretrained` fails (safetensors metadata=None)
The runner bypasses this by loading directly via `torch.load(model.pt, weights_only=False)`. No action needed — the tool wrapper handles it.

### Auto-download models fail on air-gapped server
Run each tool once on dummy data while still connected:
```bash
python scripts/run_dummy_test.py --tool totalseg_ct --cases 0002
python scripts/run_dummy_test.py --tool mrseg --cases 0002
python scripts/run_dummy_test.py --tool mriseg --cases 001
python scripts/run_dummy_test.py --tool vibeseg --cases 001
```
This triggers all auto-downloads. After this, the server can go offline.

### VoxTell output files not found
VoxTell names output files `{input_stem}_{organ}.nii.gz`, not `{organ}.nii.gz`. The tool wrapper renames them automatically.

### TextMedSeg3D `PlainConvUNet.__init__() missing num_classes`
SAT uses a custom fork of `dynamic-network-architectures` (bundled at `external/TextMedSeg3D/model/dynamic-network-architectures-main`). Do NOT install the PyPI version — it has an incompatible API.

### conda run picks up `~/.local` packages
On shared servers, `~/.local/lib/python3.10/site-packages` can shadow conda env packages. If this happens:
```bash
# Use the env's pip directly (setup_envs.sh does this automatically)
CONDA_PREFIX="$(conda info --base)"
"${CONDA_PREFIX}/envs/<env>/bin/pip" install <package>
```
