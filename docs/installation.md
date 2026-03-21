# CDW Agentic Pipeline — Installation Guide

This guide documents every step required to set up the conda environments for each
segmentation tool. All steps were done and verified on:

- **OS**: Ubuntu 22.04
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (CUDA 12.9 driver)
- **Conda**: Miniconda 3
- **Repo root**: `<repo_root>/` (referred to below as `$REPO`)

---

## Prerequisites

```bash
# 1. Conda (Miniconda3) must be installed and on PATH
conda --version   # tested with conda 24.x

# 2. CUDA driver ≥ 12.x (checked with)
nvidia-smi

# 3. huggingface_hub CLI (for checkpoint downloads)
pip install "huggingface_hub[cli]"

# 4. Login to HuggingFace (required for gated MedGemma model)
huggingface-cli login
```

---

## External Source Repos (submodules / clones)

All third-party source repos live under `external/`. They are either git submodules
or plain clones. Do **not** install packages from PyPI when a local source exists;
install from `external/<Tool>` to pin the exact version and keep it reproducible.

```bash
# Clone any missing repos (example for detectron2):
git clone https://github.com/facebookresearch/detectron2.git external/detectron2

# Or pull all submodules (if the repo uses git submodules):
git submodule update --init --recursive
```

Current repos in `external/`:
- `BiomedParse2D`, `BiomedParse3D` — Microsoft BiomedParse v1/v2
- `detectron2` — Facebook Detectron2 (backbone for BiomedParse3D)
- `MRSegmentator` — MRSegmentator (nnU-Net based)
- `MRISegmenter` — MRISegmenter (nnU-Net based)
- `NVSegmentCTMR` — NVIDIA VISTA3D Python API (mirrors `checkpoints/NVSegmentCTMR`)
- `TextMedSeg3D` — SAT / TextMedSeg3D
- `TotalSegmentator` — TotalSegmentator (nnU-Net based)
- `VIBESegmentator` — VIBESegmentator (spine segmentation)
- `VoxTell` — DKFZ VoxTell

---

## Step 1 — Download Model Checkpoints

Run the download script from the repo root. All weights go to `checkpoints/`.

```bash
bash scripts/download_checkpoints.sh
```

Some models download their weights automatically on first inference:
- TotalSegmentator → `~/.totalsegmentator/` (or specify `--weights_dir`)
- MRSegmentator → `~/.mrsegmentator/`
- MRISegmenter → `~/.mrisegmenter/`
- VIBESegmentator → downloads on first run

---

## Step 2 — Conda Environment Setup

Each tool runs in its own isolated conda environment to avoid dependency conflicts.
Create all environments with:

```bash
bash scripts/setup_envs.sh   # see scripts/setup_envs.sh for individual commands
```

Or create them individually as documented below.

---

### `cdw_totalseg` — TotalSegmentator (CT)

```bash
conda create -n cdw_totalseg python=3.10 -y
conda activate cdw_totalseg

# Install TotalSegmentator from local source
pip install -e external/TotalSegmentator

# Install remaining nnU-Net runtime deps
pip install "nnunetv2==2.6.4"
pip install requests urllib3 pyarrow xmltodict pyyaml

# batchgeneratorsv2 (not on PyPI — install from git)
pip install "git+https://github.com/MIC-DKFZ/batchgeneratorsv2.git"
```

**Notes:**
- TotalSegmentator accepts `gpu` (not `cuda:0`) as the device string.
- Weights are downloaded on first run to `~/.totalsegmentator/`.
- If `~/.local/lib/python3.10/site-packages` has an older TotalSegmentator, it
  takes precedence over the conda env. Install directly into the conda env's pip
  (`/home/$USER/miniconda3/envs/cdw_totalseg/bin/pip`) to avoid this.

---

### `cdw_mrseg` — MRSegmentator

```bash
conda create -n cdw_mrseg python=3.10 -y
conda activate cdw_mrseg

# Install from local source
pip install -e external/MRSegmentator

# PyTorch with CUDA 12.8 (required for Blackwell GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Fix for RTX Blackwell (libcusparseLt dependency)
pip install nvidia-cusparselt-cu12==0.7.1 nvidia-nvjitlink-cu12==12.8.93

# nnU-Net patch: allow loading legacy checkpoints in PyTorch 2.6+
# Edit the following line in nnunetv2/inference/predict_from_raw_data.py:
#   checkpoint = torch.load(..., weights_only=False)   # add weights_only=False
```

**nnU-Net checkpoint compatibility patch** (PyTorch 2.6+ changed default):
```python
# File: $CONDA/envs/cdw_mrseg/lib/python3.10/site-packages/nnunetv2/inference/predict_from_raw_data.py
# Find the torch.load() call and add weights_only=False:
checkpoint = torch.load(
    join(model_training_output_dir, f'fold_{f}', checkpoint_name),
    map_location=torch.device('cpu'),
    weights_only=False,   # ← add this
)
```

**Notes:**
- MRSegmentator device string: `gpu` (not `cuda:0` or `gpu0`).
- Weights are downloaded on first run.

---

### `cdw_mriseg` — MRISegmenter

```bash
conda create -n cdw_mriseg python=3.10 -y
conda activate cdw_mriseg

# PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Fix for RTX Blackwell
pip install nvidia-cusparselt-cu12==0.7.1 nvidia-nvjitlink-cu12==12.8.93

# Install MRISegmenter from local source
pip install -e external/MRISegmenter
```

**Notes:**
- MRISegmenter device string: `gpu` (not `cuda:0` or `gpu0`).
- Outputs a multilabel NIfTI (`*_seg.nii.gz`); the tool wrapper splits it per-organ.

---

### `cdw_nvseg` — VISTA3D (NVSegmentCTMR)

```bash
conda create -n cdw_nvseg python=3.10 -y
conda activate cdw_nvseg

# PyTorch with CUDA 12.8
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Fix for RTX Blackwell
pip install nvidia-cusparselt-cu12==0.7.1 nvidia-nvjitlink-cu12==12.8.93 "sympy>=1.13.3"

# MONAI (required by VISTA3D pipeline)
pip install "monai[all]==1.5.0"

# HuggingFace transformers pinned to version in model config
pip install "transformers==4.46.3"

# Install VISTA3D Python API from local NVSegmentCTMR
pip install -e external/NVSegmentCTMR
```

**Notes:**
- Model weights live at `checkpoints/NVSegmentCTMR/vista3d_pretrained_model/model.pt`.
- `from_pretrained` fails because `model.safetensors` has no metadata field. The
  runner bypasses this by using `torch.load(model_pt, weights_only=False)` directly.
- Pipeline outputs multilabel NIfTI at `{output}/{stem}/{stem}_seg.nii.gz`; the
  runner splits it by label index using `metadata.json` channel_def.
- Label map verified against `checkpoints/NVSegmentCTMR/metadata.json`.

---

### `cdw_voxtell` — VoxTell

```bash
conda create -n cdw_voxtell python=3.10 -y
conda activate cdw_voxtell

# PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Fix for RTX Blackwell
pip install nvidia-cusparselt-cu12==0.7.1 nvidia-nvjitlink-cu12==12.8.93

# Install VoxTell from local source
pip install -e external/VoxTell
```

**Notes:**
- VoxTell names output files `{input_stem}_{organ}.nii.gz`; the wrapper renames to
  `{organ}.nii.gz`.
- Model directory: `checkpoints/VoxTell/` (contains `plans.json` + `fold_0/`).
- Set `VOXTELL_MODEL_DIR` env var or pass `model_dir=` to `VoxTellTool`.

---

### `cdw_vibeseg` — VIBESegmentator

```bash
conda create -n cdw_vibeseg python=3.10 -y
conda activate cdw_vibeseg

# PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Fix for RTX Blackwell
pip install nvidia-cusparselt-cu12==0.7.1 nvidia-nvjitlink-cu12==12.8.93

# VIBESegmentator cannot be pip-installed (uses poetry build system).
# Instead, add it to sys.path at runtime (done in the tool runner).
# Install its dependencies:
pip install "TPTBox>=0.2.0" "ruamel.yaml" "configargparse" nibabel numpy scipy

# nnU-Net runtime
pip install "nnunetv2==2.6.4"
```

**Notes:**
- VIBESegmentator is added to `sys.path` at runtime via the runner (not installed).
- Path: `external/VIBESegmentator` is injected into `sys.path` before import.
- Outputs a multilabel NIfTI; the wrapper splits it per-organ.

---

### `cdw_biomedparse3d` — BiomedParse3D ⚠️ SKIPPED

**Status: Not used in production.** BiomedParse3D requires `detectron2` as a
backbone dependency. The detectron2 wheel build is fragile (CUDA version mismatch
between the compiled PyTorch CUDA runtime and the system CUDA toolkit).

**Why skipped:** BiomedParse 2D covers lung segmentation only; 3D text-prompted
segmentation is fully covered by **VoxTell** and **TextMedSeg3D (SAT)**, which
install cleanly and perform reliably. There is no capability gap.

If you need to attempt installation anyway:
```bash
# CUDA 12.4 toolkit required in the build environment
conda create -n cdw_biomedparse3d python=3.10 -y
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# Detectron2 — must match the CUDA version used to compile PyTorch
# Try without CUDA extension compilation first:
FORCE_CUDA=0 pip install --no-build-isolation -e external/detectron2

pip install hydra-core omegaconf nibabel numpy scipy einops timm
pip install -e external/BiomedParse3D
```

---

### `cdw_textmedseg` — TextMedSeg3D (SAT)

```bash
conda create -n cdw_textmedseg python=3.10 -y
conda activate cdw_textmedseg

# PyTorch with CUDA 12.8
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Fix for RTX Blackwell
pip install nvidia-cusparselt-cu12==0.7.1 nvidia-nvjitlink-cu12==12.8.93 "sympy>=1.13.3"

# MONAI + transformers
pip install "monai[all]" "transformers>=4.40.0"

# SAT-specific requirements
pip install positional_encodings einops pandas openpyxl nibabel scipy

# SAT uses a CUSTOMIZED fork of dynamic-network-architectures (bundled in repo)
# Do NOT use the PyPI version — it has a different API (num_classes argument added)
pip install -e external/TextMedSeg3D/model/dynamic-network-architectures-main

# PyTorch 2.6+ patch: torch.load now defaults to weights_only=True — patch SAT's build_model.py
sed -i 's/torch\.load(\(.*\), map_location=device)/torch.load(\1, map_location=device, weights_only=False)/g' \
    external/TextMedSeg3D/model/build_model.py
```

**Checkpoints:**
- **SAT-Nano**: `checkpoints/TextMedSeg3D/Nano/nano.pth` + `nano_text_encoder.pth`
  (backbone: `UNET`, ~20M params)
- **SAT-Pro** (recommended): `checkpoints/TextMedSeg3D/Pro/SAT_Pro.pth` + `text_encoder.pth`
  (backbone: `UNET-L`, ~450M params, better quality)

The tool auto-detects SAT-Pro if the `Pro/SAT_Pro.pth` file exists, falling back to Nano.

**Notes:**
- SAT's `inference.py` uses DDP (torch.distributed). The tool wrapper passes DDP env vars
  via torchrun (`--nproc_per_node=1` for single-GPU inference).
- Output structure: `{rcd_dir}/{dataset_name}/seg_{stem}/{organ}.nii.gz` — wrapper
  collects and moves to canonical `seg_dir/{organ}.nii.gz`.

---

### `cdw_llm` — LLM / Planner

```bash
conda create -n cdw_llm python=3.10 -y
conda activate cdw_llm

# vLLM with CUDA 12.8 (serves MedGemma + Qwen3)
pip install "vllm" --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.50.0" accelerate

# API dependencies
pip install fastapi uvicorn pydantic
```

---

## Step 3 — Environment Variables

Set these in `config/constants.py` or your shell environment:

```bash
# VoxTell model directory
export VOXTELL_MODEL_DIR="$REPO/checkpoints/VoxTell"

# VISTA3D / NVSegmentCTMR
export NVSEG_DIR="$REPO/checkpoints/NVSegmentCTMR"

# BiomedParse3D repo root
export BIOMEDPARSE3D_DIR="$REPO/external/BiomedParse3D"

# TextMedSeg3D checkpoint dir
export TEXTMEDSEG3D_DIR="$REPO/checkpoints/TextMedSeg3D"

# VISTA3D weights file
export BIOMEDPARSE3D_WEIGHTS="$REPO/checkpoints/BiomedParse2d3d/biomedparse_v2.ckpt"
```

---

## Troubleshooting

### `libcusparseLt.so.0: cannot open shared object file`
This occurs on RTX Blackwell GPUs (CUDA 12.9 driver) with PyTorch cu128.
Fix by installing the missing NVIDIA runtime library:
```bash
pip install nvidia-cusparselt-cu12==0.7.1 nvidia-nvjitlink-cu12==12.8.93
```

### `libnvJitLink.so.12: cannot open shared object file`
Same cause — PyTorch cu121 binary on Blackwell GPU. Switch to cu128:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### `nnU-Net: weights_only=True` error (PyTorch 2.6+)
PyTorch 2.6+ changed `torch.load` default to `weights_only=True`, which blocks
loading legacy nnU-Net checkpoints that contain numpy arrays.
Patch the nnU-Net file:
```bash
# Find the file
python -c "import nnunetv2; print(nnunetv2.__file__)"
# Edit: nnunetv2/inference/predict_from_raw_data.py
# Add weights_only=False to the torch.load() call that loads fold checkpoints
```

### TotalSegmentator device string
TotalSegmentator accepts `gpu` not `cuda:0`. The tool wrapper handles this.

### MRISegmenter / MRSegmentator device string
Both accept `gpu` not `gpu0` or `cuda:0`. The tool wrapper handles this.

### VISTA3D `from_pretrained` fails (safetensors metadata=None)
The `model.safetensors` file in the NVSegmentCTMR checkpoint has no `format`
metadata. Bypass `from_pretrained` by loading directly:
```python
state_dict = torch.load("vista3d_pretrained_model/model.pt", weights_only=False)
model.network.load_state_dict(state_dict)
```

### conda run picks up `~/.local` packages
User site-packages in `~/.local/lib/python3.10/site-packages` take precedence
over the conda env. Use the env's pip binary directly to avoid this:
```bash
/home/$USER/miniconda3/envs/<env>/bin/pip install <package>
```

### VoxTell output files not found
VoxTell names output files `{input_stem}_{organ}.nii.gz`, not `{organ}.nii.gz`.
The tool wrapper renames them automatically after inference.

### TextMedSeg3D `PlainConvUNet.__init__() missing num_classes`
The PyPI `dynamic-network-architectures` added a `num_classes` argument that breaks SAT.
SAT bundles its own fork at `external/TextMedSeg3D/model/dynamic-network-architectures-main`.
Install that instead of the PyPI version:
```bash
pip install -e external/TextMedSeg3D/model/dynamic-network-architectures-main
```

### TextMedSeg3D `weights_only=True` checkpoint error
Same PyTorch 2.6+ issue as nnU-Net. Patch `external/TextMedSeg3D/model/build_model.py`:
```bash
sed -i 's/torch\.load(\(.*\), map_location=device)/torch.load(\1, map_location=device, weights_only=False)/g' \
    external/TextMedSeg3D/model/build_model.py
```

### BiomedParse3D Hydra config path
`hydra.initialize()` only accepts relative config paths. Use instead:
```python
hydra.initialize_config_dir(config_dir="/absolute/path/to/configs/model")
```
The correct config name is `biomedparse_3D` (file: `configs/model/biomedparse_3D.yaml`).
