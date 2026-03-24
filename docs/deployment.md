# CDW Agentic Pipeline — Full Deployment Guide (8×L40S)

This is the comprehensive guide to deploying the entire codebase onto a fresh multi-GPU node (specifically optimized and tested for an 8×46 GB L40S system, Ubuntu 22.04+, CUDA driver ≥ 12.4). All setup relies entirely on user-land configurations (`conda`) and requires **ZERO sudo/root access**. It ensures complete reproducibility, preserving submodules, explicit environment isolations, and exact package versions out-of-the-box.

---

## 1. Initial System Sanity & Prerequisites

Ensure the target node meets the basic prerequisites:
- **CUDA Driver visibility**: Verify with `nvidia-smi` (you should see your 8×L40S GPUs).
- **Disk Space**: At least 100-150 GB free (checkpoints are ~80 GB, plus space for inputs/outputs/conda environments).
- **Conda**: A working installation of Miniconda or Anaconda on your user `PATH`.

### Sudo-free System Packages
If your dataset requires Git LFS (Large File Storage) and the system does not have it, do **not** use `apt`. Install it cleanly into your base conda environment:
```bash
conda install -c conda-forge git-lfs
git lfs install
```

---

## 2. Cloning the Repository & Submodules (CRITICAL Step)

The pipeline integrates heavily patched third-party submodules (like `TextMedSeg3D` and `BiomedParse2D`). To fetch these exact modified versions without breaking the pipeline, you **must** use a recursive clone.

```bash
# Clone the repository and initialize all nested submodules preserving their patched states
git clone --recurse-submodules <repo_url> agentic-cdw
cd agentic-cdw

# If you accidentally cloned without --recurse-submodules, fix it with:
# git submodule update --init --recursive
```

*Note: All external dependencies live in `/external/`. Do not pip-install upstream versions of these tools; our setup scripts will build the specific patched local source code.*

---

## 3. Creating Isolated Conda Environments

To prevent dependency hell and prevent your local `PIP_USER` environment from bleeding into the pipeline tools, use our automated provisioning script. We use isolated `cdw_*` environments for each segment of the pipeline.

```bash
# Optional but highly recommended: Prevent user-site packages from interfering
export PYTHONNOUSERSITE=1

# Create all environments from scratch
bash scripts/setup_envs.sh

# Or provision a single specific environment, e.g., for TotalSegmentator:
# bash scripts/setup_envs.sh --env cdw_totalseg
```

### What `setup_envs.sh` Does:
1. Provisions heavily isolated python environments (`cdw_totalseg`, `cdw_textmedseg`, `cdw_llm`, `cdw_radiomics`, etc.).
2. Uses PyTorch `cu128` wheels mapped for L40S/Blackwell architecture compatibility.
3. Automatically patches inference bugs (like PyTorch ≥2.6 `weights_only=True` loading blocks in SAT and nnU-Net).
4. Compiles the local submodules in `/external/` directly into their respective environments.

---

## 4. Downloading Model Checkpoints (Offline Ready)

To allow the server to run fully offline later, download all model weights immediately. 
You will need your HuggingFace User Access Token (to access gated models like MedGemma-27B).

First, install the HuggingFace CLI in your base environment:
```bash
pip install -U "huggingface_hub[cli]"

# Login using your HF Access Token
huggingface-cli login
```

Once authenticated, execute the unified checkpoint downloader:
```bash
bash scripts/download_checkpoints.sh
```

**What this fetches (`/checkpoints/`):**
- Foundation LLMs (Qwen3, Llama, MedGemma).
- Core visual/segmentation weights (VoxTell, VISTA3D, TextMedSeg3D).
- *Note: Some models like TotalSegmentator or MRISegmentator may dynamically download their lightweight configs upon first execution, but the heavyweight checkpoints will be localized.*

---

## 5. Environment & Resource Configuration

With the code and weights localized, link them to the system globally via `config/constants.py`.

Open `config/constants.py` and modify the cluster paths to map to your system's data drives:
- `DICOM_ROOT`: Absolute path to your raw DICOM storage.
- `SEGMENTATION_ROOT`: Absolute path where output NIfTI arrays will be saved.
- `FILELIST_CT` / `FILELIST_MRI`: Absolute paths to your JSON manifests.
- `PLANNER_DEVICE` / `SEGMENTATION_DEVICE`: Defaults to `cuda` or `gpu`.

### Model Directory Overrides
If you placed the `checkpoints/` folder on a different massive storage drive (e.g., `/mnt/data/models`), expose them via environment variables before running any code:
```bash
export VOXTELL_MODEL_DIR="/mnt/data/models/VoxTell"
export NVSEG_DIR="/mnt/data/models/NVSegmentCTMR"
export TEXTMEDSEG3D_DIR="/mnt/data/models/TextMedSeg3D"
```

---

## 6. Verification and Smoke Testing

Always run a sanity check to verify the 8 GPUs are visible inside the insulated conda environments and that no core python paths are broken.

**Device Check:**
```bash
conda run -n cdw_totalseg python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | Device 0: {torch.cuda.get_device_name(0)}')"
```

**End-to-End Pipeline Smoke Test:**
This simulates the tool wrappers on dummy metadata without taking hours.
```bash
# Test fundamental tool wrappers
python scripts/run_dummy_test.py --tool totalseg_ct --device gpu:0
python scripts/run_dummy_test.py --tool voxtell --device gpu:0 --cases 0004

# Test orchestrator planner logic (Text & Vision modes)
conda run -n cdw_radiomics python tests/run_e2e.py --mode string
conda run -n cdw_radiomics python tests/run_e2e.py --mode image
```
Success will yield a clean `tests/results/e2e_report.json` reporting no fatal stack traces.

---

## 7. Launching Production Batches (8×L40S)

When running the actual data across the 8×L40S architecture, use the built-in batch orchestrator. The runner distributes patient sub-lists round-robin.

Example for parallel CT processing:
```bash
python -m orchestrator.batch_runner \
  --filelist /mnt/data/ct_axial_cases.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers 2 \
  --segmentation_device gpu
```
*Tuning Advice:*
- **TotalSegmentator**: Fast, supports 2-3 workers per GPU (`--workers 2`).
- **NVSegment / VoxTell / SAT**: VRAM heavy, restrict to 1 worker per GPU.
- **LLM Workers**: A single L40S (46GB) can comfortably fit Qwen3-8B context. For Qwen3-32B or MedGemma-27B, spawn a dedicated `vLLM` server spanning multiple GPUs using tensor parallelism (`--tensor-parallel-size 2`) and update `VLLM_ENDPOINT` in `constants.py`.

---

## 8. Common Troubleshooting

- **`libcusparseLt.so.0` / `libnvJitLink.so` missing**: Ensure you ran `setup_envs.sh`, which explicitly installs the `nvidia-cusparselt-cu12` and `nvidia-nvjitlink-cu12` wheels for Blackwell/Ada-Lovelace architectures.
- **`EOFError: Ran out of input` / `weights_only=True` PyTorch errors**: Usually means you ran an upstream inference script natively without our patches. Rerun `setup_envs.sh --env <env>` to re-apply the local `weights_only=False` band-aids.
- **No module named 'X' / Wrong Version loaded**: Your system's `PIP_USER_SITE` is hijacking imports. Run `export PYTHONNOUSERSITE=1` and try again.

