#!/usr/bin/env bash
# ================================================================================
# CDW Agentic Pipeline — Conda Environment Setup
# ================================================================================
# Creates and installs all per-tool conda environments.
# Run from the repo root:
#
#   bash scripts/setup_envs.sh [--env <name>]  # setup specific env
#   bash scripts/setup_envs.sh                 # setup all envs
#
# Requirements:
#   - Miniconda3 with conda on PATH
#   - NVIDIA GPU with CUDA driver ≥ 12.x (RTX Blackwell: CUDA 12.9)
#   - All external repos present under external/
#     (see scripts/clone_externals.sh or git submodule update --init)
#
# See docs/installation.md for detailed notes and troubleshooting.
# ================================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL="${REPO_ROOT}/external"

# PyTorch index URLs
CU124_URL="https://download.pytorch.org/whl/cu124"
CU128_URL="https://download.pytorch.org/whl/cu128"

# NVIDIA Blackwell runtime fix (libcusparseLt.so.0 / libnvJitLink)
BLACKWELL_PKGS="nvidia-cusparselt-cu12==0.7.1 nvidia-nvjitlink-cu12==12.8.93"

# ── Helper ────────────────────────────────────────────────────────────────────

setup_env() {
    local ENV=$1
    shift
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Setting up: $ENV"
    echo "════════════════════════════════════════════════════════"
    "$@"
}

pip_in() {
    # Install packages using the conda env's own pip binary (avoids ~/.local clash)
    local ENV=$1
    shift
    local CONDA_PREFIX
    CONDA_PREFIX="$(conda info --base)"
    "${CONDA_PREFIX}/envs/$ENV/bin/pip" install "$@"
}

create_env() {
    local ENV=$1
    if conda env list | grep -q "^${ENV} "; then
        echo "[INFO] Environment '$ENV' already exists — skipping creation."
    else
        conda create -n "$ENV" python=3.10 -y
    fi
}

# ── Patch nnU-Net predict_from_raw_data.py ───────────────────────────────────
patch_nnunet() {
    local ENV=$1
    local CONDA_PREFIX
    CONDA_PREFIX="$(conda info --base)"
    local SITE="${CONDA_PREFIX}/envs/$ENV/lib/python3.10/site-packages"
    local FILE="${SITE}/nnunetv2/inference/predict_from_raw_data.py"
    if [ ! -f "$FILE" ]; then
        echo "[WARN] nnunetv2 not found in $ENV — skipping patch."
        return
    fi
    if grep -q "weights_only=False" "$FILE"; then
        echo "[INFO] nnU-Net patch already applied in $ENV."
        return
    fi
    # Add weights_only=False to torch.load inside predict_from_raw_data.py
    sed -i "s/torch\.load(\(.*checkpoint_name\))/torch.load(\1, weights_only=False)/" "$FILE"
    echo "[INFO] nnU-Net weights_only patch applied to $ENV."
}

# ── Environments ──────────────────────────────────────────────────────────────

setup_cdw_totalseg() {
    create_env cdw_totalseg
    pip_in cdw_totalseg install torch torchvision --index-url "${CU128_URL}"
    pip_in cdw_totalseg install $BLACKWELL_PKGS "sympy>=1.13.3"
    pip_in cdw_totalseg install -e "${EXTERNAL}/TotalSegmentator"
    pip_in cdw_totalseg install "nnunetv2==2.6.4" requests urllib3 pyarrow xmltodict pyyaml
    # scikit-image and timm needed by TotalSegmentator custom_trainers
    pip_in cdw_totalseg install scikit-image timm huggingface_hub
    pip_in cdw_totalseg install "git+https://github.com/MIC-DKFZ/batchgeneratorsv2.git"
    # nnU-Net legacy checkpoint compatibility for PyTorch 2.6+
    patch_nnunet cdw_totalseg
    echo "[OK] cdw_totalseg ready."
}

setup_cdw_mrseg() {
    create_env cdw_mrseg
    pip_in cdw_mrseg install torch torchvision --index-url "${CU128_URL}"
    pip_in cdw_mrseg install $BLACKWELL_PKGS "sympy>=1.13.3"
    pip_in cdw_mrseg install -e "${EXTERNAL}/MRSegmentator"
    # scikit-image required by batchgenerators (pulled from ~/.local if not in env)
    pip_in cdw_mrseg install scikit-image
    patch_nnunet cdw_mrseg
    echo "[OK] cdw_mrseg ready."
}

setup_cdw_mriseg() {
    create_env cdw_mriseg
    pip_in cdw_mriseg install torch torchvision --index-url "${CU128_URL}"
    pip_in cdw_mriseg install $BLACKWELL_PKGS "sympy>=1.13.3"
    pip_in cdw_mriseg install -e "${EXTERNAL}/MRISegmenter"
    # scikit-image required by batchgenerators (pulled from ~/.local if not in env)
    pip_in cdw_mriseg install scikit-image
    echo "[OK] cdw_mriseg ready."
}

setup_cdw_nvseg() {
    create_env cdw_nvseg
    pip_in cdw_nvseg install "torch==2.10.0" torchvision --index-url "${CU128_URL}"
    pip_in cdw_nvseg install $BLACKWELL_PKGS "sympy>=1.13.3"
    pip_in cdw_nvseg install "monai[all]==1.5.0"
    pip_in cdw_nvseg install "transformers==4.46.3"
    pip_in cdw_nvseg install -e "${EXTERNAL}/NVSegmentCTMR"
    echo "[OK] cdw_nvseg ready."
}

setup_cdw_voxtell() {
    create_env cdw_voxtell
    pip_in cdw_voxtell install torch torchvision --index-url "${CU128_URL}"
    pip_in cdw_voxtell install $BLACKWELL_PKGS "sympy>=1.13.3"
    pip_in cdw_voxtell install -e "${EXTERNAL}/VoxTell"
    # VoxTell uses transformers which requires huggingface-hub <1.0; pin it
    pip_in cdw_voxtell install "huggingface-hub>=0.34,<1.0" "tokenizers>=0.22,<=0.23"
    echo "[OK] cdw_voxtell ready."
}

setup_cdw_vibeseg() {
    create_env cdw_vibeseg
    pip_in cdw_vibeseg install torch torchvision --index-url "${CU128_URL}"
    pip_in cdw_vibeseg install $BLACKWELL_PKGS "sympy>=1.13.3"
    # VIBESegmentator cannot be pip-installed (poetry build); add to sys.path at runtime
    pip_in cdw_vibeseg install "TPTBox>=0.2.0" "ruamel.yaml" "configargparse" nibabel numpy scipy
    pip_in cdw_vibeseg install "nnunetv2==2.6.4" scikit-image
    # dynamic_network_architectures required transitively by nnU-Net; install explicitly
    pip_in cdw_vibeseg install dynamic_network_architectures
    patch_nnunet cdw_vibeseg
    echo "[OK] cdw_vibeseg ready."
}

setup_cdw_biomedparse3d() {
    # NOTE: BiomedParse3D is SKIPPED — detectron2 build is too fragile.
    # Coverage: BiomedParse 2D handles lungs only; 3D text-prompted segmentation
    # is covered by VoxTell and TextMedSeg3D (SAT) which are more reliable.
    echo "[SKIP] cdw_biomedparse3d — skipped (use cdw_voxtell or cdw_textmedseg for 3D)."
}

setup_cdw_textmedseg() {
    create_env cdw_textmedseg
    pip_in cdw_textmedseg install "torch==2.10.0" torchvision --index-url "${CU128_URL}"
    pip_in cdw_textmedseg install $BLACKWELL_PKGS "sympy>=1.13.3"
    pip_in cdw_textmedseg install "monai[all]" "transformers>=4.40.0"
    # SAT requirements
    pip_in cdw_textmedseg install positional_encodings einops pandas openpyxl nibabel scipy
    # SAT requires its own fork of dynamic-network-architectures (bundled in repo)
    pip_in cdw_textmedseg install -e "${EXTERNAL}/TextMedSeg3D/model/dynamic-network-architectures-main"
    # Patch torch.load calls in SAT to add weights_only=False (PyTorch 2.6+ compat)
    sed -i 's/torch\.load(\(.*\), map_location=device)/torch.load(\1, map_location=device, weights_only=False)/g' \
        "${EXTERNAL}/TextMedSeg3D/model/build_model.py"
    echo "[OK] cdw_textmedseg ready."
}

setup_cdw_llm() {
    create_env cdw_llm
    pip_in cdw_llm install vllm --index-url "${CU128_URL}"
    pip_in cdw_llm install "transformers>=4.50.0" accelerate
    pip_in cdw_llm install fastapi uvicorn pydantic
    echo "[OK] cdw_llm ready."
}

setup_cdw_radiomics() {
    create_env cdw_radiomics
    # PyRadiomics from GitHub (pip version is outdated)
    pip_in cdw_radiomics install "git+https://github.com/AIM-Harvard/pyradiomics.git"
    pip_in cdw_radiomics install SimpleITK nibabel scipy pandas numpy
    echo "[OK] cdw_radiomics ready."
}

# ── Entry point ───────────────────────────────────────────────────────────────

ALL_ENVS=(
    cdw_totalseg
    cdw_mrseg
    cdw_mriseg
    cdw_nvseg
    cdw_voxtell
    cdw_vibeseg
    cdw_biomedparse3d
    cdw_textmedseg
    cdw_llm
    cdw_radiomics
)

TARGET="${1:-all}"

if [ "$TARGET" = "all" ] || [ -z "${1:-}" ]; then
    for env in "${ALL_ENVS[@]}"; do
        setup_env "$env" "setup_${env}"
    done
elif [ "$TARGET" = "--env" ] && [ -n "${2:-}" ]; then
    ENV_NAME="$2"
    if declare -f "setup_${ENV_NAME}" > /dev/null; then
        setup_env "$ENV_NAME" "setup_${ENV_NAME}"
    else
        echo "[ERROR] Unknown environment: $ENV_NAME"
        echo "Available: ${ALL_ENVS[*]}"
        exit 1
    fi
else
    echo "Usage: $0 [--env <env_name>]"
    echo "Environments: ${ALL_ENVS[*]}"
    exit 1
fi

echo ""
echo "All done. Verify with:"
echo "  conda env list | grep cdw"
