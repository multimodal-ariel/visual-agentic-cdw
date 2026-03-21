#!/usr/bin/env bash
# ================================================================================
# CDW Agentic Pipeline — Checkpoint Downloader
# ================================================================================
# Downloads all LLM checkpoints into checkpoints/ at the repo root.
# Run from the repo root:
#
#   bash scripts/download_checkpoints.sh
#
# Requirements:
#   pip install huggingface_hub[cli]
#   huggingface-cli login         # required for gated models (MedGemma)
#
# Note: MedGemma-27B is a gated model on HuggingFace. You must:
#   1. Accept the model license at https://huggingface.co/google/medgemma-27b-text-it
#   2. Run `huggingface-cli login` with a token that has accepted the license.
#   Qwen3 models are open-access — no token needed.
# ================================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINTS_DIR="${REPO_ROOT}/checkpoints"

mkdir -p "${CHECKPOINTS_DIR}"
echo "Checkpoints directory: ${CHECKPOINTS_DIR}"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# MedGemma-27B-text-it (GATED — requires HF login and license acceptance)
# ──────────────────────────────────────────────────────────────────────────────

echo "==> Downloading google/medgemma-27b-text-it ..."
echo "    (Gated model. Make sure you have accepted the license and are logged in.)"
huggingface-cli download \
    google/medgemma-27b-text-it \
    --local-dir "${CHECKPOINTS_DIR}/medgemma-27b-text-it" \
    --local-dir-use-symlinks False
echo "    Done: ${CHECKPOINTS_DIR}/medgemma-27b-text-it"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Qwen3-8B (open-access — default light model)
# ──────────────────────────────────────────────────────────────────────────────

echo "==> Downloading Qwen/Qwen3-8B ..."
huggingface-cli download \
    Qwen/Qwen3-8B \
    --local-dir "${CHECKPOINTS_DIR}/qwen3-8b" \
    --local-dir-use-symlinks False
echo "    Done: ${CHECKPOINTS_DIR}/qwen3-8b"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Qwen3-32B (open-access — heavy reasoning model)
# ──────────────────────────────────────────────────────────────────────────────

echo "==> Downloading Qwen/Qwen3-32B ..."
huggingface-cli download \
    Qwen/Qwen3-32B \
    --local-dir "${CHECKPOINTS_DIR}/qwen3-32b" \
    --local-dir-use-symlinks False
echo "    Done: ${CHECKPOINTS_DIR}/qwen3-32b"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Segmentation Tool Model Weights
# ──────────────────────────────────────────────────────────────────────────────

# BiomedParse 2D+3D (open-access)
echo "==> Downloading microsoft/BiomedParse (2D+3D checkpoints) ..."
huggingface-cli download \
    microsoft/BiomedParse \
    --local-dir "${CHECKPOINTS_DIR}/BiomedParse2d3d" \
    --local-dir-use-symlinks False
echo "    Done: ${CHECKPOINTS_DIR}/BiomedParse2d3d"
echo ""

# VoxTell v1.1 (open-access, DKFZ)
echo "==> Downloading mrokuss/VoxTell ..."
huggingface-cli download \
    mrokuss/VoxTell \
    --local-dir "${CHECKPOINTS_DIR}/VoxTell" \
    --local-dir-use-symlinks False
echo "    Done: ${CHECKPOINTS_DIR}/VoxTell"
echo ""

# NV-Segment-CTMR / VISTA3D (open-access, NVIDIA)
echo "==> Downloading nvidia/NV-Segment-CTMR (VISTA3D weights) ..."
huggingface-cli download \
    nvidia/NV-Segment-CTMR \
    --local-dir "${CHECKPOINTS_DIR}/NVSegmentCTMR" \
    --local-dir-use-symlinks False
echo "    Done: ${CHECKPOINTS_DIR}/NVSegmentCTMR"
echo ""

# TextMedSeg3D / SAT-Nano (open-access)
echo "==> Downloading zzh99/SAT (TextMedSeg3D — Nano + Pro checkpoints) ..."
huggingface-cli download \
    zzh99/SAT \
    --local-dir "${CHECKPOINTS_DIR}/TextMedSeg3D" \
    --local-dir-use-symlinks False
# Download SAT-Pro separately (not included in the full snapshot by default)
echo "    Downloading SAT-Pro checkpoint ..."
huggingface-cli download \
    zzh99/SAT \
    Pro/SAT_Pro.pth Pro/text_encoder.pth \
    --local-dir "${CHECKPOINTS_DIR}/TextMedSeg3D" \
    --local-dir-use-symlinks False
echo "    Done: ${CHECKPOINTS_DIR}/TextMedSeg3D"
echo "      Nano: ${CHECKPOINTS_DIR}/TextMedSeg3D/Nano/nano.pth"
echo "      Pro:  ${CHECKPOINTS_DIR}/TextMedSeg3D/Pro/SAT_Pro.pth"
echo ""

# MRSegmentator (downloads automatically on first run via the pip package)
# MRISegmenter (downloads automatically on first run via the pip package)
# TotalSegmentator (downloads automatically on first run via the pip package)
# VIBESegmentator (downloads automatically on first run via the pip package)

echo "All checkpoints downloaded to ${CHECKPOINTS_DIR}"
