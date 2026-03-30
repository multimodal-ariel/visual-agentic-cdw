"""
================================================================================
CDW Agentic Pipeline — Constants & Configuration
================================================================================
Single source of truth for all paths, model names, and pipeline settings.
Change paths here, not in individual scripts.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# DIRECTORY ROOTS
# ──────────────────────────────────────────────────────────────────────────────

# Raw DICOM root (source data, read-only)
DICOM_ROOT = "/data/RAD"

# NIfTI + segmentation output root
SEGMENTATION_ROOT = "/data/soumitri/segmentations_3d"

# Pipeline working directory
PIPELINE_ROOT = Path(__file__).resolve().parent.parent  # agentic_cdw/

# ──────────────────────────────────────────────────────────────────────────────
# FILELISTS
# ──────────────────────────────────────────────────────────────────────────────

# Master filelist (all scans, all modalities)
FILELIST_ALL = "/data/soumitri/new_data_paths/3d_scans_list.json"

# CT-only filtered filelist (post-filter, diagnostic axial only)
FILELIST_CT_AXIAL = "/data/soumitri/new_data_paths/ct_axial_cases.json"

# Testing the pipeline on a small subset of random images
FILELIST_TESTING = "/data/soumitri/test_pipeline_demo/filelist.json"

# Cases needing NIfTI reconversion
FILELIST_RECONVERT = "/data/soumitri/new_data_paths/reconvert_cases.json"

# ──────────────────────────────────────────────────────────────────────────────
# PATH REMAPPING (DICOM → Segmentation directory)
# ──────────────────────────────────────────────────────────────────────────────

# Used by load_filelist() when input paths are under DICOM_ROOT
# but segmentations live under SEGMENTATION_ROOT
SRC_PREFIX = f"{DICOM_ROOT}/"
DST_PREFIX = f"{SEGMENTATION_ROOT}/"

# ──────────────────────────────────────────────────────────────────────────────
# EXPECTED DIRECTORY LAYOUT (per case)
# ──────────────────────────────────────────────────────────────────────────────
# <case_path>/
#     image_nifti.nii.gz                    — converted NIfTI volume
#     segmentations_separate/               — per-organ masks from TotalSeg etc.
#         liver.nii.gz
#         spleen.nii.gz
#         ...
#         statistics.json                   — TotalSegmentator stats (volume in mm³)
#     segmentations_<tool_name>/            — outputs from other tools (future)

IMAGE_FILENAME = "image_nifti.nii.gz"
SEGMENTATION_DIRNAME = "segmentations_separate"
STATISTICS_FILENAME = "statistics.json"

# ──────────────────────────────────────────────────────────────────────────────
# LLM CHECKPOINTS
# ──────────────────────────────────────────────────────────────────────────────

# Root directory for all downloaded model checkpoints
CHECKPOINT_DIR = PIPELINE_ROOT / "checkpoints"

# MedGemma-27B — primary planner (on-demand transformers, no server needed)
# Download: huggingface-cli download google/medgemma-27b-text-it \
#           --local-dir checkpoints/medgemma-27b-text-it
MEDGEMMA_27B_CHECKPOINT = str(CHECKPOINT_DIR / "medgemma-27b-text-it")

# Qwen3-8B — light/fast general-purpose LLM (~16 GB, 1 GPU, ~8 s load)
# Download: huggingface-cli download Qwen/Qwen3-8B \
#           --local-dir checkpoints/qwen3-8b
QWEN3_8B_CHECKPOINT = str(CHECKPOINT_DIR / "qwen3-8b")

# Qwen3-32B — heavier general-purpose LLM (~64 GB, 2 GPUs, ~30 s load)
# Download: huggingface-cli download Qwen/Qwen3-32B \
#           --local-dir checkpoints/qwen3-32b
QWEN3_32B_CHECKPOINT = str(CHECKPOINT_DIR / "qwen3-32b")

# ──────────────────────────────────────────────────────────────────────────────
# LLM / PLANNER SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

# Shared generation settings (used by both LLM clients as defaults)
PLANNER_MODEL_NAME = "google/medgemma-27b-text-it"   # kept for back-compat
PLANNER_DEVICE = "auto"          # "auto", "cuda:0", "cuda:0,1", etc.
PLANNER_TORCH_DTYPE = "bfloat16" # "bfloat16" (default) or "float16"
PLANNER_MAX_NEW_TOKENS = 2048
PLANNER_TEMPERATURE = 0.1

# Optional: vLLM server endpoint (not the default — local transformers is primary)
# Only used when explicitly calling PlannerLLM.from_vllm() or Qwen3(mode="api").
# Start server: vllm serve Qwen/Qwen3-32B --host 127.0.0.1 --port 41260
#               --generation-config vllm  (required for Qwen3)
# Note: NO_PROXY='*' or proxies bypass required — Apache intercepts localhost.
VLLM_ENDPOINT = "http://127.0.0.1:41260/v1/chat/completions"

# ──────────────────────────────────────────────────────────────────────────────
# SEGMENTATION TOOLS
# ──────────────────────────────────────────────────────────────────────────────

# Default GPU for segmentation models
SEGMENTATION_DEVICE = "gpu:0"

# Where each tool writes its output (relative to case_path)
# Keys must match the `name` field in tool_registry.json exactly.
TOOL_OUTPUT_DIRS = {
    # Fixed-class tools
    "TotalSegmentator_CT":  "segmentations_totalseg_ct",
    "TotalSegmentator_MR":  "segmentations_totalseg_mr",
    "MRSegmentator":        "segmentations_mrseg",
    "MRISegmenter":         "segmentations_mrisegmenter",
    "VIBESegmentator":      "segmentations_vibeseg",
    # Text-promptable tools
    "VoxTell":              "segmentations_voxtell",
    "BiomedParse3D":        "segmentations_biomedparse3d",
    "TextMedSeg3D":         "segmentations_textmedseg3d",
    # Label-prompted tools
    "VISTA3D":              "segmentations_vista3d",
    # Deferred (2D / interactive)
    "BiomedParse2D":        "segmentations_biomedparse2d",
    "HybridGNet":           "segmentations_hybridgnet",
    "nnInteractive":        "segmentations_nninteractive",
}

# Conda environment for each tool (subprocess isolation via `conda run -n <env>`)
# Keys must match TOOL_OUTPUT_DIRS keys exactly.
TOOL_CONDA_ENVS = {
    # Fixed-class tools
    "TotalSegmentator_CT":  "cdw_totalseg",
    "TotalSegmentator_MR":  "cdw_totalseg",
    "MRSegmentator":        "cdw_mrseg",
    "MRISegmenter":         "cdw_mriseg",
    "VIBESegmentator":      "cdw_vibeseg",
    # Text-promptable tools
    "VoxTell":              "cdw_voxtell",
    "BiomedParse3D":        "cdw_biomedparse3d",
    "TextMedSeg3D":         "cdw_textmedseg",
    # Label-prompted tools
    "VISTA3D":              "cdw_nvseg",
    # Deferred
    "BiomedParse2D":        "cdw_biomedparse2d",
    "HybridGNet":           "cdw_hybridgnet",
    "nnInteractive":        "cdw_nninteractive",
}

# ──────────────────────────────────────────────────────────────────────────────
# QC SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

# Severity thresholds
QC_WARN_MAX_FLAGS = 2    # <= this many flags = WARN
QC_FAIL_MIN_FLAGS = 3    # >= this many flags = FAIL

# Connected component: minimum fraction for largest component
QC_LARGEST_CC_MIN_FRACTION = 0.85

# CSV flush interval (cases between disk writes)
QC_CSV_FLUSH_EVERY = 50

# ──────────────────────────────────────────────────────────────────────────────
# RADIOMICS
# ──────────────────────────────────────────────────────────────────────────────

# Feature extractor
RADIOMICS_BACKEND = "pyradiomics"  # "pyradiomics" or "curadiomics"

# PyRadiomics config file (if any)
PYRADIOMICS_CONFIG = None  # path to .yaml params file, or None for defaults

# ──────────────────────────────────────────────────────────────────────────────
# PARALLELISM
# ──────────────────────────────────────────────────────────────────────────────

# Default worker counts (override via CLI --workers)
DEFAULT_QC_WORKERS = 16
DEFAULT_SEG_WORKERS = 3    # per GPU, for TotalSegmentator
DEFAULT_RADIOMICS_WORKERS = 8

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

LOG_DIR = str(PIPELINE_ROOT / "logs")
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
