# Running the CDW Agentic Pipeline

Self-contained guide for running the pipeline on clinical servers.
All commands assume you are in the repo root (`agentic-cdw/`).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Configuration](#2-configuration)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Running the Full Pipeline (Production)](#4-running-the-full-pipeline-production)
5. [Running Individual Components](#5-running-individual-components)
6. [Enabling / Disabling Segmentation Tools](#6-enabling--disabling-segmentation-tools)
7. [Testing & Validation](#7-testing--validation)
8. [Output Structure](#8-output-structure)
9. [LLM Setup](#9-llm-setup)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### Conda Environments

Each segmentation tool runs in its own isolated conda env. Set them all up:

```bash
bash scripts/setup_envs.sh
```

| Env | Tools | VRAM |
|-----|-------|------|
| `cdw_totalseg` | TotalSegmentator CT/MR | 12 GB |
| `cdw_mrseg` | MRSegmentator | 12 GB |
| `cdw_mriseg` | MRISegmenter | 12 GB |
| `cdw_nvseg` | VISTA3D | 16 GB |
| `cdw_vibeseg` | VIBESegmentator | 16 GB |
| `cdw_voxtell` | VoxTell | 32 GB |
| `cdw_textmedseg` | TextMedSeg3D | 36 GB |
| `cdw_llm` | Qwen3-8B + MedGemma-27B | 70 GB (both loaded) |
| `cdw_radiomics` | PyRadiomics, scipy, nibabel | CPU only |

### Model Checkpoints

```bash
# Login to HuggingFace (MedGemma is gated — accept license first)
huggingface-cli login

# Download all checkpoints to checkpoints/
bash scripts/download_checkpoints.sh
```

Expected layout after download:
```
checkpoints/
  qwen3-8b/                    # ~16 GB, planner LLM
  medgemma-27b-text-it/        # ~54 GB, clinical LLM
  qwen3-32b/                   # ~64 GB, optional heavy reasoning
```

---

## 2. Configuration

All paths and settings live in **one file**: `config/constants.py`.

Edit these for your environment:

```python
# Where raw DICOMs live (read-only)
DICOM_ROOT = "/data/RAD"

# Where NIfTI volumes + segmentation outputs go
SEGMENTATION_ROOT = "/data/soumitri/segmentations_3d"

# Filelist (JSON list of case paths — see format below)
FILELIST_CT_AXIAL = "/data/soumitri/new_data_paths/ct_axial_cases.json"
```

Everything else (tool output dirs, conda envs, QC thresholds, LLM settings) has sensible defaults.

### Other config files (rarely need editing)

| File | What it controls |
|------|------------------|
| `config/tool_registry.json` | Tool metadata: names, modalities, class paths, CLI commands |
| `config/organ_reference.json` | QC priors: volume ranges, paired ratios, anatomy-to-organ mapping |
| `config/radiomics_params.yaml` | PyRadiomics settings: resampling, normalization, feature classes |
| `config/prompts/example_prompts.py` | LLM prompt templates (validated, shouldn't need changes) |

---

## 3. Pipeline Overview

Each case flows through these steps:

```
Input: case_path (directory with image_nifti.nii.gz)
  │
  ├─ Step 1: Metadata Extraction ──────── Qwen3-8B (or path heuristic)
  │           → modality, anatomy, is_diagnostic, contrast_status
  │
  ├─ Step 2: Tool Selection ───────────── Qwen3-8B (or registry fallback)
  │           → which segmentation tools to run
  │
  ├─ Step 3: Organ List Generation ────── organ_reference.json lookup
  │           → expected organs for QC (based on anatomy)
  │
  ├─ Step 4: Segmentation ─────────────── each tool in its conda env
  │           → per-organ .nii.gz masks
  │
  ├─ Step 5: Postprocessing ───────────── largest connected component, hole fill
  │           → cleaned masks
  │
  ├─ Step 6: Tier 1 QC (Geometric) ────── pure math, no LLM
  │           → volume, CC count, paired ratios per organ
  │
  ├─ Step 7: Tier 2 QC (Multi-tool) ───── pure math, no LLM
  │           → pairwise Dice, volume divergence across tools
  │
  ├─ Step 8: Tier 3 QC (Interpretation) ─ MedGemma-27B (or rule-based fallback)
  │           → clinical quality assessment, per-organ usability
  │
  ├─ Step 9: Radiomics Gating ─────────── MedGemma-27B (or rule-based fallback)
  │           → which organs/features to extract
  │
  ├─ Step 10: Radiomics Extraction ────── PyRadiomics (CPU)
  │           → ~107 features per organ (firstorder + shape + texture)
  │
  └─ Output: per-case JSON log + row in aggregate CSV
```

**Two-model LLM architecture:**
- **Qwen3-8B** (planner): metadata, tool selection — fast, 1 GPU, <1s/query
- **MedGemma-27B** (clinical): QC interpretation, radiomics gating — 2 GPUs, ~5s/query
- Both are optional — the pipeline has heuristic/rule-based fallbacks for every LLM step

---

## 4. Running the Full Pipeline (Production)

### Filelist Format

The batch runner takes a JSON file — either a flat list or `{"cases": [...]}`:

```json
[
  "/data/soumitri/segmentations_3d/RHEUM/PT0003/CT_ABD_PELVIS",
  "/data/soumitri/segmentations_3d/LUPUS/PT0004/CT_CHEST_ABD",
  "/data/soumitri/segmentations_3d/RHEUM/PT001/MRI_ABD_PELVIS"
]
```

Each path must be a directory containing `image_nifti.nii.gz`.

### Batch Runner

```bash
# Full pipeline with LLM (production)
python -m orchestrator.batch_runner \
  --filelist /data/soumitri/new_data_paths/ct_axial_cases.json \
  --gpus 0,1 \
  --workers 2

# Without LLM (uses path heuristics + rule-based QC)
python -m orchestrator.batch_runner \
  --filelist /data/soumitri/new_data_paths/ct_axial_cases.json \
  --gpus 0,1 \
  --workers 2 \
  --no-llm

# Dry run (mock segmentation, real QC on existing masks)
python -m orchestrator.batch_runner \
  --filelist /data/soumitri/new_data_paths/ct_axial_cases.json \
  --dry-run

# Skip radiomics (just segmentation + QC)
python -m orchestrator.batch_runner \
  --filelist /data/soumitri/new_data_paths/ct_axial_cases.json \
  --gpus 0 \
  --skip-radiomics

# Resume failed cases from a previous run
python -m orchestrator.batch_runner \
  --filelist /data/soumitri/new_data_paths/ct_axial_cases.json \
  --retry-failed
```

**CLI arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--filelist` | (required) | JSON file with case paths |
| `--gpus` | `0` | Comma-separated GPU indices (e.g. `0,1,2`) |
| `--workers` | `1` | Parallel workers (should be <= number of GPUs) |
| `--state-file` | `logs/pipeline_state.json` | Persistent state for resume |
| `--output-csv` | `logs/pipeline_results.csv` | Aggregate results CSV |
| `--dry-run` | off | Mock segmentation, real QC |
| `--no-llm` | off | Skip all LLM calls |
| `--skip-radiomics` | off | Skip radiomics extraction |
| `--retry-failed` | off | Re-process previously failed cases |

### 8×L40S GPU Allocation

| Scenario | GPUs | Workers | Notes |
|----------|------|---------|-------|
| Segmentation only (no LLM) | 0-7 | 8 | 1 worker/GPU, safest |
| Seg + Qwen3-8B planner | 0-5 (seg), 6 (Qwen3) | 6 | Reserve 1 GPU for LLM |
| Seg + Qwen3-8B + MedGemma-27B | 0-5 (seg), 6-7 (MedGemma) | 6 | MedGemma needs 2 GPUs |

### Single-Case (Python)

```python
from planner.llm_client import PlannerLLM
from orchestrator.pipeline import CasePipeline

# With LLM
planner = PlannerLLM.from_local("checkpoints/qwen3-8b")
clinical = PlannerLLM.from_local("checkpoints/medgemma-27b-text-it")
planner.load()
clinical.load()

pipeline = CasePipeline(planner_llm=planner, clinical_llm=clinical)
result = pipeline.run("/data/soumitri/segmentations_3d/RHEUM/PT0003/CT_ABD_PELVIS")

planner.unload()
clinical.unload()

# Without LLM
pipeline = CasePipeline(no_llm=True)
result = pipeline.run("/data/soumitri/segmentations_3d/RHEUM/PT0003/CT_ABD_PELVIS")
```

---

## 5. Running Individual Components

### Metadata Extraction (Qwen3-8B)

```python
from planner.llm_client import PlannerLLM
from planner.metadata_extractor import MetadataExtractor

llm = PlannerLLM.from_local("checkpoints/qwen3-8b")
llm.load()

meta = MetadataExtractor(llm).extract(
    "image_nifti.nii.gz",
    shape=[512, 512, 237],
)
# → {"modality": "CT", "anatomy": "abdomen_pelvis", "is_diagnostic": true, ...}

llm.unload()
```

### Tool Selection (Qwen3-8B)

```python
from planner.tool_selector import ToolSelector

# llm already loaded (see above)
selector = ToolSelector(llm)
plan = selector.select({
    "modality": "CT",
    "anatomy": "abdomen_pelvis",
    "is_diagnostic": True,
    "shape": "3D",
})
# → {"primary_tools": ["TotalSegmentator_CT", "MRSegmentator", "VISTA3D"],  # all CT-compatible
#     "secondary_tools": [],  # always empty (maximum coverage principle)
#     "targeted_tools": [{"tool": "TextMedSeg3D", "organs": [...]}, {"tool": "VoxTell", "organs": [...]}],
#     "qc_organs": [...],
#     "reasoning": "..."}
```

### Organ List Generation (no LLM needed)

```python
from planner.organ_list_generator import OrganListGenerator

gen = OrganListGenerator()  # uses organ_reference.json lookup
organs = gen.organs_for_qc("abdomen_pelvis")
# → ["liver", "spleen", "kidney_left", "kidney_right", "pancreas", ...]
```

### Geometric QC (Tier 1 — no LLM)

```python
from qc.geometric_qc import GeometricQC

gqc = GeometricQC()
result = gqc.run(
    case_path="/data/.../PT0003/CT_ABD_PELVIS",
    seg_dir="/data/.../PT0003/CT_ABD_PELVIS/segmentations_totalseg_ct",
    tool_name="TotalSegmentator_CT",
    expected_organs=["liver", "spleen", "kidney_left"],  # from organ list
    modality="CT",
)
# result.worst_severity → "PASS" / "WARN" / "FAIL"
# result.organ_results → list of per-organ QC results with flags
```

### Multi-Tool QC (Tier 2 — no LLM)

```python
from qc.multi_tool_qc import MultiToolQC

mtqc = MultiToolQC()
agreement = mtqc.run(
    case_path="/data/.../PT0003/CT_ABD_PELVIS",
    seg_dirs={
        "TotalSegmentator_CT": ".../segmentations_totalseg_ct",
        "MRSegmentator": ".../segmentations_mrseg",
        "VISTA3D": ".../segmentations_vista3d",
    },
)
# agreement.mean_dice → 0.85
# agreement.organs_with_agreement → ["liver", "spleen", ...]
```

### QC Interpretation (Tier 3 — MedGemma-27B)

```python
from planner.llm_client import PlannerLLM
from qc.qc_interpreter import QCInterpreter

llm = PlannerLLM.from_local("checkpoints/medgemma-27b-text-it")
llm.load()

interpreter = QCInterpreter(llm)
interp = interpreter.interpret(
    tool_result,                          # ToolQCResult from Tier 1
    study_description="CT ABD/PELVIS",
    anatomy="abdomen_pelvis",
)
# interp.overall_quality → "GOOD" / "ACCEPTABLE" / "POOR" / "UNUSABLE"
# interp.extract → [{"organ": "liver", "features_allowed": [...], "postprocessing_needed": [...]}]
# interp.skip → [{"organ": "pancreas", "reason": "CC=5, FRAG=40%"}]

llm.unload()
```

### Radiomics Extraction (CPU, no LLM)

```python
from radiomics.pyradiomics import extract_pyradiomics

result = extract_pyradiomics(
    image_path="/data/.../image_nifti.nii.gz",
    mask_path="/data/.../segmentations_totalseg_ct/liver.nii.gz",
    organ="liver",
)
# result.feature_count → 107
# result.features → dict of feature_name: value
```

Requires the `cdw_radiomics` env:
```bash
conda run -n cdw_radiomics python -c "from radiomics.pyradiomics import extract_pyradiomics; ..."
```

---

## 6. Enabling / Disabling Segmentation Tools

Tool availability is controlled via the `"deferred"` flag in `config/tool_registry.json`.

### Check Current Status

```bash
# Show all tools and their deferred status
python -c "
import json
with open('config/tool_registry.json') as f:
    registry = json.load(f)
for tool in registry['tools']:
    status = 'DEFERRED' if tool.get('deferred') else 'ACTIVE'
    reason = f' — {tool[\"deferred_reason\"]}' if tool.get('deferred_reason') else ''
    print(f'  [{status:8s}] {tool[\"name\"]}{reason}')
"
```

### Disable a Tool

Edit `config/tool_registry.json` and add `"deferred": true` to the tool entry:

```json
{
  "name": "BiomedParse3D",
  "deferred": true,
  "deferred_reason": "detectron2 CUDA build too fragile",
  ...
}
```

The planner's `ToolSelector` automatically skips deferred tools — they won't appear in any tool selection plan.

### Enable a Tool

Remove (or set to `false`) the `"deferred"` field:

```json
{
  "name": "BiomedParse3D",
  ...
}
```

### Currently Deferred Tools

| Tool | Reason |
|------|--------|
| BiomedParse3D | detectron2 CUDA build fragility; VoxTell + TextMedSeg3D provide equivalent text-prompted 3D coverage |
| BiomedParse2D | 2D slice-level; pipeline focuses on 3D volumes |
| nnInteractive | Requires interactive click prompts; intended as post-hoc refinement |
| HybridGNet | 2D chest X-ray only; pipeline processes 3D CT/MRI |

### How It Works

In `planner/tool_selector.py`, the `_index_registry()` method filters out deferred tools:

```python
if tool.get("deferred"):
    continue  # skip this tool entirely
```

This means deferred tools are invisible to both the LLM planner and the rule-based fallback. No other configuration changes are needed — just toggle the flag and re-run.

---

## 7. Testing & Validation

### E2E Test Harness

Tests the full pipeline on 6 dummy cases (3 CT + 3 MRI) with pre-computed masks.

```bash
# String pipeline only (planning chain, no images, ~30s)
conda run -n cdw_radiomics python tests/run_e2e.py --mode string

# Image pipeline (QC + radiomics on real masks, ~15min)
conda run -n cdw_radiomics python tests/run_e2e.py --mode image

# Both
conda run -n cdw_radiomics python tests/run_e2e.py --mode full

# With LLM (agentic mode — requires checkpoints + GPU)
conda run -n cdw_llm python tests/run_e2e.py --mode string --use-llm
conda run -n cdw_llm python tests/run_e2e.py --mode image --use-llm
conda run -n cdw_llm python tests/run_e2e.py --mode full --use-llm
```

**What `--use-llm` changes:**
- String mode: Qwen3-8B for metadata extraction + tool selection
- Image mode: Qwen3-8B (planner) + MedGemma-27B (clinical QC + radiomics gating)
- Without `--use-llm`: path heuristics for metadata, registry fallback for tools, rule-based QC

**Output:** `tests/results/e2e_report.json`

### LLM Pipeline Validation

Tests each LLM decision point independently (not the full pipeline):

```bash
conda run -n cdw_llm python tests/test_llm_pipeline.py
```

Tests 5 steps on 4 test cases (2 CT, 2 MRI):
1. Metadata extraction (Qwen3-8B) — modality, anatomy, is_diagnostic
2. Tool selection (Qwen3-8B) — all compatible tools selected, no tiering
3. Organ prediction (MedGemma-27B) — correct organs for anatomy
4. QC interpretation (MedGemma-27B) — quality assessment from flags
5. Radiomics gating (MedGemma-27B) — per-organ extract/skip decisions

### Individual Tool Smoke Test

```bash
# Test all active tools on dummy data
python scripts/run_dummy_test.py

# Test one specific tool
python scripts/run_dummy_test.py --tool TotalSegmentator_CT
```

---

## 8. Output Structure

### Per-Case Directory

After the pipeline runs on a case, its directory looks like:

```
<case_path>/
  image_nifti.nii.gz                          # Input NIfTI volume

  segmentations_totalseg_ct/                   # TotalSegmentator CT output (117 organs)
    liver.nii.gz                               #   per-organ binary mask (uint8)
    spleen.nii.gz
    kidney_left.nii.gz
    ...
    statistics.json                            #   volume stats (mm³)
    manifest.json                              #   tool metadata

  segmentations_totalseg_mr/                   # TotalSegmentator MR output (50 organs, MRI only)
    ...

  segmentations_mrseg/                         # MRSegmentator output (40 organs)
    liver.nii.gz
    ...

  segmentations_mriseg/                        # MRISegmenter output (62 organs, MRI only)
    ...

  segmentations_vibeseg/                       # VIBESegmentator output (72 organs)
    ...

  segmentations_vista3d/                       # VISTA3D output (345+ organs)
    ...

  segmentations_voxtell/                       # VoxTell output (text-prompted)
    ...

  segmentations_textmedseg3d/                  # TextMedSeg3D output (text-prompted)
    ...
```

Tool output directory names are defined in `config/constants.py` → `TOOL_OUTPUT_DIRS`.

### Pipeline Logs

```
logs/
  pipeline_state.json                          # Case tracker (pending/running/completed/failed)
  pipeline_results.csv                         # One row per case (aggregate)
  cases/
    PT0003_log.json                            # Per-case decision audit log
    PT0004_log.json
    ...
```

### Per-Case Log Format (`logs/cases/<case_id>_log.json`)

```json
{
  "case_id": "PT0003",
  "case_path": "/data/.../PT0003/CT_ABD_PELVIS",
  "status": "completed",
  "total_time_s": 142.5,
  "metadata": {
    "modality": "CT",
    "anatomy": "abdomen_pelvis",
    "is_diagnostic": true,
    "contrast_status": "post",
    "series_type": "axial soft tissue CT with contrast",
    "mri_sequence": null
  },
  "is_diagnostic": true,
  "selected_tools": ["TotalSegmentator_CT", "MRSegmentator", "VISTA3D"],
  "expected_organs": ["liver", "spleen", "kidney_left", "kidney_right", "pancreas", ...],
  "tool_outputs": [...],
  "qc_report": {
    "overall_severity": "WARN",
    "num_tools": 3,
    "per_tool": [
      {
        "tool": "TotalSegmentator_CT",
        "severity": "PASS",
        "num_organs": 45,
        "num_in_fov": 38,
        "num_flagged": 2,
        "total_flags": 3
      },
      ...
    ],
    "agreement": {
      "mean_dice": 0.871,
      "organs_agree": ["liver", "spleen", ...],
      "organs_disagree": ["gallbladder"]
    },
    "interpretation": {
      "overall_quality": "ACCEPTABLE",
      "usable_for_radiomics": ["liver", "spleen", "kidney_left", ...],
      "unusable_organs": ["pancreas"],
      "extract": [
        {"organ": "liver", "features_allowed": ["first_order", "shape", "texture"], "postprocessing_needed": ["none"]},
        ...
      ],
      "skip": [
        {"organ": "pancreas", "reason": "CC=5, FRAG=40%"}
      ],
      "explanation": "Most organs segment well. Pancreas shows severe fragmentation..."
    }
  },
  "radiomics_summary": [
    {"organ": "liver", "feature_count": 107},
    {"organ": "spleen", "feature_count": 107}
  ],
  "step_times": {
    "metadata": 0.8,
    "tool_selection": 0.5,
    "organ_list": 0.01,
    "seg_TotalSegmentator_CT": 45.2,
    "seg_MRSegmentator": 38.1,
    "seg_VISTA3D": 52.3,
    "qc": 2.1,
    "qc_interpretation": 5.4,
    "radiomics": 12.3
  },
  "error": "",
  "warnings": []
}
```

### Pipeline State (`logs/pipeline_state.json`)

Tracks which cases are done. The batch runner reads this on startup and skips completed cases automatically:

```json
{
  "PT0003": "completed",
  "PT0004": "completed",
  "PT0005": "running",
  "PT0006": "failed",
  "PT0007": "pending"
}
```

Use `--retry-failed` to re-process failed cases.

### Aggregate CSV (`logs/pipeline_results.csv`)

One row per case, flushed every 50 cases. Columns include case_id, status, modality, anatomy, tools, QC severity, radiomics organ count, total time.

### E2E Test Output (`tests/results/e2e_report.json`)

Contains per-case results + validation checks (PASS/FAIL for each assertion).

---

## 9. LLM Setup

### Default: Local Transformers (recommended)

Models are loaded on-demand and freed after use. No server needed.

```python
# The pipeline does this internally:
planner = PlannerLLM.from_local("checkpoints/qwen3-8b")     # ~16 GB VRAM
clinical = PlannerLLM.from_local("checkpoints/medgemma-27b-text-it")  # ~54 GB VRAM
```

Both models fit on a single node with ~91 GB free VRAM. If running sequentially, only ~54 GB is needed at peak.

### Optional: vLLM Server

If you want to run the LLM as a persistent server (e.g. shared across multiple pipeline instances):

```bash
# Terminal 1: start vLLM server
vllm serve Qwen/Qwen3-32B \
  --host 127.0.0.1 \
  --port 41260 \
  --generation-config vllm

# Terminal 2: use from pipeline
python -c "
from planner.llm_client import PlannerLLM
llm = PlannerLLM.from_vllm('http://127.0.0.1:41260', model_id='Qwen/Qwen3-32B')
print(llm.query(system_prompt='', user_prompt='Hello'))
"
```

Note: On clinical servers behind Apache proxy, localhost requests may be intercepted.
The pipeline sets `proxies={"http": None, "https": None}` to bypass this.

### No LLM Mode

Every LLM step has a fallback:
- **Metadata**: parsed from directory path (e.g. `CT_ABD_PELVIS` → modality=CT, anatomy=abdomen_pelvis)
- **Tool selection**: all registry-compatible tools for the modality
- **Organ list**: lookup from `organ_reference.json`
- **QC interpretation**: rule-based severity mapping (PASS→GOOD, WARN→ACCEPTABLE, FAIL→POOR)
- **Radiomics gating**: rule-based feature gating from geometric QC flags

```bash
python -m orchestrator.batch_runner --filelist cases.json --no-llm
```

---

## 10. Troubleshooting

### "Model not loaded" error
Call `llm.load()` before querying (or use `with` context manager).

### Qwen3-8B returns empty JSON
Qwen3 uses `<think>` reasoning blocks that consume the token budget. The pipeline strips these automatically. If you still get empty responses, increase `max_new_tokens` (default: 4096).

### PyRadiomics import error in `cdw_llm` env
The local `radiomics/` directory in the repo shadows the pip-installed `pyradiomics` package. Run radiomics extraction in the `cdw_radiomics` env, not `cdw_llm`.

### Apache proxy intercepting localhost vLLM requests
Set `NO_PROXY='*'` or use the pipeline's built-in proxy bypass (already handled in `VLLMClient`).

### GPU OOM
- Qwen3-8B needs ~16 GB, MedGemma-27B needs ~54 GB
- If both must be loaded simultaneously: ~70 GB
- The pipeline loads/unloads models on-demand to minimize peak usage
- For segmentation tools, check per-tool VRAM in the env table above

### Resuming interrupted runs
The batch runner persists state to `logs/pipeline_state.json`. Just re-run the same command — completed cases are skipped. Add `--retry-failed` to also re-process failed cases.
