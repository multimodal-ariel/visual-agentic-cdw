# CDW: Agentic AI Clinical Imaging Pipeline


**A Comprehensive Data Workflow for uncurated clinical imaging data.**

End-to-end visual processing framework with LLM-guided orchestration for:
- Automated metadata extraction and scan classification
- Multi-model segmentation with intelligent tool selection
- Reference-free quality assessment of segmentations
- Gated radiomics feature extraction


## Authors

#### Soumitri Chattopadhyay, Basar Demir, Yinzhu Jin, Marc Niethammer
#### UCSD Biomedical Image Analysis Group

## Architecture

```
Input (~26k raw DICOM/NIfTI)
    │
    ▼
┌─────────────────────────┐
│  Metadata Extraction     │  ← Qwen3-8B: modality, anatomy, shape, is_diagnostic
│  (Qwen3-8B)              │     fast rule-following, <1s/case, 1 GPU
└────────────┬────────────┘
             │ is_diagnostic=false → discard (~21k cases filtered here)
             ▼
┌─────────────────────────┐
│  Tool Selection          │  ← Qwen3-8B + tool_registry: which models to run
│  (Qwen3-8B)              │     + organ list generation for text-prompted tools
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Segmentation Execution  │  ← TotalSeg, MRSeg, MRISeg, VISTA3D,
│  (Per-tool conda envs)   │     VIBESeg, VoxTell, SAT-Pro, ...
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Quality Check (3-Tier)  │  ← T1: Geometric (volume, CC, ratio, overlap)
│                          │     T2: Multi-tool agreement (Dice, STAPLE)
│                          │     T3: MedGemma-27B clinical interpretation
└────────────┬────────────┘     + per-organ radiomics gating
             ▼
┌─────────────────────────┐
│  Radiomics Extraction    │  ← PyRadiomics (CPU), gated by QC
│  (Gated by QC)           │     only on QC-passing organ masks
└─────────────────────────┘
```

## Directory Structure

```
agentic_cdw/
├── README.md
├── config/
│   ├── constants.py                # All paths, model names, settings (edit this)
│   ├── tool_registry.json          # Available segmentation tools + capabilities
│   ├── organ_reference.json        # Volume ranges, anatomy→organ mappings for QC
│   ├── radiomics_params.yaml       # PyRadiomics feature extractor config
│   └── prompts/
│       └── example_prompts.py      # All LLM prompt templates (validated)
├── llms/                           # Standalone LLM clients (on-demand local inference)
│   ├── base_llm.py                 # BaseLLM ABC + LLMResponse dataclass + CDW system prompt
│   ├── medgemma27b_text.py         # MedGemma-27B-text-it (transformers, no vLLM required)
│   ├── qwen3.py                    # Qwen3 unified client (size="8b"/"32b", mode=local/api)
│   └── utils.py                    # _unwrap_input_ids: transformers 5.x BatchEncoding compat
├── planner/
│   ├── llm_client.py               # PlannerLLM (local transformers default, optional vLLM backend)
│   ├── metadata_extractor.py       # Path → modality/anatomy/shape/is_diagnostic
│   ├── tool_selector.py            # Metadata → tool list + organ prompts
│   └── organ_list_generator.py     # Anatomy → expected organs (lookup + LLM fallback)
├── tools/
│   ├── base_tool.py                # Abstract interface: ToolInput → ToolOutput
│   ├── totalsegmentator_ct.py      # TotalSegmentator CT (117 organs)
│   ├── totalsegmentator_mr.py      # TotalSegmentator MR (50 organs)
│   ├── mrsegmentator.py            # MRSegmentator CT+MRI (40 organs)
│   ├── mrisegmenter.py             # MRISegmenter T1w abdominal MRI (62 organs)
│   ├── vista3d.py                  # NVIDIA VISTA3D CT+MRI (345+ organs)
│   ├── voxtell.py                  # VoxTell free-text 3D (open-set)
│   ├── vibesegmentator.py          # VIBESegmentator MRI+CT (72 organs)
│   ├── textmedseg3d.py             # TextMedSeg3D/SAT-Pro text-prompted (497 classes)
│   ├── biomedparse3d.py            # BiomedParse3D (SKIPPED — detectron2 build)
│   └── runners/
│       ├── run_vista3d.py          # Subprocess runner for VISTA3D (env isolation)
│       └── run_biomedparse3d.py    # Subprocess runner for BiomedParse3D
├── qc/                             # 3-tier reference-free segmentation QC
│   ├── dataclasses.py              # OrganQCResult, ToolQCResult, OrganAgreement, MultiToolQCResult, QCInterpretation, CaseQCReport
│   ├── geometric_qc.py            # Tier 1: volume plausibility, paired ratios, CC, overlap
│   ├── multi_tool_qc.py           # Tier 2: pairwise Dice, volume divergence, STAPLE consensus
│   └── qc_interpreter.py          # Tier 3: LLM clinical interpretation + radiomics gating
├── radiomics/
│   ├── pyradiomics.py              # CPU feature extraction (PyRadiomics); RadiomicsResult — primary backend
│   └── curadiomics.py              # GPU-accelerated extraction (cuRadiomics CUDA op) — optional, not in pipeline
├── processing/
│   ├── preprocessing.py            # NIfTI loading, orientation, resampling
│   ├── postprocessing.py           # LCC, closing, hole fill, mask merging, STAPLE fusion
│   └── format_utils.py             # save_organ_mask(), split_multilabel_nifti() — canonical output format
├── orchestrator/                   # End-to-end pipeline execution
│   ├── pipeline.py                # CasePipeline — single-case flow (metadata → seg → QC → radiomics)
│   ├── batch_runner.py            # BatchRunner — parallel batch with GPU pool, CSV output, CLI
│   └── case_tracker.py            # CaseTracker — JSON-backed persistent state, resume support
├── tests/
│   ├── run_e2e.py                   # E2E test harness: string + image modes on 3 CT + 3 MRI
│   ├── results/                     # E2E output: e2e_report.json with full decision trace
│   ├── test_base_tool.py           # BaseSegmentationTool unit tests
│   ├── test_format_utils.py        # format_utils unit tests (synthetic NIfTI, no real data)
│   └── test_llm_dry_run.py         # LLM dry-run tests (Qwen3-8B/32B, MedGemma × 4 prompt types)
├── scripts/
│   ├── setup_envs.sh               # One-shot conda env setup for all tools
│   ├── download_checkpoints.sh     # Downloads all model weights
│   ├── run_dummy_test.py           # End-to-end test: real cases, real inference
│   └── add_tool_submodule.sh       # Helper for adding new external tool repos
├── docs/
│   └── installation.md             # Full install guide with troubleshooting
├── external/                       # All tool source repos (installed from here)
│   ├── TotalSegmentator/
│   ├── MRSegmentator/
│   ├── MRISegmenter/
│   ├── NVSegmentCTMR/              # VISTA3D
│   ├── VoxTell/
│   ├── VIBESegmentator/
│   ├── TextMedSeg3D/               # SAT — includes patched dynamic-network-architectures
│   ├── cuRadiomics/                # CUDA radiomics op (libRadiomics.so, TF1/TF2 compat)
│   └── ...
└── checkpoints/                    # All downloaded model weights
    ├── medgemma-27b-text-it/
    ├── qwen3-8b/
    ├── qwen3-32b/
    ├── NVSegmentCTMR/              # VISTA3D weights
    ├── VoxTell/voxtell_v1.1/
    └── TextMedSeg3D/
        ├── Nano/nano.pth
        └── Pro/SAT_Pro.pth         # ← preferred (better quality)
```

## Segmentation Tools

### Fixed-class tools (run all eligible structures automatically)

| Tool | Env | Modality | Structures | Status |
|------|-----|----------|------------|--------|
| TotalSegmentator_CT | cdw_totalseg | CT | 117 | ✓ Tested (0002, 0004) |
| TotalSegmentator_MR | cdw_totalseg | MRI | 50 | wrapper done |
| MRSegmentator | cdw_mrseg | CT + MRI | 40 | ✓ Tested (CT + MRI) |
| MRISegmenter | cdw_mriseg | MRI T1w abdominal | 62 | ✓ Tested (001, 002) |
| VIBESegmentator | cdw_vibeseg | MRI + CT | 72 | ✓ Tested (001, 002) |
| VISTA3D | cdw_nvseg | CT + MRI | 345+ | ✓ Tested (all 4 cases) |

### Text-promptable tools (LLM generates organ list at runtime)

| Tool | Env | Modality | Structures | Status |
|------|-----|----------|------------|--------|
| VoxTell | cdw_voxtell | CT / MRI / PET | open-set | ✓ Tested (0002, 002) |
| TextMedSeg3D / SAT-Pro | cdw_textmedseg | CT / MRI / PET | 497 | ✓ Tested (0002, 001) |

### Deferred / skipped

| Tool | Reason |
|------|--------|
| BiomedParse3D | detectron2 CUDA build too fragile; VoxTell + SAT provide equivalent 3D coverage |
| BiomedParse2D | 2D only; pipeline handles 3D volumes |
| HybridGNet | 2D chest X-ray only |
| nnInteractive | Interactive refinement; planned as post-hoc add-on |

## Conda Environments

Each tool runs in an isolated conda env via `conda run -n <env>`. **One unified env will not work** due to dependency conflicts across tools.

| Env | Tools | PyTorch |
|-----|-------|---------|
| cdw_totalseg | TotalSegmentator CT + MR | 2.10+cu128 |
| cdw_mrseg | MRSegmentator | 2.10+cu128 |
| cdw_mriseg | MRISegmenter | 2.10+cu128 |
| cdw_nvseg | VISTA3D | 2.10+cu128 + MONAI 1.5 |
| cdw_voxtell | VoxTell | 2.10+cu128 |
| cdw_vibeseg | VIBESegmentator | 2.10+cu128 |
| cdw_textmedseg | TextMedSeg3D / SAT | 2.10+cu128 |
| cdw_biomedparse3d | BiomedParse3D (env built but tool skipped) | 2.10+cu128 |
| cdw_llm | MedGemma, Qwen3 (inference) | 2.10+cu128 |
| cdw_radiomics | PyRadiomics, cuRadiomics | — (CPU + CUDA via TF) |

Setup all envs:
```bash
bash scripts/setup_envs.sh           # all envs
bash scripts/setup_envs.sh --env cdw_totalseg  # single env
```

## LLM Model Routing

Two-model architecture, empirically validated on CDW prompt types (2026-03-20).

**Key finding:** Qwen3-8B outperforms MedGemma-27B on structured rule-following tasks (5/5 vs 4/5 on `is_diagnostic` edge cases). MedGemma-27B's medical domain training becomes a liability for exact-rule tasks — it reasons about clinical meaning and overrides explicit prompt rules. For clinical interpretation tasks, domain knowledge genuinely helps.

| Task | Model | Score | Why |
|------|-------|-------|-----|
| Metadata extraction (`is_diagnostic`, modality, anatomy) | **Qwen3-8B** | 5/5 edge cases | Rule-following; fast; 1 GPU; <1s/case |
| Tool selection | **Qwen3-8B** | 5/5 | Structured matching, not medical reasoning |
| QC interpretation (per-feature-tier safety) | **MedGemma-27B** | 9/10 vs 7/10 | Differentiates first_order/shape/texture per organ |
| Radiomics gating (conservative clinical judgment) | **MedGemma-27B** | 9/10 vs 7/10 | Conservative on flagged organs; clinical nuance |

**Known MedGemma limitation:** Misclassifies `_SUB` MRI subtraction maps as `is_diagnostic: true`. Root cause: medical pretraining recognises T1 VIBE as diagnostically valuable and overrides the suffix rule. Architectural fix: use Qwen3-8B for `is_diagnostic`; MedGemma never sees that decision.

**Operational flow:**
- Qwen3-8B handles the high-throughput pass (26k cases → metadata + tool selection) — loads in ~2s, fits on 1× L40S
- MedGemma-27B loads once for the ~2-5k cases that reach QC interpretation — ~14s load, 2× L40S, then unloads

## LLM Clients

Two layers, unified design:

**`llms/`** — standalone on-demand clients (no pipeline coupling):
```python
from llms import MedGemma27bText, Qwen3

with MedGemma27bText() as llm:
    resp = llm.query("Describe the anatomy visible in this CT.")

with Qwen3(size="8b") as llm:           # or size="32b"
    resp = llm.query("Select the best tool for this scan.")
    print(resp.content, resp.total_tokens)
```

**`planner/llm_client.py`** — pipeline-integrated wrappers with JSON extraction:
```python
from planner.llm_client import PlannerLLM
from planner.metadata_extractor import MetadataExtractor
from planner.tool_selector import ToolSelector

# Metadata + tool selection: use Qwen3-8B (fast, rule-following)
llm = PlannerLLM.from_local("checkpoints/qwen3-8b")
llm.load()
meta = MetadataExtractor(llm).extract("/data/RAD/.../CT_ABD/image_nifti.nii.gz", shape=[512,512,237])
plan = ToolSelector(llm).select(meta)
llm.unload()

# QC interpretation + radiomics gating: use MedGemma-27B (clinical domain knowledge)
llm = PlannerLLM.from_local("checkpoints/medgemma-27b-text-it")
llm.load()
interp = QCInterpreter(llm).interpret(t1_result, study_description="CT ABD/PELVIS")
llm.unload()

# Optional: vLLM server (if running externally, e.g. Qwen3-32B on g2)
llm = PlannerLLM.from_vllm("http://127.0.0.1:41260", model_id="Qwen/Qwen3-32B")
data = llm.query_json(system_prompt="", user_prompt=prompt)   # strips <think> blocks + JSON fences
```

Supported LLMs:

| Model | Class | VRAM | Role |
|-------|-------|------|------|
| Qwen3-8B | `Qwen3(size="8b")` | ~16 GB (1× L40S) | **Metadata extraction, tool selection** — fast, rule-following, 5/5 is_diagnostic edge cases |
| MedGemma-27B-text-it | `MedGemma27bText` | ~54 GB (2× L40S) | **QC interpretation, radiomics gating** — clinical domain knowledge; gated (HF license required) |
| Qwen3-32B | `Qwen3(size="32b")` | ~64 GB (2× L40S) | Batch metadata; heavy reasoning; local or optional vLLM server |

## Radiomics

Primary backend: **PyRadiomics (CPU)** — 93 features across 6 texture classes, gated by QC pass/fail per organ. Fast enough for production (~1-2s per organ on dummy data).

```python
from radiomics.pyradiomics import extract_pyradiomics, RadiomicsResult

result = extract_pyradiomics("image.nii.gz", "liver_mask.nii.gz", organ="liver")
print(result.feature_count, result.features)   # 93 features: firstorder + GLCM/GLRLM/GLSZM/GLDM/NGTDM
```

Config: `config/radiomics_params.yaml` — 1mm isotropic resampling, z-score normalisation, firstorder + GLCM/GLRLM/GLSZM/GLDM/NGTDM.
Env: `cdw_radiomics` — PyRadiomics (`git+https://github.com/AIM-Harvard/pyradiomics.git`), SimpleITK, nibabel, scipy, pandas.

> **Optional: cuRadiomics (GPU)** — available in `radiomics/curadiomics.py` but not used in the pipeline. Only supports GLCM + first-order (41 features, 2D per-slice with aggregation). Requires TensorFlow + recompiled CUDA `.so` (originally built for TF 1.12 / CUDA 9.2). Use only if GPU acceleration is needed for very large volumes.

## Quality Check (3-Tier Reference-Free QC)

No ground truth required. All checks are reference-free — masks are validated against anatomical priors and cross-tool agreement.

### Tier 1: Geometric QC (`GeometricQC`)
Per-tool, per-organ checks ported from `scripts/qc_segs_old.py` and generalized to any modality/tool:
- **Volume plausibility** — organ volume in mL vs anatomical reference ranges (29 organs)
- **Paired volume ratios** — bilateral symmetry: kidneys, adrenals, iliacs, lungs (4 pairs + lung composite)
- **Connected components** — fragment count + largest-CC fraction (21 organs with CC expectations)
- **Mask overlap** — pairwise voxel overlap between organ masks

### Tier 2: Multi-Tool Agreement (`MultiToolQC`)
Cross-tool pairwise checks when ≥2 tools segment the same case:
- **Dice coefficient** — flags organs where tools disagree (threshold: 0.70)
- **Volume divergence** — flags >50% volume difference between tools
- **Presence mismatch** — one tool found the organ, the other didn't
- **Consensus generation** — STAPLE or majority-vote fusion via `processing/postprocessing.py`

### Tier 3: LLM Interpretation (`QCInterpreter`)
Plugs Tier 1 flags into validated LLM prompts for clinical assessment:
- **QC Interpretation** (`QC_INTERPRETATION` prompt) → overall quality (GOOD/ACCEPTABLE/POOR/UNUSABLE), per-organ usability, feature-class safety
- **Radiomics Gating** (`RADIOMICS_GATING` prompt) → per-organ extract/skip decisions with postprocessing recommendations
- **Fallback** — rule-based gating when LLM is unavailable (PASS→full extraction, WARN→first_order+LCC, FAIL→skip)

```python
from qc import GeometricQC, MultiToolQC, QCInterpreter, CaseQCReport
from planner import PlannerLLM

# Tier 1: geometric checks for one tool
gqc = GeometricQC()
t1_result = gqc.run(
    case_path="/data/soumitri/segmentations_3d/RHEUM/PT123/CT_ABD",
    seg_dir="/data/.../segmentations_totalseg_ct",
    tool_name="TotalSegmentator_CT",
)
print(t1_result.worst_severity, t1_result.num_organs_flagged)

# Tier 1: multiple tools at once
t1_results = gqc.run_all_tools(case_path, seg_dirs={
    "TotalSegmentator_CT": "/path/to/segmentations_totalseg_ct",
    "MRSegmentator": "/path/to/segmentations_mrseg",
})

# Tier 2: cross-tool agreement
mtqc = MultiToolQC()
t2_result = mtqc.run(case_path, seg_dirs={...})
print(t2_result.mean_dice, t2_result.organs_with_disagreement)

# Tier 2: generate consensus mask for an organ
consensus = mtqc.generate_consensus(seg_dirs, organ="liver", method="staple")

# Tier 3: LLM interpretation + radiomics gating
llm = PlannerLLM.from_local("checkpoints/qwen3-8b")
interpreter = QCInterpreter(llm)
interp = interpreter.interpret(t1_result, study_description="CT ABD/PELVIS", anatomy="ABDOMEN")
print(interp.overall_quality)       # GOOD / ACCEPTABLE / POOR / UNUSABLE
print(interp.extract)               # [{organ, features_allowed, postprocessing_needed}, ...]
print(interp.skip)                  # [{organ, reason}, ...]

# Full case report (all tiers combined)
report = CaseQCReport(case_id="PT123", case_path=case_path)
report.tool_results = t1_results
report.agreement = t2_result
report.interpretation = interp
report.compute_overall()
print(report.overall_severity)
```

Per-organ result granularity — every `OrganQCResult` carries individual flags, severity (PASS/WARN/FAIL), volume, CC count, LCC fraction. Radiomics gating operates at the organ level, not case level.

## Orchestrator

End-to-end pipeline execution with GPU pool allocation, persistent state tracking, and automatic resume.

### Single-case pipeline (`CasePipeline`)

```python
from orchestrator.pipeline import CasePipeline
from planner import PlannerLLM

# Two-model production flow
planner = PlannerLLM.from_local("checkpoints/qwen3-8b")       # fast, 1 GPU
clinical = PlannerLLM.from_local("checkpoints/medgemma-27b-text-it")  # 2 GPUs
pipeline = CasePipeline(planner_llm=planner, clinical_llm=clinical, device="gpu:0")
result = pipeline.run("/data/soumitri/segmentations_3d/RHEUM/PT123/CT_ABD")

# Dry-run (mock segmentation, real QC on existing masks, no LLM)
pipeline = CasePipeline(dry_run=True, no_llm=True)
result = pipeline.run("dummy_outputs/0004")
print(result.status, result.qc_report["overall_severity"])
```

Pipeline flow per case:
1. Metadata extraction (LLM or path heuristic)
2. `is_diagnostic` check (skip non-diagnostic)
3. Tool selection (LLM or default per-modality)
4. Organ list generation (lookup + LLM fallback)
5. Segmentation per tool (skip if already done)
6. Postprocess masks (LCC, hole fill)
7. Tier 1+2 QC (geometric + multi-tool agreement)
8. Tier 3 QC interpretation (LLM or rule-based fallback)
9. Radiomics extraction (PyRadiomics on QC-passing organs)
10. Save per-case log JSON to `logs/cases/{case_id}_log.json`

### Batch runner (`BatchRunner`)

```bash
# CLI: 5,376 CT cases on 2 GPUs, 2 parallel workers
python -m orchestrator.batch_runner \
    --filelist /data/soumitri/new_data_paths/ct_axial_cases.json \
    --gpus 0,1 --workers 2 --no-llm

# Dry-run validation on dummy data
python -m orchestrator.batch_runner \
    --filelist test_cases.json --dry-run --no-llm --skip-radiomics

# Resume after crash (auto-skips completed cases)
python -m orchestrator.batch_runner \
    --filelist /data/.../ct_axial_cases.json --gpus 0,1 --workers 2

# Re-process failed cases
python -m orchestrator.batch_runner \
    --filelist /data/.../ct_axial_cases.json --retry-failed
```

Key features:
- **GPU pool**: explicit `--gpus 0,1` allocation prevents two tools from fighting for the same GPU
- **Resume**: JSON state file (`logs/pipeline_state.json`) tracks every case as pending/running/completed/failed — on restart, completed cases are skipped automatically
- **Aggregate CSV**: `logs/pipeline_results.csv` — one row per case with modality, tools, QC severity, radiomics count, timing
- **Per-case audit log**: `logs/cases/{case_id}_log.json` — metadata, tools, QC, radiomics decisions, timing, errors

## Quick Start

```bash
# 1. Set up environments
bash scripts/setup_envs.sh

# 2. Download checkpoints
bash scripts/download_checkpoints.sh   # requires HF login for MedGemma

# 3. Edit paths for your environment
vim config/constants.py

# 4. Run per-tool tests on dummy data
python scripts/run_dummy_test.py --tool TotalSegmentator_CT
python scripts/run_dummy_test.py  # all tools

# 5. Run full E2E test harness (3 CT + 3 MRI, decision-traced)
conda run -n cdw_totalseg python tests/run_e2e.py --mode string   # planning only
conda run -n cdw_radiomics python tests/run_e2e.py --mode image   # full pipeline
conda run -n cdw_radiomics python tests/run_e2e.py --mode full    # both modes
# → Report: tests/results/e2e_report.json

# 6. Dry-run batch pipeline (validates wiring without GPU)
python -m orchestrator.batch_runner \
    --filelist test_cases.json --dry-run --no-llm --skip-radiomics

# 7. Production run
python -m orchestrator.batch_runner \
    --filelist /data/soumitri/new_data_paths/ct_axial_cases.json \
    --gpus 0,1 --workers 2
```

## Data

| Dataset | Cases | NIfTIs | Filter |
|---------|-------|--------|--------|
| RHEUM + LUPUS + CONTR | ~26k volumes | ~24k | all modalities |
| CT axial diagnostic | 5,376 | 5,376 | `ct_axial_cases.json` |

Paths: `DICOM_ROOT = /data/RAD/`, `SEGMENTATION_ROOT = /data/soumitri/segmentations_3d/`

## End-to-End Testing

`tests/run_e2e.py` validates the full pipeline on 3 CT + 3 MRI dummy cases with complete decision tracing.

### Modes

| Mode | What it tests | Env |
|------|--------------|-----|
| `--mode string` | Planning chain only (metadata → tool selection → organ list) on simulated production paths | `cdw_totalseg` |
| `--mode image` | Full pipeline on dummy_outputs (QC on real masks, radiomics on real images) | `cdw_radiomics` |
| `--mode full` | Both modes sequentially | `cdw_radiomics` |

### Test Cases

| Case | Modality | Anatomy | Tools Selected |
|------|----------|---------|---------------|
| 0003 | CT | abdomen_pelvis | TotalSegmentator_CT, VISTA3D |
| 0004 | CT | chest_abdomen_pelvis | TotalSegmentator_CT, VISTA3D |
| 0005 | CT | abdomen | TotalSegmentator_CT, VISTA3D |
| 0002 | MRI | abdomen | MRSegmentator, VIBESegmentator |
| 001 | MRI | abdomen_pelvis | MRSegmentator, VIBESegmentator |
| 002 | MRI | chest | MRSegmentator, VIBESegmentator |

### Decision Trace

The report (`tests/results/e2e_report.json`) tracks every pipeline decision per case:
- **Metadata**: modality, anatomy, is_diagnostic
- **Tool selection**: which segmentation tools and why
- **QC severity**: per-tool Tier 1 results, cross-tool Tier 2 agreement
- **QC interpretation**: LLM clinical assessment (or rule-based fallback)
- **Radiomics gating**: which organs are safe for feature extraction
- **Radiomics results**: organs extracted, feature counts, backend used
- **Timing**: per-step and total elapsed time

### Validated Results (2026-03-20, no-LLM mode)

**String pipeline: 6/6 modality correct** (100% accuracy on path heuristics)

| Case | Modality | Anatomy | Organs in FOV |
|------|----------|---------|---------------|
| 0003 | CT | abdomen_pelvis | 22 |
| 0004 | CT | chest_abdomen_pelvis | 29 |
| 0005 | CT | abdomen | 20 |
| 0002 | MRI | abdomen | 20 |
| 001 | MRI | abdomen_pelvis | 22 |
| 002 | MRI | chest | 14 |

**Image pipeline: 6/6 completed, 290 organs × 93 features = 26,970 radiomic features**

| Case | Mod | QC | Tools OK | Radiomics Organs | Time |
|------|-----|------|----------|-----------------|------|
| 0003 | CT | WARN | 1/2 | 74 | 326s |
| 0004 | CT | WARN | 2/2 | 54 | 85s |
| 0005 | CT | WARN | 1/2 | 75 | 196s |
| 0002 | MRI | FAIL | 2/2 | 27 | 85s |
| 001 | MRI | WARN | 2/2 | 35 | 80s |
| 002 | MRI | FAIL | 2/2 | 25 | 65s |

Notes:
- QC WARN/FAIL is expected — dummy masks have geometric irregularities by design
- VISTA3D `_dry_run` fails on cases without pre-existing masks (non-blocking; uses TotalSeg masks)
- MRI cases show FAIL severity because MRSeg/VIBESeg masks on CT-derived dummy data have expected mismatches
- Radiomics fallback gating extracts all PASS+WARN organs (no LLM → conservative rule-based)

## Changelog

| Date | Update |
|------|--------|
| 2026-03-18 | Milestone 1: Directory skeleton, constants, base tool, tool registries |
| 2026-03-18 | QC pipeline v1: geometric checks (volume, CC, ratio, overlap) |
| 2026-03-18 | HU calibration audit across 24k NIfTIs |
| 2026-03-18 | MedGemma-27B validated as planner (metadata, tool selection, QC interpretation) |
| 2026-03-20 | All 9 segmentation tool wrappers implemented (`tools/`) |
| 2026-03-20 | All tool-specific conda envs set up and tested (cdw_totalseg through cdw_textmedseg) |
| 2026-03-20 | Blackwell GPU fix: `nvidia-cusparselt-cu12==0.7.1 nvidia-nvjitlink-cu12==12.8.93` |
| 2026-03-20 | BiomedParse3D skipped: detectron2 CUDA build incompatible with Blackwell driver |
| 2026-03-20 | SAT-Pro checkpoint (`Pro/SAT_Pro.pth`) downloaded; `textmedseg3d.py` auto-selects Pro |
| 2026-03-20 | All 7 active tools live-tested on 2 CT + 2 MRI cases via `run_dummy_test.py` |
| 2026-03-20 | `processing/postprocessing.py`: LCC, morphological ops, STAPLE/majority-vote fusion |
| 2026-03-20 | `processing/format_utils.py`: canonical per-organ mask output (save/load/split-multilabel) |
| 2026-03-20 | `scripts/setup_envs.sh`, `download_checkpoints.sh`, `docs/installation.md` |
| 2026-03-20 | `planner/llm_client.py`: TransformersLLM + VLLMClient + PlannerLLM (JSON extraction, think-block stripping) |
| 2026-03-20 | `planner/metadata_extractor.py`, `tool_selector.py`, `organ_list_generator.py` |
| 2026-03-20 | `llms/`: BaseLLM ABC + LLMResponse; MedGemma27bText; Qwen3 unified client (8B/32B, local/api) |
| 2026-03-20 | `llms/utils.py`: transformers 5.x compat — `dtype=` kwarg, `_unwrap_input_ids` for BatchEncoding |
| 2026-03-20 | `cdw_llm` conda env: Python 3.11, torch 2.10+cu128, transformers 5.3 |
| 2026-03-20 | Checkpoints downloaded locally: MedGemma-27B, Qwen3-8B, Qwen3-32B |
| 2026-03-20 | LLM dry run validated: all 3 models × 4 prompt types on real CDW file paths |
| 2026-03-20 | `radiomics/pyradiomics.py`: CPU extractor (primary), RadiomicsResult, sys.path collision fix for `radiomics` name shadow |
| 2026-03-20 | `radiomics/curadiomics.py`: GPU wrapper (optional) — TF1/TF2 compat, per-slice→volume aggregation |
| 2026-03-20 | `config/radiomics_params.yaml`: PyRadiomics config — 1mm iso resampling, 5 feature classes, 93 features total |
| 2026-03-20 | `cdw_radiomics` env: Python 3.10, PyRadiomics from GitHub (`pip install git+...`), SimpleITK, nibabel, pandas |
| 2026-03-20 | `qc/` 3-tier QC: dataclasses, GeometricQC (Tier 1), MultiToolQC (Tier 2), QCInterpreter (Tier 3 LLM + fallback) |
| 2026-03-20 | Bugfix: `RADIOMICS_GATING` prompt contains literal `{organ, ...}` braces — switched to `.replace()` in qc_interpreter + tests |
| 2026-03-20 | `tests/test_qc.py`: 22 unit tests across all 3 QC tiers — synthetic NIfTI, no real data, no LLM required |
| 2026-03-20 | `tests/test_llm_dry_run.py` extended: added QC_INTERPRETATION + RADIOMICS_GATING to Qwen3-8B test suite |
| 2026-03-20 | Qwen3-8B validated across all 5 CDW prompt types — correct outputs, clinically sensible radiomics gating decisions |
| 2026-03-20 | PyRadiomics smoke-tested on dummy data: liver + kidneys, 93 features per organ in ~1-2s |
| 2026-03-20 | cuRadiomics demoted to optional (TF dependency, 2D-only, fewer features) — PyRadiomics is primary backend |
| 2026-03-20 | `radiomics/__init__.py`: lazy imports to fix pyradiomics naming collision |
| 2026-03-20 | `orchestrator/case_tracker.py`: JSON-backed persistent state, thread-safe, atomic writes, resume support |
| 2026-03-20 | `orchestrator/pipeline.py`: CasePipeline — full single-case flow (metadata → seg → postprocess → QC → radiomics) |
| 2026-03-20 | `orchestrator/batch_runner.py`: BatchRunner — GPU pool, ThreadPoolExecutor, CSV aggregation, CLI |
| 2026-03-20 | Orchestrator dry-run validated on dummy data (2 cases, 2 tools, Tier 1+2 QC, resume tested) |
| 2026-03-20 | **Two-model architecture validated**: Qwen3-8B (5/5 is_diagnostic edge cases) for structured tasks; MedGemma-27B (9/10) for clinical interpretation |
| 2026-03-20 | `tests/test_llm_dry_run.py`: added `test_is_diagnostic_edge_cases()` — 5 hard cases × 3 models, `--edge-cases-only` flag, per-model scoring table |
| 2026-03-20 | `tests/run_e2e.py`: E2E test harness — string + image modes, 3 CT + 3 MRI, full decision trace to `e2e_report.json` |
| 2026-03-20 | Pipeline fixes: anatomy normalization (ABD→abdomen), pyradiomics import caching, VISTA3D _dry_run signature, metadata_override for testing |
| 2026-03-20 | **E2E validated**: 6/6 string (100% modality accuracy), 6/6 image (290 organs, 26,970 radiomics features), full decision trace |
| 2026-03-20 | Architecture updated: Qwen3-8B is now the planner for metadata/tool selection; MedGemma-27B reserved for QC interpretation + radiomics gating |
