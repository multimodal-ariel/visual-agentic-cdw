# CDW: Agentic AI Clinical Imaging Pipeline

End-to-end, LLM-orchestrated processing for uncurated clinical CT/MRI:

```metadata → planning → tool selection → segmentation → QC → radiomics.``` 

All heavy models live in tool-specific conda environments and external submodules; only code is tracked here.

## Authors

#### Soumitri Chattopadhyay, Basar Demir, Yinzhu Jin, Marc Niethammer
#### UCSD Biomedical Image Analysis Group

## High-level flow
1) **Metadata + planning (Qwen3-8B)**: modality, anatomy, shape, is_diagnostic → tool list + organ prompts.
2) **Segmentation (per-tool envs)**: TotalSeg CT/MR, MRSegmentator, MRISegmenter, VISTA3D, VIBESeg, VoxTell, TextMedSeg3D; runners isolate deps.
3) **Postprocess**: LCC, closing, hole fill, optional STAPLE/majority fusion.
4) **QC (3 tiers)**: geometric checks → multi-tool agreement → LLM clinical interpretation (MedGemma-27B) + radiomics gating.
5) **Radiomics**: PyRadiomics CPU (primary); cuRadiomics optional.

## Repo layout (essentials)
- `config/` paths, tool registry, QC priors, PyRadiomics params, prompts.
- `tools/` wrappers + subprocess runners (one env per tool).
- `planner/` LLM planner (metadata, tool selection, organ lists).
- `orchestrator/` pipeline + batch runner + case tracker.
- `processing/` preprocessing, postprocessing, format helpers.
- `qc/` geometric QC, multi-tool QC, LLM interpretation.
- `radiomics/` PyRadiomics + optional cuRadiomics.
- `scripts/` env setup, checkpoint downloads, dummy/e2e runners, submodule helper.
- `external/` tool codebases (git submodules); `checkpoints/` weights (ignored).

## Key tools & envs (conda)
- `cdw_totalseg`: TotalSegmentator CT/MR
- `cdw_mrseg`: MRSegmentator
- `cdw_mriseg`: MRISegmenter
- `cdw_nvseg`: VISTA3D (NV-Segment-CTMR)
- `cdw_vibeseg`: VIBESegmentator
- `cdw_voxtell`: VoxTell
- `cdw_textmedseg`: TextMedSeg3D / SAT-Pro
- `cdw_llm`: MedGemma-27B, Qwen3-8B/32B
- `cdw_radiomics`: PyRadiomics (primary), cuRadiomics optional

## Quick start
```bash
# 1) Envs
bash scripts/setup_envs.sh

# 2) Weights (requires HF auth for MedGemma)
bash scripts/download_checkpoints.sh

# 3) Edit paths
vim config/constants.py

# 4) Smoke tests
python scripts/run_dummy_test.py            # runs tool wrappers on dummy data
conda run -n cdw_radiomics python tests/run_e2e.py --mode string  # planner-only
conda run -n cdw_radiomics python tests/run_e2e.py --mode image   # full on dummy

# 5) Batch (production example)
/home/soumitri/env/miniconda3/bin/conda run -n cdw_llm python -m orchestrator.batch_runner --filelist /data/soumitri/test_pipeline_new_2_llm/filelist_testing_remapped.json --gpus 0 --skip-radiomics --consensus
```

Full server bring-up (multi-GPU) is summarized in `docs/deployment.md`.

## Running LLMs
- Planner (metadata/tool selection): Qwen3-8B (fast, rule-following).
- QC interpretation + radiomics gating: MedGemma-27B (clinical domain).
- Optional vLLM server supported via `planner/llm_client.py`.

## Data & outputs
- Inputs expected under `DICOM_ROOT` / `SEGMENTATION_ROOT` (set in `config/constants.py`).
- Per-case outputs: segmentations per tool, QC CSV/JSON, radiomics features; logs under `logs/`.

## Git / submodules / weights
- Code only is tracked; `checkpoints/`, data, and outputs are ignored.
- Tool repos live in `external/` as submodules (e.g., detectron2, TotalSegmentator, VISTA3D, VoxTell, etc.).
- After clone: `git submodule update --init --recursive`.

## Tests
- Unit: `tests/test_base_tool.py`, `tests/test_qc.py`, `tests/test_llm_dry_run.py`.
- E2E: `tests/run_e2e.py` (string/image/full modes) on 3 CT + 3 MRI dummy cases with full decision trace → `tests/results/e2e_report.json`.
- Latest results (2026-03-23): 6/6 string (100% modality), 6/6 image (89 organs × 107 features = 9,523), **163/163 ground-truth validation checks ALL PASS**.

## Notes
- One unified env will not work; always invoke via `conda run -n <env>` per tool.
- BiomedParse3D is skipped (detectron2 CUDA fragility); VoxTell + SAT-Pro cover 3D text-prompted use cases.
- MRI geometric QC skips volume/ratio checks (CT-calibrated); relies on CC + overlap + multi-tool agreement.
- Detailed pipeline docs: see `PIPELINE.md`.
