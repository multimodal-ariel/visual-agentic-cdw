# CDW Agentic Pipeline — Testing Helpers

**Date**: 2026-03-25

Base environment: **`cdw_radiomics`** for all commands below. Individual segmentation tools run in their own conda envs via subprocess (`conda run -n cdw_totalseg`, etc.) — you only need `cdw_radiomics` activated.

```bash
conda activate cdw_radiomics
```

---

## 0. Test Directory Setup (preprocessing)

The pipeline writes tool outputs **into** `case_path` directly (`seg_dir = os.path.join(case_path, output_dirname)`). There is no separate `--output-dir` flag. This script creates flattened directories under `OUTPUT_PATH_TESTING`, symlinks each `image_nifti.nii.gz`, and writes a remapped filelist.

**Edit `OUTPUT_PATH_TESTING` and `FILELIST_TESTING` before running.**

```bash
python3 -c "
import json, os

OUTPUT_PATH_TESTING = '/data/soumitri/testing_output'       # ← CHANGE THIS
FILELIST_TESTING = '/data/soumitri/new_data_paths/test_cases.json'  # ← CHANGE THIS

with open(FILELIST_TESTING) as f:
    paths = json.load(f)
    if isinstance(paths, dict):
        paths = paths['cases']

new_paths = []
for p in paths:
    # Flatten: /data/RAD/RHEUM/PT123/CT_ABD → _data_RAD_RHEUM_PT123_CT_ABD
    flat = p.strip('/').replace('/', '_')
    dest = os.path.join(OUTPUT_PATH_TESTING, flat)
    os.makedirs(dest, exist_ok=True)

    # Symlink image_nifti.nii.gz from original location
    src_image = os.path.join(p, 'image_nifti.nii.gz')
    dst_image = os.path.join(dest, 'image_nifti.nii.gz')
    if os.path.exists(src_image) and not os.path.exists(dst_image):
        os.symlink(src_image, dst_image)
        print(f'  linked: {src_image} -> {dst_image}')

    new_paths.append(dest)

out_filelist = os.path.join(OUTPUT_PATH_TESTING, 'filelist_testing_remapped.json')
with open(out_filelist, 'w') as f:
    json.dump(new_paths, f, indent=2)
print(f'\nWrote {len(new_paths)} paths to {out_filelist}')
"
```

After this, use `filelist_testing_remapped.json` for all subsequent commands. The remapped paths don't start with `SRC_PREFIX` (`/data/RAD/`), so `batch_runner._load_filelist()` won't re-remap them.

---

## 1.(a) Metadata Extraction using heuristic

Tests the path-based heuristic (hard-coded) that extracts modality, anatomy, is_diagnostic.

```bash
python -c "
from orchestrator.pipeline import CasePipeline, CaseResult
from config.constants import IMAGE_FILENAME, FILELIST_TESTING
from pathlib import Path
import json, os

pipeline = CasePipeline(dry_run=True, no_llm=True)
with open(FILELIST_TESTING) as f:
    cases = json.load(f)

for case_path in cases:
    case_id = Path(case_path).name
    image_path = os.path.join(case_path, IMAGE_FILENAME)
    result = CaseResult(case_id=case_id, case_path=case_path)
    meta = pipeline._step_metadata(case_path, image_path, result)
    print(f'{case_path}: {meta}')
"
```

## 1.(b) Metadata Extraction using LLM Planner (Actual Pipeline should have this)

Tests the LLM-driven modality, anatomy, is_diagnostic extraction

```bash
python -c "
from planner.metadata_extractor import MetadataExtractor
from planner import PlannerLLM
import json, os
from config.constants import IMAGE_FILENAME, FILELIST_TESTING

planner = PlannerLLM.from_local('checkpoints/qwen3-8b')
planner.load()
extractor = MetadataExtractor(planner)

with open(FILELIST_TESTING) as f:
    cases = json.load(f)

for case_path in cases:
    meta = extractor.extract(os.path.join(case_path, IMAGE_FILENAME))
    print(f'{case_path}: {meta}')
"
```

---

## 2. Tool Selection

Tests which segmentation tools get selected for a given case's metadata.

```bash
python -c "
from planner.metadata_extractor import MetadataExtractor
from planner.tool_selector import ToolSelector
from config.constants import IMAGE_FILENAME, FILELIST_TESTING
from pathlib import Path
import json, os

from planner import PlannerLLM
planner = PlannerLLM.from_local('checkpoints/qwen3-8b')
planner.load()
extractor = MetadataExtractor(planner)
tool_selector = ToolSelector(planner)

with open(FILELIST_TESTING) as f:
    cases = json.load(f)

for case_path in cases:
    meta = extractor.extract(os.path.join(case_path, IMAGE_FILENAME))
    tools = tool_selector.select(meta)
    print(case_path)
    print(f'{meta}; Selected tools: {tools}')
    print()
"
```

---

## 3. Single-Case Dry Run (full pipeline, mock segmentation)

Runs the entire pipeline for one case with mock segmentation outputs and rule-based fallbacks (no LLM, no GPU).

```bash
/home/soumitri/env/miniconda3/bin/conda run -n cdw_llm python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

CASE = '/data/soumitri/test_pipeline_2/data_soumitri_segmentations_3d_LUPUS_87B83317A40322D15E6BD16FDF2B126B_20241007_CT_ABDOMEN_PELVIS_WO_CONTRAST_Abdomen_Pelvis_3mm_Axial_ST_iMAR'

from planner import PlannerLLM
planner_llm = PlannerLLM.from_local('checkpoints/qwen3-8b')
planner_llm.load()
clinical_llm = PlannerLLM.from_local('checkpoints/medgemma-27b-text-it')
clinical_llm.load()

from orchestrator.pipeline import CasePipeline
pipeline = CasePipeline(planner_llm=planner_llm, clinical_llm=clinical_llm, skip_radiomics=False)
result = pipeline.run(CASE)  # ← CHANGE THIS
print(f'Tools run: {result.selected_tools}')
print(f'Tool outputs: {[t[\"tool_name\"] for t in result.tool_outputs]}')
print(f'Status: {result.status}')
print(f'QC severity: {result.qc_report.get(\"overall_severity\", \"N/A\")}')
print(f'Time: {result.total_time_s}s')
"
```

---

## 4. QC Only (on existing segmentations)

Runs Tier 1 (geometric per-tool) and Tier 2 (cross-tool agreement) QC on segmentation directories that already exist in a case directory.

```bash
/home/soumitri/env/miniconda3/bin/conda run -n cdw_llm python -c "
from qc.geometric_qc import GeometricQC
from qc.multi_tool_qc import MultiToolQC
import os

case_path = '/path/to/flattened/case'  # ← CHANGE THIS
seg_dirs = {}
for d in os.listdir(case_path):
    if d.startswith('segmentations_') and os.path.isdir(os.path.join(case_path, d)):
        seg_dirs[d] = os.path.join(case_path, d)

# Tier 1: per-tool geometric QC
gqc = GeometricQC()
for tool_name, seg_dir in seg_dirs.items():
    result = gqc.run(case_path, seg_dir, tool_name)
    print(f'{tool_name}: {result.severity} ({result.num_flagged} flagged)')

# Tier 2: cross-tool agreement (needs >= 2 tools)
if len(seg_dirs) >= 2:
    mqc = MultiToolQC()
    result = mqc.run(case_path, seg_dirs)
    print(f'Mean Dice: {result.mean_dice:.3f}, Disagreements: {result.organs_with_disagreement}')
else:
    print('Skipping Tier 2: need >= 2 segmentation directories')
"
```

---

## 5. STAPLE Consensus (on existing segmentations)

Generates STAPLE consensus masks for organs present in >= 2 tools. Outputs to `segmentations_consensus/` inside the case directory.

```bash
python -m processing.staple_consensus \
    --case /path/to/flattened/case \
    --method staple \
    --min-tools 2 \
    -v
```

Majority vote alternative:

```bash
python -m processing.staple_consensus \
    --case /path/to/flattened/case \
    --method majority_vote \
    --min-tools 2 \
    -v
```

---

## 6. Radiomics Extraction (on existing segmentations)

Extracts pyradiomics features for specific organs from a single tool's segmentation output.

```bash
python -c "
from radiomics.feature_extractor import extract_features

result = extract_features(
    image_path='/path/to/flattened/case/image_nifti.nii.gz',
    seg_dir='/path/to/flattened/case/segmentations_totalseg_ct',
    organs=['liver', 'spleen', 'kidney_left'],
)
print(f'Extracted features for {len(result)} organs')
for organ, features in result.items():
    print(f'  {organ}: {len(features)} features')
"
```

---

## 7. End-to-End Batch Runner

### Dry run (no GPU, no LLM, mock segmentation)

```bash
python -m orchestrator.batch_runner \
    --filelist /path/to/filelist_testing_remapped.json \
    --gpus 0 --workers 1 \
    --dry-run --no-llm
```

### Real segmentation, no LLM (rule-based fallbacks)

```bash
python -m orchestrator.batch_runner \
    --filelist /path/to/filelist_testing_remapped.json \
    --gpus 0,1,2,3,4,5,6,7 --workers 8 \
    --no-llm
```

### Full pipeline with LLMs

```bash
python -m orchestrator.batch_runner \
    --filelist /path/to/filelist_testing_remapped.json \
    --gpus 0,1,2,3,4,5,6,7 --workers 8
```

### With STAPLE consensus enabled

```bash
python -m orchestrator.batch_runner \
    --filelist /path/to/filelist_testing_remapped.json \
    --gpus 0,1,2,3,4,5,6,7 --workers 8 \
    --no-llm --consensus
```

### Retry previously failed cases

```bash
python -m orchestrator.batch_runner \
    --filelist /path/to/filelist_testing_remapped.json \
    --gpus 0,1,2,3,4,5,6,7 --workers 8 \
    --no-llm --retry-failed
```

---

## Notes

- **`/path/to/filelist_testing_remapped.json`** and **`/path/to/flattened/case`** are placeholders — replace with your actual paths after running Section 0.
- Pipeline state is saved to `logs/pipeline_state.json` — delete it to start fresh, or use `--retry-failed` to reprocess failures.
- Aggregate results CSV goes to `logs/pipeline_results.csv`.
- Each segmentation tool runs in its own conda env via subprocess. If a tool's env is missing, it will fail gracefully and the pipeline continues with the remaining tools.
