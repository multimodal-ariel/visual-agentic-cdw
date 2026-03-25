# STAPLE Consensus & Organ Name Normalization

## Overview

When the pipeline runs multiple segmentation tools on the same case, each tool produces independent organ masks. **STAPLE consensus** fuses these masks into a single higher-quality segmentation per organ, weighting each tool by its estimated reliability.

**Organ name normalization** is a prerequisite — tools use different naming conventions for the same organ, and names must match before masks can be fused.

---

## Organ Name Normalization

### The Problem

Different tools use inconsistent naming for the same organ:

| Organ | TotalSegmentator | VISTA3D | VIBESegmentator |
|-------|-----------------|---------|-----------------|
| Left kidney | `kidney_left` | `left_kidney` | `kidney_left` |
| Left adrenal | `adrenal_gland_left` | `left_adrenal_gland` | `adrenal_gland_left` |
| Small bowel | `small_bowel` | `small_bowel` | `intestine` |
| Intervertebral discs | (individual) | `intervertebral_discs` | `IVD` |
| Urinary bladder | `urinary_bladder` | `bladder` | `urinary_bladder` |

Without normalization, `kidney_left` and `left_kidney` are treated as different organs — they never get compared in cross-tool QC and never get fused in STAPLE.

### The Solution

`processing/format_utils.py` provides `normalize_organ_name(name)` which:

1. **Lowercases** everything (`vertebrae_C4` → `vertebrae_c4`)
2. **Normalizes separators** (spaces/hyphens → underscores)
3. **Swaps directional prefixes to suffixes** — the TotalSegmentator/majority convention:
   - `left_kidney` → `kidney_left`
   - `left_adrenal_gland` → `adrenal_gland_left`
   - `left_rib_10` → `rib_left_10` (special case: ribs keep `rib_{side}_{number}`)
4. **Resolves synonyms**:
   - `intestine` → `small_bowel`
   - `IVD` / `ivd` → `intervertebral_discs`
   - `bladder` → `urinary_bladder`
   - `spinal_channel` → `spinal_canal`

### Impact

After normalization, cross-tool organ matching on CT case 0004:

| Tool pair | Shared organs (before) | Shared organs (after) |
|-----------|----------------------|---------------------|
| TotalSeg CT ∩ VISTA3D | ~50 | **81** |
| TotalSeg CT ∩ VIBESeg | ~50 | **55** |
| MRSeg ∩ VISTA3D | ~18 | **23** |

### Where it's applied

- `qc/multi_tool_qc.py` — `_load_tool_masks()` normalizes names before pairwise Dice/volume comparison
- `processing/staple_consensus.py` — `build_normalized_mask_index()` normalizes before STAPLE fusion
- Both use the same canonical form, so QC metrics and consensus are consistent

### Adding new synonyms

Edit `_ORGAN_SYNONYMS` dict in `processing/format_utils.py`:

```python
_ORGAN_SYNONYMS: Dict[str, str] = {
    "intestine": "small_bowel",
    "ivd": "intervertebral_discs",
    "bladder": "urinary_bladder",
    "spinal_channel": "spinal_canal",
    # Add new entries here:
    # "some_tool_name": "canonical_name",
}
```

---

## STAPLE Consensus

### What is STAPLE?

**STAPLE** (Simultaneous Truth and Performance Level Estimation) by Warfield, Zou & Wells (2004) is a probabilistic algorithm that takes multiple binary segmentations of the same structure and produces:

1. **A consensus segmentation** — statistically optimal fusion, not simple averaging
2. **Per-rater sensitivity/specificity estimates** — how reliable each tool is

### How it works (EM algorithm)

```
Initialize: assume all raters have equal sensitivity (0.99) and specificity (0.99)

Repeat until convergence:
    E-step: For each voxel, compute P(true foreground | all rater labels, current sensitivity/specificity)
    M-step: Re-estimate each rater's sensitivity and specificity from the current truth estimate

Output: probability map → threshold at 0.5 → binary consensus mask
```

**Key difference from majority vote**: STAPLE weights tools by their estimated accuracy. If TotalSegmentator consistently agrees with other tools while VoxTell is noisier, STAPLE upweights TotalSeg automatically.

### Backend: SimpleITK

The pipeline uses **SimpleITK STAPLE** (`SimpleITK.STAPLE()`), which is already installed in `cdw_radiomics`. Key detail: SimpleITK STAPLE requires `uint16` input for 3D volumes (not `float32`).

Fallback chain:
1. **SimpleITK** (preferred — lightweight, already a dependency)
2. **ITK** (if installed — heavier but more configurable)
3. **Probabilistic mean** (pure numpy — equivalent to majority vote for binary masks)

### Performance

Tested on CT case 0004 (203×203×280 volumes):

| Organ | Tools fused | Consensus voxels | Time |
|-------|-------------|-----------------|------|
| liver | 2 (TotalSeg, MRSeg) | 158,034 | 3.0s |
| spleen | 2 (TotalSeg, MRSeg) | 17,257 | 4.0s |

Full case (68 organs from 5 tools): ~5-10 minutes total.

---

## Usage

### Standalone CLI

```bash
# Generate STAPLE consensus for a case
conda run -n cdw_radiomics python -m processing.staple_consensus \
    --case dummy_outputs/0004 \
    --method staple \
    --min-tools 2 \
    -v

# Majority vote instead of STAPLE
conda run -n cdw_radiomics python -m processing.staple_consensus \
    --case dummy_outputs/0004 \
    --method majority_vote
```

### Python API

```python
from processing.staple_consensus import generate_case_consensus

result = generate_case_consensus(
    case_path="dummy_outputs/0004",
    method="staple",    # or "majority_vote"
    min_tools=2,        # minimum tools per organ to generate consensus
)

print(result["consensus_dir"])   # segmentations_consensus/
print(result["organs_fused"])    # ['aorta', 'kidney_left', 'liver', ...]
print(result["per_organ"])       # [{organ, num_tools, tools, voxel_count}, ...]
```

### In the pipeline

```bash
# Enable STAPLE consensus in batch pipeline (opt-in)
python -m orchestrator.batch_runner \
    --filelist cases.json \
    --gpus 0,1,2,3,4,5,6,7 --workers 8 \
    --consensus

# Python
pipeline = CasePipeline(consensus=True, consensus_method="staple")
result = pipeline.run(case_path)
```

When `--consensus` is enabled:
1. After Tier 2 QC, STAPLE runs on all organs present in ≥2 tools
2. Consensus masks are saved to `segmentations_consensus/`
3. Radiomics prefers consensus masks over single-tool masks

**Default: OFF** — existing pipeline behavior is completely unchanged unless you explicitly pass `--consensus`.

### Output structure

```
case_dir/
├── image_nifti.nii.gz
├── segmentations_totalseg_ct/
│   ├── liver.nii.gz
│   ├── kidney_left.nii.gz
│   └── ...
├── segmentations_vista3d/
│   ├── liver.nii.gz
│   ├── left_kidney.nii.gz    # VISTA3D naming
│   └── ...
├── segmentations_consensus/   # ← NEW (only when --consensus)
│   ├── liver.nii.gz           # STAPLE fusion of all tools
│   ├── kidney_left.nii.gz     # canonical names
│   ├── manifest.json          # per-organ: tools used, method, voxel count
│   └── ...
```

---

## Architecture

```
Multi-tool segmentation outputs
    │
    ▼
┌──────────────────────────┐
│  Organ Name Normalization │  ← normalize_organ_name()
│  (format_utils.py)        │     left_kidney → kidney_left
└───────────┬──────────────┘     intestine → small_bowel
            │
            ▼
┌──────────────────────────┐
│  Cross-Tool Matching      │  ← build_normalized_mask_index()
│  (staple_consensus.py)    │     canonical name → file path
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  STAPLE Fusion            │  ← SimpleITK.STAPLE(uint16 masks)
│  (postprocessing.py)      │     EM algorithm → probability map → threshold
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│  Consensus Masks          │  ← segmentations_consensus/{organ}.nii.gz
│  (per-organ NIfTI)        │     + manifest.json
└──────────────────────────┘
```

---

## Files

| File | Role |
|------|------|
| `processing/format_utils.py` | `normalize_organ_name()`, `build_normalized_mask_index()` |
| `processing/staple_consensus.py` | `generate_case_consensus()`, CLI entrypoint |
| `processing/postprocessing.py` | `staple_fusion()`, `majority_vote()` — low-level fusion |
| `qc/multi_tool_qc.py` | Uses normalization in `_load_tool_masks()` for cross-tool Dice |
| `orchestrator/pipeline.py` | `_step_consensus()` — opt-in via `consensus=True` |
| `orchestrator/batch_runner.py` | `--consensus` CLI flag |

## Dependencies

- **SimpleITK** — already in `cdw_radiomics` (provides STAPLE filter)
- **nibabel** — already in all envs (NIfTI I/O)
- **numpy** — already in all envs (array ops)
- **ITK** (optional) — installed in `cdw_radiomics` as fallback
