# CDW Agentic Pipeline — Critical Analysis & Future Direction

**Date**: 2026-03-25
**Authors**: Soumitri Chattopadhyay, Claude (Opus 4.6)

A self-critical examination of the pipeline's LLM integration, whether it qualifies as "agentic," and concrete proposals for genuine agentic behavior.

---

## 1. Current LLM Roles — Honest Assessment

| Step | LLM | What it does | Has fallback? | LLM adds value? |
|------|-----|-------------|---------------|-----------------|
| Metadata extraction | Qwen3-8B | Path → modality, anatomy, is_diagnostic | Yes — path heuristic (100% on test cases) | **Marginal** — heuristic works for structured paths |
| Tool selection | Qwen3-8B | Metadata → tool list | Yes — registry lookup (identical output) | **Zero** — prompt forces all tools anyway |
| Organ list (text tools) | Qwen3-8B | Anatomy → organ names for VoxTell/SAT | Yes — organ_reference.json lookup | **Moderate** — helps for unusual anatomies |
| QC interpretation | MedGemma-27B | QC flags → clinical assessment | Yes — rule-based severity mapping | **Moderate** — adds clinical nuance |
| Radiomics gating | MedGemma-27B | QC results → extract/skip per organ | Yes — rule-based LCC gating | **Moderate** — can distinguish pathology from artifact |

### Detailed Breakdown

**Metadata Extraction (Qwen3-8B)**

The path-based heuristic (`_metadata_from_path()` in `pipeline.py`) achieves 100% accuracy on all 6 test cases. For well-structured DICOM paths like `/data/RAD/RHEUM/PT123/CT_ABD/image_nifti.nii.gz`, the heuristic trivially extracts modality=CT, anatomy=abdomen. The LLM is redundant here unless paths are ambiguous or unstructured. In production on 26k cases from RHEUM/LUPUS/CONTR cohorts, the paths are consistently structured — the LLM adds no value over string parsing.

**Tool Selection (Qwen3-8B)**

This is the most egregious case of LLM theater. The `TOOL_SELECTION` prompt in `config/prompts/example_prompts.py` contains mandatory rules:

```
Selection rules (MANDATORY — follow these exactly):
1. Put ALL modality-compatible fixed-class tools in primary_tools.
2. For CT: primary_tools MUST include TotalSegmentator_CT, MRSegmentator, VISTA3D
3. For MRI: primary_tools MUST include TotalSegmentator_MR, MRSegmentator, MRISegmenter, VIBESegmentator, VISTA3D
```

The LLM literally cannot make a different decision than the hardcoded fallback. And even if it tried, `pipeline.py:_all_compatible_tools()` unions the LLM response with the full registry. The `_fallback()` function in `tool_selector.py` produces the exact same output without any LLM call. This is a rubber stamp, not a decision.

**Organ List Generation (Qwen3-8B)**

This is the most genuinely useful LLM role. When anatomy is unusual or not in `organ_reference.json`, the LLM can reason about what structures VoxTell and TextMedSeg3D should target. The `_ORGAN_LIST_PROMPT` asks the LLM to identify structures "NOT already covered by the fixed-class tools" — a real reasoning task. However, for standard anatomies (abdomen, chest, abdomen_pelvis), the lookup table works fine. The LLM adds value primarily for edge cases.

**QC Interpretation (MedGemma-27B)**

The rule-based fallback maps severity levels mechanically: PASS→GOOD, WARN→ACCEPTABLE, FAIL→POOR. MedGemma adds nuance — for example, it can differentiate "pancreas fragmentation is likely real pathology in a lupus patient" from "pancreas fragmentation means bad segmentation." It can also make per-organ feature-class safety decisions (e.g., "shape features unreliable for this liver, but first_order is fine"). The rules can't do this. However, in `--no-llm` mode, the severity-aware fallback gating (which uses LCC fraction to gate feature classes) produces reasonable results for most cases.

**Radiomics Gating (MedGemma-27B)**

Similar to QC interpretation — the LLM can make clinically informed extract/skip decisions. The rule-based fallback (PASS→full, WARN→severity-aware, FAIL→skip) works for typical cases. MedGemma adds value on edge cases where clinical context matters.

---

## 2. Is This Pipeline Truly Agentic?

**No. Not in any meaningful sense.**

### Criteria for an agentic system

An agentic system has:
1. **Autonomy**: makes decisions that alter the execution flow
2. **Closed-loop reasoning**: observes results and adapts behavior
3. **Tool use driven by judgment**: selects and invokes tools based on its own reasoning
4. **Error recovery**: diagnoses failures and tries alternative approaches

### What the pipeline actually is

The pipeline is a **fixed-order directed acyclic graph (DAG) with LLM garnish**:

```
metadata → tools → segmentation → QC → radiomics
```

Every step always runs in the same order. The LLM never:
- **Observes output and decides to re-run** — if TotalSeg fragments the pancreas, it doesn't say "let me try VISTA3D specifically for pancreas"
- **Adapts strategy mid-case** — it doesn't adjust postprocessing based on what it sees
- **Chooses between alternative approaches** — tool selection is hardcoded to "run everything"
- **Recovers from errors** — if a tool crashes, it logs a warning and moves on silently
- **Learns from previous cases** — each case is processed independently with no memory

### The smoking gun

The `--no-llm` mode produces **nearly identical results** on all 6 E2E test cases (6/6 completed, 163/163 validation checks pass). If removing the LLM changes nothing, the LLM isn't making real decisions.

### What IS somewhat agentic (small wins)

1. **Organ list generation** — the LLM reasons about anatomy to decide what structures VoxTell/TextMedSeg3D should target. This is real (small) agency — the decision genuinely affects what gets segmented.

2. **QC clinical nuance** — a rule says "fragmented pancreas = WARN", but MedGemma can say "fragmentation in a lupus patient's kidneys may reflect real pathology, not segmentation error — extract features cautiously." Rules can't make this distinction.

3. **Radiomics gating edge cases** — the LLM can differentiate "this liver has 3 connected components because of a portal vein artifact" vs "because the mask is garbage." The rule-based fallback treats all fragmentation the same.

### What is pure theater

1. **Tool selection** — the prompt says "MUST include ALL tools." The `_fallback()` function produces the exact same output. The LLM is a rubber stamp.

2. **Metadata extraction on structured paths** — string parsing gets 100% accuracy. The LLM call costs ~1 GPU-second per case for zero marginal benefit.

---

## 3. What Would Make This Genuinely Agentic

### Proposal 1: Closed-Loop Segmentation Refinement (highest impact)

**Current behavior**: Run all tools → QC flags problems → log them → move on.

**Agentic behavior**:
```
Run TotalSeg → QC finds pancreas fragmented (3 CCs, LCC fraction 0.4)
    │
    LLM reasons: "Pancreas fragmentation with LCC < 0.5 suggests
    segmentation failure, not pathology. TotalSeg is known to
    struggle with small pancreas in low-contrast CT."
    │
    LLM decides: "Re-run pancreas with VISTA3D (345 classes,
    better at small organs). If still fragmented, try morphological
    closing before declaring failure."
    │
    Run VISTA3D (pancreas only) → QC passes → use VISTA3D mask
```

This is a **genuine agent loop**: observe → reason → act → observe again. The current pipeline can't do this because it runs all tools upfront and never revisits decisions.

**Implementation complexity**: Medium. Requires per-organ re-run infrastructure and a feedback prompt that gives the LLM QC results + asks for corrective actions. The segmentation tools already support `target_organs` for selective re-runs.

### Proposal 2: Adaptive Tool Selection (saves 30-50% compute)

**Current behavior**: Run 6-7 tools per case regardless.

**Agentic behavior**:
```
LLM sees: CT abdomen, low contrast, 512x512x237
    │
    LLM reasons: "For standard CT abdomen, TotalSeg covers
    117 structures. MRSegmentator adds only 5 unique organs
    (all already covered by TotalSeg). VIBESeg adds IVD and
    body composition but those aren't clinically relevant here."
    │
    LLM selects: TotalSeg_CT + VISTA3D + VoxTell(targeted)
    │
    Skips: MRSegmentator, VIBESeg (strict subsets for this anatomy)
    │
    After QC: if Dice between TotalSeg and VISTA3D is low on
    kidneys, THEN run MRSegmentator as tie-breaker
```

This requires the LLM to understand organ coverage overlap between tools — achievable by providing per-tool organ lists in the prompt. Qwen3-8B would be sufficient for this structured matching task.

**Implementation complexity**: Low. Just requires rewriting the `TOOL_SELECTION` prompt to allow reduction, and providing per-tool organ coverage statistics. The `ToolSelector` infrastructure already handles LLM responses correctly.

**Tradeoff**: Fewer tools means fewer inputs for STAPLE consensus and cross-tool Dice validation. The conditional "run tie-breaker if QC flags disagreement" partially mitigates this.

### Proposal 3: Error-Aware Recovery (essential for production)

**Current behavior**: Tool crashes → warning logged → case continues without that tool's output → no retry.

**Agentic behavior**:
```
VISTA3D crashes with CUDA OOM on case with 800x800x500 volume
    │
    LLM diagnoses: "Volume too large (800x800x500 = 320M voxels).
    VISTA3D requires ~8GB VRAM for standard volumes."
    │
    LLM decides: "Resample to 256x256x167 (1/8 voxels),
    run VISTA3D, resample output masks back to original space."
    │
    Execute → works → continue pipeline
```

Other recovery strategies the LLM could select:
- **Timeout**: "Tool exceeded 10min — likely stuck. Kill and skip, we have 5 other tools."
- **Dependency error**: "Missing checkpoint — skip this tool, flag for admin."
- **Shape mismatch**: "Output masks are 128x128x64 but input is 512x512x237 — tool used internal resampling. Apply `resample_from_to()` (same fix as TextMedSeg3D)."

**Implementation complexity**: Medium. Requires wrapping tool execution in a try/except that passes error context to the LLM, plus a library of recovery actions the LLM can select from.

### Proposal 4: Clinical Context-Driven Prioritization

**Current behavior**: All organs treated equally. All tools run with the same parameters.

**Agentic behavior**:
```
LLM sees: patient from RHEUM cohort (lupus study)
    │
    LLM reasons: "Lupus nephritis is the primary concern.
    Kidney segmentation quality is critical. Spleen volume
    correlates with disease activity. Joint structures
    (sacroiliac, hip) are secondary targets."
    │
    LLM actions:
    - Run kidneys through ALL tools (maximum consensus)
    - Run VoxTell targeted: "renal cortex", "renal medulla"
      (sub-organ structures not in fixed-class tools)
    - Apply stricter QC thresholds for kidneys (Dice > 0.85)
    - Flag any kidney QC failure for manual review
```

**Implementation complexity**: Low-medium. Requires a "clinical priority" prompt that takes cohort context and outputs per-organ priority levels. QC thresholds could be adjusted per-organ based on priority.

### Proposal 5: Post-Hoc Radiomics Interpretation (highest clinical value-add)

**Current behavior**: Extract 107 features → dump to CSV → done.

**Agentic behavior**:
```
After radiomics extraction, LLM sees:
    liver_firstorder_Entropy = 4.2 (normal: 3.5-4.0)
    kidney_left_shape_Sphericity = 0.65 (normal: 0.80-0.95)
    spleen_shape_Volume = 450mL (normal: 100-300mL)
    │
    LLM reasons: "Liver entropy elevated — may indicate
    heterogeneous parenchyma (fibrosis? steatosis?).
    Left kidney sphericity low — suggests irregular shape
    (cyst? mass? poor segmentation?). Spleen enlarged —
    consistent with splenomegaly in lupus."
    │
    LLM generates: per-case clinical summary with:
    - Abnormal findings flagged with confidence levels
    - Correlation with known disease patterns
    - Recommendations for manual review
```

This is where MedGemma-27B's clinical training would genuinely shine — not in deciding which tools to run, but in **interpreting what the numbers mean**. No other component in the pipeline can do this. No rule system can reason about the clinical significance of radiomics features in context.

**Implementation complexity**: Low. Just a new post-radiomics step with a prompt that takes features + clinical context and produces a structured summary. No pipeline changes needed.

---

## 4. The Fundamental Tension

Making the pipeline more agentic means making it **less predictable**. In clinical deployment:

- Deterministic behavior is a feature, not a bug
- "The LLM decided to skip MRSegmentator for this case" is hard to audit
- Reproducibility matters more than marginal quality gains
- Regulatory and IRB considerations may require explainable, repeatable processing

The sweet spot is agentic reasoning that **adds value without sacrificing reproducibility**:

| Approach | Agentic? | Reproducible? | Worth it? |
|----------|----------|---------------|-----------|
| Fixed tool selection (current) | No | Yes | Safe default |
| LLM-guided tool reduction | Somewhat | Yes (with fixed seed) | Saves compute |
| Closed-loop refinement | Yes | Harder to guarantee | High value for flagged organs |
| Clinical radiomics interpretation | Yes | Deterministic input | **Highest value-add** |
| Error recovery | Yes | Case-dependent | Essential for production |
| Clinical context prioritization | Somewhat | Yes (deterministic prompt) | Moderate value |

---

## 5. Prioritized Recommendations

### Priority 1: Clinical Radiomics Interpretation (Proposal 5)

**Why first**: This is where LLMs add value that no rule system can replicate. MedGemma-27B reading radiomics features and producing a clinical summary is the killer feature for this pipeline. It doesn't change any upstream decisions (no reproducibility risk) — it's a pure value-add on top of deterministic outputs.

### Priority 2: Adaptive Tool Selection (Proposal 2)

**Why second**: Straightforward prompt change, saves 30-50% runtime at scale (significant for 26k cases), testable with dry runs. The `ToolSelector` infrastructure already exists — only the prompt needs rewriting. Combined with the conditional tie-breaker re-run, this maintains quality while reducing compute.

### Priority 3: Error Recovery (Proposal 3)

**Why third**: Essential for production at 26k scale. Currently, a single CUDA OOM or checkpoint corruption kills a case permanently. With error recovery, the LLM can diagnose and attempt fixes — converting hard failures into recoverable situations.

### Priority 4: Closed-Loop Refinement (Proposal 1)

**Why fourth**: The most genuinely "agentic" proposal but also the hardest to validate. Requires per-organ re-run infrastructure and careful testing to ensure the LLM's corrective actions actually improve outcomes. Worth implementing after priorities 1-3 are validated.

### Priority 5: Clinical Context Prioritization (Proposal 4)

**Why fifth**: Valuable for disease-specific cohorts (RHEUM/LUPUS) but requires cohort-specific prompt engineering. Lower priority until the pipeline is processing real clinical data and researchers can evaluate whether uniform vs. prioritized processing matters for their analyses.

---

## 6. Conclusion

The pipeline in its current state is a **well-engineered batch processing system** that uses LLMs at five points, of which two (tool selection, metadata extraction) provide zero marginal value over their rule-based fallbacks. The remaining three (organ list generation, QC interpretation, radiomics gating) provide moderate value on edge cases.

Calling the pipeline "agentic" is aspirational. The path to genuine agency is:
1. Let the LLM's decisions actually change what happens next (adaptive tool selection, closed-loop refinement)
2. Let the LLM reason about results, not just process inputs (radiomics interpretation)
3. Let the LLM recover from failures instead of silently logging them (error recovery)

The most impactful change — clinical radiomics interpretation — requires no pipeline modifications, only a new post-processing step. It would transform the pipeline from "extracts numbers" to "explains what the numbers mean clinically" — which is the entire point of building this for a medical research group.
