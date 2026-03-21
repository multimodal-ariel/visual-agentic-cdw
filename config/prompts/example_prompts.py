# MedGemma-27B Prompt Templates
# Validated through testing on real clinical data from the CDW pipeline
# Model: google/medgemma-27b-text-it
# Temperature: 0.1 (deterministic)
# Max tokens: 2048-4096 depending on task

# ============================================================================
# PROMPT 1: METADATA EXTRACTION (single case)
# ============================================================================
# Input: file path + shape
# Output: JSON with modality, anatomy, shape, is_diagnostic, series_type,
#         contrast_status, mri_sequence
# Score: 17/20 on is_diagnostic, 10/10 on modality/anatomy
# ============================================================================

METADATA_EXTRACTION_SINGLE = """You are a medical imaging metadata parser. Given a DICOM file path and shape, extract structured information.

Return ONLY a JSON object with these fields:
- modality: one of [CT, MRI, PET_CT, NM, CR, US, FL, UNKNOWN]
- anatomy: one of [head, neck, chest, abdomen, pelvis, abdomen_pelvis, chest_abdomen_pelvis, spine, extremity, whole_body, cardiac, UNKNOWN]
- shape: one of [2D, 3D, 4D]
- is_diagnostic: true/false (see rules below)
- series_type: brief description (e.g. "axial soft tissue CT with contrast", "T1 VIBE Dixon MRI")
- contrast_status: one of [pre, post, with, without, unknown]
- mri_sequence: if MRI, the sequence type (e.g. T1, T2, FLAIR, DWI, VIBE, Dixon, SWI, HASTE, TSE). Otherwise null.

Rules for is_diagnostic:
- Default TRUE (most series are diagnostic)
- Set FALSE only if the series name suggests any of:
  * Scout or localizer
  * Screen save or screen capture
  * MIP, MinIP, or mIP projection
  * Coronal or sagittal reformat (COR_, SAG_, _cor, _sag in series name)
  * Bone window reconstruction (_BW, _BONE)
  * Metal artifact reduction (iMAR alone without a diagnostic base series)
  * Subtraction maps (_SUB)
  * Phase images (_Pha)
  * Fluoroscopy
  * PET NAC (non-attenuation corrected)
  * Dose report
  * UNKNOWN series name
  * PET reformats (PET_AC_Cor, PET_AC_Sag, CORONAL/SAGITTAL as standalone series names)
- Note: iMAR combined with a diagnostic series name (e.g. Abdomen_Pelvis_3mm_Axial_ST_iMAR) is still diagnostic
- Note: _BW at end of series name = bone window = NOT diagnostic
- Note: Lung window (_LW) is a valid diagnostic reconstruction

File path: {file_path}
Shape: {shape}"""


# ============================================================================
# PROMPT 2: METADATA EXTRACTION (batch)
# ============================================================================
# Input: multiple file paths with shapes
# Output: JSON array
# Score: 8/10 on is_diagnostic for batch (27B model)
# ============================================================================

METADATA_EXTRACTION_BATCH = """You are a medical imaging metadata parser. For each file path below, return a JSON array. Each element must have: modality, anatomy, shape, is_diagnostic, series_type, contrast_status, mri_sequence.

Rules for is_diagnostic:
- Default TRUE
- FALSE if: scout/localizer, screen save, MIP/MinIP/mIP, coronal/sagittal reformat (COR_, SAG_, _cor, _sag in series name), bone window (_BW, _BONE), iMAR alone, subtraction maps (_SUB), phase images (_Pha), fluoroscopy, PET NAC, dose report, UNKNOWN, angiography reformats (PET_AC_Cor, PET_AC_Sag, CORONAL/SAGITTAL as standalone series names)
- Note: iMAR combined with a diagnostic series (e.g. Abdomen_Pelvis_3mm_Axial_ST_iMAR) is still diagnostic
- Note: _BW = bone window = NOT diagnostic

{numbered_paths_with_shapes}"""


# ============================================================================
# PROMPT 3: TOOL SELECTION
# ============================================================================
# Input: case metadata JSON
# Output: JSON with tools list + reasoning
# Score: 8/10 (composes ensembles, picks VIBESeg for VIBE MRI)
# ============================================================================

TOOL_SELECTION = """You are a medical imaging pipeline planner. Given case metadata, select which segmentation tools to run.

Available tools:
- TotalSegmentator_CT: CT only (3D). 104 structures. Best for standard axial soft-tissue CT.
- TotalSegmentator_MR: MRI only (3D). 80 structures. Sequence-independent. Use flag -ta total_mr.
- MRSegmentator: CT and MRI (3D). 40 classes. Good for abdominal organs across both modalities.
- MRISegmenter: MRI only (3D). 62 structures. Specifically for T1-weighted abdominal MRI.
- VIBESegmentator: MRI and CT (3D). 71 MRI / 72 CT structures. Full torso. Works on multiple MRI sequences.
- SAT: CT, MRI, PET (3D). 497 text-prompted classes. Use for rare structures not covered by fixed-class tools.
- VoxTell: CT and MRI (3D). Free-text prompted. Use for targeted segmentation of specific structures.
- NV_Segment_CTMR: CT and MRI (3D). 345+ structures including detailed brain parcellation.
- BiomedParseV2: Any modality (2D only). Text-prompted. Use for 2D radiographs (X-ray, mammography).

Selection principles:
1. Always include at least one Tier 1 tool (TotalSegmentator or NV_Segment_CTMR) as the primary model
2. For abdomen/pelvis CT: prefer TotalSegmentator_CT as primary, add MRSegmentator for ensemble
3. For abdomen MRI with VIBE/Dixon: include VIBESegmentator
4. For T1w abdominal MRI: include MRISegmenter
5. For brain: prefer NV_Segment_CTMR (has detailed brain parcellation)
6. For 2D data (X-ray, mammography): use BiomedParseV2
7. For structures not covered by fixed-class tools: add SAT or VoxTell with specific organ prompts
8. For PET/CT: segment the CT component with CT tools, then project masks onto PET

Case metadata:
{case_metadata_json}

Return a JSON object with:
- primary_tools: list of Tier 1 tools to run
- secondary_tools: list of Tier 2 tools to run (for ensemble/second opinion)
- targeted_tools: list of Tier 3 tools with specific text prompts (if needed)
- qc_organs: list of organs to check in QC based on the detected anatomy
- reasoning: one sentence explaining the selection"""


# ============================================================================
# PROMPT 4: QC INTERPRETATION
# ============================================================================
# Input: QC flags for a case
# Output: JSON with quality assessment + per-organ usability
# Score: 9/10 (excellent clinical reasoning)
# ============================================================================

QC_INTERPRETATION = """You are a medical imaging QC interpreter for clinicians. Given QC flags from an automated segmentation pipeline, assess quality and determine which organs are usable for radiomics feature extraction.

Case: {study_description}
Anatomy: {anatomy}
Segmentation tool: {tool_name}
Organs in FOV: {num_organs_in_fov}

Flagged organs:
{flagged_organs_formatted}

Unflagged organs (passed all checks):
{passed_organs_list}

QC flag definitions:
- UNDER_VOLUME: organ volume below expected range (may indicate partial FOV or under-segmentation)
- OVER_VOLUME: organ volume above expected range (may indicate over-segmentation or leakage)
- CC=N: N connected components found (expected 1 for most organs; indicates fragmentation)
- FRAG=X%: largest component is X% of total volume (low = severe fragmentation)
- RATIO_OUTLIER: paired organ volume ratio outside expected range

Return a JSON object with:
- overall_quality: one of [GOOD, ACCEPTABLE, POOR, UNUSABLE]
- usable_for_radiomics: list of organ names where segmentation quality is sufficient for feature extraction
- unusable_organs: list of organ names where segmentation is too poor for reliable radiomics
- first_order_safe: list of organs safe for first-order features (intensity, volume)
- shape_safe: list of organs safe for shape features (sphericity, surface area)
- texture_safe: list of organs safe for texture features (GLCM, GLRLM)
- explanation: 2-3 sentences a clinician can understand"""


# ============================================================================
# PROMPT 5: RADIOMICS GATING
# ============================================================================
# Input: QC report for a case
# Output: JSON with per-organ radiomics extraction decisions
# ============================================================================

RADIOMICS_GATING = """You are a medical imaging radiomics expert. Given the QC assessment for a case, decide which organs should have radiomics features extracted and which should be skipped.

Case: {case_id}
Study: {study_description}
Overall QC quality: {overall_quality}

Per-organ QC results:
{per_organ_qc_json}

Decision criteria:
1. Only extract radiomics from organs that passed QC or have minor flags (CC=2 with largest component >90%)
2. For organs with UNDER_VOLUME at edge of FOV: skip shape features, allow intensity features only if >50% of expected volume is present
3. For organs with CC=2-3: apply largest connected component postprocessing first, then extract
4. For organs with CC>=4 or FRAG<50%: do not extract radiomics
5. For organs with OVER_VOLUME: flag for review but allow extraction (may indicate pathology, not segmentation error)

Return a JSON object with:
- extract: list of objects, each with {organ, features_allowed: [first_order, shape, texture], postprocessing_needed: [lcc, closing, hole_fill, none]}
- skip: list of objects, each with {organ, reason}
- notes: any clinical observations about the case"""


# ============================================================================
# PROMPT 6: CLINICAL VERIFICATION / EXPERT CONSULTATION
# ============================================================================
# Input: aggregate QC statistics
# Output: structured clinical assessment
# Used for: validating pipeline results with domain expertise
# ============================================================================

CLINICAL_VERIFICATION = """You are a senior {specialty} advising on a large-scale clinical imaging study. A research team has run automated segmentation on {n_cases} {modality} scans from a {dataset_description} and performed automated quality assessment.

{qc_summary}

Provide your expert assessment on:
1. Are these quality rates expected for {tool_name} on uncurated clinical data?
2. Which organs are safe for radiomics (first-order, shape, texture features separately)?
3. Is largest-connected-component postprocessing safe per organ?
4. For the key organs in this study ({key_organs}), what are the estimated usable cohort sizes after postprocessing?
5. Overall, is this dataset viable for the intended research ({research_question})?
6. What minimum cohort size do you recommend for robust radiomics studies?

Return structured JSON with these 6 sections."""


# ============================================================================
# PROMPT 7: ORGAN LIST GENERATION (anatomy → expected organs for QC)
# ============================================================================
# Input: detected anatomy
# Output: list of organs to check in QC
# Note: This can also be done via organ_reference.json lookup (faster, deterministic)
#       Use LLM version only for unusual/ambiguous anatomies
# ============================================================================

ORGAN_LIST_GENERATION = """You are a medical imaging anatomy expert. Given the detected body region of a scan, list all organs and structures that should be visible and checkable in a segmentation quality assessment.

Only list organs that would be fully or substantially within the field of view for this body region. Do not list organs that would only be partially captured at the edges.

Detected anatomy: {anatomy}
Modality: {modality}
Series type: {series_type}

Return a JSON object with:
- primary_organs: list of organs fully within FOV (must check these in QC)
- edge_organs: list of organs partially within FOV (check with relaxed thresholds)
- exclude_organs: list of organs NOT expected in this FOV (skip in QC)"""


# ============================================================================
# USAGE EXAMPLE (Python)
# ============================================================================
#
# from planner.llm_client import PlannerLLM
#
# llm = PlannerLLM("google/medgemma-27b-text-it", device="auto")
# llm.load()
#
# # Metadata extraction
# prompt = METADATA_EXTRACTION_SINGLE.format(
#     file_path="/data/RAD/RHEUM/.../CT_ABDOMEN_PELVIS_W_CONTRAST/ABD_PELVIS",
#     shape="[512, 512, 237]"
# )
# result = llm.query(system_prompt="", user_prompt=prompt)
#
# # Tool selection
# prompt = TOOL_SELECTION.format(
#     case_metadata_json='{"modality": "CT", "anatomy": "abdomen_pelvis", ...}'
# )
# result = llm.query(system_prompt="", user_prompt=prompt)
#
# llm.unload()
#
# ============================================================================
# CALLING VIA vLLM (for interactive testing on g2)
# ============================================================================
#
# python3 -c "
# import requests, json
# prompt = open('/tmp/prompt.txt').read()
# r = requests.post('http://127.0.0.1:41260/v1/chat/completions',
#     json={
#         'model': 'google/medgemma-27b-text-it',
#         'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}],
#         'max_tokens': 2048,
#         'temperature': 0.1
#     },
#     proxies={'http': None, 'https': None})
# print(r.json()['choices'][0]['message']['content'])
# "