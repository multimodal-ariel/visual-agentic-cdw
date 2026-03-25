#!/usr/bin/env python3
"""
Dataset path parser for the CDW imaging pipeline.

Parses JSON files containing DICOM paths and classifies each entry by:
  - cohort (from path: /data/RAD/{COHORT}/...)
  - modality (CT, MRI, XR, MAMMO, US, NM, PET_CT, CTA, MRA, FL, IR, DEXA, OTHER)
  - anatomy (head, neck, chest, abdomen, pelvis, abdomen_pelvis, spine, extremity,
             whole_body, cardiac, breast, OTHER)
  - anatomy_detail (finer-grained: e.g. knee, wrist, hand, ankle, etc.)

Handles two JSON formats:
  Format A: [ [path, [d1,d2,d3]], ... ]   (path + shape tuples)
  Format B: [ path1, path2, ... ]          (plain strings)

For very large files, uses ijson for streaming if available,
otherwise falls back to standard json.load.

Install ijson for streaming support:
    pip install ijson
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import ijson
    _HAS_IJSON = True
except ImportError:
    _HAS_IJSON = False


# ── Modality classification ─────────────────────────────────────────────────

_MODALITY_RULES = [
    # Order matters: more specific prefixes first
    (r"^PET[_ ]CT",           "PET_CT"),
    (r"^CTA[_ ]",             "CTA"),
    (r"^CT[_ ]ANGIO",         "CTA"),
    (r"^MRA[_ ]",             "MRA"),
    (r"^MRV[_ ]",             "MRA"),
    (r"^MRCP",                "MRI"),
    (r"^MRI[_ ]",             "MRI"),
    (r"^CT[_ ]",              "CT"),
    (r"^XR[_ ]",              "XR"),
    (r"^CR[_ ]",              "XR"),
    (r"^DR[_ ]",              "XR"),
    (r"^DG[_ ]",              "XR"),
    (r"^IC[_ ]DR",            "XR"),
    (r"^CXR",                 "XR"),
    (r"^MAMMO",               "MAMMO"),
    (r"^MG[_ ]",              "MAMMO"),
    (r"^SCREENING_DIRECT_DIG","MAMMO"),
    (r"^MAMMOGRAPHY",         "MAMMO"),
    (r"^NM[_ ]",              "NM"),
    (r"^CARDIAC",             "NM"),
    (r"^PARATHYROID",         "NM"),
    (r"^RENAL_SCAN",          "NM"),
    (r"^LUNG_VQ",             "NM"),
    (r"^LUNG_PERFUSION",      "NM"),
    (r"^US[_ ]",              "US"),
    (r"^ULS[_ ]",             "US"),
    (r"^FL[_ ]",              "FL"),
    (r"^IR[_ ]",              "IR"),
    (r"^DEXA",                "DEXA"),
    (r"^DXA",                 "DEXA"),
    (r"^QDR",                 "DEXA"),
    (r"^PVL[_ ]",             "US"),
]

# Plain-film / XR study names that don't start with a modality prefix
_XR_KEYWORDS = [
    "CHEST_1V", "CHEST_2V", "CHEST_PA", "PORTABLE_CHEST", "ADULT_CHEST",
    "ANKLE", "KNEE", "FOOT", "HAND", "WRIST", "SHOULDER", "ELBOW", "HIP",
    "FEMUR", "TIBIA", "FOREARM", "HUMERUS", "FINGER", "TOE", "CLAVICLE",
    "PELVIS_AP", "SCOLIOSIS", "SPINE_LUMBAR", "SPINE_CERVICAL", "SPINE__",
    "ABDOMEN_1", "ABDOMEN_2", "ABDOMEN_MIN", "ABDOMEN_ACUTE",
    "ABDOMEN_COMPLETE", "ABDOMEN_PORTABLE", "ABDOMEN_SUPINE",
    "ABD_ABDOMEN", "ABDS_ACUTE", "ACUTE_ABDOMINAL",
    "KUB", "BILATERAL_KNEES", "BONE_AGE", "BONE_SURVEY", "BONE SURVEY",
    "MYELOMA_SURVEY", "SHUNT_SERIES", "RIBS_", "SACROILIAC", "SACRUM",
    "FOREARM_", "TIBIA_", "TIBULA_", "SCOLIOSIS_",
    "CPII_ORTHO", "HIPS_BILATERAL",
    "RECOSUPINE",
]

# NM keywords for studies without NM_ prefix
_NM_KEYWORDS = [
    "RENAL_SCAN", "PARATHYROID_SCAN", "LUNG_VQ", "LUNG_PERFUSION",
    "NEPHROURETERAL", "CYSTOGRAM",
]

_modality_re = [(re.compile(pat, re.IGNORECASE), mod) for pat, mod in _MODALITY_RULES]


def classify_modality(study: str) -> str:
    s = study.upper().strip()
    for pat, mod in _modality_re:
        if pat.search(s):
            return mod
    for kw in _XR_KEYWORDS:
        if kw in s:
            return "XR"
    for kw in _NM_KEYWORDS:
        if kw in s:
            return "NM"
    # CT/MRI with spaces (outside-film reads)
    if s.startswith("CT "):
        return "CT"
    if s.startswith("CTA "):
        return "CTA"
    if s.startswith("MRI "):
        return "MRI"
    if s.startswith("PET "):
        return "PET_CT"
    if s.startswith("XR "):
        return "XR"
    if "OUTSIDE_FILM" in s or "OUTSIDE FILM" in s or "COMPARISON" in s:
        # Try to infer modality from prefix
        if "CT" in s.split("_")[0] or "CT" in s.split(" ")[0]:
            return "CT"
        if "MRI" in s or "NEURO" in s:
            return "MRI"
        if "MAMMO" in s:
            return "MAMMO"
        if "PEDS" in s or "X-RAY" in s or "XR" in s:
            return "XR"
    if "FLUORO" in s or "STATISTIC_ONLY_FLUORO" in s:
        return "FL"
    if "VCUG" in s or "UGI_" in s or "UPPER_GI" in s or "BARIUM" in s:
        return "FL"
    if "INSERT" in s and ("PICC" in s or "TUNNELED" in s or "CVC" in s or "PORT" in s):
        return "IR"
    return "OTHER"


# ── Anatomy classification ───────────────────────────────────────────────────

_ANATOMY_DETAIL_RULES = [
    # Extremities — fine-grained (check before broad anatomy)
    (r"KNEE",                   "knee",         "extremity"),
    (r"ANKLE",                  "ankle",        "extremity"),
    (r"FOOT",                   "foot",         "extremity"),
    (r"TOE",                    "toes",         "extremity"),
    (r"HAND",                   "hand",         "extremity"),
    (r"WRIST",                  "wrist",        "extremity"),
    (r"FINGER",                 "finger",       "extremity"),
    (r"SHOULDER",               "shoulder",     "extremity"),
    (r"ELBOW",                  "elbow",        "extremity"),
    (r"HIP",                    "hip",          "extremity"),
    (r"FEMUR",                  "femur",        "extremity"),
    (r"TIBI|FIBUL",             "tibia_fibula", "extremity"),
    (r"FOREARM",                "forearm",      "extremity"),
    (r"HUMERUS",                "humerus",      "extremity"),
    (r"CLAVICLE",               "clavicle",     "extremity"),
    (r"RING_FINGER",            "finger",       "extremity"),
    (r"SACROILIAC",             "sacroiliac",   "extremity"),
    (r"SACRUM|COCCYX",          "sacrum",       "spine"),
    (r"UPPER.?EXTREMITY|UPR.?EXTREMITY|EXTREM.*UPR", "upper_extremity", "extremity"),
    (r"LOWER.?EXTREMITY|EXTREM.*LOW", "lower_extremity", "extremity"),
    (r"BONE.?AGE",              "hand",         "extremity"),
    (r"PELVIS",                 "pelvis",       "pelvis"),

    # Spine
    (r"CERVICAL.*THORACIC.*LUMBAR|TOTAL.*CTL|TOTAL_BODY.*SPINE", "whole_spine", "spine"),
    (r"SCOLIOSIS|THORACOLUMBAR","thoracolumbar","spine"),
    (r"CERVICAL.?SPINE|C.?SPINE","cervical_spine","spine"),
    (r"THORACIC.?SPINE|T.?SPINE","thoracic_spine","spine"),
    (r"LUMBAR.?SPINE|L.?SPINE", "lumbar_spine", "spine"),
    (r"SPINE",                  "spine_other",  "spine"),

    # Head/Brain
    (r"BRAIN|HEAD(?!.*NECK)",   "brain",        "head"),
    (r"NEURO",                  "brain",        "head"),
    (r"SINUS",                  "sinus",        "head"),
    (r"TEMPORAL.?BONE",         "temporal_bone","head"),
    (r"MAXILLOFACIAL",          "maxillofacial","head"),
    (r"TMJ",                    "tmj",          "head"),
    (r"ORBIT",                  "orbit",        "head"),
    (r"SKULL",                  "skull",        "head"),
    (r"SHUNT.?SERIES",          "brain",        "head"),

    # Neck
    (r"NECK",                   "neck",         "neck"),

    # Torso combinations
    (r"CHEST.*ABD.*PEL|ABD.*PEL.*CHEST", "chest_abdomen_pelvis", "chest_abdomen_pelvis"),
    (r"ABD.*PEL|ABDOMEN.*PELVIS|AB.?PEL",  "abdomen_pelvis", "abdomen_pelvis"),

    # Chest
    (r"CHEST|LUNG|PULMONARY|CXR|THORAX", "chest", "chest"),
    (r"RIB",                    "ribs",         "chest"),

    # Abdomen
    (r"ABDOMEN|ABD|RENAL|KIDNEY|HEPATOBILIARY|GASTRIC|KUB|MESENTERIC|UROGRAM|STONE",
                                "abdomen",      "abdomen"),

    # Cardiac
    (r"CARDIAC|MYOCARDIAL|HEART", "cardiac",    "cardiac"),

    # Breast
    (r"MAMMO|BREAST|SCREENING_DIRECT_DIG|MAMMOGRAPHY", "breast", "breast"),

    # Whole body
    (r"WHOLE.?BODY|TOTAL.?BODY|BONE.?SURVEY|MYELOMA.?SURVEY|SKULL.?BASE.?TO.?THIGH|WB",
                                "whole_body",   "whole_body"),

    # Vascular (US)
    (r"CAROTID|DUPLEX|VESSEL_MAPPING|VENOUS|DOPPLER", "vascular", "vascular"),

    # GI/GU
    (r"VOIDING|CYSTOGRAM|CYST|UPPER_GI|UGI|BARIUM|SWALLOW|GASTRIC_EMPTYING",
                                "gi_gu",        "abdomen"),

    # Catch-all: outside film reads with modality hints
    (r"BONE.?MARROW|BONE.?DENSITY|BONE.?SURVEY|DEXA|DXA|QDR",
                                "skeletal",     "whole_body"),
    (r"SPECT|PERFUSION|VENTILATION", "nuclear", "chest"),
    (r"NEPHRO|RENAL|DRAIN.*PERITONEAL|G.?TUBE|STENT.*VEIN",
                                "abdomen",      "abdomen"),
    (r"PICC|TUNNELED|CVC|PORT|NONTUNNELED|CATHETER",
                                "vascular_access", "chest"),
]

_anatomy_re = [(re.compile(pat, re.IGNORECASE), detail, broad) for pat, detail, broad in _ANATOMY_DETAIL_RULES]


def classify_anatomy(study: str) -> tuple[str, str]:
    """Return (anatomy_detail, anatomy_broad)."""
    s = study.upper().strip()
    for pat, detail, broad in _anatomy_re:
        if pat.search(s):
            return detail, broad
    return "other", "OTHER"


# ── Path parsing ─────────────────────────────────────────────────────────────

def parse_path(path: str) -> dict:
    """Extract structured fields from a DICOM path.

    Path format: /data/RAD/{COHORT}/{PATIENT_ID}/{DATE}/{STUDY_DESCRIPTION}/{SERIES_NAME}
    """
    parts = path.strip().split("/")
    # Minimum: ['', 'data', 'RAD', COHORT, PATIENT, DATE, STUDY, SERIES]
    cohort = parts[3] if len(parts) > 3 else "UNKNOWN"
    patient_id = parts[4] if len(parts) > 4 else "UNKNOWN"
    date = parts[5] if len(parts) > 5 else "UNKNOWN"
    study = parts[6] if len(parts) > 6 else "UNKNOWN"
    series = parts[7] if len(parts) > 7 else "UNKNOWN"

    modality = classify_modality(study)
    anatomy_detail, anatomy_broad = classify_anatomy(study)

    return {
        "path": path,
        "cohort": cohort,
        "patient_id": patient_id,
        "date": date,
        "study_description": study,
        "series_name": series,
        "modality": modality,
        "anatomy": anatomy_broad,
        "anatomy_detail": anatomy_detail,
    }


def load_json_paths(json_path: str) -> list[str]:
    """Load paths from a JSON file, handling both formats.

    Uses ijson for streaming if available (handles very large files
    without loading the entire JSON into memory).
    Falls back to standard json.load otherwise.
    """
    if _HAS_IJSON:
        return _load_paths_streaming(json_path)

    with open(json_path) as f:
        data = json.load(f)
    paths = []
    for item in data:
        if isinstance(item, list):
            paths.append(item[0])
        elif isinstance(item, str):
            paths.append(item)
    return paths


def _load_paths_streaming(json_path: str) -> list[str]:
    """Stream-parse a JSON array of paths or [path, shape] tuples via ijson."""
    paths = []
    with open(json_path, "rb") as f:
        # Peek at structure: first non-whitespace after '[' determines format
        # ijson.items at "item" level yields each top-level array element
        for item in ijson.items(f, "item"):
            if isinstance(item, list):
                paths.append(item[0])
            elif isinstance(item, str):
                paths.append(item)
    return paths


def parse_json_to_dataframe(json_path: str) -> pd.DataFrame:
    """Parse a JSON file of paths into a classified DataFrame."""
    paths = load_json_paths(json_path)
    records = [parse_path(p) for p in paths]
    return pd.DataFrame(records)


def parse_multiple_jsons(json_paths: list[str]) -> pd.DataFrame:
    """Parse multiple JSON files, dedup by path, return combined DataFrame."""
    frames = []
    for jp in json_paths:
        df = parse_json_to_dataframe(jp)
        df["source_file"] = Path(jp).name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["path"])
    return combined


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse CDW dataset paths and print summary statistics")
    parser.add_argument("inputs", nargs="+", help="JSON files to parse")
    parser.add_argument("--output-csv", default=None, help="Save parsed records to CSV")
    args = parser.parse_args()

    df = parse_multiple_jsons(args.inputs)

    print(f"Total entries: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Unique studies: {df['study_description'].nunique()}")
    print()

    print("=== Cohort Distribution ===")
    print(df["cohort"].value_counts().to_string())
    print()

    print("=== Modality Distribution ===")
    print(df["modality"].value_counts().to_string())
    print()

    print("=== Anatomy (Broad) Distribution ===")
    print(df["anatomy"].value_counts().to_string())
    print()

    print("=== Anatomy (Detail) Distribution ===")
    print(df["anatomy_detail"].value_counts().to_string())

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")
