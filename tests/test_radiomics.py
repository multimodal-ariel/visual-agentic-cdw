#!/usr/bin/env python
"""
Quick smoke test for radiomics extraction on dummy data.
Runs PyRadiomics (CPU) and cuRadiomics (GPU) on liver + kidneys from case 0004.
Saves results as JSON inside the corresponding dummy_output subdirectory.

Usage (from repo root):
    conda run -n cdw_radiomics python scripts/test_radiomics.py
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Paths ────────────────────────────────────────────────────────────────────

CASE_ID = "0004"
CASE_DIR = os.path.join(REPO_ROOT, "dummy_outputs", CASE_ID)
IMAGE_PATH = os.path.join(CASE_DIR, "image_nifti.nii.gz")
SEG_DIR = os.path.join(CASE_DIR, "segmentations_totalseg_ct")

ORGANS = ["liver", "kidney_left", "kidney_right"]

OUTPUT_DIR = os.path.join(CASE_DIR, "radiomics_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_pyradiomics():
    """Test PyRadiomics CPU extraction."""
    from radiomics.pyradiomics import extract_pyradiomics

    results = {}
    for organ in ORGANS:
        mask_path = os.path.join(SEG_DIR, f"{organ}.nii.gz")
        if not os.path.isfile(mask_path):
            logger.warning("Mask not found: %s — skipping", mask_path)
            continue

        logger.info("PyRadiomics: extracting %s ...", organ)
        t0 = time.time()
        result = extract_pyradiomics(
            image_path=IMAGE_PATH,
            mask_path=mask_path,
            organ=organ,
        )
        elapsed = time.time() - t0
        logger.info(
            "  → %d features in %.1f s  |  warnings: %d",
            result.feature_count, elapsed, len(result.warnings),
        )
        results[organ] = {
            "organ": result.organ,
            "backend": result.backend,
            "feature_count": result.feature_count,
            "elapsed_s": round(elapsed, 2),
            "features": result.features,
            "warnings": result.warnings,
        }

    out_path = os.path.join(OUTPUT_DIR, "pyradiomics_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("PyRadiomics results saved → %s", out_path)
    return results


def test_curadiomics():
    """Test cuRadiomics GPU extraction."""
    # cuRadiomics requires TensorFlow + a compiled CUDA .so.
    # The local radiomics/ namespace collides with the installed pyradiomics
    # package after _get_pyradiomics_featureextractor() swaps sys.modules,
    # so we import the module file directly.
    try:
        import importlib.util, types
        _cura_path = os.path.join(REPO_ROOT, "radiomics", "curadiomics.py")
        spec = importlib.util.spec_from_file_location(
            "cdw_curadiomics", _cura_path,
            submodule_search_locations=[],
        )
        _cura_mod = importlib.util.module_from_spec(spec)
        _cura_mod.__package__ = "cdw_curadiomics"
        sys.modules["cdw_curadiomics"] = _cura_mod
        spec.loader.exec_module(_cura_mod)
        extract_curadiomics_nifti = _cura_mod.extract_curadiomics_nifti
    except (ImportError, OSError, AttributeError) as e:
        logger.warning("cuRadiomics not available: %s", e)
        logger.warning("Skipping cuRadiomics test (expected if TF not installed or .so not built)")
        return None

    results = {}
    for organ in ORGANS:
        mask_path = os.path.join(SEG_DIR, f"{organ}.nii.gz")
        if not os.path.isfile(mask_path):
            logger.warning("Mask not found: %s — skipping", mask_path)
            continue

        logger.info("cuRadiomics: extracting %s ...", organ)
        t0 = time.time()
        try:
            result = extract_curadiomics_nifti(
                image_path=IMAGE_PATH,
                mask_path=mask_path,
                organ=organ,
            )
            elapsed = time.time() - t0
            logger.info(
                "  → %d features in %.1f s  |  warnings: %d",
                result.feature_count, elapsed, len(result.warnings),
            )
            results[organ] = {
                "organ": result.organ,
                "backend": result.backend,
                "feature_count": result.feature_count,
                "elapsed_s": round(elapsed, 2),
                "features": result.features,
                "warnings": result.warnings,
            }
        except Exception as e:
            logger.error("cuRadiomics failed for %s: %s", organ, e)
            results[organ] = {"organ": organ, "backend": "curadiomics", "error": str(e)}

    out_path = os.path.join(OUTPUT_DIR, "curadiomics_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("cuRadiomics results saved → %s", out_path)
    return results


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Radiomics smoke test — case %s", CASE_ID)
    logger.info("Image: %s", IMAGE_PATH)
    logger.info("Seg dir: %s", SEG_DIR)
    logger.info("Organs: %s", ORGANS)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    # PyRadiomics (CPU)
    logger.info("")
    logger.info("--- PyRadiomics (CPU) ---")
    pyrad_results = test_pyradiomics()

    # cuRadiomics (GPU)
    logger.info("")
    logger.info("--- cuRadiomics (GPU) ---")
    curad_results = test_curadiomics()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for organ in ORGANS:
        if organ in pyrad_results:
            pr = pyrad_results[organ]
            logger.info(
                "  %s | pyradiomics: %d features (%.1fs)",
                organ, pr["feature_count"], pr["elapsed_s"],
            )
        if curad_results and organ in curad_results:
            cr = curad_results[organ]
            if "error" in cr:
                logger.info("  %s | curadiomics: FAILED — %s", organ, cr["error"])
            else:
                logger.info(
                    "  %s | curadiomics: %d features (%.1fs)",
                    organ, cr["feature_count"], cr["elapsed_s"],
                )
    logger.info("Results saved to: %s", OUTPUT_DIR)
