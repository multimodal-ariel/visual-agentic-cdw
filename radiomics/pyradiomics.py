"""
================================================================================
CDW Agentic Pipeline — PyRadiomics Wrapper
================================================================================
CPU-based radiomic feature extraction using the PyRadiomics library.

API
───
    result = extract_pyradiomics(image_path, mask_path, organ, ...)
    result.features          # dict[str, float]  — all extracted features
    result.organ             # str
    result.image_path        # str
    result.mask_path         # str
    result.backend           # "pyradiomics"
    result.feature_count     # int
    result.warnings          # list[str]

Environment: cdw_radiomics  (conda run -n cdw_radiomics python ...)
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Naming-collision workaround
# ──────────────────────────────────────────────────────────────────────────────
# Our local radiomics/ package shadows the installed `radiomics` (pyradiomics)
# package when the CDW root is on sys.path.  We resolve this by loading the
# installed package directly from its on-disk location using importlib.

def _get_pyradiomics_featureextractor():
    """
    Return the RadiomicsFeatureExtractor class from the *installed* pyradiomics
    library, bypassing the local radiomics/ package that has the same import name.

    Strategy: temporarily remove the CDW root from sys.path and evict any cached
    local 'radiomics' module so that Python's normal import resolves to the site-
    packages installation.  sys.path is restored afterwards; the installed module
    stays in sys.modules as the canonical 'radiomics' entry.
    """
    import sys

    _our_dir = os.path.dirname(os.path.abspath(__file__))   # .../radiomics/
    _cdw_root = os.path.dirname(_our_dir)                   # .../agentic-cdw/

    # Determine if the cached 'radiomics' is our local package
    _cached_local = {
        k: v for k, v in list(sys.modules.items())
        if (k == "radiomics" or k.startswith("radiomics."))
        and os.path.dirname(os.path.abspath(getattr(v, "__file__", "") or "")) == _our_dir
    }

    # Temporarily remove CDW root from sys.path and evict local modules
    _removed_positions = [i for i, p in enumerate(sys.path)
                          if os.path.abspath(p) == _cdw_root]
    for i in sorted(_removed_positions, reverse=True):
        sys.path.pop(i)
    for k in _cached_local:
        sys.modules.pop(k, None)

    try:
        import radiomics as _pyradiomics_lib  # noqa: F401 — now resolves to site-packages
        from radiomics import featureextractor as _fe
        return _fe.RadiomicsFeatureExtractor
    except ImportError as e:
        raise ImportError(
            "PyRadiomics is not installed in this environment.\n"
            "Install it with:\n"
            "  pip install --no-cache-dir git+https://github.com/AIM-Harvard/pyradiomics.git\n"
            f"Original: {e}"
        ) from e
    finally:
        # Restore sys.path (installed pyradiomics stays in sys.modules as 'radiomics')
        for i, pos in enumerate(sorted(_removed_positions)):
            sys.path.insert(pos + i, _cdw_root)


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass (shared schema with curadiomics)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RadiomicsResult:
    """Unified output schema for both pyradiomics and curadiomics backends."""

    organ: str
    image_path: str
    mask_path: str
    backend: str                              # "pyradiomics" or "curadiomics"
    features: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return (
            f"RadiomicsResult(organ={self.organ!r}, backend={self.backend!r}, "
            f"features={self.feature_count}, warnings={len(self.warnings)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Default params file
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_PARAMS = str(
    Path(__file__).resolve().parent.parent / "config" / "radiomics_params.yaml"
)


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction function
# ──────────────────────────────────────────────────────────────────────────────

def extract_pyradiomics(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    organ: str,
    *,
    params_file: Optional[Union[str, Path]] = None,
    label: int = 1,
    verbose: bool = False,
) -> RadiomicsResult:
    """
    Extract radiomic features from a NIfTI volume + binary mask.

    Parameters
    ----------
    image_path : str | Path
        NIfTI image file (.nii or .nii.gz).
    mask_path : str | Path
        NIfTI binary mask file (.nii or .nii.gz).
        Voxels with value == `label` are included.
    organ : str
        Free-form label for the anatomical structure (e.g. "liver", "spleen").
        Stored in the result for downstream tracking; not used for computation.
    params_file : str | Path | None
        Path to a PyRadiomics YAML parameter file.
        Defaults to config/radiomics_params.yaml in this repo.
    label : int
        Mask label to extract features from. Default 1.
    verbose : bool
        If True, enable PyRadiomics INFO-level logging.

    Returns
    -------
    RadiomicsResult
        .features: dict mapping feature name → float value.
        Includes all enabled feature classes (firstorder, glcm, glrlm, …).
    """
    # Load the installed pyradiomics (bypass our local radiomics/ package)
    RadiomicsFeatureExtractor = _get_pyradiomics_featureextractor()

    # Also grab the top-level radiomics module for setVerbosity
    import sys as _sys
    import radiomics as _pyradiomics  # after _get_pyradiomics_featureextractor, this is the installed one

    image_path = str(image_path)
    mask_path = str(mask_path)

    # Validate inputs exist
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Set PyRadiomics log level
    _pyradiomics.setVerbosity(logging.INFO if verbose else logging.WARNING)

    # Resolve params file
    params = str(params_file) if params_file else _DEFAULT_PARAMS
    if not os.path.isfile(params):
        logger.warning("Params file not found (%s); using PyRadiomics built-in defaults.", params)
        params = None

    # Build extractor
    extractor_kwargs: dict = {"label": label}
    if params:
        extractor = RadiomicsFeatureExtractor(params, **extractor_kwargs)
    else:
        extractor = RadiomicsFeatureExtractor(**extractor_kwargs)

    # Run extraction
    collected_warnings: List[str] = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        raw = extractor.execute(image_path, mask_path)
        for warning in w:
            collected_warnings.append(str(warning.message))

    # Filter to numeric features only (skip diagnostics / metadata keys)
    features: Dict[str, float] = {}
    for key, value in raw.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            features[key] = float(value)
        except (TypeError, ValueError):
            pass  # skip non-numeric outputs

    logger.info(
        "PyRadiomics: organ=%s  features=%d  warnings=%d",
        organ, len(features), len(collected_warnings),
    )

    return RadiomicsResult(
        organ=organ,
        image_path=image_path,
        mask_path=mask_path,
        backend="pyradiomics",
        features=features,
        warnings=collected_warnings,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Batch helper
# ──────────────────────────────────────────────────────────────────────────────

def extract_pyradiomics_batch(
    cases: List[Dict],
    *,
    params_file: Optional[Union[str, Path]] = None,
    label: int = 1,
    verbose: bool = False,
) -> List[RadiomicsResult]:
    """
    Extract features for a list of (image, mask, organ) triples.

    Parameters
    ----------
    cases : list of dict
        Each dict must have keys: "image_path", "mask_path", "organ".
        Optional key: "label" (overrides the global label for that case).

    Returns
    -------
    list of RadiomicsResult
        One result per input case. Failures are returned as results with
        an empty features dict and a warning entry describing the error.
    """
    results: List[RadiomicsResult] = []
    for i, case in enumerate(cases):
        try:
            result = extract_pyradiomics(
                image_path=case["image_path"],
                mask_path=case["mask_path"],
                organ=case.get("organ", f"case_{i}"),
                params_file=params_file,
                label=case.get("label", label),
                verbose=verbose,
            )
        except Exception as exc:
            logger.error("PyRadiomics failed for case %d (%s): %s", i, case, exc)
            result = RadiomicsResult(
                organ=case.get("organ", f"case_{i}"),
                image_path=str(case.get("image_path", "")),
                mask_path=str(case.get("mask_path", "")),
                backend="pyradiomics",
                features={},
                warnings=[f"Extraction failed: {exc}"],
            )
        results.append(result)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point (for quick smoke tests)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="PyRadiomics feature extraction")
    parser.add_argument("--image",   required=True, help="NIfTI image path")
    parser.add_argument("--mask",    required=True, help="NIfTI mask path")
    parser.add_argument("--organ",   default="unknown", help="Organ name (label)")
    parser.add_argument("--params",  default=None, help="PyRadiomics YAML params")
    parser.add_argument("--label",   type=int, default=1, help="Mask label value")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    result = extract_pyradiomics(
        image_path=args.image,
        mask_path=args.mask,
        organ=args.organ,
        params_file=args.params,
        label=args.label,
        verbose=args.verbose,
    )
    print(json.dumps({"organ": result.organ, "feature_count": result.feature_count,
                      "features": result.features, "warnings": result.warnings}, indent=2))
    sys.exit(0)
