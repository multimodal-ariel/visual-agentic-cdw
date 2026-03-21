"""
CDW Agentic Pipeline — Radiomics
=================================
Primary backend: PyRadiomics (CPU) — fast, full-featured, 93 features across 6 classes.

Optional backend: cuRadiomics (GPU) — available in radiomics/curadiomics.py but not
used in the pipeline. Requires TensorFlow + recompiled CUDA .so. Only supports
GLCM + first-order (41 features, 2D per-slice). Import directly if needed:
    from radiomics.curadiomics import extract_curadiomics_nifti

Standard usage:
    from radiomics.pyradiomics import extract_pyradiomics, RadiomicsResult
"""


def __getattr__(name):
    """Lazy imports to avoid pyradiomics naming collision at init time."""
    _pyrad_names = {"RadiomicsResult", "extract_pyradiomics", "extract_pyradiomics_batch"}

    if name in _pyrad_names:
        from radiomics.pyradiomics import (
            RadiomicsResult,
            extract_pyradiomics,
            extract_pyradiomics_batch,
        )
        return locals()[name]

    raise AttributeError(f"module 'radiomics' has no attribute {name!r}")


__all__ = [
    "RadiomicsResult",
    "extract_pyradiomics",
    "extract_pyradiomics_batch",
]
