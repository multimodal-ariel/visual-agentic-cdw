"""
CDW Agentic Pipeline — QC Package
==================================
Comprehensive reference-free segmentation quality assessment.

Tiers
-----
Tier 1: GeometricQC        — per-tool geometric checks (volume, CC, overlap, ratios)
Tier 2: MultiToolQC        — cross-tool pairwise agreement (Dice, volume divergence)
Tier 3: QCInterpreter      — LLM clinical interpretation + radiomics gating

Data structures
---------------
OrganQCResult, ToolQCResult, OrganAgreement, MultiToolQCResult,
QCInterpretation, CaseQCReport
"""

from qc.dataclasses import (
    CaseQCReport,
    MultiToolQCResult,
    OrganAgreement,
    OrganQCResult,
    QCInterpretation,
    ToolQCResult,
)
from qc.geometric_qc import GeometricQC
from qc.multi_tool_qc import MultiToolQC
from qc.qc_interpreter import QCInterpreter

__all__ = [
    # Engines
    "GeometricQC",
    "MultiToolQC",
    "QCInterpreter",
    # Data structures
    "OrganQCResult",
    "ToolQCResult",
    "OrganAgreement",
    "MultiToolQCResult",
    "QCInterpretation",
    "CaseQCReport",
]
