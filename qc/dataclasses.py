"""
CDW Agentic Pipeline — QC Data Structures
==========================================
Shared dataclasses for all QC tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1: Per-organ geometric QC result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OrganQCResult:
    """QC result for a single organ mask from a single tool."""

    case_id: str
    organ: str
    tool_name: str = ""
    case_path: str = ""
    mask_path: str = ""

    # Volume
    volume_ml: float = 0.0
    volume_in_range: bool = True
    volume_flag: str = ""

    # Paired ratio
    ratio_partner: str = ""
    ratio_value: float = -1.0
    ratio_in_range: bool = True
    ratio_flag: str = ""

    # Connected components
    num_components: int = 0
    largest_component_fraction: float = 1.0
    cc_flag: str = ""

    # Overlap
    has_overlap: bool = False
    overlap_flag: str = ""

    # Aggregate
    num_flags: int = 0
    severity: str = "PASS"  # PASS / WARN / FAIL

    def compute_severity(self, warn_max: int = 2) -> list[str]:
        """Count flags and set severity. Returns the list of active flags."""
        flags = []
        if not self.volume_in_range and self.volume_flag:
            flags.append(self.volume_flag)
        if not self.ratio_in_range and self.ratio_flag:
            flags.append(self.ratio_flag)
        if self.cc_flag:
            flags.append(self.cc_flag)
        if self.has_overlap:
            flags.append(self.overlap_flag)
        self.num_flags = len(flags)
        if self.num_flags == 0:
            self.severity = "PASS"
        elif self.num_flags <= warn_max:
            self.severity = "WARN"
        else:
            self.severity = "FAIL"
        return flags


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1: Per-case QC result (aggregated over all organs from one tool)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolQCResult:
    """QC result for one tool's output on one case."""

    case_id: str
    case_path: str
    tool_name: str = ""
    seg_dir: str = ""
    num_organs: int = 0
    num_organs_in_fov: int = 0
    num_organs_flagged: int = 0
    total_flags: int = 0
    worst_severity: str = "PASS"
    overlap_pairs: list[str] = field(default_factory=list)
    organ_results: list[OrganQCResult] = field(default_factory=list)
    error: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Tier 2: Multi-tool agreement result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OrganAgreement:
    """Pairwise agreement between two tools for one organ."""

    organ: str
    tool_a: str
    tool_b: str
    dice: float = 0.0
    volume_a_ml: float = 0.0
    volume_b_ml: float = 0.0
    agrees: bool = True  # True if Dice >= threshold
    flag: str = ""


@dataclass
class MultiToolQCResult:
    """Tier 2 QC result: cross-tool agreement for one case."""

    case_id: str
    case_path: str
    tool_pairs_checked: int = 0
    organs_with_agreement: int = 0
    organs_with_disagreement: int = 0
    mean_dice: float = 0.0
    agreements: list[OrganAgreement] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Tier 3: LLM QC interpretation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QCInterpretation:
    """LLM-generated clinical QC interpretation for a case."""

    case_id: str
    case_path: str
    tool_name: str = ""

    # From QC_INTERPRETATION prompt
    overall_quality: str = ""          # GOOD / ACCEPTABLE / POOR / UNUSABLE
    usable_for_radiomics: list[str] = field(default_factory=list)
    unusable_organs: list[str] = field(default_factory=list)
    first_order_safe: list[str] = field(default_factory=list)
    shape_safe: list[str] = field(default_factory=list)
    texture_safe: list[str] = field(default_factory=list)
    explanation: str = ""

    # From RADIOMICS_GATING prompt
    extract: list[dict] = field(default_factory=list)   # [{organ, features_allowed, postprocessing_needed}]
    skip: list[dict] = field(default_factory=list)       # [{organ, reason}]
    notes: str = ""

    raw_response: str = ""  # raw LLM output for debugging


# ──────────────────────────────────────────────────────────────────────────────
# Full case QC report (all tiers combined)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseQCReport:
    """Complete QC report for one case, across all tools and tiers."""

    case_id: str
    case_path: str

    # Tier 1: per-tool geometric QC
    tool_results: list[ToolQCResult] = field(default_factory=list)

    # Tier 2: multi-tool agreement
    agreement: Optional[MultiToolQCResult] = None

    # Tier 3: LLM interpretation
    interpretation: Optional[QCInterpretation] = None

    # Overall verdict
    overall_severity: str = "PASS"     # worst across all tiers
    radiomics_ready_organs: list[str] = field(default_factory=list)

    def compute_overall(self) -> None:
        """Set overall severity from all tier results."""
        severities = []
        for tr in self.tool_results:
            severities.append(tr.worst_severity)
        if self.agreement and self.agreement.organs_with_disagreement > 0:
            severities.append("WARN")
        if self.interpretation:
            quality_map = {"GOOD": "PASS", "ACCEPTABLE": "WARN", "POOR": "WARN", "UNUSABLE": "FAIL"}
            severities.append(quality_map.get(self.interpretation.overall_quality, "WARN"))

        if "FAIL" in severities:
            self.overall_severity = "FAIL"
        elif "WARN" in severities:
            self.overall_severity = "WARN"
        elif "ERROR" in severities:
            self.overall_severity = "ERROR"
        else:
            self.overall_severity = "PASS"
