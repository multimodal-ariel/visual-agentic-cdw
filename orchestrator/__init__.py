"""
CDW Agentic Pipeline — Orchestrator
====================================
End-to-end pipeline execution.

Modules
-------
pipeline        CasePipeline — single-case end-to-end flow
batch_runner    BatchRunner  — parallel batch execution with GPU pool
case_tracker    CaseTracker  — JSON-backed persistent state + resume
"""

from orchestrator.case_tracker import CaseTracker
from orchestrator.pipeline import CasePipeline, CaseResult
from orchestrator.batch_runner import BatchRunner

__all__ = [
    "CasePipeline",
    "CaseResult",
    "BatchRunner",
    "CaseTracker",
]
