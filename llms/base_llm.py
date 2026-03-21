"""
================================================================================
Base LLM Interface
================================================================================
Abstract base class for all LLM clients in the CDW pipeline.

All LLM wrappers expose the same query() interface so the planner,
orchestrator, and QC interpreter don't care which model they use.

Usage pattern (context manager — recommended):
    with MedGemma27bText() as llm:
        response = llm.query(user_prompt="Classify this DICOM series: ...")

Usage pattern (manual load/unload):
    llm = MedGemma27bText()
    llm.load()
    response = llm.query(user_prompt="...")
    llm.unload()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# CDW System Prompt
# Embedded here so every LLM client inherits the same default context.
# Override per-call by passing system_prompt= to query().
# ──────────────────────────────────────────────────────────────────────────────

CDW_SYSTEM_PROMPT = """\
You are an expert clinical imaging AI assistant embedded in an automated \
medical image analysis pipeline. Your role is to reason precisely about \
radiological metadata, segmentation quality, and anatomical plausibility.

Pipeline context:
- Dataset: ~26k clinical volumes from a rheumatology/lupus cohort (RHEUM, LUPUS, CONTR)
- Modalities: CT, MRI (T1w, T2w, VIBE, Dixon, HASTE, SWI, TSE), PET-CT, \
nuclear medicine (NM), fluoroscopy (FL)
- Anatomies: head/neck, chest, abdomen, pelvis, abdomen_pelvis, whole_body, extremities
- Segmentation tools available: TotalSegmentator (CT/MRI), MRSegmentator, \
MRISegmenter, VIBESegmentator, VoxTell, BiomedParse3D, TextMedSeg3D (SAT), VISTA3D
- Quality checks: volume plausibility (29 organs), connected component count, \
paired organ volume ratio, mask overlap

Behavioural rules:
1. Be concise and structured. Return only what is asked — no extra commentary.
2. When asked for JSON, return ONLY valid JSON with no prose, no markdown fences.
3. State confidence explicitly when uncertain (high / medium / low).
4. Treat missing or ambiguous metadata conservatively: \
unknown modality → do not assume CT unless there is explicit evidence.
5. Clinical safety first: when a segmentation is borderline, flag it rather \
than silently pass it.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Response container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Structured response from any CDW LLM query."""
    content: str                         # raw text output from the model
    model_name: str                      # which model produced this
    prompt_tokens: int = 0               # input token count (if available)
    completion_tokens: int = 0           # output token count (if available)
    finish_reason: str = ""              # "stop", "length", etc.

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """
    Abstract base for all CDW LLM clients.

    Subclasses must implement:  load(), unload(), query()
    Context-manager protocol (__enter__ / __exit__) is provided here.
    """

    model_name: str = ""

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory (GPU/CPU). No-op if already loaded."""

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free GPU memory."""

    @abstractmethod
    def query(
        self,
        user_prompt: str,
        system_prompt: str = CDW_SYSTEM_PROMPT,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Single-turn query. Returns LLMResponse with .content (the text reply).

        Args:
            user_prompt:    The task-specific prompt (DICOM metadata, QC flags, etc.)
            system_prompt:  Override the default CDW system context if needed.
            max_new_tokens: Maximum tokens to generate.
            temperature:    Sampling temperature. Low = deterministic (use for
                            structured outputs). High = creative (use for reports).
        """

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "BaseLLM":
        self.load()
        return self

    def __exit__(self, *args) -> None:
        self.unload()

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True if model weights are currently in memory. Override in subclass."""
        return False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, loaded={self.is_loaded})"
        )
