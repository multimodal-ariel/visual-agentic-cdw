"""
llms/ — CDW LLM Clients
========================
Provides a uniform query() interface over different language models.

Available clients
─────────────────
  MedGemma27bText  — google/medgemma-27b-text-it, on-demand transformers.
                     Primary planner: metadata extraction, tool selection,
                     QC interpretation. Requires 2× L40S (~54 GB bfloat16).

  Qwen3            — Qwen3 family, general-purpose text reasoning.
                     Supplementary: JSON normalisation, report formatting,
                     light classification, fallback when MedGemma is occupied.
                     Supports multiple sizes:
                         Qwen3()           → 8B, local (default, 1 GPU, ~16 GB)
                         Qwen3(size="32b") → 32B, local (2 GPUs, ~64 GB)
                         Qwen3(mode="api") → 8B via vLLM server

All clients share BaseLLM + LLMResponse from base_llm.py and the CDW-specific
default system prompt CDW_SYSTEM_PROMPT.
"""

from .base_llm import BaseLLM, LLMResponse, CDW_SYSTEM_PROMPT
from .medgemma27b_text import MedGemma27bText
from .qwen3 import Qwen3

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "CDW_SYSTEM_PROMPT",
    "MedGemma27bText",
    "Qwen3",
]
