"""
================================================================================
Qwen3 Text Client
================================================================================
General-purpose (non-medical) LLM for supplementary reasoning tasks.
Supports multiple model sizes via a single class.

Good for: JSON extraction / schema validation, report formatting,
          structured output normalisation, light classification tasks,
          fallback when MedGemma is occupied.
Not for:  tasks requiring medical domain knowledge → use MedGemma27bText.

Available sizes
───────────────
  "8b"  — Qwen/Qwen3-8B   (~16 GB bfloat16, 1× L40S, ~8 s load)   ← default
  "32b" — Qwen/Qwen3-32B  (~64 GB bfloat16, 2× L40S, ~30 s load)

  Use "8b"  for: fast/light tasks — JSON cleanup, binary classification,
                 gating decisions, report summarisation.
  Use "32b" for: heavier reasoning — multi-step planning, ensemble selection,
                 complex QC interpretation when MedGemma is unavailable.

Two backends
────────────
  "local" — on-demand transformers (DEFAULT). No server needed. Fully offline.
  "api"   — OpenAI-compatible vLLM endpoint (optional, when server available).
            Stateless, does not consume local GPU memory.

Qwen3 thinking mode
───────────────────
  thinking=False (default) — direct response (/no_think). Fast, structured.
  thinking=True            — chain-of-thought (/think). Deeper, slower.

  Local mode uses apply_chat_template(enable_thinking=) where supported
  (transformers >= 4.51), with automatic fallback to the /no_think prefix
  on older installs. <think>…</think> blocks are stripped in either case.

Usage:
    from llms.qwen3 import Qwen3

    # Light text processing (default — 8B, local)
    with Qwen3() as llm:
        resp = llm.query("Reformat this as JSON: ...")

    # Heavier reasoning (32B, local)
    with Qwen3(size="32b") as llm:
        resp = llm.query("Select tools for this case: ...")

    # API mode (vLLM server running, any size)
    llm = Qwen3(size="32b", mode="api")
    resp = llm.query("Summarise these QC flags: ...")
"""

import gc
import logging
from typing import Dict, Literal

import torch

from .base_llm import BaseLLM, CDW_SYSTEM_PROMPT, LLMResponse
from .utils import _unwrap_input_ids
from config.constants import (
    QWEN3_8B_CHECKPOINT,
    QWEN3_32B_CHECKPOINT,
    VLLM_ENDPOINT,
    PLANNER_MAX_NEW_TOKENS,
    PLANNER_TEMPERATURE,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry — add new sizes here, zero other changes needed
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Dict] = {
    "8b": {
        "hf_id":       "Qwen/Qwen3-8B",
        "checkpoint":  QWEN3_8B_CHECKPOINT,
        "vram_gb":     16,
        "load_time_s": 8,
    },
    "32b": {
        "hf_id":       "Qwen/Qwen3-32B",
        "checkpoint":  QWEN3_32B_CHECKPOINT,
        "vram_gb":     64,
        "load_time_s": 30,
    },
}


class Qwen3(BaseLLM):
    """
    Qwen3 family — general-purpose textual reasoning for the CDW pipeline.

    Intended as a lightweight complement to MedGemma27bText for tasks that
    do not require medical specialisation.

    Examples:
        Qwen3()              → 8B, local  (fast, 1 GPU, fully offline)
        Qwen3(size="32b")    → 32B, local (heavy reasoning, 2 GPUs)
        Qwen3(mode="api")    → 8B via vLLM server
    """

    def __init__(
        self,
        size: Literal["8b", "32b"] = "8b",
        mode: Literal["local", "api"] = "local",
        checkpoint: str = "",
        api_endpoint: str = VLLM_ENDPOINT,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        thinking: bool = False,
    ):
        """
        Args:
            size:         Model size: "8b" (default, fast) or "32b" (heavy).
            mode:         "local" (transformers, default) or "api" (vLLM server).
            checkpoint:   Override local checkpoint path. Defaults to the path
                          registered for the selected size in constants.py.
            api_endpoint: vLLM OpenAI-compatible endpoint URL (api mode only).
            device:       Device map for local mode ("auto", "cuda:0", …).
            torch_dtype:  Precision for local mode ("bfloat16" / "float16").
            thinking:     Enable Qwen3 chain-of-thought mode (default False).
        """
        if size not in _REGISTRY:
            raise ValueError(
                f"Unknown Qwen3 size {size!r}. Available: {list(_REGISTRY)}"
            )

        self.size = size
        self.mode = mode
        self._meta = _REGISTRY[size]
        self.model_name = self._meta["hf_id"]
        self.checkpoint = checkpoint or self._meta["checkpoint"]
        self.api_endpoint = api_endpoint
        # In API mode the served model name must match --served-model-name on
        # the vLLM server; default to the HF ID which is the standard choice.
        self.api_model_name = self._meta["hf_id"]
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.thinking = thinking
        self._model = None
        self._tokenizer = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """No-op in API mode. Loads transformers model in local mode."""
        if self.mode == "api":
            return
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(
            f"Loading {self.model_name} ({self.size.upper()}) "
            f"from {self.checkpoint!r} — "
            f"~{self._meta['vram_gb']} GB, ~{self._meta['load_time_s']} s ..."
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            device_map=self.device,
            dtype=self.torch_dtype,          # transformers >= 5.0; replaces torch_dtype
        )
        self._model.eval()
        logger.info(f"{self.model_name} ready.")

    def unload(self) -> None:
        """No-op in API mode. Releases GPU memory in local mode."""
        if self.mode == "api" or self._model is None:
            return
        logger.info(f"Unloading {self.model_name} ...")
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"{self.model_name} unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self.mode == "api" or self._model is not None

    # ── Public query ──────────────────────────────────────────────────────────

    def query(
        self,
        user_prompt: str,
        system_prompt: str = CDW_SYSTEM_PROMPT,
        max_new_tokens: int = PLANNER_MAX_NEW_TOKENS,
        temperature: float = PLANNER_TEMPERATURE,
    ) -> LLMResponse:
        if self.mode == "api":
            return self._query_api(
                user_prompt, system_prompt, max_new_tokens, temperature
            )
        return self._query_local(
            user_prompt, system_prompt, max_new_tokens, temperature
        )

    # ── API backend ───────────────────────────────────────────────────────────

    def _query_api(
        self,
        user_prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """POST to the vLLM OpenAI-compatible endpoint."""
        import requests

        think_prefix = "" if self.thinking else "/no_think\n"
        payload = {
            "model": self.api_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": think_prefix + user_prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        # Bypass Apache reverse proxy that intercepts localhost on g2.
        proxies = {"http": None, "https": None}
        try:
            resp = requests.post(
                self.api_endpoint,
                json=payload,
                proxies=proxies,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Qwen3-{self.size.upper()} API call to {self.api_endpoint!r} failed: {e}"
            ) from e

        choice = data["choices"][0]
        content = choice["message"]["content"].strip()
        content = _strip_think_block(content, self.thinking)

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            model_name=self.api_model_name,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
        )

    # ── Local backend ─────────────────────────────────────────────────────────

    def _query_local(
        self,
        user_prompt: str,
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Run inference locally via transformers."""
        if self._model is None:
            raise RuntimeError(
                f"Qwen3-{self.size.upper()} is not loaded. "
                "Call load() first or use it as a context manager."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        # enable_thinking= added in transformers 4.51. Fall back to message
        # prefix on older installs — same effect, different mechanism.
        # transformers >= 5.0 returns BatchEncoding; _unwrap_input_ids() handles both.
        try:
            result = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=self.thinking,
            )
        except TypeError:
            think_prefix = "" if self.thinking else "/no_think\n"
            messages[-1]["content"] = think_prefix + messages[-1]["content"]
            result = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        input_ids = _unwrap_input_ids(result).to(self._model.device)
        prompt_len = input_ids.shape[-1]

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][prompt_len:]
        response_text = self._tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        response_text = _strip_think_block(response_text, self.thinking)

        return LLMResponse(
            content=response_text,
            model_name=self.model_name,
            prompt_tokens=prompt_len,
            completion_tokens=new_tokens.shape[-1],
            finish_reason="stop",
        )

    def __repr__(self) -> str:
        return (
            f"Qwen3(size={self.size!r}, mode={self.mode!r}, "
            f"thinking={self.thinking}, loaded={self.is_loaded})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _strip_think_block(text: str, thinking: bool) -> str:
    """Remove residual <think>…</think> when thinking mode is off."""
    if thinking or "<think>" not in text:
        return text
    end = text.find("</think>")
    if end != -1:
        return text[end + len("</think>"):].strip()
    return text
