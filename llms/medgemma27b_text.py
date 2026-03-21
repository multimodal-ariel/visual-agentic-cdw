"""
================================================================================
MedGemma-27B Text Client
================================================================================
Loads google/medgemma-27b-text-it on-demand via HuggingFace transformers.
No vLLM server required — suitable for batch pipeline use.

Hardware:   2× L40S 46GB  (device_map="auto" splits across both GPUs)
Load time:  ~30 s
Memory:     ~54 GB (bfloat16)

Validated performance (see project notes, 2026-03-18):
    Modality extraction     10/10
    Anatomy extraction      10/10
    is_diagnostic           17/20
    MRI sequence recog.     10/10
    Contrast status          9/10
    Tool selection           8/10
    QC interpretation        9/10

Usage:
    from llms.medgemma27b_text import MedGemma27bText

    with MedGemma27bText() as llm:
        resp = llm.query("Classify this DICOM series: CT_CHEST_AX_DIAG ...")
        print(resp.content)

    # Or explicit load/unload:
    llm = MedGemma27bText()
    llm.load()
    resp = llm.query(user_prompt="...", temperature=0.0)
    llm.unload()
"""

import gc
import logging
from typing import Optional

import torch

from .base_llm import BaseLLM, CDW_SYSTEM_PROMPT, LLMResponse
from .utils import _unwrap_input_ids
from config.constants import (
    MEDGEMMA_27B_CHECKPOINT,
    PLANNER_DEVICE,
    PLANNER_MAX_NEW_TOKENS,
    PLANNER_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class MedGemma27bText(BaseLLM):
    """
    MedGemma-27B (text-only instruction-tuned).

    This is the primary planner model for the CDW pipeline. Use it for all
    tasks that require medical domain knowledge: metadata extraction, modality
    classification, tool selection, and QC interpretation.

    For non-medical supplementary tasks (formatting, summarisation, JSON
    normalisation) prefer Qwen332b — it works offline in local mode and
    optionally via a vLLM API when a server is available.
    """

    model_name = "google/medgemma-27b-text-it"

    def __init__(
        self,
        checkpoint: str = MEDGEMMA_27B_CHECKPOINT,
        device: str = PLANNER_DEVICE,
        torch_dtype: str = "bfloat16",
    ):
        """
        Args:
            checkpoint:   Path to local checkpoint directory (or HF model ID).
            device:       Device map string: "auto", "cuda:0", "cuda:0,1", etc.
            torch_dtype:  Tensor precision: "bfloat16" (default) or "float16".
        """
        self.checkpoint = checkpoint
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self._model = None
        self._tokenizer = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model + tokenizer into GPU memory. No-op if already loaded."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading {self.model_name} from {self.checkpoint!r} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            device_map=self.device,
            dtype=self.torch_dtype,          # transformers >= 5.0; replaces torch_dtype
        )
        self._model.eval()
        logger.info(f"{self.model_name} ready.")

    def unload(self) -> None:
        """Delete model from GPU memory and run gc + cuda cache clear."""
        if self._model is None:
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
        return self._model is not None

    # ── Inference ─────────────────────────────────────────────────────────────

    def query(
        self,
        user_prompt: str,
        system_prompt: str = CDW_SYSTEM_PROMPT,
        max_new_tokens: int = PLANNER_MAX_NEW_TOKENS,
        temperature: float = PLANNER_TEMPERATURE,
    ) -> LLMResponse:
        """
        Run a single-turn inference pass and return the model's response.

        Raises:
            RuntimeError: If called before load() (or outside a context manager).
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} is not loaded. "
                "Call load() first or use it as a context manager."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        # Apply the model's chat template.
        # transformers >= 5.0 returns BatchEncoding; earlier versions return a
        # raw tensor. _unwrap_input_ids() handles both transparently.
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

        # Decode only the newly generated tokens (exclude the prompt)
        new_tokens = output_ids[0][prompt_len:]
        response_text = self._tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        return LLMResponse(
            content=response_text,
            model_name=self.model_name,
            prompt_tokens=prompt_len,
            completion_tokens=new_tokens.shape[-1],
            finish_reason="stop",
        )
