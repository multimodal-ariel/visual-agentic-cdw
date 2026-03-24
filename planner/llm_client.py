"""
CDW Agentic Pipeline — LLM Planner Client
==========================================
Two backends, one API:

  TransformersLLM — on-demand local loading via HuggingFace Transformers (default).
      Loads model + tokenizer on llm.load(), frees GPU memory on llm.unload().
      Use for MedGemma-27B, Qwen3-8B, Qwen3-32B.

  VLLMClient — OpenAI-compatible HTTP client for a running vLLM server (optional).
      Zero GPU overhead on the client side.
      Use when a vLLM server is already running externally.

  PlannerLLM — factory that selects the right backend from a model path or
      endpoint URL and exposes a unified query() / query_json() API.

Usage
-----
# Default: local transformers (on-demand load/unload)
llm = PlannerLLM.from_local("checkpoints/medgemma-27b-text-it", device="auto")
llm.load()
result = llm.query(system_prompt="You are an expert.", user_prompt="What is CT?")
llm.unload()

# Optional: vLLM server (if running externally)
llm = PlannerLLM.from_vllm("http://127.0.0.1:41260", model_id="Qwen/Qwen3-32B")
result = llm.query(system_prompt="", user_prompt="Select tools for this case: ...")

# JSON extraction (strips markdown fences + Qwen3 <think> blocks)
data = llm.query_json(system_prompt="", user_prompt=prompt)
"""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_json_fence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` markdown fences."""
    text = text.strip()
    # Remove leading ```json or ``` fence
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    # Remove trailing ``` fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (Qwen3 thinking mode).
    Also handles unclosed <think> blocks (truncated output)."""
    # Remove closed think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Remove unclosed think block (truncated — no </think> found)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    return text


def _parse_json_from_response(text: str) -> dict:
    """
    Extract a JSON object from a model response.
    Handles:
      - ```json ... ``` fences
      - <think>...</think> preambles (Qwen3)
      - Trailing commentary after the closing }
    """
    text = _strip_think_blocks(text)
    text = _strip_json_fence(text)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Find a JSON array [ ... ]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from model response:\n{text[:500]}")


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class _BaseLLM(ABC):

    @abstractmethod
    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """Run a chat query and return the raw text response."""

    def query_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> Any:
        """Run a query and parse the response as JSON. Returns dict or list."""
        raw = self.query(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return _parse_json_from_response(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Transformers backend
# ──────────────────────────────────────────────────────────────────────────────

class TransformersLLM(_BaseLLM):
    """
    On-demand HuggingFace Transformers model.
    Call load() before querying and unload() after to free GPU memory.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        """
        Args:
            model_path: Local path to the model directory (e.g. checkpoints/medgemma-27b-text-it).
            device: "auto" distributes across all available GPUs. Or "cuda:0", etc.
            torch_dtype: "bfloat16" (default) or "float16".
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load the model and tokenizer into GPU memory."""
        if self._model is not None:
            logger.debug("TransformersLLM.load(): already loaded, skipping.")
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)

        logger.info("Loading LLM from %s (dtype=%s, device=%s) ...", self.model_path, self.torch_dtype, self.device)
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Use 'dtype' kwarg for transformers >= 5.x, fall back to 'torch_dtype'
        import transformers
        tf_major = int(transformers.__version__.split(".")[0])
        dtype_kwarg = "dtype" if tf_major >= 5 else "torch_dtype"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **{dtype_kwarg: dtype},
            device_map=self.device,
        )
        self._model.eval()

        elapsed = time.time() - t0
        logger.info("LLM loaded in %.1fs.", elapsed)

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is None:
            return
        logger.info("Unloading LLM from GPU memory.")
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def is_loaded(self) -> bool:
        return self._model is not None

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call TransformersLLM.load() first.")

        import torch

        # Build messages list; omit system turn if empty (some models don't support it)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Tokenize using the model's chat template
        tokenized = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Handle both old (raw tensor) and new (BatchEncoding) transformers APIs
        if isinstance(tokenized, dict) or hasattr(tokenized, "input_ids"):
            input_ids = tokenized["input_ids"]
        else:
            input_ids = tokenized

        # Move to model's device
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        input_len = input_ids.shape[-1]

        generate_kwargs: dict = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(**generate_kwargs)

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_len:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()


# ──────────────────────────────────────────────────────────────────────────────
# vLLM backend (OpenAI-compatible HTTP)
# ──────────────────────────────────────────────────────────────────────────────

class VLLMClient(_BaseLLM):
    """
    Client for a running vLLM server (OpenAI-compatible /v1/chat/completions).

    The server is started separately, e.g.:
        vllm serve Qwen/Qwen3-32B --host 127.0.0.1 --port 41260 --generation-config vllm
    """

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        timeout: int = 300,
    ):
        """
        Args:
            endpoint: Full URL to the chat completions endpoint,
                      e.g. "http://127.0.0.1:41260/v1/chat/completions".
            model_id: Model name as registered in the vLLM server
                      (e.g. "Qwen/Qwen3-32B").
            timeout: HTTP request timeout in seconds.
        """
        self.endpoint = endpoint
        self.model_id = model_id
        self.timeout = timeout

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        import requests

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
                proxies={"http": None, "https": None},  # bypass Apache proxy on localhost
            )
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"vLLM request to {self.endpoint} failed: {e}") from e

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ──────────────────────────────────────────────────────────────────────────────
# Unified PlannerLLM
# ──────────────────────────────────────────────────────────────────────────────

class PlannerLLM(_BaseLLM):
    """
    Unified LLM client. Wraps either TransformersLLM or VLLMClient.

    Preferred constructors:
        PlannerLLM.from_local(model_path, ...)   → TransformersLLM
        PlannerLLM.from_vllm(base_url, ...)      → VLLMClient
        PlannerLLM.from_constants()              → reads config/constants.py defaults

    Direct constructor (for backwards compatibility with example_prompts.py):
        PlannerLLM(model_path, device="auto")    → TransformersLLM
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        """Direct constructor — creates a TransformersLLM backend."""
        self._backend = TransformersLLM(model_path, device=device, torch_dtype=torch_dtype)

    # ── Alternate constructors ──────────────────────────────────────────────

    @classmethod
    def from_local(
        cls,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
    ) -> "PlannerLLM":
        """Create a PlannerLLM backed by a local HuggingFace model."""
        obj = cls.__new__(cls)
        obj._backend = TransformersLLM(model_path, device=device, torch_dtype=torch_dtype)
        return obj

    @classmethod
    def from_vllm(
        cls,
        base_url: str,
        model_id: str,
        timeout: int = 300,
    ) -> "PlannerLLM":
        """
        Create a PlannerLLM backed by a running vLLM server.

        Args:
            base_url: Base URL of the vLLM server,
                      e.g. "http://127.0.0.1:41260".
                      The /v1/chat/completions path is appended automatically.
            model_id: Model name as served (e.g. "Qwen/Qwen3-32B").
            timeout: HTTP timeout in seconds.
        """
        obj = cls.__new__(cls)
        endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        obj._backend = VLLMClient(endpoint=endpoint, model_id=model_id, timeout=timeout)
        return obj

    @classmethod
    def from_constants(cls) -> "PlannerLLM":
        """
        Create a PlannerLLM using defaults from config/constants.py.
        Uses TransformersLLM with MEDGEMMA_27B_CHECKPOINT.
        """
        from config.constants import (
            MEDGEMMA_27B_CHECKPOINT,
            PLANNER_DEVICE,
            PLANNER_TORCH_DTYPE,
        )
        return cls.from_local(
            MEDGEMMA_27B_CHECKPOINT,
            device=PLANNER_DEVICE,
            torch_dtype=PLANNER_TORCH_DTYPE,
        )

    # ── Lifecycle (only meaningful for TransformersLLM backend) ────────────

    def load(self) -> None:
        """Load the model into GPU memory (TransformersLLM only)."""
        if isinstance(self._backend, TransformersLLM):
            self._backend.load()

    def unload(self) -> None:
        """Free GPU memory (TransformersLLM only)."""
        if isinstance(self._backend, TransformersLLM):
            self._backend.unload()

    def is_loaded(self) -> bool:
        if isinstance(self._backend, TransformersLLM):
            return self._backend.is_loaded()
        return True  # vLLM is always "ready"

    # ── Query ───────────────────────────────────────────────────────────────

    def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        return self._backend.query(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    # ── Context manager ─────────────────────────────────────────────────────

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()
