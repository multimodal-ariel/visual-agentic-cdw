"""
llms/utils.py — Internal helpers shared across LLM clients.
"""

import torch


def _unwrap_input_ids(result) -> torch.Tensor:
    """
    Normalize the output of tokenizer.apply_chat_template(..., return_tensors="pt").

    transformers < 5.0  → returns a raw torch.Tensor directly.
    transformers >= 5.0 → returns a BatchEncoding (dict-like) with an
                          "input_ids" key containing the tensor.

    This helper transparently handles both so the calling code doesn't
    need to branch on the transformers version.
    """
    if isinstance(result, torch.Tensor):
        return result
    # BatchEncoding / dict-like
    return result["input_ids"]
