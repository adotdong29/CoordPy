"""Token accounting for the CASR benchmark.

We want a single consistent estimator across legs of the benchmark. The exact
numbers don't need to match any one model's tokenizer; they need to be
comparable across "full", "casr", and "ablation" runs.

Strategy:
  1. Try tiktoken's cl100k_base encoder (standard, fast, deterministic).
  2. If tiktoken is not installed, fall back to len(text) // 4, the canonical
     ballpark estimate for English text.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_encoder():
    """Cached accessor. Returns either a tiktoken encoding or None (fallback)."""
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # ImportError, or any runtime error loading the BPE file.
        return None


def count_tokens(text: str, model_hint: str = "qwen") -> int:
    """Count tokens in `text`. `model_hint` is accepted for API symmetry but
    the estimator is the same across models — we just need consistency across
    benchmark legs."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is None:
        return max(0, len(text) // 4)
    return len(enc.encode(text))


def count_prompt(prompt: str) -> int:
    return count_tokens(prompt)


def count_completion(completion: str) -> int:
    return count_tokens(completion)
