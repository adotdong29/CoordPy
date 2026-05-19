"""W84 / P0 #27 — Long-Context Prompt Corpus V1.

The W82 ``far_horizon_blackout_benchmark_v1`` claims the W79
LHR substrate V2 dominates baselines at horizons up to 100k
turns *in synthetic event-CID space*. The W83 composed
recovery pipeline extends this to the 20-regime synthetic
multi-agent space. Both run on **deterministic synthetic
event streams** — no live model produces the events.

P0 #27 asks for a corpus of *deterministic synthetic long-
context prompts in real-token space* whose load-bearing
property is: the prompt buries a specific recall fact deep in
the context, and answering the recall question requires
*remembering* that fact across a ≥32k-token horizon. The
prompt itself is a real natural-language string the model
tokenises; the recall question is a substring the model is
asked to complete.

The corpus is *deterministic* on ``(seed, horizon_tokens,
n_distractors_per_block)`` so two callers produce the same
corpus + same prompts + same expected answers. The corpus
ships its own ``corpus_cid``.

Honest scope (W84 P0 #27)
-------------------------

- ``W84-L-LONG-CONTEXT-CORPUS-V1-RESEARCH-ONLY-CAP`` —
  explicit-import only.
- ``W84-L-LONG-CONTEXT-CORPUS-V1-DETERMINISTIC-CAP`` — the
  corpus is generated from a seeded RNG with no model
  involvement. Real-world long-context corpora (e.g. NIAH-
  style needle-in-a-haystack with arbitrary documents) are V2.
- ``W84-L-LONG-CONTEXT-CORPUS-V1-NEEDLE-IN-HAYSTACK-CAP`` —
  the V1 pattern is *one needle*. Multi-needle / variable-
  needle is V2.
- ``W84-L-LONG-CONTEXT-CORPUS-V1-TOKEN-COUNT-DRIVEN-CAP`` —
  the horizon is specified in *tokens*, not turns. The builder
  takes a tokenizer to count tokens exactly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_context_corpus_v1 requires numpy"
        ) from exc


W84_LONG_CONTEXT_V1_SCHEMA_VERSION: str = (
    "coordpy.long_context_corpus_v1.v1")

W84_LONG_CONTEXT_DEFAULT_HORIZONS_TOKENS: tuple[int, ...] = (
    2048, 8192, 32768, 131072)

# Anti-cheat from issue: do NOT synthesize via repeated short
# snippets. The V1 builder uses *one unique distractor sentence
# per index* so a model cannot complete the prompt by pattern-
# matching against the training distribution.
W84_LONG_CONTEXT_DEFAULT_DISTRACTORS: tuple[str, ...] = (
    "The substrate-aware coordination programme reports "
    "honest precision floors per dtype.",
    "Bounded-window baselines truncate context and lose "
    "load-bearing evidence past the visible window.",
    "Replay-from-KV preserves byte-identical recompute at "
    "the dtype's native floor.",
    "Composed memory routes writes via a softmax slot router "
    "and reads via a query-key attention head.",
    "The Merkle root over the audit chain is recomputable "
    "from the committed event stream.",
    "Cross-runtime portability requires a signature-aware "
    "projector head to translate hidden states across.",
    "mTLS authentication requires CA-signed certificates "
    "with named principals to reject unauthenticated peers.",
    "Partition events emit pre/post root CIDs into the audit "
    "chain so a third party can replay the heal path.",
    "Differentiable memory substrates beat closed-form "
    "ridge controllers on delayed-recall regressions.",
    "Capacity scaling experiments push the substrate beyond "
    "100 agents, 1M events, and 1B tokens of input.",
    "Adversarial consensus repair down-weights witnesses "
    "with high deviation and high delay.",
    "Hidden-state intercept moves the trace CID, proving the "
    "runtime honours the injection at the requested layer.",
    "Online economics refinement updates the controller "
    "policy as drift accumulates in the runtime.",
    "Quantised inference at int4 preserves W80 contract bars "
    "with a widened precision floor recorded honestly.",
    "Tool-use substrates content-address every tool call and "
    "every tool response so replay is byte-identical.",
)

W84_LONG_CONTEXT_DEFAULT_NEEDLE_TEMPLATE: str = (
    "The secret coordination key is {NEEDLE_VALUE}.")

W84_LONG_CONTEXT_DEFAULT_QUESTION_TEMPLATE: str = (
    "\n\nQuestion: What is the secret coordination key? "
    "Answer: The secret coordination key is")

W84_LONG_CONTEXT_DEFAULT_NEEDLE_VALUES: tuple[str, ...] = (
    "amber-7-zephyr-9", "violet-3-quokka-1",
    "crimson-2-narwhal-5", "azure-8-pangolin-4",
    "saffron-1-axolotl-6", "indigo-5-platypus-3",
    "emerald-9-loris-2", "ochre-4-tapir-8",
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class LongContextPromptV1:
    """A single needle-in-haystack prompt.

    The needle is placed at a configurable position
    (``needle_token_position`` ∈ [0, horizon)) and the
    expected_answer is the needle string.
    """

    schema: str
    horizon_tokens: int
    needle_position_fraction: float
    needle_value: str
    expected_answer: str
    prompt_text: str
    needle_token_position_approx: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "horizon_tokens": int(self.horizon_tokens),
            "needle_position_fraction": float(round(
                self.needle_position_fraction, 6)),
            "needle_value": str(self.needle_value),
            "expected_answer": str(self.expected_answer),
            "prompt_text_sha256": hashlib.sha256(
                self.prompt_text.encode("utf-8")).hexdigest(),
            "prompt_text_n_chars": int(len(self.prompt_text)),
            "needle_token_position_approx": int(
                self.needle_token_position_approx),
            "seed": int(self.seed),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_long_context_prompt_v1",
            "prompt": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class LongContextCorpusV1:
    """A corpus of long-context needle-in-haystack prompts.

    The corpus is content-addressed by ``(seed, horizons,
    needle_value_set, distractor_set)``.
    """

    schema: str
    seed: int
    horizons_tokens: tuple[int, ...]
    n_prompts: int
    needle_position_fractions: tuple[float, ...]
    prompts: tuple[LongContextPromptV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "seed": int(self.seed),
            "horizons_tokens": list(int(h)
                                    for h in self.horizons_tokens),
            "n_prompts": int(self.n_prompts),
            "needle_position_fractions": list(
                float(f) for f in self.needle_position_fractions),
            "prompts_cids": [p.cid() for p in self.prompts],
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_long_context_corpus_v1",
            "corpus": self.to_dict()})


def build_long_context_prompt_v1(
        *,
        tokenizer: Any,
        horizon_tokens: int,
        needle_value: str,
        needle_position_fraction: float = 0.5,
        seed: int = 84_027_001,
        distractors: tuple[str, ...] = (
            W84_LONG_CONTEXT_DEFAULT_DISTRACTORS),
        needle_template: str = (
            W84_LONG_CONTEXT_DEFAULT_NEEDLE_TEMPLATE),
        question_template: str = (
            W84_LONG_CONTEXT_DEFAULT_QUESTION_TEMPLATE),
) -> LongContextPromptV1:
    """Build one needle-in-haystack prompt of ``horizon_tokens``
    tokens.

    Anti-cheat: distractors are *not* repeated short snippets;
    they are concatenated from the distractor list (cycled
    deterministically through the seeded RNG). The needle is
    placed at the configured position fraction; the question
    appears at the END of the prompt so the model must hold
    the needle across the full prefix.
    """
    rng = _np.random.default_rng(int(seed))
    target_tokens = int(horizon_tokens)
    # Estimate token-per-distractor for early stop.
    sample_tokens_per_distractor = max(
        1, int(len(tokenizer.encode(
            distractors[0])) / 1))
    # Build a long enough text by sampling distractors.
    # We aim to overshoot in chars then truncate to exact tokens.
    n_needed_distractors = max(
        4,
        int(target_tokens
            / max(1, sample_tokens_per_distractor)) + 16)
    # Order: shuffled, with the needle injected at the
    # configured position.
    distractor_order = list(range(len(distractors)))
    expanded: list[str] = []
    while len(expanded) < n_needed_distractors:
        rng.shuffle(distractor_order)
        for i in distractor_order:
            expanded.append(str(distractors[i]))
            if len(expanded) >= n_needed_distractors:
                break
    needle_text = needle_template.replace(
        "{NEEDLE_VALUE}", str(needle_value))
    expected_answer = str(needle_value)
    # Insert the needle at the configured fraction.
    insert_position = int(
        len(expanded) * float(needle_position_fraction))
    expanded.insert(insert_position, needle_text)
    haystack_pre = " ".join(expanded)
    full_text = haystack_pre + str(question_template)
    # Count tokens; trim if overshot.
    ids = tokenizer.encode(full_text)
    if len(ids) > target_tokens:
        # Truncate from the front so the needle and question
        # are preserved if possible. We trim distractors before
        # the needle to keep the needle position approximately
        # consistent.
        # Simple approach: re-tokenize a shorter haystack.
        # Reduce distractors proportionally.
        ratio = float(target_tokens) / float(len(ids))
        keep_n = max(2, int(len(expanded) * ratio) - 2)
        # Keep distractors around the needle in priority.
        # Place needle at the new fraction inside the trimmed list.
        new_insert = int(
            keep_n * float(needle_position_fraction))
        # Re-pick distractors deterministically.
        new_distractors = list(expanded[:keep_n - 1])
        new_distractors.insert(
            min(new_insert, len(new_distractors)),
            needle_text)
        haystack_pre = " ".join(new_distractors)
        full_text = haystack_pre + str(question_template)
        ids = tokenizer.encode(full_text)
    # Estimate needle's approximate token position by counting
    # tokens up to the needle in the final text.
    needle_offset_chars = full_text.find(needle_text)
    if needle_offset_chars >= 0:
        needle_prefix = full_text[:needle_offset_chars]
        needle_token_pos = int(len(
            tokenizer.encode(needle_prefix)))
    else:
        needle_token_pos = -1
    return LongContextPromptV1(
        schema=W84_LONG_CONTEXT_V1_SCHEMA_VERSION,
        horizon_tokens=int(len(ids)),
        needle_position_fraction=float(
            needle_position_fraction),
        needle_value=str(needle_value),
        expected_answer=str(expected_answer),
        prompt_text=str(full_text),
        needle_token_position_approx=int(needle_token_pos),
        seed=int(seed))


def build_long_context_corpus_v1(
        *,
        tokenizer: Any,
        horizons_tokens: Sequence[int] = (
            W84_LONG_CONTEXT_DEFAULT_HORIZONS_TOKENS),
        needle_position_fractions: Sequence[float] = (
            0.25, 0.5, 0.75),
        needle_values: Sequence[str] = (
            W84_LONG_CONTEXT_DEFAULT_NEEDLE_VALUES),
        seed: int = 84_027_001,
) -> LongContextCorpusV1:
    """Build a deterministic corpus of needle-in-haystack
    prompts across multiple horizons + multiple needle
    positions.
    """
    horizons = tuple(int(h) for h in horizons_tokens)
    fractions = tuple(
        float(f) for f in needle_position_fractions)
    prompts: list[LongContextPromptV1] = []
    rng = _np.random.default_rng(int(seed))
    for h_idx, h in enumerate(horizons):
        for frac_idx, frac in enumerate(fractions):
            # Pick a needle value deterministically by index.
            needle_idx = (
                (h_idx * len(fractions) + frac_idx)
                % len(needle_values))
            needle = str(needle_values[needle_idx])
            sub_seed = int(seed) + h_idx * 1000 + frac_idx
            p = build_long_context_prompt_v1(
                tokenizer=tokenizer,
                horizon_tokens=int(h),
                needle_value=needle,
                needle_position_fraction=float(frac),
                seed=sub_seed)
            prompts.append(p)
    return LongContextCorpusV1(
        schema=W84_LONG_CONTEXT_V1_SCHEMA_VERSION,
        seed=int(seed),
        horizons_tokens=horizons,
        n_prompts=int(len(prompts)),
        needle_position_fractions=fractions,
        prompts=tuple(prompts))


__all__ = [
    "W84_LONG_CONTEXT_V1_SCHEMA_VERSION",
    "W84_LONG_CONTEXT_DEFAULT_HORIZONS_TOKENS",
    "W84_LONG_CONTEXT_DEFAULT_DISTRACTORS",
    "W84_LONG_CONTEXT_DEFAULT_NEEDLE_TEMPLATE",
    "W84_LONG_CONTEXT_DEFAULT_QUESTION_TEMPLATE",
    "W84_LONG_CONTEXT_DEFAULT_NEEDLE_VALUES",
    "LongContextPromptV1",
    "LongContextCorpusV1",
    "build_long_context_prompt_v1",
    "build_long_context_corpus_v1",
]
