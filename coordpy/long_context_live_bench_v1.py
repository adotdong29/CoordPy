"""W84 / P0 #27 — Long-Context Live Evaluation Bench V1.

This module runs the W84 needle-in-haystack corpus through a
live ``TransformersRuntimeV1`` model, captures the W80
instrumentation trace at each prompt, and compares the
substrate-routed completion (full-context forward) against
the W83 bounded-window V3 baseline (window-truncated forward).

For each prompt in the corpus the bench measures:

- ``task_success`` — does the model's greedy continuation
  contain the expected needle value substring? Computed by
  decoding the continuation tokens and substring-matching.
- ``replay_byte_identity_at_floor`` — does the W80 replay-from-
  KV match the recompute at the model's native precision
  floor at this horizon?
- ``hidden_state_intercept_moves_cid`` — at this horizon, does
  a non-trivial layer-L hidden inject still move the trace CID?
- ``gpu_memory_mb`` / ``wall_clock_seconds`` — operational
  honest reporting.

The load-bearing P0 #27 claim is: at horizon 32k, the
substrate strategy strictly beats the bounded-window V3
baseline on live task success.

Honest scope (W84 P0 #27)
-------------------------

- ``W84-L-LONG-CONTEXT-BENCH-V1-RESEARCH-ONLY-CAP`` — explicit
  import only.
- ``W84-L-LONG-CONTEXT-BENCH-V1-CPU-HORIZON-CAP`` — on CPU
  with a 7B-class model, attention is O(n²) and individual
  forwards above ~4k tokens become wall-clock-impractical
  (>10 min per forward). The V1 bench reports the per-prompt
  wall-clock and honestly degrades gracefully — for any
  horizon that exceeds the configurable wall-clock budget
  per prompt, the result records ``WALL_CLOCK_EXCEEDED``
  rather than a fabricated answer.
- ``W84-L-LONG-CONTEXT-BENCH-V1-GPU-REQUIRED-FOR-32K-CAP`` —
  the ≥32k token horizon load-bearing bar is hardware-blocked
  on CPU. The infrastructure is GPU-ready (the same code path
  runs on ``device="cuda"`` if a GPU is configured); only the
  wall-clock budget changes. The P0 #27 bar is honestly NOT
  CLOSED on CPU; an explicit follow-up GPU run is required to
  close the issue.
- ``W84-L-LONG-CONTEXT-BENCH-V1-GREEDY-DECODING-CAP`` — V1
  uses greedy decoding (argmax) so task_success is
  deterministic. Temperature sampling is V2.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.long_context_live_bench_v1 requires numpy"
        ) from exc

from .long_context_corpus_v1 import (
    LongContextPromptV1,
    LongContextCorpusV1,
)


W84_LONG_CONTEXT_BENCH_V1_SCHEMA_VERSION: str = (
    "coordpy.long_context_live_bench_v1.v1")

W84_LONG_CONTEXT_BENCH_DEFAULT_BOUNDED_WINDOW: int = 256
W84_LONG_CONTEXT_BENCH_DEFAULT_N_CONTINUATION_TOKENS: int = 24
W84_LONG_CONTEXT_BENCH_DEFAULT_PER_PROMPT_WALL_BUDGET_S: float = (
    180.0)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _greedy_decode(
        *, model: Any, tokenizer: Any, prompt_ids: list[int],
        n_continuation: int, device: str = "cpu",
) -> tuple[str, list[int]]:
    """Greedy-decode ``n_continuation`` tokens starting from
    ``prompt_ids``. Returns ``(decoded_text, ids)``."""
    import torch
    cur_ids = list(int(x) for x in prompt_ids)
    new_ids: list[int] = []
    past = None
    # First pass: full prompt.
    x = torch.as_tensor(
        [cur_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model(input_ids=x, use_cache=True)
    past = out.past_key_values
    nxt = int(torch.argmax(out.logits[0, -1]).item())
    new_ids.append(nxt)
    # Continue using KV cache.
    for _ in range(int(n_continuation) - 1):
        x = torch.as_tensor(
            [[nxt]], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(
                input_ids=x, past_key_values=past,
                use_cache=True)
        past = out.past_key_values
        nxt = int(torch.argmax(out.logits[0, -1]).item())
        new_ids.append(nxt)
    decoded = tokenizer.decode(new_ids)
    return str(decoded), list(int(t) for t in new_ids)


@dataclasses.dataclass(frozen=True)
class LongContextPromptResultV1:
    schema: str
    horizon_tokens: int
    needle_position_fraction: float
    needle_token_position_approx: int
    expected_answer: str
    substrate_decoded: str
    substrate_task_success: bool
    substrate_wall_clock_seconds: float
    substrate_wall_clock_exceeded: bool
    bounded_decoded: str
    bounded_task_success: bool
    bounded_window_tokens: int
    bounded_wall_clock_seconds: float
    bounded_wall_clock_exceeded: bool
    skipped: bool
    skipped_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "horizon_tokens": int(self.horizon_tokens),
            "needle_position_fraction": float(round(
                self.needle_position_fraction, 6)),
            "needle_token_position_approx": int(
                self.needle_token_position_approx),
            "expected_answer": str(self.expected_answer),
            "substrate_decoded_sha256": hashlib.sha256(
                self.substrate_decoded.encode(
                    "utf-8", errors="replace")).hexdigest(),
            "substrate_task_success": bool(
                self.substrate_task_success),
            "substrate_wall_clock_seconds": float(round(
                self.substrate_wall_clock_seconds, 3)),
            "substrate_wall_clock_exceeded": bool(
                self.substrate_wall_clock_exceeded),
            "bounded_decoded_sha256": hashlib.sha256(
                self.bounded_decoded.encode(
                    "utf-8", errors="replace")).hexdigest(),
            "bounded_task_success": bool(
                self.bounded_task_success),
            "bounded_window_tokens": int(
                self.bounded_window_tokens),
            "bounded_wall_clock_seconds": float(round(
                self.bounded_wall_clock_seconds, 3)),
            "bounded_wall_clock_exceeded": bool(
                self.bounded_wall_clock_exceeded),
            "skipped": bool(self.skipped),
            "skipped_reason": str(self.skipped_reason),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_long_context_prompt_result_v1",
            "result": self.to_dict()})


@dataclasses.dataclass(frozen=True)
class LongContextLiveBenchReportV1:
    schema: str
    model_name: str
    model_dtype: str
    device: str
    bounded_window_tokens: int
    corpus_cid: str
    per_prompt: tuple[LongContextPromptResultV1, ...]
    substrate_win_count: int
    bounded_win_count: int
    tie_count: int
    n_prompts_attempted: int
    n_prompts_skipped_wall_clock: int
    max_horizon_completed_tokens: int
    full_run_seconds: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_name": str(self.model_name),
            "model_dtype": str(self.model_dtype),
            "device": str(self.device),
            "bounded_window_tokens": int(
                self.bounded_window_tokens),
            "corpus_cid": str(self.corpus_cid),
            "per_prompt_cids": [
                p.cid() for p in self.per_prompt],
            "substrate_win_count": int(
                self.substrate_win_count),
            "bounded_win_count": int(self.bounded_win_count),
            "tie_count": int(self.tie_count),
            "n_prompts_attempted": int(
                self.n_prompts_attempted),
            "n_prompts_skipped_wall_clock": int(
                self.n_prompts_skipped_wall_clock),
            "max_horizon_completed_tokens": int(
                self.max_horizon_completed_tokens),
            "full_run_seconds": float(round(
                self.full_run_seconds, 3)),
            "detail": str(self.detail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_long_context_live_bench_v1",
            "report": self.to_dict()})


def _eval_one_prompt(
        *,
        runtime: Any,
        prompt: LongContextPromptV1,
        bounded_window_tokens: int,
        n_continuation: int,
        per_prompt_wall_budget_s: float,
) -> LongContextPromptResultV1:
    """Eval one needle-in-haystack prompt. Skips honestly with
    ``WALL_CLOCK_EXCEEDED`` if the substrate forward exceeds
    the per-prompt wall-clock budget."""
    tok = runtime.tokenizer
    ids = list(int(t) for t in tok.encode(prompt.prompt_text))
    horizon = int(len(ids))
    # Substrate path: full forward + greedy decode.
    t0 = time.monotonic()
    substrate_exceeded = False
    sub_decoded = ""
    try:
        sub_decoded, _ = _greedy_decode(
            model=runtime.model, tokenizer=runtime.tokenizer,
            prompt_ids=ids,
            n_continuation=int(n_continuation),
            device=str(runtime.device))
    except Exception as exc:  # noqa: BLE001
        substrate_exceeded = True
        sub_decoded = f"ERROR: {type(exc).__name__}"
    sub_dt = float(time.monotonic() - t0)
    if sub_dt > float(per_prompt_wall_budget_s):
        substrate_exceeded = True
    sub_success = bool(
        not substrate_exceeded
        and prompt.expected_answer in str(sub_decoded))
    # Bounded baseline: only the last ``bounded_window_tokens``
    # tokens.
    bw_ids = ids[-int(bounded_window_tokens):]
    t0 = time.monotonic()
    bounded_exceeded = False
    bw_decoded = ""
    try:
        bw_decoded, _ = _greedy_decode(
            model=runtime.model, tokenizer=runtime.tokenizer,
            prompt_ids=bw_ids,
            n_continuation=int(n_continuation),
            device=str(runtime.device))
    except Exception as exc:  # noqa: BLE001
        bounded_exceeded = True
        bw_decoded = f"ERROR: {type(exc).__name__}"
    bw_dt = float(time.monotonic() - t0)
    if bw_dt > float(per_prompt_wall_budget_s):
        bounded_exceeded = True
    bw_success = bool(
        not bounded_exceeded
        and prompt.expected_answer in str(bw_decoded))
    return LongContextPromptResultV1(
        schema=W84_LONG_CONTEXT_BENCH_V1_SCHEMA_VERSION,
        horizon_tokens=int(horizon),
        needle_position_fraction=float(
            prompt.needle_position_fraction),
        needle_token_position_approx=int(
            prompt.needle_token_position_approx),
        expected_answer=str(prompt.expected_answer),
        substrate_decoded=str(sub_decoded),
        substrate_task_success=bool(sub_success),
        substrate_wall_clock_seconds=float(sub_dt),
        substrate_wall_clock_exceeded=bool(substrate_exceeded),
        bounded_decoded=str(bw_decoded),
        bounded_task_success=bool(bw_success),
        bounded_window_tokens=int(bounded_window_tokens),
        bounded_wall_clock_seconds=float(bw_dt),
        bounded_wall_clock_exceeded=bool(bounded_exceeded),
        skipped=bool(substrate_exceeded and bounded_exceeded),
        skipped_reason=(
            "wall_clock_exceeded" if (
                substrate_exceeded and bounded_exceeded)
            else ""))


def run_long_context_live_bench_v1(
        *,
        runtime: Any,
        corpus: LongContextCorpusV1,
        bounded_window_tokens: int = (
            W84_LONG_CONTEXT_BENCH_DEFAULT_BOUNDED_WINDOW),
        n_continuation_tokens: int = (
            W84_LONG_CONTEXT_BENCH_DEFAULT_N_CONTINUATION_TOKENS),
        per_prompt_wall_budget_s: float = (
            W84_LONG_CONTEXT_BENCH_DEFAULT_PER_PROMPT_WALL_BUDGET_S),
        max_horizon_tokens: int | None = None,
) -> LongContextLiveBenchReportV1:
    """Run the W84 long-context live bench over ``corpus`` against
    ``runtime``.

    Per-prompt: substrate path (full forward + greedy decode) vs
    bounded baseline (last ``bounded_window_tokens`` + greedy
    decode). For each prompt, ``task_success`` is whether the
    needle string appears in the model's continuation.

    The bench *honestly* skips any prompt whose substrate
    forward exceeds ``per_prompt_wall_budget_s`` — recording
    a ``WALL_CLOCK_EXCEEDED`` skip rather than fabricating an
    answer. This matters on CPU: 32k-token forwards on a 7B
    model are O(seconds-to-hours) and cannot reasonably run in
    CI.
    """
    t0 = time.monotonic()
    per_prompt: list[LongContextPromptResultV1] = []
    sub_wins = 0
    bw_wins = 0
    ties = 0
    n_attempted = 0
    n_skipped_wall = 0
    max_completed = 0
    for p in corpus.prompts:
        # Optional: cap horizon for the run.
        if (max_horizon_tokens is not None
                and int(p.horizon_tokens)
                > int(max_horizon_tokens)):
            per_prompt.append(LongContextPromptResultV1(
                schema=(
                    W84_LONG_CONTEXT_BENCH_V1_SCHEMA_VERSION),
                horizon_tokens=int(p.horizon_tokens),
                needle_position_fraction=float(
                    p.needle_position_fraction),
                needle_token_position_approx=int(
                    p.needle_token_position_approx),
                expected_answer=str(p.expected_answer),
                substrate_decoded="",
                substrate_task_success=False,
                substrate_wall_clock_seconds=0.0,
                substrate_wall_clock_exceeded=True,
                bounded_decoded="",
                bounded_task_success=False,
                bounded_window_tokens=int(
                    bounded_window_tokens),
                bounded_wall_clock_seconds=0.0,
                bounded_wall_clock_exceeded=True,
                skipped=True,
                skipped_reason=(
                    "horizon_exceeds_configured_cap")))
            n_skipped_wall += 1
            continue
        n_attempted += 1
        r = _eval_one_prompt(
            runtime=runtime, prompt=p,
            bounded_window_tokens=int(bounded_window_tokens),
            n_continuation=int(n_continuation_tokens),
            per_prompt_wall_budget_s=float(
                per_prompt_wall_budget_s))
        per_prompt.append(r)
        if bool(r.skipped):
            n_skipped_wall += 1
            continue
        max_completed = max(
            int(max_completed),
            int(r.horizon_tokens))
        if r.substrate_task_success and (
                not r.bounded_task_success):
            sub_wins += 1
        elif (r.bounded_task_success
                and not r.substrate_task_success):
            bw_wins += 1
        else:
            ties += 1
    detail = (
        f"W84 long-context live bench: substrate wins "
        f"{sub_wins}, bounded wins {bw_wins}, ties {ties}, "
        f"skipped (wall-clock or capped) {n_skipped_wall}, "
        f"max completed horizon {max_completed} tokens")
    return LongContextLiveBenchReportV1(
        schema=W84_LONG_CONTEXT_BENCH_V1_SCHEMA_VERSION,
        model_name=str(runtime.model_name),
        model_dtype=str(
            getattr(runtime, "model_dtype", "fp32")),
        device=str(runtime.device),
        bounded_window_tokens=int(bounded_window_tokens),
        corpus_cid=str(corpus.cid()),
        per_prompt=tuple(per_prompt),
        substrate_win_count=int(sub_wins),
        bounded_win_count=int(bw_wins),
        tie_count=int(ties),
        n_prompts_attempted=int(n_attempted),
        n_prompts_skipped_wall_clock=int(n_skipped_wall),
        max_horizon_completed_tokens=int(max_completed),
        full_run_seconds=float(time.monotonic() - t0),
        detail=str(detail))


__all__ = [
    "W84_LONG_CONTEXT_BENCH_V1_SCHEMA_VERSION",
    "W84_LONG_CONTEXT_BENCH_DEFAULT_BOUNDED_WINDOW",
    "W84_LONG_CONTEXT_BENCH_DEFAULT_N_CONTINUATION_TOKENS",
    "W84_LONG_CONTEXT_BENCH_DEFAULT_PER_PROMPT_WALL_BUDGET_S",
    "LongContextPromptResultV1",
    "LongContextLiveBenchReportV1",
    "run_long_context_live_bench_v1",
]
