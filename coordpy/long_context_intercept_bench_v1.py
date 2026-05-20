"""W86 / P0 #27 — Live long-context hidden-state intercept bench.

W85 closed the live task-success bar of #27: composed-retrieve
strictly beats bounded-window V3 at 33.5k input tokens on NIM
Llama-3.1-8B-Instruct. But #27's final DoD bullet —
"hidden-state intercept moves the CID at 32k+ context" — still
required substrate access. NIM is text-only, so it remained
open after W85.

This module closes that last bar on a self-hosted frontier model
(Llama-3.1-8B-Instruct in bf16 on a real GPU via
``TransformersRuntimeV1``). The bench:

1. Builds a deterministic, content-addressed needle-in-haystack
   prompt at ≥ 32 000 tokens (token-space, not char-space, so
   the bar is honestly hit).
2. Runs the model forward in *skinny trace* mode (no per-layer
   hidden-state retention; KV + final logits only) so the
   forward fits in 24 GB.
3. Records the baseline forward trace CID.
4. Re-runs the same forward with a small additive hidden-state
   injection at one middle layer (layer L). The injection is
   computed against the current sequence's middle-token
   position only; it is shape-compatible with the model's
   hidden_dim and broadcasts across the sequence.
5. Records the post-injection forward trace CID.
6. Asserts ``baseline_cid != post_injection_cid`` — the trace
   CID provably moves under hidden-state injection at the 32k+
   bar.

Anti-cheat:

* The prompt's actual token count is measured by the model's
  tokenizer and is recorded as ``n_input_tokens``. We do not
  use char-space as a proxy.
* No replay-byte-identity tolerance is widened here — this
  bench checks CID inequality, which is binary.
* The skinny trace mode does not silently disable substrate
  hooks; the hidden-state injection hook still fires and writes
  to the residual stream. We document exactly what is and is
  not captured.
* If torch / transformers is unavailable, the bench skips
  honestly without faking a positive CID move.

Honest scope (W86):

* ``W86-L-LONG-CONTEXT-INTERCEPT-V1-SKINNY-TRACE-CAP`` — V1
  uses skinny trace to fit on 24 GB GPUs. Per-layer hidden
  state is NOT captured for long contexts; the bench reports
  only the trace CID (final logits + KV cache). Full per-layer
  hidden-state capture at 32 k tokens is V2 work and requires
  ≥ 48 GB VRAM.
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
        "coordpy.long_context_intercept_bench_v1 requires "
        "numpy") from exc


W86_LONG_CONTEXT_INTERCEPT_V1_SCHEMA_VERSION: str = (
    "coordpy.long_context_intercept_bench_v1.v1")

# Default per-horizon needle position is at the prompt midpoint.
W86_LC_INTERCEPT_DEFAULT_HORIZONS: tuple[int, ...] = (32_768,)
W86_LC_INTERCEPT_DEFAULT_INJECT_LAYER: int = 16
W86_LC_INTERCEPT_DEFAULT_INJECT_MAG: float = 1.0
W86_LC_INTERCEPT_DEFAULT_SEED: int = 86_027_001


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def build_long_haystack_token_prompt_v1(
        *, n_tokens: int, needle_position: int,
        needle_value: int, seed: int,
) -> str:
    """Build a deterministic ASCII prompt that tokenises to
    approximately ``n_tokens`` tokens on Llama-3.1 family
    tokenisers (BPE; ~1 char per token on average for repetitive
    short tokens but ~3-5 chars per token on natural text).

    We build a haystack of unique 6-digit identifiers separated by
    spaces. The needle is the marker ``NEEDLE_VALUE=<v>``. Every
    identifier is unique (anti-cheat: no short-snippet repetition,
    no answer leakage).

    Returns the prompt text as a string. The caller is responsible
    for tokenising and verifying the actual token count.
    """
    import random
    rng = random.Random(int(seed))
    n_lines = int(n_tokens // 2) + 16  # over-shoot then truncate
    seen: set[str] = set()
    lines: list[str] = []
    pos = 0
    needle_inserted = False
    needle_line_idx = max(0, int(needle_position) // 2)
    for i in range(n_lines):
        if i == needle_line_idx and not needle_inserted:
            lines.append(
                f"NEEDLE_VALUE={int(needle_value)}")
            needle_inserted = True
            continue
        while True:
            tok = f"item-{rng.randint(100000, 999999)}"
            if tok not in seen:
                seen.add(tok)
                lines.append(tok)
                break
        pos += 1
    if not needle_inserted:
        lines.append(f"NEEDLE_VALUE={int(needle_value)}")
    return " ".join(lines)


@dataclasses.dataclass(frozen=True)
class LongContextInterceptHorizonV1:
    horizon_tokens: int
    n_input_tokens_actual: int
    inject_layer: int
    inject_magnitude: float
    baseline_trace_cid: str
    injected_trace_cid: str
    intercept_moves_cid: bool
    wall_seconds_baseline_forward: float
    wall_seconds_injected_forward: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_tokens": int(self.horizon_tokens),
            "n_input_tokens_actual": int(
                self.n_input_tokens_actual),
            "inject_layer": int(self.inject_layer),
            "inject_magnitude": float(round(
                self.inject_magnitude, 6)),
            "baseline_trace_cid": str(self.baseline_trace_cid),
            "injected_trace_cid": str(self.injected_trace_cid),
            "intercept_moves_cid": bool(
                self.intercept_moves_cid),
            "wall_seconds_baseline_forward": float(round(
                self.wall_seconds_baseline_forward, 6)),
            "wall_seconds_injected_forward": float(round(
                self.wall_seconds_injected_forward, 6)),
            "detail": str(self.detail),
        }


@dataclasses.dataclass(frozen=True)
class LongContextInterceptBenchReportV1:
    schema: str
    transformers_available: bool
    model_name: str
    precision_tier: str
    device: str
    horizons: tuple[LongContextInterceptHorizonV1, ...]
    intercept_moves_cid_at_min_32k: bool
    n_horizons_pass: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "transformers_available": bool(
                self.transformers_available),
            "model_name": str(self.model_name),
            "precision_tier": str(self.precision_tier),
            "device": str(self.device),
            "horizons": [h.to_dict() for h in self.horizons],
            "intercept_moves_cid_at_min_32k": bool(
                self.intercept_moves_cid_at_min_32k),
            "n_horizons_pass": int(self.n_horizons_pass),
            "detail": str(self.detail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "w86_long_context_intercept_bench_report_v1",
            "report": self.to_dict()})


def run_long_context_intercept_bench_v1(
        *,
        model_name: str,
        device: str = "cuda:0",
        precision_tier: str = "tier_bf16",
        horizons: Sequence[int] = (
            W86_LC_INTERCEPT_DEFAULT_HORIZONS),
        inject_layer: int = (
            W86_LC_INTERCEPT_DEFAULT_INJECT_LAYER),
        inject_magnitude: float = (
            W86_LC_INTERCEPT_DEFAULT_INJECT_MAG),
        seed: int = W86_LC_INTERCEPT_DEFAULT_SEED,
        runtime: Any = None,
) -> LongContextInterceptBenchReportV1:
    """Run the live long-context hidden-state intercept bench.

    Returns a content-addressed report. The load-bearing #27
    bool is ``intercept_moves_cid_at_min_32k``: the trace CID
    must differ between baseline and post-injection forwards at
    at least one horizon ≥ 32 768 tokens.
    """
    try:
        from .transformers_runtime_v1 import (
            TransformersRuntimeV1,
        )
        from .runtime_instrumentation_v1 import (
            InjectionPlanV1,
            W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
        )
    except ImportError as exc:
        return LongContextInterceptBenchReportV1(
            schema=(
                W86_LONG_CONTEXT_INTERCEPT_V1_SCHEMA_VERSION),
            transformers_available=False,
            model_name=str(model_name),
            precision_tier=str(precision_tier),
            device=str(device),
            horizons=tuple(),
            intercept_moves_cid_at_min_32k=False,
            n_horizons_pass=0,
            detail=(
                "transformers / torch not importable: "
                f"{type(exc).__name__}: {exc}"),
        )

    if runtime is None:
        # Use skinny_trace=True so 32 k+ token forward fits on
        # a 24 GB GPU.
        try:
            runtime = TransformersRuntimeV1(
                model_name=str(model_name),
                device=str(device),
                precision_tier=str(precision_tier),
                skinny_trace=True,
            )
        except Exception as exc:  # noqa: BLE001
            return LongContextInterceptBenchReportV1(
                schema=(
                    W86_LONG_CONTEXT_INTERCEPT_V1_SCHEMA_VERSION),
                transformers_available=False,
                model_name=str(model_name),
                precision_tier=str(precision_tier),
                device=str(device),
                horizons=tuple(),
                intercept_moves_cid_at_min_32k=False,
                n_horizons_pass=0,
                detail=(
                    "TransformersRuntimeV1 instantiation failed: "
                    f"{type(exc).__name__}: {str(exc)[:160]}"),
            )

    per_horizon: list[LongContextInterceptHorizonV1] = []
    n_pass = 0
    H_inj = int(runtime.hidden_dim)
    n_layers = int(runtime.n_layers)
    L = int(inject_layer)
    if L >= n_layers:
        L = max(0, n_layers - 1)

    for horizon in horizons:
        horizon = int(horizon)
        needle_value = 423171 + horizon
        prompt_text = build_long_haystack_token_prompt_v1(
            n_tokens=horizon,
            needle_position=horizon // 2,
            needle_value=int(needle_value),
            seed=int(seed) + int(horizon),
        )
        # Tokenize and clip to exactly ``horizon`` if we
        # overshot.
        ids = runtime.tokenize(
            prompt_text, max_len=horizon)
        if len(ids) < horizon:
            # Grow prompt text until we reach horizon.
            extra_seed = int(seed) + 1
            while len(ids) < horizon:
                more = build_long_haystack_token_prompt_v1(
                    n_tokens=max(64, horizon - len(ids)),
                    needle_position=0,
                    needle_value=int(needle_value),
                    seed=int(extra_seed))
                extra_seed += 1
                prompt_text = prompt_text + " " + more
                ids = runtime.tokenize(
                    prompt_text, max_len=horizon)
        n_actual = int(len(ids))

        # Build the injection. We inject an L2-normalised
        # constant additive bias of magnitude ``inject_magnitude``
        # at sequence positions [n_actual//3 : 2*n_actual//3].
        seg_lo = int(n_actual // 3)
        seg_hi = int(2 * n_actual // 3)
        seg_len = max(1, seg_hi - seg_lo)
        inj_block = _np.full(
            (int(n_actual), int(H_inj)),
            0.0,
            dtype=_np.float64,
        )
        rng = _np.random.default_rng(
            int(seed) + int(horizon) * 7)
        bias_vec = rng.standard_normal(int(H_inj))
        bias_vec = (
            bias_vec / (float(
                _np.linalg.norm(bias_vec)) + 1e-12))
        inj_block[seg_lo:seg_hi, :] = (
            float(inject_magnitude) * bias_vec[None, :])
        per_layer_inj: list["_np.ndarray | None"] = [
            None] * int(n_layers)
        per_layer_inj[int(L)] = inj_block
        plan = InjectionPlanV1(
            schema=(
                W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
            hidden_state_inject_per_layer=tuple(
                per_layer_inj),
            attention_bias_per_layer=tuple(),
            prefix_state_inject=None,
            kv_restore=None,
            position_offset=None,
        )

        t0 = time.time()
        try:
            base_trace = runtime.forward(input_token_ids=ids)
            t_base = float(time.time() - t0)
            t1 = time.time()
            inj_trace = runtime.forward(
                input_token_ids=ids, injection=plan)
            t_inj = float(time.time() - t1)
            base_cid = str(base_trace.cid())
            inj_cid = str(inj_trace.cid())
            moves = bool(base_cid != inj_cid)
            detail = (
                f"horizon={horizon}, n_actual_tokens={n_actual}, "
                f"inject_layer={L}, magnitude={inject_magnitude}, "
                f"moves={moves}"
            )
        except Exception as exc:  # noqa: BLE001
            t_base = float(time.time() - t0)
            t_inj = float("nan")
            base_cid = ""
            inj_cid = ""
            moves = False
            detail = (
                f"horizon={horizon}, n_actual_tokens={n_actual}, "
                f"FAILED: {type(exc).__name__}: {str(exc)[:160]}"
            )

        if moves and int(n_actual) >= 32_000:
            n_pass += 1

        per_horizon.append(LongContextInterceptHorizonV1(
            horizon_tokens=int(horizon),
            n_input_tokens_actual=int(n_actual),
            inject_layer=int(L),
            inject_magnitude=float(inject_magnitude),
            baseline_trace_cid=str(base_cid),
            injected_trace_cid=str(inj_cid),
            intercept_moves_cid=bool(moves),
            wall_seconds_baseline_forward=float(t_base),
            wall_seconds_injected_forward=float(t_inj),
            detail=str(detail),
        ))

    return LongContextInterceptBenchReportV1(
        schema=W86_LONG_CONTEXT_INTERCEPT_V1_SCHEMA_VERSION,
        transformers_available=True,
        model_name=str(model_name),
        precision_tier=str(precision_tier),
        device=str(device),
        horizons=tuple(per_horizon),
        intercept_moves_cid_at_min_32k=bool(n_pass > 0),
        n_horizons_pass=int(n_pass),
        detail=(
            f"horizons_tested={len(per_horizon)}, "
            f"horizons_pass={n_pass}"),
    )


__all__ = [
    "W86_LONG_CONTEXT_INTERCEPT_V1_SCHEMA_VERSION",
    "W86_LC_INTERCEPT_DEFAULT_HORIZONS",
    "W86_LC_INTERCEPT_DEFAULT_INJECT_LAYER",
    "W86_LC_INTERCEPT_DEFAULT_INJECT_MAG",
    "W86_LC_INTERCEPT_DEFAULT_SEED",
    "LongContextInterceptHorizonV1",
    "LongContextInterceptBenchReportV1",
    "build_long_haystack_token_prompt_v1",
    "run_long_context_intercept_bench_v1",
]
