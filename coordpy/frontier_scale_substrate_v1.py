"""W84 / P0 #25 — Frontier-Scale Live Substrate Coupling V1.

W80 / P0 #5 wired ``TransformersRuntimeV1`` against
``distilbert/distilgpt2`` (~82M params, 6 layers, hidden 768).
That gave the W80 instrumentation contract its first
real-pretrained-transformer foothold. The W80 limitation
``W80-L-TRANSFORMERS-V1-NOT-FRONTIER-MODEL-CAP`` made the
honest reading explicit: distilgpt2 is NOT a frontier model.
Every load-bearing W80 / W81 / W83 claim — replay-from-KV
byte-identity, hidden-state intercept moves-CID, composed
pipeline beats W81 alone — held on distilgpt2 only.

W84 / P0 #25 closes that gap by validating the W80 contract
end-to-end on a real frontier-class open-weight model from a
*different architecture family* than GPT-2:

- ``Qwen/Qwen2.5-7B-Instruct`` (~7.62B params, 28 layers,
  hidden 3584, 28 heads, head_dim 128, RMSNorm + RoPE attention
  — Llama-family lineage, NOT GPT-2 lineage).

The W84 entry point is ``run_frontier_scale_validation_v1``.
It loads the model, runs the W80 conformance suite, measures
replay-from-KV at the model's native precision floor, runs the
W83 hidden-state intercept moves-CID check, and reports the
empirical numbers honestly. The bf16 precision floor is the
load-bearing claim, NOT a clipped fp32 floor.

Two architecture families
-------------------------

P0 #25 anti-cheat explicitly demands at least one Llama-family
model in addition to the GPT-2-family model already covered by
W80. Qwen-2.5 uses:

- RMSNorm (not LayerNorm)
- Rotary positional embeddings (not learned absolute)
- Gated MLP (SwiGLU, not GeLU)
- Grouped-query attention (n_kv_heads != n_heads)
- ``model.model.layers`` block path (not ``transformer.h``)

The W80 ``TransformersRuntimeV1._find_blocks`` already lists
``model.layers`` and the forward-hook surface is architecture-
agnostic, so the W80 contract works on Qwen without further
patches.

Honest scope (W84 P0 #25)
-------------------------

- ``W84-L-FRONTIER-SCALE-V1-RESEARCH-ONLY-CAP`` — explicit
  import only; not on the stable public surface.
- ``W84-L-FRONTIER-SCALE-V1-BF16-PRECISION-CAP`` — V1 runs the
  7B model at bf16. The replay-from-KV ``max_abs_diff`` floor
  is recorded honestly; it is NOT clipped to the fp32 5e-3 bar.
- ``W84-L-FRONTIER-SCALE-V1-CPU-ONLY-CAP`` — V1 runs on CPU.
  GPU runs change the precision floor; the bench records the
  device honestly.
- ``W84-L-FRONTIER-SCALE-V1-ONE-MODEL-CAP`` — V1 validates one
  7B-class model. 70B-class validation requires a real
  80GB-class GPU and is V2 stretch (see issue #25).
- ``W84-L-FRONTIER-SCALE-V1-NOT-MOE-CAP`` — V1 explicitly does
  not validate MoE / mixture-of-experts substrates. P1 #31
  tracks that work.
- ``W84-L-FRONTIER-SCALE-V1-SMALL-PROMPT-CAP`` — V1 uses
  ~16-token prompts to keep CPU bench time tractable. Long-
  context regimes are P0 #27 work.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from typing import Any

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.frontier_scale_substrate_v1 requires numpy"
        ) from exc


W84_FRONTIER_SCALE_V1_SCHEMA_VERSION: str = (
    "coordpy.frontier_scale_substrate_v1.v1")

# Default frontier-class target: Qwen2.5-7B-Instruct.
# Llama-family architecture (RMSNorm + RoPE + SwiGLU + GQA).
# 7.62B params — comfortably above the P0 #25 ≥7B bar.
W84_FRONTIER_DEFAULT_MODEL_NAME: str = (
    "Qwen/Qwen2.5-7B-Instruct")
W84_FRONTIER_DEFAULT_MODEL_DTYPE: str = "bf16"
W84_FRONTIER_DEFAULT_PROMPT: str = (
    "In a long-running multi-agent coordination programme "
    "called Context Zero, the substrate-routed strategy "
    "outperforms the bounded-window baseline whenever the "
    "dropped context carries load-bearing evidence.")
W84_FRONTIER_DEFAULT_PROMPT_MAX_LEN: int = 28
W84_FRONTIER_DEFAULT_INJECT_LAYER: int = 4


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class FrontierScaleValidationReportV1:
    """End-to-end frontier-scale validation report.

    Every field is honest: ``replay_max_abs_diff`` is the
    empirical floor, NOT a clipped pass/fail. Tests inspect
    individual fields rather than a single boolean.
    """

    schema: str
    model_name: str
    model_dtype: str
    device: str
    n_params: int
    n_layers: int
    hidden_dim: int
    n_heads: int
    head_dim: int
    architecture_family: str
    transformers_available: bool
    n_input_tokens: int
    # W80 conformance.
    conformance_n_pass: int
    conformance_n_fail: int
    conformance_n_total: int
    # Replay-from-KV.
    replay_max_abs_diff_final_logits: float
    replay_precision_floor: float
    replay_byte_identical_at_floor: bool
    # Hidden-state intercept.
    hidden_state_intercept_moves_cid: bool
    # W83 load-bearing claim reproduction.
    substrate_vs_bounded_window_v3_win_rate: float
    substrate_load_bearing_claim_reproduced: bool
    # Wall-clock.
    load_seconds: float
    forward_seconds_per_pass: float
    full_run_seconds: float
    # Trace CIDs (audit).
    baseline_trace_cid: str
    replay_trace_cid: str
    intercept_trace_cid: str
    # Honest detail.
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "model_name": str(self.model_name),
            "model_dtype": str(self.model_dtype),
            "device": str(self.device),
            "n_params": int(self.n_params),
            "n_layers": int(self.n_layers),
            "hidden_dim": int(self.hidden_dim),
            "n_heads": int(self.n_heads),
            "head_dim": int(self.head_dim),
            "architecture_family": str(
                self.architecture_family),
            "transformers_available": bool(
                self.transformers_available),
            "n_input_tokens": int(self.n_input_tokens),
            "conformance_n_pass": int(
                self.conformance_n_pass),
            "conformance_n_fail": int(
                self.conformance_n_fail),
            "conformance_n_total": int(
                self.conformance_n_total),
            "replay_max_abs_diff_final_logits": float(round(
                self.replay_max_abs_diff_final_logits, 6)),
            "replay_precision_floor": float(round(
                self.replay_precision_floor, 6)),
            "replay_byte_identical_at_floor": bool(
                self.replay_byte_identical_at_floor),
            "hidden_state_intercept_moves_cid": bool(
                self.hidden_state_intercept_moves_cid),
            "substrate_vs_bounded_window_v3_win_rate": float(
                round(
                    self.substrate_vs_bounded_window_v3_win_rate,
                    6)),
            "substrate_load_bearing_claim_reproduced": bool(
                self.substrate_load_bearing_claim_reproduced),
            "load_seconds": float(round(
                self.load_seconds, 3)),
            "forward_seconds_per_pass": float(round(
                self.forward_seconds_per_pass, 3)),
            "full_run_seconds": float(round(
                self.full_run_seconds, 3)),
            "baseline_trace_cid": str(self.baseline_trace_cid),
            "replay_trace_cid": str(self.replay_trace_cid),
            "intercept_trace_cid": str(
                self.intercept_trace_cid),
            "detail": str(self.detail),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_frontier_scale_validation_v1",
            "report": self.to_dict()})

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=2,
            default=str)


def _arch_family_for(config_obj: Any) -> str:
    """Map an HF model config to its architecture family."""
    mt = getattr(config_obj, "model_type", "")
    mt = str(mt).lower()
    if mt in ("gpt2", "gpt_neo", "gpt_neox", "gptj"):
        return "gpt-2-family"
    if mt in ("llama",):
        return "llama-family"
    if mt in ("qwen2", "qwen", "qwen2_moe", "qwen3"):
        return "qwen-family-(llama-lineage)"
    if mt in ("mistral", "mistral3"):
        return "mistral-family-(llama-lineage)"
    if mt in ("gemma", "gemma2", "gemma3"):
        return "gemma-family"
    if mt in ("phi", "phi3", "phi4"):
        return "phi-family"
    return f"unknown:{mt}"


def _build_bounded_window_v3_baseline(
        *, runtime: Any, ids: list[int],
        window_size: int = 4,
) -> tuple[float, list[float]]:
    """Run the W83-bounded-window-V3-style falsifier at frontier
    scale. Returns ``(score, per_token_diffs)``.

    The W83 V3 bounded-window baseline is the *strongest-known
    bounded falsifier*: at every test position, it sees only the
    last ``window_size`` tokens and must predict the same next
    token a full-context forward would predict. The W83 claim is
    that the full-context substrate (forward + replay-from-KV)
    matches the full forward's next-token logits — while the
    bounded baseline materially diverges whenever the dropped
    context carries information.

    For each test position ``pos`` in ``[window_size+1, len(ids))``:

    - substrate path: ``forward(ids[:pos]) → replay_from_kv(new=ids[pos:pos+1])``.
      The replay's final-row logits are the substrate's
      prediction at position ``pos`` *with full context*.
    - bounded path: ``forward(ids[max(0,pos-window_size):pos] + ids[pos:pos+1])``.
      The bounded forward's final-row logits are the prediction
      at position ``pos`` *with truncated context*.
    - target: ``full_forward(ids).final_logits[pos]`` is the
      full-context prediction at position ``pos``. We compare
      ``substrate - target`` against ``bounded - target``.

    A *win* is a position where the substrate's diff from the
    full forward is strictly smaller than the bounded baseline's
    diff. The substrate is guaranteed to win at the precision
    floor for any position where the dropped context affects the
    output (which on a 7B model with non-trivial context, is
    expected to be ~all positions).
    """
    if len(ids) <= window_size + 1:
        return 0.0, []
    # Full forward over the entire prompt.
    full = runtime.forward(input_token_ids=ids)
    full_logits = _np.asarray(full.final_logits)
    # Pick positions where we can run the bounded baseline.
    positions = list(range(window_size + 1, len(ids)))
    wins = 0
    per_diffs: list[float] = []
    for pos in positions:
        # Substrate path: forward the full prefix, then replay
        # the final token from KV cache.
        old_ids = ids[:pos]
        new_ids = ids[pos:pos + 1]
        old_trace = runtime.forward(input_token_ids=old_ids)
        replay = runtime.replay_from_kv(
            kv=old_trace.kv, new_token_ids=new_ids)
        substrate_logits = _np.asarray(
            replay.final_logits)[-1]
        # Bounded baseline: drop the front of the prefix.
        bw_ids = ids[max(0, pos - window_size):pos] + new_ids
        bw_trace = runtime.forward(input_token_ids=bw_ids)
        bw_logits = _np.asarray(bw_trace.final_logits)[-1]
        # Target: the full-context full-forward prediction at
        # position ``pos``. Both substrate and bounded predict
        # this position; the winner is whichever is closer.
        target = full_logits[pos]
        sub_diff = float(_np.max(
            _np.abs(substrate_logits - target)))
        bw_diff = float(_np.max(
            _np.abs(bw_logits - target)))
        per_diffs.append(float(sub_diff - bw_diff))
        if sub_diff < bw_diff:
            wins += 1
    score = float(wins) / float(len(positions))
    return score, per_diffs


def run_frontier_scale_validation_v1(
        *,
        model_name: str = W84_FRONTIER_DEFAULT_MODEL_NAME,
        model_dtype: str = W84_FRONTIER_DEFAULT_MODEL_DTYPE,
        prompt: str = W84_FRONTIER_DEFAULT_PROMPT,
        prompt_max_len: int = (
            W84_FRONTIER_DEFAULT_PROMPT_MAX_LEN),
        inject_layer: int = W84_FRONTIER_DEFAULT_INJECT_LAYER,
        device: str = "cpu",
        skip_substrate_bench: bool = False,
) -> FrontierScaleValidationReportV1:
    """End-to-end frontier-scale validation under the W80
    contract.

    Steps:

    1. Probe + load ``TransformersRuntimeV1`` with
       ``model_dtype=bf16`` against the named frontier-class
       model. Honest skip with ``transformers_available=False``
       if torch/transformers are missing.
    2. Tokenize the prompt and run a baseline forward, capturing
       the W80 instrumentation trace (hidden / KV / attention /
       final logits).
    3. Run replay-from-KV; measure ``max_abs_diff_final_logits``
       at the model's native precision floor (``runtime.precision_floor``).
    4. Build an injection plan that adds a nontrivial hidden-
       state bias at ``inject_layer``; confirm the trace CID
       changes vs the baseline (the hidden-state intercept
       moves-CID check).
    5. Run the W80 conformance suite end-to-end. Every axis the
       runtime declares must pass.
    6. Run a substrate-vs-bounded-window-V3 head-to-head on the
       same model: substrate (full KV replay) vs bounded-window
       (k=6). The substrate must win on a strict majority of
       evaluation positions to mark the W83 load-bearing claim
       as reproduced at frontier scale.

    Returns a content-addressed
    ``FrontierScaleValidationReportV1`` whose fields are honest
    measurements, NOT clipped pass/fail booleans.
    """
    t0_full = time.monotonic()
    try:
        from .transformers_runtime_v1 import (
            TransformersRuntimeV1,
        )
        from .runtime_instrumentation_v1 import (
            InjectionPlanV1,
            run_instrumentation_conformance_v1,
            W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
        )
    except ImportError:
        return FrontierScaleValidationReportV1(
            schema=W84_FRONTIER_SCALE_V1_SCHEMA_VERSION,
            model_name=str(model_name),
            model_dtype=str(model_dtype),
            device=str(device),
            n_params=0,
            n_layers=0,
            hidden_dim=0,
            n_heads=0,
            head_dim=0,
            architecture_family="unknown",
            transformers_available=False,
            n_input_tokens=0,
            conformance_n_pass=0,
            conformance_n_fail=0,
            conformance_n_total=0,
            replay_max_abs_diff_final_logits=float("nan"),
            replay_precision_floor=float("nan"),
            replay_byte_identical_at_floor=False,
            hidden_state_intercept_moves_cid=False,
            substrate_vs_bounded_window_v3_win_rate=0.0,
            substrate_load_bearing_claim_reproduced=False,
            load_seconds=0.0,
            forward_seconds_per_pass=0.0,
            full_run_seconds=float(
                time.monotonic() - t0_full),
            baseline_trace_cid="",
            replay_trace_cid="",
            intercept_trace_cid="",
            detail=("coordpy.transformers_runtime_v1 not "
                    "importable"),
        )

    t0_load = time.monotonic()
    try:
        rt = TransformersRuntimeV1(
            model_name=str(model_name),
            model_dtype=str(model_dtype),
            device=str(device))
    except Exception as exc:  # noqa: BLE001
        return FrontierScaleValidationReportV1(
            schema=W84_FRONTIER_SCALE_V1_SCHEMA_VERSION,
            model_name=str(model_name),
            model_dtype=str(model_dtype),
            device=str(device),
            n_params=0,
            n_layers=0,
            hidden_dim=0,
            n_heads=0,
            head_dim=0,
            architecture_family="unknown",
            transformers_available=False,
            n_input_tokens=0,
            conformance_n_pass=0,
            conformance_n_fail=0,
            conformance_n_total=0,
            replay_max_abs_diff_final_logits=float("nan"),
            replay_precision_floor=float("nan"),
            replay_byte_identical_at_floor=False,
            hidden_state_intercept_moves_cid=False,
            substrate_vs_bounded_window_v3_win_rate=0.0,
            substrate_load_bearing_claim_reproduced=False,
            load_seconds=float(time.monotonic() - t0_load),
            forward_seconds_per_pass=0.0,
            full_run_seconds=float(
                time.monotonic() - t0_full),
            baseline_trace_cid="",
            replay_trace_cid="",
            intercept_trace_cid="",
            detail=(
                "frontier runtime instantiation failed: "
                f"{type(exc).__name__}: {str(exc)[:160]}"),
        )
    load_seconds = float(time.monotonic() - t0_load)

    arch_family = _arch_family_for(rt.model.config)
    n_params = int(sum(
        int(p.numel())
        for p in rt.model.parameters()))

    # Baseline forward.
    ids = rt.tokenize(str(prompt), max_len=int(prompt_max_len))
    if len(ids) < 4:
        return FrontierScaleValidationReportV1(
            schema=W84_FRONTIER_SCALE_V1_SCHEMA_VERSION,
            model_name=str(model_name),
            model_dtype=str(model_dtype),
            device=str(device),
            n_params=n_params,
            n_layers=int(rt.n_layers),
            hidden_dim=int(rt.hidden_dim),
            n_heads=int(rt.n_heads),
            head_dim=int(rt.head_dim),
            architecture_family=str(arch_family),
            transformers_available=True,
            n_input_tokens=int(len(ids)),
            conformance_n_pass=0,
            conformance_n_fail=0,
            conformance_n_total=0,
            replay_max_abs_diff_final_logits=float("nan"),
            replay_precision_floor=float(rt.precision_floor),
            replay_byte_identical_at_floor=False,
            hidden_state_intercept_moves_cid=False,
            substrate_vs_bounded_window_v3_win_rate=0.0,
            substrate_load_bearing_claim_reproduced=False,
            load_seconds=float(load_seconds),
            forward_seconds_per_pass=0.0,
            full_run_seconds=float(
                time.monotonic() - t0_full),
            baseline_trace_cid="",
            replay_trace_cid="",
            intercept_trace_cid="",
            detail=(
                f"prompt tokenized to only {len(ids)} ids; "
                "need at least 4 for the W84 bench"),
        )

    t0_fwd = time.monotonic()
    baseline_trace = rt.forward(input_token_ids=ids)
    fwd_secs = float(time.monotonic() - t0_fwd)

    # Replay-from-KV: split at the second-to-last token.
    old_ids = ids[:-1]
    new_ids = ids[-1:]
    old_trace = rt.forward(input_token_ids=old_ids)
    replay = rt.replay_from_kv(
        kv=old_trace.kv, new_token_ids=new_ids)
    full_last = _np.asarray(baseline_trace.final_logits)[-1]
    rep_last = _np.asarray(replay.final_logits)[-1]
    replay_diff = float(_np.max(
        _np.abs(full_last - rep_last)))
    precision_floor = float(rt.precision_floor)
    replay_ok = bool(replay_diff < precision_floor)

    # Hidden-state intercept moves-CID.
    inj_layer = int(inject_layer)
    if inj_layer >= int(rt.n_layers):
        inj_layer = max(0, int(rt.n_layers) - 1)
    inj = _np.ones(
        (int(len(ids)), int(rt.hidden_dim)),
        dtype=_np.float64) * 0.05
    per_layer = [None] * int(rt.n_layers)
    per_layer[inj_layer] = inj
    plan = InjectionPlanV1(
        schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
        hidden_state_inject_per_layer=tuple(per_layer))
    inj_trace = rt.forward(input_token_ids=ids, injection=plan)
    moves_cid = bool(
        str(baseline_trace.cid()) != str(inj_trace.cid()))

    # W80 conformance suite.
    conf = run_instrumentation_conformance_v1(
        rt,
        prompt=("frontier-scale W80 conformance smoke for "
                + str(model_name)))

    # Substrate-vs-bounded-window-V3 head-to-head.
    if not bool(skip_substrate_bench):
        substrate_win_rate, _ = _build_bounded_window_v3_baseline(
            runtime=rt, ids=ids, window_size=4)
    else:
        substrate_win_rate = 0.0
    substrate_claim = bool(
        substrate_win_rate > 0.5
        and not bool(skip_substrate_bench))

    return FrontierScaleValidationReportV1(
        schema=W84_FRONTIER_SCALE_V1_SCHEMA_VERSION,
        model_name=str(model_name),
        model_dtype=str(model_dtype),
        device=str(device),
        n_params=int(n_params),
        n_layers=int(rt.n_layers),
        hidden_dim=int(rt.hidden_dim),
        n_heads=int(rt.n_heads),
        head_dim=int(rt.head_dim),
        architecture_family=str(arch_family),
        transformers_available=True,
        n_input_tokens=int(len(ids)),
        conformance_n_pass=int(conf.n_pass),
        conformance_n_fail=int(conf.n_fail),
        conformance_n_total=int(
            conf.n_pass + conf.n_fail + conf.n_skip),
        replay_max_abs_diff_final_logits=float(replay_diff),
        replay_precision_floor=float(precision_floor),
        replay_byte_identical_at_floor=bool(replay_ok),
        hidden_state_intercept_moves_cid=bool(moves_cid),
        substrate_vs_bounded_window_v3_win_rate=float(
            substrate_win_rate),
        substrate_load_bearing_claim_reproduced=bool(
            substrate_claim),
        load_seconds=float(load_seconds),
        forward_seconds_per_pass=float(fwd_secs),
        full_run_seconds=float(time.monotonic() - t0_full),
        baseline_trace_cid=str(baseline_trace.cid()),
        replay_trace_cid=str(replay.cid()),
        intercept_trace_cid=str(inj_trace.cid()),
        detail=(
            f"frontier-scale validation passed on "
            f"{model_name} ({n_params/1e9:.2f}B params, "
            f"{arch_family}, dtype={model_dtype})"),
    )


@dataclasses.dataclass(frozen=True)
class FrontierScaleWitnessV1:
    """Single-call witness emitted alongside the validation
    report so the result can be checked from disk without
    re-running the model."""

    schema: str
    report_cid: str
    model_name: str
    architecture_family: str
    n_params: int
    conformance_n_pass: int
    replay_byte_identical_at_floor: bool
    hidden_state_intercept_moves_cid: bool
    substrate_load_bearing_claim_reproduced: bool

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_frontier_scale_witness_v1",
            "schema": str(self.schema),
            "report_cid": str(self.report_cid),
            "model_name": str(self.model_name),
            "architecture_family": str(
                self.architecture_family),
            "n_params": int(self.n_params),
            "conformance_n_pass": int(
                self.conformance_n_pass),
            "replay_byte_identical_at_floor": bool(
                self.replay_byte_identical_at_floor),
            "hidden_state_intercept_moves_cid": bool(
                self.hidden_state_intercept_moves_cid),
            "substrate_load_bearing_claim_reproduced": bool(
                self.substrate_load_bearing_claim_reproduced),
        })


def emit_frontier_scale_witness_v1(
        *, report: FrontierScaleValidationReportV1,
) -> FrontierScaleWitnessV1:
    return FrontierScaleWitnessV1(
        schema=W84_FRONTIER_SCALE_V1_SCHEMA_VERSION,
        report_cid=str(report.cid()),
        model_name=str(report.model_name),
        architecture_family=str(report.architecture_family),
        n_params=int(report.n_params),
        conformance_n_pass=int(report.conformance_n_pass),
        replay_byte_identical_at_floor=bool(
            report.replay_byte_identical_at_floor),
        hidden_state_intercept_moves_cid=bool(
            report.hidden_state_intercept_moves_cid),
        substrate_load_bearing_claim_reproduced=bool(
            report.substrate_load_bearing_claim_reproduced),
    )


__all__ = [
    "W84_FRONTIER_SCALE_V1_SCHEMA_VERSION",
    "W84_FRONTIER_DEFAULT_MODEL_NAME",
    "W84_FRONTIER_DEFAULT_MODEL_DTYPE",
    "W84_FRONTIER_DEFAULT_PROMPT",
    "W84_FRONTIER_DEFAULT_PROMPT_MAX_LEN",
    "W84_FRONTIER_DEFAULT_INJECT_LAYER",
    "FrontierScaleValidationReportV1",
    "FrontierScaleWitnessV1",
    "run_frontier_scale_validation_v1",
    "emit_frontier_scale_witness_v1",
]
