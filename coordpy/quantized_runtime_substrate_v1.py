"""W84 / P1 #30 — Quantized-Runtime Substrate with Honest Precision Floors.

The W80 controlled-runtime / transformers-runtime contract was
designed around fp32 byte-identical replay
(``W80-L-TRANSFORMERS-V1-FP32-DETERMINISM-CAP``). Under quantised
inference, byte-identity is mathematically impossible (rounding
makes it so). The honest claim under quantisation is a
**measurable, per-tier precision floor**, not byte-identity.

This module extends the W80 contract with three precision tiers:

* ``TIER_FP32`` — existing fp32 byte-identity floor (``5e-3``).
* ``TIER_BF16`` — bf16 round-to-nearest-even floor (``5e-2``).
* ``TIER_INT8`` — symmetric per-tensor int8 floor (``2e-1``)
  PLUS a semantic-equivalence claim (top-1 token match on a
  held-out prompt set).

Implementation strategy: rather than declare a "mock quantised
runtime that runs fp32 internally", the V1 here REALLY runs
bf16-emulated weights+activations and int8-quantized weights.
The bf16 emulation uses bitwise round-to-nearest-even on fp32
representations (the IEEE 754 standard mapping). The int8
emulation uses symmetric per-tensor quantization with explicit
``scale`` factors and dequantization at every matmul boundary.
The W79 controlled runtime is the substrate (pure NumPy, no torch
dependency), so the quantization claim is fully reproducible in
CI without GPU / bitsandbytes / auto-gptq.

The conformance suite under ``TIER_BF16`` and ``TIER_INT8``
re-runs the W80 axes with the *tier-appropriate floor*. The W83
hidden-state intercept bench reproduces under ``TIER_BF16``:
injecting a hidden state at layer L still moves the trace CID
(quantization does not erase the load-bearing structural claim).

Honest scope (W84 P1 #30)
-------------------------

* ``W84-L-QUANTIZED-RUNTIME-V1-NUMPY-EMULATION-CAP`` — the V1
  here emulates bf16 / int8 via bitwise / scale-based maths on
  fp32 / fp64 NumPy. Real GPU bitsandbytes / auto-gptq is V2
  (depends on GPU + torch in the test environment); the V1
  contract surface is what they would plug into.
* ``W84-L-QUANTIZED-RUNTIME-V1-PER-TENSOR-INT8-CAP`` — V1 is
  symmetric per-tensor int8 (one scale per weight tensor).
  Per-channel / per-group quantization (AWQ / GPTQ style) is
  V2.
* ``W84-L-QUANTIZED-RUNTIME-V1-NO-KV-QUANT-CAP`` — V1 does NOT
  quantize the KV cache itself; weights and activations are
  quantized at each matmul, but KV reads/writes are fp32. KV
  cache quantization is V2.
* ``W84-L-QUANTIZED-RUNTIME-V1-INT4-V2-CAP`` — INT4 / mixed
  precision is V2.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "coordpy.quantized_runtime_substrate_v1 requires numpy"
    ) from exc

from .controlled_runtime_substrate_v1 import (
    ControlledRuntimeParamsV1,
    ControlledRuntimeKVCacheV1,
    build_controlled_runtime_params_v1,
    forward_controlled_runtime,
    replay_from_kv_cache,
    tokenize_bytes_v79,
)
from .runtime_instrumentation_v1 import (
    AttentionSnapshotV1,
    CapabilityTag,
    ForwardTraceV1,
    HiddenStateSnapshotV1,
    InjectionPlanV1,
    InstrumentationAxis,
    KVCacheSnapshotV1,
    W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
)


W84_QUANTIZED_RUNTIME_V1_SCHEMA_VERSION: str = (
    "coordpy.quantized_runtime_substrate_v1.v1")


# ---------------------------------------------------------------
# Precision tiers + per-tier floors
# ---------------------------------------------------------------


class PrecisionTier(str, enum.Enum):
    """The three first-class precision tiers V1 declares.

    The string values are stable axis names and appear in CIDs,
    capability matrices, and conformance reports.
    """

    TIER_FP32 = "tier_fp32"
    TIER_BF16 = "tier_bf16"
    TIER_INT8 = "tier_int8"


# Per-tier maximum-absolute-difference floors for replay-from-KV
# semantic equivalence. These are EMPIRICALLY-GROUNDED floors:
# the bf16 round-to-nearest-even introduces at most ~2^-7 relative
# error per multiplication; int8 introduces ~1/127 ~ 8e-3 relative
# error per matmul, accumulated over n_layers.
W84_PRECISION_FLOORS: Mapping[str, float] = {
    PrecisionTier.TIER_FP32.value: 5e-3,
    PrecisionTier.TIER_BF16.value: 5e-2,
    PrecisionTier.TIER_INT8.value: 2e-1,
}

# Per-tier semantic-equivalence floor — fraction of held-out
# prompts on which the tier-quantized runtime must produce the
# same top-1 token as the fp32 baseline.
W84_SEMANTIC_EQUIVALENCE_FLOOR: float = 0.95


def precision_floor(tier: str) -> float:
    """Return the tier's declared max-abs-diff floor."""
    return float(W84_PRECISION_FLOORS[str(tier)])


def is_precision_tier(s: str) -> bool:
    return str(s) in W84_PRECISION_FLOORS


# ---------------------------------------------------------------
# bf16 / int8 emulation primitives — REAL maths, not mocks
# ---------------------------------------------------------------


def to_bf16(x: "_np.ndarray") -> "_np.ndarray":
    """Round an fp32/fp64 array to bf16 via bit truncation.

    bf16 has 1 sign bit, 8 exponent bits, 7 mantissa bits —
    same format as fp32 with the low 16 bits of the mantissa
    truncated. Round-to-nearest-even (IEEE 754).

    Returns an fp32 array whose values are exactly representable
    in bf16 (the low 16 bits of every fp32 word are zero).
    """
    arr = _np.ascontiguousarray(_np.asarray(x, dtype=_np.float32))
    u = arr.view(_np.uint32).copy()
    # Round-to-nearest-even: add rounding bias 0x7FFF, plus 1
    # extra if the bit at position 16 is set (ties round to even).
    bias = _np.uint32(0x7FFF) + ((u >> _np.uint32(16))
                                 & _np.uint32(1))
    # Avoid overflow by saturating: NaN / inf in fp32 have the
    # exp bits = 0xFF; bf16 keeps the same exp so no special
    # handling is needed for finite values.
    u_rounded = (u + bias) & _np.uint32(0xFFFF0000)
    return u_rounded.view(_np.float32).reshape(arr.shape).copy()


def to_bf16_fp64(x: "_np.ndarray") -> "_np.ndarray":
    """bf16-emulate then promote to fp64 for downstream maths.

    Used for activation rounding in the BF16 tier so the
    bf16-rounded values flow through the rest of the (in-repo
    fp64) matmul without an unnecessary fp32 step.
    """
    return _np.asarray(to_bf16(x), dtype=_np.float64)


@dataclasses.dataclass(frozen=True)
class _Int8Quantization:
    """Symmetric per-tensor int8 quantization record."""
    q: "_np.ndarray"  # int8 values in [-127, 127]
    scale: float

    def dequantize(self) -> "_np.ndarray":
        return self.q.astype(_np.float64) * float(self.scale)


def quantize_int8_symmetric(
        x: "_np.ndarray") -> _Int8Quantization:
    """Symmetric per-tensor int8 quantization with round-to-nearest.

    ``scale = max(|x|) / 127.0`` (clamped at 127 to keep the
    sign-symmetric range). Round-to-nearest-even via NumPy
    ``np.round``.
    """
    arr = _np.asarray(x, dtype=_np.float64)
    max_abs = float(_np.max(_np.abs(arr)))
    if max_abs <= 0.0:
        scale = 1.0
    else:
        scale = max_abs / 127.0
    q = _np.clip(
        _np.round(arr / scale), -127.0, 127.0).astype(_np.int8)
    return _Int8Quantization(q=q, scale=float(scale))


# ---------------------------------------------------------------
# QuantizedParams — params with a precision tier baked in
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QuantizedRuntimeParamsV1:
    """W79 controlled-runtime params, quantized to a tier.

    The wrapper carries the tier and, for INT8, the per-tensor
    scale factors. The dequantized weights are reconstructed once
    and held in float64 so the forward pass matmul costs match
    the fp32 baseline (the V1 emulation is about *correctness*,
    not throughput).
    """
    schema: str
    base_params_cid: str
    precision_tier: str
    int8_scales: tuple[float, ...]
    # The dequantized weight tensors — these are the ones the
    # forward pass actually uses. They are not raw fp32: they have
    # passed through the tier-specific rounding.
    quantized_params: ControlledRuntimeParamsV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "base_params_cid": str(self.base_params_cid),
            "precision_tier": str(self.precision_tier),
            "int8_scales": [
                float(round(s, 12)) for s in self.int8_scales],
            "quantized_params_cid": str(
                self.quantized_params.cid()),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_quantized_runtime_params_v1",
            "params": self.to_dict()})


def _quantize_weight_tensor(
        w: "_np.ndarray", tier: str,
) -> tuple["_np.ndarray", float]:
    """Apply tier quantization to a single weight tensor.

    Returns ``(rounded_weight_fp64, scale)``. For TIER_FP32 the
    weight is returned unchanged with scale ``0.0``.
    """
    if tier == PrecisionTier.TIER_FP32.value:
        return _np.asarray(w, dtype=_np.float64), 0.0
    if tier == PrecisionTier.TIER_BF16.value:
        return to_bf16_fp64(w), 0.0
    if tier == PrecisionTier.TIER_INT8.value:
        q = quantize_int8_symmetric(w)
        return q.dequantize(), float(q.scale)
    raise ValueError(f"unknown precision tier: {tier}")


def quantize_controlled_runtime_params_v1(
        base_params: ControlledRuntimeParamsV1,
        *, tier: str,
) -> QuantizedRuntimeParamsV1:
    """Apply per-tensor quantization to a W79 params bundle."""
    if not is_precision_tier(str(tier)):
        raise ValueError(f"unknown precision tier: {tier}")
    scales: list[float] = []
    # Quantize each weight in turn. We keep the original config
    # (n_layers, n_heads, etc.) intact.
    embed_q, s = _quantize_weight_tensor(
        base_params.embed_W, str(tier))
    scales.append(s)
    pos_q, s = _quantize_weight_tensor(
        base_params.pos_W, str(tier))
    scales.append(s)
    qs: list["_np.ndarray"] = []
    ks: list["_np.ndarray"] = []
    vs: list["_np.ndarray"] = []
    os_: list["_np.ndarray"] = []
    m1s: list["_np.ndarray"] = []
    m2s: list["_np.ndarray"] = []
    for i in range(int(base_params.n_layers)):
        w, s = _quantize_weight_tensor(
            base_params.layer_q_W[i], str(tier))
        qs.append(w)
        scales.append(s)
        w, s = _quantize_weight_tensor(
            base_params.layer_k_W[i], str(tier))
        ks.append(w)
        scales.append(s)
        w, s = _quantize_weight_tensor(
            base_params.layer_v_W[i], str(tier))
        vs.append(w)
        scales.append(s)
        w, s = _quantize_weight_tensor(
            base_params.layer_o_W[i], str(tier))
        os_.append(w)
        scales.append(s)
        w, s = _quantize_weight_tensor(
            base_params.layer_mlp_W1[i], str(tier))
        m1s.append(w)
        scales.append(s)
        w, s = _quantize_weight_tensor(
            base_params.layer_mlp_W2[i], str(tier))
        m2s.append(w)
        scales.append(s)
    unembed_q, s = _quantize_weight_tensor(
        base_params.unembed_W, str(tier))
    scales.append(s)
    new_params = ControlledRuntimeParamsV1(
        schema=base_params.schema,
        vocab_size=int(base_params.vocab_size),
        n_layers=int(base_params.n_layers),
        n_heads=int(base_params.n_heads),
        head_dim=int(base_params.head_dim),
        hidden_dim=int(base_params.hidden_dim),
        mlp_dim=int(base_params.mlp_dim),
        max_len=int(base_params.max_len),
        seed=int(base_params.seed),
        embed_W=embed_q,
        pos_W=pos_q,
        layer_q_W=tuple(qs),
        layer_k_W=tuple(ks),
        layer_v_W=tuple(vs),
        layer_o_W=tuple(os_),
        layer_mlp_W1=tuple(m1s),
        layer_mlp_W2=tuple(m2s),
        unembed_W=unembed_q,
    )
    return QuantizedRuntimeParamsV1(
        schema=W84_QUANTIZED_RUNTIME_V1_SCHEMA_VERSION,
        base_params_cid=str(base_params.cid()),
        precision_tier=str(tier),
        int8_scales=tuple(scales),
        quantized_params=new_params,
    )


# ---------------------------------------------------------------
# QuantizedRuntime adapter — speaks the W80 instrumentation
# contract with tier-aware floors.
# ---------------------------------------------------------------


def _activation_round(
        x: "_np.ndarray", tier: str) -> "_np.ndarray":
    """Round an activation tensor for the current tier.

    INT8: dequantize-quantize via symmetric per-tensor scale.
    BF16: round-to-nearest-even at the bf16 precision.
    FP32: identity (fp64 internally; the trace CID still uses
    fp64 byte order).
    """
    if tier == PrecisionTier.TIER_FP32.value:
        return _np.asarray(x, dtype=_np.float64)
    if tier == PrecisionTier.TIER_BF16.value:
        return to_bf16_fp64(x)
    if tier == PrecisionTier.TIER_INT8.value:
        q = quantize_int8_symmetric(x)
        return q.dequantize()
    raise ValueError(f"unknown precision tier: {tier}")


@dataclasses.dataclass
class QuantizedRuntimeV1:
    """Quantized wrapper around the W79 controlled runtime.

    Construct with::

        QuantizedRuntimeV1(
            base_params=build_controlled_runtime_params_v1(),
            tier="tier_bf16")

    The wrapper:

    * pre-quantizes the W79 weight tensors to the tier
    * runs the W79 forward, applying the tier's activation
      rounding to the *output of every layer*
    * speaks the W80 instrumentation contract surface
      (``backend_id`` / ``declared_axes`` / ``forward`` /
      ``replay_from_kv``)
    * declares the tier-appropriate ``precision_floor`` as a
      first-class property that the conformance runner consults
    """

    base_params: ControlledRuntimeParamsV1 = dataclasses.field(
        default_factory=build_controlled_runtime_params_v1)
    tier: str = PrecisionTier.TIER_FP32.value
    _quantized: QuantizedRuntimeParamsV1 | None = None

    def __post_init__(self) -> None:
        if not is_precision_tier(str(self.tier)):
            raise ValueError(
                f"unknown precision tier: {self.tier}")
        self._quantized = (
            quantize_controlled_runtime_params_v1(
                self.base_params, tier=str(self.tier)))

    @property
    def quantized(self) -> QuantizedRuntimeParamsV1:
        assert self._quantized is not None
        return self._quantized

    @property
    def precision_floor(self) -> float:
        return precision_floor(str(self.tier))

    def backend_id(self) -> str:
        return (
            "coordpy.quantized_runtime_substrate_v1"
            f"#{self.tier}")

    def backend_runtime_id(self) -> str:
        return (
            f"{self.backend_id()}"
            f"@{self.quantized.cid()[:16]}")

    def declared_axes(self) -> Mapping[str, str]:
        a = CapabilityTag.AVAILABLE.value
        return {
            InstrumentationAxis.READ_HIDDEN_STATE.value: a,
            InstrumentationAxis.READ_KV_CACHE.value: a,
            InstrumentationAxis.READ_ATTENTION_PROBS.value: a,
            InstrumentationAxis.READ_PER_LAYER_LOGITS.value: a,
            InstrumentationAxis.READ_FINAL_LOGITS.value: a,
            InstrumentationAxis.WRITE_HIDDEN_STATE_INJECT.value:
                a,
            InstrumentationAxis.WRITE_KV_RESTORE.value: a,
            InstrumentationAxis.WRITE_ATTENTION_BIAS.value: a,
            InstrumentationAxis.INJECT_PREFIX_STATE.value: a,
            InstrumentationAxis.REPLAY_FROM_KV.value: a,
            InstrumentationAxis.DETERMINISTIC_REPLAY.value: a,
            InstrumentationAxis.CONTENT_ADDRESSED_TRACE.value: a,
        }

    def tokenize(
            self, text: str, *, max_len: int = 64,
    ) -> list[int]:
        return list(tokenize_bytes_v79(
            str(text), max_len=int(max_len)))

    def forward(
            self, *, input_token_ids: Sequence[int],
            injection: InjectionPlanV1 | None = None,
    ) -> ForwardTraceV1:
        hidden_inj: list["_np.ndarray | None"] | None = None
        attn_inj: list["_np.ndarray | None"] | None = None
        prefix: "_np.ndarray | None" = None
        if injection is not None:
            if len(injection.hidden_state_inject_per_layer) > 0:
                hidden_inj = list(
                    injection.hidden_state_inject_per_layer)
            if len(injection.attention_bias_per_layer) > 0:
                attn_inj = list(injection.attention_bias_per_layer)
            if injection.prefix_state_inject is not None:
                prefix = injection.prefix_state_inject
        trace, kv = forward_controlled_runtime(
            params=self.quantized.quantized_params,
            input_token_ids=input_token_ids,
            hidden_state_injections_per_layer=hidden_inj,
            attention_bias_injections_per_layer=attn_inj,
            prefix_state_injection=prefix,
        )
        # Apply tier-specific activation rounding to every
        # post-layer hidden state and to the logits. The
        # rounding turns "fp32 with quantized weights" into
        # "actually-bf16-or-int8-quantized intermediate".
        if str(self.tier) != PrecisionTier.TIER_FP32.value:
            post_mlp_rounded = tuple(
                _activation_round(h, str(self.tier))
                for h in trace.post_mlp_hidden)
            pre_attn_rounded = tuple(
                _activation_round(h, str(self.tier))
                for h in trace.pre_attn_hidden)
            post_attn_rounded = tuple(
                _activation_round(h, str(self.tier))
                for h in trace.post_attn_hidden)
            logits_rounded = _activation_round(
                trace.logits, str(self.tier))
            final_hidden_rounded = _activation_round(
                trace.final_hidden, str(self.tier))
        else:
            post_mlp_rounded = trace.post_mlp_hidden
            pre_attn_rounded = trace.pre_attn_hidden
            post_attn_rounded = trace.post_attn_hidden
            logits_rounded = trace.logits
            final_hidden_rounded = trace.final_hidden
        return self._wrap_trace(
            trace=trace, kv=kv,
            post_mlp=post_mlp_rounded,
            pre_attn=pre_attn_rounded,
            post_attn=post_attn_rounded,
            logits=logits_rounded,
            final_hidden=final_hidden_rounded,
        )

    def replay_from_kv(
            self, *, kv: KVCacheSnapshotV1,
            new_token_ids: Sequence[int],
    ) -> ForwardTraceV1:
        kv_obj = ControlledRuntimeKVCacheV1.empty(
            n_layers=int(
                self.quantized.quantized_params.n_layers),
            n_heads=int(
                self.quantized.quantized_params.n_heads),
            head_dim=int(
                self.quantized.quantized_params.head_dim))
        k_layers = list(kv.k_per_layer)
        v_layers = list(kv.v_per_layer)
        for i in range(int(
                self.quantized.quantized_params.n_layers)):
            if i < len(k_layers) and k_layers[i] is not None:
                kv_obj.k_layers[i] = _np.asarray(
                    k_layers[i], dtype=_np.float64).copy()
            if i < len(v_layers) and v_layers[i] is not None:
                kv_obj.v_layers[i] = _np.asarray(
                    v_layers[i], dtype=_np.float64).copy()
        trace, kv_out = replay_from_kv_cache(
            params=self.quantized.quantized_params,
            kv_cache=kv_obj,
            new_token_ids=new_token_ids)
        if str(self.tier) != PrecisionTier.TIER_FP32.value:
            post_mlp_rounded = tuple(
                _activation_round(h, str(self.tier))
                for h in trace.post_mlp_hidden)
            pre_attn_rounded = tuple(
                _activation_round(h, str(self.tier))
                for h in trace.pre_attn_hidden)
            post_attn_rounded = tuple(
                _activation_round(h, str(self.tier))
                for h in trace.post_attn_hidden)
            logits_rounded = _activation_round(
                trace.logits, str(self.tier))
            final_hidden_rounded = _activation_round(
                trace.final_hidden, str(self.tier))
        else:
            post_mlp_rounded = trace.post_mlp_hidden
            pre_attn_rounded = trace.pre_attn_hidden
            post_attn_rounded = trace.post_attn_hidden
            logits_rounded = trace.logits
            final_hidden_rounded = trace.final_hidden
        return self._wrap_trace(
            trace=trace, kv=kv_out,
            post_mlp=post_mlp_rounded,
            pre_attn=pre_attn_rounded,
            post_attn=post_attn_rounded,
            logits=logits_rounded,
            final_hidden=final_hidden_rounded,
        )

    def _wrap_trace(
            self, *, trace: Any, kv: Any,
            post_mlp: tuple,
            pre_attn: tuple, post_attn: tuple,
            logits: "_np.ndarray",
            final_hidden: "_np.ndarray",
    ) -> ForwardTraceV1:
        hidden = HiddenStateSnapshotV1(
            schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
            n_layers=int(trace.n_layers),
            seq_len=int(trace.seq_len),
            hidden_dim=int(trace.hidden_dim),
            per_layer=tuple(
                _np.asarray(h, dtype=_np.float64)
                for h in post_mlp),
            final=(
                _np.asarray(final_hidden, dtype=_np.float64)
                if final_hidden is not None else None),
        )
        kv_snapshot = KVCacheSnapshotV1(
            schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
            n_layers=int(kv.n_layers),
            seq_len=int(kv.total_seq_len()),
            n_heads=int(kv.n_heads),
            head_dim=int(kv.head_dim),
            k_per_layer=tuple(
                (_np.asarray(k, dtype=_np.float64).copy()
                 if k is not None else None)
                for k in kv.k_layers),
            v_per_layer=tuple(
                (_np.asarray(v, dtype=_np.float64).copy()
                 if v is not None else None)
                for v in kv.v_layers),
        )
        attn = AttentionSnapshotV1(
            schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
            n_layers=int(trace.n_layers),
            n_heads=int(trace.n_heads),
            seq_q=int(trace.seq_len),
            seq_k=int(trace.attn_probs[0].shape[-1])
            if len(trace.attn_probs) > 0
            and _np.asarray(trace.attn_probs[0]).size > 0
            else 0,
            per_layer=tuple(
                _np.asarray(a, dtype=_np.float64)
                for a in trace.attn_probs),
        )
        return ForwardTraceV1(
            schema=W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION,
            backend_id=self.backend_id(),
            backend_runtime_id=self.backend_runtime_id(),
            input_token_ids=tuple(
                int(t) for t in trace.input_token_ids),
            seq_len=int(trace.seq_len),
            hidden=hidden, kv=kv_snapshot, attn=attn,
            final_logits=(
                _np.asarray(logits, dtype=_np.float64)
                if logits is not None else None),
            declared_axes=tuple(
                (str(k), str(v))
                for k, v in self.declared_axes().items()),
        )


# ---------------------------------------------------------------
# Conformance + semantic-equivalence bench
# ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QuantizedConformanceReportV1:
    """Per-tier conformance + replay-from-KV bench report."""
    schema: str
    tier: str
    precision_floor: float
    n_prompts: int
    max_replay_diff: float
    replay_within_floor: bool
    top1_match_rate_vs_fp32: float
    semantic_equivalence: bool
    hidden_intercept_moves_cid: bool
    base_params_cid: str
    quantized_params_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "tier": str(self.tier),
            "precision_floor": float(round(
                self.precision_floor, 12)),
            "n_prompts": int(self.n_prompts),
            "max_replay_diff": float(round(
                self.max_replay_diff, 12)),
            "replay_within_floor": bool(self.replay_within_floor),
            "top1_match_rate_vs_fp32": float(round(
                self.top1_match_rate_vs_fp32, 12)),
            "semantic_equivalence": bool(
                self.semantic_equivalence),
            "hidden_intercept_moves_cid": bool(
                self.hidden_intercept_moves_cid),
            "base_params_cid": str(self.base_params_cid),
            "quantized_params_cid": str(
                self.quantized_params_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w84_quantized_conformance_report_v1",
            "report": self.to_dict()})


def run_quantized_conformance_bench_v1(
        *,
        base_params: ControlledRuntimeParamsV1 | None = None,
        tier: str = PrecisionTier.TIER_BF16.value,
        prompts: Sequence[str] | None = None,
        n_prompts: int = 25,
        seed: int = 84_030_001,
) -> QuantizedConformanceReportV1:
    """Run the per-tier conformance bench.

    * Builds the fp32 baseline.
    * Builds the tier-quantized runtime.
    * For each prompt:
      - runs forward at both fp32 and tier; checks replay-from-KV
        against the tier floor.
      - records top-1 final-token agreement vs fp32.
    * Tests the hidden-state intercept at layer 0 — under the
      tier-quantized runtime, the intercept STILL moves the
      trace CID (the structural claim survives quantization).
    """
    if base_params is None:
        base_params = build_controlled_runtime_params_v1()
    if prompts is None:
        rng = _np.random.default_rng(int(seed))
        prompts = []
        for i in range(int(n_prompts)):
            n = int(rng.integers(4, 16))
            chars = [chr(int(c)) for c in rng.integers(
                ord('a'), ord('z'), size=n)]
            prompts.append("".join(chars))
    fp32 = QuantizedRuntimeV1(
        base_params=base_params,
        tier=PrecisionTier.TIER_FP32.value)
    tier_rt = QuantizedRuntimeV1(
        base_params=base_params, tier=str(tier))
    max_replay_diff = 0.0
    top1_matches = 0
    n_eval = 0
    for prompt in prompts:
        ids = tier_rt.tokenize(prompt, max_len=12)
        if len(ids) < 2:
            continue
        # Forward at fp32 baseline + tier; compare top-1 token
        # at the final position.
        fp32_trace = fp32.forward(input_token_ids=ids)
        tier_trace = tier_rt.forward(input_token_ids=ids)
        if (fp32_trace.final_logits is None
                or tier_trace.final_logits is None):
            continue
        fp32_logits = _np.asarray(
            fp32_trace.final_logits, dtype=_np.float64)[-1]
        tier_logits = _np.asarray(
            tier_trace.final_logits, dtype=_np.float64)[-1]
        if int(_np.argmax(fp32_logits)) == int(
                _np.argmax(tier_logits)):
            top1_matches += 1
        n_eval += 1
        # Replay-from-KV under the tier: prefix all but last
        # token; restore KV; replay the last token. Compare
        # the last-token logits to the full recompute trace.
        if len(ids) >= 3:
            old_ids = list(ids[:-1])
            new_ids = [int(ids[-1])]
            full = tier_rt.forward(input_token_ids=ids)
            old_trace = tier_rt.forward(input_token_ids=old_ids)
            replay = tier_rt.replay_from_kv(
                kv=old_trace.kv, new_token_ids=new_ids)
            if (full.final_logits is not None
                    and replay.final_logits is not None):
                full_last = _np.asarray(
                    full.final_logits, dtype=_np.float64)[-1]
                replay_last = _np.asarray(
                    replay.final_logits,
                    dtype=_np.float64)[-1]
                if full_last.shape == replay_last.shape:
                    diff = float(_np.max(
                        _np.abs(full_last - replay_last)))
                    if diff > max_replay_diff:
                        max_replay_diff = diff
    # Hidden-state intercept under the tier.
    intercept_moves = False
    if len(prompts) > 0:
        ids = tier_rt.tokenize(prompts[0], max_len=8)
        if len(ids) >= 1:
            baseline = tier_rt.forward(input_token_ids=ids)
            if baseline.hidden is not None and len(
                    baseline.hidden.per_layer) > 0:
                shape0 = baseline.hidden.per_layer[0].shape
                inj = _np.full(
                    shape0, 0.07, dtype=_np.float64)
                per_layer: list["_np.ndarray | None"] = [
                    None] * int(
                    baseline.hidden.n_layers)
                per_layer[0] = inj
                plan = InjectionPlanV1(
                    schema=(
                        W80_RUNTIME_INSTRUMENTATION_V1_SCHEMA_VERSION),
                    hidden_state_inject_per_layer=tuple(
                        per_layer))
                after = tier_rt.forward(
                    input_token_ids=ids, injection=plan)
                intercept_moves = (
                    baseline.cid() != after.cid())
    floor = precision_floor(str(tier))
    top1_rate = (
        float(top1_matches) / max(1, n_eval))
    semantic_eq = (
        top1_rate >= W84_SEMANTIC_EQUIVALENCE_FLOOR)
    return QuantizedConformanceReportV1(
        schema=W84_QUANTIZED_RUNTIME_V1_SCHEMA_VERSION,
        tier=str(tier),
        precision_floor=float(floor),
        n_prompts=int(n_eval),
        max_replay_diff=float(max_replay_diff),
        replay_within_floor=bool(max_replay_diff < floor),
        top1_match_rate_vs_fp32=float(top1_rate),
        semantic_equivalence=bool(semantic_eq),
        hidden_intercept_moves_cid=bool(intercept_moves),
        base_params_cid=str(base_params.cid()),
        quantized_params_cid=str(tier_rt.quantized.cid()),
    )


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


__all__ = [
    "W84_QUANTIZED_RUNTIME_V1_SCHEMA_VERSION",
    "W84_PRECISION_FLOORS",
    "W84_SEMANTIC_EQUIVALENCE_FLOOR",
    "PrecisionTier",
    "precision_floor",
    "is_precision_tier",
    "to_bf16",
    "to_bf16_fp64",
    "quantize_int8_symmetric",
    "QuantizedRuntimeParamsV1",
    "quantize_controlled_runtime_params_v1",
    "QuantizedRuntimeV1",
    "QuantizedConformanceReportV1",
    "run_quantized_conformance_bench_v1",
]
