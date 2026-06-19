"""W63 M1 — Tiny Transformer Runtime V8.

Strictly extends W62's ``coordpy.tiny_substrate_v7``. V8 keeps every
V7 invariant (byte-determinism under same params, GQA, RMSNorm /
SwiGLU, content-addressable cache_keys axis, hidden_write_trace
channel, replay_age channel, forward_count integer, cross-layer
coupling, cache-write ledger, logit-lens probe, attention-receive
delta, replay-trust ledger) and adds **five** new substrate-load-
bearing axes that W63's trainable controllers and bridges exploit:

* **Default 10 layers** (vs V7's 9). Same GQA (8 query / 4 KV).
* **Per-(layer, head, slot) hidden-vs-KV contention tensor** —
  ``TinyV8KVCache.hidden_vs_kv_contention`` of shape ``(L, H, T)``
  records per-slot ``|hidden_write| - |kv_write|`` (signed). The
  W63 hidden-vs-KV head reads this when scoring which arm should
  win for a given (layer, head, slot).
* **Per-layer hidden-state confidence probe** —
  ``forward_tiny_substrate_v8`` emits
  ``hidden_state_confidence_per_layer`` of shape ``(L,)`` where
  each layer's hidden state's softmax entropy over the logit lens
  is mapped to a confidence in [0, 1] (high entropy → low
  confidence). The replay controller V4 reads this as a regime
  feature.
* **Per-(layer, head, position) replay-determinism channel** —
  ``TinyV8KVCache.replay_determinism_channel`` of shape
  ``(L, H, T)`` records 1.0 where the V7 cache_write_ledger value
  has been stable across the last K forwards and 0.0 otherwise.
* **Per-(layer, head) prefix-state reuse trust ledger** —
  ``TinyV8KVCache.prefix_reuse_trust`` of shape ``(L, H)``
  tracks an EMA over prefix bridge V7 reuse decisions
  (+1 for REUSE_SUCCESS, -1 for REUSE_DRIFT, 0 for RECOMPUTE).
* **Cross-layer-head coupling matrix** —
  ``TinyV8ForwardTrace.cross_layer_head_coupling`` of shape
  ``(L, H, L, H)``. Diagnoses which (layer, head) pairs are
  cross-correlated across the V8 forward.

V8 still preserves the V7 multi-segment partial-prefix reuse and
extends the write log with a ``schema=v8`` tag plus the new axis
deltas.

Honest scope (do-not-overstate, W63)
------------------------------------

* Still NOT a frontier model. Default config:
  ``10 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``. V8 is
  richer than V7 but still a research substrate in pure NumPy on
  CPU. ``W63-L-NUMPY-CPU-V8-SUBSTRATE-CAP`` documents.
* V8 still does NOT bridge to third-party hosted models.
  ``W63-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The hidden-vs-KV contention channel is a *diagnostic*, not a
  detector. Bridges WRITE; controllers READ.
* The hidden-state confidence probe is a *deterministic* mapping
  from logit-lens entropy. It is a diagnostic, not a calibrated
  probability distribution.
* The prefix-reuse trust ledger updates by an external setter;
  the substrate does NOT itself emit prefix decisions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence

try:
    import numpy as _np  # type: ignore
except ImportError as exc:  # pragma: no cover - numpy required.
    raise ImportError(
        "coordpy.tiny_substrate_v8 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
)
from .tiny_substrate_v6 import (
    W61_DEFAULT_V6_D_KEY,
)
from .tiny_substrate_v7 import (
    TinyV7ForwardTrace, TinyV7KVCache,
    TinyV7SubstrateConfig, TinyV7SubstrateParams,
    W62_DEFAULT_V7_D_KEY,
    W62_DEFAULT_V7_TRUST_EMA,
    forward_tiny_substrate_v7,
    logit_lens_entropy_per_layer,
)


W63_TINY_SUBSTRATE_V8_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v8.v1")

W63_TINY_V8_VOCAB_SIZE: int = 259
W63_DEFAULT_V8_D_MODEL: int = 64
W63_DEFAULT_V8_N_HEADS: int = 8
W63_DEFAULT_V8_N_KV_HEADS: int = 4
W63_DEFAULT_V8_N_LAYERS: int = 10
W63_DEFAULT_V8_FF_HIDDEN: int = 192
W63_DEFAULT_V8_MAX_LEN: int = 128
W63_DEFAULT_V8_INIT_SCALE: float = 0.04
W63_DEFAULT_V8_SEED: int = 63012345
W63_DEFAULT_V8_ROPE_BASE: float = 10000.0
W63_DEFAULT_V8_D_KEY: int = 8
W63_DEFAULT_V8_TRUST_EMA: float = 0.5
W63_DEFAULT_V8_REPLAY_DETERMINISM_TOL: float = 1e-6


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class TinyV8SubstrateConfig:
    vocab_size: int = W63_TINY_V8_VOCAB_SIZE
    d_model: int = W63_DEFAULT_V8_D_MODEL
    n_heads: int = W63_DEFAULT_V8_N_HEADS
    n_kv_heads: int = W63_DEFAULT_V8_N_KV_HEADS
    n_layers: int = W63_DEFAULT_V8_N_LAYERS
    ff_hidden: int = W63_DEFAULT_V8_FF_HIDDEN
    max_len: int = W63_DEFAULT_V8_MAX_LEN
    init_scale: float = W63_DEFAULT_V8_INIT_SCALE
    seed: int = W63_DEFAULT_V8_SEED
    rope_base: float = W63_DEFAULT_V8_ROPE_BASE
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    d_key: int = W63_DEFAULT_V8_D_KEY
    trust_ema: float = W63_DEFAULT_V8_TRUST_EMA
    replay_determinism_tol: float = (
        W63_DEFAULT_V8_REPLAY_DETERMINISM_TOL)
    expose_hidden_vs_kv_contention: bool = True
    expose_hidden_state_confidence: bool = True
    expose_replay_determinism_channel: bool = True
    expose_prefix_reuse_trust: bool = True
    expose_cross_layer_head_coupling: bool = True

    def to_v7_config(self) -> TinyV7SubstrateConfig:
        return TinyV7SubstrateConfig(
            vocab_size=int(self.vocab_size),
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_kv_heads=int(self.n_kv_heads),
            n_layers=int(self.n_layers),
            ff_hidden=int(self.ff_hidden),
            max_len=int(self.max_len),
            init_scale=float(self.init_scale),
            seed=int(self.seed),
            rope_base=float(self.rope_base),
            use_rope=bool(self.use_rope),
            use_rmsnorm=bool(self.use_rmsnorm),
            use_swiglu=bool(self.use_swiglu),
            d_key=int(self.d_key),
            trust_ema=float(self.trust_ema))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W63_TINY_SUBSTRATE_V8_SCHEMA_VERSION,
            "vocab_size": int(self.vocab_size),
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "n_kv_heads": int(self.n_kv_heads),
            "n_layers": int(self.n_layers),
            "ff_hidden": int(self.ff_hidden),
            "max_len": int(self.max_len),
            "init_scale": float(self.init_scale),
            "seed": int(self.seed),
            "rope_base": float(self.rope_base),
            "use_rope": bool(self.use_rope),
            "use_rmsnorm": bool(self.use_rmsnorm),
            "use_swiglu": bool(self.use_swiglu),
            "d_key": int(self.d_key),
            "trust_ema": float(round(self.trust_ema, 12)),
            "replay_determinism_tol": float(round(
                self.replay_determinism_tol, 12)),
            "expose_hidden_vs_kv_contention": bool(
                self.expose_hidden_vs_kv_contention),
            "expose_hidden_state_confidence": bool(
                self.expose_hidden_state_confidence),
            "expose_replay_determinism_channel": bool(
                self.expose_replay_determinism_channel),
            "expose_prefix_reuse_trust": bool(
                self.expose_prefix_reuse_trust),
            "expose_cross_layer_head_coupling": bool(
                self.expose_cross_layer_head_coupling),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v8_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV8SubstrateParams:
    config: TinyV8SubstrateConfig
    v7_params: TinyV7SubstrateParams
    # Fixed random projection for the "hidden-state confidence
    # probe" - maps logit-lens entropy to confidence in [0, 1].
    confidence_calibration: "_np.ndarray"   # (L, 2)

    @classmethod
    def init(
            cls, config: TinyV8SubstrateConfig | None = None,
    ) -> "TinyV8SubstrateParams":
        if config is None:
            config = TinyV8SubstrateConfig()
        v7 = TinyV7SubstrateParams.init(config.to_v7_config())
        rng = _np.random.default_rng(
            int(config.seed) ^ 0xCAFEBABE_63)
        # Per-layer affine [slope, bias] for the confidence map.
        cal = rng.standard_normal(
            (int(config.n_layers), 2)) * 0.2
        cal[:, 0] = -0.3
        cal[:, 1] = 0.5
        return cls(
            config=config, v7_params=v7,
            confidence_calibration=cal.astype(_np.float64))

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return self.v7_params.v3_params

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v8_substrate_params",
            "config_cid": self.config.cid(),
            "v7_params_cid": self.v7_params.cid(),
            "confidence_calibration_cid": _ndarray_cid(
                self.confidence_calibration),
        })


@dataclasses.dataclass
class TinyV8KVCache:
    """V8 cache. Wraps a V7 cache + five new substrate-internal
    axes. The hidden_vs_kv_contention, replay_determinism_channel
    and prefix_reuse_trust ledgers are maintained by V8 bridges and
    the V63 replay controller.
    """
    v7_cache: TinyV7KVCache
    hidden_vs_kv_contention: "_np.ndarray"  # (L, H, T)
    replay_determinism_channel: "_np.ndarray"  # (L, H, T)
    prefix_reuse_trust: "_np.ndarray"  # (L, H)
    prev_cache_write_ledger: "_np.ndarray"  # (L, H, T)
    write_log_v8: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
            d_key: int = W61_DEFAULT_V6_D_KEY,
    ) -> "TinyV8KVCache":
        v7 = TinyV7KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            max_len=int(max_len), d_key=int(d_key))
        return cls(
            v7_cache=v7,
            hidden_vs_kv_contention=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            replay_determinism_channel=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            prefix_reuse_trust=_np.zeros(
                (int(n_layers), int(n_heads)),
                dtype=_np.float64),
            prev_cache_write_ledger=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            write_log_v8=[])

    def n_tokens(self) -> int:
        return int(self.v7_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v7_cache.n_layers())

    def clone(self) -> "TinyV8KVCache":
        return TinyV8KVCache(
            v7_cache=self.v7_cache.clone(),
            hidden_vs_kv_contention=(
                self.hidden_vs_kv_contention.copy()),
            replay_determinism_channel=(
                self.replay_determinism_channel.copy()),
            prefix_reuse_trust=self.prefix_reuse_trust.copy(),
            prev_cache_write_ledger=(
                self.prev_cache_write_ledger.copy()),
            write_log_v8=list(self.write_log_v8),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v8_kv_cache",
            "v7_cache_cid": self.v7_cache.cid(),
            "hidden_vs_kv_contention_cid": _ndarray_cid(
                self.hidden_vs_kv_contention),
            "replay_determinism_channel_cid": _ndarray_cid(
                self.replay_determinism_channel),
            "prefix_reuse_trust_cid": _ndarray_cid(
                self.prefix_reuse_trust),
            "prev_cache_write_ledger_cid": _ndarray_cid(
                self.prev_cache_write_ledger),
            "write_log_v8": list(self.write_log_v8),
        })


@dataclasses.dataclass
class TinyV8ForwardTrace:
    v7_trace: TinyV7ForwardTrace
    hidden_state_confidence_per_layer: "_np.ndarray"  # (L,)
    cross_layer_head_coupling: "_np.ndarray"  # (L, H, L, H)
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v7_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v8_forward_trace",
            "v7_trace_cid": self.v7_trace.cid(),
            "hidden_state_confidence_cid": _ndarray_cid(
                self.hidden_state_confidence_per_layer),
            "cross_layer_head_coupling_cid": _ndarray_cid(
                self.cross_layer_head_coupling),
        })


def _compute_hidden_state_confidence(
        v7_trace: TinyV7ForwardTrace,
        confidence_calibration: "_np.ndarray",
) -> "_np.ndarray":
    """Confidence = sigmoid(slope * entropy + bias) per layer."""
    ent = logit_lens_entropy_per_layer(v7_trace)
    L = int(confidence_calibration.shape[0])
    if ent.size == 0:
        return _np.zeros((L,), dtype=_np.float64)
    n = int(min(L, ent.size))
    out = _np.zeros((L,), dtype=_np.float64)
    for li in range(n):
        x = (
            float(confidence_calibration[li, 0]) * float(ent[li])
            + float(confidence_calibration[li, 1]))
        out[li] = 1.0 / (1.0 + math.exp(-x))
    return _np.round(out, decimals=12)


def _compute_cross_layer_head_coupling(
        v7_trace: TinyV7ForwardTrace,
) -> "_np.ndarray":
    """Cross-layer-head coupling = cosine over attention-receive
    rows. Output: (L, H, L, H) with diagonal entries = 1.0."""
    ar = v7_trace.v6_trace.v5_trace.attention_receive_per_layer
    if not ar:
        return _np.zeros((0, 0, 0, 0), dtype=_np.float64)
    L = int(len(ar))
    arrs = [_np.asarray(a, dtype=_np.float64) for a in ar]
    H = int(arrs[0].shape[0]) if arrs and arrs[0].ndim == 2 else 0
    if H == 0:
        return _np.zeros((L, 0, L, 0), dtype=_np.float64)
    # Pad each layer to the same T (use max).
    T_max = int(max(a.shape[-1] for a in arrs))
    feats = _np.zeros((L, H, T_max), dtype=_np.float64)
    for li, a in enumerate(arrs):
        if a.ndim == 2:
            T = int(a.shape[-1])
            feats[li, :, :T] = a
    # Flatten to vectors per (layer, head): F[li, hi] = T_max-dim.
    out = _np.zeros((L, H, L, H), dtype=_np.float64)
    for la in range(L):
        for ha in range(H):
            va = feats[la, ha]
            na = float(_np.linalg.norm(va))
            for lb in range(L):
                for hb in range(H):
                    vb = feats[lb, hb]
                    nb = float(_np.linalg.norm(vb))
                    if na > 0.0 and nb > 0.0:
                        out[la, ha, lb, hb] = float(
                            float(_np.dot(va, vb)) / (na * nb))
                    elif la == lb and ha == hb:
                        out[la, ha, lb, hb] = 1.0
    return _np.round(out, decimals=12)


def forward_tiny_substrate_v8(
        params: TinyV8SubstrateParams,
        token_ids: Sequence[int],
        *,
        v8_kv_cache: TinyV8KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> tuple[TinyV8ForwardTrace, TinyV8KVCache]:
    cfg = params.config
    base_v7 = (
        v8_kv_cache.v7_cache if v8_kv_cache is not None else None)
    v7_trace, new_v7 = forward_tiny_substrate_v7(
        params.v7_params, list(token_ids),
        v7_kv_cache=base_v7,
        attention_bias_per_layer=attention_bias_per_layer)
    if v8_kv_cache is None:
        v8_new = TinyV8KVCache.empty(
            int(cfg.n_layers), n_heads=int(cfg.n_heads),
            max_len=int(cfg.max_len), d_key=int(cfg.d_key))
    else:
        v8_new = v8_kv_cache.clone()
    # Snapshot prev cache_write_ledger BEFORE replacing v7_cache.
    prev_cwl = v8_new.v7_cache.cache_write_ledger.copy()
    v8_new.v7_cache = new_v7
    v8_new.prev_cache_write_ledger = prev_cwl
    # Update replay-determinism channel: 1.0 where cache_write
    # ledger is stable (diff < tol).
    new_cwl = new_v7.cache_write_ledger
    L = int(new_cwl.shape[0])
    H = int(new_cwl.shape[1])
    T_new = int(new_cwl.shape[2])
    T_prev = int(prev_cwl.shape[2])
    T_min = int(min(T_new, T_prev))
    if T_new > int(v8_new.replay_determinism_channel.shape[2]):
        pad = _np.zeros(
            (L, H,
             T_new - int(
                 v8_new.replay_determinism_channel.shape[2])),
            dtype=_np.float64)
        v8_new.replay_determinism_channel = _np.concatenate(
            [v8_new.replay_determinism_channel, pad], axis=-1)
        v8_new.hidden_vs_kv_contention = _np.concatenate(
            [v8_new.hidden_vs_kv_contention, pad.copy()],
            axis=-1)
    if T_min > 0:
        delta = _np.abs(
            new_cwl[:, :, :T_min] - prev_cwl[:, :, :T_min])
        stable = (delta < float(cfg.replay_determinism_tol))
        v8_new.replay_determinism_channel[
            :, :, :T_min] = stable.astype(_np.float64)
    # Hidden-state confidence per layer.
    conf = _compute_hidden_state_confidence(
        v7_trace, params.confidence_calibration)
    # Cross-layer-head coupling.
    cc = _compute_cross_layer_head_coupling(v7_trace)
    v8_new.write_log_v8.append({
        "schema": W63_TINY_SUBSTRATE_V8_SCHEMA_VERSION,
        "kind": "forward_v8",
        "n_new_tokens": int(len(list(token_ids))),
        "hidden_state_confidence_mean": float(conf.mean())
            if conf.size else 0.0,
        "cross_layer_head_coupling_mean": float(cc.mean())
            if cc.size else 0.0,
    })
    trace = TinyV8ForwardTrace(
        v7_trace=v7_trace,
        hidden_state_confidence_per_layer=conf,
        cross_layer_head_coupling=cc,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v8_new


def record_hidden_vs_kv_contention_v8(
        cache: TinyV8KVCache, *,
        layer_index: int, head_index: int,
        slot: int,
        hidden_write_abs: float, kv_write_abs: float,
) -> None:
    """Record per-(layer, head, slot) signed contention =
    hidden_write_abs - kv_write_abs."""
    L, H, T = cache.hidden_vs_kv_contention.shape
    if not (0 <= int(layer_index) < L
            and 0 <= int(head_index) < H):
        return
    if int(slot) >= T:
        pad = _np.zeros(
            (L, H, int(slot) - T + 1), dtype=_np.float64)
        cache.hidden_vs_kv_contention = _np.concatenate(
            [cache.hidden_vs_kv_contention, pad], axis=-1)
    val = float(hidden_write_abs) - float(kv_write_abs)
    cache.hidden_vs_kv_contention[
        int(layer_index), int(head_index),
        int(slot)] = float(
            cache.hidden_vs_kv_contention[
                int(layer_index), int(head_index),
                int(slot)]) + val
    cache.write_log_v8.append({
        "schema": W63_TINY_SUBSTRATE_V8_SCHEMA_VERSION,
        "kind": "hidden_vs_kv_contention_write",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "slot": int(slot),
        "value": float(round(val, 12)),
    })


W63_PREFIX_DECISION_REUSE_SUCCESS_FLAG: float = 1.0
W63_PREFIX_DECISION_REUSE_DRIFT_FLAG: float = -1.0
W63_PREFIX_DECISION_RECOMPUTE_FLAG: float = 0.0


_PREFIX_DECISION_TO_FLAG: dict[str, float] = {
    "prefix_reuse_success": W63_PREFIX_DECISION_REUSE_SUCCESS_FLAG,
    "prefix_reuse_drift": W63_PREFIX_DECISION_REUSE_DRIFT_FLAG,
    "prefix_recompute": W63_PREFIX_DECISION_RECOMPUTE_FLAG,
}


def record_prefix_reuse_decision_v8(
        cache: TinyV8KVCache, *,
        layer_index: int, head_index: int,
        decision: str,
        trust_ema: float = W63_DEFAULT_V8_TRUST_EMA,
) -> None:
    """EMA-update the prefix-reuse trust ledger for (layer, head)."""
    L = int(cache.prefix_reuse_trust.shape[0])
    H = int(cache.prefix_reuse_trust.shape[1])
    if not (0 <= int(layer_index) < L
            and 0 <= int(head_index) < H):
        return
    flag = float(_PREFIX_DECISION_TO_FLAG.get(
        str(decision),
        W63_PREFIX_DECISION_RECOMPUTE_FLAG))
    alpha = float(max(0.0, min(1.0, float(trust_ema))))
    prev = float(cache.prefix_reuse_trust[
        int(layer_index), int(head_index)])
    cache.prefix_reuse_trust[
        int(layer_index), int(head_index)] = (
        alpha * flag + (1.0 - alpha) * prev)
    cache.write_log_v8.append({
        "schema": W63_TINY_SUBSTRATE_V8_SCHEMA_VERSION,
        "kind": "prefix_reuse_decision",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "decision": str(decision),
        "flag": float(round(flag, 12)),
    })


def hidden_vs_kv_contention_summary(
        cache: TinyV8KVCache,
) -> dict[str, float]:
    """Summary of the hidden-vs-KV contention tensor."""
    c = _np.asarray(
        cache.hidden_vs_kv_contention, dtype=_np.float64)
    pos = float(_np.sum(_np.where(c > 0.0, c, 0.0)))
    neg = float(_np.sum(_np.where(c < 0.0, -c, 0.0)))
    return {
        "hidden_dominant_l1": float(round(pos, 12)),
        "kv_dominant_l1": float(round(neg, 12)),
        "net_contention": float(round(pos - neg, 12)),
    }


# ---------------------------------------------------------------------------
# Witness
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TinyV8SubstrateForwardWitness:
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    v7_witness_cid: str
    hidden_state_confidence_cid: str
    hidden_state_confidence_mean: float
    cross_layer_head_coupling_cid: str
    cross_layer_head_coupling_mean: float
    hidden_vs_kv_contention_l1: float
    replay_determinism_channel_l1: float
    prefix_reuse_trust_l1: float
    n_layers: int
    n_heads: int
    forward_count: int
    schema: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "v7_witness_cid": str(self.v7_witness_cid),
            "hidden_state_confidence_cid": str(
                self.hidden_state_confidence_cid),
            "hidden_state_confidence_mean": float(round(
                self.hidden_state_confidence_mean, 12)),
            "cross_layer_head_coupling_cid": str(
                self.cross_layer_head_coupling_cid),
            "cross_layer_head_coupling_mean": float(round(
                self.cross_layer_head_coupling_mean, 12)),
            "hidden_vs_kv_contention_l1": float(round(
                self.hidden_vs_kv_contention_l1, 12)),
            "replay_determinism_channel_l1": float(round(
                self.replay_determinism_channel_l1, 12)),
            "prefix_reuse_trust_l1": float(round(
                self.prefix_reuse_trust_l1, 12)),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "forward_count": int(self.forward_count),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v8_substrate_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v8_forward_witness(
        trace: TinyV8ForwardTrace,
        cache: TinyV8KVCache,
) -> TinyV8SubstrateForwardWitness:
    from .tiny_substrate_v7 import (
        emit_tiny_substrate_v7_forward_witness,
    )
    v7w = emit_tiny_substrate_v7_forward_witness(
        trace.v7_trace, cache.v7_cache)
    return TinyV8SubstrateForwardWitness(
        schema=W63_TINY_SUBSTRATE_V8_SCHEMA_VERSION,
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(int(t) for t in v7w.token_ids),
        v7_witness_cid=str(v7w.cid()),
        hidden_state_confidence_cid=_ndarray_cid(
            _np.asarray(
                trace.hidden_state_confidence_per_layer,
                dtype=_np.float64)),
        hidden_state_confidence_mean=float(
            trace.hidden_state_confidence_per_layer.mean())
            if trace.hidden_state_confidence_per_layer.size
            else 0.0,
        cross_layer_head_coupling_cid=_ndarray_cid(
            _np.asarray(
                trace.cross_layer_head_coupling,
                dtype=_np.float64)),
        cross_layer_head_coupling_mean=float(
            trace.cross_layer_head_coupling.mean())
            if trace.cross_layer_head_coupling.size else 0.0,
        hidden_vs_kv_contention_l1=float(_np.linalg.norm(
            cache.hidden_vs_kv_contention.ravel(), ord=1)),
        replay_determinism_channel_l1=float(_np.linalg.norm(
            cache.replay_determinism_channel.ravel(), ord=1)),
        prefix_reuse_trust_l1=float(_np.linalg.norm(
            cache.prefix_reuse_trust.ravel(), ord=1)),
        n_layers=int(cache.n_layers()),
        n_heads=int(cache.prefix_reuse_trust.shape[1]),
        forward_count=int(
            cache.v7_cache.v6_cache.forward_count),
    )


def build_default_tiny_substrate_v8(
        *, seed: int = W63_DEFAULT_V8_SEED,
) -> TinyV8SubstrateParams:
    return TinyV8SubstrateParams.init(
        TinyV8SubstrateConfig(seed=int(seed)))


def tokenize_bytes_v8(
        text: str, *,
        max_len: int = W63_DEFAULT_V8_MAX_LEN,
        add_bos: bool = True,
) -> list[int]:
    from .tiny_substrate_v7 import tokenize_bytes_v7
    return tokenize_bytes_v7(
        text, max_len=int(max_len), add_bos=bool(add_bos))


def detokenize_bytes_v8(token_ids: Sequence[int]) -> str:
    from .tiny_substrate_v7 import detokenize_bytes_v7
    return detokenize_bytes_v7(list(token_ids))


__all__ = [
    "W63_TINY_SUBSTRATE_V8_SCHEMA_VERSION",
    "W63_TINY_V8_VOCAB_SIZE",
    "W63_DEFAULT_V8_D_MODEL",
    "W63_DEFAULT_V8_N_HEADS",
    "W63_DEFAULT_V8_N_KV_HEADS",
    "W63_DEFAULT_V8_N_LAYERS",
    "W63_DEFAULT_V8_FF_HIDDEN",
    "W63_DEFAULT_V8_MAX_LEN",
    "W63_DEFAULT_V8_INIT_SCALE",
    "W63_DEFAULT_V8_SEED",
    "W63_DEFAULT_V8_ROPE_BASE",
    "W63_DEFAULT_V8_D_KEY",
    "W63_DEFAULT_V8_TRUST_EMA",
    "W63_DEFAULT_V8_REPLAY_DETERMINISM_TOL",
    "W63_PREFIX_DECISION_REUSE_SUCCESS_FLAG",
    "W63_PREFIX_DECISION_REUSE_DRIFT_FLAG",
    "W63_PREFIX_DECISION_RECOMPUTE_FLAG",
    "TinyV8SubstrateConfig",
    "TinyV8SubstrateParams",
    "TinyV8KVCache",
    "TinyV8ForwardTrace",
    "TinyV8SubstrateForwardWitness",
    "forward_tiny_substrate_v8",
    "record_hidden_vs_kv_contention_v8",
    "record_prefix_reuse_decision_v8",
    "hidden_vs_kv_contention_summary",
    "emit_tiny_substrate_v8_forward_witness",
    "build_default_tiny_substrate_v8",
    "tokenize_bytes_v8", "detokenize_bytes_v8",
]
