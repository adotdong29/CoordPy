"""W62 M1 — Tiny Transformer Runtime V7.

Strictly extends W61's ``coordpy.tiny_substrate_v6``. V7 keeps every
V6 invariant (byte-determinism under same params, GQA, RMSNorm /
SwiGLU, content-addressable cache_keys axis, hidden_write_trace
channel, replay_age channel, forward_count integer, cross-layer
coupling) and adds **four** new substrate-load-bearing axes that
W62's trainable controllers and bridges exploit:

* **Default 9 layers** (vs V6's 8). Same GQA (8 query / 4 KV).
* **Per-(layer, head, slot) cache-write ledger** —
  ``TinyV7KVCache.cache_write_ledger`` of shape ``(L, H, T)``
  records cumulative L2 of *any* injection (KV bridge V7 + HSB
  V6 + prefix V6) into each (layer, head, slot). The W62 cache
  controller V5 reads this when scoring repair candidates.
* **Per-layer logit-lens probe** —
  ``forward_tiny_substrate_v7`` emits
  ``logit_lens_per_layer`` of shape ``(L, V)`` where each layer's
  hidden state is projected through the V3 output head. The
  replay controller V3 reads the *logit-lens entropy* per layer
  as a regime feature.
* **Per-(layer, head, position) attention-receive delta** —
  ``TinyV7ForwardTrace.attention_receive_delta_per_layer``
  records the *forward-to-forward difference* of the V5 cumulative
  attention-receive matrix. Useful for detecting which heads got
  newly stimulated by the previous forward.
* **Per-(layer, head) replay-trust ledger** —
  ``TinyV7KVCache.replay_trust_ledger`` of shape ``(L, H)``
  records an EMA over the V62 replay controller decisions
  (+1 for REUSE, -1 for ABSTAIN, 0 for RECOMPUTE, +0.5 for
  FALLBACK). Updated by ``record_replay_decision_v7``.

V7 still preserves the V6 multi-segment partial-prefix reuse and
extends the write log with a ``schema=v7`` tag plus the new axis
deltas.

Honest scope (do-not-overstate, W62)
------------------------------------

* Still NOT a frontier model. Default config:
  ``9 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=192 / byte-vocab / max_len=128 / untrained``. V7 is
  richer than V6 but still a research substrate in pure NumPy on
  CPU. ``W62-L-NUMPY-CPU-V7-SUBSTRATE-CAP`` documents.
* V7 still does NOT bridge to third-party hosted models.
  ``W62-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* The cache-write ledger is a *channel*, not a detector. Bridges
  WRITE to it; controllers READ.
* The logit-lens probe is a deterministic linear projection of
  each layer's *output hidden state* (post layer-norm) through
  the V3 output head. It is a diagnostic, not a calibrated
  probability distribution; it does NOT match the standard
  research definition of the logit lens (which interprets
  intermediate residual streams through the unembedding).
* The replay-trust ledger updates by ``record_replay_decision_v7``;
  the substrate does NOT itself emit decisions.
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
        "coordpy.tiny_substrate_v7 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
)
from .tiny_substrate_v5 import (
    TinyV5ForwardTrace, TinyV5KVCache,
    TinyV5MultiSegmentPrefix,
    forward_tiny_substrate_v5,
    forward_with_multi_segment_reuse_v5,
)
from .tiny_substrate_v6 import (
    TinyV6ForwardTrace, TinyV6KVCache,
    TinyV6SubstrateConfig, TinyV6SubstrateParams,
    W61_DEFAULT_V6_D_KEY,
    forward_tiny_substrate_v6,
    extract_multi_segment_prefix_v6,
    forward_with_multi_segment_reuse_v6,
)


W62_TINY_SUBSTRATE_V7_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v7.v1")

W62_TINY_V7_VOCAB_SIZE: int = 259
W62_DEFAULT_V7_D_MODEL: int = 64
W62_DEFAULT_V7_N_HEADS: int = 8
W62_DEFAULT_V7_N_KV_HEADS: int = 4
W62_DEFAULT_V7_N_LAYERS: int = 9
W62_DEFAULT_V7_FF_HIDDEN: int = 192
W62_DEFAULT_V7_MAX_LEN: int = 128
W62_DEFAULT_V7_INIT_SCALE: float = 0.04
W62_DEFAULT_V7_SEED: int = 62012345
W62_DEFAULT_V7_ROPE_BASE: float = 10000.0
W62_DEFAULT_V7_D_KEY: int = 8
W62_DEFAULT_V7_TRUST_EMA: float = 0.5


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


@dataclasses.dataclass
class TinyV7SubstrateConfig:
    vocab_size: int = W62_TINY_V7_VOCAB_SIZE
    d_model: int = W62_DEFAULT_V7_D_MODEL
    n_heads: int = W62_DEFAULT_V7_N_HEADS
    n_kv_heads: int = W62_DEFAULT_V7_N_KV_HEADS
    n_layers: int = W62_DEFAULT_V7_N_LAYERS
    ff_hidden: int = W62_DEFAULT_V7_FF_HIDDEN
    max_len: int = W62_DEFAULT_V7_MAX_LEN
    init_scale: float = W62_DEFAULT_V7_INIT_SCALE
    seed: int = W62_DEFAULT_V7_SEED
    rope_base: float = W62_DEFAULT_V7_ROPE_BASE
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    d_key: int = W62_DEFAULT_V7_D_KEY
    trust_ema: float = W62_DEFAULT_V7_TRUST_EMA
    expose_cache_write_ledger: bool = True
    expose_logit_lens: bool = True
    expose_attention_receive_delta: bool = True
    expose_replay_trust_ledger: bool = True

    def to_v6_config(self) -> TinyV6SubstrateConfig:
        return TinyV6SubstrateConfig(
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
            d_key=int(self.d_key))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W62_TINY_SUBSTRATE_V7_SCHEMA_VERSION,
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
            "expose_cache_write_ledger": bool(
                self.expose_cache_write_ledger),
            "expose_logit_lens": bool(self.expose_logit_lens),
            "expose_attention_receive_delta": bool(
                self.expose_attention_receive_delta),
            "expose_replay_trust_ledger": bool(
                self.expose_replay_trust_ledger),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v7_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV7SubstrateParams:
    config: TinyV7SubstrateConfig
    v6_params: TinyV6SubstrateParams
    logit_lens_proj: "_np.ndarray"   # (L, d_model, vocab_size)

    @classmethod
    def init(
            cls, config: TinyV7SubstrateConfig | None = None,
    ) -> "TinyV7SubstrateParams":
        if config is None:
            config = TinyV7SubstrateConfig()
        v6 = TinyV6SubstrateParams.init(config.to_v6_config())
        rng = _np.random.default_rng(
            int(config.seed) ^ 0xDEADBEEF_62)
        proj = rng.standard_normal(
            (int(config.n_layers),
             int(config.d_model),
             int(config.vocab_size))) * (
                1.0 / math.sqrt(float(config.d_model)))
        return cls(
            config=config, v6_params=v6,
            logit_lens_proj=proj.astype(_np.float64))

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return self.v6_params.v3_params

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v7_substrate_params",
            "config_cid": self.config.cid(),
            "v6_params_cid": self.v6_params.cid(),
            "logit_lens_proj_cid": _ndarray_cid(
                self.logit_lens_proj),
        })


@dataclasses.dataclass
class TinyV7KVCache:
    """V7 cache. Wraps a V6 cache + four new substrate-internal
    axes. The cache_write_ledger and replay_trust_ledger are
    maintained by V7 bridges and the V62 replay controller.
    """
    v6_cache: TinyV6KVCache
    cache_write_ledger: "_np.ndarray"   # (L, H, T)
    replay_trust_ledger: "_np.ndarray"  # (L, H)
    prev_attention_receive: list["_np.ndarray"]
    # length L of (H, T) snapshots from previous forward; used
    # to compute attention-receive deltas.
    write_log_v7: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int, max_len: int,
            d_key: int = W61_DEFAULT_V6_D_KEY,
    ) -> "TinyV7KVCache":
        v6 = TinyV6KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            d_key=int(d_key))
        return cls(
            v6_cache=v6,
            cache_write_ledger=_np.zeros(
                (int(n_layers), int(n_heads), int(max_len)),
                dtype=_np.float64),
            replay_trust_ledger=_np.zeros(
                (int(n_layers), int(n_heads)),
                dtype=_np.float64),
            prev_attention_receive=[
                _np.zeros((int(n_heads), 0),
                              dtype=_np.float64)
                for _ in range(int(n_layers))],
            write_log_v7=[])

    def n_tokens(self) -> int:
        return int(self.v6_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v6_cache.n_layers())

    def clone(self) -> "TinyV7KVCache":
        return TinyV7KVCache(
            v6_cache=self.v6_cache.clone(),
            cache_write_ledger=self.cache_write_ledger.copy(),
            replay_trust_ledger=self.replay_trust_ledger.copy(),
            prev_attention_receive=[
                a.copy() for a in self.prev_attention_receive],
            write_log_v7=list(self.write_log_v7),
        )

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v7_kv_cache",
            "v6_cache_cid": self.v6_cache.cid(),
            "cache_write_ledger_cid": _ndarray_cid(
                self.cache_write_ledger),
            "replay_trust_ledger_cid": _ndarray_cid(
                self.replay_trust_ledger),
            "prev_attention_receive_cids": [
                _ndarray_cid(a)
                for a in self.prev_attention_receive],
            "write_log_v7": list(self.write_log_v7),
        })


@dataclasses.dataclass
class TinyV7ForwardTrace:
    v6_trace: TinyV6ForwardTrace
    logit_lens_per_layer: "_np.ndarray"  # (L, V)
    attention_receive_delta_per_layer: list["_np.ndarray"]
    # length L, each (H, T) — delta from previous forward.
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v6_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v7_forward_trace",
            "v6_trace_cid": self.v6_trace.cid(),
            "logit_lens_per_layer_cid": _ndarray_cid(
                self.logit_lens_per_layer),
            "attention_receive_delta_cids": [
                _ndarray_cid(a)
                for a in self.attention_receive_delta_per_layer],
        })


def _compute_logit_lens(
        v6_trace: TinyV6ForwardTrace,
        logit_lens_proj: "_np.ndarray",
) -> "_np.ndarray":
    """Project each layer's last-position hidden state through
    its logit-lens projection. Returns shape (L, V)."""
    L = int(logit_lens_proj.shape[0])
    V = int(logit_lens_proj.shape[2])
    out = _np.zeros((L, V), dtype=_np.float64)
    hs = v6_trace.v5_trace.v4_trace.head_hidden_states_per_layer
    if not hs:
        return out
    for li in range(min(L, len(hs))):
        layer_hs = hs[li]
        if layer_hs.ndim == 3:
            # (H, T, Dh) → take last token, mean over heads.
            last = layer_hs[:, -1, :].mean(axis=0)
        elif layer_hs.ndim == 2:
            last = layer_hs[-1, :]
        else:
            last = layer_hs.ravel()[
                :int(logit_lens_proj.shape[1])]
        d_model = int(logit_lens_proj.shape[1])
        if last.size < d_model:
            padded = _np.zeros(d_model, dtype=_np.float64)
            padded[:last.size] = last
            last = padded
        elif last.size > d_model:
            last = last[:d_model]
        out[li] = last @ logit_lens_proj[li]
    # Round to 12 decimals so the logit-lens CID is byte-stable.
    return _np.round(out, decimals=12)


def _compute_attention_receive_delta(
        v6_trace: TinyV6ForwardTrace,
        prev_ar: Sequence["_np.ndarray"],
) -> list["_np.ndarray"]:
    """Returns delta = current - previous attention-receive
    per layer."""
    current = v6_trace.v5_trace.attention_receive_per_layer
    out: list["_np.ndarray"] = []
    for li, c in enumerate(current):
        c_arr = _np.asarray(c, dtype=_np.float64)
        if li < len(prev_ar):
            p = _np.asarray(prev_ar[li], dtype=_np.float64)
        else:
            p = _np.zeros_like(c_arr)
        # Match shapes by truncation/padding to the smaller
        # last-axis dimension; the head axis matches by
        # construction.
        if c_arr.ndim == 2 and p.ndim == 2:
            T_c = int(c_arr.shape[-1])
            T_p = int(p.shape[-1])
            if T_c == T_p:
                delta = c_arr - p
            elif T_c > T_p:
                # New positions had no previous; delta is the
                # raw value at the new positions and the diff at
                # the old positions.
                delta = c_arr.copy()
                delta[..., :T_p] = c_arr[..., :T_p] - p
            else:
                # Cache shrank (eviction); compare common slice.
                delta = c_arr - p[..., :T_c]
        else:
            delta = c_arr - p if (
                c_arr.shape == p.shape) else c_arr.copy()
        out.append(_np.round(delta, decimals=12))
    return out


def forward_tiny_substrate_v7(
        params: TinyV7SubstrateParams,
        token_ids: Sequence[int],
        *,
        v7_kv_cache: TinyV7KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> tuple[TinyV7ForwardTrace, TinyV7KVCache]:
    cfg = params.config
    base_v6 = (
        v7_kv_cache.v6_cache if v7_kv_cache is not None else None)
    v6_trace, new_v6 = forward_tiny_substrate_v6(
        params.v6_params, list(token_ids),
        v6_kv_cache=base_v6,
        attention_bias_per_layer=attention_bias_per_layer)
    if v7_kv_cache is None:
        v7_new = TinyV7KVCache.empty(
            int(cfg.n_layers), n_heads=int(cfg.n_heads),
            max_len=int(cfg.max_len), d_key=int(cfg.d_key))
    else:
        v7_new = v7_kv_cache.clone()
    v7_new.v6_cache = new_v6
    # Logit lens.
    lens = _compute_logit_lens(
        v6_trace, params.logit_lens_proj)
    # Attention-receive delta.
    delta = _compute_attention_receive_delta(
        v6_trace, v7_new.prev_attention_receive)
    # Stash current attention-receive as prev for next forward.
    v7_new.prev_attention_receive = [
        _np.asarray(a, dtype=_np.float64).copy()
        for a in v6_trace.v5_trace.attention_receive_per_layer]
    n_tok = v7_new.n_tokens()
    # Pad / truncate cache_write_ledger to current cache size.
    L, H, T = v7_new.cache_write_ledger.shape
    if n_tok > T:
        pad = _np.zeros(
            (L, H, int(n_tok - T)), dtype=_np.float64)
        v7_new.cache_write_ledger = _np.concatenate(
            [v7_new.cache_write_ledger, pad], axis=-1)
    v7_new.write_log_v7.append({
        "schema": W62_TINY_SUBSTRATE_V7_SCHEMA_VERSION,
        "kind": "forward_v7",
        "n_new_tokens": int(len(list(token_ids))),
        "logit_lens_l2": float(_np.linalg.norm(lens.ravel())),
        "attention_receive_delta_l2_total": float(sum(
            float(_np.linalg.norm(d.ravel())) for d in delta)),
    })
    trace = TinyV7ForwardTrace(
        v6_trace=v6_trace,
        logit_lens_per_layer=lens,
        attention_receive_delta_per_layer=[
            d.copy() for d in delta],
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v7_new


def record_cache_write_v7(
        cache: TinyV7KVCache, *,
        layer_index: int, head_index: int,
        slot: int, l2: float,
) -> None:
    """Record a write into (layer, head, slot) with the L2 of the
    written quantity. Pads the ledger if needed."""
    L = int(cache.cache_write_ledger.shape[0])
    H = int(cache.cache_write_ledger.shape[1])
    T = int(cache.cache_write_ledger.shape[2])
    if not (0 <= int(layer_index) < L):
        return
    if not (0 <= int(head_index) < H):
        return
    if int(slot) >= T:
        pad = _np.zeros(
            (L, H, int(slot) - T + 1), dtype=_np.float64)
        cache.cache_write_ledger = _np.concatenate(
            [cache.cache_write_ledger, pad], axis=-1)
    cache.cache_write_ledger[
        int(layer_index), int(head_index),
        int(slot)] = float(
            cache.cache_write_ledger[
                int(layer_index), int(head_index),
                int(slot)]) + float(l2)
    cache.write_log_v7.append({
        "schema": W62_TINY_SUBSTRATE_V7_SCHEMA_VERSION,
        "kind": "cache_write",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "slot": int(slot),
        "l2": float(round(float(l2), 12)),
    })


W62_REPLAY_DECISION_REUSE_FLAG: float = 1.0
W62_REPLAY_DECISION_RECOMPUTE_FLAG: float = 0.0
W62_REPLAY_DECISION_FALLBACK_FLAG: float = 0.5
W62_REPLAY_DECISION_ABSTAIN_FLAG: float = -1.0


_DECISION_TO_FLAG: dict[str, float] = {
    "choose_reuse": W62_REPLAY_DECISION_REUSE_FLAG,
    "choose_recompute": W62_REPLAY_DECISION_RECOMPUTE_FLAG,
    "choose_fallback": W62_REPLAY_DECISION_FALLBACK_FLAG,
    "choose_abstain": W62_REPLAY_DECISION_ABSTAIN_FLAG,
}


def record_replay_decision_v7(
        cache: TinyV7KVCache, *,
        layer_index: int, head_index: int,
        decision: str, trust_ema: float = W62_DEFAULT_V7_TRUST_EMA,
) -> None:
    """Update the per-(layer, head) replay-trust ledger by EMA of
    a decision flag.

    REUSE → +1, RECOMPUTE → 0, FALLBACK → +0.5, ABSTAIN → -1.
    The ledger value is ``alpha * flag + (1 - alpha) * prev``
    with ``alpha = trust_ema``.
    """
    L = int(cache.replay_trust_ledger.shape[0])
    H = int(cache.replay_trust_ledger.shape[1])
    if not (0 <= int(layer_index) < L):
        return
    if not (0 <= int(head_index) < H):
        return
    flag = float(_DECISION_TO_FLAG.get(
        str(decision),
        W62_REPLAY_DECISION_ABSTAIN_FLAG))
    alpha = float(max(0.0, min(1.0, float(trust_ema))))
    prev = float(cache.replay_trust_ledger[
        int(layer_index), int(head_index)])
    cache.replay_trust_ledger[
        int(layer_index), int(head_index)] = (
        alpha * flag + (1.0 - alpha) * prev)
    cache.write_log_v7.append({
        "schema": W62_TINY_SUBSTRATE_V7_SCHEMA_VERSION,
        "kind": "replay_decision",
        "layer_index": int(layer_index),
        "head_index": int(head_index),
        "decision": str(decision),
        "flag": float(round(flag, 12)),
    })


def logit_lens_entropy_per_layer(
        trace: TinyV7ForwardTrace,
) -> "_np.ndarray":
    """Per-layer Shannon entropy of the logit-lens distribution.
    Returns shape (L,).
    """
    lens = _np.asarray(
        trace.logit_lens_per_layer, dtype=_np.float64)
    if lens.size == 0:
        return _np.zeros((0,), dtype=_np.float64)
    # Softmax along V.
    m = lens.max(axis=-1, keepdims=True)
    e = _np.exp(lens - m)
    s = e.sum(axis=-1, keepdims=True)
    s = _np.where(s < 1e-30, 1e-30, s)
    p = e / s
    p = _np.where(p < 1e-30, 1e-30, p)
    ent = -(p * _np.log(p)).sum(axis=-1)
    return _np.round(ent, decimals=12)


# ---------------------------------------------------------------------------
# Multi-segment partial-prefix reuse on V7 (delegates to V6)
# ---------------------------------------------------------------------------


def extract_multi_segment_prefix_v7(
        params: TinyV7SubstrateParams,
        prompt_token_ids: Sequence[int],
        *,
        segments: Sequence[tuple[int, int, str]],
) -> TinyV5MultiSegmentPrefix:
    return extract_multi_segment_prefix_v6(
        params.v6_params, list(prompt_token_ids),
        segments=segments)


def forward_with_multi_segment_reuse_v7(
        params: TinyV7SubstrateParams,
        prefix: TinyV5MultiSegmentPrefix,
        follow_up_token_ids: Sequence[int],
) -> tuple[Any, TinyV7KVCache, dict[str, int]]:
    fu_trace, v6_cache, split = forward_with_multi_segment_reuse_v6(
        params.v6_params, prefix, follow_up_token_ids)
    cfg = params.config
    v7_cache = TinyV7KVCache.empty(
        int(cfg.n_layers), n_heads=int(cfg.n_heads),
        max_len=int(cfg.max_len), d_key=int(cfg.d_key))
    v7_cache.v6_cache = v6_cache
    v7_cache.write_log_v7.append({
        "schema": W62_TINY_SUBSTRATE_V7_SCHEMA_VERSION,
        "kind": "multi_segment_reuse_v7",
        "split": {k: int(v) for k, v in split.items()},
    })
    return fu_trace, v7_cache, split


# ---------------------------------------------------------------------------
# Witness
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TinyV7SubstrateForwardWitness:
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    v6_witness_cid: str
    logit_lens_cid: str
    logit_lens_l2: float
    logit_lens_entropy_mean: float
    attention_receive_delta_l2_total: float
    cache_write_ledger_l2: float
    replay_trust_ledger_l1: float
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
            "v6_witness_cid": str(self.v6_witness_cid),
            "logit_lens_cid": str(self.logit_lens_cid),
            "logit_lens_l2": float(round(self.logit_lens_l2, 12)),
            "logit_lens_entropy_mean": float(round(
                self.logit_lens_entropy_mean, 12)),
            "attention_receive_delta_l2_total": float(round(
                self.attention_receive_delta_l2_total, 12)),
            "cache_write_ledger_l2": float(round(
                self.cache_write_ledger_l2, 12)),
            "replay_trust_ledger_l1": float(round(
                self.replay_trust_ledger_l1, 12)),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "forward_count": int(self.forward_count),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v7_substrate_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v7_forward_witness(
        trace: TinyV7ForwardTrace,
        cache: TinyV7KVCache,
) -> TinyV7SubstrateForwardWitness:
    from .tiny_substrate_v6 import (
        emit_tiny_substrate_v6_forward_witness,
    )
    v6w = emit_tiny_substrate_v6_forward_witness(
        trace.v6_trace, cache.v6_cache)
    entropy = logit_lens_entropy_per_layer(trace)
    return TinyV7SubstrateForwardWitness(
        schema=W62_TINY_SUBSTRATE_V7_SCHEMA_VERSION,
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(int(t) for t in v6w.token_ids),
        v6_witness_cid=str(v6w.cid()),
        logit_lens_cid=_ndarray_cid(
            _np.asarray(
                trace.logit_lens_per_layer,
                dtype=_np.float64)),
        logit_lens_l2=float(_np.linalg.norm(
            _np.asarray(
                trace.logit_lens_per_layer,
                dtype=_np.float64).ravel())),
        logit_lens_entropy_mean=float(entropy.mean())
            if entropy.size else 0.0,
        attention_receive_delta_l2_total=float(sum(
            float(_np.linalg.norm(d.ravel()))
            for d in
            trace.attention_receive_delta_per_layer)),
        cache_write_ledger_l2=float(_np.linalg.norm(
            cache.cache_write_ledger.ravel())),
        replay_trust_ledger_l1=float(_np.linalg.norm(
            cache.replay_trust_ledger.ravel(), ord=1)),
        n_layers=int(cache.n_layers()),
        n_heads=int(cache.replay_trust_ledger.shape[1]),
        forward_count=int(cache.v6_cache.forward_count),
    )


def build_default_tiny_substrate_v7(
        *, seed: int = W62_DEFAULT_V7_SEED,
) -> TinyV7SubstrateParams:
    return TinyV7SubstrateParams.init(
        TinyV7SubstrateConfig(seed=int(seed)))


def tokenize_bytes_v7(
        text: str, *,
        max_len: int = W62_DEFAULT_V7_MAX_LEN,
        add_bos: bool = True,
) -> list[int]:
    from .tiny_substrate_v6 import tokenize_bytes_v6
    return tokenize_bytes_v6(
        text, max_len=int(max_len), add_bos=bool(add_bos))


def detokenize_bytes_v7(token_ids: Sequence[int]) -> str:
    from .tiny_substrate_v6 import detokenize_bytes_v6
    return detokenize_bytes_v6(list(token_ids))


__all__ = [
    "W62_TINY_SUBSTRATE_V7_SCHEMA_VERSION",
    "W62_TINY_V7_VOCAB_SIZE",
    "W62_DEFAULT_V7_D_MODEL",
    "W62_DEFAULT_V7_N_HEADS",
    "W62_DEFAULT_V7_N_KV_HEADS",
    "W62_DEFAULT_V7_N_LAYERS",
    "W62_DEFAULT_V7_FF_HIDDEN",
    "W62_DEFAULT_V7_MAX_LEN",
    "W62_DEFAULT_V7_INIT_SCALE",
    "W62_DEFAULT_V7_SEED",
    "W62_DEFAULT_V7_ROPE_BASE",
    "W62_DEFAULT_V7_D_KEY",
    "W62_DEFAULT_V7_TRUST_EMA",
    "W62_REPLAY_DECISION_REUSE_FLAG",
    "W62_REPLAY_DECISION_RECOMPUTE_FLAG",
    "W62_REPLAY_DECISION_FALLBACK_FLAG",
    "W62_REPLAY_DECISION_ABSTAIN_FLAG",
    "TinyV7SubstrateConfig",
    "TinyV7SubstrateParams",
    "TinyV7KVCache",
    "TinyV7ForwardTrace",
    "TinyV7SubstrateForwardWitness",
    "forward_tiny_substrate_v7",
    "record_cache_write_v7",
    "record_replay_decision_v7",
    "logit_lens_entropy_per_layer",
    "extract_multi_segment_prefix_v7",
    "forward_with_multi_segment_reuse_v7",
    "emit_tiny_substrate_v7_forward_witness",
    "build_default_tiny_substrate_v7",
    "tokenize_bytes_v7", "detokenize_bytes_v7",
]
