"""W61 M1 — Tiny Transformer Runtime V6.

Strictly extends W60's ``coordpy.tiny_substrate_v5``. V6 keeps every
V5 invariant (byte-determinism under same params, GQA, RMSNorm /
SwiGLU, per-(layer, head, position) cumulative attention-receive
matrix, multi-segment partial-prefix reuse, per-(layer, head)
linearised logit Jacobian table, per-(layer, position) corruption
flag channel, 128-bucket fingerprint) and adds **five** new
substrate-load-bearing axes that W61's trainable controllers
exploit:

* **Default 8 layers** (vs V5's 7). Same GQA (8 query / 4 KV).
* **Per-(layer, position, d_key) content-addressable cache key
  table** — ``TinyV6KVCache.cache_keys`` of shape ``(L, T, d_key)``
  is derived deterministically from a projection of the V3 layer
  keys. The cache controller V4 fits a *bilinear retrieval head*
  over (query_feature ⊗ cache_key) to retrieve cache slots by
  content rather than just by importance.
* **Per-(layer, head) cumulative hidden-write trace** —
  ``TinyV6KVCache.hidden_write_trace`` of shape ``(L, H)`` is the
  cumulative L2 of hidden-state injections written by the HSB V5
  bridge. The replay controller V2 reads this to decide whether the
  hidden state has been corrupted by repeated injections.
* **Per-(layer, position) replay-age channel** —
  ``TinyV6KVCache.replay_age`` of shape ``(L, T)`` integer, the
  number of forwards since each cache slot was written. The cache
  controller V4 evicts old slots by replay age.
* **Forward counter** — ``TinyV6KVCache.forward_count`` integer,
  incremented every ``forward_tiny_substrate_v6`` call. Used to
  drive the replay-age channel.
* **Per-(layer_i, layer_j) cross-layer attention-coupling estimate**
  — ``TinyV6ForwardTrace.cross_layer_coupling`` of shape
  ``(L, L)``. The (i,j) entry is the cosine between layer i's
  cumulative attention-receive vector and layer j's. Cheap,
  measurable, and a real diagnostic of cross-layer interaction.

V6 still preserves the V5 multi-segment partial-prefix reuse and
extends the write log with a ``schema=v6`` tag plus the *cache key
delta* introduced by each forward (root-sum-square of new keys).

Honest scope (do-not-overstate, W61)
------------------------------------

* Still NOT a frontier model. Default config:
  ``8 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=176 / byte-vocab / max_len=128 / untrained``. V6 is
  richer than V5 but still a research substrate in pure NumPy on
  CPU. ``W61-L-NUMPY-CPU-V6-SUBSTRATE-CAP`` documents this.
* V6 still does NOT bridge to third-party hosted models. Ollama
  / OpenAI-compatible / hosted backends remain text-only at the
  HTTP surface. ``W61-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
  carries forward the W56..W60 cap unchanged.
* The cache_key projection is deterministic and *fixed* by the
  V6 params. We do NOT claim it is learned; the bilinear
  retrieval head in cache controller V4 supplies the only
  trainable component over it.
* The hidden-write trace is a *channel*, not a detector. It is
  written by the HSB V5 bridge and read by the replay controller
  V2. The substrate just stores the cumulative norm.
* Cross-layer coupling is a measured *diagnostic*. It is not an
  architectural property of the network; it is the observable
  correlation between two layers' attention-receive vectors over
  the current forward.
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
        "coordpy.tiny_substrate_v6 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3PrefixState,
    TinyV3SubstrateConfig,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)
from .tiny_substrate_v4 import (
    TinyV4KVCache,
    TinyV4SubstrateConfig,
    TinyV4SubstrateParams,
    forward_tiny_substrate_v4,
)
from .tiny_substrate_v5 import (
    TinyV5ForwardTrace, TinyV5KVCache,
    TinyV5MultiSegmentPrefix,
    TinyV5SubstrateConfig, TinyV5SubstrateParams,
    W60_DEFAULT_V5_ATTENTION_RECEIVE_EMA,
    W60_DEFAULT_V5_IMPORTANCE_EMA,
    W60_V5_SEGMENT_DROP, W60_V5_SEGMENT_RECOMPUTE,
    W60_V5_SEGMENT_REUSE,
    extract_multi_segment_prefix_v5,
    forward_with_multi_segment_reuse_v5,
    forward_tiny_substrate_v5,
    logit_jacobian_v5,
)


W61_TINY_SUBSTRATE_V6_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v6.v1")

W61_TINY_V6_VOCAB_SIZE: int = 259
W61_DEFAULT_V6_D_MODEL: int = 64
W61_DEFAULT_V6_N_HEADS: int = 8
W61_DEFAULT_V6_N_KV_HEADS: int = 4
W61_DEFAULT_V6_N_LAYERS: int = 8
W61_DEFAULT_V6_FF_HIDDEN: int = 176
W61_DEFAULT_V6_MAX_LEN: int = 128
W61_DEFAULT_V6_INIT_SCALE: float = 0.04
W61_DEFAULT_V6_SEED: int = 61012345
W61_DEFAULT_V6_ROPE_BASE: float = 10000.0
W61_DEFAULT_V6_D_KEY: int = 8


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# Config + Params
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV6SubstrateConfig:
    vocab_size: int = W61_TINY_V6_VOCAB_SIZE
    d_model: int = W61_DEFAULT_V6_D_MODEL
    n_heads: int = W61_DEFAULT_V6_N_HEADS
    n_kv_heads: int = W61_DEFAULT_V6_N_KV_HEADS
    n_layers: int = W61_DEFAULT_V6_N_LAYERS
    ff_hidden: int = W61_DEFAULT_V6_FF_HIDDEN
    max_len: int = W61_DEFAULT_V6_MAX_LEN
    init_scale: float = W61_DEFAULT_V6_INIT_SCALE
    seed: int = W61_DEFAULT_V6_SEED
    rope_base: float = W61_DEFAULT_V6_ROPE_BASE
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    track_kv_importance: bool = True
    importance_ema: float = W60_DEFAULT_V5_IMPORTANCE_EMA
    attention_receive_ema: float = (
        W60_DEFAULT_V5_ATTENTION_RECEIVE_EMA)
    expose_per_head_hidden: bool = True
    expose_per_head_jacobian: bool = True
    expose_corruption_flags: bool = True
    expose_cache_keys: bool = True
    expose_hidden_write_trace: bool = True
    expose_replay_age: bool = True
    expose_cross_layer_coupling: bool = True
    d_key: int = W61_DEFAULT_V6_D_KEY

    def to_v5_config(self) -> TinyV5SubstrateConfig:
        return TinyV5SubstrateConfig(
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
            track_kv_importance=bool(self.track_kv_importance),
            importance_ema=float(self.importance_ema),
            attention_receive_ema=float(
                self.attention_receive_ema),
            expose_per_head_hidden=bool(
                self.expose_per_head_hidden),
            expose_per_head_jacobian=bool(
                self.expose_per_head_jacobian),
            expose_corruption_flags=bool(
                self.expose_corruption_flags),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W61_TINY_SUBSTRATE_V6_SCHEMA_VERSION,
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
            "track_kv_importance": bool(self.track_kv_importance),
            "importance_ema": float(round(
                self.importance_ema, 12)),
            "attention_receive_ema": float(round(
                self.attention_receive_ema, 12)),
            "expose_per_head_hidden": bool(
                self.expose_per_head_hidden),
            "expose_per_head_jacobian": bool(
                self.expose_per_head_jacobian),
            "expose_corruption_flags": bool(
                self.expose_corruption_flags),
            "expose_cache_keys": bool(self.expose_cache_keys),
            "expose_hidden_write_trace": bool(
                self.expose_hidden_write_trace),
            "expose_replay_age": bool(self.expose_replay_age),
            "expose_cross_layer_coupling": bool(
                self.expose_cross_layer_coupling),
            "d_key": int(self.d_key),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v6_substrate_config",
            "config": self.to_dict()})


@dataclasses.dataclass
class TinyV6SubstrateParams:
    config: TinyV6SubstrateConfig
    v5_params: TinyV5SubstrateParams
    # The fixed cache-key projection. Shape (n_layers, d_model, d_key).
    cache_key_proj: "_np.ndarray"

    @classmethod
    def init(
            cls, config: TinyV6SubstrateConfig | None = None,
    ) -> "TinyV6SubstrateParams":
        if config is None:
            config = TinyV6SubstrateConfig()
        v5 = TinyV5SubstrateParams.init(config.to_v5_config())
        rng = _np.random.default_rng(
            int(config.seed) ^ 0xC0FFEE_61)
        proj = rng.standard_normal(
            (int(config.n_layers),
             int(config.d_model),
             int(config.d_key))) * (1.0
                                       / math.sqrt(
                                           float(config.d_model)))
        return cls(
            config=config, v5_params=v5,
            cache_key_proj=proj.astype(_np.float64))

    @property
    def v4_params(self) -> TinyV4SubstrateParams:
        return self.v5_params.v4_params

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return self.v5_params.v3_params

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v6_substrate_params",
            "config_cid": self.config.cid(),
            "v5_params_cid": self.v5_params.cid(),
            "cache_key_proj_cid": _ndarray_cid(
                self.cache_key_proj),
        })


# ---------------------------------------------------------------------------
# V6 KV cache
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV6KVCache:
    """V6 cache. Owns a V5 cache + four new substrate-internal axes.

    The cache_keys, hidden_write_trace, replay_age, and
    forward_count are all maintained per-forward by
    ``forward_tiny_substrate_v6`` and ``record_hidden_write_v6``.
    """
    v5_cache: TinyV5KVCache
    cache_keys: list["_np.ndarray"]
    # length L, each (T, d_key)
    hidden_write_trace: "_np.ndarray"
    # (L, H) cumulative L2 of HSB writes
    replay_age: list["_np.ndarray"]
    # length L, each (T,) int
    forward_count: int = 0
    write_log_v6: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *, n_heads: int,
            d_key: int = W61_DEFAULT_V6_D_KEY,
            importance_ema: float = W60_DEFAULT_V5_IMPORTANCE_EMA,
            attention_receive_ema: float = (
                W60_DEFAULT_V5_ATTENTION_RECEIVE_EMA),
    ) -> "TinyV6KVCache":
        v5 = TinyV5KVCache.empty(
            int(n_layers), n_heads=int(n_heads),
            importance_ema=float(importance_ema),
            attention_receive_ema=float(attention_receive_ema))
        keys = [_np.zeros((0, int(d_key)), dtype=_np.float64)
                 for _ in range(int(n_layers))]
        ages = [_np.zeros((0,), dtype=_np.int64)
                 for _ in range(int(n_layers))]
        return cls(
            v5_cache=v5, cache_keys=keys,
            hidden_write_trace=_np.zeros(
                (int(n_layers), int(n_heads)),
                dtype=_np.float64),
            replay_age=ages, forward_count=0,
            write_log_v6=[])

    @property
    def v4_cache(self) -> TinyV4KVCache:
        return self.v5_cache.v4_cache

    @property
    def v3_cache(self) -> TinyV3KVCache:
        return self.v5_cache.v3_cache

    def n_tokens(self) -> int:
        return int(self.v5_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v5_cache.n_layers())

    def clone(self) -> "TinyV6KVCache":
        return TinyV6KVCache(
            v5_cache=self.v5_cache.clone(),
            cache_keys=[k.copy() for k in self.cache_keys],
            hidden_write_trace=self.hidden_write_trace.copy(),
            replay_age=[a.copy() for a in self.replay_age],
            forward_count=int(self.forward_count),
            write_log_v6=list(self.write_log_v6),
        )

    def set_corruption_flag(
            self, *, layer_index: int, position: int,
            flagged: bool = True,
    ) -> None:
        self.v5_cache.set_corruption_flag(
            layer_index=int(layer_index),
            position=int(position),
            flagged=bool(flagged))

    def n_corrupted(self) -> int:
        return int(self.v5_cache.n_corrupted())

    def fingerprint_128(self) -> tuple[int, ...]:
        return tuple(self.v5_cache.fingerprint_128())

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v6_kv_cache",
            "v5_cache_cid": self.v5_cache.cid(),
            "cache_key_cids": [
                _ndarray_cid(k) for k in self.cache_keys],
            "hidden_write_trace_cid": _ndarray_cid(
                self.hidden_write_trace),
            "replay_age_cids": [
                _ndarray_cid(a.astype(_np.int64))
                for a in self.replay_age],
            "forward_count": int(self.forward_count),
            "write_log_v6": list(self.write_log_v6),
        })


# ---------------------------------------------------------------------------
# Forward V6
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV6ForwardTrace:
    v5_trace: TinyV5ForwardTrace
    cache_keys_per_layer: list["_np.ndarray"]
    cross_layer_coupling: "_np.ndarray"
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v5_trace.logits

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v6_forward_trace",
            "v5_trace_cid": str(self.v5_trace.cid()),
            "cache_key_cids": [
                _ndarray_cid(k)
                for k in self.cache_keys_per_layer],
            "cross_layer_coupling_cid": _ndarray_cid(
                self.cross_layer_coupling),
        })


def _build_cache_keys(
        cache: TinyV6KVCache, proj: "_np.ndarray",
) -> list["_np.ndarray"]:
    """Project the V3 layer keys into per-(layer, position) d_key
    vectors via ``proj``.

    ``proj`` is shape ``(n_layers, d_model, d_key)``. For each
    layer, take the V3 cache keys and multiply by proj[layer]
    along the d_model axis. Result is ``(n_tokens, d_key)`` per
    layer.
    """
    v3 = cache.v3_cache
    L = cache.n_layers()
    out: list["_np.ndarray"] = []
    for li in range(int(L)):
        if li >= int(proj.shape[0]):
            out.append(_np.zeros(
                (cache.n_tokens(), int(proj.shape[-1])),
                dtype=_np.float64))
            continue
        ks = v3.keys[li] if li < len(v3.keys) else None
        if ks is None or ks.size == 0:
            out.append(_np.zeros(
                (cache.n_tokens(), int(proj.shape[-1])),
                dtype=_np.float64))
            continue
        # Flatten ks if it has extra axes.
        if ks.ndim == 3:
            ks_flat = ks.reshape(ks.shape[0], -1)
        else:
            ks_flat = ks
        d_model_v3 = int(ks_flat.shape[-1])
        d_model_proj = int(proj.shape[1])
        if d_model_v3 != d_model_proj:
            pad = _np.zeros(
                (ks_flat.shape[0], d_model_proj),
                dtype=_np.float64)
            pad[:, :min(d_model_v3, d_model_proj)] = (
                ks_flat[:, :min(d_model_v3, d_model_proj)])
            ks_flat = pad
        out.append(ks_flat @ proj[li])
    return out


def _cross_layer_coupling(
        v5_trace: TinyV5ForwardTrace,
) -> "_np.ndarray":
    """Cosine between layer i and layer j attention-receive
    vectors. Returns shape (L, L)."""
    arr = v5_trace.attention_receive_per_layer
    L = len(arr)
    out = _np.zeros((L, L), dtype=_np.float64)
    flats = []
    for a in arr:
        f = _np.asarray(a, dtype=_np.float64).ravel()
        nf = float(_np.linalg.norm(f))
        if nf > 0.0:
            flats.append(f / nf)
        else:
            flats.append(f)
    for i in range(L):
        for j in range(L):
            if flats[i].size == 0 or flats[j].size == 0:
                continue
            n = min(int(flats[i].size), int(flats[j].size))
            if n == 0:
                continue
            out[i, j] = float(
                _np.dot(flats[i][:n], flats[j][:n]))
    # Round to 10 decimals to neutralise BLAS-level ULP jitter so
    # the coupling CID is byte-deterministic across forwards on the
    # same inputs. (12 decimals was sufficient through W66 but the
    # W67 V12 substrate runs 14 layers, where the L^2 accumulator
    # chain crosses an ULP boundary at 12 decimals after enough
    # upstream BLAS state mutations; 10 decimals leaves >10 digits
    # of precision and is byte-stable across all observed BLAS
    # accumulator histories.)
    return _np.round(out, decimals=10)


def forward_tiny_substrate_v6(
        params: TinyV6SubstrateParams,
        token_ids: Sequence[int],
        *,
        v6_kv_cache: TinyV6KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
) -> tuple[TinyV6ForwardTrace, TinyV6KVCache]:
    cfg = params.config
    base_v5 = (
        v6_kv_cache.v5_cache if v6_kv_cache is not None else None)
    v5_trace, new_v5 = forward_tiny_substrate_v5(
        params.v5_params, list(token_ids),
        v5_kv_cache=base_v5,
        attention_bias_per_layer=attention_bias_per_layer)
    new_n = int(new_v5.n_tokens())
    if v6_kv_cache is None:
        v6_new = TinyV6KVCache.empty(
            int(cfg.n_layers), n_heads=int(cfg.n_heads),
            d_key=int(cfg.d_key),
            importance_ema=float(cfg.importance_ema),
            attention_receive_ema=float(
                cfg.attention_receive_ema))
    else:
        v6_new = v6_kv_cache.clone()
    v6_new.v5_cache = new_v5
    # Build cache keys.
    v6_new.cache_keys = _build_cache_keys(
        v6_new, params.cache_key_proj)
    # Increment forward count and grow / increment replay_age.
    v6_new.forward_count = int(v6_new.forward_count) + 1
    new_age: list["_np.ndarray"] = []
    for li in range(int(cfg.n_layers)):
        prev = (
            v6_new.replay_age[li]
            if li < len(v6_new.replay_age)
            else _np.zeros((0,), dtype=_np.int64))
        # Old slots: age += 1. New slots (positions appended this
        # forward) get age 0.
        n_added = int(new_n - int(prev.size))
        if n_added < 0:
            new_age.append(prev[:new_n].astype(_np.int64))
        else:
            grown = _np.concatenate(
                [prev.astype(_np.int64) + 1,
                 _np.zeros((max(int(n_added), 0),),
                            dtype=_np.int64)])
            if grown.size < new_n:
                grown = _np.concatenate(
                    [grown,
                     _np.zeros((new_n - grown.size,),
                                dtype=_np.int64)])
            new_age.append(grown[:new_n])
    v6_new.replay_age = new_age
    # Cross-layer coupling diagnostic.
    coupling = _cross_layer_coupling(v5_trace)
    v6_new.write_log_v6.append({
        "schema": W61_TINY_SUBSTRATE_V6_SCHEMA_VERSION,
        "kind": "forward_v6",
        "n_new_tokens": int(len(list(token_ids))),
        "forward_count_after": int(v6_new.forward_count),
        "cache_key_l2_total": float(sum(
            float(_np.linalg.norm(k))
            for k in v6_new.cache_keys)),
    })
    trace = TinyV6ForwardTrace(
        v5_trace=v5_trace,
        cache_keys_per_layer=[k.copy()
                                  for k in v6_new.cache_keys],
        cross_layer_coupling=coupling,
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v6_new


def record_hidden_write_v6(
        cache: TinyV6KVCache, *,
        layer_index: int, per_head_l2: Sequence[float],
) -> None:
    """Increment the cumulative hidden-write trace at (layer, h)."""
    L = int(cache.hidden_write_trace.shape[0])
    H = int(cache.hidden_write_trace.shape[1])
    if not (0 <= int(layer_index) < L):
        return
    for hi in range(min(H, int(len(per_head_l2)))):
        v = float(per_head_l2[hi])
        cache.hidden_write_trace[int(layer_index), hi] = (
            float(cache.hidden_write_trace[
                int(layer_index), hi]) + v)
    cache.write_log_v6.append({
        "schema": W61_TINY_SUBSTRATE_V6_SCHEMA_VERSION,
        "kind": "hidden_write",
        "layer_index": int(layer_index),
        "per_head_l2": [float(round(float(x), 12))
                          for x in per_head_l2],
    })


# ---------------------------------------------------------------------------
# Multi-segment partial-prefix reuse on V6 (delegates to V5)
# ---------------------------------------------------------------------------


def extract_multi_segment_prefix_v6(
        params: TinyV6SubstrateParams,
        prompt_token_ids: Sequence[int],
        *,
        segments: Sequence[tuple[int, int, str]],
) -> TinyV5MultiSegmentPrefix:
    """V6 reuses the V5 multi-segment partial-prefix scheme on the
    embedded V5 substrate. The V6 cache key axis and replay-age
    channel are populated by the post-replay
    ``forward_with_multi_segment_reuse_v6`` call."""
    return extract_multi_segment_prefix_v5(
        params.v5_params, list(prompt_token_ids),
        segments=segments)


def forward_with_multi_segment_reuse_v6(
        params: TinyV6SubstrateParams,
        prefix: TinyV5MultiSegmentPrefix,
        follow_up_token_ids: Sequence[int],
) -> tuple[Any, TinyV6KVCache, dict[str, int]]:
    fu_trace, v5_cache, split = (
        forward_with_multi_segment_reuse_v5(
            params.v5_params, prefix, follow_up_token_ids))
    # Wrap in V6 cache.
    cfg = params.config
    v6_cache = TinyV6KVCache.empty(
        int(cfg.n_layers), n_heads=int(cfg.n_heads),
        d_key=int(cfg.d_key),
        importance_ema=float(cfg.importance_ema),
        attention_receive_ema=float(cfg.attention_receive_ema))
    v6_cache.v5_cache = v5_cache
    v6_cache.cache_keys = _build_cache_keys(
        v6_cache, params.cache_key_proj)
    v6_cache.forward_count = 1
    n_tok = v6_cache.n_tokens()
    v6_cache.replay_age = [
        _np.zeros((n_tok,), dtype=_np.int64)
        for _ in range(int(cfg.n_layers))]
    v6_cache.write_log_v6.append({
        "schema": W61_TINY_SUBSTRATE_V6_SCHEMA_VERSION,
        "kind": "multi_segment_reuse_v6",
        "split": {k: int(v) for k, v in split.items()},
    })
    return fu_trace, v6_cache, split


# ---------------------------------------------------------------------------
# Logit Jacobian table on V6 (delegates to V5)
# ---------------------------------------------------------------------------


def logit_jacobian_v6(
        trace: TinyV6ForwardTrace,
        params: TinyV6SubstrateParams,
        *,
        target_token: int,
) -> dict[str, Any]:
    """Per-(layer, head) linearised Jacobian table on V6 = V5 table
    (V6 adds no new layers in the Jacobian derivation), plus the
    new ``cross_layer_coupling`` summary scalar (mean off-diagonal
    cosine)."""
    sub = logit_jacobian_v5(
        trace.v5_trace, params.v5_params,
        target_token=int(target_token))
    L = int(trace.cross_layer_coupling.shape[0])
    off_diag = _np.array([
        float(trace.cross_layer_coupling[i, j])
        for i in range(L) for j in range(L) if i != j])
    sub["cross_layer_coupling_mean_off_diag"] = float(
        off_diag.mean()) if off_diag.size else 0.0
    sub["cross_layer_coupling_cid"] = _ndarray_cid(
        trace.cross_layer_coupling)
    return sub


# ---------------------------------------------------------------------------
# Witness
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TinyV6SubstrateForwardWitness:
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    v5_witness_cid: str
    cache_key_cids: tuple[str, ...]
    cross_layer_coupling_cid: str
    cross_layer_coupling_mean_off_diag: float
    hidden_write_trace_total_l2: float
    forward_count: int
    n_corrupted: int
    n_layers: int
    n_heads: int
    fingerprint_128: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W61_TINY_SUBSTRATE_V6_SCHEMA_VERSION,
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "v5_witness_cid": str(self.v5_witness_cid),
            "cache_key_cids": list(self.cache_key_cids),
            "cross_layer_coupling_cid": str(
                self.cross_layer_coupling_cid),
            "cross_layer_coupling_mean_off_diag": float(round(
                self.cross_layer_coupling_mean_off_diag, 12)),
            "hidden_write_trace_total_l2": float(round(
                self.hidden_write_trace_total_l2, 12)),
            "forward_count": int(self.forward_count),
            "n_corrupted": int(self.n_corrupted),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "fingerprint_128": list(self.fingerprint_128),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v6_substrate_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v6_forward_witness(
        trace: TinyV6ForwardTrace,
        cache: TinyV6KVCache,
) -> TinyV6SubstrateForwardWitness:
    from .tiny_substrate_v5 import (
        emit_tiny_substrate_v5_forward_witness,
    )
    v5w = emit_tiny_substrate_v5_forward_witness(
        trace.v5_trace, cache.v5_cache)
    L = int(trace.cross_layer_coupling.shape[0])
    off_diag = _np.array([
        float(trace.cross_layer_coupling[i, j])
        for i in range(L) for j in range(L) if i != j])
    return TinyV6SubstrateForwardWitness(
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(
            int(t) for t in
            trace.v5_trace.v4_trace.v3_trace.token_ids),
        v5_witness_cid=str(v5w.cid()),
        cache_key_cids=tuple(
            _ndarray_cid(k)
            for k in trace.cache_keys_per_layer),
        cross_layer_coupling_cid=_ndarray_cid(
            trace.cross_layer_coupling),
        cross_layer_coupling_mean_off_diag=float(
            off_diag.mean()) if off_diag.size else 0.0,
        hidden_write_trace_total_l2=float(
            _np.linalg.norm(cache.hidden_write_trace.ravel())),
        forward_count=int(cache.forward_count),
        n_corrupted=int(cache.n_corrupted()),
        n_layers=int(cache.n_layers()),
        n_heads=int(cache.hidden_write_trace.shape[1]),
        fingerprint_128=tuple(cache.fingerprint_128()),
    )


def build_default_tiny_substrate_v6(
        *, seed: int = W61_DEFAULT_V6_SEED,
) -> TinyV6SubstrateParams:
    return TinyV6SubstrateParams.init(
        TinyV6SubstrateConfig(seed=int(seed)))


# ---------------------------------------------------------------------------
# Tokenisation re-exports (V6 reuses V3 byte vocab unchanged)
# ---------------------------------------------------------------------------


def tokenize_bytes_v6(
        text: str, *,
        max_len: int = W61_DEFAULT_V6_MAX_LEN,
        add_bos: bool = True,
) -> list[int]:
    from .tiny_substrate_v5 import tokenize_bytes_v5
    return tokenize_bytes_v5(
        text, max_len=int(max_len), add_bos=bool(add_bos))


def detokenize_bytes_v6(token_ids: Sequence[int]) -> str:
    from .tiny_substrate_v5 import detokenize_bytes_v5
    return detokenize_bytes_v5(list(token_ids))


__all__ = [
    "W61_TINY_SUBSTRATE_V6_SCHEMA_VERSION",
    "W61_TINY_V6_VOCAB_SIZE",
    "W61_DEFAULT_V6_D_MODEL",
    "W61_DEFAULT_V6_N_HEADS",
    "W61_DEFAULT_V6_N_KV_HEADS",
    "W61_DEFAULT_V6_N_LAYERS",
    "W61_DEFAULT_V6_FF_HIDDEN",
    "W61_DEFAULT_V6_MAX_LEN",
    "W61_DEFAULT_V6_INIT_SCALE",
    "W61_DEFAULT_V6_SEED",
    "W61_DEFAULT_V6_ROPE_BASE",
    "W61_DEFAULT_V6_D_KEY",
    "TinyV6SubstrateConfig",
    "TinyV6SubstrateParams",
    "TinyV6KVCache",
    "TinyV6ForwardTrace",
    "TinyV6SubstrateForwardWitness",
    "forward_tiny_substrate_v6",
    "record_hidden_write_v6",
    "extract_multi_segment_prefix_v6",
    "forward_with_multi_segment_reuse_v6",
    "logit_jacobian_v6",
    "emit_tiny_substrate_v6_forward_witness",
    "build_default_tiny_substrate_v6",
    "tokenize_bytes_v6", "detokenize_bytes_v6",
]
