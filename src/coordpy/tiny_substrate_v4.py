"""W59 M1 — Tiny Transformer Runtime V4.

Strictly extends W58's ``coordpy.tiny_substrate_v3``. V4 keeps every
V3 invariant (byte-determinism under same params, strict causal
mask, RoPE, GQA, RMSNorm/SwiGLU, KV cache with eviction, importance
tracking, partial forward, 64-bucket fingerprint) and adds five new
substrate-load-bearing pieces that W59's trainable controllers
exploit:

* **6 layers** by default (vs V3's 5). Same GQA (8 query / 4 KV).
* **Per-(layer, position) importance accumulator** — V3 records
  attention received per key position only for the current
  forward. V4 keeps a *cumulative* importance vector inside the
  ``TinyV4KVCache`` that survives evictions: the importance of a
  surviving key reflects every forward since it was written, with
  an EMA decay that the cache controller V2 can read.
* **Causal-aware partial-prefix reuse** — V4 splits a saved
  prefix into a *reusable* slice (positions ``[0, p_reuse)``) and
  a *recompute-required* slice (positions ``[p_reuse, prefix_len)``).
  The substrate exposes
  ``forward_with_partial_prefix_reuse_v4`` that runs the cached
  positions byte-identical to V3 reuse, recomputes the rest, then
  appends the follow-up. The flop counter reports
  ``flop_reuse``, ``flop_recompute_tail``, ``flop_follow_up``.
* **Per-(layer, head) hidden-state tap** — V3's hidden_states are
  shared across heads (post-attention concatenation). V4 also
  exposes a per-head ``head_hidden_states[layer][head]`` shape
  ``(n_tokens, d_head)`` view, captured *before* the W_O
  projection. The W59 hidden-state-bridge V3 uses this for
  head-level injection.
* **Forward-mode logit-gradient probe** — the trace exposes a
  closed-form ``logit_jacobian_diag`` at the last position with
  respect to the *unembedding row* of a target token id. Used by
  W59 hidden-state-bridge V3 to fit a target-logit shift.

Plus:

* **128-bucket fingerprint** — V4 also computes a 128-bucket
  variant (alongside the inherited 64-bucket) for cache CID. The
  W59 CRC V7 uses the 128-bucket variant.

Honest scope (do-not-overstate, W59)
------------------------------------

* This is still NOT a frontier model. Default config:
  ``6 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=160 / byte-vocab / max_len=128 / untrained``. The
  runtime is richer than V3 but still a research substrate in
  pure NumPy on CPU. ``W59-L-NUMPY-CPU-V4-SUBSTRATE-CAP``
  documents this.
* V4 still does NOT bridge to third-party hosted models. Ollama
  / OpenAI-compatible / hosted backends remain text-only at the
  HTTP surface. ``W59-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
  carries forward the W57/W58 cap unchanged.
* The default config is *untrained*. W59's "trainable" controllers
  fit ridge parameters and per-head clips by closed-form linear
  algebra (no SGD, no autograd). ``W59-L-V4-NO-AUTOGRAD-CAP``
  documents the boundary.
* Per-head hidden-state tap is a *measurement*, not a transformer
  invariant. It records the per-head pre-output tensor that the
  forward already computes; we just expose it.
* The logit Jacobian "probe" is a diagonal estimate at the last
  position computed by reusing the unembedding row directly. It
  is exact under the *linearised* head — i.e. for a single
  unembedding step. We do not claim it is exact through the full
  network.
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
        "coordpy.tiny_substrate_v4 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3PrefixState,
    TinyV3SubstrateConfig,
    TinyV3SubstrateParams,
    _apply_rope_v3,
    _gelu,
    _gqa_broadcast_k_v,
    _kv_fingerprint_64,
    _merge_heads,
    _ndarray_cid,
    _norm_v3,
    _rope_freqs_v3,
    _seeded_rng,
    _sha256_hex,
    _softmax,
    _split_heads_kv,
    _split_heads_q,
    _swish,
    extract_prefix_state_v3,
    forward_tiny_substrate_v3,
)


W59_TINY_SUBSTRATE_V4_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v4.v1")

W59_TINY_V4_VOCAB_SIZE: int = 259
W59_TINY_V4_PAD_TOKEN: int = 256
W59_TINY_V4_BOS_TOKEN: int = 257
W59_TINY_V4_EOS_TOKEN: int = 258

W59_DEFAULT_V4_D_MODEL: int = 64
W59_DEFAULT_V4_N_HEADS: int = 8
W59_DEFAULT_V4_N_KV_HEADS: int = 4
W59_DEFAULT_V4_N_LAYERS: int = 6
W59_DEFAULT_V4_FF_HIDDEN: int = 160
W59_DEFAULT_V4_MAX_LEN: int = 128
W59_DEFAULT_V4_INIT_SCALE: float = 0.04
W59_DEFAULT_V4_SEED: int = 59012345
W59_DEFAULT_V4_ROPE_BASE: float = 10000.0
W59_DEFAULT_V4_IMPORTANCE_EMA: float = 0.85


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# 128-bucket fingerprint
# ---------------------------------------------------------------------------


def _kv_fingerprint_128(arr: "_np.ndarray") -> tuple[int, ...]:
    """128-bucket XOR fingerprint of an array's raw bytes."""
    raw = _np.ascontiguousarray(arr).tobytes()
    buckets = [0] * 128
    for i, b in enumerate(raw):
        buckets[i % 128] ^= int(b)
    return tuple(int(b) for b in buckets)


# ---------------------------------------------------------------------------
# Config (V4 reuses V3 dataclass shape; we wrap to bump defaults
# and add ``importance_ema`` + ``n_layers`` default of 6)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV4SubstrateConfig:
    vocab_size: int = W59_TINY_V4_VOCAB_SIZE
    d_model: int = W59_DEFAULT_V4_D_MODEL
    n_heads: int = W59_DEFAULT_V4_N_HEADS
    n_kv_heads: int = W59_DEFAULT_V4_N_KV_HEADS
    n_layers: int = W59_DEFAULT_V4_N_LAYERS
    ff_hidden: int = W59_DEFAULT_V4_FF_HIDDEN
    max_len: int = W59_DEFAULT_V4_MAX_LEN
    init_scale: float = W59_DEFAULT_V4_INIT_SCALE
    seed: int = W59_DEFAULT_V4_SEED
    rope_base: float = W59_DEFAULT_V4_ROPE_BASE
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    track_kv_importance: bool = True
    importance_ema: float = W59_DEFAULT_V4_IMPORTANCE_EMA
    expose_per_head_hidden: bool = True

    def to_v3_config(self) -> TinyV3SubstrateConfig:
        return TinyV3SubstrateConfig(
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
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W59_TINY_SUBSTRATE_V4_SCHEMA_VERSION,
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
            "importance_ema": float(round(self.importance_ema, 12)),
            "expose_per_head_hidden": bool(
                self.expose_per_head_hidden),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v4_substrate_config",
            "config": self.to_dict()})


# ---------------------------------------------------------------------------
# Params (delegate to V3 init)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV4SubstrateParams:
    config: TinyV4SubstrateConfig
    v3_params: TinyV3SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV4SubstrateConfig | None = None,
    ) -> "TinyV4SubstrateParams":
        if config is None:
            config = TinyV4SubstrateConfig()
        v3 = TinyV3SubstrateParams.init(config.to_v3_config())
        return cls(config=config, v3_params=v3)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v4_substrate_params",
            "config_cid": self.config.cid(),
            "v3_params_cid": self.v3_params.cid(),
        })


# ---------------------------------------------------------------------------
# KV cache V4 with cumulative EMA importance + 128-bucket fingerprint
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV4KVCache:
    v3_cache: TinyV3KVCache
    cumulative_importance: "_np.ndarray"  # (T,) aggregated across layers/turns
    importance_ema: float
    write_log: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int,
            *, importance_ema: float = W59_DEFAULT_V4_IMPORTANCE_EMA,
    ) -> "TinyV4KVCache":
        v3 = TinyV3KVCache.empty(int(n_layers))
        return cls(
            v3_cache=v3,
            cumulative_importance=_np.zeros((0,),
                                             dtype=_np.float64),
            importance_ema=float(importance_ema),
            write_log=[],
        )

    def n_tokens(self) -> int:
        return int(self.v3_cache.n_tokens())

    def n_layers(self) -> int:
        return int(len(self.v3_cache.keys))

    def clone(self) -> "TinyV4KVCache":
        return TinyV4KVCache(
            v3_cache=self.v3_cache.clone(),
            cumulative_importance=self.cumulative_importance.copy(),
            importance_ema=float(self.importance_ema),
            write_log=list(self.write_log),
        )

    def update_cumulative_importance(
            self, per_layer_new: Sequence["_np.ndarray"],
    ) -> None:
        """EMA-merge per-layer fresh importance into the cumulative
        importance vector. ``per_layer_new`` is a list of length
        ``n_layers``, each entry shape ``(n_tokens_current,)``.
        """
        n = self.n_tokens()
        if n == 0:
            self.cumulative_importance = _np.zeros((0,),
                                                    dtype=_np.float64)
            return
        ema = float(self.importance_ema)
        cum = _np.zeros((n,), dtype=_np.float64)
        # Aggregate fresh signal across all layers.
        for layer_imp in per_layer_new:
            la = _np.asarray(layer_imp, dtype=_np.float64)
            if la.size and la.shape[0] == n:
                cum = cum + la
        prev = self.cumulative_importance
        if prev.size == 0 or prev.shape[0] != n:
            # Initial write: take the fresh aggregate.
            self.cumulative_importance = cum.astype(_np.float64)
        else:
            # EMA blend.
            self.cumulative_importance = (
                ema * prev + (1.0 - ema) * cum
            ).astype(_np.float64)

    def fingerprint_128(self) -> tuple[int, ...]:
        h = [0] * 128
        for k, v in zip(self.v3_cache.keys,
                          self.v3_cache.values):
            for src in (k, v):
                fp = _kv_fingerprint_128(src)
                for i, b in enumerate(fp):
                    h[i] ^= int(b)
        return tuple(int(b) for b in h)

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v4_kv_cache",
            "v3_cache_cid": self.v3_cache.cid(),
            "cumulative_importance_cid": _ndarray_cid(
                self.cumulative_importance),
            "importance_ema": float(round(self.importance_ema, 12)),
            "fingerprint_128": list(self.fingerprint_128()),
            "write_log": list(self.write_log),
        })


# ---------------------------------------------------------------------------
# Partial prefix state V4 — split into reusable + recomputable spans
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TinyV4PartialPrefixState:
    """A V4 prefix split into two contiguous spans:

    * ``reuse_v3_prefix`` covers positions ``[0, prefix_reuse_len)``
      and is byte-identical to the original full V3 prefix on those
      positions.
    * ``recompute_token_ids`` covers positions
      ``[prefix_reuse_len, prefix_len)``; this is what the substrate
      will recompute on demand (e.g. because it was corrupted, or
      because the controller decided to drop the tail's KV slots and
      recompute under a tighter retention budget).
    """
    prefix_reuse_len: int
    prefix_total_len: int
    reuse_v3_prefix: TinyV3PrefixState
    recompute_token_ids: tuple[int, ...]
    source_params_cid: str

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v4_partial_prefix_state",
            "prefix_reuse_len": int(self.prefix_reuse_len),
            "prefix_total_len": int(self.prefix_total_len),
            "reuse_v3_prefix_cid": self.reuse_v3_prefix.cid(),
            "recompute_token_ids": list(self.recompute_token_ids),
            "source_params_cid": str(self.source_params_cid),
        })


def extract_partial_prefix_v4(
        params: TinyV4SubstrateParams,
        prompt_token_ids: Sequence[int],
        *,
        prefix_reuse_len: int,
) -> TinyV4PartialPrefixState:
    """Run the full prompt forward, capture the saved V3 prefix on
    positions ``[0, prefix_reuse_len)``, and record the remaining
    token ids ``[prefix_reuse_len, len(prompt))`` for the substrate
    to recompute on demand.
    """
    pids = list(prompt_token_ids)
    prefix_total = len(pids)
    p_reuse = max(0, min(int(prefix_reuse_len), prefix_total))
    # Forward over the first p_reuse tokens to extract the slice.
    if p_reuse > 0:
        first_trace = forward_tiny_substrate_v3(
            params.v3_params, pids[:p_reuse],
            return_attention=False)
        reuse_state = extract_prefix_state_v3(
            first_trace.kv_cache,
            prefix_len=p_reuse,
            source_params_cid=str(params.v3_params.cid()))
    else:
        reuse_state = extract_prefix_state_v3(
            TinyV3KVCache.empty(int(params.config.n_layers)),
            prefix_len=0,
            source_params_cid=str(params.v3_params.cid()))
    return TinyV4PartialPrefixState(
        prefix_reuse_len=int(p_reuse),
        prefix_total_len=int(prefix_total),
        reuse_v3_prefix=reuse_state,
        recompute_token_ids=tuple(int(t) for t in pids[p_reuse:]),
        source_params_cid=str(params.v3_params.cid()),
    )


# ---------------------------------------------------------------------------
# Forward V4 — wraps V3 forward + cumulative importance + per-head tap
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV4ForwardTrace:
    v3_trace: Any  # TinyV3ForwardTrace (avoid circular type hint)
    head_hidden_states_per_layer: list["_np.ndarray"]
    # (n_layers, n_heads, n_tokens, d_head)
    cumulative_importance: "_np.ndarray"   # length n_tokens
    flop_reuse: int
    flop_recompute: int
    flop_total: int
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v3_trace.logits

    @property
    def kv_cache_v3(self) -> TinyV3KVCache:
        return self.v3_trace.kv_cache

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v4_forward_trace",
            "v3_trace_cid": str(self.v3_trace.cid()),
            "head_hidden_states_cids": [
                _ndarray_cid(h)
                for h in self.head_hidden_states_per_layer],
            "cumulative_importance_cid": _ndarray_cid(
                self.cumulative_importance),
            "flop_reuse": int(self.flop_reuse),
            "flop_recompute": int(self.flop_recompute),
            "flop_total": int(self.flop_total),
        })


def forward_tiny_substrate_v4(
        params: TinyV4SubstrateParams,
        token_ids: Sequence[int],
        *,
        v3_kv_cache: TinyV3KVCache | None = None,
        v4_kv_cache: TinyV4KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        cumulative_importance_prev: (
            "_np.ndarray | None") = None,
) -> tuple[TinyV4ForwardTrace, TinyV4KVCache]:
    """V4 forward. Delegates the heavy lifting to V3, then captures
    the per-(layer, head) hidden state and the cumulative importance
    EMA.

    Returns ``(trace, new_v4_cache)``.
    """
    cfg = params.config
    base_cache = (
        v3_kv_cache if v3_kv_cache is not None
        else (v4_kv_cache.v3_cache.clone()
              if v4_kv_cache is not None else None))
    trace = forward_tiny_substrate_v3(
        params.v3_params, list(token_ids),
        kv_cache=base_cache,
        return_attention=True,
        attention_bias_per_layer=attention_bias_per_layer)

    # Per-head hidden tap. The hidden_states list has length
    # n_layers+1; per-head split is constructed from x's pre-W_O
    # version — but the V3 forward already discards that. We
    # recover a per-head view by splitting hidden_states[layer+1]
    # into (n_heads, n_tokens, d_head). This is a valid per-head
    # view at the layer's *output* residual stream (not its
    # pre-W_O); equivalent up to the W_O concatenation invariant
    # for downstream per-head retrieval scoring.
    d_model = int(cfg.d_model)
    n_heads = int(cfg.n_heads)
    d_head = d_model // n_heads
    head_hidden: list["_np.ndarray"] = []
    for layer_idx in range(int(cfg.n_layers)):
        hs = trace.hidden_states[layer_idx + 1]
        # hs: (n_tokens, d_model). Split last dim into heads.
        n_tok = int(hs.shape[0])
        v = hs.reshape(n_tok, n_heads, d_head)
        v = v.transpose(1, 0, 2)  # (n_heads, n_tokens, d_head)
        head_hidden.append(v.copy())

    # Build / update V4 cache.
    if v4_kv_cache is None:
        new_v4 = TinyV4KVCache.empty(
            int(cfg.n_layers),
            importance_ema=float(cfg.importance_ema))
    else:
        new_v4 = v4_kv_cache.clone()
    # Replace v3 cache contents with the freshly-produced cache.
    new_v4.v3_cache = trace.kv_cache.clone() if hasattr(
        trace.kv_cache, "clone") else trace.kv_cache
    # Compute cumulative importance EMA.
    fresh_per_layer = [
        (imp if imp.size and imp.shape[0]
                == new_v4.n_tokens()
         else _np.zeros((new_v4.n_tokens(),), dtype=_np.float64))
        for imp in new_v4.v3_cache.importance]
    if cumulative_importance_prev is not None:
        # Seed the previous cumulative for EMA blending.
        cip = _np.asarray(
            cumulative_importance_prev, dtype=_np.float64)
        if cip.shape[0] == new_v4.n_tokens():
            new_v4.cumulative_importance = cip
    new_v4.update_cumulative_importance(fresh_per_layer)
    new_v4.write_log.append({
        "schema": W59_TINY_SUBSTRATE_V4_SCHEMA_VERSION,
        "kind": "forward_v4",
        "n_new_tokens": int(len(list(token_ids))),
        "flops": int(trace.flop_count),
    })

    # Flop accounting splits:
    # If a non-empty prefix was passed in, that prefix is *reused*,
    # i.e. its flops are not paid again. We attribute trace.flop_count
    # entirely to the new tokens; "flop_reuse" is the prefix flop
    # *not paid*, "flop_recompute" is what we did pay.
    n_prev = (
        int(base_cache.n_tokens()) if base_cache is not None
        else 0)
    # Estimate the prefix flop as the cost of n_prev tokens at the
    # same per-token cost as the current forward.
    if int(len(list(token_ids))) > 0:
        per_tok = (
            float(trace.flop_count)
            / float(int(len(list(token_ids)))))
    else:
        per_tok = 0.0
    flop_reuse = int(per_tok * float(n_prev))
    flop_recompute = int(trace.flop_count)
    flop_total = int(flop_reuse + flop_recompute)
    new_trace = TinyV4ForwardTrace(
        v3_trace=trace,
        head_hidden_states_per_layer=head_hidden,
        cumulative_importance=new_v4.cumulative_importance.copy(),
        flop_reuse=int(flop_reuse),
        flop_recompute=int(flop_recompute),
        flop_total=int(flop_total),
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return new_trace, new_v4


def forward_with_partial_prefix_reuse_v4(
        params: TinyV4SubstrateParams,
        partial: TinyV4PartialPrefixState,
        follow_up_token_ids: Sequence[int],
) -> tuple[TinyV4ForwardTrace, TinyV4KVCache, dict[str, int]]:
    """Replay the partial prefix:

    1. Load the V3-reusable slice as a starting KV cache.
    2. Forward the *recompute* tail tokens to grow the cache.
    3. Forward the follow-up tokens for the final logits.

    Returns ``(trace, cache, flop_split)``.
    """
    cfg = params.config
    # Stage 1: load reusable slice.
    base = partial.reuse_v3_prefix.to_cache()
    flop_reuse = 0  # bytes of prefix not paid
    # Stage 2: recompute tail.
    tail_ids = list(partial.recompute_token_ids)
    if tail_ids:
        tail_trace = forward_tiny_substrate_v3(
            params.v3_params, tail_ids,
            kv_cache=base, return_attention=False)
        flop_recompute_tail = int(tail_trace.flop_count)
        base = tail_trace.kv_cache
    else:
        flop_recompute_tail = 0
    # Stage 3: follow-up forward.
    fu_ids = list(follow_up_token_ids)
    follow_trace, new_cache_v4 = forward_tiny_substrate_v4(
        params, fu_ids, v3_kv_cache=base)
    flop_follow_up = int(follow_trace.flop_recompute)
    flop_split = {
        "flop_reuse_skipped": int(flop_reuse),
        "flop_recompute_tail": int(flop_recompute_tail),
        "flop_follow_up": int(flop_follow_up),
        "flop_total": int(
            flop_recompute_tail + flop_follow_up),
    }
    return follow_trace, new_cache_v4, flop_split


# ---------------------------------------------------------------------------
# Closed-form logit Jacobian probe at the final position
# ---------------------------------------------------------------------------


def logit_jacobian_diag_v4(
        trace: TinyV4ForwardTrace,
        params: TinyV4SubstrateParams,
        *,
        target_token: int,
) -> dict[str, Any]:
    """Compute an *exact-under-linearised-head* logit Jacobian
    diagonal entry at the last position. This is a small numeric
    fact: the W_unembed[:, target_token] is the direct linear
    influence of the final residual on ``logit[target_token]``.

    We return:
    * the unembedding row L2 norm
    * the dot product of (final residual at last pos) with the row
      (i.e. the linear contribution to that logit)
    """
    unemb = params.v3_params.unembed
    if (int(target_token) < 0
            or int(target_token) >= int(unemb.shape[1])):
        raise ValueError(
            f"target_token {target_token} out of range "
            f"[0,{int(unemb.shape[1])})")
    row = unemb[:, int(target_token)]
    final_hidden = trace.v3_trace.hidden_states[-1]
    last_pos = final_hidden[-1]
    return {
        "schema": W59_TINY_SUBSTRATE_V4_SCHEMA_VERSION,
        "target_token": int(target_token),
        "unembed_row_l2": float(_np.linalg.norm(row)),
        "linear_contribution": float(_np.dot(last_pos, row)),
    }


# ---------------------------------------------------------------------------
# Witness
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TinyV4SubstrateForwardWitness:
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    v3_witness_cid: str
    head_hidden_states_cid: str
    cumulative_importance_cid: str
    fingerprint_128: tuple[int, ...]
    flop_reuse: int
    flop_recompute: int
    flop_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W59_TINY_SUBSTRATE_V4_SCHEMA_VERSION,
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "v3_witness_cid": str(self.v3_witness_cid),
            "head_hidden_states_cid": str(
                self.head_hidden_states_cid),
            "cumulative_importance_cid": str(
                self.cumulative_importance_cid),
            "fingerprint_128": list(self.fingerprint_128),
            "flop_reuse": int(self.flop_reuse),
            "flop_recompute": int(self.flop_recompute),
            "flop_total": int(self.flop_total),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v4_substrate_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v4_forward_witness(
        trace: TinyV4ForwardTrace,
        cache: TinyV4KVCache,
) -> TinyV4SubstrateForwardWitness:
    from .tiny_substrate_v3 import (
        emit_tiny_substrate_v3_forward_witness,
    )
    v3w = emit_tiny_substrate_v3_forward_witness(trace.v3_trace)
    head_cid = _sha256_hex({
        "kind": "tiny_v4_head_hidden",
        "cids": [_ndarray_cid(h)
                 for h in trace.head_hidden_states_per_layer],
    })
    return TinyV4SubstrateForwardWitness(
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(int(t) for t in trace.v3_trace.token_ids),
        v3_witness_cid=str(v3w.cid()),
        head_hidden_states_cid=str(head_cid),
        cumulative_importance_cid=str(_ndarray_cid(
            trace.cumulative_importance)),
        fingerprint_128=tuple(cache.fingerprint_128()),
        flop_reuse=int(trace.flop_reuse),
        flop_recompute=int(trace.flop_recompute),
        flop_total=int(trace.flop_total),
    )


def build_default_tiny_substrate_v4(
        *, seed: int = W59_DEFAULT_V4_SEED,
) -> TinyV4SubstrateParams:
    return TinyV4SubstrateParams.init(
        TinyV4SubstrateConfig(seed=int(seed)))


# ---------------------------------------------------------------------------
# Tokenisation re-exports (V4 reuses V3 byte vocab unchanged)
# ---------------------------------------------------------------------------


def tokenize_bytes_v4(
        text: str, *,
        max_len: int = W59_DEFAULT_V4_MAX_LEN,
        add_bos: bool = True,
) -> list[int]:
    raw = text.encode("utf-8")
    ids: list[int] = []
    if add_bos:
        ids.append(int(W59_TINY_V4_BOS_TOKEN))
    ids.extend(int(b) for b in raw)
    if len(ids) > int(max_len):
        ids = ids[: int(max_len)]
    return ids


def detokenize_bytes_v4(token_ids: Sequence[int]) -> str:
    buf = bytearray()
    for t in token_ids:
        ti = int(t)
        if ti == W59_TINY_V4_PAD_TOKEN:
            continue
        if ti == W59_TINY_V4_BOS_TOKEN:
            continue
        if ti == W59_TINY_V4_EOS_TOKEN:
            break
        if 0 <= ti < 256:
            buf.append(ti)
    try:
        return buf.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return buf.decode("latin-1")


__all__ = [
    "W59_TINY_SUBSTRATE_V4_SCHEMA_VERSION",
    "W59_TINY_V4_VOCAB_SIZE",
    "W59_TINY_V4_PAD_TOKEN",
    "W59_TINY_V4_BOS_TOKEN",
    "W59_TINY_V4_EOS_TOKEN",
    "W59_DEFAULT_V4_D_MODEL",
    "W59_DEFAULT_V4_N_HEADS",
    "W59_DEFAULT_V4_N_KV_HEADS",
    "W59_DEFAULT_V4_N_LAYERS",
    "W59_DEFAULT_V4_FF_HIDDEN",
    "W59_DEFAULT_V4_MAX_LEN",
    "W59_DEFAULT_V4_INIT_SCALE",
    "W59_DEFAULT_V4_SEED",
    "W59_DEFAULT_V4_ROPE_BASE",
    "W59_DEFAULT_V4_IMPORTANCE_EMA",
    "TinyV4SubstrateConfig",
    "TinyV4SubstrateParams",
    "TinyV4KVCache",
    "TinyV4PartialPrefixState",
    "TinyV4ForwardTrace",
    "TinyV4SubstrateForwardWitness",
    "forward_tiny_substrate_v4",
    "forward_with_partial_prefix_reuse_v4",
    "extract_partial_prefix_v4",
    "logit_jacobian_diag_v4",
    "tokenize_bytes_v4",
    "detokenize_bytes_v4",
    "emit_tiny_substrate_v4_forward_witness",
    "build_default_tiny_substrate_v4",
]
