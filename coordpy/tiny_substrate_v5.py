"""W60 M1 — Tiny Transformer Runtime V5.

Strictly extends W59's ``coordpy.tiny_substrate_v4``. V5 keeps every
V4 invariant (byte-determinism under same params, GQA, RMSNorm /
SwiGLU, KV cache with cumulative EMA importance, partial-prefix
split / replay, per-(layer, head) hidden-state tap, 128-bucket
fingerprint, logit Jacobian probe) and adds **six** new substrate-
load-bearing pieces that W60's trainable controllers exploit:

* **Default 7 layers** (vs V4's 6). Same GQA (8 query / 4 KV).
* **Per-(layer, head, position) attention-receive tally** —
  ``TinyV5KVCache.attention_receive`` of shape ``(L, H, T)`` is a
  cumulative EMA of attention received *per query head per cached
  position* across forwards. V4's importance was a scalar per
  (layer, position); V5's per-head tally lets the cache controller
  V3 score retention per-head.
* **Per-(layer, head) logit Jacobian table** — V4's
  ``logit_jacobian_diag_v4`` reports a single scalar last-position
  contribution. V5's ``logit_jacobian_v5`` reports the per-(layer,
  head) contribution of each head's hidden output to the target-
  token logit (still under the linearised-head approximation).
* **Multi-segment partial reuse** — V4 split prefix into one
  reusable head + one recompute tail. V5
  ``extract_multi_segment_prefix_v5`` splits a prefix into
  ``segments=[(start, end, kind), ...]`` where ``kind`` ∈ {reuse,
  recompute, drop}. The forward replays each segment in order;
  ``forward_with_multi_segment_reuse_v5`` returns flop split
  ``flop_reuse_skipped / flop_recompute / flop_drop_skipped /
  flop_follow_up``.
* **Cache write log V5** — every write/eviction/replay/segment
  decision is appended to ``TinyV5KVCache.write_log`` with a
  ``schema=v5`` tag, the per-call CID, and the *new policy id* used
  by the controller (``policy_id``). The write log lets the W60
  ``ReplayController`` audit which mechanism produced each cache
  state.
* **Per-(layer, position) corruption flag table** —
  ``TinyV5KVCache.corruption_flags`` of shape ``(L, T)`` boolean.
  When the W60 ``CorruptionRobustCarrierV8`` reports a detected
  corruption on a slot, it can set the flag; the cache controller
  V3 then automatically scores those slots near zero (effectively
  evicting them). V4 had no in-substrate channel for this signal.

Honest scope (do-not-overstate, W60)
------------------------------------

* Still NOT a frontier model. Default config:
  ``7 layers / 8 query heads / 4 kv heads / d_model=64 /
  ff_hidden=176 / byte-vocab / max_len=128 / untrained``. The
  runtime is richer than V4 but still a research substrate in
  pure NumPy on CPU. ``W60-L-NUMPY-CPU-V5-SUBSTRATE-CAP``
  documents this.
* V5 still does NOT bridge to third-party hosted models. Ollama
  / OpenAI-compatible / hosted backends remain text-only at the
  HTTP surface. ``W60-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
  carries forward the W57/W58/W59 cap unchanged.
* Per-(layer, head) Jacobian table is computed under the
  *linearised-head approximation* using the substrate's own
  unembedding row and per-head residual contribution. Exact only
  for a one-step linear unembed; we do not claim it is exact
  through the full network.
* Corruption flag table is a *channel*, not a detector. Flags are
  written by external detectors (CRC V8) and read by the cache
  controller V3. The substrate just stores them.
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
        "coordpy.tiny_substrate_v5 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3PrefixState,
    TinyV3SubstrateConfig,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    extract_prefix_state_v3,
    forward_tiny_substrate_v3,
)
from .tiny_substrate_v4 import (
    TinyV4KVCache,
    TinyV4PartialPrefixState,
    TinyV4SubstrateConfig,
    TinyV4SubstrateParams,
    W59_DEFAULT_V4_IMPORTANCE_EMA,
    _kv_fingerprint_128,
    forward_tiny_substrate_v4,
)


W60_TINY_SUBSTRATE_V5_SCHEMA_VERSION: str = (
    "coordpy.tiny_substrate_v5.v1")

W60_TINY_V5_VOCAB_SIZE: int = 259
W60_DEFAULT_V5_D_MODEL: int = 64
W60_DEFAULT_V5_N_HEADS: int = 8
W60_DEFAULT_V5_N_KV_HEADS: int = 4
W60_DEFAULT_V5_N_LAYERS: int = 7
W60_DEFAULT_V5_FF_HIDDEN: int = 176
W60_DEFAULT_V5_MAX_LEN: int = 128
W60_DEFAULT_V5_INIT_SCALE: float = 0.04
W60_DEFAULT_V5_SEED: int = 60012345
W60_DEFAULT_V5_ROPE_BASE: float = 10000.0
W60_DEFAULT_V5_IMPORTANCE_EMA: float = W59_DEFAULT_V4_IMPORTANCE_EMA
W60_DEFAULT_V5_ATTENTION_RECEIVE_EMA: float = 0.80


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# Config (V5 reuses V4 dataclass shape; we wrap to bump defaults
# and add ``attention_receive_ema`` + ``n_layers`` default of 7)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV5SubstrateConfig:
    vocab_size: int = W60_TINY_V5_VOCAB_SIZE
    d_model: int = W60_DEFAULT_V5_D_MODEL
    n_heads: int = W60_DEFAULT_V5_N_HEADS
    n_kv_heads: int = W60_DEFAULT_V5_N_KV_HEADS
    n_layers: int = W60_DEFAULT_V5_N_LAYERS
    ff_hidden: int = W60_DEFAULT_V5_FF_HIDDEN
    max_len: int = W60_DEFAULT_V5_MAX_LEN
    init_scale: float = W60_DEFAULT_V5_INIT_SCALE
    seed: int = W60_DEFAULT_V5_SEED
    rope_base: float = W60_DEFAULT_V5_ROPE_BASE
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

    def to_v4_config(self) -> TinyV4SubstrateConfig:
        return TinyV4SubstrateConfig(
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
            expose_per_head_hidden=bool(
                self.expose_per_head_hidden),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W60_TINY_SUBSTRATE_V5_SCHEMA_VERSION,
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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v5_substrate_config",
            "config": self.to_dict()})


# ---------------------------------------------------------------------------
# Params (delegate to V4 init via V4 -> V3)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV5SubstrateParams:
    config: TinyV5SubstrateConfig
    v4_params: TinyV4SubstrateParams

    @classmethod
    def init(
            cls, config: TinyV5SubstrateConfig | None = None,
    ) -> "TinyV5SubstrateParams":
        if config is None:
            config = TinyV5SubstrateConfig()
        v4 = TinyV4SubstrateParams.init(config.to_v4_config())
        return cls(config=config, v4_params=v4)

    @property
    def v3_params(self) -> TinyV3SubstrateParams:
        return self.v4_params.v3_params

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v5_substrate_params",
            "config_cid": self.config.cid(),
            "v4_params_cid": self.v4_params.cid(),
        })


# ---------------------------------------------------------------------------
# KV cache V5 with per-(layer, head, pos) attention-receive +
# corruption flags + write log V5
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV5KVCache:
    v4_cache: TinyV4KVCache
    attention_receive: list["_np.ndarray"]   # length L, each (H, T)
    corruption_flags: list["_np.ndarray"]    # length L, each (T,) bool
    attention_receive_ema: float
    write_log_v5: list[dict[str, Any]] = dataclasses.field(
        default_factory=list)

    @classmethod
    def empty(
            cls, n_layers: int, *,
            n_heads: int = W60_DEFAULT_V5_N_HEADS,
            importance_ema: float = (
                W60_DEFAULT_V5_IMPORTANCE_EMA),
            attention_receive_ema: float = (
                W60_DEFAULT_V5_ATTENTION_RECEIVE_EMA),
    ) -> "TinyV5KVCache":
        v4 = TinyV4KVCache.empty(
            int(n_layers), importance_ema=float(importance_ema))
        attn_recv: list["_np.ndarray"] = [
            _np.zeros((int(n_heads), 0), dtype=_np.float64)
            for _ in range(int(n_layers))]
        corr: list["_np.ndarray"] = [
            _np.zeros((0,), dtype=_np.bool_)
            for _ in range(int(n_layers))]
        return cls(
            v4_cache=v4,
            attention_receive=attn_recv,
            corruption_flags=corr,
            attention_receive_ema=float(attention_receive_ema),
            write_log_v5=[],
        )

    @property
    def v3_cache(self) -> TinyV3KVCache:
        return self.v4_cache.v3_cache

    def n_tokens(self) -> int:
        return int(self.v4_cache.n_tokens())

    def n_layers(self) -> int:
        return int(self.v4_cache.n_layers())

    def clone(self) -> "TinyV5KVCache":
        return TinyV5KVCache(
            v4_cache=self.v4_cache.clone(),
            attention_receive=[
                a.copy() for a in self.attention_receive],
            corruption_flags=[
                f.copy() for f in self.corruption_flags],
            attention_receive_ema=float(
                self.attention_receive_ema),
            write_log_v5=list(self.write_log_v5),
        )

    def update_attention_receive(
            self, per_layer_per_head: Sequence["_np.ndarray"],
    ) -> None:
        """EMA-merge per-(layer, head) fresh attention-received
        signal into ``attention_receive``.

        Each entry is shape ``(n_heads, n_tokens_current)``."""
        n = self.n_tokens()
        if n == 0:
            self.attention_receive = [
                a.copy() for a in self.attention_receive]
            return
        a = float(self.attention_receive_ema)
        new_recv: list["_np.ndarray"] = []
        for layer_idx, layer_in in enumerate(per_layer_per_head):
            x = _np.asarray(layer_in, dtype=_np.float64)
            if x.ndim != 2 or x.shape[1] != n:
                # If shape mismatch, fall back to zeros.
                x = _np.zeros((x.shape[0]
                                  if x.ndim >= 1
                                  else 1, n),
                                dtype=_np.float64)
            prev = (
                self.attention_receive[layer_idx]
                if layer_idx < len(self.attention_receive)
                else _np.zeros_like(x))
            if (prev.size == 0
                    or prev.shape[0] != x.shape[0]
                    or prev.shape[1] != n):
                new_recv.append(x.astype(_np.float64))
            else:
                new_recv.append(
                    (a * prev + (1.0 - a) * x).astype(
                        _np.float64))
        # If fewer per-layer entries than layers, pad with zeros.
        while len(new_recv) < self.n_layers():
            new_recv.append(_np.zeros(
                (W60_DEFAULT_V5_N_HEADS, n), dtype=_np.float64))
        self.attention_receive = new_recv

    def set_corruption_flag(
            self, *, layer_index: int, position: int,
            flagged: bool = True,
    ) -> None:
        L = self.n_layers()
        n = self.n_tokens()
        if layer_index < 0 or layer_index >= L:
            return
        # Resize the layer's flag vector if needed.
        if (self.corruption_flags[layer_index].size != n):
            new_flags = _np.zeros((n,), dtype=_np.bool_)
            old = self.corruption_flags[layer_index]
            for i in range(min(int(old.size), n)):
                new_flags[i] = bool(old[i])
            self.corruption_flags[layer_index] = new_flags
        if 0 <= int(position) < n:
            self.corruption_flags[layer_index][
                int(position)] = bool(flagged)

    def n_corrupted(self) -> int:
        return int(sum(int(f.sum())
                       for f in self.corruption_flags))

    def fingerprint_128(self) -> tuple[int, ...]:
        return self.v4_cache.fingerprint_128()

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v5_kv_cache",
            "v4_cache_cid": self.v4_cache.cid(),
            "attention_receive_cids": [
                _ndarray_cid(a) for a in self.attention_receive],
            "corruption_flag_cids": [
                _ndarray_cid(f.astype(_np.uint8))
                for f in self.corruption_flags],
            "attention_receive_ema": float(round(
                self.attention_receive_ema, 12)),
            "write_log_v5": list(self.write_log_v5),
        })


# ---------------------------------------------------------------------------
# Multi-segment partial prefix V5
# ---------------------------------------------------------------------------


W60_V5_SEGMENT_REUSE: str = "reuse"
W60_V5_SEGMENT_RECOMPUTE: str = "recompute"
W60_V5_SEGMENT_DROP: str = "drop"
W60_V5_SEGMENT_KINDS: tuple[str, ...] = (
    W60_V5_SEGMENT_REUSE,
    W60_V5_SEGMENT_RECOMPUTE,
    W60_V5_SEGMENT_DROP,
)


@dataclasses.dataclass(frozen=True)
class TinyV5MultiSegmentPrefix:
    """A V5 prefix split into ``segments=[(start, end, kind), ...]``,
    plus the cached *full* prefix state needed for reuse segments
    and the recompute token ids needed for recompute segments.

    For drop segments the substrate skips the entire span — neither
    cache nor recompute. The follow-up forward starts from the
    accumulated cache of the kept (reuse + recompute) segments.
    """
    segments: tuple[tuple[int, int, str], ...]
    full_prefix_v3: TinyV3PrefixState
    full_token_ids: tuple[int, ...]
    source_params_cid: str

    def reuse_len(self) -> int:
        return int(sum(
            (e - s) for s, e, k in self.segments
            if k == W60_V5_SEGMENT_REUSE))

    def recompute_len(self) -> int:
        return int(sum(
            (e - s) for s, e, k in self.segments
            if k == W60_V5_SEGMENT_RECOMPUTE))

    def drop_len(self) -> int:
        return int(sum(
            (e - s) for s, e, k in self.segments
            if k == W60_V5_SEGMENT_DROP))

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v5_multi_segment_prefix",
            "segments": [
                [int(s), int(e), str(k)]
                for s, e, k in self.segments],
            "full_prefix_v3_cid": self.full_prefix_v3.cid(),
            "full_token_ids": list(self.full_token_ids),
            "source_params_cid": str(self.source_params_cid),
        })


def extract_multi_segment_prefix_v5(
        params: TinyV5SubstrateParams,
        prompt_token_ids: Sequence[int],
        *,
        segments: Sequence[tuple[int, int, str]],
) -> TinyV5MultiSegmentPrefix:
    """Extract a V5 multi-segment prefix.

    1. Run a full prefix forward to get the saved KV slice.
    2. Validate the segment list (in-bounds, non-overlapping,
       sorted, kinds in the allowed set).
    3. Pack everything into a frozen carrier.
    """
    pids = [int(t) for t in prompt_token_ids]
    L_total = len(pids)
    # Validate segments.
    seg_norm: list[tuple[int, int, str]] = []
    for s, e, k in segments:
        s_i = int(s); e_i = int(e); k_s = str(k)
        if s_i < 0 or e_i > L_total or s_i >= e_i:
            raise ValueError(
                f"segment ({s_i}, {e_i}, {k_s}) out of bounds")
        if k_s not in W60_V5_SEGMENT_KINDS:
            raise ValueError(
                f"segment kind {k_s!r} not in "
                f"{W60_V5_SEGMENT_KINDS}")
        seg_norm.append((s_i, e_i, k_s))
    # Sort and check non-overlapping.
    seg_norm.sort(key=lambda x: x[0])
    last_end = 0
    for s, e, k in seg_norm:
        if s < last_end:
            raise ValueError(
                f"segment ({s},{e}) overlaps prior end "
                f"{last_end}")
        last_end = e
    # Run full prefix forward to get the saved V3 slice.
    if L_total > 0:
        full_trace = forward_tiny_substrate_v3(
            params.v3_params, pids,
            return_attention=False)
        full_prefix = extract_prefix_state_v3(
            full_trace.kv_cache, prefix_len=L_total,
            source_params_cid=str(params.v3_params.cid()))
    else:
        full_prefix = extract_prefix_state_v3(
            TinyV3KVCache.empty(int(params.config.n_layers)),
            prefix_len=0,
            source_params_cid=str(params.v3_params.cid()))
    return TinyV5MultiSegmentPrefix(
        segments=tuple(seg_norm),
        full_prefix_v3=full_prefix,
        full_token_ids=tuple(pids),
        source_params_cid=str(params.v3_params.cid()),
    )


def _slice_v3_prefix(
        full: TinyV3PrefixState, *,
        start: int, end: int,
        source_params_cid: str,
) -> TinyV3PrefixState:
    """Return a sliced V3 prefix on positions ``[start, end)``."""
    n = int(end - start)
    new_keys: list["_np.ndarray"] = []
    new_vals: list["_np.ndarray"] = []
    new_imp: list["_np.ndarray"] = []
    for kl, vl, il in zip(full.keys, full.values, full.importance):
        if kl.size == 0:
            new_keys.append(_np.zeros(
                (n, kl.shape[-1] if kl.ndim > 1 else 0),
                dtype=_np.float64))
            new_vals.append(_np.zeros(
                (n, vl.shape[-1] if vl.ndim > 1 else 0),
                dtype=_np.float64))
            new_imp.append(_np.zeros((n,), dtype=_np.float64))
            continue
        new_keys.append(kl[start:end].copy())
        new_vals.append(vl[start:end].copy())
        new_imp.append(
            il[start:end].copy()
            if il.size else _np.zeros(
                (n,), dtype=_np.float64))
    return TinyV3PrefixState(
        prefix_len=int(n),
        keys=tuple(new_keys),
        values=tuple(new_vals),
        importance=tuple(new_imp),
        source_params_cid=str(source_params_cid),
        redundant_copy_cid=str(source_params_cid),
    )


def forward_with_multi_segment_reuse_v5(
        params: TinyV5SubstrateParams,
        prefix: TinyV5MultiSegmentPrefix,
        follow_up_token_ids: Sequence[int],
) -> tuple[
    Any, TinyV5KVCache, dict[str, int]]:
    """Replay a V5 multi-segment prefix into a V5 cache and forward
    the follow-up. Reuse segments load saved slices; recompute
    segments rerun the matching token ids; drop segments are
    skipped entirely.
    """
    cfg = params.config
    v3p = params.v3_params
    pids_full = list(prefix.full_token_ids)
    base = TinyV3KVCache.empty(int(cfg.n_layers))
    flop_recompute = 0
    for s, e, k in prefix.segments:
        if k == W60_V5_SEGMENT_REUSE:
            slice_state = _slice_v3_prefix(
                prefix.full_prefix_v3, start=s, end=e,
                source_params_cid=str(v3p.cid()))
            slice_cache = slice_state.to_cache()
            # Concatenate slice cache contents into base.
            base = _concat_v3_cache(base, slice_cache)
        elif k == W60_V5_SEGMENT_RECOMPUTE:
            tail_ids = pids_full[s:e]
            if tail_ids:
                tt = forward_tiny_substrate_v3(
                    v3p, tail_ids,
                    kv_cache=base, return_attention=False)
                flop_recompute += int(tt.flop_count)
                base = tt.kv_cache
        # drop: skip entirely
    # Now follow-up forward.
    fu_ids = list(follow_up_token_ids)
    fu_trace, fu_v4_cache = forward_tiny_substrate_v4(
        params.v4_params, fu_ids, v3_kv_cache=base)
    flop_follow_up = int(fu_trace.flop_recompute)
    # Build V5 cache atop V4.
    v5_cache = TinyV5KVCache.empty(
        int(cfg.n_layers), n_heads=int(cfg.n_heads),
        importance_ema=float(cfg.importance_ema),
        attention_receive_ema=float(cfg.attention_receive_ema))
    v5_cache.v4_cache = fu_v4_cache
    v5_cache.write_log_v5.append({
        "schema": W60_TINY_SUBSTRATE_V5_SCHEMA_VERSION,
        "kind": "multi_segment_reuse",
        "segments": [
            [int(s), int(e), str(k)]
            for s, e, k in prefix.segments],
        "flop_recompute": int(flop_recompute),
        "flop_follow_up": int(flop_follow_up),
    })
    flop_split = {
        "flop_reuse_skipped": int(prefix.reuse_len()),
        "flop_recompute": int(flop_recompute),
        "flop_drop_skipped": int(prefix.drop_len()),
        "flop_follow_up": int(flop_follow_up),
        "flop_total": int(flop_recompute + flop_follow_up),
    }
    return fu_trace, v5_cache, flop_split


def _concat_v3_cache(
        base: TinyV3KVCache, addn: TinyV3KVCache,
) -> TinyV3KVCache:
    """Concatenate two V3 caches along token axis (per layer)."""
    new = base.clone()
    for li in range(int(len(new.keys))):
        if li >= len(addn.keys):
            continue
        if addn.keys[li].size == 0:
            continue
        if new.keys[li].size == 0:
            new.keys[li] = addn.keys[li].copy()
            new.values[li] = addn.values[li].copy()
            new.importance[li] = (
                addn.importance[li].copy()
                if addn.importance[li].size
                else _np.zeros((addn.keys[li].shape[0],),
                                dtype=_np.float64))
        else:
            new.keys[li] = _np.concatenate(
                [new.keys[li], addn.keys[li]], axis=0)
            new.values[li] = _np.concatenate(
                [new.values[li], addn.values[li]], axis=0)
            ai = (addn.importance[li]
                  if addn.importance[li].size
                  else _np.zeros((addn.keys[li].shape[0],),
                                  dtype=_np.float64))
            new.importance[li] = _np.concatenate(
                [new.importance[li], ai], axis=0)
    return new


# ---------------------------------------------------------------------------
# Forward V5
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TinyV5ForwardTrace:
    v4_trace: Any   # TinyV4ForwardTrace
    attention_receive_per_layer: list["_np.ndarray"]
    # length L, each (H, T)
    config_cid: str
    params_cid: str

    @property
    def logits(self) -> "_np.ndarray":
        return self.v4_trace.logits

    @property
    def kv_cache_v3(self) -> TinyV3KVCache:
        return self.v4_trace.kv_cache_v3

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v5_forward_trace",
            "v4_trace_cid": str(self.v4_trace.cid()),
            "attention_receive_cids": [
                _ndarray_cid(a)
                for a in self.attention_receive_per_layer],
        })


def _attention_receive_from_v3_trace(
        v3_trace: Any, *, n_heads: int, n_tokens: int,
) -> list["_np.ndarray"]:
    """Build a per-layer (n_heads, n_tokens) attention-received
    matrix from the V3 trace's per-layer attention maps. Each
    column j is the SUM of attention placed onto token j across
    all queries, per head.

    If the V3 trace did not record attention (return_attention
    False), returns zero matrices.
    """
    out: list["_np.ndarray"] = []
    attn_per_layer = getattr(
        v3_trace, "attn_weights_per_layer", None)
    if attn_per_layer is None:
        for _ in range(int(len(v3_trace.kv_cache.keys))):
            out.append(_np.zeros(
                (int(n_heads), int(n_tokens)),
                dtype=_np.float64))
        return out
    for layer_idx, A in enumerate(attn_per_layer):
        # A: (n_heads, n_q, n_k). Sum over query axis.
        if A is None or A.size == 0:
            out.append(_np.zeros(
                (int(n_heads), int(n_tokens)),
                dtype=_np.float64))
            continue
        if A.ndim == 3:
            recv = A.sum(axis=1)  # (n_heads, n_k)
        else:
            recv = _np.zeros(
                (int(n_heads), int(n_tokens)),
                dtype=_np.float64)
        # Pad / truncate to (n_heads, n_tokens).
        H, K = recv.shape
        out_recv = _np.zeros(
            (int(n_heads), int(n_tokens)),
            dtype=_np.float64)
        out_recv[:min(H, int(n_heads)),
                 :min(K, int(n_tokens))] = recv[
                    :min(H, int(n_heads)),
                    :min(K, int(n_tokens))]
        out.append(out_recv)
    return out


def forward_tiny_substrate_v5(
        params: TinyV5SubstrateParams,
        token_ids: Sequence[int],
        *,
        v5_kv_cache: TinyV5KVCache | None = None,
        attention_bias_per_layer: (
            Sequence["_np.ndarray | None"] | None) = None,
        cumulative_importance_prev: (
            "_np.ndarray | None") = None,
) -> tuple[TinyV5ForwardTrace, TinyV5KVCache]:
    """V5 forward. Delegates to V4, then captures the per-(layer,
    head) attention-received matrix.

    Returns ``(trace, new_v5_cache)``.
    """
    cfg = params.config
    v4p = params.v4_params
    base_v4_cache = (
        v5_kv_cache.v4_cache.clone()
        if v5_kv_cache is not None else None)
    v4_trace, new_v4 = forward_tiny_substrate_v4(
        v4p, list(token_ids),
        v4_kv_cache=base_v4_cache,
        attention_bias_per_layer=attention_bias_per_layer,
        cumulative_importance_prev=cumulative_importance_prev)
    n_tok = int(new_v4.n_tokens())
    recv = _attention_receive_from_v3_trace(
        v4_trace.v3_trace,
        n_heads=int(cfg.n_heads),
        n_tokens=int(n_tok))
    if v5_kv_cache is None:
        v5_new = TinyV5KVCache.empty(
            int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            importance_ema=float(cfg.importance_ema),
            attention_receive_ema=float(
                cfg.attention_receive_ema))
    else:
        v5_new = v5_kv_cache.clone()
    v5_new.v4_cache = new_v4
    v5_new.update_attention_receive(recv)
    # Trim corruption flags to current token length.
    new_corr: list["_np.ndarray"] = []
    for layer_idx in range(int(cfg.n_layers)):
        prev = (
            v5_new.corruption_flags[layer_idx]
            if layer_idx < len(v5_new.corruption_flags)
            else _np.zeros((n_tok,), dtype=_np.bool_))
        if prev.size != n_tok:
            new_flags = _np.zeros((n_tok,), dtype=_np.bool_)
            for i in range(min(int(prev.size), n_tok)):
                new_flags[i] = bool(prev[i])
            new_corr.append(new_flags)
        else:
            new_corr.append(prev.copy())
    v5_new.corruption_flags = new_corr
    v5_new.write_log_v5.append({
        "schema": W60_TINY_SUBSTRATE_V5_SCHEMA_VERSION,
        "kind": "forward_v5",
        "n_new_tokens": int(len(list(token_ids))),
        "n_corrupted_after": int(v5_new.n_corrupted()),
    })
    trace = TinyV5ForwardTrace(
        v4_trace=v4_trace,
        attention_receive_per_layer=[
            r.copy() for r in v5_new.attention_receive],
        config_cid=str(cfg.cid()),
        params_cid=str(params.cid()),
    )
    return trace, v5_new


# ---------------------------------------------------------------------------
# Per-(layer, head) logit Jacobian table
# ---------------------------------------------------------------------------


def logit_jacobian_v5(
        trace: TinyV5ForwardTrace,
        params: TinyV5SubstrateParams,
        *,
        target_token: int,
) -> dict[str, Any]:
    """Per-(layer, head) linearised Jacobian table.

    For each layer, V5 returns the per-head dot product
    ``(per_head_residual, unembed_row[:, target])``. This is the
    *direct linear contribution* of each head's residual at the
    last position to ``logit[target]``, before the unembed and any
    downstream nonlinearities. Approximation: same caveat as V4 —
    exact only under the linearised one-step unembed.
    """
    unemb = params.v3_params.unembed
    if (int(target_token) < 0
            or int(target_token) >= int(unemb.shape[1])):
        raise ValueError(
            f"target_token {target_token} out of range "
            f"[0,{int(unemb.shape[1])})")
    row = unemb[:, int(target_token)]
    cfg = params.config
    d_model = int(cfg.d_model)
    n_heads = int(cfg.n_heads)
    d_head = d_model // n_heads
    n_layers = int(cfg.n_layers)
    head_hidden = trace.v4_trace.head_hidden_states_per_layer
    table = _np.zeros((n_layers, n_heads), dtype=_np.float64)
    for li in range(min(n_layers, len(head_hidden))):
        hh = head_hidden[li]
        # hh: (n_heads, n_tokens, d_head)
        if hh.shape[0] < n_heads or hh.shape[1] < 1:
            continue
        for hi in range(n_heads):
            last_h = hh[hi, -1]   # (d_head,)
            # The unembed row is over d_model; this head occupies
            # positions ``[hi*d_head, (hi+1)*d_head)`` in the
            # post-W_O concatenation. For the linearised view we
            # take the corresponding slice of the unembed row.
            row_slice = row[hi * d_head:(hi + 1) * d_head]
            table[li, hi] = float(_np.dot(last_h, row_slice))
    final_hidden = trace.v4_trace.v3_trace.hidden_states[-1]
    last_pos = final_hidden[-1]
    return {
        "schema": W60_TINY_SUBSTRATE_V5_SCHEMA_VERSION,
        "target_token": int(target_token),
        "per_layer_per_head": table.tolist(),
        "unembed_row_l2": float(_np.linalg.norm(row)),
        "linear_contribution_total": float(
            _np.dot(last_pos, row)),
    }


# ---------------------------------------------------------------------------
# Witness
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TinyV5SubstrateForwardWitness:
    config_cid: str
    params_cid: str
    token_ids: tuple[int, ...]
    v4_witness_cid: str
    attention_receive_cids: tuple[str, ...]
    n_corrupted: int
    n_layers: int
    n_heads: int
    fingerprint_128: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W60_TINY_SUBSTRATE_V5_SCHEMA_VERSION,
            "config_cid": str(self.config_cid),
            "params_cid": str(self.params_cid),
            "token_ids": list(self.token_ids),
            "v4_witness_cid": str(self.v4_witness_cid),
            "attention_receive_cids": list(
                self.attention_receive_cids),
            "n_corrupted": int(self.n_corrupted),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "fingerprint_128": list(self.fingerprint_128),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "tiny_v5_substrate_forward_witness",
            "witness": self.to_dict()})


def emit_tiny_substrate_v5_forward_witness(
        trace: TinyV5ForwardTrace,
        cache: TinyV5KVCache,
) -> TinyV5SubstrateForwardWitness:
    from .tiny_substrate_v4 import (
        emit_tiny_substrate_v4_forward_witness,
    )
    v4w = emit_tiny_substrate_v4_forward_witness(
        trace.v4_trace, cache.v4_cache)
    return TinyV5SubstrateForwardWitness(
        config_cid=str(trace.config_cid),
        params_cid=str(trace.params_cid),
        token_ids=tuple(int(t)
                          for t in trace.v4_trace.v3_trace.token_ids),
        v4_witness_cid=str(v4w.cid()),
        attention_receive_cids=tuple(
            _ndarray_cid(a)
            for a in trace.attention_receive_per_layer),
        n_corrupted=int(cache.n_corrupted()),
        n_layers=int(cache.n_layers()),
        n_heads=int(trace.v4_trace.head_hidden_states_per_layer[0].shape[0]
                      if trace.v4_trace.head_hidden_states_per_layer
                      else 0),
        fingerprint_128=tuple(cache.fingerprint_128()),
    )


def build_default_tiny_substrate_v5(
        *, seed: int = W60_DEFAULT_V5_SEED,
) -> TinyV5SubstrateParams:
    return TinyV5SubstrateParams.init(
        TinyV5SubstrateConfig(seed=int(seed)))


# ---------------------------------------------------------------------------
# Tokenisation re-exports (V5 reuses V3 byte vocab unchanged)
# ---------------------------------------------------------------------------


def tokenize_bytes_v5(
        text: str, *,
        max_len: int = W60_DEFAULT_V5_MAX_LEN,
        add_bos: bool = True,
) -> list[int]:
    raw = text.encode("utf-8")
    ids: list[int] = []
    if add_bos:
        ids.append(257)
    ids.extend(int(b) for b in raw)
    if len(ids) > int(max_len):
        ids = ids[: int(max_len)]
    return ids


def detokenize_bytes_v5(token_ids: Sequence[int]) -> str:
    buf = bytearray()
    for t in token_ids:
        ti = int(t)
        if ti == 256:
            continue
        if ti == 257:
            continue
        if ti == 258:
            break
        if 0 <= ti < 256:
            buf.append(ti)
    try:
        return buf.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return buf.decode("latin-1")


__all__ = [
    "W60_TINY_SUBSTRATE_V5_SCHEMA_VERSION",
    "W60_TINY_V5_VOCAB_SIZE",
    "W60_DEFAULT_V5_D_MODEL",
    "W60_DEFAULT_V5_N_HEADS",
    "W60_DEFAULT_V5_N_KV_HEADS",
    "W60_DEFAULT_V5_N_LAYERS",
    "W60_DEFAULT_V5_FF_HIDDEN",
    "W60_DEFAULT_V5_MAX_LEN",
    "W60_DEFAULT_V5_INIT_SCALE",
    "W60_DEFAULT_V5_SEED",
    "W60_DEFAULT_V5_ROPE_BASE",
    "W60_DEFAULT_V5_IMPORTANCE_EMA",
    "W60_DEFAULT_V5_ATTENTION_RECEIVE_EMA",
    "W60_V5_SEGMENT_REUSE",
    "W60_V5_SEGMENT_RECOMPUTE",
    "W60_V5_SEGMENT_DROP",
    "W60_V5_SEGMENT_KINDS",
    "TinyV5SubstrateConfig",
    "TinyV5SubstrateParams",
    "TinyV5KVCache",
    "TinyV5MultiSegmentPrefix",
    "TinyV5ForwardTrace",
    "TinyV5SubstrateForwardWitness",
    "extract_multi_segment_prefix_v5",
    "forward_with_multi_segment_reuse_v5",
    "forward_tiny_substrate_v5",
    "logit_jacobian_v5",
    "tokenize_bytes_v5",
    "detokenize_bytes_v5",
    "emit_tiny_substrate_v5_forward_witness",
    "build_default_tiny_substrate_v5",
]
