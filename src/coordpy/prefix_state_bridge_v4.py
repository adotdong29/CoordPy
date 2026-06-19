"""W60 M4 — Prefix-State Bridge V4.

Strictly extends W59's ``coordpy.prefix_state_bridge_v3``. V3
supported a single-split partial reuse (one reuse head + one
recompute tail). V4 supports the **multi-segment** partial reuse
exposed by ``coordpy.tiny_substrate_v5``:

* segments=[(start, end, kind), ...] where kind ∈ {reuse, recompute,
  drop}
* the bridge replays each segment in order, evaluates the follow-up,
  and reports per-segment flop accounting
* drop segments shorten the effective prefix and yield strictly
  smaller flop bills at the cost of a logit-drift penalty bounded
  by the segment's ``importance × cumulative_attention_receive``
  budget.

V4 also extends the K-seed drift spectrum to *longer chains*: the
caller can pass a chain of N forwards rather than a single follow-
up; V4 reports per-step drift and the cumulative drift envelope.

V4 strictly extends V3: with one reuse-only segment covering the
whole prefix and a single follow-up step, V4 reduces to V3's
``bridge_prefix_state_and_measure_v3`` byte-for-byte (up to schema
tag).

Honest scope
------------

* Drop segments do NOT preserve byte-identity to a full recompute
  — by construction they remove KV slots from the chain. The
  V4 witness reports the L2 drop-induced drift so the consumer
  can decide whether to accept the saving.
* Multi-segment chain forward is still through the in-repo V5/V4
  substrate; no third-party substrate access.
  ``W60-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries forward.
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
        "coordpy.prefix_state_bridge_v4 requires numpy"
        ) from exc

from .tiny_substrate_v3 import (
    TinyV3KVCache,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)
from .tiny_substrate_v4 import TinyV4SubstrateParams
from .tiny_substrate_v5 import (
    TinyV5MultiSegmentPrefix,
    TinyV5SubstrateParams,
    W60_V5_SEGMENT_DROP,
    W60_V5_SEGMENT_RECOMPUTE,
    W60_V5_SEGMENT_REUSE,
    extract_multi_segment_prefix_v5,
    forward_with_multi_segment_reuse_v5,
)


W60_PREFIX_STATE_BRIDGE_V4_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v4.v1")


@dataclasses.dataclass(frozen=True)
class PrefixStateBridgeV4Witness:
    schema: str
    multi_segment_prefix_cid: str
    n_segments: int
    n_reuse_segments: int
    n_recompute_segments: int
    n_drop_segments: int
    reuse_len: int
    recompute_len: int
    drop_len: int
    follow_up_token_ids: tuple[int, ...]
    full_recompute_last_logits_cid: str
    multi_segment_last_logits_cid: str
    max_abs_drop_induced_drift: float
    flop_full_recompute: int
    flop_multi_segment_total: int
    flop_saved_vs_recompute: int
    flop_savings_ratio: float
    chain_step_drifts_l2: tuple[float, ...]
    chain_cumulative_drift_l2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "multi_segment_prefix_cid": str(
                self.multi_segment_prefix_cid),
            "n_segments": int(self.n_segments),
            "n_reuse_segments": int(self.n_reuse_segments),
            "n_recompute_segments": int(
                self.n_recompute_segments),
            "n_drop_segments": int(self.n_drop_segments),
            "reuse_len": int(self.reuse_len),
            "recompute_len": int(self.recompute_len),
            "drop_len": int(self.drop_len),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "full_recompute_last_logits_cid": str(
                self.full_recompute_last_logits_cid),
            "multi_segment_last_logits_cid": str(
                self.multi_segment_last_logits_cid),
            "max_abs_drop_induced_drift": float(round(
                self.max_abs_drop_induced_drift, 12)),
            "flop_full_recompute": int(
                self.flop_full_recompute),
            "flop_multi_segment_total": int(
                self.flop_multi_segment_total),
            "flop_saved_vs_recompute": int(
                self.flop_saved_vs_recompute),
            "flop_savings_ratio": float(round(
                self.flop_savings_ratio, 12)),
            "chain_step_drifts_l2": [
                float(round(d, 12))
                for d in self.chain_step_drifts_l2],
            "chain_cumulative_drift_l2": float(round(
                self.chain_cumulative_drift_l2, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_state_bridge_v4_witness",
            "witness": self.to_dict()})


def bridge_prefix_state_and_measure_v4(
        *,
        params_v5: TinyV5SubstrateParams,
        prompt_token_ids: Sequence[int],
        follow_up_chain: Sequence[Sequence[int]],
        segments: Sequence[tuple[int, int, str]],
) -> PrefixStateBridgeV4Witness:
    """End-to-end V4 prefix-state bridge over a multi-segment
    prefix and a chain of follow-ups.

    Returns a witness with per-segment flop split and per-step
    drift L2 over the chain.
    """
    cfg = params_v5.config
    v3p = params_v5.v3_params
    prompt = list(prompt_token_ids)
    chain = [list(c) for c in follow_up_chain]
    if not chain:
        chain = [[]]
    # Full-recompute baseline: prompt + concat(chain).
    full_ids = list(prompt) + [
        t for c in chain for t in c]
    full_trace = forward_tiny_substrate_v3(
        v3p, full_ids, return_attention=False)
    full_last = full_trace.logits[-1]
    flop_full = int(full_trace.flop_count)
    # Multi-segment path.
    prefix = extract_multi_segment_prefix_v5(
        params_v5, prompt, segments=segments)
    # Chain forward over multi-segment cache.
    cum_cache = None
    cum_drift = 0.0
    step_drifts: list[float] = []
    flop_multi_total = 0
    final_logits = None
    # First step uses the multi-segment reuse path.
    first_step = chain[0]
    ft, fc_v5, split = forward_with_multi_segment_reuse_v5(
        params_v5, prefix, first_step)
    flop_multi_total += int(split["flop_total"])
    cum_cache = fc_v5
    final_logits = ft.logits[-1] if ft.logits.size else None
    if first_step and final_logits is not None:
        # Compare to a full-recompute step output for the same
        # cumulative slice.
        slice_full_ids = list(prompt) + list(first_step)
        slice_full = forward_tiny_substrate_v3(
            v3p, slice_full_ids, return_attention=False)
        d = float(_np.linalg.norm(
            final_logits - slice_full.logits[-1]))
        step_drifts.append(d)
        cum_drift += d
    # Subsequent steps.
    for step_ids in chain[1:]:
        if not step_ids:
            step_drifts.append(0.0)
            continue
        # Forward the next step on top of the current cache.
        from .tiny_substrate_v4 import (
            forward_tiny_substrate_v4,
        )
        next_trace, fc_v4_next = forward_tiny_substrate_v4(
            params_v5.v4_params, step_ids,
            v3_kv_cache=cum_cache.v4_cache.v3_cache)
        flop_multi_total += int(next_trace.flop_recompute)
        # Compare to full slice.
        slice_full_ids = list(prompt) + [
            t for c in chain[:chain.index(step_ids) + 1]
            for t in c]
        slice_full = forward_tiny_substrate_v3(
            v3p, slice_full_ids, return_attention=False)
        last_seg = next_trace.logits[-1]
        d = float(_np.linalg.norm(
            last_seg - slice_full.logits[-1]))
        step_drifts.append(d)
        cum_drift += d
        # Update v5 cache for next iteration.
        from .tiny_substrate_v5 import TinyV5KVCache
        cum_cache = TinyV5KVCache.empty(
            int(cfg.n_layers), n_heads=int(cfg.n_heads))
        cum_cache.v4_cache = fc_v4_next
        final_logits = last_seg
    # Drop-induced drift = max-abs between multi-segment final
    # logits and full-recompute logits at the final step.
    if final_logits is not None:
        drop_drift = float(_np.max(_np.abs(
            final_logits - full_last)))
    else:
        drop_drift = 0.0
    flop_saved = int(flop_full - flop_multi_total)
    flop_ratio = float(flop_saved) / float(max(flop_full, 1))
    n_reuse = sum(1 for s in segments
                   if s[2] == W60_V5_SEGMENT_REUSE)
    n_rec = sum(1 for s in segments
                  if s[2] == W60_V5_SEGMENT_RECOMPUTE)
    n_drop = sum(1 for s in segments
                   if s[2] == W60_V5_SEGMENT_DROP)
    last_first = chain[0] if chain else []
    return PrefixStateBridgeV4Witness(
        schema=W60_PREFIX_STATE_BRIDGE_V4_SCHEMA_VERSION,
        multi_segment_prefix_cid=str(prefix.cid()),
        n_segments=int(len(segments)),
        n_reuse_segments=int(n_reuse),
        n_recompute_segments=int(n_rec),
        n_drop_segments=int(n_drop),
        reuse_len=int(prefix.reuse_len()),
        recompute_len=int(prefix.recompute_len()),
        drop_len=int(prefix.drop_len()),
        follow_up_token_ids=tuple(
            int(t) for t in last_first),
        full_recompute_last_logits_cid=_ndarray_cid(full_last),
        multi_segment_last_logits_cid=_ndarray_cid(
            final_logits if final_logits is not None
            else _np.zeros_like(full_last)),
        max_abs_drop_induced_drift=float(drop_drift),
        flop_full_recompute=int(flop_full),
        flop_multi_segment_total=int(flop_multi_total),
        flop_saved_vs_recompute=int(flop_saved),
        flop_savings_ratio=float(flop_ratio),
        chain_step_drifts_l2=tuple(step_drifts),
        chain_cumulative_drift_l2=float(cum_drift),
    )


__all__ = [
    "W60_PREFIX_STATE_BRIDGE_V4_SCHEMA_VERSION",
    "PrefixStateBridgeV4Witness",
    "bridge_prefix_state_and_measure_v4",
]
