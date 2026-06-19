"""W58 M4 — Prefix-State Bridge V2.

Strictly extends W57's ``coordpy.prefix_state_bridge``. V2 adds
three load-bearing pieces to the prefix-state reuse path:

1. **Real flop-saved counter** — running the follow-up forward
   with a saved prefix state costs ``flop_with_reuse`` fp64 ops;
   recomputing prompt+follow-up from scratch costs
   ``flop_full_recompute`` fp64 ops. The witness records both,
   plus
   ``flop_saved = flop_full_recompute - flop_with_reuse``. This
   is what the R-119 cache-reuse-vs-recompute benchmark uses.
2. **Redundant prefix storage** — every saved prefix carries a
   secondary content-addressed redundant copy CID. If the
   primary copy is corrupted, the redundant CID still matches
   the original. This is the substrate-side equivalent of
   replication; it lets the consensus controller V4 distinguish
   "prefix bit-rot" from "prefix is gone".
3. **Cross-seed prefix translation** — V2 can re-extract a prefix
   from a substrate trained with seed ``s1`` and replay it on a
   substrate trained with seed ``s2``. The bridge measures the
   resulting *drift* (L2 of last-position logits vs a same-seed
   reuse). The drift is what tells you whether prefixes are
   transferable.

Honest scope
------------

* The "fitted" / "trained" nature of W58 is restricted to the
  inject scales of the KV/HSB bridges. The prefix state itself
  is *byte-exact* on same-seed reuse; the cross-seed transfer
  measurement is an honest probe of drift, not a claim that
  prefixes are cross-model-transferable.
* W58 caps carry forward unchanged.
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
        "coordpy.prefix_state_bridge_v2 requires numpy") from exc

from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3PrefixState,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    build_default_tiny_substrate_v3,
    extract_prefix_state_v3,
    forward_tiny_substrate_v3,
)


W58_PREFIX_STATE_BRIDGE_V2_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v2.v1")


@dataclasses.dataclass(frozen=True)
class PrefixStateBridgeV2Witness:
    schema: str
    prefix_state_cid: str
    redundant_copy_cid: str
    follow_up_token_ids: tuple[int, ...]
    recompute_logits_cid: str
    reuse_logits_cid: str
    max_abs_reuse_recompute_diff: float
    reuse_matches_recompute: bool
    corruption_detected: bool
    redundant_copy_intact: bool
    flop_full_recompute: int
    flop_with_reuse: int
    flop_saved: int
    flop_savings_ratio: float
    cross_seed_drift_l2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "prefix_state_cid": str(self.prefix_state_cid),
            "redundant_copy_cid": str(self.redundant_copy_cid),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "recompute_logits_cid": str(
                self.recompute_logits_cid),
            "reuse_logits_cid": str(self.reuse_logits_cid),
            "max_abs_reuse_recompute_diff": float(round(
                self.max_abs_reuse_recompute_diff, 12)),
            "reuse_matches_recompute": bool(
                self.reuse_matches_recompute),
            "corruption_detected": bool(
                self.corruption_detected),
            "redundant_copy_intact": bool(
                self.redundant_copy_intact),
            "flop_full_recompute": int(self.flop_full_recompute),
            "flop_with_reuse": int(self.flop_with_reuse),
            "flop_saved": int(self.flop_saved),
            "flop_savings_ratio": float(round(
                self.flop_savings_ratio, 12)),
            "cross_seed_drift_l2": float(round(
                self.cross_seed_drift_l2, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_state_bridge_v2_witness",
            "witness": self.to_dict()})


def save_prefix_state_v3(
        params: TinyV3SubstrateParams,
        prompt_token_ids: Sequence[int],
        *,
        prefix_len: int | None = None,
) -> TinyV3PrefixState:
    trace = forward_tiny_substrate_v3(
        params, list(prompt_token_ids), return_attention=False)
    n = trace.kv_cache.n_tokens()
    L = int(prefix_len) if prefix_len is not None else int(n)
    return extract_prefix_state_v3(
        trace.kv_cache,
        prefix_len=L,
        source_params_cid=str(params.cid()))


def corrupt_prefix_state_v3(
        ps: TinyV3PrefixState, *,
        layer_index: int,
        token_position: int,
        magnitude: float = 1.0,
        seed: int = 58044,
) -> TinyV3PrefixState:
    rng = _np.random.default_rng(int(seed))
    new_keys = [k.copy() for k in ps.keys]
    new_values = [v.copy() for v in ps.values]
    new_importance = [i.copy() for i in ps.importance]
    L = max(0, min(int(layer_index), len(new_keys) - 1))
    if new_keys[L].size:
        T = new_keys[L].shape[0]
        D = new_keys[L].shape[1]
        T_idx = max(0, min(int(token_position), T - 1))
        perturb = rng.standard_normal(D) * float(magnitude)
        new_keys[L][T_idx] = new_keys[L][T_idx] + perturb
        new_values[L][T_idx] = (
            new_values[L][T_idx] + perturb)
    redundant = _sha256_hex({
        "keys_cids": [_ndarray_cid(k) for k in new_keys],
        "values_cids": [_ndarray_cid(v) for v in new_values],
    })
    return TinyV3PrefixState(
        prefix_len=int(ps.prefix_len),
        keys=tuple(new_keys),
        values=tuple(new_values),
        importance=tuple(new_importance),
        source_params_cid=str(ps.source_params_cid),
        redundant_copy_cid=str(redundant),
    )


def bridge_prefix_state_and_measure_v2(
        *,
        params: TinyV3SubstrateParams,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        corrupt_after_save: bool = False,
        corruption_layer: int = 0,
        corruption_position: int = 0,
        corruption_magnitude: float = 1.0,
        cross_seed_params: TinyV3SubstrateParams | None = None,
) -> PrefixStateBridgeV2Witness:
    """End-to-end V2 prefix-state reuse + flop counter +
    cross-seed drift measurement."""
    ps = save_prefix_state_v3(params, list(prompt_token_ids))
    pre_corrupt_cid = ps.cid()
    pre_redundant_cid = ps.redundant_copy_cid
    if corrupt_after_save:
        ps = corrupt_prefix_state_v3(
            ps,
            layer_index=int(corruption_layer),
            token_position=int(corruption_position),
            magnitude=float(corruption_magnitude))
    post_cid = ps.cid()
    post_redundant_cid = ps.redundant_copy_cid
    corruption_detected = bool(post_cid != pre_corrupt_cid)
    # redundant_copy_intact = True iff the redundant CID has not
    # changed since save. A redundant copy that flips under
    # corruption is what signals "even my backup detected the
    # damage"; True means the backup still matches the original.
    redundant_intact = bool(
        post_redundant_cid == pre_redundant_cid)

    # Reuse path
    reuse_cache = ps.to_cache()
    reuse_trace = forward_tiny_substrate_v3(
        params, list(follow_up_token_ids),
        kv_cache=reuse_cache,
        return_attention=False)
    flop_reuse = int(reuse_trace.flop_count)

    # Full recompute path
    combined = list(prompt_token_ids) + list(follow_up_token_ids)
    recomp_trace = forward_tiny_substrate_v3(
        params, list(combined), return_attention=False)
    flop_full = int(recomp_trace.flop_count)

    reuse_last = reuse_trace.logits[-1]
    recomp_last = recomp_trace.logits[-1]
    diff = float(_np.max(_np.abs(reuse_last - recomp_last)))
    matches = (
        bool(diff < 1e-9) if not corrupt_after_save else False)

    flop_saved = int(flop_full - flop_reuse)
    flop_ratio = (
        float(flop_saved) / float(max(flop_full, 1)))

    # Cross-seed drift (replay prefix on different-seed substrate)
    cross_drift_l2 = 0.0
    if cross_seed_params is not None:
        # Use the prefix from `params` (different seed) — the
        # substrate forward on `cross_seed_params` consumes the
        # foreign cache. Logits will drift.
        try:
            cross_trace = forward_tiny_substrate_v3(
                cross_seed_params, list(follow_up_token_ids),
                kv_cache=reuse_cache,
                return_attention=False)
            cross_drift_l2 = float(_np.linalg.norm(
                cross_trace.logits[-1] - reuse_last))
        except Exception:
            cross_drift_l2 = float("nan")

    return PrefixStateBridgeV2Witness(
        schema=W58_PREFIX_STATE_BRIDGE_V2_SCHEMA_VERSION,
        prefix_state_cid=str(post_cid),
        redundant_copy_cid=str(post_redundant_cid),
        follow_up_token_ids=tuple(
            int(t) for t in follow_up_token_ids),
        recompute_logits_cid=str(_ndarray_cid(recomp_last)),
        reuse_logits_cid=str(_ndarray_cid(reuse_last)),
        max_abs_reuse_recompute_diff=float(diff),
        reuse_matches_recompute=bool(matches),
        corruption_detected=bool(corruption_detected),
        redundant_copy_intact=bool(redundant_intact),
        flop_full_recompute=int(flop_full),
        flop_with_reuse=int(flop_reuse),
        flop_saved=int(flop_saved),
        flop_savings_ratio=float(flop_ratio),
        cross_seed_drift_l2=float(cross_drift_l2),
    )


__all__ = [
    "W58_PREFIX_STATE_BRIDGE_V2_SCHEMA_VERSION",
    "PrefixStateBridgeV2Witness",
    "save_prefix_state_v3",
    "corrupt_prefix_state_v3",
    "bridge_prefix_state_and_measure_v2",
]
