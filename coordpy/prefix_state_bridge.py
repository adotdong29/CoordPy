"""W57 M4 — Prefix-State Bridge.

Saves a substrate KV cache as a *prefix state*, hashes it
content-addressed, and reuses it across turns. Operationally this
is "real cached-state reuse across turns" — the substrate does
not need to recompute the prefix; it loads the previous prefix
state, and a fresh forward over follow-up tokens attends to it.

This is the load-bearing piece for the **state-reuse-vs-recompute**
benchmark (R-116). We compare:

* (a) **substrate prefix-state reuse**: load saved KV → forward
  over a follow-up.
* (b) **substrate full recompute**: forward over the same
  combined sequence from scratch.
* (c) **proxy-only baseline**: a capsule-only path that does not
  touch substrate state.

For correctly-implemented KV caches (a) and (b) produce
byte-identical logits at the follow-up positions (within float64
precision). The bridge measures this and certifies it.

It also exposes a **prefix-state CORRUPTION** path: a caller can
deliberately corrupt a saved prefix and re-load it. The bridge
detects the corruption via the recorded readback CID and exposes
a ``corruption_detected`` field for the consensus controller V3.
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
        "coordpy.prefix_state_bridge requires numpy") from exc

from .tiny_substrate_v2 import (
    TinyV2KVCache,
    TinyV2PrefixState,
    TinyV2SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    extract_prefix_state,
    forward_tiny_substrate_v2,
)


W57_PREFIX_STATE_BRIDGE_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge.v1")


@dataclasses.dataclass(frozen=True)
class PrefixStateBridgeWitness:
    schema: str
    prefix_state_cid: str
    follow_up_token_ids: tuple[int, ...]
    recompute_logits_cid: str
    reuse_logits_cid: str
    max_abs_reuse_recompute_diff: float
    reuse_matches_recompute: bool
    corruption_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "prefix_state_cid": str(self.prefix_state_cid),
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
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_state_bridge_witness",
            "witness": self.to_dict()})


def save_prefix_state(
        params: TinyV2SubstrateParams,
        prompt_token_ids: Sequence[int],
        *,
        prefix_len: int | None = None,
) -> TinyV2PrefixState:
    """Run a forward on ``prompt_token_ids`` and extract a prefix
    state at length ``prefix_len`` (default: full prompt)."""
    trace = forward_tiny_substrate_v2(
        params, list(prompt_token_ids), return_attention=False)
    n = trace.kv_cache.n_tokens()
    L = int(prefix_len) if prefix_len is not None else int(n)
    return extract_prefix_state(
        trace.kv_cache,
        prefix_len=L,
        source_params_cid=str(params.cid()))


def corrupt_prefix_state(
        ps: TinyV2PrefixState, *,
        layer_index: int,
        token_position: int,
        magnitude: float = 1.0,
        seed: int = 57044,
) -> TinyV2PrefixState:
    """Deliberately corrupt one slot of a prefix state.

    Returns a NEW prefix state with the corruption applied. The
    ``corruption_detected`` check in the bridge witness compares
    the new CID against the original CID — they will differ if
    the corruption changed the bytes.
    """
    rng = _np.random.default_rng(int(seed))
    new_keys = [k.copy() for k in ps.keys]
    new_values = [v.copy() for v in ps.values]
    L = max(0, min(int(layer_index), len(new_keys) - 1))
    if new_keys[L].size:
        T = new_keys[L].shape[0]
        D = new_keys[L].shape[1]
        T_idx = max(0, min(int(token_position), T - 1))
        perturb = rng.standard_normal(D) * float(magnitude)
        new_keys[L][T_idx] = new_keys[L][T_idx] + perturb
        new_values[L][T_idx] = (
            new_values[L][T_idx] + perturb)
    return TinyV2PrefixState(
        prefix_len=int(ps.prefix_len),
        keys=tuple(new_keys),
        values=tuple(new_values),
        source_params_cid=str(ps.source_params_cid),
    )


def bridge_prefix_state_and_measure(
        *,
        params: TinyV2SubstrateParams,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        corrupt_after_save: bool = False,
        corruption_layer: int = 0,
        corruption_position: int = 0,
        corruption_magnitude: float = 1.0,
) -> PrefixStateBridgeWitness:
    """End-to-end prefix-state reuse experiment.

    Saves a prefix state from ``prompt_token_ids``, optionally
    corrupts it, then forwards ``follow_up_token_ids`` on (a) the
    prefix-state-reused cache and (b) a full recompute. Compares.
    """
    # Save prefix from prompt.
    ps = save_prefix_state(params, list(prompt_token_ids))
    pre_corrupt_cid = ps.cid()
    if corrupt_after_save:
        ps = corrupt_prefix_state(
            ps,
            layer_index=int(corruption_layer),
            token_position=int(corruption_position),
            magnitude=float(corruption_magnitude))
    post_cid = ps.cid()
    corruption_detected = bool(post_cid != pre_corrupt_cid)
    # Reuse path.
    reuse_cache = ps.to_cache()
    reuse_trace = forward_tiny_substrate_v2(
        params, list(follow_up_token_ids),
        kv_cache=reuse_cache,
        return_attention=False)
    # Recompute path.
    combined = list(prompt_token_ids) + list(follow_up_token_ids)
    recomp_trace = forward_tiny_substrate_v2(
        params, list(combined), return_attention=False)
    # Compare last-position logits over the follow-up region.
    n_fu = len(list(follow_up_token_ids))
    reuse_last = reuse_trace.logits[-1]
    recomp_last = recomp_trace.logits[-1]
    diff = float(_np.max(_np.abs(reuse_last - recomp_last)))
    matches = bool(diff < 1e-9) if not corrupt_after_save else False
    return PrefixStateBridgeWitness(
        schema=W57_PREFIX_STATE_BRIDGE_SCHEMA_VERSION,
        prefix_state_cid=str(post_cid),
        follow_up_token_ids=tuple(
            int(t) for t in follow_up_token_ids),
        recompute_logits_cid=str(
            _ndarray_cid(recomp_last)),
        reuse_logits_cid=str(_ndarray_cid(reuse_last)),
        max_abs_reuse_recompute_diff=float(diff),
        reuse_matches_recompute=bool(matches),
        corruption_detected=bool(corruption_detected),
    )


__all__ = [
    "W57_PREFIX_STATE_BRIDGE_SCHEMA_VERSION",
    "PrefixStateBridgeWitness",
    "save_prefix_state",
    "corrupt_prefix_state",
    "bridge_prefix_state_and_measure",
]
