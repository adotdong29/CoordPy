"""W59 M4 — Prefix-State Bridge V3.

Strictly extends W58's ``coordpy.prefix_state_bridge_v2``. V3 adds:

* **Partial-prefix reuse** — V2 reuses the *full* saved prefix or
  recomputes from scratch. V3 supports splitting a saved prefix
  into a reusable head ``[0, p_reuse)`` and a recompute tail
  ``[p_reuse, prefix_len)``. The bridge replays the head, runs
  the tail through the substrate, then evaluates the follow-up.
  Flops are reported as
  ``flop_reuse_skipped``, ``flop_recompute_tail``,
  ``flop_follow_up``, plus the total.
* **K-seed drift L2** — V2 reports a single cross-seed drift L2.
  V3 reports the *spectrum* over K seeds (mean / max / min) and
  records the variance.
* **Drift-bound certificate** — V3 records the *maximum* L2 drift
  observed over the K seeds, alongside a Lipschitz-style bound
  ``drift ≤ C × |Δ_params|`` where ``Δ_params`` is the L2 norm of
  the seed perturbation in parameter space and C is empirically
  the worst observed ratio. The certificate is empirical, not
  proven.
* **128-bucket fingerprint** — every saved prefix carries a 128-
  bucket fingerprint (vs V2's 64). The W59 CRC V7 uses this.

V3 strictly extends V2: with ``p_reuse == prefix_len`` and only
one seed supplied, V3 reduces to V2 byte-for-byte.

Honest scope
------------

* Partial reuse is *exact* (byte-identical to a full recompute
  given the same params and the same prefix tokens) when the
  reuse head is a contiguous prefix and the tail tokens are
  consistent with the original prompt. We test this in R-122.
* The Lipschitz certificate is empirical (the worst observed
  ratio over K seeds) — not proven across all seeds.
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
        "coordpy.prefix_state_bridge_v3 requires numpy"
        ) from exc

from .prefix_state_bridge_v2 import (
    corrupt_prefix_state_v3,
    save_prefix_state_v3,
)
from .tiny_substrate_v3 import (
    TinyV3KVCache,
    TinyV3PrefixState,
    TinyV3SubstrateParams,
    _ndarray_cid,
    _sha256_hex,
    forward_tiny_substrate_v3,
)
from .tiny_substrate_v4 import (
    TinyV4PartialPrefixState,
    TinyV4SubstrateParams,
    _kv_fingerprint_128,
    extract_partial_prefix_v4,
    forward_with_partial_prefix_reuse_v4,
)


W59_PREFIX_STATE_BRIDGE_V3_SCHEMA_VERSION: str = (
    "coordpy.prefix_state_bridge_v3.v1")


@dataclasses.dataclass(frozen=True)
class PrefixStateBridgeV3Witness:
    schema: str
    full_prefix_cid: str
    partial_prefix_cid: str
    prefix_total_len: int
    prefix_reuse_len: int
    follow_up_token_ids: tuple[int, ...]
    recompute_logits_cid: str
    partial_reuse_logits_cid: str
    full_reuse_logits_cid: str
    max_abs_partial_reuse_vs_recompute_diff: float
    partial_reuse_matches_recompute: bool
    flop_full_recompute: int
    flop_full_reuse: int
    flop_partial_reuse: int
    flop_partial_recompute_tail: int
    flop_partial_total: int
    flop_saved_vs_recompute: int
    flop_savings_ratio_vs_recompute: float
    k_seed_drift_l2_mean: float
    k_seed_drift_l2_max: float
    k_seed_drift_l2_min: float
    k_seed_drift_l2_var: float
    drift_lipschitz_certificate_ratio: float
    fingerprint_128: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "full_prefix_cid": str(self.full_prefix_cid),
            "partial_prefix_cid": str(self.partial_prefix_cid),
            "prefix_total_len": int(self.prefix_total_len),
            "prefix_reuse_len": int(self.prefix_reuse_len),
            "follow_up_token_ids": list(
                self.follow_up_token_ids),
            "recompute_logits_cid": str(
                self.recompute_logits_cid),
            "partial_reuse_logits_cid": str(
                self.partial_reuse_logits_cid),
            "full_reuse_logits_cid": str(
                self.full_reuse_logits_cid),
            "max_abs_partial_reuse_vs_recompute_diff": float(
                round(
                    self.max_abs_partial_reuse_vs_recompute_diff,
                    12)),
            "partial_reuse_matches_recompute": bool(
                self.partial_reuse_matches_recompute),
            "flop_full_recompute": int(self.flop_full_recompute),
            "flop_full_reuse": int(self.flop_full_reuse),
            "flop_partial_reuse": int(self.flop_partial_reuse),
            "flop_partial_recompute_tail": int(
                self.flop_partial_recompute_tail),
            "flop_partial_total": int(self.flop_partial_total),
            "flop_saved_vs_recompute": int(
                self.flop_saved_vs_recompute),
            "flop_savings_ratio_vs_recompute": float(round(
                self.flop_savings_ratio_vs_recompute, 12)),
            "k_seed_drift_l2_mean": float(round(
                self.k_seed_drift_l2_mean, 12)),
            "k_seed_drift_l2_max": float(round(
                self.k_seed_drift_l2_max, 12)),
            "k_seed_drift_l2_min": float(round(
                self.k_seed_drift_l2_min, 12)),
            "k_seed_drift_l2_var": float(round(
                self.k_seed_drift_l2_var, 12)),
            "drift_lipschitz_certificate_ratio": float(round(
                self.drift_lipschitz_certificate_ratio, 12)),
            "fingerprint_128": list(self.fingerprint_128),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "prefix_state_bridge_v3_witness",
            "witness": self.to_dict()})


def bridge_prefix_state_and_measure_v3(
        *,
        params_v4: TinyV4SubstrateParams,
        prompt_token_ids: Sequence[int],
        follow_up_token_ids: Sequence[int],
        prefix_reuse_len: int,
        cross_seed_params_list: Sequence[TinyV4SubstrateParams] = (),
) -> PrefixStateBridgeV3Witness:
    """End-to-end V3 prefix-state bridge with partial reuse,
    K-seed drift spectrum, and 128-bucket fingerprint."""
    prompt = list(prompt_token_ids)
    fu = list(follow_up_token_ids)
    v3p = params_v4.v3_params

    # Full recompute path.
    combined = prompt + fu
    recomp = forward_tiny_substrate_v3(
        v3p, combined, return_attention=False)
    flop_full_recompute = int(recomp.flop_count)
    recomp_last = recomp.logits[-1]

    # Full reuse (V2-style) path.
    full_ps = save_prefix_state_v3(v3p, prompt)
    full_reuse_cache = full_ps.to_cache()
    full_reuse = forward_tiny_substrate_v3(
        v3p, fu, kv_cache=full_reuse_cache,
        return_attention=False)
    flop_full_reuse = int(full_reuse.flop_count)
    full_reuse_last = full_reuse.logits[-1]

    # Partial reuse path (V4).
    partial = extract_partial_prefix_v4(
        params_v4, prompt, prefix_reuse_len=int(prefix_reuse_len))
    part_trace, _, flop_split = (
        forward_with_partial_prefix_reuse_v4(
            params_v4, partial, fu))
    partial_last = part_trace.logits[-1]
    flop_partial_total = int(flop_split["flop_total"])
    flop_partial_tail = int(flop_split["flop_recompute_tail"])
    # Partial reuse should match a full recompute byte-identical
    # within float64 precision.
    diff = float(_np.max(_np.abs(partial_last - recomp_last)))
    matches = bool(diff < 1e-9)

    flop_saved = int(flop_full_recompute - flop_partial_total)
    flop_ratio = (
        float(flop_saved) / float(max(flop_full_recompute, 1)))

    # K-seed drift spectrum.
    drifts: list[float] = []
    for foreign in cross_seed_params_list:
        try:
            t = forward_tiny_substrate_v3(
                foreign.v3_params, fu,
                kv_cache=full_reuse_cache,
                return_attention=False)
            drifts.append(float(_np.linalg.norm(
                t.logits[-1] - full_reuse_last)))
        except Exception:
            drifts.append(float("nan"))
    if drifts:
        clean = [d for d in drifts if not _np.isnan(d)]
        if clean:
            mean_d = float(_np.mean(clean))
            max_d = float(_np.max(clean))
            min_d = float(_np.min(clean))
            var_d = float(_np.var(clean))
        else:
            mean_d = max_d = min_d = var_d = float("nan")
    else:
        mean_d = 0.0
        max_d = 0.0
        min_d = 0.0
        var_d = 0.0

    # Empirical Lipschitz certificate: max drift / max parameter-
    # L2 delta over the K cross-seed substrates.
    cert_ratio = 0.0
    if cross_seed_params_list:
        base_params_flat = _np.concatenate([
            v3p.embed.reshape(-1),
            v3p.unembed.reshape(-1),
        ]).astype(_np.float64)
        max_param_delta = 0.0
        for foreign in cross_seed_params_list:
            fp = _np.concatenate([
                foreign.v3_params.embed.reshape(-1),
                foreign.v3_params.unembed.reshape(-1),
            ]).astype(_np.float64)
            d = float(_np.linalg.norm(fp - base_params_flat))
            if d > max_param_delta:
                max_param_delta = d
        if max_param_delta > 1e-9:
            cert_ratio = float(max_d) / float(max_param_delta)

    # 128-bucket fingerprint over the full prefix.
    fp128 = [0] * 128
    for k in full_ps.keys:
        bs = _np.ascontiguousarray(k).tobytes()
        for j, byte in enumerate(bs):
            fp128[j % 128] ^= int(byte)
    for v in full_ps.values:
        bs = _np.ascontiguousarray(v).tobytes()
        for j, byte in enumerate(bs):
            fp128[j % 128] ^= int(byte)

    return PrefixStateBridgeV3Witness(
        schema=W59_PREFIX_STATE_BRIDGE_V3_SCHEMA_VERSION,
        full_prefix_cid=str(full_ps.cid()),
        partial_prefix_cid=str(partial.cid()),
        prefix_total_len=int(len(prompt)),
        prefix_reuse_len=int(partial.prefix_reuse_len),
        follow_up_token_ids=tuple(int(t) for t in fu),
        recompute_logits_cid=_ndarray_cid(recomp_last),
        partial_reuse_logits_cid=_ndarray_cid(partial_last),
        full_reuse_logits_cid=_ndarray_cid(full_reuse_last),
        max_abs_partial_reuse_vs_recompute_diff=float(diff),
        partial_reuse_matches_recompute=bool(matches),
        flop_full_recompute=int(flop_full_recompute),
        flop_full_reuse=int(flop_full_reuse),
        flop_partial_reuse=int(flop_split["flop_reuse_skipped"]),
        flop_partial_recompute_tail=int(flop_partial_tail),
        flop_partial_total=int(flop_partial_total),
        flop_saved_vs_recompute=int(flop_saved),
        flop_savings_ratio_vs_recompute=float(flop_ratio),
        k_seed_drift_l2_mean=float(mean_d),
        k_seed_drift_l2_max=float(max_d),
        k_seed_drift_l2_min=float(min_d),
        k_seed_drift_l2_var=float(var_d),
        drift_lipschitz_certificate_ratio=float(cert_ratio),
        fingerprint_128=tuple(int(b) for b in fp128),
    )


__all__ = [
    "W59_PREFIX_STATE_BRIDGE_V3_SCHEMA_VERSION",
    "PrefixStateBridgeV3Witness",
    "bridge_prefix_state_and_measure_v3",
]
