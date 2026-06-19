"""W61 M16 — Uncertainty Layer V9.

Strictly extends W60's ``coordpy.uncertainty_layer_v8``. V9 adds an
**eighth confidence weighting axis**: ``attention_pattern_fidelity``
(in ``[0, 1]``). The W61 composite is now

  weighted_composite = Σ_i w_i × confidence_i

where w_i is the product of eight per-component factors normalised.

V9 reduces to V8 byte-for-byte when ``attention_pattern_fidelities``
is all 1.0.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence


W61_UNCERTAINTY_V9_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v9.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class AttentionPatternAwareWeightedComposite:
    composite: float
    weighted_composite: float
    pessimistic_composite: float
    optimistic_composite: float
    n_components: int
    hidden_aware: bool
    cache_aware: bool
    retrieval_aware: bool
    replay_aware: bool
    attention_pattern_aware: bool
    sensitivity_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W61_UNCERTAINTY_V9_SCHEMA_VERSION,
            "composite": float(round(self.composite, 12)),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "pessimistic_composite": float(round(
                self.pessimistic_composite, 12)),
            "optimistic_composite": float(round(
                self.optimistic_composite, 12)),
            "n_components": int(self.n_components),
            "hidden_aware": bool(self.hidden_aware),
            "cache_aware": bool(self.cache_aware),
            "retrieval_aware": bool(self.retrieval_aware),
            "replay_aware": bool(self.replay_aware),
            "attention_pattern_aware": bool(
                self.attention_pattern_aware),
            "sensitivity_max": float(round(
                self.sensitivity_max, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v9_composite",
            "composite": self.to_dict()})


def _weighted_composite_v9(
        cs: Sequence[float], ts: Sequence[float],
        sfs: Sequence[float], hfs: Sequence[float],
        cfs: Sequence[float], rfs: Sequence[float],
        rps: Sequence[float], aps: Sequence[float],
) -> float:
    w = [(float(t) * float(s) * float(h) * float(c)
            * float(r) * float(rp) * float(ap))
          for t, s, h, c, r, rp, ap
          in zip(ts, sfs, hfs, cfs, rfs, rps, aps)]
    z = float(sum(w)) or 1.0
    w = [x / z for x in w]
    return float(sum(w[i] * float(cs[i])
                      for i in range(len(cs))))


def compose_uncertainty_report_v9(
        *,
        confidences: Sequence[float],
        trusts: Sequence[float],
        substrate_fidelities: Sequence[float],
        hidden_state_fidelities: Sequence[float],
        cache_reuse_fidelities: Sequence[float],
        retrieval_fidelities: Sequence[float],
        replay_fidelities: Sequence[float],
        attention_pattern_fidelities: Sequence[float] | None = None,
) -> AttentionPatternAwareWeightedComposite:
    n = len(confidences)
    if attention_pattern_fidelities is None:
        attention_pattern_fidelities = [1.0] * n
    ts = list(trusts) + [1.0] * (n - len(trusts))
    sfs = list(substrate_fidelities) + [1.0] * (
        n - len(substrate_fidelities))
    hfs = list(hidden_state_fidelities) + [1.0] * (
        n - len(hidden_state_fidelities))
    cfs = list(cache_reuse_fidelities) + [1.0] * (
        n - len(cache_reuse_fidelities))
    rfs = list(retrieval_fidelities) + [1.0] * (
        n - len(retrieval_fidelities))
    rps = list(replay_fidelities) + [1.0] * (
        n - len(replay_fidelities))
    aps = list(attention_pattern_fidelities) + [1.0] * (
        n - len(attention_pattern_fidelities))
    comp = (
        float(sum(confidences)) / float(n) if n else 0.0)
    w_comp = _weighted_composite_v9(
        confidences, ts, sfs, hfs, cfs, rfs, rps, aps)
    pess = (
        float(min(confidences)) if confidences else 0.0)
    opt = (
        float(max(confidences)) if confidences else 0.0)
    # Sensitivity: per-axis max range across components.
    axes = [ts, sfs, hfs, cfs, rfs, rps, aps]
    sens = float(
        max((max(a) - min(a)) for a in axes if a)) if n else 0.0
    return AttentionPatternAwareWeightedComposite(
        composite=float(comp),
        weighted_composite=float(w_comp),
        pessimistic_composite=float(pess),
        optimistic_composite=float(opt),
        n_components=int(n),
        hidden_aware=any(x < 1.0 - 1e-12 for x in hfs),
        cache_aware=any(x < 1.0 - 1e-12 for x in cfs),
        retrieval_aware=any(x < 1.0 - 1e-12 for x in rfs),
        replay_aware=any(x < 1.0 - 1e-12 for x in rps),
        attention_pattern_aware=any(
            x < 1.0 - 1e-12 for x in aps),
        sensitivity_max=float(sens),
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyV9Witness:
    schema: str
    composite_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v9_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v9_witness(
        composite: AttentionPatternAwareWeightedComposite,
) -> UncertaintyV9Witness:
    return UncertaintyV9Witness(
        schema=W61_UNCERTAINTY_V9_SCHEMA_VERSION,
        composite_cid=str(composite.cid()),
    )


__all__ = [
    "W61_UNCERTAINTY_V9_SCHEMA_VERSION",
    "AttentionPatternAwareWeightedComposite",
    "UncertaintyV9Witness",
    "compose_uncertainty_report_v9",
    "emit_uncertainty_v9_witness",
]
