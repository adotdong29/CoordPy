"""W60 M16 — Uncertainty Layer V8.

Strictly extends W59's ``coordpy.uncertainty_layer_v7``. V8 adds a
**seventh confidence weighting axis**: ``replay_fidelity`` (in
``[0, 1]``). The W60 composite is now

  weighted_composite =
      Σ_i w_i × confidence_i

where the weights are

  w_i = trust_i × substrate_fidelity_i × hidden_state_fidelity_i
        × cache_reuse_fidelity_i × retrieval_fidelity_i
        × replay_fidelity_i

normalised. The pessimistic/optimistic brackets and per-axis
sensitivity (now 7 axes) carry forward from V7.

V8 reduces to V7 byte-for-byte when ``replay_fidelities`` is all
1.0.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping, Sequence


W60_UNCERTAINTY_V8_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v8.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class ReplayAwareWeightedComposite:
    composite: float
    weighted_composite: float
    pessimistic_composite: float
    optimistic_composite: float
    n_components: int
    hidden_aware: bool
    cache_aware: bool
    retrieval_aware: bool
    replay_aware: bool
    sensitivity_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W60_UNCERTAINTY_V8_SCHEMA_VERSION,
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
            "sensitivity_max": float(round(
                self.sensitivity_max, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v8_composite",
            "composite": self.to_dict()})


def _weighted_composite_v8(
        cs: Sequence[float], ts: Sequence[float],
        sfs: Sequence[float], hfs: Sequence[float],
        cfs: Sequence[float], rfs: Sequence[float],
        rps: Sequence[float],
) -> float:
    w = [(float(t) * float(s) * float(h) * float(c)
            * float(r) * float(rp))
          for t, s, h, c, r, rp
          in zip(ts, sfs, hfs, cfs, rfs, rps)]
    z = float(sum(w)) or 1.0
    w = [x / z for x in w]
    return float(sum(w[i] * float(cs[i])
                      for i in range(len(cs))))


def compose_uncertainty_report_v8(
        *,
        component_confidences: Mapping[str, float],
        trust_weights: Mapping[str, float],
        substrate_fidelities: Mapping[str, float],
        hidden_state_fidelities: Mapping[str, float],
        cache_reuse_fidelities: Mapping[str, float],
        retrieval_fidelities: Mapping[str, float],
        replay_fidelities: Mapping[str, float],
        adversarial_radius: float = 0.05,
) -> ReplayAwareWeightedComposite:
    if not component_confidences:
        return ReplayAwareWeightedComposite(
            composite=1.0,
            weighted_composite=1.0,
            pessimistic_composite=1.0,
            optimistic_composite=1.0,
            n_components=0,
            hidden_aware=False,
            cache_aware=False,
            retrieval_aware=False,
            replay_aware=False,
            sensitivity_max=0.0,
        )
    keys = list(component_confidences.keys())
    cs = [float(component_confidences[k]) for k in keys]
    ts = [float(trust_weights.get(k, 1.0)) for k in keys]
    sfs = [float(substrate_fidelities.get(k, 1.0))
            for k in keys]
    hfs = [float(hidden_state_fidelities.get(k, 1.0))
            for k in keys]
    cfs = [float(cache_reuse_fidelities.get(k, 1.0))
            for k in keys]
    rfs = [float(retrieval_fidelities.get(k, 1.0))
            for k in keys]
    rps = [float(replay_fidelities.get(k, 1.0)) for k in keys]
    composite = float(
        sum(cs[i] for i in range(len(cs)))
        / float(len(cs)))
    weighted = _weighted_composite_v8(
        cs, ts, sfs, hfs, cfs, rfs, rps)
    eps = max(0.0, float(adversarial_radius))
    pess = _weighted_composite_v8(
        [max(0.0, c - eps) for c in cs],
        ts, sfs, hfs, cfs, rfs, rps)
    opti = _weighted_composite_v8(
        [min(1.0, c + eps) for c in cs],
        ts, sfs, hfs, cfs, rfs, rps)
    sens = 0.0
    delta = 1e-3
    base = weighted
    for axis_idx in range(7):
        for i in range(len(cs)):
            cs2 = list(cs); ts2 = list(ts); sfs2 = list(sfs)
            hfs2 = list(hfs); cfs2 = list(cfs); rfs2 = list(rfs)
            rps2 = list(rps)
            if axis_idx == 0:
                cs2[i] = float(max(0.0, min(1.0,
                                              cs[i] + delta)))
            elif axis_idx == 1:
                ts2[i] = float(max(0.0, min(1.0,
                                              ts[i] + delta)))
            elif axis_idx == 2:
                sfs2[i] = float(max(0.0, min(
                    1.0, sfs[i] + delta)))
            elif axis_idx == 3:
                hfs2[i] = float(max(0.0, min(
                    1.0, hfs[i] + delta)))
            elif axis_idx == 4:
                cfs2[i] = float(max(0.0, min(
                    1.0, cfs[i] + delta)))
            elif axis_idx == 5:
                rfs2[i] = float(max(0.0, min(
                    1.0, rfs[i] + delta)))
            else:
                rps2[i] = float(max(0.0, min(
                    1.0, rps[i] + delta)))
            perturbed = _weighted_composite_v8(
                cs2, ts2, sfs2, hfs2, cfs2, rfs2, rps2)
            j = abs(perturbed - base) / float(delta)
            if j > sens:
                sens = float(j)
    hidden_distinct = (
        len(set(hfs)) > 1
        or any(h != 1.0 for h in hfs))
    cache_distinct = (
        len(set(cfs)) > 1
        or any(c != 1.0 for c in cfs))
    retrieval_distinct = (
        len(set(rfs)) > 1
        or any(r != 1.0 for r in rfs))
    replay_distinct = (
        len(set(rps)) > 1
        or any(r != 1.0 for r in rps))
    return ReplayAwareWeightedComposite(
        composite=float(composite),
        weighted_composite=float(weighted),
        pessimistic_composite=float(pess),
        optimistic_composite=float(opti),
        n_components=int(len(cs)),
        hidden_aware=bool(hidden_distinct),
        cache_aware=bool(cache_distinct),
        retrieval_aware=bool(retrieval_distinct),
        replay_aware=bool(replay_distinct),
        sensitivity_max=float(sens),
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyLayerV8Witness:
    schema: str
    composite_cid: str
    n_components: int
    hidden_aware: bool
    cache_aware: bool
    retrieval_aware: bool
    replay_aware: bool
    pessimistic_le_weighted: bool
    weighted_le_optimistic: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "n_components": int(self.n_components),
            "hidden_aware": bool(self.hidden_aware),
            "cache_aware": bool(self.cache_aware),
            "retrieval_aware": bool(self.retrieval_aware),
            "replay_aware": bool(self.replay_aware),
            "pessimistic_le_weighted": bool(
                self.pessimistic_le_weighted),
            "weighted_le_optimistic": bool(
                self.weighted_le_optimistic),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v8_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v8_witness(
        *, composite: ReplayAwareWeightedComposite,
) -> UncertaintyLayerV8Witness:
    p_le_w = bool(
        composite.pessimistic_composite
        <= composite.weighted_composite + 1e-9)
    w_le_o = bool(
        composite.weighted_composite
        <= composite.optimistic_composite + 1e-9)
    return UncertaintyLayerV8Witness(
        schema=W60_UNCERTAINTY_V8_SCHEMA_VERSION,
        composite_cid=str(composite.cid()),
        n_components=int(composite.n_components),
        hidden_aware=bool(composite.hidden_aware),
        cache_aware=bool(composite.cache_aware),
        retrieval_aware=bool(composite.retrieval_aware),
        replay_aware=bool(composite.replay_aware),
        pessimistic_le_weighted=bool(p_le_w),
        weighted_le_optimistic=bool(w_le_o),
    )


__all__ = [
    "W60_UNCERTAINTY_V8_SCHEMA_VERSION",
    "ReplayAwareWeightedComposite",
    "UncertaintyLayerV8Witness",
    "compose_uncertainty_report_v8",
    "emit_uncertainty_v8_witness",
]
