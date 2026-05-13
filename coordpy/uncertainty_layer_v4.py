"""W56 M-supporting — Uncertainty Layer V4.

Extends W55 Uncertainty V3 with:

* **substrate_fidelity** — a new per-component confidence input
  derived from the tiny substrate's hidden state cosine to a
  reference.
* **substrate-weighted composite** — each component's confidence
  is scaled by its substrate fidelity in addition to its trust
  scalar; low-substrate components are down-weighted.

V4 reduces to V3 when substrate_fidelity = 1.0 for all
components.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


W56_UNCERTAINTY_V4_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v4.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class SubstrateWeightedComposite:
    """Composite confidence with substrate fidelity weighting.

    Computes:
      composite = geomean(c_i for i in components)
      weighted_composite = sum_i (w_i × c_i) where
        w_i = (trust_i × substrate_fidelity_i) /
              sum_j (trust_j × substrate_fidelity_j)
    """

    composite: float
    weighted_composite: float
    n_components: int
    substrate_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W56_UNCERTAINTY_V4_SCHEMA_VERSION,
            "composite": float(round(self.composite, 12)),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "n_components": int(self.n_components),
            "substrate_aware": bool(self.substrate_aware),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v4_composite",
            "composite": self.to_dict()})


def compose_uncertainty_report_v4(
        *,
        component_confidences: Mapping[str, float],
        trust_weights: Mapping[str, float],
        substrate_fidelities: Mapping[str, float],
) -> SubstrateWeightedComposite:
    if not component_confidences:
        return SubstrateWeightedComposite(
            composite=1.0,
            weighted_composite=1.0,
            n_components=0,
            substrate_aware=False,
        )
    keys = list(component_confidences.keys())
    cs = [
        float(component_confidences.get(k, 0.0))
        for k in keys]
    ts = [
        float(trust_weights.get(k, 1.0))
        for k in keys]
    sfs = [
        float(substrate_fidelities.get(k, 1.0))
        for k in keys]
    substrate_aware = any(abs(s - 1.0) > 1e-9 for s in sfs)
    log_sum = 0.0
    for c in cs:
        log_sum += math.log(max(1e-12, float(c)))
    composite = float(math.exp(log_sum / float(len(cs))))
    w_unn = [float(t) * float(s) for t, s in zip(ts, sfs)]
    z = sum(w_unn) or 1.0
    ws = [float(w / z) for w in w_unn]
    weighted = float(sum(
        w * c for w, c in zip(ws, cs)))
    weighted = float(max(0.0, min(1.0, weighted)))
    return SubstrateWeightedComposite(
        composite=float(max(0.0, min(1.0, composite))),
        weighted_composite=float(weighted),
        n_components=int(len(cs)),
        substrate_aware=bool(substrate_aware),
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyLayerV4Witness:
    schema: str
    composite_cid: str
    composite: float
    weighted_composite: float
    substrate_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "composite": float(round(self.composite, 12)),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "substrate_aware": bool(self.substrate_aware),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v4_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v4_witness(
        composite: SubstrateWeightedComposite,
) -> UncertaintyLayerV4Witness:
    return UncertaintyLayerV4Witness(
        schema=W56_UNCERTAINTY_V4_SCHEMA_VERSION,
        composite_cid=composite.cid(),
        composite=composite.composite,
        weighted_composite=composite.weighted_composite,
        substrate_aware=composite.substrate_aware,
    )


__all__ = [
    "W56_UNCERTAINTY_V4_SCHEMA_VERSION",
    "SubstrateWeightedComposite",
    "UncertaintyLayerV4Witness",
    "compose_uncertainty_report_v4",
    "emit_uncertainty_v4_witness",
]
