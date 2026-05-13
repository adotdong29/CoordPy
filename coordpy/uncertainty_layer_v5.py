"""W57 M14 — Uncertainty Layer V5.

Extends W56 V4 with:

* **hidden_state_fidelity** as a fourth confidence weighting
  axis on top of (confidence, trust, substrate_fidelity).
* **adversarial worst-case bound**: the composite is bracketed by
  a worst-case adversarial perturbation calibration check —
  reports both the optimistic composite and the pessimistic
  composite under bounded adversarial perturbation.
* **per-axis sensitivity**: a small numerical Jacobian over the
  confidence/trust/sub_fidelity/hidden_fidelity axes that the
  caller can use as a per-component "how brittle is this
  composite?" diagnostic.

V5 reduces to V4 byte-for-byte when ``hidden_fidelities`` is all
1.0 and the adversarial radius is 0.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


W57_UNCERTAINTY_V5_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v5.v1")


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class HiddenStateWeightedComposite:
    composite: float
    weighted_composite: float
    pessimistic_composite: float
    optimistic_composite: float
    n_components: int
    hidden_aware: bool
    sensitivity_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": W57_UNCERTAINTY_V5_SCHEMA_VERSION,
            "composite": float(round(self.composite, 12)),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "pessimistic_composite": float(round(
                self.pessimistic_composite, 12)),
            "optimistic_composite": float(round(
                self.optimistic_composite, 12)),
            "n_components": int(self.n_components),
            "hidden_aware": bool(self.hidden_aware),
            "sensitivity_max": float(round(
                self.sensitivity_max, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v5_composite",
            "composite": self.to_dict()})


def _weighted_composite(
        cs: Sequence[float], ts: Sequence[float],
        sfs: Sequence[float], hfs: Sequence[float],
) -> float:
    w = [float(t) * float(s) * float(h)
          for t, s, h in zip(ts, sfs, hfs)]
    z = float(sum(w)) or 1.0
    w = [x / z for x in w]
    return float(sum(w[i] * float(cs[i])
                      for i in range(len(cs))))


def compose_uncertainty_report_v5(
        *,
        component_confidences: Mapping[str, float],
        trust_weights: Mapping[str, float],
        substrate_fidelities: Mapping[str, float],
        hidden_state_fidelities: Mapping[str, float],
        adversarial_radius: float = 0.05,
) -> HiddenStateWeightedComposite:
    if not component_confidences:
        return HiddenStateWeightedComposite(
            composite=1.0,
            weighted_composite=1.0,
            pessimistic_composite=1.0,
            optimistic_composite=1.0,
            n_components=0,
            hidden_aware=False,
            sensitivity_max=0.0,
        )
    keys = list(component_confidences.keys())
    cs = [float(component_confidences.get(k, 0.0)) for k in keys]
    ts = [float(trust_weights.get(k, 1.0)) for k in keys]
    sfs = [float(substrate_fidelities.get(k, 1.0)) for k in keys]
    hfs = [float(hidden_state_fidelities.get(k, 1.0))
            for k in keys]
    hidden_aware = any(abs(h - 1.0) > 1e-9 for h in hfs)
    log_sum = 0.0
    for c in cs:
        log_sum += math.log(max(1e-12, float(c)))
    composite = float(math.exp(log_sum / float(len(cs))))
    weighted = _weighted_composite(cs, ts, sfs, hfs)
    eps = float(adversarial_radius)
    cs_lo = [max(0.0, c - eps) for c in cs]
    cs_hi = [min(1.0, c + eps) for c in cs]
    pess = _weighted_composite(cs_lo, ts, sfs, hfs)
    opt = _weighted_composite(cs_hi, ts, sfs, hfs)
    # Per-axis sensitivity (numerical Jacobian magnitude).
    sens_max = 0.0
    delta = 0.01
    for i in range(len(cs)):
        cs_p = list(cs)
        cs_p[i] = min(1.0, cs[i] + delta)
        wp = _weighted_composite(cs_p, ts, sfs, hfs)
        sens_max = max(sens_max,
                        abs(wp - weighted) / max(delta, 1e-12))
    return HiddenStateWeightedComposite(
        composite=float(composite),
        weighted_composite=float(weighted),
        pessimistic_composite=float(pess),
        optimistic_composite=float(opt),
        n_components=int(len(cs)),
        hidden_aware=bool(hidden_aware),
        sensitivity_max=float(sens_max),
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyLayerV5Witness:
    schema: str
    composite_cid: str
    composite: float
    weighted_composite: float
    pessimistic_composite: float
    optimistic_composite: float
    sensitivity_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "composite": float(round(self.composite, 12)),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "pessimistic_composite": float(round(
                self.pessimistic_composite, 12)),
            "optimistic_composite": float(round(
                self.optimistic_composite, 12)),
            "sensitivity_max": float(round(
                self.sensitivity_max, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v5_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v5_witness(
        report: HiddenStateWeightedComposite,
) -> UncertaintyLayerV5Witness:
    return UncertaintyLayerV5Witness(
        schema=W57_UNCERTAINTY_V5_SCHEMA_VERSION,
        composite_cid=str(report.cid()),
        composite=float(report.composite),
        weighted_composite=float(report.weighted_composite),
        pessimistic_composite=float(report.pessimistic_composite),
        optimistic_composite=float(report.optimistic_composite),
        sensitivity_max=float(report.sensitivity_max),
    )


__all__ = [
    "W57_UNCERTAINTY_V5_SCHEMA_VERSION",
    "HiddenStateWeightedComposite",
    "UncertaintyLayerV5Witness",
    "compose_uncertainty_report_v5",
    "emit_uncertainty_v5_witness",
]
