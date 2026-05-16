"""W63 M18 — Uncertainty Layer V11.

Strictly extends W62's ``coordpy.uncertainty_layer_v10``. V10 had
9 weighting axes. V11 adds a 10th: ``hidden_wins_fidelity``.

When ``hidden_wins_fidelities = None`` (or all 1.0) V11 reduces
to V10 (no hidden-wins awareness).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .uncertainty_layer_v10 import (
    ReplayDominanceAwareWeightedComposite,
    compose_uncertainty_report_v10,
)
from .tiny_substrate_v3 import _sha256_hex


W63_UNCERTAINTY_V11_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v11.v1")


@dataclasses.dataclass(frozen=True)
class HiddenWinsAwareWeightedComposite:
    schema: str
    inner_v10: ReplayDominanceAwareWeightedComposite
    weighted_composite: float
    hidden_wins_aware: bool
    n_axes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v10_cid": str(self.inner_v10.cid()),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "hidden_wins_aware": bool(self.hidden_wins_aware),
            "n_axes": int(self.n_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v11_composite",
            "composite": self.to_dict()})


def compose_uncertainty_report_v11(
        *,
        confidences: Sequence[float],
        trusts: Sequence[float],
        substrate_fidelities: Sequence[float],
        hidden_state_fidelities: Sequence[float],
        cache_reuse_fidelities: Sequence[float],
        retrieval_fidelities: Sequence[float],
        replay_fidelities: Sequence[float],
        attention_pattern_fidelities: (
            Sequence[float] | None) = None,
        replay_dominance_fidelities: (
            Sequence[float] | None) = None,
        hidden_wins_fidelities: (
            Sequence[float] | None) = None,
) -> HiddenWinsAwareWeightedComposite:
    inner = compose_uncertainty_report_v10(
        confidences=list(confidences),
        trusts=list(trusts),
        substrate_fidelities=list(substrate_fidelities),
        hidden_state_fidelities=list(hidden_state_fidelities),
        cache_reuse_fidelities=list(cache_reuse_fidelities),
        retrieval_fidelities=list(retrieval_fidelities),
        replay_fidelities=list(replay_fidelities),
        attention_pattern_fidelities=(
            list(attention_pattern_fidelities)
            if attention_pattern_fidelities is not None
            else None),
        replay_dominance_fidelities=(
            list(replay_dominance_fidelities)
            if replay_dominance_fidelities is not None
            else None),
    )
    n = len(confidences)
    if hidden_wins_fidelities is None:
        hws = [1.0] * n
    else:
        hws = list(hidden_wins_fidelities)
        hws = hws + [1.0] * max(0, n - len(hws))
    hw_aware = any(float(x) < 1.0 - 1e-12 for x in hws[:n])
    if n == 0:
        w_comp = 0.0
    else:
        mean_hw = float(sum(hws[:n])) / float(n)
        w_comp = float(inner.weighted_composite) * float(mean_hw)
    return HiddenWinsAwareWeightedComposite(
        schema=W63_UNCERTAINTY_V11_SCHEMA_VERSION,
        inner_v10=inner,
        weighted_composite=float(w_comp),
        hidden_wins_aware=bool(hw_aware),
        n_axes=10,
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyV11Witness:
    schema: str
    composite_cid: str
    n_axes: int
    hidden_wins_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "n_axes": int(self.n_axes),
            "hidden_wins_aware": bool(self.hidden_wins_aware),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v11_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v11_witness(
        composite: HiddenWinsAwareWeightedComposite,
) -> UncertaintyV11Witness:
    return UncertaintyV11Witness(
        schema=W63_UNCERTAINTY_V11_SCHEMA_VERSION,
        composite_cid=str(composite.cid()),
        n_axes=int(composite.n_axes),
        hidden_wins_aware=bool(composite.hidden_wins_aware),
    )


__all__ = [
    "W63_UNCERTAINTY_V11_SCHEMA_VERSION",
    "HiddenWinsAwareWeightedComposite",
    "compose_uncertainty_report_v11",
    "UncertaintyV11Witness",
    "emit_uncertainty_v11_witness",
]
