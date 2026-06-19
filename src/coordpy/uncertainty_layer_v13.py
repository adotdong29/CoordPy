"""W65 M17 — Uncertainty Layer V13.

Strictly extends W64's ``coordpy.uncertainty_layer_v12``. V12 had
11 weighting axes. V13 adds a 12th: ``team_coordination_fidelity``.

When ``team_coordination_fidelities = None`` (or all 1.0) V13
reduces to V12 (no team-coordination awareness).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .uncertainty_layer_v12 import (
    ReplayDominancePrimaryAwareWeightedComposite,
    compose_uncertainty_report_v12,
)
from .tiny_substrate_v3 import _sha256_hex


W65_UNCERTAINTY_V13_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v13.v1")


@dataclasses.dataclass(frozen=True)
class TeamCoordinationAwareWeightedComposite:
    schema: str
    inner_v12: ReplayDominancePrimaryAwareWeightedComposite
    weighted_composite: float
    team_coordination_aware: bool
    n_axes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v12_cid": str(self.inner_v12.cid()),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "team_coordination_aware": bool(
                self.team_coordination_aware),
            "n_axes": int(self.n_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v13_composite",
            "composite": self.to_dict()})


def compose_uncertainty_report_v13(
        *, confidences: Sequence[float],
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
        replay_dominance_primary_fidelities: (
            Sequence[float] | None) = None,
        team_coordination_fidelities: (
            Sequence[float] | None) = None,
) -> TeamCoordinationAwareWeightedComposite:
    inner = compose_uncertainty_report_v12(
        confidences=list(confidences), trusts=list(trusts),
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
        hidden_wins_fidelities=(
            list(hidden_wins_fidelities)
            if hidden_wins_fidelities is not None else None),
        replay_dominance_primary_fidelities=(
            list(replay_dominance_primary_fidelities)
            if replay_dominance_primary_fidelities is not None
            else None),
    )
    n = len(confidences)
    if team_coordination_fidelities is None:
        tcs = [1.0] * n
    else:
        tcs = list(team_coordination_fidelities)
        tcs = tcs + [1.0] * max(0, n - len(tcs))
    tc_aware = any(float(x) < 1.0 - 1e-12 for x in tcs[:n])
    if n == 0:
        w_comp = 0.0
    else:
        mean_tc = float(sum(tcs[:n])) / float(n)
        w_comp = float(
            inner.weighted_composite) * float(mean_tc)
    return TeamCoordinationAwareWeightedComposite(
        schema=W65_UNCERTAINTY_V13_SCHEMA_VERSION,
        inner_v12=inner,
        weighted_composite=float(w_comp),
        team_coordination_aware=bool(tc_aware),
        n_axes=12,
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyV13Witness:
    schema: str
    composite_cid: str
    n_axes: int
    team_coordination_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "n_axes": int(self.n_axes),
            "team_coordination_aware": bool(
                self.team_coordination_aware),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v13_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v13_witness(
        composite: TeamCoordinationAwareWeightedComposite,
) -> UncertaintyV13Witness:
    return UncertaintyV13Witness(
        schema=W65_UNCERTAINTY_V13_SCHEMA_VERSION,
        composite_cid=str(composite.cid()),
        n_axes=int(composite.n_axes),
        team_coordination_aware=bool(
            composite.team_coordination_aware),
    )


__all__ = [
    "W65_UNCERTAINTY_V13_SCHEMA_VERSION",
    "TeamCoordinationAwareWeightedComposite",
    "compose_uncertainty_report_v13",
    "UncertaintyV13Witness",
    "emit_uncertainty_v13_witness",
]
