"""W66 M17 — Uncertainty Layer V14.

Strictly extends W65's ``coordpy.uncertainty_layer_v13``. V13 had
12 weighting axes. V14 adds a 13th: ``team_failure_recovery_fidelity``.

When ``team_failure_recovery_fidelities = None`` (or all 1.0) V14
reduces to V13 (no team-failure-recovery awareness).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .uncertainty_layer_v13 import (
    TeamCoordinationAwareWeightedComposite,
    compose_uncertainty_report_v13,
)
from .tiny_substrate_v3 import _sha256_hex


W66_UNCERTAINTY_V14_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v14.v1")


@dataclasses.dataclass(frozen=True)
class TeamFailureRecoveryAwareWeightedComposite:
    schema: str
    inner_v13: TeamCoordinationAwareWeightedComposite
    weighted_composite: float
    team_failure_recovery_aware: bool
    n_axes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v13_cid": str(self.inner_v13.cid()),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "team_failure_recovery_aware": bool(
                self.team_failure_recovery_aware),
            "n_axes": int(self.n_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v14_composite",
            "composite": self.to_dict()})


def compose_uncertainty_report_v14(
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
        team_failure_recovery_fidelities: (
            Sequence[float] | None) = None,
) -> TeamFailureRecoveryAwareWeightedComposite:
    inner = compose_uncertainty_report_v13(
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
        team_coordination_fidelities=(
            list(team_coordination_fidelities)
            if team_coordination_fidelities is not None
            else None),
    )
    n = len(confidences)
    if team_failure_recovery_fidelities is None:
        tfrs = [1.0] * n
    else:
        tfrs = list(team_failure_recovery_fidelities)
        tfrs = tfrs + [1.0] * max(0, n - len(tfrs))
    tfr_aware = any(
        float(x) < 1.0 - 1e-12 for x in tfrs[:n])
    if n == 0:
        w_comp = 0.0
    else:
        mean_tfr = float(sum(tfrs[:n])) / float(n)
        w_comp = float(
            inner.weighted_composite) * float(mean_tfr)
    return TeamFailureRecoveryAwareWeightedComposite(
        schema=W66_UNCERTAINTY_V14_SCHEMA_VERSION,
        inner_v13=inner,
        weighted_composite=float(w_comp),
        team_failure_recovery_aware=bool(tfr_aware),
        n_axes=13,
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyV14Witness:
    schema: str
    composite_cid: str
    n_axes: int
    team_failure_recovery_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "n_axes": int(self.n_axes),
            "team_failure_recovery_aware": bool(
                self.team_failure_recovery_aware),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v14_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v14_witness(
        composite: TeamFailureRecoveryAwareWeightedComposite,
) -> UncertaintyV14Witness:
    return UncertaintyV14Witness(
        schema=W66_UNCERTAINTY_V14_SCHEMA_VERSION,
        composite_cid=str(composite.cid()),
        n_axes=int(composite.n_axes),
        team_failure_recovery_aware=bool(
            composite.team_failure_recovery_aware),
    )


__all__ = [
    "W66_UNCERTAINTY_V14_SCHEMA_VERSION",
    "TeamFailureRecoveryAwareWeightedComposite",
    "compose_uncertainty_report_v14",
    "UncertaintyV14Witness",
    "emit_uncertainty_v14_witness",
]
