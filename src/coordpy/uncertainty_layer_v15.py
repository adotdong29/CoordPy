"""W67 M17 — Uncertainty Layer V15.

Strictly extends W66's ``coordpy.uncertainty_layer_v14``. V14 had
13 weighting axes. V15 adds a 14th:
``branch_merge_reconciliation_fidelity``.

When ``branch_merge_reconciliation_fidelities = None`` (or all
1.0) V15 reduces to V14 (no branch-merge awareness).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .uncertainty_layer_v14 import (
    TeamFailureRecoveryAwareWeightedComposite,
    compose_uncertainty_report_v14,
)
from .tiny_substrate_v3 import _sha256_hex


W67_UNCERTAINTY_V15_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v15.v1")


@dataclasses.dataclass(frozen=True)
class BranchMergeAwareWeightedComposite:
    schema: str
    inner_v14: TeamFailureRecoveryAwareWeightedComposite
    weighted_composite: float
    branch_merge_aware: bool
    n_axes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v14_cid": str(self.inner_v14.cid()),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "branch_merge_aware": bool(self.branch_merge_aware),
            "n_axes": int(self.n_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v15_composite",
            "composite": self.to_dict()})


def compose_uncertainty_report_v15(
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
        branch_merge_reconciliation_fidelities: (
            Sequence[float] | None) = None,
) -> BranchMergeAwareWeightedComposite:
    inner = compose_uncertainty_report_v14(
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
        team_failure_recovery_fidelities=(
            list(team_failure_recovery_fidelities)
            if team_failure_recovery_fidelities is not None
            else None),
    )
    n = len(confidences)
    if branch_merge_reconciliation_fidelities is None:
        bmrs = [1.0] * n
    else:
        bmrs = list(branch_merge_reconciliation_fidelities)
        bmrs = bmrs + [1.0] * max(0, n - len(bmrs))
    bmr_aware = any(
        float(x) < 1.0 - 1e-12 for x in bmrs[:n])
    if n == 0:
        w_comp = 0.0
    else:
        mean_bmr = float(sum(bmrs[:n])) / float(n)
        w_comp = float(
            inner.weighted_composite) * float(mean_bmr)
    return BranchMergeAwareWeightedComposite(
        schema=W67_UNCERTAINTY_V15_SCHEMA_VERSION,
        inner_v14=inner,
        weighted_composite=float(w_comp),
        branch_merge_aware=bool(bmr_aware),
        n_axes=14,
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyV15Witness:
    schema: str
    composite_cid: str
    n_axes: int
    branch_merge_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "n_axes": int(self.n_axes),
            "branch_merge_aware": bool(self.branch_merge_aware),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v15_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v15_witness(
        composite: BranchMergeAwareWeightedComposite,
) -> UncertaintyV15Witness:
    return UncertaintyV15Witness(
        schema=W67_UNCERTAINTY_V15_SCHEMA_VERSION,
        composite_cid=str(composite.cid()),
        n_axes=int(composite.n_axes),
        branch_merge_aware=bool(composite.branch_merge_aware),
    )


__all__ = [
    "W67_UNCERTAINTY_V15_SCHEMA_VERSION",
    "BranchMergeAwareWeightedComposite",
    "compose_uncertainty_report_v15",
    "UncertaintyV15Witness",
    "emit_uncertainty_v15_witness",
]
