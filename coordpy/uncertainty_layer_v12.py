"""W64 M17 — Uncertainty Layer V12.

Strictly extends W63's ``coordpy.uncertainty_layer_v11``. V11 had
10 weighting axes. V12 adds an 11th: ``replay_dominance_primary_fidelity``.

When ``replay_dominance_primary_fidelities = None`` (or all 1.0)
V12 reduces to V11 (no replay-dominance-primary awareness).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .uncertainty_layer_v11 import (
    HiddenWinsAwareWeightedComposite,
    compose_uncertainty_report_v11,
)
from .tiny_substrate_v3 import _sha256_hex


W64_UNCERTAINTY_V12_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v12.v1")


@dataclasses.dataclass(frozen=True)
class ReplayDominancePrimaryAwareWeightedComposite:
    schema: str
    inner_v11: HiddenWinsAwareWeightedComposite
    weighted_composite: float
    replay_dominance_primary_aware: bool
    n_axes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v11_cid": str(self.inner_v11.cid()),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "replay_dominance_primary_aware": bool(
                self.replay_dominance_primary_aware),
            "n_axes": int(self.n_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v12_composite",
            "composite": self.to_dict()})


def compose_uncertainty_report_v12(
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
        replay_dominance_primary_fidelities: (
            Sequence[float] | None) = None,
) -> ReplayDominancePrimaryAwareWeightedComposite:
    inner = compose_uncertainty_report_v11(
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
        hidden_wins_fidelities=(
            list(hidden_wins_fidelities)
            if hidden_wins_fidelities is not None
            else None),
    )
    n = len(confidences)
    if replay_dominance_primary_fidelities is None:
        rdps = [1.0] * n
    else:
        rdps = list(replay_dominance_primary_fidelities)
        rdps = rdps + [1.0] * max(0, n - len(rdps))
    rdp_aware = any(
        float(x) < 1.0 - 1e-12 for x in rdps[:n])
    if n == 0:
        w_comp = 0.0
    else:
        mean_rdp = float(sum(rdps[:n])) / float(n)
        w_comp = float(
            inner.weighted_composite) * float(mean_rdp)
    return ReplayDominancePrimaryAwareWeightedComposite(
        schema=W64_UNCERTAINTY_V12_SCHEMA_VERSION,
        inner_v11=inner,
        weighted_composite=float(w_comp),
        replay_dominance_primary_aware=bool(rdp_aware),
        n_axes=11,
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyV12Witness:
    schema: str
    composite_cid: str
    n_axes: int
    replay_dominance_primary_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "n_axes": int(self.n_axes),
            "replay_dominance_primary_aware": bool(
                self.replay_dominance_primary_aware),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v12_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v12_witness(
        composite: ReplayDominancePrimaryAwareWeightedComposite,
) -> UncertaintyV12Witness:
    return UncertaintyV12Witness(
        schema=W64_UNCERTAINTY_V12_SCHEMA_VERSION,
        composite_cid=str(composite.cid()),
        n_axes=int(composite.n_axes),
        replay_dominance_primary_aware=bool(
            composite.replay_dominance_primary_aware),
    )


__all__ = [
    "W64_UNCERTAINTY_V12_SCHEMA_VERSION",
    "ReplayDominancePrimaryAwareWeightedComposite",
    "compose_uncertainty_report_v12",
    "UncertaintyV12Witness",
    "emit_uncertainty_v12_witness",
]
