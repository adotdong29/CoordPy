"""W62 — Uncertainty Layer V10.

Strictly extends W61's ``coordpy.uncertainty_layer_v9``. V9 had
8 weighting axes. V10 adds a 9th: ``replay_dominance_fidelity``.

When ``replay_dominance_fidelities = None`` (or all 1.0) V10
reduces to V9 (no replay-dominance awareness).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

from .uncertainty_layer_v9 import (
    AttentionPatternAwareWeightedComposite,
    compose_uncertainty_report_v9,
    W61_UNCERTAINTY_V9_SCHEMA_VERSION,
)
from .tiny_substrate_v3 import _sha256_hex


W62_UNCERTAINTY_V10_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v10.v1")


@dataclasses.dataclass(frozen=True)
class ReplayDominanceAwareWeightedComposite:
    schema: str
    inner_v9: AttentionPatternAwareWeightedComposite
    weighted_composite: float
    replay_dominance_aware: bool
    n_axes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v9_cid": str(self.inner_v9.cid()),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "replay_dominance_aware": bool(
                self.replay_dominance_aware),
            "n_axes": int(self.n_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v10_composite",
            "composite": self.to_dict()})


def compose_uncertainty_report_v10(
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
) -> ReplayDominanceAwareWeightedComposite:
    inner = compose_uncertainty_report_v9(
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
    )
    n = len(confidences)
    if replay_dominance_fidelities is None:
        rds = [1.0] * n
    else:
        rds = list(replay_dominance_fidelities)
        rds = rds + [1.0] * max(0, n - len(rds))
    rd_aware = any(float(x) < 1.0 - 1e-12 for x in rds[:n])
    # Replay-dominance-weighted composite: scale V9 by mean rd.
    if n == 0:
        w_comp = 0.0
    else:
        mean_rd = float(sum(rds[:n])) / float(n)
        w_comp = float(inner.weighted_composite) * float(mean_rd)
    return ReplayDominanceAwareWeightedComposite(
        schema=W62_UNCERTAINTY_V10_SCHEMA_VERSION,
        inner_v9=inner,
        weighted_composite=float(w_comp),
        replay_dominance_aware=bool(rd_aware),
        n_axes=9,
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyV10Witness:
    schema: str
    composite_cid: str
    n_axes: int
    replay_dominance_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "n_axes": int(self.n_axes),
            "replay_dominance_aware": bool(
                self.replay_dominance_aware),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v10_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v10_witness(
        composite: ReplayDominanceAwareWeightedComposite,
) -> UncertaintyV10Witness:
    return UncertaintyV10Witness(
        schema=W62_UNCERTAINTY_V10_SCHEMA_VERSION,
        composite_cid=str(composite.cid()),
        n_axes=int(composite.n_axes),
        replay_dominance_aware=bool(
            composite.replay_dominance_aware),
    )


__all__ = [
    "W62_UNCERTAINTY_V10_SCHEMA_VERSION",
    "ReplayDominanceAwareWeightedComposite",
    "compose_uncertainty_report_v10",
    "UncertaintyV10Witness",
    "emit_uncertainty_v10_witness",
]
