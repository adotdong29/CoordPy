"""W69 M15 — Uncertainty Layer V17.

Strictly extends W68's ``coordpy.uncertainty_layer_v16``. V16 had
15 weighting axes. V17 adds a 16th:
``multi_branch_rejoin_resolution_fidelity``.

When ``multi_branch_rejoin_resolution_fidelities = None`` (or all
1.0) V17 reduces to V16 (no multi-branch-rejoin awareness).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .uncertainty_layer_v16 import (
    PartialContradictionAwareWeightedComposite,
    compose_uncertainty_report_v16,
)
from .tiny_substrate_v3 import _sha256_hex


W69_UNCERTAINTY_V17_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v17.v1")


@dataclasses.dataclass(frozen=True)
class MultiBranchRejoinAwareWeightedComposite:
    schema: str
    inner_v16: PartialContradictionAwareWeightedComposite
    weighted_composite: float
    multi_branch_rejoin_aware: bool
    n_axes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v16_cid": str(self.inner_v16.cid()),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "multi_branch_rejoin_aware": bool(
                self.multi_branch_rejoin_aware),
            "n_axes": int(self.n_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v17_composite",
            "composite": self.to_dict()})


def compose_uncertainty_report_v17(
        *,
        multi_branch_rejoin_resolution_fidelities: (
            Sequence[float] | None) = None,
        **v16_kwargs: Any,
) -> MultiBranchRejoinAwareWeightedComposite:
    inner = compose_uncertainty_report_v16(**v16_kwargs)
    confidences = list(v16_kwargs.get("confidences", []))
    n = len(confidences)
    if multi_branch_rejoin_resolution_fidelities is None:
        mbrs = [1.0] * n
    else:
        mbrs = list(multi_branch_rejoin_resolution_fidelities)
        mbrs = mbrs + [1.0] * max(0, n - len(mbrs))
    mbr_aware = any(
        float(x) < 1.0 - 1e-12 for x in mbrs[:n])
    if n == 0:
        w_comp = 0.0
    else:
        mean_mbr = float(sum(mbrs[:n])) / float(n)
        w_comp = float(
            inner.weighted_composite) * float(mean_mbr)
    return MultiBranchRejoinAwareWeightedComposite(
        schema=W69_UNCERTAINTY_V17_SCHEMA_VERSION,
        inner_v16=inner,
        weighted_composite=float(w_comp),
        multi_branch_rejoin_aware=bool(mbr_aware),
        n_axes=16,
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyV17Witness:
    schema: str
    composite_cid: str
    n_axes: int
    multi_branch_rejoin_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "n_axes": int(self.n_axes),
            "multi_branch_rejoin_aware": bool(
                self.multi_branch_rejoin_aware),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v17_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v17_witness(
        composite: MultiBranchRejoinAwareWeightedComposite,
) -> UncertaintyV17Witness:
    return UncertaintyV17Witness(
        schema=W69_UNCERTAINTY_V17_SCHEMA_VERSION,
        composite_cid=str(composite.cid()),
        n_axes=int(composite.n_axes),
        multi_branch_rejoin_aware=bool(
            composite.multi_branch_rejoin_aware),
    )


__all__ = [
    "W69_UNCERTAINTY_V17_SCHEMA_VERSION",
    "MultiBranchRejoinAwareWeightedComposite",
    "compose_uncertainty_report_v17",
    "UncertaintyV17Witness",
    "emit_uncertainty_v17_witness",
]
