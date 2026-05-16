"""W68 M16 — Uncertainty Layer V16.

Strictly extends W67's ``coordpy.uncertainty_layer_v15``. V15 had
14 weighting axes. V16 adds a 15th:
``partial_contradiction_resolution_fidelity``.

When ``partial_contradiction_resolution_fidelities = None`` (or all
1.0) V16 reduces to V15 (no partial-contradiction awareness).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

from .uncertainty_layer_v15 import (
    BranchMergeAwareWeightedComposite,
    compose_uncertainty_report_v15,
)
from .tiny_substrate_v3 import _sha256_hex


W68_UNCERTAINTY_V16_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v16.v1")


@dataclasses.dataclass(frozen=True)
class PartialContradictionAwareWeightedComposite:
    schema: str
    inner_v15: BranchMergeAwareWeightedComposite
    weighted_composite: float
    partial_contradiction_aware: bool
    n_axes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v15_cid": str(self.inner_v15.cid()),
            "weighted_composite": float(round(
                self.weighted_composite, 12)),
            "partial_contradiction_aware": bool(
                self.partial_contradiction_aware),
            "n_axes": int(self.n_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v16_composite",
            "composite": self.to_dict()})


def compose_uncertainty_report_v16(
        *,
        partial_contradiction_resolution_fidelities: (
            Sequence[float] | None) = None,
        **v15_kwargs: Any,
) -> PartialContradictionAwareWeightedComposite:
    inner = compose_uncertainty_report_v15(**v15_kwargs)
    confidences = list(v15_kwargs.get("confidences", []))
    n = len(confidences)
    if partial_contradiction_resolution_fidelities is None:
        pcrs = [1.0] * n
    else:
        pcrs = list(partial_contradiction_resolution_fidelities)
        pcrs = pcrs + [1.0] * max(0, n - len(pcrs))
    pcr_aware = any(
        float(x) < 1.0 - 1e-12 for x in pcrs[:n])
    if n == 0:
        w_comp = 0.0
    else:
        mean_pcr = float(sum(pcrs[:n])) / float(n)
        w_comp = float(
            inner.weighted_composite) * float(mean_pcr)
    return PartialContradictionAwareWeightedComposite(
        schema=W68_UNCERTAINTY_V16_SCHEMA_VERSION,
        inner_v15=inner,
        weighted_composite=float(w_comp),
        partial_contradiction_aware=bool(pcr_aware),
        n_axes=15,
    )


@dataclasses.dataclass(frozen=True)
class UncertaintyV16Witness:
    schema: str
    composite_cid: str
    n_axes: int
    partial_contradiction_aware: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "composite_cid": str(self.composite_cid),
            "n_axes": int(self.n_axes),
            "partial_contradiction_aware": bool(
                self.partial_contradiction_aware),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "uncertainty_v16_witness",
            "witness": self.to_dict()})


def emit_uncertainty_v16_witness(
        composite: PartialContradictionAwareWeightedComposite,
) -> UncertaintyV16Witness:
    return UncertaintyV16Witness(
        schema=W68_UNCERTAINTY_V16_SCHEMA_VERSION,
        composite_cid=str(composite.cid()),
        n_axes=int(composite.n_axes),
        partial_contradiction_aware=bool(
            composite.partial_contradiction_aware),
    )


__all__ = [
    "W68_UNCERTAINTY_V16_SCHEMA_VERSION",
    "PartialContradictionAwareWeightedComposite",
    "compose_uncertainty_report_v16",
    "UncertaintyV16Witness",
    "emit_uncertainty_v16_witness",
]
