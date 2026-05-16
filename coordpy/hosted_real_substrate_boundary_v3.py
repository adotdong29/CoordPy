"""W70 H5 — Hosted ↔ Real-Substrate Boundary V3.

Strictly extends W69's
``coordpy.hosted_real_substrate_boundary_v2``. V3 adds:

* **Three new blocked-axes at the hosted surface** —
  ``repair_trajectory_cid``, ``dominant_repair_per_layer``,
  ``budget_primary_gate_per_layer``.
* **Repair-dominance falsifier** — same shape as V2, but for the
  new W70 V15 axes.
* **Frontier-blocked axes** — V3 keeps W69's frontier-blocked set
  unchanged (third-party transformer hidden-state read, KV bytes
  read, attention weights read).

Honest scope (W70)
------------------

* The wall remains a **structural** assertion.
* W70 does NOT pierce the hosted substrate boundary; the boundary
  V3 records this as a content-addressed invariant.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_real_substrate_boundary import (
    W68_HOSTED_PLANE_AVAILABLE_AXES,
)
from .hosted_real_substrate_boundary_v2 import (
    HostedRealSubstrateBoundaryV2,
    W69_HOSTED_PLANE_BLOCKED_AXES_V2,
    W69_FRONTIER_BLOCKED_AXES,
    build_default_hosted_real_substrate_boundary_v2,
)
from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v15 import (
    W70_SUBSTRATE_V15_CAPABILITY_AXES,
)
from .tiny_substrate_v3 import _sha256_hex


W70_HOSTED_REAL_SUBSTRATE_BOUNDARY_V3_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary_v3.v1")

# Axes blocked at the hosted surface (V70 adds 3 new V15 axes).
W70_HOSTED_PLANE_BLOCKED_AXES_V3_NEW: tuple[str, ...] = (
    "repair_trajectory_cid",
    "dominant_repair_per_layer",
    "budget_primary_gate_per_layer",
)
W70_HOSTED_PLANE_BLOCKED_AXES_V3: tuple[str, ...] = (
    *W69_HOSTED_PLANE_BLOCKED_AXES_V2,
    *W70_HOSTED_PLANE_BLOCKED_AXES_V3_NEW,
)

# Frontier-blocked axes — what V15 also does not satisfy.
W70_FRONTIER_BLOCKED_AXES: tuple[str, ...] = (
    *W69_FRONTIER_BLOCKED_AXES,
)


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV3:
    schema: str
    inner_v2: HostedRealSubstrateBoundaryV2
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v15_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    frontier_blocked_axes: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v2_cid": str(self.inner_v2.cid()),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v15_axes": list(
                self.real_substrate_v15_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "frontier_blocked_axes": list(
                self.frontier_blocked_axes),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_v3",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary_v3(
) -> HostedRealSubstrateBoundaryV3:
    inner_v2 = (
        build_default_hosted_real_substrate_boundary_v2())
    return HostedRealSubstrateBoundaryV3(
        schema=(
            W70_HOSTED_REAL_SUBSTRATE_BOUNDARY_V3_SCHEMA_VERSION),
        inner_v2=inner_v2,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W70_HOSTED_PLANE_BLOCKED_AXES_V3,
        real_substrate_v15_axes=tuple(
            W70_SUBSTRATE_V15_CAPABILITY_AXES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        frontier_blocked_axes=W70_FRONTIER_BLOCKED_AXES,
        rationale=(
            "Hosted APIs expose text, optional logprobs, and "
            "optional prefix-cache hit accounting at the HTTP "
            "surface. They do NOT expose hidden states, KV-cache "
            "bytes, attention weights, or any per-(layer, head, "
            "slot) tensor — including the four W69 V14 axes "
            "AND the three new W70 V15 axes "
            "(repair_trajectory_cid, dominant_repair_per_layer, "
            "budget_primary_gate_per_layer). The W70 V15 in-repo "
            "substrate is the only runtime that honestly exposes "
            "the full V15 capability set. The third-party-hosted-"
            "model substrate access remains blocked at the "
            "frontier; W70 carries the W69 frontier_blocked_axes "
            "set forward unchanged."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV3Falsifier:
    schema: str
    boundary_v3_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v3_cid": str(self.boundary_v3_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_boundary_v3_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_substrate_boundary_v3_falsifier(
        *, boundary: HostedRealSubstrateBoundaryV3,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryV3Falsifier:
    """Returns 0 iff the claim is consistent with the V3 wall;
    1 if the claim violates the V3 wall."""
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryV3Falsifier(
        schema=(
            W70_HOSTED_REAL_SUBSTRATE_BOUNDARY_V3_SCHEMA_VERSION),
        boundary_v3_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReportV3:
    schema: str
    boundary_v3_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v3_cid": str(self.boundary_v3_cid),
            "hosted_solvable_axes": list(
                self.hosted_solvable_axes),
            "real_substrate_only_axes": list(
                self.real_substrate_only_axes),
            "blocked_at_frontier_axes": list(
                self.blocked_at_frontier_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_wall_report_v3",
            "report": self.to_dict()})


def build_wall_report_v3(
        *, boundary: HostedRealSubstrateBoundaryV3,
) -> HostedRealSubstrateWallReportV3:
    real_only = tuple(
        a for a in boundary.real_substrate_v15_axes
        if a not in tuple(boundary.available_axes))
    return HostedRealSubstrateWallReportV3(
        schema=(
            W70_HOSTED_REAL_SUBSTRATE_BOUNDARY_V3_SCHEMA_VERSION),
        boundary_v3_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        blocked_at_frontier_axes=tuple(
            boundary.frontier_blocked_axes),
    )


__all__ = [
    "W70_HOSTED_REAL_SUBSTRATE_BOUNDARY_V3_SCHEMA_VERSION",
    "W70_HOSTED_PLANE_BLOCKED_AXES_V3_NEW",
    "W70_HOSTED_PLANE_BLOCKED_AXES_V3",
    "W70_FRONTIER_BLOCKED_AXES",
    "HostedRealSubstrateBoundaryV3",
    "build_default_hosted_real_substrate_boundary_v3",
    "HostedRealSubstrateBoundaryV3Falsifier",
    "probe_hosted_real_substrate_boundary_v3_falsifier",
    "HostedRealSubstrateWallReportV3",
    "build_wall_report_v3",
]
