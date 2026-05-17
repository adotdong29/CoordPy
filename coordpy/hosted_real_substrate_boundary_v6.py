"""W73 H5 — Hosted ↔ Real-Substrate Boundary V6.

Strictly extends W72's
``coordpy.hosted_real_substrate_boundary_v5``. V6 adds:

* **Three new blocked-axes at the hosted surface** —
  ``replacement_repair_trajectory_cid``,
  ``replacement_after_ctr_per_layer``,
  ``replacement_pressure_gate_per_layer``.
* **Replacement-pressure falsifier** — same shape as V5, but for
  the new W73 V18 axes.
* **Frontier-blocked axes** — V6 keeps W70's frontier-blocked set
  unchanged (third-party transformer hidden-state read, KV bytes
  read, attention weights read).

Honest scope (W73)
------------------

* The wall remains a **structural** assertion.
* W73 does NOT pierce the hosted substrate boundary; the boundary
  V6 records this as a content-addressed invariant.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_real_substrate_boundary import (
    W68_HOSTED_PLANE_AVAILABLE_AXES,
)
from .hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from .hosted_real_substrate_boundary_v5 import (
    HostedRealSubstrateBoundaryV5,
    W72_HOSTED_PLANE_BLOCKED_AXES_V5,
    build_default_hosted_real_substrate_boundary_v5,
)
from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v18 import (
    W73_SUBSTRATE_V18_CAPABILITY_AXES,
)
from .tiny_substrate_v3 import _sha256_hex


W73_HOSTED_REAL_SUBSTRATE_BOUNDARY_V6_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary_v6.v1")

# Axes blocked at the hosted surface (V73 adds 3 new V18 axes).
W73_HOSTED_PLANE_BLOCKED_AXES_V6_NEW: tuple[str, ...] = (
    "replacement_repair_trajectory_cid",
    "replacement_after_ctr_per_layer",
    "replacement_pressure_gate_per_layer",
)
W73_HOSTED_PLANE_BLOCKED_AXES_V6: tuple[str, ...] = (
    *W72_HOSTED_PLANE_BLOCKED_AXES_V5,
    *W73_HOSTED_PLANE_BLOCKED_AXES_V6_NEW,
)

# Frontier-blocked axes — what V18 also does not satisfy.
W73_FRONTIER_BLOCKED_AXES: tuple[str, ...] = (
    *W70_FRONTIER_BLOCKED_AXES,
)


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV6:
    schema: str
    inner_v5: HostedRealSubstrateBoundaryV5
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v18_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    frontier_blocked_axes: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v5_cid": str(self.inner_v5.cid()),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v18_axes": list(
                self.real_substrate_v18_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "frontier_blocked_axes": list(
                self.frontier_blocked_axes),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_v6",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary_v6(
) -> HostedRealSubstrateBoundaryV6:
    inner_v5 = (
        build_default_hosted_real_substrate_boundary_v5())
    return HostedRealSubstrateBoundaryV6(
        schema=(
            W73_HOSTED_REAL_SUBSTRATE_BOUNDARY_V6_SCHEMA_VERSION),
        inner_v5=inner_v5,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W73_HOSTED_PLANE_BLOCKED_AXES_V6,
        real_substrate_v18_axes=tuple(
            W73_SUBSTRATE_V18_CAPABILITY_AXES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        frontier_blocked_axes=W73_FRONTIER_BLOCKED_AXES,
        rationale=(
            "Hosted APIs expose text, optional logprobs, and "
            "optional prefix-cache hit accounting at the HTTP "
            "surface. They do NOT expose hidden states, KV-cache "
            "bytes, attention weights, or any per-(layer, head, "
            "slot) tensor — including the four W69 V14 axes, the "
            "three W70 V15 axes, the three W71 V16 axes, the three "
            "W72 V17 axes, AND the three new W73 V18 axes "
            "(replacement_repair_trajectory_cid, "
            "replacement_after_ctr_per_layer, "
            "replacement_pressure_gate_per_layer). The W73 V18 "
            "in-repo substrate is the only runtime that honestly "
            "exposes the full V18 capability set. The third-party-"
            "hosted-model substrate access remains blocked at the "
            "frontier; W73 carries the W70 frontier_blocked_axes "
            "set forward unchanged."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV6Falsifier:
    schema: str
    boundary_v6_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v6_cid": str(self.boundary_v6_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_boundary_v6_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_substrate_boundary_v6_falsifier(
        *, boundary: HostedRealSubstrateBoundaryV6,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryV6Falsifier:
    """Returns 0 iff the claim is consistent with the V6 wall;
    1 if the claim violates the V6 wall."""
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryV6Falsifier(
        schema=(
            W73_HOSTED_REAL_SUBSTRATE_BOUNDARY_V6_SCHEMA_VERSION),
        boundary_v6_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReportV6:
    schema: str
    boundary_v6_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v6_cid": str(self.boundary_v6_cid),
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
                "hosted_real_substrate_wall_report_v6",
            "report": self.to_dict()})


def build_wall_report_v6(
        *, boundary: HostedRealSubstrateBoundaryV6,
) -> HostedRealSubstrateWallReportV6:
    real_only = tuple(
        a for a in boundary.real_substrate_v18_axes
        if a not in tuple(boundary.available_axes))
    return HostedRealSubstrateWallReportV6(
        schema=(
            W73_HOSTED_REAL_SUBSTRATE_BOUNDARY_V6_SCHEMA_VERSION),
        boundary_v6_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        blocked_at_frontier_axes=tuple(
            boundary.frontier_blocked_axes),
    )


__all__ = [
    "W73_HOSTED_REAL_SUBSTRATE_BOUNDARY_V6_SCHEMA_VERSION",
    "W73_HOSTED_PLANE_BLOCKED_AXES_V6_NEW",
    "W73_HOSTED_PLANE_BLOCKED_AXES_V6",
    "W73_FRONTIER_BLOCKED_AXES",
    "HostedRealSubstrateBoundaryV6",
    "build_default_hosted_real_substrate_boundary_v6",
    "HostedRealSubstrateBoundaryV6Falsifier",
    "probe_hosted_real_substrate_boundary_v6_falsifier",
    "HostedRealSubstrateWallReportV6",
    "build_wall_report_v6",
]
