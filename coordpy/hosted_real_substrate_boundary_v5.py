"""W72 H5 — Hosted ↔ Real-Substrate Boundary V5.

Strictly extends W71's
``coordpy.hosted_real_substrate_boundary_v4``. V5 adds:

* **Three new blocked-axes at the hosted surface** —
  ``restart_repair_trajectory_cid``,
  ``delayed_rejoin_after_restart_per_layer``,
  ``rejoin_pressure_gate_per_layer``.
* **Rejoin-pressure falsifier** — same shape as V4, but for the
  new W72 V17 axes.
* **Frontier-blocked axes** — V5 keeps W70's frontier-blocked set
  unchanged (third-party transformer hidden-state read, KV bytes
  read, attention weights read).

Honest scope (W72)
------------------

* The wall remains a **structural** assertion.
* W72 does NOT pierce the hosted substrate boundary; the boundary
  V5 records this as a content-addressed invariant.
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
from .hosted_real_substrate_boundary_v4 import (
    HostedRealSubstrateBoundaryV4,
    W71_HOSTED_PLANE_BLOCKED_AXES_V4,
    build_default_hosted_real_substrate_boundary_v4,
)
from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v17 import (
    W72_SUBSTRATE_V17_CAPABILITY_AXES,
)
from .tiny_substrate_v3 import _sha256_hex


W72_HOSTED_REAL_SUBSTRATE_BOUNDARY_V5_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary_v5.v1")

# Axes blocked at the hosted surface (V72 adds 3 new V17 axes).
W72_HOSTED_PLANE_BLOCKED_AXES_V5_NEW: tuple[str, ...] = (
    "restart_repair_trajectory_cid",
    "delayed_rejoin_after_restart_per_layer",
    "rejoin_pressure_gate_per_layer",
)
W72_HOSTED_PLANE_BLOCKED_AXES_V5: tuple[str, ...] = (
    *W71_HOSTED_PLANE_BLOCKED_AXES_V4,
    *W72_HOSTED_PLANE_BLOCKED_AXES_V5_NEW,
)

# Frontier-blocked axes — what V17 also does not satisfy.
W72_FRONTIER_BLOCKED_AXES: tuple[str, ...] = (
    *W70_FRONTIER_BLOCKED_AXES,
)


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV5:
    schema: str
    inner_v4: HostedRealSubstrateBoundaryV4
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v17_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    frontier_blocked_axes: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v4_cid": str(self.inner_v4.cid()),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v17_axes": list(
                self.real_substrate_v17_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "frontier_blocked_axes": list(
                self.frontier_blocked_axes),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_v5",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary_v5(
) -> HostedRealSubstrateBoundaryV5:
    inner_v4 = (
        build_default_hosted_real_substrate_boundary_v4())
    return HostedRealSubstrateBoundaryV5(
        schema=(
            W72_HOSTED_REAL_SUBSTRATE_BOUNDARY_V5_SCHEMA_VERSION),
        inner_v4=inner_v4,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W72_HOSTED_PLANE_BLOCKED_AXES_V5,
        real_substrate_v17_axes=tuple(
            W72_SUBSTRATE_V17_CAPABILITY_AXES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        frontier_blocked_axes=W72_FRONTIER_BLOCKED_AXES,
        rationale=(
            "Hosted APIs expose text, optional logprobs, and "
            "optional prefix-cache hit accounting at the HTTP "
            "surface. They do NOT expose hidden states, KV-cache "
            "bytes, attention weights, or any per-(layer, head, "
            "slot) tensor — including the four W69 V14 axes, the "
            "three W70 V15 axes, the three W71 V16 axes, AND the "
            "three new W72 V17 axes "
            "(restart_repair_trajectory_cid, "
            "delayed_rejoin_after_restart_per_layer, "
            "rejoin_pressure_gate_per_layer). The W72 V17 in-repo "
            "substrate is the only runtime that honestly exposes "
            "the full V17 capability set. The third-party-hosted-"
            "model substrate access remains blocked at the "
            "frontier; W72 carries the W70 frontier_blocked_axes "
            "set forward unchanged."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV5Falsifier:
    schema: str
    boundary_v5_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v5_cid": str(self.boundary_v5_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_boundary_v5_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_substrate_boundary_v5_falsifier(
        *, boundary: HostedRealSubstrateBoundaryV5,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryV5Falsifier:
    """Returns 0 iff the claim is consistent with the V5 wall;
    1 if the claim violates the V5 wall."""
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryV5Falsifier(
        schema=(
            W72_HOSTED_REAL_SUBSTRATE_BOUNDARY_V5_SCHEMA_VERSION),
        boundary_v5_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReportV5:
    schema: str
    boundary_v5_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v5_cid": str(self.boundary_v5_cid),
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
                "hosted_real_substrate_wall_report_v5",
            "report": self.to_dict()})


def build_wall_report_v5(
        *, boundary: HostedRealSubstrateBoundaryV5,
) -> HostedRealSubstrateWallReportV5:
    real_only = tuple(
        a for a in boundary.real_substrate_v17_axes
        if a not in tuple(boundary.available_axes))
    return HostedRealSubstrateWallReportV5(
        schema=(
            W72_HOSTED_REAL_SUBSTRATE_BOUNDARY_V5_SCHEMA_VERSION),
        boundary_v5_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        blocked_at_frontier_axes=tuple(
            boundary.frontier_blocked_axes),
    )


__all__ = [
    "W72_HOSTED_REAL_SUBSTRATE_BOUNDARY_V5_SCHEMA_VERSION",
    "W72_HOSTED_PLANE_BLOCKED_AXES_V5_NEW",
    "W72_HOSTED_PLANE_BLOCKED_AXES_V5",
    "W72_FRONTIER_BLOCKED_AXES",
    "HostedRealSubstrateBoundaryV5",
    "build_default_hosted_real_substrate_boundary_v5",
    "HostedRealSubstrateBoundaryV5Falsifier",
    "probe_hosted_real_substrate_boundary_v5_falsifier",
    "HostedRealSubstrateWallReportV5",
    "build_wall_report_v5",
]
