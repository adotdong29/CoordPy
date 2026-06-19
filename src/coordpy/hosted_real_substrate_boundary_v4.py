"""W71 H5 — Hosted ↔ Real-Substrate Boundary V4.

Strictly extends W70's
``coordpy.hosted_real_substrate_boundary_v3``. V4 adds:

* **Three new blocked-axes at the hosted surface** —
  ``delayed_repair_trajectory_cid``,
  ``restart_dominance_per_layer``,
  ``delayed_repair_gate_per_layer``.
* **Restart-dominance falsifier** — same shape as V3, but for the
  new W71 V16 axes.
* **Frontier-blocked axes** — V4 keeps W70's frontier-blocked set
  unchanged (third-party transformer hidden-state read, KV bytes
  read, attention weights read).

Honest scope (W71)
------------------

* The wall remains a **structural** assertion.
* W71 does NOT pierce the hosted substrate boundary; the boundary
  V4 records this as a content-addressed invariant.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_real_substrate_boundary import (
    W68_HOSTED_PLANE_AVAILABLE_AXES,
)
from .hosted_real_substrate_boundary_v3 import (
    HostedRealSubstrateBoundaryV3,
    W70_HOSTED_PLANE_BLOCKED_AXES_V3,
    W70_FRONTIER_BLOCKED_AXES,
    build_default_hosted_real_substrate_boundary_v3,
)
from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v16 import (
    W71_SUBSTRATE_V16_CAPABILITY_AXES,
)
from .tiny_substrate_v3 import _sha256_hex


W71_HOSTED_REAL_SUBSTRATE_BOUNDARY_V4_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary_v4.v1")

# Axes blocked at the hosted surface (V71 adds 3 new V16 axes).
W71_HOSTED_PLANE_BLOCKED_AXES_V4_NEW: tuple[str, ...] = (
    "delayed_repair_trajectory_cid",
    "restart_dominance_per_layer",
    "delayed_repair_gate_per_layer",
)
W71_HOSTED_PLANE_BLOCKED_AXES_V4: tuple[str, ...] = (
    *W70_HOSTED_PLANE_BLOCKED_AXES_V3,
    *W71_HOSTED_PLANE_BLOCKED_AXES_V4_NEW,
)

# Frontier-blocked axes — what V16 also does not satisfy.
W71_FRONTIER_BLOCKED_AXES: tuple[str, ...] = (
    *W70_FRONTIER_BLOCKED_AXES,
)


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV4:
    schema: str
    inner_v3: HostedRealSubstrateBoundaryV3
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v16_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    frontier_blocked_axes: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v3_cid": str(self.inner_v3.cid()),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v16_axes": list(
                self.real_substrate_v16_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "frontier_blocked_axes": list(
                self.frontier_blocked_axes),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_v4",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary_v4(
) -> HostedRealSubstrateBoundaryV4:
    inner_v3 = (
        build_default_hosted_real_substrate_boundary_v3())
    return HostedRealSubstrateBoundaryV4(
        schema=(
            W71_HOSTED_REAL_SUBSTRATE_BOUNDARY_V4_SCHEMA_VERSION),
        inner_v3=inner_v3,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W71_HOSTED_PLANE_BLOCKED_AXES_V4,
        real_substrate_v16_axes=tuple(
            W71_SUBSTRATE_V16_CAPABILITY_AXES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        frontier_blocked_axes=W71_FRONTIER_BLOCKED_AXES,
        rationale=(
            "Hosted APIs expose text, optional logprobs, and "
            "optional prefix-cache hit accounting at the HTTP "
            "surface. They do NOT expose hidden states, KV-cache "
            "bytes, attention weights, or any per-(layer, head, "
            "slot) tensor — including the four W69 V14 axes, the "
            "three W70 V15 axes, AND the three new W71 V16 axes "
            "(delayed_repair_trajectory_cid, "
            "restart_dominance_per_layer, "
            "delayed_repair_gate_per_layer). The W71 V16 in-repo "
            "substrate is the only runtime that honestly exposes "
            "the full V16 capability set. The third-party-hosted-"
            "model substrate access remains blocked at the "
            "frontier; W71 carries the W70 frontier_blocked_axes "
            "set forward unchanged."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV4Falsifier:
    schema: str
    boundary_v4_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v4_cid": str(self.boundary_v4_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_boundary_v4_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_substrate_boundary_v4_falsifier(
        *, boundary: HostedRealSubstrateBoundaryV4,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryV4Falsifier:
    """Returns 0 iff the claim is consistent with the V4 wall;
    1 if the claim violates the V4 wall."""
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryV4Falsifier(
        schema=(
            W71_HOSTED_REAL_SUBSTRATE_BOUNDARY_V4_SCHEMA_VERSION),
        boundary_v4_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReportV4:
    schema: str
    boundary_v4_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v4_cid": str(self.boundary_v4_cid),
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
                "hosted_real_substrate_wall_report_v4",
            "report": self.to_dict()})


def build_wall_report_v4(
        *, boundary: HostedRealSubstrateBoundaryV4,
) -> HostedRealSubstrateWallReportV4:
    real_only = tuple(
        a for a in boundary.real_substrate_v16_axes
        if a not in tuple(boundary.available_axes))
    return HostedRealSubstrateWallReportV4(
        schema=(
            W71_HOSTED_REAL_SUBSTRATE_BOUNDARY_V4_SCHEMA_VERSION),
        boundary_v4_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        blocked_at_frontier_axes=tuple(
            boundary.frontier_blocked_axes),
    )


__all__ = [
    "W71_HOSTED_REAL_SUBSTRATE_BOUNDARY_V4_SCHEMA_VERSION",
    "W71_HOSTED_PLANE_BLOCKED_AXES_V4_NEW",
    "W71_HOSTED_PLANE_BLOCKED_AXES_V4",
    "W71_FRONTIER_BLOCKED_AXES",
    "HostedRealSubstrateBoundaryV4",
    "build_default_hosted_real_substrate_boundary_v4",
    "HostedRealSubstrateBoundaryV4Falsifier",
    "probe_hosted_real_substrate_boundary_v4_falsifier",
    "HostedRealSubstrateWallReportV4",
    "build_wall_report_v4",
]
