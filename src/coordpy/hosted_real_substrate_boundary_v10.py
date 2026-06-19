"""W77 H5 — Hosted ↔ Real-Substrate Boundary V10.

Strictly extends W76's
``coordpy.hosted_real_substrate_boundary_v9``. V10 adds:

* **Three new blocked-axes at the hosted surface** —
  ``replacement_after_restart_after_compound_chain_trajectory_cid``,
  ``replacement_after_restart_after_compound_chain_length_per_layer``,
  ``replacement_after_restart_after_compound_chain_pressure_gate_per_layer``.
* **Post-restart-replacement-pressure falsifier** — same shape as
  V9, but for the new W77 V22 axes.
* **Frontier-blocked axes** — V10 keeps W70's frontier-blocked set
  unchanged (third-party transformer hidden-state read, KV bytes
  read, attention weights read).

Honest scope (W77)
------------------

* The wall remains a **structural** assertion.
* W77 does NOT pierce the hosted substrate boundary; the boundary
  V10 records this as a content-addressed invariant.
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
from .hosted_real_substrate_boundary_v9 import (
    HostedRealSubstrateBoundaryV9,
    W76_HOSTED_PLANE_BLOCKED_AXES_V9,
    build_default_hosted_real_substrate_boundary_v9,
)
from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v22 import (
    W77_SUBSTRATE_V22_CAPABILITY_AXES,
)
from .tiny_substrate_v3 import _sha256_hex


W77_HOSTED_REAL_SUBSTRATE_BOUNDARY_V10_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary_v10.v1")

# Axes blocked at the hosted surface (W77 adds 3 new V22 axes).
W77_HOSTED_PLANE_BLOCKED_AXES_V10_NEW: tuple[str, ...] = (
    "replacement_after_restart_after_compound_chain_trajectory_cid",
    "replacement_after_restart_after_compound_chain_length_per_layer",
    "replacement_after_restart_after_compound_chain_pressure_gate_per_layer",
)
W77_HOSTED_PLANE_BLOCKED_AXES_V10: tuple[str, ...] = (
    *W76_HOSTED_PLANE_BLOCKED_AXES_V9,
    *W77_HOSTED_PLANE_BLOCKED_AXES_V10_NEW,
)

# Frontier-blocked axes — V10 carries W70's set forward unchanged.
W77_FRONTIER_BLOCKED_AXES: tuple[str, ...] = (
    *W70_FRONTIER_BLOCKED_AXES,
)


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV10:
    schema: str
    inner_v9: HostedRealSubstrateBoundaryV9
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v22_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    frontier_blocked_axes: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v9_cid": str(self.inner_v9.cid()),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v22_axes": list(
                self.real_substrate_v22_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "frontier_blocked_axes": list(
                self.frontier_blocked_axes),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_v10",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary_v10(
) -> HostedRealSubstrateBoundaryV10:
    inner_v9 = (
        build_default_hosted_real_substrate_boundary_v9())
    return HostedRealSubstrateBoundaryV10(
        schema=(
            W77_HOSTED_REAL_SUBSTRATE_BOUNDARY_V10_SCHEMA_VERSION),
        inner_v9=inner_v9,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W77_HOSTED_PLANE_BLOCKED_AXES_V10,
        real_substrate_v22_axes=tuple(
            W77_SUBSTRATE_V22_CAPABILITY_AXES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        frontier_blocked_axes=W77_FRONTIER_BLOCKED_AXES,
        rationale=(
            "Hosted APIs expose text, optional logprobs, and "
            "optional prefix-cache hit accounting at the HTTP "
            "surface. They do NOT expose hidden states, KV-cache "
            "bytes, attention weights, or any per-(layer, head, "
            "slot) tensor — including the W69-W76 V14..V21 axes "
            "AND the three new W77 V22 axes "
            "(replacement_after_restart_after_compound_chain_"
            "trajectory_cid, replacement_after_restart_after_"
            "compound_chain_length_per_layer, replacement_after_"
            "restart_after_compound_chain_pressure_gate_per_layer). "
            "The W77 V22 in-repo substrate is the only runtime "
            "that honestly exposes the full V22 capability set. "
            "The third-party-hosted-model substrate access "
            "remains blocked at the frontier; W77 carries the W70 "
            "frontier_blocked_axes set forward unchanged."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV10Falsifier:
    schema: str
    boundary_v10_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v10_cid": str(self.boundary_v10_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_boundary_v10_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_substrate_boundary_v10_falsifier(
        *, boundary: HostedRealSubstrateBoundaryV10,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryV10Falsifier:
    """Returns 0 iff the claim is consistent with the V10 wall;
    1 if the claim violates the V10 wall."""
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryV10Falsifier(
        schema=(
            W77_HOSTED_REAL_SUBSTRATE_BOUNDARY_V10_SCHEMA_VERSION),
        boundary_v10_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReportV10:
    schema: str
    boundary_v10_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v10_cid": str(self.boundary_v10_cid),
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
                "hosted_real_substrate_wall_report_v10",
            "report": self.to_dict()})


def build_wall_report_v10(
        *, boundary: HostedRealSubstrateBoundaryV10,
) -> HostedRealSubstrateWallReportV10:
    real_only = tuple(
        a for a in boundary.real_substrate_v22_axes
        if a not in tuple(boundary.available_axes))
    return HostedRealSubstrateWallReportV10(
        schema=(
            W77_HOSTED_REAL_SUBSTRATE_BOUNDARY_V10_SCHEMA_VERSION),
        boundary_v10_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        blocked_at_frontier_axes=tuple(
            boundary.frontier_blocked_axes),
    )


__all__ = [
    "W77_HOSTED_REAL_SUBSTRATE_BOUNDARY_V10_SCHEMA_VERSION",
    "W77_HOSTED_PLANE_BLOCKED_AXES_V10_NEW",
    "W77_HOSTED_PLANE_BLOCKED_AXES_V10",
    "W77_FRONTIER_BLOCKED_AXES",
    "HostedRealSubstrateBoundaryV10",
    "build_default_hosted_real_substrate_boundary_v10",
    "HostedRealSubstrateBoundaryV10Falsifier",
    "probe_hosted_real_substrate_boundary_v10_falsifier",
    "HostedRealSubstrateWallReportV10",
    "build_wall_report_v10",
]
