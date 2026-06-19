"""W78 H5 — Hosted ↔ Real-Substrate Boundary V11.

Strictly extends W77's
``coordpy.hosted_real_substrate_boundary_v10``. V11 adds:

* **Three new blocked-axes at the hosted surface** —
  ``long_horizon_reconstruction_trajectory_cid``,
  ``long_horizon_reconstruction_length_per_layer``,
  ``long_horizon_reconstruction_pressure_gate_per_layer``.
* **Long-horizon-reconstruction-pressure falsifier** — same
  shape as V10, but for the new W78 V23 axes.
* **Frontier-blocked axes** — V11 keeps W70's frontier-blocked
  set unchanged.

Honest scope (W78)
------------------

* The wall remains a **structural** assertion.
* W78 does NOT pierce the hosted substrate boundary; the
  boundary V11 records this as a content-addressed invariant.
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
from .hosted_real_substrate_boundary_v10 import (
    HostedRealSubstrateBoundaryV10,
    W77_HOSTED_PLANE_BLOCKED_AXES_V10,
    build_default_hosted_real_substrate_boundary_v10,
)
from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v23 import (
    W78_SUBSTRATE_V23_CAPABILITY_AXES,
)
from .tiny_substrate_v3 import _sha256_hex


W78_HOSTED_REAL_SUBSTRATE_BOUNDARY_V11_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary_v11.v1")

W78_HOSTED_PLANE_BLOCKED_AXES_V11_NEW: tuple[str, ...] = (
    "long_horizon_reconstruction_trajectory_cid",
    "long_horizon_reconstruction_length_per_layer",
    "long_horizon_reconstruction_pressure_gate_per_layer",
)
W78_HOSTED_PLANE_BLOCKED_AXES_V11: tuple[str, ...] = (
    *W77_HOSTED_PLANE_BLOCKED_AXES_V10,
    *W78_HOSTED_PLANE_BLOCKED_AXES_V11_NEW,
)

W78_FRONTIER_BLOCKED_AXES: tuple[str, ...] = (
    *W70_FRONTIER_BLOCKED_AXES,
)


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV11:
    schema: str
    inner_v10: HostedRealSubstrateBoundaryV10
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v23_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    frontier_blocked_axes: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v10_cid": str(self.inner_v10.cid()),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v23_axes": list(
                self.real_substrate_v23_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "frontier_blocked_axes": list(
                self.frontier_blocked_axes),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_v11",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary_v11(
) -> HostedRealSubstrateBoundaryV11:
    inner_v10 = (
        build_default_hosted_real_substrate_boundary_v10())
    return HostedRealSubstrateBoundaryV11(
        schema=(
            W78_HOSTED_REAL_SUBSTRATE_BOUNDARY_V11_SCHEMA_VERSION),
        inner_v10=inner_v10,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W78_HOSTED_PLANE_BLOCKED_AXES_V11,
        real_substrate_v23_axes=tuple(
            W78_SUBSTRATE_V23_CAPABILITY_AXES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        frontier_blocked_axes=W78_FRONTIER_BLOCKED_AXES,
        rationale=(
            "Hosted APIs expose text, optional logprobs, and "
            "optional prefix-cache hit accounting at the HTTP "
            "surface. They do NOT expose hidden states, KV-cache "
            "bytes, attention weights, or any per-(layer, head, "
            "slot) tensor — including the W69-W77 V14..V22 axes "
            "AND the three new W78 V23 axes "
            "(long_horizon_reconstruction_trajectory_cid, "
            "long_horizon_reconstruction_length_per_layer, "
            "long_horizon_reconstruction_pressure_gate_per_layer). "
            "The W78 V23 in-repo substrate is the only runtime "
            "that honestly exposes the full V23 capability set. "
            "The third-party-hosted-model substrate access "
            "remains blocked at the frontier; W78 carries the W70 "
            "frontier_blocked_axes set forward unchanged."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV11Falsifier:
    schema: str
    boundary_v11_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v11_cid": str(self.boundary_v11_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_boundary_v11_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_substrate_boundary_v11_falsifier(
        *, boundary: HostedRealSubstrateBoundaryV11,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryV11Falsifier:
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryV11Falsifier(
        schema=(
            W78_HOSTED_REAL_SUBSTRATE_BOUNDARY_V11_SCHEMA_VERSION),
        boundary_v11_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReportV11:
    schema: str
    boundary_v11_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v11_cid": str(self.boundary_v11_cid),
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
                "hosted_real_substrate_wall_report_v11",
            "report": self.to_dict()})


def build_wall_report_v11(
        *, boundary: HostedRealSubstrateBoundaryV11,
) -> HostedRealSubstrateWallReportV11:
    real_only = tuple(
        a for a in boundary.real_substrate_v23_axes
        if a not in tuple(boundary.available_axes))
    return HostedRealSubstrateWallReportV11(
        schema=(
            W78_HOSTED_REAL_SUBSTRATE_BOUNDARY_V11_SCHEMA_VERSION),
        boundary_v11_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        blocked_at_frontier_axes=tuple(
            boundary.frontier_blocked_axes),
    )


__all__ = [
    "W78_HOSTED_REAL_SUBSTRATE_BOUNDARY_V11_SCHEMA_VERSION",
    "W78_HOSTED_PLANE_BLOCKED_AXES_V11_NEW",
    "W78_HOSTED_PLANE_BLOCKED_AXES_V11",
    "W78_FRONTIER_BLOCKED_AXES",
    "HostedRealSubstrateBoundaryV11",
    "build_default_hosted_real_substrate_boundary_v11",
    "HostedRealSubstrateBoundaryV11Falsifier",
    "probe_hosted_real_substrate_boundary_v11_falsifier",
    "HostedRealSubstrateWallReportV11",
    "build_wall_report_v11",
]
