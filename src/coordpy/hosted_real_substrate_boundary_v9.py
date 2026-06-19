"""W76 H5 — Hosted ↔ Real-Substrate Boundary V9.

Strictly extends W75's
``coordpy.hosted_real_substrate_boundary_v8``. V9 adds:

* **Three new blocked-axes at the hosted surface** —
  ``compound_chain_then_restart_trajectory_cid``,
  ``compound_chain_then_restart_length_per_layer``,
  ``compound_chain_then_restart_pressure_gate_per_layer``.
* **Chain-then-restart-pressure falsifier** — same shape as V8,
  but for the new W76 V21 axes.
* **Frontier-blocked axes** — V9 keeps W70's frontier-blocked set
  unchanged (third-party transformer hidden-state read, KV bytes
  read, attention weights read).

Honest scope (W76)
------------------

* The wall remains a **structural** assertion.
* W76 does NOT pierce the hosted substrate boundary; the boundary
  V9 records this as a content-addressed invariant.
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
from .hosted_real_substrate_boundary_v8 import (
    HostedRealSubstrateBoundaryV8,
    W75_HOSTED_PLANE_BLOCKED_AXES_V8,
    build_default_hosted_real_substrate_boundary_v8,
)
from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v21 import (
    W76_SUBSTRATE_V21_CAPABILITY_AXES,
)
from .tiny_substrate_v3 import _sha256_hex


W76_HOSTED_REAL_SUBSTRATE_BOUNDARY_V9_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary_v9.v1")

# Axes blocked at the hosted surface (W76 adds 3 new V21 axes).
W76_HOSTED_PLANE_BLOCKED_AXES_V9_NEW: tuple[str, ...] = (
    "compound_chain_then_restart_trajectory_cid",
    "compound_chain_then_restart_length_per_layer",
    "compound_chain_then_restart_pressure_gate_per_layer",
)
W76_HOSTED_PLANE_BLOCKED_AXES_V9: tuple[str, ...] = (
    *W75_HOSTED_PLANE_BLOCKED_AXES_V8,
    *W76_HOSTED_PLANE_BLOCKED_AXES_V9_NEW,
)

# Frontier-blocked axes — what V21 also does not satisfy.
W76_FRONTIER_BLOCKED_AXES: tuple[str, ...] = (
    *W70_FRONTIER_BLOCKED_AXES,
)


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV9:
    schema: str
    inner_v8: HostedRealSubstrateBoundaryV8
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v21_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    frontier_blocked_axes: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v8_cid": str(self.inner_v8.cid()),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v21_axes": list(
                self.real_substrate_v21_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "frontier_blocked_axes": list(
                self.frontier_blocked_axes),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_v9",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary_v9(
) -> HostedRealSubstrateBoundaryV9:
    inner_v8 = (
        build_default_hosted_real_substrate_boundary_v8())
    return HostedRealSubstrateBoundaryV9(
        schema=(
            W76_HOSTED_REAL_SUBSTRATE_BOUNDARY_V9_SCHEMA_VERSION),
        inner_v8=inner_v8,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W76_HOSTED_PLANE_BLOCKED_AXES_V9,
        real_substrate_v21_axes=tuple(
            W76_SUBSTRATE_V21_CAPABILITY_AXES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        frontier_blocked_axes=W76_FRONTIER_BLOCKED_AXES,
        rationale=(
            "Hosted APIs expose text, optional logprobs, and "
            "optional prefix-cache hit accounting at the HTTP "
            "surface. They do NOT expose hidden states, KV-cache "
            "bytes, attention weights, or any per-(layer, head, "
            "slot) tensor — including the W69-W75 V14..V20 axes "
            "AND the three new W76 V21 axes "
            "(compound_chain_then_restart_trajectory_cid, "
            "compound_chain_then_restart_length_per_layer, "
            "compound_chain_then_restart_pressure_gate_per_layer). "
            "The W76 V21 in-repo substrate is the only runtime "
            "that honestly exposes the full V21 capability set. "
            "The third-party-hosted-model substrate access "
            "remains blocked at the frontier; W76 carries the W70 "
            "frontier_blocked_axes set forward unchanged."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV9Falsifier:
    schema: str
    boundary_v9_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v9_cid": str(self.boundary_v9_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_boundary_v9_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_substrate_boundary_v9_falsifier(
        *, boundary: HostedRealSubstrateBoundaryV9,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryV9Falsifier:
    """Returns 0 iff the claim is consistent with the V9 wall;
    1 if the claim violates the V9 wall."""
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryV9Falsifier(
        schema=(
            W76_HOSTED_REAL_SUBSTRATE_BOUNDARY_V9_SCHEMA_VERSION),
        boundary_v9_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReportV9:
    schema: str
    boundary_v9_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v9_cid": str(self.boundary_v9_cid),
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
                "hosted_real_substrate_wall_report_v9",
            "report": self.to_dict()})


def build_wall_report_v9(
        *, boundary: HostedRealSubstrateBoundaryV9,
) -> HostedRealSubstrateWallReportV9:
    real_only = tuple(
        a for a in boundary.real_substrate_v21_axes
        if a not in tuple(boundary.available_axes))
    return HostedRealSubstrateWallReportV9(
        schema=(
            W76_HOSTED_REAL_SUBSTRATE_BOUNDARY_V9_SCHEMA_VERSION),
        boundary_v9_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        blocked_at_frontier_axes=tuple(
            boundary.frontier_blocked_axes),
    )


__all__ = [
    "W76_HOSTED_REAL_SUBSTRATE_BOUNDARY_V9_SCHEMA_VERSION",
    "W76_HOSTED_PLANE_BLOCKED_AXES_V9_NEW",
    "W76_HOSTED_PLANE_BLOCKED_AXES_V9",
    "W76_FRONTIER_BLOCKED_AXES",
    "HostedRealSubstrateBoundaryV9",
    "build_default_hosted_real_substrate_boundary_v9",
    "HostedRealSubstrateBoundaryV9Falsifier",
    "probe_hosted_real_substrate_boundary_v9_falsifier",
    "HostedRealSubstrateWallReportV9",
    "build_wall_report_v9",
]
