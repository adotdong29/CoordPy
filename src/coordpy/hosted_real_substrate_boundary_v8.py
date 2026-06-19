"""W75 H5 — Hosted ↔ Real-Substrate Boundary V8.

Strictly extends W74's
``coordpy.hosted_real_substrate_boundary_v7``. V8 adds:

* **Three new blocked-axes at the hosted surface** —
  ``compound_chain_repair_trajectory_cid``,
  ``compound_chain_length_per_layer``,
  ``compound_chain_pressure_gate_per_layer``.
* **Compound-chain-pressure falsifier** — same shape as V7, but
  for the new W75 V20 axes.
* **Frontier-blocked axes** — V8 keeps W70's frontier-blocked set
  unchanged (third-party transformer hidden-state read, KV bytes
  read, attention weights read).

Honest scope (W75)
------------------

* The wall remains a **structural** assertion.
* W75 does NOT pierce the hosted substrate boundary; the boundary
  V8 records this as a content-addressed invariant.
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
from .hosted_real_substrate_boundary_v7 import (
    HostedRealSubstrateBoundaryV7,
    W74_HOSTED_PLANE_BLOCKED_AXES_V7,
    build_default_hosted_real_substrate_boundary_v7,
)
from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v20 import (
    W75_SUBSTRATE_V20_CAPABILITY_AXES,
)
from .tiny_substrate_v3 import _sha256_hex


W75_HOSTED_REAL_SUBSTRATE_BOUNDARY_V8_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary_v8.v1")

# Axes blocked at the hosted surface (V75 adds 3 new V20 axes).
W75_HOSTED_PLANE_BLOCKED_AXES_V8_NEW: tuple[str, ...] = (
    "compound_chain_repair_trajectory_cid",
    "compound_chain_length_per_layer",
    "compound_chain_pressure_gate_per_layer",
)
W75_HOSTED_PLANE_BLOCKED_AXES_V8: tuple[str, ...] = (
    *W74_HOSTED_PLANE_BLOCKED_AXES_V7,
    *W75_HOSTED_PLANE_BLOCKED_AXES_V8_NEW,
)

# Frontier-blocked axes — what V20 also does not satisfy.
W75_FRONTIER_BLOCKED_AXES: tuple[str, ...] = (
    *W70_FRONTIER_BLOCKED_AXES,
)


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV8:
    schema: str
    inner_v7: HostedRealSubstrateBoundaryV7
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v20_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    frontier_blocked_axes: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v7_cid": str(self.inner_v7.cid()),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v20_axes": list(
                self.real_substrate_v20_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "frontier_blocked_axes": list(
                self.frontier_blocked_axes),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_v8",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary_v8(
) -> HostedRealSubstrateBoundaryV8:
    inner_v7 = (
        build_default_hosted_real_substrate_boundary_v7())
    return HostedRealSubstrateBoundaryV8(
        schema=(
            W75_HOSTED_REAL_SUBSTRATE_BOUNDARY_V8_SCHEMA_VERSION),
        inner_v7=inner_v7,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W75_HOSTED_PLANE_BLOCKED_AXES_V8,
        real_substrate_v20_axes=tuple(
            W75_SUBSTRATE_V20_CAPABILITY_AXES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        frontier_blocked_axes=W75_FRONTIER_BLOCKED_AXES,
        rationale=(
            "Hosted APIs expose text, optional logprobs, and "
            "optional prefix-cache hit accounting at the HTTP "
            "surface. They do NOT expose hidden states, KV-cache "
            "bytes, attention weights, or any per-(layer, head, "
            "slot) tensor — including the four W69 V14 axes, the "
            "three W70 V15 axes, the three W71 V16 axes, the three "
            "W72 V17 axes, the three W73 V18 axes, the three W74 "
            "V19 axes, AND the three new W75 V20 axes "
            "(compound_chain_repair_trajectory_cid, "
            "compound_chain_length_per_layer, "
            "compound_chain_pressure_gate_per_layer). The W75 V20 "
            "in-repo substrate is the only runtime that honestly "
            "exposes the full V20 capability set. The third-party-"
            "hosted-model substrate access remains blocked at the "
            "frontier; W75 carries the W70 frontier_blocked_axes "
            "set forward unchanged."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV8Falsifier:
    schema: str
    boundary_v8_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v8_cid": str(self.boundary_v8_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_boundary_v8_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_substrate_boundary_v8_falsifier(
        *, boundary: HostedRealSubstrateBoundaryV8,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryV8Falsifier:
    """Returns 0 iff the claim is consistent with the V8 wall;
    1 if the claim violates the V8 wall."""
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryV8Falsifier(
        schema=(
            W75_HOSTED_REAL_SUBSTRATE_BOUNDARY_V8_SCHEMA_VERSION),
        boundary_v8_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReportV8:
    schema: str
    boundary_v8_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v8_cid": str(self.boundary_v8_cid),
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
                "hosted_real_substrate_wall_report_v8",
            "report": self.to_dict()})


def build_wall_report_v8(
        *, boundary: HostedRealSubstrateBoundaryV8,
) -> HostedRealSubstrateWallReportV8:
    real_only = tuple(
        a for a in boundary.real_substrate_v20_axes
        if a not in tuple(boundary.available_axes))
    return HostedRealSubstrateWallReportV8(
        schema=(
            W75_HOSTED_REAL_SUBSTRATE_BOUNDARY_V8_SCHEMA_VERSION),
        boundary_v8_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        blocked_at_frontier_axes=tuple(
            boundary.frontier_blocked_axes),
    )


__all__ = [
    "W75_HOSTED_REAL_SUBSTRATE_BOUNDARY_V8_SCHEMA_VERSION",
    "W75_HOSTED_PLANE_BLOCKED_AXES_V8_NEW",
    "W75_HOSTED_PLANE_BLOCKED_AXES_V8",
    "W75_FRONTIER_BLOCKED_AXES",
    "HostedRealSubstrateBoundaryV8",
    "build_default_hosted_real_substrate_boundary_v8",
    "HostedRealSubstrateBoundaryV8Falsifier",
    "probe_hosted_real_substrate_boundary_v8_falsifier",
    "HostedRealSubstrateWallReportV8",
    "build_wall_report_v8",
]
