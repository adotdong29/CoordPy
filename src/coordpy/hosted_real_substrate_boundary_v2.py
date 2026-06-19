"""W69 H6 — Hosted ↔ Real-Substrate Boundary V2.

Strictly extends W68's ``coordpy.hosted_real_substrate_boundary``.
V2 adds:

* **Four new blocked-axes at the hosted surface** —
  ``multi_branch_rejoin_witness``, ``silent_corruption_witness``,
  ``substrate_self_checksum``, ``v14_gate_score``.
* **Aggregate Plane A/B handoff envelope CID** — V2 records the
  handoff-envelope chain across a run so that the wall + the
  handoff policy together are content-addressed.
* **Frontier-blocked axes** — V2 enumerates a small set of axes
  that the V14 in-repo substrate also does not satisfy (e.g.
  third-party transformer hidden-state read); these are an honest
  no-progress marker.

Honest scope (W69)
------------------

* The wall remains a **structural** assertion.
* ``W69-T-HOSTED-REAL-SUBSTRATE-BOUNDARY-V2`` documents the V2
  assertion.
* W69 does NOT pierce the hosted substrate boundary; the boundary
  V2 records this as a content-addressed invariant.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_real_substrate_boundary import (
    HostedRealSubstrateBoundary,
    W68_HOSTED_PLANE_AVAILABLE_AXES,
    W68_HOSTED_PLANE_BLOCKED_AXES,
)
from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v14 import (
    W69_SUBSTRATE_V14_CAPABILITY_AXES,
)
from .tiny_substrate_v3 import _sha256_hex


W69_HOSTED_REAL_SUBSTRATE_BOUNDARY_V2_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary_v2.v1")

# Axes blocked at the hosted surface (V69 adds 4 new V14 axes).
W69_HOSTED_PLANE_BLOCKED_AXES_V2_NEW: tuple[str, ...] = (
    "multi_branch_rejoin_witness",
    "silent_corruption_witness",
    "substrate_self_checksum",
    "v14_gate_score",
)
W69_HOSTED_PLANE_BLOCKED_AXES_V2: tuple[str, ...] = (
    *W68_HOSTED_PLANE_BLOCKED_AXES,
    *W69_HOSTED_PLANE_BLOCKED_AXES_V2_NEW,
)

# Frontier-blocked axes — what V14 also does not satisfy.
W69_FRONTIER_BLOCKED_AXES: tuple[str, ...] = (
    "third_party_hosted_model_hidden_state_read",
    "third_party_hosted_model_kv_bytes_read",
    "third_party_hosted_model_attention_weights_read",
)


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV2:
    schema: str
    inner_v1: HostedRealSubstrateBoundary
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v14_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    frontier_blocked_axes: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v1_cid": str(self.inner_v1.cid()),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v14_axes": list(
                self.real_substrate_v14_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "frontier_blocked_axes": list(
                self.frontier_blocked_axes),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_v2",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary_v2(
) -> HostedRealSubstrateBoundaryV2:
    from .hosted_real_substrate_boundary import (
        build_default_hosted_real_substrate_boundary,
    )
    inner_v1 = build_default_hosted_real_substrate_boundary()
    return HostedRealSubstrateBoundaryV2(
        schema=(
            W69_HOSTED_REAL_SUBSTRATE_BOUNDARY_V2_SCHEMA_VERSION),
        inner_v1=inner_v1,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W69_HOSTED_PLANE_BLOCKED_AXES_V2,
        real_substrate_v14_axes=tuple(
            W69_SUBSTRATE_V14_CAPABILITY_AXES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        frontier_blocked_axes=W69_FRONTIER_BLOCKED_AXES,
        rationale=(
            "Hosted APIs expose text, optional logprobs, and "
            "optional prefix-cache hit accounting at the HTTP "
            "surface. They do NOT expose hidden states, KV-cache "
            "bytes, attention weights, or any per-(layer, head, "
            "slot) tensor — including the four new W69 V14 axes "
            "(multi_branch_rejoin_witness, silent_corruption_"
            "witness, substrate_self_checksum, v14_gate_score). "
            "The W69 V14 in-repo substrate is the only runtime "
            "that honestly exposes the full V14 capability set. "
            "The third-party-hosted-model substrate access remains "
            "blocked at the frontier; W69 codifies this as the "
            "frontier_blocked_axes set."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV2Falsifier:
    schema: str
    boundary_v2_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v2_cid": str(self.boundary_v2_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_boundary_v2_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_substrate_boundary_v2_falsifier(
        *, boundary: HostedRealSubstrateBoundaryV2,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryV2Falsifier:
    """Returns 0 iff the claim is consistent with the V2 wall;
    1 if the claim violates the V2 wall."""
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryV2Falsifier(
        schema=(
            W69_HOSTED_REAL_SUBSTRATE_BOUNDARY_V2_SCHEMA_VERSION),
        boundary_v2_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReportV2:
    schema: str
    boundary_v2_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_v2_cid": str(self.boundary_v2_cid),
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
                "hosted_real_substrate_wall_report_v2",
            "report": self.to_dict()})


def build_wall_report_v2(
        *, boundary: HostedRealSubstrateBoundaryV2,
) -> HostedRealSubstrateWallReportV2:
    real_only = tuple(
        a for a in boundary.real_substrate_v14_axes
        if a not in tuple(boundary.available_axes))
    return HostedRealSubstrateWallReportV2(
        schema=(
            W69_HOSTED_REAL_SUBSTRATE_BOUNDARY_V2_SCHEMA_VERSION),
        boundary_v2_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        blocked_at_frontier_axes=tuple(
            boundary.frontier_blocked_axes),
    )


__all__ = [
    "W69_HOSTED_REAL_SUBSTRATE_BOUNDARY_V2_SCHEMA_VERSION",
    "W69_HOSTED_PLANE_BLOCKED_AXES_V2_NEW",
    "W69_HOSTED_PLANE_BLOCKED_AXES_V2",
    "W69_FRONTIER_BLOCKED_AXES",
    "HostedRealSubstrateBoundaryV2",
    "build_default_hosted_real_substrate_boundary_v2",
    "HostedRealSubstrateBoundaryV2Falsifier",
    "probe_hosted_real_substrate_boundary_v2_falsifier",
    "HostedRealSubstrateWallReportV2",
    "build_wall_report_v2",
]
