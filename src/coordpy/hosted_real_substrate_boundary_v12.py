"""W79 — Hosted ↔ Real-Substrate Boundary V12.

Strictly extends W78's
``coordpy.hosted_real_substrate_boundary_v11``. V12 adds:

* **Three new blocked-axes at the hosted surface** — the W79 V24
  substrate axes (replacement_then_restart_after_long_delay_*).
* **Seven new ``controlled_runtime`` axes** — these are
  explicitly **available on the controlled-runtime substrate**
  but still **blocked at the third-party hosted surface**. The
  W79 milestone makes this distinction load-bearing: a substrate
  axis can be hosted-blocked, in-repo-available, AND controlled-
  runtime-available simultaneously.
* **Replacement-then-restart-after-long-delay falsifier** —
  same shape as V11, but for the W79 axis.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .hosted_real_substrate_boundary import (
    W68_HOSTED_PLANE_AVAILABLE_AXES,
)
from .hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from .hosted_real_substrate_boundary_v11 import (
    HostedRealSubstrateBoundaryV11,
    W78_HOSTED_PLANE_BLOCKED_AXES_V11,
    build_default_hosted_real_substrate_boundary_v11,
)
from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v24 import (
    W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES,
    W79_SUBSTRATE_V24_CAPABILITY_AXES,
)


W79_HOSTED_REAL_SUBSTRATE_BOUNDARY_V12_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary_v12.v1")

W79_HOSTED_PLANE_BLOCKED_AXES_V12_NEW: tuple[str, ...] = (
    "replacement_then_restart_after_long_delay_trajectory_cid",
    "replacement_then_restart_after_long_delay_length_per_layer",
    "replacement_then_restart_after_long_delay_pressure_gate_per_layer",
    # The controlled-runtime axes are ALSO blocked at the hosted
    # plane — the controlled runtime is the W79 attack on the
    # wall, not an exception.
    *W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES,
)
W79_HOSTED_PLANE_BLOCKED_AXES_V12: tuple[str, ...] = (
    *W78_HOSTED_PLANE_BLOCKED_AXES_V11,
    *W79_HOSTED_PLANE_BLOCKED_AXES_V12_NEW,
)
W79_FRONTIER_BLOCKED_AXES: tuple[str, ...] = (
    *W70_FRONTIER_BLOCKED_AXES,
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV12:
    schema: str
    inner_v11: HostedRealSubstrateBoundaryV11
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v24_axes: tuple[str, ...]
    controlled_runtime_v1_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    frontier_blocked_axes: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "inner_v11_cid": str(self.inner_v11.cid()),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v24_axes": list(
                self.real_substrate_v24_axes),
            "controlled_runtime_v1_axes": list(
                self.controlled_runtime_v1_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "frontier_blocked_axes": list(
                self.frontier_blocked_axes),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_v12",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary_v12(
) -> HostedRealSubstrateBoundaryV12:
    inner_v11 = (
        build_default_hosted_real_substrate_boundary_v11())
    return HostedRealSubstrateBoundaryV12(
        schema=(
            W79_HOSTED_REAL_SUBSTRATE_BOUNDARY_V12_SCHEMA_VERSION),
        inner_v11=inner_v11,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W79_HOSTED_PLANE_BLOCKED_AXES_V12,
        real_substrate_v24_axes=tuple(
            W79_SUBSTRATE_V24_CAPABILITY_AXES),
        controlled_runtime_v1_axes=tuple(
            W79_CONTROLLED_RUNTIME_AXES_AS_CAPABILITIES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        frontier_blocked_axes=W79_FRONTIER_BLOCKED_AXES,
        rationale=(
            "Hosted APIs expose text, optional logprobs, and "
            "optional prefix-cache hit accounting at the HTTP "
            "surface. They do NOT expose hidden states, KV cache "
            "bytes, attention weights, the W69..W78 axes, the "
            "three new W79 V24 axes "
            "(replacement_then_restart_after_long_delay_*), or "
            "the seven new W79 controlled-runtime axes. The W79 "
            "controlled-runtime substrate V1 IS a second "
            "substrate runtime that DOES expose those axes — "
            "but the third-party hosted-model substrate access "
            "remains blocked at the frontier; W79 carries the "
            "W70 frontier_blocked_axes set forward unchanged."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryV12Falsifier:
    schema: str
    boundary_v12_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def cid(self) -> str:
        return _sha256_hex({
            "kind":
                "hosted_real_substrate_boundary_v12_falsifier",
            "schema": str(self.schema),
            "boundary_v12_cid": str(self.boundary_v12_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        })


def probe_hosted_real_substrate_boundary_v12_falsifier(
        *, boundary: HostedRealSubstrateBoundaryV12,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryV12Falsifier:
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryV12Falsifier(
        schema=(
            W79_HOSTED_REAL_SUBSTRATE_BOUNDARY_V12_SCHEMA_VERSION),
        boundary_v12_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReportV12:
    schema: str
    boundary_v12_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    controlled_runtime_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_wall_report_v12",
            "schema": str(self.schema),
            "boundary_v12_cid": str(self.boundary_v12_cid),
            "hosted_solvable_axes": list(
                self.hosted_solvable_axes),
            "real_substrate_only_axes": list(
                self.real_substrate_only_axes),
            "controlled_runtime_only_axes": list(
                self.controlled_runtime_only_axes),
            "blocked_at_frontier_axes": list(
                self.blocked_at_frontier_axes),
        })


def build_wall_report_v12(
        *, boundary: HostedRealSubstrateBoundaryV12,
) -> HostedRealSubstrateWallReportV12:
    real_only = tuple(
        a for a in boundary.real_substrate_v24_axes
        if a not in tuple(boundary.available_axes)
        and a not in tuple(boundary.controlled_runtime_v1_axes))
    ctrl_only = tuple(
        a for a in boundary.controlled_runtime_v1_axes
        if a not in tuple(boundary.available_axes))
    return HostedRealSubstrateWallReportV12(
        schema=(
            W79_HOSTED_REAL_SUBSTRATE_BOUNDARY_V12_SCHEMA_VERSION),
        boundary_v12_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        controlled_runtime_only_axes=ctrl_only,
        blocked_at_frontier_axes=tuple(
            boundary.frontier_blocked_axes),
    )


__all__ = [
    "W79_HOSTED_REAL_SUBSTRATE_BOUNDARY_V12_SCHEMA_VERSION",
    "W79_HOSTED_PLANE_BLOCKED_AXES_V12_NEW",
    "W79_HOSTED_PLANE_BLOCKED_AXES_V12",
    "W79_FRONTIER_BLOCKED_AXES",
    "HostedRealSubstrateBoundaryV12",
    "build_default_hosted_real_substrate_boundary_v12",
    "HostedRealSubstrateBoundaryV12Falsifier",
    "probe_hosted_real_substrate_boundary_v12_falsifier",
    "HostedRealSubstrateWallReportV12",
    "build_wall_report_v12",
]
