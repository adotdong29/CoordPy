"""W68 H6 — Hosted ↔ Real-Substrate Boundary (the Wall).

The explicit **architecture-wall assertion** between Plane A (hosted
control plane) and Plane B (real substrate plane). This module
codifies the W68 honest claim:

* Plane A can route, fuse-logprobs, plan caches, filter providers,
  and plan budgets *at the HTTP text surface*.
* Plane A canNOT honestly access hidden states, KV cache bytes,
  attention weights, per-(layer, head) tensors, prefix-reuse
  counters, branch-merge witnesses, partial-contradiction
  witnesses, agent-replacement flags, or any V13 substrate-load-
  bearing axis.
* Plane B can do all of the above *only* on runtimes we control
  (in-repo V13 substrate; future local-transformers /
  llama.cpp / vLLM / MLX runtimes if added honestly).

This module ships a content-addressed
``HostedRealSubstrateBoundary`` object that names every blocked
axis at the hosted surface. It is used by the W68 limitation
reproduction and by R-155 (hosted-vs-real wall benchmark family).

Honest scope (W68)
------------------

* This boundary is a **structural** assertion. It tracks the
  research line's honest claim and is exercised by the W68
  falsifier test.
* ``W68-T-HOSTED-REAL-SUBSTRATE-BOUNDARY`` documents the assertion.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .hosted_router_controller import (
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .substrate_adapter_v13 import (
    W68_SUBSTRATE_V13_CAPABILITY_AXES,
)
from .tiny_substrate_v3 import _sha256_hex


W68_HOSTED_REAL_SUBSTRATE_BOUNDARY_SCHEMA_VERSION: str = (
    "coordpy.hosted_real_substrate_boundary.v1")

# Axes available at the hosted surface (Plane A).
W68_HOSTED_PLANE_AVAILABLE_AXES: tuple[str, ...] = (
    "text",
    "logprobs",
    "prefix_cache",
    "data_policy_declared",
    "cost_estimate_declared",
    "latency_estimate_declared",
)

# Axes blocked at the hosted surface (require Plane B).
W68_HOSTED_PLANE_BLOCKED_AXES: tuple[str, ...] = (
    "hidden_state_read",
    "hidden_state_write",
    "kv_bytes_read",
    "kv_bytes_write",
    "attention_weights_read",
    "attention_weights_write",
    "per_layer_head_tensor_read",
    "branch_merge_witness",
    "role_dropout_recovery_flag",
    "substrate_snapshot_fork",
    "v12_gate_score",
    "partial_contradiction_witness",
    "agent_replacement_flag",
    "prefix_reuse_counter",
    "v13_gate_score",
)


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundary:
    schema: str
    available_axes: tuple[str, ...]
    blocked_axes: tuple[str, ...]
    real_substrate_v13_axes: tuple[str, ...]
    hosted_tiers: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "available_axes": list(self.available_axes),
            "blocked_axes": list(self.blocked_axes),
            "real_substrate_v13_axes": list(
                self.real_substrate_v13_axes),
            "hosted_tiers": list(self.hosted_tiers),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary",
            "boundary": self.to_dict()})


def build_default_hosted_real_substrate_boundary(
) -> HostedRealSubstrateBoundary:
    return HostedRealSubstrateBoundary(
        schema=W68_HOSTED_REAL_SUBSTRATE_BOUNDARY_SCHEMA_VERSION,
        available_axes=W68_HOSTED_PLANE_AVAILABLE_AXES,
        blocked_axes=W68_HOSTED_PLANE_BLOCKED_AXES,
        real_substrate_v13_axes=tuple(
            W68_SUBSTRATE_V13_CAPABILITY_AXES),
        hosted_tiers=(
            W68_HOSTED_TIER_TEXT_ONLY,
            W68_HOSTED_TIER_LOGPROBS,
            W68_HOSTED_TIER_PREFIX_CACHE,
            W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
        ),
        rationale=(
            "Hosted APIs (OpenRouter, Groq, OpenAI, OpenAI-compat) "
            "expose text, optional logprobs (top-k), and optional "
            "prefix-cache hit accounting at the HTTP surface. They "
            "do NOT expose hidden states, KV-cache bytes, attention "
            "weights, or any per-(layer, head, slot) tensor. The "
            "W68 V13 substrate (in-repo) is the only runtime that "
            "honestly exposes the full V13 capability set including "
            "partial-contradiction witness, agent-replacement flag, "
            "prefix-reuse counter, and the V13 composite gate score."
        ),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateBoundaryFalsifier:
    schema: str
    boundary_cid: str
    claimed_axis: str
    claim_satisfied_at_hosted: bool
    falsifier_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_cid": str(self.boundary_cid),
            "claimed_axis": str(self.claimed_axis),
            "claim_satisfied_at_hosted": bool(
                self.claim_satisfied_at_hosted),
            "falsifier_score": float(round(
                self.falsifier_score, 12)),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_boundary_falsifier",
            "falsifier": self.to_dict()})


def probe_hosted_real_substrate_boundary_falsifier(
        *, boundary: HostedRealSubstrateBoundary,
        claimed_axis: str,
        claim_satisfied_at_hosted: bool,
) -> HostedRealSubstrateBoundaryFalsifier:
    """Returns 0 iff the claim is consistent with the boundary
    (claimed-blocked-and-not-satisfied OR claimed-available-and-
    satisfied). Returns 1 iff the claim violates the boundary
    (claimed-blocked-but-allegedly-satisfied at hosted)."""
    in_available = (
        str(claimed_axis) in tuple(boundary.available_axes))
    in_blocked = (
        str(claimed_axis) in tuple(boundary.blocked_axes))
    score = 0.0
    if in_blocked and bool(claim_satisfied_at_hosted):
        score = 1.0
    if in_available and not bool(claim_satisfied_at_hosted):
        score = 1.0
    return HostedRealSubstrateBoundaryFalsifier(
        schema=W68_HOSTED_REAL_SUBSTRATE_BOUNDARY_SCHEMA_VERSION,
        boundary_cid=str(boundary.cid()),
        claimed_axis=str(claimed_axis),
        claim_satisfied_at_hosted=bool(
            claim_satisfied_at_hosted),
        falsifier_score=float(score),
    )


@dataclasses.dataclass(frozen=True)
class HostedRealSubstrateWallReport:
    schema: str
    boundary_cid: str
    hosted_solvable_axes: tuple[str, ...]
    real_substrate_only_axes: tuple[str, ...]
    blocked_at_frontier_axes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "boundary_cid": str(self.boundary_cid),
            "hosted_solvable_axes": list(
                self.hosted_solvable_axes),
            "real_substrate_only_axes": list(
                self.real_substrate_only_axes),
            "blocked_at_frontier_axes": list(
                self.blocked_at_frontier_axes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "hosted_real_substrate_wall_report",
            "report": self.to_dict()})


def build_wall_report(
        *, boundary: HostedRealSubstrateBoundary,
) -> HostedRealSubstrateWallReport:
    real_only = tuple(
        a for a in boundary.real_substrate_v13_axes
        if a not in tuple(boundary.available_axes))
    # Frontier-blocked axes: those we don't have on any runtime we
    # control today (we list none ⇒ V13 in-repo covers all V13 axes).
    blocked_frontier = tuple()
    return HostedRealSubstrateWallReport(
        schema=W68_HOSTED_REAL_SUBSTRATE_BOUNDARY_SCHEMA_VERSION,
        boundary_cid=str(boundary.cid()),
        hosted_solvable_axes=tuple(boundary.available_axes),
        real_substrate_only_axes=real_only,
        blocked_at_frontier_axes=blocked_frontier,
    )


__all__ = [
    "W68_HOSTED_REAL_SUBSTRATE_BOUNDARY_SCHEMA_VERSION",
    "W68_HOSTED_PLANE_AVAILABLE_AXES",
    "W68_HOSTED_PLANE_BLOCKED_AXES",
    "HostedRealSubstrateBoundary",
    "build_default_hosted_real_substrate_boundary",
    "HostedRealSubstrateBoundaryFalsifier",
    "probe_hosted_real_substrate_boundary_falsifier",
    "HostedRealSubstrateWallReport",
    "build_wall_report",
]
