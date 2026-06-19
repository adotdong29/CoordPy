"""W69 R-159 benchmark family — Hosted ↔ Real Handoff Coordinator.

The new W69 Plane A↔B bridge benchmark family. R-159 exercises the
hosted_real_handoff_coordinator, the boundary V2 wall (≥ 19 blocked
axes + frontier-blocked axes), and the cross-plane handoff savings.

H450..H458 cell families (9 H-bars):

* H450   Handoff envelope is content-addressed (same input → same
         CID)
* H451   Hosted-only request routes to plane A
* H452   Substrate-only request routes to plane B
* H453   Both-required request routes to plane A+B-audit
* H454   Falsifier returns 0 on honest claim and 1 on dishonest
         claim
* H455   Cross-plane handoff token saving ≥ 40 %
* H456   Boundary V2 enumerates ≥ 19 blocked axes (V14)
* H457   Boundary V2 enumerates ≥ 3 frontier-blocked axes
* H458   Boundary V2 wall-report has ≥ 60 real-substrate-only V14
         axes
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_real_handoff_coordinator import (
    HandoffRequest, HostedRealHandoffCoordinator,
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
    hosted_real_handoff_cross_plane_savings,
    probe_hosted_real_handoff_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v2 import (
    W69_FRONTIER_BLOCKED_AXES,
    build_default_hosted_real_substrate_boundary_v2,
    build_wall_report_v2,
)


R159_SCHEMA_VERSION: str = "coordpy.r159_benchmark.v1"


def family_h450_handoff_envelope_content_addressed(
        seed: int) -> dict[str, Any]:
    c1 = HostedRealHandoffCoordinator()
    c2 = HostedRealHandoffCoordinator()
    req = HandoffRequest(
        request_cid=f"h450_{int(seed)}",
        needs_text_only=True,
        needs_substrate_state_access=False)
    e1 = c1.decide(req=req, substrate_self_checksum_cid="x")
    e2 = c2.decide(req=req, substrate_self_checksum_cid="x")
    return {
        "schema": R159_SCHEMA_VERSION,
        "name": "h450_handoff_envelope_content_addressed",
        "passed": bool(e1.cid() == e2.cid()),
    }


def family_h451_text_only_routes_to_plane_a(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinator()
    e = c.decide(
        req=HandoffRequest(
            request_cid=f"h451_{int(seed)}",
            needs_text_only=True,
            needs_substrate_state_access=False))
    return {
        "schema": R159_SCHEMA_VERSION,
        "name": "h451_text_only_routes_to_plane_a",
        "passed": bool(
            e.decision == W69_HANDOFF_DECISION_HOSTED_ONLY
            and e.plane == "A"),
    }


def family_h452_substrate_only_routes_to_plane_b(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinator()
    e = c.decide(
        req=HandoffRequest(
            request_cid=f"h452_{int(seed)}",
            needs_text_only=False,
            needs_substrate_state_access=True))
    return {
        "schema": R159_SCHEMA_VERSION,
        "name": "h452_substrate_only_routes_to_plane_b",
        "passed": bool(
            e.decision
            == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
            and e.plane == "B"),
    }


def family_h453_both_required_routes_to_audit(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinator()
    e = c.decide(
        req=HandoffRequest(
            request_cid=f"h453_{int(seed)}",
            needs_text_only=True,
            needs_substrate_state_access=True))
    return {
        "schema": R159_SCHEMA_VERSION,
        "name": "h453_both_required_routes_to_audit",
        "passed": bool(
            e.decision
            == W69_HANDOFF_DECISION_HOSTED_WITH_REAL_AUDIT
            and e.plane == "A+B-audit"),
    }


def family_h454_falsifier_honest_vs_dishonest(
        seed: int) -> dict[str, Any]:
    c = HostedRealHandoffCoordinator()
    e = c.decide(
        req=HandoffRequest(
            request_cid=f"h454_{int(seed)}",
            needs_text_only=True,
            needs_substrate_state_access=False))
    honest = probe_hosted_real_handoff_falsifier(
        envelope=e,
        claim_kind="hosted_provides_substrate_state",
        claim_satisfied=False)
    dishonest = probe_hosted_real_handoff_falsifier(
        envelope=e,
        claim_kind="hosted_provides_substrate_state",
        claim_satisfied=True)
    return {
        "schema": R159_SCHEMA_VERSION,
        "name": "h454_falsifier_honest_vs_dishonest",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


def family_h455_cross_plane_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_real_handoff_cross_plane_savings(n_turns=12)
    return {
        "schema": R159_SCHEMA_VERSION,
        "name": "h455_cross_plane_savings",
        "passed": bool(sav["saving_ratio"] >= 0.4),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h456_boundary_v2_blocked_axes(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v2()
    return {
        "schema": R159_SCHEMA_VERSION,
        "name": "h456_boundary_v2_blocked_axes",
        "passed": bool(len(b.blocked_axes) >= 19),
        "n_blocked": int(len(b.blocked_axes)),
    }


def family_h457_boundary_v2_frontier_blocked_axes(
        seed: int) -> dict[str, Any]:
    return {
        "schema": R159_SCHEMA_VERSION,
        "name": "h457_boundary_v2_frontier_blocked_axes",
        "passed": bool(len(W69_FRONTIER_BLOCKED_AXES) >= 3),
        "n_frontier": int(len(W69_FRONTIER_BLOCKED_AXES)),
    }


def family_h458_boundary_v2_real_substrate_only_axes(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v2()
    wr = build_wall_report_v2(boundary=b)
    return {
        "schema": R159_SCHEMA_VERSION,
        "name": "h458_boundary_v2_real_substrate_only_axes",
        "passed": bool(len(wr.real_substrate_only_axes) >= 60),
        "n_real_only": int(len(wr.real_substrate_only_axes)),
    }


_R159_FAMILIES: tuple[Any, ...] = (
    family_h450_handoff_envelope_content_addressed,
    family_h451_text_only_routes_to_plane_a,
    family_h452_substrate_only_routes_to_plane_b,
    family_h453_both_required_routes_to_audit,
    family_h454_falsifier_honest_vs_dishonest,
    family_h455_cross_plane_savings,
    family_h456_boundary_v2_blocked_axes,
    family_h457_boundary_v2_frontier_blocked_axes,
    family_h458_boundary_v2_real_substrate_only_axes,
)


def run_r159(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R159_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R159_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R159_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R159_SCHEMA_VERSION", "run_r159"]
