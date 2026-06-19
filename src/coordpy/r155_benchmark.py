"""W68 R-155 benchmark family — Hosted-vs-Real Wall.

The explicit cross-plane "wall" benchmark: what Plane A (hosted
control plane) can solve, what only Plane B (real substrate plane)
can solve, and where the wall sits.

H350..H355 cell families (6 H-bars):

* H350   Wall is content-addressed (boundary CID stable)
* H351   Wall lists ≥ 15 blocked axes at the hosted surface
* H352   Wall lists ≥ 60 real-substrate-only V13 axes
* H353   Hosted-cache-aware planner ≥ 50 % savings at hit_rate=1.0
* H354   Real-substrate branch-merge primitive ≥ 60 % flops saving
* H355   Falsifier triggers on dishonest hosted hidden-state claim
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_cache_aware_planner import (
    hosted_cache_aware_savings_vs_recompute,
)
from coordpy.hosted_real_substrate_boundary import (
    build_default_hosted_real_substrate_boundary,
    build_wall_report,
    probe_hosted_real_substrate_boundary_falsifier,
)
from coordpy.tiny_substrate_v12 import (
    substrate_branch_merge_flops_v12,
)


R155_SCHEMA_VERSION: str = "coordpy.r155_benchmark.v1"


def family_h350_wall_content_addressed(
        seed: int) -> dict[str, Any]:
    b1 = build_default_hosted_real_substrate_boundary()
    b2 = build_default_hosted_real_substrate_boundary()
    return {
        "schema": R155_SCHEMA_VERSION,
        "name": "h350_wall_content_addressed",
        "passed": bool(b1.cid() == b2.cid()),
        "cid": b1.cid(),
    }


def family_h351_wall_blocked_axes(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary()
    return {
        "schema": R155_SCHEMA_VERSION,
        "name": "h351_wall_blocked_axes",
        "passed": bool(len(b.blocked_axes) >= 15),
        "n_blocked": len(b.blocked_axes),
    }


def family_h352_wall_real_substrate_only_axes(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary()
    wr = build_wall_report(boundary=b)
    return {
        "schema": R155_SCHEMA_VERSION,
        "name": "h352_wall_real_substrate_only_axes",
        "passed": bool(len(wr.real_substrate_only_axes) >= 60),
        "n_real_only": len(wr.real_substrate_only_axes),
    }


def family_h353_hosted_cache_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_cache_aware_savings_vs_recompute(
        n_turns=10, prefix_tokens_per_turn=500,
        role_tokens_per_turn=100,
        hosted_cache_hit_rate=1.0)
    return {
        "schema": R155_SCHEMA_VERSION,
        "name": "h353_hosted_cache_savings",
        "passed": bool(sav["saving_ratio"] >= 0.5),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h354_real_substrate_branch_merge(
        seed: int) -> dict[str, Any]:
    res = substrate_branch_merge_flops_v12(
        n_tokens=128, n_branches=4)
    return {
        "schema": R155_SCHEMA_VERSION,
        "name": "h354_real_substrate_branch_merge",
        "passed": bool(res["saving_ratio"] >= 0.6),
        "ratio": float(res["saving_ratio"]),
    }


def family_h355_dishonest_claim_falsifier(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary()
    honest = probe_hosted_real_substrate_boundary_falsifier(
        boundary=b, claimed_axis="hidden_state_read",
        claim_satisfied_at_hosted=False)
    dishonest = (
        probe_hosted_real_substrate_boundary_falsifier(
            boundary=b, claimed_axis="hidden_state_read",
            claim_satisfied_at_hosted=True))
    return {
        "schema": R155_SCHEMA_VERSION,
        "name": "h355_dishonest_claim_falsifier",
        "passed": bool(
            honest.falsifier_score == 0.0
            and dishonest.falsifier_score == 1.0),
    }


_R155_FAMILIES: tuple[Any, ...] = (
    family_h350_wall_content_addressed,
    family_h351_wall_blocked_axes,
    family_h352_wall_real_substrate_only_axes,
    family_h353_hosted_cache_savings,
    family_h354_real_substrate_branch_merge,
    family_h355_dishonest_claim_falsifier,
)


def run_r155(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R155_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R155_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R155_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R155_SCHEMA_VERSION", "run_r155"]
