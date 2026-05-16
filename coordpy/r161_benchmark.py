"""W70 R-161 benchmark family — Hosted Control Plane V3 (Plane A).

Exercises the W70 hosted control plane V3 modules: router V3,
logprob router V3 (abstain-when-disagree), cache-aware planner V3
(per-role staggered+rotated), cost planner V3
(cost-per-team-success-under-budget + abstain), boundary V3
(≥ 22 blocked axes).

H470..H478 cell families (10 H-bars):

* H470   Router V3 determinism on (registry CID, request CID, budget)
* H470b  Router V3 budget-efficiency score > 0
* H471   Logprob V3 abstain-when-disagree fires under high entropy
* H471b  Logprob V3 per-budget tiebreak narrows top-k under tight
* H472   Cache planner V3 per-role staggered+rotated CIDs are content-
         addressed
* H472b  Cache planner V3 ≥ 65 % saving on 8×8 at hit_rate=1.0
* H473   Cost planner V3 cost-per-team-success-under-budget is finite
* H473b  Cost planner V3 abstain-when-budget-violated fires
* H474   Boundary V3 enumerates ≥ 22 blocked axes (V15)
* H474b  Boundary V3 wall-report has ≥ 60 real-substrate-only V15 axes
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_router_controller import (
    HostedRoutingRequest, default_hosted_registry,
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
)
from coordpy.hosted_router_controller_v2 import (
    HostedRoutingRequestV2,
)
from coordpy.hosted_router_controller_v3 import (
    HostedRouterControllerV3, HostedRoutingRequestV3,
)
from coordpy.hosted_logprob_router_v3 import (
    HostedLogprobRouterV3,
)
from coordpy.hosted_logprob_router import TopKLogprobsPayload
from coordpy.hosted_cache_aware_planner_v3 import (
    HostedCacheAwarePlannerV3,
    hosted_cache_aware_savings_v3_vs_recompute,
)
from coordpy.hosted_cost_planner import HostedCostPlanSpec
from coordpy.hosted_cost_planner_v2 import HostedCostPlanSpecV2
from coordpy.hosted_cost_planner_v3 import (
    HostedCostPlanSpecV3, plan_hosted_cost_v3,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    build_default_hosted_real_substrate_boundary_v3,
    build_wall_report_v3,
)


R161_SCHEMA_VERSION: str = "coordpy.r161_benchmark.v1"


def family_h470_router_v3_determinism(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc1 = HostedRouterControllerV3.init(
        reg, {"openrouter_paid": 0.85, "openai_paid": 0.92})
    rc2 = HostedRouterControllerV3.init(
        reg, {"openrouter_paid": 0.85, "openai_paid": 0.92})
    req = HostedRoutingRequestV3(
        inner_v2=HostedRoutingRequestV2(
            inner_v1=HostedRoutingRequest(
                request_cid=f"r470_{int(seed)}",
                input_tokens=1000,
                expected_output_tokens=300,
                require_logprobs=True,
                require_prefix_cache=True,
                data_policy_required="no_log",
                max_latency_ms=2000.0, max_cost_usd=50.0),
            weight_cost=1.0, weight_latency=0.5,
            weight_success=0.3),
        visible_token_budget=128,
        baseline_token_cost=512,
        repair_dominance_label=1)
    d1 = rc1.decide_v3(req)
    d2 = rc2.decide_v3(req)
    return {
        "schema": R161_SCHEMA_VERSION,
        "name": "h470_router_v3_determinism",
        "passed": bool(
            d1.chosen_provider == d2.chosen_provider
            and d1.cid() == d2.cid()),
    }


def family_h470b_router_v3_budget_efficiency_score(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc = HostedRouterControllerV3.init(
        reg, {"openrouter_paid": 0.85, "openai_paid": 0.92})
    req = HostedRoutingRequestV3(
        inner_v2=HostedRoutingRequestV2(
            inner_v1=HostedRoutingRequest(
                request_cid=f"r470b_{int(seed)}",
                input_tokens=1000,
                expected_output_tokens=300,
                require_logprobs=True,
                require_prefix_cache=True,
                data_policy_required="no_log",
                max_latency_ms=2000.0, max_cost_usd=50.0),
            weight_cost=1.0, weight_latency=0.5,
            weight_success=0.3),
        visible_token_budget=256,
        baseline_token_cost=512,
        repair_dominance_label=0,
        weight_budget_efficiency=1.0)
    d = rc.decide_v3(req)
    return {
        "schema": R161_SCHEMA_VERSION,
        "name": "h470b_router_v3_budget_efficiency_score",
        "passed": bool(d.budget_efficiency_score > 0.0),
        "score": float(d.budget_efficiency_score),
    }


def family_h471_logprob_v3_abstain(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV3(
        abstain_entropy_floor=0.5)
    # High-entropy distribution.
    payloads = [
        TopKLogprobsPayload("p1", tuple(
            (chr(ord("a") + i), -0.1 - i * 0.01)
            for i in range(20))),
        TopKLogprobsPayload("p2", tuple(
            (chr(ord("a") + i), -0.1 - i * 0.005)
            for i in range(20))),
    ]
    res = lr.fuse_v3(
        payloads, visible_token_budget=256,
        baseline_token_cost=512)
    return {
        "schema": R161_SCHEMA_VERSION,
        "name": "h471_logprob_v3_abstain",
        "passed": bool(
            "abstain_v3" in str(res.get("fusion_kind", ""))),
    }


def family_h471b_logprob_v3_per_budget_tiebreak(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV3(
        abstain_entropy_floor=10.0)
    payloads = [
        TopKLogprobsPayload("p1",
            (("a", -0.1), ("b", -0.5), ("c", -1.0))),
        TopKLogprobsPayload("p2",
            (("a", -0.2), ("b", -0.4), ("c", -1.1))),
    ]
    # Tight budget → smaller effective_top_k.
    res_tight = lr.fuse_v3(
        payloads, visible_token_budget=32,
        baseline_token_cost=512, top_k=10)
    res_full = lr.fuse_v3(
        payloads, visible_token_budget=512,
        baseline_token_cost=512, top_k=10)
    return {
        "schema": R161_SCHEMA_VERSION,
        "name": "h471b_logprob_v3_per_budget_tiebreak",
        "passed": bool(
            int(res_tight.get("effective_top_k", 0))
            < int(res_full.get("effective_top_k", 0))),
    }


def family_h472_cache_v3_prefix_cid_content_addressed(
        seed: int) -> dict[str, Any]:
    cp1 = HostedCacheAwarePlannerV3()
    cp2 = HostedCacheAwarePlannerV3()
    planned1, _ = (
        cp1.plan_per_role_staggered_and_rotated(
            shared_prefix_text="W70 shared prefix " * 8,
            per_role_blocks={
                "plan": ["a", "b"], "write": ["c", "d"]}))
    planned2, _ = (
        cp2.plan_per_role_staggered_and_rotated(
            shared_prefix_text="W70 shared prefix " * 8,
            per_role_blocks={
                "plan": ["a", "b"], "write": ["c", "d"]}))
    ok = (
        planned1[0].rotated_prefix_cids
        == planned2[0].rotated_prefix_cids)
    return {
        "schema": R161_SCHEMA_VERSION,
        "name": "h472_cache_v3_prefix_cid_content_addressed",
        "passed": bool(ok),
    }


def family_h472b_cache_v3_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_cache_aware_savings_v3_vs_recompute(
        n_roles=8, n_turns=8, hosted_cache_hit_rate=1.0)
    return {
        "schema": R161_SCHEMA_VERSION,
        "name": "h472b_cache_v3_savings",
        "passed": bool(sav["saving_ratio"] >= 0.65),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h473_cost_v3_cps_under_budget(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    # Use a generous per-turn budget so we don't abstain.
    spec_v3 = HostedCostPlanSpecV3(
        inner_v2=HostedCostPlanSpecV2(
            inner_v1=HostedCostPlanSpec(
                n_turns=5, input_tokens_per_turn=200,
                output_tokens_per_turn=100,
                max_cost_per_turn_usd=5.0,
                max_latency_per_turn_ms=2500.0,
                min_quality_score=0.5),
            allow_provider_rotation=True),
        visible_token_budget_per_turn=512,
        baseline_token_cost_per_turn=512,
        abstain_when_budget_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v3(
        registry=reg, provider_quality_scores=qs,
        spec_v3=spec_v3)
    return {
        "schema": R161_SCHEMA_VERSION,
        "name": "h473_cost_v3_cps_under_budget",
        "passed": bool(
            rep.cost_per_team_success_under_budget >= 0.0
            and rep.cost_per_team_success_under_budget
                < float("inf")),
    }


def family_h473b_cost_v3_abstain_budget_violated(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    # Tight per-turn budget < input+output ⇒ violation.
    spec_v3 = HostedCostPlanSpecV3(
        inner_v2=HostedCostPlanSpecV2(
            inner_v1=HostedCostPlanSpec(
                n_turns=5, input_tokens_per_turn=500,
                output_tokens_per_turn=200,
                max_cost_per_turn_usd=5.0,
                max_latency_per_turn_ms=2500.0,
                min_quality_score=0.5),
            allow_provider_rotation=True),
        visible_token_budget_per_turn=256,
        baseline_token_cost_per_turn=512,
        abstain_when_budget_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v3(
        registry=reg, provider_quality_scores=qs,
        spec_v3=spec_v3)
    return {
        "schema": R161_SCHEMA_VERSION,
        "name": "h473b_cost_v3_abstain_budget_violated",
        "passed": bool(
            rep.budget_violated
            and "abstain_v3" in str(rep.rationale_v3)),
    }


def family_h474_boundary_v3_blocked_axes(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v3()
    return {
        "schema": R161_SCHEMA_VERSION,
        "name": "h474_boundary_v3_blocked_axes",
        "passed": bool(len(b.blocked_axes) >= 22),
        "n_blocked": int(len(b.blocked_axes)),
    }


def family_h474b_boundary_v3_real_substrate_only_axes(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v3()
    wr = build_wall_report_v3(boundary=b)
    return {
        "schema": R161_SCHEMA_VERSION,
        "name": "h474b_boundary_v3_real_substrate_only_axes",
        "passed": bool(len(wr.real_substrate_only_axes) >= 60),
        "n_real_only": int(len(wr.real_substrate_only_axes)),
    }


_R161_FAMILIES: tuple[Any, ...] = (
    family_h470_router_v3_determinism,
    family_h470b_router_v3_budget_efficiency_score,
    family_h471_logprob_v3_abstain,
    family_h471b_logprob_v3_per_budget_tiebreak,
    family_h472_cache_v3_prefix_cid_content_addressed,
    family_h472b_cache_v3_savings,
    family_h473_cost_v3_cps_under_budget,
    family_h473b_cost_v3_abstain_budget_violated,
    family_h474_boundary_v3_blocked_axes,
    family_h474b_boundary_v3_real_substrate_only_axes,
)


def run_r161(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R161_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R161_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R161_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R161_SCHEMA_VERSION", "run_r161"]
