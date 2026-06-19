"""W71 R-165 benchmark family — Hosted Control Plane V4 (Plane A).

Exercises the W71 hosted control plane V4 modules: router V4
(restart-pressure + delayed-repair match), logprob router V4
(restart-aware abstain floor + per-budget+restart tiebreak),
cache-aware planner V4 (two-layer rotated), cost planner V4
(cost-per-repair-success-under-budget + abstain-when-restart-
pressure-violated), boundary V4 (≥ 25 blocked axes), provider
filter V3 (restart-aware drop).

H540..H548 cell families (10 H-bars):

* H540   Router V4 determinism on (registry CID, request V4 CID,
         budget, restart_pressure, repair_dominance_label)
* H540b  Router V4 restart-pressure score > 0 under high pressure
* H541   Logprob V4 abstain floor more aggressive under restart
* H541b  Logprob V4 per-budget+restart tiebreak narrows top-k
* H542   Cache planner V4 two-layer rotated CIDs content-addressed
* H542b  Cache planner V4 ≥ 72 % saving on 10×8 at hit_rate=1.0
* H543   Cost planner V4 cost-per-repair-success-under-budget finite
* H543b  Cost planner V4 abstain-when-restart-violated fires
* H544   Boundary V4 enumerates ≥ 25 blocked axes (V16)
* H544b  Provider filter V3 drops noisy provider under high restart
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_cache_aware_planner_v4 import (
    HostedCacheAwarePlannerV4,
    hosted_cache_aware_savings_v4_vs_recompute,
)
from coordpy.hosted_cost_planner import HostedCostPlanSpec
from coordpy.hosted_cost_planner_v2 import HostedCostPlanSpecV2
from coordpy.hosted_cost_planner_v3 import HostedCostPlanSpecV3
from coordpy.hosted_cost_planner_v4 import (
    HostedCostPlanSpecV4, plan_hosted_cost_v4,
)
from coordpy.hosted_logprob_router import TopKLogprobsPayload
from coordpy.hosted_logprob_router_v4 import HostedLogprobRouterV4
from coordpy.hosted_provider_filter import (
    HostedProviderFilterSpec,
)
from coordpy.hosted_provider_filter_v2 import (
    HostedProviderFilterSpecV2,
)
from coordpy.hosted_provider_filter_v3 import (
    HostedProviderFilterSpecV3, filter_hosted_registry_v3,
)
from coordpy.hosted_real_substrate_boundary_v4 import (
    build_default_hosted_real_substrate_boundary_v4,
)
from coordpy.hosted_router_controller import (
    HostedRoutingRequest, default_hosted_registry,
)
from coordpy.hosted_router_controller_v2 import (
    HostedRoutingRequestV2,
)
from coordpy.hosted_router_controller_v3 import (
    HostedRoutingRequestV3,
)
from coordpy.hosted_router_controller_v4 import (
    HostedRouterControllerV4, HostedRoutingRequestV4,
)


R165_SCHEMA_VERSION: str = "coordpy.r165_benchmark.v1"


def _build_req_v4(
        seed: int, restart_pressure: float = 0.7,
        visible_token_budget: int = 128,
) -> HostedRoutingRequestV4:
    return HostedRoutingRequestV4(
        inner_v3=HostedRoutingRequestV3(
            inner_v2=HostedRoutingRequestV2(
                inner_v1=HostedRoutingRequest(
                    request_cid=f"r540_{int(seed)}",
                    input_tokens=1000,
                    expected_output_tokens=300,
                    require_logprobs=True,
                    require_prefix_cache=True,
                    data_policy_required="no_log",
                    max_latency_ms=2000.0,
                    max_cost_usd=50.0),
                weight_cost=1.0, weight_latency=0.5,
                weight_success=0.3),
            visible_token_budget=int(visible_token_budget),
            baseline_token_cost=512,
            repair_dominance_label=1),
        restart_pressure=float(restart_pressure),
        weight_restart_pressure=0.6,
        weight_delayed_repair_match=0.4)


def family_h540_router_v4_determinism(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc1 = HostedRouterControllerV4.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    rc2 = HostedRouterControllerV4.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    req = _build_req_v4(int(seed))
    d1 = rc1.decide_v4(req)
    d2 = rc2.decide_v4(req)
    return {
        "schema": R165_SCHEMA_VERSION,
        "name": "h540_router_v4_determinism",
        "passed": bool(
            d1.chosen_provider == d2.chosen_provider
            and d1.cid() == d2.cid()),
    }


def family_h540b_router_v4_restart_pressure_score(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc = HostedRouterControllerV4.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    req = _build_req_v4(int(seed), restart_pressure=0.8)
    d = rc.decide_v4(req)
    return {
        "schema": R165_SCHEMA_VERSION,
        "name": "h540b_router_v4_restart_pressure_score",
        "passed": bool(d.restart_pressure_score > 0.0),
        "score": float(d.restart_pressure_score),
    }


def family_h541_logprob_v4_abstain_floor_aggressive(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV4(
        restart_pressure_floor=0.5,
        restart_abstain_floor_delta=1.5)
    payloads = [
        TopKLogprobsPayload("p1", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.1)
            for i in range(8))),
        TopKLogprobsPayload("p2", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.05)
            for i in range(8))),
    ]
    no = lr.fuse_v4(
        payloads, visible_token_budget=256,
        baseline_token_cost=512, restart_pressure=0.0)
    high = lr.fuse_v4(
        payloads, visible_token_budget=256,
        baseline_token_cost=512, restart_pressure=0.9)
    return {
        "schema": R165_SCHEMA_VERSION,
        "name": "h541_logprob_v4_abstain_floor_aggressive",
        "passed": bool(
            high["effective_abstain_entropy_floor"]
            < no["effective_abstain_entropy_floor"]),
        "no": float(no["effective_abstain_entropy_floor"]),
        "high": float(high["effective_abstain_entropy_floor"]),
    }


def family_h541b_logprob_v4_per_budget_restart_tiebreak(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV4(
        restart_pressure_floor=0.5,
        restart_abstain_floor_delta=0.0)
    payloads = [
        TopKLogprobsPayload(
            "p1", (("a", -0.1), ("b", -0.5), ("c", -1.0))),
        TopKLogprobsPayload(
            "p2", (("a", -0.2), ("b", -0.4), ("c", -1.1))),
    ]
    no_restart = lr.fuse_v4(
        payloads, visible_token_budget=512,
        baseline_token_cost=512, top_k=10,
        restart_pressure=0.0)
    with_restart = lr.fuse_v4(
        payloads, visible_token_budget=512,
        baseline_token_cost=512, top_k=10,
        restart_pressure=0.9)
    return {
        "schema": R165_SCHEMA_VERSION,
        "name":
            "h541b_logprob_v4_per_budget_restart_tiebreak",
        "passed": bool(
            int(with_restart.get("effective_top_k_v4", 0))
            < int(no_restart.get("effective_top_k_v4", 0))),
    }


def family_h542_cache_v4_two_layer_cid_content_addressed(
        seed: int) -> dict[str, Any]:
    cp1 = HostedCacheAwarePlannerV4()
    cp2 = HostedCacheAwarePlannerV4()
    planned1, _ = (
        cp1.plan_per_role_two_layer_rotated(
            shared_prefix_text="W71 shared prefix " * 10,
            per_role_blocks={
                "plan": ["a", "b"], "write": ["c", "d"]}))
    planned2, _ = (
        cp2.plan_per_role_two_layer_rotated(
            shared_prefix_text="W71 shared prefix " * 10,
            per_role_blocks={
                "plan": ["a", "b"], "write": ["c", "d"]}))
    ok = (
        planned1[0].coarse_rotated_prefix_cids
        == planned2[0].coarse_rotated_prefix_cids)
    return {
        "schema": R165_SCHEMA_VERSION,
        "name":
            "h542_cache_v4_two_layer_cid_content_addressed",
        "passed": bool(ok),
    }


def family_h542b_cache_v4_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_cache_aware_savings_v4_vs_recompute(
        n_roles=10, n_turns=8,
        hosted_cache_hit_rate=1.0)
    return {
        "schema": R165_SCHEMA_VERSION,
        "name": "h542b_cache_v4_savings",
        "passed": bool(sav["saving_ratio"] >= 0.72),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h543_cost_v4_cps_under_budget(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v4 = HostedCostPlanSpecV4(
        inner_v3=HostedCostPlanSpecV3(
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
            abstain_when_budget_violated=True),
        restart_pressure=0.3,
        restart_pressure_cap=0.7,
        abstain_when_restart_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v4(
        registry=reg, provider_quality_scores=qs,
        spec_v4=spec_v4)
    return {
        "schema": R165_SCHEMA_VERSION,
        "name": "h543_cost_v4_cps_under_budget",
        "passed": bool(
            rep.cost_per_repair_success_under_budget >= 0.0
            and rep.cost_per_repair_success_under_budget
                < float("inf")
            and not rep.restart_violated),
    }


def family_h543b_cost_v4_abstain_restart_violated(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v4 = HostedCostPlanSpecV4(
        inner_v3=HostedCostPlanSpecV3(
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
            abstain_when_budget_violated=True),
        restart_pressure=0.9,
        restart_pressure_cap=0.7,
        abstain_when_restart_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v4(
        registry=reg, provider_quality_scores=qs,
        spec_v4=spec_v4)
    return {
        "schema": R165_SCHEMA_VERSION,
        "name": "h543b_cost_v4_abstain_restart_violated",
        "passed": bool(
            rep.restart_violated
            and "abstain_v4" in str(rep.rationale_v4)),
    }


def family_h544_boundary_v4_blocked_axes(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v4()
    return {
        "schema": R165_SCHEMA_VERSION,
        "name": "h544_boundary_v4_blocked_axes",
        "passed": bool(len(b.blocked_axes) >= 25),
        "n_blocked": int(len(b.blocked_axes)),
    }


def family_h544b_provider_filter_v3_restart_drop(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v3 = HostedProviderFilterSpecV3(
        inner_v2=HostedProviderFilterSpecV2(
            inner_specs=(
                HostedProviderFilterSpec(
                    require_data_policy="no_log",
                    allowed_tiers=(
                        "logprobs",
                        "logprobs_and_prefix_cache",
                        "prefix_cache",
                        "text_only"),
                    max_p50_latency_ms=10_000.0,
                    max_cost_per_1k_output=100.0),
            ),
            combine="all",
            tier_weights={"text_only": 0.5}),
        restart_pressure=0.9,
        restart_pressure_floor=0.5,
        max_restart_noise_per_provider={
            "openrouter_paid": 0.3,
            "openai_paid": 1.0})
    # Set noise for a "noisy" provider above its cap.
    pn = {
        "openrouter_paid": 0.5,  # > 0.3 cap → drop
        "openai_paid": 0.1,
    }
    filt, rep = filter_hosted_registry_v3(reg, spec_v3, pn)
    return {
        "schema": R165_SCHEMA_VERSION,
        "name": "h544b_provider_filter_v3_restart_drop",
        "passed": bool(
            int(rep.get("n_dropped_under_restart", 0)) >= 1
            and "openrouter_paid"
            in list(rep.get("dropped_under_restart", []))),
        "n_dropped": int(rep.get(
            "n_dropped_under_restart", 0)),
    }


_R165_FAMILIES: tuple[Any, ...] = (
    family_h540_router_v4_determinism,
    family_h540b_router_v4_restart_pressure_score,
    family_h541_logprob_v4_abstain_floor_aggressive,
    family_h541b_logprob_v4_per_budget_restart_tiebreak,
    family_h542_cache_v4_two_layer_cid_content_addressed,
    family_h542b_cache_v4_savings,
    family_h543_cost_v4_cps_under_budget,
    family_h543b_cost_v4_abstain_restart_violated,
    family_h544_boundary_v4_blocked_axes,
    family_h544b_provider_filter_v3_restart_drop,
)


def run_r165(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R165_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R165_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R165_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R165_SCHEMA_VERSION", "run_r165"]
