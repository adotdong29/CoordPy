"""W72 R-169 benchmark family — Hosted Control Plane V5 (Plane A).

Exercises the W72 hosted control plane V5 modules: router V5
(rejoin-pressure + delayed-rejoin match), logprob router V5
(rejoin-aware abstain floor + per-budget+restart+rejoin tiebreak),
cache-aware planner V5 (three-layer rotated), cost planner V5
(cost-per-rejoin-success-under-budget + abstain-when-rejoin-
pressure-violated), boundary V5 (≥ 28 blocked axes), provider
filter V4 (rejoin-aware drop).

H700..H708 cell families (10 H-bars):

* H700   Router V5 determinism on (registry CID, request V5 CID,
         budget, restart_pressure, rejoin_pressure,
         repair_dominance_label)
* H700b  Router V5 rejoin-pressure score > 0 under high pressure
* H701   Logprob V5 abstain floor more aggressive under rejoin
* H701b  Logprob V5 per-budget+restart+rejoin tiebreak narrows
         top-k further
* H702   Cache planner V5 three-layer rotated CIDs content-addressed
* H702b  Cache planner V5 ≥ 80 % saving on 12×8 at hit_rate=1.0
* H703   Cost planner V5 cost-per-rejoin-success-under-budget finite
* H703b  Cost planner V5 abstain-when-rejoin-violated fires
* H704   Boundary V5 enumerates ≥ 28 blocked axes (V17)
* H704b  Provider filter V4 drops noisy provider under high rejoin
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_cache_aware_planner_v5 import (
    HostedCacheAwarePlannerV5,
    hosted_cache_aware_savings_v5_vs_recompute,
)
from coordpy.hosted_cost_planner import HostedCostPlanSpec
from coordpy.hosted_cost_planner_v2 import HostedCostPlanSpecV2
from coordpy.hosted_cost_planner_v3 import HostedCostPlanSpecV3
from coordpy.hosted_cost_planner_v4 import HostedCostPlanSpecV4
from coordpy.hosted_cost_planner_v5 import (
    HostedCostPlanSpecV5, plan_hosted_cost_v5,
)
from coordpy.hosted_logprob_router import TopKLogprobsPayload
from coordpy.hosted_logprob_router_v5 import HostedLogprobRouterV5
from coordpy.hosted_provider_filter import (
    HostedProviderFilterSpec,
)
from coordpy.hosted_provider_filter_v2 import (
    HostedProviderFilterSpecV2,
)
from coordpy.hosted_provider_filter_v3 import (
    HostedProviderFilterSpecV3,
)
from coordpy.hosted_provider_filter_v4 import (
    HostedProviderFilterSpecV4, filter_hosted_registry_v4,
)
from coordpy.hosted_real_substrate_boundary_v5 import (
    build_default_hosted_real_substrate_boundary_v5,
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
    HostedRoutingRequestV4,
)
from coordpy.hosted_router_controller_v5 import (
    HostedRouterControllerV5, HostedRoutingRequestV5,
)


R169_SCHEMA_VERSION: str = "coordpy.r169_benchmark.v1"


def _build_req_v5(
        seed: int, restart_pressure: float = 0.7,
        rejoin_pressure: float = 0.7,
        visible_token_budget: int = 128,
) -> HostedRoutingRequestV5:
    return HostedRoutingRequestV5(
        inner_v4=HostedRoutingRequestV4(
            inner_v3=HostedRoutingRequestV3(
                inner_v2=HostedRoutingRequestV2(
                    inner_v1=HostedRoutingRequest(
                        request_cid=f"r700_{int(seed)}",
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
            weight_delayed_repair_match=0.4),
        rejoin_pressure=float(rejoin_pressure),
        weight_rejoin_pressure=0.6,
        weight_delayed_rejoin_match=0.4)


def family_h700_router_v5_determinism(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc1 = HostedRouterControllerV5.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    rc2 = HostedRouterControllerV5.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    req = _build_req_v5(int(seed))
    d1 = rc1.decide_v5(req)
    d2 = rc2.decide_v5(req)
    return {
        "schema": R169_SCHEMA_VERSION,
        "name": "h700_router_v5_determinism",
        "passed": bool(
            d1.chosen_provider == d2.chosen_provider
            and d1.cid() == d2.cid()),
    }


def family_h700b_router_v5_rejoin_pressure_score(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc = HostedRouterControllerV5.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    req = _build_req_v5(int(seed), rejoin_pressure=0.8)
    d = rc.decide_v5(req)
    return {
        "schema": R169_SCHEMA_VERSION,
        "name": "h700b_router_v5_rejoin_pressure_score",
        "passed": bool(d.rejoin_pressure_score > 0.0),
        "score": float(d.rejoin_pressure_score),
    }


def family_h701_logprob_v5_abstain_floor_aggressive(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV5(
        rejoin_pressure_floor=0.5,
        rejoin_abstain_floor_delta=1.5)
    payloads = [
        TopKLogprobsPayload("p1", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.1)
            for i in range(8))),
        TopKLogprobsPayload("p2", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.05)
            for i in range(8))),
    ]
    no = lr.fuse_v5(
        payloads, visible_token_budget=256,
        baseline_token_cost=512,
        restart_pressure=0.0, rejoin_pressure=0.0)
    high = lr.fuse_v5(
        payloads, visible_token_budget=256,
        baseline_token_cost=512,
        restart_pressure=0.0, rejoin_pressure=0.9)
    return {
        "schema": R169_SCHEMA_VERSION,
        "name": "h701_logprob_v5_abstain_floor_aggressive",
        "passed": bool(
            high["effective_abstain_entropy_floor"]
            < no["effective_abstain_entropy_floor"]),
        "no": float(no["effective_abstain_entropy_floor"]),
        "high": float(high["effective_abstain_entropy_floor"]),
    }


def family_h701b_logprob_v5_combined_tiebreak(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV5(
        rejoin_pressure_floor=0.5,
        rejoin_abstain_floor_delta=0.0)
    payloads = [
        TopKLogprobsPayload(
            "p1", (("a", -0.1), ("b", -0.5), ("c", -1.0))),
        TopKLogprobsPayload(
            "p2", (("a", -0.2), ("b", -0.4), ("c", -1.1))),
    ]
    no_rejoin = lr.fuse_v5(
        payloads, visible_token_budget=512,
        baseline_token_cost=512, top_k=10,
        restart_pressure=0.0, rejoin_pressure=0.0)
    with_rejoin = lr.fuse_v5(
        payloads, visible_token_budget=512,
        baseline_token_cost=512, top_k=10,
        restart_pressure=0.0, rejoin_pressure=0.9)
    return {
        "schema": R169_SCHEMA_VERSION,
        "name": "h701b_logprob_v5_combined_tiebreak",
        "passed": bool(
            int(with_rejoin.get("effective_top_k_v5", 0))
            < int(no_rejoin.get("effective_top_k_v5", 0))),
    }


def family_h702_cache_v5_three_layer_cid_content_addressed(
        seed: int) -> dict[str, Any]:
    cp1 = HostedCacheAwarePlannerV5()
    cp2 = HostedCacheAwarePlannerV5()
    planned1, _ = (
        cp1.plan_per_role_three_layer_rotated(
            shared_prefix_text="W72 shared prefix " * 12,
            per_role_blocks={
                "plan": ["a", "b"], "write": ["c", "d"]}))
    planned2, _ = (
        cp2.plan_per_role_three_layer_rotated(
            shared_prefix_text="W72 shared prefix " * 12,
            per_role_blocks={
                "plan": ["a", "b"], "write": ["c", "d"]}))
    ok = (
        planned1[0].ultra_coarse_rotated_prefix_cids
        == planned2[0].ultra_coarse_rotated_prefix_cids)
    return {
        "schema": R169_SCHEMA_VERSION,
        "name":
            "h702_cache_v5_three_layer_cid_content_addressed",
        "passed": bool(ok),
    }


def family_h702b_cache_v5_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_cache_aware_savings_v5_vs_recompute(
        n_roles=12, n_turns=8,
        hosted_cache_hit_rate=1.0)
    return {
        "schema": R169_SCHEMA_VERSION,
        "name": "h702b_cache_v5_savings",
        "passed": bool(sav["saving_ratio"] >= 0.80),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h703_cost_v5_cps_under_budget(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v5 = HostedCostPlanSpecV5(
        inner_v4=HostedCostPlanSpecV4(
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
            abstain_when_restart_violated=True),
        rejoin_pressure=0.3,
        rejoin_pressure_cap=0.7,
        abstain_when_rejoin_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v5(
        registry=reg, provider_quality_scores=qs,
        spec_v5=spec_v5)
    return {
        "schema": R169_SCHEMA_VERSION,
        "name": "h703_cost_v5_cps_under_budget",
        "passed": bool(
            rep.cost_per_rejoin_success_under_budget >= 0.0
            and rep.cost_per_rejoin_success_under_budget
                < float("inf")
            and not rep.rejoin_violated),
    }


def family_h703b_cost_v5_abstain_rejoin_violated(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v5 = HostedCostPlanSpecV5(
        inner_v4=HostedCostPlanSpecV4(
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
            abstain_when_restart_violated=True),
        rejoin_pressure=0.9,
        rejoin_pressure_cap=0.7,
        abstain_when_rejoin_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v5(
        registry=reg, provider_quality_scores=qs,
        spec_v5=spec_v5)
    return {
        "schema": R169_SCHEMA_VERSION,
        "name": "h703b_cost_v5_abstain_rejoin_violated",
        "passed": bool(
            rep.rejoin_violated
            and "abstain_v5" in str(rep.rationale_v5)),
    }


def family_h704_boundary_v5_blocked_axes(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v5()
    return {
        "schema": R169_SCHEMA_VERSION,
        "name": "h704_boundary_v5_blocked_axes",
        "passed": bool(len(b.blocked_axes) >= 28),
        "n_blocked": int(len(b.blocked_axes)),
    }


def family_h704b_provider_filter_v4_rejoin_drop(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v4 = HostedProviderFilterSpecV4(
        inner_v3=HostedProviderFilterSpecV3(
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
            restart_pressure=0.0,
            restart_pressure_floor=0.5,
            max_restart_noise_per_provider={
                "openrouter_paid": 1.0,
                "openai_paid": 1.0}),
        rejoin_pressure=0.9,
        rejoin_pressure_floor=0.5,
        max_rejoin_noise_per_provider={
            "openrouter_paid": 0.25,
            "openai_paid": 1.0})
    pn_rejoin = {
        "openrouter_paid": 0.4,  # > 0.25 cap → drop
        "openai_paid": 0.1,
    }
    filt, rep = filter_hosted_registry_v4(
        reg, spec_v4,
        provider_restart_noise={},
        provider_rejoin_noise=pn_rejoin)
    return {
        "schema": R169_SCHEMA_VERSION,
        "name": "h704b_provider_filter_v4_rejoin_drop",
        "passed": bool(
            int(rep.get("n_dropped_under_rejoin", 0)) >= 1
            and "openrouter_paid"
            in list(rep.get("dropped_under_rejoin", []))),
        "n_dropped": int(rep.get(
            "n_dropped_under_rejoin", 0)),
    }


_R169_FAMILIES: tuple[Any, ...] = (
    family_h700_router_v5_determinism,
    family_h700b_router_v5_rejoin_pressure_score,
    family_h701_logprob_v5_abstain_floor_aggressive,
    family_h701b_logprob_v5_combined_tiebreak,
    family_h702_cache_v5_three_layer_cid_content_addressed,
    family_h702b_cache_v5_savings,
    family_h703_cost_v5_cps_under_budget,
    family_h703b_cost_v5_abstain_rejoin_violated,
    family_h704_boundary_v5_blocked_axes,
    family_h704b_provider_filter_v4_rejoin_drop,
)


def run_r169(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R169_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R169_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R169_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R169_SCHEMA_VERSION", "run_r169"]
