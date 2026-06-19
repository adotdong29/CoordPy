"""W74 R-177 benchmark family — Hosted Control Plane V7 (Plane A).

Exercises the W74 hosted control plane V7 modules: router V7
(compound-pressure + compound-repair-after-DRTR match), logprob
router V7 (compound-aware abstain floor + per-budget+restart+
rejoin+replacement+compound tiebreak), cache-aware planner V7
(five-layer rotated), cost planner V7 (cost-per-compound-success-
under-budget + abstain-when-compound-pressure-violated), boundary
V7 (≥ 34 blocked axes), provider filter V6 (compound-aware drop).

H880..H888 cell families (10 H-bars):

* H880   Router V7 determinism on (registry CID, request V7 CID,
         budget, restart_pressure, rejoin_pressure,
         replacement_pressure, compound_pressure,
         repair_dominance_label)
* H880b  Router V7 compound-pressure score > 0 under high pressure
* H881   Logprob V7 abstain floor more aggressive under compound
* H881b  Logprob V7 per-budget+...+compound tiebreak narrows top-k
* H882   Cache planner V7 five-layer rotated CIDs content-addressed
* H882b  Cache planner V7 ≥ 0.85 saving on 16×8 at hit_rate=1.0
* H883   Cost planner V7 cost-per-compound-success-under-budget
         finite
* H883b  Cost planner V7 abstain-when-compound-violated fires
* H884   Boundary V7 enumerates ≥ 34 blocked axes (V19)
* H884b  Provider filter V6 drops noisy provider under high
         compound
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_cache_aware_planner_v7 import (
    HostedCacheAwarePlannerV7,
    hosted_cache_aware_savings_v7_vs_recompute,
)
from coordpy.hosted_cost_planner import HostedCostPlanSpec
from coordpy.hosted_cost_planner_v2 import HostedCostPlanSpecV2
from coordpy.hosted_cost_planner_v3 import HostedCostPlanSpecV3
from coordpy.hosted_cost_planner_v4 import HostedCostPlanSpecV4
from coordpy.hosted_cost_planner_v5 import HostedCostPlanSpecV5
from coordpy.hosted_cost_planner_v6 import HostedCostPlanSpecV6
from coordpy.hosted_cost_planner_v7 import (
    HostedCostPlanSpecV7, plan_hosted_cost_v7,
)
from coordpy.hosted_logprob_router import TopKLogprobsPayload
from coordpy.hosted_logprob_router_v7 import HostedLogprobRouterV7
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
    HostedProviderFilterSpecV4,
)
from coordpy.hosted_provider_filter_v5 import (
    HostedProviderFilterSpecV5,
)
from coordpy.hosted_provider_filter_v6 import (
    HostedProviderFilterSpecV6, filter_hosted_registry_v6,
)
from coordpy.hosted_real_substrate_boundary_v7 import (
    build_default_hosted_real_substrate_boundary_v7,
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
    HostedRoutingRequestV5,
)
from coordpy.hosted_router_controller_v6 import (
    HostedRoutingRequestV6,
)
from coordpy.hosted_router_controller_v7 import (
    HostedRouterControllerV7, HostedRoutingRequestV7,
)


R177_SCHEMA_VERSION: str = "coordpy.r177_benchmark.v1"


def _build_req_v7(
        seed: int, restart_pressure: float = 0.7,
        rejoin_pressure: float = 0.7,
        replacement_pressure: float = 0.7,
        compound_pressure: float = 0.7,
        visible_token_budget: int = 128,
) -> HostedRoutingRequestV7:
    return HostedRoutingRequestV7(
        inner_v6=HostedRoutingRequestV6(
            inner_v5=HostedRoutingRequestV5(
                inner_v4=HostedRoutingRequestV4(
                    inner_v3=HostedRoutingRequestV3(
                        inner_v2=HostedRoutingRequestV2(
                            inner_v1=HostedRoutingRequest(
                                request_cid=f"r880_{int(seed)}",
                                input_tokens=1000,
                                expected_output_tokens=300,
                                require_logprobs=True,
                                require_prefix_cache=True,
                                data_policy_required="no_log",
                                max_latency_ms=2000.0,
                                max_cost_usd=50.0),
                            weight_cost=1.0,
                            weight_latency=0.5,
                            weight_success=0.3),
                        visible_token_budget=int(
                            visible_token_budget),
                        baseline_token_cost=512,
                        repair_dominance_label=1),
                    restart_pressure=float(restart_pressure),
                    weight_restart_pressure=0.6,
                    weight_delayed_repair_match=0.4),
                rejoin_pressure=float(rejoin_pressure),
                weight_rejoin_pressure=0.6,
                weight_delayed_rejoin_match=0.4),
            replacement_pressure=float(replacement_pressure),
            weight_replacement_pressure=0.6,
            weight_replacement_after_ctr_match=0.4),
        compound_pressure=float(compound_pressure),
        weight_compound_pressure=0.6,
        weight_compound_repair_drtr_match=0.4)


def family_h880_router_v7_determinism(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc1 = HostedRouterControllerV7.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    rc2 = HostedRouterControllerV7.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    req = _build_req_v7(int(seed))
    d1 = rc1.decide_v7(req)
    d2 = rc2.decide_v7(req)
    return {
        "schema": R177_SCHEMA_VERSION,
        "name": "h880_router_v7_determinism",
        "passed": bool(
            d1.chosen_provider == d2.chosen_provider
            and d1.cid() == d2.cid()),
    }


def family_h880b_router_v7_compound_pressure_score(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc = HostedRouterControllerV7.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    req = _build_req_v7(int(seed), compound_pressure=0.8)
    d = rc.decide_v7(req)
    return {
        "schema": R177_SCHEMA_VERSION,
        "name": "h880b_router_v7_compound_pressure_score",
        "passed": bool(d.compound_pressure_score > 0.0),
        "score": float(d.compound_pressure_score),
    }


def family_h881_logprob_v7_abstain_floor_aggressive(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV7(
        compound_pressure_floor=0.5,
        compound_abstain_floor_delta=1.8)
    payloads = [
        TopKLogprobsPayload("p1", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.1)
            for i in range(8))),
        TopKLogprobsPayload("p2", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.05)
            for i in range(8))),
    ]
    no = lr.fuse_v7(
        payloads, visible_token_budget=256,
        baseline_token_cost=512,
        restart_pressure=0.0, rejoin_pressure=0.0,
        replacement_pressure=0.0, compound_pressure=0.0)
    high = lr.fuse_v7(
        payloads, visible_token_budget=256,
        baseline_token_cost=512,
        restart_pressure=0.0, rejoin_pressure=0.0,
        replacement_pressure=0.0, compound_pressure=0.9)
    return {
        "schema": R177_SCHEMA_VERSION,
        "name": "h881_logprob_v7_abstain_floor_aggressive",
        "passed": bool(
            high["effective_abstain_entropy_floor"]
            < no["effective_abstain_entropy_floor"]),
        "no": float(no["effective_abstain_entropy_floor"]),
        "high": float(high["effective_abstain_entropy_floor"]),
    }


def family_h881b_logprob_v7_combined_tiebreak(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV7(
        compound_pressure_floor=0.5,
        compound_abstain_floor_delta=0.0)
    payloads = [
        TopKLogprobsPayload(
            "p1", (("a", -0.1), ("b", -0.5), ("c", -1.0))),
        TopKLogprobsPayload(
            "p2", (("a", -0.2), ("b", -0.4), ("c", -1.1))),
    ]
    no_cmp = lr.fuse_v7(
        payloads, visible_token_budget=512,
        baseline_token_cost=512, top_k=10,
        restart_pressure=0.0, rejoin_pressure=0.0,
        replacement_pressure=0.0, compound_pressure=0.0)
    with_cmp = lr.fuse_v7(
        payloads, visible_token_budget=512,
        baseline_token_cost=512, top_k=10,
        restart_pressure=0.0, rejoin_pressure=0.0,
        replacement_pressure=0.0, compound_pressure=0.9)
    return {
        "schema": R177_SCHEMA_VERSION,
        "name": "h881b_logprob_v7_combined_tiebreak",
        "passed": bool(
            int(with_cmp.get("effective_top_k_v7", 0))
            < int(no_cmp.get("effective_top_k_v7", 0))),
    }


def family_h882_cache_v7_five_layer_cid_content_addressed(
        seed: int) -> dict[str, Any]:
    cp1 = HostedCacheAwarePlannerV7()
    cp2 = HostedCacheAwarePlannerV7()
    planned1, _ = (
        cp1.plan_per_role_five_layer_rotated(
            shared_prefix_text="W74 shared prefix " * 16,
            per_role_blocks={
                "plan": ["a", "b"], "write": ["c", "d"]}))
    planned2, _ = (
        cp2.plan_per_role_five_layer_rotated(
            shared_prefix_text="W74 shared prefix " * 16,
            per_role_blocks={
                "plan": ["a", "b"], "write": ["c", "d"]}))
    ok = (
        planned1[0].giga_coarse_rotated_prefix_cids
        == planned2[0].giga_coarse_rotated_prefix_cids)
    return {
        "schema": R177_SCHEMA_VERSION,
        "name":
            "h882_cache_v7_five_layer_cid_content_addressed",
        "passed": bool(ok),
    }


def family_h882b_cache_v7_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_cache_aware_savings_v7_vs_recompute(
        n_roles=16, n_turns=8,
        hosted_cache_hit_rate=1.0)
    return {
        "schema": R177_SCHEMA_VERSION,
        "name": "h882b_cache_v7_savings",
        "passed": bool(sav["saving_ratio"] >= 0.85),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h883_cost_v7_cps_under_budget(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v7 = HostedCostPlanSpecV7(
        inner_v6=HostedCostPlanSpecV6(
            inner_v5=HostedCostPlanSpecV5(
                inner_v4=HostedCostPlanSpecV4(
                    inner_v3=HostedCostPlanSpecV3(
                        inner_v2=HostedCostPlanSpecV2(
                            inner_v1=HostedCostPlanSpec(
                                n_turns=5,
                                input_tokens_per_turn=200,
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
                abstain_when_rejoin_violated=True),
            replacement_pressure=0.3,
            replacement_pressure_cap=0.7,
            abstain_when_replacement_violated=True),
        compound_pressure=0.3,
        compound_pressure_cap=0.7,
        abstain_when_compound_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v7(
        registry=reg, provider_quality_scores=qs,
        spec_v7=spec_v7)
    return {
        "schema": R177_SCHEMA_VERSION,
        "name": "h883_cost_v7_cps_under_budget",
        "passed": bool(
            rep.cost_per_compound_success_under_budget
            >= 0.0
            and rep.cost_per_compound_success_under_budget
                < float("inf")
            and not rep.compound_violated),
    }


def family_h883b_cost_v7_abstain_compound_violated(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v7 = HostedCostPlanSpecV7(
        inner_v6=HostedCostPlanSpecV6(
            inner_v5=HostedCostPlanSpecV5(
                inner_v4=HostedCostPlanSpecV4(
                    inner_v3=HostedCostPlanSpecV3(
                        inner_v2=HostedCostPlanSpecV2(
                            inner_v1=HostedCostPlanSpec(
                                n_turns=5,
                                input_tokens_per_turn=200,
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
                abstain_when_rejoin_violated=True),
            replacement_pressure=0.3,
            replacement_pressure_cap=0.7,
            abstain_when_replacement_violated=True),
        compound_pressure=0.9,
        compound_pressure_cap=0.7,
        abstain_when_compound_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v7(
        registry=reg, provider_quality_scores=qs,
        spec_v7=spec_v7)
    return {
        "schema": R177_SCHEMA_VERSION,
        "name": "h883b_cost_v7_abstain_compound_violated",
        "passed": bool(
            rep.compound_violated
            and "abstain_v7" in str(rep.rationale_v7)),
    }


def family_h884_boundary_v7_blocked_axes(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v7()
    return {
        "schema": R177_SCHEMA_VERSION,
        "name": "h884_boundary_v7_blocked_axes",
        "passed": bool(len(b.blocked_axes) >= 34),
        "n_blocked": int(len(b.blocked_axes)),
    }


def family_h884b_provider_filter_v6_compound_drop(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v6 = HostedProviderFilterSpecV6(
        inner_v5=HostedProviderFilterSpecV5(
            inner_v4=HostedProviderFilterSpecV4(
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
                rejoin_pressure=0.0,
                rejoin_pressure_floor=0.5,
                max_rejoin_noise_per_provider={
                    "openrouter_paid": 1.0,
                    "openai_paid": 1.0}),
            replacement_pressure=0.0,
            replacement_pressure_floor=0.5,
            max_replacement_noise_per_provider={
                "openrouter_paid": 1.0,
                "openai_paid": 1.0}),
        compound_pressure=0.9,
        compound_pressure_floor=0.5,
        max_compound_noise_per_provider={
            "openrouter_paid": 0.18,
            "openai_paid": 1.0})
    pn_compound = {
        "openrouter_paid": 0.35,  # > 0.18 cap → drop
        "openai_paid": 0.05,
    }
    filt, rep = filter_hosted_registry_v6(
        reg, spec_v6,
        provider_restart_noise={},
        provider_rejoin_noise={},
        provider_replacement_noise={},
        provider_compound_noise=pn_compound)
    return {
        "schema": R177_SCHEMA_VERSION,
        "name": "h884b_provider_filter_v6_compound_drop",
        "passed": bool(
            int(rep.get("n_dropped_under_compound", 0)) >= 1
            and "openrouter_paid"
            in list(rep.get("dropped_under_compound", []))),
        "n_dropped": int(rep.get(
            "n_dropped_under_compound", 0)),
    }


_R177_FAMILIES: tuple[Any, ...] = (
    family_h880_router_v7_determinism,
    family_h880b_router_v7_compound_pressure_score,
    family_h881_logprob_v7_abstain_floor_aggressive,
    family_h881b_logprob_v7_combined_tiebreak,
    family_h882_cache_v7_five_layer_cid_content_addressed,
    family_h882b_cache_v7_savings,
    family_h883_cost_v7_cps_under_budget,
    family_h883b_cost_v7_abstain_compound_violated,
    family_h884_boundary_v7_blocked_axes,
    family_h884b_provider_filter_v6_compound_drop,
)


def run_r177(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R177_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R177_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R177_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R177_SCHEMA_VERSION", "run_r177"]
