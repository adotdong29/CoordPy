"""W76 R-185 benchmark family — Hosted Control Plane V9 (Plane A).

Exercises the W76 hosted control plane V9 modules: router V9
(chain-then-restart-pressure + chain-then-restart-after-RTR
match), logprob router V9 (chain-then-restart-aware abstain floor
+ per-budget+restart+rejoin+replacement+compound+chain+chain-then-
restart tiebreak), cache-aware planner V9 (seven-layer rotated),
cost planner V9 (cost-per-chain-then-restart-success-under-budget
+ abstain-when-chain-then-restart-pressure-violated), boundary V9
(≥ 40 blocked axes), provider filter V8 (chain-then-restart-aware
drop).

H1080..H1088 cell families (10 H-bars):

* H1080   Router V9 determinism on (registry CID, request V9
          CID, budget, all pressures, repair_dominance_label)
* H1080b  Router V9 chain-then-restart-pressure score > 0 under
          high pressure
* H1081   Logprob V9 abstain floor more aggressive under
          chain-then-restart
* H1081b  Logprob V9 per-budget+...+chain-then-restart tiebreak
          narrows top-k
* H1082   Cache planner V9 seven-layer rotated CIDs content-
          addressed
* H1082b  Cache planner V9 ≥ 0.88 saving on 20×8 at hit_rate=1.0
* H1083   Cost planner V9 cost-per-chain-then-restart-success-
          under-budget finite
* H1083b  Cost planner V9 abstain-when-chain-then-restart-
          violated fires
* H1084   Boundary V9 enumerates ≥ 40 blocked axes (V21)
* H1084b  Provider filter V8 drops noisy provider under high
          chain-then-restart pressure
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_cache_aware_planner_v9 import (
    HostedCacheAwarePlannerV9,
    hosted_cache_aware_savings_v9_vs_recompute,
)
from coordpy.hosted_cost_planner import HostedCostPlanSpec
from coordpy.hosted_cost_planner_v2 import HostedCostPlanSpecV2
from coordpy.hosted_cost_planner_v3 import HostedCostPlanSpecV3
from coordpy.hosted_cost_planner_v4 import HostedCostPlanSpecV4
from coordpy.hosted_cost_planner_v5 import HostedCostPlanSpecV5
from coordpy.hosted_cost_planner_v6 import HostedCostPlanSpecV6
from coordpy.hosted_cost_planner_v7 import HostedCostPlanSpecV7
from coordpy.hosted_cost_planner_v8 import HostedCostPlanSpecV8
from coordpy.hosted_cost_planner_v9 import (
    HostedCostPlanSpecV9, plan_hosted_cost_v9,
)
from coordpy.hosted_logprob_router import TopKLogprobsPayload
from coordpy.hosted_logprob_router_v9 import HostedLogprobRouterV9
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
    HostedProviderFilterSpecV6,
)
from coordpy.hosted_provider_filter_v7 import (
    HostedProviderFilterSpecV7,
)
from coordpy.hosted_provider_filter_v8 import (
    HostedProviderFilterSpecV8, filter_hosted_registry_v8,
)
from coordpy.hosted_real_substrate_boundary_v9 import (
    build_default_hosted_real_substrate_boundary_v9,
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
    HostedRoutingRequestV7,
)
from coordpy.hosted_router_controller_v8 import (
    HostedRoutingRequestV8,
)
from coordpy.hosted_router_controller_v9 import (
    HostedRouterControllerV9, HostedRoutingRequestV9,
)


R185_SCHEMA_VERSION: str = "coordpy.r185_benchmark.v1"


def _build_req_v9(
        seed: int, restart_pressure: float = 0.7,
        rejoin_pressure: float = 0.7,
        replacement_pressure: float = 0.7,
        compound_pressure: float = 0.7,
        compound_chain_pressure: float = 0.7,
        compound_chain_then_restart_pressure: float = 0.7,
) -> HostedRoutingRequestV9:
    return HostedRoutingRequestV9(
        inner_v8=HostedRoutingRequestV8(
            inner_v7=HostedRoutingRequestV7(
                inner_v6=HostedRoutingRequestV6(
                    inner_v5=HostedRoutingRequestV5(
                        inner_v4=HostedRoutingRequestV4(
                            inner_v3=HostedRoutingRequestV3(
                                inner_v2=HostedRoutingRequestV2(
                                    inner_v1=HostedRoutingRequest(
                                        request_cid=f"r185-{seed}",
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
                                visible_token_budget=128,
                                baseline_token_cost=512,
                                repair_dominance_label=1),
                            restart_pressure=float(
                                restart_pressure),
                            weight_restart_pressure=0.6,
                            weight_delayed_repair_match=0.4),
                        rejoin_pressure=float(rejoin_pressure),
                        weight_rejoin_pressure=0.6,
                        weight_delayed_rejoin_match=0.4),
                    replacement_pressure=float(
                        replacement_pressure),
                    weight_replacement_pressure=0.6,
                    weight_replacement_after_ctr_match=0.4),
                compound_pressure=float(compound_pressure),
                weight_compound_pressure=0.6,
                weight_compound_repair_drtr_match=0.4),
            compound_chain_pressure=float(
                compound_chain_pressure),
            weight_compound_chain_pressure=0.6,
            weight_compound_chain_repair_rtr_match=0.4),
        compound_chain_then_restart_pressure=float(
            compound_chain_then_restart_pressure),
        weight_compound_chain_then_restart_pressure=0.6,
        weight_chain_then_restart_after_rtr_match=0.4)


def _build_provider_filter_v8() -> HostedProviderFilterSpecV8:
    return HostedProviderFilterSpecV8(
        inner_v7=HostedProviderFilterSpecV7(
            inner_v6=HostedProviderFilterSpecV6(
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
                                tier_weights={
                                    "logprobs_and_prefix_cache": 1.0,
                                    "logprobs": 0.8,
                                    "prefix_cache": 0.7,
                                    "text_only": 0.5}),
                            restart_pressure=0.7,
                            restart_pressure_floor=0.5,
                            max_restart_noise_per_provider={
                                "openrouter_paid": 0.3,
                                "openai_paid": 1.0},
                            restart_tier_weights={
                                "logprobs_and_prefix_cache": 1.0}),
                        rejoin_pressure=0.7,
                        rejoin_pressure_floor=0.5,
                        max_rejoin_noise_per_provider={
                            "openrouter_paid": 0.25,
                            "openai_paid": 1.0},
                        rejoin_tier_weights={
                            "logprobs_and_prefix_cache": 1.0}),
                    replacement_pressure=0.7,
                    replacement_pressure_floor=0.5,
                    max_replacement_noise_per_provider={
                        "openrouter_paid": 0.20,
                        "openai_paid": 1.0},
                    replacement_tier_weights={
                        "logprobs_and_prefix_cache": 1.0}),
                compound_pressure=0.7,
                compound_pressure_floor=0.5,
                max_compound_noise_per_provider={
                    "openrouter_paid": 0.18,
                    "openai_paid": 1.0},
                compound_tier_weights={
                    "logprobs_and_prefix_cache": 1.0}),
            compound_chain_pressure=0.7,
            compound_chain_pressure_floor=0.5,
            max_compound_chain_noise_per_provider={
                "openrouter_paid": 0.15,
                "openai_paid": 1.0},
            compound_chain_tier_weights={
                "logprobs_and_prefix_cache": 1.0}),
        chain_then_restart_pressure=0.85,
        chain_then_restart_pressure_floor=0.5,
        max_chain_then_restart_noise_per_provider={
            "openrouter_paid": 0.12,
            "openai_paid": 1.0},
        chain_then_restart_tier_weights={
            "logprobs_and_prefix_cache": 1.0})


def run_r185(*, seeds: Sequence[int]) -> dict[str, Any]:
    reg = default_hosted_registry()
    cells: dict[str, Any] = {}
    # H1080: Router V9 determinism.
    ctrl = HostedRouterControllerV9.init(
        reg, {"openrouter_paid": 0.86, "openai_paid": 0.93})
    seen_routing_cids: dict[int, str] = {}
    ctr_scores: list[float] = []
    for s in seeds:
        req = _build_req_v9(int(s))
        d = ctrl.decide_v9(req)
        seen_routing_cids[int(s)] = (
            d.per_budget_restart_rejoin_replacement_compound_chain_then_restart_routing_cid)
        ctr_scores.append(
            float(d.chain_then_restart_pressure_score))
    # Re-run to check determinism.
    ctrl2 = HostedRouterControllerV9.init(
        reg, {"openrouter_paid": 0.86, "openai_paid": 0.93})
    deterministic = True
    for s in seeds:
        req = _build_req_v9(int(s))
        d = ctrl2.decide_v9(req)
        if (str(d
                .per_budget_restart_rejoin_replacement_compound_chain_then_restart_routing_cid)
                != str(seen_routing_cids[int(s)])):
            deterministic = False
            break
    cells["H1080"] = bool(deterministic)
    cells["H1080b"] = bool(
        all(float(x) > 0.0 for x in ctr_scores))
    # H1081: Logprob V9 abstain floor under chain-then-restart.
    payloads = [
        TopKLogprobsPayload(
            provider="openai_paid",
            token_to_logprob=(
                ("a", -0.1), ("b", -2.0), ("c", -3.0))),
    ]
    router = HostedLogprobRouterV9()
    res_low = router.fuse_v9(
        payloads, chain_then_restart_pressure=0.0)
    res_high = router.fuse_v9(
        payloads, chain_then_restart_pressure=0.9)
    cells["H1081"] = bool(
        float(res_high.get(
            "effective_abstain_entropy_floor", 1e9))
        < float(res_low.get(
            "effective_abstain_entropy_floor", 0.0)))
    cells["H1081b"] = bool(
        int(res_high.get("effective_top_k_v9", 1000))
        <= int(res_low.get("effective_top_k_v9", 0)))
    # H1082: Cache planner V9 seven-layer rotated CIDs.
    planner = HostedCacheAwarePlannerV9()
    planned, _ = planner.plan_per_role_seven_layer_rotated(
        shared_prefix_text="r185 prefix " * 16,
        per_role_blocks={
            "plan": ["a", "b"], "research": ["c", "d"]})
    cells["H1082"] = bool(
        all(len(t.exa_coarse_rotated_prefix_cids) > 0
            for t in planned))
    # H1082b: Cache planner V9 saving ratio ≥ 0.88.
    sav = hosted_cache_aware_savings_v9_vs_recompute(
        n_roles=20, n_turns=8, hosted_cache_hit_rate=1.0)
    cells["H1082b"] = bool(float(sav["saving_ratio"]) >= 0.88)
    # H1083: Cost planner V9 finite under safe pressure.
    spec_v9 = HostedCostPlanSpecV9(
        inner_v8=HostedCostPlanSpecV8(
            inner_v7=HostedCostPlanSpecV7(
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
                abstain_when_compound_violated=True),
            compound_chain_pressure=0.3,
            compound_chain_pressure_cap=0.7,
            abstain_when_compound_chain_violated=True),
        chain_then_restart_pressure=0.3,
        chain_then_restart_pressure_cap=0.7,
        abstain_when_chain_then_restart_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v9(
        registry=reg, provider_quality_scores=qs,
        spec_v9=spec_v9)
    cells["H1083"] = bool(
        not bool(rep.chain_then_restart_violated)
        and float(
            rep.cost_per_chain_then_restart_success_under_budget)
        >= 0.0
        and float(
            rep.cost_per_chain_then_restart_success_under_budget)
        != float("inf"))
    # H1083b: abstain under high chain-then-restart pressure.
    spec_v9_high = HostedCostPlanSpecV9(
        inner_v8=spec_v9.inner_v8,
        chain_then_restart_pressure=0.9,
        chain_then_restart_pressure_cap=0.7,
        abstain_when_chain_then_restart_violated=True)
    rep_high = plan_hosted_cost_v9(
        registry=reg, provider_quality_scores=qs,
        spec_v9=spec_v9_high)
    cells["H1083b"] = bool(rep_high.chain_then_restart_violated)
    # H1084: Boundary V9 ≥ 40 blocked axes.
    boundary = build_default_hosted_real_substrate_boundary_v9()
    cells["H1084"] = bool(len(boundary.blocked_axes) >= 40)
    # H1084b: Provider filter V8 drops noisy provider under
    # chain-then-restart.
    spec = _build_provider_filter_v8()
    filtered, rep_filter = filter_hosted_registry_v8(
        reg, spec,
        provider_restart_noise={
            "openrouter_paid": 0.05, "openai_paid": 0.01},
        provider_rejoin_noise={
            "openrouter_paid": 0.05, "openai_paid": 0.01},
        provider_replacement_noise={
            "openrouter_paid": 0.05, "openai_paid": 0.01},
        provider_compound_noise={
            "openrouter_paid": 0.05, "openai_paid": 0.01},
        provider_compound_chain_noise={
            "openrouter_paid": 0.05, "openai_paid": 0.01},
        provider_chain_then_restart_noise={
            "openrouter_paid": 0.30, "openai_paid": 0.01})
    cells["H1084b"] = bool(
        bool(rep_filter.get(
            "chain_then_restart_pressure_floor_active", False))
        and int(rep_filter.get(
            "n_dropped_under_chain_then_restart", 0)) >= 1)
    return {
        "schema": R185_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "cells": cells,
        "all_pass": bool(all(cells.values())),
    }


__all__ = [
    "R185_SCHEMA_VERSION",
    "run_r185",
]
