"""W75 R-181 benchmark family — Hosted Control Plane V8 (Plane A).

Exercises the W75 hosted control plane V8 modules: router V8
(compound-chain-pressure + compound-repair-after-RTR match),
logprob router V8 (compound-chain-aware abstain floor + per-
budget+restart+rejoin+replacement+compound+chain tiebreak), cache-
aware planner V8 (six-layer rotated), cost planner V8 (cost-per-
compound-chain-success-under-budget + abstain-when-compound-chain-
pressure-violated), boundary V8 (≥ 37 blocked axes), provider
filter V7 (compound-chain-aware drop).

H970..H978 cell families (10 H-bars):

* H970   Router V8 determinism on (registry CID, request V8 CID,
         budget, restart_pressure, rejoin_pressure,
         replacement_pressure, compound_pressure,
         compound_chain_pressure, repair_dominance_label)
* H970b  Router V8 compound-chain-pressure score > 0 under high
         pressure
* H971   Logprob V8 abstain floor more aggressive under chain
* H971b  Logprob V8 per-budget+...+chain tiebreak narrows top-k
* H972   Cache planner V8 six-layer rotated CIDs content-addressed
* H972b  Cache planner V8 ≥ 0.87 saving on 18×8 at hit_rate=1.0
* H973   Cost planner V8 cost-per-compound-chain-success-under-
         budget finite
* H973b  Cost planner V8 abstain-when-compound-chain-violated
         fires
* H974   Boundary V8 enumerates ≥ 37 blocked axes (V20)
* H974b  Provider filter V7 drops noisy provider under high
         compound-chain
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_cache_aware_planner_v8 import (
    HostedCacheAwarePlannerV8,
    hosted_cache_aware_savings_v8_vs_recompute,
)
from coordpy.hosted_cost_planner import HostedCostPlanSpec
from coordpy.hosted_cost_planner_v2 import HostedCostPlanSpecV2
from coordpy.hosted_cost_planner_v3 import HostedCostPlanSpecV3
from coordpy.hosted_cost_planner_v4 import HostedCostPlanSpecV4
from coordpy.hosted_cost_planner_v5 import HostedCostPlanSpecV5
from coordpy.hosted_cost_planner_v6 import HostedCostPlanSpecV6
from coordpy.hosted_cost_planner_v7 import HostedCostPlanSpecV7
from coordpy.hosted_cost_planner_v8 import (
    HostedCostPlanSpecV8, plan_hosted_cost_v8,
)
from coordpy.hosted_logprob_router import TopKLogprobsPayload
from coordpy.hosted_logprob_router_v8 import HostedLogprobRouterV8
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
    HostedProviderFilterSpecV7, filter_hosted_registry_v7,
)
from coordpy.hosted_real_substrate_boundary_v8 import (
    build_default_hosted_real_substrate_boundary_v8,
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
    HostedRouterControllerV8, HostedRoutingRequestV8,
)


R181_SCHEMA_VERSION: str = "coordpy.r181_benchmark.v1"


def _build_req_v8(
        seed: int, restart_pressure: float = 0.7,
        rejoin_pressure: float = 0.7,
        replacement_pressure: float = 0.7,
        compound_pressure: float = 0.7,
        compound_chain_pressure: float = 0.7,
) -> HostedRoutingRequestV8:
    return HostedRoutingRequestV8(
        inner_v7=HostedRoutingRequestV7(
            inner_v6=HostedRoutingRequestV6(
                inner_v5=HostedRoutingRequestV5(
                    inner_v4=HostedRoutingRequestV4(
                        inner_v3=HostedRoutingRequestV3(
                            inner_v2=HostedRoutingRequestV2(
                                inner_v1=HostedRoutingRequest(
                                    request_cid=f"r181-{seed}",
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
            weight_compound_repair_drtr_match=0.4),
        compound_chain_pressure=float(compound_chain_pressure),
        weight_compound_chain_pressure=0.6,
        weight_compound_chain_repair_rtr_match=0.4)


def _build_provider_filter_v7() -> HostedProviderFilterSpecV7:
    return HostedProviderFilterSpecV7(
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
        compound_chain_pressure=0.8,
        compound_chain_pressure_floor=0.5,
        max_compound_chain_noise_per_provider={
            "openrouter_paid": 0.15,
            "openai_paid": 1.0},
        compound_chain_tier_weights={
            "logprobs_and_prefix_cache": 1.0})


def run_r181(*, seeds: Sequence[int]) -> dict[str, Any]:
    reg = default_hosted_registry()
    cells: dict[str, Any] = {}
    # H970: Router V8 determinism.
    ctrl = HostedRouterControllerV8.init(
        reg, {"openrouter_paid": 0.86, "openai_paid": 0.93})
    seen_routing_cids: dict[int, str] = {}
    chain_scores: list[float] = []
    for s in seeds:
        req = _build_req_v8(int(s))
        d = ctrl.decide_v8(req)
        seen_routing_cids[int(s)] = (
            d
            .per_budget_restart_rejoin_replacement_compound_chain_routing_cid)
        chain_scores.append(
            float(d.compound_chain_pressure_score))
    # Re-run to check determinism.
    ctrl2 = HostedRouterControllerV8.init(
        reg, {"openrouter_paid": 0.86, "openai_paid": 0.93})
    deterministic = True
    for s in seeds:
        req = _build_req_v8(int(s))
        d = ctrl2.decide_v8(req)
        if (str(d
                .per_budget_restart_rejoin_replacement_compound_chain_routing_cid)
                != str(seen_routing_cids[int(s)])):
            deterministic = False
            break
    cells["H970"] = bool(deterministic)
    cells["H970b"] = bool(
        all(float(x) > 0.0 for x in chain_scores))
    # H971: Logprob V8 abstain floor under chain pressure.
    payloads = [
        TopKLogprobsPayload(
            provider="openai_paid",
            token_to_logprob=(
                ("a", -0.1), ("b", -2.0), ("c", -3.0))),
    ]
    router = HostedLogprobRouterV8()
    res_low = router.fuse_v8(
        payloads, compound_chain_pressure=0.0)
    res_high = router.fuse_v8(
        payloads, compound_chain_pressure=0.9)
    cells["H971"] = bool(
        float(res_high.get(
            "effective_abstain_entropy_floor", 1e9))
        < float(res_low.get(
            "effective_abstain_entropy_floor", 0.0)))
    cells["H971b"] = bool(
        int(res_high.get("effective_top_k_v8", 1000))
        <= int(res_low.get("effective_top_k_v8", 0)))
    # H972: Cache planner V8 six-layer rotated CIDs.
    planner = HostedCacheAwarePlannerV8()
    planned, _ = planner.plan_per_role_six_layer_rotated(
        shared_prefix_text="r181 prefix " * 16,
        per_role_blocks={
            "plan": ["a", "b"], "research": ["c", "d"]})
    cells["H972"] = bool(
        all(len(t.peta_coarse_rotated_prefix_cids) > 0
            for t in planned))
    # H972b: Cache planner V8 saving ratio ≥ 0.87.
    sav = hosted_cache_aware_savings_v8_vs_recompute(
        n_roles=18, n_turns=8, hosted_cache_hit_rate=1.0)
    cells["H972b"] = bool(float(sav["saving_ratio"]) >= 0.87)
    # H973: Cost planner V8 finite under safe pressure.
    spec_v8 = HostedCostPlanSpecV8(
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
        abstain_when_compound_chain_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v8(
        registry=reg, provider_quality_scores=qs,
        spec_v8=spec_v8)
    cells["H973"] = bool(
        not bool(rep.compound_chain_violated)
        and float(rep.cost_per_compound_chain_success_under_budget)
        >= 0.0
        and float(rep.cost_per_compound_chain_success_under_budget)
        != float("inf"))
    # H973b: abstain under high chain pressure.
    spec_v8_high = HostedCostPlanSpecV8(
        inner_v7=spec_v8.inner_v7,
        compound_chain_pressure=0.9,
        compound_chain_pressure_cap=0.7,
        abstain_when_compound_chain_violated=True)
    rep_high = plan_hosted_cost_v8(
        registry=reg, provider_quality_scores=qs,
        spec_v8=spec_v8_high)
    cells["H973b"] = bool(rep_high.compound_chain_violated)
    # H974: Boundary V8 ≥ 37 blocked axes.
    boundary = build_default_hosted_real_substrate_boundary_v8()
    cells["H974"] = bool(len(boundary.blocked_axes) >= 37)
    # H974b: Provider filter V7 drops noisy provider under chain.
    # Tune below all earlier per-layer caps so openrouter survives
    # V2..V6 and is dropped exactly by V7's compound-chain layer.
    spec = _build_provider_filter_v7()
    filtered, rep_filter = filter_hosted_registry_v7(
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
            "openrouter_paid": 0.30, "openai_paid": 0.01})
    cells["H974b"] = bool(
        bool(rep_filter.get(
            "compound_chain_pressure_floor_active", False))
        and int(rep_filter.get(
            "n_dropped_under_compound_chain", 0)) >= 1)
    return {
        "schema": R181_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "cells": cells,
        "all_pass": bool(all(cells.values())),
    }


__all__ = [
    "R181_SCHEMA_VERSION",
    "run_r181",
]
