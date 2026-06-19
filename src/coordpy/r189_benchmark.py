"""W77 R-189 benchmark family — Hosted Control Plane V10 (Plane A).

Exercises the W77 hosted control plane V10 modules: router V10
(post-restart-replacement-pressure + PCR match), logprob router
V10 (post-restart-replacement-aware abstain floor + 8-way
tiebreak), cache-aware planner V10 (eight-layer rotated), cost
planner V10 (cost-per-post-restart-replacement-success-under-
budget + abstain-when-pcr-violated), boundary V10 (≥ 43 blocked
axes), provider filter V9 (post-restart-replacement-aware drop).

H1170..H1179 cell families (10 H-bars):

* H1170   Router V10 determinism on (registry CID, request V10 CID,
          budget, all pressures, repair_dominance_label)
* H1170b  Router V10 post-restart-replacement-pressure score > 0
* H1171   Logprob V10 abstain floor more aggressive under PCR
* H1171b  Logprob V10 per-...+PCR tiebreak narrows top-k
* H1172   Cache planner V10 eight-layer rotated CIDs content-
          addressed
* H1172b  Cache planner V10 ≥ 0.89 saving on 20×8 at hit_rate=1.0
* H1173   Cost planner V10 cost-per-post-restart-replacement-
          success-under-budget finite
* H1173b  Cost planner V10 abstain-when-pcr-violated fires
* H1174   Boundary V10 enumerates ≥ 43 blocked axes (V22)
* H1174b  Provider filter V9 drops noisy provider under high PCR
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_cache_aware_planner_v10 import (
    HostedCacheAwarePlannerV10,
    hosted_cache_aware_savings_v10_vs_recompute,
)
from coordpy.hosted_cost_planner import HostedCostPlanSpec
from coordpy.hosted_cost_planner_v2 import HostedCostPlanSpecV2
from coordpy.hosted_cost_planner_v3 import HostedCostPlanSpecV3
from coordpy.hosted_cost_planner_v4 import HostedCostPlanSpecV4
from coordpy.hosted_cost_planner_v5 import HostedCostPlanSpecV5
from coordpy.hosted_cost_planner_v6 import HostedCostPlanSpecV6
from coordpy.hosted_cost_planner_v7 import HostedCostPlanSpecV7
from coordpy.hosted_cost_planner_v8 import HostedCostPlanSpecV8
from coordpy.hosted_cost_planner_v9 import HostedCostPlanSpecV9
from coordpy.hosted_cost_planner_v10 import (
    HostedCostPlanSpecV10, plan_hosted_cost_v10,
)
from coordpy.hosted_logprob_router import TopKLogprobsPayload
from coordpy.hosted_logprob_router_v10 import (
    HostedLogprobRouterV10,
)
from coordpy.hosted_provider_filter import HostedProviderFilterSpec
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
    HostedProviderFilterSpecV8,
)
from coordpy.hosted_provider_filter_v9 import (
    HostedProviderFilterSpecV9, filter_hosted_registry_v9,
)
from coordpy.hosted_real_substrate_boundary_v10 import (
    build_default_hosted_real_substrate_boundary_v10,
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
    HostedRoutingRequestV9,
)
from coordpy.hosted_router_controller_v10 import (
    HostedRouterControllerV10, HostedRoutingRequestV10,
)


R189_SCHEMA_VERSION: str = "coordpy.r189_benchmark.v1"


def _build_req_v10(
        *, rc: str = "req",
        post_restart_replacement_pressure: float = 0.9,
) -> HostedRoutingRequestV10:
    req_v9 = HostedRoutingRequestV9(
        inner_v8=HostedRoutingRequestV8(
            inner_v7=HostedRoutingRequestV7(
                inner_v6=HostedRoutingRequestV6(
                    inner_v5=HostedRoutingRequestV5(
                        inner_v4=HostedRoutingRequestV4(
                            inner_v3=HostedRoutingRequestV3(
                                inner_v2=HostedRoutingRequestV2(
                                    inner_v1=HostedRoutingRequest(
                                        request_cid=str(rc),
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
                            restart_pressure=0.7,
                            weight_restart_pressure=0.6,
                            weight_delayed_repair_match=0.4),
                        rejoin_pressure=0.7,
                        weight_rejoin_pressure=0.6,
                        weight_delayed_rejoin_match=0.4),
                    replacement_pressure=0.7,
                    weight_replacement_pressure=0.6,
                    weight_replacement_after_ctr_match=0.4),
                compound_pressure=0.7,
                weight_compound_pressure=0.6,
                weight_compound_repair_drtr_match=0.4),
            compound_chain_pressure=0.8,
            weight_compound_chain_pressure=0.6,
            weight_compound_chain_repair_rtr_match=0.4),
        compound_chain_then_restart_pressure=0.85,
        weight_compound_chain_then_restart_pressure=0.6,
        weight_chain_then_restart_after_rtr_match=0.4)
    return HostedRoutingRequestV10(
        inner_v9=req_v9,
        post_restart_replacement_pressure=float(
            post_restart_replacement_pressure),
        weight_post_restart_replacement_pressure=0.6,
        weight_post_restart_replacement_after_pcr_match=0.4)


def run_r189(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    reg = default_hosted_registry()
    router_v10 = HostedRouterControllerV10.init(
        reg, {"openrouter_paid": 0.87, "openai_paid": 0.94})
    # H1170: determinism — same request → same per-routing CID.
    req_a = _build_req_v10(rc="req-a")
    req_b = _build_req_v10(rc="req-a")
    dec_a = router_v10.decide_v10(req_a)
    dec_b = router_v10.decide_v10(req_b)
    cells["H1170"] = bool(
        dec_a.per_routing_cid_v10 == dec_b.per_routing_cid_v10)
    # H1170b: PCR pressure score > 0.
    cells["H1170b"] = bool(
        float(dec_a.post_restart_replacement_pressure_score) > 0.0)
    # H1171: logprob V10 abstain floor more aggressive under PCR.
    logprob_v10 = HostedLogprobRouterV10()
    payloads = [
        TopKLogprobsPayload(
            provider="openai_paid",
            token_to_logprob=(
                ("a", -0.5), ("b", -1.0), ("c", -2.0))),
    ]
    res_low = logprob_v10.fuse_v10(
        payloads, post_restart_replacement_pressure=0.0,
        visible_token_budget=256, baseline_token_cost=512)
    res_high = logprob_v10.fuse_v10(
        payloads, post_restart_replacement_pressure=0.95,
        visible_token_budget=256, baseline_token_cost=512)
    cells["H1171"] = bool(
        float(res_high["effective_abstain_entropy_floor_v10"])
        < float(res_low.get(
            "effective_abstain_entropy_floor_v10",
            res_low.get(
                "effective_abstain_entropy_floor",
                float(res_low["effective_abstain_entropy_floor_v10"])))))
    # H1171b: top-k narrower under high PCR.
    cells["H1171b"] = bool(
        int(res_high["effective_top_k_v10"])
        <= int(res_low["effective_top_k_v10"]))
    # H1172: cache planner V10 eight-layer CIDs content-addressed.
    planner_v10 = HostedCacheAwarePlannerV10()
    planned, _ = planner_v10.plan_per_role_eight_layer_rotated(
        shared_prefix_text=("W77 shared prefix " * 16),
        per_role_blocks={
            "plan": ["t0"], "research": ["r0"]})
    cells["H1172"] = bool(
        all(
            isinstance(p.zetta_coarse_rotated_prefix_cids, tuple)
            for p in planned)
        and all(
            len(p.zetta_coarse_rotated_prefix_cids) >= 1
            for p in planned))
    # H1172b: ≥ 0.89 saving on 20×8 at hit_rate=1.0.
    sav = hosted_cache_aware_savings_v10_vs_recompute(
        n_roles=20, n_turns=8, hosted_cache_hit_rate=1.0)
    cells["H1172b"] = bool(float(sav["saving_ratio"]) >= 0.89)
    # H1173: cost planner V10 finite cost.
    spec_v10 = HostedCostPlanSpecV10(
        inner_v9=HostedCostPlanSpecV9(
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
            abstain_when_chain_then_restart_violated=True),
        post_restart_replacement_pressure=0.3,
        post_restart_replacement_pressure_cap=0.7,
        abstain_when_post_restart_replacement_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v10(
        registry=reg, provider_quality_scores=qs,
        spec_v10=spec_v10)
    cells["H1173"] = bool(
        not bool(rep.post_restart_replacement_violated)
        and float(
            rep.cost_per_post_restart_replacement_success_under_budget)
        != float("inf"))
    # H1173b: abstain-when-PCR-violated.
    spec_violated = HostedCostPlanSpecV10(
        inner_v9=spec_v10.inner_v9,
        post_restart_replacement_pressure=0.99,
        post_restart_replacement_pressure_cap=0.5,
        abstain_when_post_restart_replacement_violated=True)
    rep_v = plan_hosted_cost_v10(
        registry=reg,
        provider_quality_scores=qs,
        spec_v10=spec_violated)
    cells["H1173b"] = bool(
        str(rep_v.rationale_v10).startswith("abstain_v10_"))
    # H1174: boundary V10 ≥ 43 blocked axes.
    boundary = build_default_hosted_real_substrate_boundary_v10()
    cells["H1174"] = bool(len(boundary.blocked_axes) >= 43)
    # H1174b: provider filter V9 drops under high PCR.
    spec_pf = HostedProviderFilterSpecV9(
        inner_v8=HostedProviderFilterSpecV8(
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
                                    "logprobs_and_prefix_cache": 1.0,
                                    "logprobs": 0.7,
                                    "prefix_cache": 0.6,
                                    "text_only": 0.4}),
                            rejoin_pressure=0.7,
                            rejoin_pressure_floor=0.5,
                            max_rejoin_noise_per_provider={
                                "openrouter_paid": 0.25,
                                "openai_paid": 1.0},
                            rejoin_tier_weights={
                                "logprobs_and_prefix_cache": 1.0,
                                "logprobs": 0.65,
                                "prefix_cache": 0.55,
                                "text_only": 0.35}),
                        replacement_pressure=0.7,
                        replacement_pressure_floor=0.5,
                        max_replacement_noise_per_provider={
                            "openrouter_paid": 0.20,
                            "openai_paid": 1.0},
                        replacement_tier_weights={
                            "logprobs_and_prefix_cache": 1.0,
                            "logprobs": 0.60,
                            "prefix_cache": 0.50,
                            "text_only": 0.30}),
                    compound_pressure=0.7,
                    compound_pressure_floor=0.5,
                    max_compound_noise_per_provider={
                        "openrouter_paid": 0.18,
                        "openai_paid": 1.0},
                    compound_tier_weights={
                        "logprobs_and_prefix_cache": 1.0,
                        "logprobs": 0.55,
                        "prefix_cache": 0.45,
                        "text_only": 0.25}),
                compound_chain_pressure=0.7,
                compound_chain_pressure_floor=0.5,
                max_compound_chain_noise_per_provider={
                    "openrouter_paid": 0.15,
                    "openai_paid": 1.0},
                compound_chain_tier_weights={
                    "logprobs_and_prefix_cache": 1.0,
                    "logprobs": 0.50,
                    "prefix_cache": 0.40,
                    "text_only": 0.20}),
            chain_then_restart_pressure=0.8,
            chain_then_restart_pressure_floor=0.5,
            max_chain_then_restart_noise_per_provider={
                "openrouter_paid": 0.12,
                "openai_paid": 1.0},
            chain_then_restart_tier_weights={
                "logprobs_and_prefix_cache": 1.0,
                "logprobs": 0.45,
                "prefix_cache": 0.35,
                "text_only": 0.15}),
        post_restart_replacement_pressure=0.9,
        post_restart_replacement_pressure_floor=0.5,
        max_post_restart_replacement_noise_per_provider={
            "openrouter_paid": 0.10,
            "openai_paid": 1.0},
        post_restart_replacement_tier_weights={
            "logprobs_and_prefix_cache": 1.0,
            "logprobs": 0.40,
            "prefix_cache": 0.30,
            "text_only": 0.10})
    _, filter_rep = filter_hosted_registry_v9(
        reg, spec_pf,
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
            "openrouter_paid": 0.05, "openai_paid": 0.01},
        provider_post_restart_replacement_noise={
            "openrouter_paid": 0.50, "openai_paid": 0.01})
    cells["H1174b"] = bool(
        int(filter_rep.get(
            "n_dropped_under_post_restart_replacement", 0)) >= 1)
    return {
        "schema": R189_SCHEMA_VERSION,
        "n_seeds": int(len(seeds)),
        "cells": cells,
        "all_pass": bool(all(cells.values())),
    }


__all__ = ["R189_SCHEMA_VERSION", "run_r189"]
