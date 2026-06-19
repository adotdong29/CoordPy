"""W78 R-193 benchmark family — Hosted Control Plane V11 (Plane A).

Exercises the W78 hosted control plane V11 modules: router V11
(long-horizon-reconstruction-pressure + LHR match), logprob router
V11 (long-horizon-reconstruction-aware abstain floor + 9-way
tiebreak), cache-aware planner V11 (nine-layer rotated), cost
planner V11 (cost-per-long-horizon-reconstruction-success-under-
budget + abstain-when-lhr-violated), boundary V11 (≥ 46 blocked
axes), provider filter V10 (long-horizon-reconstruction-aware
drop).

H1190..H1199 cell families (10 H-bars).
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_cache_aware_planner_v11 import (
    HostedCacheAwarePlannerV11,
    hosted_cache_aware_savings_v11_vs_recompute,
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
from coordpy.hosted_cost_planner_v10 import HostedCostPlanSpecV10
from coordpy.hosted_cost_planner_v11 import (
    HostedCostPlanSpecV11, plan_hosted_cost_v11,
)
from coordpy.hosted_logprob_router import TopKLogprobsPayload
from coordpy.hosted_logprob_router_v11 import (
    HostedLogprobRouterV11,
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
    HostedProviderFilterSpecV9,
)
from coordpy.hosted_provider_filter_v10 import (
    HostedProviderFilterSpecV10, filter_hosted_registry_v10,
)
from coordpy.hosted_real_substrate_boundary_v11 import (
    build_default_hosted_real_substrate_boundary_v11,
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
    HostedRoutingRequestV10,
)
from coordpy.hosted_router_controller_v11 import (
    HostedRouterControllerV11, HostedRoutingRequestV11,
)


R193_SCHEMA_VERSION: str = "coordpy.r193_benchmark.v1"


def _build_req_v11(
        *, rc: str = "req",
        long_horizon_reconstruction_pressure: float = 0.95,
) -> HostedRoutingRequestV11:
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
    req_v10 = HostedRoutingRequestV10(
        inner_v9=req_v9,
        post_restart_replacement_pressure=0.9,
        weight_post_restart_replacement_pressure=0.6,
        weight_post_restart_replacement_after_pcr_match=0.4)
    return HostedRoutingRequestV11(
        inner_v10=req_v10,
        long_horizon_reconstruction_pressure=float(
            long_horizon_reconstruction_pressure),
        weight_long_horizon_reconstruction_pressure=0.6,
        weight_long_horizon_reconstruction_after_pcr_match=0.4)


def _build_spec_v11(
        *, lhr_pressure: float = 0.3,
        lhr_cap: float = 0.7,
) -> HostedCostPlanSpecV11:
    return HostedCostPlanSpecV11(
        inner_v10=HostedCostPlanSpecV10(
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
            abstain_when_post_restart_replacement_violated=True),
        long_horizon_reconstruction_pressure=float(lhr_pressure),
        long_horizon_reconstruction_pressure_cap=float(lhr_cap),
        abstain_when_long_horizon_reconstruction_violated=True)


def _build_spec_pf_v10() -> HostedProviderFilterSpecV10:
    """Reuse the W77 default V9 filter spec from the W77 team."""
    from coordpy.w77_team import W77Params
    w77_p = W77Params.build_default(seed=78900)
    return HostedProviderFilterSpecV10(
        inner_v9=w77_p.hosted_provider_filter_v9,
        long_horizon_reconstruction_pressure=0.95,
        long_horizon_reconstruction_pressure_floor=0.5,
        max_long_horizon_reconstruction_noise_per_provider={
            "openrouter_paid": 0.08,
            "openai_paid": 1.0},
        long_horizon_reconstruction_tier_weights={})


def run_r193(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    reg = default_hosted_registry()
    router_v11 = HostedRouterControllerV11.init(
        reg, {"openrouter_paid": 0.88, "openai_paid": 0.95})
    # H1190: determinism.
    req_a = _build_req_v11(rc="req-a")
    req_b = _build_req_v11(rc="req-a")
    dec_a = router_v11.decide_v11(req_a)
    dec_b = router_v11.decide_v11(req_b)
    cells["H1190"] = bool(
        dec_a.per_routing_cid_v11 == dec_b.per_routing_cid_v11)
    # H1190b: LHR pressure score > 0.
    cells["H1190b"] = bool(
        float(dec_a.long_horizon_reconstruction_pressure_score)
        > 0.0)
    # H1191: logprob V11 abstain floor more aggressive under LHR.
    logprob_v11 = HostedLogprobRouterV11()
    payloads = [
        TopKLogprobsPayload(
            provider="openai_paid",
            token_to_logprob=(
                ("a", -0.5), ("b", -1.0), ("c", -2.0))),
    ]
    res_low = logprob_v11.fuse_v11(
        payloads,
        long_horizon_reconstruction_pressure=0.0,
        visible_token_budget=256, baseline_token_cost=512)
    res_high = logprob_v11.fuse_v11(
        payloads,
        long_horizon_reconstruction_pressure=0.95,
        visible_token_budget=256, baseline_token_cost=512)
    cells["H1191"] = bool(
        float(res_high["effective_abstain_entropy_floor_v11"])
        < float(res_low["effective_abstain_entropy_floor_v11"]))
    cells["H1191b"] = bool(
        int(res_high["effective_top_k_v11"])
        <= int(res_low["effective_top_k_v11"]))
    # H1192: cache planner V11 nine-layer CIDs content-addressed.
    planner_v11 = HostedCacheAwarePlannerV11()
    planned, _ = planner_v11.plan_per_role_nine_layer_rotated(
        shared_prefix_text=("W78 shared prefix " * 16),
        per_role_blocks={
            "plan": ["t0"], "research": ["r0"]})
    cells["H1192"] = bool(
        all(
            isinstance(
                p.yotta_coarse_rotated_prefix_cids, tuple)
            for p in planned)
        and all(
            len(p.yotta_coarse_rotated_prefix_cids) >= 1
            for p in planned))
    # H1192b: ≥ 0.89 saving on 20×8 at hit_rate=1.0.
    sav = hosted_cache_aware_savings_v11_vs_recompute(
        n_roles=20, n_turns=8, hosted_cache_hit_rate=1.0)
    cells["H1192b"] = bool(float(sav["saving_ratio"]) >= 0.89)
    # H1193: cost planner V11 finite cost.
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v11(
        registry=reg, provider_quality_scores=qs,
        spec_v11=_build_spec_v11(lhr_pressure=0.3, lhr_cap=0.7))
    cells["H1193"] = bool(
        not bool(rep.long_horizon_reconstruction_violated)
        and float(
            rep.cost_per_long_horizon_reconstruction_success_under_budget)
        != float("inf"))
    # H1193b: abstain-when-LHR-violated.
    rep_v = plan_hosted_cost_v11(
        registry=reg, provider_quality_scores=qs,
        spec_v11=_build_spec_v11(lhr_pressure=0.99, lhr_cap=0.5))
    cells["H1193b"] = bool(
        str(rep_v.rationale_v11).startswith("abstain_v11_"))
    # H1194: boundary V11 ≥ 46 blocked axes.
    boundary = build_default_hosted_real_substrate_boundary_v11()
    cells["H1194"] = bool(len(boundary.blocked_axes) >= 46)
    # H1194b: provider filter V10 drops under high LHR.
    spec_pf = _build_spec_pf_v10()
    _, pf_rep = filter_hosted_registry_v10(
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
            "openrouter_paid": 0.05, "openai_paid": 0.01},
        provider_long_horizon_reconstruction_noise={
            "openrouter_paid": 0.50, "openai_paid": 0.01})
    cells["H1194b"] = bool(
        int(pf_rep.get(
            "n_dropped_under_long_horizon_reconstruction", 0))
        >= 1)
    return {
        "schema": R193_SCHEMA_VERSION,
        "seeds": list(seeds),
        "cells": dict(cells),
        "all_pass": bool(all(bool(v) for v in cells.values())),
    }


__all__ = ["R193_SCHEMA_VERSION", "run_r193"]
