"""W73 R-173 benchmark family — Hosted Control Plane V6 (Plane A).

Exercises the W73 hosted control plane V6 modules: router V6
(replacement-pressure + replacement-after-CTR match), logprob
router V6 (replacement-aware abstain floor + per-budget+restart+
rejoin+replacement tiebreak), cache-aware planner V6 (four-layer
rotated), cost planner V6 (cost-per-replacement-rejoin-success-
under-budget + abstain-when-replacement-pressure-violated),
boundary V6 (≥ 31 blocked axes), provider filter V5
(replacement-aware drop).

H800..H808 cell families (10 H-bars):

* H800   Router V6 determinism on (registry CID, request V6 CID,
         budget, restart_pressure, rejoin_pressure,
         replacement_pressure, repair_dominance_label)
* H800b  Router V6 replacement-pressure score > 0 under high
         pressure
* H801   Logprob V6 abstain floor more aggressive under
         replacement
* H801b  Logprob V6 per-budget+restart+rejoin+replacement tiebreak
         narrows top-k further
* H802   Cache planner V6 four-layer rotated CIDs content-
         addressed
* H802b  Cache planner V6 ≥ 0.85 saving on 14×8 at hit_rate=1.0
* H803   Cost planner V6 cost-per-replacement-rejoin-success-
         under-budget finite
* H803b  Cost planner V6 abstain-when-replacement-violated fires
* H804   Boundary V6 enumerates ≥ 31 blocked axes (V18)
* H804b  Provider filter V5 drops noisy provider under high
         replacement
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_cache_aware_planner_v6 import (
    HostedCacheAwarePlannerV6,
    hosted_cache_aware_savings_v6_vs_recompute,
)
from coordpy.hosted_cost_planner import HostedCostPlanSpec
from coordpy.hosted_cost_planner_v2 import HostedCostPlanSpecV2
from coordpy.hosted_cost_planner_v3 import HostedCostPlanSpecV3
from coordpy.hosted_cost_planner_v4 import HostedCostPlanSpecV4
from coordpy.hosted_cost_planner_v5 import HostedCostPlanSpecV5
from coordpy.hosted_cost_planner_v6 import (
    HostedCostPlanSpecV6, plan_hosted_cost_v6,
)
from coordpy.hosted_logprob_router import TopKLogprobsPayload
from coordpy.hosted_logprob_router_v6 import HostedLogprobRouterV6
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
    HostedProviderFilterSpecV5, filter_hosted_registry_v5,
)
from coordpy.hosted_real_substrate_boundary_v6 import (
    build_default_hosted_real_substrate_boundary_v6,
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
    HostedRouterControllerV6, HostedRoutingRequestV6,
)


R173_SCHEMA_VERSION: str = "coordpy.r173_benchmark.v1"


def _build_req_v6(
        seed: int, restart_pressure: float = 0.7,
        rejoin_pressure: float = 0.7,
        replacement_pressure: float = 0.7,
        visible_token_budget: int = 128,
) -> HostedRoutingRequestV6:
    return HostedRoutingRequestV6(
        inner_v5=HostedRoutingRequestV5(
            inner_v4=HostedRoutingRequestV4(
                inner_v3=HostedRoutingRequestV3(
                    inner_v2=HostedRoutingRequestV2(
                        inner_v1=HostedRoutingRequest(
                            request_cid=f"r800_{int(seed)}",
                            input_tokens=1000,
                            expected_output_tokens=300,
                            require_logprobs=True,
                            require_prefix_cache=True,
                            data_policy_required="no_log",
                            max_latency_ms=2000.0,
                            max_cost_usd=50.0),
                        weight_cost=1.0, weight_latency=0.5,
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
        weight_replacement_after_ctr_match=0.4)


def family_h800_router_v6_determinism(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc1 = HostedRouterControllerV6.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    rc2 = HostedRouterControllerV6.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    req = _build_req_v6(int(seed))
    d1 = rc1.decide_v6(req)
    d2 = rc2.decide_v6(req)
    return {
        "schema": R173_SCHEMA_VERSION,
        "name": "h800_router_v6_determinism",
        "passed": bool(
            d1.chosen_provider == d2.chosen_provider
            and d1.cid() == d2.cid()),
    }


def family_h800b_router_v6_replacement_pressure_score(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc = HostedRouterControllerV6.init(
        reg, {"openrouter_paid": 0.85,
              "openai_paid": 0.92})
    req = _build_req_v6(int(seed), replacement_pressure=0.8)
    d = rc.decide_v6(req)
    return {
        "schema": R173_SCHEMA_VERSION,
        "name": "h800b_router_v6_replacement_pressure_score",
        "passed": bool(d.replacement_pressure_score > 0.0),
        "score": float(d.replacement_pressure_score),
    }


def family_h801_logprob_v6_abstain_floor_aggressive(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV6(
        replacement_pressure_floor=0.5,
        replacement_abstain_floor_delta=1.8)
    payloads = [
        TopKLogprobsPayload("p1", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.1)
            for i in range(8))),
        TopKLogprobsPayload("p2", tuple(
            (chr(ord("a") + i), -1.0 - i * 0.05)
            for i in range(8))),
    ]
    no = lr.fuse_v6(
        payloads, visible_token_budget=256,
        baseline_token_cost=512,
        restart_pressure=0.0, rejoin_pressure=0.0,
        replacement_pressure=0.0)
    high = lr.fuse_v6(
        payloads, visible_token_budget=256,
        baseline_token_cost=512,
        restart_pressure=0.0, rejoin_pressure=0.0,
        replacement_pressure=0.9)
    return {
        "schema": R173_SCHEMA_VERSION,
        "name": "h801_logprob_v6_abstain_floor_aggressive",
        "passed": bool(
            high["effective_abstain_entropy_floor"]
            < no["effective_abstain_entropy_floor"]),
        "no": float(no["effective_abstain_entropy_floor"]),
        "high": float(high["effective_abstain_entropy_floor"]),
    }


def family_h801b_logprob_v6_combined_tiebreak(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV6(
        replacement_pressure_floor=0.5,
        replacement_abstain_floor_delta=0.0)
    payloads = [
        TopKLogprobsPayload(
            "p1", (("a", -0.1), ("b", -0.5), ("c", -1.0))),
        TopKLogprobsPayload(
            "p2", (("a", -0.2), ("b", -0.4), ("c", -1.1))),
    ]
    no_rep = lr.fuse_v6(
        payloads, visible_token_budget=512,
        baseline_token_cost=512, top_k=10,
        restart_pressure=0.0, rejoin_pressure=0.0,
        replacement_pressure=0.0)
    with_rep = lr.fuse_v6(
        payloads, visible_token_budget=512,
        baseline_token_cost=512, top_k=10,
        restart_pressure=0.0, rejoin_pressure=0.0,
        replacement_pressure=0.9)
    return {
        "schema": R173_SCHEMA_VERSION,
        "name": "h801b_logprob_v6_combined_tiebreak",
        "passed": bool(
            int(with_rep.get("effective_top_k_v6", 0))
            < int(no_rep.get("effective_top_k_v6", 0))),
    }


def family_h802_cache_v6_four_layer_cid_content_addressed(
        seed: int) -> dict[str, Any]:
    cp1 = HostedCacheAwarePlannerV6()
    cp2 = HostedCacheAwarePlannerV6()
    planned1, _ = (
        cp1.plan_per_role_four_layer_rotated(
            shared_prefix_text="W73 shared prefix " * 14,
            per_role_blocks={
                "plan": ["a", "b"], "write": ["c", "d"]}))
    planned2, _ = (
        cp2.plan_per_role_four_layer_rotated(
            shared_prefix_text="W73 shared prefix " * 14,
            per_role_blocks={
                "plan": ["a", "b"], "write": ["c", "d"]}))
    ok = (
        planned1[0].mega_coarse_rotated_prefix_cids
        == planned2[0].mega_coarse_rotated_prefix_cids)
    return {
        "schema": R173_SCHEMA_VERSION,
        "name":
            "h802_cache_v6_four_layer_cid_content_addressed",
        "passed": bool(ok),
    }


def family_h802b_cache_v6_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_cache_aware_savings_v6_vs_recompute(
        n_roles=14, n_turns=8,
        hosted_cache_hit_rate=1.0)
    return {
        "schema": R173_SCHEMA_VERSION,
        "name": "h802b_cache_v6_savings",
        "passed": bool(sav["saving_ratio"] >= 0.85),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h803_cost_v6_cps_under_budget(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v6 = HostedCostPlanSpecV6(
        inner_v5=HostedCostPlanSpecV5(
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
            abstain_when_rejoin_violated=True),
        replacement_pressure=0.3,
        replacement_pressure_cap=0.7,
        abstain_when_replacement_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v6(
        registry=reg, provider_quality_scores=qs,
        spec_v6=spec_v6)
    return {
        "schema": R173_SCHEMA_VERSION,
        "name": "h803_cost_v6_cps_under_budget",
        "passed": bool(
            rep.cost_per_replacement_rejoin_success_under_budget
            >= 0.0
            and rep.cost_per_replacement_rejoin_success_under_budget
                < float("inf")
            and not rep.replacement_violated),
    }


def family_h803b_cost_v6_abstain_replacement_violated(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v6 = HostedCostPlanSpecV6(
        inner_v5=HostedCostPlanSpecV5(
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
            abstain_when_rejoin_violated=True),
        replacement_pressure=0.9,
        replacement_pressure_cap=0.7,
        abstain_when_replacement_violated=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v6(
        registry=reg, provider_quality_scores=qs,
        spec_v6=spec_v6)
    return {
        "schema": R173_SCHEMA_VERSION,
        "name": "h803b_cost_v6_abstain_replacement_violated",
        "passed": bool(
            rep.replacement_violated
            and "abstain_v6" in str(rep.rationale_v6)),
    }


def family_h804_boundary_v6_blocked_axes(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary_v6()
    return {
        "schema": R173_SCHEMA_VERSION,
        "name": "h804_boundary_v6_blocked_axes",
        "passed": bool(len(b.blocked_axes) >= 31),
        "n_blocked": int(len(b.blocked_axes)),
    }


def family_h804b_provider_filter_v5_replacement_drop(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v5 = HostedProviderFilterSpecV5(
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
        replacement_pressure=0.9,
        replacement_pressure_floor=0.5,
        max_replacement_noise_per_provider={
            "openrouter_paid": 0.20,
            "openai_paid": 1.0})
    pn_replacement = {
        "openrouter_paid": 0.4,  # > 0.20 cap → drop
        "openai_paid": 0.1,
    }
    filt, rep = filter_hosted_registry_v5(
        reg, spec_v5,
        provider_restart_noise={},
        provider_rejoin_noise={},
        provider_replacement_noise=pn_replacement)
    return {
        "schema": R173_SCHEMA_VERSION,
        "name": "h804b_provider_filter_v5_replacement_drop",
        "passed": bool(
            int(rep.get("n_dropped_under_replacement", 0)) >= 1
            and "openrouter_paid"
            in list(rep.get("dropped_under_replacement", []))),
        "n_dropped": int(rep.get(
            "n_dropped_under_replacement", 0)),
    }


_R173_FAMILIES: tuple[Any, ...] = (
    family_h800_router_v6_determinism,
    family_h800b_router_v6_replacement_pressure_score,
    family_h801_logprob_v6_abstain_floor_aggressive,
    family_h801b_logprob_v6_combined_tiebreak,
    family_h802_cache_v6_four_layer_cid_content_addressed,
    family_h802b_cache_v6_savings,
    family_h803_cost_v6_cps_under_budget,
    family_h803b_cost_v6_abstain_replacement_violated,
    family_h804_boundary_v6_blocked_axes,
    family_h804b_provider_filter_v5_replacement_drop,
)


def run_r173(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R173_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R173_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R173_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R173_SCHEMA_VERSION", "run_r173"]
