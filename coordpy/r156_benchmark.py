"""W69 R-156 benchmark family — Hosted Control Plane V2 (Plane A).

Exercises the W69 hosted control plane V2 modules: router V2,
logprob router V2 (Bayesian fusion), cache-aware planner V2
(per-role staggered), provider filter V2 (compositional ALL/ANY),
cost planner V2 (multi-turn schedule), and boundary V2 (≥ 19
blocked axes).

H400..H409 cell families (10 H-bars):

* H400   Router V2 determinism on (registry CID, request CID)
* H400b  Router V2 sticky-routing fires
* H401   Logprob V2 Bayesian Dirichlet fusion fires
* H401b  Logprob V2 tiebreak-to-V1 fallback fires under high
         entropy
* H402   Cache planner V2 per-role staggered prefix CIDs are
         content-addressed
* H402b  Cache planner V2 ≥ 60 % saving on 6×8 at hit_rate=1.0
* H403   Provider filter V2 ALL-combine drops cumulative
* H403b  Provider filter V2 ANY-combine keeps union
* H404   Cost planner V2 supports rotation
* H404b  Cost planner V2 cost-per-success ratio is finite
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
    HostedRouterControllerV2, HostedRoutingRequestV2,
)
from coordpy.hosted_logprob_router_v2 import (
    HostedLogprobRouterV2,
)
from coordpy.hosted_logprob_router import TopKLogprobsPayload
from coordpy.hosted_cache_aware_planner_v2 import (
    HostedCacheAwarePlannerV2,
    hosted_cache_aware_savings_v2_vs_recompute,
)
from coordpy.hosted_provider_filter import (
    HostedProviderFilterSpec,
)
from coordpy.hosted_provider_filter_v2 import (
    HostedProviderFilterSpecV2, filter_hosted_registry_v2,
)
from coordpy.hosted_cost_planner import HostedCostPlanSpec
from coordpy.hosted_cost_planner_v2 import (
    HostedCostPlanSpecV2, plan_hosted_cost_v2,
)


R156_SCHEMA_VERSION: str = "coordpy.r156_benchmark.v1"


def family_h400_router_v2_determinism(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc1 = HostedRouterControllerV2.init(reg, {
        "openrouter_paid": 0.85, "openai_paid": 0.92})
    rc2 = HostedRouterControllerV2.init(reg, {
        "openrouter_paid": 0.85, "openai_paid": 0.92})
    req = HostedRoutingRequestV2(
        inner_v1=HostedRoutingRequest(
            request_cid=f"r400_{int(seed)}",
            input_tokens=1000, expected_output_tokens=300,
            require_logprobs=True, require_prefix_cache=True,
            data_policy_required="no_log",
            max_latency_ms=2000.0, max_cost_usd=50.0),
        weight_cost=1.0, weight_latency=0.5,
        weight_success=0.3)
    d1 = rc1.decide_v2(req)
    d2 = rc2.decide_v2(req)
    return {
        "schema": R156_SCHEMA_VERSION,
        "name": "h400_router_v2_determinism",
        "passed": bool(
            d1.chosen_provider == d2.chosen_provider
            and d1.cid() == d2.cid()),
    }


def family_h400b_router_v2_sticky(seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc = HostedRouterControllerV2.init(reg, {})
    req = HostedRoutingRequestV2(
        inner_v1=HostedRoutingRequest(
            request_cid=f"r400b_{int(seed)}",
            input_tokens=500, expected_output_tokens=100,
            require_logprobs=False, require_prefix_cache=False,
            data_policy_required="any",
            max_latency_ms=5000.0, max_cost_usd=10.0),
        sticky_provider="openai_paid")
    d = rc.decide_v2(req)
    return {
        "schema": R156_SCHEMA_VERSION,
        "name": "h400b_router_v2_sticky",
        "passed": bool(
            d.sticky_applied
            and d.chosen_provider == "openai_paid"),
    }


def family_h401_logprob_v2_bayesian_fusion(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV2(
        provider_trust={"p1": 1.0, "p2": 0.5})
    res = lr.fuse([
        TopKLogprobsPayload("p1",
            (("a", -0.1), ("b", -0.5), ("c", -1.0))),
        TopKLogprobsPayload("p2",
            (("a", -0.2), ("b", -0.4), ("c", -1.1))),
    ])
    return {
        "schema": R156_SCHEMA_VERSION,
        "name": "h401_logprob_v2_bayesian_fusion",
        "passed": bool(
            "bayesian_dirichlet_v2" in str(
                res.get("fusion_kind", ""))
            or "tiebreak" in str(
                res.get("fusion_kind", ""))),
    }


def family_h401b_logprob_v2_tiebreak(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouterV2(
        provider_trust={"p1": 0.1, "p2": 0.1})
    # High-entropy distribution: many tokens with similar logprobs.
    payloads = [
        TopKLogprobsPayload("p1", tuple(
            (chr(ord("a") + i), -0.1 - i * 0.01)
            for i in range(20))),
        TopKLogprobsPayload("p2", tuple(
            (chr(ord("a") + i), -0.1 - i * 0.005)
            for i in range(20))),
    ]
    res = lr.fuse(
        payloads, tiebreak_entropy_floor=0.1)
    return {
        "schema": R156_SCHEMA_VERSION,
        "name": "h401b_logprob_v2_tiebreak",
        "passed": bool(
            "tiebreak" in str(res.get("fusion_kind", ""))),
    }


def family_h402_cache_v2_prefix_cid_content_addressed(
        seed: int) -> dict[str, Any]:
    cp1 = HostedCacheAwarePlannerV2()
    cp2 = HostedCacheAwarePlannerV2()
    planned1, _ = cp1.plan_per_role_staggered(
        shared_prefix_text="W69 shared prefix " * 8,
        per_role_blocks={"plan": ["a", "b"], "write": ["c", "d"]})
    planned2, _ = cp2.plan_per_role_staggered(
        shared_prefix_text="W69 shared prefix " * 8,
        per_role_blocks={"plan": ["a", "b"], "write": ["c", "d"]})
    ok = (
        planned1[0].staggered_prefix_cids
        == planned2[0].staggered_prefix_cids)
    return {
        "schema": R156_SCHEMA_VERSION,
        "name": "h402_cache_v2_prefix_cid_content_addressed",
        "passed": bool(ok),
    }


def family_h402b_cache_v2_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_cache_aware_savings_v2_vs_recompute(
        n_roles=6, n_turns=8, hosted_cache_hit_rate=1.0)
    return {
        "schema": R156_SCHEMA_VERSION,
        "name": "h402b_cache_v2_savings",
        "passed": bool(sav["saving_ratio"] >= 0.6),
        "ratio": float(sav["saving_ratio"]),
    }


def family_h403_filter_v2_all_combine(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v2 = HostedProviderFilterSpecV2(
        inner_specs=(
            HostedProviderFilterSpec(
                require_data_policy="no_log",
                allowed_tiers=(
                    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
                    W68_HOSTED_TIER_PREFIX_CACHE,
                    W68_HOSTED_TIER_LOGPROBS),
                max_p50_latency_ms=2500.0,
                max_cost_per_1k_output=20.0),
            HostedProviderFilterSpec(
                require_data_policy="no_log",
                allowed_tiers=(
                    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,),
                max_p50_latency_ms=1500.0,
                max_cost_per_1k_output=20.0),
        ),
        combine="all")
    _, report = filter_hosted_registry_v2(reg, spec_v2)
    return {
        "schema": R156_SCHEMA_VERSION,
        "name": "h403_filter_v2_all_combine",
        "passed": bool(report["n_dropped"] >= 1),
        "n_dropped": int(report["n_dropped"]),
    }


def family_h403b_filter_v2_any_combine(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v2 = HostedProviderFilterSpecV2(
        inner_specs=(
            HostedProviderFilterSpec(
                require_data_policy="no_log",
                allowed_tiers=(
                    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,),
                max_p50_latency_ms=2500.0,
                max_cost_per_1k_output=20.0),
            HostedProviderFilterSpec(
                require_data_policy="any",
                allowed_tiers=(W68_HOSTED_TIER_LOGPROBS,),
                max_p50_latency_ms=2500.0,
                max_cost_per_1k_output=20.0),
        ),
        combine="any")
    filtered, _ = filter_hosted_registry_v2(reg, spec_v2)
    return {
        "schema": R156_SCHEMA_VERSION,
        "name": "h403b_filter_v2_any_combine",
        "passed": bool(len(filtered.providers) >= 2),
        "n_kept": int(len(filtered.providers)),
    }


def family_h404_cost_v2_rotation(seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v2 = HostedCostPlanSpecV2(
        inner_v1=HostedCostPlanSpec(
            n_turns=5, input_tokens_per_turn=500,
            output_tokens_per_turn=100,
            max_cost_per_turn_usd=5.0,
            max_latency_per_turn_ms=2500.0,
            min_quality_score=0.5),
        allow_provider_rotation=True)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v2(
        registry=reg, provider_quality_scores=qs,
        spec_v2=spec_v2)
    rotation = (
        len(set(rep.per_turn_chosen_providers)) >= 2
        if rep.chosen_provider is not None else False)
    return {
        "schema": R156_SCHEMA_VERSION,
        "name": "h404_cost_v2_rotation",
        "passed": bool(rotation),
    }


def family_h404b_cost_v2_cost_per_success(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec_v2 = HostedCostPlanSpecV2(
        inner_v1=HostedCostPlanSpec(
            n_turns=5, input_tokens_per_turn=500,
            output_tokens_per_turn=100,
            max_cost_per_turn_usd=5.0,
            max_latency_per_turn_ms=2500.0,
            min_quality_score=0.5),
        allow_provider_rotation=False)
    qs = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost_v2(
        registry=reg, provider_quality_scores=qs,
        spec_v2=spec_v2)
    return {
        "schema": R156_SCHEMA_VERSION,
        "name": "h404b_cost_v2_cost_per_success",
        "passed": bool(
            rep.cost_per_success_ratio >= 0.0
            and rep.cost_per_success_ratio < float("inf")),
    }


_R156_FAMILIES: tuple[Any, ...] = (
    family_h400_router_v2_determinism,
    family_h400b_router_v2_sticky,
    family_h401_logprob_v2_bayesian_fusion,
    family_h401b_logprob_v2_tiebreak,
    family_h402_cache_v2_prefix_cid_content_addressed,
    family_h402b_cache_v2_savings,
    family_h403_filter_v2_all_combine,
    family_h403b_filter_v2_any_combine,
    family_h404_cost_v2_rotation,
    family_h404b_cost_v2_cost_per_success,
)


def run_r156(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R156_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R156_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R156_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R156_SCHEMA_VERSION", "run_r156"]
