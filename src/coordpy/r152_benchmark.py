"""W68 R-152 benchmark family — Hosted Control Plane (Plane A).

Exercises the W68 hosted control plane: router decisions,
logprob fusion, prefix-cache-aware planning, provider filtering,
cost/latency optimisation, hosted-vs-real wall. **Plane A only.**

H300..H309 cell families (10 H-bars):

* H300   HostedRouterController determinism
* H300b  HostedRouterController rejects logprob-blind provider
* H301   HostedLogprobRouter shared-top-k fusion fires
* H301b  HostedLogprobRouter text-only quorum fallback
* H302   HostedCacheAwarePlanner shared_prefix_cid is content-
         addressed
* H302b  HostedCacheAwarePlanner ≥ 50 % token savings on 4 turns
* H303   HostedProviderFilter drops train-policy providers
* H304   HostedCostPlanner picks free provider when quality
         threshold met
* H304b  HostedCostPlanner abstains when no provider passes
* H305   HostedRealSubstrateBoundary blocks hidden_state_read
"""

from __future__ import annotations

from typing import Any, Sequence

from .hosted_router_controller import (
    HostedRouterController,
    HostedRoutingRequest,
    default_hosted_registry,
    W68_HOSTED_TIER_LOGPROBS,
    W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
    W68_HOSTED_TIER_PREFIX_CACHE,
    W68_HOSTED_TIER_TEXT_ONLY,
)
from .hosted_logprob_router import (
    HostedLogprobRouter, TopKLogprobsPayload,
)
from .hosted_cache_aware_planner import (
    HostedCacheAwarePlanner,
    hosted_cache_aware_savings_vs_recompute,
)
from .hosted_provider_filter import (
    HostedProviderFilterSpec, filter_hosted_registry,
)
from .hosted_cost_planner import (
    HostedCostPlanSpec, plan_hosted_cost,
)
from .hosted_real_substrate_boundary import (
    build_default_hosted_real_substrate_boundary,
    probe_hosted_real_substrate_boundary_falsifier,
)


R152_SCHEMA_VERSION: str = "coordpy.r152_benchmark.v1"


def family_h300_router_determinism(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc1 = HostedRouterController(registry=reg)
    rc2 = HostedRouterController(registry=reg)
    req = HostedRoutingRequest(
        request_cid=f"req_{int(seed)}",
        input_tokens=1000,
        expected_output_tokens=300,
        require_logprobs=True,
        require_prefix_cache=True,
        data_policy_required="no_log",
        max_latency_ms=2000.0, max_cost_usd=50.0)
    d1 = rc1.decide(req)
    d2 = rc2.decide(req)
    return {
        "schema": R152_SCHEMA_VERSION,
        "name": "h300_router_determinism",
        "passed": bool(
            d1.chosen_provider == d2.chosen_provider
            and d1.cid() == d2.cid()),
    }


def family_h300b_router_rejects_no_logprobs(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    rc = HostedRouterController(registry=reg)
    req = HostedRoutingRequest(
        request_cid=f"req_lpb_{int(seed)}",
        input_tokens=500, expected_output_tokens=100,
        require_logprobs=True,
        require_prefix_cache=False,
        data_policy_required="any",
        max_latency_ms=5000.0, max_cost_usd=10.0)
    d = rc.decide(req)
    # ollama_local does not support logprobs ⇒ must be rejected.
    rejected_names = {n for n, _r in d.rejected_providers}
    return {
        "schema": R152_SCHEMA_VERSION,
        "name": "h300b_router_rejects_no_logprobs",
        "passed": bool(
            "ollama_local" in rejected_names
            and "openrouter_free" in rejected_names),
        "rejected": sorted(rejected_names),
    }


def family_h301_logprob_fusion_shared_top_k(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouter()
    res = lr.fuse([
        TopKLogprobsPayload("p1",
            (("a", -0.1), ("b", -0.5), ("c", -1.0))),
        TopKLogprobsPayload("p2",
            (("a", -0.2), ("b", -0.4), ("c", -1.1))),
    ])
    return {
        "schema": R152_SCHEMA_VERSION,
        "name": "h301_logprob_fusion_shared_top_k",
        "passed": bool(
            res["fusion_kind"] == "weighted_mean_shared_top_k"
            and len(res["fused_distribution"]) >= 1
            and res["fused_distribution"][0][0] == "a"),
    }


def family_h301b_logprob_text_only_fallback(
        seed: int) -> dict[str, Any]:
    lr = HostedLogprobRouter()
    res = lr.fuse(
        [], text_only_responses=["yes", "yes", "no"])
    return {
        "schema": R152_SCHEMA_VERSION,
        "name": "h301b_logprob_text_only_fallback",
        "passed": bool(
            res["kind"] == "text_only_quorum"
            and res["decision"] == "yes"),
    }


def family_h302_prefix_cid_is_content_addressed(
        seed: int) -> dict[str, Any]:
    cp = HostedCacheAwarePlanner()
    planned1, _ = cp.plan(
        shared_prefix_text="A common prefix",
        role_blocks=["x", "y"])
    cp2 = HostedCacheAwarePlanner()
    planned2, _ = cp2.plan(
        shared_prefix_text="A common prefix",
        role_blocks=["x", "y"])
    return {
        "schema": R152_SCHEMA_VERSION,
        "name": "h302_prefix_cid_is_content_addressed",
        "passed": bool(
            planned1[0].shared_prefix_cid
            == planned2[0].shared_prefix_cid),
    }


def family_h302b_cache_aware_savings(
        seed: int) -> dict[str, Any]:
    sav = hosted_cache_aware_savings_vs_recompute(
        n_turns=10, prefix_tokens_per_turn=500,
        role_tokens_per_turn=100,
        hosted_cache_hit_rate=1.0)
    return {
        "schema": R152_SCHEMA_VERSION,
        "name": "h302b_cache_aware_savings",
        "passed": bool(sav["saving_ratio"] >= 0.5),
        "saving_ratio": float(sav["saving_ratio"]),
    }


def family_h303_filter_drops_train_policy(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec = HostedProviderFilterSpec(
        require_data_policy="no_log",
        allowed_tiers=(W68_HOSTED_TIER_LOGPROBS_AND_PREFIX_CACHE,
                       W68_HOSTED_TIER_PREFIX_CACHE,
                       W68_HOSTED_TIER_LOGPROBS),
        max_p50_latency_ms=2500.0,
        max_cost_per_1k_output=20.0)
    _, report = filter_hosted_registry(reg, spec)
    return {
        "schema": R152_SCHEMA_VERSION,
        "name": "h303_filter_drops_train_policy",
        "passed": bool(report["n_dropped"] >= 1),
        "n_dropped": int(report["n_dropped"]),
    }


def family_h304_cost_planner_picks_free(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec = HostedCostPlanSpec(
        n_turns=5, input_tokens_per_turn=500,
        output_tokens_per_turn=100,
        max_cost_per_turn_usd=5.0,
        max_latency_per_turn_ms=2500.0,
        min_quality_score=0.5)
    quality = {p.name: 0.8 for p in reg.providers}
    rep = plan_hosted_cost(
        registry=reg,
        provider_quality_scores=quality, spec=spec)
    return {
        "schema": R152_SCHEMA_VERSION,
        "name": "h304_cost_planner_picks_free",
        "passed": bool(
            rep.chosen_provider is not None
            and rep.estimated_total_cost_usd == 0.0),
        "chosen": (
            str(rep.chosen_provider)
            if rep.chosen_provider is not None else None),
    }


def family_h304b_cost_planner_abstains(
        seed: int) -> dict[str, Any]:
    reg = default_hosted_registry()
    spec = HostedCostPlanSpec(
        n_turns=5, input_tokens_per_turn=500,
        output_tokens_per_turn=100,
        max_cost_per_turn_usd=0.001,
        max_latency_per_turn_ms=50.0,
        min_quality_score=0.99)
    quality = {p.name: 0.5 for p in reg.providers}
    rep = plan_hosted_cost(
        registry=reg, provider_quality_scores=quality,
        spec=spec)
    return {
        "schema": R152_SCHEMA_VERSION,
        "name": "h304b_cost_planner_abstains",
        "passed": bool(rep.chosen_provider is None),
    }


def family_h305_boundary_blocks_hidden_state(
        seed: int) -> dict[str, Any]:
    b = build_default_hosted_real_substrate_boundary()
    fal_honest = probe_hosted_real_substrate_boundary_falsifier(
        boundary=b, claimed_axis="hidden_state_read",
        claim_satisfied_at_hosted=False)
    fal_dishonest = (
        probe_hosted_real_substrate_boundary_falsifier(
            boundary=b, claimed_axis="hidden_state_read",
            claim_satisfied_at_hosted=True))
    return {
        "schema": R152_SCHEMA_VERSION,
        "name": "h305_boundary_blocks_hidden_state",
        "passed": bool(
            fal_honest.falsifier_score == 0.0
            and fal_dishonest.falsifier_score == 1.0),
    }


_R152_FAMILIES: tuple[Any, ...] = (
    family_h300_router_determinism,
    family_h300b_router_rejects_no_logprobs,
    family_h301_logprob_fusion_shared_top_k,
    family_h301b_logprob_text_only_fallback,
    family_h302_prefix_cid_is_content_addressed,
    family_h302b_cache_aware_savings,
    family_h303_filter_drops_train_policy,
    family_h304_cost_planner_picks_free,
    family_h304b_cost_planner_abstains,
    family_h305_boundary_blocks_hidden_state,
)


def run_r152(
        seeds: Sequence[int] = (199, 299, 399),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in seeds:
        results: dict[str, Any] = {}
        for fn in _R152_FAMILIES:
            try:
                r = fn(int(s))
            except Exception as e:
                r = {
                    "schema": R152_SCHEMA_VERSION,
                    "name": getattr(fn, "__name__", "unknown"),
                    "passed": False,
                    "exception": str(e),
                }
            results[r["name"]] = r
        out.append({
            "schema": R152_SCHEMA_VERSION,
            "seed": int(s),
            "family_results": results,
        })
    return out


__all__ = ["R152_SCHEMA_VERSION", "run_r152"]
