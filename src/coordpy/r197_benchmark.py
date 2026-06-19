"""W79 R-197 benchmark family — Hosted control-plane V12.

H1280..H1289 cell families (10 H-bars):

* H1280  Router V12 deterministic on identical V12 request
* H1281  Logprob router V12 carries V11 inner state
* H1282  Cache planner V12 ten-layer rotated; ≥ 90 % savings
* H1283  Cost planner V12 abstain flag preserved
* H1284  Boundary V12 ≥ 56 blocked axes
* H1285  Boundary V12 falsifier honest=0 on a V24 axis
* H1286  Provider filter V11 drops under RTRLD pressure
* H1287  Handoff V11 RTRLD promotion (high pressure → Plane B)
* H1288  Handoff V11 controlled-runtime promotion when
         needs_controlled_runtime=True
* H1289  Handoff V11 cross-plane savings ≥ 0.88 at default config
"""

from __future__ import annotations

from typing import Any, Sequence

from coordpy.hosted_cache_aware_planner_v12 import (
    HostedCacheAwarePlannerV12,
    W79_HOSTED_CACHE_AWARE_PLANNER_V12_DEFAULT_SAVINGS,
    emit_hosted_cache_aware_planner_v12_witness,
)
from coordpy.hosted_cost_planner_v12 import (
    HostedCostPlannerV12,
)
from coordpy.hosted_logprob_router_v12 import (
    HostedLogprobRouterV12,
)
from coordpy.hosted_provider_filter_v11 import (
    HostedProviderFilterSpecV11, filter_hosted_registry_v11,
)
from coordpy.hosted_provider_filter_v10 import (
    HostedProviderFilterSpecV10,
)
from coordpy.hosted_provider_filter_v9 import (
    HostedProviderFilterSpecV9,
)
from coordpy.hosted_real_handoff_coordinator_v11 import (
    HandoffEnvelopeV11, HandoffRequestV11,
    HostedRealHandoffCoordinatorV11,
    W79_HANDOFF_DECISION_CONTROLLED_RUNTIME,
    W79_HANDOFF_DECISION_REPLACEMENT_THEN_RESTART_AFTER_LONG_DELAY_FALLBACK,
    hosted_real_handoff_v11_replacement_then_restart_after_long_delay_aware_savings,
)
from coordpy.hosted_real_handoff_coordinator import (
    HandoffRequest, W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
)
from coordpy.hosted_real_handoff_coordinator_v2 import (
    HandoffRequestV2,
)
from coordpy.hosted_real_handoff_coordinator_v3 import (
    HandoffRequestV3,
)
from coordpy.hosted_real_handoff_coordinator_v4 import (
    HandoffRequestV4,
)
from coordpy.hosted_real_handoff_coordinator_v5 import (
    HandoffRequestV5,
)
from coordpy.hosted_real_handoff_coordinator_v6 import (
    HandoffRequestV6,
)
from coordpy.hosted_real_handoff_coordinator_v7 import (
    HandoffRequestV7,
)
from coordpy.hosted_real_handoff_coordinator_v8 import (
    HandoffRequestV8,
)
from coordpy.hosted_real_handoff_coordinator_v9 import (
    HandoffRequestV9,
)
from coordpy.hosted_real_handoff_coordinator_v10 import (
    HandoffRequestV10,
)
from coordpy.hosted_real_substrate_boundary_v12 import (
    build_default_hosted_real_substrate_boundary_v12,
    probe_hosted_real_substrate_boundary_v12_falsifier,
)
from coordpy.hosted_router_controller import (
    default_hosted_registry,
)
from coordpy.hosted_router_controller_v12 import (
    HostedRouterControllerV12, HostedRoutingRequestV12,
)
from coordpy.hosted_router_controller_v11 import (
    HostedRoutingRequestV11,
)
from coordpy.hosted_router_controller_v10 import (
    HostedRoutingRequestV10,
)
from coordpy.hosted_router_controller_v9 import (
    HostedRoutingRequestV9,
)
from coordpy.hosted_router_controller_v8 import (
    HostedRoutingRequestV8,
)
from coordpy.hosted_router_controller_v7 import (
    HostedRoutingRequestV7,
)
from coordpy.hosted_router_controller_v6 import (
    HostedRoutingRequestV6,
)
from coordpy.hosted_router_controller_v5 import (
    HostedRoutingRequestV5,
)
from coordpy.hosted_router_controller_v4 import (
    HostedRoutingRequestV4,
)
from coordpy.hosted_router_controller_v3 import (
    HostedRoutingRequestV3,
)
from coordpy.hosted_router_controller_v2 import (
    HostedRoutingRequestV2,
)
from coordpy.hosted_router_controller import (
    HostedRoutingRequest,
)


R197_SCHEMA_VERSION: str = "coordpy.r197_benchmark.v1"


def _make_router_req_v12() -> HostedRoutingRequestV12:
    return HostedRoutingRequestV12(
        inner_v11=HostedRoutingRequestV11(
            inner_v10=HostedRoutingRequestV10(
                inner_v9=HostedRoutingRequestV9(
                    inner_v8=HostedRoutingRequestV8(
                        inner_v7=HostedRoutingRequestV7(
                            inner_v6=HostedRoutingRequestV6(
                                inner_v5=HostedRoutingRequestV5(
                                    inner_v4=HostedRoutingRequestV4(
                                        inner_v3=HostedRoutingRequestV3(
                                            inner_v2=HostedRoutingRequestV2(
                                                inner_v1=HostedRoutingRequest(
                                                    request_cid="r197",
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
                    weight_chain_then_restart_after_rtr_match=0.4),
                post_restart_replacement_pressure=0.9,
                weight_post_restart_replacement_pressure=0.6,
                weight_post_restart_replacement_after_pcr_match=0.4),
            long_horizon_reconstruction_pressure=0.95,
            weight_long_horizon_reconstruction_pressure=0.6,
            weight_long_horizon_reconstruction_after_pcr_match=0.4),
        replacement_then_restart_after_long_delay_pressure=0.97,
        weight_replacement_then_restart_after_long_delay_pressure=0.6,
        weight_replacement_then_restart_after_long_delay_match=0.4)


def _make_req_v11(
        rc: str,
        rtrld_pressure: float = 0.0,
        rtrld_trajectory_cid: str = "",
        long_delay_blackout: int = 0,
        needs_controlled_runtime: bool = False,
        needs_text_only: bool = True,
        needs_substrate_state_access: bool = False,
) -> HandoffRequestV11:
    base_v10 = HandoffRequestV10(
        inner_v9=HandoffRequestV9(
            inner_v8=HandoffRequestV8(
                inner_v7=HandoffRequestV7(
                    inner_v6=HandoffRequestV6(
                        inner_v5=HandoffRequestV5(
                            inner_v4=HandoffRequestV4(
                                inner_v3=HandoffRequestV3(
                                    inner_v2=HandoffRequestV2(
                                        inner_v1=HandoffRequest(
                                            request_cid=str(rc),
                                            needs_text_only=bool(
                                                needs_text_only),
                                            needs_substrate_state_access=bool(
                                                needs_substrate_state_access)),
                                        visible_token_budget=128,
                                        baseline_token_cost=512,
                                        dominant_repair_label=0),
                                    restart_pressure=0.0,
                                    delayed_repair_trajectory_cid="",
                                    delay_turns=0,
                                    expected_substrate_trust=0.7),
                                rejoin_pressure=0.0,
                                restart_repair_trajectory_cid="",
                                rejoin_lag_turns=0,
                                expected_substrate_trust_v4=0.7),
                            replacement_pressure=0.0,
                            replacement_repair_trajectory_cid="",
                            replacement_lag_turns=0,
                            expected_substrate_trust_v5=0.7),
                        compound_pressure=0.0,
                        compound_repair_trajectory_cid="",
                        compound_window_turns=0,
                        expected_substrate_trust_v6=0.7),
                    compound_chain_pressure=0.0,
                    compound_chain_repair_trajectory_cid="",
                    compound_chain_window_turns=0,
                    expected_substrate_trust_v7=0.7),
                compound_chain_then_restart_pressure=0.0,
                compound_chain_then_restart_trajectory_cid="",
                post_compound_chain_restart_window_turns=0,
                expected_substrate_trust_v8=0.7),
            post_restart_replacement_pressure=0.0,
            post_restart_replacement_trajectory_cid="",
            post_restart_replacement_window_turns=0,
            expected_substrate_trust_v9=0.7),
        long_horizon_reconstruction_pressure=0.0,
        long_horizon_reconstruction_trajectory_cid="",
        long_horizon_blackout_window_turns=0,
        expected_substrate_trust_v10=0.7)
    return HandoffRequestV11(
        inner_v10=base_v10,
        replacement_then_restart_after_long_delay_pressure=float(
            rtrld_pressure),
        replacement_then_restart_after_long_delay_trajectory_cid=str(
            rtrld_trajectory_cid),
        long_delay_blackout_window_turns=int(long_delay_blackout),
        needs_controlled_runtime=bool(needs_controlled_runtime),
        expected_substrate_trust_v11=0.7)


def run_r197(*, seeds: Sequence[int]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    reg = default_hosted_registry()
    router = HostedRouterControllerV12.init(reg, {
        "openrouter_paid": 0.88, "openai_paid": 0.95})
    req = _make_router_req_v12()
    d1 = router.decide_v12(req)
    d2 = router.decide_v12(req)
    cells["H1280"] = bool(
        str(d1.chosen_provider or "")
        == str(d2.chosen_provider or ""))
    logprob_v12 = HostedLogprobRouterV12()
    cells["H1281"] = bool(
        logprob_v12.inner_v11 is not None)
    planner = HostedCacheAwarePlannerV12()
    plan, rep = planner.plan_per_role_ten_layer_rotated(
        shared_prefix_text="R197 long shared prefix " * 24,
        per_role_blocks={"plan": ["a", "b"], "research": ["c"]})
    cells["H1282"] = bool(
        float(rep.get("v12_expected_savings_ratio", 0.0))
        >= float(W79_HOSTED_CACHE_AWARE_PLANNER_V12_DEFAULT_SAVINGS))
    cost = HostedCostPlannerV12()
    cells["H1283"] = bool(
        bool(cost.replacement_then_restart_after_long_delay_violation_abstain))
    boundary = build_default_hosted_real_substrate_boundary_v12()
    cells["H1284"] = bool(len(boundary.blocked_axes) >= 56)
    fbo = probe_hosted_real_substrate_boundary_v12_falsifier(
        boundary=boundary,
        claimed_axis=(
            "replacement_then_restart_after_long_delay_trajectory_cid"),
        claim_satisfied_at_hosted=False)
    cells["H1285"] = bool(float(fbo.falsifier_score) == 0.0)
    # Reuse the W78 default provider filter V10 (which itself
    # carries the V9..V2 wrappers byte-for-byte) so the V11
    # filter test focuses on the W79 axis.
    from coordpy.w78_team import W78Params as _W78P
    _w78p = _W78P.build_default(seed=78001)
    spec_v11 = HostedProviderFilterSpecV11(
        inner_v10=_w78p.hosted_provider_filter_v10,
        replacement_then_restart_after_long_delay_pressure=0.9,
        replacement_then_restart_after_long_delay_pressure_floor=0.5,
        max_replacement_then_restart_after_long_delay_noise_per_provider={
            "openrouter_paid": 0.06,
            "openai_paid": 1.0})
    _, frep = filter_hosted_registry_v11(
        reg, spec_v11,
        provider_replacement_then_restart_after_long_delay_noise={
            "openrouter_paid": 0.5,
            "openai_paid": 0.1})
    cells["H1286"] = bool(
        int(frep.get(
            "n_dropped_under_replacement_then_restart_after_long_delay",
            0)) >= 1)
    coord = HostedRealHandoffCoordinatorV11()
    e_rtrld = coord.decide_v11(
        req_v11=_make_req_v11(
            "rc-rtrld", rtrld_pressure=0.97,
            rtrld_trajectory_cid="rtrld_x",
            long_delay_blackout=100))
    cells["H1287"] = bool(
        e_rtrld.decision_v11
        == W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY
        and float(
            e_rtrld.replacement_then_restart_after_long_delay_alignment)
        == 1.0)
    e_ctrl = coord.decide_v11(
        req_v11=_make_req_v11(
            "rc-ctrl", needs_controlled_runtime=True))
    cells["H1288"] = bool(
        e_ctrl.decision_v11
        == W79_HANDOFF_DECISION_CONTROLLED_RUNTIME
        and bool(e_ctrl.controlled_runtime_active))
    sv = (
        hosted_real_handoff_v11_replacement_then_restart_after_long_delay_aware_savings(
            n_turns=100))
    cells["H1289"] = bool(float(sv["saving_ratio"]) >= 0.88)
    return {
        "schema": R197_SCHEMA_VERSION,
        "seeds": list(seeds),
        "cells": dict(cells),
        "all_pass": bool(all(bool(v) for v in cells.values())),
    }


__all__ = ["R197_SCHEMA_VERSION", "run_r197"]
