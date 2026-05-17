"""W75 tests — hosted handoff V7 + boundary V8."""

from __future__ import annotations

from coordpy.hosted_real_handoff_coordinator import (
    HandoffRequest,
    W69_HANDOFF_DECISION_HOSTED_ONLY,
    W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY,
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
    HandoffRequestV7, HostedRealHandoffCoordinatorV7,
    W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK,
    hosted_real_handoff_v7_compound_chain_aware_savings,
    probe_hosted_real_handoff_v7_compound_chain_falsifier,
)
from coordpy.hosted_real_substrate_boundary_v3 import (
    W70_FRONTIER_BLOCKED_AXES,
)
from coordpy.hosted_real_substrate_boundary_v8 import (
    W75_FRONTIER_BLOCKED_AXES,
    W75_HOSTED_PLANE_BLOCKED_AXES_V8,
    build_default_hosted_real_substrate_boundary_v8,
    probe_hosted_real_substrate_boundary_v8_falsifier,
)


def _build_req(**kw) -> HandoffRequestV7:
    return HandoffRequestV7(
        inner_v6=HandoffRequestV6(
            inner_v5=HandoffRequestV5(
                inner_v4=HandoffRequestV4(
                    inner_v3=HandoffRequestV3(
                        inner_v2=HandoffRequestV2(
                            inner_v1=HandoffRequest(
                                request_cid=kw.get("rc", "t"),
                                needs_text_only=kw.get(
                                    "text", True),
                                needs_substrate_state_access=kw
                                .get("sub", False)),
                            visible_token_budget=256,
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
        compound_chain_pressure=kw.get(
            "compound_chain_pressure", 0.0),
        compound_chain_repair_trajectory_cid=kw.get(
            "compound_chain_repair_trajectory_cid", ""),
        compound_chain_window_turns=kw.get(
            "compound_chain_window_turns", 0),
        expected_substrate_trust_v7=0.7)


def test_handoff_v7_text_only_to_plane_a() -> None:
    c = HostedRealHandoffCoordinatorV7()
    env = c.decide_v7(req_v7=_build_req(rc="text"))
    assert env.decision_v7 == W69_HANDOFF_DECISION_HOSTED_ONLY


def test_handoff_v7_substrate_only_to_plane_b() -> None:
    c = HostedRealHandoffCoordinatorV7()
    env = c.decide_v7(
        req_v7=_build_req(rc="sub", text=False, sub=True))
    assert env.decision_v7 == (
        W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)


def test_handoff_v7_compound_chain_promotion() -> None:
    c = HostedRealHandoffCoordinatorV7()
    env = c.decide_v7(
        req_v7=_build_req(
            rc="chain", compound_chain_pressure=0.9,
            compound_chain_repair_trajectory_cid="cid",
            compound_chain_window_turns=4))
    assert env.decision_v7 == (
        W69_HANDOFF_DECISION_REAL_SUBSTRATE_ONLY)
    assert env.compound_chain_alignment == 1.0


def test_handoff_v7_compound_chain_fallback() -> None:
    c = HostedRealHandoffCoordinatorV7()
    env = c.decide_v7(
        req_v7=_build_req(
            rc="fb",
            compound_chain_pressure=0.0,
            compound_chain_repair_trajectory_cid="cid",
            compound_chain_window_turns=4))
    assert env.decision_v7 == (
        W75_HANDOFF_DECISION_COMPOUND_CHAIN_REPAIR_FALLBACK)
    assert env.compound_chain_repair_rtr_fallback_active


def test_handoff_v7_cross_plane_savings_above_84() -> None:
    r = hosted_real_handoff_v7_compound_chain_aware_savings(
        n_turns=100)
    assert r["saving_ratio"] >= 0.84


def test_handoff_v7_chain_falsifier_dishonest_scored_1() -> None:
    c = HostedRealHandoffCoordinatorV7()
    env_text = c.decide_v7(req_v7=_build_req(rc="text"))
    f_dishonest = (
        probe_hosted_real_handoff_v7_compound_chain_falsifier(
            envelope_v7=env_text,
            claim_satisfied=True))
    assert f_dishonest.falsifier_score == 1.0


def test_boundary_v8_blocked_axes_ge_37() -> None:
    b = build_default_hosted_real_substrate_boundary_v8()
    assert len(b.blocked_axes) >= 37
    assert len(W75_HOSTED_PLANE_BLOCKED_AXES_V8) >= 37


def test_boundary_v8_frontier_blocked_unchanged() -> None:
    assert (
        tuple(W75_FRONTIER_BLOCKED_AXES)
        == tuple(W70_FRONTIER_BLOCKED_AXES))


def test_boundary_v8_falsifier_honest_zero_dishonest_one() -> None:
    b = build_default_hosted_real_substrate_boundary_v8()
    honest = probe_hosted_real_substrate_boundary_v8_falsifier(
        boundary=b,
        claimed_axis="compound_chain_repair_trajectory_cid",
        claim_satisfied_at_hosted=False)
    dishonest = probe_hosted_real_substrate_boundary_v8_falsifier(
        boundary=b,
        claimed_axis="compound_chain_repair_trajectory_cid",
        claim_satisfied_at_hosted=True)
    assert honest.falsifier_score == 0.0
    assert dishonest.falsifier_score == 1.0
