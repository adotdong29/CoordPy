"""W83 — team orchestrator tests."""

from __future__ import annotations


def test_w83_team_trivial_passthrough():
    from coordpy.w83_team import (
        W83Params, W83Team, verify_w83_handoff,
    )
    params = W83Params.build_trivial()
    team = W83Team(params=params)
    env = team.run_team_turn(
        w79_outer_cid="W79_outer_test_cid")
    # Trivial pass-through carries the W79 outer CID forward.
    assert env.w79_outer_cid == "W79_outer_test_cid"
    # No mechanism runs in trivial mode, so the load-bearing
    # claims are False — verify_w83_handoff should report fails.
    ok, fails = verify_w83_handoff(
        env, params, "W79_outer_test_cid")
    assert not ok
    assert (
        "w83_composed_memory_does_not_beat_baselines" in fails)


def test_w83_team_default_build_and_run():
    from coordpy.w83_team import (
        build_default_w83_team, verify_w83_handoff,
    )
    team = build_default_w83_team()
    env = team.run_team_turn(w79_outer_cid="seed_w79_cid")
    assert env.schema == "coordpy.w83_team.v1"
    assert env.composed_memory_beats_baselines
    assert env.slot_reconstruction_beats_baselines
    assert env.online_economics_beats_offline
    assert env.integrity_trust_consensus_beats_w81
    assert env.compose_pipeline_audit_verifiable_rate >= 1.0
    assert env.compose_pipeline_lowers_w81_error
    assert env.bounded_window_v3_failure_rate >= 1.0 - 1e-12
    assert env.cross_runtime_projector_beats_w82
    assert env.distributed_gateway_merkle_match
    assert env.hosted_audit_merkle_root_matches
    assert env.recovery_overall_task_success_rate >= 0.75
    assert env.recovery_new_regime_task_success_rate >= 0.50
    ok, fails = verify_w83_handoff(
        env, team.params, env.w79_outer_cid)
    assert ok, fails


def test_w83_team_envelope_drift_detected():
    from coordpy.w83_team import (
        build_default_w83_team, verify_w83_handoff,
    )
    team = build_default_w83_team()
    env = team.run_team_turn(w79_outer_cid="cid_a")
    # Verify against a different expected W79 outer CID.
    ok, fails = verify_w83_handoff(
        env, team.params, "cid_b_different")
    assert not ok
    assert (
        "w83_envelope_w79_outer_cid_drift" in fails)


def test_w83_team_envelope_cid_deterministic_on_seed():
    from coordpy.w83_team import (
        W83Params, W83Team,
    )
    p_a = W83Params.build_default(seed=83_000)
    p_b = W83Params.build_default(seed=83_000)
    assert str(p_a.cid()) == str(p_b.cid())


def test_w83_team_failure_modes_listed():
    from coordpy.w83_team import W83_FAILURE_MODES
    assert len(W83_FAILURE_MODES) >= 12
    for mode in W83_FAILURE_MODES:
        assert mode.startswith("w83_")
