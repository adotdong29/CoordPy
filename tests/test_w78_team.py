"""W78 — team orchestrator tests + W77 envelope chaining."""

from __future__ import annotations


def test_w78_team_runs_and_verifies():
    from coordpy.w78_team import (
        build_default_w78_team, verify_w78_handoff,
    )
    team = build_default_w78_team()
    env = team.run_team_turn(w77_outer_cid="w77_outer_x")
    ok, fails = verify_w78_handoff(
        env, team.params, "w77_outer_x")
    assert ok, fails
    assert bool(env.substrate_v23_used)
    assert bool(env.twenty_three_way_used)
    assert float(env.masc_v14_v23_beats_v22_rate) >= 0.5
    assert float(env.long_horizon_reconstruction_success_rate) >= (
        0.5)
    assert bool(env.bounded_window_all_fixed_k_failed)


def test_w78_team_w77_envelope_invariant_trivial():
    """W77 envelope's outer_cid is byte-for-byte preserved under
    trivial passthrough."""
    from coordpy.w78_team import W78Params, W78Team
    team = W78Team(params=W78Params.build_trivial())
    env = team.run_team_turn(w77_outer_cid="w77_dummy_cid")
    assert str(env.w77_outer_cid) == "w77_dummy_cid"
    # Trivial path emits an envelope, but its substrate flags are
    # all False / empty.
    assert not bool(env.substrate_v23_used)
    assert not bool(env.twenty_three_way_used)


def test_w78_team_n_failure_modes_enumerated():
    from coordpy.w78_team import W78_FAILURE_MODES
    assert len(W78_FAILURE_MODES) >= 50


def test_w78_failure_modes_disjoint():
    from coordpy.w78_team import W78_FAILURE_MODES
    assert len(set(W78_FAILURE_MODES)) == len(W78_FAILURE_MODES)


def test_w78_verify_detects_drift():
    from coordpy.w78_team import (
        build_default_w78_team, verify_w78_handoff,
    )
    team = build_default_w78_team()
    env = team.run_team_turn(w77_outer_cid="w77_x")
    # Drift the w77_outer_cid → verify should fail.
    ok, fails = verify_w78_handoff(env, team.params, "different")
    assert not ok
    assert "w78_outer_envelope_w77_outer_cid_drift" in fails


def test_w78_stable_boundary():
    """coordpy.__version__ and SDK_VERSION must be unchanged."""
    import coordpy
    assert coordpy.__version__ == "1.2.0"
    assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"
