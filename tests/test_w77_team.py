"""W77 tests — W77Team end-to-end smoke."""

from __future__ import annotations

from coordpy.w77_team import (
    W77_FAILURE_MODES, W77_SCHEMA_VERSION,
    W77HandoffEnvelope, W77Params, W77Team,
    build_default_w77_team, verify_w77_handoff,
)


def test_w77_team_trivial_envelope() -> None:
    p = W77Params.build_trivial()
    team = W77Team(params=p)
    env = team.run_team_turn(w76_outer_cid="x")
    assert env.schema == W77_SCHEMA_VERSION
    assert env.w76_outer_cid == "x"
    ok, fails = verify_w77_handoff(env, p, "x")
    assert ok
    assert fails == []


def test_w77_team_full_envelope_chain() -> None:
    team = build_default_w77_team(seed=77100)
    env = team.run_team_turn(w76_outer_cid="w76-fake")
    assert isinstance(env, W77HandoffEnvelope)
    assert env.substrate_v22_used
    assert env.twenty_two_way_used
    assert env.masc_v13_v22_beats_v21_rate >= 0.5
    assert env.masc_v13_tsc_v22_beats_tsc_v21_rate >= 0.5
    assert len(env.post_restart_replacement_trajectory_cid) > 0
    assert env.hosted_router_v10_chosen in (
        "openrouter_paid", "openai_paid")


def test_w77_team_envelope_content_addressed() -> None:
    team_a = build_default_w77_team(seed=77200)
    team_b = build_default_w77_team(seed=77200)
    env_a = team_a.run_team_turn(w76_outer_cid="z")
    env_b = team_b.run_team_turn(w76_outer_cid="z")
    # Same seed → same envelope CID.
    assert env_a.cid() == env_b.cid()


def test_w77_failure_modes_canonical() -> None:
    assert "w77_outer_envelope_schema_mismatch" in (
        W77_FAILURE_MODES)
    assert "w77_substrate_v22_n_layers_off" in W77_FAILURE_MODES
    assert (
        "w77_handoff_v9_cross_plane_savings_below_86_percent"
        in W77_FAILURE_MODES)
    assert len(W77_FAILURE_MODES) >= 50


def test_w77_no_version_bump() -> None:
    from coordpy import SDK_VERSION, __version__
    assert __version__ == "1.2.0"
    assert SDK_VERSION == "coordpy.sdk.v3.43"


def test_w77_team_carries_forward_w76_outer_cid() -> None:
    team = build_default_w77_team(seed=77300)
    env = team.run_team_turn(w76_outer_cid="w76-content-cid")
    assert env.w76_outer_cid == "w76-content-cid"
