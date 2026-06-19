"""W76 tests — W76Team end-to-end smoke."""

from __future__ import annotations

from coordpy.w76_team import (
    W76_FAILURE_MODES, W76_SCHEMA_VERSION,
    W76HandoffEnvelope, W76Params, W76Team,
    build_default_w76_team, verify_w76_handoff,
)


def test_w76_team_trivial_envelope() -> None:
    p = W76Params.build_trivial()
    team = W76Team(params=p)
    env = team.run_team_turn(w75_outer_cid="x")
    assert env.schema == W76_SCHEMA_VERSION
    assert env.w75_outer_cid == "x"
    ok, fails = verify_w76_handoff(env, p, "x")
    assert ok
    assert fails == []


def test_w76_team_full_envelope_chain() -> None:
    team = build_default_w76_team(seed=76100)
    env = team.run_team_turn(w75_outer_cid="w75-fake")
    assert isinstance(env, W76HandoffEnvelope)
    assert env.substrate_v21_used
    assert env.twenty_one_way_used
    assert env.masc_v12_v21_beats_v20_rate >= 0.5
    assert env.masc_v12_tsc_v21_beats_tsc_v20_rate >= 0.5
    assert len(env.chain_then_restart_trajectory_cid) > 0
    assert env.hosted_router_v9_chosen in (
        "openrouter_paid", "openai_paid")


def test_w76_team_envelope_content_addressed() -> None:
    team_a = build_default_w76_team(seed=76200)
    team_b = build_default_w76_team(seed=76200)
    env_a = team_a.run_team_turn(w75_outer_cid="z")
    env_b = team_b.run_team_turn(w75_outer_cid="z")
    # Same seed → same envelope CID.
    assert env_a.cid() == env_b.cid()


def test_w76_failure_modes_canonical() -> None:
    assert "w76_outer_envelope_schema_mismatch" in (
        W76_FAILURE_MODES)
    assert "w76_substrate_v21_n_layers_off" in W76_FAILURE_MODES
    assert (
        "w76_handoff_v8_cross_plane_savings_below_85_percent"
        in W76_FAILURE_MODES)
    assert len(W76_FAILURE_MODES) >= 50


def test_w76_no_version_bump() -> None:
    from coordpy import SDK_VERSION, __version__
    assert __version__ == "1.2.1"
    assert SDK_VERSION == "coordpy.sdk.v3.43"


def test_w76_team_carries_forward_w75_outer_cid() -> None:
    team = build_default_w76_team(seed=76300)
    env = team.run_team_turn(w75_outer_cid="w75-content-cid")
    assert env.w75_outer_cid == "w75-content-cid"
