"""W75 tests — W75Team end-to-end smoke."""

from __future__ import annotations

from coordpy.w75_team import (
    W75_FAILURE_MODES, W75_SCHEMA_VERSION,
    W75HandoffEnvelope, W75Params, W75Team,
    build_default_w75_team, verify_w75_handoff,
)


def test_w75_team_trivial_envelope() -> None:
    p = W75Params.build_trivial()
    team = W75Team(params=p)
    env = team.run_team_turn(w74_outer_cid="x")
    assert env.schema == W75_SCHEMA_VERSION
    assert env.w74_outer_cid == "x"
    ok, fails = verify_w75_handoff(env, p, "x")
    assert ok
    assert fails == []


def test_w75_team_full_envelope_chain() -> None:
    team = build_default_w75_team(seed=75100)
    env = team.run_team_turn(w74_outer_cid="w74-fake")
    assert isinstance(env, W75HandoffEnvelope)
    assert env.substrate_v20_used
    assert env.twenty_way_used
    assert env.masc_v11_v20_beats_v19_rate >= 0.5
    assert env.masc_v11_tsc_v20_beats_tsc_v19_rate >= 0.5
    assert len(env.compound_chain_repair_trajectory_cid) > 0
    assert env.hosted_router_v8_chosen in (
        "openrouter_paid", "openai_paid")


def test_w75_team_envelope_is_content_addressed() -> None:
    team_a = build_default_w75_team(seed=75200)
    team_b = build_default_w75_team(seed=75200)
    env_a = team_a.run_team_turn(w74_outer_cid="z")
    env_b = team_b.run_team_turn(w74_outer_cid="z")
    # Same seed → same envelope CID.
    assert env_a.cid() == env_b.cid()


def test_w75_failure_modes_has_known_canonical_set() -> None:
    assert "w75_outer_envelope_schema_mismatch" in (
        W75_FAILURE_MODES)
    assert "w75_substrate_v20_n_layers_off" in W75_FAILURE_MODES
    assert (
        "w75_handoff_v7_cross_plane_savings_below_84_percent"
        in W75_FAILURE_MODES)
    assert len(W75_FAILURE_MODES) >= 50


def test_w75_no_version_bump() -> None:
    from coordpy import SDK_VERSION, __version__
    assert __version__ == "1.2.0"
    assert SDK_VERSION == "coordpy.sdk.v3.43"
