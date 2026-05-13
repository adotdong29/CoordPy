"""W55 envelope chain test: verifier soundness on a full team run."""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w55_team import (
    W55Team,
    W55_ENVELOPE_VERIFIER_FAILURE_MODES,
    build_w55_registry,
    verify_w55_handoff,
)


def _build_team(seed: int):
    backend = SyntheticLLMClient(
        model_tag=f"synth.w55t.{seed}",
        default_response="env-chain")
    agents = [
        Agent(name=f"a_{seed}", instructions="",
              role="r0", backend=backend,
              temperature=0.0, max_tokens=20),
    ]
    reg = build_w55_registry(
        schema_cid=f"w55_env_{seed}",
        role_universe=("r0",))
    team = W55Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    return team, reg


def test_w55_envelope_clean_verify_ok() -> None:
    team, reg = _build_team(seed=11)
    r = team.run("env-chain clean")
    v = verify_w55_handoff(
        r.w55_envelope,
        expected_w54_outer_cid=r.w54_outer_cid,
        expected_params_cid=r.w55_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg,
        persistent_v7_state_cids=r.persistent_v7_state_cids)
    assert v["ok"] is True
    assert v["failures"] == []


def test_w55_envelope_verifier_rejects_w54_outer_tamper() -> None:
    team, reg = _build_team(seed=13)
    r = team.run("env-chain tamper w54")
    v = verify_w55_handoff(
        r.w55_envelope,
        expected_w54_outer_cid="ff" * 32,
        expected_params_cid=r.w55_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w55_w54_outer_cid_mismatch" in v["failures"]


def test_w55_envelope_verifier_rejects_params_tamper() -> None:
    team, reg = _build_team(seed=17)
    r = team.run("env-chain tamper params")
    v = verify_w55_handoff(
        r.w55_envelope,
        expected_w54_outer_cid=r.w54_outer_cid,
        expected_params_cid="ff" * 32,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w55_params_cid_mismatch" in v["failures"]


def test_w55_envelope_verifier_has_33_failure_modes() -> None:
    assert len(W55_ENVELOPE_VERIFIER_FAILURE_MODES) == 33


def test_w55_envelope_outer_cid_stable_across_runs() -> None:
    team1, _ = _build_team(seed=19)
    team2, _ = _build_team(seed=19)
    r1 = team1.run("stable run")
    r2 = team2.run("stable run")
    assert r1.w55_outer_cid == r2.w55_outer_cid
    assert r1.w54_outer_cid == r2.w54_outer_cid


def test_w55_envelope_composite_confidence_in_bounds() -> None:
    team, reg = _build_team(seed=23)
    r = team.run("composite bound")
    assert 0.0 <= r.composite_confidence_mean_v3 <= 1.0
    assert 0.0 <= r.trust_weighted_composite_mean <= 1.0
    assert 0.0 <= r.twcc_quorum_rate <= 1.0


def test_w55_envelope_persistent_v7_state_count_matches() -> None:
    team, reg = _build_team(seed=29)
    r = team.run("v7 chain")
    assert (
        len(r.persistent_v7_state_cids) == r.n_turns)


def test_w55_envelope_mlsc_v3_audit_trail_cid_nonempty() -> None:
    team, reg = _build_team(seed=31)
    r = team.run("mlsc v3 audit")
    assert (
        len(r.w55_envelope.mlsc_v3_audit_trail_cid) == 64)


def test_w55_envelope_twcc_audit_cid_nonempty() -> None:
    team, reg = _build_team(seed=37)
    r = team.run("twcc audit")
    assert (
        len(r.w55_envelope.twcc_audit_trail_cid) == 64)


def test_w55_envelope_algebra_trace_cid_nonempty() -> None:
    team, reg = _build_team(seed=41)
    r = team.run("algebra trace")
    assert (
        len(r.w55_envelope.disagreement_algebra_trace_cid)
        == 64)
