"""W56 envelope chain test: verifier soundness on a full team run."""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w56_team import (
    W56Team,
    W56_ENVELOPE_VERIFIER_FAILURE_MODES,
    build_w56_registry,
    verify_w56_handoff,
)


def _build_team(seed: int):
    backend = SyntheticLLMClient(
        model_tag=f"synth.w56t.{seed}",
        default_response="env-chain")
    agents = [
        Agent(name=f"a_{seed}", instructions="",
              role="r0", backend=backend,
              temperature=0.0, max_tokens=20),
    ]
    reg = build_w56_registry(
        schema_cid=f"w56_env_{seed}",
        role_universe=("r0",))
    team = W56Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    return team, reg


def test_w56_envelope_clean_verify_ok() -> None:
    team, reg = _build_team(seed=11)
    r = team.run("env-chain clean")
    v = verify_w56_handoff(
        r.w56_envelope,
        expected_w55_outer_cid=r.w55_outer_cid,
        expected_params_cid=r.w56_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg,
        persistent_v8_state_cids=r.persistent_v8_state_cids)
    assert v["ok"] is True
    assert v["failures"] == []


def test_w56_envelope_verifier_rejects_w55_outer_tamper() -> None:
    team, reg = _build_team(seed=13)
    r = team.run("env-chain tamper w55")
    v = verify_w56_handoff(
        r.w56_envelope,
        expected_w55_outer_cid="ff" * 32,
        expected_params_cid=r.w56_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w56_w55_outer_cid_mismatch" in v["failures"]


def test_w56_envelope_verifier_rejects_params_tamper() -> None:
    team, reg = _build_team(seed=17)
    r = team.run("env-chain tamper params")
    v = verify_w56_handoff(
        r.w56_envelope,
        expected_w55_outer_cid=r.w55_outer_cid,
        expected_params_cid="ff" * 32,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w56_params_cid_mismatch" in v["failures"]


def test_w56_envelope_verifier_has_38_failure_modes() -> None:
    assert len(W56_ENVELOPE_VERIFIER_FAILURE_MODES) == 38


def test_w56_envelope_outer_cid_stable_across_runs() -> None:
    team1, _ = _build_team(seed=19)
    team2, _ = _build_team(seed=19)
    r1 = team1.run("stable run")
    r2 = team2.run("stable run")
    assert r1.w56_outer_cid == r2.w56_outer_cid
    assert r1.w55_outer_cid == r2.w55_outer_cid


def test_w56_envelope_persistent_v8_state_count_matches() -> None:
    team, reg = _build_team(seed=29)
    r = team.run("v8 chain")
    assert (
        len(r.persistent_v8_state_cids) == r.n_turns)


def test_w56_envelope_substrate_adapter_matrix_cid_nonempty() -> None:
    team, reg = _build_team(seed=31)
    r = team.run("adapter")
    assert len(r.w56_envelope.substrate_adapter_matrix_cid) == 64


def test_w56_envelope_substrate_used_flag() -> None:
    team, reg = _build_team(seed=37)
    r = team.run("substrate")
    assert r.substrate_used is True
    assert r.w56_envelope.substrate_used is True


def test_w56_envelope_composite_confidence_in_bounds() -> None:
    team, reg = _build_team(seed=41)
    r = team.run("composite bound")
    assert 0.0 <= r.composite_confidence_v4_mean <= 1.0
