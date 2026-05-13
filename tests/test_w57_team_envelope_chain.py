"""W57 envelope chain test: verifier soundness on a full team run."""

from __future__ import annotations

import hashlib

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w57_team import (
    W57Team,
    W57_ENVELOPE_VERIFIER_FAILURE_MODES,
    build_w57_registry,
    verify_w57_handoff,
)


def _build_team(seed: int):
    backend = SyntheticLLMClient(
        model_tag=f"synth.w57t.{seed}",
        default_response="env-chain")
    agents = [
        Agent(name=f"a_{seed}", instructions="",
              role="r0", backend=backend,
              temperature=0.0, max_tokens=20),
    ]
    sc = hashlib.sha256(
        f"w57_env_{seed}".encode("utf-8")).hexdigest()
    reg = build_w57_registry(
        schema_cid=sc, role_universe=("r0",))
    team = W57Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    return team, reg


def test_w57_envelope_clean_verify_ok() -> None:
    team, reg = _build_team(seed=11)
    r = team.run("env-chain clean")
    v = verify_w57_handoff(
        r.w57_envelope,
        expected_w56_outer_cid=r.w56_outer_cid,
        expected_params_cid=r.w57_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg,
        persistent_v9_state_cids=r.persistent_v9_state_cids)
    assert v["ok"] is True
    assert v["failures"] == []


def test_w57_envelope_verifier_rejects_w56_outer_tamper() -> None:
    team, reg = _build_team(seed=13)
    r = team.run("env-chain tamper w56")
    v = verify_w57_handoff(
        r.w57_envelope,
        expected_w56_outer_cid="ff" * 32,
        expected_params_cid=r.w57_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w57_w56_outer_cid_mismatch" in v["failures"]


def test_w57_envelope_verifier_rejects_params_tamper() -> None:
    team, reg = _build_team(seed=17)
    r = team.run("env-chain tamper params")
    v = verify_w57_handoff(
        r.w57_envelope,
        expected_w56_outer_cid=r.w56_outer_cid,
        expected_params_cid="ff" * 32,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w57_params_cid_mismatch" in v["failures"]


def test_w57_envelope_verifier_has_at_least_40_failure_modes() -> None:
    assert len(W57_ENVELOPE_VERIFIER_FAILURE_MODES) >= 40


def test_w57_envelope_outer_cid_stable_across_runs() -> None:
    team1, _ = _build_team(seed=19)
    team2, _ = _build_team(seed=19)
    r1 = team1.run("stable run")
    r2 = team2.run("stable run")
    assert r1.w57_outer_cid == r2.w57_outer_cid


def test_w57_envelope_substrate_v2_used_flag_true() -> None:
    team, _ = _build_team(seed=23)
    r = team.run("substrate v2 flag")
    assert r.substrate_v2_used is True


def test_w57_envelope_bidirectional_used_flag_true() -> None:
    team, _ = _build_team(seed=29)
    r = team.run("bidirectional flag")
    assert r.bidirectional_used is True
