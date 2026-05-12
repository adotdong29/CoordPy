"""W52 envelope chain test: verifier soundness on a full team run."""

from __future__ import annotations

import pytest

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w52_team import (
    W52Team,
    W52_ENVELOPE_VERIFIER_FAILURE_MODES,
    build_w52_registry,
    verify_w52_handoff,
)


def _build_team(seed: int) -> tuple[W52Team, "W52Registry"]:
    from coordpy.w52_team import W52Registry  # local alias
    backend = SyntheticLLMClient(
        model_tag=f"synth.w52t.{seed}",
        default_response="env-chain")
    agents = [
        Agent(name=f"a_{seed}", instructions="",
              role="r0", backend=backend,
              temperature=0.0, max_tokens=20),
    ]
    reg = build_w52_registry(
        schema_cid=f"w52_env_{seed}",
        role_universe=("r0",))
    team = W52Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    return team, reg


def test_w52_envelope_clean_verify_ok() -> None:
    team, reg = _build_team(seed=11)
    r = team.run("env-chain clean")
    v = verify_w52_handoff(
        r.w52_envelope,
        expected_w51_outer_cid=r.w51_outer_cid,
        expected_params_cid=r.w52_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg,
        persistent_v4_state_cids=r.persistent_v4_state_cids)
    assert v["ok"] is True
    assert v["failures"] == []


def test_w52_envelope_verifier_rejects_w51_outer_tamper() -> None:
    team, reg = _build_team(seed=13)
    r = team.run("env-chain tamper w51")
    v = verify_w52_handoff(
        r.w52_envelope,
        expected_w51_outer_cid="ff" * 32,
        expected_params_cid=r.w52_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w52_w51_outer_cid_mismatch" in v["failures"]


def test_w52_envelope_verifier_rejects_params_tamper() -> None:
    team, reg = _build_team(seed=17)
    r = team.run("env-chain tamper params")
    v = verify_w52_handoff(
        r.w52_envelope,
        expected_w51_outer_cid=r.w51_outer_cid,
        expected_params_cid="ff" * 32,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w52_params_cid_mismatch" in v["failures"]


def test_w52_envelope_verifier_has_26_failure_modes() -> None:
    assert len(W52_ENVELOPE_VERIFIER_FAILURE_MODES) == 26


def test_w52_envelope_outer_cid_stable_across_runs() -> None:
    team1, _ = _build_team(seed=19)
    team2, _ = _build_team(seed=19)
    r1 = team1.run("stable run")
    r2 = team2.run("stable run")
    assert r1.w52_outer_cid == r2.w52_outer_cid
    assert r1.w51_outer_cid == r2.w51_outer_cid
