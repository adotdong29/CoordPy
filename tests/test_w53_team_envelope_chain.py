"""W53 envelope chain test: verifier soundness on a full team run."""

from __future__ import annotations

import pytest

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w53_team import (
    W53Team,
    W53_ENVELOPE_VERIFIER_FAILURE_MODES,
    build_w53_registry,
    verify_w53_handoff,
)


def _build_team(seed: int):
    backend = SyntheticLLMClient(
        model_tag=f"synth.w53t.{seed}",
        default_response="env-chain")
    agents = [
        Agent(name=f"a_{seed}", instructions="",
              role="r0", backend=backend,
              temperature=0.0, max_tokens=20),
    ]
    reg = build_w53_registry(
        schema_cid=f"w53_env_{seed}",
        role_universe=("r0",))
    team = W53Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    return team, reg


def test_w53_envelope_clean_verify_ok() -> None:
    team, reg = _build_team(seed=11)
    r = team.run("env-chain clean")
    v = verify_w53_handoff(
        r.w53_envelope,
        expected_w52_outer_cid=r.w52_outer_cid,
        expected_params_cid=r.w53_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg,
        persistent_v5_state_cids=r.persistent_v5_state_cids)
    assert v["ok"] is True
    assert v["failures"] == []


def test_w53_envelope_verifier_rejects_w52_outer_tamper() -> None:
    team, reg = _build_team(seed=13)
    r = team.run("env-chain tamper w52")
    v = verify_w53_handoff(
        r.w53_envelope,
        expected_w52_outer_cid="ff" * 32,
        expected_params_cid=r.w53_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w53_w52_outer_cid_mismatch" in v["failures"]


def test_w53_envelope_verifier_rejects_params_tamper() -> None:
    team, reg = _build_team(seed=17)
    r = team.run("env-chain tamper params")
    v = verify_w53_handoff(
        r.w53_envelope,
        expected_w52_outer_cid=r.w52_outer_cid,
        expected_params_cid="ff" * 32,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w53_params_cid_mismatch" in v["failures"]


def test_w53_envelope_verifier_has_29_failure_modes() -> None:
    assert len(W53_ENVELOPE_VERIFIER_FAILURE_MODES) == 29


def test_w53_envelope_outer_cid_stable_across_runs() -> None:
    team1, _ = _build_team(seed=19)
    team2, _ = _build_team(seed=19)
    r1 = team1.run("stable run")
    r2 = team2.run("stable run")
    assert r1.w53_outer_cid == r2.w53_outer_cid
    assert r1.w52_outer_cid == r2.w52_outer_cid


def test_w53_envelope_composite_confidence_in_bounds() -> None:
    team, reg = _build_team(seed=23)
    r = team.run("composite bound")
    assert 0.0 <= r.composite_confidence_mean <= 1.0
    assert 0.0 <= r.arbiter_pick_rate_shared_mean <= 1.0


def test_w53_envelope_persistent_v5_state_count_matches() -> None:
    team, reg = _build_team(seed=29)
    r = team.run("v5 chain")
    # When persistent_v5 is enabled and team produces N turns,
    # we should have N persistent_v5 states.
    assert (
        len(r.persistent_v5_state_cids) == r.n_turns)


def test_w53_envelope_mlsc_audit_trail_cid_nonempty() -> None:
    team, reg = _build_team(seed=31)
    r = team.run("mlsc audit")
    # Even with default trivial-ish content, the audit trail
    # CID must be a valid sha256 hex.
    assert (
        len(r.w53_envelope.mlsc_audit_trail_cid) == 64)
