"""Tests for the W50 team envelope chain + verifier."""

from __future__ import annotations

import pytest

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w50_team import (
    W50_ENVELOPE_VERIFIER_FAILURE_MODES,
    W50_SCHEMA_VERSION,
    W50HandoffEnvelope,
    W50Params,
    W50Team,
    W50TurnWitnessBundle,
    build_trivial_w50_registry,
    build_w50_registry,
    verify_w50_handoff,
)


def _build_agents(role: str, backend) -> list[Agent]:
    return [
        Agent(name=f"a_{role}", instructions="", role=role,
              backend=backend, temperature=0.0, max_tokens=20),
    ]


def test_w50_default_params_builds_full_surfaces() -> None:
    p = W50Params.build_default(role_universe=("r0", "r1"))
    assert p.cross_backend_params is not None
    assert p.deep_proxy_stack is not None
    assert p.adaptive_codebook is not None
    assert p.adaptive_gate is not None
    assert p.cross_bank_transfer is not None
    assert p.eviction_v2 is not None
    assert p.role_reuse_map is not None
    assert p.reconstruction_v2_head is not None
    assert p.cross_backend_enabled is True
    assert p.deep_stack_enabled is True
    assert p.adaptive_compression_enabled is True
    assert p.cross_bank_transfer_enabled is True
    assert p.shared_latent_carrier_v2_enabled is True


def test_w50_trivial_params_all_none_and_disabled() -> None:
    p = W50Params.build_trivial()
    assert p.cross_backend_params is None
    assert p.deep_proxy_stack is None
    assert p.adaptive_codebook is None
    assert p.adaptive_gate is None
    assert p.cross_bank_transfer is None
    assert p.eviction_v2 is None
    assert p.role_reuse_map is None
    assert p.reconstruction_v2_head is None
    assert p.all_flags_disabled is True


def test_w50_params_cid_deterministic() -> None:
    p1 = W50Params.build_default(seed=1)
    p2 = W50Params.build_default(seed=1)
    assert p1.cid() == p2.cid()


def test_w50_team_default_run_produces_envelope() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="hello")
    agents = _build_agents("r0", backend)
    reg = build_w50_registry(
        schema_cid="schema_v1", role_universe=("r0",))
    team = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("task")
    assert r.w50_outer_cid != ""
    assert r.w49_root_cid != ""
    assert r.n_turns >= 1
    assert r.anchor_status == "synthetic_only"
    assert len(r.turn_witness_bundles) == r.n_turns


def test_w50_verifier_n_failure_modes_is_20() -> None:
    """H10 cumulative: 20 disjoint failure modes at W50."""
    assert len(W50_ENVELOPE_VERIFIER_FAILURE_MODES) == 20


def test_w50_verify_passes_on_clean_envelope() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="x")
    agents = _build_agents("r0", backend)
    reg = build_w50_registry(
        schema_cid="schema_v2", role_universe=("r0",))
    team = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("task")
    v = verify_w50_handoff(
        r.w50_envelope,
        expected_w49_root_cid=r.w49_root_cid,
        expected_params_cid=r.w50_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is True


def test_w50_verify_detects_w49_root_cid_tamper() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="x")
    agents = _build_agents("r0", backend)
    reg = build_w50_registry(
        schema_cid="schema_v3", role_universe=("r0",))
    team = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("task")
    v = verify_w50_handoff(
        r.w50_envelope,
        expected_w49_root_cid="00" * 32,
        expected_params_cid=r.w50_params_cid)
    assert v["ok"] is False
    assert "w50_w49_root_cid_mismatch" in v["failures"]


def test_w50_verify_detects_params_cid_tamper() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="x")
    agents = _build_agents("r0", backend)
    reg = build_w50_registry(
        schema_cid="schema_v4", role_universe=("r0",))
    team = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("task")
    v = verify_w50_handoff(
        r.w50_envelope,
        expected_w49_root_cid=r.w49_root_cid,
        expected_params_cid="ff" * 32)
    assert v["ok"] is False
    assert "w50_params_cid_mismatch" in v["failures"]


def test_w50_verify_detects_bundle_count_mismatch() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="x")
    agents = _build_agents("r0", backend)
    reg = build_w50_registry(
        schema_cid="schema_v5", role_universe=("r0",))
    team = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("task")
    # Pass an empty bundle list — count mismatch
    v = verify_w50_handoff(
        r.w50_envelope, bundles=())
    assert v["ok"] is False
    assert "w50_per_turn_bundle_count_mismatch" in v["failures"]


def test_w50_envelope_outer_cid_changes_when_anchor_changes() -> None:
    """Tamper detection: changing anchor_status changes outer CID."""
    e1 = W50HandoffEnvelope(
        schema_version=W50_SCHEMA_VERSION,
        w49_root_cid="abc",
        params_cid="def",
        turn_witness_bundle_cid="ghi",
        w49_envelope_count=1,
        w50_carrier_chain_cid="jkl",
        realism_anchor_payload_cid="mno",
        anchor_status="synthetic_only")
    e2 = W50HandoffEnvelope(
        schema_version=W50_SCHEMA_VERSION,
        w49_root_cid="abc",
        params_cid="def",
        turn_witness_bundle_cid="ghi",
        w49_envelope_count=1,
        w50_carrier_chain_cid="jkl",
        realism_anchor_payload_cid="mno",
        anchor_status="real_llm_anchor")
    assert e1.cid() != e2.cid()


def test_w50_verify_detects_schema_mismatch() -> None:
    e = W50HandoffEnvelope(
        schema_version="bogus.v1",
        w49_root_cid="x", params_cid="y",
        turn_witness_bundle_cid="z", w49_envelope_count=1,
        w50_carrier_chain_cid="a",
        realism_anchor_payload_cid="b",
        anchor_status="synthetic_only")
    v = verify_w50_handoff(e)
    assert v["ok"] is False
    assert "w50_schema_mismatch" in v["failures"]


def test_w50_verify_detects_anchor_status_invalid() -> None:
    e = W50HandoffEnvelope(
        schema_version=W50_SCHEMA_VERSION,
        w49_root_cid="x", params_cid="y",
        turn_witness_bundle_cid="z", w49_envelope_count=1,
        w50_carrier_chain_cid="a",
        realism_anchor_payload_cid="b",
        anchor_status="BOGUS_STATUS_XYZ")
    v = verify_w50_handoff(e)
    assert v["ok"] is False
    assert "w50_anchor_status_invalid" in v["failures"]


def test_w50_team_two_runs_byte_identical() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="z")
    agents = _build_agents("r0", backend)
    reg = build_w50_registry(
        schema_cid="schema_repl", role_universe=("r0",))
    team1 = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r1 = team1.run("repl_task")
    team2 = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r2 = team2.run("repl_task")
    assert r1.w49_root_cid == r2.w49_root_cid
    assert r1.w50_outer_cid == r2.w50_outer_cid


def test_w50_chain_walk_depth_grows_with_turns() -> None:
    # Use a 2-agent team to get >1 turn
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="hi")
    agents = [
        Agent(name="a1", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=20),
        Agent(name="a2", instructions="", role="r1",
              backend=backend, temperature=0.0, max_tokens=20),
    ]
    reg = build_w50_registry(
        schema_cid="schema_chain", role_universe=("r0", "r1"))
    team = W50Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("chain task")
    # Each turn produces a carrier; chain grows.
    assert len(r.carrier_chain_cids) == r.n_turns
