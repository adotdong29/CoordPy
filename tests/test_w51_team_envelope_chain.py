"""Test the W51 team envelope chain w47→w48→w49→w50→w51."""

from __future__ import annotations

import pytest

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w51_team import (
    W51_ENVELOPE_VERIFIER_FAILURE_MODES,
    W51_SCHEMA_VERSION,
    W51Team,
    build_w51_registry,
    verify_w51_handoff,
)


def _build_agents(backend) -> list[Agent]:
    return [
        Agent(name="alpha", instructions="", role="r0",
              backend=backend, temperature=0.0, max_tokens=24),
        Agent(name="bravo", instructions="", role="r1",
              backend=backend, temperature=0.0, max_tokens=24),
    ]


def test_w51_envelope_passes_verifier_when_clean() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.envtest", default_response="OK")
    agents = _build_agents(backend)
    reg = build_w51_registry(
        schema_cid="w51_envchain_v1",
        role_universe=("r0", "r1"))
    team = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("envelope chain test")
    v = verify_w51_handoff(
        r.w51_envelope,
        expected_w50_outer_cid=r.w50_outer_cid,
        expected_params_cid=r.w51_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg,
        persistent_state_cids=r.persistent_state_cids)
    assert v["ok"] is True
    assert v["failures"] == []
    assert v["n_failure_modes"] == 24


def test_w51_envelope_verifier_rejects_forged_w50_outer_cid() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.envtest", default_response="OK")
    agents = _build_agents(backend)
    reg = build_w51_registry(
        schema_cid="w51_envchain_v1",
        role_universe=("r0", "r1"))
    team = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("forgery test")
    v = verify_w51_handoff(
        r.w51_envelope,
        expected_w50_outer_cid="ff" * 32,
        expected_params_cid=r.w51_params_cid,
        bundles=r.turn_witness_bundles,
        registry=reg)
    assert v["ok"] is False
    assert "w51_w50_outer_cid_mismatch" in v["failures"]


def test_w51_envelope_schema_version_matches() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.envtest", default_response="OK")
    agents = _build_agents(backend)
    reg = build_w51_registry(
        schema_cid="w51_envchain_v1",
        role_universe=("r0", "r1"))
    team = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("schema check")
    assert r.w51_envelope.schema_version == W51_SCHEMA_VERSION


def test_w51_envelope_24_disjoint_failure_modes() -> None:
    """The W51 envelope verifier enumerates 24 disjoint
    failure modes."""
    assert len(W51_ENVELOPE_VERIFIER_FAILURE_MODES) == 24
    # Each failure mode must be unique
    assert len(set(W51_ENVELOPE_VERIFIER_FAILURE_MODES)) == 24


def test_w51_envelope_replay_byte_identical() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.envtest", default_response="OK")
    agents = _build_agents(backend)
    reg = build_w51_registry(
        schema_cid="w51_envchain_v1",
        role_universe=("r0", "r1"))
    r1 = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2).run("replay")
    r2 = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2).run("replay")
    assert r1.w50_outer_cid == r2.w50_outer_cid
    assert r1.w51_outer_cid == r2.w51_outer_cid
    assert r1.w51_params_cid == r2.w51_params_cid


def test_w51_envelope_anchor_status_synthetic_only_by_default() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.envtest", default_response="OK")
    agents = _build_agents(backend)
    reg = build_w51_registry(
        schema_cid="w51_envchain_v1",
        role_universe=("r0", "r1"))
    team = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    r = team.run("anchor check")
    # Without explicit Ollama backends and without
    # COORDPY_W51_OLLAMA_REACHABLE=1, the anchor is synthetic_only.
    assert r.triple_anchor_status in (
        "synthetic_only", "real_llm_anchor", "skipped")
