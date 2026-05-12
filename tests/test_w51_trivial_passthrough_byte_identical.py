"""Critical guard test: W51 trivial-passthrough must equal W50
byte-for-byte.

Reproduces the H1 / W51-L-TRIVIAL-W51-PASSTHROUGH hypothesis.
"""

from __future__ import annotations

import pytest

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w50_team import (
    W50Team,
    build_trivial_w50_registry,
)
from coordpy.w51_team import (
    W51Team,
    build_trivial_w51_registry,
)


def _build_agents(role: str, backend) -> list[Agent]:
    return [
        Agent(name=f"a_{role}", instructions="",
              role=role, backend=backend,
              temperature=0.0, max_tokens=20),
    ]


def test_w51_trivial_team_w50_outer_cid_equals_w50_team_outer_cid() -> None:
    """The critical guard: W51 trivial wraps W50 byte-identically."""
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="hello")
    agents = _build_agents("r0", backend)

    reg50 = build_trivial_w50_registry(schema_cid="w51_pt_v1")
    team50 = W50Team(
        agents=agents, backend=backend, registry=reg50,
        max_visible_handoffs=2)
    result50 = team50.run("guard task")

    reg51 = build_trivial_w51_registry(schema_cid="w51_pt_v1")
    team51 = W51Team(
        agents=agents, backend=backend, registry=reg51,
        max_visible_handoffs=2)
    result51 = team51.run("guard task")

    assert result51.w50_outer_cid == result50.w50_outer_cid
    assert result51.final_output == result50.final_output


def test_w51_trivial_passthrough_two_runs_byte_identical() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="abc")
    agents = _build_agents("r0", backend)
    reg51 = build_trivial_w51_registry()
    team1 = W51Team(
        agents=agents, backend=backend, registry=reg51,
        max_visible_handoffs=2)
    r1 = team1.run("two-run guard")
    team2 = W51Team(
        agents=agents, backend=backend, registry=reg51,
        max_visible_handoffs=2)
    r2 = team2.run("two-run guard")
    assert r1.w50_outer_cid == r2.w50_outer_cid
    assert r1.w51_outer_cid == r2.w51_outer_cid
    assert r1.final_output == r2.final_output


def test_w51_trivial_persistent_state_chain_empty() -> None:
    """Trivial W51 has no persistent state chain entries."""
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="x")
    agents = _build_agents("r0", backend)
    reg51 = build_trivial_w51_registry()
    team = W51Team(
        agents=agents, backend=backend, registry=reg51,
        max_visible_handoffs=2)
    r = team.run("trivial state chain check")
    assert len(r.persistent_state_cids) == 0


def test_w51_trivial_witness_bundle_cids_all_empty() -> None:
    """Trivial W51 has empty witness CIDs for every flag."""
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="x")
    agents = _build_agents("r0", backend)
    reg51 = build_trivial_w51_registry()
    team = W51Team(
        agents=agents, backend=backend, registry=reg51,
        max_visible_handoffs=2)
    r = team.run("trivial witness check")
    for b in r.turn_witness_bundles:
        assert b.persistent_state_witness_cid == ""
        assert b.triple_backend_witness_cid == ""
        assert b.deep_stack_v2_forward_witness_cid == ""
        assert b.hierarchical_compression_witness_cid == ""
        assert b.cramming_witness_v3_cid == ""
        assert b.long_horizon_reconstruction_witness_cid == ""
        assert b.branch_cycle_memory_witness_cid == ""
