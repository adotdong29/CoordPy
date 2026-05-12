"""Critical guard test: W52 trivial-passthrough must equal W51
byte-for-byte.

Reproduces the H1 / W52-L-TRIVIAL-W52-PASSTHROUGH hypothesis.
"""

from __future__ import annotations

import pytest

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w51_team import (
    W51Team,
    build_trivial_w51_registry,
)
from coordpy.w52_team import (
    W52Team,
    build_trivial_w52_registry,
)


def _build_agents(role: str, backend) -> list[Agent]:
    return [
        Agent(name=f"a_{role}", instructions="",
              role=role, backend=backend,
              temperature=0.0, max_tokens=20),
    ]


def test_w52_trivial_team_w51_outer_cid_equals_w51_team_outer_cid() -> None:
    """The critical guard: W52 trivial wraps W51 byte-identically."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w52", default_response="hello")
    agents = _build_agents("r0", backend)

    reg51 = build_trivial_w51_registry(schema_cid="w52_pt_v1")
    team51 = W51Team(
        agents=agents, backend=backend, registry=reg51,
        max_visible_handoffs=2)
    result51 = team51.run("guard task")

    reg52 = build_trivial_w52_registry(schema_cid="w52_pt_v1")
    team52 = W52Team(
        agents=agents, backend=backend, registry=reg52,
        max_visible_handoffs=2)
    result52 = team52.run("guard task")

    assert result52.w51_outer_cid == result51.w51_outer_cid
    assert result52.final_output == result51.final_output


def test_w52_trivial_passthrough_two_runs_byte_identical() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test.w52", default_response="abc")
    agents = _build_agents("r0", backend)
    reg52 = build_trivial_w52_registry()
    team1 = W52Team(
        agents=agents, backend=backend, registry=reg52,
        max_visible_handoffs=2)
    r1 = team1.run("two-run guard")
    team2 = W52Team(
        agents=agents, backend=backend, registry=reg52,
        max_visible_handoffs=2)
    r2 = team2.run("two-run guard")
    assert r1.w51_outer_cid == r2.w51_outer_cid
    assert r1.w52_outer_cid == r2.w52_outer_cid
    assert r1.final_output == r2.final_output


def test_w52_trivial_persistent_v4_chain_empty() -> None:
    """Trivial W52 has no persistent V4 chain entries."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w52", default_response="x")
    agents = _build_agents("r0", backend)
    reg52 = build_trivial_w52_registry()
    team = W52Team(
        agents=agents, backend=backend, registry=reg52,
        max_visible_handoffs=2)
    r = team.run("trivial v4 chain check")
    assert len(r.persistent_v4_state_cids) == 0


def test_w52_trivial_witness_bundle_cids_all_empty() -> None:
    """Trivial W52 has empty witness CIDs for every flag."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w52", default_response="x")
    agents = _build_agents("r0", backend)
    reg52 = build_trivial_w52_registry()
    team = W52Team(
        agents=agents, backend=backend, registry=reg52,
        max_visible_handoffs=2)
    r = team.run("trivial witness check")
    for b in r.turn_witness_bundles:
        assert b.persistent_v4_witness_cid == ""
        assert b.multi_hop_witness_cid == ""
        assert b.deep_stack_v3_forward_witness_cid == ""
        assert b.quantised_compression_witness_cid == ""
        assert b.cramming_witness_v4_cid == ""
        assert b.long_horizon_v4_witness_cid == ""
        assert b.branch_cycle_memory_v2_witness_cid == ""
        assert b.role_graph_witness_cid == ""
        assert b.transcript_vs_shared_witness_cid == ""
