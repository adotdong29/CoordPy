"""Critical guard test: W55 trivial-passthrough must equal W54
byte-for-byte.

Reproduces the H1 / W55-L-TRIVIAL-W55-PASSTHROUGH hypothesis.
"""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w54_team import (
    W54Team,
    build_trivial_w54_registry,
)
from coordpy.w55_team import (
    W55Team,
    build_trivial_w55_registry,
)


def _build_agents(role: str, backend) -> list[Agent]:
    return [
        Agent(name=f"a_{role}", instructions="",
              role=role, backend=backend,
              temperature=0.0, max_tokens=20),
    ]


def test_w55_trivial_team_w54_outer_cid_equals_w54_team_outer_cid(
) -> None:
    """W55 trivial wraps W54 byte-identically."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w55",
        default_response="hello")
    agents = _build_agents("r0", backend)

    reg54 = build_trivial_w54_registry(
        schema_cid="w55_pt_v1")
    team54 = W54Team(
        agents=agents, backend=backend, registry=reg54,
        max_visible_handoffs=2)
    result54 = team54.run("guard task")

    reg55 = build_trivial_w55_registry(
        schema_cid="w55_pt_v1")
    team55 = W55Team(
        agents=agents, backend=backend, registry=reg55,
        max_visible_handoffs=2)
    result55 = team55.run("guard task")

    assert result55.w54_outer_cid == result54.w54_outer_cid
    assert result55.final_output == result54.final_output


def test_w55_trivial_passthrough_two_runs_byte_identical() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test.w55",
        default_response="abc")
    agents = _build_agents("r0", backend)
    reg55 = build_trivial_w55_registry()
    team1 = W55Team(
        agents=agents, backend=backend, registry=reg55,
        max_visible_handoffs=2)
    r1 = team1.run("two-run guard")
    team2 = W55Team(
        agents=agents, backend=backend, registry=reg55,
        max_visible_handoffs=2)
    r2 = team2.run("two-run guard")
    assert r1.w54_outer_cid == r2.w54_outer_cid
    assert r1.w55_outer_cid == r2.w55_outer_cid
    assert r1.final_output == r2.final_output


def test_w55_trivial_persistent_v7_chain_empty() -> None:
    """Trivial W55 has no persistent V7 chain entries."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w55", default_response="x")
    agents = _build_agents("r0", backend)
    reg55 = build_trivial_w55_registry()
    team = W55Team(
        agents=agents, backend=backend, registry=reg55,
        max_visible_handoffs=2)
    r = team.run("trivial v7 chain check")
    assert len(r.persistent_v7_state_cids) == 0
    assert len(r.mlsc_v3_capsule_cids) == 0


def test_w55_trivial_witness_bundle_cids_all_empty() -> None:
    """Trivial W55 has empty witness CIDs for every flag."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w55", default_response="x")
    agents = _build_agents("r0", backend)
    reg55 = build_trivial_w55_registry()
    team = W55Team(
        agents=agents, backend=backend, registry=reg55,
        max_visible_handoffs=2)
    r = team.run("trivial witness check")
    for b in r.turn_witness_bundles:
        assert b.persistent_v7_witness_cid == ""
        assert b.hept_translator_witness_cid == ""
        assert b.mlsc_v3_witness_cid == ""
        assert b.twcc_witness_cid == ""
        assert b.deep_stack_v6_witness_cid == ""
        assert b.ecc_v7_compression_witness_cid == ""
        assert b.long_horizon_v7_witness_cid == ""
        assert b.crc_v3_witness_cid == ""
        assert b.tvs_arbiter_v4_witness_cid == ""
        assert b.uncertainty_v3_witness_cid == ""
        assert b.disagreement_algebra_witness_cid == ""
