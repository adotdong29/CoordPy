"""Critical guard test: W54 trivial-passthrough must equal W53
byte-for-byte.

Reproduces the H1 / W54-L-TRIVIAL-W54-PASSTHROUGH hypothesis.
"""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w53_team import (
    W53Team,
    build_trivial_w53_registry,
)
from coordpy.w54_team import (
    W54Team,
    build_trivial_w54_registry,
)


def _build_agents(role: str, backend) -> list[Agent]:
    return [
        Agent(name=f"a_{role}", instructions="",
              role=role, backend=backend,
              temperature=0.0, max_tokens=20),
    ]


def test_w54_trivial_team_w53_outer_cid_equals_w53_team_outer_cid(
) -> None:
    """The critical guard: W54 trivial wraps W53 byte-identically."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w54",
        default_response="hello")
    agents = _build_agents("r0", backend)

    reg53 = build_trivial_w53_registry(
        schema_cid="w54_pt_v1")
    team53 = W53Team(
        agents=agents, backend=backend, registry=reg53,
        max_visible_handoffs=2)
    result53 = team53.run("guard task")

    reg54 = build_trivial_w54_registry(
        schema_cid="w54_pt_v1")
    team54 = W54Team(
        agents=agents, backend=backend, registry=reg54,
        max_visible_handoffs=2)
    result54 = team54.run("guard task")

    assert result54.w53_outer_cid == result53.w53_outer_cid
    assert result54.final_output == result53.final_output


def test_w54_trivial_passthrough_two_runs_byte_identical() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test.w54",
        default_response="abc")
    agents = _build_agents("r0", backend)
    reg54 = build_trivial_w54_registry()
    team1 = W54Team(
        agents=agents, backend=backend, registry=reg54,
        max_visible_handoffs=2)
    r1 = team1.run("two-run guard")
    team2 = W54Team(
        agents=agents, backend=backend, registry=reg54,
        max_visible_handoffs=2)
    r2 = team2.run("two-run guard")
    assert r1.w53_outer_cid == r2.w53_outer_cid
    assert r1.w54_outer_cid == r2.w54_outer_cid
    assert r1.final_output == r2.final_output


def test_w54_trivial_persistent_v6_chain_empty() -> None:
    """Trivial W54 has no persistent V6 chain entries."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w54", default_response="x")
    agents = _build_agents("r0", backend)
    reg54 = build_trivial_w54_registry()
    team = W54Team(
        agents=agents, backend=backend, registry=reg54,
        max_visible_handoffs=2)
    r = team.run("trivial v6 chain check")
    assert len(r.persistent_v6_state_cids) == 0
    assert len(r.mlsc_v2_capsule_cids) == 0


def test_w54_trivial_witness_bundle_cids_all_empty() -> None:
    """Trivial W54 has empty witness CIDs for every flag."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w54", default_response="x")
    agents = _build_agents("r0", backend)
    reg54 = build_trivial_w54_registry()
    team = W54Team(
        agents=agents, backend=backend, registry=reg54,
        max_visible_handoffs=2)
    r = team.run("trivial witness check")
    for b in r.turn_witness_bundles:
        assert b.persistent_v6_witness_cid == ""
        assert b.hex_translator_witness_cid == ""
        assert b.mlsc_v2_witness_cid == ""
        assert b.consensus_controller_witness_cid == ""
        assert b.deep_stack_v5_witness_cid == ""
        assert b.ecc_v6_compression_witness_cid == ""
        assert b.long_horizon_v6_witness_cid == ""
        assert b.crc_v2_witness_cid == ""
        assert b.tvs_arbiter_v3_witness_cid == ""
        assert b.uncertainty_v2_witness_cid == ""
