"""Critical guard test: W53 trivial-passthrough must equal W52
byte-for-byte.

Reproduces the H1 / W53-L-TRIVIAL-W53-PASSTHROUGH hypothesis.
"""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w52_team import (
    W52Team,
    build_trivial_w52_registry,
)
from coordpy.w53_team import (
    W53Team,
    build_trivial_w53_registry,
)


def _build_agents(role: str, backend) -> list[Agent]:
    return [
        Agent(name=f"a_{role}", instructions="",
              role=role, backend=backend,
              temperature=0.0, max_tokens=20),
    ]


def test_w53_trivial_team_w52_outer_cid_equals_w52_team_outer_cid(
) -> None:
    """The critical guard: W53 trivial wraps W52 byte-identically."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w53",
        default_response="hello")
    agents = _build_agents("r0", backend)

    reg52 = build_trivial_w52_registry(
        schema_cid="w53_pt_v1")
    team52 = W52Team(
        agents=agents, backend=backend, registry=reg52,
        max_visible_handoffs=2)
    result52 = team52.run("guard task")

    reg53 = build_trivial_w53_registry(
        schema_cid="w53_pt_v1")
    team53 = W53Team(
        agents=agents, backend=backend, registry=reg53,
        max_visible_handoffs=2)
    result53 = team53.run("guard task")

    assert result53.w52_outer_cid == result52.w52_outer_cid
    assert result53.final_output == result52.final_output


def test_w53_trivial_passthrough_two_runs_byte_identical() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test.w53",
        default_response="abc")
    agents = _build_agents("r0", backend)
    reg53 = build_trivial_w53_registry()
    team1 = W53Team(
        agents=agents, backend=backend, registry=reg53,
        max_visible_handoffs=2)
    r1 = team1.run("two-run guard")
    team2 = W53Team(
        agents=agents, backend=backend, registry=reg53,
        max_visible_handoffs=2)
    r2 = team2.run("two-run guard")
    assert r1.w52_outer_cid == r2.w52_outer_cid
    assert r1.w53_outer_cid == r2.w53_outer_cid
    assert r1.final_output == r2.final_output


def test_w53_trivial_persistent_v5_chain_empty() -> None:
    """Trivial W53 has no persistent V5 chain entries."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w53", default_response="x")
    agents = _build_agents("r0", backend)
    reg53 = build_trivial_w53_registry()
    team = W53Team(
        agents=agents, backend=backend, registry=reg53,
        max_visible_handoffs=2)
    r = team.run("trivial v5 chain check")
    assert len(r.persistent_v5_state_cids) == 0
    assert len(r.mlsc_capsule_cids) == 0


def test_w53_trivial_witness_bundle_cids_all_empty() -> None:
    """Trivial W53 has empty witness CIDs for every flag."""
    backend = SyntheticLLMClient(
        model_tag="synth.test.w53", default_response="x")
    agents = _build_agents("r0", backend)
    reg53 = build_trivial_w53_registry()
    team = W53Team(
        agents=agents, backend=backend, registry=reg53,
        max_visible_handoffs=2)
    r = team.run("trivial witness check")
    for b in r.turn_witness_bundles:
        assert b.persistent_v5_witness_cid == ""
        assert b.quint_translator_witness_cid == ""
        assert b.mlsc_witness_cid == ""
        assert b.deep_stack_v4_forward_witness_cid == ""
        assert b.ecc_compression_witness_cid == ""
        assert b.long_horizon_v5_witness_cid == ""
        assert b.branch_merge_memory_v3_witness_cid == ""
        assert b.corruption_robust_carrier_witness_cid == ""
        assert (
            b.transcript_vs_shared_arbiter_v2_witness_cid
            == "")
        assert b.uncertainty_layer_witness_cid == ""
