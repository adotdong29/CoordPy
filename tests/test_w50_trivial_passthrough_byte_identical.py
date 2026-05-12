"""Critical guard test: W50 trivial-passthrough must equal W49
byte-for-byte.

Reproduces the H1 / W50-L-TRIVIAL-W50-PASSTHROUGH hypothesis.
"""

from __future__ import annotations

import pytest

from coordpy.agents import Agent
from coordpy.multi_block_proxy import (
    MultiBlockProxyTeam,
    build_trivial_multi_block_proxy_registry,
)
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w50_team import (
    W50Team,
    build_trivial_w50_registry,
)


def _build_agents(role: str, backend) -> list[Agent]:
    return [
        Agent(name=f"a_{role}", instructions="",
              role=role, backend=backend,
              temperature=0.0, max_tokens=20),
    ]


def test_w50_trivial_team_w49_root_cid_equals_w49_team_root_cid() -> None:
    """The critical guard: W50 trivial wraps W49 byte-identically."""
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="hello")
    agents = _build_agents("r0", backend)

    reg49 = build_trivial_multi_block_proxy_registry()
    team49 = MultiBlockProxyTeam(
        agents=agents, backend=backend, registry=reg49,
        max_visible_handoffs=2)
    result49 = team49.run("guard task")

    reg50 = build_trivial_w50_registry()
    team50 = W50Team(
        agents=agents, backend=backend, registry=reg50,
        max_visible_handoffs=2)
    result50 = team50.run("guard task")

    assert result50.w49_root_cid == result49.root_cid
    assert result50.final_output == result49.final_output


def test_w50_trivial_passthrough_two_runs_byte_identical() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="abc")
    agents = _build_agents("r0", backend)
    reg50 = build_trivial_w50_registry()
    team1 = W50Team(
        agents=agents, backend=backend, registry=reg50,
        max_visible_handoffs=2)
    r1 = team1.run("task42")
    team2 = W50Team(
        agents=agents, backend=backend, registry=reg50,
        max_visible_handoffs=2)
    r2 = team2.run("task42")
    assert r1.w49_root_cid == r2.w49_root_cid
    assert r1.w50_outer_cid == r2.w50_outer_cid
    assert r1.w50_params_cid == r2.w50_params_cid


def test_w50_trivial_outer_cid_does_not_equal_w49_root_cid() -> None:
    """The W50 outer CID *wraps* the W49 root CID — they should
    NOT be byte-equal even in trivial mode."""
    backend = SyntheticLLMClient(
        model_tag="synth.test", default_response="x")
    agents = _build_agents("r0", backend)
    reg50 = build_trivial_w50_registry()
    team = W50Team(
        agents=agents, backend=backend, registry=reg50,
        max_visible_handoffs=2)
    r = team.run("task")
    assert r.w50_outer_cid != r.w49_root_cid


def test_w50_trivial_registry_is_trivial_property() -> None:
    reg50 = build_trivial_w50_registry()
    assert reg50.is_trivial is True
    assert reg50.params.all_flags_disabled is True
