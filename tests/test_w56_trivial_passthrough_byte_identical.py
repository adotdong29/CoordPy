"""W56 trivial passthrough — byte-identical to W55 outer."""

from __future__ import annotations

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w56_team import (
    W56Team,
    build_trivial_w56_registry,
)


def _build(seed: int):
    backend = SyntheticLLMClient(
        model_tag=f"synth.w56tp.{seed}",
        default_response="trivial")
    reg = build_trivial_w56_registry(
        schema_cid=f"w56_triv_{seed}")
    agents = [
        Agent(name=f"a_{seed}", instructions="",
              role="r0", backend=backend,
              temperature=0.0, max_tokens=20),
    ]
    team = W56Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2)
    return team


def test_w56_trivial_passthrough_byte_identical() -> None:
    t1 = _build(11)
    t2 = _build(11)
    r1 = t1.run("trivial")
    r2 = t2.run("trivial")
    # Two independent runs with the same trivial config must
    # produce byte-identical envelopes.
    assert r1.w56_outer_cid == r2.w56_outer_cid
    assert r1.w55_outer_cid == r2.w55_outer_cid
    assert r1.substrate_used is False


def test_w56_trivial_passthrough_substrate_unused() -> None:
    team = _build(13)
    r = team.run("trivial substrate off")
    assert r.substrate_used is False
    assert (
        r.w56_envelope.substrate_used is False)


def test_w56_trivial_passthrough_no_v8_state() -> None:
    team = _build(17)
    r = team.run("trivial v8 off")
    # V8 disabled → no V8 states should be stored
    assert len(r.persistent_v8_state_cids) == 0
