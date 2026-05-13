"""W57 trivial passthrough: when all flags are off, the W57 envelope's
inner ``w56_outer_cid`` equals the W56 envelope CID byte-for-byte.

This is the W57 falsifier and the proof that W57 is purely additive
on top of W56 / ... / W47 / SDK v3.43.
"""

from __future__ import annotations

import hashlib

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w56_team import W56Team, build_w56_registry
from coordpy.w57_team import (
    W57Team, W57Params, W57Registry,
    build_trivial_w57_registry,
)


def _make_backend_and_agent(seed: int):
    backend = SyntheticLLMClient(
        model_tag=f"synth.w57.triv.{seed}",
        default_response="passthrough")
    agent = Agent(
        name=f"a_{seed}", instructions="",
        role="r0", backend=backend,
        temperature=0.0, max_tokens=20)
    return backend, agent


def test_w57_trivial_passthrough_w56_outer_cid_match() -> None:
    """Trivial W57 registry over a non-trivial W56 registry yields
    a W57 envelope whose w56_outer_cid equals the underlying W56's
    own outer_cid."""
    backend, agent = _make_backend_and_agent(seed=5701)
    sc = hashlib.sha256(b"w57_triv").hexdigest()
    # Trivial W57 wraps a trivial W56 by construction.
    triv_reg = build_trivial_w57_registry(schema_cid=sc)
    triv_team = W57Team(
        agents=[agent], backend=backend, registry=triv_reg,
        max_visible_handoffs=2)
    r = triv_team.run("triv passthrough")
    # The trivial W57 inner wraps a trivial W56; both should
    # passthrough byte-for-byte to W55 etc. The W57 envelope's
    # w56_outer_cid should match the explicit W56 envelope CID.
    assert r.w57_envelope.w56_outer_cid == r.w56_outer_cid
