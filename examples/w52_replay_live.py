"""W52 replay-live driver — best-effort quad-backend Ollama
anchor probe.

This script demonstrates the H8 realism-anchor path:

* When ``COORDPY_W52_OLLAMA_REACHABLE=1`` is set in the
  environment AND ``OllamaBackend`` is reachable, the
  quad-backend probe runs four real model tags through the
  multi-hop translator and records bounded fidelity.

* Otherwise the probe records
  ``anchor_status = "synthetic_only"`` and the
  ``W52-L-CROSS-TOKENIZER-QUAD-CAP`` conjecture carries
  forward — the W52 envelope chain still seals
  byte-identically.

Pure-Python only. The default invocation prints the W52 result
with the skip path; setting
``COORDPY_W52_OLLAMA_REACHABLE=1`` opts into the real probe.
"""

from __future__ import annotations

import os

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w52_team import (
    W52Team,
    build_w52_registry,
)


W52_OLLAMA_ENV_VAR = "COORDPY_W52_OLLAMA_REACHABLE"


def _make_anchor_backends() -> tuple[
        SyntheticLLMClient | None,
        SyntheticLLMClient | None,
        SyntheticLLMClient | None,
        SyntheticLLMClient | None]:
    """Construct four backends for the quad anchor.

    When the env flag is set, attempt to instantiate four
    Ollama backends — but fall back to None if Ollama is not
    reachable in this environment.
    """
    if os.environ.get(W52_OLLAMA_ENV_VAR, "").strip() != "1":
        return None, None, None, None
    try:
        from coordpy.llm_backend import (  # type: ignore
            OllamaBackend,
        )
    except ImportError:
        return None, None, None, None
    try:
        url = os.environ.get(
            "COORDPY_OLLAMA_URL", "http://localhost:11434")
        a = OllamaBackend(model="qwen2.5:0.5b", url=url)
        b = OllamaBackend(model="qwen2.5:0.5b", url=url)
        c = OllamaBackend(model="qwen2.5:0.5b", url=url)
        d = OllamaBackend(model="qwen2.5:0.5b", url=url)
        return a, b, c, d  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None, None, None, None


def main() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.w52.replay",
        default_response="W52_REPLAY")
    agents = [
        Agent(name="alpha", instructions="probe", role="r0",
              backend=backend, temperature=0.0, max_tokens=24),
        Agent(name="bravo", instructions="probe", role="r1",
              backend=backend, temperature=0.0, max_tokens=24),
        Agent(name="charlie", instructions="probe", role="r2",
              backend=backend, temperature=0.0, max_tokens=24),
        Agent(name="delta", instructions="probe", role="r3",
              backend=backend, temperature=0.0, max_tokens=24),
    ]
    reg = build_w52_registry(
        schema_cid="w52_replay_live_v1",
        role_universe=("r0", "r1", "r2", "r3"))
    ba, bb, bc, bd = _make_anchor_backends()
    team = W52Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2,
        quad_backend_a=ba,
        quad_backend_b=bb,
        quad_backend_c=bc,
        quad_backend_d=bd,
        quad_anchor_n_turns=4)
    result = team.run("W52 replay-live probe — quad anchor")
    print("=" * 72)
    print("W52 replay-live quad anchor probe")
    print("=" * 72)
    print(f"w52_outer_cid:                 {result.w52_outer_cid}")
    print(f"multi_hop_anchor_status:       "
          f"{result.multi_hop_anchor_status}")
    print(f"persistent_v4_chain:           "
          f"{len(result.persistent_v4_state_cids)} states")
    print(f"n_turns:                       {result.n_turns}")
    print()
    print(f"Anchor env flag check: "
          f"{W52_OLLAMA_ENV_VAR}="
          f"{os.environ.get(W52_OLLAMA_ENV_VAR, '<unset>')!r}")
    if result.multi_hop_anchor_status == "synthetic_only":
        print()
        print("Synthetic-only path — W52-L-CROSS-TOKENIZER-QUAD-CAP")
        print("carries forward unchanged. Set "
              f"{W52_OLLAMA_ENV_VAR}=1 and run with an Ollama "
              "daemon for the real-LLM probe.")
    else:
        print()
        print("Real-LLM anchor recorded — see the multi-hop "
              "translator witness CID for bounded fidelity.")


if __name__ == "__main__":
    main()
