"""W51 replay-live driver — best-effort triple-backend
Ollama anchor probe.

This script demonstrates the H7 realism-anchor path:

* When ``COORDPY_W51_OLLAMA_REACHABLE=1`` is set in the
  environment AND ``OllamaBackend`` is reachable, the
  triple-backend probe runs three real model tags through
  the triple translator and records bounded fidelity.

* Otherwise the probe records
  ``anchor_status = "synthetic_only"`` and the
  ``W51-L-CROSS-TOKENIZER-TRIPLE-CAP`` conjecture carries
  forward — the W51 envelope chain still seals
  byte-identically.

Pure-Python only. The default invocation prints the W51
result with the skip path; setting
``COORDPY_W51_OLLAMA_REACHABLE=1`` opts into the real probe.
"""

from __future__ import annotations

import os

from coordpy.agents import Agent
from coordpy.cross_backend_translator import (
    W51_OLLAMA_ENV_VAR,
)
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w51_team import (
    W51Team,
    build_w51_registry,
)


def _make_anchor_backends() -> tuple[
        SyntheticLLMClient | None,
        SyntheticLLMClient | None,
        SyntheticLLMClient | None]:
    """Construct three backends for the triple anchor.

    When the env flag is set, attempt to instantiate three
    Ollama backends — but fall back to synthetic backends if
    Ollama is not reachable in this environment.
    """
    if os.environ.get(W51_OLLAMA_ENV_VAR, "").strip() != "1":
        return None, None, None
    try:
        from coordpy.llm_backend import (  # type: ignore
            OllamaBackend,
        )
    except ImportError:
        return None, None, None
    try:
        a = OllamaBackend(
            model="qwen2.5:0.5b", url=os.environ.get(
                "COORDPY_OLLAMA_URL",
                "http://localhost:11434"))
        b = OllamaBackend(
            model="qwen2.5:0.5b", url=os.environ.get(
                "COORDPY_OLLAMA_URL",
                "http://localhost:11434"))
        c = OllamaBackend(
            model="qwen2.5:0.5b", url=os.environ.get(
                "COORDPY_OLLAMA_URL",
                "http://localhost:11434"))
        return a, b, c  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None, None, None


def main() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.w51.replay",
        default_response="W51_REPLAY")
    agents = [
        Agent(name="alpha", instructions="probe", role="r0",
              backend=backend, temperature=0.0, max_tokens=24),
        Agent(name="bravo", instructions="probe", role="r1",
              backend=backend, temperature=0.0, max_tokens=24),
        Agent(name="charlie", instructions="probe", role="r2",
              backend=backend, temperature=0.0, max_tokens=24),
    ]
    reg = build_w51_registry(
        schema_cid="w51_replay_live_v1",
        role_universe=("r0", "r1", "r2"))
    ba, bb, bc = _make_anchor_backends()
    team = W51Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2,
        triple_backend_a=ba,
        triple_backend_b=bb,
        triple_backend_c=bc,
        triple_anchor_n_turns=4)
    result = team.run("W51 replay-live probe — triple anchor")
    print("=" * 72)
    print("W51 replay-live triple anchor probe")
    print("=" * 72)
    print(f"w51_outer_cid:             {result.w51_outer_cid}")
    print(f"triple_anchor_status:      {result.triple_anchor_status}")
    print(f"persistent_state_chain:    {len(result.persistent_state_cids)} states")
    print(f"n_turns:                   {result.n_turns}")
    print()
    print(f"Anchor env flag check: "
          f"{W51_OLLAMA_ENV_VAR}="
          f"{os.environ.get(W51_OLLAMA_ENV_VAR, '<unset>')!r}")
    if result.triple_anchor_status == "synthetic_only":
        print()
        print("Synthetic-only path — W51-L-CROSS-TOKENIZER-TRIPLE-CAP")
        print("carries forward unchanged. Set "
              f"{W51_OLLAMA_ENV_VAR}=1 and run with an Ollama "
              "daemon for the real-LLM probe.")
    else:
        print()
        print(f"Real-LLM anchor recorded — see the triple "
              f"translator witness CID for bounded fidelity.")


if __name__ == "__main__":
    main()
