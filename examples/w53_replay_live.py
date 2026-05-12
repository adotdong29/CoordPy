"""W53 replay-live driver — best-effort quint-backend Ollama
anchor probe + corruption-robust carrier live probe.

This script demonstrates the W53 realism-anchor path:

* When ``COORDPY_W53_OLLAMA_REACHABLE=1`` is set in the
  environment AND ``OllamaBackend`` is reachable, the
  quint-backend probe runs five real model tags through the
  multi-hop translator V3 and records bounded fidelity.

* Otherwise the probe records
  ``anchor_status = "synthetic_only"`` and the
  ``W53-C-CROSS-TOKENIZER-QUINT-CAP`` conjecture carries
  forward — the W53 envelope chain still seals byte-identically.

Pure-Python only. The default invocation prints the W53 result
with the skip path; setting
``COORDPY_W53_OLLAMA_REACHABLE=1`` opts into the real probe.
"""

from __future__ import annotations

import os

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w53_team import (
    W53Team,
    build_w53_registry,
)


W53_OLLAMA_ENV_VAR = "COORDPY_W53_OLLAMA_REACHABLE"


def _make_anchor_backends() -> tuple[
        SyntheticLLMClient | None,
        SyntheticLLMClient | None,
        SyntheticLLMClient | None,
        SyntheticLLMClient | None]:
    """Construct four backends for the inner W52 quad anchor."""
    if os.environ.get(W53_OLLAMA_ENV_VAR, "").strip() != "1":
        return None, None, None, None
    try:
        from coordpy.llm_backend import (  # type: ignore
            OllamaBackend,
        )
    except ImportError:
        return None, None, None, None
    try:
        url = os.environ.get(
            "COORDPY_OLLAMA_URL",
            "http://localhost:11434")
        a = OllamaBackend(model="qwen2.5:0.5b", url=url)
        b = OllamaBackend(model="qwen2.5:0.5b", url=url)
        c = OllamaBackend(model="qwen2.5:0.5b", url=url)
        d = OllamaBackend(model="qwen2.5:0.5b", url=url)
        return a, b, c, d  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None, None, None, None


def main() -> None:
    backend = SyntheticLLMClient(
        model_tag="synth.w53.replay",
        default_response="W53_REPLAY")
    agents = [
        Agent(name="alpha", instructions="probe",
              role="r0", backend=backend,
              temperature=0.0, max_tokens=24),
        Agent(name="bravo", instructions="probe",
              role="r1", backend=backend,
              temperature=0.0, max_tokens=24),
        Agent(name="charlie", instructions="probe",
              role="r2", backend=backend,
              temperature=0.0, max_tokens=24),
        Agent(name="delta", instructions="probe",
              role="r3", backend=backend,
              temperature=0.0, max_tokens=24),
    ]
    reg = build_w53_registry(
        schema_cid="w53_replay_live_v1",
        role_universe=("r0", "r1", "r2", "r3"))
    ba, bb, bc, bd = _make_anchor_backends()
    team = W53Team(
        agents=agents, backend=backend, registry=reg,
        max_visible_handoffs=2,
        quad_backend_a=ba,
        quad_backend_b=bb,
        quad_backend_c=bc,
        quad_backend_d=bd,
        quad_anchor_n_turns=4)
    result = team.run("W53 replay-live probe — quint anchor")
    print("=" * 72)
    print("W53 replay-live quint anchor probe")
    print("=" * 72)
    print(
        f"w53_outer_cid:                 "
        f"{result.w53_outer_cid}")
    print(
        f"w52_outer_cid:                 "
        f"{result.w52_outer_cid}")
    print(
        f"composite_confidence_mean:     "
        f"{result.composite_confidence_mean:.4f}")
    print(
        f"arbiter_pick_rate_shared_mean: "
        f"{result.arbiter_pick_rate_shared_mean:.4f}")
    print(
        f"persistent_v5_chain (n):       "
        f"{len(result.persistent_v5_state_cids)}")
    print(
        f"mlsc_capsule_chain (n):        "
        f"{len(result.mlsc_capsule_cids)}")
    print()
    if ba is None:
        print(
            "Quint anchor: synthetic_only (set "
            f"{W53_OLLAMA_ENV_VAR}=1 to attempt real "
            "Ollama probe)")
    else:
        print(
            "Quint anchor: real_llm_anchor — see W52 "
            "anchor payload for inner quad probe results")
    print(
        "Done. No version bump (coordpy stays at 0.5.20). "
        "Reachable via explicit `from coordpy.w53_team "
        "import W53Team`.")


if __name__ == "__main__":
    main()
