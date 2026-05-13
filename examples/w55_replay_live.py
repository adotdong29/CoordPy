"""W55 replay-live driver — best-effort hept-backend Ollama
anchor probe + CRC V3 BCH-corrected live probe.

This script demonstrates the W55 realism-anchor path:

* When ``COORDPY_W55_OLLAMA_REACHABLE=1`` is set in the
  environment AND ``OllamaBackend`` is reachable, the inner
  W54 hex anchor runs (which in turn calls the W53 quint /
  W52 quad anchor). W55 inherits that anchor result via the
  W54 envelope.

* Otherwise the W54 inner anchor records
  ``anchor_status = "synthetic_only"``; the W55 envelope still
  seals byte-identically and the
  ``W55-C-CROSS-TOKENIZER-HEPT-CAP`` conjecture carries
  forward.

Pure-Python only. The default invocation prints the W55 result
with the skip path; setting ``COORDPY_W55_OLLAMA_REACHABLE=1``
opts into the real probe.
"""

from __future__ import annotations

import os

from coordpy.agents import Agent
from coordpy.synthetic_llm import SyntheticLLMClient
from coordpy.w55_team import (
    W55Team,
    build_w55_registry,
)


W55_OLLAMA_ENV_VAR = "COORDPY_W55_OLLAMA_REACHABLE"


def _make_anchor_backends() -> tuple[
        SyntheticLLMClient | None,
        SyntheticLLMClient | None,
        SyntheticLLMClient | None,
        SyntheticLLMClient | None]:
    """Construct four backends for the inner W54 hex anchor."""
    if (os.environ.get(W55_OLLAMA_ENV_VAR, "").strip()
            != "1"):
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
        model_tag="synth.w55.replay",
        default_response="W55_REPLAY")
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
    reg = build_w55_registry(
        schema_cid="w55_replay_schema_v1",
        role_universe=("r0", "r1", "r2", "r3"))
    qa, qb, qc, qd = _make_anchor_backends()
    team = W55Team(
        agents=agents,
        backend=backend,
        registry=reg,
        max_visible_handoffs=2,
        quad_backend_a=qa, quad_backend_b=qb,
        quad_backend_c=qc, quad_backend_d=qd,
        quad_anchor_n_turns=3)
    result = team.run(
        "W55 replay-live probe: 7-backend hept transitivity + "
        "trust-weighted compromise + BCH(15,7) double-bit "
        "corrected ECC V7 + MLSC V3 trust-decay + TWCC "
        "5-stage controller + 5-arm arbiter + disagreement "
        "algebra")
    print("=" * 72)
    print(
        f"W55 replay-live driver — "
        f"{len(result.turn_witness_bundles)} turns")
    print("=" * 72)
    anchor_env = (
        "ollama_reachable"
        if os.environ.get(W55_OLLAMA_ENV_VAR, "").strip()
        == "1" else "synthetic_only")
    print(f"anchor_env:                    {anchor_env}")
    print(
        f"final_output:                  "
        f"{result.final_output!r}")
    print(
        f"w54_outer_cid:                 "
        f"{result.w54_outer_cid}")
    print(
        f"w55_outer_cid:                 "
        f"{result.w55_outer_cid}")
    print(
        f"composite_confidence_v3_mean:  "
        f"{result.composite_confidence_mean_v3:.4f}")
    print(
        f"trust_weighted_composite_mean: "
        f"{result.trust_weighted_composite_mean:.4f}")
    print(
        f"twcc_quorum_rate:              "
        f"{result.twcc_quorum_rate:.4f}")
    print(
        f"persistent_v7_chain (n):       "
        f"{len(result.persistent_v7_state_cids)}")
    print(
        f"mlsc_v3_capsule_chain (n):     "
        f"{len(result.mlsc_v3_capsule_cids)}")
    print()
    print(
        "Done. No version bump (coordpy stays at 0.5.20). "
        "Set COORDPY_W55_OLLAMA_REACHABLE=1 to enable the "
        "inner W54 hex-backend real-LLM anchor probe.")


if __name__ == "__main__":
    main()
