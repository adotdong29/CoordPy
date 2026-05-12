"""W50 replay-live realism anchor probe.

Best-effort: when ``COORDPY_W50_OLLAMA_REACHABLE=1`` and a small
Ollama model is reachable, runs a real-LLM cross-backend
alignment probe and records the bounded fidelity. Otherwise
records ``anchor_status: "synthetic_only"`` and exits with a
deterministic skip-witness — the W50-L-CROSS-BACKEND-TOKENIZER-
CAP carry-forward.

This is the W50 ``family_cross_backend_alignment_realism_probe``
realism anchor, exposed as an example so users can run it
locally without modifying coordpy.
"""

from __future__ import annotations

import os
import sys

from coordpy.cross_backend_alignment import (
    W50_ANCHOR_STATUS_REAL_LLM,
    W50_ANCHOR_STATUS_SYNTHETIC,
    W50_DEFAULT_OLLAMA_MODEL,
    W50_OLLAMA_ENV_VAR,
    run_realism_anchor_probe,
)


def main() -> int:
    print("=" * 60)
    print("W50 replay-live realism anchor probe")
    print("=" * 60)
    env_flag = os.environ.get(W50_OLLAMA_ENV_VAR, "")
    print(f"{W50_OLLAMA_ENV_VAR} = {env_flag!r}")

    if env_flag.strip() not in ("1", "true", "True"):
        # Skip path — deterministic synthetic-only witness
        payload = run_realism_anchor_probe()
        print(f"anchor_status:   {payload['anchor_status']}")
        print(f"skipped_ok:      {payload['skipped_ok']}")
        print(f"reason:          {payload.get('reason', '')}")
        print()
        print(
            "Set COORDPY_W50_OLLAMA_REACHABLE=1 and ensure an "
            "Ollama daemon is running locally with "
            f"{W50_DEFAULT_OLLAMA_MODEL} pulled to attempt the "
            "real-LLM anchor probe.")
        return 0

    # Real-LLM probe path: try to construct an OllamaBackend.
    try:
        from coordpy.llm_backend import OllamaBackend
    except ImportError as exc:  # pragma: no cover
        print(f"OllamaBackend unavailable: {exc}")
        print("Falling back to synthetic_only.")
        payload = run_realism_anchor_probe()
        print(f"anchor_status: {payload['anchor_status']}")
        return 0

    from coordpy.synthetic_llm import SyntheticLLMClient

    primary = OllamaBackend(model=W50_DEFAULT_OLLAMA_MODEL)
    synthetic = SyntheticLLMClient(
        model_tag="synth.w50.anchor",
        default_response="W50_OK_anchor")
    payload = run_realism_anchor_probe(
        primary_backend=primary,
        synthetic_backend=synthetic,
        n_turns=10,
        require_real=True)
    print(f"anchor_status:   {payload['anchor_status']}")
    print(f"n_turns:         {payload['n_turns']}")
    print(f"fidelity:        {payload['fidelity']}")
    if payload["anchor_status"] == W50_ANCHOR_STATUS_REAL_LLM:
        print(f"primary_model_tag:   {payload.get('primary_model_tag')}")
        print(f"synthetic_model_tag: {payload.get('synthetic_model_tag')}")
    print()
    print(
        "W50 does NOT claim closure of the "
        "cross-tokenizer conjecture. This probe BOUNDS — not "
        "closes — `W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
