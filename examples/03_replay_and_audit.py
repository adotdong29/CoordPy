"""Example 03 — replay a sealed run and re-verify the chain.

Runs a tiny three-agent team, dumps the four-file sealed bundle,
then immediately replays the manifest against a fresh backend
instance and re-hashes the new chain. The replay-side
``prompt_sha256`` for every turn must match the original; per-turn
``temperature`` and ``max_tokens`` are read from the manifest, not
substituted with the loader's defaults.

This is the "audit my run" demo. It uses the same primitives the
``coordpy-team replay`` CLI uses, but in 60 lines of Python so you
can see the contract.

Pick a backend with the usual ``COORDPY_BACKEND`` /
``COORDPY_MODEL`` / ``COORDPY_OLLAMA_URL`` /
``COORDPY_API_KEY`` env vars. With local Ollama::

    COORDPY_BACKEND=ollama \\
    COORDPY_MODEL=qwen2.5:0.5b \\
    COORDPY_OLLAMA_URL=http://localhost:11434 \\
        python3 examples/03_replay_and_audit.py
"""

from __future__ import annotations

import json
import sys
import tempfile

from coordpy import (
    AgentTeam, agent, backend_from_env, replay_team_result,
    verify_chain_from_view_dict,
)


def main() -> int:
    try:
        original_backend = backend_from_env()
    except Exception as exc:
        print(
            "could not build backend: " + repr(exc),
            file=sys.stderr,
        )
        return 2

    team = AgentTeam(
        [
            agent(
                "planner", "Break the task into 2-3 short steps.",
                temperature=0.0, max_tokens=180,
            ),
            agent(
                "researcher", "Gather the key facts.",
                temperature=0.0, max_tokens=240,
            ),
            agent(
                "writer", "Write the final answer in one paragraph.",
                temperature=0.3, max_tokens=300,
            ),
        ],
        backend=original_backend,
        team_instructions=(
            "Bounded-context team. Reuse the visible handoffs."
        ),
        max_visible_handoffs=2,
        task_summary="Answer briefly using the prior handoffs.",
    )

    print("=== ORIGINAL RUN ===")
    result = team.run(
        "What is bounded-context transfer in one paragraph?")
    print(result.final_output.strip())

    with tempfile.TemporaryDirectory() as td:
        paths = result.dump(td)
        print()
        print(f"sealed bundle: {td}")
        for k, v in paths.items():
            print(f"  {k:<14s} {v}")

        # Spot-check the manifest carries per-turn generation params.
        with open(paths["team_result"], "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        print()
        print("=== PER-TURN GENERATION PARAMS (from manifest) ===")
        for t in manifest["turns"]:
            print(
                f"  {t['role']:<14s} "
                f"temperature={t['temperature']:.2f}  "
                f"max_tokens={t['max_tokens']:>4d}"
            )

        # Replay against a fresh backend instance. The persisted
        # (temperature, max_tokens) tuples are restored faithfully.
        replay_backend = backend_from_env()
        print()
        print("=== REPLAY AGAINST FRESH BACKEND ===")
        replayed = replay_team_result(
            paths["team_result"], backend=replay_backend)
        print(replayed.final_output.strip())

        print()
        print("=== AUDIT ===")
        for original, redo in zip(result.turns, replayed.turns):
            ok = original.prompt_sha256 == redo.prompt_sha256
            print(
                f"  {redo.role:<14s} prompt_sha256_match={ok}"
            )
        print(f"  capsule_root_original={result.root_cid}")
        print(f"  capsule_root_replay  ={replayed.root_cid}")
        chain_ok = (
            replayed.capsule_view is not None
            and verify_chain_from_view_dict(replayed.capsule_view)
        )
        print(f"  chain_recompute_ok   ={chain_ok}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
