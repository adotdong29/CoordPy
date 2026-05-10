"""Example 01 — quickstart: a three-agent team with sealed audit.

The smallest stable provider-backed agent-team example. Shows the
bounded-context handoff story end-to-end:

* Three agents (planner -> researcher -> writer) share state through
  ``TEAM_HANDOFF`` capsules, not through full transcripts.
* Each agent sees the team instructions plus the latest N visible
  handoffs (here N=3) plus a one-line ``task_summary`` (after the
  first turn).
* The runtime seals every handoff into a content-addressed capsule
  chain you can re-verify from disk.

Pick a backend by setting ``COORDPY_BACKEND`` plus the matching
credentials/endpoint:

    # Local Ollama
    export COORDPY_BACKEND=ollama
    export COORDPY_MODEL=qwen2.5:0.5b
    export COORDPY_OLLAMA_URL=http://localhost:11434

    # OpenAI-compatible provider
    export COORDPY_BACKEND=openai
    export COORDPY_MODEL=gpt-4o-mini
    export COORDPY_API_KEY=...
    # optional for non-default compatible providers:
    # export COORDPY_API_BASE_URL=https://your-provider.example/v1

Then::

    python3 examples/01_quickstart.py
"""

from __future__ import annotations

from coordpy import AgentTeam, agent


def main() -> None:
    team = AgentTeam.from_env(
        [
            agent("planner", "Break the task into 2-3 crisp steps."),
            agent("researcher", "Gather the facts that matter."),
            agent("writer", "Write the final answer for the user."),
        ],
        team_instructions=(
            "Work as a bounded-context team. Reuse the visible "
            "handoffs above instead of restating the full task."
        ),
        max_visible_handoffs=3,
        task_summary=(
            "Answer the user briefly using only the prior handoffs."
        ),
    )
    result = team.run("Explain what CoordPy does in plain English.")

    print(result.final_output)
    print()

    # Bounded-context savings vs naive token cramming.
    cramming = result.cramming_estimate()
    print(
        f"savings: {cramming['saved_words']} words "
        f"({cramming['savings_pct']:.1f}%) "
        f"~{cramming['estimated_tokens_saved']} tokens"
    )
    if result.root_cid:
        print(f"capsule_root={result.root_cid}")


if __name__ == "__main__":
    main()
