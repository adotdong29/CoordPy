"""Minimal agent-team example.

Two common configurations.

Local Ollama:

    export COORDPY_BACKEND=ollama
    export COORDPY_MODEL=qwen2.5:0.5b
    export COORDPY_OLLAMA_URL=http://localhost:11434

OpenAI-compatible provider:

    export COORDPY_BACKEND=openai
    export COORDPY_MODEL=gpt-4o-mini
    export COORDPY_API_KEY=...
    # Optional, for non-default providers:
    # export COORDPY_API_BASE_URL=https://your-provider.example/v1
"""

from __future__ import annotations

from coordpy import AgentTeam, agent


def main() -> None:
    team = AgentTeam.from_env(
        [
            agent("planner", "Break the task into 2-3 concrete steps."),
            agent("researcher", "Gather the facts that matter."),
            agent("writer", "Write the final answer for the user."),
        ],
        team_instructions=(
            "Reuse visible handoffs instead of restating the task."
        ),
        max_visible_handoffs=3,
    )
    result = team.run("Explain what coordpy does.")
    print(result.final_output)
    if result.root_cid:
        print(f"capsule_root={result.root_cid}")


if __name__ == "__main__":
    main()
