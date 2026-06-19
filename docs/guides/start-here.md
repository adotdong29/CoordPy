# Start here

CoordPy is a **Python-first agent development kit (ADK)** for building LLM
agents and agent teams — with the now-familiar `Agent` / `Tool` / `Runner` /
`Session` / `State` / `Memory` / `Artifacts` model — that gives you
**content-addressed capsule audit, provenance, and replay for free**
underneath. You write import-and-code ADK; every model call, tool call, and
handoff automatically seals into a hash-chained capsule you can re-verify
from bytes and replay against another model.

This page is the fastest path to using it. For the architecture, see
[`ARCHITECTURE.md`](../../ARCHITECTURE.md); for the full reference, see
[`docs/reference/`](../reference/).

## Install

Requires Python 3.10+. The only required dependency is NumPy.

```bash
pip install coordpy-ai          # import name: coordpy
```

Optional extras: `coordpy-ai[scientific]`, `[crypto]`, `[dl]`, `[heavy]`,
`[docker]`, `[dev]`.

## Your first agent (the front door)

`coordpy.adk` is import-and-code. A tool is a plain typed, docstring'd
function (a trailing `tool_context` is auto-injected and hidden from the
model); a turn is a stream of events; the answer is the one that satisfies
`is_final_response()`.

```python
from coordpy.adk import Agent, InMemoryRunner, ToolContext
from coordpy.llm_backend import backend_from_env

def lookup_population(city: str, tool_context: ToolContext) -> dict:
    """Return the population of a city."""
    table = {"tokyo": 37_400_000, "paris": 11_100_000}
    tool_context.state["last_city"] = city
    pop = table.get(city.lower())
    return ({"status": "success", "city": city, "population": pop}
            if pop else {"status": "error", "error": f"unknown city {city!r}"})

assistant = Agent(
    name="geo_assistant",
    model=backend_from_env(),          # any Ollama / OpenAI-compatible backend
    instruction="Answer population questions; use the lookup tool.",
    tools=[lookup_population],
    output_key="answer",
)

runner = InMemoryRunner(agent=assistant, app_name="quickstart")
runner.session_service.create_session(
    app_name="quickstart", user_id="u1", session_id="s1")

for event in runner.run(user_id="u1", session_id="s1",
                        new_message="How big is Tokyo?"):
    if event.is_final_response():
        print(event.text)

# The CoordPy guarantee underneath — for free:
assert runner.verify_session("s1")            # re-verified from bytes
print(runner.session_root_cid("s1"))          # deterministic for identical inputs
```

No-network demos that need no model:
[`examples/adk_quickstart.py`](../../examples/adk_quickstart.py) and
`python -m coordpy.adk.examples.research_assistant` (researcher → writer with
tools, artifacts, and memory).

### Configure a backend

```bash
# Local Ollama (no API key)
export COORDPY_BACKEND=ollama
export COORDPY_MODEL=qwen2.5:14b
export COORDPY_OLLAMA_URL=http://localhost:11434

# OpenAI-compatible provider
export COORDPY_BACKEND=openai
export COORDPY_MODEL=gpt-4o-mini
export COORDPY_API_KEY=...
# export COORDPY_API_BASE_URL=https://your-provider.example/v1   # optional
```

## Secondary surfaces

The ADK is the front door. These build on the same capsule machinery:

* **CLI** — `coordpy-team run/replay/sweep/compare` drives curated preset
  teams; `coordpy-capsule verify-view` re-hashes a sealed chain from disk.
* **`AgentTeam`** — `AgentTeam.from_env([...])` for bounded-context preset
  teams in Python (the legacy `coordpy.Agent` / `create_team` surface,
  unchanged and additive to `coordpy.adk.Agent`).
* **`RunSpec` → `RunReport`** — the structured, profile-driven evaluation
  path (`coordpy.run(coordpy.RunSpec(profile="local_smoke", ...))`).

See the [README](../../README.md) for runnable examples of each.

## Stable vs experimental

`dir(coordpy)` lists the **stable** surface: `coordpy.adk`, the curated
`coordpy` SDK, and the version/schema constants. The console scripts and the
on-disk schemas (`coordpy.capsule_view.v1`, `coordpy.team_result.v1`,
`coordpy.provenance.v1`, `phase45.product_report.v2`) are stable too.

The research programme (the "Context Zero" multi-agent / substrate /
manifold ladder behind the papers) ships under `coordpy.__experimental__`:
importable as `coordpy.<name>` for reproducibility and audit, but **not** in
`dir(coordpy)` or `from coordpy import *`, and with no stability promise.

Run it, don't read it:

```bash
coordpy-subject check        # prints the canonical subject + the S1..S5
                             # stable/experimental/historical tier map and
                             # runs the hermetic stable-contract harness
```

Machine-readable tier map:
[`docs/reference/W144_COORDPY_SUBJECT_REGISTRY.json`](../reference/W144_COORDPY_SUBJECT_REGISTRY.json).

## Where to read next

* [README](../../README.md) — the full tour (CLI, teams, replay, audit).
* [`ARCHITECTURE.md`](../../ARCHITECTURE.md) — the layered design and the
  Capsule Contract.
* [`docs/reference/`](../reference/) — capsule formalism + the subject registry.
* [`CONTRIBUTING.md`](../../CONTRIBUTING.md) · [`RELEASING.md`](../../RELEASING.md) · [`SECURITY.md`](../../SECURITY.md).
* The research archive lives in [`docs/archive/`](../archive/) and the active
  research head in [`docs/research/active/`](../research/active/).
