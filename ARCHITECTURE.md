# CoordPy Architecture

## What CoordPy is

CoordPy (`pip install coordpy-ai`, import `coordpy`) is a **Python-first
agent development kit (ADK)** for building LLM agents and agent teams. Its
front door is `coordpy.adk`, which exposes a familiar
**Agent · Tool · Runner · Session · State · Memory · Artifacts** mental
model. You write plain import-and-code ADK; underneath, every model call,
tool call, and handoff seals into a typed, content-addressed, hash-chained
**capsule** trail you can re-verify from bytes and replay against another
model. That capsule layer — content-addressed audit, provenance, and replay
*by construction* — is the guarantee underneath the ADK surface, not a
separate thing you opt into. The command-line interface (`coordpy-team`, …)
is a secondary runtime surface.

## Layered architecture

CoordPy is four layers. The top layer is what you import; each lower layer is
a stable contract the layer above rests on.

```
  ┌───────────────────────────────────────────────────────────────┐
  │  coordpy.adk        Agent / Runner / Session / State / Memory   │  ← you write this
  │                     Artifacts / Tools / Sequential|Parallel|Loop│
  ├───────────────────────────────────────────────────────────────┤
  │  bridge             _capsule_trail: every run auto-seals a      │  ← automatic
  │                     re-verifiable capsule chain (root_cid)      │
  ├───────────────────────────────────────────────────────────────┤
  │  Capsule Contract   ContextCapsule / CapsuleLedger / CapsuleView│  ← the guarantee
  │                     C1..C4 construction+admission · T-1..T-7    │
  ├───────────────────────────────────────────────────────────────┤
  │  product surfaces   AgentTeam / create_team presets ·           │  ← higher-level,
  │                     RunSpec → run → RunReport profile path      │    reuses the core
  └───────────────────────────────────────────────────────────────┘
```

### Top — the `coordpy.adk` surface

`coordpy.adk` is the v1 library front door
(`ADK_SURFACE_SCHEMA == "coordpy.adk.v1"`). The shape is import-and-code: a
tool is a typed function, a turn is an event stream, and the answer is the
event that satisfies `is_final_response()`.

```python
from coordpy.adk import Agent, InMemoryRunner, ToolContext
from coordpy.llm_backend import backend_from_env

def lookup_population(city: str, tool_context: ToolContext) -> dict:
    """Return the population of a city."""
    tool_context.state["last_city"] = city
    return {"status": "success", "city": city, "population": 37_400_000}

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
```

The reference Runner is synchronous (with an async `run_async` wrapper) over a
text tool-call protocol and in-memory services — a faithful, bounded subset,
not a clone of every ADK feature. The members of the surface are: 

* **Agents.** `Agent` (alias `LlmAgent`) is a model-backed agent with an
  `instruction`, a list of `tools`, and an optional `output_key`. `BaseAgent`
  is the shared base. Workflow agents `SequentialAgent`, `ParallelAgent`, and
  `LoopAgent` compose sub-agents without a model of their own.
* **Tools.** A tool is a plain typed Python function. A trailing
  `tool_context` parameter is auto-injected and hidden from the model;
  `FunctionTool` / `to_function_tool` wrap a callable, and `BaseTool` is the
  base type.
* **Runner.** `Runner(agent=…, app_name=…, session_service=…)` drives one
  agent. `runner.run(...)` yields a **stream of typed `Event`s**; the
  user-facing answer is the event that satisfies `event.is_final_response()`.
  `InMemoryRunner(agent=…)` wires zero-config in-memory services — the
  fastest path to a running agent. Swapping in persistent services later is a
  one-line change with identical agent code (the Runner seam).
* **Session & State.** A `Session` holds the event history; `State` is a
  mutable key/value store with three reserved prefixes — `app:`
  (`APP_PREFIX`, app-wide), `user:` (`USER_PREFIX`, per-user), and `temp:`
  (`TEMP_PREFIX`, per-invocation, not persisted). Services
  (`BaseSessionService` / `InMemorySessionService`) own how state is merged
  and persisted.
* **Artifacts & Memory.** `Artifact` plus `BaseArtifactService` /
  `InMemoryArtifactService` store binary/text blobs; `MemoryEntry` /
  `SearchMemoryResponse` plus `BaseMemoryService` / `InMemoryMemoryService`
  provide retrievable long-term memory.
* **Events.** `Event` / `EventActions` carry the author, content, and a
  `state_delta` that the session service applies on append.
* **Context hierarchy.** Tools and callbacks receive a context object whose
  capability widens with trust:
  `ReadonlyContext ⊂ CallbackContext ⊂ ToolContext`. A read-only context can
  observe; a callback context can mutate state; a tool context additionally
  reaches artifacts and memory. `InvocationContext` is the per-run root.

### Bridge — every run auto-seals a capsule trail

The Runner needs no separate audit call. As it streams events it persists
each one via `session_service.append_event` (applying the event's
`state_delta`) and seals it onto a hash-chained `CapsuleTrail` (the
`_capsule_trail` bridge onto the core capsule primitives below). The result
is a re-verifiable capsule chain reachable directly from the Runner:

```python
view = runner.session_capsule_view("s1")   # → a coordpy.capsule_view.v1 chain
assert runner.verify_session("s1")          # re-verified from bytes; tamper-evident
print(runner.session_root_cid("s1"))        # deterministic for identical inputs
```

The run-report summary that fixes `root_cid` deliberately excludes the
volatile `invocation_id`, so two runs over byte-identical inputs produce the
same `root_cid`. The per-run report schema is `coordpy.adk.run.v1`.

### Core — the Capsule Contract

A **Context Capsule** is a typed, content-addressed, lifecycle-bounded,
budget-bounded, provenance-carrying unit of coordination. Every piece of
context that crosses a role, layer, or run boundary is a capsule, never a raw
prompt string. The core types live in `coordpy.capsule`:

* `ContextCapsule` — the frozen capsule (`kind`, `payload`, `budget`,
  `parents`, `cid`).
* `CapsuleKind` — the closed vocabulary of semantic kinds; unknown kinds are
  rejected at construction (`CapsuleKind.ALL`).
* `CapsuleLifecycle` — the lifecycle states `PROPOSED → ADMITTED → SEALED`
  (plus optional `RETIRED`).
* `CapsuleBudget` — an explicit byte / token / parent / round budget checked
  at admission.
* `CapsuleLedger` — the append-only, hash-chained container that admits,
  seals, and exposes capsules.
* `CapsuleView` — a serialisable slice of the capsule graph, safe to embed in
  a report and re-verify.

**Construction & admission invariants** (`coordpy/capsule.py`):

* **C1 — Identity.** A capsule's `cid` is the SHA-256 of its canonical
  content, so identical payloads have identical CIDs.
* **C2 — Typed claim.** Every capsule has a `kind` drawn from
  `CapsuleKind.ALL`; untyped capsules are illegal.
* **C3 — Budgeted admission.** Admission honours the capsule's byte / parent
  budget; an oversize capsule raises `CapsuleAdmissionError`.
* **C4 — Monotonic seal.** `seal` is monotonic — a sealed capsule is
  immutable and its CID is fixed for all time.

Provenance rides on the same mechanism: a capsule's `parents` must already be
in the ledger, the ledger keeps a hash chain so any retroactive insert breaks
verification, and the `RUN_REPORT` capsule's CID is the durable identifier
for a run.

**Team-coordination lifecycle** (`coordpy/lifecycle_audit.py`,
`coordpy/team_coord.py`). Three kinds carry multi-agent coordination —
`TEAM_HANDOFF` (a capsule-native handoff, born as a capsule), `ROLE_VIEW`
(a role's per-round admitted view, whose `max_parents` is the role's inbox
capacity), and `TEAM_DECISION` (one decision per coordination round) —
governed by invariants T-1..T-7:

* **T-1..T-3** — every `TEAM_HANDOFF` declares one source role, one target
  role, and parents reachable from the source's `ROLE_VIEW`.
* **T-4..T-5** — every `TEAM_DECISION` has parents that include every
  `TEAM_HANDOFF` it adjudicates.
* **T-6..T-7** — `ROLE_VIEW` and `TEAM_DECISION` transitions are monotonic;
  once sealed they stay reachable from the run's `RUN_REPORT` capsule.

### Product surfaces that reuse the same core

Two higher-level surfaces sit on the same capsule machinery:

* **Preset teams.** `AgentTeam` / `create_team` (plus `agent` and the curated
  `coordpy.presets`, e.g. `quant_desk_team`) build bounded-context teams:
  each agent sees the team instructions plus the latest N visible handoffs
  (default `N=4`), and `result.dump(...)` writes a four-file replayable
  bundle. The `coordpy-team` CLI is the same path from the shell.
* **Structured profile path.** `coordpy.run(coordpy.RunSpec(profile=…,
  out_dir=…))` runs a reproducible, profile-driven evaluation and returns a
  `RunReport` dict (readiness, provenance, capsule chain, `root_cid`),
  writing the matching artefacts to disk. `SweepSpec` / `run_sweep` drive
  parameter sweeps over the same machinery.

### On-disk schemas (stable)

All four schemas are content-addressed and re-verifiable from bytes alone via
`coordpy-capsule`.

| Schema constant | What it is |
|---|---|
| `coordpy.capsule_view.v1` | The sealed, hash-chained capsule chain for a run — re-verified by `runner.verify_session()` and `coordpy-capsule verify-view`. |
| `coordpy.team_result.v1` | The team manifest (per-turn prompt, generation params, capsule CIDs) used to replay a team run on another backend/model. |
| `coordpy.provenance.v1` | The provenance manifest: git SHA, package version, input SHA-256, argv, artifact list. One per run. |
| `phase45.product_report.v2` | The `RunSpec → run` product report (same shape as the returned dict). |

## Stable vs experimental boundary

The stable surface is everything you can rely on across releases; the
research ladder ships in the same wheel for reproducibility and audit but
carries no stability promise.

| Surface | Stability |
|---|---|
| `coordpy.adk` ADK (library front door): `Agent`/`LlmAgent`, `Runner`/`InMemoryRunner`, `Session`/`State`, session/artifact/memory services, `FunctionTool`/`ToolContext`/`CallbackContext`, `SequentialAgent`/`ParallelAgent`/`LoopAgent`, `Event`/`EventActions`, `ADK_SURFACE_SCHEMA` | Stable (v1) |
| Curated `coordpy` SDK (`dir(coordpy)`): `RunSpec`/`run`/`RunReport`, `SweepSpec`/`run_sweep`, `AgentTeam`/`create_team`/`agent`/`presets`, capsule primitives, backends, schema constants | Stable |
| Console scripts: `coordpy-team`, `coordpy-capsule`, `coordpy`, `coordpy-import`, `coordpy-ci`, `coordpy-subject` | Stable |
| On-disk schemas: `coordpy.capsule_view.v1`, `coordpy.team_result.v1`, `coordpy.provenance.v1`, `phase45.product_report.v2` | Stable |
| `coordpy.__experimental__` (importable, **not** in `__all__`) | Experimental — may move or disappear between releases |

The research ladder is exposed under the `coordpy.__experimental__` attribute
(a tuple of names) rather than `__all__`, so it is importable for audit and
reproducibility but is not part of the public surface. For a one-shot
orientation to what is stable vs experimental vs historical, run
`coordpy-subject` (or `python -m coordpy.subject`); the machine-readable tier
map is
[`docs/reference/W144_COORDPY_SUBJECT_REGISTRY.json`](docs/reference/W144_COORDPY_SUBJECT_REGISTRY.json).

## Design principles

* **Bounded context, not token cramming.** A role sees the team instructions
  plus the latest N visible handoffs, never the full transcript. Every team
  run reports the real-token savings.
* **Auditable by construction.** Audit is not a logging add-on: every
  boundary-crossing artefact *is* a content-addressed, hash-chained capsule,
  so a run is reconstructible and tamper-evident from its bytes.
* **Replayable across models.** A run's manifest records each turn's prompt,
  sampling params, and capsule CID, so the same prompts can be re-run on a
  different backend/model at the original settings.
* **Pure-Python core.** The capsule core and ADK surface are pure Python;
  NumPy is the only required dependency. Heavier capabilities are optional
  extras.

---

The full research-programme architecture — Context Zero: the W22..W145
substrate/manifold ladder, phase history, scale-projection operators, and the
exact-memory / retrieval / planner substrate — is preserved in
[`docs/research/active/ARCHITECTURE_VISION.md`](docs/research/active/ARCHITECTURE_VISION.md).
