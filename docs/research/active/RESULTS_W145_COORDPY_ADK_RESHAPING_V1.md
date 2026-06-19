# RESULTS — W145: CoordPy ADK reshaping (library-first product surface)

**Milestone class:** product-shape / library-usability. **Earns no new
empirical result; retires nothing.** Stable boundary held: 0.5.20 /
coordpy.sdk.v3.43; no PyPI; gate `258b6ed7` invariant; `coordpy/__init__.py`
gains only an additive `adk` re-export.

**One line.** W145 re-centres the CoordPy *product* as a Python-first **agent
development kit (ADK)** modelled on the official Google ADK product shape
(read live from `adk.dev`), copying the conceptual surface and developer
ergonomics — not prose or code — while keeping CoordPy's capsule / provenance
/ replay / audit guarantees underneath. The library (`coordpy.adk`) becomes
the front door; the CLI becomes a secondary runtime surface.

---

## 1. What W145 is / is not

**Is:** a product-shape milestone; a Python-first public-surface tightening; a
library-usability milestone; one coherent thing a normal developer can pick up
without archaeology.

**Is not:** a benchmark hunt; the Latent State Transition architecture branch
(intentionally NOT started); a CLI surface expansion; a docs-only pass.

**Product/programme split (the rule that makes this legal).** The canonical
docs say Context Zero is "not reducible to a framework, a library, or a
product." That discipline is about the **programme**. W145 reshapes only the
**product** surface: CoordPy gets a clean ADK shape; Context Zero stays the
research programme. The "library-first" framing applies to the product only.

---

## 2. Lane α — official-ADK product-shape audit and CoordPy remap

### 2.1 The ADK primitive set (locked BEFORE building)

Read from the official Google ADK docs (`adk.dev`; `google.github.io/adk-docs`
301-redirects there; PyPI pkg `google-adk`): home, `get-started/about`,
`get-started/python`, `tutorials/agent-team`, `sessions/`, `sessions/state`,
`sessions/memory`, `artifacts/`, `callbacks/types-of-callbacks`,
`agents/workflow-agents`, `tools-custom/function-tools`, `context/`, `events/`.

The canonical primitives: **Agent / LlmAgent**, **Tool / FunctionTool**,
**Runner / InMemoryRunner**, **Session**, **SessionService**
(InMemory/Database/VertexAi), **State** (prefixes `user:` / `app:` / `temp:` /
none), **Memory / MemoryService**, **Artifact / ArtifactService**,
**ReadonlyContext ⊂ CallbackContext ⊂ ToolContext / InvocationContext**,
**sub_agents** (LLM transfer), **SequentialAgent / ParallelAgent / LoopAgent**,
**Events / EventActions**, six **callbacks**. The turn model: `runner.run(...)`
yields a *stream* of typed `Event`s; the displayable one is
`event.is_final_response()`. Front door: **library-first** (`pip install
google-adk`; import `Agent`); the CLI (`adk run` / `adk web` / `adk api_server`)
is a secondary accelerant over the same objects.

### 2.2 Google-ADK → CoordPy concept map

| Google ADK | CoordPy ADK (`coordpy.adk`) | CoordPy guarantee underneath |
|---|---|---|
| `Agent` / `LlmAgent` | `Agent` (alias `LlmAgent`) | each model step seals PROMPT/LLM_RESPONSE capsules |
| `FunctionTool` / plain fn | `FunctionTool` / plain fn in `tools=` | each tool call seals a capsule |
| `Runner` / `InMemoryRunner` | `Runner` / `InMemoryRunner` | drives the event loop; builds a `CapsuleLedger`; `session_capsule_view()` + `verify_session()` |
| `Session` | `Session` | event + state container |
| `BaseSessionService` / `InMemorySessionService` | same names | `append_event` applies prefix-routed `state_delta` |
| `State` (`user:`/`app:`/`temp:`/none) | `State` (same prefixes) | delta-tracked; persistence encoded in the key |
| `Memory` / `MemoryService` / `InMemoryMemoryService` | `BaseMemoryService` / `InMemoryMemoryService` | keyword recall over ingested past sessions |
| `Artifact` / `ArtifactService` / `InMemoryArtifactService` | `Artifact` / `InMemoryArtifactService` | save → ARTIFACT capsule (content-addressed, on-disk-verifiable) |
| `ReadonlyContext`⊂`CallbackContext`⊂`ToolContext`, `InvocationContext` | same four | carry the live `State` + services |
| `Event` / `EventActions` | `Event` / `EventActions` | each Event seals one hash-chained capsule |
| `sub_agents` (LLM transfer) | `sub_agents` (transfer) | transfer seals a TEAM_HANDOFF capsule |
| `SequentialAgent`/`ParallelAgent`/`LoopAgent` | same | workflow over shared `State` via `output_key` |
| `output_key` | `output_key` | the inter-step data bus |
| six callbacks | same six | guardrail / cache / filter hooks |
| CLI `adk run`/`web`/`api_server` (secondary) | `coordpy-team` CLI (now SECONDARY) | unchanged, still works |

**The distinctive CoordPy layer Google ADK does not emphasize** — and which
W145 makes automatic under the ADK surface: content-addressed capsules,
provenance, deterministic CIDs, replayable runs, tamper-evident auditable
handoffs.

### 2.3 Candidate product framings — brainstormed and killed

10+ framings considered; the winner is #3.

1. "Context-capsule runtime" (W144 status quo) — accurate, but research-shaped:
   says what's *underneath*, not what a developer *does first*. **Kept as the
   underneath-guarantee; killed as the front-door framing.**
2. "Auditable agent framework" — "framework" is vague and collides with the
   "not a framework" discipline. **Killed.**
3. **"Python-first ADK, with capsule audit underneath" — WINNER.** Names the
   verb (build agents), the mental model (ADK), and the differentiator (audit).
4. "Provenance layer for LLM agents" — too narrow; reads like a logging plugin.
   **Killed.**
5. "Multi-agent orchestration library (`coordpy-team`)" — keeps the team/CLI as
   the story; narrow and CLI-shaped. **Killed.**
6. "Reproducible eval harness for code benchmarks" — that is the *programme*,
   not the product (S3 machinery). **Killed.**
7. "Drop-in Google ADK clone" — overclaims feature-parity; dishonest. **Killed.**
8. "Capsule SDK / content-addressed context store" — infra-shaped, not
   agent-shaped. **Killed.**
9. "Agent runtime with replay" — "runtime" reads passive; loses the build verb.
   **Killed** in favour of the ADK verb.
10. "LangChain-style chains over capsules" — wrong mental model (chains, not
    agents). **Killed.**
11. "Tool-calling agent SDK" — too narrow (drops sessions/state/memory/teams).
    **Killed.**
12. "Verifiable agent OS / latent OS" — that is the S2 north-star / architecture
    branch, out of scope for the product. **Killed.**

### 2.4 Failure-mode enumerations (used to keep the winner honest)

**≥8 ways a framing is still too CLI-shaped** (all avoided): (1) leading the
README/quickstart with `coordpy-team run --preset …`; (2) describing the
product by its console scripts; (3) a "five-minute first run" that is a CLI
invocation, not an import; (4) front door = `coordpy-team`; (5) output framed
as "writes four files into out-dir"; (6) configuration via flags rather than
constructor kwargs; (7) replay framed as `coordpy-team replay`; (8)
verification framed as a `coordpy-capsule verify-view` shell step rather than
`runner.verify_session()`.

**≥8 ways a framing is still too research-shaped** (all avoided): (1) leading
with "context-capsule runtime / C1..C6"; (2) front-loading the W22..W42
dense-control ladder; (3) surfacing `__experimental__` as if it were the API;
(4) describing the product via theorems (W3-39/40/41) and retirements
(W89/W105/W142b); (5) centering "bounded-context vs token cramming"; (6) naming
the product after a milestone chain; (7) pointing newcomers at
RESEARCH_STATUS / THEOREM_REGISTRY first; (8) exposing the substrate version
ladders (`tiny_substrate_v*`, `persistent_latent_v*`) as the surface.

**≥6 ways a framing is too broad/vague** (all avoided): (1) "an agent
framework" (unbounded; collides with "not a universal agent platform"); (2) "a
platform for AI"; (3) "coordinate any LLM agents"; (4) "production-ready agent
infrastructure" (overclaims); (5) "a library for context" (too abstract); (6)
"everything is a capsule" (true but says nothing about what you build); (7)
"an SDK and CLI" (the W144 phrasing — generic; picks no verb or mental model).

### 2.5 Canonical ADK subject statement (decision)

> **CoordPy is a Python-first agent development kit (ADK):** you build Agents,
> Tools, Sessions, and sub-agents and run them with a Runner, while every
> model call, tool call, and handoff automatically seals into a typed,
> content-addressed, lifecycle-bounded, provenance-carrying capsule you can
> re-verify and replay. The ADK ergonomics are the front door (`coordpy.adk`);
> the **Capsule Contract (C1..C6)** is the guarantee underneath; the CLI is
> secondary.

This is mirrored verbatim in `coordpy/subject.py` and
`docs/W144_COORDPY_SUBJECT_REGISTRY.json`.

### 2.6 Keep / reshape / demote / promote

* **KEEP (stable, underneath, untouched):** `capsule`, `capsule_runtime`,
  `lifecycle_audit`, `provenance`, `llm_backend`, `synthetic_llm`,
  `run`/`runtime`, `presets`, and the legacy `agents.py`
  (`Agent`/`AgentTeam`/`create_team`/`replay_team_result`) — back-compat.
* **PROMOTE (new front door):** `coordpy.adk` — the library ADK surface +
  example + capsule bridge. It is the recommended door for *building*.
* **RESHAPE (story):** README + `START_HERE` + the subject one-liner +
  `front_doors` → ADK-first / import-and-code.
* **DEMOTE (kept, no longer the story):** the six console scripts. `coordpy-team`
  stays the CLI usage door; `coordpy-subject` stays orientation.
* **LEAVE SECONDARY / explicit-import-only (unchanged):** S2 manifold lineage +
  W140..W143 chain; S3 benchmark machinery; S4 historical ladders; S5
  keep-for-record. No research module is promoted into the user API.

---

## 3. Lane β — the library-first CoordPy ADK surface (build inventory)

New package `coordpy/adk/` (additive). Modules:

| Module | Surface |
|---|---|
| `sessions.py` | `State` (prefix-routed, delta-tracked), `Event`/`EventActions`, `Session`, `BaseSessionService`, `InMemorySessionService` |
| `artifacts.py` | `Artifact`, `BaseArtifactService`, `InMemoryArtifactService` (versioned; `user:`-scope by filename) |
| `memory.py` | `MemoryEntry`, `SearchMemoryResponse`, `BaseMemoryService`, `InMemoryMemoryService` (keyword recall) |
| `context.py` | `InvocationContext`, `ReadonlyContext` ⊂ `CallbackContext` ⊂ `ToolContext` |
| `tools.py` | `BaseTool`, `FunctionTool` (schema by reflection; auto-injected `tool_context`), `to_function_tool` |
| `agents.py` | `BaseAgent`, `Agent`/`LlmAgent` (text tool-call + transfer protocol, six callbacks, `output_key`), `SequentialAgent`/`ParallelAgent`/`LoopAgent` |
| `_capsule_trail.py` | `CapsuleTrail` — seals each event onto `coordpy.capsule` (`CapsuleLedger`/`ContextCapsule`/`render_view`); no new capsule kinds |
| `runner.py` | `Runner`, `InMemoryRunner`, `session_capsule_view()`, `verify_session()`, `session_root_cid()`, async `run_async` |
| `examples/research_assistant.py` | a researcher→writer `SequentialAgent` (tools + artifacts + memory); `root_agent`; `run_demo()`; `python -m …` |

Plus `examples/adk_quickstart.py` (repo-level, hermetic) and
`tests/test_w145_adk_surface_v1.py` (16 tests).

**The capsule guarantee, automatic.** The Runner persists every event through
`session_service.append_event` (applying its `state_delta`) and seals it into a
hash-chained `CapsuleLedger`. `runner.session_capsule_view(session_id)` returns
a `coordpy.capsule_view.v1` chain; `runner.verify_session(...)` recomputes the
chain from bytes (tamper-evident); the sealed payload is content-only, so
`root_cid` is **deterministic for identical inputs**.

**Honest scope (a faithful, bounded subset — not a Google ADK clone).** The
reference Runner is synchronous with an async `run_async` wrapper; tool-calling
uses a text protocol (`TOOL_CALL: {…}` / `TRANSFER: <name>`) that works with any
text backend (incl. the hermetic synthetic/scripted backends); services are
in-memory. `DatabaseSessionService` / persistent artifact+memory backends /
streaming events / function-calling-native tool dispatch are explicit future
work (W146), not claimed here.

**Validation.** `tests/test_w145_adk_surface_v1.py` (16 tests) + the W144
subject harness (`tests/test_w144_subject_harness_v1.py`, 23 tests) both PASS,
hermetic — **no NIM, no network, no model, $0**. Coverage: surface import +
back-compat; single-turn event stream; function tool + `ToolContext` state;
unknown-tool handling; `State` prefix routing (`temp:` dropped, `user:`
cross-session) + `output_key` data bus; versioned artifacts; `sub_agents`
transfer; `SequentialAgent`/`ParallelAgent`/`LoopAgent` (escalate + max-iters);
callback short-circuit (before_model, before_tool); capsule trail verify +
tamper-evidence; deterministic `root_cid`; memory recall; the example app.

---

## 4. Lane γ — docs / story / examples / truth-surface consolidation

Files changed: `README.md` (Python-first; ADK quickstart + capsule-for-free
lead; CLI demoted; public-surface table adds the `coordpy.adk` row),
`docs/START_HERE.md` (ADK-first orientation), `coordpy/subject.py` + the
machine-readable `docs/W144_COORDPY_SUBJECT_REGISTRY.json` (new `library` front
door; ADK one-liner; `adk` in S1; provenance → W145), `ARCHITECTURE.md` (W145
library-front-door callout), `CHANGELOG.md` (W145 entry), this RESULTS doc,
`docs/RUNBOOK_W145.md`, and `linear_github_mapping.json`. graphify refreshed at
start (HEAD `aff4d70`) and end. `coordpy/__init__.py` gains a single additive
line (`from . import adk`).

---

## 5. Front-door decision

**`coordpy.adk` (library, import-and-code) = the BUILD front door (new,
primary).** `coordpy-team` (CLI) is demoted to a secondary runtime surface.
`coordpy-subject` stays the orientation/verification door. This revises W144's
CLI-leaning emphasis while preserving W144's S1..S5 tier map and the
product/programme split.

---

## 6. What this does NOT claim (anti-overstatement)

* W145 earns **no new empirical result** and **retires nothing**. **W89
  (+5.56) + W105 (+7.00) remain the only two MULTI-AGENT retirements; W142b is
  the distinct discover-then-amortize retirement; W143 closed the composition
  line.**
* CoordPy is **not** "production-ready"; multi-agent context is **not**
  "solved"; the ADK surface does **not** beat Google's ADK and is **not**
  feature-complete vs it (it is a faithful, bounded subset). The capsule layer
  does **not** reach transformer internals. Bounded-context / compaction /
  summarization remain anti-patterns — not what the ADK surface is.
* The "library-first" framing is about the **product**; the **programme**
  (Context Zero) is still "not reducible to a library."
* The tests prove the ADK surface works hermetically and the capsule trail
  verifies — not that it is exhaustively complete.

---

## 7. W146 branch logic

By the end of W145, W146 is one of: (1) **deepen the ADK product** — persistent
`DatabaseSessionService` / artifact + memory backends, streaming events, an
`adk`-native CLI front door; (2) **begin the Latent State Transition
architecture branch** (the W144-named next move), now that the product has a
clean front door to host it; (3) **re-open COO-9** (third-retirement benchmark)
if benchmark spend is greenlit and a code-competent model whose efficient form
is NOT i.i.d.-reachable is available. W145 chooses none; **COO-9 remains the
lead *research* path** (W145 is a product milestone, orthogonal to the frontier).

---

## 8. Stable-boundary invariant

**No version bump (0.5.20 / coordpy.sdk.v3.43); no PyPI; gate `258b6ed7`
invariant; `coordpy/__init__.py` gains only an additive `adk` re-export. W89
(+5.56) + W105 (+7.00) remain the only two multi-agent retirements.**
