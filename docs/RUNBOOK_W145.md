# RUNBOOK — W145: CoordPy ADK reshaping (library-first product surface)

**Status:** locked before implementation. One RUNBOOK, one truth surface
(`docs/RESULTS_W145_COORDPY_ADK_RESHAPING_V1.md`), one decision path.

**Operator greenlight:** W145 re-centres CoordPy (the *product*) as a
Python-first **Agent Development Kit (ADK)** whose developer ergonomics
mirror Google's official ADK product shape (read from `https://adk.dev/`),
while CoordPy's distinctive guarantees — content-addressed capsules,
provenance, replay, auditable handoffs — stay underneath. This is **not**
the Latent State Transition architecture branch and **not** a benchmark
milestone.

**What W145 IS:** a CoordPy-as-ADK product-shape milestone; a Python-first
public-surface tightening; a library usability milestone; a "make it useful
to a normal developer" milestone; one coherent thing.

**What W145 IS NOT:** another benchmark hunt; the architecture branch;
another CLI surface expansion; a docs-only pass; a research-ladder nostalgia
trip.

---

## 0. Product/programme split (the rule that makes this legal)

The canonical docs say Context Zero is "not reducible to a framework, a
library, or a product" and CoordPy is "not a universal agent platform."
That discipline is about the **programme**, not the **product**. W145 only
reshapes the *product surface*:

* **Context Zero = research programme** (the W1..W144 arc). Unchanged.
* **CoordPy = shipped product.** W145 gives the product a clean,
  library-first ADK shape. It does **not** claim the programme is "just a
  library," does **not** claim multi-agent context is "solved," and does
  **not** retire anything.

The Capsule Contract (C1..C6) remains the centre of gravity. The ADK shape
is the *front door* to it, not a replacement for it.

---

## 1. α / β / γ branch logic (exact)

**Lane α — official-ADK product-shape audit + CoordPy remap (synthesis).**
1. Read the current official Google ADK docs FIRST; treat them as the
   product-shape reference. Copy the SHAPE and ergonomics, never prose/code.
2. Lock the target ADK primitive set BEFORE building (§3).
3. Map the current CoordPy public surface honestly: keep / reshape / demote /
   promote (§4 of RESULTS).
4. Brainstorm ≥10 candidate framings; ≥8 "still too CLI-shaped"; ≥8 "still
   too research-shaped"; ≥6 "too broad/vague"; kill the weak ones (RESULTS §3).
5. Produce the Google-ADK→CoordPy concept map (§3 here, expanded in RESULTS).
6. α succeeds iff it lands: a canonical ADK-style subject statement + a
   concept map + an honest keep/reshape/demote list.

**Lane β — build the library-first CoordPy ADK surface (construction).**
1. Land a real, import-and-code subpackage `coordpy/adk/` exposing the locked
   primitive set, simple enough to teach in one short example.
2. Additive only: `coordpy.Agent` / `create_team` / `AgentTeam` /
   `replay_team_result` keep their existing signatures (the ADK `Agent` lives
   at `coordpy.adk.Agent`, a DIFFERENT object — no rename, no break).
3. Keep capsule/audit underneath: every run automatically seals a
   hash-chained capsule trail the user can verify and replay, without having
   to learn the research ladder.
4. Land ≥1 genuinely useful canonical example app (research / document-triage
   assistant), not a benchmark toy, not a thin script wrapper.
5. β succeeds iff: the library-first surface lands, the hermetic tests pass,
   and the result reads as one usable ADK.

**Lane γ — docs/story/examples/truth-surface consolidation (mandatory).**
1. README + START_HERE become Python-first (import-and-code leads; CLI
   secondary).
2. Subject one-liner + front doors + registry updated to the ADK story
   (additively; W144 tier map + product/programme split survive).
3. CHANGELOG + ARCHITECTURE + `linear_github_mapping.json` + Linear synced.
4. graphify refreshed at start and end.
5. γ alone is NOT success — W145 must land executable library/API/example/
   test assets.

---

## 2. Official-doc research rule

* Source of truth: the live official Google ADK docs at `https://adk.dev/`
  (canonical; `google.github.io/adk-docs` 301-redirects there; PyPI pkg
  `google-adk`). Pages mined: home, `get-started/about`, `get-started/python`,
  `tutorials/agent-team`, `sessions/`, `sessions/state`, `sessions/memory`,
  `artifacts/`, `callbacks/types-of-callbacks`, `agents/workflow-agents`,
  `tools-custom/function-tools`, `context/`, `events/`.
* Copy the **conceptual surface and ergonomics only** (primitive names,
  constructor keyword shape, the event-stream turn model, state prefixes).
  Do **not** copy Google's prose or code verbatim. No vendored Google code.
* The CoordPy implementation is original Python over CoordPy's own capsule /
  backend machinery.

---

## 3. ADK-concept mapping rule (locked primitive set + concept map)

CoordPy's canonical equivalents (namespace `coordpy.adk`, mirroring
`from google.adk... import ...`):

| Google ADK | CoordPy ADK (`coordpy.adk`) | CoordPy guarantee underneath |
|---|---|---|
| `Agent` / `LlmAgent` | `Agent` (alias `LlmAgent`) | each model step seals PROMPT/LLM_RESPONSE capsules |
| `FunctionTool` / plain fn tools | `FunctionTool` / plain fn in `tools=` | each tool call seals a capsule |
| `Runner` / `InMemoryRunner` | `Runner` / `InMemoryRunner` | drives the event loop; builds a `CapsuleLedger`; `runner.session_capsule_view()` + `verify` |
| `Session` | `Session` | event + state container |
| `BaseSessionService` / `InMemorySessionService` | same names | `append_event` applies prefix-routed `state_delta` |
| `State` (`user:`/`app:`/`temp:`/none) | `State` (same prefixes) | delta-tracked; persistence encoded in the key |
| `Memory` / `MemoryService` / `InMemoryMemoryService` | `BaseMemoryService` / `InMemoryMemoryService` | keyword search over ingested past sessions |
| `Artifact` / `ArtifactService` / `InMemoryArtifactService` | `Artifact` / `InMemoryArtifactService` | save → ARTIFACT capsule (content-addressed, on-disk-verifiable) |
| `ReadonlyContext`⊂`CallbackContext`⊂`ToolContext`, `InvocationContext` | same four | carry the live `State` + services |
| `Event` / `EventActions` | `Event` / `EventActions` | each Event seals one hash-chained capsule |
| `sub_agents` (LLM transfer) | `sub_agents` (transfer) | transfer seals a TEAM_HANDOFF/ROLE_VIEW capsule |
| `SequentialAgent`/`ParallelAgent`/`LoopAgent` | same | workflow over shared `State` via `output_key` |
| `output_key` | `output_key` | the inter-step data bus |
| 6 callbacks (before/after agent·model·tool) | same 6 | guardrail / cache / filter hooks |
| CLI `adk run`/`web`/`api_server` (secondary) | `coordpy-team` CLI (now SECONDARY) | unchanged, still works |

The turn model is faithful: `runner.run(...)` yields a stream of typed
`Event`s; the displayable one is `event.is_final_response()`.

**Distinctive CoordPy layer Google ADK does not emphasize** — kept and made
automatic under the ADK surface: content-addressed capsules, provenance,
deterministic CIDs, replayable runs, auditable handoffs.

---

## 4. Public-surface reshaping rule

* **Additive + backwards-compatible.** No existing public import breaks. New
  surface is the `coordpy.adk` subpackage; `coordpy/__init__.py` gains exactly
  one additive line (`from . import adk`) plus the smoke-symbol `"adk"`.
* **No rename of `coordpy.Agent`.** The legacy product `Agent`
  (`instructions=`, `role=`, `backend=`) stays. The ADK `Agent`
  (`model=`, `instruction=`, `tools=`, `sub_agents=`, `output_key=`) is
  `coordpy.adk.Agent` — a deliberately separate object.
* **Demote, don't delete.** The six console scripts remain fully functional;
  they stop being the recommended front door. `coordpy-team` is the CLI usage
  door; `coordpy-subject` is orientation.
* **Research stays explicit-import-only.** S2/S3/S4/S5 unchanged. No research
  module is promoted into the user-facing ADK API.
* **No version bump, no PyPI.** `__version__` stays `0.5.20`; `SDK_VERSION`
  stays `coordpy.sdk.v3.43`. Gate `258b6ed7` untouched.

---

## 5. Validation rule

* New hermetic test file `tests/test_w145_adk_surface_v1.py` — **no network,
  no model, no NIM spend** (uses `SyntheticLLMClient`). Must cover: surface
  import; back-compat of `coordpy.Agent`/`create_team`; Agent+Runner+Session
  single turn via `is_final_response()`; function-tool call + `ToolContext`
  state write; `State` prefix routing (`temp:` not persisted) + `output_key`
  data bus; artifacts (save→version, load, list); `sub_agents` transfer;
  `SequentialAgent` pipeline; `LoopAgent` escalate; **capsule-underneath**
  (`session_capsule_view` → `verify_chain_from_view_dict` True; tamper → False);
  memory add+search; deterministic replay (stable CIDs).
* The existing W144 subject harness test must still pass unchanged
  (`run_harness()` stays at 4 checks).
* The example app must run hermetically (`python -m coordpy.adk.examples.research_assistant`).
* Run the focused W145 + W144 tests; do not run the 30-min full sweep.

---

## 6. Anti-overstatement rule

* W145 is a **product-shape milestone**. It earns **no new empirical result**
  and **retires nothing**. The standing truth holds verbatim: **W89 (+5.56)
  + W105 (+7.00) remain the only two MULTI-AGENT retirements; W142b is the
  distinct discover-then-amortize retirement; W143 closed the composition
  line.**
* Do NOT claim: CoordPy is "production-ready"; multi-agent context is
  "solved"; the ADK surface beats Google's ADK; the capsule layer reaches
  transformer internals. Bounded-context / compaction / summarization remain
  anti-patterns; they are not what the ADK surface is.
* The ADK reshaping is a *product front door*, explicitly NOT a claim about
  the programme. The "library-first" framing applies to the product only.
* The new tests prove the ADK surface works hermetically and the capsule
  trail verifies — not that it is feature-complete vs Google ADK (it is a
  faithful, bounded subset: synchronous reference Runner, JSON tool-call
  protocol, in-memory services).

---

## 7. graphify deliverables

* `graphify update .` at the **start** (done; built from HEAD `aff4d70`,
  no topology change) and again at the **end** after code/doc changes, so
  `graphify-out/` matches repo truth.
* During the milestone, use `graphify explain` / `graphify path` /
  `graphify affected` to confirm: the new `adk` package's edges into
  `capsule` / `llm_backend`; that `adk` does NOT pull in the research ladder;
  and that `coordpy → adk` is a clean additive edge.
* `graphify-out/` stays an untracked local artifact (not committed).

---

## 8. Truth-surface consolidation rule

* One result note: `docs/RESULTS_W145_COORDPY_ADK_RESHAPING_V1.md` (the α
  audit + concept map + brainstorm/kills + β build inventory + γ changes +
  anti-overstatement). One RUNBOOK (this file).
* Synced surfaces: `README.md`, `docs/START_HERE.md`, `ARCHITECTURE.md`,
  `coordpy/subject.py` + `docs/W144_COORDPY_SUBJECT_REGISTRY.json` (additive
  `library` front door + ADK one-liner + `adk` in S1), `CHANGELOG.md`,
  `linear_github_mapping.json`, Linear (new W145 issue + COO-9/COO-6 comment).
* The W144 subject registry stays the orientation source of truth; W145
  updates it in place (the milestone provenance becomes W145; the W144 tier
  map and split survive).

---

## 9. W146 branch logic

By the end of W145, W146 is one of:
1. **Deepen the ADK product** — add `DatabaseSessionService` / persistent
   artifact + memory backends / streaming events / an `adk`-native CLI
   front door (`coordpy adk run`) — continue the usability line.
2. **Begin the Latent State Transition architecture branch** (the W144-named
   next move), now that the product has a clean front door to host it.
3. **Re-open COO-9** (third-retirement benchmark) only if benchmark spend is
   greenlit and a code-competent model whose efficient form is NOT
   i.i.d.-reachable is available.

W145 does not choose; it leaves all three open and records that **COO-9
remains the lead *research* path** (W145 is a product milestone, orthogonal
to the retirement frontier).

---

## Stable-boundary invariant (close every surface with this)

**No version bump (0.5.20 / coordpy.sdk.v3.43); no PyPI; gate `258b6ed7`
invariant; `coordpy/__init__.py` gains only an additive `adk` re-export.
W89 (+5.56) + W105 (+7.00) remain the only two multi-agent retirements.**
