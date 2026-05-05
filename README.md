# CoordPy

**A Python-first SDK and CLI for building auditable AI agent teams
with structured, content-addressed context.** CoordPy is built for
the bounded-context problem: instead of cramming ever-larger prompt
transcripts into every agent step, it compresses cross-agent context
into typed, content-addressed **capsules** with explicit budgets,
provenance, and lifecycle rules. One `RunSpec` in, one reproducible
`RunReport` out, and that report is the root of a sealed capsule graph
you can audit, replay, and trust.

> **Status — SDK v3.43, the final release of the CoordPy SDK v3.4x
> line (May 2026).** This is the first public release of CoordPy.
> See [Final release scope](#final-release-scope-v343) below for
> what is stable, what is experimental, and what is explicitly out
> of scope.

## What CoordPy is

CoordPy is the shipped product surface from the **Context Zero**
research programme. It gives you four things you would otherwise
build by hand on every project that wires multiple LLM agents
together:

* **A capsule contract.** Every cross-boundary artefact (prompt,
  response, parse outcome, role handoff, run report) is a typed
  object with a content-derived ID, declared parents, an explicit
  budget, and a closed lifecycle. The contract is checked at
  runtime; tampering is detectable.
* **A capsule-native runtime.** `coordpy.run(RunSpec(...))` produces a
  `RunReport` whose root is a sealed capsule DAG, written to disk
  alongside a `provenance.json` reproducibility manifest and a
  detached `meta_manifest.json` witness.
* **A team-coordination surface.** Agents exchange `TEAM_HANDOFF` /
  `ROLE_VIEW` / `TEAM_DECISION` capsules with a mechanically-checked
  T-1..T-7 lifecycle audit, so multi-agent runs are reproducible,
  bounded-context, and audit-friendly.
* **A research-grade evaluation harness.** Reproducible profiles
  (`local_smoke`, `bundled_57`, `aspen_mac1_coder`,
  `aspen_mac2_frontier`, `public_jsonl`, …), a `coordpy-ci` gate that
  consumes the report, and a `coordpy-capsule verify` CLI that
  re-hashes the on-disk capsule chain end-to-end.

## Who it is for

CoordPy is for developers and researchers building **AI agent teams**
rather than a single prompt loop. It is a good fit when you need one or
more of these:

* a stable Python SDK and CLI for multi-agent runs;
* structured handoffs between agents instead of ad hoc prompt strings;
* auditable run artifacts that can be re-verified from disk;
* a clean stable core plus an opt-in experimental research surface.

## Why CoordPy

Most multi-agent stacks treat context as text — prompts, JSON
records, ad-hoc tool traces. That works until something breaks, and
then the failure is a vague "the model was confused." CoordPy treats
context as **objects**: typed, content-addressed, lifecycle-bounded.
The result is a runtime where you can ask "what evidence did the
team actually have?", get a sealed DAG, and re-verify it from the
bytes on disk. Reproducibility, auditability, and a clean integration
boundary for downstream tools come along for free.

Just as importantly, CoordPy is about **bounded-context compression**.
The point is not only to keep a clean audit trail; it is to stop
token-cramming every full transcript into every downstream agent. The
runtime moves compact, typed capsules across role and run boundaries so
agent teams can preserve the context that matters without paying the
cost of replaying everything as raw text.

## What makes it different

CoordPy is **not** another agent-orchestration framework. It is a
runtime contract — small, stable, opinionated — under which
existing agent code becomes auditable. Capsules are the load-bearing
abstraction; everything else (CLIs, profiles, the team harness, the
research-grade trust ladder) hangs off them. The individual
primitives (content addressing, hash-chained logs, typed claim
kinds, capability-style typed references) are inherited from Git /
Merkle DAGs / IPFS / actor systems / session types. What CoordPy
contributes is the *unification* — one contract, implemented
end-to-end in a runnable SDK.

## Install

```bash
pip install coordpy
# Just the CLIs in an isolated env:
pipx install coordpy
```

Or from a clone (development install):

```bash
git clone https://github.com/adotdong29/context-zero.git
cd context-zero
pip install -e .
```

The only required dependency is NumPy. The optional LLM-agent demo
talks HTTP to a local Ollama instance — no Python binding required.

For model access, CoordPy now supports two simple stable paths:

* **Local Ollama** via `COORDPY_BACKEND=ollama` and
  `COORDPY_OLLAMA_URL=http://localhost:11434`
* **OpenAI-compatible providers** via `COORDPY_BACKEND=openai`,
  `COORDPY_MODEL=...`, and `COORDPY_API_KEY=...`

Optional extras: `coordpy[scientific]`, `coordpy[dl]`, `coordpy[heavy]`,
`coordpy[crypto]`, `coordpy[docker]` (Docker-first sandbox), `coordpy[dev]`.

Installing the package registers four console scripts:

```bash
coordpy --profile local_smoke --out-dir /tmp/coordpy-smoke
coordpy-import   --jsonl /path/to/swe_bench_lite.jsonl --out /tmp/audit.json
coordpy-ci       --report /tmp/coordpy-smoke/product_report.json --min-pass-at-1 1.0
coordpy-capsule  view   --report /tmp/coordpy-smoke/product_report.json
coordpy-capsule  verify --report /tmp/coordpy-smoke/product_report.json
```

## Quickstart

```python
import coordpy

report = coordpy.run(coordpy.RunSpec(profile="local_smoke",
                                     out_dir="/tmp/coordpy-smoke"))
assert report["readiness"]["ready"]
assert report["provenance"]["schema"] == "coordpy.provenance.v1"

# Every run ships a sealed capsule graph.
cv = report["capsules"]
assert cv["schema"] == "coordpy.capsule_view.v1"
assert cv["chain_ok"]
print(f"RUN_REPORT CID = {cv['root_cid']}")
print(report["summary_text"])
```


A first real-LLM team is one extra line:

```bash
COORDPY_OLLAMA_URL=http://localhost:11434 \
    coordpy --profile local_smoke --acknowledge-heavy --out-dir /tmp/coordpy-smoke
```

See [`docs/START_HERE.md`](docs/START_HERE.md) for the onboarding
path and [`examples/`](examples/) for short standalone programs.

If you only want the product surface, you can stop after this section
plus [Stable vs experimental — at a glance](#stable-vs-experimental--at-a-glance).
Everything below is deeper release-scope detail and historical research context.

## Create your first agent team

The easiest stable path is now the lightweight agent/team API:

```python
from coordpy import AgentTeam, agent

team = AgentTeam.from_env(
    [
        agent("planner", "Break the task into 2-3 crisp steps."),
        agent("researcher", "Gather the facts that matter."),
        agent("writer", "Write the final answer for the user."),
    ],
    model="gpt-4o-mini",          # or qwen2.5:0.5b with COORDPY_BACKEND=ollama
    backend_name="openai",        # or "ollama"
    team_instructions=(
        "Work as a bounded-context team. Reuse visible handoffs "
        "instead of restating the full task each time."
    ),
)
result = team.run("Explain what CoordPy does in plain English.")
print(result.final_output)
```

Environment variables:

```bash
# Local Ollama
export COORDPY_BACKEND=ollama
export COORDPY_MODEL=qwen2.5:0.5b
export COORDPY_OLLAMA_URL=http://localhost:11434

# OpenAI-compatible provider
export COORDPY_BACKEND=openai
export COORDPY_MODEL=gpt-4o-mini
export COORDPY_API_KEY=...
# Optional for non-default compatible providers:
# export COORDPY_API_BASE_URL=https://your-provider.example/v1
```

See [`examples/agent_team.py`](examples/agent_team.py) for the full
runnable version.

Other practical entry paths:

* **Smallest stable product path**: run `coordpy --profile local_smoke`
  and inspect the resulting `RunReport` plus sealed capsule graph.
  This is the fastest way to see the shipped SDK and CLI in action.
* **Direct capsule-level team coordination API (experimental)**:
  [`docs/CAPSULE_TEAM_FORMALISM.md`](docs/CAPSULE_TEAM_FORMALISM.md)
  plus `coordpy.__experimental__`. This is where
  `TEAM_HANDOFF`, `ROLE_VIEW`, `TEAM_DECISION`, and the lower-level
  team-coordination machinery live if you want to build directly on the
  research surface rather than stay on the stable runtime.

## Stable vs experimental — at a glance

| Surface | What you get | Stability |
|---|---|---|
| `coordpy` SDK — `RunSpec`, `run`, `RunReport`, `SweepSpec`, `run_sweep`, `CoordPyConfig`, `Agent`, `AgentTeam`, `agent`, `create_team`, `profiles`, `ci_gate`, `import_data`, `extensions`, capsule primitives, schema constants, `OpenAICompatibleBackend`, `backend_from_env` | The product / runtime contract | **Stable** |
| Console scripts — `coordpy`, `coordpy-import`, `coordpy-ci`, `coordpy-capsule` | The CLI surface | **Stable v3** |
| Capsule view / provenance / report schemas — `coordpy.capsule_view.v1`, `coordpy.provenance.v1`, `phase45.product_report.v2` | On-disk contracts | **Stable** |
| `coordpy.__experimental__` — W22..W42 trust-adjudication / multi-agent-coordination ladder, R-69..R-89 benchmark drivers, bounded live cross-host probes | Research surface, included for audit and reproduction | **Experimental** — may move, rename, or be withdrawn as the next programme starts |
| Transformer-internal trust transfer (`W42-C-NATIVE-LATENT`); K+1-host disjoint topology beyond the two-Mac pair (`W42-C-MULTI-HOST`) | Architecture-bound open frontiers | **Out of scope for this release** — see [Out of scope](#out-of-scope-for-this-release) |

The full stability matrix lives further down in
[Stability matrix](#stability-matrix); the canonical, file-by-file
status is in [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)
and [`docs/THEOREM_REGISTRY.md`](docs/THEOREM_REGISTRY.md).

## Current reader path

If you are new here, this is the recommended order:

1. Read this README through [Create your first agent team](#create-your-first-agent-team).
2. Read [`docs/START_HERE.md`](docs/START_HERE.md) for the concise
   product/research split.
3. Run one example or the `local_smoke` profile.
4. Only then drop into the release result, stability matrix, or
   historical milestone material below.

## The released result, in one paragraph

The v3.43 line closes the **capsule-layer-only research programme**
inside Context Zero. Its strongest internal result is a measured
strict trust-precision recovery on a regime where the prior best
(W41) tied at 0.500: on `R-89-ROLE-INVARIANT-RECOVER`, W42 raises
trust precision from 0.500 to **1.000 across 5/5 seeds**
(`Δ_trust_precision = +0.500`, min = max). This is the first
capsule-native multi-agent-coordination method in the programme
that materially **bounds** `W41-L-COMPOSITE-COLLUSION-CAP` at the
capsule layer via a third orthogonal evidence axis (the
role-handoff invariance axis). W42 is closed-form, deterministic,
zero-parameter, and capsule-layer; it does **not** add a
transformer-internal mechanism, does **not** close
`W42-L-FULL-COMPOSITE-COLLUSION-CAP` (a newly proved-conditional
limitation theorem), and does **not** claim universal solution of
multi-agent context. Live cross-host evidence at temperature 0 on
the two-Mac topology (`localhost` gemma2:9b + `192.168.12.191`
qwen2.5:14b) shows **4/4 paraphrase-invariant gold-correlated
agreement** across K=4 paraphrases of one arithmetic prompt — the
first measured cross-host paraphrase-invariance result in the
programme. Full results note:
[`docs/RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md`](docs/RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md);
pre-committed success bar:
[`docs/SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md`](docs/SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md);
paper:
[`papers/context_as_objects.md`](papers/context_as_objects.md).

## Final release scope (v3.43)

The SDK v3.43 line is the **final release of the CoordPy SDK v3.4x
research line** -- the **end-of-line for the capsule-layer-only
research programme** in the Context Zero project.  The boundary
between what is stable, what is experimental but included, and
what is explicitly out of scope is now final and frozen for this
release.

### Stable and shipped

* The product/runtime contract: one ``coordpy.RunSpec`` in, one
  reproducible ``RunReport`` out, where the report is the root of
  a sealed capsule graph that can be audited and replayed.  This
  contract is byte-for-byte unchanged from earlier v3.x releases
  and is what users and downstream tools should depend on.
* Public CLIs: ``coordpy``, ``coordpy-import``, ``coordpy-ci``,
  ``coordpy-capsule`` (see ``[project.scripts]`` in
  ``pyproject.toml``).
* Capsule contract types and ``coordpy.run`` /
  ``coordpy.RunReport`` orchestration (W3-7..W3-31 and W3-32..W3-41).
* Public package version: ``coordpy.__version__ = 0.5.16`` ==
  ``pyproject.toml`` ``project.version = 0.5.16``;
  ``SDK_VERSION = "coordpy.sdk.v3.43"``.

### Experimental but included

Everything under ``coordpy.__experimental__`` is
**experimental research surface** included in the release for
audit, reproduction, and downstream research.  This covers the
entire capsule-layer trust-adjudication / multi-agent-coordination
research ladder:

* The W22..W42 capsule-layer research surface (every symbol in
  the cumulative ``__experimental__`` tuple — orchestrators,
  registries, envelopes, verifiers, signature CIDs, manifest
  versions v6 through v12, decision selectors, named decision
  branches, and the 196 cumulative enumerated trust-boundary
  failure modes).
* The R-69..R-89 benchmark family drivers
  (``coordpy._internal.experiments`` (research)
  ``phase89_role_invariant_synthesis``) and the matching unit
  research notes that document the ladder.
* The bounded live cross-host probes
  (``phase8x_xllm_*`` and ``phase89_xllm_role_invariance_probe``).

These symbols may move, rename, or get withdrawn as the next
research programme starts.  Downstream code that depends on them
should pin against the experimental tuple, not assume API
stability.

### Out of scope for this release

Two open frontiers are explicitly **out of capsule-layer scope**
and are not addressed by the CoordPy SDK v3.4x line.  They are
preserved as named conjectures in
``docs/THEOREM_REGISTRY.md`` so future work has a clean handle on
them, but they are not blockers to the v3.43 final release:

* ``W42-C-NATIVE-LATENT`` — transformer-internal trust transfer.
  Closing this needs hidden-state / KV-cache / attention access that
  the capsule-layer runtime does not have.
* ``W42-C-MULTI-HOST`` — broader K+1-host disjoint topology beyond the
  current two-host lab setup. This release ships on the strongest
  honest live topology available here.

## Historical and research material

The old milestone-by-milestone build-up, legacy substrate notes, and
long-form research history are still in the repo, but they are no
longer part of the main README reading path.

If you want the historical or research-heavy material, start here:

* [`docs/START_HERE.md`](docs/START_HERE.md)
* [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)
* [`docs/THEOREM_REGISTRY.md`](docs/THEOREM_REGISTRY.md)
* [`docs/context_zero_master_plan.md`](docs/context_zero_master_plan.md)
* [`ARCHITECTURE.md`](ARCHITECTURE.md)
* [`papers/context_as_objects.md`](papers/context_as_objects.md)
* [`docs/archive/`](docs/archive/)
* [`docs/`](docs/) research notes

## Honest caveats

- **This release is the capsule-layer line, not the final universal
  answer to context.** The released result is strong within its proven
  scope, but it does not claim transformer-internal / native-latent
  transfer.
- **The research ladder is included, but not stable.** Everything under
  `coordpy.__experimental__` remains research API.
- **Live multi-host evidence is still bounded by the available lab
  topology.** The released system ships on the strongest evidence
  available here, not on a hypothetical broader host substrate.
- **Not yet peer-reviewed.** The code, tests, results notes, theorem
  registry, and paper draft are public precisely so they can be
  challenged.

---

## Project status

CoordPy is now at its **first public release**:

- stable Python-first SDK and CLI for auditable agent-team runs
- stable lightweight `Agent` / `AgentTeam` surface
- stable capsule/provenance/report contracts
- experimental W22..W42 research ladder preserved in the same repo
- paper draft and theorem registry aligned with the released scope

The next programme is explicit and separate from this release:
native-latent / transformer-internal access and broader real multi-host
substrate work sit beyond the capsule-layer line shipped here.

---

## License

MIT.
