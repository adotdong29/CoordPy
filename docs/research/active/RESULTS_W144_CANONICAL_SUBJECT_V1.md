# RESULTS W144 — CoordPy canonical subject audit (Lane α)

**Status: FINAL.** Date 2026-06-12. Consolidation milestone (retires nothing; no new empirical claim).
Runbook: `docs/RUNBOOK_W144.md`. Machine-readable registry: `docs/W144_COORDPY_SUBJECT_REGISTRY.json`.
Stable boundary held: `0.5.20` / `coordpy.sdk.v3.43`; no PyPI; gate `258b6ed7` invariant.

## 1. The question

Not "what else can the research do?" but: **what is CoordPy, exactly, as one coherent testable
subject, and which parts of this repo are truly load-bearing to that subject?** The repo is large
(871 top-level `coordpy/*.py` modules, 378 docs, a 1654-line `__init__.py`, a 700-line milestone dump
in START_HERE), so the cost of *not* answering this is archaeology on every future visit.

## 2. Canonical subject (decision)

> **CoordPy is a context-capsule runtime.** Every inter-role, inter-layer, inter-run artefact is a
> typed, content-addressed, lifecycle-bounded, budget-bounded, provenance-carrying **capsule** — never
> a raw prompt string. The **Capsule Contract (C1..C6)** is the centre of gravity; the operational
> surfaces are the team runtime, the provenance/audit layer, the reproducible profile runner, and the
> CI/import gates.

This is **not a new framing** — it is the framing the master plan §10.0 and ARCHITECTURE.md already
named as the SDK-v3 centre of gravity. W144's job was to *confirm it against the actual code*, make it
*executable*, and draw the *stable-vs-experimental-vs-historical* line cleanly. Context Zero is the
research **programme**; CoordPy is the shipped **product**. W144 does not flatten that split.

## 3. Framing brainstorm (≥12 candidates; weak ones killed)

14 candidate centres of gravity were enumerated (full list in `RUNBOOK_W144.md §2.1`). The kills:

- **Too benchmark-shaped (S3, not the product):** "no-oracle self-tutoring / discover-then-amortize
  engine" (W141/W142b result), "contamination-resistant code-benchmark harness" (W120–W143), "a
  multi-agent coordination *research platform*", "profile-driven evaluation SDK" (the superseded
  Slice-1 outcome-name), and any framing centred on HumanEval/MBPP/ICPC margins, retirements, NIM
  pilots, or pass@k. These are Context-Zero programme outputs.
- **Too architecture-shaped (S2/S4, or the next branch):** "Latent State Transition lab",
  "transformer-substrate proxy lab" (W48–W66), "trust-adjudication / dense-control ratification
  library" (W22–W42), and any framing where KV-proxies / pseudo-attention / learned manifolds are the
  *centre*. `Latent State Transition` is the **north star of the separate architecture branch**, not
  the shipped product.
- **Too diffuse to test:** "bounded-context orchestration", "auditable agent teams", "an agent
  platform", "an LLM framework", "a context toolkit", "everything in the repo". All outcome-shaped or
  unbounded; none yield a single executable contract. (Master plan §10.0 explicitly retired the
  outcome-shaped names in favour of the mechanism-shaped capsule contract.)

**Survivor:** *context-capsule runtime* (C1..C6) — mechanism-shaped, already documented, directly
testable by re-hashing a sealed capsule chain.

## 4. The S1..S5 tier map (classification rule + result)

Rule (locked before results, extends the §10.1 stability matrix; full text in the registry):

| Tier | Meaning | How identified |
|---|---|---|
| **S1 stable-core** | the released SDK/CLI contract | stable (non-`__experimental__`) `__init__` symbol / console script / on-disk schema / `_internal.product` / `extensions`; contract-test-locked |
| **S2 canonical-experimental** | explicit-import-only but central to identity | architecture north-star lineage OR current W140..W143 load-bearing markers; ≥2 of {named, recent-marker, has-test, cross-imported} |
| **S3 benchmark/research-support** | the eval machinery | `*_bench_v*` / `*_corpus_v*` / `*_battlefield_v*` / `*_loader_v*` / `*_executor_v*` / dataset benches |
| **S4 historical/archive-only** | kept verbatim for provenance | W22–W42 dense-control ladder (188 `__experimental__` symbols) + W49–W139 substrate version ladders |
| **S5 blocked/dead/keep-for-record** | killed or superseded; KEPT | superseded version tails; W136-killed learned-memory trio |

**Result (census of all 871 top-level modules, `/tmp/w144_census.json`):**
- **S1 ≈ 21 modules** (config, provenance, run, runtime, llm_backend, capsule, capsule_runtime,
  lifecycle_audit, api_layers, agents, presets, synthetic_llm, capsule_policy(+bundle),
  capsule_decoder(+v2), team_coord, team_policy, extensions, `_cli`, `_version`, + the new `subject`)
  + 4 `_internal.product` modules; **6 console scripts**; **7 on-disk schemas**.
- **S2 = 11 modules**: the **Latent State Transition lineage** (product_manifold, live_manifold,
  learned_manifold, autograd_manifold, shared_state_proxy) + the **W140..W143 discover-then-amortize
  chain** (family_tutor_compiler_v1, self_tutoring_controller_v1, self_tutoring_technique_extractor_v1,
  no_oracle_verifier_v1/v2, multi_agent_discover_amortize_v1).
- **S3/S4 = the bulk** (~750 modules): version ladders up to v29 (`long_horizon_retention` 29,
  `persistent_latent` 28, `mergeable_latent_capsule` 27, `tiny_substrate`/`kv_bridge` 24 each, …).
- **S5 = a handful**: only **4 true orphans repo-wide** (`__main__`, `_pretty` = legitimate internals;
  `hosted_cost_planner_v10`, `hosted_logprob_router_v10` = superseded tails). **Nothing dead was
  deleted** — consolidation is classification + front-door + harness, not a purge.

## 5. Graphify evidence (graph built from `466cf445`, 87,284 nodes)

The tiers are real graph structure, not just naming:
- `graphify path coordpy multi_agent_discover_amortize_v1` → **no path** (the W143 hub is a pure
  explicit-import leaf, disconnected from the package front door) ⇒ S2, not S1.
- `graphify path agents shared_state_proxy` → **1 hop** (the W48 north-star `imports_from` the stable
  `agents.py`) ⇒ S2 architecture lineage builds *on* the stable Agent surface.
- `graphify path capsule_runtime self_tutoring_controller_v1` → 4 hops (indirect, via `__init__`).
- `graphify explain` shows the stable core in communities 4056/148/168, the manifold lineage in
  12/21/143, and the W140..W143 chain in 4109/136/71 — three distinct clusters.

## 6. Front-door decision

- **USAGE front door = `coordpy-team`** (unchanged; already documented as "the recommended front door
  for new users").
- **ORIENTATION front door = `coordpy-subject` (NEW)** — the door for understanding/verifying CoordPy.
  It prints the subject + S1..S5 tier map and runs the hermetic harness, and explicitly names
  `coordpy-team` as the usage door. One decision, two clearly-scoped surfaces, no ambiguity.

## 7. Executable subject harness (Lane β summary)

`coordpy/subject.py` + the `coordpy-subject` console script (+ `python -m coordpy.subject`) run four
hermetic checks (no network/model), all **PASS**:
1. **stable_smoke** — import, version pin (0.5.20 / v3.43), 35 stable public symbols present, 188
   `__experimental__` flagged.
2. **team_runtime** — a synthetic 2-agent `AgentTeam` seals a `coordpy.capsule_view.v1`.
3. **capsule_verify** — `verify_chain_from_view_dict` re-hashes that chain from the view bytes
   (`chain_ok=True`). The check is load-bearing: a `None` view fails, not silently passes.
4. **s2_exemplars** — all 7 curated canonical-experimental exemplars import with documented symbols.

`tests/test_w144_subject_harness_v1.py` (20 tests) locks the contract and **closes the coverage gap**
on the 3 previously-untested W140..W143 chain modules.

## 8. What this does NOT claim

W144 retires nothing and earns no new empirical result. W89 (+5.56) + W105 (+7.00) remain the only two
MULTI-AGENT retirements; W142b the distinct discover-then-amortize retirement; W143 closed the
composition line. The harness verifies that the stable contract smoke-tests pass and the subject is
legible — not that CoordPy is "production-ready" or that multi-agent context is "solved". The S2 label
is an identity judgement, not a stability promise; those modules stay explicit-import-only. The
separate `Latent State Transition` architecture branch was intentionally **not** started.
