# Context Zero — Reference Architecture

**Wevra** is the shipped **context-capsule runtime** produced by the
**Context Zero** research programme. Every piece of context that
crosses a role boundary, a layer boundary, or a run boundary in Wevra
is a typed, content-addressed, lifecycle-bounded, budget-bounded,
provenance-carrying **capsule** — never a raw prompt string. As of
SDK v3.3 (April 2026), capsules drive execution at the run boundary,
inside the inner sweep loop, AND on the parser axis (PARSE_OUTCOME
capsule sealed before every PATCH_PROPOSAL — Theorem W3-39). A
runtime-checkable lifecycle audit mechanically verifies eight
invariants L-1..L-8 over every finished run (Theorem W3-40), and
deterministic-mode replay opt-in (`RunSpec(deterministic=True)`)
collapses the full capsule DAG byte-for-byte across runs of the same
logical input (Theorem W3-41). Meta-artefacts have a formally-defined
detached-witness boundary (Theorem W3-36). Substantive on-disk
artifacts are content-addressed at write time and re-verifiable at
audit time. This document is the programme's architectural
reference: it covers the full substrate (routing, exact memory,
retrieval, planner, runtime calibration, typed handoffs) and the
Wevra product surface built on top of it, now *centred* on the
Capsule Contract and *driven* by it. For a one-pass orientation,
start with [`docs/START_HERE.md`](docs/START_HERE.md). For canonical
research status see
[`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md); for the
canonical theorem registry see
[`docs/THEOREM_REGISTRY.md`](docs/THEOREM_REGISTRY.md); for the
do-not-overstate rule book see
[`docs/HOW_NOT_TO_OVERSTATE.md`](docs/HOW_NOT_TO_OVERSTATE.md).

## The Capsule Contract (SDK v3 centre of gravity)

Wevra's durable top-level description is:

> **Wevra is a context-capsule runtime.** Every inter-role,
> inter-layer, and inter-run artefact satisfies a six-invariant
> contract:
>
>   **C1  Identity.**     Stable content-address (SHA-256) over
>                          `(kind, payload, budget, parents)`.
>   **C2  Typed claim.**   Closed vocabulary of `CapsuleKind`.
>   **C3  Lifecycle.**     `PROPOSED → ADMITTED → SEALED` (+ optional
>                          `RETIRED`); illegal transitions are refused.
>   **C4  Budget.**        Explicit `CapsuleBudget` checked at admit
>                          time.
>   **C5  Provenance.**    Parents must be in the ledger; the ledger
>                          keeps a hash chain so any retroactive
>                          insert breaks `verify_chain()`.
>   **C6  Frozen.**        A sealed capsule's CID is fixed for all
>                          time.

The Phase-19 `Handle`, Phase-31 `TypedHandoff`, Phase-35
`ThreadResolution`, Phase-36 `AdaptiveEdge`, every `SweepSpec` /
sweep-cell, every `ARTIFACT` on disk, and the `RUN_REPORT` itself
are all capsule-shaped. The `CapsuleLedger` is their shared
append-only, hash-chained container. The `RUN_REPORT` capsule's CID
is the durable identifier for a Wevra run — send someone that CID
plus `product_report.json` and they can reproduce every upstream
capsule, verify the chain end-to-end, and know the bytes haven't
drifted.

Reference implementation: `vision_mvp/wevra/capsule.py`. Theory note:
[`docs/archive/wevra-milestones/RESULTS_WEVRA_CAPSULE.md`](docs/archive/wevra-milestones/RESULTS_WEVRA_CAPSULE.md).
Contract tests: `vision_mvp/tests/test_wevra_capsules.py`
(invariants C1..C6 individually + end-to-end).

### Capsule-native execution (SDK v3.1)

The Capsule Contract above (C1..C6) describes *what a capsule is*.
SDK v3.1 adds the **execution-contract** layer: capsules drive
runtime, not just describe it.

```
                                                       (sealed in flight)
   start_run    seal_readiness  seal_sweep_spec  seal_sweep_cell  seal_provenance  seal_run_report
       │              │                │                │                │                │
       ▼              ▼                ▼                ▼                ▼                ▼
   PROFILE  →  READINESS_CHECK   →  SWEEP_SPEC  →   SWEEP_CELL    →  PROVENANCE   →   RUN_REPORT
                                          ↑                                ↑
                                          │                                │
                              seal_and_write_artifact            seal_and_write_artifact
                              (sweep_result.json)               (provenance.json,
                                                                 readiness_verdict.json)
                                       (every substantive artefact gets an
                                        ARTIFACT capsule whose payload SHA-256
                                        is verified against the on-disk file's
                                        bytes by re-read.)
```

A stage that fails leaves a typed entry in the runtime's *in-flight
register* that never reaches the ledger. Downstream stages refuse
to seal because the parent CID is missing (Capsule Contract C5).
The capsule layer is therefore the runtime's typed execution
contract for the run-boundary stages (W3-32, W3-35).

### Intra-cell capsule-native + detached witness (SDK v3.2)

SDK v3.2 extends the capsule-native slice past the cell boundary.
Inside every sweep cell, each (task, strategy) parse→apply→test
transition seals two more capsules in flight:

```
                              SWEEP_SPEC
                              ├── SWEEP_CELL_1   ··· SWEEP_CELL_n
                              ├── PATCH_PROPOSAL_1   (parent: SWEEP_SPEC)
                              │       └── TEST_VERDICT_1   (parent: PATCH_PROPOSAL_1)
                              ├── PATCH_PROPOSAL_2
                              │       └── TEST_VERDICT_2
                              └── ...

                                         (post-fixed-point, secondary ledger)
                              RUN_REPORT
                                  └── (cross-ref) META_MANIFEST
                                                   meta_artifacts:
                                                     product_report.json    SHA
                                                     capsule_view.json      SHA
                                                     product_summary.txt    SHA
```

The lifecycle ordering ``patch → verdict`` is enforced at the
type level (Theorem W3-32-extended). The meta-artefact set is
formally a *circularity slice* (Theorem W3-36 — no extension of
the primary ledger can authenticate a file whose bytes encode
the rendered view), so the META_MANIFEST sits in a *secondary*
ledger and is the one-hop trust unit beyond the primary view.
``wevra-capsule verify`` (v3.2) recomputes the chain from
on-disk header bytes (Theorem W3-37) and re-hashes every
ARTIFACT and meta-artefact at audit time (Theorem W3-38).

Reference implementation:
`vision_mvp/wevra/capsule_runtime.py::CapsuleNativeRunContext`
(``seal_patch_proposal`` / ``seal_test_verdict`` /
``seal_meta_manifest``); hooks plumbed through
`vision_mvp/tasks/swe_sandbox.py::run_swe_loop_sandboxed`. Theory
notes:
[`docs/archive/wevra-milestones/RESULTS_WEVRA_CAPSULE_NATIVE.md`](docs/archive/wevra-milestones/RESULTS_WEVRA_CAPSULE_NATIVE.md)
(W3-32..W3-35) and
[`docs/archive/wevra-milestones/RESULTS_WEVRA_INTRA_CELL.md`](docs/archive/wevra-milestones/RESULTS_WEVRA_INTRA_CELL.md)
(W3-32-extended / W3-36 / W3-37 / W3-38). Contract tests:
`vision_mvp/tests/test_wevra_capsule_native.py` (16 tests, v3.1)
and `vision_mvp/tests/test_wevra_capsule_native_intra_cell.py`
(16 tests, v3.2).

The post-hoc `build_report_ledger` adapter is retained for third
parties who fold finished `product_report` dicts (no runtime
context available); the two paths produce CID-equivalent ledgers
for the spine kinds (Theorem W3-34, preserved under the v3.2
intra-cell extension because intra-cell capsules are siblings of
the spine, not modifications of it).

### Sub-intra-cell parse-outcome + lifecycle audit + determinism (SDK v3.3)

SDK v3.3 extends the discipline one further structural layer with a
PARSE_OUTCOME capsule sealed *before* every PATCH_PROPOSAL:

```
                              SWEEP_SPEC
                              ├── SWEEP_CELL_1   ··· SWEEP_CELL_n
                              ├── PARSE_OUTCOME_1   (parent: SWEEP_SPEC)
                              │       └── PATCH_PROPOSAL_1
                              │               (parents: SWEEP_SPEC + PARSE_OUTCOME_1)
                              │               └── TEST_VERDICT_1
                              ├── PARSE_OUTCOME_2
                              │       └── PATCH_PROPOSAL_2
                              │               └── TEST_VERDICT_2
                              └── ...
```

The parser's structured outcome — `ok` boolean, closed-vocabulary
`failure_kind`, `recovery` label, substitutions count, bounded
detail — becomes a typed witness on the capsule DAG. The parse →
patch → verdict chain is enforced at the type level (Theorem W3-39).

The **lifecycle audit** (`vision_mvp/wevra/lifecycle_audit.py`)
mechanically verifies eight invariants L-1..L-8 over a finished
ledger:

  * L-1 No orphan capsules.
  * L-2 PATCH_PROPOSAL parents include SWEEP_SPEC.
  * L-3 TEST_VERDICT parent is exactly one sealed PATCH_PROPOSAL.
  * L-4 PARSE_OUTCOME parent is exactly SWEEP_SPEC.
  * L-5 SWEEP_CELL parent is exactly SWEEP_SPEC.
  * L-6 PATCH_PROPOSAL ↔ TEST_VERDICT ↔ PARSE_OUTCOME coordinates
    are equinumerous.
  * L-7 PATCH_PROPOSAL coordinates match its PARSE_OUTCOME parent's.
  * L-8 TEST_VERDICT is sealed strictly after its PATCH_PROPOSAL.

The audit returns OK / BAD / EMPTY plus typed counterexamples. It is
runnable from a `CapsuleNativeRunContext` (in-process) or from an
on-disk `capsule_view.json` alone (forensic). Theorem W3-40 anchors
the audit's soundness.

**Deterministic-mode replay** (`RunSpec(deterministic=True)`) strips
per-run / host-local / wall-clock fields from the
PROVENANCE / READINESS_CHECK / RUN_REPORT capsule payloads and the
ARTIFACT capsule paths so two runs of the same deterministic
profile (mock mode, `in_process` / `subprocess` sandbox, frozen
JSONL) produce byte-identical full-DAG CIDs and chain head
(Theorem W3-41). On-disk product_report.json still records
wall-clock fields for forensic context — the determinism is on
the capsule graph, not on wall clock.

Reference implementation:
`vision_mvp/wevra/capsule_runtime.py::CapsuleNativeRunContext.seal_parse_outcome`,
`vision_mvp/wevra/lifecycle_audit.py`,
`vision_mvp/product/runner.py::_canonicalise_for_determinism`.
Theory note:
[`docs/archive/wevra-milestones/RESULTS_WEVRA_DEEP_INTRA_CELL.md`](docs/archive/wevra-milestones/RESULTS_WEVRA_DEEP_INTRA_CELL.md).
Contract tests: `vision_mvp/tests/test_wevra_capsule_native_deeper.py`
(18 tests).

### How capsules relate to the older CASR / substrate / handoff work

| Older primitive                         | Phase | Capsule kind it instantiates |
|---                                      |---    |---                           |
| `context_ledger.Handle`                  | 19    | `HANDLE`                     |
| `role_handoff.TypedHandoff`             | 31    | `HANDOFF`                    |
| `dynamic_comm.ThreadResolution`         | 35    | `THREAD_RESOLUTION`          |
| `adaptive_sub.AdaptiveEdge`             | 36    | `ADAPTIVE_EDGE`              |
| `wevra.runtime.SweepSpec`               | —     | `SWEEP_SPEC`                 |
| per-cell sweep report (`wevra.sweep.v2`) | —     | `SWEEP_CELL`                 |
| `phase44_public_readiness` verdict      | 44    | `READINESS_CHECK`            |
| `wevra.provenance.v1` manifest          | —     | `PROVENANCE`                 |
| on-disk `product_report.json` etc.       | —     | `ARTIFACT`                   |
| resolved profile dict                    | —     | `PROFILE`                    |
| the run itself                           | —     | `RUN_REPORT`                 |

The older primitives are **byte-for-byte unchanged**. The capsule
layer names the contract they already satisfied, lifts them under
one ledger, and makes that ledger the SDK's new public centre. None
of this is retrofitted cryptography: the hash-chaining that
`HandoffLog` already did (Phase 31), the content-addressing that
`MerkleDAG` / `ContextLedger.put` already did (Phase 19), and the
provenance manifest that every run already carried are the existing
evidence; SDK v3 recognises that they were instances of one shared
thing.

> **Naming.** `Context Zero` is the research programme; `Wevra` is the first
> finished product produced by it. The original substrate proposal — **CASR**
> (Causal-Abstraction Scale-Renormalized Routing) — lives in
> `vision_mvp.core.*` as research-grade code and grounds Wevra's O(log N)
> bounded-context claim (Theorem 3 in
> [`docs/archive/pre-wevra-theory/PROOFS.md`](docs/archive/pre-wevra-theory/PROOFS.md)).
> The programme's phase-by-
> phase diary lives in `vision_mvp/RESULTS_PHASE*.md`; the Wevra SDK boundary
> lives under `vision_mvp/wevra/` and is the stable public contract.
>
> **Wevra SDK boundary (Slice 1 + v3 + v3.1).** The stable public
> surface is: `RunSpec` (with `capsule_native: bool = True`),
> `run`, `WevraConfig`, `profiles`, `report`, `ci_gate`,
> `import_data`, `build_manifest`, the capsule primitives
> (`ContextCapsule`, `CapsuleLedger`, `CapsuleView`,
> `build_report_ledger`, every `capsule_from_*` adapter),
> the capsule-native runtime symbols (`CapsuleNativeRunContext`,
> `seal_and_write_artifact`, `ContentAddressMismatch`,
> `CONSTRUCTION_IN_FLIGHT`, `CONSTRUCTION_POST_HOC`), and the schema
> constants (`wevra.provenance.v1`, `phase45.product_report.v2`,
> `wevra.capsule_view.v1`, `phase46.ci_verdict.v1`,
> `phase46.import_audit.v1`). See the **Stability matrix** in
> `README.md` and in `docs/context_zero_master_plan.md` for the
> durable classification of every layer (Wevra SDK · capsule
> primitives · capsule-native runtime · core substrate · legacy
> product path · plugin/extension system · unified runtime ·
> Docker sandbox · research shards). Anything not on the SDK
> surface is research-grade or boundary/next-slice and may change
> without notice.

> **How to read the rest of this file.** The phase-by-phase
> callouts immediately below (Phases 26 → 44) are a *historical
> incremental record* of how the substrate was built up. They are
> kept verbatim for provenance; each claim is anchored to a
> `RESULTS_PHASE*.md` note and to tests. If you want the durable
> architecture, skip the phase callouts and read the layered
> substrate diagram further down (five substrate layers + render
> mode + runtime calibration + typed-handoff team layer), then § 3
> ("Architecture of the solution") in
> [`docs/context_zero_master_plan.md`](docs/context_zero_master_plan.md).
> For the Wevra product surface specifically, see § 10 of the master
> plan and [`docs/START_HERE.md`](docs/START_HERE.md).

> **Architecture as of Phase 27: five substrate layers + a render
> mode + a snippet-scale runtime-calibration observer (Phase 26) +
> a *corpus-scale* runtime-calibration observer (Phase 27). Conservative
> intraprocedural + interprocedural semantic code analysis sits in
> the ingestion layer; the new runtime-calibration layer observes
> instrumented execution of the same code against a per-predicate
> probe set and reports the analyzer-vs-runtime divergence matrix.**
> The original CASR spec (below) covers the *routing* and *trigger*
> layers. Phases 19–21 added three more layers — *exact external
> memory*, *retrieval*, and *computation/planning* — that handle
> the content and aggregation sides of context. Phase 22 generalised
> the substrate to real Python codebases (AST-derived typed
> metadata) and introduced the **direct-exact** render path that
> bypasses the LLM when the planner has the answer. Phase 23
> validated the Phase-22 result across six real Python corpora
> (research / utility / test / CLI-framework / stdlib) with a
> reusable multi-corpus registry. Phase 24 extended the direct-
> exact guarantee from syntactic structure to conservative
> *intraprocedural* static-semantic predicates (`may_raise`,
> `is_recursive`, `may_write_global`,
> `calls_subprocess`/`filesystem`/`network`), computed from the AST
> by `core/code_semantics`; direct-exact scored 44 / 44 (100 %, σ
> = 0) on the semantic battery across six corpora. Phase 25
> extended the exact slice to conservative *interprocedural*
> semantic predicates — transitive closures over a local call graph
> plus Tarjan-SCC recursion-cycle detection — via
> `core/code_interproc`; direct-exact scored **50 / 50 (100 %, σ =
> 0)** on the Phase-25 interprocedural battery across the same six
> corpora with zero LLM calls and zero prompt chars. **Phase 26
> introduces a separate truth axis — *runtime-truth calibration* —
> via instrumented probes that observe how a function actually
> behaves when executed. The runtime layer is ADDITIVE: it does
> NOT replace the analyzer or planner; it sits alongside them as
> an observer that reports analyzer-vs-runtime divergence per
> predicate. On a 21-snippet executable corpus spanning 8 families,
> the analyzer agrees with runtime observation on 123 / 126
> (97.6 %) applicable measurements; every divergence lands on a
> Phase-24 pre-documented boundary condition. Analyzer-gold
> exactness and runtime-truth calibration are formalised as
> independent axes (Theorem P26-1); the direct-exact planner's
> 126 / 126 round-trip to the analyzer demonstrates the substrate
> guarantee is independent of analyzer calibration.**
>
> **Phase 28 extends the runtime-calibration axis along two
> orthogonal dimensions: (a) runtime calibration is run over the
> full local Phase-23 corpus set (`vision-core`, `vision-tasks`,
> `vision-tests`, `vision-experiments`) with coverage reported
> as a first-class cross-corpus variable (`ready_fraction` ranges
> from 2.9 % to 80.2 %), and (b) the analyzer's `may_raise` axis
> is split — the Phase-24 contract is preserved unchanged as
> `may_raise_explicit` (sound, FN = 0 across all four corpora),
> and a new conservative sound-over-precision predicate
> `may_raise_implicit` is added for implicit-raise propagation
> from builtin operations (soundness: FN = 1 / 116 runtime-
> positives on the pooled entered slice). The substrate layer is
> unchanged — Phase 28 touches the analyzer (`code_semantics`,
> `code_interproc`), the runtime observer (`code_runtime_calibration`,
> `code_corpus_runtime`), and adds the benchmark
> `phase28_multi_corpus_runtime_calibration`. See Theorems
> P28-1..P28-4 in `RESULTS_PHASE28.md`.**
>
> **Phase 29 adds two couples-but-independent pieces. First, a
> task-scale causal-relevance harness (`tasks/task_scale_swe` +
> `experiments/phase29_task_scale_falsifiability`) that runs the
> routing / substrate stack over a multi-role SWE-style task
> distribution and measures, per (task, role, event), whether the
> event is *causally relevant* under an analyzer-derived oracle.
> On 80 queries / 5 718 events across four corpora, the pooled
> aggregator-role causal-relevance fraction under naive broadcast
> is **4.54 %**; the substrate collapses aggregator context by
> **1 007×** at **100 %** correctness on matched tasks. This is
> the first task-scale test of the core thesis; falsifiability
> decision on the ROADMAP gate: **CONFIRMED** (Theorems P29-1 /
> P29-2 / P29-3 / P29-4). Second, a conservative method-instance
> auto-construction recipe (extends `code_corpus_runtime`):
> methods on safely-zero-arg-constructable classes (no custom
> `__init__`, or `__init__` with only self + defaulted params,
> or `@dataclass`-all-defaulted) promote to a new `ready_method`
> status; the probe constructs the instance under the Phase-26
> sandbox + Phase-27 budget tracer. Runtime `ready_fraction` on
> `vision-tests` lifts 2.9 % → 98.8 %; pooled entered slice grows
> 4.83× (306 → 1 477) with `may_raise_explicit` FN preserved at 0
> and construct-failed < 1 % (Theorem P29-5). The substrate layer
> is unchanged; Phase 29 touches `code_corpus_runtime` (method
> coverage) and adds the task-scale harness. See Theorems
> P29-1..P29-8 in `RESULTS_PHASE29.md`.**
>
> **Phase 31 adds a new substrate layer on the *team-communication*
> axis — typed, content-addressed, role-scoped handoffs between
> agents — and ships the programme's first *non-code* task-scale
> benchmark. The new module (`core/role_handoff.py`) provides
> `TypedHandoff`, `RoleSubscriptionTable`, bounded `RoleInbox`,
> hash-chained `HandoffLog`, per-(source_role, to_role,
> claim_kind) `DeliveryAccount`, and a `HandoffRouter`. The layer
> sits one level above the Phase-1/29 role-keyed Bloom routing: it
> routes by *claim kind* (e.g. `SLOW_QUERY_OBSERVED`,
> `DISK_FILL_CRITICAL`), so downstream roles can subscribe to
> load-bearing content without reading the payload. The companion
> benchmark (`tasks/incident_triage`) runs a five-role operational
> incident-triage team across five scenario kinds and four
> delivery strategies; substrate prompt size is **flat at 196
> tokens** across distractor densities k ∈ {6, 20, 60, 120}
> (event-stream 40 → 440 events), while naive collapses from 100 %
> → 20 % at k=120 under truncation. Theorems P31-1..P31-5 + two
> conjectures formalise the role-conditioned relevance
> factorisation, communication-sparsity lower bound, bounded-
> context upper bound, correctness preservation under subscription
> coverage, and a provable separation from any single-agent
> compression of the event stream (P31-5). See Theorems
> P31-1..P31-5 in `RESULTS_PHASE31.md`.**
>
> **Phase 39 adds a multi-role SWE-bench-style bridge
> *strictly above* the Phase-31 typed-handoff substrate
> and ships the first real-LLM data point on the
> Phase-38 prompt-variant pipeline:
> (a) `tasks/swe_bench_bridge` — a `SWEBenchStyleTask`
> schema that mirrors the public SWE-bench instance shape
> (`instance_id`, `repo`, `base_commit`,
> `problem_statement`, `gold_patch`, `test_source`); a
> four-instance hand-authored `MiniSWEBank` whose patches
> are line-anchored substitutions and whose hidden tests
> run in a fresh `exec` namespace (no shell, no
> subprocess, no network); a four-role team
> (`issue_reader` / `code_searcher` / `patch_generator`
> / `test_runner`) wired through the unchanged Phase-31
> `HandoffRouter`; a `SWEBenchAdapter.from_dict` shim
> documenting the schema mapping for a future real-
> SWE-bench loader. **Theorem P39-3** (substrate
> bounded-context preservation) — the patch_generator's
> prompt size is independent of `n_distractors` (842
> chars at every distractor count) while naive grows
> from 949 → 1936; **Theorem P39-4** (schema
> mappability) — the gap to public SWE-bench is
> adapter-shaped, not architectural.
> (b) `experiments/phase39_swe_bridge` — a runnable
> driver supporting `--mode mock` (deterministic oracle
> generator; sub-second) and `--mode real` (Ollama LLM
> patch generator).
> (c) `experiments/phase39_frontier_substrate` — a
> bounded cross-family sweep on Phase-31 incident triage
> across `llama3.1:8b`, `gemma2:9b`, `qwen2.5-coder:7b`.
> (d) Real-LLM data point on the Phase-38 prompt
> calibration pipeline (the existing
> `phase38_prompt_calibration --mode real` driver).
> **Theorem P39-1**: on `qwen2.5:0.5b`, four of five
> Phase-38 variants reproduce the Phase-37 default
> distribution to within ±0 calls; the bias is
> *model-shaped, not prompt-shaped* on this size class.
> **Theorem P39-2** (regime taxonomy): every team-
> shaped task admits a *communication-bounded* vs
> *transcription-bounded* decomposition; the substrate
> is the gating constraint only when the synthesis
> layer is order-preserving on the typed bundle. No
> Phase-31 through Phase-38 primitive is modified. See
> `RESULTS_PHASE39.md`.**
>
> **Phase 43 adds a semantic-failure taxonomy layer
> *strictly above* the Phase-42 parser-compliance layer,
> a public-style loader self-test, a frontier-model run
> (``qwen3.5:35b`` 36B-MoE on the ASPEN cluster), and one
> byte-safe trailing-delimiter pattern added to the Phase-42
> ``_strip_trailing_prose`` list.** Four coupled additions,
> all *strictly above* the Phase-42 layer (every Phase-42
> default preserves Phase-42 byte-for-byte):
> (a) ``vision_mvp/tasks/swe_semantic_taxonomy.py`` (NEW) —
> nine-label closed vocabulary (``SEM_OK`` / ``SEM_PARSE_FAIL``
> / ``SEM_WRONG_EDIT_SITE`` / ``SEM_RIGHT_SITE_WRONG_LOGIC``
> / ``SEM_INCOMPLETE_MULTI_HUNK`` / ``SEM_TEST_OVERFIT`` /
> ``SEM_STRUCTURAL_SEMANTIC_INERT`` / ``SEM_SYNTAX_INVALID``
> / ``SEM_NO_MATCH_RESIDUAL``) with a pure deterministic
> classifier and ``SemanticCounter`` aggregator. Sits
> strictly above the Phase-42 parser-compliance counter in
> the analysis stack.
> (b) ``vision_mvp/experiments/phase43_frontier_headroom.py``
> (NEW) — Phase-43 analysis driver. Ingests Phase-42-shape
> artifacts, re-derives per-cell semantic labels, emits
> cross-model summary JSON. Includes
> ``verify_public_style_loader`` that round-trips every
> bank instance through the loader + strict matcher under
> the oracle (57/57 saturation on the bundled bank).
> (c) ``vision_mvp/core/llm_client.py`` (EXTENDED) —
> ``LLMClient(think=…)`` threads Ollama's ``/api/generate``
> ``think`` field for Qwen3-class thinking models so their
> output budget is not consumed by internal reasoning.
> Default ``None`` preserves Phase-42 byte-for-byte.
> (d) ``vision_mvp/tasks/swe_patch_parser.py`` (one-pattern
> regression fix) — ``_PROSE_TAILS`` gains one pattern
> ``\n\s*<{2,4}\s*\Z`` that strips partial / full trailing
> delimiters (``<<``, ``<<<``, ``<<<<``). Surfaced by the
> ``qwen3.5:35b`` cluster run's unclosed_new failure shape.
> Byte-safe under Theorem P42-2.
>
> **Phase 43 theory**: Theorem P43-1 (bounded-context
> preservation on the external-validity bank — substrate
> 205.9 tokens flat across the full
> parser × matcher × distractor cross product); Theorem
> P43-2 (post-parser-recovery semantic residue is
> structurally classifiable — nine-label taxonomy is total,
> exhaustive, deterministic); Theorem P43-3 (semantic-ceiling
> separation on coder-finetuned models at N ≥ 50 —
> substrate-vs-naive gap is 0 pp on every measured
> coder-finetuned model, per-strategy failure-mix
> histograms are byte-identical, and the dominant residue
> label is ``SEM_WRONG_EDIT_SITE`` on coder-finetuned
> models vs ``SEM_SYNTAX_INVALID`` on general-purpose
> models of matched parameter class). Four conjectures
> (C43-1..C43-4). The programme's durable substrate claim
> is now unambiguous: *bounded active context per role*, not
> pass@1 lift. See ``RESULTS_PHASE43.md``.
>
> **Phase 44 adds raw-text residue capture, a refined semantic
> taxonomy (v2 classifier), and a validated public-SWE-bench-
> Lite drop-in readiness pipeline — *strictly above* the
> Phase-43 analysis layer.** Four coupled additions, all
> strictly additive (every Phase-43 default preserves
> Phase-43 byte-for-byte):
> (a) ``vision_mvp/tasks/swe_raw_capture.py`` (NEW) —
> ``RawCaptureRecord`` / ``RawCaptureStore`` with schema
> version ``phase44.v1``. Each record persists the raw LLM
> bytes + SHA-256, the ``ParseOutcome`` dict, the proposed
> substitutions, the applied substitutions after the matcher,
> the patched-source SHA-256, and the downstream verdict.
> ``make_capturing_generator`` wraps a bridge generator or a
> fresh ``llm_call`` and plumbs raw text into the store
> while preserving the Phase-42 LLM-output cache discipline.
> (b) ``vision_mvp/tasks/swe_semantic_taxonomy.py``
> (EXTENDED) — five new sub-labels
> (``SEM_RIGHT_FILE_WRONG_SPAN``, ``SEM_RIGHT_SPAN_WRONG_LOGIC``,
> ``SEM_PARTIAL_MULTI_HUNK_SUCCESS``,
> ``SEM_NARROW_FIX_TEST_OVERFIT``, ``SEM_STRUCTURAL_VALID_INERT``)
> partition the Phase-43 coarse buckets when raw bytes are
> available. ``classify_semantic_outcome_v2`` subsumes the v1
> classifier on sentinel inputs (Theorem P44-2).
> ``REFINEMENT_MAP`` is reflexive so the sentinel path
> remains a legal v2 classification.
> (c) ``vision_mvp/experiments/phase44_semantic_residue.py``
> (NEW) — sweep mode runs the Phase-42-shape experiment with
> raw capture on; analyse-only mode consumes (parent,
> capture) pairs and emits a ``phase44.summary.v1`` JSON
> with per-cell coarse + refined counters and a
> ``coarse_to_refined_partition`` audit.
> (d) ``vision_mvp/experiments/phase44_public_readiness.py``
> (NEW) — five-check CI-gate validator (schema / adapter /
> parser / matcher / test_runner) on any local JSONL.
> Emits ``{"ready": true, "n": 57, ...}`` on the bundled
> bank in ~5 s wall (Theorem P44-3).
>
> **Phase 44 theory**: Theorem P44-1 (raw capture is a
> lossless projection of pipeline state); Theorem P44-2
> (refined classifier is monotone on sentinel inputs —
> backwards-compatibility with Phase 43 is a theorem, not an
> aspiration); Theorem P44-3 (public-readiness saturates on
> the bundled bank at external-validity scale — the
> externalisation gap is now purely data-availability).
> Four conjectures (C44-1..C44-4) frame the sharper
> residue-composition questions raw capture makes
> measurable. See ``RESULTS_PHASE44.md``.
>
> **Phase 42 adds the parser-compliance attribution layer
> on top of the Phase-41 matcher axis and grows the
> SWE-bench-Lite-style bank past the ≥ 50-instance
> external-validity threshold.** Three coupled additions,
> all *strictly above* the Phase-41 layer (every Phase-41
> default preserves Phase-41 byte-for-byte):
> (a) `tasks/swe_patch_parser` (NEW) — a
> `parse_patch_block(text, mode, unified_diff_parser)`
> entry point with three modes (`PARSER_STRICT` = Phase-41
> baseline; `PARSER_ROBUST` = Phase-42 default with five
> named recovery heuristics; `PARSER_UNIFIED` = diff-only),
> a closed ten-label failure taxonomy (`PARSE_OK`,
> `PARSE_EMPTY_OUTPUT`, `PARSE_NO_BLOCK`,
> `PARSE_UNCLOSED_NEW`, `PARSE_UNCLOSED_OLD`,
> `PARSE_MALFORMED_DIFF`, `PARSE_EMPTY_PATCH`,
> `PARSE_MULTI_BLOCK`, `PARSE_PROSE_ONLY`,
> `PARSE_FENCED_ONLY`), and a six-label recovery enum
> (`RECOVERY_NONE`, `RECOVERY_CLOSED_AT_EOS`,
> `RECOVERY_FENCED_CODE`, `RECOVERY_LABEL_PREFIX`,
> `RECOVERY_UNIFIED_DIFF`, `RECOVERY_LOOSE_DELIM`).
> `ParserComplianceCounter` exposes `compliance_rate` /
> `raw_compliance_rate` / `recovery_lift` per cell.
> (b) `tasks/swe_bench_bridge` (EXTENDED) —
> `llm_patch_generator(..., parser_mode=…,
> parser_counter=…, prompt_style=…)` routes the parser axis
> from the bridge boundary; `None` preserves the Phase-41
> regex byte-for-byte. `build_patch_generator_prompt(…,
> prompt_style="block" | "unified_diff")` opts into a
> unified-diff output contract. Re-exports
> `parse_patch_block` / `ParseOutcome` /
> `ParserComplianceCounter` so one import gives the
> caller the full Phase-42 surface.
> (c) `tasks/data/swe_lite_style_bank.jsonl`
> (REGENERATED) — the Phase-41 28-instance bank grown
> with 29 new instances covering string manipulation,
> numeric guards, sequence construction, dict helpers,
> recursion/iteration, exception handling, set algebra,
> class state transitions (`StopLight` multi-hunk,
> `Stack.pop`), binary search off-by-one, graph walk
> reachability, and default argument correction. Every
> new instance validated via the same oracle-round-trip
> precondition as Phase 41.
> (d) `core/llm_client` (EXTENDED) —
> `LLMClient(base_url=None)` plumbs the ASPEN cluster
> endpoints (macbook-1 `http://192.168.12.191:11434`,
> macbook-2 `http://192.168.12.248:11434`); default
> `None` preserves the Phase-41 localhost semantics.
> (e) `experiments/phase42_parser_sweep` (NEW) — sweeps
> `(parser_mode × apply_mode × n_distractors)` with an
> LLM-output cache keyed per
> `(instance_id, strategy_proxy, n_distractors,
> prompt_style)` so the parser-mode axis re-parses
> cached text; emits the per-strategy
> `{recovered, regressed, unchanged_pass,
> unchanged_fail}` set delta between strict and each
> non-strict parser. **Theorem P42-1** (parser-compliance
> attribution: `Δ pass@1 = |R_recovered_parser| −
> |R_regressed_parser|` under every matcher × strategy ×
> distractor cell; promotes Conjecture C41-5 to theorem).
> **Theorem P42-2** (parser recovery cannot produce a
> false pass — byte-provenance argument). **Theorem
> P42-3** (robust parser dominates on format-
> noncompliant generators). Combined with Theorem P41-3
> and Theorem P39-2, the programme now has a
> **three-axis attribution surface**
> (parser × matcher × substrate). Phase-42 mock
> reproduces Theorem P41-1 on the 57-instance bank
> (substrate prompt 205.9 tokens flat, naive 197 → 527,
> 1 368 sandboxed measurements in 122 s). See
> `RESULTS_PHASE42.md`.
>
> **Phase 41 moves the Phase-40 real SWE loop to first
> larger-N data with a two-axis attribution surface.**
> Three coupled additions, all *strictly above* the
> Phase-40 layer (every Phase-40 artifact reruns
> byte-for-byte under the Phase-41 defaults):
> (a) `tasks/data/swe_lite_style_bank.jsonl` (NEW) —
> a 28-instance SWE-bench-Lite-shape JSONL bank
> (~4.7× the Phase-40 mini bank) covering a disciplined
> spectrum of edit shapes: operator-typo, off-by-one,
> wrong-branch, seed-wrong, aggregate-missing, mutation-
> vs-copy, multi-hunk (one class touches two methods),
> parity-partition, slice-direction, index-return,
> polarity-flipped, empty-guard, type-conversion,
> unicode edge, ambiguous comparator. A bank-builder
> (`_build_swe_lite_bank.py`) round-trips every instance
> through `parse_unified_diff + apply_patch +
> run_patched_test` before writing; refuses to register
> any instance whose diff doesn't parse, whose OLD blocks
> aren't uniquely anchored, or whose oracle-patched
> source doesn't pass the hidden test. The JSONL is the
> reproducibility precondition: Phase-41 evaluation runs
> offline in seconds.
> (b) `tasks/swe_bench_bridge` + `tasks/swe_sandbox`
> (EXTENDED) — `apply_patch` accepts an `apply_mode`
> kwarg ∈ {`strict` (default, Phase-40 byte-exact),
> `lstrip` (leading-whitespace drift tolerance),
> `ws_collapse` (internal-whitespace drift),
> `line_anchored` (trailing-whitespace drift)}. All three
> permissive modes retain **unique-match discipline**
> (a normalised OLD that appears more than once in the
> normalised source is rejected as `old_ambiguous`).
> `apply_mode` is threaded through `run_swe_loop`,
> every `Sandbox.run(...)` backend, and
> `run_swe_loop_sandboxed`; `SWEReport.config` records
> it for audit.
> (c) `experiments/phase41_swe_lite_sweep` (NEW) — the
> attribution-aware driver. Caches each LLM call per
> `(instance_id, strategy, n_distractors)` so permissive
> cells reuse strict cells' proposals (no extra LLM
> wall on the matcher axis); emits a per-strategy
> `{recovered, regressed, unchanged_pass,
> unchanged_fail}` set delta between each permissive
> mode and the strict baseline. **Theorem P41-1**
> (bounded-context preservation at 4.7× scale —
> substrate 746.4 chars flat, naive 806.8 → 2 125.8
> across `n_distractors ∈ {0, 6, 12, 24}` on 672
> sandboxed measurements). **Theorem P41-2** (oracle-
> ceiling is matcher-mode-invariant — permissive
> matching subtracts no correctness from a byte-exact
> generator). **Theorem P41-3** (matcher-permissiveness
> attribution decomposition: `Δ pass@1 = |R_recovered|
> − |R_regressed|`). Combined with Theorem P39-2, the
> programme now has a **two-axis attribution surface**
> for any real SWE loop — substrate delivery × matcher
> precision. Real-LLM sweeps on `qwen2.5-coder:7b`
> (28 instances) and `gemma2:9b` (subset) populate the
> attribution tables. See `RESULTS_PHASE41.md`.
>
> **Phase 40 makes the Phase-39 SWE bridge a real
> external task loop.** Three coupled additions, all
> *strictly above* the Phase-39 schema layer:
> (a) `tasks/swe_bench_bridge` extension —
> `parse_unified_diff` (a tolerant `git diff` parser),
> `SWEBenchAdapter.from_swe_bench_dict` (the real-shape
> adapter that derives `buggy_function` from the diff
> hunk and promotes a `test_patch` into a runnable
> `test_source`), `load_jsonl_bank` (hermetic JSONL
> loader with per-instance file namespacing), and a
> bundled six-instance JSONL artifact
> (`vision_mvp/tasks/data/swe_real_shape_mini.jsonl`).
> (b) `tasks/swe_sandbox` (NEW) — a `Sandbox` protocol
> with three backends: `InProcessSandbox` (Phase-39
> wrapped), `SubprocessSandbox` (new — wall-clock
> timeout, tempdir cwd, sanitised env, JSON outcome
> protocol so test-level vs sandbox-level failures are
> attributable), `DockerSandbox` (new — optional;
> `--network=none --read-only` rootfs, `tmpfs /work`,
> `--stop-timeout`). `select_sandbox("auto")` picks
> Docker → subprocess → in-process by availability;
> `run_swe_loop_sandboxed` is the sandbox-aware
> substrate runner.
> (c) `experiments/phase40_real_swe_bridge` (NEW) — the
> end-to-end driver that composes loader + substrate +
> sandbox + (optional) real LLM patch generator. Mock
> run: 72 sandboxed measurements, pass@1 = 1.000 on
> every (strategy, distractor) cell. Real-LLM runs:
> qwen2.5:0.5b (transcription-bounded, every cell hits
> patch_no_match) and qwen2.5-coder:7b (5/6 under
> naive/routing, 4/6 under substrate — honest variance
> at small N inside the P39-2 transcription-bounded
> regime). **Theorem P40-1** (unidiff round-trip),
> **Theorem P40-2** (real-shape substrate bounded-
> context preservation — substrate prompt 813 chars
> across n_distractors ∈ {0, 6, 12, 24}; naive grows
> 826 → 2 145), **Theorem P40-3** (sandbox-boundary
> preservation — InProcessSandbox and SubprocessSandbox
> deliver pass@1 = 1.000 on the oracle ceiling on the
> mini bank and the real-shape JSONL bank). The
> external-validity gap to public SWE-bench is now
> *empirical*, not infrastructural. See
> `RESULTS_PHASE40.md`.**
>
> **Phase 38 extends the coordination-primitive layer with
> four composition-level additions that close the two-layer
> ensemble, minimum-primitive-ablation, and prompt-variant
> frontier items named by Phase 37's conjectures:
> (a) `core/two_layer_ensemble` — a
> `PathUnionCausalityExtractor` with three combiner modes
> (`path_dual_agree` / `path_union_root` / `path_verified`)
> that sits strictly above any per-path noise wrapper, plus
> a `TwoLayerDefense` descriptor record. Theorem P38-2
> shows that `path_union_root` closes the Phase-37
> `adv_drop_root` cell where every reply-axis ensemble
> alone is powerless. (b) `core/extractor_adversary` —
> a `DropGoldClaimExtractor` adversarial layer-1 wrapper,
> a deterministic `NarrativeSecondaryExtractor` that
> catches dropped claims via service-tag matching, and a
> `UnionClaimExtractor` bridging the two. Theorem P38-1:
> the composition
> `UnionClaimExtractor ∘ EnsembleReplier(MODE_DUAL_AGREE)`
> is the unique configuration that recovers the joint
> layer-1 + layer-2 attack on the Phase-35 bank.
> (c) `core/primitive_ablation` — feature-flagged
> `AblatedFeatures` and thread runners (`run_ablated_thread_
> contested`, `run_ablated_thread_nested`) that toggle each
> of {`typed_vocab`, `bounded_witness`,
> `terminating_resolution`, `round_aware_state`,
> `frozen_membership`}. Theorem P38-3 presents the
> ablation-table falsifier for Phase-37 Conjecture C37-4.
> (d) `core/prompt_variants` — five surgical prompt
> variants (default, contrastive, few_shot, rubric,
> forced_order) + a `build_thread_reply_prompt_variant`
> dispatcher + a `VariantLLMThreadReplier` wrapper. Every
> variant preserves the Phase-36 typed-reply contract
> (allowed kinds, witness cap, UNCERTAIN fallback). A
> sibling `core/two_layer_ensemble` addition — `TwoLayer
> Defense` — is a descriptor record that records which
> layers are active for reporting. One surgical addition
> to `tasks/contested_incident`: an optional
> `claim_extractor` parameter on the handoff-protocol
> runners so Phase-38 layer-1 adversaries compose without
> modifying the Phase-35 decoder. No Phase-31 through
> Phase-37 primitive is modified. See `RESULTS_PHASE38.md`.**
>
> **Phase 37 extends the coordination-primitive layer with
> three composition-level additions, strictly above the
> Phase-36 reply primitives:
> (a) `core/reply_calibration` — a `CalibratingReplier` that
> wraps any `LLMThreadReplier` with a per-call oracle
> comparator and records every call into a 9-bucket
> correctness taxonomy (correct / malformed / oov / six
> semantic confusions) plus an orthogonal
> `witness_truncated` counter (Theorem P37-1: real LLMs
> produce 100 % well-formed JSON but 90 % semantic
> mislabel — the Phase-36 synthetic `malformed_prob` knob
> is a near-useless surrogate). (b) `core/reply_ensemble`
> — three pluggable ensemble modes (`dual_agree` AND-gated
> parallel; `primary_fallback` chatty-primary + fallback;
> `verified` primary + deterministic verifier), all
> matching the `LLMThreadReplier` shape so they drop into
> `causality_extractor_from_replier`. Theorems P37-2
> (biased-primary recovery), P37-3 (syntactic-noise
> recovery), P37-4 (structural limit — ensembles cannot
> recover extractor-output-level noise applied below
> them). (c) `tasks/nested_contested_incident` — a harder
> task family where round-1 replies are insufficient; a
> two-round thread harness (`run_nested_two_round_thread`)
> and a two-round adaptive-sub harness
> (`run_nested_two_round_adaptive_sub`) that uses a new
> `CLAIM_COORDINATION_BRIEFING` kind for inter-round
> auditor-to-producer briefings. Theorem P37-5: accuracy
> equivalence EXTENDS to nested contests at 0 pp gap, but
> the thread uses 0 inter-round briefings while
> adaptive_sub_2r uses 18 — a structural-complexity
> separation beneath the accuracy equivalence. No Phase-35
> or Phase-36 primitive is modified. See
> `RESULTS_PHASE37.md`.**
>
> **Phase 36 extends the dynamic-coordination layer with three
> sibling modules at the coordination-primitive layer (above
> `core/role_handoff`, parallel to `core/dynamic_comm`):
> (a) `core/reply_noise` — parameterised Bernoulli drop /
> mislabel wrappers and an adversarial reply wrapper targeting
> the gold `INDEPENDENT_ROOT` reply on a per-scenario budget
> (Theorems P36-1 graceful i.i.d. degradation and P36-2
> targeted-adversarial collapse). (b) `core/llm_thread_replier`
> — an `LLMThreadReplier` that drives a narrow, bounded LLM
> call per (producer, candidate) and returns a typed reply
> filtered against the Phase-35 reply-kind enum (Theorem P36-3
> LLM-replier substitutivity). (c) `core/adaptive_sub` — a
> bounded, TTL-expiring subscription-edit primitive
> (`AdaptiveSubscriptionTable` + `AdaptiveSubRouter` +
> `AdaptiveEdge`) offered as a serious comparison point to the
> Phase-35 escalation thread (Theorem P36-4 empirical
> equivalence). On the Phase-35 contested bank × the Phase-36
> noise × k × seed grid (96 paired measurements), the
> dynamic-thread vs adaptive-sub accuracy gap is 0.000 pp at
> every cell; token overhead is +12 %. The Phase-35 primitive
> is unchanged byte-for-byte. See `RESULTS_PHASE36.md`.**
>
> **Phase 35 adds a single new substrate layer strictly above
> Phase 31's typed-handoff layer and strictly below any
> unrestricted group-chat layer: the *escalation thread*
> (`core/dynamic_comm.EscalationThread` +
> `ThreadReply` + `ThreadResolution` + `DynamicCommRouter`). A
> thread has a frozen member set, a typed `issue_kind`
> (`RESOLVE_ROOT_CAUSE_CONFLICT` / `RESOLVE_SEVERITY_CONFLICT` /
> `RESOLVE_VERDICT_QUORUM` / `CONFIRM_CLAIM`), a bounded tuple
> of candidate claims, and three bounded budgets: `max_rounds`,
> `max_replies_per_member`, `witness_token_cap`. Member roles
> post typed replies from a small enumerated vocabulary
> (`INDEPENDENT_ROOT` / `DOWNSTREAM_SYMPTOM` / `UNCERTAIN` /
> `AGREE` / `DISAGREE` / `DEFER_TO`); the thread closes on
> quorum-on-agree, max-round exhaustion, or explicit opener
> close. The thread's single public output is a
> `CLAIM_THREAD_RESOLUTION` handoff routed through the
> unchanged Phase-31 `HandoffRouter`; thread-internal events
> (`THREAD:OPEN` / `THREAD:REPLY` / `THREAD:CLOSE`) are hash-
> chained in the existing `HandoffLog` for audit but never
> enter non-member inboxes (Theorem P35-4). Bounded-context is
> preserved with an additive `T·R_max·W` per role per round
> (Theorem P35-2), independent of |X|. The companion benchmark
> (`tasks/contested_incident`) runs a 6-scenario bank — 4
> contested root-cause pairs where static priority is
> inverted — showing the dynamic strategy at 100 % contested
> accuracy (flat at 246 tokens) vs static handoffs at 0 %
> contested accuracy (Theorem P35-1 separation). See Theorems
> P35-1..P35-4 + Conjectures C35-5, C35-6 in
> `RESULTS_PHASE35.md`.**
>
> **Phase 34 extends Arc 8 with (a) per-role-adaptive calibration
> (`core/extractor_calibration.per_role_audit_summary` +
> `core/extractor_noise.PerRoleNoiseConfig` +
> `per_role_noisy_extractor`): the pooled quadruple is now
> decomposed into per-role (δ̂_k, ε̂_k, μ̂_k, π̂_k) with a
> *limiting-role* argmax; on Phase-34's mock benchmark the per-role
> drop-rate spread is ≥ 0.33 across all three domains, confirming
> Conjecture C33-3's "pooled i.i.d. hides structure" on every
> domain; (b) an adversarial extractor wrapper
> (`core/extractor_noise.adversarial_extractor`) with three target
> modes — load-bearing claim drop with priority ordering, role
> silencing, severity-escalation injection — that provably beats
> i.i.d. at matched nominal budget (Theorem P34-2: at budget = 1 on
> all three domains the adversary collapses substrate accuracy to
> 0 % while matched i.i.d. preserves 20 %–80 %, gap +0.47 pp pooled);
> (c) the programme's first meaningful regex + LLM ensemble result
> (`core/ensemble_extractor.UnionExtractor`) on a compliance
> *mixed* bank (5 canonical + 5 narrative scenarios where regex and
> LLM have genuinely complementary coverage): regex 50 % / LLM 0 % /
> ensemble 100 % at pooled δ_u = 0.00 ≤ δ_r · δ_l = 0.188 —
> Conjecture C33-4 promoted to Theorem P34-3; (d) three theorems
> (P34-1 role-limited accuracy; P34-2 adversarial-vs-iid separation;
> P34-3 ensemble union lower bound) and two conjectures (C34-4
> typed-handoff ensemble-vs-adversary; C34-5 per-role replay as
> tighter predictor than pooled). The substrate primitive
> (`core/role_handoff`) remains byte-unchanged. See
> `RESULTS_PHASE34.md`.**
>
> **Phase 33 extends Arc 8 with (a) an LLM-driven extractor path
> (`core/llm_extractor`) — a drop-in replacement for any
> Phase-31/32 regex extractor that calls a
> ``Callable[[str], str]`` LLM per (role, scenario) boundary,
> parses the reply into typed ``(kind, payload, evids)`` tuples,
> and filters against ``known_kinds_by_role`` so the substrate's
> type-safety invariants are preserved under hallucination — the
> substrate primitive (`core/role_handoff`) is unchanged
> byte-for-byte; (b) a real-vs-synthetic noise calibration layer
> (`core/extractor_calibration`) that measures the empirical
> ``(δ̂ drop, ε̂ spurious, μ̂ mislabel, π̂ payload-corrupt)``
> quadruple against a gold causal chain and maps it to the
> closest Phase-32 synthetic sweep point — ``qwen2.5:0.5b`` on
> compliance review is 0.70 / 0.12 / 0.40 / 0.60, Phase-32
> closest-match predicts substrate accuracy / recall / precision
> within max-abs-gap 0.10 ⇒ verdict "approximates"; (c) a *third*
> non-code domain — security-audit escalation
> (`tasks/security_escalation`) — with a five-role cast (SOC /
> IR / threat intel / data steward / CISO), 15 claim kinds, and a
> novel **max-ordinal severity + claim-set classification**
> decoder (structurally distinct from Phase 31 priority-order and
> Phase 32 monotone-verdict shapes). Substrate flat at 242
> tokens / 100 % accuracy across k ∈ {6, 20, 60, 120}; naive
> collapses 100 % → 20 % at k = 120 under truncation; (d) three
> theorems (P33-1 LLM-extractor subsumption under the Phase-32
> sweep; P33-2 cross-domain correctness at K = 3; P33-3
> two-regime bound on max-ordinal decoders) and two conjectures
> (C33-3 role-heterogeneous noise; C33-4 ensemble-extractor
> composition). See `RESULTS_PHASE33.md`.**
>
> **Phase 32 extends Arc 8 with (a) a second non-code domain —
> vendor-onboarding compliance review (`tasks/compliance_review`)
> with a distinct role cast (legal / security / privacy / finance
> / compliance officer) and a priority-monotone-verdict + strict-
> set-flags decoder — that confirms the substrate's behaviour is
> domain-agnostic (substrate flat at 171 tokens / 100 % accuracy
> across k ∈ {6, 20, 60, 120}, same signature as Phase 31); (b) a
> parameterised extractor-noise module (`core/extractor_noise`)
> with five noise axes (drop / spurious / mislabel /
> payload_corrupt / seed) and a 96-point controlled sweep across
> both domains, confirming the Theorem-P32-2 two-regime
> graceful-degradation bound; (c) Theorem P32-1 (cross-domain
> correctness preservation), Theorem P32-2 (noisy-extractor
> graceful degradation, promoting C31-7 to theorem in the monotone
> regime), Theorem P32-3 (token-bound preservation under bounded
> noise — the inbox capacity is the regulariser); and (d) a
> frontier-model spot check with `qwen2.5-coder:7b` on both
> non-code benchmarks at k = 6. See Theorems P32-1..P32-3 +
> Conjectures C32-4, C32-5 in `RESULTS_PHASE32.md`.**
>
> **Phase 27 extends the runtime-calibration axis from the curated
> 21-snippet corpus to REAL CORPUS FUNCTIONS. The Phase-27 observer
> classifies every function in a corpus into a callability state
> (`ready_no_args` / `ready_typed` / `ready_curated` or one of
> several `unsupported_*` states), synthesises recipe-compatible
> arguments via a `SafeRecipeRegistry`, and runs the Phase-26 probes
> with additional `sys.settrace`-based entry detection and per-call
> wall-time budgeting. On `vision-core` (~791 functions) the ready
> slice is ~35.7 %; the remaining 64 % is structurally unprobable
> under the default recipe strategy (methods without auto-
> constructed instances, variadic args, generators, async, untyped
> positional params). Theorem P27-1 formalises this as a strict
> inclusion $F_R \subseteq F_A$; Theorem P27-2 shows corpus-scale
> runtime coverage is witness-availability-bounded, not planner-
> exactness-bounded — the planner round-trip remains 100 % on
> every predicate across every corpus.** The full
> architecture composes as:
>
> ```
> Routing  (who talks to whom; O(log N))             — lossy by design
>     ↓
> Trigger  (when to refine)                          — lossy by design
>     ↓
> Exact external memory (Merkle DAG)                 — LOSSLESS, content-addressed
>     ↓                  ┌─ text chunks (Phases 19–21)
>     ↓                  ├─ source files + AST metadata (Phase 22)
>     ↓                  ├─ source files + AST structural metadata
>     ↓                  │    + conservative intraprocedural metadata (Phase 24)
>     ↓                  └─ source files + AST structural metadata
>     ↓                       + conservative intraprocedural metadata
>     ↓                       + conservative INTERPROCEDURAL metadata (Phase 25)
> Retrieval (dense + lexical RRF + multi-hop)        — lossy in ranking, never in content
>     ↓
> Computation / planning (typed operators + planner) — LOSSLESS, deterministic
>     ↓                  ┌─ structural patterns (count / list / top / join)
>     ↓                  ├─ intraprocedural patterns (may_raise / recursive / io) [P24]
>     ↓                  └─ INTERPROCEDURAL patterns (trans_may_raise /
>     ↓                                                 participates_in_cycle /
>     ↓                                                 trans_calls_* / unresolved) [P25]
> Render: { wrap_llm | direct }                      — direct path: zero LLM, zero prompt
>     ↓
> Bounded active context fed to the LLM (only when
> the wrap path or retrieval fallback is used)       — bytes are exact slices of memory
>
>     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
>     Phase-31 typed-handoff substrate (cross-role content channel)
>     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
>     role A's events → role A's extractor → TypedHandoff
>                      (claim_kind, payload, src_event_ids, cid)
>                      ↓
>                      RoleSubscriptionTable[(src_role, claim_kind)]
>                         → set(consumer roles)
>                      ↓
>                      bounded RoleInbox (dedup by payload_cid,
>                         overflow accounted, wrong_role rejected)
>                      ↓
>                      hash-chained HandoffLog
>                         (SHA-256 over (prev_chain_hash, handoff
>                          fields); tamper / truncation detector)
>                      ↓
>                      per-(src_role, to_role, claim_kind)
>                         DeliveryAccount counters for the benchmark
>     (Phase-31 is additive: the layer sits alongside routing and
>      ingestion; teams that do not need typed handoffs can ignore
>      it. The handoff layer lifts load-bearing content into routing
>      headers so downstream roles can subscribe by claim-kind — the
>      mechanism by which the Phase-29 "routing-by-type cannot rescue
>      the aggregator" observation is resolved for general teams.)
>     (Phase-32 adds a controlled noise wrapper
>      `core/extractor_noise.noisy_extractor` that sits between any
>      extractor and the router to exercise Theorem P32-2's
>      graceful-degradation regimes; production runs use identity
>      noise, the Phase-32 sweep uses non-trivial parameters.)
>
>     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
>     Phase-35 dynamic-coordination layer (strictly above P31 layer)
>     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
>     Auditor detects contested candidates in its RoleInbox
>                      ↓
>                      open_thread(issue_kind, frozen(members),
>                                   candidate_claims, max_rounds,
>                                   max_replies_per_member,
>                                   quorum, witness_token_cap)
>                      ↓
>                      member roles post typed ThreadReply messages:
>                      {INDEPENDENT_ROOT, DOWNSTREAM_SYMPTOM,
>                       UNCERTAIN, AGREE, DISAGREE, DEFER_TO}
>                      ↓
>                      close_thread → ThreadResolution:
>                      {SINGLE_INDEPENDENT_ROOT, QUORUM_AGREE,
>                       CONFLICT, NO_CONSENSUS, TIMEOUT}
>                      ↓
>                      emit(CLAIM_THREAD_RESOLUTION, payload="kind=...
>                           winner=role/kind losers=r/k,...")
>                         ↓  (through unchanged HandoffRouter)
>                      RoleInbox(auditor) — single public output
>     (Phase-35 is strictly additive: thread-internal events live in
>      the existing HandoffLog but no inbox subscribes to the
>      THREAD:* internal claim kinds; non-member roles see zero
>      thread traffic. Bounded-context invariant extends with an
>      additive T·R_max·W per role per round — Theorem P35-2.)
>
>     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
>     Phase-26 runtime-calibration observer (additive, off-path)
>     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
>     Source bytes ┄┄┄→ instrumented execution probes
>                       (monkeypatched subprocess / filesystem /
>                        network APIs; sys.settrace for cycles)
>                       ↓
>                       per-predicate RuntimeObservation:
>                         runtime_flag, n_runs, n_triggered,
>                         witnesses, decidable, applicable
>                       ↓
>                       calibration summary: FP, FN, fp_rate,
>                       fn_rate, per-family breakdown
>     (source bytes and analyzer flags flow in; the runtime observer
>      reports a second truth value per predicate; the planner's
>      direct-exact path is unchanged.)
>
>     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
>     Phase-27 corpus-scale runtime-calibration observer (additive)
>     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
>     Real corpus  ┄┄┄→ CorpusFunctionCandidate per qname:
>                       {ready_no_args | ready_typed | ready_curated
>                        | unsupported_*}  (AST + inspect.signature
>                        + SafeRecipeRegistry lookup)
>                       ↓
>                       InvocationRecipe per ready candidate:
>                       (no_args | typed from fuzz pool | curated)
>                       ↓
>                       sandbox + entry-and-budget tracer:
>                         sys.settrace counts enter_count on
>                         target code object; time.monotonic()
>                         check every line event; sentinel on
>                         budget expiry.
>                       ↓
>                       per-predicate CorpusObservation:
>                         runtime_flag, n_runs, n_triggered,
>                         n_entered, n_timeout, witnesses,
>                         recipe_kind, applicable, entered, timeout
>                       ↓
>                       coverage account: per-status buckets +
>                       ready_fraction, calibrated_fraction;
>                       per-predicate metrics restricted to
>                       entered=True.
> ```
>
> See `vision_mvp/RESULTS_PHASE19.md`, `RESULTS_PHASE20.md`,
> `RESULTS_PHASE21.md`, `RESULTS_PHASE22.md`, `RESULTS_PHASE23.md`,
> `RESULTS_PHASE24.md`, and `RESULTS_PHASE25.md` for the cumulative
> evidence: an exact byte-store + bounded-context worker beats
> summarise-then-pool on long-document needle questions; hybrid
> retrieval + structural multi-hop expansion close most of the
> remaining recall gap; a typed operator pipeline answers
> aggregation queries the retrieval layer cannot reach (91 % vs 64
> % on synthetic aggregation, beating even oracle on that slice);
> on a real Python codebase the direct-exact path achieves **7/7
> correct with zero LLM calls and zero prompt chars**, while
> retrieval-only conditions score **0/7** because aggregation is
> structurally unreachable by top-k retrieval; across **six real
> Python corpora** direct-exact scores **65/65 (100 %, σ = 0)** on
> the structural battery, **44 / 44 (100 %, σ = 0)** on the Phase-
> 24 intraprocedural semantic battery, and **50 / 50 (100 %, σ =
> 0)** on the Phase-25 interprocedural semantic battery with zero
> LLM calls; retrieval-mediated paths average **19.7 % (σ = 17.6)**
> on structural aggregation, **49.6 % (σ = 15.8)** on Phase-24
> semantic, and **38.0 % (σ = 23.1)** on Phase-25 interprocedural.
> **The exact slice now covers syntactic code structure AND
> conservative intraprocedural-semantic code properties AND
> conservative interprocedural-semantic code properties — the last
> including transitive effect propagation over a local call graph
> and exact SCC-based recursion-cycle detection.** The CASR spec
> below is unchanged for the routing/trigger layers.

---

---

## Design Principles

1. **Routing decisions must not require reading message content.** The Bloom filter operates on event type metadata (O(1)), not event bodies. Reading content for routing decisions would negate the efficiency gains.

2. **Scale assignments are declarative, set at agent instantiation.** Scales do not change mid-task. Dynamic scale inference is a research question deferred to Phase 3.

3. **The world model updates are asynchronous.** The surprise filter does not block message delivery. Updates to M_i happen in a background process.

4. **No information is destroyed, only routed.** The event log is append-only. Any agent can replay the full event history if needed. CASR is a filter on delivery, not on storage.

5. **Fail open.** When uncertain (Bloom filter positive hit, world model not yet trained), deliver the message. Over-delivery is safer than under-delivery.

---

## Agent Interface

Every agent in a CASR-enabled team exposes this interface at instantiation:

```python
@dataclass
class AgentConfig:
    agent_id: str
    role: str                    # Human-readable role name
    task_description: str        # Current task at instantiation
    scale: int                   # 0=Token, 1=Statement, 2=Function, 3=Module, 4=System
    distortion_budget: float     # Acceptable task-error probability increase (0.0 to 1.0)
    causal_footprint: BloomFilter  # Pre-computed relevance filter for this role
    world_model: Optional[GenerativeModel]  # None until trained in Phase 2
    surprise_threshold: float    # τᵢ — KL threshold for transmission (0.0 disables filter)
```

**Scale semantics:**

| Scale Value | Granularity | Example Events Visible | Example Roles |
|-------------|-------------|----------------------|---------------|
| 0 | Token | Every token, syntax error, formatting diff | Linter, formatter, syntax checker |
| 1 | Statement | Individual tool calls, single code lines, test results | Code writer, unit tester, file editor |
| 2 | Function | Function completions, subtask results, local test pass/fail | Subagent, debugger, function-level reviewer |
| 3 | Module | Subsystem changes, integration test results, cross-function state | Orchestrator, module-level planner |
| 4 | System | Architectural decisions, goal completions, global constraints | Meta-orchestrator, project planner |

**Distortion budget:** Expressed as maximum acceptable probability of the agent taking a suboptimal action due to missing context. Conservative agents (planners) use low budget (~0.01). Monitoring agents (checking for catastrophic failures only) use high budget (~0.20).

---

## The Scale Projection Operators

For each scale s, the projection operator P_s maps a full event to its representation at scale s.

**Required property (composability):**
```
P_{s1}(P_{s2}(e)) = P_{max(s1,s2)}(e)  for all events e
```

Applying two projections in sequence gives the coarser projection. This ensures consistency across the hierarchy.

**Fixed-point events** (preserved at all scales, P_s(e) = e for all s):
- Task goal specification messages
- Hard constraint declarations
- Error/failure events (any unhandled exception or task failure)
- Final output/completion events

**Projection implementations by scale transition:**

```
scale 0 → 1: Aggregate consecutive tokens into statement-level summaries.
             Discard whitespace, formatting, comments.
             Preserve: variable names, control flow, function calls.

scale 1 → 2: Aggregate statements into function-level summaries.
             Discard: intermediate variable states, loop iterations.
             Preserve: function signature, return value, side effects, errors.

scale 2 → 3: Aggregate function results into module-level summaries.
             Discard: internal function logic.
             Preserve: module interface changes, integration test results, exported state.

scale 3 → 4: Aggregate module changes into system-level summaries.
             Discard: implementation details.
             Preserve: architectural decisions, constraint violations, goal progress.
```

**Implementation note:** In the MVP, these projections are implemented as LLM calls with structured output schemas. In Phase 2, they can be replaced with fine-tuned smaller models for efficiency.

---

## Message Bus Architecture

The central component is an event-sourced message bus. All agents are publishers and subscribers.

```
┌─────────────────────────────────────────────────────────┐
│                    EVENT BUS                             │
│                                                         │
│  ┌──────────────┐   ┌──────────────────────────────┐   │
│  │  Event Log   │   │     Subscriber Registry      │   │
│  │ (append-only)│   │ agent_id → AgentConfig       │   │
│  └──────────────┘   └──────────────────────────────┘   │
│                                                         │
│  On new event e published by agent aⱼ:                 │
│  For each subscriber aᵢ:                               │
│    1. B_i(e.type) → if "definitely not": skip          │
│    2. P_{sᵢ}(e) → compute scale projection             │
│    3. δᵢ(e) = KL(M_i.predict() || e.embedding)        │
│       if δ < τᵢ and M_i is trained: skip               │
│    4. Deliver P_{sᵢ}(e) to aᵢ's context queue         │
└─────────────────────────────────────────────────────────┘
```

**Event schema:**

```python
@dataclass
class Event:
    event_id: str           # UUID
    event_type: str         # Enumerated type (tool_call, message, state_change, error, goal_update)
    sender_id: str          # Sending agent
    timestamp: float        # Unix timestamp
    scale_level: int        # Scale of the originating agent
    body: dict              # Full event content (not read during routing decision)
    embedding: np.ndarray   # Precomputed embedding for world model comparison
    is_fixed_point: bool    # If True, delivered to all agents unmodified
```

**Delivery guarantee:** At-least-once delivery. Events that pass all three CASR stages are queued for delivery. If an agent's queue is full (context window filling), the bus falls back to delivering only fixed-point events until the agent processes its queue.

---

## Bloom Filter Specification

**Construction (offline, per agent role):**

```
Input: Set of (event_type, is_relevant) pairs for this role
Output: BloomFilter with target false positive rate p = 0.01

false_positive_rate = 0.01  (1% of irrelevant events pass the filter)
n = number of event types in the system
m = -n * ln(p) / (ln(2))^2  (filter size in bits)
k = (m/n) * ln(2)           (number of hash functions)
```

**At runtime:**
```
query(event_type) → {DEFINITELY_NOT_RELEVANT, POSSIBLY_RELEVANT}
```

If DEFINITELY_NOT_RELEVANT: drop the event without reading its body.
If POSSIBLY_RELEVANT: proceed to scale projection.

**Staleness mitigation:** Bloom filters are rebuilt at each task phase transition (e.g., when the orchestrator changes the global task state). Between transitions, the filter is immutable.

**Conservative initialization:** Before any empirical data is collected, initialize the Bloom filter to include all event types (100% pass rate). Refine using empirical footprint estimation once data is available.

---

## World Model Specification

The world model M_i for agent aᵢ is a lightweight model that predicts the next event's embedding given the agent's current context:

```
M_i : (current_context, recent_events) → predicted_event_embedding
```

**Stage 2 implementation (Phase 2+):**
- Small transformer (≤7B parameters) or frozen large model with a fine-tuned prediction head
- Input: last K events in aᵢ's context, projected to scale sᵢ
- Output: predicted embedding of next event in aᵢ's context
- Training: minimize L2 distance between predicted and actual event embeddings

**Surprise computation:**
```python
def surprise(M_i, event_e):
    predicted = M_i.predict(current_context)
    actual = event_e.embedding
    return kl_divergence(predicted, actual)
    # or simpler: cosine_distance(predicted, actual)
```

**World model disabled (MVP):** In the MVP, M_i is not trained. Set τᵢ = 0, which delivers all events that pass the Bloom filter. The surprise filter is enabled incrementally in Phase 2.

---

## Failure Modes and Mitigations

| Failure Mode | Cause | Detection | Mitigation |
|---|---|---|---|
| Missing critical context | Bloom filter false negative (impossible by construction) | N/A | None needed — Bloom filters have no false negatives |
| Context starvation | τᵢ too high, world model over-predicts | Agent produces incorrect output despite low context | Decrease τᵢ or trigger full-sync |
| Bloom filter staleness | New event type introduced after filter construction | Agent fails to respond to new event types | Rebuild filters at phase transitions; default-include unknown event types |
| World model drift | Team behavior diverges from training distribution | Surprise distribution shifts systematically | Periodic re-training of M_i on recent event logs |
| Scale mismatch | Event from scale-0 agent delivered to scale-4 agent without projection | Scale-4 agent context fills with low-level detail | Scale projection is mandatory for all cross-scale delivery |
| Orchestrator overload | All N workers complete simultaneously, flood orchestrator | Orchestrator queue depth spikes | Rate-limit delivery to orchestrator; batch completions within a time window |

**Full-state synchronization:** Every K rounds (K is a hyperparameter, default 50), each agent receives the unfiltered projection of all current state at its scale, bypassing all CASR filters. This corrects accumulated errors from stale Bloom filters and miscalibrated world models. K should be set to the expected task-phase length.

---

## Event Type Registry

A centralized registry of all event types and their default scale assignments. This is the source of truth for Bloom filter construction.

```
Core event types:

TOOL_CALL           scale=1  (statement level by default)
TOOL_RESULT         scale=1
FILE_EDIT           scale=1
FILE_CREATE         scale=2  (function/module level)
TEST_RUN            scale=2
TEST_RESULT         scale=2
FUNCTION_COMPLETE   scale=2
MODULE_COMPLETE     scale=3
TASK_GOAL_UPDATE    scale=4, is_fixed_point=True
HARD_CONSTRAINT     scale=4, is_fixed_point=True
ERROR_UNHANDLED     scale=4, is_fixed_point=True  (always delivers to all)
TASK_COMPLETE       scale=4, is_fixed_point=True
AGENT_SPAWN         scale=3
AGENT_TERMINATE     scale=3
MESSAGE_AGENT       scale=2  (default; overridden by sender scale)
```

**Custom event types:** Teams can register domain-specific event types with explicit scale assignments and relevance mappings per role.

---

## Scaling Characteristics

| Team Size | History Depth | Naive Context (tokens) | CASR Context (tokens) | Reduction |
|-----------|--------------|----------------------|----------------------|-----------|
| 5 agents  | 50 rounds    | ~12,500              | ~2,500               | 5x        |
| 10 agents | 100 rounds   | ~100,000             | ~6,600               | 15x       |
| 20 agents | 200 rounds   | ~800,000             | ~14,600              | 55x       |
| 50 agents | 500 rounds   | ~12,500,000          | ~46,000              | 272x      |

*Estimates based on O(H·log(N)) vs O(N·H²) scaling, with k=50 tokens per event, branching factor b=5.*

These are theoretical. Empirical validation is the primary goal of Phase 1 (MVP).

---

## Interface with Existing Frameworks

CASR is designed as a drop-in message bus layer for existing multi-agent frameworks.

**AutoGen integration:** Replace AutoGen's GroupChat or nested conversation patterns with the CASR event bus. Agent-to-agent messages become events; the bus handles routing.

**LangGraph integration:** Add a CASR routing layer to each graph edge. Before a LangGraph node receives its input state, run the state update through the CASR pipeline.

**CrewAI integration:** Intercept the task context assembly step. Instead of assembling full context for each agent, assemble CASR-filtered context.

The goal is not to replace these frameworks but to add principled context routing as a layer beneath their agent orchestration logic.


---

## Phase-45 Product Surface (operator entrypoint)

Phase 45 added a thin orchestration surface on top of the Phase
31..44 stack at `vision_mvp/product/`:

- `vision_mvp/product/profiles.py` — six stable, versioned
  profiles (`local_smoke`, `bundled_57`, `bundled_57_mock_sweep`,
  `aspen_mac1_coder`, `aspen_mac2_frontier`, `public_jsonl`).
  Schema: `phase45.profile.v1`.
- `vision_mvp/product/runner.py` — `run_profile(...)` composes
  readiness → sweep → report. Readiness is a hard gate unless
  overridden (Theorem P45-2). Real-LLM sweeps are *recorded* as
  a launch command rather than forked from inside the runner.
- `vision_mvp/product/report.py` — summary renderer;
  reusable on any stored `product_report.json`.
- One command:
  `python3 -m vision_mvp.product --profile <name> --out-dir <d>`

The product surface adds no new substrate semantics; see
`vision_mvp/RESULTS_PHASE45.md` and
`docs/context_zero_master_plan.md` §9 for the Finished-Product
Checklist and release criteria.


---

## Phase-46 Boundary Surface (external-exercise readiness)

Phase 46 adds a boundary layer between the Phase-45 product
surface and the outside world:

- `vision_mvp/product/import_data.py` — `audit_jsonl(...)`:
  schema classification (native / hermetic / ambiguous /
  unusable), duplicate-id detection, decode / non-object /
  empty-bank failure modes, delegated Theorem-P44-3 readiness.
  CLI exit codes distinguish *missing file* (2) from *blocker*
  (1) from *clean* (0).
- `vision_mvp/product/ci_gate.py` — `evaluate_report(...)` +
  `aggregate(...)`: five-check CI verdict over one or more
  `product_report.json` files. Threshold knobs for readiness
  fraction and per-cell pass@1; profile-whitelist support;
  machine-readable `phase46.ci_verdict.v1`.
- Frontier-model slot: `aspen_mac1_coder_70b` profile +
  `profiles.model_availability()` declarative check. Runner
  attaches `model_metadata` to recorded launches so downstream
  consumers can distinguish *slot_pending_availability* from
  *assumed_resident* without probing Ollama.

The boundary layer does not change any programme-internal
semantics; see `vision_mvp/RESULTS_PHASE46.md` and
`docs/context_zero_master_plan.md` §9.9 for the endogenous /
exogenous split.
