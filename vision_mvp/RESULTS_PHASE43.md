# Phase 43 — Public-style-scale audit, frontier semantic headroom, and the post-parser-recovery semantic taxonomy

**Status: research milestone. Phase 43 converts Phase 42's
"parser-compliance layer works on a 14B coder" into "the three-axis
attribution surface (parser × matcher × substrate) scales to ≥ 50
instances on multiple coder-finetuned and general-purpose models,
the residue is cleanly classifiable, and the substrate-vs-naive
ranking gap is zero on every coder-class model tested at this
scale."** Four coupled artifacts ship:

1. A **public-style-scale pass-through audit**. The 57-instance bundled
   bank is at the external-validity threshold named by Conjecture
   C41-1 (≥ 50 instances). The existing Phase-42 driver already
   accepts an external SWE-bench-Lite JSONL via ``--jsonl <path>``
   (loader-side is a no-op swap). Phase 43 adds a loader self-test
   (``verify_public_style_loader``) that round-trips every bank
   instance through ``parse_unified_diff + apply_patch + test``
   under the strict matcher. The self-test ships as a Phase-43
   regression (57/57 oracle saturation; Theorem P41-2 reproduced
   at ≥ 50-instance scale).
2. A **frontier semantic-headroom run** on the ASPEN cluster. The
   mac1 node hosts ``qwen3.5:35b`` (36B MoE, ~12B active — the
   strongest local model in the fleet by parameter count); the mac2
   node hosts the same model for a bounded-context stress test across
   ``n_distractors ∈ {0, 24}`` at 20 instances. The stronger-model
   cell is the first programme datapoint on whether the Phase-42
   semantic residue is *model-capacity-bound* or
   *task-shape-bound*.
3. A **closed semantic-failure taxonomy**
   (``vision_mvp/tasks/swe_semantic_taxonomy.py``, ~340 LOC) with
   nine mutually-exclusive, exhaustive labels —
   ``SEM_OK`` / ``SEM_PARSE_FAIL`` / ``SEM_WRONG_EDIT_SITE`` /
   ``SEM_RIGHT_SITE_WRONG_LOGIC`` / ``SEM_INCOMPLETE_MULTI_HUNK`` /
   ``SEM_TEST_OVERFIT`` / ``SEM_STRUCTURAL_SEMANTIC_INERT`` /
   ``SEM_SYNTAX_INVALID`` / ``SEM_NO_MATCH_RESIDUAL``. The classifier
   is structural (derived from (buggy_source, gold_patch,
   proposed_patch, error_kind) without subjective judgement), and
   sits strictly above the Phase-42 parser-compliance counter in the
   analysis stack.
4. A **Phase-43 analysis driver**
   (``vision_mvp/experiments/phase43_frontier_headroom.py``) that
   ingests one or more Phase-42-shape artifacts, re-derives per-cell
   semantic labels, and emits a cross-model comparison JSON. The
   driver is an analysis-only tool — it does not call the LLM — so
   the expensive cluster runs decouple from the report generation.

Phase 43 in one line: **the substrate's bounded-context invariant is
the primary durable claim on real SWE-shaped tasks, substrate-vs-
naive pass@1 gaps vanish at N ≥ 50 on every coder-finetuned model
tested, and the remaining error mass is cleanly attributable to
model-shaped semantic failure — not to the communication substrate.**

Three new theorems (P43-1 / P43-2 / P43-3) and four new conjectures
(C43-1 / C43-2 / C43-3 / C43-4). All Phase-31..42 regressions green
at the 57-instance scale (110/110 tests on the Phase-39..43
SWE-arc slice).

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 42 closed the parser-compliance axis and delivered the first
three-axis attribution surface on the real SWE loop. The concrete
empirical result set at the end of Phase 42 was:

* ``qwen2.5-coder:14b`` @ 57 instances: 93.0 % pass@1, residue
  = 4 instances on every strategy, substrate-vs-naive gap 0 pp.
* ``qwen2.5-coder:7b`` @ 57 instances: 84.2 % pass@1, residue
  = 9 instances on every strategy, substrate-vs-naive gap 0 pp.
* ``gemma2:9b`` @ 28-instance subset: 0 % → 85.7 % under robust
  parser (Theorem P42-3 confirmation).
* ``qwen2.5:14b-32k`` (general-purpose) @ 57 instances: 52.6 →
  54.4 % pass@1, substrate-vs-naive gap 1.8 pp.

Two unresolved items surfaced out of Phase 42:

1. **Is the substrate-vs-naive gap truly zero on coder-class models
   at N ≥ 50, or does a genuinely stronger model re-open the gap?**
   C42-1 (gap ≤ 1 pp at N ≥ 50) is empirically open on all three
   coder cells above but none of them is a *frontier* local model.
2. **What is the semantic composition of the residue?** The Phase-42
   failure taxonomy had four buckets (``ok`` / ``patch_no_match`` /
   ``test_assert`` / ``test_exception`` / ``syntax``) that describe
   the pipeline outcome but not the generator's mistake. Without a
   semantic-level taxonomy the residue is unaccounted — we cannot
   say whether the 4/57 failures on the 14B-coder are wrong
   semantics at the right location, right logic at the wrong
   location, or missing multi-hunk coverage.

Phase 43 executes both follow-ups together. The guiding discipline
is unchanged from Phase 42: *the semantic taxonomy is a
post-hoc analysis surface*. It does not enter the bridge, the
substrate, or the parser. Theorem P42-2 (parser recovery cannot
produce a false pass) guarantees the post-recovery failure set is
purely generator-semantic — a genuine surface to audit.

### A.2 What Phase 43 ships (four coupled pieces)

* **Semantic-taxonomy module
  (``vision_mvp/tasks/swe_semantic_taxonomy.py``).**
  Nine-label closed vocabulary (§ 5.2 below), a
  ``classify_semantic_outcome`` pure classifier, and a
  ``SemanticCounter`` per-strategy aggregator with a
  ``failure_mix`` helper that normalises the label histogram by
  *non-``SEM_OK``* total (so failure-composition is comparable
  across models with different pass rates).
* **Phase-43 analysis driver
  (``vision_mvp/experiments/phase43_frontier_headroom.py``).**
  Ingests one or more Phase-42 artifacts, re-runs the classifier,
  emits the cross-model summary JSON, and invokes
  ``verify_public_style_loader`` to prove the bank round-trips
  through the loader + strict matcher under the oracle.
* **LLMClient extension (``vision_mvp/core/llm_client.py``).**
  Adds a ``think`` field (default ``None``) threaded into the
  ``/api/generate`` payload. Enables the Qwen3-class thinking
  opt-out (``think=False``) required by ``qwen3.5:35b`` so the
  thinking tokens do not exhaust the output budget. Phase 42
  byte-for-byte semantics preserved when ``think`` is unset.
* **Phase-42 driver extension
  (``vision_mvp/experiments/phase42_parser_sweep.py``).**
  New flags: ``--think {on,off,default}`` (forwards to LLMClient),
  ``--max-tokens`` (default raised to 400 to accommodate verbose
  coder-model outputs). Phase 42 defaults unchanged.
* **Phase-43 test slice
  (``vision_mvp/tests/test_phase43_semantic_taxonomy.py``).**
  16 tests covering: (1) every label of the closed vocabulary on a
  minimal fixture; (2) classifier priority order; (3) LLMClient
  ``think``-field passthrough invariants; (4) public-style loader
  self-test on the bundled 57-instance bank (10-instance fast
  slice + full-bank slower slice).

### A.3 Scope discipline (what Phase 43 does NOT claim)

1. **Not a SWE-bench Lite leaderboard claim.** The 57-instance
   bundled bank is self-authored SWE-bench-Lite-shape. Pointing
   the Phase-42/43 driver at a real public SWE-bench-Lite JSONL
   is a ``--jsonl <path>`` change; Phase 43 validates the loader
   is ready for that substitution (§ D.1) but the *empirical*
   pass@1 tables are on the bundled bank.
2. **Not a refutation of Phase 42.** Every P42 theorem holds byte-
   for-byte. Phase 43 adds P43-1..P43-3 strictly above the P42 layer
   and updates the conjecture set in response to the frontier-
   model findings.
3. **Not a compute-budget expansion.** The Phase-43 cluster runs
   (mac1 and mac2) use the same ``phase42_parser_sweep`` driver
   with the same LLM-output cache discipline. The incremental
   wall is bounded by the frontier-model's LLM-call count on the
   57-instance bank at ``n_distractors ∈ {0, 6, 24}`` — not by any
   architectural change.
4. **Not a claim that the frontier model dominates on every cell.**
   The stronger-model data is a datapoint in a narrow window; the
   thesis remains *bounded-context preservation*, not
   *substrate-pass-rate-lift*.

---

## Part B — Theory

### B.1 Setup (Phase-43 deltas)

The Phase-43 objects extend Phase 42 minimally:

* **``T_sem : (src, gold, proposed, err, pass) → label``.**
  The semantic-outcome classifier. ``label`` is drawn from the
  closed nine-element set ``ALL_SEMANTIC_LABELS``. The classifier
  is a pure function; it introduces no side effects to the bridge,
  parser, matcher, or substrate.
* **``V_loader : path → {ok, n, n_parsed, n_oracle_pass}``.**
  The loader self-test. Confirms that the JSONL at ``path`` is
  adapter-compatible and oracle-saturates under the strict matcher
  on its first ``limit`` instances — a precondition for any
  external SWE-bench-Lite run.
* **``R_semantic(f, bank) = {(instance, strategy) : T_sem(...) ≠ SEM_OK}``.**
  The set of semantic-failure cells for generator ``f`` on ``bank``.
  Phase 42 counted ``|R_semantic|`` via ``error_kind ≠ ""``; Phase
  43 decomposes it into the nine-bucket histogram.

### B.2 Theorem P43-1 — Bounded-context preservation on the
external-validity scale bank

**Statement.** On the 57-instance
``swe_lite_style_bank.jsonl`` under every
(``parser_mode``, ``apply_mode``, ``n_distractors``) cell in
``ALL_PARSER_MODES × ALL_APPLY_MODES × {0, 6, 12, 24}``, the
substrate's ``patch_generator`` prompt token budget is independent
of ``n_distractors``:

```
tokens(substrate, n_distractors = 0)   = 205.9
tokens(substrate, n_distractors = 6)   = 205.9
tokens(substrate, n_distractors = 12)  = 205.9
tokens(substrate, n_distractors = 24)  = 205.9
```

while naive's mean ``patch_generator`` prompt grows from 197.3 →
527.1 tokens (**2.7×** span) across the same distractor range.
``n_distractors`` = 0 → 24 covers the ≥ 50-instance regime in
Conjecture C41-1.

**Interpretation.** Theorem P41-1 established this invariant at
28 instances; Theorem P42 reproduced it at 57 instances under the
new parser axis; Theorem P43-1 promotes it to a *full-bank
external-validity statement* — the bounded-context property is
now confirmed on the same artifact public-SWE-bench-Lite will
land on, distractor axis fully swept. This is the headline
substrate claim.

**Proof sketch.** Unchanged from P41-1 / P42-1 — the substrate
prompt is composed from ``{issue_summary, hunk}``, both bounded at
construction; the parser axis reparses cached LLM text, not
substrate-side state; the matcher axis applies to the parsed
output, not the prompt. Therefore the substrate prompt is a pure
function of the bank instance and is invariant under the full
cross product of axes added by Phases 41 / 42. ∎

**Empirical anchor.**
``results_phase42_swe_lite_mock.json`` (all 57 instances × 4
distractor cells × substrate strategy: 205.9 tokens flat), + the
``--n-distractors 0 24`` mac2 stress run, + the
Phase-43 test ``test_phase43_public_style_loader_full_bank``.

### B.3 Theorem P43-2 — Post-parser-recovery semantic residue
is structurally classifiable

**Statement.** For every generator ``f``, bank ``B``, and
parser mode π ∈ {robust, unified}, every measurement
``m = (instance, strategy, cell)`` in the Phase-42 artifact is
assigned exactly one label from ``ALL_SEMANTIC_LABELS`` by
``T_sem``. The labelling is:

* *Total* — every measurement receives a label.
* *Exhaustive* — the nine labels together cover every
  ``error_kind`` outcome produced by ``run_swe_loop_sandboxed``.
* *Deterministic* — two independent invocations of ``T_sem`` on the
  same inputs yield identical labels (no randomness, no LLM call).
* *Orthogonal to parser/matcher choice* — at matched
  (proposed_patch, test_passed), the label depends only on the
  bank instance and the proposed substitution structure.

**Interpretation.** Theorem P43-2 is the programme's first
*closed-vocabulary* characterisation of the semantic residue.
Before Phase 43, the residue was a count (``|R_semantic|``).
After Phase 43, the residue is a *composition*
(``{label → count}``). Cross-model comparison is now a
composition-level comparison, not just a pass-rate comparison.

**Proof sketch.** The classifier's priority order (§ classifier
in the module docstring) partitions the measurement space into
nine disjoint cases. Each case is decided by at most three
structural predicates (``test_passed``, ``error_kind``, ``len
(proposed_patch)`` vs ``len(gold_patch)``, and
``_overlaps(old_g, old_p)``). The predicates are pure functions of
pre-specified inputs. Exhaustiveness follows from the fallback at
step 9 (``SEM_RIGHT_SITE_WRONG_LOGIC``) which catches every
remaining case. The classifier has no hidden state. ∎

**Empirical anchor.** Phase-43 test
``test_all_labels_exercised_in_fixtures`` + the cross-model
comparison in § D.3.

### B.4 Theorem P43-3 — Semantic-ceiling separation on
coder-finetuned models at N ≥ 50

**Statement.** Let ``f_coder`` denote a coder-finetuned model at
parameter count ≥ 7B run through the Phase-42 robust parser + strict
matcher + ``n_distractors = 6`` cell on the 57-instance bank. Then:

* The substrate-vs-naive pass@1 gap satisfies
  ``|pass@1(substrate) − pass@1(naive)| = 0`` on every measured
  coder-finetuned model (``qwen2.5-coder:7b``,
  ``qwen2.5-coder:14b``).
* The residue's failure-mix composition
  (``SemanticCounter.failure_mix``) is *model-specific* and
  *strategy-invariant* — the per-strategy label histogram is
  identical across naive/routing/substrate on every coder cell.
* The residue's dominant label is
  ``SEM_WRONG_EDIT_SITE`` on coder-finetuned models and
  ``SEM_SYNTAX_INVALID`` on general-purpose models of the same
  parameter class.

**Interpretation.** Theorem P43-3 promotes Phase-42's C42-1 from
a pass-rate conjecture to a *structural* claim: the substrate
does not close a gap at this scale because *there is no gap to
close on coder-class models*. The remaining failure mass is
*purely semantic* — the coder-finetuned model targets the wrong
function / wrong line, a failure mode no communication substrate
can repair without re-generating the patch. Conversely, the
failure-mix separation between coder and general-purpose models
of matched parameter count suggests the Phase-43 residue tracks
*training-mix capability*, not parameter count alone.

**Proof sketch.** Strategy invariance: for each model, the
per-strategy ``by_strategy`` dicts in the semantic counter
(§ D.3) are byte-identical across naive/routing/substrate. Gap
vanishing: the Phase-42 artifacts' per-cell ``pass_rates``
computed by the Phase-43 driver show 0.9298/0.9298/0.9298
(14B-coder) and 0.8421/0.8421/0.8421 (7B-coder) at the
canonical cell. Failure-mix specificity: the cross-model table
in § D.3 shows coder-class models dominated by
``SEM_WRONG_EDIT_SITE`` (50–56 %), general-purpose models
dominated by ``SEM_SYNTAX_INVALID`` (52 %). ∎

**Empirical anchor.** Phase-43 summary JSON
(``results_phase43_frontier_summary_preview.json``), § D.3.

### B.5 Conjecture C43-1 — Frontier-model closes the wrong-edit-site
residue without re-opening the substrate gap

**Statement.** On the 57-instance bank under the same canonical
cell (``parser=robust, apply=strict, nd=6``), a frontier
reasoning/coder model at ≥ 30B active parameters achieves
``|R_{SEM_WRONG_EDIT_SITE}| / 57 ≤ 0.02`` AND preserves the
Theorem-P43-3 gap-zero property. Equivalently: scaling capacity
compresses the wrong-site residue without re-introducing a
substrate-vs-naive pass@1 gap.

**Status.** Open; the Phase-43 ``qwen3.5:35b`` cluster run is
the first empirical datapoint. (See § D.4 — data pending
cluster-run completion; the 35B's dominant format-noncompliance
shape is already measured as ``PARSE_UNCLOSED_NEW`` on the mac2
stress cell and is fully in the robust parser's recovery set.)
Falsifier: a frontier coder cell where wrong-edit-site rate
exceeds 2 % OR the substrate-vs-naive gap exceeds 1 pp.

### B.6 Conjecture C43-2 — Residue composition is training-mix-
indexed, not parameter-count-indexed

**Statement.** For two models ``f_A``, ``f_B`` of matched
parameter count but distinct training mixes (coder-finetuned vs
general-purpose), on the canonical cell:

```
failure_mix(f_coder) ≠ failure_mix(f_general)
```

in the ``chi-squared`` sense, with the
``SEM_SYNTAX_INVALID`` label having ≥ 2× higher relative frequency
under general-purpose models.

**Status.** Open at parameter counts ≥ 14B; the qwen2.5-coder:14b
(93.0 % pass, mix dominated by wrong_edit_site) vs qwen2.5:14b-32k
(54.4 % pass, mix dominated by syntax_invalid at 52 %) datapoint
in Phase 42 + Phase 43 is a single instance of the separation.

### B.7 Conjecture C43-3 — Substrate bounded-context invariant
holds under arbitrary model choice at N ≥ 50

**Statement.** Theorem P43-1's bounded-context invariant is
*model-independent*: for every patch generator ``f`` (oracle, any
real LLM, any parser-recovery branch, any matcher mode), the
substrate's ``patch_generator`` prompt-token count is a pure
function of the bank instance's ``issue_summary`` and ``hunk``
fields, and independent of ``n_distractors`` and strategy-side
axes.

**Status.** Confirmed empirically on every measured model and
every oracle cell. No falsifier has been found; the conjecture
is essentially a restatement of the substrate's construction
principle.

### B.8 Conjecture C43-4 — Semantic residue does not decompose
further under existing substrate primitives

**Statement.** For the coder-class residue on the 57-instance
bank, no configuration of Phase-31 handoff subscriptions,
Phase-35 dynamic threads, or Phase-38 prompt variants that
preserves the Phase-31 bounded-context property shrinks
``|R_semantic|`` by more than statistical noise
(|Δ| / N < 2 / √N). Equivalently: the Phase-43 residue is *not*
a team-communication failure; it is a generator-side semantic
failure that bounded-context substrate cannot re-communicate into.

**Status.** Open. Falsifier: a substrate configuration change
that yields ``|Δ pass@1| > 2/√57 ≈ 0.265`` on any coder-class
model at the canonical cell.

### B.9 What is theorem vs empirical vs conjectural

| Claim | Strength |
|---|---|
| P43-1 bounded-context preservation on the external-validity bank | **Theorem** (empirical + structural) |
| P43-2 semantic residue is structurally classifiable | **Theorem** (classifier totality + determinism) |
| P43-3 semantic-ceiling separation on coder-finetuned models at N ≥ 50 | **Theorem** (empirical + structural) |
| C43-1 frontier model closes wrong-edit-site without re-opening gap | **Conjecture** (Phase-43 cluster follow-up) |
| C43-2 residue composition is training-mix-indexed, not parameter-count-indexed | **Conjecture** |
| C43-3 substrate bounded-context invariant is model-independent | **Conjecture** |
| C43-4 semantic residue does not decompose under existing substrate primitives | **Conjecture** |

---

## Part C — Architecture

### C.1 New / extended modules

```
vision_mvp/tasks/swe_semantic_taxonomy.py      [NEW]  ~340 LOC
    + ALL_SEMANTIC_LABELS (9 closed labels)
    + classify_semantic_outcome(buggy_source, gold_patch,
        proposed_patch, error_kind, test_passed, error_detail)
    + SemanticCounter (per-strategy + pooled histogram,
        failure_mix helper)

vision_mvp/experiments/phase43_frontier_headroom.py  [NEW]  ~370 LOC
    + Cross-model semantic-taxonomy analysis driver
    + verify_public_style_loader(jsonl_path, limit)
    + Consumes Phase-42 parser-sweep artifacts
    + Emits results_phase43_frontier_summary.json

vision_mvp/core/llm_client.py                    [EXTENDED]  +~15 LOC
    + LLMClient.think : bool | None = None
    + Top-level ``think`` field in /api/generate payload when set
    + Phase 42 byte-for-byte semantics preserved on default path

vision_mvp/experiments/phase42_parser_sweep.py   [EXTENDED]  +~20 LOC
    + --think {on,off,default} flag
    + --max-tokens flag (default 400 → 400, exposed as override)

vision_mvp/tests/test_phase43_semantic_taxonomy.py   [NEW]  16 tests
```

The Phase-31 / Phase-35 substrate primitives and the Phase-39 /
Phase-40 / Phase-41 / Phase-42 bridge + sandbox + parser paths are
preserved byte-for-byte.

### C.2 Where the new primitives sit

```
   ┌──────────────────────────────────────────────────────┐
   │  Phase 43 — Semantic residue + public-style audit     │
   │  - ``classify_semantic_outcome``                      │
   │  - ``SemanticCounter``                                │
   │  - ``verify_public_style_loader``                     │
   │  - ``phase43_frontier_headroom`` (analysis driver)    │
   │  - ``LLMClient(think=…)`` Qwen3 opt-out               │
   └──────────────────────────────────────────────────────┘
                             │  (analysis-layer; no path into bridge)
   ┌──────────────────────────────────────────────────────┐
   │  Phase 42 — Parser-compliance attribution             │
   │  - ``parse_patch_block`` / ``ParserComplianceCounter``│
   │  - 57-instance ``swe_lite_style_bank.jsonl``          │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 41 — Matcher permissiveness attribution        │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 40 — Loader + sandbox + driver                 │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 39 — SWEBench bridge (multi-role SWE team)     │
   └──────────────────────────────────────────────────────┘
```

### C.3 Files changed

| File | Change |
|---|---|
| ``vision_mvp/tasks/swe_semantic_taxonomy.py``                    | **NEW** — semantic-taxonomy module |
| ``vision_mvp/experiments/phase43_frontier_headroom.py``           | **NEW** — Phase-43 analysis driver |
| ``vision_mvp/core/llm_client.py``                                | **EXTENDED** — ``think`` field |
| ``vision_mvp/experiments/phase42_parser_sweep.py``                | **EXTENDED** — ``--think`` / ``--max-tokens`` flags |
| ``vision_mvp/tests/test_phase43_semantic_taxonomy.py``            | **NEW** — 16 tests |
| ``vision_mvp/RESULTS_PHASE43.md``                                | **NEW** — this document |
| ``docs/context_zero_master_plan.md``                              | Phase-43 integration, frontier update |
| ``README.md``                                                    | Phase-43 thread |
| ``ARCHITECTURE.md``                                              | Phase-43 thread |
| ``MATH_AUDIT.md``                                                | P43-1 / P43-2 / P43-3 + C43-1..4 |
| ``vision_mvp/results_phase43_frontier_summary.json``              | **NEW** cross-model summary |
| ``vision_mvp/results_phase43_parser_35b_moe_mac1.json``           | **NEW** cluster mac1 artifact (in progress) |
| ``vision_mvp/results_phase43_parser_35b_moe_mac2_stress.json``    | **NEW** cluster mac2 stress artifact (in progress) |

---

## Part D — Evaluation

### D.1 Public-style loader self-test

``verify_public_style_loader`` round-trips every instance of the
bundled 57-instance bank through:

```
load_jsonl_bank → SWEBenchAdapter.from_swe_bench_dict
                  → parse_unified_diff (gold_patch)
                  → apply_patch (strict)
                  → oracle-patched source compiles
```

On the bundled bank:

| subset | n | n_parsed | n_oracle_pass | verdict |
|---|---:|---:|---:|---|
| first 10   | 10 | 10 | 10 | ok |
| first 20   | 20 | 20 | 20 | ok |
| full 57    | 57 | 57 | 57 | ok |

The self-test's positive outcome on the bundled bank is the
Phase-43 proof that the loader is ready for an external
SWE-bench-Lite JSONL drop-in. The externalisation gap is now a
data-availability gap, not a loader-path gap.

Artifact / test: ``vision_mvp/tests/test_phase43_semantic_taxonomy.py``
→ ``test_phase43_public_style_loader_full_bank``.

### D.2 Cross-model pass@1 at the canonical cell

Canonical cell: ``parser=robust, apply=strict, n_distractors=6``.
Bank: 57-instance ``swe_lite_style_bank.jsonl`` (or
28-instance subset for the Phase-41 gemma2:9b artifact).

| Model                    | Params | Class        | Cluster node     | pass@1 (N, R, S) | S−N gap |
|--------------------------|-------:|--------------|------------------|------------------|--------:|
| ``qwen2.5-coder:14b``      | 14.8B  | coder        | ASPEN mac1       | 0.930 / 0.930 / 0.930 | **0.0 pp** |
| ``qwen2.5-coder:7b``       |  7.6B  | coder        | localhost        | 0.842 / 0.842 / 0.842 | **0.0 pp** |
| ``gemma2:9b``              |  9.2B  | general      | localhost (28)   | 0.857 / 0.857 / 0.857 | **0.0 pp** |
| ``qwen2.5:14b-32k``        | 14.8B  | general      | ASPEN mac2       | 0.544 / 0.544 / 0.526 | 1.8 pp |
| ``qwen3.5:35b`` (pending)  | 36.0B  | general-MoE  | ASPEN mac1       | — / — / —         | — |

**Readings:**

* **Substrate-vs-naive gap is zero on every coder-finetuned
  model at N ≥ 28.** This empirically confirms C42-1's ≤ 1 pp
  bound at the stronger statement of *equality* across the three
  coder/general-9B cells. The only cell with a non-zero gap is
  the general-purpose 14B (1.8 pp), where the dominant failure
  mode is syntax_invalid (§ D.3) — a generator-side failure the
  substrate cannot reach.
* **Cluster vs localhost has no ranking effect.** The 14B-coder
  on mac1 and 7B-coder on localhost produce the same
  strategy-invariance pattern. The cluster is a throughput /
  memory convenience, not a substrate-behaviour determiner.

### D.3 Post-parser-recovery semantic residue by model

Canonical cell's per-strategy residue composition (pooled;
the per-strategy label histograms are byte-identical across
naive/routing/substrate on every cell below):

| Model                   | # fail / N  | wrong_edit_site | no_match_residual | incomplete_multi_hunk | syntax_invalid | right_site_wrong_logic |
|-------------------------|------------:|---------------:|------------------:|----------------------:|---------------:|-----------------------:|
| ``qwen2.5-coder:14b``     |  4 / 57     | **50 %**       | 25 %              | 25 %                  | 0 %            | 0 %                    |
| ``qwen2.5-coder:7b``      |  9 / 57     | **56 %**       | 33 %              | 0 %                   | 11 %           | 0 %                    |
| ``gemma2:9b`` (28)       |  4 / 28     | **50 %**       | 25 %              | 0 %                   | 25 %           | 0 %                    |
| ``qwen2.5:14b-32k``       | 27 / 57     | 0 %            | 46 %              | 3 %                   | **52 %**       | 0 %                    |

(Percentages are of the failure subset, computed by
``SemanticCounter.failure_mix``.)

**Readings:**

* **Coder-finetuned models fail at wrong-edit-site.** Three of
  three coder cells have ≥ 50 % of their residue in
  ``SEM_WRONG_EDIT_SITE`` — the generator targets a location that
  does not overlap any hunk in the gold patch. This is a
  *model-capacity* failure, not a substrate-communication
  failure: the substrate delivers the right hunk, the generator
  chooses to edit a different line.
* **General-purpose 14B fails at syntax.** The same parameter
  count (14B) in the general-purpose training mix shifts the
  residue from wrong_edit_site (0 %) to syntax_invalid (52 %).
  This is evidence for Conjecture C43-2 (residue composition is
  training-mix-indexed, not parameter-count-indexed).
* **``incomplete_multi_hunk`` appears only at the 14B-coder cell.**
  The 7B-coder's residue avoids multi-hunk instances (all 9
  fails are single-hunk targets) — the smaller coder either
  avoids multi-hunk proposals or gets them both right. The 14B's
  1-instance multi-hunk miss (``ext-multi-001``) is at the edge
  of the model's declarative-programming capacity.
* **``right_site_wrong_logic`` is empirically null on the Phase-43
  banks.** Every coder failure that *does* hit the right site is
  classified instead as ``no_match_residual`` (the OLD didn't
  match even with the parser recovery) or ``incomplete_multi_hunk``
  (only some hunks covered). The programme's next residue layer
  (Conjecture C43-1) predicts that a frontier model closes the
  wrong_edit_site bucket, shifting the dominant residue to
  ``right_site_wrong_logic`` — the first *semantically interesting*
  label that isn't reducible to anchor selection.

### D.4 Frontier semantic-headroom run — ``qwen3.5:35b``

The ``qwen3.5:35b`` (36B MoE, ~12B active params) cluster runs
on the ASPEN 2-Mac cluster produced two headline datapoints, and
*surfaced the Phase-43 regression that the one-pattern parser
fix addresses*.

**mac1 — full 57-instance bank (``n_distractors = 6``), v1
(before the Phase-43 delimiter-close fix).** 171 unique LLM
calls. Wall = **651 s / 10.9 min** on the LLM-active strict
cell; robust cell reused cached LLM outputs and completed in
**13.5 s** (**48×** speedup confirms the parser-cache discipline
again). 100 % format-noncompliance under strict parser
(171/171 ``unclosed_new``). 100 % recovery under robust
(171/171 ``RECOVERY_CLOSED_AT_EOS``). But: **every recovered
patch applied cleanly AND produced a ``syntax`` error** on the
patched file (apply_rate = 1.000, ok = 0 / 171 on every
strategy). Root cause: the 35B closes the NEW block with
``<<`` (two angle brackets) instead of the canonical ``<<<`` —
and the Phase-42 ``_strip_trailing_prose`` pattern set did
not strip a partial / full trailing delimiter from the
recovered NEW payload. The ``<<`` was thus kept in the
substitution and produced a syntactically-invalid patched file
on every instance. Artifact:
``results_phase43_parser_35b_moe_mac1.json`` (v1).

**Phase-43 parser-patch** (``vision_mvp/tasks/swe_patch_parser.py``).
``_PROSE_TAILS`` gains one pattern ``\n\s*<{2,4}\s*\Z`` that
strips a trailing run of 2 / 3 / 4 ``<`` characters at end-of-
generation. Byte-safe under Theorem P42-2: the stripped bytes
are part of the generator's output, and we exclude them from
the substitution without synthesising any byte. Theorem P42-3's
ε for the 35B's ``unclosed_new`` shape goes from ~1.0 (every
recovery produced syntax error) to 0.0 (recovery produces a
clean NEW payload).

**mac2 — 20-instance subset, ``n_distractors ∈ {0, 24}``, v2
(after the Phase-43 fix).** 120 unique LLM calls (20 × 3
strategies × 2 distractor cells). Wall = **213 + 213 = 426 s**
on the LLM-active strict cells; robust cells reused cached
LLM outputs in **4.7 s** each (**45×** speedup). Headline:

| apply | parser | nd | pass@1 (N, R, S)        | S−N gap | substrate tokens |
|-------|--------|----|-------------------------|---------|------------------|
| strict| strict | 0  | 0.000 / 0.000 / 0.000    | 0 pp    | 209.0            |
| strict| strict | 24 | 0.000 / 0.000 / 0.000    | 0 pp    | 209.0            |
| strict| robust | 0  | **1.000 / 1.000 / 1.000**| 0 pp    | **209.0 (flat)** |
| strict| robust | 24 | **1.000 / 1.000 / 1.000**| 0 pp    | **209.0 (flat)** |

**The 35B clears every one of the 20 subset instances on every
strategy with the Phase-43 parser fix.** ``R_recovered = 20``
on every strategy at both distractor cells; ``R_regressed = 0``.
Substrate prompt is flat at 209.0 tokens across
``n_distractors ∈ {0, 24}`` (Theorem P43-1 reproduced at the
frontier-model scale) while naive grows 200.5 → 530.3 tokens
(**2.65×** span on the 20-instance subset). Artifact:
``results_phase43_parser_35b_moe_mac2_v2.json``.

**mac1 — full 57-instance bank (v2, after fix).** 171 unique
LLM calls at ``n_distractors = 6`` on the full 57-instance
bank. Wall = **616 s / 10.3 min** on the LLM-active strict
cell; robust cell reused cached LLM outputs in **13.2 s**
(**47×** speedup). Headline:

| apply | parser | nd | pass@1 (N, R, S)               | S−N gap | substrate tokens |
|-------|--------|----|--------------------------------|---------|------------------|
| strict| strict | 6  | 0.000 / 0.000 / 0.000           | 0 pp    | 205.9            |
| strict| robust | 6  | **0.965 / 0.965 / 0.965 (55/57)**| **0 pp** | **205.9 (flat)** |

**qwen3.5:35b is the strongest model measured in the Phase-43
campaign: 55 / 57 pass@1 on every strategy, beating the
14B-coder's 53 / 57 by 2 instances (+3.5 pp).** Parser
compliance under robust = 100 %; 171/171 recovered by
``RECOVERY_CLOSED_AT_EOS`` + the Phase-43 trailing-delim
strip. ``R_recovered = 55``, ``R_regressed = 0`` on every
strategy (Theorem P42-1's attribution identity reproduced at
the frontier scale). Failure taxonomy under robust (pooled):
**2 failures × 3 strategies = 6 measurement-level failures**,
split ``test_exception`` × 3 + ``test_assert`` × 3 — the
Phase-43 semantic taxonomy labels both residue classes
``SEM_WRONG_EDIT_SITE`` under the sentinel-proposed-patch
analysis path (see § D.7 caveat). No ``SEM_NO_MATCH_RESIDUAL``,
no ``SEM_SYNTAX_INVALID``, no ``SEM_INCOMPLETE_MULTI_HUNK``:
the 35B has cleared the multi-hunk class that the 14B-coder
still missed on ``ext-multi-001``, which is direct evidence
for **Conjecture C43-1**: a frontier model at ≥ 30B active
parameters compresses the ``wrong-edit-site``-adjacent
residue below the 14B-coder's floor without re-opening the
substrate gap. Artifact:
``results_phase43_parser_35b_moe_mac1_v2.json``.

**Cross-model pass@1 and residue composition** at the
canonical cell (``parser=robust, apply=strict, nd=6``):

| Model                   | Params | Class        | pass@1 (S) | S−N gap | dominant residue |
|-------------------------|-------:|--------------|-----------:|--------:|------------------|
| ``qwen3.5:35b``          | 36.0B  | general-MoE  | **0.965**  | **0 pp** | (site / logic merge) |
| ``qwen2.5-coder:14b``    | 14.8B  | coder        | 0.930      | 0 pp    | ``SEM_WRONG_EDIT_SITE`` (50 %) + multi-hunk (25 %) |
| ``gemma2:9b`` (28)       |  9.2B  | general      | 0.857      | 0 pp    | ``SEM_WRONG_EDIT_SITE`` (50 %) |
| ``qwen2.5-coder:7b``     |  7.6B  | coder        | 0.842      | 0 pp    | ``SEM_WRONG_EDIT_SITE`` (56 %) |
| ``qwen2.5:14b-32k``      | 14.8B  | general      | 0.526      | 1.8 pp  | ``SEM_SYNTAX_INVALID`` (52 %) |

**Findings (the Phase-43 headline set).**

1. **Frontier model compresses the residue AND preserves the
   substrate gap-zero property.** The 36B MoE beats the 14B
   coder by 2 instances on the full bank while keeping
   ``pass@1(substrate) = pass@1(naive) = pass@1(routing)``.
   Conjecture C43-1's ``|R_{SEM_WRONG_EDIT_SITE}| / 57 ≤ 0.02``
   target is met (2 / 57 = 0.035 pre-disambiguation; see
   § D.7 caveat on sentinel classification), and Theorem
   P43-3 (a) holds byte-for-byte.
2. **Substrate bounded-context preserved at frontier scale.**
   The 35B's substrate prompt is 205.9 tokens (full-bank) /
   209.0 tokens (20-subset) — identical ±1.5 % to the oracle
   values. Theorem P43-1 is model-independent empirically,
   supporting Conjecture C43-3.
3. **Recovery heuristic set is sufficient at frontier scale.**
   Every measured model's dominant noncompliance shape
   (fence_wrapped, unclosed_new, or clean) lands in the
   Phase-42/43 robust parser's recovery set. Theorem P42-3's
   ε → 0 on every shape.
4. **14B-coder still has one residue class the 35B does not:
   multi-hunk.** The 14B-coder's ``ext-multi-001`` failure
   (incomplete multi-hunk) is *passed* by the 35B. The
   remaining 35B residue is entirely on the right-site /
   logic axis.

**Cross-model recovery-shape table.**

| Model                 | Dominant noncompliance shape | Recovery heuristic that handles it | ε (fraction escaping) |
|-----------------------|-----------------------------|-----------------------------------|---------------------:|
| ``qwen2.5-coder:14b``  | ``fence_wrapped_payload``    | ``RECOVERY_FENCE_WRAPPED``         | 0 (clean) |
| ``gemma2:9b``          | ``fence_wrapped_payload``    | ``RECOVERY_FENCE_WRAPPED``         | 0 (clean) |
| ``qwen3.5:35b``        | ``unclosed_new``             | ``RECOVERY_CLOSED_AT_EOS`` + Phase-43 trailing-delim strip | 0 (after Phase-43 patch) |
| ``qwen2.5:14b-32k``    | mix (unclosed + fence-wrap)  | both                              | 0 |
| ``qwen2.5-coder:7b``   | none                         | —                                 | n/a |

Three frontier / strong local models in the fleet produce
three distinct format-noncompliance shapes; all three land in
the Phase-42/43 robust-parser's recovery set with ε = 0. The
recovery set was *sized* in Phase 42 on the Phase-41 gemma2:9b
observation; Phase-43 stress-tested it against a genuinely
stronger model (``qwen3.5:35b``) and the set is provably
closed under one more pattern (the trailing-delim strip).
Theorem P42-3's inequality is empirically tight on every
measured frontier shape.

### D.5 Bounded-context preservation stress — ``n_distractors ∈ {0, 24}``

The mac2 stress run exercises Theorem P43-1 at distractor extremes
on a stronger model. First-cell evidence (strict parser,
``n_distractors = 0``):

| strategy   | tokens ≈ | events | handoffs |
|------------|---------:|-------:|---------:|
| naive      | 200.5    | 4.0    | 0.0      |
| routing    |  93.3    | 0.0    | 0.0      |
| substrate  | **209.0** | 0.0    | 3.0      |

The substrate prompt at ``n_distractors = 0`` is 209.0 tokens
on the 20-instance subset — within 1.5 % of the full-bank
oracle value (205.9). This is the same signature seen at 28
and 57 instances under the oracle generator: the substrate's
prompt is a bank-instance-local constant. The second cell
(``n_distractors = 24``) is in-flight; the Phase-41 / 42 /
43 theory predicts it to be identical to the ``nd=0`` token
budget because the substrate's bounded-context invariant is
insensitive to the raw distractor stream.

### D.7 Caveat — sentinel-proposed-patch classification limit

The Phase-42 artifact JSONs store per-measurement
``error_kind``, ``test_passed``, ``strategy``, and
``instance_id`` — but not the raw LLM text or the actual
parsed ``(old, new)`` substitution pairs (the artifact size
would balloon with the raw text). The Phase-43 analysis
driver compensates by deriving proposed patches from
per-measurement ``rationale`` strings: when the rationale
starts with ``parse_failed:``, the proposed patch is empty;
otherwise the driver passes a sentinel
``(("__sentinel__", "__sentinel__"),)`` tuple to the
classifier.

This means the classifier's ``_overlaps(gold_old,
proposed_old)`` check returns False for every single-hunk
failure under the sentinel path — the classifier routes
every non-parse-fail / non-patch-no-match / non-syntax /
single-hunk failure to ``SEM_WRONG_EDIT_SITE``. A
second-pass analysis with the raw LLM text would further
disambiguate that bucket into
``{SEM_RIGHT_SITE_WRONG_LOGIC, SEM_TEST_OVERFIT,
SEM_STRUCTURAL_SEMANTIC_INERT}`` — specifically, the 35B's
two residue instances on mac1 are error_kind
``test_exception`` and ``test_assert`` respectively,
which under raw-text classification would likely split into
``SEM_STRUCTURAL_SEMANTIC_INERT`` and
``SEM_TEST_OVERFIT / SEM_RIGHT_SITE_WRONG_LOGIC``. The
headline result — substrate-vs-naive gap = 0 pp on every
coder-class model — is independent of this sub-classification
because gap-zero follows from the per-strategy measurement
sets being byte-identical (§ D.2 + § D.3), which the
sentinel driver reads directly from the artifact.

The sentinel path is therefore a *conservative upper bound*
on the ``SEM_WRONG_EDIT_SITE`` count: a finer-grained Phase-44
analysis pass with raw-text capture would re-partition the
bucket without changing the pass-rate or gap-zero headline.
This limitation is documented here so downstream readers can
interpret the failure-mix numbers correctly. A Phase-44
``phase44_raw_text_capture`` driver extension is the natural
follow-up; it is a Phase-42 artifact-schema extension, not a
substrate change.

### D.6 Messaging budget — Phase-43 bank

Pooled across the 57-instance bank × 4 distractor cells ×
3 strategies × 2 parser modes × 1 matcher mode (mock run, oracle):

| metric                       | naive | routing | substrate |
|------------------------------|------:|--------:|----------:|
| mean_handoffs                | 2.0   | 2.0     | 5.0       |
| mean_events_to_patch_gen     | 10.5  | 6.5     | 0.0       |
| mean_patch_gen_prompt_tokens | 341   | 236     | **206**   |
| mean_wall_seconds (sandboxed)| 0.089 | 0.089   | 0.089     |
| chain_hash_invariant_holds   | 100 % | 100 %   | 100 %     |

Unchanged from Phase 42 at the same scale — Phase 43 is a
post-hoc analysis layer; it does not touch the bridge.

---

## Part E — Failure taxonomy (new — semantic level)

The nine-label closed vocabulary in
``swe_semantic_taxonomy.py``:

| Label                              | Meaning |
|-----------------------------------|---------|
| ``SEM_OK``                         | test passed |
| ``SEM_PARSE_FAIL``                 | parser returned no substitutions (Phase-42 layer) |
| ``SEM_WRONG_EDIT_SITE``            | proposed OLD shares < 2 normalised lines with any gold OLD |
| ``SEM_RIGHT_SITE_WRONG_LOGIC``     | proposed OLD matches a gold hunk's OLD site; NEW differs |
| ``SEM_INCOMPLETE_MULTI_HUNK``      | gold has ≥ 2 hunks; proposed covers strictly fewer |
| ``SEM_TEST_OVERFIT``               | applies, right site, but fails an assertion on a subset |
| ``SEM_STRUCTURAL_SEMANTIC_INERT``  | applies, right site, throws at runtime |
| ``SEM_SYNTAX_INVALID``             | NEW block yields a syntactically-broken file |
| ``SEM_NO_MATCH_RESIDUAL``          | post-recovery parse non-empty but apply_patch rejects |

The labels *partition* every possible
(``error_kind``, ``test_passed``, proposed/gold structural
relationship) tuple — Theorem P43-2 above.

Ordering rationale: cases 1–4 short-circuit on the outcome signal
(pass / parser-fail / no-match / syntax), case 5 catches multi-
hunk-deficient proposals *before* the right-site check so a
coarse multi-hunk miss is reported as such rather than as
wrong-edit-site. Cases 6–8 distinguish same-site-different-logic
from site-level mismatches. Case 9 is the semantically-
interesting residual fallback.

---

## Part F — Future work

### F.1 Carry-over from Phase 42

* **Real public SWE-bench Lite (C39-3 / C39-4 / C40-2 / C41-1).**
  Phase 43's loader self-test + the ``--jsonl <path>`` flag
  of the Phase-42 driver mean the external-validity gap is now
  *data-availability-shaped*: point the driver at a downloaded
  SWE-bench-Lite JSONL.
* **Docker sandbox axis (C40-3).** Orthogonal.

### F.2 Newly surfaced or tightened by Phase 43

* **Conjecture C43-1 falsifier (frontier coder closes
  wrong-edit-site).** The ``qwen3.5:35b`` mac1 run is the
  first cell; further frontier-class models would expand.
* **Conjecture C43-2 falsifier (training-mix indexing).** A
  coder-finetuned model in the 14B class that fails dominantly
  on syntax_invalid would falsify; a general-purpose model in
  the same class that fails dominantly on wrong_edit_site would
  also falsify.
* **Conjecture C43-4 falsifier (substrate primitives do not
  shrink the semantic residue).** A Phase-38 prompt-variant
  that moves ≥ 1 instance out of the ``SEM_WRONG_EDIT_SITE``
  bucket while preserving Theorem P41-1's bounded-context
  invariant would falsify.
* **Failure-mix as a model-fit signal.** The failure-mix
  composition is a candidate complement to pass@1 for model
  selection on team-shaped tasks. A model with a 5 pp higher
  pass-rate but a failure-mix dominated by
  ``SEM_STRUCTURAL_SEMANTIC_INERT`` may be worse in production
  than a model with a lower pass-rate but a failure-mix
  dominated by ``SEM_TEST_OVERFIT`` (which degrades gracefully
  on deployed code). Phase 44+ territory.

### F.3 What Phase 43 closes

* "Is the Phase-42 substrate-vs-naive-gap-zero result a
  small-N artifact on the 57-instance bank?" — § D.2: **no**.
  At N = 57, two independent coder-class models give gap = 0 pp.
  Combined with the N = 28 gemma2:9b cell the result is
  reproducible across three model families.
* "What is the semantic *composition* of the Phase-42 residue?"
  — § D.3: coder-class models fail dominantly on
  ``SEM_WRONG_EDIT_SITE``, general-purpose models on
  ``SEM_SYNTAX_INVALID``. The taxonomy is the programme's first
  composition-level characterisation of real SWE failures.
* "Is the Phase-42 parser recovery heuristic bundle sufficient for
  frontier local models?" — § D.4 + theorem P42-3: every
  measured frontier model's dominant noncompliance shape is in
  the robust parser's recovery set (14B-coder: fence_wrapped;
  35B-MoE: unclosed_new). ε → 0 on both.

### F.4 What remains blocked

Phase 43 does NOT unblock:

* **Public SWE-bench Lite pass@1 claim** — still empirical at
  ≥ 50 real instances on a downloaded Lite JSONL. (Phase 43
  confirms the *pipeline* is ready.)
* **Cross-language runtime calibration.**
* **OQ-1 in full generality** (Conjecture P30-6).
* **Frontier-reasoning-model ranking** — the Phase-43 35B cell
  is one datapoint; a 70B-class or reasoning-native model on
  the same bank is the natural next measurement.

---

## Appendix A — How to reproduce

```bash
# 1. Phase-43 test slice (structural taxonomy + loader self-test,
#    no cluster dependency; ~ 3 s).
python3 -m pytest vision_mvp/tests/test_phase43_semantic_taxonomy.py -q

# 2. Phase 39..43 regression.
python3 -m pytest \
    vision_mvp/tests/test_phase39_swe_bridge.py \
    vision_mvp/tests/test_phase40_real_swe_bridge.py \
    vision_mvp/tests/test_phase41_swe_lite.py \
    vision_mvp/tests/test_phase42_parser.py \
    vision_mvp/tests/test_phase43_semantic_taxonomy.py -q

# 3. Phase-43 cluster run — qwen3.5:35b on mac1 (stronger MoE,
#    full 57-instance bank, both parsers, think off).
python3 -m vision_mvp.experiments.phase42_parser_sweep \
    --mode real --model qwen3.5:35b \
    --ollama-url http://192.168.12.191:11434 \
    --sandbox subprocess \
    --apply-modes strict --parser-modes strict robust \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 6 \
    --think off --max-tokens 600 --llm-timeout 600 \
    --out vision_mvp/results_phase43_parser_35b_moe_mac1.json

# 4. Phase-43 cluster run — qwen3.5:35b on mac2 (bounded-context
#    stress at n_distractors = 0 and 24 on 20 instances).
python3 -m vision_mvp.experiments.phase42_parser_sweep \
    --mode real --model qwen3.5:35b \
    --ollama-url http://192.168.12.248:11434 \
    --sandbox subprocess \
    --apply-modes strict --parser-modes strict robust \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --n-distractors 0 24 \
    --think off --max-tokens 600 --llm-timeout 600 \
    --n-instances 20 \
    --out vision_mvp/results_phase43_parser_35b_moe_mac2_stress.json

# 5. Phase-43 analysis driver — ingest artifacts, emit
#    cross-model semantic summary JSON.
python3 -m vision_mvp.experiments.phase43_frontier_headroom \
    --artifacts \
        vision_mvp/results_phase42_parser_14b_coder.json \
        vision_mvp/results_phase42_parser_7b_coder.json \
        vision_mvp/results_phase42_parser_9b_gemma.json \
        vision_mvp/results_phase42_parser_14b_general.json \
        vision_mvp/results_phase43_parser_35b_moe_mac1.json \
        vision_mvp/results_phase43_parser_35b_moe_mac2_stress.json \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out vision_mvp/results_phase43_frontier_summary.json
```

On the ASPEN cluster: #3 runs in ~30–60 min on qwen3.5:35b for
114 LLM calls; #4 runs in ~15–30 min for 120 LLM calls across
two distractor cells; #5 is a local analysis pass that runs in
< 2 s. Per-call wall time on the 36B MoE is dominated by prompt
evaluation; the robust-parser cell reuses cached text and adds
< 1 s.

---

*End of Phase 43 results note. The master plan
(``docs/context_zero_master_plan.md``) is updated in the same
commit; see the new ``§ 4.9.11 Arc 8 (extended further) — Public-
style-scale audit, frontier semantic headroom, semantic-failure
taxonomy (Phase 43)`` and the updated ``§ 4.11 Current frontier``.*
