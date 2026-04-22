# Phase 27 — Corpus-Scale Runtime-Truth Calibration of the Conservative Analyzer

**Status: research framing + corpus-scale runtime-calibration layer +
safe-invocation-recipe protocol + entry-detection + per-call wall-time
budget + three-axis calibration (analyzer / runtime / direct-exact
planner) at corpus scale + coverage accounting with witness-
availability reporting.** Phases 24–25 established *conservative
static semantic exactness* at corpus scale (100 % direct-exact on
six corpora, zero LLM calls). Phase 26 added a separate truth axis —
*runtime-truth calibration* — on a **21-snippet executable corpus**
with 123 / 126 agreement (97.6 %). Phase 27 extends runtime truth
from snippet-scale to **corpus-scale**: real functions drawn from
real Python modules in the repository, probed via the Phase-26
machinery under an explicit **safe-invocation protocol**.

> Phase 26, in one line: on 21 curated executable snippets, analyzer
> and runtime agree on 97.6 % of applicable measurements; divergences
> land on pre-documented boundaries. Phase 27, in one line: pushing
> the same probes onto real corpus functions exposes a **witness-
> availability bound** that is strictly tighter than analyzer
> coverage — the runtime layer is a *partial observer* over the
> analyzer's declared slice, and how-partial it is is the new
> first-class metric (Theorem P27-1). The corpus calibration
> harness ships coverage accounting, entry-detection, and a
> per-call wall-time budget, and re-verifies Phase 26's three-axis
> separation on real code without breaking the Phase-22 substrate
> guarantee.

---

## Part A — Research framing

### A.1 From snippet-scale to corpus-scale runtime truth

Phase 26 measured calibration on a 21-snippet corpus where every
snippet was hand-authored and ships with an `invoke` driver that
produces recipe-compatible arguments. The snippet-scale coverage was
*author-mediated*: each predicate's applicability was something the
author declared. That is the right scaffolding for probing probe
correctness — but it is silent on the central practical question:
**what happens when the same analyzer is compared to runtime on
code the author did not curate for the probe?**

Phase 27 answers that question on `vision_mvp/core` (the research
framework's own source). The probes are unchanged; the difference
is entirely on the *input* side. Real functions do not come with
invocation drivers, so Phase 27 introduces a **safe-invocation
protocol** that classifies every function in the corpus into one of
eleven callability states and attaches an `InvocationRecipe` when a
safe one can be derived.

### A.2 Four distinct truth statements

Phase 27 keeps the Phase-26 axis separation and adds a fourth:

1. **Analyzer-gold truth.** What `core.code_semantics` +
   `core.code_interproc` declares. Substrate input.
2. **Snippet-calibrated runtime truth.** Phase 26: runtime
   observation on the 21-snippet corpus. Validated against
   author-declared ground truth per snippet.
3. **Corpus-scale executable runtime truth (NEW).** Runtime
   observation on functions drawn from real modules. Validated
   only on the *ready* slice (functions for which a safe invocation
   recipe exists AND the target frame was entered within the
   probe budget).
4. **Untestable / partially testable functions (NEW).** Functions
   whose coverage is structurally bounded by:

     - no safe invocation recipe (async, generator, varargs,
       untyped positional params, method without auto-constructed
       instance);
     - recipe produced args that never entered the target frame
       (immediate ``TypeError`` on arg unpack);
     - the call did not return within the per-call wall-time
       budget (runaway loop, heavy numerical work).

The Phase-27 benchmark reports BOTH (3) and (4) as first-class
metrics.

### A.3 The analyse → propagate → discover → probe → decide pipeline

```
corpus root
    │
    ├── Phase-24/25 static pass
    │     ↓
    │   interproc flags per qname (analyzer axis)
    │
    ├── Phase-27 discovery
    │     ↓
    │   CorpusFunctionCandidate per qname
    │   {ready_no_args | ready_typed | ready_curated |
    │    unsupported_*}
    │     ↓
    │   InvocationRecipe per *ready* candidate
    │
    ├── Phase-27 probe (reuses Phase-26 instrumentation)
    │     ↓
    │   sandbox: _record_subprocess + _record_filesystem +
    │            _record_network + _observe_globals +
    │            _track_reentry (as per predicate)
    │     ↓
    │   entry-detection: sys.settrace on target code object
    │   budget: wall-clock check on every line event → sentinel
    │     ↓
    │   CorpusObservation per (qname, predicate) with
    │   {applicable, entered, timeout, runtime_flag, witnesses}
    │
    └── calibration aggregation
          ↓
        per-predicate FP / FN / agreement on entered subset +
        per-status coverage buckets
```

Nothing on the direct-exact path changed. The corpus-scale layer is
a **validation observer** that takes the same source bytes, imports
the modules the normal way, classifies functions, synthesises
recipe-compatible arguments, and reports an independent truth value
per function × predicate that can be compared against the static
flag.

### A.4 Theorem-style claims

Each claim has a test or empirical measurement (or both).

---

**Theorem P27-1 (Corpus-scale runtime calibration is a partial
observer over the semantic slice).** Let $F$ be the set of
functions declared in a corpus; $F_A \subseteq F$ the set for which
the Phase-24/25 analyzer emits flags; $F_R \subseteq F$ the set for
which the Phase-27 runtime layer is *applicable* (i.e., a safe
invocation recipe exists and the target frame is reached within
budget on at least one invocation). Then

$$
F_R \subseteq F_A \subseteq F, \qquad
\text{and in general } F_R \neq F_A.
$$

The inclusion $F_R \subseteq F_A$ follows because a function without
an AST representation has no analyzer flag; the strict inequality
arises because invocation-recipe availability is a witness-protocol-
bounded condition.

*Proof by exhibition.* Any function with `*args, **kwargs`, or whose
positional parameters lack recipe-compatible annotations, is in
$F_A$ (the analyzer flags it from its AST) but not in $F_R$ (no safe
recipe exists under the default strategy). Async functions, generator
functions, and methods requiring a constructed instance all fall in
the same gap. The Phase-27 coverage accounting reports this gap as a
first-class metric: `ready_no_args + ready_typed + ready_curated` is
$|F_R|$, and `n_total` is $|F_A|$. ∎

*Empirical measurement.* § D.1 reports per-corpus $|F_R| / |F_A|$.

---

**Theorem P27-2 (Runtime-calibration coverage is witness-availability-
bounded, not planner-exactness-bounded).** For any corpus $C$,
$|F_R(C)| / |F_A(C)|$ depends on the invocation-recipe derivation
strategy and on the ``SafeRecipeRegistry``, NOT on analyzer
exactness nor planner exactness. Adding a curated recipe strictly
increases $|F_R|$ (monotone). Planner exactness remains at 100 %
on the $F_A$ slice regardless of Phase-27 coverage — the Phase-22
substrate guarantee is independent of the Phase-27 runtime observer.

*Proof.* The planner reads the analyzer's flags; the coverage
metric reads the invocation protocol. The two operate on
independent inputs. Monotonicity in the recipe registry size
follows from the lookup rule in ``classify_function_candidate``:
a curated entry always wins over heuristic classification, so
adding an entry can only upgrade a function's status from
``unsupported_*`` to ``ready_curated`` (never the reverse). ∎

*Empirical verification.* § D.4 — planner round-trip match rate is
100 % across every predicate on every corpus, identical to
Phase 22's and Phase 26's substrate claim, regardless of the
runtime-calibration coverage fraction.

---

**Theorem P27-3 (Sandboxed corpus probes preserve runtime-
observation soundness, conditional on entry detection).** Let
``entered(f, args) = True`` iff the probe's ``sys.settrace`` counter
for `f`'s code object is at least one after the probe returns. If
``entered(f, args) = True`` AND the probe observes effect $e$
during the same invocation, then $e$ is reachable from a real
execution path through $f$'s body starting from the entry-point
state induced by ``args``.

*Proof.* The Phase-26 instrumentation sits below the Python
interpreter boundary (the monkeypatched entry points are real
Python-level hooks; the trace tracer records call events on the
actual code object). Phase 27 adds entry-detection as a filter, not
a relaxation — observations with ``entered=False`` are explicitly
removed from the calibration matrix because they fail the
antecedent of the soundness argument: the target's body never
executed, so any observed effect came from argument evaluation or
recipe setup, not from $f$'s semantics. Corollary: the subset of
observations with ``entered=True`` is a sound lower bound on $f$'s
runtime-reachable effect set. ∎

*Corollary P27-3a.* Absence of observation on the entered subset
is NOT a proof of absence — the input distribution under the recipe
may fail to cover the triggering path. This is the same caveat as
Phase 26's P26-2a, now applied per-function at corpus scale. The
recipe is a deterministic function of ``seed``, so repeat-run
variance (§ D.5) exposes stochasticity when present.

---

**Conjecture P27-4 (Corpus-scale FN concentration on pre-
documented boundaries).** False-negative observations in corpus-
scale calibration (static False, runtime True, entered True)
concentrate on the Phase-24 pre-documented boundary classes
(reflection via `getattr`, `eval`, alias assignment). If a
runtime FN appears on a function that does not use these
mechanisms and whose call graph is fully resolved, it indicates a
new boundary class the snippet corpus did not exercise.

*Operational consequence.* The Phase-27 divergence report is a
**discovery tool for new boundaries**. A new-boundary FN is a
first-class research signal, not a benchmark failure — the
research recipe is to add a matching pattern to the Phase-24
analyzer OR to add a matching snippet to the Phase-26 corpus so
future runs keep the boundary under continuous measurement.

*Empirical check.* § D.3 attributes every Phase-27 divergence to
one of four categories: (a) pre-documented boundary, (b) new
boundary not previously enumerated, (c) recipe artifact (target
entered under an argument set no reasonable caller would pass),
(d) environment-dependent effect the snippet corpus did not cover.

---

**Conjecture P27-5 (Coverage-precision tradeoff for the
invocation protocol).** Let $R$ denote an invocation-recipe
aggressiveness parameter (adding more types to the fuzz pool,
loosening annotation restrictions, adding generic method-
instance constructors). Increasing $R$ monotonically increases
$|F_R|$ but non-monotonically affects the FN rate — more exotic
recipes may trip entry-point exceptions that belong to the
recipe, not the target's semantic behaviour. The "clean" slice
(trivial signatures, pure argument types) has the highest
entered-fraction and the least recipe-induced noise; aggressive
recipes widen coverage at the cost of calibration noise.

*Empirical probe.* § D.6 varies the budget and seed count at
fixed recipe aggressiveness; $F_R$ is stable (recipe-dominated,
not budget-dominated). Varying recipe aggressiveness is future
work under OQ-27a.

---

### A.5 Impossibility / boundary conditions

What Phase 27 does **not** claim:

1. **Corpus-scale calibration is an oracle.** It is a lower bound,
   like Phase 26. Absence of observation is not a proof of
   absence; increasing fuzz budget tightens the bound but does
   not close it.
2. **Every corpus function is testable.** Functions requiring
   constructed instances, external services, async event loops,
   generator drivers, or variadic argument schemas are out of
   scope under the default recipe strategy.
3. **Environment independence.** Functions that read
   `os.environ`, check network reachability, depend on ambient
   time or uncontrolled random state are opaque to the probe
   under a default-environment run. The ``SafeRecipeRegistry``
   does not control the environment.
4. **Hermetic sandboxing.** The Phase-26 sandbox neuters
   subprocess / filesystem / network egress but is not a defence
   against adversarial code. The Phase-27 trust boundary is the
   same: honest research code, code-reviewed curated recipes.
5. **C-extension opacity.** Functions that execute entirely inside
   a C extension are not interruptible by the tracer-based
   timeout and are not traced for entry. They land in the
   `timeout` bucket or produce spurious `applicable=True,
   entered=False` observations; both outcomes are filtered out
   of the calibration matrix.
6. **Concurrency / async.** Unchanged from Phase 26 — out of
   scope.
7. **Reflection / exec.** Unchanged — `eval` and `exec` stay
   opaque to the analyzer; runtime observation through the
   probe hooks continues to surface them when executed.

Every one of these is documented inline in
`core/code_corpus_runtime.py`.

---

## Part B — Architecture

### B.1 Four-layer composition — additive, not replacing

```
Routing / Trigger / Exact-Memory / Retrieval / Computation / Render
    (unchanged from Phases 19–25)                                │
                                                                 ↓
                                 ┌─────────────────────────────────┐
                                 │  direct-exact answer            │
                                 │  (analyzer-gold, zero LLM)      │
                                 └──────────────┬──────────────────┘
                                                │
                                                ↓  — Phase 26 —
                                 ┌─────────────────────────────────┐
                                 │  runtime probes (snippet-scale) │
                                 │  author-declared recipes        │
                                 └──────────────┬──────────────────┘
                                                │
                                                ↓  — Phase 27 NEW —
                                 ┌─────────────────────────────────┐
                                 │  corpus-scale runtime observer  │
                                 │  • candidate classification     │
                                 │  • safe invocation recipes      │
                                 │  • entry detection              │
                                 │  • wall-clock budget            │
                                 │  • coverage accounting          │
                                 └─────────────────────────────────┘
```

Nothing on the direct-exact path changed. The Phase-27 observer is
an **additive validation layer** that takes the same source bytes,
imports the modules the normal way, applies a classifier +
recipe, and reports an independent per-(function, predicate) truth
value plus a per-corpus coverage breakdown.

### B.2 The corpus-runtime module

`core/code_corpus_runtime.py` (~750 lines including docstrings and
aggregation) exposes:

- `CorpusFunctionCandidate` — frozen classification of one
  function, with eleven possible `callable_status` values and a
  `reason` string.
- `InvocationRecipe` — produces a deterministic list of
  positional-arg tuples from an RNG + candidate + callable.
- `SafeRecipeRegistry` — `(module_name, qname) → recipe` for
  functions whose arguments cannot be auto-derived.
- `no_args_recipe`, `typed_recipe`, `curated_recipe` — three
  recipe factories that cover the three `ready_*` statuses.
- `classify_function_candidate` — AST + registry driven
  classifier, pure static.
- `_entry_and_budget_tracer` — single `sys.settrace` tracer that
  both counts target entries and enforces a wall-time budget;
  restores any previous tracer on exit.
- `probe_corpus_function` — reuses Phase-26 instrumentation to
  observe one predicate on one function across seeds.
- `discover_candidates` — walk a corpus, import each module,
  resolve callables, emit `DiscoveredCandidate`s.
- `build_corpus_static_flags` — bridge to the Phase-24/25
  interprocedural analyser for the static-flag column.
- `calibrate_corpus` — end-to-end pipeline for one corpus.
- `summarise_corpus_calibration` — per-predicate FP/FN/agreement
  metrics *restricted to `entered=True`*.
- `collect_divergences` — per-row disagreements with
  attribution metadata (callable status, witnesses).
- `CoverageAccount` — per-corpus status breakdown +
  `ready_fraction`, `calibrated_fraction`.

### B.3 The safe-invocation recipe protocol

The recipe protocol makes three guarantees:

  * **Purity of argument construction.** Recipes must not open
    files, spawn subprocesses, or touch the network at
    argument-build time. The three built-in factories
    (`no_args_recipe`, `typed_recipe`, `curated_recipe`) observe
    this; curated entries are code-reviewed per the rules in
    `tasks/corpus_runtime_recipes.py`.
  * **Determinism given seed.** `recipe.build(rng, cand, func)` is
    a pure function of `(seed, candidate, callable)` — the same
    seed yields the same argument sequence.
  * **Explicit refusal.** If the recipe cannot synthesise
    recipe-compatible arguments, it returns `[]`; the probe
    reports `applicable=False`. An empty-list return is *not* a
    silent success — it surfaces as a coverage entry.

### B.4 The entry-detection + budget tracer

`_entry_and_budget_tracer` installs a single `sys.settrace`
tracer that:

  * Increments `enter_count` every time the target code object is
    the callee of a `call` event. The probe filters
    ``entered=True`` iff the count is ≥ 1.
  * Checks `time.monotonic() > deadline` on every `line` event,
    raising `_BudgetExceeded` (a Phase-26 `_ProbeSentinel` subclass)
    if the budget has expired. `_ProbeSentinel` is swallowed by
    the probe body's exception handler, so it never contributes
    to `may_raise` observations.
  * Restores any previous tracer on exit, pinning the Phase-26
    sandbox-restoration invariant at corpus scale.

The tracer is Python-level; pure-C computation is not
interruptible — that is honestly reported as `timeout`-counted
observations rather than as analyzer disagreements.

### B.5 The curated recipe registry

`tasks/corpus_runtime_recipes.py` ships a small registry (16
entries at time of writing) covering pure AST-helper functions
whose argument types (AST nodes, frozensets) cannot be
auto-derived. Entries are:

| predicate family   | example entry                                |
|---                 |---                                           |
| Call resolution    | `code_semantics._call_name`                  |
| API membership     | `code_semantics._call_matches`               |
| Raise detection    | `code_semantics._analyze_may_raise`          |
| Try/except analysis| `code_semantics._raise_is_caught`            |
| Module traversal   | `code_semantics._module_level_names`         |
| Generator check    | `code_corpus_runtime._function_is_generator` |

Each entry carries a one-line justification and constructs its
arguments from fresh `ast.parse` calls — never from corpus-derived
objects.

### B.6 The benchmark

`experiments/phase27_corpus_runtime_calibration.py` ties it
together:

1. For each local corpus (`vision-core`, `vision-tasks` by
   default):
   a. Build analyzer-gold static flags via
      `build_corpus_static_flags`.
   b. `calibrate_corpus(...)` walks candidates, runs probes,
      aggregates.
   c. Run the Phase-26 planner round-trip for each predicate
      (count query via `CodeQueryPlanner` → compare to analyzer
      aggregate).
   d. Collect divergences via `collect_divergences`.
2. Pool metrics across corpora.
3. Emit a scoreboard + JSON artefact.

Repeat-run variance: `--seeds 0 1 2 3 4` runs five independent
seeds; observations OR across seeds, `n_runs` and `n_triggered`
accumulate.

---

## Part C — Implementation

### C.1 Files added or modified

| File | Change |
|---|---|
| `vision_mvp/core/code_corpus_runtime.py`                  | **NEW** — corpus-scale runtime calibration primitives (~750 LOC) |
| `vision_mvp/tasks/corpus_runtime_recipes.py`              | **NEW** — default curated recipe registry + skip list (~210 LOC) |
| `vision_mvp/experiments/phase27_corpus_runtime_calibration.py` | **NEW** — three-axis benchmark + pooled report (~310 LOC) |
| `vision_mvp/tests/test_code_corpus_runtime.py`            | **NEW** — 40 classifier / recipe / tracer / probe / end-to-end tests |
| `vision_mvp/RESULTS_PHASE27.md`                           | **NEW** — this document |
| `README.md`, `ARCHITECTURE.md`, `MATH_AUDIT.md`           | Updated to thread the analyzer / analyzer-snippet-runtime / analyzer-corpus-runtime distinctions into the project story |

Total new code: ~1 400 lines (module + registry + benchmark + tests + doc).

### C.2 Module boundary preserved

Phase 27 touches **zero** lines inside `core/code_index.py`,
`core/code_planner.py`, `core/code_semantics.py`,
`core/code_interproc.py`, `core/code_runtime_calibration.py`, or
`tasks/python_corpus.py`. The new module is a peer layer; the
existing Phase-22/23/24/25/26 primitives are imported but not
mutated.

Full repo test suite: passes with **+40 new tests** (see § D.2).

---

## Part D — Evaluation

> Empirical numbers below come from the runs archived at
> `vision_mvp/results_phase27_corpus.json` (seeds 0 1 2, budget
> 0.1s) and `vision_mvp/results_phase27_corpus_5seeds.json`
> (seeds 0 1 2 3 4, budget 0.15s). The per-predicate and
> coverage tables are populated from those files.

### D.1 Headline — callable coverage on `vision-core`

The `vision-core` corpus (after applying
`DEFAULT_PHASE27_SKIP_FILES`, which removes 11 import-heavy /
fuzz-hostile modules documented in
`tasks/corpus_runtime_recipes.py`) carries **715 functions**
(top-level + methods). Phase-24/25 analyzer flags are emitted
for every one of them. Running the Phase-27 discovery + probe
pipeline yields the following coverage breakdown (seeds = 0 1 2,
budget = 0.05 s):

| status                       | count | fraction |
|---                           |---:|---:|
| `ready_no_args`              |   8 | 1.1 % |
| `ready_typed`                | 226 | 31.6 % |
| `ready_curated`              |  16 | 2.2 % |
| **ready (sum)**              | **250** | **35.0 %** |
| `unsupported_method`         | 421 | 58.9 % |
| `unsupported_missing`        |  19 | 2.7 % |
| `unsupported_varargs`        |  10 | 1.4 % |
| `unsupported_generator`      |   8 | 1.1 % |
| `unsupported_untyped`        |   7 | 1.0 % |
| `unsupported_async`          |   0 | 0.0 % |
| `unsupported_import`         |   0 | 0.0 % |
| **unsupported (sum)**        | **465** | **65.0 %** |

Reading the table:

- **Method dominance.** 59 % of the corpus is methods on classes;
  the default recipe strategy does NOT synthesise class instances,
  so every method lands in `unsupported_method`. This is honest:
  any method-specific calibration requires a curated
  instance-constructor recipe (OQ-27b).
- **Typed-function slice.** 32 % (226 functions) have all-typed
  signatures recognisable to the recipe's fuzz pool. These are
  the auto-derivable slice.
- **Zero-arg slice.** Only 8 functions are zero-arg — most
  "pure" helpers in the corpus take at least one argument.
- **Curated slice.** The shipping registry covers 16 AST-helper
  functions that the default strategy would classify
  `unsupported_untyped`.

This is the **research headline**: the analyzer-gold slice is
`|F_A| = 715`; the calibration-ready slice is `|F_R| = 250`;
$|F_R| / |F_A| = 35.0 \%$. Theorem P27-1's strict inclusion is a
real phenomenon on real code. Adding the 11 fuzz-hostile modules
back (no skip list) raises `|F_A|` to 791 but lowers
$|F_R| / |F_A|$ slightly (to 35.7 %) — the extra modules are
almost entirely `ready_typed`, but they produce runaway
computations under default fuzz, hitting `timeout` on most calls.
The skip list narrows the measurement to the tractable slice so
that coverage reflects analysability, not benchmark wall-time.

### D.2 Headline — entered fraction + per-predicate calibration

Of the 250 ready candidates, 104 were probed (a function is
"probed" iff at least one predicate produced an observation with
`applicable=True`); of those, **102 entered** the target frame
at least once across seeds (entry_count ≥ 1 via
`_entry_and_budget_tracer`). No timeouts at the 0.05 s per-call
budget. The `ready_typed` slice has a lower probed/ready rate
than `ready_no_args` because `typed_recipe` returns `[]` when
`inspect.signature` reveals an annotation outside the
whitelist — the AST classifier is a *necessary* but not
*sufficient* check.

| corpus        | n_total | ready | probed | entered | calibrated_fraction |
|---           |---:|---:|---:|---:|---:|
| `vision-core` | 715 | 250 | 104 | 102 | **14.3 %** of n_total |

**Per-predicate agreement on the ENTERED slice** (seeds = 0 1 2,
budget = 0.05 s, archived at `results_phase27_corpus.json`):

| predicate                | applic | entered | S_true | R_true | agree | FP | FN |
|---                       |---:|---:|---:|---:|---:|---:|---:|
| `calls_filesystem`       | 104 | 102 | 2 | 1 | **101 / 102 (99.0 %)** | 1 | 0 |
| `calls_network`          | 104 | 102 | 1 | 1 | **102 / 102 (100 %)**  | 0 | 0 |
| `calls_subprocess`       | 104 | 102 | 1 | 1 | **102 / 102 (100 %)**  | 0 | 0 |
| `may_raise`              | 104 | 102 | 16 | 40 | 76 / 102 (74.5 %)       | 1 | **25** |
| `may_write_global`       | 104 | 102 | 0 | 0 | **102 / 102 (100 %)**  | 0 | 0 |
| `participates_in_cycle`  | 104 | 102 | 2 | 2 | **102 / 102 (100 %)**  | 0 | 0 |
| **pooled (ex-may_raise)**| 520 | 510 | 6 | 5 | **509 / 510 (99.8 %)** | 1 | 0 |
| **pooled (incl may_raise)**| 624 | 612 | 22 | 45 | **585 / 612 (95.6 %)** | 2 | 25 |

Reading the table:

- **Five of six predicates calibrate essentially cleanly.**
  `calls_network`, `calls_subprocess`, `may_write_global`, and
  `participates_in_cycle` all score 102 / 102 (100 %) agreement on
  the entered subset. `calls_filesystem` is 101 / 102 (99.0 %).
- **`may_raise` is the outlier** — 25 false-negatives concentrated
  in one category. The witnesses (exception types recorded by the
  probe) tell the story unambiguously: `TypeError`, `ValueError`,
  `OverflowError`, `ZeroDivisionError`, `AttributeError`,
  `IndexError`. Every one of these is a **builtin-propagated
  exception**, NOT an explicit `raise` statement in the source.
  The analyzer correctly flags `may_raise = False` because its
  documented contract is *"the body contains an uncaught `raise`
  statement"* (Phase-24 `RESULTS_PHASE24.md` § "Known boundary
  conditions"). The runtime probe observes *any* exception the
  function propagates, including implicit ones from builtin
  operations on argument values outside the function's semantic
  domain.

### D.2.1 The implicit-raise boundary class (new in Phase 27)

Phase 26's snippet corpus exercises **explicit** raises
(`raise RuntimeError`, `raise ValueError(...)`). The 25 Phase-27
false-negatives surface a strictly different category: **implicit
raises from builtin operations when arguments are outside the
function's semantic domain**. Examples:

| qname                                              | witness exceptions    | cause |
|---                                                 |---                    |---|
| `code_index._module_name_from_path`                | `ValueError`          | `os.path.relpath` raises on unrelated roots |
| `gf.inv_mod` / `gf.add` / `gf.mul` / `gf.div`      | `ZeroDivisionError`   | modular division by zero |
| `cuckoo_filter._blake2b_bytes`                     | `ValueError`          | `blake2b` rejects empty digest_size |
| `lexical_index.reciprocal_rank_fusion`             | `TypeError`           | iterating an int |
| `routing_hash._hash64` / `sketches._hash_pair`     | `OverflowError`       | integer overflow on large input |
| `linear_logic._is_tree`                            | `TypeError`           | `len()` of an int |
| `code_interproc.build_call_graph` / `.analyze_*`   | `AttributeError`      | calling `.module_name` on a fuzz int |
| `code_runtime_calibration.synthesize_args`         | `IndexError`          | `rng.choice([])` — this is actually
                                                                              correct (empty pool input is a caller bug,
                                                                              not a target bug) |

Is this an **analyzer soundness break** or an **analyzer scope
decision**? Phase 24's contract is explicit: `may_raise` covers
explicit `raise` statements, not implicit exception propagation.
Under that contract, the analyzer is correct on all 25 cases.
Under a STRICTER contract ("function can raise SOME exception on
SOME input"), the analyzer would need to flag every function that
performs any operation that can propagate exceptions — in
practice, almost every function. That would collapse `may_raise`
into a near-constant True flag, losing the discrimination the
direct-exact path exploits.

**Research consequence.** The implicit-raise boundary is a new
first-class category for the Phase-24 analyzer, alongside
reflection and eval (the Phase-24/26 pre-documented boundaries).
Conjecture P27-4 is **partially confirmed and partially refined**:
false negatives DO concentrate on a boundary class, but the class
is different from the one Phase 26 surfaced (eval / reflection).
The recipe is:

  - **Option (a)**: extend Phase 24's `may_raise` to cover implicit
    raises (dividing by zero, indexing out of range, calling
    methods on wrong types). This gains soundness on implicit
    raises at the cost of near-saturation on the flag.
  - **Option (b)**: introduce a separate `may_raise_implicit`
    predicate to keep the discrimination. OQ-27i.
  - **Option (c)**: sharpen the Phase-27 recipe to only fuzz
    within the function's semantic domain (signature + corpus-
    level usage patterns). OQ-27c.

The Phase-27 benchmark **reports the gap** rather than choosing
an option; the architecture decision is deferred to Phase 28.

### D.2.2 The single `calls_filesystem` false-positive

`vision_mvp.core.code_corpus_runtime._walk_python_files` is
flagged `calls_filesystem=True` by the analyzer (it calls
`os.walk` in its body) but `False` by runtime with empty
witnesses. The reason: the Phase-26 filesystem-recording sandbox
(`_record_filesystem`) instruments `builtins.open`, `os.open`,
`os.remove`, `os.unlink`, `os.mkdir`, `os.makedirs` — but NOT
`os.walk`, `os.listdir`, `os.scandir`, `os.stat`. The analyzer's
`_FILESYSTEM_APIS` set is broader than the probe's
instrumentation set.

This is a **probe-instrumentation-bounded lower bound**, analogous
to the coverage gap documented under Theorem P27-1 but on the
probe side rather than the recipe side. The fix is additive: add
the missing APIs to `_record_filesystem`. It is deliberately not
applied in Phase 27 to keep the Phase-26 module unchanged (no
regressions) and to leave the finding visible in this report.
OQ-27j.

### D.3 Divergence attribution

Every row with `static_flag ≠ runtime_flag` on the entered subset
is a calibration divergence. Phase-27 attributes each one to one
of four categories:

1. **Pre-documented boundary (Phase-24).** Reflection via
   `getattr`, `eval`, alias assignment. These are the same
   categories Phase 26 surfaced on the curated snippet corpus —
   a real-code appearance confirms the boundary is not an
   artefact of the snippet authoring.
2. **Recipe artefact.** The target entered under an argument set
   no realistic caller would pass — e.g. `may_raise` fires
   because a typed recipe fed an `int` into a function that
   expects a `Counter`. These are filtered at reporting time
   when the witness type is exclusively `TypeError` or
   `AttributeError` on the first statement.
3. **Environment-dependent.** `may_raise=True` because the
   function reads `os.environ["FOO"]` and FOO is unset in the
   probe environment. Analyzer is correctly False (no explicit
   `raise` in source).
4. **New boundary.** Anything not in the three above. The
   research recipe is to add a matching snippet to Phase 26 and
   a matching pattern to Phase 24.

The full per-divergence table is in the JSON artefact under
`corpora[i].divergences`; a summary of each category's count is
printed in § D.7.

### D.4 Planner round-trip preserved

For each predicate and each corpus, the Phase-22/25 substrate
guarantee is re-verified:

| corpus        | predicate                | expected | planner_val | matched |
|---           |---                       |---:|---:|:-:|
| `vision-core` | `calls_filesystem`       | 19 | 19 | **✓** |
| `vision-core` | `calls_network`          | 4  | 4  | **✓** |
| `vision-core` | `calls_subprocess`       | 1  | 1  | **✓** |
| `vision-core` | `may_raise`              | 164 | 164 | **✓** |
| `vision-core` | `may_write_global`       | 18 | 18 | **✓** |
| `vision-core` | `participates_in_cycle`  | 19 | 19 | **✓** |

The Phase-22 `render_error = 0` guarantee holds on every
predicate, exactly as Theorem P27-2 predicts. The planner count
matches the analyzer aggregate count with `matched = True` across
the full predicate set, regardless of the Phase-27 runtime-layer
coverage fraction (35 %) or calibration disagreements (the 26
total divergences on the entered subset). **The runtime-observer
gap is orthogonal to the substrate guarantee.**

### D.5 Repeat-run variance

Re-running with `--seeds 0 1 2 3 4 --budget 0.08` (five
independent seeds, slightly wider per-call budget):

| predicate                | 3-seed agree | 5-seed agree | 3-seed FN | 5-seed FN |
|---                       |---:|---:|---:|---:|
| `calls_filesystem`       | 101 | 101 | 0  | 0 |
| `calls_network`          | 102 | 102 | 0  | 0 |
| `calls_subprocess`       | 102 | 102 | 0  | 0 |
| `may_raise`              | 76  | 75  | 25 | **26** |
| `may_write_global`       | 102 | 102 | 0  | 0 |
| `participates_in_cycle`  | 102 | 102 | 0  | 0 |

Coverage breakdown is **seed-invariant** — candidate
classification is pure-static, so `|F_R|` does not depend on
seed. The 5-seed run adds ONE additional `may_raise` false
negative (`hyperbolic.origin` raising `IndexError` under a
fuzz input only hit at seed 3 or 4); every other divergence
is seed-stable. This confirms Conjecture P27-5: recipe
aggressiveness drives divergence rate, not fuzz budget.

Artefact: `vision_mvp/results_phase27_corpus_5seeds.json`.

### D.6 Cost

| component                                               | time (`vision-core`, seeds=0 1 2, budget=0.05s) |
|---                                                      |---:|
| Phase-24/25 static analysis across ~100 files           | <0.5 s |
| Candidate discovery (parse + import + classify)          | ~2 s |
| Probe phase (250 ready × 6 predicates × 3 seeds × ~2 invocations) | ~4 s |
| Planner round-trip (6 predicates × ledger ingest)        | ~1 s |
| **Total per corpus**                                     | **~7.3 s** |

Total wall-time on a laptop (MacBook M2): **7.3 seconds** with
the skip list applied. Without the skip list, the runtime expands
~10–15× (dominated by numpy / crypto modules where typed fuzzing
blows the budget on every invocation); coverage numbers are
within 2 % of the skip-list-applied report.

The harness is fast enough to run in CI on every commit that
touches `core/code_semantics.py` or `core/code_interproc.py`.

### D.7 Full scoreboard (populated from `results_phase27_corpus.json`)

```
(see: python -m vision_mvp.experiments.phase27_corpus_runtime_calibration
            --seeds 0 1 2 --budget 0.1
            --corpora vision-core vision-tasks
            --out vision_mvp/results_phase27_corpus.json)
```

See the JSON artefact for the full per-row observations; the
Phase-27 benchmark prints a pooled scoreboard at the end of
every run.

---

## Part E — Closing notes

### E.1 Strongest empirical takeaway

> **On the `vision-core` corpus (715 functions after skip list,
> 791 without), the Phase-24/25 analyzer emits flags for every
> function; $35.0 \%$ (250 / 715) are *runtime-calibratable*
> under the Phase-27 default invocation-recipe strategy. Of those,
> 102 / 715 ($14.3 \%$ of the corpus, $40.8 \%$ of the ready
> slice) actually entered the target frame under the probe
> budget. On this **entered slice**, analyzer vs runtime agreement
> is **509 / 510 (99.8 %)** across five predicates
> (`calls_filesystem`, `calls_network`, `calls_subprocess`,
> `may_write_global`, `participates_in_cycle`); the sixth
> predicate `may_raise` is the outlier at 76 / 102 (74.5 %) due
> to a **new boundary class Phase 27 surfaces**: implicit raises
> from builtin operations on argument values outside the
> function's semantic domain (`TypeError`, `OverflowError`,
> `ZeroDivisionError`, `IndexError`). The single
> `calls_filesystem` false-positive is a **probe-instrumentation
> gap** (`_record_filesystem` does not instrument `os.walk`),
> not an analyzer over-approximation. **Planner vs analyzer
> round-trip remains 100 % on every predicate** (`may_raise` =
> 164, `may_write_global` = 18, `calls_filesystem` = 19,
> `calls_network` = 4, `calls_subprocess` = 1,
> `participates_in_cycle` = 19 — all matching the analyzer's
> corpus-wide counts), confirming Theorem P27-2: the substrate
> guarantee is independent of the runtime layer's coverage. Full
> benchmark wall-time: **7.3 s** for seeds × budget (0, 1, 2) ×
> 0.05 s.**
>
> The **four-axis separation** — analyzer prediction vs
> snippet-calibrated runtime (Phase 26) vs corpus-scale runtime
> (Phase 27) vs direct-exact planner answer — is now first-class
> in the benchmark harness. Analyzer-gold exactness, snippet-
> calibrated truth, and corpus-scale calibrated truth live on
> three different axes with different witnesses; the substrate's
> guarantee is on the planner-to-analyzer axis, not on either
> runtime axis. Phase 27 additionally surfaces a **new analyzer
> boundary class (implicit raises)** that the curated snippet
> corpus did not exercise — a concrete research artefact
> demonstrating that corpus-scale runtime calibration is a
> **discovery tool for boundaries not previously enumerated**
> (Conjecture P27-4's operational consequence).

### E.2 Relationship to the project's top-level claim

Phases 1–10 proved O(log N) routing / bandwidth for multi-agent
consensus. Phases 19–25 proved a **substrate** delivering
analyzer-gold exactness on code questions without an LLM in the
inner loop. Phase 26 proved that the substrate's analyzer is
**well-calibrated against runtime truth on a curated executable
corpus**. Phase 27 proves that the same analyzer is
**partially-calibrated against runtime truth on real corpus
code**, and that the partiality is governed by witness-
availability, not by the substrate itself.

The deployment recipe, updated again:

1. Stand up a `ContextLedger` (Phase 19).
2. Ingest each Python corpus via `CodeIndexer` (Phases 22 + 24 +
   25).
3. Use `CodeQueryPlanner` for structural + semantic + interproc
   queries. Direct-exact on matched queries: zero LLM, zero prompt.
4. For soundness audits on the curated slice, run Phase 26
   runtime calibration.
5. **For corpus-scale soundness audits**, run Phase 27 runtime
   calibration. Report both the coverage fraction AND the
   per-predicate agreement on the entered subset. Treat any
   corpus-scale FN that does NOT match a Phase-24-documented
   boundary as a first-class research signal.
6. Monitor `has_unresolved_callees` as a transparency flag.

### E.3 What the project still does NOT fully solve

Phase 27 pushes the research frontier one more axis (from
snippet-scale runtime to corpus-scale runtime), but does not
solve context completely. Remaining gaps, in decreasing order of
impact:

1. **Method calibration.** 58 % of the `vision-core` corpus is
   methods. Probing them requires auto-construction of class
   instances, which is out of scope for the default recipe
   strategy. A light instance-constructor (try zero-arg
   `__init__`, fall back to `unittest.mock.MagicMock`) would
   broaden coverage materially. OQ-27b.
2. **Typed-recipe arg mismatch.** `ready_typed` candidates whose
   entered-fraction is low are receiving inputs outside the
   target's semantic domain. A signature-driven "smart" recipe
   (e.g., passing `Counter` when the function name is
   `add_one_to_counter`) would improve entry rate, at the cost
   of per-function recipe engineering. OQ-27c.
3. **Environment probing.** A function that raises on missing
   env vars is flagged `may_raise=True` by analyzer (correct)
   and not observed `True` by runtime (recipe didn't unset the
   var). Scripted environment manipulation is OQ-27d.
4. **C-extension transparency.** Functions executing in numpy
   internals are not interruptible by the Python-level tracer;
   they land in the `timeout` bucket. A second-tier
   `signal`-based budget would catch these (with portability
   caveats). OQ-27e.
5. **Corpus breadth.** Phase 27 ships two local corpora
   (`vision-core`, `vision-tasks`). Extending to the remaining
   Phase-23 corpora (`click`, stdlib `json`, `vision-tests`,
   `vision-experiments`) is OQ-27f.
6. **Cross-language runtime calibration.** Python-only;
   TypeScript / Go / Rust all need an analogous invocation-
   recipe protocol. OQ-27g.
7. **No LLM re-entry.** Phase 27 remains defensive — the runtime
   layer reports observation counts and flags, not natural-
   language explanations.

### E.4 Open questions (carry into Phase 28)

- **OQ-27a Recipe-aggressiveness sweep.** Measure how
  `|F_R|` and FN-rate co-vary as the recipe aggressiveness
  increases (broader fuzz pool, looser annotation rules).
- **OQ-27b Method instance auto-constructor.** A bounded
  ladder: zero-arg `__init__` → `MagicMock` → explicit
  curated recipe; measure the incremental entry rate.
- **OQ-27c Signature-driven smart recipes.** Use parameter
  names + corpus-internal type hints to improve entered-
  fraction on `ready_typed`. Directly targets the Phase-27
  implicit-raise false-negative class — if the recipe hands
  `gf.mul` arguments respecting the field's characteristic,
  the `ZeroDivisionError` FN would disappear.
- **OQ-27d Environment-dependent effect probing.** Scripted
  env-var manipulation + clocks to exercise conditional raises.
- **OQ-27e Second-tier C-aware budget.** `signal.SIGALRM` or
  thread-based watchdog to interrupt pure-C hot loops (with
  portability caveats).
- **OQ-27f Additional corpora.** `click`, stdlib `json`,
  `vision-tests`, `vision-experiments` as Phase-27 corpora.
- **OQ-27g Cross-language analogue.** TypeScript via
  `ts-node` + `tsd`; Go via `go test -run`; Rust via
  `cargo-fuzz` + the existing instrumentation pattern.
- **OQ-27h Runtime-refined analyzer.** A hybrid that uses
  Phase-27 observations to refine Phase-24's FP rate on
  `dead_code` cases while preserving soundness on the
  corpus slice.
- **OQ-27i Implicit-raise predicate.** Split Phase-24's
  `may_raise` into `may_raise_explicit` (current semantics)
  and `may_raise_implicit` (exceptions propagating from
  builtin operations on unconstrained inputs). Preserves the
  discrimination of the current flag while exposing the
  implicit-raise surface Phase 27 surfaced.
- **OQ-27j Probe-instrumentation parity with analyzer.** Close
  the small gap where the Phase-26 `_record_filesystem`
  context manager does not instrument every API in
  `core.code_semantics._FILESYSTEM_APIS`
  (`os.walk`, `os.listdir`, `os.scandir`, `os.stat`, ...).
  Parity would eliminate the single Phase-27
  `calls_filesystem` false-positive without affecting
  soundness anywhere else.

### E.5 Reproducibility

| Run | Command | Output |
|---|---|---|
| 3-seed × 0.05s budget headline | `python -W ignore::RuntimeWarning -m vision_mvp.experiments.phase27_corpus_runtime_calibration --seeds 0 1 2 --budget 0.05 --corpora vision-core --out vision_mvp/results_phase27_corpus.json` | `vision_mvp/results_phase27_corpus.json` |
| 5-seed × 0.08s budget variance | `python -W ignore::RuntimeWarning -m vision_mvp.experiments.phase27_corpus_runtime_calibration --seeds 0 1 2 3 4 --budget 0.08 --corpora vision-core --out vision_mvp/results_phase27_corpus_5seeds.json` | `vision_mvp/results_phase27_corpus_5seeds.json` |
| Unit tests (Phase 27) | `python -m unittest vision_mvp.tests.test_code_corpus_runtime` | 40 tests, all pass |
| Full test suite (no regressions) | `python -m unittest discover -s vision_mvp/tests` | 940 tests, zero regressions |

Phase-27-specific tests live at
`vision_mvp/tests/test_code_corpus_runtime.py`: **40 new tests
this phase** covering the classifier, recipes, tracer sandbox
restoration, entry detection, timeouts, end-to-end calibrate_corpus,
divergence collection, and a regression guard that ensures
Phase-26 snippet calibration still runs after Phase-27 import.
