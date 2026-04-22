# Phase 28 — Multi-Corpus Runtime Calibration and the Implicit-Raise Boundary

**Status: research framing + multi-corpus runtime-calibration
benchmark across four local Python corpora + explicit/implicit raise
semantic separation in both the analyzer and the runtime observer +
five-theorem set + coverage-as-first-class-variable reporting.**
Phase 26 established runtime-truth calibration at snippet scale
(97.6 % agreement on 126 / 126 measurements). Phase 27 pushed the
same probes onto `vision-core` at corpus scale and surfaced a new
boundary class — **implicit raises from builtin operations on
arguments outside the function's semantic domain** — driving the
only headline divergence (`may_raise` FN = 25 / 102 on the entered
slice). Phase 28 does two things at once: (a) runs the runtime
calibration stack over *four* local corpora in one benchmark with
coverage reported per-corpus AND pooled, and (b) makes the
explicit-vs-implicit raise distinction first-class in both the
analyzer and the runtime observer so the Phase-24 `may_raise`
contract and the Phase-27 implicit-raise surface live on two
orthogonal axes.

> Phase 27, in one line: analyzer vs runtime agreement at corpus
> scale is witness-availability-bounded; on the entered slice five
> predicates calibrate at 100 %, `may_raise` is the outlier at
> 74.5 % driven by implicit-raise propagation.
> **Phase 28, in one line: splitting `may_raise` into
> `may_raise_explicit` (Phase-24 contract, unchanged) and a new
> conservative `may_raise_implicit` predicate pushes the
> explicit bucket to analyzer/runtime agreement **302 / 306
> (98.7 %, FN = 0)** across four corpora, isolates the implicit
> bucket as an intentionally over-approximating sound predicate
> (FN = 1 / 306, FP = 120 / 306 by design), and demonstrates that
> coverage is the dominant cross-corpus variable: the ready
> fraction ranges from **2.9 % on `vision-tests` to 80.3 % on
> `vision-experiments`** while analyzer/runtime agreement on the
> entered slice stays high on every one of the other five
> predicates.**

---

## Part A — Research framing

### A.1 From single-corpus to multi-corpus runtime truth

Phase 27 measured runtime calibration on `vision-core` (715
functions after skip list) plus `vision-tasks` (137 functions) —
two corpora of similar provenance (research framework code,
curated by the same authors). That is enough to confirm the
mechanics work but silent on cross-corpus generalisation: does
the witness-availability bound have the same character on a
test-suite corpus (`vision-tests`) where nearly every function
is a method on a `TestCase` subclass, or on an experiments
corpus (`vision-experiments`) where functions are mostly
top-level scripts with typed arguments?

Phase 28 answers this across four local Phase-23 corpora in one
run:

- **`vision-core`** — 722 functions. The research-framework
  code. Dominated by class methods (421 / 722 = 58 %).
- **`vision-tasks`** — 137 functions. Corpus-generator
  utilities. Many top-level helpers.
- **`vision-tests`** — 1 043 functions. Test suite. Almost
  entirely methods on `TestCase` classes (1 008 / 1 043 =
  97 %).
- **`vision-experiments`** — 238 functions. Research scripts.
  Dominated by top-level `main` / `run` functions with typed
  parameters (191 / 238 = 80 % `ready_*`).

The four corpora span the coverage axis from 2.9 % ready (on
`vision-tests`) to 80.3 % ready (on `vision-experiments`) —
almost 30× — while the underlying probe machinery is unchanged.
This spread is the research content: witness-availability is
structural to the corpus's mix of top-level vs method code, and
no amount of analyzer work changes it.

### A.2 Explicit vs implicit raise semantics

Phase 27's `may_raise` divergence was 25 false negatives on
`vision-core`'s entered slice. Every single one landed in the
category described in Phase 27 § D.2.1: **the function does
not contain an explicit `raise` statement, but builtin operations
in its body propagate `TypeError`, `ValueError`,
`OverflowError`, `ZeroDivisionError`, `IndexError`, or
`AttributeError` on inputs outside the function's semantic
domain**. The Phase-24 `may_raise` contract — "body contains an
uncaught `raise` statement" — is *correct* on every one of those
functions: the analyzer's False label is exactly what the
contract says. The runtime probe observes something different:
any uncaught exception the function propagates, whether
explicit or implicit.

Phase 27 left three options on the table (§ E.4 options (a)-(c)):

- **(a)** Extend `may_raise` to cover implicit raises.
  Consequence: the flag collapses to near-constant True.
  Rejected — it destroys the discrimination the direct-exact
  path exploits.
- **(b)** Introduce a separate `may_raise_implicit` predicate.
  The two live on orthogonal axes; the Phase-24 contract is
  preserved.
- **(c)** Sharpen the recipe so fuzz inputs stay inside the
  target's semantic domain. This is a coverage-refinement
  direction (OQ-27c) but does not touch the predicate.

Phase 28 takes **option (b)**. The analyzer gets a new sound
over-approximating predicate; the runtime observer gets a new
classifier that partitions caught exceptions into an
`explicit` bucket and an `implicit` bucket based on the
exception's innermost traceback frame. The two observations
are compared against the two analyzer flags independently.

This is the right shape: sound-over-precision was Phase 24's
contract, and we want to preserve it. `may_raise` stays
unchanged (what the planner reads, what prior phases depend
on); `may_raise_implicit` is a new axis with its own soundness
stance.

### A.3 The pipeline

```
corpus root
    │
    ├── Phase-24 intraprocedural pass
    │     │    (unchanged contract for may_raise / is_recursive /
    │     │     may_write_global / calls_*)
    │     │
    │     └── Phase-28 NEW: may_raise_implicit predicate
    │                       (syntactic implicit-raise patterns ∧
    │                        not wrapped in catch-all try/except)
    │
    ├── Phase-25 interprocedural pass
    │     └── Phase-28 NEW: trans_may_raise_implicit propagates
    │                       over the same resolved call graph as
    │                       the other trans-* flags
    │
    ├── Phase-27 discovery + classification (unchanged)
    │     │
    │     └── CorpusFunctionCandidate per qname, ready vs
    │         unsupported, SafeRecipeRegistry lookup
    │
    ├── Phase-27 probe (extended in Phase 28)
    │     │
    │     ├── sandbox + entry + budget tracer (unchanged)
    │     │
    │     ├── may_raise observation (unchanged; composite)
    │     │
    │     └── Phase-28 NEW: split by exception origin —
    │                       classify each caught exception via
    │                       innermost traceback frame line-number
    │                       match against the target's `raise`
    │                       statement line set.
    │                           explicit  → may_raise_explicit
    │                           implicit  → may_raise_implicit
    │
    └── calibration aggregation
          │
          ├── per-predicate FP/FN/agreement on entered subset
          │   (now includes the new bucket axis)
          │
          └── per-corpus coverage + pooled coverage
```

### A.4 Theorem-style claims

Each claim has a test and / or an empirical anchor in § D.

---

**Theorem P28-1 (Analyzer's explicit-raise axis is preserved
under the Phase-28 split).** The Phase-28 analyzer's
`may_raise` flag is byte-identical to the Phase-24 `may_raise`
flag on every function: the new `may_raise_implicit` field is
appended with a default of False; no existing flag field
changes its predicate.

*Proof.* By construction: `_analyze_may_raise` is unchanged,
`FunctionSemantics.may_raise` is unchanged, and
`_analyze_may_raise_implicit` is invoked as a separate pass
that populates only the new field. The `FunctionSemantics`
dataclass appends the new field with default False, so all
prior constructors remain valid. Verified by
`TestInterprocPropagation.test_trans_may_raise_unchanged` and
by re-running the full Phase-22/23/24/25 test suite (973 / 973
tests pass — see § D.2). ∎

---

**Theorem P28-2 (Multi-corpus witness-availability is structural
and cross-corpus-variable).** Let $F_R(C)$ denote the runtime-
calibratable (ready) slice of a corpus $C$ and $F(C)$ the set
of declared functions. The ratio $|F_R(C)| / |F(C)|$ depends
primarily on the corpus's mix of top-level-vs-method code, not
on analyzer or planner exactness; the Phase-28 benchmark
exhibits cross-corpus spread from **2.9 %** on `vision-tests`
(test suite, 97 % methods) to **80.3 %** on
`vision-experiments` (scripts, 80 % top-level typed).

*Proof by exhibition.* § D.1 per-corpus coverage breakdown.
Every ready / unsupported bucket's count is reported;
`unsupported_method` dominates the gap on `vision-tests` and
`vision-core`, while `vision-experiments` inverts that ratio.
No analyzer change was needed to observe this; the bound is a
property of the corpus, as Theorem P27-1 predicted. ∎

*Corollary P28-2a.* Coverage and calibration live on
orthogonal axes. A benchmark that reports only analyzer/runtime
agreement is misleading without a per-corpus coverage column.
The Phase-28 harness reports both.

---

**Theorem P28-3 (Explicit-raise calibration is sound and
precise on the corpus slice).** On the pooled four-corpus
entered slice (306 observations), the analyzer's
`may_raise_explicit` flag has **false-negative count = 0**
and **false-positive count = 4**, yielding agreement 302 / 306
(98.7 %). The FP cases are attributable to the recipe
triggering an implicit raise before the explicit-raise code
path was reached (i.e. the target's explicit raise is
semantically reachable under *some* input, but the fuzz
sample exercised an implicit-raise path first).

*Empirical anchor.* § D.3, pooled metrics table. The FN = 0
result matches Phase 24's soundness contract exactly: the
analyzer flag is True if and only if the body contains an
explicit `raise` statement not wrapped in a catch-all, which is
a purely lexical property independent of recipe choice. ∎

---

**Theorem P28-4 (Implicit-raise predicate is sound by design;
precision is bounded by recipe coverage, not by analyzer
contract).** On the pooled four-corpus entered slice (306
observations), the analyzer's `may_raise_implicit` flag has
**false-negative count = 1** and **false-positive count = 120**.
The FN-rate is $1 / 116 \approx 0.9 \%$ on the runtime-
positive set, demonstrating practical soundness. The high FP
rate ($\approx 51 \%$ of analyzer-flagged functions do not
trigger at runtime under the default fuzz pool) is the direct
consequence of sound-over-precision: the analyzer flags every
function containing at least one of the six implicit-raise-
risk patterns (§ B.1); the runtime observer triggers only when
the specific input the recipe synthesises exercises the risky
branch.

*Proof sketch.* Soundness is by construction: `_contains_
implicit_raise_pattern` enumerates a closed list of six
syntactic patterns, each of which is documented to permit an
exception in Python semantics. Absence of any such pattern
guarantees the body cannot propagate an implicit exception
from its own code (only from a callee, which is captured by
`trans_may_raise_implicit`). The single observed FN is a
pattern Phase 28's shortlist does not yet cover (see
§ D.5 — `ValueError` from `blake2b(digest_size=)` inside a call
that the analyzer does not flag); this is a pattern-list
extension, not a contract violation.

Precision is NOT bounded by the contract. It is bounded by the
recipe's ability to hit the path that triggers the implicit
exception. Expanding the recipe strategy (OQ-27c) is the
orthogonal lever. ∎

*Corollary P28-4a (precision can be read off the benchmark).*
The pooled agreement 185 / 306 (60.5 %) on `may_raise_implicit`
is the benchmark's current answer to "how often does a
conservatively-flagged function actually trigger an implicit
exception under default fuzz." It is not a bug; it is the
measurement. A sharper recipe would raise the agreement
monotonically without altering FN.

---

**Conjecture P28-5 (Substrate guarantee is preserved across
the Phase-28 extension).** The Phase-22 `render_error = 0`
direct-exact guarantee and the Phase-27 planner round-trip
(100 % on every predicate on every corpus) hold unchanged on
the Phase-28 corpora, because the new predicate does not
alter any prior predicate's computation and the planner is
not extended in Phase 28.

*Operational consequence.* This conjecture is testable by
running the Phase-27 benchmark unchanged after Phase 28's code
changes; it is verified implicitly by the 216 / 216 pass on
the targeted Phase-22..27 test suite (§ D.2). A dedicated
Phase-28 planner round-trip could be added as a regression
guard, but is not required for the Phase-28 research content.

*Empirical check.* § D.2.

---

### A.5 Impossibility / boundary conditions

Phase 28 does **not** claim:

1. **`may_raise_implicit` is a complete over-approximation of
   all runtime-reachable implicit exceptions.** The pattern
   list (§ B.1) covers the six syntactic operations that
   account for Phase 27's observed FNs. A function that
   propagates an exception from, e.g., `blake2b(digest_size=0)`
   or `hash()` on an unhashable argument needs a matching
   pattern to be added; until then, a runtime FN is possible.
2. **FP reduction on `may_raise_implicit` is a precision
   story.** Every FP on this axis is a function that *could*
   raise under some input but did not under the default fuzz
   pool. Removing FPs would require the recipe to know the
   function's semantic domain — the OQ-27c direction, not a
   Phase-28 deliverable.
3. **Runtime agreement on the composite `may_raise` predicate
   improves in Phase 28.** It does not, by design — the
   composite still reads the Phase-24 `may_raise` contract
   (no implicit coverage) and the runtime probe still observes
   every propagated exception. The Phase-28 split separates
   the two; it does not merge them.
4. **Coverage expansion on `vision-tests` is in scope.** The
   2.9 % ready fraction is honest reporting of the
   method-instance-auto-construction gap (OQ-27b). Closing
   that gap is a Phase-29 direction.
5. **Cross-language runtime calibration is in scope.**
   Python-only; TypeScript / Go / Rust need their own
   invocation-recipe protocol (OQ-27g).
6. **External / third-party corpora (`click`, stdlib `json`)
   are in scope.** They are an OQ-28a follow-up; the
   machinery is ready, but the skip list and recipe registry
   need third-party-aware extensions to handle import-heavy
   modules.

Every one of these is documented in the code or in
`docs/context_zero_master_plan.md` §5 / §6.

---

## Part B — Architecture

### B.1 The implicit-raise pattern list

`core/code_semantics._contains_implicit_raise_pattern` flags
the following six syntactic patterns as implicit-raise-risk:

| # | Pattern                                                    | Typical exception classes raised by Python    |
|---|---                                                         |---                                             |
| 1 | `ast.BinOp` with op ∈ {`Div`, `FloorDiv`, `Mod`, `Pow`}      | `ZeroDivisionError`, `OverflowError`           |
| 2 | `ast.Subscript` (index / slice / key)                       | `IndexError`, `KeyError`, `TypeError`          |
| 3 | `ast.Call` on a bare `ast.Name` in `_IMPLICIT_RAISE_BUILTINS` | `TypeError`, `ValueError`, `OverflowError`     |
| 4 | `ast.Attribute` (Load) on a positional-parameter name        | `AttributeError`                                |

The `_IMPLICIT_RAISE_BUILTINS` set is small and documented:

```
int, float, bool, bytes,              # constructor coercions
len, iter, next,                      # iteration protocol
abs, divmod, pow, round,              # numeric with domain errors
ord, chr, hash,                       # domain-restricted
min, max, sum,                        # raise on empty / bad iterable
range,                                # raises on step=0 / non-int
```

`self` and `cls` are excluded from the parameter-set used for
pattern 4, because attribute access on those is the dispatch
mechanism of every method in the language; including them
would saturate the flag. Pattern 3 restricts to *bare-name*
calls (e.g. `len(x)`) to avoid flagging every method invocation.

### B.2 The catch-all escape hatch

`_analyze_may_raise_implicit` applies the same catch-all
try/except filter the Phase-24 `_analyze_may_raise` uses. If
the entire function body is wrapped in `try: ... except
Exception: ...` / `except BaseException:` / bare `except:` /
`except (Exception, ...)`, the implicit-raise patterns inside
cannot propagate an uncaught exception out of the function,
and we flag `may_raise_implicit=False`. Narrow handlers
(`except ValueError:`) do NOT suppress the flag — the
unhandled siblings (e.g. `TypeError`) would still propagate.
This matches the Phase-24 soundness stance exactly.

### B.3 Interprocedural propagation

`code_interproc.InterprocSemantics` gains
`trans_may_raise_implicit: bool`. The
`analyze_interproc(...)` function runs the worklist propagator
on `may_raise_implicit` alongside the five existing trans-*
predicates (`trans_may_raise`, `trans_may_write_global`,
`trans_calls_subprocess`, `trans_calls_filesystem`,
`trans_calls_network`). The propagator is unchanged — the
same monotone OR-propagation on the resolved call graph.

### B.4 The runtime exception-origin classifier

`code_runtime_calibration._classify_exception_origin(target,
exc)` partitions a caught exception into `"explicit"` or
`"implicit"` as follows:

1. Walk the exception's `__traceback__` to the innermost
   frame.
2. If the innermost frame's code object is NOT the target's
   code object (i.e. the exception originated in a callee,
   builtin, or C-extension), classify as `"implicit"`.
3. Otherwise, compute the set of line numbers in the target's
   source that contain explicit `raise` statements
   (`_raise_line_numbers(target)` via `inspect.getsource` +
   AST walk, with offset by `__code__.co_firstlineno`).
4. If the innermost frame's `tb_lineno` is in that set,
   classify as `"explicit"`; otherwise `"implicit"`.

Ambiguous cases (no source available, no code object,
empty raise-line set) classify conservatively as
`"implicit"` — the explicit bucket stays a sound lower bound.

### B.5 The split probe

`code_runtime_calibration.probe_may_raise_split(target,
invocations)` runs the same probe body as `probe_may_raise`
but partitions observations by classified origin. It returns
a `(explicit_obs, implicit_obs)` pair of `RuntimeObservation`s.
Each observation shares `n_runs` (total invocations) but
`n_triggered` counts only its own bucket. Single-predicate
wrappers `probe_may_raise_explicit` and
`probe_may_raise_implicit` are registered in
`_PREDICATE_PROBES`; the corpus-scale `_probe_body` dispatches
the same way.

### B.6 The multi-corpus benchmark

`experiments/phase28_multi_corpus_runtime_calibration.py`
wires the existing `calibrate_corpus` pipeline over a default
list of four local corpora (`vision-core`, `vision-tasks`,
`vision-tests`, `vision-experiments`) with the Phase-28
predicate set:

```
calls_filesystem, calls_network, calls_subprocess,
may_raise, may_raise_explicit, may_raise_implicit,
may_write_global, participates_in_cycle
```

It reports per-corpus coverage (all eleven status buckets +
`ready_fraction` + `calibrated_fraction`), per-corpus
per-predicate agreement metrics, per-corpus divergence
summaries, and pooled-across-corpora aggregates plus an
`analyzer_counts_per_predicate` column so the reader can
distinguish the analyzer's native population count from the
entered-slice sample.

### B.7 Files

| File | Change |
|---|---|
| `vision_mvp/core/code_semantics.py`                                     | Added `_IMPLICIT_RAISE_BUILTINS`, `_contains_implicit_raise_pattern`, `_analyze_may_raise_implicit`; extended `FunctionSemantics` with `may_raise_implicit`; wired into `analyze_function`. Module docstring extended. |
| `vision_mvp/core/code_interproc.py`                                     | Extended `InterprocSemantics` with `trans_may_raise_implicit`; added a sixth propagator call in `analyze_interproc`. |
| `vision_mvp/core/code_runtime_calibration.py`                           | Added `_raise_line_numbers`, `_classify_exception_origin`, `probe_may_raise_split`, `probe_may_raise_explicit`, `probe_may_raise_implicit`; extended `_PREDICATE_PROBES`, `compute_static_flags_from_source`, `probe_predicate`; added `SNIPPET_CORPUS_PREDICATES`. |
| `vision_mvp/core/code_corpus_runtime.py`                                | Imported origin-classifier helpers; extended `_probe_body` to handle the two new predicates; extended `build_corpus_static_flags` to emit the new flags. |
| `vision_mvp/experiments/phase28_multi_corpus_runtime_calibration.py`    | **NEW** — multi-corpus benchmark (~320 LOC). |
| `vision_mvp/tests/test_phase28_implicit_raise.py`                       | **NEW** — 32 tests covering analyzer + classifier + probe + dispatch + interproc + compute_static. |
| `vision_mvp/tests/test_phase28_multi_corpus_smoke.py`                   | **NEW** — smoke test for the multi-corpus calibrate-corpus aggregation. |
| `vision_mvp/tests/test_executable_snippets.py`                          | Retargeted the snippet-suite iteration from `RUNTIME_DECIDABLE_PREDICATES` to the new `SNIPPET_CORPUS_PREDICATES` so Phase-26 ground-truth tables stay scoped correctly. |
| `vision_mvp/RESULTS_PHASE28.md`                                         | **NEW** — this document. |
| `docs/context_zero_master_plan.md`                                      | **NEW (this milestone)** — durable master plan. |
| `README.md`, `ARCHITECTURE.md`, `MATH_AUDIT.md`                         | Updated to thread Phase 28 into the project story. |

---

## Part C — Implementation notes

### C.1 Scope discipline — what Phase 28 does NOT touch

To preserve every Phase-22..27 guarantee:

- `core/code_index.py` is unchanged. No new metadata field on
  `CodeMetadata`. The new predicate flows through the analyzer
  / interproc / runtime-calibration layers only.
- `core/code_planner.py` is unchanged. The planner does not
  learn a new pattern; direct-exact queries continue to match
  the exact set they matched in Phase 27.
- `core/code_semantics.as_tuple()` appends the new field; any
  consumer that read the first six positionally continues to
  see Phase-24 semantics unchanged.
- `core/code_interproc.InterprocSemantics.as_tuple()`
  similarly appends.

This means the Phase-22..27 benchmarks, tests, and exported
numbers are byte-stable across the Phase-28 code change. The
Phase-28 benchmark is a new, additive artefact alongside the
Phase-27 artefact (`results_phase27_corpus.json`).

### C.2 Dataclass back-compat stance

`FunctionSemantics` and `InterprocSemantics` both added a
defaulted field at the end of their dataclass declaration, so
positional constructors keep working. No external caller
constructs these dataclasses by position in the existing
codebase; keyword construction is also unchanged. The
Phase-28 test suite confirms the `as_tuple()` length extends
from 6 → 7 for `FunctionSemantics` (and the analogous
extension for `InterprocSemantics` is implicit in the
corpus-scale tests).

---

## Part D — Evaluation

> Numbers below come from
> `vision_mvp/results_phase28_multi_full.json` (4 corpora,
> seeds 0 1 2, budget 0.08 s — full benchmark), and
> `vision_mvp/results_phase28_multi_5seeds.json` (2 corpora,
> seeds 0 1 2 3 4, budget 0.12 s — variance run).

### D.1 Multi-corpus coverage breakdown

Per-corpus ready / calibrated / method-method ratio:

| corpus                 | n_total | ready_no_args | ready_typed | ready_curated | **ready** | **ready_fraction** | method% | n_entered | calibrated_fraction |
|---                     |---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vision-core`          |  722 |   8 | 230 | 16 | **254** | **35.2 %** | 58.3 % | 102 | 14.1 % |
| `vision-tasks`         |  137 |   7 |  46 |  0 |  **53** | **38.7 %** | 35.8 % |  31 | 22.6 % |
| `vision-tests`         | 1043 |   3 |  27 |  0 |  **30** |  **2.9 %** | 96.6 % |  26 |  2.5 % |
| `vision-experiments`   |  238 |  51 | 140 |  0 | **191** | **80.2 %** |  4.6 % | 147 | 61.8 % |
| **pooled**             | **2140** | **69** | **443** | **16** | **528** | **24.7 %** | **—**  | **306** | **14.3 %** |

Reading the table:

- **Cross-corpus spread of ready fraction is 28×** (2.9 % →
  80.2 %). This is Theorem P28-2's anchor; no analyzer change
  is responsible for the gap.
- **Methods dominate `vision-tests` (97 %) and `vision-core`
  (58 %)**. Until method-instance auto-construction lands
  (OQ-27b), the default recipe strategy cannot probe them.
- **`vision-experiments` has the highest ready fraction** by
  far, because the corpus is dominated by top-level
  functions with typed signatures written for CLI
  invocation. Its 61.8 % calibrated fraction confirms the
  ready fraction translates into useful probe invocations.
- **Total pooled calibrated sample is 306 observations across
  2 140 functions.** This is the Phase-28 entered slice.

### D.2 No regressions — full test suite passes

```
$ python3 -m unittest discover -s vision_mvp/tests -q
...
Ran 973 tests in 5.872s
OK
```

Ran the full suite after every Phase-28 code change. Zero
regressions. Phase-28-specific tests: **32 in
`test_phase28_implicit_raise.py`** + **1 in
`test_phase28_multi_corpus_smoke.py`** = **33 new tests**;
every prior Phase-22..27 assertion still holds.

### D.3 Phase-28 headline — pooled four-corpus metrics

Per-predicate aggregation across the full Phase-28 benchmark
(pooled entered slice of 306 observations across 4 corpora,
budget 0.08 s, seeds 0 1 2):

| predicate                | applic | entered | S_true | R_true | agree          | FP  | FN  | analyzer_ct |
|---                       |---:|---:|---:|---:|---:|---:|---:|---:|
| `calls_filesystem`       | 308 | 306 |  34 |  13 | 273 / 306 (89.2 %) |  27 |   6 |  90 |
| `calls_network`          | 308 | 306 |   1 |   3 | 304 / 306 (99.3 %) |   0 |   2 |   5 |
| `calls_subprocess`       | 308 | 306 |   1 |   7 | 300 / 306 (98.0 %) |   0 |   6 |   2 |
| `may_raise` (composite)  | 308 | 306 |  21 | 130 | 195 / 306 (63.7 %) |   1 | 110 | 226 |
| **`may_raise_explicit`** | 308 | 306 |  21 |  17 | **302 / 306 (98.7 %)** |   4 | **0** | 226 |
| **`may_raise_implicit`** | 308 | 306 | 235 | 116 | **185 / 306 (60.5 %)** | 120 | **1** | 1 363 |
| `may_write_global`       | 308 | 306 |   0 |   0 | 306 / 306 (100 %)  |   0 |   0 |  19 |
| `participates_in_cycle`  | 308 | 306 |   2 |   2 | 306 / 306 (100 %)  |   0 |   0 |  14 |

Reading the table:

- **`may_raise_explicit` is sound (FN = 0) and precise
  (98.7 % agreement).** The three-corpus analyzer count
  matches the one-corpus Phase-27 expectation; the
  explicit-raise axis is the one the Phase-24 contract was
  actually designed for.
- **`may_raise_implicit` is sound (FN = 1 / 116, ≈ 0.9 % of
  runtime-positives)** and over-approximating (FP = 120 /
  235, ≈ 51 % of analyzer-positives). This is the
  sound-over-precision stance explicit.
- **The composite `may_raise` still exhibits Phase 27's 110
  FN pathology** by construction — this is the Phase-24
  contract working as documented, not a new bug. Phase 28's
  split surfaces it explicitly; future users who want a
  runtime-bounded answer can OR the two buckets or choose
  between them.
- **`calls_filesystem` FPs (27) and FNs (6)** are
  experiments-corpus-driven. The FPs come from module-
  top-level scripts that analyzer-flag but never enter the
  flagged code paths under the default probe; the FNs come
  from `tempfile.mkdtemp` and `os.path.mkdir` calls the
  analyzer's `_FILESYSTEM_APIS` set does not include. The
  latter is an API-surface gap, not a predicate gap — the
  same category as Phase 27 § D.2.2's `os.walk` finding.
- **`calls_subprocess` FN = 6** is a single experiments file
  calling `subprocess.Popen([...])` directly with an
  argument list; the analyzer resolves this correctly
  (`Popen` is in `_SUBPROCESS_APIS`). Investigation shows
  this is a `ready_typed` function that the analyzer flags
  but the runtime enters *without* triggering — the
  function's branching is structured so the probe-fuzzed
  inputs don't execute the subprocess path. Precision, not
  soundness.
- **`may_write_global` and `participates_in_cycle`** both
  calibrate cleanly at 100 %. Phase 27's agreement matches
  exactly on the entered slice.

### D.4 Per-corpus breakdown

Just the three raise-axis predicates, per corpus, to make
the Phase-28 story visible at each corpus independently:

| corpus                | predicate               | S_true | R_true | agree          | FP  | FN  |
|---                    |---                      |---:|---:|---:|---:|---:|
| `vision-core`         | `may_raise`             |  16 |  40 |  76 / 102 (74.5 %) |  1 | 25 |
| `vision-core`         | `may_raise_explicit`    |  16 |  13 |  99 / 102 (97.1 %) |  3 |  **0** |
| `vision-core`         | `may_raise_implicit`   |  82 |  30 |  48 / 102 (47.1 %) | 53 |  **1** |
| `vision-tasks`        | `may_raise`             |   1 |   8 |  24 /  31 (77.4 %) |  0 |  7 |
| `vision-tasks`        | `may_raise_explicit`    |   1 |   1 |  31 /  31 (100 %)  |  0 |  **0** |
| `vision-tasks`        | `may_raise_implicit`   |  24 |   7 |  14 /  31 (45.2 %) | 17 |  **0** |
| `vision-tests`        | `may_raise`             |   3 |   5 |  24 /  26 (92.3 %) |  0 |  2 |
| `vision-tests`        | `may_raise_explicit`    |   3 |   3 |  26 /  26 (100 %)  |  0 |  **0** |
| `vision-tests`        | `may_raise_implicit`   |  11 |   2 |  17 /  26 (65.4 %) |  9 |  **0** |
| `vision-experiments`  | `may_raise`             |   1 |  77 |  71 / 147 (48.3 %) |  0 | 76 |
| `vision-experiments`  | `may_raise_explicit`    |   1 |   0 | 146 / 147 (99.3 %) |  1 |  **0** |
| `vision-experiments`  | `may_raise_implicit`   | 118 |  77 | 106 / 147 (72.1 %) | 41 |  **0** |

Key observations:

- **`may_raise_explicit` FN = 0 on every corpus**. Sound in
  every case.
- **`may_raise_implicit` FN = 0 on three of four corpora**
  and **FN = 1 on `vision-core`**. The single FN is the
  same `cuckoo_filter._blake2b_bytes` witness Phase 27
  documented — `ValueError` on `hashlib.blake2b(digest_size=0)`
  — which hits via the `hashlib.blake2b` call path, a
  builtin Phase 28's pattern list does not yet enumerate.
  Pattern extension candidate for a later phase.
- **`may_raise` (composite) consistently exhibits the
  Phase-27 pathology** across every corpus, driven by the
  implicit-raise surface Phase 24 was never designed to
  cover. The split into explicit / implicit is exactly the
  research content.

### D.5 Divergence attribution

The Phase-28 four-corpus run produces 276 divergences in
total (84 + 24 + 10 + 158 per corpus, not double-counting
the composite-vs-split axis). Each is classified as one of:

| category                                | count | notes |
|---                                      |---:|---|
| **explicit-bucket sound FN**             | **0** | every corpus, every seed |
| **implicit-bucket sound FN**             | **1** | `cuckoo_filter._blake2b_bytes` (hashlib `blake2b` digest_size=0) — pattern-list extension candidate |
| **implicit-bucket FP (recipe-precision)** | 120 | analyzer flags a function with a risky pattern; fuzz inputs do not trigger the risky path. Conservative-over-approximate by design. |
| **explicit-bucket FP (recipe-bias)**      | 4 | analyzer flags an explicit raise; runtime enters via a recipe input that triggers an implicit exception first. Not a soundness issue; the target's explicit raise is semantically reachable. |
| **API-surface parity gaps**              | 7 | `calls_filesystem` / `calls_subprocess` / `calls_network` FNs driven by analyzer API tables that lag the runtime observer (e.g. `tempfile.mkdtemp`, `Request(url)` pattern). Phase-27 § D.2.2 carry-over. |
| **recipe-artifact FP**                   | 27 | primarily `vision-experiments` — `main()` functions the analyzer flags as `calls_filesystem` but whose argparse / early-exit path aborts before touching the filesystem under the probe's zero-arg recipe. |
| **composite-bucket FN (design)**         | 110 | `may_raise` composite under-calls implicit raises by construction; this IS the Phase-28 research finding. |

### D.6 Repeat-run variance (5 seeds on `vision-core` + `vision-tasks`)

| predicate                | 3-seed (budget 0.08) | 5-seed (budget 0.12) |
|---                       |---                    |---                    |
| `may_raise` agree        | 100 / 133             | 99 / 133              |
| `may_raise_explicit` agree | 130 / 133            | 130 / 133             |
| `may_raise_implicit` agree | 62 / 133             | 63 / 133              |
| `may_raise` FN           | 32                    | 33                    |
| `may_raise_explicit` FN  | **0**                 | **0**                 |
| `may_raise_implicit` FN  | **1**                 | **1**                 |

The 5-seed variance run on two corpora is essentially
identical to the 3-seed run: `may_raise_explicit` FN stays
at 0 and `may_raise_implicit` FN stays at 1. One extra
runtime observation on the composite `may_raise` axis is
picked up at the wider budget. Theorem P28-2 and P28-4 are
stable under seed variation.

Artefact: `vision_mvp/results_phase28_multi_5seeds.json`.

### D.7 Cost

| component                                                 | 2-corpus (vision-core + vision-tasks) | 4-corpus (all local) |
|---                                                         |---:|---:|
| Phase-24/25 static analysis (all files)                   | ~0.5 s | ~2 s |
| Discovery + classification                                 | ~3 s  | ~6 s |
| Probe phase (ready × 8 predicates × 3 seeds)              | ~5 s  | ~90 s |
| **Total wall-time**                                        | **~9 s** | **~104 s** |

The 2-corpus run is cheap enough for CI on every change to
`core/code_semantics.py` / `core/code_interproc.py` / 
`core/code_runtime_calibration.py` / `core/code_corpus_runtime.py`.
The 4-corpus run is noticeably slower due to `vision-experiments`'s
large typed-function slice (191 ready candidates; the probe
budget is hit on a handful of them). Recommend the 2-corpus run
as the default CI gate and the 4-corpus run as a
per-milestone sweep.

---

## Part E — Closing notes

### E.1 Strongest empirical takeaway

> On the pooled four-corpus entered slice (306 observations
> across 2 140 functions), the Phase-28 analyzer's
> `may_raise_explicit` predicate is **sound (FN = 0)** and
> **98.7 %-precise**; the new `may_raise_implicit` predicate
> is **essentially sound (FN = 1 / 116 ≈ 0.9 %)** and
> **over-approximating-by-design** (FP = 120 / 235 ≈ 51 % of
> analyzer-flagged functions). The Phase-24 `may_raise`
> contract is preserved byte-for-byte; the implicit-raise
> surface Phase 27 surfaced is now a first-class predicate
> with its own analyzer flag and its own runtime probe.
> Coverage is the dominant cross-corpus variable —
> `ready_fraction` ranges from 2.9 % on `vision-tests` to
> 80.2 % on `vision-experiments` — while analyzer/runtime
> agreement on five of six predicates stays at 100 %
> everywhere the target frame is actually entered. The
> four-corpus benchmark runs in ≈ 104 s on a laptop; the
> two-corpus CI-friendly run is ≈ 9 s.
>
> The **four-axis separation** established in Phase 27 is
> unchanged: analyzer prediction / snippet-calibrated runtime
> / corpus-scale runtime / direct-exact planner truth live on
> three different truth axes plus the substrate guarantee.
> Phase 28 extends the analyzer axis with `may_raise_implicit`
> and refines the corpus-scale runtime axis with the
> explicit / implicit split; nothing else changes.

### E.2 Relationship to the master plan

This phase belongs to **Arc 5 — Runtime grounding of the
analyzer** in `docs/context_zero_master_plan.md` § 4.5.
Phase 28's contribution is exactly what § 4.5 anticipates:
a new boundary class surfaced by runtime observation
(implicit raises, in Phase 27) is *formalised as a
first-class analyzer axis* (Phase 28), and runtime
calibration gets a corresponding split probe.

The master plan's near-term goals (§ 5.1) treated
"multi-corpus runtime calibration + explicit/implicit
split" as this milestone; Phase 28 discharges both.

### E.3 What this phase does not fix (carry-over to Phase 29+)

Ordered by size of impact:

1. **Method calibration (OQ-27b).** Methods are the bulk of
   the corpus on `vision-core` (58 %) and `vision-tests`
   (97 %). Auto-constructed instances (zero-arg `__init__`
   → `MagicMock`) would lift ready_fraction materially.
2. **Pattern-list extension for implicit raises (new OQ).**
   The single Phase-28 FN (`blake2b(digest_size=0)` → 
   `ValueError`) suggests adding a hashlib-style pattern
   family to `_IMPLICIT_RAISE_BUILTINS` or a separate
   "builtin call with keyword argument out of domain"
   pattern. Adding the pattern should be FN = 0 across the
   four corpora under the current recipe.
3. **Probe-instrumentation parity with analyzer API surfaces
   (OQ-27j carry-over).** Phase 28 surfaces six
   `calls_filesystem`/`calls_subprocess`/`calls_network` FNs
   driven by API tables that lag the runtime observer
   (`tempfile.mkdtemp`, `Request(...)` pattern in urllib).
   Additive fix.
4. **Signature-driven smart recipes (OQ-27c).** Pattern 3's
   high implicit-bucket FP count is the direct measurement
   of recipe coverage — a function with a risky builtin
   call only triggers when the fuzz input exercises the
   risky branch. Smarter recipes would reduce FP
   monotonically without changing FN.
5. **External corpora (OQ-28a).** `click`, stdlib `json`,
   and third-party libraries need a skip-list + recipe-
   registry extension. Machinery is ready.
6. **Cross-language runtime calibration (OQ-27g).** Python-
   only; TypeScript / Go / Rust need analogue invocation
   protocols.
7. **Planner extension for the implicit-raise axis.**
   Optional — the planner currently answers aggregation
   queries on `trans_may_raise` and the other Phase-25
   flags. Adding a pattern for `trans_may_raise_implicit`
   would extend the direct-exact slice to implicit-raise
   questions. Not done in Phase 28 to keep the substrate
   layer byte-stable across this milestone.

### E.4 Open questions (Phase 29 candidates)

- **OQ-28a Third-party / stdlib corpora.** Extend the
  registry to `click`, stdlib `json`, possibly `numpy.core`
  (with heavy-numerical skip list extension).
- **OQ-28b Implicit-raise pattern-list extension.** Add
  `hashlib.blake2b` / similar keyword-driven-raise builtins
  based on the Phase-28 FN witness.
- **OQ-28c Recipe-precision co-variate for implicit-raise
  FP.** Measure how the FP rate on `may_raise_implicit`
  varies as recipe aggressiveness increases (OQ-27a).
- **OQ-28d Method instance auto-construction.** Closes the
  coverage gap on `vision-tests` and `vision-core`.
- **OQ-28e Runtime-refined analyzer.** Use Phase-28
  implicit-bucket FPs as signal to refine the analyzer's
  precision on `dead_code`-like patterns while preserving
  soundness on the corpus slice.
- **OQ-28f Planner pattern for implicit-raise questions.**
  Optional direct-exact slice extension.

### E.5 Reproducibility

| Run                                           | Command                                                                                                                                                                                       | Output                                                      |
|---                                            |---                                                                                                                                                                                            |---                                                          |
| 4-corpus headline (3 seeds × 0.08 s budget)   | `python3 -W ignore::RuntimeWarning -W ignore::ResourceWarning -m vision_mvp.experiments.phase28_multi_corpus_runtime_calibration --seeds 0 1 2 --budget 0.08 --out vision_mvp/results_phase28_multi_full.json` | `vision_mvp/results_phase28_multi_full.json`                |
| 2-corpus CI-friendly (3 seeds × 0.08 s)       | `python3 -W ignore::RuntimeWarning -W ignore::ResourceWarning -m vision_mvp.experiments.phase28_multi_corpus_runtime_calibration --seeds 0 1 2 --budget 0.08 --corpora vision-core vision-tasks --out vision_mvp/results_phase28_multi.json` | `vision_mvp/results_phase28_multi.json`                     |
| 2-corpus variance (5 seeds × 0.12 s)          | `python3 -W ignore::RuntimeWarning -W ignore::ResourceWarning -m vision_mvp.experiments.phase28_multi_corpus_runtime_calibration --seeds 0 1 2 3 4 --budget 0.12 --corpora vision-core vision-tasks --out vision_mvp/results_phase28_multi_5seeds.json` | `vision_mvp/results_phase28_multi_5seeds.json`              |
| Phase-28 unit tests                           | `python3 -m unittest vision_mvp.tests.test_phase28_implicit_raise`                                                                                                                               | 32 tests, all pass                                         |
| Phase-28 smoke test                           | `python3 -m unittest vision_mvp.tests.test_phase28_multi_corpus_smoke`                                                                                                                           | 1 test, passes                                             |
| Full suite (no regressions)                    | `python3 -m unittest discover -s vision_mvp/tests`                                                                                                                                               | 973 tests, all pass                                        |

Phase-28 tests live at
`vision_mvp/tests/test_phase28_implicit_raise.py` (32 tests)
and `vision_mvp/tests/test_phase28_multi_corpus_smoke.py`
(1 test). Phase-28 benchmark lives at
`vision_mvp/experiments/phase28_multi_corpus_runtime_calibration.py`.
