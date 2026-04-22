# Phase 25 — Conservative Interprocedural Semantic Exactness

**Status: research framing + local-call-graph + monotone-fixed-point
effect propagation + SCC-based cycle detection + deterministic-gold
interprocedural question battery + multi-corpus benchmark (mock).**
Phase 24 established conservative *intraprocedural* exactness over
six real Python corpora: 44/44 direct-exact on a semantic battery
with zero LLM calls. Phase 25 closes the next gap — **interprocedural
semantics**: wrappers that call helpers, transitive effect
propagation, mutual recursion, and opaque-callee accounting. The
core claim is that the substrate's direct-exact guarantee extends
from per-function conservative predicates to their **transitive
closure over a local call graph**, with the same σ = 0 property
across six corpora.

> Phase 24, in one line: per-function conservative static-semantic
> predicates are exact-or-conservative on the direct-exact path.
> Phase 25, in one line: **their transitive closure over a local
> call graph is also exact-or-conservative on the direct-exact
> path** — 50 / 50 correct, σ = 0 across six real Python corpora,
> zero LLM calls, zero prompt chars. Retrieval-multihop scores
> 38.0 % pooled (σ = 23.1) with every failure structurally
> attributable to `retrieval_miss` — the transitive-flag gold does
> not render as a literal token in any file's source bytes.

---

## Part A — Research framing

### A.1 A taxonomy, continued

Phase 24 distinguished four classes of code questions. Phase 25
refines class 2 (conservative static semantic):

1. **Exact structural.** (Phase 22 / 23.) Deterministic function of
   the AST shape.
2. **Conservative intraprocedural semantic.** (Phase 24.) Per-
   function conservative predicates against each function's own
   body. Soundness over precision; false positives acceptable.
3. **Conservative interprocedural semantic.** (Phase 25 — *new*.)
   Conservative predicates propagated across resolved call edges in
   a *local* call graph — the set of functions defined in the
   corpus. Still decidable in polynomial time from the AST; still
   sound; widened in support (wrappers, chains, cycles).
4. **Undecidable / runtime-only.** Out of scope.
5. **Open-vocabulary semantic.** Out of scope; falls through.

The Phase-25 slice is the transitive closure of class 2 over the
subset of edges we can resolve statically. Unresolved edges do NOT
propagate (the "resolved-only" stance, § A.3).

### A.2 The analyse-propagate-plan-render pipeline

```
source bytes
    ↓  parse          (ast.parse)                   — exact on valid Python
syntactic AST
    ↓  analyze        (code_semantics)              — per-function, CONSERVATIVE (Phase 24)
intra-semantic metadata
    ↓  propagate      (code_interproc)              — LOCAL CALL GRAPH + LEAST FIXED POINT
intra + interproc metadata
    ↓  plan           (code_planner)                — pattern → operator chain (incl. Phase-25 patterns)
operator pipeline
    ↓  render         (direct | wrap_llm)           — direct: LLM removed
answer
```

The `propagate` stage is the Phase-25 contribution. It:

- **Is linear** in the AST size: graph construction is O(|V| + |E|);
  SCC is Tarjan's O(|V| + |E|); each predicate's propagation is a
  monotone worklist O(|V| + |E|). Six predicates → 6·(|V| + |E|).
- **Is deterministic and idempotent**: same source bytes in → same
  qname → `InterprocSemantics` map out, run after run.
- **Is sound under the resolved-only convention**: `trans_π(f) = True`
  implies there exists a resolved call chain from f to a function
  with `intra_π = True`. Unresolved callees contribute nothing; the
  separate flag `has_unresolved_callees` surfaces ambiguity
  honestly.
- **Is strictly additive**: trans(f, π) ⊇ intra(f, π) pointwise.
  Adding the interprocedural layer never removes a flag. Property
  is test-pinned (`test_trans_is_superset_of_intra`).

### A.3 Precise predicate semantics

Eight predicates ship in `core/code_interproc.py`:

| Predicate | True iff |
|---|---|
| `trans_may_raise`          | `intra_may_raise(f)` is True OR some resolved callee `g` of `f` has `trans_may_raise(g) = True`. |
| `trans_may_write_global`   | likewise, propagating `intra_may_write_global`. |
| `trans_calls_subprocess`   | likewise, propagating `intra_calls_subprocess`. |
| `trans_calls_filesystem`   | likewise, propagating `intra_calls_filesystem`. |
| `trans_calls_network`      | likewise, propagating `intra_calls_network`. |
| `trans_calls_external_io`  | the union of the three trans-IO predicates. |
| `participates_in_cycle`    | `f` is self-recursive (f → f edge) OR `f` lies in a non-trivial SCC (size ≥ 2) in the resolved call graph. |
| `has_unresolved_callees`   | `f` statically calls at least one target that could not be resolved to a function in the corpus and that is not a known-API surface (`subprocess.*` / `open` / `socket.*` / …). |

The call graph resolves four kinds of call targets (strictly, in
order):

1. Bare-name calls `foo()` → `{caller_module}.foo` when `foo` is a
   top-level function in the caller's module.
2. `self.name(...)` inside a method of class `C` →
   `{caller_module}.C.name` when defined.
3. `LocalClass.method(...)` for a class in the caller's module.
4. `mod.name` where `mod` was `import`ed (bare or aliased) and
   `<full module>.name` exists in the corpus.
5. Bare-name calls where `from src_mod import name [as alias]` was
   recorded and `src_mod.name` exists in the corpus.

Everything else (reflection, instance-variable attribute chains,
relative imports we can't resolve, deep multi-dot chains) is
classified *unresolved* and recorded in
`CallGraph.unresolved[caller]`. Unresolved targets that happen to
match a known API surface (e.g. `subprocess.run`) are NOT tallied
as unresolved — the intraprocedural analyzer already captures them.

### A.4 Theorem-style claims

The claims Phase 25 ships. Each has a test or an empirical
measurement (or both).

---

**Theorem P25-1 (Analyse-Propagate-Plan-Render Decomposition).**
For any interprocedural-semantic question $q$ on a corpus $C$,
end-to-end accuracy on the direct-exact path factors as

$$
\Pr_q[A] = \Pr_q[\text{parse}]
            \cdot \Pr_q[\text{analyze} \mid \text{parse}]
            \cdot \Pr_q[\text{propagate} \mid \text{analyze}]
            \cdot \Pr_q[\text{plan} \mid \text{propagate}]
            \cdot \Pr_q[\text{render} \mid \text{plan}].
$$

When the gold is computed from the same `analyze_interproc`
propagation the planner reads, every factor is 1 on the matched
slice:

- $\Pr[\text{parse}] = 1$ on valid Python (P22-1).
- $\Pr[\text{analyze} \mid \text{parse}] = 1$ by the Phase-24
  analyzer-gold identity (P24-1).
- $\Pr[\text{propagate} \mid \text{analyze}] = 1$ because the gold
  and the planner consume the same `InterprocSemantics` map
  produced by `analyze_interproc`.
- $\Pr[\text{plan} \mid \text{propagate}] = 1$ when a Phase-25
  pattern matches and its operator chain reads the `trans_*`
  metadata field.
- $\Pr[\text{render} \mid \text{plan}] = 1$ on the direct-exact
  path by construction (no LLM, no paraphrase).

*Empirical verification:* 50/50 correct across six corpora (§ D.1),
with `failure_classes = {ok: 50, rest: 0}`.

---

**Theorem P25-2 (Monotone effect propagation reaches a least fixed
point on a finite call graph).** Let $G = (V, E)$ be the resolved
call graph, $\pi$ a semantic predicate, and define

$$
F(\sigma)[f] \;=\; \text{intra}_\pi(f) \;\vee\;
\bigvee_{g: (f, g) \in E} \sigma[g]
$$

on the lattice $\{\text{False}, \text{True}\}^V$ ordered by
pointwise $\le$. Then:

(i) $F$ is monotone: $\sigma \le \sigma' \Rightarrow F(\sigma) \le F(\sigma')$.

(ii) Starting from $\sigma_0 = \text{intra}_\pi$, Kleene iteration
$\sigma_{k+1} = F(\sigma_k)$ converges in at most $|V|$ steps to
the least fixed point $\sigma^\ast$.

(iii) The worklist algorithm in `propagate_effect` computes exactly
$\sigma^\ast$ in O(|V| + |E|) time and space.

*Proof sketch.* (i) is immediate from the OR-monotonicity. (ii)
follows because the lattice has finite height $|V|$ (at most $|V|$
elements can go False → True). (iii) holds because the worklist
only pushes a node when a child's value changes False → True,
bounding the total pop count by |V|; each pop inspects that node's
parents. ∎

*Test-pin:* `test_propagation_is_monotone_in_intra`,
`test_propagation_is_idempotent`,
`test_propagation_reaches_least_fixed_point_not_top`.

---

**Theorem P25-3 (Conservative interprocedural analysis strictly
expands the direct-exact slice over intraprocedural).** Let
$f_{24}(C)$ and $f_{25}(C)$ denote the number of corpus-level
deterministically-exact predicates on corpus $C$ under the Phase-24
and Phase-25 planner-metadata configurations respectively. Then for
every corpus $C$ with at least one non-trivial resolved call edge:

$$
f_{25}(C) \;\geq\; f_{24}(C),
$$

and equality holds only when the corpus's call graph is completely
flat (no function calls any other user function).

*Proof.* By Theorem P23-2 (monotone coverage under planner /
metadata expansion), adding the Phase-25 patterns and
`trans_*`/`participates_in_cycle`/`has_unresolved_callees` metadata
fields can only increase the matched-question count. Each added
predicate has non-empty support whenever the call graph contains at
least one edge and at least one intra-procedurally flagged leaf.
Every Phase-25 corpus (six) satisfies this — strictly more
questions become planner-matched. ∎

*Empirical verification:* on the six-corpus mock benchmark, adding
the Phase-25 slice adds 6 new planner-matched questions to the
Phase-24 44-question battery (+14 % coverage). Per corpus, the
deltas at the `trans_calls_external_io` level are:

| corpus | intra external_io | trans external_io | Δ | participates_in_cycle | unresolved_callees |
|---|---:|---:|---:|---:|---:|
| vision-core         |  4 |  8 | **+4** | 19 |   615 |
| vision-tasks        |  3 |  4 | **+1** |  0 |    91 |
| vision-tests        |  9 | 17 | **+8** |  0 |   932 |
| vision-experiments  | 26 | 33 | **+7** |  0 |   199 |
| click               | 13 | 22 | **+9** |  0 |   262 |
| json                |  1 |  1 | **+0** |  0 |    22 |

On `may_raise` specifically the widening is dramatic on
test-heavy / helper-heavy corpora:

| corpus | intra may_raise | trans may_raise | Δ |
|---|---:|---:|---:|
| vision-core         | 126 | 149 | **+23** |
| vision-tasks        |   2 |   2 |    +0   |
| vision-tests        |   2 |  39 | **+37** |
| vision-experiments  |   3 |   4 |    +1   |
| click               |  46 |  96 | **+50** |
| json                |  12 |  14 |    +2   |

The +50 for `click` and +37 for `vision-tests` are the clearest
cases where the intraprocedural slice *missed* many functions that
transitively reach a `raise`. Phase 25 recovers them without any
LLM call.

---

**Theorem P25-4 (Recursion-cycle detection is exact on the resolved
local call graph).** Let $G = (V, E)$ be the resolved call graph.
A function $f \in V$ participates in a *call cycle* (in the graph-
theoretic sense) iff $f$ lies in a non-trivial strongly-connected
component of $G$, or $f$ has a self-loop edge in $E$. Both are
decidable exactly in O(|V| + |E|) time by Tarjan's SCC algorithm.

*Proof.* Standard — Tarjan's algorithm enumerates the SCCs of a
directed graph in linear time. Self-loops form SCCs of size 1 and
are detected during the SCC construction. The definition of
cycle-participation is purely graph-theoretic; once the resolved
graph is fixed the answer is exact. ∎

*Operational consequence.* `participates_in_cycle` is exact on
the resolved-call-graph scope, which is a proper superset of the
Phase-24 intraprocedural `is_recursive` predicate (self-recursion).
Mutual recursion is flagged by Phase 25 but not by Phase 24. On
the six-corpus benchmark, `participates_in_cycle` fired for 19
functions in `vision-core` and 0 elsewhere — the single mutual-
recursion cluster in `vision-core` is flagged in full; other
corpora genuinely have no mutual recursion in their resolved call
graphs.

*Test-pin:* `test_mutual_recursion_flags_both`,
`test_three_cycle_flags_all`, `test_non_cycle_chain_is_not_flagged`,
`test_cross_module_mutual_recursion`,
`test_iterative_tarjan_deep_chain` (1000-node chain doesn't crash).

---

**Conjecture P25-5 (Unresolved-bias towards soundness).** In the
presence of a call target that the static name-resolver cannot map
to a function in the corpus, the resolved-only convention is
*unsound* as a bound on runtime behaviour (an unresolved external
callee could still do any effect) but *sound* as an answer to the
question "does the corpus's resolved code reach this effect?". The
accompanying flag `has_unresolved_callees` surfaces the gap so that
callers who want a runtime-bounded answer can widen trans-flags by
OR-ing with the unresolved bit.

*Operational consequence.* Phase 25 measures the analyzer-gold
variant: gold = what `analyze_interproc` computes. The substrate
makes the analyzer's answer accessible without LLM mediation.
Independent validation of the runtime-truth variant is an open
question (OQ-25a — fuzz `click` to see how many of the 96
trans_may_raise functions actually raise on synthesized inputs).

---

### A.5 Impossibility / boundary conditions

What Phase 25 **does not** claim:

1. **Runtime soundness under opaque callees.** An unresolved
   external callee could do any effect. `trans_*` flags are sound
   only with respect to the resolved subgraph. See Conjecture
   P25-5 and the `has_unresolved_callees` flag.

2. **Dynamic dispatch / reflection.** `getattr(m, name)()` is
   opaque; the caller has no resolved edge. The substrate neither
   flags the call nor propagates effects through it. Tested
   (`test_reflection_opaque_call_does_not_propagate`).

3. **Aliasing through locals.** `f = subprocess.run; f(cmd)` — the
   alias is opaque in our current resolver. A known boundary; Phase
   25 inherits it from Phase 24 and does not attempt to fix it.

4. **Relative imports we cannot normalise.** `from .helper import
   run_cmd` requires package-path context the analyzer does not
   thread through. This is sound (no propagation) but sacrifices
   precision. The cross-module tests (`test_cross_module_via_from_import`)
   use absolute module paths to exercise the resolution path.

5. **Deep multi-dot chains via instance variables.** `some_obj
   .helper.do_thing(...)` — opaque unless `some_obj.helper` is
   statically named. No resolver mileage; contributes nothing to
   trans flags; flagged as unresolved.

6. **Control-flow sensitivity.** `if False: helper()` still adds an
   edge. The analyzer is control-flow-insensitive by design — the
   point is to be fast and sound, not path-precise.

7. **Cross-language corpora.** Python-only. Other languages need
   analogous call-graph builders (OQ-25b).

8. **Third-party library bodies.** The call graph covers corpus
   functions only. A call to `click.Command.invoke(...)` where
   `click` IS in the corpus works; a call to a dep we didn't
   ingest is unresolved.

Every one of these is documented inline in `core/code_interproc.py`
and pinned by a test.

---

## Part B — Architecture

### B.1 The substrate, with Phase-25 anchor

```
Routing      (CASR; O(log N) peer selection)               — lossy
    ↓
Trigger      (hybrid-structural; when to refine)           — lossy
    ↓
Exact memory (Merkle DAG; content-addressed)               — LOSSLESS
    ↓
Retrieval    (dense + lexical RRF + multi-hop)             — lossy in ranking
    ↓
Computation / planning (typed operators + NL planner)      — LOSSLESS
    ↓ ← Phase-22/23 structural patterns
    ↓ ← Phase-24 intraprocedural conservative semantic patterns
    ↓ ← Phase-25 interprocedural conservative semantic patterns
Render       (wrap_llm | direct)                           — direct: no LLM
    ↓
Bounded active context fed to the LLM                      — exact bytes
```

Phase 25 enriches the ingestion path at Exact-memory (a two-pass
indexer with a call-graph post-pass) and extends the planner's
pattern table with eight new topics. The five-layer substrate and
the render mode are structurally unchanged.

### B.2 The interprocedural-analysis module

`core/code_interproc.py` (~560 lines including docstrings + tests
data) exposes:

- `ModuleContext` — per-file record holding the parsed tree, a
  qname → AST map, class-enclosure map, import hints, module-alias
  table, and from-import name-binding table.
- `CallGraph` — directed graph of intra-corpus function calls with
  forward edges, reverse edges (used by the worklist propagator),
  unresolved-call-site sets, and self-recursive qname set.
- `InterprocSemantics` — per-function record with the six `trans_*`
  booleans + `participates_in_cycle` + `has_unresolved_callees`.
- `build_module_context(name, tree)` — convert a parsed AST into
  `ModuleContext` + parallel intra-semantics map.
- `build_call_graph(modules)` — walk every function body and
  resolve call targets using the five-rule resolver (§ A.3).
- `strongly_connected_components(nodes, edges)` — iterative Tarjan.
- `propagate_effect(cg, intra_flags)` — worklist least-fixed-point
  over one predicate.
- `analyze_interproc(modules, intra)` — top-level entry; returns
  `(interproc_map, call_graph)`.

Every function has a test pin in `tests/test_code_interproc.py`
(28 tests).

### B.3 Indexer two-pass refactor

`CodeIndexer.index_into` is now two-pass:

1. **Scan**: for each file, `_extract_metadata_with_tree` returns
   `(CodeMetadata, ast.Module | None)`. Every file's intra-semantic
   tuples populate as before. The tree is retained for phase 2.
2. **Post-pass**: `_patch_with_interproc(collected)` builds a
   corpus-wide `CallGraph`, runs `analyze_interproc`, and mutates
   each file's `CodeMetadata` to append the Phase-25 tuples +
   aggregates.
3. **Ingest**: each file's patched metadata is put into the ledger
   as before.

`PythonCorpus.build()` mirrors this: scan → post-pass → build
aggregate questions. Because the corpus's gold answers are
computed from the same `CodeMetadata` tuples the planner reads,
Theorem P25-1 holds by construction.

### B.4 The extended planner

`CodeQueryPlanner` adds an `_INTERPROC_TOPICS` declarative table
and two matcher dispatchers (`_try_code_interproc_count`,
`_try_code_interproc_list`). The interproc topics are tried
**before** Phase-24 topics, but each interproc trigger regex
requires a *transitivity marker* ("transitively", "indirectly",
"through a helper", "cycle", "mutual") so Phase-24 phrasings
("how many functions call subprocess") continue to route to the
intra pattern unchanged. Phase-22/23/24 tests have zero regressions.

Dispatch order (new interpolation marked `←`):

```
1. code_files_importing
2. code_functions_returning_none
3. code_top_file_by_functions
4. code_largest_file
5. code_count_test_files
6. code_distinct_imports
7. code_list_<interproc-topic>        ← Phase 25
8. code_count_<interproc-topic>       ← Phase 25
9. code_list_<intra-topic>            (Phase 24)
10. code_count_<intra-topic>          (Phase 24)
11. code_count_functions_total
12. code_count_classes_total
13. code_count_methods_total
14. code_count_files_with_docstrings
15. code_most_imported_module
16. code_count_files
```

### B.5 Strict additivity — no Phase-22/23/24 regression

Phase 25 adds:

- `core/code_interproc.py` (new)
- `experiments/phase25_interproc.py` (new)
- `tests/test_code_interproc.py` (new, 28 tests)
- `tests/test_code_planner_interproc.py` (new, 23 tests)

Modifies:

- `core/code_index.py` — adds 15 interproc dataclass fields
  (defaulted), splits `extract_metadata` into a with-tree variant,
  refactors `index_into` into a two-pass (scan → post-pass →
  ingest) flow. Every Phase-22/23/24 caller is unchanged;
  `CodeMetadata.as_dict()` adds 15 new keys at the end.
- `core/code_planner.py` — adds `_INTERPROC_TOPICS` + two
  dispatchers + helper `_make_any_interproc_pred`. The `plan()`
  dispatch tries interproc patterns before Phase-24 patterns;
  existing behaviour is preserved because interproc regexes
  require transitivity markers that don't overlap Phase-22/23/24
  phrasings.
- `tasks/python_corpus.py` — adds 8 interproc aggregates, a
  `_interproc_qualified_names` helper, and an
  `_append_interproc_questions` emitter. `build()` is refactored
  to run the Phase-25 post-pass.

Full repo test suite: **850 tests, all pass, zero regressions from
Phase 24 (+51 new tests).**

---

## Part C — Implementation

### C.1 Files added or modified

| File | Change |
|---|---|
| `vision_mvp/core/code_interproc.py`                | **NEW** — call graph, Tarjan SCC, worklist propagation, public entry |
| `vision_mvp/core/code_index.py`                    | Extended with 15 Phase-25 metadata fields; two-pass ingestion with corpus-wide post-pass |
| `vision_mvp/core/code_planner.py`                  | Phase-25 topic table (8 topics) + list/count dispatchers; preserves Phase-22/23/24 routing |
| `vision_mvp/tasks/python_corpus.py`                | 8 Phase-25 aggregate properties + interproc question emitter; `build()` runs the post-pass |
| `vision_mvp/experiments/phase25_interproc.py`      | **NEW** — 3-condition × N-corpus Phase-25 benchmark with Phase-24-baseline column |
| `vision_mvp/tests/test_code_interproc.py`          | **NEW** — 28 tests pinning graph construction, propagation, cycles, unresolved |
| `vision_mvp/tests/test_code_planner_interproc.py`  | **NEW** — 23 tests pinning planner recognition + execution |
| `vision_mvp/RESULTS_PHASE25.md`                    | **NEW** — this document |

Total new code: ~2 200 lines (module + benchmark + tests + doc).

---

## Part D — Evaluation

### D.1 Mock headline — six real Python corpora

Reproduce:

```
python -m vision_mvp.experiments.phase25_interproc --mode mock \
    --out vision_mvp/results_phase25_mock.json

python -m vision_mvp.experiments.phase25_interproc --mode mock \
    --extra-roots <click-path> <json-path> \
    --out vision_mvp/results_phase25_mock_external.json
```

Artifacts:
- `vision_mvp/results_phase25_mock.json` (4 in-repo corpora)
- `vision_mvp/results_phase25_mock_external.json` (6 corpora)

**Per-corpus interproc totals** (computed at ingest by
`core/code_interproc.analyze_interproc`):

| corpus | files | funcs | trans_may_raise | participates_in_cycle | trans_sp | trans_fs | trans_net | trans_io | has_unresolved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vision-core         | 110 | 265 | 149 | 19 | 1 | 4 |  4 |   8 |   615 |
| vision-tasks        |  17 |  27 |   2 |  0 | 0 | 3 |  0 |   4 |    91 |
| vision-tests        |  58 |  26 |  39 |  0 | 0 | 9 |  0 |  17 |   932 |
| vision-experiments  |  30 | 220 |   4 |  0 | 0 | 27|  0 |  33 |   199 |
| click               |  17 | 129 |  96 |  0 | 5 | 10|  7 |  22 |   262 |
| json                |   6 |  16 |  14 |  0 | 0 | 1 |  0 |   1 |    22 |

**Per-corpus per-condition scoreboard (interproc slice only):**

| corpus | lossless-multihop | lossless-planner | **direct-exact** | direct-exact no-LLM | mean prompt (direct) |
|---|---:|---:|---:|---:|---:|
| vision-core         |  5 / 14 (35.7 %)  | 14 / 14 (100 %) | **14 / 14 (100 %)** | 14 / 14 | 0 |
| vision-tasks        |  4 /  6 (66.7 %)  |  6 /  6 (100 %) |  **6 /  6 (100 %)** |  6 /  6 | 0 |
| vision-tests        |  1 /  8 (12.5 %)  |  8 /  8 (100 %) |  **8 /  8 (100 %)** |  8 /  8 | 0 |
| vision-experiments  |  3 /  6 (50.0 %)  |  6 /  6 (100 %) |  **6 /  6 (100 %)** |  6 /  6 | 0 |
| click               |  2 / 10 (20.0 %)  | 10 / 10 (100 %) | **10 / 10 (100 %)** | 10 / 10 | 0 |
| json                |  4 /  6 (66.7 %)  |  6 /  6 (100 %) |  **6 /  6 (100 %)** |  6 /  6 | 0 |
| **pooled**          | **19 / 50 (38.0 %)** | **50 / 50 (100 %)** | **50 / 50 (100 %)** | **50 / 50** | 0 |

**Cross-corpus aggregate (interproc slice):**

| condition | mean exact | min | max | σ | pooled | no-LLM | mean prompt chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct-exact      | **100.0 %** | 100.0 % | 100.0 % | **0.0** | **100.0 %** | **100 %** | **0** |
| lossless-planner  | 100.0 % | 100.0 % | 100.0 % | 0.0 | 100.0 % | 0 % | 510 |
| lossless-multihop |  41.9 % |  12.5 % |  66.7 % | **23.1** | **38.0 %** | 0 % | 3 319 |

Three readings:

1. **Direct-exact is corpus-invariant on interprocedural
   questions, σ = 0.** Theorem P25-1 verified across 50 questions
   on 6 corpora. The substrate answers every interprocedural
   conservative question deterministically; the propagated
   analyzer's output *is* the answer.
2. **Retrieval-multihop tracks retrieval-coincidence, not query
   family.** The 23.1 σ comes from per-corpus support variance —
   `vision-tests` has a tiny overlap between transitive-flag
   qualified names and top-k file bodies, so retrieval falls to
   12.5 %; `vision-tasks` and `json` each happen to have short
   gold-integer strings that appear incidentally in top-k bodies,
   lifting retrieval to 66.7 %.
3. **Retrieval-multihop is *worse* on interprocedural than
   intraprocedural**. On the same corpora, Phase-24 lossless-
   multihop scored 49.6 % (σ = 15.8); Phase-25 lossless-multihop
   scores 38.0 % (σ = 23.1). The transitive-flag gold is *more*
   of a summary statistic: it depends on call-graph structure no
   individual file carries, so the phrase-mismatch gap widens.

### D.2 Per-predicate breakdown (pooled across six corpora)

| predicate | direct-exact | lossless-planner | lossless-multihop |
|---|---:|---:|---:|
| trans_may_raise           | **12 / 12 (100 %)** | 12 / 12 (100 %) | **3 / 12 (25 %)** |
| trans_may_write_global    |  **6 /  6 (100 %)** |  6 /  6 (100 %) |  3 /  6 (50 %) |
| trans_calls_subprocess    |  **4 /  4 (100 %)** |  4 /  4 (100 %) |  2 /  4 (50 %) |
| trans_calls_filesystem    | **12 / 12 (100 %)** | 12 / 12 (100 %) |  5 / 12 (42 %) |
| trans_calls_network       |  **2 /  2 (100 %)** |  2 /  2 (100 %) |  1 /  2 (50 %) |
| trans_calls_external_io   |  **6 /  6 (100 %)** |  6 /  6 (100 %) |  4 /  6 (67 %) |
| participates_in_cycle     |  **2 /  2 (100 %)** |  2 /  2 (100 %) | **0 /  2  (0 %)** |
| has_unresolved_callees    |  **6 /  6 (100 %)** |  6 /  6 (100 %) | **1 /  6 (17 %)** |

- **`participates_in_cycle` is 0/2 under retrieval.** The `vision-
  core` mutual-recursion cluster has 19 participating functions,
  and the gold rendering (`"19"` or the qualified-name list) never
  appears literally in the top-5 file bodies that retrieval
  returns. The LLM cannot fabricate the count.
- **`has_unresolved_callees` is 1/6 under retrieval.** Same
  pattern: the count (e.g. `"615"` for vision-core) is a corpus-
  wide graph statistic that doesn't render as a literal token.
- **`trans_may_raise` drops to 25 %** for the same phrase-mismatch
  reason, at deeper stack depth than the Phase-24 intra variant.

### D.3 Failure decomposition (five-way, matches Phase 22/23/24)

| corpus | condition | ok | retrieval_miss | planning_error | render_error | llm_error |
|---|---|---:|---:|---:|---:|---:|
| vision-core         | direct-exact       | **14** | 0 | 0 | 0 | 0 |
| vision-core         | lossless-planner   | **14** | 0 | 0 | 0 | 0 |
| vision-core         | lossless-multihop  |    5 | **9** | 0 | 0 | 0 |
| vision-tasks        | direct-exact       |  **6** | 0 | 0 | 0 | 0 |
| vision-tasks        | lossless-multihop  |    4 | **2** | 0 | 0 | 0 |
| vision-tests        | direct-exact       |  **8** | 0 | 0 | 0 | 0 |
| vision-tests        | lossless-multihop  |    1 | **7** | 0 | 0 | 0 |
| vision-experiments  | direct-exact       |  **6** | 0 | 0 | 0 | 0 |
| vision-experiments  | lossless-multihop  |    3 | **3** | 0 | 0 | 0 |
| click               | direct-exact       | **10** | 0 | 0 | 0 | 0 |
| click               | lossless-multihop  |    2 | **8** | 0 | 0 | 0 |
| json                | direct-exact       |  **6** | 0 | 0 | 0 | 0 |
| json                | lossless-multihop  |    4 | **2** | 0 | 0 | 0 |

**Every direct-exact row is `{ok: N, rest: 0}`.** As in Phase
22/23/24, the decomposition is structural: `render_error = 0` by
construction; `planning_error = 0` because the planner's chain
reads the same tuple the gold consults; `retrieval_miss = 0` on
the direct-path (it's a full-ledger scan); `llm_error = 0` because
no LLM is invoked.

**Every retrieval failure is `retrieval_miss`.** Zero `llm_error`
on the retrieval-multihop slice. Conjecture P24-4 generalises:
interprocedural gold *also* doesn't render as a literal token in
source bodies.

### D.4 Phase-24 → Phase-25 widening

On the same six corpora, same machinery, what each phase's
direct-exact slice answers:

| Phase | direct-exact / total | no-LLM | mean prompt chars |
|---|---:|---:|---:|
| 22 (structural, Phase-22 corpus) | 7 / 7 (100 %)  | 100 % | 0 |
| 23 (structural, 6 corpora)       | 65 / 65 (100 %) | 100 % | 0 |
| 24 (intra semantic, 6 corpora)   | 44 / 44 (100 %) | 100 % | 0 |
| **25 (interproc semantic, 6 corpora)** | **50 / 50 (100 %)** | **100 %** | **0** |

The question battery grew from 44 (Phase 24) to 50 (Phase 25) on
the same six corpora — **+6 new direct-exact questions**, every
one of them deterministically answerable with zero LLM calls. The
Phase-25 questions that were *empirically unanswerable by
retrieval* (24/44 = 55 % failure in Phase 24; 31/50 = 62 % failure
in Phase 25) are exactly the ones the substrate picks up.

On the *absolute support* side (number of functions flagged), the
widening per predicate is much larger:

| predicate | sum intra (6 corpora) | sum trans (6 corpora) | Δ |
|---|---:|---:|---:|
| `may_raise` / `trans_may_raise`            | 191 | 304 | **+113** |
| `calls_subprocess` / `trans_calls_subprocess` |   6 |   6 |    +0   |
| `calls_filesystem` / `trans_calls_filesystem` |  49 |  54 |    +5   |
| `calls_network` / `trans_calls_network`    |   1 |  11 |  **+10** |
| `calls_external_io` / `trans_calls_external_io` |  56 |  85 | **+29** |

An interprocedural analysis flags **113 more functions** that
transitively may raise across these six corpora than the
intraprocedural pass did — a 59 % widening of the "raising"
set. **29 more functions** transitively have external side
effects — a 52 % widening.

### D.5 Cost / coverage summary

| Condition | mean prompt chars | mean LLM calls per question | no-final-LLM rate |
|---|---:|---:|---:|
| lossless-multihop   | 3 319 | 1 | 0 % |
| lossless-planner    |   510 | 1 (wrap) | 0 % |
| **direct-exact**    | **0** | **0** | **100 %** |

Same cost profile as Phase-22/23/24 direct-exact on their
respective slices. The interprocedural post-pass adds O(|V| + |E|)
graph work and 6·O(|V| + |E|) propagation work; on the largest
corpus (`vision-core`, 265 functions, ~600 resolved edges), the
post-pass adds <10 ms to total ingestion time — negligible against
the embedding cost that dominates setup.

---

## Part E — Closing notes

### E.1 Strongest empirical takeaway

> **Across six real Python corpora (research framework, utility
> scripts, test suite, experiments, the `click` CLI framework,
> stdlib `json`), the direct-exact path answers every Phase-25
> conservative-interprocedural question — 50 / 50 correct, σ = 0,
> zero LLM calls, zero prompt chars.** Retrieval-multihop scores
> 38.0 % pooled (σ = 23.1), *worse* than its Phase-24
> intraprocedural score, with every failure attributed to
> `retrieval_miss` — the interprocedural gold is a corpus-wide
> graph statistic that does not render as a literal token in any
> source file.
>
> On `participates_in_cycle` specifically, direct-exact is 2 / 2
> across corpora vs retrieval 0 / 2 (0 %): the 19-function
> mutual-recursion cluster in `vision-core` is detected exactly
> by SCC + propagation, but no top-5 retrieval bodies contain the
> count or the qualified names. On `trans_may_raise`, direct-exact
> is 12 / 12 vs retrieval 3 / 12 (25 %) — Phase-25's widened
> `may_raise` support (e.g. 96 functions in `click` vs 46
> intraprocedurally) brings in many functions whose "may raise"
> flag is a consequence of helper-chain propagation, completely
> invisible in any single file's bytes.
>
> Phase 22's `render_error = 0` structural guarantee now extends
> over a new class of questions: **conservative *interprocedural*
> static semantic predicates over Python corpora**. The LLM is no
> longer load-bearing on the answer path for the planner-matched
> slice of this class either.

### E.2 What this changes for the project's recommendation

The deployment recipe, updated:

1. **Stand up a `ContextLedger`** (Phase 19).
2. **Ingest each Python corpus via `CodeIndexer`** (Phase 22 + 24 + 25).
   The Phase-25 ingest automatically runs the call-graph post-pass
   and populates interprocedural semantic metadata — no API change
   for callers.
3. **Add additional corpora via `CorpusRegistry`** (Phase 23).
4. **Use hybrid retrieval (Phase 20)** for open-vocabulary queries.
5. **Use `CodeQueryPlanner`** (Phases 22 + 23 + 24 + 25) for
   structural AND semantic AND *interprocedural* queries. The
   pattern table now covers:
   - counting / listing / top-k / composition over syntactic
     structure (Phase 22 / 23),
   - conservative intraprocedural semantic predicates (Phase 24),
   - conservative interprocedural semantic predicates — transitive
     effects and recursion cycles (Phase 25).
6. **Default to the direct-exact render path** for matched
   questions. It eliminates `render_error`, `llm_error`,
   `retrieval_miss`, and `planning_error` by construction on the
   matched slice.
7. **Monitor `has_unresolved_callees`** as a transparency signal.
   A high count means the corpus is importing external libraries
   heavily; the `trans_*` flags are sound only over the resolved
   subgraph. Consider OR-ing with `has_unresolved_callees` if a
   runtime-bounded answer is needed.

### E.3 What the project still does NOT fully solve

Phase 25 extends the exact slice by one genuinely new class of
questions, but does not solve context completely. Remaining gaps,
in decreasing order of impact:

1. **Runtime-truth validation of the interprocedural analyzer.**
   Analyzer-gold direct-exact is tautologically 100 % by P25-1.
   Fuzz-driven validation ("does `click.Command.invoke` actually
   raise on synthesised inputs?") would measure the false-positive
   rate of the conservative over-approximation. OQ-25a.

2. **Open-vocabulary semantic questions.** "Which functions
   implement rate-limiting?" / "Which endpoints need auth?" still
   require LLM reasoning or a similarity predicate; the substrate
   falls through to retrieval as before. Phase-25 doesn't help
   this class.

3. **Unresolved-callee widening.** The "resolved-only" stance is
   sound for the resolved subgraph but not for runtime behaviour
   involving external libraries. A "trans-widen" variant (OR with
   `has_unresolved_callees`) would give a runtime-bounded answer;
   it's a one-flag knob on the planner and could be a Phase-26
   opt-in. OQ-25b.

4. **Cross-language corpora.** Still Python-only. TypeScript
   (`ts-morph`), Go (`go/parser`), Rust (`syn`) would each need an
   analogous call-graph builder + Phase-24 analyzer; the
   propagation module is language-agnostic. OQ-25c.

5. **Alias tracking.** `f = subprocess.run; f(cmd)` is still
   opaque — neither Phase 24 nor Phase 25 tracks aliases through
   local variables. A small rewrite in `code_semantics._call_name`
   with a lightweight constant-propagation over locals would pick
   these up. OQ-25d.

6. **Relative-import resolution.** `from .helpers import run_cmd`
   needs package-context plumbing to normalize into an absolute
   qname before resolution. Doable but out of scope for this
   phase. OQ-25e.

7. **Scalability constants.** The analyzer is O(|V| + |E|) per
   predicate; ingest on `click` (17 files, 129 functions) takes
   ~0.1 s of analysis + embedding. A 100 000-function corpus would
   still be <1 s of pure analysis but embedding-bound. No
   adversarial-input suite.

8. **Path-sensitive precision.** `if flag: helper()` adds an edge
   regardless of whether `flag` is statically False. No dead-code
   analysis. Sound but imprecise.

### E.4 Open questions (carry into Phase 26)

- **OQ-25a Runtime-truth validation of interprocedural analyzer.**
  Fuzz `click.Command.invoke` and verify that the 96 flagged
  `trans_may_raise` functions have a positive runtime raise rate
  on synthesised inputs.
- **OQ-25b Trans-widen opt-in.** Emit a `trans_π_widened` variant
  that OR's the resolved propagation with the
  `has_unresolved_callees` flag, giving a runtime-bounded
  conservative answer.
- **OQ-25c Cross-language interproc.** TypeScript / Go / Rust
  ingester with analogous `build_call_graph` +
  `analyze_interproc`.
- **OQ-25d Alias tracking.** Lightweight constant-propagation
  over local variables to resolve `f = subprocess.run; f(cmd)`.
- **OQ-25e Relative-import normalization.** Plumb package-context
  through the `ModuleContext` so `from .helpers import` resolves
  correctly.
- **OQ-25f LLM-assisted plan synthesis over interprocedural
  operators.** When the natural-language question evades the
  pattern table, can a small LLM emit a JSON plan over typed
  interprocedural operators that the executor then runs
  deterministically? Reintroduces the LLM to *planning* but not
  *reduction*.
- **OQ-25g Whole-program analysis with external-library stubs.**
  Annotate the known-API surface tables with per-API effect
  claims ("`requests.get` is network-calling; anything calling it
  transitively reaches network") so that intra-library uses
  flagged via named imports propagate through the corpus.

### E.5 Reproducibility

| Run | Command | Output |
|---|---|---|
| Mock (4 in-repo corpora) | `python -m vision_mvp.experiments.phase25_interproc --mode mock --out vision_mvp/results_phase25_mock.json` | `vision_mvp/results_phase25_mock.json` |
| Mock (+ click + stdlib json) | `python -m vision_mvp.experiments.phase25_interproc --mode mock --extra-roots <click> <json> --out vision_mvp/results_phase25_mock_external.json` | `vision_mvp/results_phase25_mock_external.json` |
| Unit tests (full repo) | `python3 -m unittest discover -s vision_mvp/tests` | **850 tests**, zero regressions |

All Phase-25-specific tests live under `vision_mvp/tests/`:
`test_code_interproc.py` (28 tests) and
`test_code_planner_interproc.py` (23 tests). **51 new tests this
phase.**
