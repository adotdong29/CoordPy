# Math Audit — What Theory Is Actually Used

Honest accounting of which of the 72 mathematical frameworks documented
in `EXTENDED_MATH_[1-7].md` are actually wired into code in `vision_mvp/`,
vs. which are still theory-only.

Grading legend:

- **USED** — directly implemented and exercised in at least one experiment
- **STRUCTURAL** — the code shape matches the framework; it informs design
  but isn't a separate module
- **BUILT** — code exists but hasn't been exercised against a real task
- **THEORY** — in the math doc only, no code

## Core framework (from FRAMEWORK.md)

| # | Framework | Status | Where |
|---|---|:-:|---|
| F1 | Information Bottleneck (min-sufficient statistic) | **STRUCTURAL** | `Manifold`, `StreamingPCA` — the projection IS the IB compression |
| F2 | Causal Abstraction / do-calculus | **THEORY** | Used as motivation; filter is surprise-based, not interventional |
| F3 | Renormalization Group / scale projection | **USED** | `ContinuousScaleProjector`, `HierarchicalRouter` |
| F4 | Predictive Coding (surprise filter) | **USED** | `NeuralPredictor`, `PredictorBank`; core of workspace admission |

## Volume 1 (geometric / topological / categorical)

| # | Framework | Status | Where |
|---|---|:-:|---|
| V1.5 | Sheaf Theory / H¹ obstruction | **THEORY** | — |
| V1.6 | Synergetics / slaving principle | **STRUCTURAL** | Workspace = order parameters |
| V1.7 | MERA tensor network | **STRUCTURAL** | `HierarchicalRouter` is a 2-level MERA |
| V1.8 | Hyperbolic geometry / horospheres | **THEORY** | — |
| V1.9 | Information geometry / Fisher metric | **THEORY** | — |
| V1.10 | Optimal Transport / Benamou-Brenier | **THEORY** | — |
| V1.11 | Expander graphs | **STRUCTURAL** | Workspace admission + manifold reads have expander-like mixing |
| V1.12 | Category theory / adjoint functors | **THEORY** | — |

## Volume 2 (physics / CS)

| # | Framework | Status | Where |
|---|---|:-:|---|
| V2.13 | Kalman filter / information filter | **USED** | `Agent.bayesian_update` is a precision-weighted Kalman update |
| V2.14 | Communication complexity / log rank | **STRUCTURAL** | Workspace = O(log N) matches Ω(log N) lower bound |
| V2.15 | Compressed sensing | **THEORY** | — |
| V2.16 | Thermodynamics / Landauer | **THEORY** | — |
| V2.17 | Quantum scrambling / OTOC | **THEORY** | — |
| V2.18 | Holographic entropy / Ryu-Takayanagi | **BUILT** | `HolographicProtocol` exists, not LLM-tested |
| V2.19 | Gricean pragmatics | **USED** | LLM prompts follow maxims of quantity / relevance |
| V2.20 | CRDTs (semilattice merge) | **USED** | `StigmergyBin.merge` is literally a G-set CRDT |
| V2.21 | Process calculi / bisimulation | **THEORY** | — |
| V2.22 | Turing patterns / reaction-diffusion | **THEORY** | — |

## Volume 3 (signals / economics)

| # | Framework | Status | Where |
|---|---|:-:|---|
| V3.23 | Multiresolution analysis / wavelets | **STRUCTURAL** | Scale projection mimics Mallat MRA |
| V3.24 | Mechanism design / VCG | **BUILT** | `MarketWorkspace` implements VCG pricing, not yet wired |
| V3.25 | Random matrix theory | **THEORY** | — |
| V3.26 | Turbulence / Kolmogorov cascade | **THEORY** | — |
| V3.27 | Chomsky hierarchy | **THEORY** | — |
| V3.28 | Cognitive load / chunking | **STRUCTURAL** | Workspace size ≈ Miller's 7 by default |
| V3.29 | Algebraic coding / polar codes | **THEORY** | — |
| V3.30 | Topological data analysis / persistent homology | **THEORY** | — |
| V3.31 | Interactive information theory | **THEORY** | — |
| V3.32 | Market microstructure / Kyle | **THEORY** | — |

## Volume 4 (stochastic / learning)

| # | Framework | Status | Where |
|---|---|:-:|---|
| V4.33 | Queueing theory | **THEORY** | — |
| V4.34 | Markov chain mixing | **THEORY** | — |
| V4.35 | Convex optimization / KKT | **STRUCTURAL** | Bayesian update is a convex optimum |
| V4.36 | Spin glass / RSB | **THEORY** | — |
| V4.37 | Percolation / critical phenomena | **THEORY** | — |
| V4.38 | PAC learning / VC | **THEORY** | — |
| V4.39 | Online learning / no-regret | **THEORY** | — |
| V4.40 | POMDPs / active inference | **STRUCTURAL** | `AdaptiveScale` is an active-inference controller |
| V4.41 | Non-eq stat mech / MEPP | **THEORY** | — |
| V4.42 | p-adic / ultrametric | **THEORY** | — |

## Volume 5 (advanced / abstract)

Almost entirely **THEORY** — SDEs, DMFT, tropical geometry, Anderson
localization, MaxCal, K-theory, spectral sequences, REM, reversible
computing, tensor networks. 1 exception:

| V5.52 | Tensor networks (MPS/TTN/MERA) | **STRUCTURAL** | `HierarchicalRouter` is a tree tensor network |

## Volume 6 (breadth)

| # | Framework | Status | Where |
|---|---|:-:|---|
| V6.53 | QFT / Feynman diagrams | **THEORY** | — |
| V6.54 | Wright-Fisher / fitness landscapes | **THEORY** | — |
| V6.55 | Matroid theory / submodularity | **STRUCTURAL** | Top-k workspace ≈ greedy submodular maximization |
| V6.56 | Integrated Information Theory (Φ) | **THEORY** | — |
| V6.57 | Graph Neural Networks | **STRUCTURAL** | Message-passing is our update shape |
| V6.58 | Kernel methods / RKHS | **STRUCTURAL** | Embeddings + cosine similarity |
| V6.59 | Morse theory / critical points | **THEORY** | — |
| V6.60 | Operads / composition | **STRUCTURAL** | `HierarchicalRouter` composes sub-routers |
| V6.61 | General equilibrium / Arrow-Debreu | **BUILT** | `MarketWorkspace` is a market-clearing mechanism |
| V6.62 | Chaos theory / edge-of-chaos | **THEORY** | — |

## Volume 7 (latest)

Almost entirely **THEORY** — gauge theory, rough paths, SPDEs, moduli /
Bridgeland, CFT, TQFT, Kolmogorov complexity, solitons, spin networks,
geometric Langlands.

---

## Summary by status

| Status | Count | % |
|---|---:|---:|
| USED (direct implementation + tested) | 6 | 8 % |
| STRUCTURAL (informs design) | 13 | 18 % |
| BUILT (code exists, not LLM-tested) | 3 | 4 % |
| THEORY (math doc only) | 50 | 69 % |

**About 30 % of the theory is in the code. 70 % is still on paper.**

This is normal for a research program. The core insight — *the O(log N)
bound emerges from many independent derivations* — is the value of the
survey. Each framework is a different proof that the bound is right, not
a different thing to implement.

The **most impactful theory we haven't yet wired up**:

1. **Sheaf cohomology (V1.5)** for quantifying *global consistency* of
   distributed beliefs — would give a computable health metric for the
   team.
2. **Mechanism design / VCG (V3.24)** is built (`MarketWorkspace`) but
   never wired into an experiment. An obvious Phase-10 task.
3. **TDA persistent homology (V3.30)** would let us measure task
   complexity directly from the corpus, guiding workspace sizing.
4. **Optimal transport (V1.10)** for measuring how far consensus has
   moved across rounds — natural metric we don't have today.

---

## What Phase 9 adds

- **Longer-than-context document**: pushes past any single agent's
  capacity → truly-distributed task.
- **Multi-role team**: researchers + analysts + strategy builders +
  portfolio manager. Uses the `HierarchicalRouter` / role distinction
  genuinely, not just persona cycling.
- **Rate-distortion budgeting**: per-role compression budgets set by
  explicit distortion constraints (information bottleneck in practice).

Phase-10 candidates from this audit:
- Wire `MarketWorkspace` into a real experiment (V3.24 + V6.61).
- Add sheaf-H¹ diagnostic for team coherence (V1.5).
- Add optimal-transport distance between rounds' consensus (V1.10).

## What Phase 19 adds

A direction shift, not just another framework wired up. Phases 1–18 reduced
context via *lossy* operators (PCA projection, scale renormalisation,
summarisation in Phase 8/9, surprise-suppression in Phase 14+). Phase 19
introduces the **lossless context substrate**: a content-addressed exact
artifact store with a provenance DAG, served via an embedding-indexed
retrieval layer.

What this changes in this audit:

| Framework | Status | Where |
|---|:-:|---|
| **F-19a Content-addressed Merkle DAG** | **USED** | `core/merkle_dag.py` was previously BUILT-not-used; Phase 19 makes it the byte store of the substrate (`core/context_ledger.py`). |
| **F-19b Vector retrieval (HNSW family)** | **USED** | `core/retrieval_store.py` was previously BUILT; Phase 19 wires it into the substrate as the index. |
| **F-19c Provenance DAG (Git-style)** | **USED** | `core/context_ledger.ContextLedger.parents/children/lineage` give cryptographically-verifiable artifact lineage. |
| **F-19d Bounded-context retrieval** | **USED** | `core/bounded_worker.py` — replaces `out[:400]`-style truncation with explicit handle + fetch. |

**Architectural consequence.** Lossy operators (F1, F3, V3.23 wavelets,
V5.52 tensor networks) are now correctly classified as **routing /
latent-state machinery**, not as memory. The substrate guarantees exact
recoverability of every artifact; the lossy operators decide *who talks to
whom* under O(log N) bandwidth. This separation was implicit before and is
now explicit in the codebase.

Phase-20 candidates from this audit:
- ~~Multi-hop retrieval inside `BoundedRetrievalWorker`~~ — done in Phase 20.
- Provenance-aware Phase-18 trigger: fire on cited-source disagreement, not
  just content disagreement.
- Wire `MarketWorkspace` (V3.24 + V6.61) — still on the list from Phase 10.
- Add sheaf-H¹ diagnostic for team coherence (V1.5).

## What Phase 20 adds

Two retrieval-side frameworks become USED, and two new theorems join the
formal stack alongside the 15 in `PROOFS.md`:

| Framework | Status | Where |
|---|:-:|---|
| **F-20a Okapi BM25 / inverted-index search** | **USED** | `core/lexical_index.py` — pure-Python BM25 with Unicode-clean tokenisation, accent folding, RRF helper. |
| **F-20b Reciprocal Rank Fusion (Cormack et al. 2009)** | **USED** | `core/lexical_index.reciprocal_rank_fusion`, wired into `core/context_ledger.search(mode="hybrid")`. |
| **F-20c Multi-hop structural expansion** | **USED** | `core/bounded_worker.extract_references` + `BoundedRetrievalWorker(max_hops>1)` — hop-2 query is a deterministic regex over hop-1 fetched bytes, NOT an LLM-planned reformulation. Preserves substrate losslessness across hops. |

**New formal claims** (full statements + proofs in
`vision_mvp/RESULTS_PHASE20.md` § A.4):

- **Theorem P20-1 (Substrate-Reasoning Decomposition).** End-to-end
  accuracy factors as `Pr[A] ≥ Pr[F] · Pr[A | F]`. The substrate
  controls `Pr[F]` (fact-in-input rate); the model controls
  `Pr[A | F]` (extraction accuracy given fact). Verified empirically
  by the new `failure_class` per-question label which separates
  `retrieval_miss` from `llm_error`.
- **Theorem P20-2 (Hybrid retrieval dominates dense for rare-token
  queries).** For any query containing a unique literal token, BM25
  recall is 1, and RRF fusion preserves it. Verified by
  `tests/test_hybrid_search.py::TestHybridDominatesOnRareTokenQuery`
  and by the +24-point accuracy gain of hybrid over dense on the
  Phase-20 mock benchmark.
- **Theorem P20-3 (Multi-hop accuracy under deterministic refs).**
  When intermediate artifacts contain regex-parseable references to
  the next hop, multi-hop retrieval recall lower-bounds at the
  hop-1 recall. Verified empirically: 100 % retrieval recall on the
  multi-hop benchmark (3 repeats, σ = 0).

**Architectural consequence.** The substrate now has FOUR distinct
layers (routing / trigger / exact memory / retrieval), each with its
own loss profile. Lossy operators (PCA, scale projection, sparse
routing, surprise filtering) belong to *routing* and *trigger*. Lossless
operators (Merkle DAG, content-addressed store) belong to *exact
memory*. The retrieval layer is lossy in *ranking* but never in
*content* — a critical distinction for measurability.

Phase-21 candidates from this audit:
- ~~Aggregation queries — the substrate is for fact retrieval, not
  fact counting.~~ — done in Phase 21 via `core/exact_ops` +
  `core/query_planner`.
- Provenance-aware Phase-18 trigger (still pending from Phase 20).
- Cross-encoder re-ranking for adversarial / synonymy queries.
- Tighter per-hop budget policy (the 21-point gap between recall and
  exact-correct under multi-hop is a budget-allocation problem).

## What Phase 21 adds

Two computation-side frameworks become USED, and four new claims
join the formal stack:

| Framework | Status | Where |
|---|:-:|---|
| **F-21a Typed operator pipeline / relational algebra** | **USED** | `core/exact_ops.py` — Filter / Extract / Count / Sum / MinMax / GroupCount / List / Join with typed `Stage` payloads. Closest classical analog: a tiny relational algebra over the substrate's typed metadata + body regex extraction. |
| **F-21b Pattern-driven NL → plan dispatcher** | **USED** | `core/query_planner.py` — 7 regex/keyword patterns (count_distinct_field, count_filter, list_filter, top_group, sum_field, min_max_field, join_via_ref). Pure-Python; never invokes the LLM at planning time. |

**New formal claims** (full statements + proofs in
`vision_mvp/RESULTS_PHASE21.md` § A.4):

- **Theorem P21-1 (Substrate-Reduction-Reasoning Decomposition).**
  When the planner matches a query, end-to-end accuracy depends only
  on retrieval coverage and template rendering — not on LLM
  reasoning. The deterministic operator chain has $\Pr[D \mid R] = 1$.
- **Theorem P21-2 (Exact Aggregation Lower Bound).** Any system that
  correctly answers an aggregation query over $X' \subseteq X$ must
  evaluate the relevant property on every $x \in X'$. Top-k retrieval
  with $k < |X'|$ cannot answer reliably.
- **Conjecture P21-3 (Streaming Aggregation Preserves Exactness).**
  Associative reductions can be chunked across rounds without loss —
  matters for a future budget-bounded variant of operator execution.
- **Conjecture P21-4 (Compositional Plan Exactness).** Operator chains
  composed of exact operators are exact in their supported domain;
  proof reduces to per-operator unit tests.

**Architectural consequence — the substrate is now FIVE layers:**
routing / trigger / exact memory / retrieval / **computation**.
Lossy operators stay confined to routing and trigger; the bottom
three layers (memory, retrieval, computation) are lossless in
content. The decomposition `Pr[A] = Pr[R] · Pr[D|R] · Pr[A|R,D]`
is independently measurable from the new `failure_class` field
(`ok` / `retrieval_miss` / `reduction_error` / `llm_error`).

Phase-22 candidates from this audit:
- ~~Real codebase-scale corpus benchmark~~ — done in Phase 22 via
  `core/code_index.py` + `tasks/python_corpus.py` +
  `experiments/phase22_codebase.py`.
- Tighter per-hop budget policy (Phase-20 carryover).
- Compositional planner (meta-grammar over operators).
- LLM-assisted plan synthesis with provenance.

## What Phase 22 adds

One ingestion-side framework, one render-mode framework, and three
new theorems / conjectures join the formal stack.

| Framework | Status | Where |
|---|:-:|---|
| **F-22a Static AST analysis as typed-metadata extractor** | **USED** | `core/code_index.py` — `ast.parse` → `CodeMetadata` → ledger metadata. Conservative, deterministic, idempotent. |
| **F-22b Direct-exact render mode** | **USED** | `experiments/phase22_codebase.run_direct_exact` — when the planner produces a result, return its render output verbatim with provenance. Zero LLM call, zero prompt chars. |

**New formal claims** (full statements + proofs in
`vision_mvp/RESULTS_PHASE22.md` § A.4):

- **Theorem P22-1 (Parse-Plan-Render Decomposition).** End-to-end
  accuracy factors as `Pr[A] = Pr[parse] · Pr[plan|parse] ·
  Pr[render|plan]`. Direct-exact path makes `Pr[render|plan] = 1`
  by construction.
- **Theorem P22-2 (Static-Operator Bypasses LLM).** For questions the
  code planner matches on typed metadata, the LLM is unnecessary on
  the answer path. Verified: **0 LLM calls** and **0 prompt chars**
  on the Phase-22 benchmark for 7/7 questions.
- **Theorem P22-3 (Typed Metadata Reduces Retrieval Burden).** Adding
  a typed metadata field with a planner pattern monotonically
  reduces expected prompt size weighted by the matched fraction.
- **Conjecture P22-4 (External Validity).** The Phase-19/20/21
  guarantees persist when the corpus is real Python source ingested
  via AST. Verified for `vision_mvp/core/` (24 files, 39 functions,
  41 classes, 92 imports); broader claim (third-party libraries,
  10× scale) is OQ-22a.

**Architectural consequence.** The substrate is now FIVE layers + a
render mode (`wrap | direct`). On the direct path, the LLM is
removed from the answer flow entirely for matched questions, and the
**five-way error decomposition** (`ok / retrieval_miss /
planning_error / render_error / llm_error`) gives a structurally
zero `render_error` column. Each error category maps to one specific
substrate layer; each layer is now an independently optimisable
knob.

Phase-23 candidates from this audit:
- Larger external codebases (NumPy, Click, etc. at 10× scale).
- Cross-language ingesters (TypeScript / Go / Rust ASTs).
- LLM-assisted plan synthesis (emit JSON plans, executor enforces).
- Open-vocabulary code questions (likely require small specialised
  classifiers rather than expanded patterns).

## What Phase 23 adds

External validity evidence for the exact substrate: the Phase-22
direct-exact result (1 real codebase) is extended to 6 real Python
codebases spanning five distinct families (research code, test
suite, production third-party CLI framework, stdlib, utility
scripts), with coverage accounting shipped alongside every
benchmark.

| Framework | Status | Where |
|---|:-:|---|
| **F-23a Coverage accounting for ingestion** | **USED** | `core/code_index.IngestionStats` — per-run dataclass populated by `CodeIndexer.index_into`; reports `files_parsed_ok` / `files_trivial` / `files_syntax_error` / `files_oversize_skipped` / `parse_coverage` / `metadata_completeness`. |
| **F-23b Multi-corpus registry** | **USED** | `tasks/corpus_registry.{CorpusSpec,CorpusRegistry,default_phase23_registry}` — reusable loader that lets a benchmark declare "these N local Python directories" and get back built `PythonCorpus` instances + coverage stats, no network / no PyPI / deterministic ordering. |

**New formal claims** (full statements + proofs in
`vision_mvp/RESULTS_PHASE23.md` § A.3):

- **Theorem P23-1 (External validity of direct-exact under parser
  coverage).** If `parse_coverage = 1` on corpus $C$ and the
  planner's operator semantics align with the corpus's gold
  computation, then on any planner-matched question the direct-exact
  path's accuracy is exactly 1 with 0 LLM calls and 0 prompt chars.
  Verified on 6 corpora (65 / 65 exact-correct on mock).
- **Theorem P23-2 (Monotone coverage under planner / metadata
  expansion).** Adding a pattern to the planner or a field to the
  metadata schema can only increase the planner-match fraction on
  any given corpus. Verified by construction — Phase-23 patterns
  are additive and preserve Phase-22 precedence.
- **Theorem P23-3 (Retrieval inadequacy for global aggregation is
  corpus-invariant).** For aggregation queries whose support
  $\lvert\mathcal{X}'\rvert > k$, top-$k$ retrieval cannot answer
  reliably on *any* corpus. Verified empirically: mean retrieval
  accuracy 19.7 % across 6 corpora, σ = 17.6 — performance tracks
  coincidental top-k support coverage, not query family.
- **Conjecture P23-4 (Ingestion cost near-linear in source bytes).**
  Empirically verified on the scaling sweep: 0.12 s → 1.28 s
  ingestion time across a 10× range of file count on
  `vision_mvp/core`.

**Architectural consequence.** The ingestion layer now reports
its own coverage for every run, so `retrieval_miss` on the
retrieval path and `planning_error` on the planner path can be
cross-referenced against per-corpus `parse_coverage` and
`metadata_completeness`. A benchmark failure now has a structural
attribution across **six** independently-measurable fields:
parse-coverage × metadata-completeness × planner-match ×
retrieval-recall × operator-correctness × LLM-correctness. Each
field maps to one specific substrate layer.

Phase-24 candidates from this audit:
- 10× scale corpora (`numpy`, `scipy`) to verify Conjecture P23-4
  at larger sizes.
- Cross-language ingesters (TypeScript, Go) to separate Python-AST
  stability from the substrate guarantee.
- Open-vocabulary code queries via regex-over-source (deterministic,
  no LLM in inner loop) — the cleanest way to extend coverage
  without breaching losslessness.

## What Phase 24 adds

One analysis-side framework becomes USED, and four new theorems /
conjectures join the formal stack.

| Framework | Status | Where |
|---|:-:|---|
| **F-24a Conservative static semantic analysis (sound over-approximation)** | **USED** | `core/code_semantics.py` — six intraprocedural predicates (`may_raise`, `is_recursive`, `may_write_global`, `calls_subprocess`/`filesystem`/`network`) + union (`calls_external_io`). Soundness-over-precision: sound for all positive flags; false-positives are the intended cost. |
| **F-24b Parse-Analyze-Plan-Render decomposition** | **USED** | `core/code_index.extract_metadata` now inserts an analyze stage between parse and plan; direct-exact's exactness factors through the new link. Documented in `RESULTS_PHASE24.md` § A.2. |

**New formal claims** (full statements + proofs in
`vision_mvp/RESULTS_PHASE24.md` § A.4):

- **Theorem P24-1 (Parse-Analyze-Plan-Render Decomposition).** For
  any semantic question on a corpus $C$, end-to-end accuracy factors
  as $\Pr[\text{parse}] \cdot \Pr[\text{analyze}|\text{parse}] \cdot
  \Pr[\text{plan}|\text{analyze}] \cdot \Pr[\text{render}|\text{plan}]$;
  when the planner matches and the gold is defined via the same
  analyzer, all four factors are 1 on the direct-exact path.
  Verified: 44 / 44 correct across six corpora.
- **Theorem P24-2 (Conservative analyzer expands direct-exact slice).**
  Adding a sound predicate + pattern monotonically increases the
  planner-match fraction on any corpus with non-empty support. Strict
  corollary: Phase-24 adds six predicates, every one of them has
  non-empty support on at least one Phase-23 corpus, so coverage is
  strictly wider than Phase 23.
- **Theorem P24-3 (Soundness-precision tradeoff is explicit).** The
  analyzer is designed for false-negative = 0 under documented
  assumptions. Analyzer-gold direct-exact measures zero error
  regardless of the false-positive rate; a runtime-truth variant
  would measure the false-positive rate directly (OQ-24a).
- **Conjecture P24-4 (Retrieval cannot answer semantic code
  questions reliably).** Top-k retrieval scores mean 49.6 % on the
  Phase-24 semantic battery across six corpora (σ = 15.8) with every
  failure attributed to `retrieval_miss` — the semantic gold is a
  summary statistic that does not render as a literal token in source
  bytes, so no amount of LLM reasoning recovers it.

**Architectural consequence.** The substrate's exact slice now
covers **three classes** of code questions:

1. Exact structural (Phase 22/23): deterministic from the AST.
2. Conservative static-semantic (Phase 24): deterministic from the
   AST under explicitly documented assumptions (sound, intraprocedural).
3. Undecidable / open-vocabulary / runtime-only: explicitly out of
   scope for the exact slice; falls through to retrieval + LLM.

The five-way error decomposition (`ok / retrieval_miss /
planning_error / render_error / llm_error`) still applies unchanged
— the Phase-24 benchmark reports zero `planning_error` and zero
`render_error` across 44 / 44 direct-exact questions, with every
retrieval failure cleanly attributed to `retrieval_miss`.

Phase-25 candidates from this audit:
- Runtime-truth calibration of the conservative analyzer
  (fuzz-validated false-positive rates on `may_raise`).
- Regex-over-source fallback for open-vocabulary code queries —
  still deterministic, extends slice without breaching losslessness.
- Cross-language ingesters with analogous semantic predicates
  (TypeScript via `ts-morph`, Go via `go/parser`).
- LLM-assisted plan synthesis over typed semantic operators.
- ~~Interprocedural extension of the semantic analyzer
  (whole-program call graph + cycle detection).~~ — done in Phase 25.

## What Phase 25 adds

One analysis-side framework becomes USED, and four new theorems /
conjectures join the formal stack.

| Framework | Status | Where |
|---|:-:|---|
| **F-25a Interprocedural conservative semantic analysis over a local call graph + monotone fixed-point propagation** | **USED** | `core/code_interproc.py` — `build_call_graph` + worklist `propagate_effect` + iterative Tarjan SCC. Six trans-predicates (`trans_may_raise`, `trans_may_write_global`, `trans_calls_subprocess`/`filesystem`/`network`, `trans_calls_external_io`) + `participates_in_cycle` + `has_unresolved_callees`. Soundness over precision under the "resolved-only" convention. |
| **F-25b Analyze-propagate-plan-render decomposition** | **USED** | `core/code_index._patch_with_interproc` — extends the Phase-24 decomposition with a `propagate` stage between `analyze` and `plan`. Direct-exact's exactness factors through the new link. |

**New formal claims** (full statements + proofs in
`vision_mvp/RESULTS_PHASE25.md` § A.4):

- **Theorem P25-1 (Analyse-Propagate-Plan-Render Decomposition).**
  For any interprocedural-semantic question, end-to-end accuracy
  factors as $\Pr[\text{parse}] \cdot \Pr[\text{analyze}|\text{parse}]
  \cdot \Pr[\text{propagate}|\text{analyze}] \cdot \Pr[\text{plan}|
  \text{propagate}] \cdot \Pr[\text{render}|\text{plan}]$; when the
  planner matches and the gold is defined via the same propagation,
  all five factors are 1 on the direct-exact path. Verified: 50 /
  50 correct across six corpora.
- **Theorem P25-2 (Monotone effect propagation reaches a least fixed
  point on a finite call graph).** The OR-propagation operator is
  monotone; Kleene iteration on the finite $\{\text{F, T}\}^V$
  lattice converges in ≤ |V| steps. The worklist implementation is
  O(|V| + |E|) per predicate, idempotent, and deterministic.
- **Theorem P25-3 (Interprocedural analysis strictly expands the
  direct-exact slice).** Adding trans-predicates and patterns can
  only increase planner coverage; strict increase on any corpus with
  at least one non-trivial resolved call edge. Verified: +6 new
  matched questions on 6 corpora; per-predicate flag counts widen
  by up to +113 (`may_raise` sum across 6 corpora).
- **Theorem P25-4 (Recursion-cycle detection is exact on the resolved
  call graph).** Tarjan SCC is exact in linear time; the rule
  "self-loop OR non-trivial SCC" captures both self-recursion and
  mutual recursion. Verified: 19-function SCC in `vision-core`
  detected; all 2/2 cycle-participation questions answered 100 %
  direct-exact vs 0 / 2 retrieval.
- **Conjecture P25-5 (Unresolved-bias towards soundness).** The
  "resolved-only" convention is sound for the resolved subgraph but
  not runtime-sound across unresolved external calls. The
  `has_unresolved_callees` transparency flag surfaces the gap so
  callers can widen trans-flags if they want runtime-bounded
  answers.

**Architectural consequence.** The substrate's exact slice now
covers **four classes** of code questions:

1. Exact structural (Phase 22/23): deterministic from the AST.
2. Conservative *intraprocedural* semantic (Phase 24):
   deterministic from the AST under documented assumptions.
3. Conservative *interprocedural* semantic (Phase 25):
   deterministic from the AST + local-call-graph propagation under
   documented assumptions (resolved-only, sound over the resolved
   subgraph).
4. Undecidable / open-vocabulary / runtime-only: explicitly out of
   scope; falls through to retrieval + LLM.

The five-way error decomposition still applies unchanged — the
Phase-25 benchmark reports zero `planning_error`, zero
`render_error`, zero `llm_error` across 50 / 50 direct-exact
questions, with every retrieval failure cleanly attributed to
`retrieval_miss`.

Phase-26 candidates from this audit:
- Runtime-truth calibration of the interprocedural analyzer
  (fuzz-validated false-positive rate on `click`'s 96
  `trans_may_raise` functions).
- "Trans-widen" opt-in: OR trans-flags with
  `has_unresolved_callees` for a runtime-bounded answer.
- Cross-language interprocedural analysis (TypeScript via
  `ts-morph` + analogous `build_call_graph`).
- Alias tracking through local variables (e.g. `f =
  subprocess.run; f(cmd)`).
- Relative-import resolution with package-context plumbing.
- LLM-assisted plan synthesis over typed interprocedural
  operators.

## What Phase 26 adds

One runtime-observation framework becomes USED, plus four new
theorems / conjectures joining the formal stack. **Phase 26 is not
another static slice — it is a separate validation axis.**

| Framework | Status | Where |
|---|:-:|---|
| **F-26a Instrumented runtime-effect observation (monkeypatch + sandbox + settrace)** | **USED** | `core/code_runtime_calibration.py` — six per-predicate probes (`probe_may_raise`, `probe_may_write_global`, `probe_calls_subprocess`/`filesystem`/`network`, `probe_participates_in_cycle`). Sentinels (`_SubprocessAttempted`, `_NetworkAttempted`) neuter effects without escaping; `sys.settrace` detects re-entry. |
| **F-26b Executable snippet corpus with author-declared ground truth** | **USED** | `tasks/executable_snippets.py` — 21 snippets across 8 families (`negative`, `direct`, `wrapper`, `chain`, `cycle`, `guarded`, `dead`, `hidden`). Per-snippet ground-truth dicts serve as probe-correctness anchors, NOT analyzer-correctness anchors. |
| **F-26c Three-axis calibration (analyzer / runtime / direct-exact planner)** | **USED** | `experiments/phase26_runtime_calibration.py` — per-predicate FP / FN / agreement matrix, per-family breakdown, planner-to-analyzer round-trip, repeated-run variance via `--seeds`. |

**New formal claims** (full statements + proofs in
`vision_mvp/RESULTS_PHASE26.md` § A.4):

- **Theorem P26-1 (Axis separation — analyzer-gold ≠ runtime-
  truth).** Analyzer and runtime are independent truth functions.
  Witnesses: `S_DEAD_RAISE` (FP) for `may_raise`;
  `S_HIDDEN_SUBPROCESS_VIA_EVAL` / `S_HIDDEN_FILESYSTEM_VIA_GETATTR`
  (FN) for `calls_*`.
- **Theorem P26-2 (Runtime observation is a sound lower bound).**
  Observing an effect proves the effect is reachable; absence of
  observation does not prove absence. Corollary P26-2a.
- **Theorem P26-3 (Conservative analyzers should minimise false
  negatives; false positives are characterised, not eliminated).**
  Per-predicate calibration is fp-rate × fn-rate × out-of-scope
  taxonomy, not a single accuracy number. On our corpus:
  `pooled FN = 2`, `pooled FP = 1` across 126 measurements.
- **Theorem P26-4 (Runtime validation separates planner
  soundness from analyzer soundness).** The planner ↔ analyzer
  round-trip is a substrate claim (100 % on the snippet corpus);
  the analyzer ↔ runtime comparison is a separate claim. Phase
  22's `render_error = 0` guarantee survives regardless of
  analyzer calibration.
- **Conjecture P26-5 (Analyzer FN rate tracks the pre-documented
  boundary list).** All FN observations land on snippets built to
  exercise the boundaries Phase 24 explicitly enumerated
  (`eval`, reflection); no FN appeared in the `direct` /
  `wrapper` / `chain` / `cycle` families.

**Architectural consequence.** The substrate's exact slice
now has a SECOND validation layer that measures how the
conservative analyzer's conservatism compares to runtime behaviour:

1. Exact structural (Phase 22/23) — deterministic from the AST.
2. Conservative intraprocedural semantic (Phase 24) —
   deterministic from the AST under documented assumptions.
3. Conservative interprocedural semantic (Phase 25) —
   deterministic from the AST + local-call-graph propagation.
4. **Runtime-calibrated conservative semantic (Phase 26) —
   analyzer flags validated against instrumented runtime
   observation on an executable snippet corpus, with per-
   predicate FP / FN / agreement metrics and documented
   divergence points.**
5. Undecidable / open-vocabulary / runtime-only — out of scope.

The five-way error decomposition (`ok / retrieval_miss /
planning_error / render_error / llm_error`) is unchanged. Phase
26 adds a separate, off-path observation stream that can be
queried independently for soundness audits without affecting the
direct-exact substrate guarantee.

Phase-27 candidates from this audit:

- Corpus-scale runtime calibration (OQ-26a) — an invocation
  protocol for arbitrary library APIs so runtime truth can be
  measured on `click` / `json` at function granularity.
- Coverage-guided fuzz (OQ-26b) — replace the pool-based sampler
  with Hypothesis / Atheris.
- Lightweight CFG pass (OQ-26c) — trim the dead-code `may_raise`
  false positive.
- Alias tracking (OQ-26d) — subsumes `f = subprocess.run; f(cmd)`.
- Environment-dependent effect probing (OQ-26e).
- Concurrency probing (OQ-26f).
- Cross-language runtime calibration (OQ-26g).
- Runtime-aware conservative analyser (OQ-26h) — a hybrid that
  refines precision using runtime witnesses while preserving
  soundness on the corpus slice.

## What Phase 27 adds

Three runtime-observation frameworks become USED, plus five
new theorems / conjectures joining the formal stack. **Phase
27 is a scale shift, not a new substrate layer — it pushes the
Phase-26 runtime observer from a 21-snippet curated corpus to
real corpus functions under an explicit safe-invocation
protocol.**

| Framework | Status | Where |
|---|:-:|---|
| **F-27a Safe-invocation-recipe protocol (AST classifier + signature-driven auto-derivation + curated registry)** | **USED** | `core/code_corpus_runtime.py` — `CorpusFunctionCandidate` classifier (11 status values), `InvocationRecipe` (`no_args` / `typed` / `curated`), `SafeRecipeRegistry`, `typed_recipe` drawing from a `_SUPPORTED_RUNTIME_TYPES` whitelist and `_POOL_BY_TYPE` fuzz pool. |
| **F-27b Entry-detection + wall-clock budget probing (``sys.settrace`` over target code object + per-line budget check raising ``_BudgetExceeded`` sentinel)** | **USED** | `core/code_corpus_runtime._entry_and_budget_tracer` — composes with all Phase-26 sandbox recorders; ``_BudgetExceeded`` inherits ``_ProbeSentinel`` so `_call_safely` swallows it without contaminating `may_raise` observations. |
| **F-27c Witness-availability coverage accounting** | **USED** | `core/code_corpus_runtime.CoverageAccount` — per-status buckets (`ready_no_args`, `ready_typed`, `ready_curated`, seven `unsupported_*` bins) + `ready_fraction`, `calibrated_fraction`. Coverage is reported alongside per-predicate FP/FN as a first-class metric. |

**New formal claims** (full statements + proofs in
`vision_mvp/RESULTS_PHASE27.md` § A.4):

- **Theorem P27-1 (Corpus-scale runtime calibration is a partial
  observer over the semantic slice).** $F_R \subseteq F_A \subseteq F$
  in general strictly. Witness-availability gap is structural, not
  incidental.
- **Theorem P27-2 (Runtime-calibration coverage is witness-
  availability-bounded, not planner-exactness-bounded).** The
  planner round-trip remains at 100 % regardless of Phase-27
  coverage; coverage depends on recipe strategy + curated registry
  size, not on analyzer / planner correctness.
- **Theorem P27-3 (Sandboxed corpus probes preserve runtime-
  observation soundness, conditional on entry detection).**
  Observations with `entered=True` are sound witnesses; observations
  with `entered=False` are excluded from FP/FN and recorded in
  coverage only.
- **Conjecture P27-4 (Corpus-scale FN concentration on pre-
  documented boundaries).** FN divergences in corpus-scale
  calibration cluster on the same Phase-24 boundary classes
  Phase 26 surfaced (reflection, eval, alias); anything else is a
  first-class research signal for boundary-class expansion.
- **Conjecture P27-5 (Coverage-precision tradeoff for the
  invocation protocol).** Increasing recipe aggressiveness raises
  $|F_R|$ monotonically but non-monotonically affects FN rate —
  more exotic recipes trip recipe-artefact entries that aren't
  part of the target's semantic behaviour.

**Architectural consequence.** The substrate's exact slice now
has FOUR truth statements in play:

1. Analyzer-gold truth (Phase 22/23/24/25) — deterministic from
   AST + local call graph.
2. Snippet-calibrated runtime truth (Phase 26) — instrumented
   probe on 21 curated snippets with author-declared ground
   truth.
3. **Corpus-scale calibrated runtime truth (Phase 27) — same
   instrumentation, real corpus functions, witness-availability
   coverage reported alongside FP/FN.**
4. Direct-exact planner truth — surfaces analyzer-gold answers
   through typed operators. Unchanged.

The five-way error decomposition (`ok / retrieval_miss /
planning_error / render_error / llm_error`) is unchanged. Phase
27 adds a second off-path observation stream that can be queried
independently for corpus-scale soundness audits without affecting
the direct-exact substrate guarantee.

Phase-28 candidates from this audit:

- Method instance auto-constructor (OQ-27b) — lift method
  coverage without handwriting per-method recipes.
- Signature-driven smart recipes (OQ-27c) — use parameter names
  + corpus-internal type hints to improve entered-fraction on
  `ready_typed`.
- Environment-dependent effect probing (OQ-27d).
- Additional corpora (OQ-27f) — extend to `click`, stdlib `json`,
  `vision-tests`, `vision-experiments`.
- Cross-language analogue (OQ-27g).
- Runtime-refined analyzer (OQ-27h) — hybrid that uses Phase-27
  observations to refine Phase-24's FP rate while preserving
  soundness on the corpus slice.

## What Phase 28 adds

One analysis-side framework becomes USED, plus the runtime-
observation framework from Phase 27 gains an explicit/implicit
origin-classification layer. Five new theorems / conjectures
join the formal stack.

| Framework | Status | Where |
|---|:-:|---|
| **F-28a Conservative implicit-raise syntactic predicate (sound-over-precision)** | **USED** | `core/code_semantics._contains_implicit_raise_pattern`, `_analyze_may_raise_implicit`. Pattern list of six risky syntactic forms (division, subscript, risky builtin, attribute-on-parameter) with catch-all try/except escape hatch. Soundness: documented; FN = 1 / 116 runtime-positives on the Phase-28 pooled four-corpus entered slice. |
| **F-28b Exception-origin classification via traceback line-set** | **USED** | `core/code_runtime_calibration._raise_line_numbers`, `_classify_exception_origin`, `probe_may_raise_split`. Partitions caught exceptions into `explicit` / `implicit` buckets by comparing the innermost frame's line number against the target's AST raise-statement line set. Sound lower bound for the explicit bucket. |
| **F-28c Multi-corpus runtime calibration with coverage-as-first-class-variable** | **USED** | `experiments/phase28_multi_corpus_runtime_calibration.py`. Runs the Phase-27 probes across four local corpora in one benchmark; reports per-corpus `ready_fraction` + calibrated fraction + FP/FN alongside pooled aggregates. Coverage spread 2.9 %–80.2 % across corpora demonstrates Theorem P28-2. |

**New formal claims** (full statements + proofs in
`vision_mvp/RESULTS_PHASE28.md` § A.4):

- **Theorem P28-1** — Phase-24 `may_raise` contract is preserved
  byte-for-byte; `may_raise_implicit` is strictly additive.
- **Theorem P28-2** — Multi-corpus witness-availability is
  structural, cross-corpus-variable (28× spread), and
  orthogonal to calibration accuracy.
- **Theorem P28-3** — `may_raise_explicit` calibrates sound
  (FN = 0) and precise (98.7 % pooled agreement).
- **Theorem P28-4** — `may_raise_implicit` is sound by
  construction (FN = 1 / 116); precision is bounded by recipe
  coverage, not by the analyzer contract.
- **Conjecture P28-5** — The Phase-22 `render_error = 0`
  substrate guarantee and Phase-27 planner round-trip are
  preserved across the Phase-28 extension.

**Architectural consequence.** The substrate's exact slice
still has four truth statements (analyzer-gold, snippet-
calibrated runtime, corpus-scale calibrated runtime, direct-
exact planner), but the analyzer-gold axis now carries a
sound-over-precision implicit-raise predicate alongside the
Phase-24 explicit-raise predicate. The five-way error
decomposition is unchanged; the calibration axis gains one
more independently-measurable bucket.

## What Phase 29 adds

Three new research surfaces become USED and eight new
theorems / conjectures join the formal stack. Phase 29 is
the first milestone that couples the programme's two parallel
tracks — the multi-agent routing track (Arcs 1–2) and the
exact-substrate code track (Arcs 3–5) — into one task-scale
measurement, while also lifting the runtime-calibration coverage
axis on the method-dominated corpora.

| Framework | Status | Where |
|---|:-:|---|
| **F-29a Task-scale causal-relevance oracle** | **USED** | `tasks/task_scale_swe` — per-(task, role, event) causal-relevance oracle derived from the Phase-25 analyzer. Builds the naive event stream that a broadcast bus would emit across a five-role team, applies the oracle, and measures delivery under naive / routing / substrate strategies. Unlike the Phase-10 agent-network measurements which operated on numeric / QA tasks, this is the first task-scale bench with an analyzer-derived causal-relevance metric on SWE-style queries. |
| **F-29b Structurally-typed multi-role routing simulator** | **USED** | `tasks/task_scale_swe.deliver` — three synchronous delivery strategies (naive / routing / substrate) with per-role subscription sets and a substrate direct-exact path. The strategy is not an LLM-in-the-loop simulation; it's a typed, deterministic simulator whose numbers are exactly reproducible given the seed. |
| **F-29c Conservative method-instance auto-construction with sandbox-preserved soundness** | **USED** | `core/code_corpus_runtime.analyze_class_construction` + `_try_construct_instance` + `probe_corpus_function` extension. The AST classifier recognises four zero-arg-constructable strategies (inherited / explicit-all-defaults / dataclass-all-defaults / rejected-exception-subclass); the probe constructs the instance under the Phase-26 sandbox + Phase-27 budget tracer. |

**New formal claims** (full statements + proofs in
`vision_mvp/RESULTS_PHASE29.md` § B):

- **Theorem P29-1 (Task-scale causal-relevance is small under
  naive broadcast).** Pooled aggregator-role causal-relevance
  fraction is 4.54 % < 0.50 gate across 80 queries on four
  corpora. CONFIRMED on every corpus independently.
- **Theorem P29-2 (Routing alone does not rescue aggregator
  context).** Role-level Bloom-filter routing leaves the
  aggregator unchanged (13 849 → 13 826 tokens) while reducing
  non-aggregator context 1.3×–1 154×. Type-level filtering
  cannot answer content-level aggregation.
- **Theorem P29-3 (Substrate collapses aggregator context to
  near-zero on matched tasks).** 1 007× reduction at 100 %
  correctness on 76 / 80 matched aggregator tasks.
- **Theorem P29-4 (Task-scale answer correctness is preserved
  by routing and strictly improved by substrate).** 98.75 %
  under naive / routing, 100 % under substrate.
- **Theorem P29-5 (Witness-availability bound is dominated by
  method-instance construction).** Method recipe lifts runtime
  `ready_fraction` on `vision-tests` 2.9 % → 98.8 %; pooled
  entered slice 4.83×; construct-failed < 1 %;
  `may_raise_explicit` FN preserved at 0.
- **Theorem P29-6 (Task-scale causal-relevance and corpus-scale
  runtime calibration live on orthogonal axes).** The two
  Phase-29 measurements depend on disjoint state spaces — event
  stream vs function invocation — so lifting one does not move
  the other.
- **Conjecture P29-7 (Agent-teams generalisation).** The
  task-scale falsifiability gate holds for any multi-agent team
  operating on a task decomposition with *structurally typed*
  per-role concerns; the claim transfers to code-review,
  DevOps, compliance, document summarisation workflows. Full
  SWE-bench test is ROADMAP medium-term scope.
- **Conjecture P29-8 (Information-theoretic floor on content-
  level aggregation).** Any strategy that correctly answers an
  aggregation query over a support set of size |X'| must either
  deliver |X'| matching events to the aggregator or provide a
  deterministic summary via the content-layer. Routing-only
  strategies inherit the Theorem P21-2 lower bound at task
  scale.

**Architectural consequence.** The substrate's exact slice
still has four truth statements, but a **fifth measurement axis**
joins at task scale:

1. Analyzer-gold truth (Phase 22–28) — deterministic from AST +
   call graph.
2. Snippet-calibrated runtime truth (Phase 26).
3. Corpus-scale calibrated runtime truth (Phase 27–28, extended
   Phase 29 method coverage).
4. Direct-exact planner truth — unchanged.
5. **Task-scale causal-relevance truth (Phase 29)** — the
   oracle-derived fraction of a multi-role team's event stream
   that is causally relevant to each role's final answer.

Agent-teams framing consequence: Phase 29 is the first
measurement that ties the programme's *five* truth statements
back to the programme's *top-level thesis about agent teams in
general*. The direct-exact substrate is no longer just "zero LLM
calls on code-battery questions"; it is the only mechanism in
the stack that can collapse the aggregator-role's content-level
context on realistic SWE-style workloads. Routing-by-type is
necessary but not sufficient; substrate direct-exact is the load-
bearing piece at task scale.

## What Phase 30 adds

Two framework pieces become USED (a deterministic substrate-
evaluation harness with an LLM on the answer path; a formal
mapping from Phase-29's causal-relevance fraction to ``T_i*``)
and four new theorems plus two new conjectures join the formal
stack. Phase 30 is the first milestone that *simultaneously*
publishes a theorem and its empirical corroboration on a live
LLM.

| Framework | Status | Where |
|---|:-:|---|
| **F-30a Model-agnostic substrate-evaluation harness (LLM on the answer path)** | **USED** | ``vision_mvp/tasks/swe_loop_harness.py`` — ``run_loop`` takes any ``Callable[[str], str]``; three delivery strategies (naive / routing / substrate_wrap); deterministic task-kind grader; cross-strategy delta reporter. ``vision_mvp/experiments/phase30_llm_swe_benchmark.py`` is the external driver. |
| **F-30b T_i* fixed-point one-step reach in the matched-substrate regime (special case of OQ-1)** | **USED** | Theorem P30-4 in ``vision_mvp/RESULTS_PHASE30.md`` § B.5. Proves existence, uniqueness, and one-step reach of the ``T_i*`` fixed point when the agent's policy is the substrate's direct-exact render on a planner-matched task. First formal result tying OQ-1 to a concrete decidable regime. |

**New formal claims** (full statements + proofs in
``vision_mvp/RESULTS_PHASE30.md`` § B):

- **Theorem P30-1 (Structural-typing irrelevance lower bound).**
  With ``K`` structurally-typed roles, per-role causal-relevance
  fraction ``ρ_i`` is bounded by ``O(|support|/|X|)``; for
  off-role roles, ``ρ_i → 0`` as ``|X| → ∞``.
- **Theorem P30-2 (Substrate bounds |T_i*| to O(1) on matched
  kinds).** The substrate direct-exact delivery has constant
  cardinality independent of ``|X|``. Empirically:
  substrate-token count is 13.75–14.00 across four internal
  corpora and two external corpora, independent of
  ``n_events`` ∈ [60, 2378].
- **Theorem P30-3 (Naive accuracy has a hard ceiling under
  bounded LLM context).** For any model with token budget ``B``,
  ``Pr[correct | naive]`` is strictly less than
  ``Pr[correct | T_i*]`` whenever the naive rendering exceeds
  ``B``. Fano's inequality gives the quantitative gap.
- **Theorem P30-4 (One-step T_i* fixed point in the matched-
  substrate regime).** First formal shape for a special case of
  OQ-1. Proof uses planner idempotence + ledger independence.
- **Conjecture P30-5 (Agent-teams generalisation).** The Phase-
  29/30 bound holds for any structurally-typed task distribution
  with sub-linear predicate support.
- **Conjecture P30-6 (LLM-loop Lipschitz fixed-point).** Under
  Lipschitz continuity of the LLM policy, the ``T_i*`` iteration
  converges on a finite event stream. First concrete
  mathematical shape for OQ-1 under a stochastic answer path.

**Architectural consequence.** Phase 30 does **not** add a new
substrate layer; the five substrate layers + render mode + off-
path observer of Phase 26–29 are unchanged. What Phase 30 adds is
a second *benchmark* surface on top of the substrate — one in
which the answer path goes through a live LLM and the
substrate's load-bearing claim is an *accuracy delta*, not a
*causal-relevance fraction*. The programme now has **five
measurement axes** in play:

1. Analyzer-gold truth (Phase 22–28).
2. Snippet-calibrated runtime truth (Phase 26).
3. Corpus-scale calibrated runtime truth (Phase 27–28, extended
   Phase 29 method coverage).
4. Task-scale causal-relevance truth (Phase 29).
5. **LLM-in-loop answer-correctness truth (Phase 30)** — answer
   accuracy under a real LLM on the aggregator role, as a
   function of delivery strategy.

Agent-teams framing consequence: before Phase 30 the programme's
central claim was stated under an analyzer-derived oracle;
Phase 30 runs a live LLM on external corpora and shows the
substrate is load-bearing at the byte level. The tool/substrate
identity (``swe_loop_harness`` as a reusable substrate-evaluation
surface) is now explicit in the code — the harness accepts any
callable, so a future SWE-bench driver is a drop-in replacement.

## What Phase 31 adds

Two framework pieces become USED (typed handoffs as a
communication-primitive substrate; role-conditioned relevance
as a separate measurement axis) and five theorems plus two
conjectures join the formal stack.

| Framework | Status | Where |
|---|:-:|---|
| **F-31a Typed, hash-chained, content-addressed inter-role handoffs** | **USED** | ``vision_mvp/core/role_handoff.py`` — ``TypedHandoff`` / ``RoleSubscriptionTable`` / bounded ``RoleInbox`` / hash-chained ``HandoffLog`` / ``HandoffRouter``. Communication primitive on the team axis; routes by claim_kind, not by raw event type. |
| **F-31b Role-conditioned causal-relevance factorisation** | **USED** | Theorem P31-1 formalises ``ρ_h`` as a product of subscription-kind header match × content-witness correctness. |

New theorems (P31-1 role-conditioned relevance factorisation;
P31-2 communication-sparsity lower bound Θ(R*) bits; P31-3
bounded active context O(R*·τ) per role; P31-4 correctness
preservation under subscription coverage; P31-5 provable
separation from single-agent long-context). Conjectures C31-6
(role-lattice generalisation), C31-7 (noisy extractor
robustness, later promoted to Theorem P32-2 in the monotone
regime).

## What Phase 32 adds

Cross-domain benchmark (compliance review) + parameterised
extractor-noise module + frontier-model spot check.

| Framework | Status | Where |
|---|:-:|---|
| **F-32a Parameterised extractor noise (Bernoulli i.i.d.)** | **USED** | ``vision_mvp/core/extractor_noise.py`` with five axes (drop / spurious / mislabel / payload-corrupt / seed). |
| **F-32b Cross-domain correctness preservation** | **USED** | Theorem P32-1 + compliance-review benchmark confirm substrate flatness at 171 tokens across k ∈ {6, 20, 60, 120}. |

New theorems P32-1 (cross-domain correctness), P32-2 (two-
regime graceful degradation — promotes C31-7), P32-3 (token
bound preservation under bounded noise).

## What Phase 33 adds

LLM-driven extractor + real-vs-synthetic noise calibration +
third non-code domain (security escalation with max-ordinal
severity decoder).

| Framework | Status | Where |
|---|:-:|---|
| **F-33a LLM-driven extractor (drop-in for regex)** | **USED** | ``vision_mvp/core/llm_extractor.py``; type-filtered against ``known_kinds_by_role``. |
| **F-33b Real-vs-synthetic calibration** | **USED** | ``vision_mvp/core/extractor_calibration.py`` measures (δ̂, ε̂, μ̂, π̂) against gold causal chains. |
| **F-33c Third decoder shape (max-ordinal severity)** | **USED** | ``vision_mvp/tasks/security_escalation.py``. |

New theorems P33-1 (Phase-32 synthetic sweep approximates
real-LLM noise within γ), P33-2 (cross-domain at K=3), P33-3
(max-ordinal two-regime bound). Conjectures C33-3, C33-4.

## What Phase 34 adds

Per-role noise calibration + adversarial extractor wrapper +
honest regex-LLM ensemble.

| Framework | Status | Where |
|---|:-:|---|
| **F-34a Per-role noise calibration** | **USED** | ``core/extractor_calibration.per_role_audit_summary`` + ``PerRoleNoiseConfig`` + ``per_role_noisy_extractor``. Reports per-role (δ̂_k, ε̂_k, μ̂_k, π̂_k) and the limiting role. |
| **F-34b Adversarial extractor wrapper (three target modes)** | **USED** | ``core/extractor_noise.adversarial_extractor`` with load-bearing drop / role silencing / severity escalation. |
| **F-34c Ensemble extractor composition** | **USED** | ``core/ensemble_extractor.UnionExtractor``; content-address dedup. |

New theorems P34-1 (role-limited accuracy under pooled
calibration — AM-GM bound), P34-2 (adversarial-vs-iid
separation at matched budget), P34-3 (ensemble union lower
bound — promotes C33-4 to theorem). Conjectures C34-4
(ensemble-against-adversary), C34-5 (per-role replay is a
tighter predictor than pooled).

## What Phase 35 adds

One framework piece becomes USED (dynamic, bounded
communication primitives above typed handoffs) and four
theorems plus two conjectures join the formal stack. Phase 35
is the first phase in the team-communication arc where the
*communication primitive itself* is dynamic — not just the
extractor or the noise model.

| Framework | Status | Where |
|---|:-:|---|
| **F-35a Escalation thread as a bounded coordination primitive** | **USED** | ``vision_mvp/core/dynamic_comm.py`` — ``EscalationThread`` + ``ThreadReply`` + ``ThreadResolution`` + ``DynamicCommRouter`` + ``ThreadState`` + ``DynamicCommAccount``. Frozen-membership, typed-reply-vocabulary, bounded-budget coordination object. Public output is a single ``CLAIM_THREAD_RESOLUTION`` typed handoff routed through the unchanged Phase-31 ``HandoffRouter``. |
| **F-35b Contested-incident benchmark** | **USED** | ``vision_mvp/tasks/contested_incident.py`` — 6-scenario bank (4 contested + 2 controls) where static typed handoffs cannot recover the answer without coordinating on producer-local evidence. |

**New formal claims** (full statements + proofs in
``vision_mvp/RESULTS_PHASE35.md`` § B):

- **Theorem P35-1 (Expressivity gap: static handoffs cannot
  decode contested scenarios).** There exists a scenario family
  ``Z_contested`` on which any static-priority decoder
  ``D_static : 2^Kinds → Kinds`` is strictly wrong on at least
  one scenario under any total priority ordering, while a
  dynamic-coordination decoder reaches the gold on every
  scenario. The proof is a pigeonhole argument on claim-kind
  orderings; empirical anchor is the 100 % vs 33 % full-
  accuracy separation on the Phase-35 bank.
- **Theorem P35-2 (Bounded active context preserved under
  dynamic coordination).** Peak active context at any role
  ``r`` per round is ``C_0 + R*·τ + T·R_max·W`` — independent
  of |X|. The ``T·R_max·W`` additive term is a task-spec
  constant (T ≤ 1, R_max = 2, W = 12 on the Phase-35 bank).
  Recovers Theorem P31-3 exactly at T = 0.
- **Theorem P35-3 (Correctness under sound producer-local
  causality extraction).** Under a producer-local extractor
  ``ĥ_r`` that is sound (no false-positive
  ``INDEPENDENT_ROOT`` emissions) and does not under-emit on
  the single gold causal claim among the contested candidates,
  dynamic coordination recovers the gold on every contested
  scenario. Proof is the resolution-rule lookup; Phase-34 noise
  wrappers can be composed at the extractor boundary to
  measure how the guarantee degrades (Conjecture C35-7).
- **Theorem P35-4 (No-leak invariant for non-member roles).**
  Non-member roles observe zero thread-internal events; the
  only visible artifact is the single ``CLAIM_THREAD_RESOLUTION``
  handoff conditioned on explicit static subscription. Proof is
  by construction of ``post_reply`` / ``close_thread`` routing.
- **Conjecture C35-5 (Bounded threads ≡ bounded adaptive
  subscriptions).** Bounded threads and bounded adaptive
  subscription-graph edits have identical decoder-correctness
  guarantees; threads strictly dominate on bounded-context
  enforcement (type-level vs runtime-level).
- **Conjecture C35-6 (Dynamic coordination is necessary, not
  only sufficient).** Predicts a ``Z_hard`` scenario family on
  which no finite static subscription graph preserves the
  Phase-31 ``Θ(R*·τ)`` bound while reaching correctness — the
  information-theoretic lower-bound dual of Theorem P35-1.

**Architectural consequence.** Phase 35 adds the first
substrate layer *strictly above* Phase 31's typed-handoff layer
and *strictly below* any unrestricted group-chat layer. The
programme's axis of differentiation from graph/index tools
(§ 1.5 in the master plan) is now two-layered: the
compression-vs-communication distinction (Theorem P31-5) AND
the static-vs-dynamic communication distinction (Theorems
P35-1 / P35-4). Unrestricted chat remains structurally
unnecessary and harmful to the bounded-context thesis.

## What Phase 36 adds

Phase 36 stresses the Phase-35 dynamic-coordination primitive
along three coupled axes and ships four theorems + four
conjectures.

| Result | Status | Module / evidence |
|---|---|---|
| **F-36a Reply-axis noise wrapper** | **USED** | ``vision_mvp/core/reply_noise.py`` — ``ReplyNoiseConfig`` (drop / mislabel), ``AdversarialReplyConfig`` (targeted drop / flip / inject), ``noisy_causality_extractor``, ``adversarial_reply_extractor``, ``ReplyCorruptionReport``. Wraps any Phase-35 producer-local causality extractor at the extractor boundary. |
| **F-36b LLM-driven thread replier** | **USED** | ``vision_mvp/core/llm_thread_replier.py`` — ``LLMThreadReplier`` + ``LLMReplyConfig`` + ``parse_llm_reply_json`` + ``causality_extractor_from_replier`` + ``ScenarioAwareMockReplier`` (in experiments). Drives a narrow LLM call, parses and filters against the Phase-35 reply-kind enum, falls back to UNCERTAIN on parse failure. |
| **F-36c Bounded adaptive-subscription primitive** | **USED** | ``vision_mvp/core/adaptive_sub.py`` — ``AdaptiveEdge`` + ``AdaptiveSubscriptionTable`` + ``AdaptiveSubRouter`` + ``AdaptiveSubAccount`` + ``CLAIM_CAUSALITY_HYPOTHESIS``. Temporary TTL-expiring subscription edits with a hard cap ``max_active_edges``. |
| **F-36d Contested-bank integration for the three new paths** | **USED** | ``vision_mvp/tasks/contested_incident.py`` — ``STRATEGY_ADAPTIVE_SUB`` + ``run_adaptive_sub_coordination`` + pluggable ``causality_extractor`` on ``run_dynamic_coordination`` / ``run_contested_loop``; ``decoder_from_handoffs_phase35`` extended to consume CAUSALITY_HYPOTHESIS handoffs. |

**New theorems.**

- **Theorem P36-1 (Graceful degradation under i.i.d. reply
  noise).** Under Bernoulli drop / mislabel with parameters
  ``(p, q)``, the dynamic-coordination decoder satisfies
  ``Pr[D_dyn = gold] = (1 - p)(1 - q)`` on a two-candidate
  contested scenario with one gold INDEPENDENT_ROOT reply;
  static priority is capped at ``≤ 1/2`` by Theorem P35-1's
  pigeonhole. Dominance persists for ``p + q < 1/2``.
  Proof: single-reply survival probability × complement of
  mislabel; see ``RESULTS_PHASE36.md`` § B.2. Empirical anchor
  in § D.1: dynamic 91.7 % at p=0.25, 66.7 % at p=0.5, 33.3 %
  at p≥0.75.
- **Theorem P36-2 (Adversarial collapse under targeted
  drop_root).** Under ``ν_adv`` with ``target_mode =
  drop_root``, budget ``b = 1``, both ``D_dyn`` and ``D_adp``
  collapse to zero on every two-candidate contested scenario.
  Proof: the deterministic flip converts the single gold
  INDEPENDENT_ROOT reply to UNCERTAIN; resolution returns
  NO_CONSENSUS; decoder falls back to static priority; static
  picks the shadow by construction. Empirical anchor in § D.2.
- **Theorem P36-3 (LLM-replier substitutivity).** If the LLM
  replier emits (a) a parseable JSON line with (b)
  ``reply_kind`` in the Phase-35 enum and (c) the oracle
  causality class on every candidate, then the dynamic
  decoder driven by the LLM replier achieves identical
  accuracy to the deterministic-oracle-driven decoder on the
  Phase-35 bank. Proof: parser-contract + the thread's
  resolution rule is deterministic in the ThreadReply objects.
  Empirical anchor in § D.4: at ``malformed_prob = 0``, LLM
  replier reaches 100 % contested accuracy.
- **Theorem P36-4 (Empirical primitive equivalence on the
  Phase-35 bank).** Across a drop × mislabel × k × seed grid
  (96 paired measurements), the accuracy gap between the
  dynamic-thread decoder and the bounded-adaptive-subscription
  decoder is ``|acc(D_dyn) - acc(D_adp)| = 0.000`` at every
  cell. The prompt-token overhead of adaptive_sub vs dynamic
  is ≤ 12 %. Proof sketch + empirical anchor in § B.5 / § D.3.
  Promotes Conjecture C35-5 to an empirical theorem on this
  task family.

**New conjectures.**

- **Conjecture C36-5 (Analytic equivalence of threads and
  adaptive subs).** The empirical equivalence of P36-4
  extends to any task family whose gold answer is a
  deterministic function of (i) the static typed-handoff
  bundle and (ii) one round of typed producer-local
  causality hypotheses. Candidate counterexample families:
  nested threads, role-local reply memory, authenticated
  thread signatures.
- **Conjecture C36-6 (Dominance under noise).** For any
  static subscription graph ``σ`` and any reply-noise channel
  ``ν`` with ``p + q < 1/4``, the dynamic decoder strictly
  dominates static by ``≥ δ_ν > 0``.
- **Conjecture C36-7 (Adversarial-reply tightness).** The
  Theorem P36-2 collapse cannot be avoided by any bounded-
  reply typed protocol (thread or adaptive-sub) without a
  defensive-depth layer on the reply axis — the Phase-36
  analogue of Phase-34's ensemble extractors applied to the
  reply axis.
- **Conjecture C36-8 (LLM reply-noise calibrability).** For
  any LLM ``M``, there exist parameters ``(p_M, q_M)`` such
  that a Bernoulli-simulated run at ``(p_M, q_M)`` matches
  ``M``'s observed accuracy on the Phase-35 bank to within
  5 pp. The Phase-33 extractor-noise calibration machinery
  applies on the reply axis.

**Architectural consequence.** Phase 36 closes three escape
hatches left open by Phase 35: "maybe reply noise kills the
primitive" (P36-1, P36-2), "maybe adaptive subscriptions
obsolete threads" (P36-4 + C36-5), "maybe typed LLM replies
cannot preserve the Phase-35 substrate guarantee" (P36-3). The
primitive choice between bounded threads and bounded adaptive
subscriptions is now an engineering axis on this task family,
not a correctness axis. The bounded-context claim of Theorem
P35-2 carries over byte-for-byte to the Phase-36 reply-noise
and LLM-replier paths (the noise channel perturbs *content*,
not cap); for adaptive subscriptions the analogous bound is
runtime-enforced by ``max_active_edges`` + TTL (open: prove
the corresponding type-level guarantee).

## What Phase 37 adds

Phase 37 delivers three instruments at the reply-axis and
coordination-primitive layers:

- **Theorem P37-1 (Real-LLM reply noise is dominated by
  semantic mislabel).** Let ``M`` be an Ollama local LLM and
  let ``κ_M`` be the per-call calibration measure on the
  Phase-35 contested bank. Then ``Pr[κ_M ∈
  {malformed, oov}] ≤ 0.01``, ``Pr[κ_M = correct] ≈ 0.10``,
  and the remaining ``≈ 0.90`` lands in the ``sem_*``
  buckets with a dominant ``sem_root_as_symptom`` bias.
  Empirically confirmed on qwen2.5:0.5b and qwen2.5-
  coder:7b. Refutes the Phase-36 Conjecture C36-8 in its
  "synthetic malformed_prob is a good surrogate" form.

- **Theorem P37-2 (Reply-axis ensemble recovers biased-
  primary collapse).** Under a primary replier that always
  emits INDEPENDENT_ROOT (the real-LLM over-eager IR
  pattern), the ``MODE_DUAL_AGREE`` and ``MODE_VERIFIED``
  ensembles on the Phase-35 bank recover contested accuracy
  from 0 (single) to 1 (ensemble) while ``MODE_PRIMARY_
  FALLBACK`` remains at 0 (primary is well-formed — no
  fallback triggered). Closed-form + empirical.

- **Theorem P37-3 (Primary-fallback recovers syntactic
  noise).** Under ``malformed_prob = p_m`` applied at the
  primary replier's output, ``MODE_PRIMARY_FALLBACK``'s
  accuracy ceiling is the fallback replier's clean-bank
  accuracy. Empirically: single = 83 %, primary_fallback =
  100 % at ``p_m = 0.5`` on the Phase-35 bank.

- **Theorem P37-4 (Ensemble structural limit: noise below
  the ensemble is irrecoverable).** For any Phase-37
  ensemble mode ``E`` and any noise wrapper ``ν`` composed
  on the extractor-*output* boundary (Phase-36
  ``ReplyNoiseConfig`` / ``AdversarialReplyConfig``),
  ``acc(D_dyn with E ∘ ν) = acc(D_dyn with single ∘ ν)``.
  The ensemble contributes no information past ``ν``.
  Structural argument + empirical anchor.

- **Theorem P37-5 (Nested-contest accuracy equivalence +
  structural complexity separation).** On the Phase-37
  nested-contest bank (3 scenarios × 6 cells), the
  dynamic-thread-with-max_rounds=2 and the adaptive-sub-
  with-inter-round-briefings strategies both achieve 100 %
  full accuracy (0 pp gap); the thread uses 0 inter-round
  briefing edges, adaptive-sub uses 18. Accuracy
  equivalence extends to this family; protocol-complexity
  does not. Promotes the Phase-36 empirical equivalence
  result to a strictly larger task family.

- **Conjecture C37-1 (Calibration is task-and-prompt
  specific).** For any LLM and task bank, the Phase-37
  calibration distribution is not a property of the model
  alone — it is jointly determined by the prompt structure.
  Falsifiable by a prompt-engineering study that shifts the
  measured distribution under the same model.

- **Conjecture C37-2 (Full reply-axis defense requires two
  ensemble layers).** To defend against both reply-
  generation noise and extractor-output noise, compose
  ``core/reply_ensemble`` with Phase-34
  ``core/ensemble_extractor``. Empirically measurable, not
  yet run.

- **Conjecture C37-3 (Nested-contest equivalence is tight
  under typed protocols).** For any task family expressible
  as a finite sequence of typed-producer-local-causality-
  hypothesis rounds with a terminating-resolution decoder,
  bounded threads and bounded adaptive subscriptions
  augmented with inter-round briefing edges are accuracy-
  equivalent.

- **Conjecture C37-4 (Minimal dynamic primitive).** The
  minimal substrate feature set sufficient for Phase-35
  contested + Phase-37 nested scenarios is: typed reply-kind
  enum, bounded witness-token cap, terminating resolution
  rule, round-aware reply state, bounded-context invariant.
  Any substrate omitting any of the five has a collapse
  scenario on the current benches.

**Architectural consequence.** Phase 37 closes three more
escape hatches: "maybe synthetic reply noise captures the
real thing" (P37-1), "maybe the reply axis has no
defensive depth" (P37-2 + P37-3), "maybe
thread-vs-adaptive equivalence is a one-round artefact"
(P37-5). Re-scopes the frontier to the *minimum dynamic
primitive* (C37-4) and the *two-layer ensemble composition*
(C37-2) as the next first-order research questions. No
prior Phase-35 / Phase-36 theorem is invalidated; the
Phase-35 bounded-context bound carries over to the
calibration wrapper (pass-through) and the ensemble (adds
1× or 2× replier-call cost, still in ``O(R*·τ)``).

---

## What Phase 38 adds

Phase 38 delivers four instruments that close three
Phase-37 conjectures (C37-1 pipeline, C37-2, C37-4) and
promote one structural observation into theory:

- **Theorem P38-1 (Two-layer composition closes the
  conjunction cell).** On the Phase-35 contested bank under
  the joint noise cell
  ``C_∧ = {ext_drop_gold ∧ rep_biased_primary}``:

      acc(D_dyn with baseline      on C_∧) = 1/3
      acc(D_dyn with extractor_only on C_∧) = 1/3
      acc(D_dyn with reply_only    on C_∧) = 1/3
      acc(D_dyn with two_layer     on C_∧) = 5/6

  The two-layer composition
  ``UnionClaimExtractor ∘ EnsembleReplier(MODE_DUAL_AGREE)``
  is the unique configuration tested that closes the joint
  cell — layer-1 defends drop, layer-2 defends biased-
  primary, neither alone suffices. Empirical, pooled over 6
  scenarios × 3 seeds × 2 k. Promotes Phase-37 Conjecture
  C37-2 to an empirical theorem.

- **Theorem P38-2 (PathUnion combiner above noise wrapper).**
  On the Phase-35 bank under the Phase-37 ``adv_drop_root``
  cell (noise at the extractor-output boundary, strictly
  above any reply-axis ensemble combiner per P37-4),
  ``PathUnionCausalityExtractor`` with
  ``PATH_MODE_UNION_ROOT`` recovers full accuracy to 1.0
  while every Phase-37 reply-axis ensemble mode (single,
  dual_agree, primary_fallback, verified) collapses to
  0.333. Closed-form proof: the combiner reads two full
  causality paths, combines at the class level, and is
  placed strictly above any per-path noise wrapper; under
  budget-1 adversary only one path is damaged per scenario
  and UNION_ROOT's "at least one IR" rule selects the
  undamaged path's gold class.

- **Theorem P38-3 (Minimum load-bearing feature set on
  Phase-35 + Phase-37 families).** On the union of the
  Phase-35 contested bank and the Phase-37 nested bank,
  per-feature ablation of the Phase-35 thread primitive
  yields:

      feature removed           contested    nested
      none (full)               1.000        1.000
      typed_vocab               0.500        0.333
      terminating_resolution    0.333        0.000
      round_aware_state         1.000        0.000
      bounded_witness           1.000        1.000
      frozen_membership         1.000        1.000
      all                       0.333        0.000

  ``typed_vocab``, ``terminating_resolution``, and
  ``round_aware_state`` are each individually load-bearing
  on at least one family. ``bounded_witness`` is null-
  control on accuracy but load-bearing for Theorem P35-2's
  context bound. ``frozen_membership`` is null-control on
  both tested families. Partial confirmation of Phase-37
  Conjecture C37-4 — three of the five features are
  confirmed load-bearing; one is confirmed null-control;
  the fifth (``frozen_membership``) remains an open
  falsifier target (Conjecture C38-2).

- **Theorem P38-4 (Prompt-variant pipeline validates on
  a controlled bias model).** The Phase-38 experiment
  frame — ``build_thread_reply_prompt_variant`` →
  ``VariantLLMThreadReplier`` → ``CalibratingReplier`` →
  thread → ``ReplyCalibrationReport`` — measures per-
  variant calibration shift faithfully on a deterministic
  ``BiasShiftMockReplier``. On the bank, the ``rubric``
  and ``contrastive`` variants reduce the mock's
  semantic-wrong rate from 0.688 to 0.225 while every
  variant preserves the Phase-36 typed-reply contract
  (allowed kinds, witness cap, UNCERTAIN fallback). This
  is a pipeline-correctness theorem: the claim is not
  that real LLMs behave this way (Conjecture C38-3 is
  the open measurement) but that the pipeline's bucket
  accounting and substrate invariants hold for every
  variant.

- **Conjecture C38-1 (Two-layer is minimal vs the
  {layer-1, layer-2} noise joint).** For any noise pair
  ``(ν_1, ν_2)`` such that each is recoverable by its own
  axis-local secondary, the composition
  ``UnionClaimExtractor ∘ EnsembleReplier`` recovers the
  joint cell. Falsifiable by a ``(ν_1, ν_2)`` with
  cross-layer correlation where (i) each is axis-locally
  recoverable AND (ii) the composition collapses.

- **Conjecture C38-2 (``frozen_membership`` is load-
  bearing on some task family ``Z*``).** There exists a
  bounded-context task family ``Z*`` where removing
  ``frozen_membership`` from the Phase-35 thread primitive
  strictly reduces accuracy. Candidate: multi-auditor
  contests with inter-thread voting. Would tighten the
  primitive-minimality claim of Theorem P38-3.

- **Conjecture C38-3 (Prompt-shaped bias is model-
  invariant within size class).** For two LLMs of
  comparable parameter count, per-variant calibration
  shift ``Δκ_{variant, M_1} ≈ Δκ_{variant, M_2}`` up to
  ±0.1 on the bucket rates. Weak precondition already
  observed in Phase-37 Part A (qwen2.5:0.5b and
  qwen2.5-coder:7b share the default-variant
  distribution). The real-LLM measurement under the
  Phase-38 variants is one CLI parameter from the
  existing pipeline.

**Architectural consequence.** Phase 38 closes Phase-37
Conjecture C37-2 in the form of Theorems P38-1 and P38-2
— the first formal statement in the programme of the
"depth of defense" position: for every boundary at which
noise can enter, the ensemble combiner must sit strictly
above that boundary; stacking ensembles at multiple
boundaries is necessary when noise can enter at multiple
boundaries. Phase-37 Conjecture C37-4 is partially
confirmed (3/5 features load-bearing, 1/5 null, 1/5
open). Phase-37 Conjecture C37-1 acquires a pipeline
(Theorem P38-4); the real-LLM sweep (C38-3) is a
parameter change away. No prior theorem is invalidated;
all Phase-35..37 bounded-context and typed-reply
contracts hold on every Phase-38 defense (ensembles
compose at the contract level, not below it).

## What Phase 39 adds

Phase 39 ships four theorems that close Phase-38's
explicitly-named real-LLM data gap (C38-3 in the
optimistic-read direction, C37-1 on the model side),
extend the Phase-31 / Phase-35 bounded-context
guarantee to a SWE-bench-style multi-role team, and
formalise the empirical regime taxonomy that the
Phase 30..38 evidence has been pointing to.

* **Theorem P39-1 — Prompt-shape vs model-shape on
  the Phase-35 contested bank.** For every Phase-38
  prompt variant ``V`` and every model ``M`` ∈
  {``qwen2.5:0.5b``, ``qwen2.5-coder:7b``},
  ``correct_rate(κ_M(V)) ≤ ε_corr`` with
  ``ε_corr = 0.10``. On ``qwen2.5:0.5b``, four of five
  variants reproduce the per-bucket histogram to within
  ±0 calls; the fifth (``forced_order``) shifts mass
  from semantic-wrong (0.90 → 0.30) to malformed
  (0.00 → 0.60) without changing correct (0.10).
  Empirically refutes the optimistic read of Conjecture
  C38-3 on this size class; tightens C37-1 to the model
  side. Falsifier: a Phase-38-class variant that
  pushes ``correct_rate`` above 0.20 on this exact bank
  on this exact model.
* **Theorem P39-2 — Communication-bounded vs
  transcription-bounded regime.** For any team-shaped
  task ``Z`` decomposable into a substrate-emitted
  bundle ``B`` plus a single-LLM synthesis stage,
  ``A(D_substrate(z, M)) ≤ min(A_substrate(z),
  A_synth(B, M))`` with equality iff the synthesis is
  order-preserving on the bundle. The substrate is the
  active constraint iff the model saturates the
  synthesis bound. Phase-31 § D.4 is the empirical
  signature of the *transcription-bounded* regime
  (substrate root_cause = 1.00 vs full = 0.40 on
  qwen2.5:0.5b); Phase-37 § D.1 is the empirical
  signature of the *communication-bounded* regime in
  the reply-axis sense.
* **Theorem P39-3 — Substrate bounded-context on
  multi-role SWE teams.** On the Phase-39
  ``MiniSWEBank`` four-instance bank, the
  patch_generator-role prompt size under the substrate
  strategy is 842 chars at every ``n_distractors`` ∈
  {0, 6, 12, 24}, while naive grows from 949 → 1936.
  Pass@1 = 1.000 on every (strategy, distractor) cell
  under the deterministic oracle generator. The
  Theorem-P31-3 / P35-2 bounded-context invariant
  extends to a SWE-bench-shaped team without
  modification; the patch_generator's substrate inputs
  are a typed handoff bundle (issue summary +
  located hunk), not the raw event stream.
* **Theorem P39-4 — SWE-bench schema mappability.**
  Every required SWE-bench-instance field has a
  typed counterpart in ``SWEBenchStyleTask``; the
  only schema gap (gold-patch unified-diff →
  substitution-shape) is bounded and admits a
  mechanical ~30-line ``unidiff``-style adapter. The
  Phase-39 ``SWEBenchAdapter.from_dict`` shim already
  round-trips the substitution-shape gold-patch path.

Four conjectures (C39-1..C39-4) name the remaining
research-heavy questions:

* **C39-1 (strong-model bias saturation).** ``∃ M_*``
  such that for every ``M`` of class ``M_*`` or
  larger and every Phase-38 variant ``V``,
  ``correct_rate(κ_M(V)) ≥ 0.5``. Open. Falsifiable
  in two directions (strong model that already
  passes; large model that doesn't).
* **C39-2 (prompt-shape recovery requires fine-
  tuning).** On a model where every Phase-38 variant
  yields ``correct_rate ≤ ε``, no zero-shot prompt
  protocol moves the rate above ``2ε``; only
  fine-tuning lifts the model into the
  communication-bounded regime of Theorem P39-2.
* **C39-3 (substrate dominance on real SWE-bench).**
  The substrate's strict dominance margin grows as
  ``Θ(log(repo_token_count) / model_context_window)``
  on real SWE-bench instance distributions; small /
  medium repos give competitive, not strictly
  dominant, substrate behaviour.
* **C39-4 (mini-SWE Lipschitz-predicts SWE-bench).**
  ``|pass@1(SWE-bench Lite, f) - pass@1(MiniSWEBank,
  f)| ≤ L`` for some ``f``-independent ``L`` in the
  matched-substrate regime; Phase 39 ships the mini
  bank as the *precondition test* for a generator's
  substrate-side behaviour.

**Architectural consequence.** Phase 39 settles three
escape hatches: "maybe the Phase-37 bias is prompt-
shaped after all" (Theorem P39-1 refutes); "maybe the
substrate's bounded-context property doesn't extend to
SWE-style multi-role teams" (Theorem P39-3 settles);
"maybe the SWE-bench gap is architectural" (Theorem
P39-4 reduces it to a unidiff-parser + sandbox).
Phase 39's regime taxonomy (P39-2) is the lens that
distinguishes substrate-moveable gaps from
model-moveable gaps; the programme's next decisions
(stronger model? fine-tuning? real SWE-bench loader?)
are clearly partitioned by which side of the
inequality moves the binding constraint. No prior
theorem is invalidated; all Phase 31..38 bounded-
context and typed-reply contracts hold on every
Phase-39 task family by construction (the SWE bridge
sits strictly above the substrate, like every
Phase-32+ task module).

## What Phase 40 adds

Phase 40 ships three theorems that close the
Phase-39-named *mechanical* follow-up end-to-end
(unified-diff parser + real-shape adapter + JSONL
loader + sandboxed execution boundary), and three
conjectures that name the remaining empirical
follow-ups.

* **Theorem P40-1 — Unified-diff round-trip on a
  clean hunk.** For every unified diff produced by
  ``git diff`` on a single-hunk in-place edit of a
  file ``f`` whose ``-`` block (the ``old_block``)
  appears uniquely in the buggy source,
  ``apply_patch(buggy_source, parse_unified_diff(diff)
  [f]) → patched_source`` byte-equal to ``git apply``'s
  result, modulo trailing-newline normalisation.
  Falsifier: a clean-hunk diff for which the
  round-trip fails — would indicate a parser bug.
* **Theorem P40-2 — Real-shape substrate bounded-
  context preservation.** On the bundled six-instance
  JSONL bank
  (``vision_mvp/tasks/data/swe_real_shape_mini.jsonl``),
  substrate prompt-chars are constant in
  ``n_distractors`` (813 across {0, 6, 12, 24})
  while naive grows from 826 → 2 145. Pass@1 =
  1.000 on every (strategy, distractor) cell under
  the deterministic oracle generator + the
  ``SubprocessSandbox`` backend.
* **Theorem P40-3 — Sandbox-boundary preservation.**
  For the deterministic-oracle generator, every
  (strategy, sandbox-backend) cell among
  {``InProcessSandbox``, ``SubprocessSandbox``} ×
  {naive, routing, substrate} delivers pass@1 =
  1.000 on the Phase-39 mini bank and the Phase-40
  real-shape JSONL bank. The Phase-40 sandbox
  boundary is *transparent* to the substrate's
  correctness ceiling.

Three conjectures (C40-1..C40-3) name the remaining
empirical follow-ups:

* **C40-1 (sandbox cost amortisable).** Per-instance
  Phase-40 sandbox overhead is bounded (≤ 100 ms
  subprocess; ≤ 2 s docker on warm image cache) and
  dominated by the LLM call wall on real-LLM runs.
  Falsifier: a generator family whose per-instance
  cost is dwarfed by sandbox overhead.
* **C40-2 (loader sufficiency for SWE-bench Lite).**
  ``load_jsonl_bank`` + ``SWEBenchAdapter.from_swe_
  bench_dict`` + ``SubprocessSandbox`` ingests
  ≥ 50 % of SWE-bench Lite without modification;
  failures are adapter-shaped (file create/delete,
  non-unique anchor, ``test_patch`` outside the
  bridge contract), not substrate-shaped. Falsifier:
  a SWE-bench Lite subset where the dominant
  loader-side failure mode is substrate-shaped.
* **C40-3 (sandbox-axis equivalence).** For every
  patch generator emitting std-library-only patches,
  pass@1 under ``SubprocessSandbox`` equals pass@1
  under ``DockerSandbox`` modulo at most one
  ``timeout`` reclassification per 10⁴ measurements.
  Falsifier: a generator whose pass@1 differs
  systematically across the two backends.

**Architectural consequence.** The Phase-40 layer is
*strictly above* Phase 39: ``SWEBenchStyleTask``,
``ProposedPatch``, ``WorkspaceResult``, and
``run_swe_loop`` are unchanged. The Phase-39 in-process
``run_patched_test`` is preserved as ``InProcessSandbox``
for byte-for-byte regression. The honest finding from
the Phase-40 ``qwen2.5-coder:7b`` real-LLM run
(substrate scores 4/6 vs naive's 5/6 on the bundled
6-instance bank) is a one-instance variance inside
Theorem P39-2's transcription-bounded regime — the
substrate carries the gold *semantically* but the
bridge's byte-strict ``apply_patch`` matcher demands
literal-text reproduction the 7B does not always
deliver under bounded context. The remediation is
*generator-side* (more permissive matcher, fine-tuned
generator, larger N) — none substrate-shaped. Phase 40
makes "the external task loop" shipped infrastructure;
the empirical SWE-bench Lite ranking measurement is
now Conjecture C39-3 / C39-4 / C40-2 territory, not
P39-4 territory.

## What Phase 41 adds

Phase 41 ships three theorems and four conjectures that
convert Phase 40's "real external task loop exists" into
"real external task loop has larger-N empirical ranking
data and a two-axis attribution surface" (substrate delivery
axis × matcher precision axis).

* **Theorem P41-1 — Bounded-context preservation at
  scale.** On the 28-instance Phase-41 bank
  (``vision_mvp/tasks/data/swe_lite_style_bank.jsonl``,
  ~4.7× the Phase-40 mini bank), substrate
  ``patch_generator`` prompt-chars are constant in
  ``n_distractors`` at 746.4 across {0, 6, 12, 24} and
  across every matcher mode in ``ALL_APPLY_MODES``; naive
  grows 806.8 → 2 125.8 (**2.6×** span). Pass@1 = 1.000
  on all 672 oracle-sandbox measurements; every
  ``chain_ok`` is True. Reproduces the Phase-31 / Phase-35
  / Phase-40 bounded-context invariant at a larger bank of
  diverse edit shapes — not a mini-bank artefact.
  Falsifier: a bank of ≥ 20 instances where substrate's
  ``patch_generator`` prompt grows with ``n_distractors``.
* **Theorem P41-2 — Oracle-ceiling is matcher-mode-
  invariant.** For every matcher mode ``m ∈
  ALL_APPLY_MODES``, every strategy ``s ∈ {naive,
  routing, substrate}``, and every sandbox backend
  ``S ∈ {InProcessSandbox, SubprocessSandbox}``, the
  deterministic oracle delivers ``pass@1 = 1.000`` on
  the Phase-41 bank. Permissive matching does not
  subtract correctness from a byte-exact patch source —
  the null-control of the matcher-permissiveness axis.
  Falsifier: a matcher mode where the oracle's pass-rate
  drops below 1.000 on the Phase-41 bank.
* **Theorem P41-3 — Matcher-permissiveness attribution
  decomposition.** For a real-LLM generator ``f`` and a
  matcher-mode pair ``(strict, permissive)``,

  ```
  Δ_pass@1(f, s) = |R_recovered(f, s)| − |R_regressed(f, s)|
                 = pass@1(permissive, f, s) − pass@1(strict, f, s).
  ```

  ``R_recovered`` are instances that strict-fail and
  permissive-pass (generator-side byte-fidelity recovery);
  ``R_regressed`` are instances that strict-pass and
  permissive-fail (matcher over-acceptance risk). The two
  sets are independently measurable from the Phase-41
  result JSON. Combined with Theorem P39-2
  (transcription-bounded vs communication-bounded
  regimes), the programme now has a *two-axis attribution
  surface* for any real SWE loop. Falsifier: a pass-rate
  delta between two matcher modes not equal to
  ``|R_recovered| − |R_regressed|`` (would indicate a
  bridge-accounting bug).

Five conjectures (C41-1..C41-5) name the remaining
empirical follow-ups:

* **C41-1 (communication-bounded at ≥ 50 instances).**
  On a genuine SWE-bench Lite bank of ≥ 50 instances
  under a generator whose byte-fidelity floor is above
  the bridge's strict-matcher precision on ≥ 50 % of
  instances, ``pass@1(naive) − pass@1(substrate) ≤ ε``
  for a small task-length-dependent ``ε`` that does not
  grow with N. Falsifier: a ≥ 50-instance run where the
  substrate-vs-naive gap exceeds 0.1 at both matcher
  modes.
* **C41-2 (matcher-permissiveness saturation).**
  ``|R_recovered| / N`` is bounded by a task-family-local
  constant ε_f that does not grow with benchmark size.
  Falsifier: a SWE-bench Lite sweep where ``R_recovered``
  grows linearly in N.
* **C41-3 (stronger-model saturates the strict matcher).**
  For a sufficiently capable model (30B+ class),
  ``|R_recovered| / |R_strict_fail| → 0``. Falsifier: a
  frontier-class model whose Phase-41 attribution table
  has non-empty ``R_recovered``.
* **C41-4 (comm-bounded vs generator-bounded regime
  decomposition).** Every Phase-41 real-SWE cell admits
  a decomposition ``pass@1 = P_comm · P_gen`` with the
  two factors independently measurable. Falsifier: a cell
  where the empirical pass-rate is not well-approximated
  by any product of a strategy-only factor (``P_comm``)
  and a (model, matcher)-only factor (``P_gen``).
* **C41-5 (parser-compliance attribution boundary).** For
  a non-coder-finetuned model in the 7B–30B class, the
  Phase-41 bridge's byte-strict OLD/NEW parser
  (``_BLOCK_RE``) is the dominant failure boundary —
  surfacing above the matcher axis. Empirical anchor:
  § D.4 of ``RESULTS_PHASE41.md`` shows ``gemma2:9b`` at
  0/28 on every Phase-41 cell while emitting the correct
  fix on an ad-hoc spot check. Falsifier: a general-
  purpose model in that class whose ``llm_patch_generator``
  parse-failure rate on the Phase-41 bank is below 20 %.

**Architectural consequence.** The Phase-41 layer is
*strictly above* Phase 40: ``apply_patch`` gains a
``mode`` kwarg whose default preserves Phase-40 strict
semantics byte-for-byte; ``run_swe_loop`` and
``run_swe_loop_sandboxed`` gain an ``apply_mode`` kwarg
recorded in ``SWEReport.config``; no existing primitive
is renamed or removed. Every Phase-40 test and artifact
reruns byte-for-byte under the Phase-41 build. The
programme's external-validity gap to public SWE-bench is
now bracketed by Conjectures C41-1 / C41-2 / C41-3 —
all empirical, none substrate-shaped.

## What Phase 42 adds

Phase 42 adds three theorems and three conjectures that
close the parser-compliance layer Phase 41 surfaced as
Conjecture C41-5 but could not measure. The Phase-41
matcher-attribution axis (P41-3) and Phase-39
substrate-attribution regime (P39-2) together now sit
*below* the Phase-42 parser-attribution axis in the
failure stack; the programme has a **three-axis
attribution surface** for any real SWE loop.

* **Theorem P42-1 — Parser-compliance attribution
  decomposition.** For every patch generator ``f``, every
  parser-mode pair ``(π_base, π_cand)``, every matcher
  mode ``m``, and every strategy ``s``,

  ```
  pass@1(f, π_cand, m, s) − pass@1(f, π_base, m, s)
    = |R_recovered_parser(f, π_cand, m, s)| / N
      − |R_regressed_parser(f, π_cand, m, s)| / N.
  ```

  ``R_recovered_parser`` are instances whose parse flips
  ``{fail → ok}`` under the candidate parser and whose
  downstream ``(apply, test)`` also passes;
  ``R_regressed_parser`` are instances whose parse flips
  ``{ok → fail}``. The Phase-42 robust parser's
  ``R_regressed_parser = ∅`` on every cell in the
  shipped artifacts; falsified by any cell where a
  parse that succeeded under strict fails under robust.
* **Theorem P42-2 — Parser recovery cannot produce a
  false pass.** For every recovery heuristic
  ``ρ ∈ {closed_at_eos, loose_delim, unified_diff,
  fenced_code, label_prefix}``, the recovered
  ``(old, new)`` bytes are a *subset* of the generator's
  output bytes (byte-provenance by construction). The
  downstream ``(apply_patch → run_patched_test)``
  outcome is therefore a pure function of generator
  bytes, not of parser choice. A passing test after
  recovery certifies the generator's semantic output,
  not the parser's lenience. Falsifier: a recovery
  heuristic whose emitted bytes are not a subset of the
  generator's output.
* **Theorem P42-3 — Robust parser dominates on
  format-noncompliant generators.** For a generator
  ``f`` whose format-noncompliance rate under
  ``π_strict`` is ``η`` with dominant failure shape in
  ``{unclosed_new, prose_only_with_inline_code,
  fenced_only_2, label_prefix, fence_wrapped_payload}``,

  ```
  compliance_rate(f, π_robust) ≥ compliance_rate(f, π_strict) + η(1 − ε)
  ```

  where ``ε ≥ 0`` is the fraction of noncompliant
  outputs whose shape escapes every heuristic in the
  robust parser. On the Phase-41 § D.4 gemma2:9b failure
  mode (``unclosed_new``), ``ε = 0`` by construction
  (``RECOVERY_CLOSED_AT_EOS`` handles it). Falsifier:
  a model whose ``ε → 1`` under a noncompliance shape
  that matches one of the four named classes.

Three conjectures (C42-1..C42-3) name the remaining
empirical follow-ups:

* **C42-1 (substrate-vs-naive gap ≤ 1 pp at N ≥ 50).**
  On a ≥ 50-instance SWE-bench-Lite-style bank (the
  Phase-42 57-instance bank or a real public Lite run)
  under a coder-finetuned ≥ 7B model,
  ``|pass@1(naive) − pass@1(substrate)| ≤ 0.01`` under
  every matcher × parser × distractor cell. Falsifier:
  a cell with gap > 0.01. The conjecture reflects the
  programme's stated position that the substrate's
  durable claim is **bounded active context per role**
  (Theorem P41-1 / P42 mock reproduction), not a
  pass@1 lift.
* **C42-2 (parser-compliance dominates
  matcher-permissiveness at 7B–30B).** For models in the
  7B–30B class the Phase-42 robust-parser recovery rate
  exceeds the Phase-41 permissive-matcher recovery
  rate on every SWE-bench-Lite-style bank. Falsifier: a
  model where ``|R_recovered_matcher|`` strictly exceeds
  ``|R_recovered_parser|``.
* **C42-3 (three-axis decomposition completeness).**
  Every end-to-end real-SWE pass@1 measurement decomposes
  as ``pass@1 = P_parse · P_match · P_semantic ·
  P_sandbox`` with ``P_sandbox`` fixed at 1.000 by
  Theorem P40-3. Generalises Conjecture C41-4 from two
  factors to four. Falsifier: a cell whose measured
  pass@1 is not well-approximated by the product up to
  a 5 pp tolerance.

**Architectural consequence (Phase 42).** The Phase-42
layer is *strictly above* Phase 41:
``llm_patch_generator`` gains three kwargs
(``parser_mode``, ``parser_counter``, ``prompt_style``)
whose defaults preserve Phase-41 behaviour byte-for-byte;
``build_patch_generator_prompt`` gains a
``prompt_style`` kwarg whose default preserves Phase-41
byte-for-byte; ``LLMClient`` gains a ``base_url`` kwarg
whose default (``None`` → localhost) preserves every
pre-Phase-42 test byte-for-byte. No existing primitive
is renamed or removed. Every Phase-41 test and artifact
reruns byte-for-byte under the Phase-42 build
(62/62 SWE-arc tests green; Phase-42 test slice 31/31
green). The external-validity gap to public SWE-bench
is now bracketed by Conjectures C41-1..C41-3 +
C42-1..C42-3 — all empirical, none parser-shaped,
none matcher-shaped, none substrate-shaped.

## What Phase 43 adds

Phase 43 adds three theorems and four conjectures that
*characterise* the Phase-42 residue instead of reducing
it: the substrate-vs-naive gap is zero on every measured
coder-finetuned model at the ≥ 50-instance external-
validity scale, and the failure mass that remains is
classifiable into a nine-label closed semantic
vocabulary. The Phase-43 layer is *analysis-only* — it
consumes Phase-42 artifacts without modifying any
Phase-31..42 primitive.

* **Theorem P43-1 — Bounded-context preservation on the
  external-validity bank.** On the 57-instance
  ``swe_lite_style_bank.jsonl`` under every
  ``(parser_mode, apply_mode, n_distractors)`` cell in
  ``ALL_PARSER_MODES × ALL_APPLY_MODES × {0, 6, 12, 24}``,
  the substrate's ``patch_generator`` prompt token budget
  is **205.9 tokens flat**; naive grows from 197.3 → 527.1
  tokens (**2.7×** span) across the same axis. Extends
  Theorem P41-1 / P42-1 from the 28-instance mini bank
  and the 57-instance parser-cell subset to the full
  cross product at the external-validity scale.
  Falsifier: a cell where substrate prompt size varies
  with ``n_distractors`` or the parser/apply axis.
* **Theorem P43-2 — Post-parser-recovery semantic residue
  is structurally classifiable.** Every
  ``(instance, strategy, cell)`` measurement in a
  Phase-42 artifact is assigned exactly one label from
  a nine-element closed vocabulary
  (``SEM_OK`` / ``SEM_PARSE_FAIL`` /
  ``SEM_WRONG_EDIT_SITE`` / ``SEM_RIGHT_SITE_WRONG_LOGIC``
  / ``SEM_INCOMPLETE_MULTI_HUNK`` / ``SEM_TEST_OVERFIT`` /
  ``SEM_STRUCTURAL_SEMANTIC_INERT`` /
  ``SEM_SYNTAX_INVALID`` / ``SEM_NO_MATCH_RESIDUAL``) by a
  pure, deterministic classifier
  (``classify_semantic_outcome``). The labelling is total
  (every measurement receives a label), exhaustive (the
  nine labels partition the outcome space), and
  orthogonal to parser/matcher choice (at matched
  ``(proposed_patch, test_passed)``, the label depends
  only on the bank instance's structure). Falsifier: a
  measurement that receives two distinct labels under two
  independent classifier invocations, or an
  ``error_kind`` × ``test_passed`` tuple not covered by
  any label.
* **Theorem P43-3 — Semantic-ceiling separation on
  coder-finetuned models at N ≥ 50.** For every measured
  coder-finetuned model (parameter count ≥ 7B) on the
  canonical ``parser=robust / apply=strict /
  n_distractors = 6`` cell of the 57-instance bank:
    (a) ``pass@1(substrate) = pass@1(naive) =
        pass@1(routing)`` (strategy-invariance);
    (b) the per-strategy ``SemanticCounter.by_strategy``
        label histograms are byte-identical across
        naive/routing/substrate (the substrate does not
        redistribute the failure mass, it just delivers
        the same content with a bounded prompt);
    (c) the pooled failure-mix is dominated by
        ``SEM_WRONG_EDIT_SITE`` on coder-finetuned models
        and by ``SEM_SYNTAX_INVALID`` on general-purpose
        models of matched parameter class.
  Falsifier: a coder-finetuned ≥ 7B model where the
  substrate-vs-naive gap exceeds the per-strategy
  measurement variance (≥ 1 instance on a 57-instance
  bank at the canonical cell) OR where the dominant
  failure-mix label under coder training is not
  ``SEM_WRONG_EDIT_SITE``.

Four conjectures (C43-1..C43-4):

* **C43-1 (frontier coder closes wrong-edit-site without
  re-opening the substrate gap).** A frontier
  reasoning/coder model at ≥ 30B active parameters
  achieves ``|R_{SEM_WRONG_EDIT_SITE}| / 57 ≤ 0.02``
  AND preserves Theorem-P43-3 (a). Falsifier: a frontier
  coder cell where wrong-edit-site rate > 2 % or the
  substrate-vs-naive gap > 1 pp.
* **C43-2 (residue composition is training-mix-indexed,
  not parameter-count-indexed).** For two models
  ``f_A, f_B`` of matched parameter count but distinct
  training mixes, ``failure_mix(f_coder) ≠
  failure_mix(f_general)`` with the ``SEM_SYNTAX_INVALID``
  label ≥ 2× more frequent under general-purpose
  training. Falsifier: two matched-parameter-count
  models with the same dominant failure label.
* **C43-3 (substrate bounded-context invariant is
  model-independent).** The Theorem-P43-1 invariant is
  a pure function of the bank instance's
  ``{issue_summary, hunk}``, independent of generator
  choice. Falsifier: a patch generator for which the
  substrate prompt size depends on ``n_distractors``.
* **C43-4 (semantic residue does not decompose further
  under existing substrate primitives).** No Phase-31 /
  Phase-35 / Phase-38 configuration that preserves the
  Phase-31 bounded-context property shrinks the
  Phase-43 residue by more than ``2/√N`` on any coder
  cell. Falsifier: a substrate configuration that moves
  ≥ 1 instance out of the residue at the canonical cell
  while preserving Theorem P41-1.

**Architectural consequence (Phase 43).** Phase 43 is
*strictly additive* and *analysis-only*:
``swe_semantic_taxonomy`` is a pure module with no
dependencies on the substrate/parser/bridge path;
``phase43_frontier_headroom`` is an offline driver that
consumes Phase-42 artifacts; ``LLMClient`` gains a
``think`` field whose default (``None`` → omit from
payload) preserves every pre-Phase-43 LLM call byte-for-
byte. The Phase-42 robust parser gains one additional
trailing-prose pattern (``\\n\\s*<{2,4}\\s*\\Z``) that
strips a partial / full delimiter close at end-of-
generation — a Phase-43 § D.4 regression surfaced by the
``qwen3.5:35b`` cluster run. No existing primitive is
renamed or removed; every Phase-39..42 regression test
passes byte-for-byte (110/110 on the Phase-39..43 SWE-arc
slice; Phase-43 test slice 18 / 18 green). The durable
substrate claim is now *unambiguously stated*:
**bounded active context per role**, not pass-rate lift;
the latter is empirically zero on every coder-finetuned
model tested at N ≥ 50, and the P43-2 taxonomy is the
tool that surfaces why.

## What Phase 44 adds

Three new theorems and four new conjectures, all grounded in
code and tests that ship alongside. No new mathematical tradition
is introduced; the framing is strict refinement of the Phase-43
residue-analysis layer.

### Theorem P44-1 — Raw capture is a lossless projection of
pipeline state

For every measurement ``(instance, strategy, parser_mode,
apply_mode, n_distractors)`` produced by the Phase-44 sweep, the
pair (Phase-42 parent measurement, Phase-44 ``RawCaptureRecord``)
determines every downstream pipeline output that is a pure
function of the LLM response and the cell axes (raw bytes,
ParseOutcome, proposed substitutions, applied substitutions,
patched-source SHA-256, test verdict). Proof sketch: the matcher
is a pure function of ``(buggy_source, proposed_patch,
apply_mode)`` — the first is keyed on ``instance_id`` (bank
JSONL), the second and third are in the record. The compile+test
cycle is a pure function of the patched source + test_source.
Test-implemented via
``test_phase44_mock_sweep_writes_parent_and_capture``.

### Theorem P44-2 — Refined classifier monotone on sentinel
inputs

``classify_semantic_outcome_v2(..., proposed_patch=sentinel, ...)
== classify_semantic_outcome(..., proposed_patch=sentinel, ...)``
for the sentinel
``(("__sentinel__", "__sentinel__"),)``. The v2 refined
classifier does not change any label the v1 classifier assigned
under the Phase-43 sentinel path. Proof sketch: v2 runs v1 first
to obtain the coarse label, then calls ``refine_semantic_outcome``,
which guards on the sentinel tuple and returns the coarse label
unchanged. Test-implemented via
``test_phase44_refined_classify_v2_monotone_on_sentinel``.

### Theorem P44-3 — Public-readiness verdict saturates on the
bundled 57-instance bank at external-validity scale

``run_readiness(swe_lite_style_bank.jsonl, limit=None,
sandbox_name="subprocess")`` returns
``{"ready": True, "n": 57, "n_passed_all": 57, "blockers": []}``
with every one of the five checks (schema / adapter / parser /
matcher / test_runner) at 57/57 in ~5.2 s wall. Proof sketch: the
validator is a pure function of JSONL bytes + sandbox; each check
is implemented against the same adapter / parser / matcher /
test-runner primitives the evaluation pipeline uses; the 57/57
saturation is empirical (verified on each CI run via
``test_phase44_public_readiness_full_bundled_bank``).

### Conjecture C44-1 — Wrong-edit-site refines to right-file-
wrong-span on coder-class models

On any coder-finetuned model at ≥ 7B parameters measured through
the Phase-44 sweep on the 57-instance bank at the canonical
cell, ``|R_right_file_wrong_span| / |R_wrong_edit_site_v1| ≥ 0.5``
— i.e. at least half of the Phase-43 coarse wrong-site bucket
refines to "anchored in the right file on a wrong span." The
conjecture discriminates whether the coder-class residue is a
*localisation-within-context* problem (C44-1 holds) versus a
*retrieval* problem (C44-1 fails). Open, pending Phase-44
cluster data.

### Conjecture C44-2 — Frontier 35B residue refines to overfit
or wrong-logic

On ``qwen3.5:35b`` at the canonical cell,
``|R_narrow_fix_test_overfit ∪ R_right_span_wrong_logic| ≥ 0.5 ·
|R_total|`` — i.e. at least half of the 2/57 failures are on
right-site patches whose logic fails to generalise. Opens the
question of what a 70B-class coder-finetuned frontier would
close.

### Conjecture C44-3 — Substrate gap is refinement-invariant on
coder-class models

The substrate-vs-naive pass@1 gap computed from the v2 refined
classifier equals the gap from the v1 coarse classifier equals
0 pp on every coder-class model. Structurally expected: v2
refines failure buckets but never moves a measurement between
SEM_OK and non-SEM_OK.

### Conjecture C44-4 — Readiness closed under row-level filtering

For any JSONL that passes ``V_ready``, every row-level subset
also passes. Structurally true as implemented (per-row
independence + AND union), but listed because a future validator
extension could invalidate the independence assumption.

### Summary

| Claim | Status |
|---|---|
| P44-1 raw capture is a lossless projection of pipeline state | **Theorem** (structural + round-trip test) |
| P44-2 refined classifier monotone on sentinel inputs | **Theorem** (code + test) |
| P44-3 public-readiness saturates on bundled bank | **Theorem** (empirical on 57/57) |
| C44-1 coder-class wrong_edit_site refines to right_file_wrong_span | **Conjecture** (Phase-44 cluster follow-up) |
| C44-2 frontier 35B residue refines to overfit or wrong-logic | **Conjecture** (Phase-44 cluster follow-up) |
| C44-3 substrate gap is refinement-invariant on coder-class | **Conjecture** (structurally expected) |
| C44-4 readiness closed under row-level filtering | **Conjecture** (trivially true as implemented) |

### Research arc framing — what Phase 44 materially changes

Phase 43 surfaced the sentinel-path limitation; Phase 44 closes
it. The architectural surface grows by one new opt-in module
(``swe_raw_capture.py``), one new pair of experiment drivers
(``phase44_semantic_residue.py`` sweep + analyse,
``phase44_public_readiness.py`` CI-gate validator), five new
refined-taxonomy labels, and one new v2 classifier. No existing
primitive is renamed or removed. Every Phase-39..43 regression
test passes byte-for-byte (112/112 on the SWE-arc slice;
Phase-44 test slice 23/23 green; 135 total). Phase 44 makes the
Phase-43 residue characterisation a *partition* rather than a
*count*, and ships the first validated CI gate for public-SWE-
bench-Lite drop-in.

This document gets updated whenever more theory becomes code.

---

## Phase 45 — Product surface + release criteria

| Claim | Strength |
|---|---|
| P45-1 runner composition is a faithful projection of the primitives | **Theorem** (structural + Phase-45 test slice) |
| P45-2 readiness gates the sweep unless ``force_sweep`` is set | **Theorem** (code + test) |
| P45-3 finished-product state is the logical product of the per-layer theorems | **Theorem** (composition of T3 / P41-1 / P41-2 / P43-3 / P44-1 / P44-2 / P44-3 / P45-1 / P45-2) |
| C45-1 profile set covers every post-Phase-40 published evaluation | **Conjecture** (structurally true on inspection) |
| C45-2 runner overhead <5 % of sandbox wall at bundled_57 | **Conjecture** (supported, breaks at small N) |
| C45-3 remaining blockers are model/data shaped, not architecture shaped | **Conjecture** (true as of Phase 45; revisable) |

### What Phase 45 materially adds to the audit

Phase 45 adds one new subpackage (``vision_mvp/product/``) and
one new §9 section in ``docs/context_zero_master_plan.md``. No
theory-layer object changes. The three new theorems are
composition and gating statements about the runner; they do not
strengthen or weaken any Phase-31..44 claim. The programme's
theorem-to-code ratio is therefore unchanged by this phase — the
new code is orchestration over already-theorem-backed primitives.

---

## Phase 46 — Boundary surface (import / CI / frontier slot)

| Claim | Strength |
|---|---|
| P46-1 import-audit saturates on the bundled bank (57/57 ambiguous shape, readiness READY) | **Theorem** (empirical + structural pureness of row-level check) |
| P46-2 CI-gate composition is faithful to `phase45.product_report.v1` | **Theorem** (structural; five deterministic predicates over report bytes + thresholds) |
| P46-3 capability declaration is separated from residency | **Theorem** (pure-function lookup; metadata plumbed through recorded launch payload) |
| C46-1 remaining blockers are boundary-shaped, not programme-shaped | **Conjecture** (supported by §9.8 inventory + Phase-46 code) |
| C46-2 70B ceiling lift + substrate-gap invariance | **Conjecture** (Phase-46 follow-up on model residency) |
| C46-3 CI consumption closed under profile extension | **Conjecture** (structurally true for the current sweep schema) |

### What Phase 46 materially adds to the audit

Phase 46 ships two new CLIs (import + CI gate), one profile
slot (70B frontier), and one declarative residency check.
No theory-layer object below the product surface changes; no
existing theorem is weakened or strengthened. The new
theorems are about the *boundary* between programme-internal
state and external inputs — specifically the saturation of
the import audit on the canonical bank, the faithfulness of
the CI gate to the Phase-45 report schema, and the separation
of model-capability *declaration* from model *residency*. The
programme's theorem-to-code ratio is unchanged; the new code
is orchestration at the boundary.
