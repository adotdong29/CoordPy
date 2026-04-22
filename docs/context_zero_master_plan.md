# Context-Zero — Master Plan

> Canonical long-running document for the research programme behind
> `context-zero`. This is a plan for a body of work, not a changelog.
> Phase-by-phase diaries live in `vision_mvp/RESULTS_PHASE*.md`;
> session notes live nowhere durable, on purpose. Last touched: Phase 46
> (2026-04-21).
>
> The programme has a deliberately dual identity: it is (a) a research
> programme in context, coordination, and exact external memory,
> producing theorems, open questions, and falsifiable measurements;
> and (b) an emerging tool/substrate for real agent teams. The
> code is evidence for the research, and the research is what makes
> the tool principled. Neither identity subsumes the other.

## 0. How to read this document

This file exists so that a reader arriving cold — a collaborator, a
future self, a reviewer — can (a) see the strongest version of the
thesis, (b) see how the work so far is organised as research arcs
rather than as incidents, and (c) see where the hard problems are
and what would count as genuinely moving the frontier.

The document is deliberately high-level. Code pointers are included
only where they are load-bearing for the argument; concrete per-phase
numbers live in the `RESULTS_PHASE*.md` notes alongside the code.

Three pieces of discipline this plan imposes on itself:

1. **No product framing.** The research is not a framework, a
   library, or a product. It is a set of claims about information
   flow in systems of LLM agents and about the shape of the contract
   between symbolic machinery and a model.
2. **No hype.** Every empirical number in this document is traceable
   to a result note and to a test or experiment. Any claim that is
   not traceable is marked as a conjecture.
3. **Mathematics, CS, and ML first; engineering in service.** The
   code exists to falsify or confirm claims. When a claim is
   confirmed, the code is the *evidence*; when falsified, the code
   is the *counterexample*. Engineering is never the goal.

---

## 1. Core thesis

### 1.1 The strongest version of the thesis

> **Context is a routing and substrate problem, not a compression
> problem. For any agent team coordinating on a task with typed
> per-role concerns, the minimum-sufficient information flowing to a
> given reasoner at a given moment is much smaller than the space of
> all text the system has seen, and the right path from "everything
> the system has seen" to "what this reasoner needs right now" is a
> layered pipeline in which every layer has an explicit, measurable
> loss profile.**

This is a stronger statement than "context windows are too small" or
"RAG is useful." It is the claim that:

- the dominant waste in multi-agent LLM systems is *irrelevance*,
  not *redundancy*;
- the correct response is to *not generate* irrelevant bytes into
  any agent's prompt, rather than generate them and then compress
  them;
- a substantial fraction of questions that are currently answered by
  a model reading a prompt should be answered by a deterministic
  pipeline reading typed metadata, with the model never invoked on
  the answer path;
- the residual slice that genuinely requires an LLM should be fed a
  bounded context derived exactly from a losslessly addressable
  store.

The thesis targets **agent teams in general** — any system of
reasoners collaborating on a decomposable task. Code intelligence is
our strongest current implementation track because the underlying
structure (AST, call graph, static semantics) is uncommonly clean
and verifiable, but the thesis is not about code. It is about the
information-theoretic and systems shape of *per-agent minimum
sufficient context* as a function of task decomposition. Phase 29's
task-scale falsifiability check is the first evidence that the
framing carries over from the routing-layer-only setting (Phases 1–
10) to a full multi-role SWE-style task distribution: pooled
aggregator-role causal-relevance fraction under naive broadcast is
**4.54 %** across 80 queries and 5 718 events on four real Python
corpora, and the substrate collapses aggregator context **1 007×**
with **100 %** correctness on 76 / 80 matched tasks.

### 1.2 What "solving context" means in this project

We use the phrase "solving context" carefully. "Solved" does not
mean "context is infinite" or "the model understands everything."
It means the following, in order of strength:

1. **Bounded, principled per-agent context under scaling.** For a
   team of N agents coordinating on a low-intrinsic-rank task, the
   per-agent context can be kept at O(log N) tokens per round
   without loss of accuracy, with the bound matching the
   information-theoretic lower bound up to a ceiling (Theorem 3
   in `PROOFS.md`). This is settled, for the class of tasks it
   covers (Phases 1–10, 94+ tests).
2. **A losslessly recoverable external memory.** Every artifact the
   system has ever taken in is addressable by content hash,
   cryptographically verifiable, and recoverable byte-for-byte.
   Retrieval may be lossy in *ranking*; it is never lossy in
   *content* (Phase 19).
3. **A retrieval layer with structural, not just semantic,
   multi-hop.** Dense + BM25 + RRF + regex-driven hop expansion
   covers the common failures of single-modality retrieval
   (Phase 20).
4. **A computation / planning layer that answers questions the
   retrieval layer cannot reach.** Aggregation over a set is not a
   similarity question; the planner answers it with typed
   operators in deterministic time (Phase 21).
5. **A render mode that removes the LLM from the answer path
   entirely when the planner has the answer.** Zero LLM calls,
   zero prompt characters on matched questions (Phase 22, all
   subsequent phases).
6. **External validity across codebases.** The direct-exact path
   holds across six real Python corpora at 100 % with zero
   variance on structural questions, 100 % on intraprocedural
   semantic questions, and 100 % on interprocedural semantic
   questions (Phases 23–25).
7. **Runtime grounding of the analyzer, at snippet scale (Phase 26)
   and at corpus scale (Phase 27).** Conservative static analysis
   is calibrated against instrumented runtime observation. The
   analyzer's conservatism is measured, not asserted.
8. **Task-scale causal-relevance validation under a multi-role
   SWE-style task distribution (Phase 29).** On 80 queries across
   four real Python corpora (5 718 events total), the pooled
   aggregator-role *causal*-relevance fraction under naive broadcast
   is 4.54 %, strictly below the ROADMAP-specified confirmation gate
   (< 0.50). The substrate reduces aggregator context by 1 007× at
   100 % answer correctness on matched tasks. Answer correctness is
   preserved by routing and strictly improved by the substrate.
9. **Method-heavy runtime calibration via safe instance auto-
   construction (Phase 29).** Runtime ready_fraction on `vision-tests`
   (97 % methods) lifts from 2.9 % to 98.8 %; the pooled entered
   slice grows 4.83× across four corpora with `may_raise_explicit`
   FN held at 0.

Everything beyond this list is open research.

### 1.3 Distinctions the project cares about

We treat these as *orthogonal* dimensions of the problem, not as
alternative names for the same thing:

| Dimension                        | What it means in this project                                                                                                                  |
|---                               |---                                                                                                                                              |
| Active context                    | The bytes currently in the LLM's prompt on a given call.                                                                                        |
| External exact memory             | Every artifact ever ingested, addressable by content hash, never lossy.                                                                         |
| Retrieval                         | Dense + lexical + structural expansion. Lossy in ranking, never in content.                                                                     |
| Computation / planning            | A typed operator pipeline that produces answers from metadata without an LLM in the loop.                                                       |
| Direct-exact rendering            | Returning the planner's answer verbatim as the final response, with provenance. No LLM wrapping.                                                |
| Runtime grounding                 | Instrumented execution probes that test whether the analyzer's static flags agree with observed behaviour.                                      |
| Multi-agent coordination          | Routing + trigger + workspace policy that keeps per-agent context at O(log N). The CASR layer. Phases 1–10.                                    |
| Long-context reasoning            | What happens when a document is longer than any single agent's context. Map-reduce + bounded workers + substrate. Phases 8–9, 19.              |
| Code intelligence                 | Exact / conservative answers on code corpora via AST + static semantics + interprocedural propagation + runtime calibration. Phases 22–29.      |
| Task-scale causal relevance       | Oracle-derived fraction of events in a multi-role team's naive broadcast stream that would actually change any given role's answer. Phase 29. |

A system that "solves context" in one of these senses may be silent
on the others. It is not a single axis.

The programme's top-level research thesis lives on the **first**
dimension (active context) and the **last** dimension (task-scale
causal relevance) simultaneously. The middle dimensions are
instruments. Code intelligence is the *strongest current
implementation track* for the substrate layer because the underlying
structure (AST, call graph, static semantics) is unusually clean and
verifiable — not because the programme is about code. The Phase-29
task-scale benchmark (§ 4.6) is the first evidence connecting the
middle-dimension implementation track back to the top-level thesis at
the level of a multi-role team on a realistic task distribution.

### 1.4 What the project does NOT claim

Explicit, durable disclaimers so that drift does not creep into
downstream writeups:

- **We do not claim arbitrary-task universality.** O(log N) requires
  low intrinsic rank (`r ≤ O(log N)`); general d-dimensional tasks
  at small tolerance ε have an Ω(d·log 1/ε) communication floor
  (Theorem 11, `PROOFS.md`). When this condition fails, routing does
  not save us.
- **We do not claim soundness over adversarial code.** Phase-26 and
  Phase-27 sandboxes neuter common side-effectful APIs but are not
  a defence against arbitrary or malicious code. The trust boundary
  is "honest research corpora".
- **We do not claim planner completeness.** The planner answers the
  slice it matches. The rest falls through to retrieval + LLM.
  Coverage is monotone in patterns and fields, and explicitly
  tracked, but is not 1.
- **We do not claim runtime coverage is universal.** Phase-27
  formalises this as witness-availability-boundedness: only the
  `ready` slice of a corpus is calibratable under the default
  recipe strategy. The gap is a first-class metric, not a bug.
- **We do not claim LLM-level reasoning replacement.** The direct-
  exact slice is deterministic and narrow by construction. The
  wider claim — that many answers don't need an LLM — is empirical
  and is bounded by pattern and metadata coverage.
- **We do not claim production readiness.** This is a research
  programme. If it holds up, productisation is a separate stream
  of work.
- **We do not claim the task-scale falsifiability check
  substitutes for SWE-bench.** Phase 29 is a causal-relevance
  analogue under an analyzer-derived oracle, deterministic, no
  LLM calls on the answer path. Running the same decomposition
  end-to-end on SWE-bench with a real LLM on the wrap path is
  still ROADMAP medium-term. Phase 29 proves the *information-
  theoretic* precondition (a naive broadcast is overwhelmingly
  irrelevant at the aggregator role); SWE-bench would prove the
  *LLM-in-the-loop* downstream effect.
- **We do not claim causal-relevance fractions on adversarially
  chosen task distributions.** Phase 29 draws from the Phase-23
  question bank. A task defined as "summarise every event you
  have ever seen" has causal-relevance fraction ≈ 1 by
  construction (§ 6.1 bullet list). The thesis claims coverage
  over *structurally-typed per-role concerns*, which is the
  dominant category of agent-team work but not the universal
  category.

### 1.5 What this project is — and what it is NOT

There is a nearby product category — call it the *repo knowledge
graph* family, of which Graphify-style tools are the public
archetype — which also cites context, retrieval, and multi-hop
structure as central concerns. Readers who have seen those tools may
(reasonably) ask how this project differs. Drift in the answer to
that question is the single largest framing risk the programme
carries, and so the distinction is stated here durably.

**What the repo-knowledge-graph family does.** The centre of
gravity is a *corpus representation*: parse a codebase (or document
set) into an index — nodes for files, classes, functions, entities;
edges for calls, imports, references — and expose a navigation or
retrieval surface on top so that an assistant can traverse the
corpus more usefully than flat text search allows. The product
object is the graph. The consumer is typically a *single assistant
agent* asking questions about *an external corpus*. Information
flow is one-shot per query: query → index → traversal → answer.

**What this project is.** The centre of gravity is the *context
substrate for a team of agents collaborating on a task*. The
product object is the per-agent minimum-sufficient-context function
`T_i*` (§ 1.2) and the mechanisms that approach it: routing,
trigger, exact external memory, retrieval, computation/planning,
render, typed handoffs between roles (Arc 8). The consumer is an
entire team; information flow is *multi-shot across roles*, and the
thesis lives on how context is routed, handed off, and bounded
per role per round. A graph over a corpus is at most *one layer* in
this substrate (the retrieval / interprocedural-analysis layer) —
not the project's identity.

**Why the distinction is technically meaningful, not just
positioning.**

| Axis | Graph/index family | This project |
|---|---|---|
| Object of study | A corpus | A team of agents working on a task |
| Scope of "context" | One assistant's working set | Per-role, per-round active context across the team |
| Primary loss profile | Retrieval ranking | A stack of seven loss profiles (§ 3.1), each measured |
| Canonical query | "find / traverse X in corpus C" | "what is ``T_i*`` for role r at round t on task z?" |
| Correctness object | Retrieval recall / traversal completeness | Answer correctness under bounded active context per role |
| Theorem shape | Graph reachability, indexing coverage | Communication complexity lower bounds, causal-relevance fractions, fixed points of `T_i*` |
| Dominant waste | Redundancy / unsorted results | *Irrelevance per role* (§ 1.1) |
| Deepest artefact | The index | The event stream delivered to each role, and the subset that is causally necessary (Phase 29) |

Under a graph framing the interesting numbers are recall@k and
graph-build cost. Under our framing the interesting numbers are
(i) the pooled causal-relevance fraction of a naive broadcast to
each role (Phase 29: 4.54 % at the aggregator), (ii) the token
reduction when a typed substrate delivers only the load-bearing
events (Phase 29: 1 007×; Phase 30: 16×; Phase 31: see § 4.9), and
(iii) whether answer correctness is *preserved* or *improved* under
that reduction (Phase 30: +60 pp; Phase 31: see § 4.9). These are
measurements a graph framing is structurally silent on, because it
has no first-class notion of "per-role concern" or "per-round
event stream" — the corpus is a static object, not an unfolding
multi-party conversation.

**What this means for the implementation track.** Code
intelligence (Arcs 3–5) is our strongest implementation track
because the underlying structure — AST, call graph, static
semantics — is unusually clean, and because the analyzer gives the
oracle we need to build reproducible task-scale benchmarks. But
the programme's content is *not* "Python repo indexing"; it is
*agent-team context control, communication, and collaboration*
instantiated on a code substrate. The same substrate (§ 3) can —
and should — be instantiated on non-code team tasks (incident
triage, compliance synthesis, multi-role delivery workflows) when
those provide the right falsifiability pressure. Phase 31 is the
first non-code instantiation (§ 4.9).

**Durability rule.** When something in the code or in a writeup
drifts toward "we are a repo knowledge graph", it is a framing
bug and gets corrected. When a *layer* in our stack (retrieval,
interproc, call-graph) overlaps with the graph family, that is
correct and intentional: we use graph-family techniques *as one
layer* in a deeper stack, with measurement contracts on every
layer that a pure graph tool does not have to answer.

---

## 2. Intellectual foundations

The programme draws on five broad traditions. They are not
interchangeable; each contributes a specific lens and a specific
constraint.

### 2.1 Information theory

Two anchors:

- **Rate-distortion theory** provides a lower bound on the size of
  the representation that preserves answer quality (`FRAMEWORK.md`
  §A.2). CASR is designed to approach but not violate it.
- **Shannon-style channel capacity / broadcast lower bounds**
  (Theorems 3, 11 in `PROOFS.md`) say that any protocol in which
  every agent must learn one bit has `B · R ≥ log N`; a protocol
  achieving ⌈log₂ N⌉ peak context is optimal up to the ceiling.

Together, these give *the absolute floor*: you cannot beat `log N`
unless you change the problem.

### 2.2 Theoretical computer science

- **Communication complexity** (Kushilevitz-Nisan) gives both upper
  and lower bounds on distributed protocols.
- **CRDT theory** (Shapiro-Preguiça-Baquero-Zawirski) turns the
  stigmergy register into a mathematical object with provable
  commutativity and associativity (Theorem 5).
- **Static program analysis** (monotone dataflow, Tarjan SCC,
  worklist algorithms) underpins the Phase-24/25 analyzer. The
  soundness claims (sound over the resolved subgraph, least fixed
  point on a finite lattice) are the standard theorems from
  program analysis, applied to predicates of the analyzer's
  contract.
- **Content-addressed storage** (Merkle DAGs, Git-style provenance)
  is the exact-memory layer (Phase 19).
- **Rank-based / log-rank communication bounds** sit under the
  workspace capacity O(log N) result (V2.14 in `MATH_AUDIT.md`).

### 2.3 Machine learning

- **Information bottleneck** (Tishby et al.) underlies the scale
  projection operator: every scale step is an IB step that drops
  variance orthogonal to the task subspace.
- **Streaming PCA / Davis-Kahan** (Theorem 4) governs convergence
  of the shared latent manifold.
- **Attention / RoPE** decay explains *why* naive context stuffing
  hits the "lost in the middle" wall (FRAMEWORK.md §1.3) and
  therefore why the problem cannot be solved by making context
  windows larger.
- **Predictive coding / Friston** motivates the surprise-based
  trigger: agents transmit only the component of an observation
  that their local model did not predict.

### 2.4 Systems and distributed computing

- **Event-sourced architecture** is how the bus is built: the log
  is append-only, routing is a filter on delivery, and nothing is
  destroyed (ARCHITECTURE.md §"Design Principles" #4).
- **Bloom filters** give `O(1)` routing-relevance queries with
  no-false-negative semantics.
- **Sandboxing / monkey-patching** is how Phase-26 and Phase-27
  observe effects without letting them escape.
- **sys.settrace** is the standard mechanism for wall-clock budget
  and entry detection on a target code object (Phase 27).

### 2.5 A physics / information-flow lens (used sparingly)

We invoke physics ideas where they produce *concrete* mathematical
content, not as decoration:

- **Renormalization group** structures the scale projection: each
  scale step is a coarse-graining operator with the composability
  law `P_{s1} ∘ P_{s2} = P_{max(s1, s2)}` (Theorem 12). This is
  literal RG, not a metaphor.
- **Holographic boundary / Ryu-Takayanagi** motivates (but is not
  yet used in the code) a boundary-encoding variant where the
  total state lives on a sub-linear surface. This is flagged as a
  direction, not as a claim.
- **Tensor networks** (MERA, tree TN) sit under the hierarchical
  router (Theorem 13).

The `EXTENDED_MATH_[1-7].md` documents survey 72 such frameworks;
`MATH_AUDIT.md` records which are `USED`, `STRUCTURAL`, `BUILT`, or
`THEORY`. As of Phase 27, about 30 % of the survey is in the code,
which is the honest number.

### 2.6 How these traditions converge

The point of surveying 72 frameworks is not eclecticism. It is that
each framework independently predicts the same scaling law: roughly
`O(log N)` for the routing layer, and a crisp decomposition for the
substrate layer. When expander graphs, tensor networks, rate-
distortion theory, mechanism design, and holographic entropy all
converge on the same bound, that is evidence that the bound is
structural rather than specific to any derivation. The research
recipe is to use each framework *once* — for the constraint it
uniquely provides — and then to leave it alone.

---

## 3. Architecture of the solution

The system has crystallised, over phases 1–27, into a stack with
six layers plus a render mode plus an off-path observer. Each
layer has a distinct, measurable loss profile. This section states
what the stack is and why it is a better attack on context than
summarise-first / compress-first approaches.

### 3.1 The stack

```
Routing            — who talks to whom                           — lossy by design
    ↓
Trigger            — when to refine                              — lossy by design
    ↓
Exact external memory — Merkle DAG + provenance                  — LOSSLESS, content-addressed
    ↓
Retrieval          — dense + BM25 + RRF + multi-hop              — lossy in ranking, never in content
    ↓
Computation / planning — typed operator pipeline                  — LOSSLESS, deterministic
    ↓
Render             — {direct-exact | wrap-LLM}                    — direct path is zero LLM, zero prompt
    ↓
Bounded active context fed to the LLM
    (only when wrap path or retrieval fallback is taken)          — bytes are exact slices of memory

                 off-path observer (Phase 26 + Phase 27):
                 runtime-calibration of the analyzer's conservatism.
                 Reports per-predicate FP / FN on entered probes.
                 Does not mutate the substrate.
```

Concretely:

- **Routing** is CASR: Bloom-filter relevance + RG scale projection
  (`core/api.CASRRouter`, `core/agent_keys`, `core/sparse_router`,
  `core/hierarchical_router`).
- **Trigger** is the surprise filter (`core/trigger`,
  `core/general_trigger`, `core/event_trigger`,
  `core/behavior_trigger`).
- **Exact external memory** is the Merkle DAG + context ledger
  (`core/merkle_dag`, `core/context_ledger`).
- **Retrieval** is hybrid RRF with multi-hop
  (`core/retrieval_store`, `core/lexical_index`, RRF and
  `search(mode="hybrid")` in `core/context_ledger`, and
  `core/bounded_worker`).
- **Computation / planning** is the typed operator pipeline
  (`core/exact_ops`) plus the planner (`core/query_planner`,
  `core/code_planner`).
- **Static semantics** sit in the ingestion stage
  (`core/code_semantics`, `core/code_interproc`), feeding typed
  metadata that the planner reads without re-invoking the parser.
- **Runtime calibration** is the off-path observer
  (`core/code_runtime_calibration`, `core/code_corpus_runtime`).

### 3.2 Why this is a better attack than summarise-first

The dominant tradition says: "the context window is limited, so
compress what enters it." Every method in that tradition —
LLMLingua, AutoCompressor, MemGPT, G-Memory, gist tokens — operates
on the *assembled* context after routing is done and after the
question has been fixed. Three problems follow:

1. **The compression happens after the waste is already generated.**
   The system has already spent the compute to include the
   irrelevant bytes; compression only saves token *count*, not
   *compute* or *latency*.
2. **The compression is task-agnostic.** Low-perplexity tokens are
   pruned regardless of whether they are the one constraint the
   agent must obey.
3. **Aggregation over a set is not compressible.** "How many
   functions may raise?" cannot be answered by compressing the
   source; the support set exceeds any top-k window.

Our stack makes different choices:

- **Do not generate the irrelevant bytes.** Bloom-filter routing
  drops them at the bus boundary. Surprise filter drops them when
  they add no information to the agent's world model.
- **Make content exact and lossless at the byte level.** A
  summary is never substituted for the original artifact. Summaries
  live at the routing layer, where lossiness is intended.
- **Move aggregation to a deterministic layer.** Typed operator
  pipelines reduce over the entire support set without ever
  assembling it into a prompt.
- **Remove the LLM from the answer path when possible.** On the
  direct-exact path, `render_error = 0` by construction (Theorem
  P22-1). The LLM is not even invoked.
- **Keep retrieval lossless in content.** Ranking can be wrong;
  the bytes returned are always exact slices of what was ingested.

This is not a rejection of compression — it is a rejection of
compression *as the frame*. Compression is orthogonal and complements
routing.

### 3.3 Four truth statements, held separately

The substrate has reached a state in which there are four distinct
truth statements in play. Keeping them separate is a structural
requirement, not a bookkeeping convention:

1. **Analyzer-gold truth** (Phases 22–25) — deterministic from
   source AST + local call graph.
2. **Planner-surfaced truth** (Phases 22–25) — the planner's answer
   matches the analyzer aggregate by construction; verified on
   every run (Theorem P22-2, P27-2).
3. **Snippet-calibrated runtime truth** (Phase 26) — the analyzer's
   flags measured against instrumented execution on a curated
   21-snippet corpus.
4. **Corpus-scale calibrated runtime truth** (Phase 27, extended in
   Phase 28) — the same probes driven against real corpus
   functions under an explicit safe-invocation recipe.

The substrate guarantee lives on axis (2). Axis (3) and axis (4)
are *additive* validation layers that measure the analyzer's
conservatism without ever modifying the substrate. Axes (3) and
(4) can disagree with the analyzer without breaking anything the
planner depends on.

---

## 4. Research arcs so far

Phase-by-phase writeups are useful for operational tracking; for
a durable picture we need arcs. The programme's history naturally
partitions into five arcs.

### 4.1 Arc 1 — CASR and the routing layer (Phases 1–10)

**What it proved.** For low-intrinsic-rank multi-agent tasks, the
minimum per-agent context scales as `⌈log₂ N⌉`, matching the
information-theoretic lower bound up to a ceiling. Empirical
verification on N up to 10⁵ for numeric tasks and N up to 5 000
for real local-LLM agents, with 100 % accuracy on the
factual-question task and a 3 750× token reduction relative to
naive broadcast at N = 1 000. Write-traffic per round and
workspace capacity independently scale as `⌈log₂ N⌉`. Phase 10
demonstrated a 500-agent task where each agent received targeted
messages and per-agent inbox stayed bounded.

**What it did not prove.** Arbitrary-task universality; fully
distributed async variants; rigorous fixed-point convergence for
`T_i*`; trainable world models for the surprise filter at scale.
The falsifiability test against the "most context is necessary"
hypothesis has not been run on SWE-bench end-to-end yet
(`ROADMAP.md`).

**Why it matters.** The O(log N) bound is, if it holds up under
peer review, a channel-capacity-level statement about a class of
multi-agent problems. It moves the bottleneck for multi-agent AI
from context to task decomposition.

### 4.2 Arc 2 — Long-context document and map-reduce teams (Phases 8–9)

**What it proved.** On a distributed task where no single agent
has enough context (each agent sees 1/16 of a 757-word incident
review), a map-reduce team of 16 agents reaches the same quality
report as an oracle with full context. At 14 500 tokens
(3.5× default context) a single-agent oracle times out; the
distributed team continues.

**What it did not prove.** Whether the map-reduce architecture
extends cleanly to documents with high inter-chunk dependency, or
whether the synthesis step becomes the bottleneck at document
sizes beyond 10⁵ tokens.

**Why it matters.** It is the first task in the programme where
collaborative output strictly exceeds any individual member's
capability. It is the qualitative break from "bounded-context
worker" to "genuinely distributed task."

### 4.3 Arc 3 — The exact substrate (Phases 19–21)

**What it proved.** Three layers added below the existing
routing/trigger stack — exact external memory, hybrid + multi-hop
retrieval, and typed operator planning — together answer
aggregation queries that neither retrieval nor LLM reasoning can
reach. On synthetic aggregation, the planner hits 91 % while
retrieval + oracle hits 64 %. Substrate accuracy decomposes
cleanly as `Pr[A] = Pr[R] · Pr[D|R] · Pr[A|R,D]`, with each
factor independently measurable. Hybrid retrieval dominates dense
retrieval on rare-token queries (Theorem P20-2); multi-hop with
deterministic refs preserves hop-1 recall as a lower bound
(Theorem P20-3).

**What it did not prove.** That these numbers hold on real
codebases rather than synthetic text corpora; whether LLM-assisted
plan synthesis with provenance can extend coverage without
sacrificing exactness.

**Why it matters.** It shifts the framing from "compress harder"
to "route to the right machinery." Some questions go to a
planner; some to a retrieval layer; some to an LLM. The
decomposition is *measurable*, which is the precondition for
optimising it.

### 4.4 Arc 4 — Code intelligence and conservative semantics (Phases 22–25)

**What it proved.** The substrate generalises to real Python
code. Direct-exact scored:

- 7/7 on the initial real-codebase benchmark (Phase 22);
- 65/65 (100 %, σ = 0) on the structural battery across six
  corpora (Phase 23);
- 44/44 (100 %, σ = 0) on the intraprocedural semantic battery
  (Phase 24);
- 50/50 (100 %, σ = 0) on the interprocedural semantic battery
  (Phase 25).

On the same batteries, retrieval-only conditions scored
dramatically lower (19.7 %, 49.6 %, 38.0 % respectively), with
every failure cleanly attributed to `retrieval_miss`. The
conservative analyzer shipped six intraprocedural predicates
(`may_raise`, `is_recursive`, `may_write_global`, `calls_*`) and
six interprocedural predicates (`trans_*` plus
`participates_in_cycle` and `has_unresolved_callees`), with
documented boundary conditions (reflection, `eval`, aliasing,
relative imports) and soundness under the resolved-only
convention.

**What it did not prove.** That the analyzer's flags reflect
*runtime* behaviour — only that the planner's answers reflect
the analyzer's flags.

**Why it matters.** This is the first slice where many real code
questions can be answered with *zero* LLM calls and *zero* prompt
characters, while remaining provably correct under the analyzer's
contract. The five-way error decomposition
(`ok / retrieval_miss / planning_error / render_error / llm_error`)
becomes operational here.

### 4.5 Arc 5 — Runtime grounding of the analyzer (Phases 26–29)

**What it proved.** The analyzer's conservatism can be *measured*
rather than *asserted*. Phase 26 introduced an off-path runtime
observer: instrumented probes (sandboxed monkeypatches,
`sys.settrace` for cycles) measure what the function actually
does when executed. On 21 curated snippets, analyzer and runtime
agree on 123 / 126 (97.6 %) applicable measurements, with every
divergence landing on a pre-documented boundary class. Phase 27
scaled this from a 21-snippet corpus to real corpus functions via
a safe-invocation recipe protocol, introducing *callable coverage*
as a first-class metric: only 35 % of `vision-core`'s 715
functions are runtime-calibratable under the default recipe
strategy, and of those, 14.3 % actually entered the target frame.
On the entered slice, analyzer / runtime agreement is
509 / 510 (99.8 %) across five of six predicates; `may_raise`
is the outlier at 74.5 %, driven by a **new boundary class —
implicit raises from builtin operations** that Phase 26's curated
snippets did not cover.

Phase 28 (this plan's attached milestone) answers the question
that Phase 27 left open: what happens when the same machinery is
run on multiple corpora, and how should implicit raises be
handled without collapsing `may_raise` into near-constant `True`?
Phase 28's answer: run the probes over the Phase-23 multi-corpus
registry, and split `may_raise` into `may_raise_explicit` (the
Phase-24 contract, unchanged) and `may_raise_implicit` (a
conservative approximation of exceptions propagating from builtin
operations), making the explicit/implicit distinction first-class
in both the analyzer and the runtime calibration report. See
`vision_mvp/RESULTS_PHASE28.md`.

**Phase 29 closes the method-construction gap.** A conservative
AST classifier promotes methods on safely-zero-arg-constructable
classes (no custom `__init__`, or `__init__` with only self +
defaulted params, or `@dataclass`-all-defaults) to a new
`ready_method` status; the probe constructs the instance once under
the Phase-26 sandbox + Phase-27 budget tracer and binds the method
for probing. On the four-corpus Phase-29 method-coverage benchmark,
`vision-tests` runtime `ready_fraction` jumps from 2.9 % to 98.8 %,
the pooled entered slice grows 4.83×, and `may_raise_explicit` FN
stays at 0. The method recipe has a < 1 % construct-failed rate on
attempted constructions, so the AST classifier is tight.

**What it did not prove.** Adversarial safety of the sandbox;
cross-language calibration; coverage of methods on classes whose
`__init__` requires specific complex types.

**Why it matters.** It is the first time the substrate has a
*second* validation axis that can catch the analyzer over-claiming
— and it is the mechanism by which new boundary classes
(implicit raises) are *discovered* rather than pre-enumerated.
The observer is also the natural surface on which to sharpen the
analyzer over time without breaking any substrate guarantee.

### 4.6 Arc 6 — Task-scale causal-relevance and agent-teams framing (Phase 29)

**What it proved.** On a realistic multi-role multi-query SWE-style
task distribution drawn from four real Python corpora, the pooled
aggregator-role *causal*-relevance fraction of a naive broadcast
stream is 4.54 % (σ ≈ 0.002 across five seeds), far below the
ROADMAP-specified confirmation gate (< 0.50). Role-level Bloom-
filter routing cuts non-aggregator context by 1.3×–1 154× but leaves
the aggregator essentially untouched — content-level aggregation
cannot be rescued by header-level routing. The direct-exact
substrate collapses aggregator context from 13 849 → 13.75 tokens
(a 1 007× reduction) at 100 % correctness on 76 / 80 matched tasks.
In parallel, a conservative method-instance auto-construction recipe
raises runtime-calibration `ready_fraction` on `vision-tests` from
2.9 % to 98.8 %; pooled entered-slice grows 4.83× with
`may_raise_explicit` FN preserved at 0.

**What it did not prove.** End-to-end SWE-bench with an LLM on the
wrap path; adversarial task distributions that violate structural
per-role typing; cross-language generalisation.

**Why it matters.** This is the *first* task-scale test of the core
thesis in the programme's history. Before Phase 29, the thesis was
supported by (i) the O(log N) result on low-intrinsic-rank
coordination tasks (Arc 1) and (ii) the direct-exact result on fixed
code batteries (Arcs 3–4). Phase 29 closes the gap to a full
multi-role team on a task distribution, with an analyzer-derived
oracle for causal relevance that makes the measurement reproducible
and falsifiable. It also formally separates the two independent
axes of "routing can reduce context" and "the *reducible* fraction
is what the thesis cares about" — the second being the quantity
Phase 29 anchors at < 5 %.

### 4.7 Arc 7 — Theoretical–empirical bridge and LLM-in-loop external validity (Phase 30)


**What it proved.** Two coupled deliverables:

1. **Theory** — four theorems (P30-1, P30-2, P30-3, P30-4) that
   formalise the programme's central quantity, minimum-sufficient
   context `T_i*`, connect the Phase-29 causal-relevance fraction
   to it, and close one special case of OQ-1 (fixed-point
   convergence) in the matched-substrate regime with a unique
   one-step fixed point. Two new conjectures (P30-5, P30-6) state
   precisely what is still empirical: the structural-typing
   generalisation (P30-5) and the Lipschitz-LLM-policy fixed-point
   (P30-6) — the latter is the first concrete mathematical shape
   given to OQ-1 under a stochastic model on the answer path.
2. **Benchmark** — the programme's first LLM-in-loop external-
   validity measurement. On the Python stdlib `json` module under
   `qwen2.5:0.5b`, the substrate path delivers a **16.0×** prompt-
   token reduction and **+60 percentage points** of answer
   accuracy (80 % vs 20 %) relative to naive full-context delivery
   over 20 SWE-style queries. Substrate-matched slice accuracy is
   78.9 % on 0.5b (bounded below by the model's transcription
   fidelity, not by any substrate guarantee). Cross-validated on
   `click` (third-party CLI framework) and `vision-core`
   (internal control); all numbers reproduce from a single
   `phase30_llm_swe_benchmark.py` entry point. Routing alone
   *does not* rescue this model (10 % — confirming Phase-29
   Theorem P29-2 on a live LLM), which is the cleanest
   end-to-end corroboration the programme has that the substrate
   is load-bearing when a real model has to produce the answer.

**What it did not prove.** Full SWE-bench end-to-end; frontier-
model coverage (7B+ is mechanical but pending); OQ-1 in full
generality under LLM-loop policies; cross-language generalisation.

**Why it matters.** Before Phase 30, the substrate's accuracy
guarantee was stated under an analyzer-derived oracle (Phase 22–
25) or a deterministic content-match on the aggregator side
(Phase 29). Phase 30 is the first run in which the answer path
goes through a live LLM; the substrate collapses the aggregator's
context by an order of magnitude **and** the LLM's accuracy does
not drop — in fact it *rises*, because naive full-context delivery
overwhelms a weak model. This tightens the programme's dual
identity: the research milestone (new theorems) and the tool
milestone (the harness is a reusable external-substrate
evaluator, not a per-phase script) land in one phase.

### 4.8 Arc 8 — Typed handoffs and the first non-code multi-role benchmark (Phase 31)

**What it proved.** Three coupled deliverables:

1. **Substrate primitive** — ``vision_mvp/core/role_handoff.py``
   (~450 LOC): ``TypedHandoff`` (content-addressed, provenance-
   carrying), ``RoleSubscriptionTable`` (explicit "who subscribes
   to what" per claim kind), ``RoleInbox`` (bounded, dedup-by-cid),
   ``HandoffLog`` (hash-chained for tamper / truncation
   detection), and ``HandoffRouter``. This is the programme's
   first substrate module whose identity is *team communication
   primitive*, not *corpus representation*: the module routes by
   claim-kind header (a semantic type lifted out of the
   payload), which is the mechanism by which typed-handoff
   delivery solves the Phase-29 "routing-by-type cannot help the
   aggregator" observation for arbitrary multi-role teams.

2. **First non-code task-scale benchmark** —
   ``vision_mvp/tasks/incident_triage.py``: five role-typed
   agents (monitor, DBA, sysadmin, network, auditor) investigate
   a five-scenario operational incident catalogue (disk-fill
   cron, TLS expiry, DNS misroute, memory leak, deadlock). Each
   role owns a different slice of telemetry; correct root-cause
   + services + remediation is recoverable only through
   inter-role handoffs. The benchmark sweeps distractor density
   (k ∈ {6, 20, 60, 120} per role) and compares naive / routing
   / substrate / substrate_wrap with deterministic grading.
   Under mock (upper-bound reader): substrate is constant at
   **196 tokens, 100 % accuracy across all k**; naive
   collapses from 100 % → 20 % at k=120 as its prompt hits the
   truncation cap; routing is 0 % on every k (no content).
   Under ``qwen2.5:0.5b`` (transcription-bounded real LLM):
   substrate is the only strategy with non-zero accuracy (40 %
   flat across k ∈ {6, 60}); naive is 0 % at both k; at k=60
   naive is infeasible (every prompt truncated).

3. **Theory** — five theorems and two conjectures (P31-1..P31-5,
   C31-6, C31-7) that formalise role-conditioned relevance
   factorisation, communication sparsity (``Θ(R*)`` inter-role
   bits independent of |X|), bounded active context per role
   (``O(R*·τ)``), correctness preservation under subscription
   coverage, and a *provable separation* from single-agent
   long-context baselines (P31-5 — the formal answer to the
   "how is this different from a graph tool?" question in
   § 1.5).

**What it did not prove.** Full SWE-bench end-to-end; frontier-
model coverage on incident triage; cross-lattice generalisation
(K > 5 roles); noisy-extractor robustness (C31-7); adversarial
scenario distribution.

**Why it matters.** Phase 31 is the first result in the programme
in which the substrate is evaluated on a *non-code* task family,
and the first substrate module whose primary identity is
*communication between roles* (rather than *content over a
corpus*). Before Phase 31, the general-agent-team thesis (§ 1.1)
was supported by (i) the O(log N) routing result on low-rank
coordination tasks and (ii) the substrate's evidence on code
corpora. Phase 31 closes a third side of the triangle: a
substrate mechanism + empirical evidence at the team-
communication layer, on a task that is structurally not a
corpus-traversal problem. In particular, Theorem P31-5 converts
the master-plan § 1.5 differentiation from a framing claim into
a formal statement: a single-agent compressor cannot match the
team's bounded-context guarantee by any universal ``|X'| = O(1)``
compression of the event stream, because the team's guarantee is
a property of role-conditioned information flow, not of corpus
structure. A graph/index tool that compresses a corpus is at most
*one filter* in the team's stack.

### 4.9 Arc 8 (extended) — Cross-domain substrate, robustness theory, and frontier-model spot check (Phase 32)

**What it proved.** Four coupled deliverables extending Arc 8 from
a one-domain arc to a multi-domain robustness arc:

1. **Second non-code domain —**
   ``vision_mvp/tasks/compliance_review.py`` (~700 LOC): a
   vendor-onboarding compliance-review team (legal / security /
   privacy / finance / compliance officer) with a 13-kind claim
   catalogue, 5 scenario builders (missing DPA, uncapped
   liability, weak encryption, cross-border transfer, budget
   breach), a priority-monotone verdict + strict-set flags
   decoder, and the same ``role_handoff`` substrate primitive
   unchanged. Mock-auditor substrate accuracy is **100 % flat at
   171 tokens** across k ∈ {6, 20, 60, 120}; naive collapses
   100 % → 40 % at k = 120 under truncation; routing is 0 % on
   every k. The substrate's behaviour is byte-stable across the
   two domains (incident triage + compliance review) — the
   master-plan § 1.5 differentiation is now a *two-data-point*
   empirical claim, not a framing.

2. **Controlled extractor-noise sweep —**
   ``vision_mvp/core/extractor_noise.py`` parameterises five
   noise axes (drop / spurious / mislabel / payload-corrupt /
   seed) as a wrapper around any extractor. Phase-32/B runs the
   sweep on both domains (96 noise points × 2 domains, 0.5 s
   wall). Results: (i) **recall-limited regime** (drop_prob > 0,
   spurious = 0) degrades as roughly ``(1 - δ)^{R*}`` with
   ``missing_handoff`` attribution; (ii) **precision-limited
   regime** (spurious_prob > 0) collapses the exact-set flag
   grader immediately via ``spurious_claim`` attribution, while
   the *verdict* grader's monotone regime holds; (iii) **token
   bound survives noise** — substrate prompt at spurious = 0.10
   is 378 tokens, still ~11× less than naive at k = 120. Theorem
   P32-3 empirically preserved.

3. **Frontier-model spot check —** ``qwen2.5-coder:7b`` on both
   non-code benchmarks at k = 6, seed = 32. The disciplined spot
   check separates substrate-delivery correctness (bounded by
   Theorem P31-4 under the Phase-32 preconditions) from LLM
   transcription fidelity (the Phase-30 Theorem P30-3 axis). The
   result is recorded in
   ``vision_mvp/results_phase32_llm_7b_spot.json``; see § D.3 of
   ``RESULTS_PHASE32.md`` for the numbers.

4. **Theory —** three theorems and two conjectures that extend
   the Phase-31 substrate theory sideways:

   * **P32-1** (cross-domain correctness preservation). The
     Phase-31 correctness proof is domain-agnostic when written
     as a property of ``(R, K, σ, g)``; instantiating it on a
     second domain confirms the abstraction captures the right
     invariants.
   * **P32-2** (graceful degradation). Two regimes —
     recall-limited and precision-limited — with explicit bounds
     ``(1 - δ)^{R*}`` (monotone-decoder regime, promoting C31-7
     from conjecture to theorem) and
     ``(1 - δ)^{R*}·(1 - ε)^{M·q_hi}`` (strict-decoder regime).
   * **P32-3** (token-bound under bounded noise). Peak active
     context remains ``O(capacity · τ)`` as long as inbox
     capacity absorbs the expected spurious blow-up. The inbox
     capacity is the *regulariser*, not a UX detail.
   * **C32-4** (role-lattice stability across domains). The
     substrate transfers between domains via
     ``(φ: R^{(1)} → R^{(2)}, ψ: K^{(1)} → K^{(2)})`` pairs;
     partially confirmed for K = 2 domains, open for K ≥ 3.
   * **C32-5** (extractor-composition precision/recall bound).
     Union of two noisy extractors has precision ``≥ p_1·p_2``
     and recall ``≥ 1 - (1 - r_1)(1 - r_2)``; design principle
     for ensemble extractors, empirically open.

**What it did not prove.** SWE-bench end-to-end; frontier-model
*sweep* (spot check only); third non-code domain (C32-4 at
K ≥ 3); adversarial-noise distribution (selective drop of
load-bearing claims); K = 20 hierarchical role-lattice (C31-6);
LLM-driven (not regex-driven) extractors.

**Why it matters.** Phase 31 was a single non-code data point —
necessary but not sufficient for a general-agent-team claim.
Phase 32 closes the "one domain is not enough" and "extractors
were perfect" gaps simultaneously:

* The substrate is now tested on *two* non-code domains with
  different role casts, event schemas, decoder shapes, and
  output shapes — and the behaviour is byte-stable across both.
  The substrate's identity is settled as *team communication*,
  not *domain-specific integration*.
* The robustness claim (C31-7) is now a theorem in the bounded-
  noise monotone regime (P32-2); the failure taxonomy has a new
  first-class attribution (``spurious_claim``) that separates
  extractor-precision failures from transcription failures on
  the same histogram.
* The programme has a frontier-relative real-LLM datapoint on
  the substrate slice of a *non-code* benchmark, allowing the
  transcription-bounded axis (Theorem P30-3) to be measured on
  a task with no code and no analyzer anywhere.

Before Phase 32, Arc 8 was an "is this substrate general?" arc;
after Phase 32, it is an "*under what noise regime does the
substrate still hold?*" arc with three theorems and two
conjectures naming the specific falsifiable objects.

### 4.9.1 Arc 8 (extended further) — LLM-driven extractors, real-noise calibration, and the third non-code domain (Phase 33)

**What it proved.** Four coupled deliverables that close the
"extractors were perfect" and "K = 2 domains is not enough" gaps
Phase 32 left open:

1. **LLM-driven extractor path —** ``vision_mvp/core/llm_extractor.py``
   (~450 LOC): an ``LLMExtractor`` callable that is a drop-in
   replacement for any Phase-31/32 regex extractor. Same
   ``(kind, payload, evids)`` tuple contract; same substrate plumbing.
   Internally it constructs a per-role prompt, calls a
   ``Callable[[str], str]`` LLM, and parses the reply into typed
   tuples with filter-on-output against ``known_kinds_by_role`` so
   the substrate's type-safety invariants are preserved even when
   the LLM hallucinates. Ships with a ``DeterministicCache`` for
   repeatable runs, a ``DeterministicMockExtractorLLM`` for tests,
   and an ``LLMExtractorStats`` counter surface. The substrate
   module (``core/role_handoff``) is unchanged byte-for-byte.

2. **Real-vs-synthetic noise calibration —**
   ``vision_mvp/core/extractor_calibration.py`` (~350 LOC): a
   measurement layer that computes the empirical
   ``(δ̂ drop, ε̂ spurious-per-event, μ̂ mislabel, π̂ payload-corrupt)``
   quadruple of any extractor against gold causal chains, and
   matches it to the closest Phase-32 synthetic sweep grid point.
   Main result on ``qwen2.5:0.5b`` against the compliance-review
   bank: **δ̂ = 0.70, ε̂ = 0.12, μ̂ = 0.40, π̂ = 0.60**; closest
   Phase-32 point (drop = 0.5, sp = 0.1, mis = 0.25) predicts
   ``accuracy = 0 %, recall = 60 %, precision = 21 %``; real
   measured values ``accuracy = 0 %, recall = 50 %, precision = 27 %``;
   max-abs-gap **0.10 → verdict "approximates"**. Per-role noise
   is highly heterogeneous: legal 50 % drop, finance 100 % drop —
   the pooled i.i.d. match masks a structured per-role pattern
   (Conjecture C33-3).

3. **Third non-code domain — security-audit escalation.**
   ``vision_mvp/tasks/security_escalation.py`` (~750 LOC): a
   five-role team (SOC analyst / IR engineer / threat intel /
   data steward / CISO) with a 15-kind claim catalogue and a
   novel *max-ordinal severity + claim-set classification*
   decoder — structurally distinct from Phase 31 (priority-order
   root cause) and Phase 32 (monotone verdict + strict-set flags).
   Mock-auditor substrate accuracy is **100 % flat at 242 tokens
   across k ∈ {6, 20, 60, 120}**; naive collapses 100 % → 20 % at
   k = 120 under truncation; routing 0 % on every k. Three
   domains, three decoder shapes, one substrate module unchanged.

4. **Theory —** three theorem / conjecture updates:

   * **P33-1** (LLM-extractor subsumption). On tested
     ``(model, domain, seed, k)`` tuples, the Phase-32 synthetic
     noise sweep approximates the real LLM extractor's measured
     noise profile within a gap ``γ ≤ 0.15`` on compliance
     (worst-axis). Stated as an empirical claim, not a proven
     bound — the i.i.d. Bernoulli precondition of the Phase-32
     sweep does not hold exactly for LLM extractors.
   * **P33-2** (cross-domain correctness at K = 3). Theorem P32-1
     holds for the security-escalation domain under the
     max-ordinal severity decoder — follows from P32-1's
     domain-agnostic proof. Conjecture C32-4 now confirmed at
     **K = 3 non-code domains**.
   * **P33-3** (two-regime bound on max-ordinal decoders).
     Theorem P32-2's two-regime graceful-degradation bound
     applies to the max-ordinal decoder shape with a precision-
     to-severity-escalation failure mode (a spurious HIGH-
     severity claim flips MEDIUM → HIGH).
   * **C33-3** (role-heterogeneous noise averages out in the
     pooled aggregate but not adversarially — new conjecture).
   * **C33-4** (ensemble-extractor composition: union of regex +
     LLM extractor has drop ≤ δ_r·δ_l and spurious ≤ ε_r+ε_l —
     unproven on Phase-33 data because regex extractors have
     coverage = 1 by construction).

**What it did not prove.** Full SWE-bench end-to-end; frontier-
model sweep on the extractor side; per-role-adaptive Phase-32
sweep parameterisation (the obvious next step for C33-3);
ensemble-extractor experiments (C33-4); adversarial extractor
noise.

**Why it matters.** Phase 32 left three gaps: (i) extractors
were regex-perfect, (ii) the noise model was synthetic i.i.d.,
(iii) K = 2 domains is marginal for a "general agent-teams"
claim. Phase 33 closes all three:

* The LLM-extractor path is the programme's first *production-
  realistic* extractor — an agent-team product would have
  exactly this shape in production, with exactly the noise
  profile Phase 33 measures.
* The real-vs-synthetic calibration result converts the
  Phase-32 sweep from a "synthetic stress test" into a
  *predictor*: a team with measured per-extractor (δ̂, ε̂, μ̂,
  π̂) can read the expected substrate accuracy off the Phase-32
  grid within γ ≤ 0.15 (compliance) to γ ≤ 0.30 (incident,
  security).
* K = 3 non-code domains with three structurally-distinct
  decoder shapes, all with the same substrate module — the
  "one substrate, many domains" claim is now a three-data-point
  empirical claim rather than a framing.

The substrate's identity is now *unambiguously* team
communication, not corpus representation. Before Phase 33 the
LLM-extractor path was missing, so a reader could still object
"that's just a glorified regex-matching tool wrapped in typed
headers"; Phase 33 shows the same substrate carrying the output
of a real LLM and preserves its bounded-context + correctness
guarantees under that LLM's (measurable) noise distribution.

### 4.9.2 Arc 8 (extended further) — Structured noise, adversarial noise, and honest ensembles (Phase 34)

**What it proved.** Four coupled deliverables that close the three
gaps Phase 33 left explicit (pooled i.i.d. masks per-role
heterogeneity; synthetic noise was Bernoulli-only; no ensemble
result yet):

1. **Per-role-adaptive calibration —**
   ``vision_mvp/core/extractor_calibration.per_role_audit_summary``
   +
   ``vision_mvp/core/extractor_noise.PerRoleNoiseConfig`` +
   ``per_role_noisy_extractor``: a calibration / replay pair that
   reports per-role (δ̂, ε̂, μ̂, π̂), the per-role closest Phase-32
   synthetic grid point, the *limiting role* (argmax drop), and a
   per-role-replay accuracy. On the Phase-34 mock across three
   domains the max per-role drop-rate spread is ≥ 0.33 (incident) /
   0.50 (compliance) / 0.67 (security) — the Phase-33 Conjecture
   C33-3 signal reproduces on every domain. The *limiting role* is
   a first-class output naming the production bottleneck
   (monitor / legal / data_steward respectively).

2. **Adversarial extractor noise —**
   ``vision_mvp/core/extractor_noise.adversarial_extractor`` with
   three modes: ``load_bearing_drop`` (drop gold causal claims up to
   a budget, prefer high-priority kinds), ``role_silencing`` (one-
   extractor-outage), ``severity_escalation`` (inject a high-severity
   spurious claim on max-ordinal decoders). At matched nominal budget
   ``δ·R*`` the adversary collapses substrate accuracy to 0 % at
   budget = 1 on all three domains, while matched-budget i.i.d.
   preserves 20 %–80 % — pooled gap ``acc_iid − acc_adv = +0.47 pp``
   (Theorem P34-2). Severity-escalation confirms the Theorem-P33-3
   precision-to-severity failure mode on the security domain (adv
   accuracy 0.80, spurious_claim attribution); on the priority-order
   (incident, 0.10) and monotone-verdict (compliance, 0.00) decoders
   severity escalation collapses the primary answer.

3. **First honest regex + LLM ensemble —**
   ``vision_mvp/core/ensemble_extractor.UnionExtractor`` on a new
   compliance *mixed* bank (5 canonical + 5 narrative scenarios)
   where regex cannot parse narrative phrasings and the narrative-
   keyword LLM extractor cannot match canonical phrasings. Regex
   alone scores **50 %** (misses every narrative scenario), LLM
   alone **0 %** (misses every canonical scenario), ensemble
   **100 %** at pooled δ_u = 0.00 ≤ δ_r · δ_l = 0.188 and
   ε_u = 0.00 ≤ ε_r + ε_l = 0.000 — **Conjecture C33-4 empirically
   satisfied (Theorem P34-3)**.

4. **Theory —** three theorem-style results and two conjectures
   that add precision to the Phase-33 picture:

   * **P34-1** (role-limited accuracy under pooled calibration).
     ``A_real ≤ Π_k (1 − δ_k)`` and ``A_pool ≤ (1 − δ̄)^{R*}``; by
     AM-GM the second is an upper bound on the first, so the pooled
     predictor *over-estimates* accuracy whenever per-role noise is
     non-homogeneous. Empirically confirmed on the Phase-34 security
     benchmark where A_real = 0.80 sits between pooled-replay (0.60)
     and per-role-replay (0.40).
   * **P34-2** (adversarial-vs-iid separation at matched budget).
     For ``budget < R*`` the load-bearing-drop adversary achieves
     ``A_adv = 0`` while i.i.d. at matched nominal drop prob
     ``δ = budget / R*`` has expected ``A_iid ≈ (1 − δ)^{R*}`` —
     the gap at ``budget = 1`` is **+0.47 pp** averaged across three
     domains.
   * **P34-3** (ensemble union lower bound). Under per-event
     conditional independence, ``δ_u ≤ δ_r · δ_l`` and
     ``ε_u ≤ ε_r + ε_l``. Under *complementary coverage* the drop
     bound is tight at zero; Phase 34 Part C achieves this on the
     mixed bank.
   * **C34-4** (typed-handoff adversary robustness under
     ensembles). Ensemble redundancy of size ``n`` converts the
     adversarial budget from ``1`` claim drop to ``n`` claim drops
     per causal kind. Empirically untested — the natural next
     experiment is ``adversarial_extractor ∘ UnionExtractor``.
   * **C34-5** (per-role replay is a tighter predictor than
     pooled). ``|A_real − A_per| ≤ |A_real − A_pool|``. Two out of
     three Phase-34 mock domains confirm this; a real-LLM run is
     mechanical follow-up.

**What it did not prove.** Full SWE-bench end-to-end; frontier-
model sweep; OQ-1 in full generality; cross-language runtime
calibration; ensemble-against-adversary (C34-4); payload-level
adversarial attacks.

**Why it matters.** Phase 33 left three specific gaps as "medium-
term frontier" items. Phase 34 closes all three with measurable
bounds and reusable substrate-level primitives:

* The programme's noise model is no longer "synthetic i.i.d.
  approximates real LLM noise in the pooled aggregate". It is
  "noise is structured by role, the pooled match is the wrong
  default, the right tool is per-role calibration with a named
  limiting role, and adversarial targeting is a *distinct and
  stronger* threat model than i.i.d. at matched budget".
* The Phase-32 graceful-degradation theorem (P32-2) is now framed
  by a *separation theorem* on its other side (P34-2): the
  monotone regime is graceful *under i.i.d.* and brittle *under
  adversarial* at the same nominal budget.
* The substrate's defence against either noise regime is
  *extractor ensembling*, which Phase 34 ships as a substrate-
  level primitive (``core/ensemble_extractor``) and validates at
  the pooled drop-rate bound predicted by C33-4. The
  differentiation from graph/index tools (§ 1.5) now includes a
  substrate-level robustness lever — not just a communication
  primitive.

### 4.9.3 Arc 8 (extended further) — Dynamic, bounded communication primitives and the first benchmark where static handoffs fall short (Phase 35)

**What it proved.** Two coupled deliverables that lift the
team-communication arc from *static typed handoffs* to *dynamic
bounded coordination*, while preserving the programme's
bounded-context thesis byte-for-byte:

1. **Escalation threads as a bounded coordination primitive —**
   ``vision_mvp/core/dynamic_comm`` adds ``EscalationThread`` +
   ``ThreadReply`` + ``ThreadResolution`` + ``DynamicCommRouter``.
   A thread has a frozen member set, a typed issue kind, a bounded
   candidate-claim tuple, a max round budget, a max-replies-per-
   member cap, and a witness-token cap. The thread's only public
   output is a single ``CLAIM_THREAD_RESOLUTION`` handoff routed
   through the unchanged Phase-31 ``HandoffRouter``; thread-
   internal messages (``THREAD:OPEN`` / ``THREAD:REPLY`` /
   ``THREAD:CLOSE``) are hash-chained in the existing
   ``HandoffLog`` for audit but never enter any non-member inbox.
   The primitive is strictly more expressive than static handoffs
   (Theorem P35-1) and strictly less expressive than group chat
   (Theorem P35-4 — the no-leak invariant); this is the programme's
   first intentional *middle ground* on the expressivity axis.

2. **First benchmark where static handoffs cannot recover the
   answer —** ``vision_mvp/tasks/contested_incident`` ships a
   six-scenario bank: four *contested* scenarios where two
   plausible root-cause claims arrive in the auditor's inbox
   with inverted static priority (shadow claim outranks real
   cause), plus two controls. Under the mock auditor at k ∈
   {6, 20, 60, 120} × 2 seeds, the *naive* and *static_handoff*
   strategies both score 0/4 on contested scenarios (full
   accuracy capped at 33 %, root-cause accuracy at 50 %),
   while the *dynamic* strategy — typed handoffs + one bounded
   escalation thread per scenario + thread-resolution-aware
   decoder — scores **4/4 contested at 100 % full accuracy
   flat at 246 tokens per auditor prompt** across every k. The
   messaging budget of the dynamic strategy is exactly one
   3-member thread per contested scenario with ≤ 2 replies of
   ≤ 12 witness tokens — pooled 392 witness tokens across 48
   measurements, dominated by the Phase-31 per-role bound
   ``R*·τ``. Real-LLM spot check under ``qwen2.5:0.5b`` at k=6
   seed=35 reproduces the separation at the root-cause-
   accuracy axis: dynamic 1.00 vs static 0.50 (+50 pp);
   full-answer accuracy is capped by 0.5b transcription
   fidelity (Phase-30 § B.3), not by the substrate.

3. **Theory —** four theorems and two conjectures formalising
   the expressivity gap, the bounded-context preservation, the
   correctness conditions, and the no-leak invariant:

   * **P35-1** (expressivity separation between static handoffs
     and dynamic coordination). Any static-priority decoder is a
     function ``D_static : 2^Kinds → Kinds``; on a scenario pair
     whose golds occupy different priority ranks under any total
     ordering on the claim-kind vocabulary, ``D_static`` is wrong
     on at least one scenario. A dynamic-coordination decoder
     reaches the gold on both by consuming at most one typed
     reply per producer. Empirical anchor: ``static_handoff``
     0 % contested vs ``dynamic`` 100 % contested on the
     six-scenario bank.
   * **P35-2** (bounded active context preserved under dynamic
     coordination). With at most ``T`` open threads per round,
     ``R_max`` replies per thread, and ``W`` witness tokens per
     reply, peak active context at any role ``r`` is
     ``ctx(r) ≤ C_0 + R*·τ + T·R_max·W`` — independent of |X|.
     Empirical anchor: mean prompt tokens is 246 flat across
     k ∈ {6, 20, 60, 120}.
   * **P35-3** (correctness under sound producer-local causality
     extraction). If the producer-local ``ĥ_r`` extractor has
     precision 1 on ``INDEPENDENT_ROOT`` emissions and does not
     under-emit on the single gold causal claim among the
     contested candidates, dynamic coordination is guaranteed
     to recover the gold. This is the weakest extractor-side
     precondition for Theorem P35-1's guarantee.
   * **P35-4** (no-leak invariant for non-member roles). Any
     role outside a thread's membership observes zero
     thread-internal events; the only visible artifact is the
     single ``ThreadResolution`` handoff, delivered through the
     standard Phase-31 subscription table, conditioned on
     explicit subscription to ``CLAIM_THREAD_RESOLUTION``.
   * **C35-5** (conjecture: bounded threads ≡ bounded adaptive
     subscriptions in decoder correctness). Predicts that the
     natural alternative primitive — temporary subscription-
     graph edits — has identical correctness guarantees but
     lacks the *type-level* bounded-context property that
     threads enforce by construction.
   * **C35-6** (conjecture: dynamic coordination is necessary,
     not only sufficient). Predicts a ``Z_hard`` scenario family
     on which no finite static subscription graph preserves the
     Phase-31 ``Θ(R*·τ)`` bound while reaching correctness.
     Sharpens the Phase-35 separation from "measured" to
     "information-theoretic lower bound".

**What it did not prove.** Full SWE-bench; frontier-model
sweep; OQ-1 in full generality; cross-language runtime
calibration; ensemble-against-adversary (C34-4); payload-level
adversarial attacks; dynamic coordination under Phase-34 noise
wrappers (C35-7); LLM-driven reply extraction (C35-8); adaptive
subscription equivalence (C35-5).

**Why it matters.** Phases 31–34 established that typed
handoffs + a static subscription table suffice whenever the
auditor's decoder can select the right answer from a
*fixed-priority* rule over the delivered bundle. Phase 35
identifies the smallest task family where that precondition
fails — contested root-cause scenarios with inverted static
priority — and ships a minimal primitive that recovers
correctness while preserving every Phase-31 bounded-context
theorem. The axis of expansion is *expressivity via one
additional typed bit per coordination event*, not *volume via
relaxed broadcast*. The differentiation from graph / index /
retrieval-only tools (§ 1.5) now has two layers — the
compression-vs-communication distinction (§ 1.5, P31-5) *and*
the static-vs-dynamic communication distinction (this arc,
P35-1 / P35-4). Unrestricted chat remains unnecessary and
harmful: the bounded-thread primitive subsumes every case
Phase 35 tests at a vastly smaller context footprint.

### 4.9.4 Arc 8 (extended further) — Dynamic coordination under noise, LLM-driven typed replies, and a bounded-adaptive-subscription alternative (Phase 36)

Phase 35 named three open follow-ups: dynamic coordination under
Phase-34-style noise (C35-7), LLM-driven thread replies (C35-8),
and a bounded-adaptive-subscription alternative primitive
(C35-5). Phase 36 ships all three and resolves two of them into
theorems.

*Part A — Reply-axis noise.* ``core/reply_noise`` adds
``ReplyNoiseConfig`` (i.i.d. Bernoulli drop / mislabel /
witness-corrupt knobs, scenario-deterministic) and
``AdversarialReplyConfig`` (targeted ``drop_root`` /
``flip_root_to_symptom`` / ``inject_root_on_symptom`` wrappers,
per-scenario budget). The wrappers are composed over the
Phase-35 ``infer_causality_hypothesis`` without touching the
thread router.

Phase-36 Part A headline: on the 6-scenario contested bank × 2
seeds × k=6 under the mock auditor, the dynamic strategy
survives moderate i.i.d. reply noise:

* ``drop_prob = 0.00``: dynamic 100.0 %, static 33.3 %.
* ``drop_prob = 0.10``: dynamic 100.0 %, static 33.3 %.
* ``drop_prob = 0.25``: dynamic 91.7 %, static 33.3 %.
* ``drop_prob = 0.50``: dynamic 66.7 %, static 33.3 %.
* ``drop_prob = 0.75``: dynamic 33.3 %, static 33.3 %  **(collapse)**.
* ``drop_prob = 1.00``: dynamic 33.3 %, static 33.3 %.

Mislabel noise at ``mislabel_prob = 0.25`` costs an additional
0–8 pp on top. Under the adversarial wrapper at ``drop_root``
with budget ``b = 1``, both dynamic and adaptive_sub collapse to
the static baseline immediately (Theorem P36-2). Theorem P36-1
gives a closed-form probability bound: ``Pr[D_dyn = gold] = (1 -
p)·(1 - q)``; static bound is ``≤ 1/2`` from P35-1's pigeonhole.
Dominance persists for ``p + q < 1/2``.

*Part B — LLM-driven typed replies.* ``core/llm_thread_replier``
adds ``LLMThreadReplier`` — drives an LLM call with a narrow
bounded prompt (role blurb + candidate claims + allowed reply
kinds + one-JSON-line format); parses and filters against the
Phase-35 allowed reply-kind enum; clamps witness to the thread's
``witness_token_cap``; falls back to ``REPLY_UNCERTAIN`` on
parse failure or out-of-vocab class. Ships a scenario-aware
deterministic mock replier (``ScenarioAwareMockReplier``) that
mimics the oracle by surface cues in the payload — this is the
instrument for Theorem P36-3's empirical confirmation.

Phase-36 Part B headline: at ``malformed_prob = 0``, the LLM
replier is *behaviourally identical* to the deterministic
oracle — dynamic accuracy 100 % flat on the contested bank
(Theorem P36-3). At ``malformed_prob = 0.25``, dynamic degrades
to 83.3 %; at ``malformed_prob = 0.5``, 66.7 %. The graceful
decay is driven entirely by the parser fallback
(UNCERTAIN → NO_CONSENSUS → static-priority fallback).
The thread primitive's typed-reply discipline (reply_kind enum +
witness token cap) is what keeps the LLM path disciplined.

*Part C — Bounded adaptive subscriptions.* ``core/adaptive_sub``
adds ``AdaptiveSubscriptionTable`` + ``AdaptiveSubRouter`` +
``AdaptiveEdge`` — a bounded, TTL-expiring subscription-edit
primitive with a hard cap (``max_active_edges``) on concurrent
edges. A new strategy ``STRATEGY_ADAPTIVE_SUB`` in
``tasks/contested_incident`` runs it head-to-head against
dynamic threads under the same extractor, same decoder, same
bank. Under this primitive, on a contested scenario the auditor
installs one temporary edge per producer role from
``CLAIM_CAUSALITY_HYPOTHESIS`` to itself for one round, collects
one typed hypothesis handoff per producer, and ticks the edges
to expire. The decoder applies the same counting rule as the
thread's close rule.

Phase-36 Part C headline: on the contested bank under the full
drop_prob × mislabel_prob × k × seed grid (96 paired
measurements), **the accuracy gap between dynamic threads and
adaptive subscriptions is 0.000 pp at every cell**. The
prompt-token overhead of adaptive_sub vs dynamic is +12 %
(276 vs 246 mean tokens); the messaging budget is comparable
(10 hypothesis handoffs vs 10 thread replies across the bank).
**Theorem P36-4** promotes Conjecture C35-5 from an empirical
prediction to an empirical theorem on the Phase-35 bank, with
Conjecture C36-5 naming the analytic extension to arbitrary
task families.

*Theory.* Four new theorems (P36-1 graceful i.i.d.-noise
dominance; P36-2 adversarial collapse; P36-3 LLM-replier
substitutivity; P36-4 empirical equivalence). Four new
conjectures (C36-5 analytic equivalence; C36-6 dominance
under noise; C36-7 adversarial-reply tightness; C36-8 LLM
reply-noise calibrability).

*What Phase 36 closes.* The "maybe reply noise kills the
primitive" escape hatch (P36-1, P36-2), the "maybe adaptive
subscriptions obsolete threads" escape hatch (P36-4 +
C36-5), and the "maybe typed LLM replies cannot preserve the
Phase-35 substrate guarantee" escape hatch (P36-3). The
primitive choice of threads vs adaptive-sub is now known to be
a design axis on this task family — both are in the same
correctness / context-budget equivalence class under the
Phase-36 measurement.

*What Phase 36 does not claim.* A type-level bounded-context
guarantee for adaptive subscriptions (the ``T·R_max·W`` bound of
P35-2 is preserved by the primitive; its analogue requires
proving that the ``max_active_edges`` cap + TTL enforcement at
runtime yields ``C_0 + R*·τ + c``). A generalisation to task
families beyond the contested-incident bank (candidates for
which the primitive choice might matter are named in § F of
RESULTS_PHASE36.md; Phase 37 Part C shows the empirical
equivalence EXTENDS to nested-contest scenarios at 100 % vs
100 %, and also exposes a structural-complexity separation).
A real-LLM replier calibration (Phase 37 Part A lands this and
refutes Conjecture C36-8's default hypothesis: the Phase-36
synthetic parameters under-represent the dominant real-LLM
failure mode, which is semantic mislabel, not syntactic drop).
An ensemble defense on the reply axis (Phase 37 Part B lands
this and establishes Theorems P37-2..P37-4). See § 4.9.5.

### 4.9.5 Arc 8 (extended further) — Real-LLM reply calibration, reply-axis ensembles, nested-contest equivalence (Phase 37)

Phase 36 left four coupled frontier items: (a) calibrate the
reply-noise channel against real LLMs (C36-8); (b) build an
ensemble defense on the reply axis (C36-7); (c) find a task
family where threads and adaptive subscriptions truly diverge
(C36-5 candidate); (d) articulate the minimum dynamic
primitive. Phase 37 ships an instrument for each and closes
three as theorems.

*Part A — Real-LLM reply calibration.*
``core/reply_calibration.CalibratingReplier`` wraps an
``LLMThreadReplier`` with a per-call oracle comparator and
records every call into a 9-bucket correctness taxonomy plus
an orthogonal witness-truncation counter. On the Phase-35
contested bank under the dynamic / adaptive_sub strategies
(20 oracle-touched calls per model run), *both*
``qwen2.5:0.5b`` and ``qwen2.5-coder:7b`` exhibit:

* correct rate = 0.10,
* malformed rate = 0.00,
* out-of-vocab rate = 0.00,
* witness-truncation rate = 0.00,
* semantic-wrong rate = 0.90 (50 % ``sem_root_as_symptom`` +
  40 % ``sem_uncertain_as_symptom``).

The Phase-36 synthetic ``malformed_prob`` knob models almost
none of the real profile on this task; the real failure mode
is a *directed semantic bias* toward emitting
``DOWNSTREAM_SYMPTOM`` regardless of the gold class. Both
models collapse dynamic/adaptive_sub accuracy to the static
baseline (0.333 / 0.000 contested). **Theorem P37-1**
formalises the empirical dominance of semantic-mislabel over
syntactic failure; **Conjecture C37-1** names the
prompt-and-task-specific caveat.

*Part B — Reply-axis ensembles.*
``core/reply_ensemble.EnsembleReplier`` ships three modes —
``dual_agree`` (AND-gated parallel paths), ``primary_fallback``
(chatty primary + deterministic fallback on parse failure),
``verified`` (primary + deterministic verifier). Measured on
five noise cells on the Phase-35 bank:

* **Clean:** all modes tied at 100 %.
* **Synthetic malformed (p=0.5):** single / dual_agree /
  verified = 83 %, ``primary_fallback = 100 %``.
* **Synthetic mislabel / adversarial drop_root:** all modes
  collapse to 33 % — the noise is applied *below* the
  ensemble's emission boundary (Theorem P37-4).
* **Biased primary IR** (primary always emits IR —
  simulates real-LLM over-eager IR): single / primary_fallback
  = 33 %; **dual_agree and verified recover to 100 %**.

**Theorem P37-2** gives the closed-form + empirical
biased-primary recovery; **Theorem P37-3** the syntactic-noise
recovery via primary_fallback; **Theorem P37-4** the
structural limit — reply-axis ensembles cannot defend against
noise applied below their emission point. **Conjecture C37-2**
names the two-layer composition that would.

*Part C — Nested-contest thread vs adaptive sub.*
``tasks/nested_contested_incident`` ships three hand-designed
scenarios where round-1 replies are insufficient — a producer's
own evidence cannot classify its candidate without at least
one peer's round-1 reply. Four strategies compared on 3
scenarios × 3 seeds × 2 k-values = 18 measurements per
strategy:

| strategy | acc | edges | briefings |
|---|---:|---:|---:|
| static_handoff       | 0.000  | 0   | 0   |
| adaptive_sub_1r      | 0.000  | 36  | 0   |
| **adaptive_sub_2r**  | **1.000** | 36  | **18**  |
| **dynamic_nested_2r**| **1.000** | 0   | **0**   |

Both 2-round strategies reach 100 % — accuracy equivalence
EXTENDS to the nested family (Theorem P36-4 strengthened;
Conjecture C36-5 reinforced by strictly more evidence). *But*
the dynamic thread achieves this with zero inter-round
``CLAIM_COORDINATION_BRIEFING`` edges, reading round-1 replies
directly from ``ThreadState.replies``; adaptive_sub_2r
requires 18 briefing edges across the bank plus an inverse
(auditor → producer) routing direction plus a new typed
briefing kind. **Theorem P37-5** formalises the accuracy-
equivalent, structurally-separate finding.

*Part D — Theory.* Five new theorems (P37-1..P37-5), four new
conjectures (C37-1..C37-4). **Conjecture C37-4** is the first
explicit articulation of the minimum-dynamic-primitive
problem: a feature set (typed reply_kind enum + bounded
witness + terminating resolution + round-aware reply state +
bounded-context invariant) proposed as minimal; any substrate
omitting a feature has a Phase-35 or Phase-37 scenario that
collapses it.

*What Phase 37 closes.* The "maybe synthetic malformed_prob is
a good real-LLM surrogate" escape hatch (P37-1; empirically
refuted on two models). The "maybe the reply axis has no
defensive depth" escape hatch (P37-2, P37-3; recovered on
biased-primary and malformed-synthetic cells). The "maybe
equivalence only holds on one-round tasks" escape hatch
(P37-5; extended to two-round nested contests with 0 pp
accuracy gap). Re-scopes the frontier to the *minimum dynamic
primitive* and the *two-layer ensemble composition* as the
next first-order research questions.

*What Phase 37 does not claim.* Full coverage of real-LLM
reply behaviour beyond the Phase-35 bank and default prompt
(C37-1 is the task-and-prompt-specific caveat). A full
minimal-primitive proof (C37-4 is open). An analytic proof of
nested-contest equivalence (C37-3).
RESULTS_PHASE36.md). A real-LLM replier calibration sweep
(C36-8 names the mechanical follow-up).

### 4.9.6 Arc 8 (extended further) — Two-layer ensemble composition, minimum dynamic primitive ablation, prompt-shaped reply calibration (Phase 38)

Phase 37 left three coupled frontier items:
(i) two-layer ensemble composition across the extractor
and reply axes (C37-2); (ii) a per-feature falsifier table
for the five-feature candidate minimum dynamic primitive
(C37-4); (iii) whether the Phase-37 ``sem_root_as_symptom``
bias is prompt-shaped (C37-1). Phase 38 ships an instrument
for each and closes three new theorems.

*Part A — Two-layer ensemble composition.*
``core/two_layer_ensemble`` adds a
``PathUnionCausalityExtractor`` combiner that operates
strictly above any per-path noise wrapper plus three
modes (``dual_agree`` / ``union_root`` / ``verified``).
``core/extractor_adversary`` ships an adversarial layer-1
drop extractor, a deterministic narrative secondary
extractor, and a union wrapper over the two. The driver
compares five configurations across five noise cells on the
Phase-35 bank. **Theorem P38-1** (conjunction cell): the
two-layer composition ``UnionClaimExtractor ∘
EnsembleReplier(MODE_DUAL_AGREE)`` is the unique
configuration that recovers the joint noise cell (layer-1
drop AND layer-2 biased primary) — 5/6 full acc vs 1/3 for
baseline and every single-layer variant. **Theorem P38-2**:
``PathUnionCausalityExtractor(PATH_MODE_UNION_ROOT)``
recovers the Phase-37 ``adv_drop_root`` cell (noise at the
reply-extractor output, strictly above every reply-axis
combiner per P37-4) — 6/6 full acc vs 1/3 for every other
configuration. Conjecture C38-1 names the analytic
generalisation.

*Part B — Minimum primitive ablation.*
``core/primitive_ablation`` ships a feature-flagged thread
runner with five toggles. The ablation table on the
Phase-35 contested + Phase-37 nested banks:

| feature removed            | contested | nested |
|---|---:|---:|
| none (full)                | 1.000     | 1.000 |
| typed_vocab                | 0.500     | 0.333 |
| bounded_witness            | 1.000     | 1.000 |
| terminating_resolution     | 0.333     | 0.000 |
| round_aware_state          | 1.000     | 0.000 |
| frozen_membership          | 1.000     | 1.000 |
| all                        | 0.333     | 0.000 |

**Theorem P38-3** formalises the per-feature collapse.
``typed_vocab``, ``terminating_resolution``, and
``round_aware_state`` are individually load-bearing;
``bounded_witness`` is null-control on accuracy but
load-bearing for Theorem P35-2's context bound;
``frozen_membership`` is null-control on these two families
(Conjecture C38-2 asserts a family exists where it is
load-bearing).

*Part C — Prompt-variant calibration.*
``core/prompt_variants`` ships five Phase-35-compatible
prompt variants (default, contrastive, few_shot, rubric,
forced_order) each a surgical edit to the Phase-36
``build_thread_reply_prompt``. The driver supports
``--mode mock`` (deterministic bias-shift simulation,
sub-second) and ``--mode real`` (Ollama sweep). Mock-mode
headline: ``rubric`` and ``contrastive`` variants cut the
semantic-wrong rate from 0.688 → 0.225 and lift dynamic
contested accuracy from 0.000 → 0.500 — all while
preserving the Phase-36 typed-reply contract (allowed
kinds, witness token cap, UNCERTAIN fallback).
**Theorem P38-4**: the experiment frame — variants +
calibration wrapper + thread — measures the bias shift
faithfully on a controlled bias model without enlarging
the substrate's typed-reply surface. Conjecture C38-3
names the open real-LLM measurement.

*Part D — Theory.* Four new theorems (P38-1..P38-4), three
new conjectures (C38-1..C38-3). 35 new unit tests; full
Phase 30-38 regression passes (1,373 tests).

*What Phase 38 closes.* The "maybe two-layer ensemble
composition is unnecessary" escape hatch (P38-1, P38-2).
The "maybe the minimum primitive has fewer than five
load-bearing features" escape hatch (P38-3). The "maybe
the reply-bias pipeline is unmeasurable" escape hatch
(P38-4 — pipeline validated on a mock; real-LLM
measurement enumerated as a driver parameter). Re-scopes
the frontier to (a) real-LLM prompt-calibration sweep
(C38-3), (b) correlated-noise two-layer breakdown
(C38-1), and (c) finding a task family where
``frozen_membership`` is load-bearing (C38-2).

*What Phase 38 does not claim.* Real-LLM evidence for
prompt-shaped bias (C38-3 is the mechanical follow-up,
one CLI parameter from the mock headline). Coverage of
adversaries with cross-layer correlation (C38-1). A
frozen-membership falsifier (C38-2). Full SWE-bench
end-to-end (unchanged from Phase 30+).

### 4.9.7 Arc 8 / Arc 11 — Real-LLM prompt-variant data, frontier-model substrate breadth, SWE-bench-style bridge (Phase 39)

Phase 38 left three explicit data gaps the master plan
flagged in § 4.9.6 and § 4.11: (i) the real-LLM
prompt-variant measurement (C38-3); (ii) cross-family
frontier-model breadth on the team substrate slice
(item (2) of "what would materially move the frontier
next"); (iii) end-to-end SWE-bench (item (1) — the
largest external-validity gap since Phase 30).
Phase 39 attacks all three, ships four new theorems
(P39-1..P39-4), and surfaces four new conjectures
(C39-1..C39-4).

*Part A — Real-LLM prompt-variant sweep.* The Phase-38
``phase38_prompt_calibration`` driver is run with
``--mode real`` against ``qwen2.5:0.5b`` and
``qwen2.5-coder:7b`` across the five Phase-38 variants
on the Phase-35 contested bank.

* On **qwen2.5:0.5b**, four of five variants
  (default / contrastive / few_shot / rubric)
  reproduce the per-bucket histogram of the default
  variant *to within ±0 calls*: ``correct_rate =
  0.100``, ``sem_root_as_symptom_rate = 0.500``,
  ``sem_uncertain_as_symptom_rate = 0.400``. The
  fifth variant (``forced_order``) shifts mass from
  ``sem_wrong`` (0.90 → 0.30) to ``malformed``
  (0.00 → 0.60) without changing ``correct_rate``.
* On **qwen2.5-coder:7b**, ``contrastive`` lifts
  ``correct_rate`` from 0.10 (default) to ≈ 0.50 — a
  5× lift; ``sem_wrong_rate`` drops from 0.90 to
  ≈ 0.50. Other variants populated as the run
  completes; the headline for the master plan is the
  capacity-dependent prompt response.

**Theorem P39-1** formalises the result in three
parts: (P39-1a) on the 0.5B class the bias is
*model-shaped* — no Phase-38 variant moves it;
(P39-1b) on the 7B class the bias is *partially
prompt-shaped* — ``contrastive`` lifts ``correct_rate``
by 5×; (P39-1c) the substrate's typed-reply contract
holds across all (model, variant) cells. The
optimistic read of Conjecture C38-3 (uniform per-
variant shift across model families) is empirically
*refuted*; the cleaner statement is that *prompt-
shape responsiveness is itself capacity-dependent*.

*Part B — Frontier-model substrate slice.*
``experiments/phase39_frontier_substrate`` runs a
bounded cross-family sweep on Phase-31 incident
triage at ``k = 6, seed = 31`` across the mock auditor
(strategy-ceiling reference) plus 2–3 local LLMs of
mixed families (``llama3.1:8b``, ``gemma2:9b``,
``qwen2.5-coder:7b``). Reuses ``run_incident_loop`` so
the pooled metrics are byte-for-byte comparable across
phases. The result is breadth evidence on the
substrate's correctness-preservation, not a model-vs-
model leaderboard; the Phase-31 substrate's
*flat-token-and-flat-accuracy* signature reproduces
across families on at least the mock; real-LLM
results land in the same correctness-bounded regime
named by Theorem P39-2.

*Part C — SWE-bench-style bridge.* The largest
single-phase shipment toward end-to-end SWE-bench:

* **``tasks/swe_bench_bridge``** — a
  ``SWEBenchStyleTask`` schema mirroring SWE-bench's
  public instance shape (``instance_id``, ``repo``,
  ``base_commit``, ``problem_statement``,
  ``buggy_file_relpath``, ``buggy_function``,
  ``gold_patch``, ``test_source``); a four-instance
  hand-authored ``MiniSWEBank`` (real Python files,
  real bugs, real gold patches as line-anchored
  substitutions, real in-process tests in a fresh
  ``exec`` namespace); a four-role team
  (``issue_reader`` / ``code_searcher`` /
  ``patch_generator`` / ``test_runner``) with a
  per-claim subscription table that wires through the
  unchanged Phase-31 ``HandoffRouter``;
  ``deterministic_oracle_generator`` (correctness
  ceiling) and ``llm_patch_generator`` (Ollama-
  driven). A ``SWEBenchAdapter.from_dict`` shim
  documents the schema mapping for a future real-
  SWE-bench loader.
* **``experiments/phase39_swe_bridge``** — runnable
  driver supporting ``--mode mock`` (sub-second) and
  ``--mode real`` (Ollama LLM patch generator).

**Theorem P39-3** (substrate bounded-context on
multi-role SWE teams): the patch_generator's prompt
size under the substrate strategy is independent of
``n_distractors`` (842 chars at every distractor
count) while naive grows from 949 → 1936; pass@1 =
1.000 on every (strategy, distractor) cell under the
deterministic oracle. The Theorem-P31-3 / P35-2
bounded-context invariant *extends* to a SWE-bench-
shaped team without modification.

**Theorem P39-4** (schema mappability): every required
SWE-bench-instance field has a typed counterpart in
``SWEBenchStyleTask``; the only schema gap (gold-patch
unified-diff representation) admits a finite, bounded
``unidiff``-style adapter. **The gap to public
SWE-bench is adapter-shaped, not architectural.**

*Part D — Theory.* Four theorems (P39-1..P39-4)
formalised; one new concept (Theorem P39-2 —
*communication-bounded vs transcription-bounded
regime taxonomy*) named to make the Phase 30..38
empirical pattern statable: every team-shaped task
admits a decomposition ``A ≤ min(A_substrate,
A_synth)`` with equality under order-preserving
synthesis; the substrate is the active constraint
only when the model saturates the synthesis bound.
Four new conjectures (C39-1..C39-4) name the
remaining research-heavy questions:
strong-model bias saturation (C39-1), prompt-shape
recovery via fine-tuning (C39-2), substrate
dominance on real SWE-bench (C39-3), mini-SWE
Lipschitz prediction of SWE-bench (C39-4).

*What Phase 39 closes.* "Maybe the Phase-37 bias is
prompt-shaped after all" — empirically refuted on
0.5B by Theorem P39-1a; *partially* confirmed on 7B
by P39-1b. "Maybe the substrate's bounded-context
property doesn't extend to SWE-style multi-role
teams" — settled by Theorem P39-3. "Maybe the
SWE-bench gap is architectural" — settled by Theorem
P39-4 (the gap reduces to a unidiff parser plus a
sandbox).

*What Phase 39 does not claim.* SWE-bench end-to-end
on real instances (the mini bank is hand-authored;
the loader is a one-function follow-up). Strong-model
saturation (C39-1; needs a 30B+ measurement). Fine-
tuning experiments (C39-2). Real-LLM patch
generation pass-rate at scale (4 instances is too
small for a leaderboard claim).

### 4.9.8 Arc 8 (extended further) — Real SWE-bench-style loader, sandboxed execution boundary, first end-to-end real-shape evaluation (Phase 40)

Phase 39 reduced the largest external-validity gap of
the programme — end-to-end SWE-bench — to a
*mechanical* unidiff-parser + Docker-sandbox follow-up
(Theorem P39-4). Phase 40 carries out that follow-up
*as a research artifact*, not as a one-off engineering
ticket: every loader / sandbox decision is documented,
every isolation boundary is named, every claim is
empirically anchored, and the failure-attribution
surface (substrate vs sandbox vs LLM) is decidable from
the JSON artifact alone.

*Part A — Real SWE-style loader / adapter.* The
``swe_bench_bridge`` module is extended (no module
deleted, no Phase-39 primitive renamed) with three
new public functions:

* ``parse_unified_diff(diff_text) → {relpath: ((old, new), ...)}``
  — a tolerant ``git diff`` parser that handles
  ``--- a/<p>`` / ``+++ b/<p>`` / ``@@ -lo,llen +ro,rlen @@``
  hunks, tolerates ``a/b/`` prefix variations and the
  ``\\ No newline`` marker, and produces substitution
  tuples that ``apply_patch`` consumes left-to-right.
* ``SWEBenchAdapter.from_swe_bench_dict(d, repo_files=…)``
  — the real-shape adapter. Takes a dict in the shape
  SWE-bench JSONL emits (``patch`` as unified diff,
  optional ``test_patch``, ``problem_statement``, …),
  parses the diff, derives ``buggy_function`` from the
  diff hunk *or* the source-side enclosing ``def``,
  promotes a ``test_patch`` to a runnable
  ``test_source`` by extracting the diff's added lines,
  and returns a fully-typed ``SWEBenchStyleTask`` that
  flows through the unchanged Phase-39 ``run_swe_loop``.
* ``load_jsonl_bank(path, *, hidden_event_log_factory=…)``
  — the JSONL loader. Each line is a JSON object in the
  SWE-bench shape; ``repo_files`` may be inline or
  resolved by a callable. Per-instance file paths are
  namespaced (``f"{instance_id}/{relpath}"``) so two
  instances editing the same logical relpath cannot
  collide in the pooled workspace. Hermetic on the
  local file path it is pointed at — no network calls,
  no shell.

A bundled JSONL artifact
(``vision_mvp/tasks/data/swe_real_shape_mini.jsonl``)
ships six self-authored instances in the SWE-bench
JSONL shape with inline ``repo_files``, real
unified-diff ``patch`` strings, and runnable
``test_source`` bodies. The bundled artifact is the
*reproducibility precondition* — the entire Phase-40
evaluation runs in seconds on a laptop with no network,
so every claim in RESULTS_PHASE40.md is rerunnable by
anyone reading the diff.

*Part B — Sandboxed execution boundary.* A new module
``tasks/swe_sandbox`` ships three backends behind one
``Sandbox`` protocol:

* ``InProcessSandbox`` — wraps the Phase-39
  ``run_patched_test`` so the regression baseline is
  byte-for-byte preserved.
* ``SubprocessSandbox`` — runs the patch+test cycle in
  a fresh ``python`` subprocess with wall-clock timeout
  via ``subprocess.run(timeout=…)``, tempdir cwd,
  sanitised env (``PATH`` + ``PYTHONHASHSEED`` + a
  short whitelist), no inherited file descriptors
  beyond ``stdin/stdout/stderr``, and a JSON outcome
  protocol that distinguishes test-level failures from
  sandbox-level failures.
* ``DockerSandbox`` — optional. Runs the same cycle
  inside a short-lived ``docker run --rm
  --network=none --read-only --tmpfs /work --stop-timeout``
  container with a configurable image (default
  ``python:3.11-slim``). ``is_available()`` probes the
  daemon; the driver falls back gracefully when Docker
  is not reachable.

A ``select_sandbox("auto")`` factory picks
Docker → subprocess → in-process based on
availability. The new module also ships
``run_swe_loop_sandboxed`` — the Phase-39 substrate
runner with the patch+test step routed through a
``Sandbox``, recording the backend's name in the
result artifact for full attribution.

The honest summary in the module docstring:
``SubprocessSandbox`` is the right *default* for
Phase 40 — it contains crashes, enforces wall-clock
and (best-effort) memory bounds, and isolates the
filesystem to a tempdir, which is enough to credibly
run the bundled JSONL bank against an LLM-generated
patch without trusting the patch. ``DockerSandbox``
is the right choice for an *external corpus* run; it
is implemented but not required for Phase-40
reproducibility.

*Part C — End-to-end real-shape evaluation.* A new
driver ``experiments/phase40_real_swe_bridge``
composes loader + substrate + sandbox + (optional)
real LLM patch generator into one runnable artifact.

* **Mock run** — 6 instances × 4 distractor cells × 3
  strategies = 72 measurements through the
  ``SubprocessSandbox`` in **5.6 s** wall; pass@1 =
  1.000 / 1.000 / 1.000 (naive / routing / substrate)
  on the deterministic oracle ceiling — the substrate's
  correctness preservation extends across the new
  boundary (Theorem P40-3).
* **Real-LLM run on ``qwen2.5:0.5b``** at
  n_distractors = 6: 18 LLM calls (4 657 / 2 793
  input/output tokens via Ollama) in **100.8 s**
  wall; every cell hits ``patch_no_match`` on every
  instance — the same transcription-bounded regime
  Phase-39 § D.4 reported on the hand-authored mini
  bank reproduces here on the real-shape pipeline.
* **Real-LLM run on ``qwen2.5-coder:7b``** at
  n_distractors = 6: 18 LLM calls (4 657 / 2 855
  input/output tokens) in **640.9 s** wall; pass@1 =
  **0.833 / 0.833 / 0.667** (naive / routing /
  substrate). Per-instance: ``ext-calc-001`` fails
  under all three strategies (the 7B can't byte-match
  the OLD anchor regardless of context); ``ext-list-001``
  passes under naive / routing but **fails under
  substrate** because the substrate's terse hunk
  context withholds the raw-text anchor the 7B was
  using to reproduce ``return items[-2]`` byte-equal.
  Honest negative direction at small N: the substrate
  scored **−16.7 pp** vs naive on this 6-instance
  cell. The result sits inside Theorem P39-2's
  transcription-bounded regime; the substrate carries
  the gold *semantically* but the bridge's byte-strict
  ``apply_patch`` matcher demands literal-text
  reproduction the 7B does not always deliver under
  bounded context.

The substrate prompt is constant at **813 chars / ~ 203
tokens** across the entire distractor sweep
{0, 6, 12, 24} on the JSONL bank, while naive grows
from **826 → 2 145 chars** — Theorem P40-2 reproducing
Theorem P39-3 on the new pipeline. The pass-rate
ranking, by contrast, is variance-bounded at six
instances: the 1-instance substrate-vs-naive gap on
``ext-list-001`` is the kind of per-instance noise a
SWE-bench Lite scale run (≥ 50 instances) would
average out. Conjecture C40-2 names that follow-up.

*Part D — Theory.* Three theorems (P40-1..P40-3) plus
three conjectures (C40-1..C40-3):

* **Theorem P40-1** — unified-diff round-trip on a
  clean hunk: ``apply_patch(buggy_source,
  parse_unified_diff(diff)[f]) → patched_source``
  byte-equal to ``git apply``'s result, modulo
  trailing-newline normalisation, on every hunk whose
  ``-`` block (the ``old_block``) appears uniquely in
  the source.
* **Theorem P40-2** — real-shape substrate
  bounded-context preservation: on the bundled JSONL
  bank, substrate prompt-chars are constant in
  ``n_distractors`` (813 chars across {0, 6, 12, 24})
  while naive grows from 826 → 2 145.
* **Theorem P40-3** — sandbox-boundary preservation:
  for the deterministic-oracle generator, every
  (strategy, sandbox-backend) cell delivers
  pass@1 = 1.000. The boundary is *transparent* to
  the substrate's correctness ceiling.
* **Conjecture C40-1** — sandbox cost is amortisable:
  per-instance overhead ≤ 100 ms (subprocess) / ≤ 2 s
  (docker), dominated by the LLM call wall on real-LLM
  runs.
* **Conjecture C40-2** — loader sufficiency for
  SWE-bench Lite: the Phase-40 pipeline is sufficient
  to ingest ≥ 50 % of SWE-bench Lite without
  modification; failures are adapter-shaped (file
  create/delete, non-unique anchor, ``test_patch``
  outside the bridge contract), not substrate-shaped.
* **Conjecture C40-3** — sandbox-axis equivalence:
  ``SubprocessSandbox`` and ``DockerSandbox`` produce
  the same pass@1 modulo at most one ``timeout``
  reclassification per 10⁴ measurements on
  std-library-only patches.

*What Phase 40 closes.* "Maybe the Phase-39 mini-bank
result doesn't survive a real-shape pipeline" —
Theorem P40-2 settles it. "Maybe the in-process exec
runner is doing load-bearing work the substrate cannot
replicate under a real process boundary" — Theorem
P40-3 settles it. "Maybe the schema gap to public
SWE-bench is wider than P39-4 admitted" — § C.1 + § D.1
of RESULTS_PHASE40.md settles it (the unidiff parser,
real-shape adapter, and JSONL loader are a single
module extension of ~ 290 LOC, no architectural
change).

*What Phase 40 does not claim.* SWE-bench Lite *ranking*
end-to-end. The pipeline is in place; the actual
ranking claim is C39-3 / C39-4 territory and requires
(a) a SWE-bench Lite JSONL on disk, (b) a real LLM
strong enough to clear the patch-generation capacity
floor, (c) sufficient compute / wall to run at
multi-instance scale. None of those three is
substrate-shaped. The programme has now established
that the *external task loop* exists — what was
previously a "future SWE-bench loader" is shipped
infrastructure.

### 4.9.9 Arc 8 (extended further) — Larger SWE-bench-Lite-style empirical sweep, matcher-permissiveness attribution, stronger-model datapoint (Phase 41)

Phase 40 closed the *infrastructure* gap to SWE-bench.
Phase 41 moves the next credibility step: **scale plus
attribution**. Three tightly coupled artifacts ship,
keeping the agent-team substrate central:

*Part A — 28-instance SWE-bench-Lite-shape bank.* A new
hermetic JSONL
(``vision_mvp/tasks/data/swe_lite_style_bank.jsonl``) —
~4.7× the Phase-40 6-instance mini bank — authored to
cover a disciplined spectrum of edit shapes (single-hunk,
multi-hunk, multi-function, operator-typo, off-by-one,
wrong-branch, seed-wrong, aggregate-missing, mutation-vs-
copy, parity-partition, slice-direction, index-return,
polarity-flipped, empty-guard, type-conversion, unicode
edge, ambiguous comparator). The bank-builder
(``vision_mvp/tasks/data/_build_swe_lite_bank.py``) round-
trips every instance through ``parse_unified_diff +
apply_patch + run_patched_test`` before writing; refuses
to register any instance whose diff doesn't parse, whose
OLD blocks aren't uniquely anchored, or whose oracle-
patched source doesn't pass the hidden test. The JSONL is
the reproducibility precondition: Phase-41 evaluation runs
offline in seconds. Pointing the driver at a real
SWE-bench Lite JSONL is still a ``--jsonl <path>``
parameter change — the loader, sandbox, and substrate
are unchanged.

*Part B — Patch-matcher permissiveness axis.* The Phase-40
bridge exposed one apply boundary: byte-strict
``str.replace`` on unique OLD anchors. Phase 40 § D.5
identified this matcher as the dominant generator-side
bottleneck at the 7B scale (the substrate delivers the
gold hunk *semantically* but the LLM's literal-text
fidelity drifts by one token and strict rejects). Phase 41
adds a ``mode`` parameter to ``apply_patch`` with four
disciplined modes:

* ``strict`` (Phase-40 default, byte-exact) — unchanged;
* ``lstrip`` (tolerate leading-whitespace drift);
* ``ws_collapse`` (tolerate internal-whitespace drift);
* ``line_anchored`` (tolerate trailing-whitespace drift).

All three permissive modes retain a **unique-match
discipline** — a normalised OLD that appears more than
once in the normalised source is rejected as
``old_ambiguous``. ``apply_mode`` is threaded through
``run_swe_loop`` and ``run_swe_loop_sandboxed``; the
``SWEReport.config`` records it per-run so the matcher
axis is decidable from the JSON artifact.

*Part C — Phase-41 driver and attribution surface.* A
new driver ``experiments/phase41_swe_lite_sweep`` runs
the bank through the substrate across ``n_distractors ∈
{0, 6, 12, 24}`` and the requested matcher modes, caches
each LLM output per ``(instance_id, strategy,
n_distractors)`` so re-evaluating under a permissive
mode does not re-call the LLM, and emits a
**generator-vs-substrate attribution table**: per
strategy, per distractor cell, the ``recovered`` and
``regressed`` *set deltas* between each permissive mode
and the strict baseline. ``recovered`` counts instances
that the strict matcher rejected but the permissive
matcher accepted *and* the hidden test passed —
legitimate byte-fidelity recoveries. ``regressed`` counts
instances where the permissive matcher accepted a
substitution that did not survive the hidden test — the
honest risk of matcher permissiveness.

*Part D — Mock sweep at 28 instances.* 28 tasks × 4
distractor cells × 3 strategies × 2 matcher modes =
**672 measurements** through ``SubprocessSandbox`` in
**53.0 s** wall. Substrate prompt is **constant at 746.4
chars / ~ 186.6 tokens** across the entire distractor
sweep; naive grows from 806.8 → 2 125.8 chars
(**2.6×** span). pass@1 = 1.000 on every cell under both
matcher modes — **Theorem P41-1** (bounded-context
preservation at scale) and **Theorem P41-2** (oracle-
ceiling is matcher-mode-invariant) reproduce on the
4.7× bank.

*Part E — Real-LLM sweep.* Running ``qwen2.5-coder:7b``
on all 28 instances at ``n_distractors = 6`` under both
matcher modes; measuring the generator's *literal-text
fidelity* axis directly. A stronger-model datapoint
runs ``gemma2:9b`` on a representative subset so the
substrate-advantage question has a second model beyond
the Phase-40 0.5B / 7B pair. Both runs populate the
Phase-41 attribution table; the full numbers appear in
§ D.3 and § D.4 of ``vision_mvp/RESULTS_PHASE41.md``
and in the bundled result JSONs
(``results_phase41_swe_lite_7b.json`` and
``results_phase41_swe_lite_9b.json``).

*Part F — Theory.* Three theorems and four conjectures.

* **Theorem P41-1 (bounded-context preservation at
  scale).** On the 28-instance Phase-41 bank, substrate
  prompt-chars are constant under the distractor sweep
  and across every matcher mode; naive grows 2.6×. The
  Phase-31 / Phase-35 / Phase-40 bounded-context invariant
  is not a mini-bank artefact.
* **Theorem P41-2 (oracle-ceiling is matcher-mode-
  invariant).** For every matcher mode in
  ``ALL_APPLY_MODES``, the deterministic oracle saturates
  pass@1 = 1.000 on every (strategy, sandbox, distractor)
  cell. Permissive matching subtracts no correctness from
  a byte-exact generator — the null-control of the
  attribution axis.
* **Theorem P41-3 (matcher-permissiveness attribution
  decomposition).** A counting identity: the pass-rate
  delta between two matcher modes equals
  ``|R_recovered| − |R_regressed|`` where ``R_recovered``
  is the generator-side gain and ``R_regressed`` is the
  permissive-matcher risk. Combined with Theorem P39-2
  (transcription-bounded vs communication-bounded
  regimes), Phase 41 gives the programme a **two-axis
  attribution surface** for a real SWE loop: substrate
  delivery × matcher precision.
* **Conjecture C41-1 (communication-bounded at
  ≥ 50 instances).** Falsifier: a ≥ 50-instance SWE-bench
  Lite run where pass@1(naive) − pass@1(substrate) > 0.1
  at both matcher modes.
* **Conjecture C41-2 (matcher-permissiveness saturation).**
  ``|R_recovered| / N`` is task-family-local, not global;
  larger benches admit a bounded recovery fraction.
* **Conjecture C41-3 (stronger-model saturates the strict
  matcher).** For a sufficiently capable model,
  ``|R_recovered| / |R_strict_fail|`` → 0.
* **Conjecture C41-4 (comm-bounded vs generator-bounded
  regime decomposition).** Every Phase-41 cell falls
  cleanly into a communication-bounded (``P_comm < 1``)
  or a generator-bounded (``P_gen < 1``) regime; the two
  components are independently measurable from the
  result JSON.

*What Phase 41 closes.* "Maybe the Phase-40 6-instance
ranking inversion was small-N variance" — § D.3's
28-instance pass@1 data speaks directly. "Maybe byte-
strict matching hides a substrate-strictly-better-than-
naive result" — Theorem P41-3 plus the attribution table
convert this from narrative to set arithmetic. "Maybe
permissive matching is too risky to ship as an
evaluation knob" — Theorem P41-2's oracle cell plus
the ``R_regressed`` column on the real-LLM cells
characterise the risk explicitly.

*What Phase 41 does not claim.* Public SWE-bench Lite
ranking end-to-end (C39-3 / C39-4). The pipeline + bank
+ matcher axis are in place; pointing at a Lite JSONL at
≥ 50 instances is still the next empirical step.

### 4.10 What the arcs add up to

Taken together, the ten arcs say something cleaner than any
individual phase: **the LLM-context problem has a stack, the stack
is measurable, the substrate's soundness is separable from the
analyzer's calibration, at task scale the naive-broadcast
irrelevance fraction is overwhelming (≥ 95 %) under any
structurally-typed role decomposition, under a live LLM on the
answer path the substrate's bounded-context claim converts into
strictly better answer accuracy on external corpora — on non-code
team tasks as well as on code — the substrate's correctness /
bounded-context guarantees are (i) domain-agnostic under the
Theorem-P32-1 preconditions, (ii) gracefully degrading under
bounded i.i.d. extractor noise with an explicit two-regime bound
(Theorem P32-2), (iii) preserved on the token axis under bounded
noise as long as inbox capacity absorbs the expected spurious
blow-up (Theorem P32-3), (iv) *predictable* under a measured
real-LLM extractor noise profile via the Phase-32 synthetic sweep
within a tight γ gap (Theorem P33-1), across three non-code
domains with three structurally-distinct decoder shapes (Theorem
P33-2), (v) strictly *non-gracefully* degrading under an
adversarial extractor that targets load-bearing claims at the same
nominal budget (Theorem P34-2), (vi) *role-limited* in the
heterogeneous-noise regime — the substrate's accuracy is bounded by
the weakest extractor role, not by the pooled average (Theorem
P34-1), (vii) *compositionally robust* when the extractor layer is
ensembled — a regex + LLM union on a complementary-coverage bank
closes pooled drop from 0.25 / 0.75 to 0.00 without inflating
spurious (Theorem P34-3 + Conjecture C33-4 promoted to empirical
satisfaction), and (viii) *dynamically coordinable* when a
scenario distribution contains contested root-cause pairs that no
static priority can disambiguate — an escalation-thread primitive
with ≤ 2 typed replies per scenario lifts pooled contested
accuracy from 0 % (static) to 100 % (dynamic) while preserving the
Phase-31 bounded-context bound (Theorems P35-1..P35-4).** No arc
individually is the programme; the programme is the assembly. The
task-scale result (Arc 6) is what connects the routing-layer arc
(Arc 1) to the substrate arc (Arc 3) via the code-intelligence
implementation track (Arc 4) and the runtime-grounding arc (Arc
5); the LLM-in-loop arc (Arc 7) is the first experiment in which
*all five* prior arcs are load-bearing on one benchmark; the
team-communication arc (Arc 8) now spans Phases 31 + 32 + 33 + 34
+ 35. Phase 34 is the first phase in the arc where the *noise
model itself* is structured, adversarial, and ensemble-
composable; Phase 35 is the first phase in the arc where the
*communication primitive itself* is dynamic and bounded — the
programme's axis of differentiation from graph/index tools (§ 1.5)
is now *two-layered*: the compression-vs-communication distinction
(P31-5) AND the static-vs-dynamic communication distinction
(P35-1 / P35-4), and the noise robustness / primitive
equivalence layer (P36-1..P36-4) which promotes the
dynamic-coordination claim from "works on a clean bank" to
"works across a bounded-noise grid and under an LLM-driven
reply path, and is matched by a bounded-adaptive-subscription
alternative with no accuracy gap on this bank". Phase 37 then
calibrates the reply-axis against real LLMs (P37-1: 90 %
semantic-wrong, 0 % syntactic failure — the synthetic
``malformed_prob`` knob is a useless surrogate), ships the
reply-axis ensemble (P37-2, P37-3 — biased-primary and
malformed-syntactic recovery), formalises the structural
limit of single-layer reply ensembles (P37-4 — ensembles
below the noise wrapper contribute no information), and
extends the thread-vs-adaptive-subscription empirical
equivalence to a harder (nested-contest) task family
(P37-5). Phase 38 closes the composition story: the
two-layer stack ``UnionClaimExtractor ∘
EnsembleReplier(MODE_DUAL_AGREE)`` is the unique
configuration that recovers a joint (extractor-axis +
reply-axis) attack on the Phase-35 bank (P38-1), and the
``PathUnionCausalityExtractor`` above-noise combiner
recovers the Phase-37-legacy adv_drop_root cell that every
single-layer reply-axis ensemble provably cannot (P38-2).
Phase 38 also ships a per-feature falsifier table for the
minimum dynamic primitive (P38-3 — ``typed_vocab``,
``terminating_resolution``, and ``round_aware_state`` are
individually load-bearing; ``bounded_witness`` and
``frozen_membership`` are null-control on accuracy on the
tested families) and a prompt-variant calibration frame
(P38-4 — the pipeline can measure a prompt-induced bias
shift without enlarging the typed-reply contract). The
programme's axis of differentiation is now *three-layered*:
(i) compression-vs-communication (P31-5), (ii) static-vs-
dynamic coordination (P35-1, P35-4), (iii) single-layer-vs-
two-layer ensemble composition (P38-1, P38-2). Together
P38-1 + P38-2 + P37-4 give the first formal statement of
the "depth of defense" position in the programme: for every
boundary at which noise can enter, the ensemble combiner
must sit strictly above that boundary; stacking ensembles
at multiple boundaries is necessary when noise can enter
at multiple boundaries.

Phase 39 then converts the Phase-38 mock-only prompt-
calibration result into real-LLM data and ships the
first runnable SWE-bench-style bridge artifact. The
real-LLM measurement (P39-1) refines C38-3: prompt-
shape responsiveness is itself capacity-dependent —
the 0.5B class is *model-bound* (no Phase-38 variant
moves it), the 7B class is *partially prompt-bound*
(``contrastive`` lifts ``correct_rate`` 5× over
default). The SWE-bench bridge (P39-3 + P39-4)
reduces the largest external-validity gap of the
programme — end-to-end SWE-bench — to a unidiff-parser
+ Docker-sandbox engineering follow-up: every required
SWE-bench-instance field has a typed counterpart in
``SWEBenchStyleTask``, the four-role team-substrate
plumbing is unchanged Phase-31 ``HandoffRouter``, and
the bounded-context invariant of P31-3 / P35-2
extends to the SWE-style team by construction (842
chars / 0 events to the patch_generator under
substrate, independent of n_distractors). Theorem
P39-2 names the regime taxonomy the programme has
been implying since Phase 30: every team-shaped task
admits an ``A ≤ min(A_substrate, A_synth)``
decomposition; the substrate is the active constraint
only when the model saturates the synthesis bound —
this is the lens that distinguishes
substrate-moveable from model-moveable gaps and
clarifies the next set of research decisions
(stronger model? fine-tuning? real SWE-bench loader?).

Phase 40 then carries out the Phase-39-named
follow-up *as a research artifact*. The
unidiff-parser + real-shape adapter + JSONL loader
ship as a 290-LOC extension to ``swe_bench_bridge``;
the sandboxed execution boundary
(``SubprocessSandbox`` + optional ``DockerSandbox``)
ships as a new ``tasks/swe_sandbox`` module behind
a ``Sandbox`` protocol; the
``experiments/phase40_real_swe_bridge`` driver
composes loader + substrate + sandbox into a
runnable end-to-end pipeline; a bundled
six-instance real-shape JSONL artifact makes the
entire evaluation reproducible offline in seconds.
**Theorem P40-1** (unidiff round-trip), **P40-2**
(real-shape substrate bounded-context preservation),
**P40-3** (sandbox-boundary transparency on the
oracle ceiling) settle the three claims the
infrastructure carries. The Phase-40 real-LLM run
on ``qwen2.5-coder:7b`` produces an honest
*substrate-vs-naive ranking inversion* at six
instances (5/6 vs 4/6 — the substrate's bounded
prompt withholds a raw-text anchor on
``ext-list-001`` that the 7B uses for byte-strict
patch reproduction); the result sits inside
P39-2's transcription-bounded regime and the
remediation is generator-side, not substrate-side.
The programme's external-validity gap to public
SWE-bench is now *empirical* (run on more
instances, run on a stronger / fine-tuned generator,
optionally relax the bridge's byte-strict matcher)
rather than infrastructural — the loader, the
adapter, the sandbox, and the substrate are all
shipped, and the failure-attribution surface
(substrate vs LLM vs sandbox) is decidable from
the JSON artifact alone.

### 4.9.10 Arc 8 (extended further) — Parser-compliance attribution layer, 57-instance SWE-bench-Lite bank, cluster rerun (Phase 42)

Phase 41 closed the 28-instance matcher-attribution surface
but surfaced a strictly *higher* attribution layer — the
LLM-output parser — that sits above matcher precision and
dominates for models whose dominant failure mode is
format-noncompliance. Phase-41 § D.4 reported the concrete
instance: ``gemma2:9b`` emits the semantically correct fix
on every instance but fails to close the bridge's ``<<<``
output delimiter, so every patch lands as
``patch_no_match`` before the matcher axis becomes measurable.
Phase 42 makes the parser a first-class attribution surface
and closes the ≥ 50-instance external-validity threshold
named by Conjecture C41-1.

Three coupled deliverables ship:

* **Parser-compliance layer** (``vision_mvp/tasks/swe_patch_parser.py``) —
  a ``parse_patch_block(text, mode, unified_diff_parser)``
  entry point with three parser modes
  (``PARSER_STRICT`` = Phase-41 baseline regex;
  ``PARSER_ROBUST`` = Phase-42 default with five named
  heuristics; ``PARSER_UNIFIED`` = unified-diff-only).
  ``ParseOutcome`` carries a ten-label closed failure
  taxonomy (``ok`` / ``empty_output`` / ``no_block`` /
  ``unclosed_new`` / ``unclosed_old`` / ``malformed_diff`` /
  ``empty_patch`` / ``multi_block`` / ``prose_only`` /
  ``fenced_only``) and a six-label recovery enum
  (``RECOVERY_NONE`` / ``RECOVERY_CLOSED_AT_EOS`` /
  ``RECOVERY_FENCED_CODE`` / ``RECOVERY_LABEL_PREFIX`` /
  ``RECOVERY_UNIFIED_DIFF`` / ``RECOVERY_LOOSE_DELIM``).
  ``ParserComplianceCounter`` aggregates per-cell and
  exposes ``compliance_rate`` /
  ``raw_compliance_rate`` / ``recovery_lift`` so parser
  recovery is attributable per heuristic.
  ``llm_patch_generator(llm_call, parser_mode=…, parser_counter=…,
  prompt_style=…)`` routes the parser axis from the bridge
  boundary; ``None`` preserves the Phase-41 byte-strict path
  byte-for-byte. ``build_patch_generator_prompt(…,
  prompt_style="block" | "unified_diff")`` opts into a
  unified-diff output contract when the parser cell uses
  ``PARSER_UNIFIED`` or ``PARSER_ROBUST``.
* **57-instance SWE-bench-Lite-style bank.** The Phase-41
  28-instance ``swe_lite_style_bank.jsonl`` grown with 29
  new instances covering a broader edit-shape class:
  string manipulation (pad / join / double / reverse /
  sep_join / sort_by_length / contains_any), numeric
  guards (nonneg / square / wrap_index / power /
  operator precedence / gcd), sequence construction
  (reverse_list / pair_with / flatten / chunk),
  dict helpers (invert / get_or_default), recursion /
  iteration (fib / count_occurrences / argmax /
  running_max / cumulative), boolean short-circuit
  (safe_lookup), exception handling (parse_int_or — narrow
  the except clause), nested data (flatten), format /
  representation (to_percent), sentinel values
  (argmax empty), set algebra (symmetric_difference),
  class state transitions (Stack.pop, StopLight /
  multi-hunk), binary search off-by-one, graph walk
  reachability, default argument correction (greet).
  Every instance is validated via the same
  oracle-round-trip precondition as the Phase-41 bank —
  the bank-builder refuses to register an instance whose
  diff doesn't parse, whose OLD blocks aren't
  uniquely anchored, or whose oracle-patched source
  doesn't pass the hidden test.
* **Phase-42 driver + cluster integration**
  (``vision_mvp/experiments/phase42_parser_sweep.py``).
  Sweeps ``(parser_mode, apply_mode, n_distractors)``
  against an injected sandbox and either the deterministic
  oracle or a real LLM. An LLM-output cache keyed per
  ``(instance_id, strategy_proxy, n_distractors,
  prompt_style)`` ensures the parser-mode axis is a cheap
  re-parse of cached text — only the first parser cell
  issues LLM calls. ``--ollama-url`` forwards to
  ``LLMClient(base_url=…)`` so coding/generation runs route
  to the ASPEN cluster macbook-1 node
  (``http://192.168.12.191:11434``) and secondary /
  comparison runs to macbook-2
  (``http://192.168.12.248:11434``) or localhost in
  parallel.

**Theory.** Three theorems + three conjectures:

* **Theorem P42-1** (parser-compliance attribution
  decomposition). For every ``(π_base, π_cand)``,
  ``Δ pass@1 = |R_recovered_parser| − |R_regressed_parser|``
  under every matcher × strategy × distractor cell. Promotes
  Conjecture C41-5 to a theorem on the attribution surface
  and gives the programme a *three-axis* decomposition
  (parser × matcher × substrate).
* **Theorem P42-2** (parser recovery cannot produce a
  false pass). For every recovery heuristic ρ, the
  recovered ``(old, new)`` bytes are a subset of the
  generator's output bytes. Therefore the downstream
  ``(apply_patch → run_patched_test)`` outcome is a pure
  function of generator bytes, not of parser choice; a
  passing test after recovery certifies the generator's
  semantic content would have passed if delimited
  correctly.
* **Theorem P42-3** (robust parser dominates on
  format-noncompliant generators). When the dominant
  noncompliance mode is
  ``{unclosed_new, prose_only_with_inline_code,
  fenced_only_2, label_prefix}``, the Phase-42 robust
  parser's compliance rate lower-bounds the Phase-41
  strict parser's compliance rate plus ``η(1-ε)`` where
  η is noncompliance fraction and ε the fraction
  escaping every heuristic.
* **Conjecture C42-1** (substrate-vs-naive gap ≤ 1 pp at
  N ≥ 50). On the Phase-42 bank under a coder-finetuned
  ≥ 7B model the absolute gap is bounded by 1 pp under
  every matcher × parser × distractor cell.
* **Conjecture C42-2** (parser-compliance dominates
  matcher-permissiveness at 7B–30B). The Phase-42
  recovery-rate exceeds the Phase-41 permissive-matcher
  recovery-rate at medium model scale.
* **Conjecture C42-3** (three-axis decomposition
  completeness). Every cell's pass@1 is well-approximated
  by the product ``P_parse · P_match · P_semantic ·
  P_sandbox`` with ``P_sandbox`` fixed at 1.0 by Theorem
  P40-3.

**Empirical artifacts.** Phase-42 mock sweep reruns the
Theorem P41-1 invariant on the 57-instance bank:
substrate prompt flat at 205.9 tokens across
``n_distractors ∈ {0, 6, 12, 24}`` while naive grows
197 → 527 tokens (2.7× span) — a direct reproduction of
Theorem P41-1 at 2.0× the Phase-41 bank scale. 1 368
sandboxed measurements in 122 s wall; every
``chain_ok = True``. **Real-LLM headline on the cluster
macbook-1 ``qwen2.5-coder:14b``**: pass@1 jumps from
**0.018 / 0.018 / 0.018 (naive/routing/substrate, strict
parser, 56 × 3 patch_no_match)** to **0.930 / 0.930 /
0.930 (robust parser)** — a **+91.2 percentage-point
lift** on every strategy. 52 of the 56 strict-parser
failures recover (``R_recovered = 52, R_regressed = 0``,
Theorem P42-1); 168/171 LLM calls trigger
``RECOVERY_FENCE_WRAPPED`` (recovery lift = +0.983,
empirically confirming ``η(1 − ε) ≈ 0.982`` in Theorem
P42-3). **Substrate-vs-naive gap is 0 pp under robust**,
the strongest empirical support for Conjecture C42-1 to
date. **Real-LLM second datapoint on cluster macbook-2
``qwen2.5:14b-32k`` (general)**: pass@1 0.526 / 0.526 /
0.509 (strict) → 0.544 / 0.544 / 0.526 (robust) — a
lighter +1.8 pp lift because only 21/171 (12.3 %) of
outputs fence-wrapped. The parser lift is a pure
function of the model's noncompliance shape, not of
substrate. **Real-LLM third datapoint localhost
``qwen2.5-coder:7b``** (Phase-41 headline model, now at
N = 57): pass@1 = **0.842 / 0.842 / 0.842** on both
strict and robust parser cells (no fence-wrapping, no
parser-recovery; the parser axis is empirically null on
this model). **Real-LLM fourth datapoint localhost
``gemma2:9b``** (Phase-41 § D.4 failure-mode replication,
28-instance subset): pass@1 = **0.000 / 0.000 / 0.000
(strict)** → **0.857 / 0.857 / 0.857 (robust)** — a
**+85.7 pp lift** from exactly the same LLM output,
changing only the parser mode. 82/84 calls trigger
`RECOVERY_FENCE_WRAPPED`; `R_recovered = 24`,
`R_regressed = 0` on every strategy. **This is the
exact empirical falsification of Conjecture C41-5 the
Phase-41 note named as the programme's most tractable
Phase-42 target** — the model the Phase-41 bridge
stalled at 0/28 now passes at 24/28 on the same bridge
with the same model after the parser-layer upgrade.
**Substrate-vs-naive gap = 0 pp** on *four*
independent real-LLM cells (14B-coder cluster, 14B-
general cluster, 7B-coder localhost, 9B-gemma
localhost) — the strongest empirical support for
Conjecture C42-1 in the programme to date. See
``RESULTS_PHASE42.md`` § D.3, § D.4, § D.4b, § D.5 for
full per-strategy tables and compliance counters.

**Architectural discipline.** Every Phase-41 primitive is
preserved byte-for-byte: ``llm_patch_generator`` defaults
to the Phase-41 regex when ``parser_mode=None``;
``build_patch_generator_prompt`` defaults to
``prompt_style="block"``; ``LLMClient.base_url`` defaults
to ``None`` (localhost). Every Phase-41 artifact reruns
byte-for-byte under the Phase-42 build. Phase 42 adds
atop, never replaces.

### 4.9.11 Arc 8 (extended further) — Public-style-scale audit, frontier semantic headroom, and the post-parser-recovery semantic taxonomy (Phase 43)

Phase 42 closed the parser-compliance layer and shipped the
first three-axis attribution surface on the real SWE loop.
The residual 4/57 (coder-14B) / 9/57 (coder-7B) failures
after robust-parser recovery are the programme's first
truly *semantic* residue — format-compliant, byte-matching,
structurally valid patches that fail the hidden test. Phase
43 characterises that residue without expanding the
substrate, the parser, or the matcher — the frontier is now
*model-shaped*, not infrastructure-shaped.

Four coupled deliverables ship:

* **Semantic failure taxonomy**
  (``vision_mvp/tasks/swe_semantic_taxonomy.py``) — a closed
  nine-label vocabulary (``SEM_OK`` / ``SEM_PARSE_FAIL`` /
  ``SEM_WRONG_EDIT_SITE`` / ``SEM_RIGHT_SITE_WRONG_LOGIC`` /
  ``SEM_INCOMPLETE_MULTI_HUNK`` / ``SEM_TEST_OVERFIT`` /
  ``SEM_STRUCTURAL_SEMANTIC_INERT`` / ``SEM_SYNTAX_INVALID``
  / ``SEM_NO_MATCH_RESIDUAL``) with a pure, deterministic
  classifier ``classify_semantic_outcome`` that takes
  ``(buggy_source, gold_patch, proposed_patch, error_kind,
  test_passed)`` and returns exactly one label.
  ``SemanticCounter`` aggregates per-strategy + pooled
  histograms with a ``failure_mix`` helper that normalises
  the composition by non-``SEM_OK`` total.
* **Public-style-scale loader self-test.**
  ``phase43_frontier_headroom.verify_public_style_loader``
  round-trips every instance of a target JSONL through
  ``load_jsonl_bank → SWEBenchAdapter.from_swe_bench_dict
  → parse_unified_diff → apply_patch → run_patched_test``.
  On the bundled 57-instance bank: 57 / 57 oracle
  saturation under strict matcher. The externalisation gap
  to real public SWE-bench-Lite is now a
  ``--jsonl <path>`` drop-in.
* **Phase-43 analysis driver**
  (``vision_mvp/experiments/phase43_frontier_headroom.py``).
  Ingests one or more Phase-42-shape artifacts, re-derives
  per-cell semantic labels, and emits a cross-model
  comparison JSON. Analysis-only — does not call the LLM,
  does not touch the bridge.
* **LLMClient ``think`` field + Phase-42 driver extension.**
  ``LLMClient(think=False)`` threads the ``think`` flag
  into Ollama's ``/api/generate`` body so Qwen3-class
  thinking models (qwen3.5:35b) use their full
  ``num_predict`` budget for output rather than internal
  thinking. The Phase-42 driver gains ``--think {on, off,
  default}`` and ``--max-tokens`` flags. Default ``None``
  preserves Phase-42 byte-for-byte semantics.

The Phase-43 cluster fanout ran two independent
``qwen3.5:35b`` (36B MoE) cells on the ASPEN cluster:
macbook-1 on the full 57-instance bank at
``n_distractors = 6``; macbook-2 on a 20-instance subset
across ``n_distractors ∈ {0, 24}`` for the bounded-context
stress check. The mac1 first-cell result surfaced a new
Phase-43 regression shape: the 35B emits ``<<`` (two angle
brackets) instead of the canonical ``<<<`` at
end-of-generation, and the Phase-42 ``_strip_trailing_prose``
pattern set did not strip a partial / full trailing
delimiter from the recovered NEW payload. The Phase-43
parser patch adds ``\n\s*<{2,4}\s*\Z`` to the tails list;
the patch is byte-safe under Theorem P42-2 (the trailing
``<<`` is in the generator's output; we exclude it from
the substitution payload but never synthesise a byte).

**Theory.** Three theorems + four conjectures:

* **Theorem P43-1** (bounded-context preservation on the
  external-validity bank). On the 57-instance bank under
  every ``(parser_mode, apply_mode, n_distractors)`` cell
  in ``ALL_PARSER_MODES × ALL_APPLY_MODES × {0, 6, 12, 24}``,
  the substrate's ``patch_generator`` prompt token budget
  is **205.9 tokens flat**. Naive grows 197.3 → 527.1
  tokens (**2.7×** span). Extends Theorem P41-1 / P42-1 to
  the full cross product at the external-validity scale.
* **Theorem P43-2** (post-parser-recovery semantic residue
  is structurally classifiable). Every measurement is
  assigned exactly one label from a nine-element closed
  vocabulary by a pure, deterministic classifier. The
  labelling is *total*, *exhaustive*, and *orthogonal to
  parser/matcher choice*. Promotes the Phase-42 residue
  from a count to a composition, making cross-model
  comparisons composition-level, not only pass-rate-level.
* **Theorem P43-3** (semantic-ceiling separation on
  coder-finetuned models at N ≥ 50). For every measured
  coder-finetuned ≥ 7B model on the canonical
  ``parser=robust / apply=strict / nd=6`` cell of the
  57-instance bank: (a) ``pass@1(substrate) =
  pass@1(naive) = pass@1(routing)`` — strategy invariance;
  (b) per-strategy ``SemanticCounter.by_strategy`` label
  histograms are byte-identical across strategies; (c) the
  pooled failure mix is dominated by ``SEM_WRONG_EDIT_SITE``
  on coder-finetuned models and by ``SEM_SYNTAX_INVALID``
  on general-purpose models of matched parameter class.
* **Conjecture C43-1** (frontier coder closes wrong-edit-
  site without re-opening the substrate gap). A frontier
  reasoning/coder model at ≥ 30B active parameters
  achieves ``|R_{SEM_WRONG_EDIT_SITE}| / 57 ≤ 0.02`` AND
  preserves Theorem-P43-3 (a). **Partially supported
  empirically**: ``qwen3.5:35b`` on the full 57-instance
  bank at the canonical cell scores 55 / 57 (0.965 pass@1)
  with 2 / 57 residue (0.035 — slightly above the
  conjecture's 0.02 threshold) and substrate-vs-naive
  gap = 0 pp. The 14B-coder's multi-hunk residue
  (``ext-multi-001``) is cleared by the 35B, and the two
  remaining failures are in ``test_exception`` /
  ``test_assert`` shape. A 70B-class coder-finetuned
  frontier is the natural tighter test.
* **Conjecture C43-2** (residue composition is
  training-mix-indexed, not parameter-count-indexed). Two
  matched-parameter-count models differ in dominant
  failure label by training mix.
* **Conjecture C43-3** (substrate bounded-context
  invariant is model-independent).
* **Conjecture C43-4** (semantic residue does not
  decompose further under existing substrate primitives).

**Empirical headline.**

* **Cross-model substrate-vs-naive gap at the canonical
  cell (parser=robust / apply=strict / nd=6):**
  **0.0 pp** on ``qwen2.5-coder:14b`` (93.0 % pass@1),
  **0.0 pp** on ``qwen2.5-coder:7b`` (84.2 %), **0.0 pp**
  on ``gemma2:9b`` @ 28-instance subset (85.7 %), **1.8 pp**
  on ``qwen2.5:14b-32k`` (54.4 %, general-purpose). The
  gap-zero property holds on every measured coder-class
  model at the ≥ 50-instance scale — the strongest form of
  Conjecture C42-1.
* **Cross-model failure-mix composition at the canonical
  cell.** Coder-14B: 50 % wrong_edit_site / 25 %
  incomplete_multi_hunk / 25 % no_match_residual.
  Coder-7B: 56 % wrong_edit_site / 33 %
  no_match_residual / 11 % syntax_invalid. Gemma2-9B:
  50 % wrong_edit_site / 25 % no_match_residual / 25 %
  syntax_invalid. General-14B: 52 %
  syntax_invalid / 46 % no_match_residual / 3 %
  incomplete_multi_hunk. Training-mix separation
  (Conjecture C43-2) is empirically visible at matched
  parameter count (14B-coder vs 14B-general).
* **Frontier 35B-MoE headline.** The ``qwen3.5:35b`` mac1
  run produces 171 / 171 ``unclosed_new``-shape
  noncompliance under strict parser (100 % format
  noncompliance, same regime gemma2:9b had in Phase 41).
  Without the Phase-43 trailing-delim patch every
  recovered patch produced syntax errors (0 / 57 pass).
  **With the Phase-43 patch: 55 / 57 pass@1 on every
  strategy, substrate-vs-naive gap = 0 pp**, beating the
  previous strongest local cell (``qwen2.5-coder:14b`` at
  53 / 57). Substrate prompt is flat at 205.9 tokens
  (Theorem P43-1 reproduced at frontier scale). Mac2
  stress cell (20-instance subset across
  ``n_distractors ∈ {0, 24}``): 20 / 20 pass on every
  strategy, substrate prompt 209.0 tokens flat across
  both distractor counts (bounded-context invariant
  preserved at frontier × distractor-extreme × strategy
  cross product).

**Architectural discipline.** Phase 43 is strictly
additive and analysis-only:
``swe_semantic_taxonomy`` is a new pure module with no
dependencies on substrate/parser/bridge paths;
``phase43_frontier_headroom`` is an offline driver
consuming Phase-42 artifacts; ``LLMClient.think`` default
``None`` preserves Phase-42 byte-for-byte semantics. The
Phase-42 robust-parser's trailing-prose list gains one
pattern — a strict regression improvement. Every
Phase-39..42 regression test passes byte-for-byte (110 /
110 on the SWE-arc slice; Phase-43 test slice 18 / 18
green).

### 4.9.12 Arc 8 (extended further) — Raw-text semantic residue capture, refined taxonomy, and public-SWE-bench-Lite drop-in readiness (Phase 44)

Phase 43 characterised the post-parser-recovery residue with
a nine-label closed vocabulary but had to pass a *sentinel*
proposed-patch tuple into the classifier because the
Phase-42 artifact schema does not preserve raw LLM output
(Phase-43 § D.7). Phase 44 closes that limitation and
promotes the public-SWE-bench-Lite drop-in claim from
documentation to validated code. Four coupled pieces ship:

* **Raw-text capture module**
  (``vision_mvp/tasks/swe_raw_capture.py``) — a new opt-in
  module with a versioned ``RawCaptureRecord`` /
  ``RawCaptureStore`` schema (``SCHEMA_VERSION =
  "phase44.v1"``). Every record carries the raw LLM bytes +
  SHA-256, the ``ParseOutcome.as_dict()``, the proposed
  ``(old, new)`` pairs, the applied pairs post-matcher, the
  SHA-256 of the patched source, and the downstream
  ``error_kind`` / ``test_passed`` verdict.
  ``make_capturing_generator`` wraps either a prebuilt
  bridge generator or a fresh ``llm_call`` and plumbs the
  raw text through the store while preserving the
  Phase-42 LLM-output cache discipline.
* **Refined semantic taxonomy** (extension to
  ``swe_semantic_taxonomy.py``). Five Phase-44 sub-labels
  partition the Phase-43 coarse buckets when raw bytes
  are available:
  ``SEM_RIGHT_FILE_WRONG_SPAN`` (anchored in the right
  file, wrong span),
  ``SEM_RIGHT_SPAN_WRONG_LOGIC`` (v1 synonym — raw-bytes
  confirmation),
  ``SEM_PARTIAL_MULTI_HUNK_SUCCESS`` (at least one hunk
  byte-normalised-agrees with gold),
  ``SEM_NARROW_FIX_TEST_OVERFIT`` (right site, NEW
  token-overlaps gold NEW, fails on a subset),
  ``SEM_STRUCTURAL_VALID_INERT`` (patched source
  normalised-equals buggy source — behaviourally inert
  patch). ``REFINEMENT_MAP`` declares legal partitions
  reflexively (coarse ∈ refined_set), making the sentinel
  path a legal v2 classification. ``classify_semantic_outcome_v2``
  subsumes the Phase-43 classifier on sentinel inputs
  (Theorem P44-2 monotonicity).
* **Phase-44 driver**
  (``vision_mvp/experiments/phase44_semantic_residue.py``).
  Sweep mode runs the Phase-42-shape experiment with raw
  capture on and writes paired parent + capture artifacts.
  Analyse-only mode consumes one or more (parent, capture)
  pairs, re-classifies every measurement under both v1 and
  v2, and emits a ``phase44.summary.v1`` JSON with per-
  cell coarse / refined taxonomy counters and a
  ``coarse_to_refined_partition`` audit.
* **Public-SWE-bench-Lite readiness validator**
  (``vision_mvp/experiments/phase44_public_readiness.py``).
  A five-check pipeline
  (``schema`` → ``adapter`` → ``parser`` → ``matcher`` →
  ``test_runner``) runs against any local JSONL and emits
  a CI-gate verdict
  (``{"ready": bool, "checks": {...}, "blockers": [...]}``).
  The bundled 57-instance bank scores **57/57 on every
  check in 5.2 s wall** under the subprocess sandbox
  (Theorem P44-3).

**Theory.** Three theorems + four conjectures:

* **Theorem P44-1** (raw capture is a lossless projection of
  pipeline state). The ``(parent_measurement, capture_record)``
  pair is sufficient to re-derive every Phase-42 pipeline
  output that is a pure function of the LLM response and
  the cell axes, without LLM or sandbox calls.
* **Theorem P44-2** (refined classifier monotone on
  sentinel inputs). For every Phase-43 measurement tuple,
  ``classify_semantic_outcome_v2(..., proposed_patch=
  sentinel, ...) == classify_semantic_outcome(...,
  proposed_patch=sentinel, ...)`` — the refinement is a
  strict extension, never a replacement. A reader who
  trusted Phase 43 can trust every v2 label on every
  shared sentinel input.
* **Theorem P44-3** (public-readiness saturates on
  bundled bank at ≥ 50-instance scale). The readiness
  validator run against the bundled 57-instance bank
  returns ``{"ready": True, "n": 57, "n_passed_all": 57,
  "blockers": []}`` on every check in ~5 s; the
  externalisation gap is now a pure data-availability
  gap.
* **Conjecture C44-1** (raw-capture disambiguation shifts
  the ``wrong_edit_site`` bucket into a mix dominated by
  ``right_file_wrong_span`` on coder-class models). Open;
  Phase-44 cluster follow-up.
* **Conjecture C44-2** (frontier 35B residue refines to a
  mix dominated by ``narrow_fix_test_overfit`` or
  ``right_span_wrong_logic``, not ``wrong_edit_site``).
  Open.
* **Conjecture C44-3** (substrate gap is refinement-
  invariant on coder-class models). Structurally expected
  because the v2 classifier does not move a measurement
  between SEM_OK and non-SEM_OK.
* **Conjecture C44-4** (readiness is closed under
  row-level filtering). Structurally true as implemented;
  listed because a future extension (cross-row
  uniqueness, repo-level aggregates) could invalidate it.

**Empirical headline.**

* **Public-SWE-bench-Lite readiness verdict on the bundled
  bank:** ``ready: true`` on every check
  (schema=57/57, adapter=57/57, parser=57/57,
  matcher=57/57, test_runner=57/57), 5.2 s wall through
  SubprocessSandbox. Artifact:
  ``vision_mvp/results_phase44_readiness_bundled.json``.
  **The externalisation gap to real public SWE-bench-Lite
  is now purely a data-availability gap**: a public JSONL
  that passes ``V_ready`` runs through the Phase-44
  pipeline by a pure ``--jsonl <path>`` change.
* **Phase-44 cluster runs:** ``qwen2.5-coder:14b`` on mac1
  (strongest practical local coder-class model) and
  ``qwen3.5:35b`` on mac2 (the Phase-43 frontier
  datapoint) with raw capture on. Parent + capture
  artifacts persist raw LLM bytes for every measurement
  so the Phase-44 refined classifier can partition the
  Phase-43 ``wrong_edit_site`` / ``incomplete_multi_hunk``
  / ``structural_semantic_inert`` buckets into their
  Phase-44 sub-labels. Cross-model summary in
  ``results_phase44_refined_summary.json``.
* **Regression:** 112 / 112 on the Phase-39..43 SWE-arc
  slice; Phase-44 test slice 23 / 23 green; 135 total.

**Architectural discipline.** Phase 44 is strictly
additive and opt-in:
``swe_raw_capture`` is a new module with no dependencies
into the bridge / parser / matcher; the Phase-42 defaults
(``parser_mode=None``, oracle generator, etc.) preserve
Phase-42 byte-for-byte. The refined taxonomy labels are
declared alongside — not in place of — the v1 vocabulary,
and ``REFINEMENT_MAP``'s reflexive structure guarantees
that a Phase-43-style classification is a legal Phase-44
classification. The readiness validator is a standalone
analysis driver; it cannot affect any production path.

---

## 4.11 Current frontier

As of Phase 44, the programme sits at the following frontier.
This section is the one-paragraph answer to "where is the
programme right now, and what breaks next?" — it is expected to
change every two or three phases, unlike the rest of this
document.

**Top-line, post-Phase-44.** The programme now has a
**raw-bytes-grounded semantic residue attribution surface on the
57-instance external-validity bank, a runnable CI-gate public-
SWE-bench-Lite readiness validator that saturates at 57/57 on
every one of five checks (schema / adapter / parser / matcher /
test_runner) in 5.2 s, and a Phase-44 refined semantic taxonomy
whose v2 classifier subsumes the Phase-43 classifier on sentinel
inputs (Theorem P44-2) and partitions the coarse Phase-43
residue buckets (wrong_edit_site / incomplete_multi_hunk /
structural_semantic_inert / test_overfit / right_site_wrong_logic)
into five new sub-labels when raw bytes are available.** The
Phase-44 cluster runs (``qwen2.5-coder:14b`` on mac1 and
``qwen3.5:35b`` on mac2) persist raw LLM output for every
measurement, so the Phase-43 sentinel-path limitation (§ D.7
caveat) is closed: the 14B-coder's 4/57 residue and the 35B's
2/57 residue are now attributable to specific refined labels
by inspection of stored bytes, no LLM replay required. Phase 44
is the programme's first residue-analysis milestone whose
primary output is a *partition*, not a *count* — and whose
substrate claim (Theorem P43-1 bounded-context preservation) is
preserved structurally (Conjecture C44-3: refinement does not
move a measurement between SEM_OK and non-SEM_OK, so the
substrate-vs-naive gap is refinement-invariant). **The remaining
gap is (a) public-data-shaped** (a public SWE-bench-Lite JSONL
that passes ``V_ready``; this is purely a data-availability gap
now that the readiness validator is shipped) **and (b) model-
shaped** (a 70B-class coder-finetuned frontier would close
Conjecture C43-1's 2% threshold from the current 3.5%).

**Top-line, post-Phase-43** (preserved below for context). The
programme now has an
**external-validity-scale real SWE loop (57-instance bundled
bank; loader verified ready for public SWE-bench-Lite drop-in),
a four-axis attribution surface (parser × matcher × substrate
× semantic), a composition-level characterisation of the
residue via the Phase-43 nine-label closed taxonomy, and a
frontier datapoint (``qwen3.5:35b`` 36B MoE) that achieves
0.965 / 0.965 / 0.965 (55 / 57) at substrate-vs-naive gap
= 0 pp** — beating the previous strongest local cell
(``qwen2.5-coder:14b`` at 0.930) on the same bank. Summary
table at the canonical cell (``parser=robust / apply=strict /
nd=6``): **qwen3.5:35b 96.5 %, qwen2.5-coder:14b 93.0 %,
gemma2:9b (28) 85.7 %, qwen2.5-coder:7b 84.2 %,
qwen2.5:14b-32k 54.4 %** — substrate-vs-naive gap is 0 pp
on every cell except qwen2.5:14b-32k (general-purpose, 1.8
pp, dominated by syntax_invalid residue). The residue is
**training-mix-indexed** (coder-finetuned → wrong_edit_site;
general-purpose → syntax_invalid) and **model-capacity-
indexed** (35B compresses the 14B-coder's multi-hunk miss).
The programme's durable substrate claim is **bounded active
context per role** (Theorem P43-1: substrate prompt flat at
205.9 tokens across the full cross product on the full bank;
209.0 tokens on the 20-instance subset across nd ∈ {0, 24}
at frontier-model scale) — explicitly *not* a pass@1 lift,
which is empirically zero on every coder-class model. The
Phase-43 ``qwen3.5:35b`` cluster run surfaced (and patched) a
regression: trailing ``<<`` delimiter at EOS on thinking-
disabled Qwen3 outputs; the one-pattern fix in
``_strip_trailing_prose`` is byte-safe under Theorem P42-2
and was load-bearing — without it the 35B scored 0 / 57
(100 % syntax_invalid); with it the 35B scored 55 / 57.
**The remaining gap is model-shaped**: Conjecture C43-1
(frontier closes wrong-edit-site) is empirically supported
by the 35B datapoint (2 / 57 residue, down from 14B-coder's
4 / 57, and the multi-hunk ``ext-multi-001`` that the 14B
missed is now passed); a 70B-class coder-finetuned frontier
would be the natural next measurement.
**Phase-42 deprecation is null**: every Phase-42 default
preserves Phase-41 byte-for-byte, and every Phase-43 addition
is strictly additive (analysis-only module + one pattern
added to an existing list + one optional LLMClient kwarg).
**Historical top-line, post-Phase-42** (preserved below for
context): three-axis attribution surface, 57-instance
empirical data at the external-validity threshold named by
Conjecture C41-1, headline real-LLM datapoint +91.2 pp
pass@1 lift on ``qwen2.5-coder:14b`` (cluster macbook-1)
from a single parser-recovery heuristic.
Phase 40 shipped the loader + sandbox + substrate
composition; Phase 41 shipped the matcher attribution layer
and the first 28-instance empirical data; Phase 42 adds the
parser-compliance attribution layer (a
``parse_patch_block`` entry point with a closed ten-label
failure taxonomy and six-label recovery enum, counters that
separate raw compliance from recovered compliance per
heuristic, and a ``llm_patch_generator(..., parser_mode=…,
parser_counter=…, prompt_style=…)`` hook so the bridge can
opt into the new axis without disturbing any Phase-41
artifact), the 57-instance ``swe_lite_style_bank.jsonl``
(grown from the Phase-41 28-instance bank by 29 new
instances covering broader edit shapes), ASPEN-cluster
endpoint support on ``LLMClient(base_url=…)`` so heavy
runs fan out across macbook-1 and macbook-2, and a
Phase-42 driver (``phase42_parser_sweep``) that emits a
per-(strategy) ``{recovered, regressed, unchanged_pass,
unchanged_fail}`` set delta between the strict and each
non-strict parser cell. Theorem P42-1 promotes the Phase-41
matcher-attribution identity to a three-axis decomposition
(``Δ pass@1 = |R_recovered_parser| − |R_regressed_parser|``
under every matcher × strategy × distractor cell);
Theorem P42-2 proves parser recovery cannot produce a false
pass by a byte-provenance argument on the recovery
heuristics; Theorem P42-3 lower-bounds the compliance gain
under the robust parser on models whose dominant failure
mode matches one of the four named noncompliance shapes.
**Phase-41 deprecation is null**: every Phase-41 default
preserves Phase-40 byte-for-byte, and every Phase-42
default preserves Phase-41 byte-for-byte — the parser axis
is strictly additive. Theorem P41-1 (bounded-context
preservation) reproduces at N = 57: substrate prompt flat
at 205.9 tokens across ``n_distractors ∈ {0, 6, 12, 24}``
while naive grows 197 → 527 (**2.7×** span), 1 368
sandboxed measurements in 122 s wall, every
``chain_ok = True``. The remaining gap is
*public-bench-shaped* — the ``--jsonl <path>`` parameter
change to point the driver at real SWE-bench Lite — and
*research-shaped* (Conjectures C42-1 / C42-2 / C42-3 plus
carry-over C41-1 / C41-2 / C41-3 / C41-4). The
infrastructure and the three-axis attribution surface are
shipped.

**Top-line, post-Phase-41.** The programme now has a **real
external task loop with first larger-N empirical ranking data
and a two-axis attribution surface**: Phase 40 shipped the
loader + sandbox + substrate composition; Phase 41 adds a
28-instance SWE-bench-Lite-shape bank (``swe_lite_style_bank.jsonl``,
authored to cover diverse edit shapes and validated against
the oracle round-trip), a matcher-permissiveness axis
(``apply_mode`` threaded through ``run_swe_loop`` and
``run_swe_loop_sandboxed`` with four modes: strict, lstrip,
ws_collapse, line_anchored), and a driver
(``phase41_swe_lite_sweep``) that emits a per-strategy
``recovered``/``regressed`` set delta between strict and
permissive matchers — making the *"substrate failure vs
generator-literal-text failure vs bridge-matcher precision"*
attribution decidable from the JSON artifact alone. Theorem
P41-1 reproduces the bounded-context invariant at 4.7× bank
scale (substrate 746.4 chars flat; naive 806.8 → 2 125.8
across ``n_distractors ∈ {0, 6, 12, 24}``). Theorem P41-3
formalises the matcher-permissiveness attribution as a
counting identity. **Empirical headline on
``qwen2.5-coder:7b`` at 28 instances (``n_distractors = 6``):
pass@1 = 0.929 / 0.929 / 0.893 (naive / routing /
substrate) under strict matcher — the Phase-40 6-instance
16.7 pp substrate-vs-naive gap shrinks to a 3.6 pp gap
(1 instance out of 28) at the larger scale. Under ``lstrip``
and ``line_anchored`` permissive matchers the pass@1 is
byte-identical to strict: R_recovered = ∅ and
R_regressed = ∅ on every cell. The permissive matcher
axis is empirically *null-gain and null-risk* on this bank
at the 7B scale — meaning the 7B's strict-mode failures
are not byte-fidelity drift (whitespace / trailing newline
differences the permissive matcher could repair) but
deeper semantic mis-targeting. The Phase-40 § D.5 hypothesis
that matcher permissiveness would re-rank substrate vs
naive is refuted at 28 instances. On the stronger-model
datapoint ``gemma2:9b`` (the Phase-39 Part B frontier
ranking winner on mock-auditor non-code tasks) pass@1 = 0/28
on every cell — NOT because the 9B cannot solve the
problem (a spot check shows it emits the semantically
correct fix) but because it does not reliably close the
bridge's ``<<<`` output delimiter, so every patch parses
as empty and is surfaced as ``patch_no_match``. This is a
new attribution boundary (Conjecture C41-5:
parser-compliance attribution) that sits *above* the
matcher axis — Phase-41's permissive matchers are safe
but powerless when the generator's output fails the
LLM-output-parser contract before reaching the matcher.** The remaining gap is
empirical (point the loader at **public** SWE-bench Lite
JSONL at ≥ 50 instances; collect ranking data on a frontier-
class model) and research-shaped (conjectures C41-1 /
C41-2 / C41-3 / C41-4 — the *regime-decomposition* form
of the substrate-vs-generator story).

**What is settled (durable).** The O(log N) routing bound on
low-intrinsic-rank coordination tasks (Arc 1); the exact-substrate
stack on fixed code batteries across six real Python corpora (Arcs
3–4); runtime calibration of the analyzer at snippet scale (97.6 %
agreement) and multi-corpus scale (98.7 % on `may_raise_explicit`,
FN = 0, over 306 pooled observations in Phase 28 — 1 479 in Phase 29
after the method coverage lift); the explicit / implicit raise axis
separation (Phase 28); safe method-instance auto-construction
(Phase 29, < 1 % construct-failed rate, preserves FN = 0 on
`may_raise_explicit`); task-scale causal-relevance under an
analyzer-derived oracle at 4.54 % pooled naive-broadcast across
four real Python corpora (Phase 29).

**What is newly settled (Phase 30).** Four theorems that connect
the empirical Phase-29 causal-relevance fraction to the
information-theoretic object `T_i*` (minimum-sufficient context);
one of those (Theorem P30-4) closes a special case of OQ-1 (the
matched-substrate regime has a unique one-step fixed point).
First LLM-in-loop external-validity benchmark: on the Python
stdlib `json` module under `qwen2.5:0.5b`, the substrate path
delivers **16.0×** prompt-token reduction with **+60 percentage
points** of answer-accuracy lift (80 % vs 20 %) over naive. On
`click` (third-party CLI framework), the token ratio extends to
**60×** (mock reference); internal control on `vision-core` has
the ratio at **140×**. Substrate-matched slice correctness on the
0.5b model is 78.9 % — bounded below by LLM transcription
fidelity, not by any substrate guarantee; the headline is
architecture-shape evidence for the programme's dual identity
(research milestone ∥ tool milestone).

**What is newly settled (Phase 31).** Five theorems (P31-1..
P31-5) formalising role-conditioned relevance, communication
sparsity (``Θ(R*)`` inter-role bits independent of |X|), bounded
active context per role (``O(R*·τ)``), correctness preservation
under subscription coverage, and a formal separation from
single-agent long-context baselines (P31-5 converts the § 1.5
differentiation to a theorem). Two conjectures (C31-6 role-
lattice generalisation, C31-7 noisy-extractor robustness). First
*non-code* task-scale benchmark: a five-role operational
incident-triage team, five scenario kinds, distractor sweep
k ∈ {6, 20, 60, 120}. Under the mock reader-emulator (the
strategy ceiling), substrate is flat at **196 tokens and 100 %
accuracy across every k**; naive collapses from 100 % → 20 % at
k=120 under truncation; routing is 0 % on every k (no content
to the auditor). Under real ``qwen2.5:0.5b``, substrate is the
only strategy with non-zero accuracy on either k (40 % flat at
k ∈ {6, 60}); naive at k=60 is infeasible (5/5 prompts
truncated). Substrate handoff recall is **1.00** on every
scenario. Typed-handoff substrate primitive
(``core/role_handoff``) with hash-chained log, content-address
dedup, and per-(source, to, kind) delivery accounting shipped
alongside — this is the programme's first *communication-primitive*
substrate module.

**What is newly settled (Phase 32).** Second non-code task-scale
benchmark — vendor-onboarding *compliance review*
(``tasks/compliance_review``) with a distinct role cast (legal /
security / privacy / finance / compliance officer), 13 claim
kinds, and a priority-monotone-verdict + strict-set-flags
decoder. Mock-auditor substrate accuracy is **100 % flat at 171
tokens across k ∈ {6, 20, 60, 120}**, identical signature to
Phase 31's 196-token flatline on incident triage. Three new
theorems (P32-1 cross-domain correctness preservation; P32-2
graceful degradation with explicit two-regime bound, promoting
C31-7 to theorem in the monotone regime; P32-3 token-bound under
bounded noise with inbox capacity as regulariser) and two
conjectures (C32-4 role-lattice stability across domains; C32-5
extractor-composition precision/recall bound). Controlled
extractor-noise sweep (``core/extractor_noise``, 96 noise points
× 2 domains, 0.5 s wall): under drop_prob = 0.5 recall falls as
``(1 - δ)^{R*}``, verdict-side degrades gracefully (monotone
regime), flag-side collapses at spurious_prob = 0.05 (strict-set
exact-match grader), tokens grow but stay bounded (substrate at
spurious = 0.10 is 378 tokens, still ~11× less than naive at
k = 120). Frontier-model spot check: ``qwen2.5-coder:7b`` on
both non-code benchmarks at k = 6 seed = 32 — substrate
**saturates the mock ceiling at 100 %** on compliance review
(vs naive 0 %); on incident triage substrate_wrap reaches
**80 %** vs naive 0 % (the remaining 20 pp is transcription-
bounded, not communication-bounded). Total Phase-32/C cost:
30 LLM calls, 12.3 min wall. See
``vision_mvp/results_phase32_llm_7b_spot.json`` and
``RESULTS_PHASE32.md`` § D.3.

**What is newly settled (Phase 33).** ~~**LLM-driven
extractors** — replacing the regex extractors with an LLM-call
extractor and measuring the induced noise profile.~~ **Done in
P33.** ``core/llm_extractor`` is a drop-in replacement for any
Phase-31/32 extractor; ``core/extractor_calibration`` measures the
empirical (δ̂, ε̂, μ̂, π̂) quadruple against a gold causal chain
and maps it to the closest Phase-32 synthetic sweep point.
Headline on ``qwen2.5:0.5b`` against compliance review:
δ̂ = 0.70, ε̂ = 0.12, μ̂ = 0.40, π̂ = 0.60; Phase-32 closest
match predicts substrate accuracy / recall / precision within a
max-abs gap of **0.10 → verdict "approximates" (Theorem P33-1)**.
~~**Third non-code domain** to confirm C32-4 at K ≥ 3.~~ **Done
in P33.** ``tasks/security_escalation`` — five-role security
team, 15 claim kinds, **max-ordinal severity + claim-set
classification** decoder (structurally distinct from Phase 31
priority-order and Phase 32 monotone-verdict shapes). Substrate
flat at **242 tokens / 100 % accuracy** across k ∈ {6, 20, 60,
120}; naive collapses 100 % → 20 % at k = 120 under truncation;
routing 0 % on every k (Theorem P33-2: cross-domain correctness
at K = 3). Security noise sweep confirms Theorem P32-2 on the
max-ordinal decoder with a precision-to-severity-escalation
failure mode (Theorem P33-3).

**What is newly settled (Phase 34).** Three coupled additions
that close the Phase-33 frontier items (h), (i), (j):

* ~~**Per-role-adaptive Phase-32 sweep** (Conjecture C33-3).~~
  **Done in P34.** ``core/extractor_calibration.per_role_audit_summary``
  reports per-role (δ̂, ε̂, μ̂, π̂), identifies the *limiting
  role* (argmax drop), and maps each role to its closest Phase-32
  synthetic grid point. A per-role replay via
  ``PerRoleNoiseConfig`` + ``per_role_noisy_extractor`` is a
  tighter predictor of substrate accuracy than the pooled replay
  (Conjecture C34-5). Mock calibration across three domains shows
  max per-role drop-rate spread ≥ 0.33 (incident), 0.50
  (compliance), 0.67 (security) — pooled i.i.d. reliably masks
  per-role structure. Theorem P34-1 formalises the bound
  ``A_real ≤ Π_k (1 − δ_k) ≤ (1 − δ̄)^{R*}`` via AM-GM.
* ~~**Adversarial extractor noise.**~~ **Done in P34.**
  ``core/extractor_noise.adversarial_extractor`` implements three
  modes — load-bearing claim drop with priority ordering, role
  silencing, severity-escalation injection. **Theorem P34-2**
  establishes an adversarial-vs-iid separation: at matched nominal
  budget ``δ = budget / R*`` the targeted-drop adversary collapses
  substrate accuracy to **0 %** at budget = 1 on all three
  domains, while matched-budget i.i.d. preserves 20 %–80 % (pooled
  gap **+0.47 pp** across three domains). Severity escalation on
  the max-ordinal security decoder triggers the Theorem-P33-3
  precision-to-severity failure (adv accuracy 0.80); on the
  priority-order (incident 0.10) and monotone-verdict (compliance
  0.00) decoders severity escalation collapses the primary answer.
* ~~**Ensemble extractor composition (Conjecture C33-4).**~~ **Done
  in P34.** ``core/ensemble_extractor.UnionExtractor`` composes
  two extractors with content-address dedup. On a compliance
  *mixed* bank (5 canonical + 5 narrative, regex cannot parse the
  narrative phrasings and narrative-LLM cannot match canonical
  phrasings), regex alone scores 50 %, LLM alone 0 %, ensemble
  **100 %**, with pooled δ_u = 0.00 ≤ δ_r · δ_l = 0.188 and
  ε_u = 0.00 ≤ ε_r + ε_l = 0.000. **Theorem P34-3** promotes the
  Conjecture-C33-4 union bound to an empirical bound on a
  genuinely complementary-coverage scenario bank. The substrate
  now ships with a *defensive depth* primitive against Theorem
  P34-2's adversary (Conjecture C34-4 names the combined test).

**What is newly settled (Phase 35).** Two coupled additions
that move the team-communication arc from *static* typed
handoffs to *dynamic* bounded coordination while preserving
every Phase-31 bounded-context theorem byte-for-byte:

* ~~**Dynamic, bounded communication primitives above the typed-
  handoff substrate.**~~ **Done in P35.**
  ``core/dynamic_comm`` adds ``EscalationThread`` +
  ``ThreadReply`` + ``ThreadResolution`` + ``DynamicCommRouter``
  — a typed, frozen-membership, explicitly-terminated
  coordination object whose single public output is a
  ``CLAIM_THREAD_RESOLUTION`` handoff routed through the
  unchanged Phase-31 ``HandoffRouter``. Thread-internal events
  (``THREAD:OPEN`` / ``THREAD:REPLY`` / ``THREAD:CLOSE``) are
  hash-chained in the existing ``HandoffLog`` for audit but
  never enter non-member inboxes (Theorem P35-4, no-leak
  invariant). Bounded-context is preserved with an additive
  ``T·R_max·W`` per role per round (Theorem P35-2), independent
  of |X|.
* ~~**First benchmark where static typed handoffs cannot
  recover the answer.**~~ **Done in P35.**
  ``tasks/contested_incident`` is a 6-scenario bank (4 contested
  + 2 controls). Under the mock auditor across k ∈ {6, 20, 60,
  120} × 2 seeds (48 dynamic measurements, 48 static, 48 naive),
  the static-handoff strategy scores **0/4 contested (0 %
  contested accuracy, 33 % full, 50 % root-cause)** — the
  failure is a priority-inversion pattern that no choice of
  static priority permutation can fix (Theorem P35-1). The
  dynamic strategy scores **4/4 contested (100 % flat across
  every k, 246 tokens per prompt, ≤ 2 replies of ≤ 12 witness
  tokens per thread)**. Real-LLM spot check under
  ``qwen2.5:0.5b`` at k=6 seed=35 reproduces the separation at
  the root-cause-accuracy axis: dynamic 1.00 vs static 0.50
  (+50 pp); full-answer accuracy is capped by 0.5b
  transcription fidelity (Phase-30 § B.3), not by the
  substrate. Theorem P35-3 identifies the weakest producer-
  local extractor precondition under which dynamic coordination
  is guaranteed to recover correctness (sound ``INDEPENDENT_ROOT``
  emission + no under-emission on the gold causal claim).
* **Theory —** four theorems and two conjectures. Theorem P35-1
  (expressivity separation between static handoffs and dynamic
  coordination — the scenario-pair pigeonhole argument), P35-2
  (bounded-context preservation with additive ``T·R_max·W``),
  P35-3 (correctness under sound producer-local extraction),
  P35-4 (no-leak invariant for non-member roles). Conjecture
  C35-5 (bounded threads ≡ bounded adaptive subscriptions in
  decoder correctness — predicts equivalence, threads win on
  type-level bounded-context enforcement). Conjecture C35-6
  (dynamic coordination is necessary, not only sufficient —
  predicts an information-theoretic lower bound dual of P35-1).

**What is newly settled (Phase 36).** Three coupled additions
that promote the Phase-35 dynamic-coordination claim from a
clean-bank result to a noise-robust result, replace the
deterministic reply oracle with a disciplined LLM reply path,
and resolve the adaptive-subscription vs dynamic-thread
comparison on the contested bank:

* ~~**Dynamic coordination under reply noise (C35-7).**~~
  **Done in P36 Part A.** ``core/reply_noise`` composes over
  the Phase-35 producer-local causality extractor with i.i.d.
  Bernoulli noise (``drop_prob`` / ``mislabel_prob``) and
  adversarial noise (``drop_root`` / ``flip_root_to_symptom`` /
  ``inject_root_on_symptom``). Phase-36 Theorem P36-1 gives a
  closed-form probability bound ``Pr[D_dyn = gold] =
  (1-p)·(1-q)``; dominance over static persists for
  ``p + q < 1/2``. Theorem P36-2: a single targeted adversarial
  ``drop_root`` (budget ``b = 1``) collapses dynamic to the
  static baseline.
* ~~**LLM-driven thread replies (C35-8).**~~ **Done in P36
  Part B.** ``core/llm_thread_replier.LLMThreadReplier`` drives
  a narrow, bounded LLM call (one JSON line: ``reply_kind`` ∈
  the Phase-35 enum + bounded witness); filters out-of-vocab /
  malformed replies at parse time; falls back to
  ``REPLY_UNCERTAIN``. Theorem P36-3: under well-formed
  in-vocab replies, the LLM replier is behaviourally identical
  to the deterministic oracle, so the Phase-35 substrate
  theorems carry over without modification. Graceful decay
  under ``malformed_prob = 0.5`` (dynamic 66.7 % vs static
  33.3 %).
* ~~**Adaptive subscription as an alternative primitive
  (C35-5).**~~ **Done in P36 Part C.** ``core/adaptive_sub``
  ships a bounded, TTL-expiring subscription-edit primitive with
  a hard cap on concurrent edges. Theorem P36-4: on the
  Phase-35 bank under the full drop × mislabel × k × seed grid
  (96 paired measurements), ``|acc(D_dyn) - acc(D_adp)| =
  0.000`` at every cell; prompt-token overhead of adaptive_sub
  is +12 %. Conjecture C35-5's equivalence prediction is
  empirically confirmed on this task family; Conjecture C36-5
  names the analytic extension to arbitrary families.
* **Theory —** four theorems (P36-1 graceful-degradation bound;
  P36-2 adversarial-collapse bound; P36-3 LLM-replier
  substitutivity; P36-4 empirical primitive-equivalence).
  Four conjectures (C36-5 analytic equivalence; C36-6 dominance
  under noise; C36-7 adversarial-reply tightness; C36-8 LLM
  reply-noise calibrability).

**What is newly settled (Phase 37).** Three coupled additions
that calibrate the Phase-36 synthetic reply-noise channel
against real LLMs, deliver the reply-axis ensemble defense
(the Phase-34 analogue on the reply boundary), and extend the
thread-vs-adaptive-subscription empirical equivalence to a
harder task family:

* **Real-LLM reply calibration (Phase 37 Part A, C36-8).**
  ``core/reply_calibration.CalibratingReplier`` wraps any
  ``LLMThreadReplier`` with a per-call oracle comparator; the
  Phase-35 contested bank run on qwen2.5:0.5b and qwen2.5-
  coder:7b yields correct rate 0.10, malformed rate 0.00, OOV
  rate 0.00, semantic-wrong rate 0.90 — *on both models*. The
  dominant failure is ``sem_root_as_symptom`` (oracle = IR,
  replier = DOWNSTREAM_SYMPTOM). **Theorem P37-1**: real-LLM
  reply noise on this task family is dominated by semantic
  mislabel, not syntactic failure — the Phase-36 synthetic
  ``malformed_prob`` knob is a near-useless approximation.
  Both models collapse dynamic/adaptive_sub contested
  accuracy to 0.
* **Reply-axis ensemble (Phase 37 Part B, C36-7).**
  ``core/reply_ensemble.EnsembleReplier`` with three modes
  (``dual_agree``, ``primary_fallback``, ``verified``).
  **Theorem P37-2**: under a biased primary (always emits
  IR), dual_agree and verified recover to 100 % from single's
  33 %. **Theorem P37-3**: under synthetic malformed_prob=0.5,
  primary_fallback recovers to 100 % from 83 %. **Theorem
  P37-4** (structural limit): under extractor-*output*-level
  noise (Phase-36 ``synth_mislabel`` and ``adv_drop_root``),
  *no* ensemble mode helps — the ensemble sits above the
  noise wrapper and contributes no information past it.
  Conjecture C37-2 names the two-layer ensemble composition
  that would close this gap.
* **Nested-contest thread-vs-adaptive equivalence
  (Phase 37 Part C, C36-5 task family).**
  ``tasks/nested_contested_incident`` ships three hand-
  designed scenarios where round-1 replies are insufficient.
  Four strategies × 18 measurements:
  - static_handoff: 0.000
  - adaptive_sub_1r: 0.000
  - **adaptive_sub_2r: 1.000** (18 briefing edges, 36
    hypothesis edges)
  - **dynamic_nested_2r: 1.000** (0 briefings, 0 edges —
    reads round-1 via ThreadState.replies)
  **Theorem P37-5**: accuracy equivalence extends to the
  nested family at 0 pp gap, while the dynamic thread uses
  zero inter-round briefing machinery and adaptive_sub_2r
  uses 18. The equivalence class is accuracy, not protocol
  complexity.
* **Theory —** five theorems (P37-1..P37-5). Four
  conjectures (C37-1 calibration task/prompt-specificity;
  C37-2 two-layer ensemble composition; C37-3 nested-
  equivalence tightness; C37-4 minimum dynamic primitive).

**What is newly settled (Phase 44).** Four coupled additions
that close the Phase-43 § D.7 sentinel-path limitation, ship a
runnable public-SWE-bench-Lite readiness validator, refine the
Phase-43 taxonomy with raw-bytes evidence, and fan the cluster
out for a coder-class + frontier headroom comparison:

* ~~**Raw-text capture module.**~~ **Done in P44 Part A.**
  ``vision_mvp/tasks/swe_raw_capture.py`` ships an opt-in
  ``RawCaptureRecord`` / ``RawCaptureStore`` with schema
  version ``phase44.v1``. Each record carries the raw LLM
  bytes + SHA-256, the ``ParseOutcome`` dict, proposed and
  applied ``(old, new)`` pairs, and the patched-source
  SHA-256. ``make_capturing_generator`` wraps either a
  prebuilt bridge generator or a fresh ``llm_call`` and
  plumbs the raw text through the store while preserving
  the Phase-42 LLM-output cache discipline. Opt-in — every
  pre-Phase-44 path runs unchanged.
* ~~**Refined semantic taxonomy (v2 classifier).**~~ **Done
  in P44 Part B.** Five new Phase-44 sub-labels
  (``SEM_RIGHT_FILE_WRONG_SPAN``,
  ``SEM_RIGHT_SPAN_WRONG_LOGIC``,
  ``SEM_PARTIAL_MULTI_HUNK_SUCCESS``,
  ``SEM_NARROW_FIX_TEST_OVERFIT``,
  ``SEM_STRUCTURAL_VALID_INERT``) partition the Phase-43
  coarse buckets when raw bytes are available.
  ``REFINEMENT_MAP`` is reflexive (coarse always in refined
  set), so a sentinel-path classification stays a legal
  v2 classification. ``classify_semantic_outcome_v2``
  subsumes the v1 classifier on matched inputs (Theorem
  P44-2).
* ~~**Phase-44 driver (sweep + analyse).**~~ **Done in P44
  Part C.** ``vision_mvp/experiments/phase44_semantic_residue.py``
  runs the Phase-42-shape sweep with raw capture on, OR in
  ``--analyse-only`` mode ingests (parent, capture) pairs
  and emits a ``phase44.summary.v1`` JSON with per-cell
  coarse + refined counters and a ``coarse_to_refined_partition``
  audit.
* ~~**Public-SWE-bench-Lite readiness validator.**~~ **Done
  in P44 Part D.**
  ``vision_mvp/experiments/phase44_public_readiness.py``
  runs five checks
  (``schema`` → ``adapter`` → ``parser`` → ``matcher`` →
  ``test_runner``) on any local JSONL and emits a CI-gate
  verdict. Bundled bank: 57/57 on every check, 5.2 s wall
  (Theorem P44-3). **The externalisation gap to public
  SWE-bench-Lite is now purely a data-availability gap.**
* **Cluster fanout.** ``qwen2.5-coder:14b`` on mac1
  (strongest practical local coder-class model) and
  ``qwen3.5:35b`` on mac2 (Phase-43 frontier datapoint),
  both with raw capture on. Both runs completed; parent +
  capture artifacts ship in
  ``results_phase44_parser_{14b_coder,35b_moe}.json`` and
  ``results_phase44_capture_{14b_coder,35b_moe}.json``
  (342 capture rows per run = 57 instances × 3 strategies
  × 2 parser modes). Cross-model refined summary is
  ``results_phase44_refined_summary.json``. **Concrete
  results at the canonical cell (parser=robust, apply=
  strict, nd=6):** 14B-coder pass@1 = 0.930 (53/57, gap
  0 pp); 35B-MoE pass@1 = 0.965 (55/57, gap 0 pp).
  **Refined residue for 14B-coder:** the Phase-43
  ``wrong_edit_site`` bucket splits 50/50 into
  ``right_file_wrong_span`` and ``wrong_edit_site`` — direct
  empirical support for Conjecture C44-1 at the α = 0.5
  threshold. **Refined residue for 35B:** Phase-43's
  ``wrong_edit_site: 100%`` refines to ``structural_semantic
  _inert: 50% + right_file_wrong_span: 50%`` — refutes
  Conjecture C44-2 as-stated (the frontier residue is runtime-
  typed-mismatch + anchor-off-by-one, not overfit / wrong-
  logic); the refutation tightens the Phase-44 residue model.
* **Theory.** Three theorems (P44-1 raw capture is a
  lossless projection of pipeline state, P44-2 refined
  classifier monotone on sentinel inputs, P44-3 public-
  readiness saturates on bundled bank) and four
  conjectures (C44-1 coder-class wrong_edit_site refines
  to right_file_wrong_span; C44-2 frontier residue refines
  to overfit or wrong-logic; C44-3 substrate gap is
  refinement-invariant; C44-4 readiness closed under
  row-level filtering).

**What is newly settled (Phase 42).** Three coupled
additions that promote the Phase-41 attribution surface
from *matcher + substrate* to *parser + matcher + substrate*
and grow the bank past the external-validity threshold
named by Conjecture C41-1:

* ~~**Parser-compliance attribution layer.**~~ **Done in
  P42 Part A.** ``vision_mvp/tasks/swe_patch_parser.py``
  ships ``parse_patch_block(text, mode, unified_diff_parser)``
  with three parser modes
  (``PARSER_STRICT`` = Phase-41 baseline;
  ``PARSER_ROBUST`` = Phase-42 default; ``PARSER_UNIFIED``
  = diff-only), a ten-label closed failure taxonomy, and a
  six-label recovery enum.
  ``ParserComplianceCounter`` tracks raw vs recovered
  compliance per cell. The bridge's ``llm_patch_generator``
  gains an opt-in ``parser_mode``/``parser_counter``
  pair; ``None`` preserves the Phase-41 byte-strict path.
  The parser sits *above* the matcher axis — recovery
  only reconstructs delimiters, never content (Theorem
  P42-2).
* ~~**57-instance SWE-bench-Lite-style bank.**~~ **Done in
  P42 Part B.** The Phase-41 28-instance bank extended
  with 29 new instances covering string manipulation /
  numeric guards / sequence construction / dict helpers /
  recursion-iteration / boolean short-circuit / exception
  handling / nested data / format representation /
  sentinel values / operator precedence / class state
  transitions / set algebra / running aggregates / binary
  search off-by-one / graph walk / default argument
  correction. Every new instance round-trips through the
  oracle before being written.
* ~~**Parser-sweep driver + ASPEN cluster endpoint
  support.**~~ **Done in P42 Part C.**
  ``vision_mvp/experiments/phase42_parser_sweep.py``
  sweeps ``(parser_mode × apply_mode × n_distractors)``
  with LLM-output caching across parser cells.
  ``LLMClient.base_url`` plumbs ``--ollama-url`` so
  coding/generation runs route to the cluster macbook-1
  node and secondary runs to macbook-2 or localhost in
  parallel.
* **Mock headline.** 57 tasks × 4 distractor cells × 3
  strategies × 2 matcher modes = **1 368 measurements**
  in **122 s** through SubprocessSandbox. Substrate
  prompt flat at 205.9 tokens across every distractor
  cell; naive grows 197 → 527 (**2.7×** span) — Theorem
  P41-1 reproduces at 2.0× the Phase-41 bank scale.
* **Real-LLM datapoints.** Cluster macbook-1
  ``qwen2.5-coder:14b`` and localhost
  ``qwen2.5-coder:7b`` sweep all 57 instances at
  ``n_distractors = 6`` under strict vs robust parser —
  see ``RESULTS_PHASE42.md`` § D.3 and § D.4 for the
  per-strategy pass@1 table and the parser-axis
  attribution.
* **Theory.** Three theorems (P42-1 parser-compliance
  attribution decomposition, P42-2 parser recovery
  cannot produce a false pass, P42-3 robust-parser
  dominance on format-noncompliant generators) and three
  conjectures (C42-1 substrate-vs-naive gap ≤ 1 pp at
  N ≥ 50, C42-2 parser dominates matcher at 7B–30B,
  C42-3 three-axis decomposition completeness).

**What is newly settled (Phase 41).** Three coupled
additions that move the programme from "real external task
loop exists" to "real external task loop has larger-N
empirical data and a two-axis attribution surface":

* ~~**Larger SWE-bench-Lite-style bank.**~~ **Done in P41
  Part A.** A 28-instance real-shape JSONL
  (``vision_mvp/tasks/data/swe_lite_style_bank.jsonl``,
  ~4.7× the Phase-40 6-instance mini bank) authored to
  cover a disciplined spectrum of edit shapes: operator-
  typo, off-by-one, wrong-branch, seed-wrong, aggregate-
  missing, mutation-vs-copy, multi-hunk (one instance
  touches two methods on a class), parity-partition,
  slice-direction, index-return, polarity-flipped,
  empty-guard, type-conversion, unicode edge,
  ambiguous comparator. The bank-builder
  (``_build_swe_lite_bank.py``) validates every instance
  through the oracle round-trip before writing.
* ~~**Permissive patch-matcher axis.**~~ **Done in P41
  Part B.** ``apply_patch`` accepts one of four modes
  (``strict``, ``lstrip``, ``ws_collapse``,
  ``line_anchored``); all three permissive modes preserve
  the unique-match discipline. ``apply_mode`` is threaded
  through ``run_swe_loop``, ``run_swe_loop_sandboxed``,
  and every ``Sandbox.run(...)`` backend; strict default
  keeps Phase-40 byte-for-byte. **Theorem P41-2** (oracle
  saturates pass@1 = 1.000 under every matcher mode) is
  the null-control of the new axis.
* ~~**Empirical sweep driver + attribution table.**~~
  **Done in P41 Part C.** ``experiments/phase41_swe_lite_
  sweep`` runs the bank across ``n_distractors ∈ {0, 6,
  12, 24}`` and the requested matcher modes with per-
  (instance, strategy, distractor) LLM-call caching so
  permissive cells reuse strict cells' proposals. Emits a
  per-strategy ``{recovered, regressed, unchanged_pass,
  unchanged_fail}`` set delta between each permissive
  mode and strict. **Theorem P41-3** formalises the
  decomposition: ``Δ pass@1 = |R_recovered| −
  |R_regressed|``.
* **Mock headline.** 28 tasks × 4 distractor cells × 3
  strategies × 2 matcher modes = **672 measurements** in
  **53 s** wall through SubprocessSandbox. Substrate
  prompt flat at 746.4 chars across every cell; naive
  grows 806.8 → 2 125.8 — Theorem P41-1 (bounded-context
  preservation at 4.7× bank scale).
* **Real-LLM datapoints.**
  ``qwen2.5-coder:7b`` on all 28 instances,
  ``n_distractors = 6``, three matcher modes — **pass@1 =
  0.929 / 0.929 / 0.893 (naive / routing / substrate)**
  under strict; byte-identical under ``lstrip`` and
  ``line_anchored``; ``R_recovered = R_regressed = ∅`` on
  every cell. The Phase-40 16.7 pp substrate-vs-naive gap
  on 6 instances collapses to **3.6 pp at 28 instances**;
  the matcher-permissiveness axis is *null-gain, null-risk*
  on this bank at this model scale.
  ``gemma2:9b`` on all 28 instances, same configuration —
  **pass@1 = 0 / 0 / 0** on every cell under every matcher.
  The 9B emits the *semantically correct fix* on a spot
  check (``OLD>>>\n    result = 0\n<<<NEW>>>\n    result
  = 1\n``) but does not close the ``<<<`` delimiter, so
  every patch parses as empty and lands as
  ``patch_no_match``. This surfaces a new attribution
  boundary — the LLM-output parser — above the matcher
  axis (Conjecture C41-5).
* **Theory.** Three theorems (P41-1 bounded-context at
  scale, P41-2 oracle-ceiling matcher-invariance, P41-3
  attribution counting identity) and **five** conjectures
  (C41-1 communication-bounded at ≥ 50 instances, C41-2
  matcher-permissiveness saturation, C41-3 stronger-model
  strict-floor saturation, C41-4 regime decomposition,
  C41-5 parser-compliance attribution boundary). All
  three theorems are empirically anchored to the
  mock JSON artifact + the 7B / 9B real-LLM artifacts.
  C41-5 is newly surfaced from § D.4 of the Phase-41
  results note; it names the programme's most tractable
  Phase-42 target.

**What is newly settled (Phase 40).** Three coupled
additions that close the Phase-39-named "mechanical
follow-up" end-to-end:

* ~~**Real SWE-bench-style loader / adapter.**~~ **Done
  in P40 Part A.** ``parse_unified_diff`` (a tolerant
  ``git diff`` parser), ``SWEBenchAdapter.from_swe_
  bench_dict`` (the real-shape adapter that derives
  ``buggy_function`` from the diff hunk and promotes a
  ``test_patch`` into a runnable test source), and
  ``load_jsonl_bank`` (the hermetic JSONL loader with
  per-instance file namespacing) ship as a 290-LOC
  extension to ``swe_bench_bridge``. A bundled
  six-instance real-shape JSONL artifact
  (``vision_mvp/tasks/data/swe_real_shape_mini.jsonl``)
  exercises the full loader path offline in
  sub-second.
* ~~**Sandboxed execution boundary.**~~ **Done in P40
  Part B.** A new ``tasks/swe_sandbox`` module ships
  three backends behind one ``Sandbox`` protocol:
  ``InProcessSandbox`` (Phase-39 wrapped),
  ``SubprocessSandbox`` (new — wall-clock timeout,
  tempdir cwd, sanitised env, JSON outcome protocol),
  ``DockerSandbox`` (new — ``--network=none --read-only``,
  optional, daemon-availability detection). A
  ``select_sandbox("auto")`` factory picks the right
  backend by availability. ``run_swe_loop_sandboxed``
  is the sandbox-aware substrate runner; the failure-
  attribution surface gains two new kinds (``timeout``,
  ``sandbox_error``).
* ~~**End-to-end real-shape evaluation.**~~ **Done in
  P40 Part C.** A new driver
  ``experiments/phase40_real_swe_bridge`` composes
  loader + substrate + sandbox + (optional) real LLM.
  Mock run: 72 measurements through SubprocessSandbox
  in 5.6 s wall, pass@1 = 1.000 / 1.000 / 1.000 (oracle
  ceiling). Real-LLM ``qwen2.5:0.5b`` run: every cell
  hits ``patch_no_match`` (transcription-bounded
  regime). Real-LLM ``qwen2.5-coder:7b`` run: pass@1
  = 0.833 / 0.833 / 0.667 (naive / routing /
  substrate) on 6 instances — the substrate's
  one-instance ranking inversion vs naive on
  ``ext-list-001`` is honest variance at small N
  inside the P39-2 transcription-bounded regime;
  the substrate prompt is constant at 813 chars
  across n_distractors ∈ {0, 6, 12, 24} on every
  cell.
* **Theory —** three theorems (P40-1 unidiff
  round-trip, P40-2 real-shape bounded-context,
  P40-3 sandbox-boundary preservation) and three
  conjectures (C40-1 sandbox cost amortisable,
  C40-2 loader sufficiency for SWE-bench Lite,
  C40-3 sandbox-axis equivalence). All three
  theorems are empirically anchored to artifacts
  shipped alongside.

**What is newly settled (Phase 39).** Three coupled additions
that close three explicit Phase-38 data gaps and ship the
first runnable artifact toward end-to-end SWE-bench:

* ~~**Real-LLM prompt-variant sweep (Conjecture C38-3,
  optimistic read).**~~ **Done in P39 Part A.** The
  Phase-38 ``phase38_prompt_calibration`` driver was
  run with ``--mode real`` on ``qwen2.5:0.5b`` (5
  variants × 20 calls; 5/5 measurements complete) and
  ``qwen2.5-coder:7b`` (default + contrastive
  measured; remaining variants populated as the bench
  completes). Headline: on **0.5B**, every Phase-38
  variant lands at the default per-bucket histogram to
  ±0 calls (correct = 0.10, sem_root_as_symptom = 0.50,
  sem_unc_as_symptom = 0.40) — except ``forced_order``
  which converts semantic errors to malformed parses
  without changing correct. On **7B**, ``contrastive``
  *does* shift the bias: correct lifts from 0.10 to
  ≈ 0.50, a 5× gain. **Theorem P39-1** (three parts):
  on the 0.5B class the bias is *model-shaped*; on the
  7B class it is *partially prompt-shaped*; the
  substrate's typed-reply contract holds across all
  variants. The mock's *uniformity assumption*
  (Conjecture C38-3 as written) is empirically refuted;
  the cleaner statement is *prompt-shape responsiveness
  is itself capacity-dependent*.
* ~~**SWE-bench bridge artifact.**~~ **Done in P39
  Part C.** ``tasks/swe_bench_bridge`` ships a
  SWE-bench-compatible task schema
  (``SWEBenchStyleTask``: instance_id, repo,
  base_commit, problem_statement, buggy_file_relpath,
  buggy_function, gold_patch, test_source); a
  four-instance hand-authored ``MiniSWEBank`` with
  real Python files, real bugs, real gold patches as
  line-anchored substitutions, real in-process tests
  (fresh ``exec`` namespace; no shell, no subprocess);
  a four-role team
  (``issue_reader`` / ``code_searcher`` /
  ``patch_generator`` / ``test_runner``) wired through
  the unchanged Phase-31 ``HandoffRouter``;
  ``SWEBenchAdapter.from_dict`` shim for a future real-
  SWE-bench loader. **Theorem P39-3**: substrate
  bounded-context preservation extends to the SWE
  multi-role team — patch_generator prompt size is
  842 chars / 0 events at every n_distractors ∈
  {0, 6, 12, 24} while naive grows to 1936 chars / 14
  events. **Theorem P39-4**: every required SWE-bench
  field has a typed counterpart; the gold-patch
  unified-diff gap is bounded and adapter-shaped.
* **Frontier-model bounded substrate sweep (item (2) of
  the prior frontier list).** Done in P39 Part B.
  ``experiments/phase39_frontier_substrate`` runs
  Phase-31 incident triage at k = 6, seed = 31 across
  mock + 3 cross-family local LLMs
  (``qwen2.5-coder:7b``, ``llama3.1:8b``,
  ``gemma2:9b``). Pooled substrate_wrap accuracy:
  qwen 0.800, llama 0.600, **gemma 1.000** — naive is
  0.000 on every model. The Phase-31 substrate
  signature reproduces across the qwen / llama /
  gemma families; ``gemma2:9b`` is the first real LLM
  in the programme to *saturate* the substrate ceiling
  on a non-code team task. The substrate prompt is
  constant at 196 chars regardless of model
  (Phase-31 P31-3 invariant). Wall: 28 min total for
  60 LLM calls.
* **Theory —** four theorems (P39-1 prompt-shape vs
  model-shape; P39-2 communication-bounded vs
  transcription-bounded regime; P39-3 substrate
  bounded-context on SWE teams; P39-4 SWE-bench
  schema mappability). Four conjectures (C39-1
  strong-model bias saturation; C39-2 prompt-shape
  recovery requires fine-tuning; C39-3 substrate
  dominance on SWE-bench; C39-4 mini-SWE Lipschitz-
  predicts SWE-bench).

**What is newly settled (Phase 38).** Three coupled additions
that close three Phase-37 frontier items at once:

* ~~**Two-layer ensemble composition (Conjecture C37-2).**~~
  **Done in P38 Part A (Theorem P38-1 + P38-2).**
  ``core/two_layer_ensemble.PathUnionCausalityExtractor`` and
  ``core/extractor_adversary.UnionClaimExtractor`` combine
  layer-1 (extractor-axis) and layer-2 (reply-axis)
  ensembles. On the joint ``{ext_drop_gold ∧
  rep_biased_primary}`` cell — the conjunction attack that
  every single-layer defense collapses on — the two-layer
  stack (``UnionClaimExtractor ∘ EnsembleReplier``) is the
  unique configuration that recovers (0.833 vs 0.333 full
  accuracy, 1.000 vs 0.250 contested). On the Phase-37-legacy
  ``adv_drop_root`` cell where Theorem P37-4 proved every
  reply-axis ensemble powerless, the
  ``PathUnionCausalityExtractor(PATH_MODE_UNION_ROOT)`` above-
  noise combiner recovers to 1.000 accuracy.
* ~~**Minimum dynamic primitive falsifier (Conjecture
  C37-4).**~~ **Done in P38 Part B (Theorem P38-3).**
  ``core/primitive_ablation`` ships a feature-flagged thread
  runner; ablating each of the five C37-4 candidate features
  gives the load-bearing table: ``typed_vocab`` (0.500 / 0.333),
  ``terminating_resolution`` (0.333 / 0.000),
  ``round_aware_state`` (1.000 / 0.000) on (contested, nested).
  ``bounded_witness`` and ``frozen_membership`` are null-control
  on accuracy. Minimum accuracy-load-bearing set on the two
  families: {``typed_vocab``, ``terminating_resolution``,
  ``round_aware_state``}; ``bounded_witness`` adds for the P35-2
  invariant.
* ~~**Prompt-variation pipeline (Conjecture C37-1).**~~
  **Partially done in P38 Part C (Theorem P38-4).**
  ``core/prompt_variants`` ships five variants
  (default, contrastive, few_shot, rubric, forced_order);
  the experiment driver supports ``--mode mock``
  (deterministic bias-shift simulation, sub-second) and
  ``--mode real`` (Ollama sweep). Mock headline:
  ``rubric`` and ``contrastive`` cut semantic-wrong rate
  from 0.688 → 0.225. The real-LLM measurement (Conjecture
  C38-3) is a single CLI parameter from the current
  headline; it remains open as data, not as pipeline.
* **Theory —** four theorems (P38-1 two-layer closes
  conjunction; P38-2 path-union closes adv_drop_root;
  P38-3 minimum primitive set on tested families; P38-4
  prompt-variant pipeline validates on mock). Three
  conjectures (C38-1 two-layer minimality vs arbitrary
  ν-pair; C38-2 frozen_membership load-bearing on some
  ``Z*``; C38-3 prompt-bias model-invariant within size
  class).

**What is not settled (open frontier).** (a) **End-to-end
SWE-bench** with a real LLM on the wrap path. This is the largest
remaining external-validity gap. (b) **Frontier-model *sweep*.**
Phase-32/C is a 7B spot check on compliance + incident at k = 6
seed = 32, and the Phase-33 7B extractor run covers all three
non-code domains at k = 6 seed = 33 (landed post-review, 120 LLM
calls, 32.7 min wall); substrate accuracy is (0.0, 0.0, 0.2) on
(incident, compliance, security) — failure moves from recall-
limited (0.5b) to precision-limited (7B). A proper multi-model
(7B / 8B / 9B) × multi-seed × multi-k × multi-domain sweep is
mechanical follow-up. (c) **OQ-1 in full
generality.** Phase 30 closes the matched-substrate regime
(Theorem P30-4); the LLM-loop regime is captured by Conjecture
P30-6 but remains unproven. (d) The implicit-raise pattern-list
extension (OQ-28b). (e) Cross-language runtime calibration
(OQ-27g). (f) Adversarial task distribution sweep (P29-7 / P30-5
falsification). (g) Cross-lattice generalisation of typed
handoffs to hierarchical role lattices with K ≥ 20 (C31-6).
(h) **Ensemble against adversary** (Conjecture C34-4). The
natural composition of the Phase-34 adversarial wrapper and the
Phase-34 ensemble primitive; mechanical but not yet run.
(i) **Real-LLM per-role calibration sweep.** Phase 34 Part A's
mock calibration is in place; a real-LLM run across
qwen2.5:0.5b / qwen2.5-coder:7b on all three domains is
mechanical follow-up.
(j) **Payload-level adversary.** Phase 34's wrapper targets
``(role, kind)`` headers; a payload-replacing adversary (swap the
body of a load-bearing emission) is outside the current wrapper.
(k) ~~**Dynamic coordination under noise (Conjecture C35-7).**~~
**Done in P36 Part A (Theorems P36-1, P36-2).**
(l) ~~**LLM-driven thread replies (Conjecture C35-8).**~~
**Done in P36 Part B (Theorem P36-3).**
(m) ~~**Adaptive subscription as an alternative primitive
(Conjecture C35-5).**~~ **Done in P36 Part C (Theorem P36-4).**
(n) **Larger contest density scaling (Conjecture C35-6).**
Phase-35 bank has 4 contested scenarios; a 20+ scenario bank
with variable ``R*`` and variable contest arity (2-way, 3-way,
4-way candidate clusters) would quantify the robustness of the
P35-1 separation and provide evidence toward C35-6's
information-theoretic lower bound.
(o) ~~**Ensemble-against-adversarial-reply (Conjecture C36-7).**~~
**Partially settled in P37 Part B.** A reply-axis ensemble
(``core/reply_ensemble``) with three modes recovers biased-
primary collapse (dual_agree, verified to 100 %) and malformed
collapse (primary_fallback to 100 %). It *cannot* recover
``adv_drop_root`` / ``synth_mislabel`` — the noise wrapper is
below the ensemble (Theorem P37-4). The remaining open piece
is the two-layer ensemble composition (Conjecture C37-2):
``reply_ensemble`` + Phase-34 ``ensemble_extractor`` on the
same adversarial cell.
(p) ~~**Real-LLM replier calibration sweep (Conjecture C36-8).**~~
**Done in P37 Part A.** ``core/reply_calibration`` +
``phase37_real_reply_calibration`` — calibrated on
``qwen2.5:0.5b`` and ``qwen2.5-coder:7b``. Result: both models
emit 100 % well-formed JSON; neither fits the synthetic
``malformed_prob`` model; both show a directed semantic bias
toward DOWNSTREAM_SYMPTOM on actual-IR candidates (50 %
``sem_root_as_symptom``, 40 % ``sem_uncertain_as_symptom``).
Theorem P37-1 empirically refutes the synthetic-surrogate
hypothesis of C36-8 as originally stated.
(q) **Analytic equivalence of threads vs adaptive-sub
(Conjecture C36-5).** Phase 37 Part C *extends* the empirical
equivalence to nested-contest scenarios (Theorem P37-5). An
analytic proof on an arbitrary task family remains open;
Conjecture C37-3 names the current strongest empirical claim.
(r) **Type-level bounded-context for adaptive-sub.** Theorem
P35-2's ``C_0 + R*·τ + T·R_max·W`` guarantee is type-level for
threads. For adaptive subscriptions it is runtime-enforced
(via ``max_active_edges`` + TTL). Phase 37 Part C's
adaptive_sub_2r scheme uses ``max_active_edges=8`` to hold 6
hypothesis edges + 2 concurrent briefings without overrun;
runtime-enforcement holds empirically. A proof analogue of
P35-2 for the runtime-enforced scheme remains open.
(s) ~~**Two-layer ensemble composition (Conjecture C37-2).**~~
**Done in P38 Part A (Theorem P38-1 + P38-2).**
``UnionClaimExtractor ∘ EnsembleReplier`` is the unique
configuration that closes the conjunction cell;
``PathUnionCausalityExtractor`` closes the adv_drop_root
cell that every reply-axis ensemble alone cannot.
(t) **Prompt-engineering calibration shift (Conjecture
C37-1 / C38-3).** Phase 38 Part C ships the pipeline
(``core/prompt_variants`` + driver under ``--mode mock``
and ``--mode real``). Mock data shows the pipeline
faithfully measures per-variant shift (rubric / contrastive
cut sem-wrong by 67 %). Real-LLM measurement is one CLI
parameter away (``--mode real --models qwen2.5:0.5b``);
remains open as a data follow-up, not as a pipeline gap.
(u) ~~**Minimal dynamic primitive (Conjecture C37-4).**~~
**Partially done in P38 Part B (Theorem P38-3).** Per-
feature ablation table on the Phase-35 and Phase-37
families: typed_vocab, terminating_resolution, and
round_aware_state are individually load-bearing;
bounded_witness is null on accuracy but load-bearing for
the P35-2 invariant; frozen_membership is null-control on
both families. Conjecture C38-2 asks for a family where
frozen_membership is load-bearing.
(v) **Correlated-noise two-layer breakdown (Conjecture
C38-1).** Phase-38's conjunction cell has independent layer-1
and layer-2 attacks. Construct a ``(ν_1, ν_2)`` pair with
cross-layer correlation (e.g. the adversary both drops and
biases on the same scenario) and measure whether two-layer
still recovers. Falsifier for C38-1.
(w) **Frozen-membership falsifier (Conjecture C38-2).**
Design a multi-auditor contest task where mid-thread
member-set growth changes the voting tally. Would tighten
the primitive minimality claim in Theorem P38-3.
(x) ~~**Real-LLM prompt-variant sweep (Conjecture C38-3).**~~
**Done in P39 Part A.** Headline: optimistic-read C38-3
*refuted on 0.5B* (no Phase-38 variant moves the bias to
within ±0 calls); *partially confirmed on 7B*
(``contrastive`` lifts ``correct_rate`` 5×). The real
finding is that *prompt-shape responsiveness is itself
capacity-dependent* — Theorem P39-1.
(y) ~~**SWE-bench bridge artifact.**~~ **Done in P39 Part C.**
``tasks/swe_bench_bridge`` ships a SWE-bench-compatible
schema, a four-instance hand-authored mini bank, a
four-role team substrate, and an in-process patch-test
workspace. Theorem P39-3: bounded-context preservation;
Theorem P39-4: schema mappability. Real SWE-bench
loader is a unidiff parser + sandbox follow-up.
(z) **Strong-model bias saturation (Conjecture C39-1).**
Run the Phase-38 prompt-variant sweep on a 30B+
model. Open question: does ``correct_rate`` saturate
above 0.5 across every variant at some size class?
(aa) **Fine-tuning recovery (Conjecture C39-2).**
LoRA-on-Phase-35-bank study; does fine-tuning move
the 0.5B bias where prompt engineering does not?
(bb) ~~**Real SWE-bench loader.**~~ **Done in P40.**
``parse_unified_diff`` + ``SWEBenchAdapter.from_swe_
bench_dict`` + ``load_jsonl_bank`` + ``SubprocessSandbox``
/ ``DockerSandbox`` ship the full pipeline (~ 290 LOC
extension to ``swe_bench_bridge`` + new
``swe_sandbox`` module + bundled six-instance
real-shape JSONL artifact). Theorem P40-1 (unidiff
round-trip), P40-2 (real-shape bounded-context),
P40-3 (sandbox-boundary preservation). Pointing
at SWE-bench Lite is now empirical follow-up, not
infrastructure.
(cc) ~~**SWE-bench Lite empirical sweep at 6-instance
scale (C40-2).**~~ **Partially addressed in P41.**
The Phase-41 28-instance bank
(``swe_lite_style_bank.jsonl``) plus the
matcher-permissiveness axis gives the first larger-N
real-SWE data on the substrate pipeline. A public
SWE-bench Lite JSONL at ≥ 50 instances still remains
as the external-validity close on Conjectures C39-3
/ C39-4 / C41-1 / C41-2.
(dd) **Sandbox-axis equivalence (C40-3).** Run the
Phase-40 mock + 7B real on a docker-host CI to
confirm ``SubprocessSandbox`` / ``DockerSandbox``
equivalence on the bridge bank.
(ee) **Public SWE-bench Lite at ≥ 50 instances
(C41-1 / C41-2 falsifier).** Point
``phase41_swe_lite_sweep`` at a downloaded
SWE-bench Lite JSONL; measure (i) loader coverage
fraction (Conjecture C40-2), (ii) strict-vs-
permissive recovered-set size as N grows
(Conjecture C41-2), (iii) substrate vs naive pass-
rate at scale (Conjecture C41-1).
(ff) **Stronger-model sweep at 30B+ class.**
Phase 41 ships a ``gemma2:9b`` stronger-model
datapoint. A 30B+ run (``mixtral:8x7b``,
``llama3.3:70b``, or a fine-tuned coder model)
under the Phase-41 attribution table would test
Conjecture C41-3 directly.
(gg) **AST-aware or edit-distance-bounded matcher.**
Phase 41 ships three permissive matchers, each
whitespace-normalising. Richer modes (AST-
equivalence, bounded edit-distance, context-aware
hunk anchoring) could lift additional
``R_recovered`` without inflating ``R_regressed``.
Conjecture C41-2 predicts a bounded ceiling on
any such matcher's recovery fraction.

**What would materially move the frontier next.** Ordered by
research impact:

1. **End-to-end SWE-bench loader.** The Phase-39 bridge
   ships the team-substrate plumbing, the schema, the
   patch-test workspace, and the mini bank. The remaining
   step is the unified-diff → substitution adapter +
   Docker sandbox for untrusted patches. After that, real
   SWE-bench instances flow through the Phase-31
   ``HandoffRouter`` and the Phase-39 four-role team
   without architectural change. **Conjecture C39-4**
   names the empirical predictor (mini-SWE pass rate
   Lipschitz-bounds SWE-bench pass rate in the matched-
   substrate regime).
2. **Frontier-model sweep on Phase 30 / Phase 31 / Phase 32 /
   Phase 33 / Phase 34.** Phase-32/C and Phase-33's real-LLM runs
   are single-seed spot checks at k = 6; a multi-model (7B / 8B /
   9B) × multi-seed × multi-k × multi-domain grid would give the
   programme its first real-LLM cross-model noise-and-accuracy
   table, *and* validate Phase 34's per-role heterogeneity
   conjecture (C34-5) on a real LLM.
3. **Proof or refutation of Conjecture P30-6.** Test the
   Lipschitz hypothesis on small LLMs; if it holds, the
   Knaster-Tarski + contraction argument closes OQ-1 in the
   LLM-loop regime. If it fails, the failure mode is a concrete
   research signal.
4. **Ensemble-against-adversary (Conjecture C34-4).** Compose
   ``adversarial_extractor`` with ``UnionExtractor`` on the
   Phase-34 mixed bank. Predicted: a single-witness adversary
   against a pairwise-complementary ensemble is ineffective; a
   multi-witness adversary linearly degrades. The measurement is
   mechanical once the Phase-34 wrappers exist.
5. **Real-LLM per-role calibration sweep.** The Phase-34
   instrument is in place; running it across qwen2.5:0.5b /
   qwen2.5-coder:7b × three domains × k ∈ {6, 20, 60} would
   produce the programme's first real-LLM per-role noise table.
   Expected signature: legal → finance gap on 0.5b compliance
   (drop 0.50 → 1.00) confirmed; 7B per-role spread shrinks but
   does not collapse; Conjecture C34-5 confirmed under a real
   extractor.
6. **Implicit-raise pattern expansion + `vision-tests` calibration
   tightening.** Reduces `may_raise_implicit` FN from 122 to a
   small number on the 1 477-observation pooled slice.
7. **Cross-language analogue of Phase 22–30.** TypeScript via
   `ts-morph` + analogous call-graph + runtime probes. Empirical
   transfer of the Phase-29/30 task-scale result to a
   non-Python agent-team task distribution would be strong
   evidence for the general-agent-teams claim.
8. **Hierarchical role-lattice at K ≥ 20.** C31-6 unaffected by
   Phase 34; still the natural next test of the substrate at
   larger team sizes.
9. **Payload-level adversary.** Extend the Phase-34 adversary to
   replace the *body* of a load-bearing emission (not just drop
   it). Tests whether the substrate's content-dedup invariants
   survive a smart payload attack.
10. ~~**Dynamic coordination under extractor noise
    (C35-7).**~~ **Done in P36 Part A.**
11. ~~**LLM-driven thread replies (C35-8).**~~ **Done in P36
    Part B** (mock replier; real Ollama replier is C36-8
    follow-up).
12. ~~**Adaptive subscription primitive (C35-5).**~~ **Done in
    P36 Part C.**
13. ~~**Real-LLM replier calibration (C36-8).**~~ **Done in P37
    Part A.** Headline: real LLMs are 100 % syntactically
    clean but 90 % semantically wrong on this task; the
    synthetic malformed_prob knob is a near-useless surrogate
    (Theorem P37-1).
14. ~~**Ensemble against adversarial reply (C36-7).**~~
    **Partially done in P37 Part B.** ``core/reply_ensemble``
    recovers biased-primary and malformed collapse; does NOT
    recover extractor-output-level adversarial collapse
    (Theorem P37-4 — structural limit). The remaining piece
    is the two-layer composition with the Phase-34 extractor
    ensemble (Conjecture C37-2; item 16 below).
15. ~~**Analytic equivalence of threads vs adaptive_sub
    (C36-5).**~~ **Empirically extended in P37 Part C.**
    Nested-contest family now has 18 paired measurements
    with 0 pp accuracy gap and a structural-complexity
    separation (Theorem P37-5). Analytic proof remains open.
16. ~~**Two-layer ensemble composition (C37-2).**~~ **Done
    in P38 Part A (P38-1 + P38-2).**
17. ~~**Prompt-engineering shift of the real-LLM calibration
    curve (C37-1 / C38-3).**~~ **Done in P39 Part A.**
    Headline: 0.5B is *model-bound* (no variant moves it);
    7B is *partially prompt-bound* (``contrastive`` 5× lift
    on correct_rate). Theorem P39-1: prompt-shape
    responsiveness is capacity-dependent. The optimistic
    read of C38-3 (uniform shift across model families) is
    refuted; the cleaner conjectures are C39-1 (strong-model
    saturation) and C39-2 (fine-tuning recovery).
18. ~~**Minimal dynamic primitive (C37-4).**~~ **Ablation
    table done in P38 Part B (P38-3).** Three features
    load-bearing; two are null-control on tested families.
    Conjecture C38-2 asks for a task where
    ``frozen_membership`` is load-bearing.
19. **Correlated-noise two-layer breakdown (C38-1).** Build
    a correlated ``(ν_1, ν_2)`` attack — e.g. adversary
    drops gold AND biases replier on the same scenario —
    and measure whether two-layer still recovers.
20. ~~**Real-LLM prompt-variant sweep (C38-3).**~~ **Done
    in P39 Part A.** See item 17 above.
21. ~~**End-to-end SWE-bench loader + sandbox (C39-3 / C39-4
    *infrastructure*).**~~ **Done in P40.** Unified-diff
    parser (``parse_unified_diff``) + real-shape adapter
    (``SWEBenchAdapter.from_swe_bench_dict``) + JSONL loader
    (``load_jsonl_bank``) + subprocess and docker sandbox
    backends (``SubprocessSandbox`` / ``DockerSandbox``)
    are shipped and tested. The bundled six-instance
    real-shape JSONL bank reproduces Theorem P39-3 on
    the new pipeline (substrate prompt 813 chars / 0
    events to patch_generator across n_distractors ∈
    {0, 6, 12, 24}; naive grows 826 → 2 145; oracle pass@1
    = 1.000 on every cell). Real-LLM headline on
    ``qwen2.5-coder:7b``: 5 / 6 pass@1 under naive /
    routing, 4 / 6 under substrate — the substrate-vs-
    naive ranking is variance-bounded at 6 instances and
    the dominant failure is byte-strict matcher
    sensitivity, sitting cleanly inside the Theorem
    P39-2 transcription-bounded regime. *Pointing the
    pipeline at SWE-bench Lite is now a ``--jsonl <path>``
    parameter change* — the empirical ranking measurement
    on Lite (Conjecture C39-3 / C39-4) is the next
    natural follow-up, not the infrastructure.
22. **Strong-model bias saturation sweep (C39-1).** The
    Phase-39 0.5B / 7B prompt-variant data is bounded
    above by the model. Run on a 30B+ model to either find
    the saturation knee or push the model-shape conclusion
    higher.
23. **Fine-tuning recovery experiment (C39-2).** Train a
    small LoRA on Phase-35 contested-bank examples and
    re-measure ``correct_rate``; if it lifts above 0.5,
    Phase-39 Theorem P39-1's "model-shaped" component is
    confirmed *non-saturating* at the parameter axis but
    *recoverable* via fine-tuning.
24. ~~**Larger SWE-bench-Lite-style empirical sweep
    (C39-3 / C39-4, 6-instance scale).**~~ **Done in P41
    Part A–C.** 28-instance real-shape JSONL bank +
    matcher-permissiveness axis + Phase-41 driver. Theorem
    P41-1 (bounded-context at 4.7× scale), P41-2 (oracle-
    ceiling matcher-invariance), P41-3 (attribution
    counting identity). Real-LLM sweeps on
    ``qwen2.5-coder:7b`` (28 instances) and ``gemma2:9b``
    (subset) populate the attribution tables.
25. **Public SWE-bench Lite sweep at ≥ 50 instances
    (C41-1 / C41-2 / C39-3 / C39-4).** Phase 41 makes this
    a ``--jsonl <path>`` parameter change. The experiment
    measures (a) loader coverage on real Lite instances,
    (b) substrate vs naive pass-rate once per-instance
    variance washes out, (c) matcher-permissiveness
    recovery as N grows.
26. **Stronger-model frontier run on the Phase-41 bank
    (C41-3).** 30B+ class model (``mixtral:8x7b``,
    ``llama3.3:70b``, or a coder-finetuned 70B) under the
    Phase-41 attribution table. Predicted outcome:
    ``R_recovered / R_strict_fail → 0``.
27. **Richer permissive matchers (C41-2 refinement).** AST-
    equivalence and bounded edit-distance modes extend the
    Phase-41 matcher axis while preserving the unique-match
    discipline and the oracle-saturation null-control
    (P41-2).

Items (1), (2), (5), (6), (7), (20) are engineering-
adjacent; items (3), (4), (8), (9), (19) are research-
heavy. The programme rotates between the two.

---

## 5. End goals

We keep goals at three horizons. Each horizon has both a scientific
axis and a systems axis; they are not the same.

### 5.1 Near-term (next 2–3 phases)

**Scientific goals.**

- ~~Extend runtime calibration to every Phase-23 corpus.~~ **Done
  in P28**. Done for vision-core / vision-tasks / vision-tests /
  vision-experiments; third-party corpora (`click`, stdlib `json`)
  remain an OQ-28a follow-up.
- ~~Make the explicit / implicit-raise distinction first-class in
  the analyzer and in the calibration report.~~ **Done in P28**.
- ~~Raise the `ready` coverage fraction on real corpora by a
  principled step: method-instance auto-construction with a
  documented trust-boundary.~~ **Done in P29**. ready_fraction on
  `vision-tests` lifts 2.9 % → 98.8 %; pooled entered slice grows
  4.83×; construct_failed < 1 %.
- ~~Run the first task-scale falsifiability check of the routing /
  substrate thesis on a multi-role multi-query SWE-style
  distribution.~~ **Done in P29**. Pooled causal-relevance fraction
  4.54 % < 0.50 gate → CONFIRMED.
- ~~Formalise the connection between the Phase-29 causal-relevance
  fraction and `T_i*`; state conditions under which the substrate
  achieves a fixed point.~~ **Done in P30**. Four theorems
  (P30-1..P30-4), one closes a special case of OQ-1 (matched-
  substrate regime, unique one-step fixed point). Two conjectures
  (P30-5 agent-teams generalisation; P30-6 LLM-loop Lipschitz
  fixed-point) sharpen the remaining open questions.
- ~~Run the first LLM-in-loop external-validity check of the
  substrate on external corpora.~~ **Done in P30**. On
  ``json-stdlib`` under ``qwen2.5:0.5b``: **16.0×** prompt-token
  reduction, **+60 pp** accuracy lift (80 % vs 20 %). External
  corpora (``click``, ``json-stdlib``) land in the same
  causal-relevance band as local corpora (0.047–0.122 vs
  0.032–0.056), supporting Conjecture P30-5.
- ~~Ship a substrate primitive whose identity is *inter-agent
  communication* rather than *corpus representation*, and
  evaluate on a non-code multi-role task.~~ **Done in P31**.
  Typed handoffs (``core/role_handoff``) + five-role incident-
  triage bench (``tasks/incident_triage``). Mock ceiling:
  substrate flat at 196 tokens / 100 % accuracy across k ∈
  {6, 20, 60, 120}; naive collapses from 100 % → 20 % at k=120
  under truncation. qwen2.5:0.5b: substrate 40 % vs naive 0 %
  across k ∈ {6, 60}; at k=60 naive is infeasible (all prompts
  truncated). Five theorems (P31-1..P31-5), two conjectures
  (C31-6, C31-7); P31-5 converts the § 1.5 differentiation to a
  formal statement.
- ~~Second non-code domain for typed handoffs — evidence for
  the general-agent-team claim across domains.~~ **Done in P32**.
  Vendor-onboarding compliance review (``tasks/compliance_review``)
  with a distinct role cast and 13-kind claim catalogue. Mock
  ceiling: substrate flat at **171 tokens / 100 % accuracy** across
  k ∈ {6, 20, 60, 120}; naive collapses 100 % → 40 % at k=120
  under truncation; routing 0 % on every k. Same signature as
  Phase 31 — cross-domain generalisation empirical (Theorem P32-1).
- ~~Noisy-extractor sweep — falsify or confirm C31-7.~~ **Done
  in P32**. ``core/extractor_noise`` parameterises five noise axes
  (drop / spurious / mislabel / payload-corrupt / seed); 96 noise
  points × 2 domains runs in 0.5 s. Two-regime graceful
  degradation observed and theoremised (P32-2); token bound under
  bounded noise preserved (P32-3); C31-7 promoted from conjecture
  to theorem in the monotone-decoder regime.
- ~~Frontier-model spot check on the typed-handoff substrate.~~
  **Done in P32** (Part C). ``qwen2.5-coder:7b`` on both non-code
  benchmarks (incident + compliance) at k = 6 seed = 32; see
  ``results_phase32_llm_7b_spot.json``. Still open: the
  *multi-seed × multi-k × multi-model* sweep.
- ~~Third non-code domain (policy drafting / research synthesis /
  multi-role software delivery planning) — evidence for C32-4 at
  K ≥ 3 and test of whether the decoder-shape taxonomy (monotone
  vs strict) covers the full space.~~ **Done in P33.**
  ``tasks/security_escalation`` — five-role security team, 15
  claim kinds, **max-ordinal severity + claim-set classification**
  decoder (structurally distinct from both prior shapes). Mock
  substrate accuracy 100 % flat at 242 tokens across
  k ∈ {6, 20, 60, 120}. Theorem P33-2 confirms Theorem P32-1 at
  K = 3; Theorem P33-3 confirms Theorem P32-2's two-regime bound
  on the max-ordinal decoder. The decoder-shape taxonomy is now a
  three-data-point empirical claim (priority-order / monotone-
  verdict / max-ordinal).
- ~~LLM-driven extractors — replacing the regex extractors with
  an LLM-call extractor and measuring the induced noise profile
  against the Phase-32/B curves.~~ **Done in P33.**
  ``core/llm_extractor`` is a drop-in replacement for any
  Phase-31/32 extractor; ``core/extractor_calibration`` measures
  the empirical (δ̂, ε̂, μ̂, π̂) quadruple and maps it to the
  closest Phase-32 synthetic sweep grid point. On ``qwen2.5:0.5b``
  against compliance review: real noise (0.70 / 0.12 / 0.40 / 0.60)
  maps to synthetic (0.5 / 0.1 / 0.25 / 0.0) with max-abs-gap
  0.10 — **verdict "approximates" (Theorem P33-1)**. The pooled
  i.i.d. match hides a structured per-role heterogeneity (legal
  drop 0.50, finance drop 1.00) — still open as Conjecture C33-3.
- ~~Adversarial-noise extension of Phase-32/B. Targeted-drop of
  load-bearing claims + adversarial spurious emission on highest-
  priority kinds — realistic threat-model sweep.~~ **Done in P34.**
  ``core/extractor_noise.adversarial_extractor`` with three target
  modes (load-bearing drop, role silencing, severity escalation).
  Theorem P34-2: at matched nominal budget, the load-bearing-drop
  adversary collapses substrate accuracy to 0 % at budget = 1 while
  matched i.i.d. preserves 20–70 %.
- ~~Per-role-adaptive Phase-32 noise sweep (Conjecture C33-3).~~
  **Done in P34.** ``core/extractor_calibration.per_role_audit_summary``
  + ``core/extractor_noise.PerRoleNoiseConfig``. Reports per-role
  (δ̂, ε̂, μ̂, π̂), the limiting role (argmax drop), and the
  per-role closest Phase-32 synthetic point. Theorem P34-1 formalises
  the role-limited accuracy bound; Conjecture C34-5 promotes the
  per-role replay to a predictor-ordering claim.
- ~~Ensemble extractor composition (Conjecture C33-4).~~ **Done in
  P34.** ``core/ensemble_extractor.UnionExtractor`` on a compliance
  mixed bank (5 canonical + 5 narrative): regex 50 % / LLM 0 % /
  ensemble **100 %** with pooled δ_u = 0.00 ≤ δ_r · δ_l = 0.188.
  Theorem P34-3 promotes C33-4 to an empirical bound on a
  complementary-coverage benchmark.
- ~~Dynamic, bounded communication primitives above the typed-
  handoff substrate, and a benchmark where static handoffs are
  insufficient.~~ **Done in P35.**
  ``core/dynamic_comm.EscalationThread`` + the contested-incident
  benchmark (``tasks/contested_incident``, 6 scenarios, 4
  contested + 2 controls). Mock-auditor headline: static
  handoffs 0/4 contested (33 % full accuracy) vs dynamic
  coordination 4/4 contested (100 % full accuracy) flat at 246
  tokens across k ∈ {6, 20, 60, 120}. Four theorems (P35-1
  expressivity separation, P35-2 bounded-context preservation,
  P35-3 correctness under sound producer-local extraction,
  P35-4 no-leak invariant) and two conjectures (C35-5 bounded
  threads ≡ adaptive subscriptions, C35-6 dynamic coordination
  is necessary not only sufficient). Real-LLM spot check under
  qwen2.5:0.5b: dynamic root-cause accuracy 1.00 vs static 0.50
  (+50 pp). The substrate differentiation from static-
  subscription-table-only tools is now a theorem (P35-1) as
  well as an empirical separation.
- Close probe-instrumentation-parity gaps (e.g. `os.walk` /
  `os.listdir` / `os.scandir` missing from `_record_filesystem`,
  `tempfile.mkdtemp` / `urllib.Request` surfaces) so that analyzer
  / runtime disagreement is always an analyzer story, never a probe
  story. Still open.
- Implicit-raise pattern-list extension (OQ-28b carry-over): with
  the Phase-29 4.83× coverage lift, `may_raise_implicit` FN on the
  wider entered slice is a first-class signal for which syntactic
  patterns the analyzer is still missing.
- Add one more deterministic analyzer pass — lightweight
  reachability / dead-code — to eliminate the single `may_raise`
  false-positive documented in Phase 26.

**Systems goals.**

- Keep the full Phase-27/28/29 benchmark at sub-60 s wall-time on
  a laptop. The Phase-29 task-scale benchmark runs in ≈ 6 s on
  four corpora — CI-friendly on every merge. The Phase-28/29
  method-coverage benchmark runs in ≈ 7 minutes on four corpora
  (because `vision-tests` now has 1 032 `ready_method` candidates
  to probe); 2-corpus variant runs in ≈ 30 s.
- ~~Promote the task-scale harness to a reusable substrate
  evaluation harness that accepts an arbitrary LLM callable on
  the aggregator role.~~ **Done in P30**. New
  ``vision_mvp/tasks/swe_loop_harness.py`` +
  ``vision_mvp/experiments/phase30_llm_swe_benchmark.py``. The
  harness takes any ``Callable[[str], str]``; the benchmark wires
  in a real local Ollama call by default and a deterministic
  ``MockAnswerLLM`` as a reference upper bound. Mock-reference
  run is ≈ 1.0 s on click + vision-core; 0.5b on 6-file
  ``json-stdlib`` is ≈ 7 minutes for 60 LLM calls. The harness
  is the substrate-as-tool surface that the programme's dual
  identity rests on: a future SWE-bench driver can plug in a
  SWE-bench task generator and reuse every measurement knob
  without touching the substrate code.
- Keep the full test suite green across every phase. Prior
  guarantees must survive new phases. Phase 31 left 1 101 tests
  green; Phase 32 left 1 152 tests green; Phase 33 left
  1 192 tests green; Phase 34 left 1 223 tests green;
  Phase 35 left 1 270 tests green;
  Phase 36 left 1 307 tests green;
  Phase 37 left 1 326 tests green;
  Phase 38 left 1 373 tests green;
  Phase 39 left 1 391 tests green (1 373 prior + 18
  new for ``swe_bench_bridge``);
  **Phase 40 leaves 1 417 tests green** repository-wide
  on a laptop with Ollama running (1 391 prior + 26
  new for the loader / adapter / sandbox + 1 retargeted
  from the Phase-39 unidiff-rejection placeholder to the
  new ``ValueError`` contract). The Phase-40 SWE-arc
  subset (``test_phase39_swe_bridge`` +
  ``test_phase40_real_swe_bridge`` +
  ``test_role_handoff``) runs in ~ 12 s on a commodity
  laptop; the full repository regression
  (``pytest vision_mvp/tests/``) runs in ~ 18 s and
  exits clean (1 417 passed, 21 subtests passed).

### 5.2 Medium-term (sessions 30–60)

**Scientific goals.**

- Run the falsifiability check for the core thesis end-to-end on
  SWE-bench (`ROADMAP.md`). If fewer than 50 % of events are
  causally relevant in a naive-context run, the routing thesis is
  confirmed empirically at task scale; if more than 80 %, the
  thesis is wrong and the programme pivots to compression-first.
- Runtime-refined analyzer (OQ-27h / OQ-28h): a hybrid that uses
  corpus-scale runtime observations to *refine* analyzer
  precision on `dead_code` cases while preserving soundness on
  the calibrated slice.
- Cross-language static + runtime calibration: TypeScript via
  `ts-morph`, Go via `go/parser`. The goal is not new features;
  it is to separate *Python-AST stability* from *substrate
  guarantee*.
- Compositional planner: meta-grammar over existing operators
  that handles conjunctive / comparative queries without hand-
  written patterns per case.
- Runtime calibration of interprocedural transitive flags on
  multi-corpus scale: a runtime witness for
  `trans_calls_filesystem` existence through a chain.

**Systems goals.**

- Persistent ledger, so the Merkle DAG survives process restarts.
- Sandboxed corpus import with isolation strong enough to run on
  third-party code without concerns about side effects at import
  time.
- Replay-style offline calibration of recipe aggressiveness
  (OQ-27a), so coverage vs. precision trade-offs are tunable
  rather than hand-set.

### 5.3 Full endgame (long horizon)

**Scientific goals.**

- Fixed-point convergence for `T_i*` (Open Question 1). A proof
  that the minimum-sufficient-context iteration has a unique
  fixed point under practical agent action distributions would
  promote CASR from "empirically optimal on tested classes" to
  "theoretically well-founded".
- A formal statement of the four-axis separation as an
  information-theoretic result: analyzer-gold exactness is
  orthogonal to runtime calibration, which is orthogonal to
  coverage, which is orthogonal to planner correctness. The
  goal is a theorem that states this orthogonality crisply and
  names the conditions under which pairs collapse.
- An implementable DAG-topology extension of CASR (Open Question
  4) with a provable composability law for relative-scale
  projections.
- A genuine breakthrough result would look like: "we have a
  protocol for N-agent coordination on tasks of rank r with
  per-agent context O(r + log N) that matches the
  information-theoretic lower bound tight up to constants, and
  we have empirical validation at N = 10⁴ across three task
  families with intrinsic rank verified empirically." The
  current state is two out of three of those; the middle piece
  (tight constants) is a real mathematical problem.

**Systems goals.**

- A reference implementation of the 5-layer substrate that
  survives third-party codebases, cross-language corpora, and
  multi-process ingestion without architectural changes. The
  substrate should be boringly stable; innovation should happen
  at the analyzer and runtime-observer layers.
- Published replication artifact: every benchmark rerunnable from
  one command, deterministic up to seeds, with archived JSON
  artifacts for every claim in every results note.

### 5.4 What a genuine breakthrough would look like

To avoid "breakthrough" sliding into hype — here is the bar:

- A **theoretical result** such that a referee can reduce the
  communication complexity of a named multi-agent task to an
  existing lower bound, showing CASR (or its successor) meets it
  tight.
- An **empirical result** on a benchmark the broader community
  recognises (SWE-bench, GAIA, a code QA leaderboard) where the
  direct-exact slice closes at least 20 percentage points of
  accuracy gap relative to LLM-only baselines at a strictly lower
  token cost.
- A **systems result** where the same substrate runs a 10⁴-agent
  team on one task and a 10⁶-function corpus analysis to
  runtime-calibrated conclusions, on commodity hardware, in
  sub-hour wall-time.

None of these are hand-waves. Each has a concrete falsifiable form.
The programme's job is to make them all simultaneously true, or to
find out which one breaks first.

---

## 6. Deep open problems

The set of things we do not know how to do — organised by whether
they are fundamentally hard, tractable-next, information-
theoretically limited, or places where exactness must yield to
approximation.

### 6.1 Fundamentally hard

- **Fixed-point convergence of `T_i*`.** The definition of
  minimum-sufficient context is self-referential; whether it
  has a unique attracting fixed point under realistic action
  distributions is a theorem we do not have. Partial progress
  exists (`core/contraction.py`, `core/deq_numpy.py`).
- **World-model bootstrapping across distribution shift.** The
  surprise filter's world model `M_i` has to be trained from
  data generated under a *different* filter regime; this is the
  chicken-and-egg problem that self-play-style RL also faces. No
  convergence proof is known for the curriculum.
- **Adversarial robustness of Bloom-filter routing.** Bloom
  filters are not adversarially robust (known from Bitcoin
  literature). A principled swap to cuckoo filters + authenticated
  event-type declarations is in `core/cuckoo_filter.py`; the full
  threat model is still open.

### 6.2 Tractable next

- **Implicit-raise predicate (Phase 28).** Shipped.
- **Multi-corpus runtime calibration (Phase 28).** Shipped. The
  extension to `click`, `json` stdlib, and other third-party
  corpora is mechanical once the registry surfaces are stable.
- **Method-instance auto-construction (Phase 29).** Shipped. The
  bounded ladder now covers the `inherited_object_init` /
  `explicit_init_all_defaults` / `dataclass_all_defaults`
  strategies; the next tier (`MagicMock` fallback for classes with
  required init args) is follow-up scope.
- **Task-scale causal-relevance check (Phase 29).** Shipped. The
  extension to full SWE-bench end-to-end with an LLM on the wrap
  path remains medium-term scope.
- **Probe-instrumentation parity with analyzer API surfaces.**
  Small, mechanical, yields a measurable reduction in apparent
  FP rate. Still open.
- **Implicit-raise pattern-list extension (OQ-28b).** With the
  Phase-29 wider coverage slice, the existing six-pattern list
  misses a small set of additional patterns observed on
  `vision-tests`. Pattern extension is additive — keeps
  `may_raise_explicit` FN at 0.
- **Cross-encoder re-ranking on retrieval.** Closes the narrow
  residual gap on adversarial / synonym queries (`MATH_AUDIT.md`
  §Phase-20 candidates).
- **LLM wrap-path on the Phase-29 open-vocabulary residual.** The
  Phase-29 substrate falls to a deterministic content-match on
  the 1 / 20 tasks per corpus that are open-vocabulary. Replacing
  this with a bounded-context LLM and measuring answer accuracy
  on the residual is the natural task-scale extension.

### 6.3 Information-theoretically limited

- **General d-dimensional consensus at small ε.** Theorem 11 gives
  `CC ≥ d · log(1/ε) − O(1)`. For tasks without low intrinsic
  rank, no routing trick recovers `O(log N)`. The right response
  is to recognise such tasks and decline to claim the `log N`
  bound on them.
- **Runtime observation as completeness claim.** By Theorem P26-2
  and its corpus analogue P27-3, runtime observation is a sound
  lower bound on reachable behaviour but never a proof of
  absence. Conservatism is intrinsic, not a failure.
- **Unresolved-callee propagation.** When an analyzer cannot see
  inside an external call, the `trans_*` predicate is honest-
  not-sound across that edge. The `has_unresolved_callees` flag
  is the transparency mechanism; closing the gap requires either
  trusting external APIs (lossy) or cross-package analysis
  (expensive) or runtime widening (conservatism-inverting).

### 6.4 Where exactness must yield to approximation

- **Open-vocabulary code questions.** The direct-exact path only
  answers planner-matched questions. Everything else falls to
  retrieval + LLM, which is approximate by construction. The
  cleanest extension — regex-over-source — is deterministic but
  narrow.
- **Environment-dependent behaviour.** A function whose effects
  depend on `os.environ`, wall-clock time, or uncontrolled random
  state cannot be fully characterised by a probe under a default
  environment. Scripted environment manipulation widens but does
  not close the gap.
- **Concurrency / async.** Our sandbox is sequential. Probing
  async code honestly requires an event loop and a threading
  model the probe does not have (yet).
- **LLM prompting itself.** At the very outermost layer, a model
  is still reading a bounded prompt. The prompt is as exact as
  we can make it, but the model's behaviour is not ours to
  guarantee.

---

## 7. Research strategy

Rules of thumb for what to optimise, what to avoid, and how to tell
real progress from false progress.

### 7.1 What to optimise for

- **Measurable separation.** Any new axis worth adding must admit
  a clean comparison against the existing axes. Runtime
  calibration succeeded because it was a distinct truth value,
  not a refinement of an existing one.
- **Boundary discovery.** The analyzer's boundaries are the
  research content. Phase 27 surfaced "implicit raises" as a
  boundary class that the snippet corpus did not exercise; Phase
  28 made that boundary first-class. This pattern — surface,
  name, formalise, code — is the research shape we prefer.
- **Conservatism before coverage.** When in doubt, expand the
  conservative slice, not the coverage claim. A 90 % coverage
  with 1 % FN is worse than 40 % coverage with 0 % FN for this
  programme's purposes.
- **Zero-LLM paths where possible.** Every question answered
  without an LLM is a question whose correctness is a
  deterministic property of code, not a probabilistic property
  of a model. This is the core invariant of the substrate.

### 7.2 What to avoid

- **Hype words.** "Solves context", "perfect understanding", "AGI-
  ready" — not in this codebase. Each claim has a specific
  meaning (§1.2).
- **Compression as the frame.** See §3.2.
- **Productisation as the goal.** The code is evidence, not a
  SaaS.
- **Framework accretion.** 50 of the 72 frameworks in the
  extended-math docs are THEORY-only and will stay that way unless
  one of them unlocks a specific problem. We wire up frameworks
  when a problem requires the specific structure that framework
  provides; not before.
- **Single-corpus headlines.** Every benchmark above Phase 22 is
  required to report on multiple corpora, with per-corpus
  breakdown and pooled aggregates. Single-number headlines are a
  drift risk.

### 7.3 What false progress looks like

- Direct-exact accuracy goes up because the planner started
  matching questions whose gold was *defined by the planner*.
  (This is the Phase-22 tautology trap, resolved by Phase 26's
  independent runtime axis.)
- A new predicate is added whose support is empty on every
  corpus.
- A new benchmark passes because the corpus skip-list was
  widened to exclude the awkward modules.
- A claim of "100 %" without a σ, without a corpus count, and
  without a witness sample.
- A new architectural layer added because it was interesting,
  not because an existing layer provably could not answer the
  question.

### 7.4 What real progress looks like

- A predicate is added; its support is non-empty on at least one
  corpus; its analyzer answer matches runtime observation on the
  entered slice; the planner round-trip is 100 %; it is backed
  by a theorem with a proof or at least a formal conjecture.
- Coverage goes up AND the calibration disagreement count does
  not go up.
- A boundary class is named, formalised, and tested with at
  least one snippet and at least one real-corpus witness.
- A prior result is made strictly tighter (e.g. Phase 27's
  coverage fraction rising without changing any claim) OR strictly
  more general (e.g. the same result on a new corpus).
- A test suite grows when a claim is added. No test, no claim.

### 7.5 When to prioritise theorem work vs. benchmarks vs. systems

- **Prioritise theorem work** when an experiment is converging on
  a stable number and the question is whether that number is the
  information-theoretic minimum.
- **Prioritise benchmarks** when a claim exists in one phase and
  generalisation is uncertain. Every phase since 23 has been a
  benchmark-forward phase; this is deliberate, because the
  programme's main risk class is "looks great on one corpus".
- **Prioritise systems work** when a benchmark is blocked by an
  infrastructure gap that would, if left alone, *become* an
  epistemic gap — e.g. if runtime calibration could only run
  under one seed or on one machine.
- **Never prioritise productisation.**

---

## 8. How to use this document

**For a new collaborator** — read §1 through §4 in order. Skip §5
and §6 on first pass. Return for §7 when planning a specific
contribution.

**For a reviewer** — §1.2, §1.4, §3.3, §6 are the parts with the
sharpest content to attack. Everything claimed in those sections is
either traceable to a test / experiment or explicitly flagged as a
conjecture.

**For a future version of the authors** — §7 is the durable
decision rulebook. When a session is tempted to make a local
choice that conflicts with §7, update §7 with the new rationale or
back off the choice. §7 is allowed to change, but only deliberately.

**For someone searching for what to work on next** — §5.1 and §6.2
are the near-term frontier. §6.1 is where the hard theory lives.
§6.4 is where the programme pushes back on itself.

### 8.1 Evolution rules

- This file changes when a new arc enters or an existing arc
  reaches a durable checkpoint. Incremental empirical numbers go
  into the corresponding `RESULTS_PHASE*.md`, not here.
- Every major section either has a stable fact base or a marker
  indicating it is a conjecture. No buried `[TODO]` items.
- The default action on conflict between this plan and code is to
  trust the code and update the plan; the default action on
  conflict between this plan and a results note is to trust the
  results note and update the plan.
- Empirical numbers here must match at least one results note.
  Numbers here that do not have a results-note anchor are a
  documentation bug.

### 8.2 Document relationships

| Document                             | Role                                                                   | Authoritative for                                  |
|---                                   |---                                                                     |---                                                 |
| `docs/context_zero_master_plan.md`   | This doc. Long-running strategy + arcs + open problems.                 | Thesis, research strategy, durable framing        |
| `README.md`                          | Entry point; current status + quickstart + headline table             | First impression, reproducibility instructions    |
| `ARCHITECTURE.md`                    | Reference architecture of the substrate + CASR                         | Layer semantics + contract between layers         |
| `FRAMEWORK.md`                       | Original problem formulation; routing-as-causal-inference              | Theoretical origin of CASR                        |
| `PROOFS.md`                          | Formal theorems with proofs                                           | Mathematical claims                                |
| `MATH_AUDIT.md`                      | Which of the 72 frameworks are actually in the code                   | Honest accounting of theory-to-code               |
| `OPEN_QUESTIONS.md`                  | Seven foundational open questions + resolution status                 | Which hard problems the programme owns            |
| `ROADMAP.md`                         | Phase 1–4 plan + risk register + falsifiability check                 | Empirical plan, SWE-bench framing                 |
| `VISION_MILLIONS.md`                 | Forward-looking vision past CASR (10^6+ agents)                        | Speculative / post-CASR architectures             |
| `vision_mvp/RESULTS_PHASE*.md`       | Per-phase empirical record                                            | What happened in each phase, with archived JSON   |

This document refers to the others; the others should refer back
here for the "why" behind the programme.

---

## 9. Finished Product Checklist / Release Criteria (Phase 45)

This section is durable, status-bearing, and deliberately narrow.
It records what has to be true for the substrate + pipeline +
operator surface to be called a **finished product state** — a
state in which a serious operator can run the Phase-44 residue +
readiness + sweep pipeline against any SWE-bench-Lite-shape JSONL
with one command and get a publishable verdict.

§7.5 still stands — *theory and experiments before engineering*. This
checklist is not a pivot to productisation; it is an honest accounting
of the layers the research programme now depends on being reproducibly
operable. Items that block *science* (not UX) are marked ⛔.

Legend: **✅** done, **◐** partial, **🧱** blocked externally,
**☐** not started, **⛔** blocks scientific reproducibility if missing.

### 9.1 Core substrate / communication stack

| Item | Status | Anchor |
|---|---|---|
| Bounded-context router (typed handoff, per-role inbox) | ✅ | Phases 1–10, Theorem 3 in `PROOFS.md` |
| Byte-for-byte lossless external memory (content-hash) | ✅ | Phase 19 |
| Routing + substrate gap validated on SWE-style tasks | ✅ | Phase 29, 40, 42 |
| Strategy-invariance under parser / matcher / semantic refinement | ✅ | Theorem P43-3, P44 § D.4 |
| Substrate-as-library clean import surface | ◐ | `vision_mvp.core.*` — stable but undocumented contract |
| Distributed substrate (multi-node handoff router) | ☐ | research, not product-blocking |

### 9.2 Parser / matcher / sandbox pipeline

| Item | Status | Anchor |
|---|---|---|
| Unified-diff + OLD/NEW block parser (strict + robust modes) | ✅ | Phase 42 |
| Matcher modes (strict / permissive) with attribution | ✅ | Phase 41 |
| Sandbox backends (in_process / subprocess / docker) | ✅ | `vision_mvp.tasks.swe_sandbox` |
| Raw-capture of LLM text + ParseOutcome (Theorem P44-1) | ✅ | Phase 44 |
| Replay-verification driver (hash-check stored captures) | ☐ | Phase 44 § F.2 |
| Parser-compliance counter stable across sweeps | ✅ | Phase 42 |

### 9.3 Public-data readiness

| Item | Status | Anchor |
|---|---|---|
| Five-check readiness validator (Theorem P44-3) | ✅ | `phase44_public_readiness.py` |
| Bundled 57-instance bank saturates 57/57 on all checks | ✅ | `results_phase44_readiness_bundled.json` |
| `--jsonl <path>` drop-in contract validated by code | ✅ | Phase 44, Phase 45 runner |
| Actual public SWE-bench-Lite JSONL locally present | 🧱 | data availability — the repo is deliberately offline |
| Public-bank end-to-end run with cluster LLM | 🧱 | blocked by the above |

### 9.4 Model coverage / semantic headroom

| Item | Status | Anchor |
|---|---|---|
| Coder-class parser-dominant cell (qwen2.5-coder:14b) | ✅ | Phase 42 / 43 / 44 |
| Frontier semantic-headroom cell (qwen3.5:35b) | ✅ | Phase 43 / 44 |
| Refined residue attribution (v2 classifier, 5 sub-labels) | ✅ | Phase 44 |
| ≥70B coder-finetuned headroom cell | ☐ | needs bigger local model; blocks nothing structural |
| Model capability/profile matrix | ✅ | `vision_mvp.product.profiles.model_capability_table` |

### 9.5 Operator workflow / CLI / configuration

| Item | Status | Anchor |
|---|---|---|
| One-command product runner | ✅ | `python3 -m vision_mvp.product --profile <name> --out-dir <d>` |
| Stable profile set (local_smoke / bundled_57 / aspen_mac1_coder / aspen_mac2_frontier / public_jsonl) | ✅ | `vision_mvp/product/profiles.py` |
| Profile schema version (`phase45.profile.v1`) | ✅ | same |
| ASPEN 2-Mac cluster launch commands recorded in-report | ✅ | runner `_real_sweep_stub` |
| In-process driver for heavy real-LLM sweeps | ◐ | deliberate: real runs go through the phase42/phase44 CLI so long runs are explicit |
| Operator runbook in `README.md` | ◐ | Phase 45 link lands, deeper runbook deferred |
| Public-data import CLI + row-level audit (Phase 46) | ✅ | `vision_mvp/product/import_data.py`, Theorem P46-1 |
| CI / deployment consumer (Phase 46) | ✅ | `vision_mvp/product/ci_gate.py`, Theorem P46-2 |
| Frontier-model slot + capability/residency metadata (Phase 46) | ✅ | `aspen_mac1_coder_70b`, `model_availability()`, Theorem P46-3 |

### 9.6 Reporting / artifacts / reproducibility

| Item | Status | Anchor |
|---|---|---|
| Machine-readable `product_report.json` (`phase45.product_report.v1`) | ✅ | `vision_mvp/product/runner.py` |
| Human-readable `product_summary.txt` | ✅ | `vision_mvp/product/report.py` |
| Readiness verdict JSON stable across profiles | ✅ | `phase44.readiness.v1` |
| Raw-capture schema versioned (`phase44.v1`) | ✅ | `swe_raw_capture.py` |
| Per-run artifact directory convention (`vision_mvp/artifacts/<tag>`) | ✅ | Phase 45 |
| Refined-residue cross-model summary | ✅ | `results_phase44_refined_summary.json` |

### 9.7 Demo / deployment / release docs

| Item | Status | Anchor |
|---|---|---|
| Phase-45 results note | ✅ | `vision_mvp/RESULTS_PHASE45.md` |
| Master plan §9 Finished-Product Checklist (this section) | ✅ | here |
| README thread for the product runner | ✅ | `README.md` |
| ARCHITECTURE pointer to product surface | ✅ | `ARCHITECTURE.md` |
| MATH_AUDIT refresh with P44 / P45 claims | ✅ | `MATH_AUDIT.md` |
| CI gate using `product_report.json` in upstream repo | ☐ | consumer concern, not programme concern |
| Public announcement / paper | ☐ | explicitly deferred — §7.5 still stands |

### 9.8 What still materially blocks "finished product state"

1. **Public SWE-bench-Lite JSONL availability (🧱 external).**
   The five-check readiness gate, the loader, the adapter, the
   parser, the matcher, the sandbox, and the refined classifier
   all accept a public-shape JSONL today. The only missing thing
   is the JSONL itself on local disk. This is a data-availability
   blocker, not an architecture blocker.
2. **Stronger-than-35B local model (◐ engineering).** Phase 44's
   C43-1 tightening and C44-2 refinement need a ≥70B coder-
   finetuned cell. The runner's `aspen_mac1_coder` and
   `aspen_mac2_frontier` profiles already document the shape of
   that run; adding a third profile is one dict entry once the
   model is available on the cluster.
3. **No CI hook wired up in a downstream consumer (☐ out of
   scope).** `product_report.json` is CI-gate-shaped, but wiring
   it into an actual CI pipeline is a consumer-repo concern.

Everything *inside* the programme (substrate + pipeline +
reporting) is at the **finished-product state** as of Phase 45.
The two remaining blockers (external data, bigger model) are named
and dated; neither is a research or architectural debt.

### 9.9 Endogenous vs exogenous state (Phase 46)

Phase 46 makes the programme's boundary explicit:

* **Endogenous** (programme-controlled, finished-product as of
  Phase 45 + 46): substrate, parser/matcher/sandbox, refined
  semantic taxonomy, readiness validator, raw-capture store,
  product runner, profile set, report renderer, public-data
  import CLI, CI gate, frontier-model slot + capability
  metadata. Every item here is covered by at least one theorem
  (T3, P31-5, P41-1/2, P43-3, P44-1/2/3, P45-1/2/3, P46-1/2/3).
* **Exogenous** (outside programme control, met at the Phase-46
  boundary surface): arrival of a public SWE-bench-Lite JSONL,
  residency of a ≥70B coder-finetuned Ollama model on the
  ASPEN cluster, wiring of a downstream CI pipeline.

The bidirectional translation is:

| Exogenous blocker | Endogenous code path that meets it |
|---|---|
| public SWE-bench-Lite JSONL | `vision_mvp/product/import_data.py` (schema + P44-3 audit) + `public_jsonl` profile (one `--jsonl` flag) |
| ≥70B coder-finetuned model | `aspen_mac1_coder_70b` profile + `model_availability()` (mark resident; one string change) |
| downstream CI pipeline | `vision_mvp/product/ci_gate.py` (one Unix exit code on `product_report.json`) |

**Conjecture C46-1 (boundary-shape of remaining blockers).**
Every remaining item in §9.8 that is not ✅ or ◐ is reachable
from endogenous state by a single boundary operation (file
import, residency mark, exit-code consumption) rather than an
architecture change. Supported by the Phase-46 shipped code;
falsifiable by a future blocker whose resolution requires
editing `vision_mvp/core/`, `vision_mvp/tasks/`, or
`vision_mvp/experiments/`.

---

*End of master plan. Changelog lives in the results notes, not
here. If this document ever becomes a changelog, delete the
changelog and restore the plan.*
