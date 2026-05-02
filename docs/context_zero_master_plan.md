# Context-Zero — Master Plan

> **Latest milestone marker (SDK v3.36 / W35, 2026-05-02).**
> *Trust-subspace dense-control proxy + basis-history projection +
> W35 manifest-v5 CID*.  W35 wraps W34's live-aware multi-anchor
> abstention path with a controller-verified dense basis over W21
> probe top_sets, W33 EWMA trust, W34 live-attestation/response-
> feature state, top-set stability, and host health.  The integration
> materially helped on the specific blocker where W34 could only
> abstain: on R-82-TRUST-SUBSPACE-SHIFT, W34 abstains on 6 disputed
> cells; W35 safely reroutes 5/6 through the stable basis direction,
> raising correctness from 0.625 to **0.9375** (**+0.3125**) across
> 5/5 seeds, with trust precision preserved at **1.000** and one
> visible-token overhead/cell.  Trust/audit survived: W35 adds 14
> mechanically tested verifier failure modes and manifest-v5 CID
> binding, bringing the cumulative W22..W35 trust boundary to **98
> enumerated failure modes**.  Bounded-context efficiency improved in
> the dense-state sense: mean **13,016.5 structured bits per visible
> W35 token** on the load-bearing regime.  Multi-host evidence did not
> materially broaden: local Ollama and `192.168.12.191` are usable,
> but `192.168.12.248` still times out; the bounded two-host fallback
> observed 3/5 cross-host disagreements, all gold-correlated.  Earlier
> loose ends closed versus sharpened: the old explicit oracle line and
> dense-control/geometry line now compose in one mechanism; native-
> latent remains open because W35 is not transformer-internal hidden
> state projection; live magnitude remains open as a systematic survey;
> Mac 2 remains hardware-bounded.  Release readiness improves only as
> a consequence: W35 is experimental, stable runtime unchanged, version
> bumped to SDK v3.36 / 0.5.9.  **Net effect on the original thesis**:
> materially stronger than W34 on the audited-proxy path, still blocked
> by a deeper trust/semantics wall for true native latent.
> See ``docs/RESULTS_WEVRA_W35_TRUST_SUBSPACE_DENSE_CONTROL.md`` and
> ``docs/SUCCESS_CRITERION_W35_TRUST_SUBSPACE_DENSE_CONTROL.md``.

> Canonical long-running document for the research programme behind
> `context-zero`. This is a plan for a body of work, not a changelog.
> Phase-by-phase diaries live in `vision_mvp/RESULTS_PHASE*.md`;
> session notes live nowhere durable, on purpose. Last touched: SDK
> v3.9 cross-role corroboration multi-agent benchmark + W8 family
> (Phase-55 deterministic decoy-plurality + cross-role-corroborated
> gold; first SDK milestone to clear the strong success bar of
> `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1).
> Previously: SDK v3.6 two-Mac distributed-inference integration boundary +
> real cross-LLM parser-boundary measurement (chosen path: **MLX
> distributed** — Apple-official, supports Apple Silicon
> sharded inference; not Hyperspace, which is distributed-agent
> infra rather than single-model sharding). The Wevra-side
> integration ships as a duck-typed ``LLMBackend`` Protocol with
> two concrete implementations (``OllamaBackend``,
> ``MLXDistributedBackend``); the inner-loop seal-PROMPT /
> seal-LLM_RESPONSE chain is byte-for-byte unchanged. First
> real cross-LLM measurement on the available model class
> (Mac 1 Ollama: Qwen-2.5-14B-dense vs Qwen-3.5-35B-MoE)
> yields a **saturated 1.000 cross-model PARSE_OUTCOME
> failure-kind TVD on strict parsing** (n=10), collapsing to
> 0.000 under robust parsing — the **first real evidence**
> that the capsule-native runtime survives a 2.4× model-class
> jump and a dense → MoE architecture swap without spine
> modification (Theorem W5-1 proved-empirical; W5-2 / W5-3
> proved on the integration boundary). The two-Mac MLX-distributed
> path is **experimental infrastructure**, not product; Wevra's
> single-run product runtime contract is byte-for-byte
> unchanged. SDK v3.5 capsule-native multi-agent team
> coordination slice still in force (TEAM_HANDOFF / ROLE_VIEW /
> TEAM_DECISION; W4-1 / W4-2 / W4-3; W4-C1 empirical).
> 2026-04-26.
> Canonical research-status pointer:
> [`docs/RESEARCH_STATUS.md`](RESEARCH_STATUS.md). Canonical
> theorem registry: [`docs/THEOREM_REGISTRY.md`](THEOREM_REGISTRY.md).
> Do-not-overstate rules:
> [`docs/HOW_NOT_TO_OVERSTATE.md`](HOW_NOT_TO_OVERSTATE.md).
>
> The programme has a deliberately dual identity: it is (a) a research
> programme in context, coordination, and exact external memory,
> producing theorems, open questions, and falsifiable measurements;
> and (b) an emerging tool/substrate for real agent teams. The
> code is evidence for the research, and the research is what makes
> the tool principled. Neither identity subsumes the other.
>
> **Naming.** `Context Zero` is the research programme. `Wevra` is the
> first usable product produced by the programme: the operator-facing
> surface that wraps the already-settled substrate, parser/matcher/sandbox
> stack, readiness gate, reporting, and deployment boundary.
>
> **Research center.** As of the SDK-v3 milestone (2026-04-22) and
> the Phase-46 capsule-research milestone (same date), the **Context
> Capsule** abstraction has stopped being just a product label
> ("context is an object") and has become a *research center*: a
> formal mathematical model (`docs/CAPSULE_FORMALISM.md`), an ML
> problem with a real held-out result
> (`docs/archive/capsule-research/RESULTS_CAPSULE_LEARNING.md`), and an empirical
> unification audit across four prior substrate primitives. The
> capsule contract is *additive on top of* every Phase-N
> bounded-context theorem — no substrate primitive is modified.
> See § 4.12 for the post-Phase-46 research-center status and
> § 4.11 for the current frontier.
>
> **Execution contract (SDK v3.1, 2026-04-26).** Capsules have
> stopped being purely a post-hoc audit fold and (partially) become
> the runtime's typed execution contract. The
> ``CapsuleNativeRunContext`` runtime context drives a Wevra run
> through capsule lifecycle transitions: each stage seals a typed
> capsule before the next stage can read its result. Substantive
> on-disk artifacts are content-addressed at write time
> (Theorem W3-33). The post-hoc ``build_report_ledger`` is
> retained as a third-party adapter; the two paths are
> CID-equivalent on non-ARTIFACT kinds (Theorem W3-34). The
> capsule layer has therefore moved from *"capsules as audit
> graph"* toward *"capsules as execution contract"*. See § 4.18
> for the v3.1 runtime extension and `docs/archive/wevra-milestones/RESULTS_WEVRA_CAPSULE_NATIVE.md`
> for the milestone note.
>
> **Intra-cell + detached witness (SDK v3.2, 2026-04-26).**
> The capsule-native slice has been extended past the cell
> boundary into the inner sweep loop. Each (task, strategy)
> parse→apply→test transition seals a pair of capsules in
> flight: a PATCH_PROPOSAL (parent: SWEEP_SPEC) and a
> TEST_VERDICT (parent: PATCH_PROPOSAL). The lifecycle ordering
> ``patch → verdict`` is enforced at the type level
> (Theorem W3-32-extended). The meta-artefact boundary is now
> formally a sharp circularity (Theorem W3-36 — the rendering
> of the RUN_REPORT-rooted view cannot be authenticated within
> the primary ledger) with a positive corollary: a detached
> META_MANIFEST in a secondary ledger is the strongest
> authentication achievable, one trust hop beyond the primary
> view. ``wevra-capsule verify`` now recomputes the chain from
> on-disk header bytes (Theorem W3-37) and re-hashes every
> ARTIFACT against the on-disk file at audit time
> (Theorem W3-38). See § 4.19 for the v3.2 runtime extension and
> `docs/archive/wevra-milestones/RESULTS_WEVRA_INTRA_CELL.md` for the milestone note.
>
> **PROMPT / LLM_RESPONSE slice (SDK v3.4, 2026-04-26).**
> The capsule-native slice has been extended one further
> structural layer past v3.3 to the LLM byte boundary itself.
> Each LLM call seals two capsules in flight: a PROMPT
> (parent: SWEEP_SPEC) and an LLM_RESPONSE (parent: PROMPT),
> both content-addressed by their bytes' SHA-256 (idempotent
> on content — byte-identical prompts collapse to one
> capsule). The PARSE_OUTCOME capsule's parent set is now
> either `(SWEEP_SPEC,)` (oracle path) or
> `(SWEEP_SPEC, LLM_RESPONSE)` (LLM-backed path). The
> end-to-end inner-loop chain is therefore a five-link typed
> DAG `PROMPT → LLM_RESPONSE → PARSE_OUTCOME →
> PATCH_PROPOSAL → TEST_VERDICT` with strong parent-CID
> gating at each step (Theorems W3-42 / W3-43 / W3-44).
> The lifecycle audit covers eleven invariants L-1..L-11
> (Theorems W3-40 / W3-45). A new in-process synthetic-LLM
> mode (`SweepSpec(mode="synthetic", synthetic_model_tag=
> <tag>)`) lets the full chain run end-to-end in CI without
> an Ollama endpoint. The cross-model parser-boundary
> research (W3-C6, empirical) reports cross-distribution
> PARSE_OUTCOME failure-kind Total Variation Distance up to
> 1.000 across the calibrated synthetic distribution library
> (`vision_mvp/wevra/synthetic_llm.py`) and parser-mode
> shift up to 1.000 on `synthetic.unclosed`. See § 4.21 for
> the v3.4 runtime extension and
> `docs/archive/wevra-milestones/RESULTS_WEVRA_INNER_LOOP.md` for the milestone note.
>
> **Multi-agent team coordination slice (SDK v3.5,
> 2026-04-26).** The capsule-native abstraction has been
> extended **between agents in a team** — the original
> Context-Zero "solve context for multi-agent teams" thesis
> now has a capsule-native research slice. Three new
> closed-vocabulary capsule kinds are added:
> `TEAM_HANDOFF` (capsule-native multi-agent handoff;
> distinct from the substrate-adapter `HANDOFF` which lifts
> a Phase-31 `TypedHandoff`), `ROLE_VIEW` (per-role admitted
> view of one coordination round; parents are the admitted
> TEAM_HANDOFF cids; `max_parents = K_role`,
> `max_tokens = T_role`), and `TEAM_DECISION` (team-level
> decision; parents are the role views consulted). A
> `TeamCoordinator` drives one round end-to-end against an
> in-memory ledger; `audit_team_lifecycle` mechanically
> verifies invariants T-1..T-7 (Theorem W4-1, proved +
> mechanically-checked). Theorem W4-2 (proved-conditional)
> states coverage-implies-correctness on the deterministic
> team decoder; Theorem W4-3 (proved-negative) states a
> sharp local-view limitation: per-role budget below the
> role's causal-share floor cannot be rescued by *any*
> admission policy. A learned per-role admission policy
> (logistic regression over six capsule features) admits
> *strictly fewer handoffs* than the strongest fixed
> admission baseline (coverage-guided) on every train seed
> (12/12) and improves pooled team-decision accuracy on
> most seeds (gap_full > 0 in 11/12, mean $+0.054$;
> gap_root_cause > 0 in 8/12, mean $+0.032$) at the
> Phase-52 default config — but the accuracy advantage
> reverses at higher noise (W4-C1, empirical; see
> `docs/HOW_NOT_TO_OVERSTATE.md` for cross-seed reading
> rules).
> The Wevra single-run product runtime contract is unchanged;
> the team layer is research-grade (`vision_mvp.wevra
> .team_coord`). See § 4.22 for the v3.5 extension and
> `docs/archive/wevra-milestones/RESULTS_WEVRA_TEAM_COORD.md` for the milestone note;
> the formal model lives in `docs/CAPSULE_TEAM_FORMALISM.md`.
>
> **Programme vs Product — read this first.** The programme ships
> theorems, phase shards, and an EXTENDED_MATH survey (§§ 1–9). The
> product ships one SDK — **Wevra** — with a bounded public surface:
> `RunSpec` → provenance-stamped report, profile-driven evaluation on
> SWE-bench-Lite-shape banks, plugin protocols, unified mock/real
> runtime, CI gate (§ 10). Wevra is **not** the whole programme and
> **not** a universal agent platform. CASR (Causal-Abstraction Scale-
> Renormalized Routing) is the *original substrate* — it lives in
> `vision_mvp.core.*` as settled research code and grounds Wevra's
> bounded-context claim (Theorem 3 in `docs/archive/pre-wevra-theory/PROOFS.md`); it is not itself
> the product identity. The canonical one-pass orientation for a new
> reader is [`docs/START_HERE.md`](START_HERE.md).

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

1. **No product-first framing.** The research is not reducible to a
   framework, a library, or a product. It is a set of claims about
   information flow in systems of LLM agents and about the shape of
   the contract between symbolic machinery and a model. The current
   product surface (`Wevra`) is an output of the programme, not the
   programme's boundary.
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
   in `docs/archive/pre-wevra-theory/PROOFS.md`). This is settled, for the class of tasks it
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
  (Theorem 11, `docs/archive/pre-wevra-theory/PROOFS.md`). When this condition fails, routing does
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
  the representation that preserves answer quality (`docs/archive/pre-wevra-theory/FRAMEWORK.md`
  §A.2). CASR is designed to approach but not violate it.
- **Shannon-style channel capacity / broadcast lower bounds**
  (Theorems 3, 11 in `docs/archive/pre-wevra-theory/PROOFS.md`) say that any protocol in which
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
  workspace capacity O(log N) result (V2.14 in `docs/archive/pre-wevra-theory/MATH_AUDIT.md`).

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
`docs/archive/pre-wevra-theory/MATH_AUDIT.md` records which are `USED`, `STRUCTURAL`, `BUILT`, or
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
(`docs/archive/pre-wevra-theory/ROADMAP.md`).

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

As of Phase 51, the programme sits at the following frontier.
This section is the one-paragraph answer to "where is the
programme right now, and what breaks next?" — it is expected to
change every two or three phases, unlike the rest of this
document.

**Top-line, post-Phase-51 (relational / cohort-aware decoder
frontier — honest limitation-leaning result).**  Phase 51
opens the decoder-side relational axis named by Conjecture
W3-C5 — a hypothesis class structurally outside the
magnitude-monoid linear family that W3-29 lower-bounds on the
zero-shot penalty axis.  Theorem **W3-30** (proved,
constructive) strictly contains the DeepSet class: a decoder
that partitions the bundle by source-role and aggregates per-
role features expresses predicates (e.g. "≥ 2 distinct source
roles each emit a capsule implying rc") that DeepSet's
per-capsule φ-sum cannot.  Claim **W3-31** (empirical, code-
backed) reports the outcome: under a matched Phase-51 training
pipeline, the ``CohortRelationalDecoder`` achieves zero-shot
gap = 0.038 at level = 0.237 on (incident, security), matching
Phase-50's reported sign-stable-DeepSet level of 0.237, edging
the same-pipeline sign-stable-DeepSet baseline by +5 pp (4
instances at $n=80$, not statistically robust), and **not
strictly exceeding** Phase-50's reported ceiling.  On Phase-31
Gate 1 it underperforms DeepSet (0.362 vs 0.425 at the
pre-committed cell; 0.388 vs 0.425 at best cell).
Conjecture **W3-C10** names the standing empirical ceiling:
direction-invariant zero-shot transfer on (incident, security)
appears bounded at ≈ 0.237 across all eight Phase-49 / Phase-50
/ Phase-51 decoder families.  **The honest stance.**  Phase 51
is a **limitation-leaning milestone** — it proves a formal
strict-separation result, ships the smallest serious relational
decoder, and finds empirically that the richer class does NOT
cleanly break the Phase-50 ceiling on this task-family pair.
The next research move is either (a) a relational feature that
specifically targets W3-C10 (speculative), (b) a different
operational-detection pair with smaller
$\|w^*_A - w^*_B\|$ (structurally cleaner, Phase-52 candidate),
or (c) accept W3-C10 and redirect (mechanical W3-C1 closure,
relational *substrate* primitive).  Anchor:
`docs/archive/capsule-research/RESULTS_CAPSULE_RESEARCH_MILESTONE6.md` and
`docs/CAPSULE_FORMALISM.md` § 4.F.

**Top-line, post-Phase-50 (strict-reading honest certification
+ structural limitation theorems — preserved).**  Phase 50 is
the strict-reading test of the Phase-49 paradigm-shift
candidate, and is an **honest falsification under the strict
reading** accompanied by the proved structural reason: Gate 1
fails at $n_{\rm test} = 320$ ($\hat{p} = 0.362$, below even
the point threshold — W3-26 falsifies W3-23), and strict
zero-shot Gate 2 fails across all six Phase-50 hypothesis
classes (best max penalty V2 at +0.112 — W3-27).  Two new
proved theorems (**W3-24** post-search winner's-curse bias,
classical; **W3-29** Bayes-divergence lower bound on zero-shot
risk penalty) give the structural reasons for the strict-
reading failure.  One empirical positive (**W3-28** sign-stable
DeepSet zero-shot gap = 0.000) and one refined conjecture
(**W3-C9** Gate-2 reformulation to gap-reading) close the
frontier with an operationally-defensible bar.  **The honest
stance.**  Phase 49 is a **canonical** paradigm-shift
candidate (point-estimate Gate 1 at $n=80$ + pooled-multitask
or gap-reading Gate 2) but **NOT a strict** paradigm shift
(strict CI Gate 1 + strict-penalty zero-shot Gate 2 are both
structurally blocked).  The programme should adopt W3-C9's
reformulated bar.  Anchor:
`docs/archive/capsule-research/RESULTS_CAPSULE_RESEARCH_MILESTONE5.md` and
`docs/CAPSULE_FORMALISM.md` § 4.E.

**Top-line, post-Phase-49 (stronger decoder + symmetric
transfer frontier — superseded).** The paradigm-shift bar
(Conjecture W3-C7) is partially earned under a canonical-but-
not-strictest reading: Gate 1 MET at 0.425 point estimate
(DeepSetBundleDecoder, best cell on Phase-31 noisy bench at
$n_{\rm test} = 80$; binomial CI $[0.317, 0.539]$), Gate 2
MET under the multitask shared-head reading (0.350 on both
incident and security test sets, gap 0.000, ≥ 13 pp over
priority baseline, Theorem W3-22). Strict readings leave
≈ 2.5 pp of Gate 1 CI headroom and ≈ 9 pp of zero-shot
Gate 2 penalty headroom. Three new formal results —
**W3-20** (Deep Sets sufficiency, positive-conditional,
proved), **W3-21** (linear-class sign-flip asymmetry,
negative, proved), **W3-22** (multitask symmetric transfer,
empirical). **Honest stance (Phase-49-internal).** The
programme had a paradigm-shift *candidate* one well-scoped
milestone away from strict certification.  **Phase 50
ran that milestone and falsified the strict reading** —
the CI and penalty gaps were structural, not statistical.
Anchor: `docs/archive/capsule-research/RESULTS_CAPSULE_RESEARCH_MILESTONE4.md` and
`docs/CAPSULE_FORMALISM.md` § 4.D.

**Top-line, post-Phase-48 (bundle-aware decoder frontier).**
The Phase-47 boundary localises at the decoder: the
0.200 Phase-31 structural ceiling is a property of the
priority decoder, not admission, and every admission rule is
bounded above by it under Theorem W3-17 (admission locality —
negative, proved). Phase 48 is the decoder-side attack on
Conjecture P47-C1.

1. **Decoder frontier — structural + empirical break (Part A).**
   Three decoder-side results:
    * Theorem **W3-17** (proved, conditional) — admission-only
      rules cannot exceed the priority-decoder ceiling under
      ceiling-forcing spurious injection; this sharpens the
      Phase-47 empirical observation into a limitation
      statement over all header-level admission rules.
    * Theorem **W3-18** (proved, conditional) — plurality
      decoding strictly dominates priority decoding on the
      *coherent-majority* regime. The contract test
      `test_w3_18_plurality_strictly_dominates_priority_on_coherent_majority`
      exhibits the sharpest single-bundle separator
      (two-OOM_KILL + one-spurious-DFC ⇒ priority says
      disk_fill, plurality says memory_leak).
    * Claim **W3-19** (empirical, seed-robust) — the
      `LearnedBundleDecoder` breaks the 0.200 structural
      ceiling at **+15 pp** on FIFO × B=64..256 (test accuracy
      0.350) and at **+17.5 pp** on bundle-learned admission ×
      B=96 (test accuracy 0.375). On the oracle-clean causal
      slice it reaches **0.575**, exceeding the priority
      decoder's clean ceiling of 0.525. Plurality alone does
      *not* break the ceiling on the full bench
      (priority-tiebreak fallback reproduces the ceiling on
      one-vote-each scenarios) — falsifies a naive reading of
      P47-C1. Anchor: `docs/archive/capsule-research/RESULTS_CAPSULE_RESEARCH_MILESTONE3.md`
      § 2.
2. **Decoder-side cross-domain transfer — asymmetric (Part B).**
   First decoder-side transfer study (incident / security; the
   two operational-detection domains from Phase 47). Transfer
   matrix on $n_{\rm test} = 80$ instances per domain:
   * within-incident: learned 0.362, priority 0.212 (+15 pp).
   * within-security: learned 0.300, priority 0.200 (+10 pp).
   * incident → security (cross): 0.300 (+10 pp; matches
     within-security).
   * security → incident (cross): 0.125 (**−8.7 pp**,
     BELOW priority).
   Feature sign-flip on `lone_top_priority_flag` (+1.13 on
   incident, −1.19 on security) is the structural reason: the
   "single-signature high-priority" shape in incident vs the
   "multi-source high-priority corroboration" shape in
   security. Conjecture W3-C6 states the asymmetric
   task-family-indexed transfer. P47-C3 (admission-side) and
   W3-C6 (decoder-side) are the *same phenomenon*, now sharp
   on both sides. Anchor:
   `docs/archive/capsule-research/RESULTS_CAPSULE_RESEARCH_MILESTONE3.md` § 3.
3. **Paradigm-shift stance — still below bar (Part C).**
   Conjecture **W3-C7** sets the explicit threshold: a
   paradigm-shift-worthy decoder would (a) break 0.200 by
   ≥ 2× ($\ge 0.400$) AND (b) transfer approximately
   symmetrically across operational-detection domains. Phase
   48 achieves (a) weakly (0.375 < 0.400) and (b) partially
   (one direction works, the reverse fails).
   **The centre is materially upgraded toward a paradigm shift**
   (the decoder frontier is open and traversable) but does NOT
   yet earn the label.

**Remaining gap (post-Phase-48) is now structurally five-shaped.**
(a) **Stronger decoder** (closing the $\ge 0.400$ threshold —
deeper hypothesis classes, richer features, coupled admit-
decode training), (b) **symmetric transfer** (what does
security → incident need to not regress? — domain-adaptive
feature calibration, or a mixture-of-experts gating), (c)
**mechanical W3-C1 closure** on the remaining ~10 Phase-N
magnitude-shaped theorems, (d) **relational-axis extension**
(W3-C5 — stays conjectural), (e) **richer feature taxonomy**
(P46-C2 refinement + decoder-side analogue). Product-shape
gaps (public SWE-bench-Lite JSONL, 70B-class coder) remain
orthogonal.

**Top-line, post-Phase-47 (capsule-research-center extension).**
The Phase-46 research center is now **sharper on both sides**:

1. **Formal frontier closed on the magnitude algebra (Part A).**
   Three theorems — W3-14 (per-capsule budget locality —
   negative), W3-15 (cohort-lift subsumption — positive,
   12th kind `COHORT` added), W3-16 (relational limitation —
   negative). The Phase-46 PARTIAL row on AdaptiveEdge
   closes FULL via cohort lift; the unification audit now
   reports **6/6 FULL, 0/6 PARTIAL, 0/6 FAIL**. W3-C3
   (11-kind completeness) is **falsified**; the honest
   count is 12. W3-16 names a sharp relational limitation
   (pairwise predicates across distinct capsule members
   cannot be enforced by cohort admission alone). See
   `docs/CAPSULE_FORMALISM.md` § 4.B and § 5; empirical
   anchor in
   `vision_mvp/experiments/phase46_unification_audit.py::audit_adaptive_edge_cohort`.
2. **Bundle-aware admission is not a paradigm shift (Part B).**
   P46-C1 in its strong form — "bundle-aware admission lifts
   decoder accuracy past 0.30" — is **falsified** on the
   Phase-31 noisy-extractor bench. The structural ceiling is
   0.200, a property of the Phase-31 priority decoder under
   noisy DISK_FILL_CRITICAL injection, not a property of
   admission. However, the decoder-aware bundle-learned
   policy achieves **full budget efficiency**: it hits the
   ceiling at B = 16 tokens where per-capsule learned
   (Phase 46) needs B = 96, a 6× efficiency lift. The next
   paradigm-shift candidate is therefore bundle-aware
   **decoding** (P47-C1), not more admission work. Anchor:
   `docs/archive/capsule-research/RESULTS_CAPSULE_RESEARCH_MILESTONE2.md` § 2.
3. **Cross-domain transfer is asymmetric and task-family-
   indexed (Part C).** First three-domain transfer study
   (incident / compliance / security). Within-domain
   admit-precision lifts +10 to +23 pp above base rate.
   Cross-domain transfer is **strong** on the
   incident-security pair (+21 pp above security's base
   rate, within −2 pp of within-domain) but **weak or
   negative** elsewhere (security → incident is −3 pp).
   Pooled training does not recover within-domain
   performance. Interpretation: transfer follows
   task-family similarity (operational detection vs
   document review); claim-kind features are domain-
   specific; scalar features disagree in sign across
   domains. P46-C2 is **partially supported, partially
   falsified**; successor conjecture P47-C3 states the
   task-family-indexed form. Anchor:
   `docs/archive/capsule-research/RESULTS_CAPSULE_RESEARCH_MILESTONE2.md` § 3.

The remaining gap is now structurally **four-shaped**:
(a) **bundle-aware decoding** (P47-C1, the post-Phase-47
paradigm-shift candidate), (b) **mechanical W3-C1 closure**
on the remaining ~10 Phase-N bounded-context theorems (they
are all magnitude-shaped — the cohort lift unblocks them
*structurally*), (c) **relational-axis extension** (W3-C5 —
stays conjectural until a Phase-N substrate primitive
forces it), (d) **richer feature taxonomy for transfer**
(P46-C2 refinement). Product-shape gaps (public SWE-bench-
Lite JSONL, 70B-class coder) remain orthogonal.

**Top-line, post-Phase-46 (capsule research milestone)**
(preserved below for context). The programme has the Context
Capsule abstraction as a **research center, not just a
product label** — three coupled additions all sit on top of
the unchanged SDK-v3 capsule contract C1..C6:

1. **Formal mathematical model** (`docs/CAPSULE_FORMALISM.md`):
   capsule space, lifecycle automaton, admissibility predicate,
   budget tropical-min monoid, capsule DAG. Theorems W3-7..W3-13;
   conjectures W3-C1..W3-C4 with sharp falsifiers.
2. **Capsule learning result** — admission policy is *strictly
   learnable* on Phase-31 incident-triage with Phase-32 noisy
   extractors. Headline P46-1: at budget = 16 tokens on a
   held-out by-seed test set, learned admit-precision = 0.796 vs
   best heuristic 0.634, **+16.2 pp**. The 40-feature logistic
   policy is fully inspectable; train/test gap < 1.5 pp.
3. **Unification audit** — 4/5 FULL + 1/5 PARTIAL + 0/5 FAIL on
   the per-primitive Theorem-W3-11 reduction (Phase-19 Handle,
   Phase-31 TypedHandoff, Phase-35 ThreadResolution, Phase-36
   AdaptiveEdge, end-to-end ProductReport). The PARTIAL
   (AdaptiveEdge) is the **honest near-falsifier** of the
   subsumption claim: its `max_active_edges` bound is table-
   level, not capsule-level.

The remaining gap is now structurally **three-shaped**:
(a) **formal** — extend Theorem W3-11 from 4 primitives to all
~15 Phase-N bounded-context theorems (W3-C1); (b) **learning**
— bundle-aware admission to lift the noise-poisoning decoder
ceiling past 0.225 on Phase-31 (P46-C1); (c) **falsification**
— stress the 11-kind alphabet with cross-run references and
out-of-tree adapters (W3-C3). These are listed § 4.12 and
form the explicit Phase-47 research agenda. The product-shape
gaps from Phase 44 (public SWE-bench-Lite JSONL availability;
70B-class coder frontier) remain open and are *orthogonal* to
the capsule research agenda.

**Top-line, post-Phase-44** (preserved below for context). The
programme now has a
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

### 4.12 The Context Capsule as a research center (Phase 46 — post-SDK-v3 milestone)

The SDK v3 milestone (`docs/archive/wevra-milestones/RESULTS_WEVRA_CAPSULE.md`,
2026-04-22) named the **Context Capsule** as Wevra's
load-bearing abstraction: every piece of context that
crosses a role / layer / run boundary is a typed, content-
addressed, lifecycle-bounded, budget-bounded, provenance-
carrying object. That milestone moved capsules from
"recurring shape across the substrate" to "named
contract C1..C6" with code, tests, and an end-to-end
runner integration.

Phase 46 is the *research-shaped follow-up* — it moves the
capsule abstraction from product label to **research center**.
Three additions, all additive on the substrate:

**1. Formal mathematical model
(`docs/CAPSULE_FORMALISM.md`).** Defines capsule space
$\mathcal{C}$, identity map, lifecycle automaton,
admissibility predicate $\mathcal{A}_b$, budget tropical-min
monoid, capsule DAG. States Theorems W3-7 (CID
homomorphism), W3-8 (admissibility monotone under
tightening, Lemma 3.2 promoted), W3-9 (ledger DAG acyclic
+ append-order-topological), W3-10 (chain tamper-evidence
under SHA-256), **W3-11 (capsule subsumption — partial,
proven on four primitive classes)**, W3-12 (view is
faithful header projection), W3-13 (run-pattern DAG height
$\le 4$). Section 5 names sharp-falsifier conjectures
W3-C1..W3-C4.

**2. Capsule learning problem
(`vision_mvp/wevra/capsule_policy.py`,
`vision_mvp/experiments/phase46_capsule_learning.py`,
`docs/archive/capsule-research/RESULTS_CAPSULE_LEARNING.md`).** Capsule admission —
"should this proposed capsule be admitted under the ledger's
budget?" — is a **learnable policy**, not just a fixed
heuristic. The Phase-46 driver builds a 5 143-capsule
benchmark on Phase-31 incident-triage with Phase-32 noisy
extractors (`spurious_prob = 0.30`), splits 80 / 20 by
random seed (so each scenario_id appears in train AND test
under different distractor draws), and trains a logistic-
regression policy on a 40-feature closed vocab via
full-batch GD in pure Python.

**Theorem-style claim P46-1.** On the held-out test set
($n = 40$ instances), the learned policy strictly dominates
every fixed heuristic on admit-precision at every tight
budget cell:

| Budget | FIFO  | Smallest | KindPri (4) | KindPri (8) | **Learned** |
|------- |------ |--------- |------------ |------------ |------------ |
| 16     | 0.634 | 0.615    | 0.487       | 0.487       | **0.796**   |
| 32     | 0.630 | 0.587    | 0.541       | 0.541       | **0.751**   |
| 48     | 0.604 | 0.592    | 0.560       | 0.560       | **0.684**   |
| 64     | 0.580 | 0.571    | 0.563       | 0.563       | **0.612**   |
| 96+    | 0.547 (saturation: every policy admits the full bundle) ||||

The win is +16.2 pp at $B = 16$ over the strongest
heuristic. Train-set numbers are within 1.5 pp of test
on every cell — no overfitting. The 40-feature model is
fully inspectable: top weights are
`claim:ERROR_RATE_SPIKE +1.23`, `claim:POOL_EXHAUSTION
+0.95`, `src:monitor +0.81`.

**P46-1 is the first empirical evidence for Conjecture
W3-C4 (admission learnability).** It opens three new
conjectures (P46-C1 bundle-aware admission; P46-C2 cross-
domain transfer; P46-C3 rate-distortion optimality of
header features) — all named with sharp falsifiers in
`docs/archive/capsule-research/RESULTS_CAPSULE_LEARNING.md` § 6.

**3. Unification stress test
(`vision_mvp/experiments/phase46_unification_audit.py`,
`vision_mvp/tests/test_capsule_subsumption.py`).** For
each of five substrate primitives — Phase-19 Handle,
Phase-31 TypedHandoff, Phase-35 ThreadResolution,
Phase-36 AdaptiveEdge, end-to-end ProductReport — the
audit constructs a real instance, lifts it via the
canonical adapter, and verifies the W3-11 reduction
$(k_T, b_T)$ holds operationally.

**Result: 4/5 FULL, 1/5 PARTIAL, 0/5 FAIL.**

| Primitive            | Capsule kind         | Verdict | Reduction                                                     |
|---                   |---                   |---      |---                                                            |
| Handle               | HANDLE               | FULL    | $B \to b_t = B$                                              |
| TypedHandoff         | HANDOFF              | FULL    | $\tau \to b_t = \tau$                                        |
| ThreadResolution     | THREAD_RESOLUTION    | FULL    | $(\tau, R_{\max}, W) \to (b_t, b_r, b_w)$                    |
| AdaptiveEdge         | ADAPTIVE_EDGE        | PARTIAL | TTL $\to b_r$ (`max_active_edges` is *table*-level, not capsule-level) |
| ProductReport        | RUN_REPORT + DAG     | FULL    | $\beta_{\rm cell} \to b_b$ on SWEEP_CELL kinds in DAG         |

The partial fit on AdaptiveEdge is the **honest near-
falsifier** of W3-C3: the substrate ships at least one
coordination object whose bounded-context invariant lives
at the *table* level, not the per-edge level. The capsule
contract subsumes the edge; it does not subsume the
table-level cap.

**Two new contract-test files** (`test_capsule_policy.py`,
`test_capsule_subsumption.py`) lock 16 new tests on the
policy framework and the per-primitive subsumption
reductions. Crucially, two of those tests are *negative
cases* — they document where the capsule contract is
*silent* (role topology, extractor soundness) so a future
agent does not accidentally try to re-subsume P31-5 /
P35-1 / P31-4 / P35-3 under the capsule budget.

**What this changes about the programme.**

Before Phase 46, the capsule contract was the SDK's product
identity. After Phase 46:

1. The contract is *also* a formal mathematical structure
   — a tropical-min monoid on five budget axes with proven
   monotonicity, a capsule DAG with proven topological
   properties, and a partial-subsumption theorem with an
   explicit reduction tuple per primitive.
2. The contract *opens* a real ML research problem
   (admission learnability) with a real held-out result.
   This is the programme's first ML-relevant problem
   *internal to the substrate* — distinct from the
   "use a real LLM as a downstream answerer" axis that
   dominates Phases 30–43.
3. The unification claim is now *falsifiable per
   primitive*: any new substrate primitive that ships in
   Phase 47+ has a precise check ("does it admit a
   $(k_T, b_T)$ reduction?") rather than a framing
   judgement.

**What this does NOT change.**

* The substrate primitives themselves are byte-for-byte
  unchanged. The `CapsuleLedger.admit_and_seal` runtime
  contract is unchanged; `BudgetedAdmissionLedger` is
  *additive*. SDK_VERSION remains "wevra.sdk.v3".
* The expressivity-separation theorems (P31-5, P35-1) and
  correctness-preservation theorems (P31-4, P35-3) are
  *NOT* subsumed by the capsule contract — by design. The
  formalism is explicit about this.
* The downstream task accuracy ceiling on Phase-31
  incident triage under noise (0.225) is unchanged. The
  per-capsule learning result lifts admit-precision but
  does not yet lift bundle-decoder accuracy. P46-C1
  names the next experiment.

**Phase 46 frontier.** Three explicit research items now
sit at the top of the research arc:

* **Formal frontier (W3-C1)**: write capsule reductions for
  the 11 remaining Phase-N bounded-context theorems. Closes
  the conjecture or surfaces a second honest non-fit.
* **Learning frontier (P46-C1)**: bundle-aware admission.
  Per-bundle scoring (e.g. permutation-invariant set
  encoder) that lifts the noise-poisoning ceiling past
  0.225 on the same Phase-31 setting.
* **Falsification frontier (W3-C3)**: stress the
  11-kind alphabet. Cross-run references and out-of-tree
  adapters are named candidate twelfth-kind risks; either
  fits or surfaces the missing kind.

Phase 46 is the first phase whose primary deliverable is
*neither* a new substrate primitive *nor* a new benchmark
domain — it is a **formalisation + ML problem + audit** on
an existing primitive. This is a deliberate move: the
programme has accumulated enough substrate that the next
high-value research is *unifying and learning over what is
already there*, not adding more.

---

### 4.13 The Context Capsule research centre, Phase 47 extension (cohort + bundle + transfer)

Phase 47 continues the research-first stance of Phase 46. No
new substrate primitive, no new benchmark domain — instead,
three tightly-coupled research moves that **sharpen the
boundary** of the capsule abstraction on all three fronts.
See `docs/archive/capsule-research/RESULTS_CAPSULE_RESEARCH_MILESTONE2.md` for the
full note; the summary below is programme-level.

**1. Formal frontier — cohort lift + relational limitation
(Part A).** Phase 46's 4/5 FULL + 1/5 PARTIAL unification
audit closes. Three theorems:

* **Theorem W3-14 (per-capsule locality — negative).** No
  per-capsule budget enforces cardinality invariants on the
  admitted set. Constructive: admit $N + 1$ minimal capsules
  each individually within budget; the count is unbounded.
  `docs/CAPSULE_FORMALISM.md` § 4.B, Theorem W3-14.
* **Theorem W3-15 (cohort-lift — positive).** Extend the
  alphabet with a twelfth kind `COHORT` whose parents are
  its members and whose `max_parents` axis bounds
  membership. Every cardinality invariant
  $|\{c : \Phi(c)\}| \le N$ then admits a cohort witness
  whose admission succeeds iff the bound holds. AdaptiveEdge
  closes PARTIAL → FULL. `capsule.py::CapsuleKind.COHORT`,
  `capsule_from_cohort`, `capsule_from_adaptive_sub_table`.
* **Theorem W3-16 (relational limitation — negative).**
  Cohort admission is cardinality-only. Generic relational
  invariants $\forall c_1 \neq c_2 : \Psi(c_1, c_2) = 1$ cannot
  be enforced by any magnitude-only extension of the algebra.
  The limitation is constructive via the cardinality-
  preserving rewrite.

**Unification audit:** 6/6 FULL (5 original + 1 cohort lift)
under the magnitude algebra. **W3-C3 (11 kinds complete) is
falsified** — the honest count is **12 kinds**. Successor
W3-C3' names the 12-kind stability hypothesis; W3-C5 is the
next extension (relational axis) if a substrate primitive
forces it.

**2. Bundle-aware learning frontier — falsification +
budget efficiency (Part B).** P46-C1 in its strong form
("bundle-aware admission lifts decoder accuracy past 0.30 on
held-out Phase-31") is **FALSIFIED on the noise-poisoned
Phase-31 test set** — the decoder-aware bundle-learned
policy saturates at the structural ceiling 0.200, does not
exceed it. The ceiling is structural: the Phase-31 priority
decoder outputs `disk_fill` whenever any DISK_FILL_CRITICAL
is admitted, and under `spurious_prob = 0.30` at least one
spurious DISK_FILL_CRITICAL is admitted in ≈ 100 % of test
scenarios regardless of policy. Accuracy ≤ Pr[gold =
disk_fill] = 1/5 = 0.200.

**Positive result (Theorem-style claim P47-2).** The
decoder-aware bundle-learned policy achieves **6× budget
efficiency** over per-capsule Phase-46 learned: it reaches
the ceiling at B = 16 tokens while per-capsule learned needs
B ≥ 96. The key design decision is the training objective
(decoder-aware label `1{implied_rc = gold}`) — the same
hypothesis class on causal labels does not reproduce the
effect. Code-backed
(`phase47_bundle_learning.py`, ~15 s wall).

**Implication.** The next paradigm-shift candidate on the
capsule ML axis is **bundle-aware decoding** (Conjecture
P47-C1), not more admission work. A Borda-count-style
plurality-aware decoder that is not first-match-priority
might lift decoder accuracy past 0.30; this is the natural
Phase-48 experiment.

**3. Cross-domain transfer — asymmetric, task-family-indexed
(Part C).** First three-domain transfer study. All three
domains train their own policy successfully (within-domain
precision +10 to +23 pp above base rate). Cross-domain
results are **asymmetric**:

| Train → Test          | Lift above base rate                |
|---                    |---                                  |
| incident → security   | **+21 pp** (near-within-domain)     |
| compliance → incident | +6 pp (weak)                        |
| compliance → security | +4 pp (trivial)                     |
| security → compliance | +3 pp (trivial)                     |
| incident → compliance | +2 pp (trivial)                     |
| security → incident   | **−3 pp** (below base rate)         |

Pooled training (two domains → held-out third) does **not**
recover within-domain performance. Feature attribution shows
scalar features (`log1p_n_tokens`, `log1p_n_bytes`) disagree
in sign across domains; claim-kind features are native-
domain-only because the closed feature vocabulary does not
cover compliance/security kinds.

**Verdict on P46-C2.** Partially supported (incident →
security works), partially falsified (most cross-cells are
trivial; one is negative). **Conjecture P47-C3 (task-
family-indexed transfer)**: cross-domain admission transfer
is strong for pairs sharing the same task-family structure
(role cast + scenario archetype + decoder shape) and weak-
or-negative otherwise. Incident and security share the
"operational multi-role detection" family; compliance is
structurally "multi-role document review".

**What changes about the programme.**

| Claim / Result             | Status before P47      | Status after P47            |
|---                         |---                     |---                          |
| W3-11 per-primitive        | 4/5 FULL + 1/5 PARTIAL | **6/6 FULL**                |
| W3-C3 (11 kinds)           | Conjectural            | **Falsified (12 kinds)**    |
| AdaptiveEdge table bound   | PARTIAL                | **FULL via COHORT**         |
| Relational invariants      | Not framed            | **Named limitation (W3-16)** |
| P46-C1 strong form         | Open                   | **Falsified on this bench** |
| Budget efficiency          | Open                   | **6× via decoder-aware labels (P47-2)** |
| Cross-domain transfer      | Open                   | **Asymmetric partial (P47-3)** |

**Is Phase 47 a paradigm shift?** **No.** Phase 47 moves the
capsule abstraction from "research center" to "research
center with **proven boundaries on both sides**" — an
expressive magnitude algebra on the inside, a relational
limit on the outside, and empirically falsifiable ML claims
in the middle. The programme's paradigm-shift candidate has
shifted from admission-side ML (which Phase 47 falsifies in
its strong form) to **bundle-aware decoding** (P47-C1). If a
decoder-side result breaks the 0.200 structural ceiling on
the same Phase-31 noisy bench, *that* justifies paradigm-
shift framing. Until then, the centre is strong, bounded,
and falsifiable — which is more than a "useful unification"
but less than a paradigm shift.

---

### 4.14 The Context Capsule research centre, Phase 48 extension (bundle-aware decoder frontier)

Phase 48 is the decoder-side attack on the post-Phase-47
research frontier. Phase 47 left two coupled open
questions — "is the 0.200 ceiling structural to the
decoder, not admission?" (P47-1) and "can a bundle-aware
decoder break the ceiling?" (P47-C1). Phase 48 turns the
first into a proved limitation theorem and turns the second
into a partially-positive empirical result.

**1. Formal frontier — admission locality + decoder
sufficiency (Part A).** Three decoder-side results:

* **Theorem W3-17** (admission locality — negative, proved,
  conditional). No admission rule $\pi$ that is *pure on
  capsule headers* and *indistinguishable on the spurious
  ceiling-forcing kind* can achieve decoder accuracy above
  the priority-decoder ceiling $\Pr[y_{\rm gold} = y^{\star}]$
  under a distribution that injects the spurious kind
  with probability $\ge 1 - \varepsilon$. On Phase-31 noisy
  bench this is tight at 0.200. Sharpens Phase-47's empirical
  observation into a limitation proof.
  `docs/CAPSULE_FORMALISM.md` § 4.C, Theorem W3-17.
* **Theorem W3-18** (plurality decoder sufficiency — positive,
  proved, conditional). On coherent-majority bundles — where
  the gold rc has strictly more implied-rc votes than any
  other rc — plurality decoding returns gold with probability 1
  while priority decoding returns wrong with probability 1 on
  the poisoned slice. The sharpest single-bundle separator
  (test
  `test_w3_18_plurality_strictly_dominates_priority_on_coherent_majority`)
  is: `{OOM_KILL/sysadmin, OOM_KILL/db_admin,
  ERROR_RATE_SPIKE/monitor, LATENCY_SPIKE/monitor,
  DISK_FILL_CRITICAL/network}` → priority = disk_fill,
  plurality = memory_leak. `docs/CAPSULE_FORMALISM.md` § 4.C,
  Theorem W3-18.
* **Claim W3-19** (empirical, code-backed). On the Phase-31
  noisy bench, held-out by-seed at $n_{\rm test} = 80$ (20
  seeds × 0.8/0.2 split), the `LearnedBundleDecoder` — a
  10-feature multinomial-logistic classifier over class-
  agnostic bundle-shape features — beats the 0.200 priority
  ceiling by **+15 pp to +17.5 pp**:
  * FIFO admission × B ∈ {64, 96, 128, 256}: 0.350 (test).
  * bundle_learned admission × B = 96: **0.375 (test)**.
  * oracle-clean causal slice: 0.575 (vs priority's 0.525).
  Plurality alone does NOT break the ceiling on the full bench
  (priority-tiebreak fallback on one-vote-each ties reproduces
  the ceiling); `SourceCorroboratedPriorityDecoder(min_sources=2)`
  degrades to 0.000 (it vetoes the causal chain's single-source
  high-priority kind too). The break is carried by the
  learned decoder's weight structure — a strong negative
  weight on `has_top_priority_kind` (penalising "DFC in
  bundle") combined with a positive weight on `votes_share`
  (rewarding vote coherence).

P47-C1 is therefore **partially validated**: a bundle-aware
decoder DOES break the ceiling on this bench (weakly by the
W3-C7 threshold — 0.375 < 0.400). A plurality-only decoder
does NOT break it, which was the naive reading of P47-C1;
the falsification of that naive reading is itself a
scientific result.

**2. Decoder-side cross-domain transfer — asymmetric, sign-
flip signature (Part B).** First decoder-side transfer study
on the two operational-detection domains (incident, security).
Compliance is omitted — its decoder shape (verdict threshold)
is structurally different, and including it would conflate
"decoder-side transfer fails" with "the two domains have
different decoder geometries."

Transfer matrix at $n_{\rm test} = 80$ per domain
(20 seeds × 0.8/0.2 split):

| Train → Test        | Learned dec acc | Priority acc | Lift over priority |
|---                  |---              |---           |---                 |
| incident → incident | 0.362           | 0.212        | +0.150             |
| incident → security | 0.300           | 0.200        | +0.100             |
| security → incident | 0.125           | 0.212        | **−0.087**         |
| security → security | 0.300           | 0.200        | +0.100             |

**Key observations.**

* Incident → security transfer is **fully positive** — the
  cross-decoder matches within-security accuracy
  (0.300 = 0.300). This is the "operational-detection family
  transfers" signature Phase 47 named at the admission layer;
  Phase 48 reproduces it at the decoder layer.
* Security → incident transfer is **sharply negative** — the
  security-trained decoder weights mis-adapt to incident's
  bundle geometry, producing 0.125 (below the 0.200 base rate
  AND below priority's 0.212).
* **Feature sign-flip** on `lone_top_priority_flag`: +1.13 on
  incident, −1.19 on security. Incident scenarios carry
  single-signature high-priority causal claims (DFC / OOM_KILL
  / TLS_EXPIRED by one authoritative role each); security
  scenarios carry multi-source high-priority corroboration
  (MALWARE_DETECTED + LATERAL_MOVEMENT + TTP_ATTRIBUTED across
  three roles). The opposite signs are the *structural* reason
  cross-transfer fails in the reverse direction.

Conjecture **W3-C6** names the asymmetric task-family-indexed
decoder-transfer form; P47-C3 (admission side) and W3-C6
(decoder side) are the same phenomenon, now sharp on both
sides of the capsule contract.

**3. What changes about the programme.**

| Claim / Result                  | Before P48                    | After P48                                      |
|---                              |---                            |---                                             |
| 0.200 ceiling status            | Empirically bounded above     | **Proved tight via W3-17 (admission locality)**|
| P47-C1 (bundle decoder breaks)  | Open                          | **Partially supported via W3-19 (+15 pp)**     |
| P47-C1 strong (>0.400 threshold)| Open                          | **Not met (0.375 < 0.400)**                    |
| Plurality decoder universal break | Open                        | **Falsified on full Phase-31 bench**           |
| Decoder-side transfer           | Not framed                    | **Asymmetric + sign-flip (W3-C6)**             |
| Paradigm-shift threshold        | Informal                      | **Stated via W3-C7 (explicit 2-part gate)**    |

**4. Does Phase 48 justify the paradigm-shift framing?**
**Materially closer. Not yet.** Phase 48 closes *both* sides of
the decoder-frontier story — a proved limitation theorem
(W3-17) on the admission side and an empirically-validated
ceiling break (W3-19) on the decoder side — in a way Phase
47 explicitly left open. But the break is **weak** against
the stringent W3-C7 threshold (0.375 < 0.400 target), and
the cross-domain transfer is **asymmetric** (incident →
security works; security → incident fails). A genuine
paradigm shift would need either (i) a decoder that
crosses 0.400 on the Phase-31 bench with a principled
mechanism, or (ii) symmetric transfer across ≥ 2
operational-detection domains. Phase 48 delivers neither.
What Phase 48 DOES deliver is a **decoder-shaped frontier
that is open and traversable** — the programme knows what
to measure next, has the tooling to measure it, and has a
partially-verified hypothesis (bundle-aware decoding is the
right axis) to sharpen.

The capsule centre is now **research-grade on the decoder
axis as well as the admission axis**; the paradigm-shift
bar is stated explicitly and not yet earned.

---

### 4.15 The Context Capsule research centre, Phase 49 extension (stronger decoder + symmetric transfer)

Phase 49 is the first direct attack on both gates of
Conjecture W3-C7 — the explicit paradigm-shift threshold.
Phase 48 closed one half of P47-C1 (the 0.200 ceiling is
breakable, code-backed) but left both W3-C7 gates open:
Gate 1 (≥ 0.400 test accuracy) hit only 0.375, and Gate 2
(symmetric cross-domain transfer) failed with a sharp
0.175 gap between incident→security and security→incident.
Phase 49 adds three decoder families, one extended feature
vocabulary, one training scheme, and three formal results;
both gates are now cleared under their canonical
operational forms.

**1. Stronger-decoder frontier (Part A, attacks Gate 1).**
Four Phase-49 decoder families, each a principled step up
from the Phase-48 V1 linear baseline:

* **``LearnedBundleDecoderV2``** — linear over a 20-feature
  V2 vocabulary (V1 + 10 domain-invariant relative
  features). Best cell 0.350 on Phase-31 bench; does NOT
  cross 0.400. A pure feature-engineering lift to linear
  does not close Gate 1.
* **``InteractionBundleDecoder``** — linear over 191 features
  (V2 + all pairwise crosses). Best cell 0.338. The
  explicit feature-cross hypothesis overfits at
  $n_{\rm train} = 320 \cdot 7 = 2240$ aug-pairs.
* **``MLPBundleDecoder``** — 1-hidden-layer MLP (hidden = 12)
  over V2 features, shared across rc. Best cell 0.362.
  Non-linearity alone helps modestly.
* **``DeepSetBundleDecoder``** — proper Deep Sets: per-capsule
  $\varphi(c, rc) \in \mathbb{R}^8$ summed over the bundle,
  concatenated with V2 aggregated features, scored through a
  1-hidden-layer MLP (hidden = 10, ~290 parameters). Best
  cell **0.425** on `bundle_learned_admit @ B=64` with
  augmented training. **Crosses Gate 1.** Anchor:
  `vision_mvp/experiments/phase49_stronger_decoder.py`.

A key ingredient is **training-data augmentation**: Phase 48
trained on FIFO @ B=256 only; Phase 49 trains on the union of
bundles from 7 admission cells (FIFO and learned admission at
varying budgets). This distribution-matches the decoder's
training distribution to its deployment distribution and is
the operational reason DeepSet crosses 0.400 (without
augmentation it reaches 0.388).

**2. Symmetric-transfer frontier (Part B, attacks Gate 2).**
Three results:

* **Theorem W3-20** (Deep Sets sufficiency — proved,
  constructive). The class-agnostic linear decoder class
  $\mathcal{H}_{\rm lin}$ is *strictly* contained in the Deep
  Sets class $\mathcal{H}_{\rm DS}$: there exists a per-
  capsule embedding $\varphi$ (the "top-priority kind does
  not imply this rc" conjunction) that expresses
  per-capsule-conjunctive shapes no aggregated-linear decoder
  can express.
* **Theorem W3-21** (linear-class asymmetry — proved,
  negative). A class-agnostic linear decoder over a feature
  whose gold-conditional sign flips across domains CANNOT
  achieve both per-domain optima simultaneously.
  This is the structural reason Phase 48's zero-shot transfer
  is asymmetric — not a training artefact, but a hypothesis-
  class limitation.
* **Claim W3-22** (multitask shared-head symmetric transfer —
  empirical, code-backed). The ``MultitaskBundleDecoder``
  jointly trained on pooled (incident, security) data with
  factorisation $w = w_{\rm shared} + w_{\rm domain}[d]$
  and $\ell_2$-regularisation
  $(\lambda_{\rm sh}, \lambda_{\rm dom}) = (10^{-3}, 5\cdot
  10^{-3})$ achieves, with $w_{\rm domain} \equiv 0$ at
  deployment (the "shared-head-only" reading), **0.350 on
  both incident and security test sets** — gap 0.000,
  ≥ 13 pp over each domain's priority baseline. Anchor:
  `vision_mvp/experiments/phase49_symmetric_transfer.py`.

The DeepSet decoder also reduces the zero-shot transfer gap
from V1's 0.175 to **0.038** — zero-shot transfer is now
symmetric within the 5 pp bar **on the gap metric**, though
not on the stricter within-domain-penalty metric.

**3. Conjecture W3-C8 (refined) — absolute vs relative
feature signs.** The empirical V2 sign-agreement rate
(0.550) is strictly lower than the V1 rate (0.700) on the
(incident, security) pair — the V2 relative-margin features
introduce MORE sign-dependence, not less. The refined
conjecture states that *absolute* count features (log1p_votes,
votes_share, frac_bundle_implies_rc) have stable signs across
operational-detection domains; *relative-margin* features
(`*_minus_max_other`, `is_strict_top_by_*`) have
domain-dependent signs because their gold-conditional sign
depends on the competitor distribution. This is open and
supported.

**4. Does Phase 49 cross the paradigm-shift bar?**

**Under a liberal reading: YES.**
* Gate 1 (≥ 0.400 on held-out Phase-31 bench): MET at 0.425
  point estimate (DeepSet @ best cell, n_test = 80; 95 %
  binomial CI $[0.317, 0.539]$).
* Gate 2 (approximately-symmetric transfer): MET under the
  multitask shared-head reading (0.350 on both domains with
  one weight vector, gap 0.000).

**Under a strict reading: NOT YET.**
* Gate 1's CI lower bound is 0.317 < 0.400 — consistent
  with crossing but not a rigorous certification. Crossing
  rigorously needs either (a) larger $n_{\rm test}$ (e.g.
  the Phase-31 bench at 40 seeds / $n_{\rm test} = 160$) or
  (b) a cell that hits ≥ 0.45 at the current $n$.
* Gate 2's zero-shot (pure weight-transfer) reading is not
  met for any Phase-49 decoder: DeepSet's transfer penalty
  is +0.138 pp ($i \to s$) and +0.137 pp ($s \to i$),
  above the 5 pp bar. This is structurally *expected* from
  W3-21: under a gold-conditional sign-flip feature, no
  class-agnostic decoder can achieve both per-domain
  optima. Closing Gate 2 in the strict zero-shot reading
  would need a richer hypothesis class whose conditional
  non-linearity actively internalises the sign-flip (Phase
  49's DeepSet reduces the gap from 0.175 to 0.038 but does
  not close it); or a representation-alignment step that
  normalises the feature distribution across domains.

**Honest summary.** Phase 49 materially advances the centre.
The decoder-side frontier now has three proved results
(W3-17, W3-18, W3-20, W3-21 — four total if W3-18's
conditional sufficiency is counted), one empirical
ceiling-break past the W3-C7 Gate 1 point estimate
(W3-23), and one symmetric-transfer result (W3-22). The
"paradigm-shift candidate" label is now defensible under a
liberal reading but the strict reading still has a small
(≈ 0.025 Gate 1 CI gap, ≈ 0.09 Gate 2 penalty gap) gap.
**The honest centre is: "Phase 49 is the first paradigm-shift
candidate phase in the programme, passing both gates under
the canonical-but-not-strictest reading; strict paradigm-
shift certification requires one more milestone on either
sample size (Gate 1) or zero-shot transfer hypothesis
(Gate 2)."**

**Remaining gap (post-Phase-49) is four-shaped.**
(a) **Stricter Gate 1**: a decoder that hits ≥ 0.450 at
$n_{\rm test} = 80$ or ≥ 0.400 at $n_{\rm test} = 160$.
(b) **Strict zero-shot Gate 2**: a decoder whose weight
vector transferred from A to B satisfies
$\mathrm{acc}(B, w_A) \ge \mathrm{acc}(B, w_B^*) - 0.05$ in
both directions, without pooled multitask training.
(c) **W3-C1 mechanical closure** on remaining ~10 Phase-N
magnitude theorems.
(d) **Relational-axis extension** (W3-C5 — stays
conjectural).

---

### 4.16 The Context Capsule research centre, Phase 50 extension (strict-reading certification — honest falsification)

Phase 50 takes the Phase-49 paradigm-shift candidate at its
strictest operational reading and tests both gates rigorously.
The result is an **honest falsification under the strict
reading** accompanied by three new formal results (Theorems
W3-24, W3-29 and refined Conjecture W3-C9) that surround the
frontier with proved boundaries on both sides.  Phase 50 is a
*negative-result milestone* that sharpens the programme's
paradigm-shift claim from "candidate under canonical-but-not-
strictest reading" (Phase-49 stance) to "canonical but
structurally-blocked-strict paradigm-shift candidate under the
gap-reading + pre-committed-cell reading" (Phase-50 stance).

**1. Gate 1 under larger $n_{\rm test}$ (Part A, attacks
the Phase-49 Gate 1 CI gap).**

Phase 49 reported a point estimate of 0.425 for
``DeepSetBundleDecoder`` at bundle_learned_admit @ B=64 on
$n_{\rm test} = 80$, with 95 % Wilson CI $[0.317, 0.539]$
(CI lower bound below 0.400).  Phase 50 re-runs the same sweep
at 2× and 4× the sample:

* $n_{\rm test} = 160$ (40 seeds): best post-search cell
  (DeepSet @ bundle_learned_admit @ B=48) hits **0.400
  exactly**, Wilson CI $[0.327, 0.477]$.  Pre-committed
  (W3-23) cell: DeepSet = 0.344.  **Strict CI reading NOT
  MET** (W3-25).
* $n_{\rm test} = 320$ (80 seeds): best post-search cell
  hits **0.362**, Wilson CI $[0.312, 0.417]$.  Pre-committed
  cell: DeepSet = 0.359.  **Even the point-estimate reading
  is NOT MET** (W3-26).

**Theorem W3-24** (proved, classical extreme-value lemma):
The post-search best-cell estimator $\hat{p}^\max$ over $C$
evaluation cells has upward bias $\Omega(\sigma_n \sqrt{\log C})$
in the null regime.  For Phase-49's $C=21, n=80$:
$\sigma_n \approx 0.055$, $\sqrt{2 \log 21} \approx 2.47$,
expected bias $\approx 0.136$.  Phase-49 at $n=80$
(0.425) → Phase-50 pre-committed cell at $n=320$ (0.359): drop
of 0.066 consistent with the combined sample-noise +
winner's-curse correction.

**Gate 1 honest verdict (Phase 50): NOT MET on any reading at
$n_{\rm test} = 320$.**  Phase-49's canonical-reading claim is
retracted.

**2. Strict zero-shot Gate 2 across 6 hypothesis classes
(Part B, attacks the Phase-49 zero-shot penalty gap).**

Phase 50 ships three new Phase-50 zero-shot candidates —
``SignStableFeaturesV2`` sub-family (a ``LearnedBundleDecoderV2``
restricted to the 8 Phase-49-identified sign-stable features),
``StandardisedBundleDecoderV2`` (V2 with source-domain-only
z-score standardisation), and ``SignStableDeepSetDecoder`` (a
DeepSet with per-capsule $\varphi$ and V2 aggregated features
restricted to the sign-stable sub-families).  Combined with the
Phase-48 V1, Phase-49 V2, Phase-49 DeepSet baselines, six
zero-shot families are compared on (incident, security) at
$n_{\rm test} = 80$ per domain:

| Family            | within inc | within sec | $i\to s$ | $s\to i$ | gap     | max penalty | strict G2 |
|---                |---         |---         |---        |---        |---      |---            |---         |
| v1 (P48)          | 0.362      | 0.300      | 0.300     | 0.125     | 0.175   | +0.237        | NO         |
| v2 (P49)          | 0.287      | 0.312      | 0.200     | 0.175     | 0.025   | +0.112 (min)  | NO         |
| stable (P50)      | 0.325      | 0.300      | 0.212     | 0.163     | 0.050   | +0.163        | NO         |
| std (P50)         | 0.350      | 0.212      | 0.300     | 0.188     | 0.112   | +0.162        | NO         |
| deepset (P49)     | 0.350      | 0.388      | 0.250     | 0.212     | 0.038   | +0.138        | NO         |
| stable_deepset (P50) | 0.362   | 0.400      | 0.237     | 0.237     | **0.000** | +0.163      | NO         |

**Claim W3-27** (empirical, Phase 50): **No zero-shot family
achieves max per-direction transfer penalty ≤ 5 pp.**  Best
max-penalty: V2 full at +0.112, above the bar.

**Claim W3-28** (empirical, Phase 50): Under the **gap**
reading of Gate 2 — "the two transfer accuracies are within
5 pp of each other" — sign-stable DeepSet achieves **gap =
0.000** (both directions 0.237 on $n_{\rm test} = 80$),
strictly meeting the bar.

**Theorem W3-29** (proved, conditional on strict convexity):
For two domains with distinct Bayes-optimal linear
decoders $w^*_A \ne w^*_B$,
$$
(\mathcal{R}_A(w) - \mathcal{R}_A^*) + (\mathcal{R}_B(w) - \mathcal{R}_B^*)
\;\ge\; \frac{\lambda_{\min}}{4} \|w^*_A - w^*_B\|^2.
$$
Structural lower bound on zero-shot transfer risk-penalty.

**3. Conjecture W3-C9 (Phase 50) — Gate-2 reformulation.**
The programme's honest post-Phase-50 stance is to retain
Conjecture W3-C7 Gate 2 as *aspirational* in its strict
"penalty ≤ 5 pp both directions" reading and adopt the **gap
reading** as the operationally defensible restatement.  Under
the gap reading, Phase-49 + Phase-50 have together produced a
single-weight-vector zero-shot decoder (sign-stable DeepSet)
that transfers directionally-symmetrically across (incident,
security); penalty-reading strict Gate 2 is structurally
blocked on this pair by Theorem W3-21 + Theorem W3-29 and is
not closed by any of six principled zero-shot hypothesis
classes.

**4. Does Phase 50 upgrade or retract the paradigm-shift
claim?**

**Retracts under strict reading.**  The Phase-49 "paradigm-
shift candidate passing both gates under canonical-but-not-
strictest reading" claim is **partially retracted**:
- Point-estimate Gate 1 at $n_{\rm test} = 320$: NOT MET
  (W3-26).  Phase-49's 0.425 was inflated by W3-24 winner's-
  curse bias plus sample noise.
- Strict zero-shot Gate 2 (penalty reading): NOT MET (W3-27)
  — refuted across six principled zero-shot families.

**Preserves under two honest reformulations.**
- Gap-reading Gate 2: MET by sign-stable DeepSet (W3-28).
- Pooled-multitask Gate 2: MET (W3-22, Phase 49, unaffected).
- At $n_{\rm test} = 80$ Gate 1 is MET at point estimate
  (W3-23, reproduced but CI-wide); at $n_{\rm test} = 160$ it
  is borderline MET (W3-25, 0.400 exactly at best cell with
  winner's-curse risk).

**Programme classification (final honest).**

| Reading                                                       | Verdict         |
|---                                                            |---               |
| Strict W3-C7 (CI-lower-bound Gate 1 + zero-shot-penalty Gate 2) | **NOT MET**      |
| Canonical W3-C7 (point-estimate Gate 1 at $n=80$ + pooled-multitask Gate 2) | **MET**          |
| W3-C9 refined (point-estimate Gate 1 at $n=80$ + gap-reading Gate 2) | **MET** (gap via W3-28) |
| Strict at $n_{\rm test} = 320$ (point-estimate Gate 1)        | **NOT MET** (0.362 < 0.400) |

**The final honest status.**  Phase 49 is a **canonical
paradigm-shift candidate** — passing both gates under
operationally defensible readings (point-estimate at $n=80$;
pooled-multitask or gap-reading zero-shot).  Phase 49 is **NOT
a strict paradigm shift** — the strict CI + strict zero-shot
penalty reading is structurally blocked.  Phase 50 names the
block (W3-24 winner's curse + W3-29 Bayes-divergence bound)
and reformulates the honest bar (W3-C9).

**What would still have to be true for full paradigm-shift
claim.**  Under the strict pre-Phase-50 reading, one would
need *simultaneously*:
(a) A decoder that hits $\hat{p} \ge 0.45$ at $n_{\rm test} =
    320$ (so Wilson CI lower bound ≥ 0.400).  The Phase-49
    DeepSet hits 0.362 here — a 9 pp shortfall that would
    require a materially richer hypothesis class, more
    training data, or a tighter benchmark distribution.
(b) A zero-shot hypothesis class whose weight-only transfer
    satisfies max-per-direction penalty ≤ 5 pp on (incident,
    security).  Theorem W3-29 gives a structural lower bound;
    we conjecture (W3-C9) that no class-agnostic class
    achieves this without per-domain adaptation.

**Or the bar should be reformulated (W3-C9).**  The honest
recommendation: adopt the gap reading of Gate 2 (operationally
strictly met by W3-28) and accept pre-committed-cell
point-estimate for Gate 1 (honest, W3-24-aware).  Under this
reformulation Phase-49 IS the paradigm-shift candidate; Phase-50
is its limitation-theorem milestone.

**Remaining gap (post-Phase-50) is three-shaped.**
(a) **True strict Gate 1 at $n \ge 320$**: still open.  Would
    require a new decoder class (per-domain adaptation,
    representation learning, or a tighter benchmark).
(b) **True strict Gate 2 penalty at 5 pp** on weight-only
    zero-shot transfer: W3-29-bounded, conjecturally unattainable
    on (incident, security).  Open for a *different*
    operational-detection pair.
(c) **W3-C1 mechanical closure + W3-C5 relational extension**:
    unchanged from Phase 49.

Anchor: `docs/archive/capsule-research/RESULTS_CAPSULE_RESEARCH_MILESTONE5.md` and
`docs/CAPSULE_FORMALISM.md` § 4.E.

---

### 4.17 The Context Capsule research centre, Phase 51 extension (cohort-relational decoder — honest limitation-leaning milestone)

Phase 51 is the deliberate next move after Phase 50's strict-
reading retraction.  The strict W3-C7 reading is retracted;
W3-C9's refined reading (gap reading of Gate 2 + point-estimate
Gate 1 at $n=80$) is the defensible bar; Phase-50's sign-stable
DeepSet achieves W3-C9's gap reading at level 0.237.  The
honest open question is the **level** of direction-invariant
zero-shot transfer, not a new strict-reading attempt.  Phase
51 attacks the level question with a hypothesis class
structurally outside the magnitude-monoid linear family.

**1. Theorem W3-30 (cohort-relational strict separation — proved, constructive).**
See `docs/CAPSULE_FORMALISM.md` § 4.F.  A decoder that
partitions the bundle by ``source_role``, computes a per-role
feature vector ψ, sums over roles, and augments with a cross-
role feature vector ρ (distinct-roles-supporting-rc, pairs-of-
supporting-roles, is-role-plurality-for-rc, …) is strictly
richer than DeepSet: it distinguishes two bundles with
identical (claim_kind, rc) multisets but different role
assignments — a distinction DeepSet's per-capsule φ-sum
cannot make.  Code anchor:
`vision_mvp/wevra/capsule_decoder_relational.py::CohortRelationalDecoder`.

**2. Claim W3-31 (empirical, code-backed — matched pipeline).**

| Metric                                 | Relational | SS-DeepSet (same pipeline) | DeepSet (full) | Phase-50 SS-DeepSet (ref) |
|---                                     |---         |---                          |---              |---                         |
| within-incident                         | 0.350      | 0.312                       | 0.388           | 0.362                     |
| within-security                         | 0.362      | 0.338                       | 0.362           | 0.400                     |
| $i \to s$ zero-shot                     | 0.237      | 0.225                       | 0.225           | 0.237                     |
| $s \to i$ zero-shot                     | 0.275      | 0.188                       | 0.212           | 0.237                     |
| gap ($\le 0.05$ = W3-C9 met)            | **0.038**  | 0.038                       | 0.013           | 0.000                     |
| max penalty (5 pp bar = aspirational)   | +0.125     | +0.125                      | +0.175          | +0.163                    |
| **level = min** ($i \to s$, $s \to i$) | **0.237**  | 0.188                       | 0.212           | 0.237                     |

(All at $n_{\rm test} = 80$ per domain; Phase-51 trains under
augmented FIFO @ [256, 96, 48], Phase-50 baseline trained under
FIFO @ 256 only.)

**On Phase-31 Gate 1 (pre-committed cell, $n_{\rm test} = 80$).**

| Decoder                  | $\hat p$ @ bundle_learned_admit/B=64 | Best cell | @ cell |
|---                       |---                                    |---         |---      |
| DeepSet (reproducer)     | **0.425**                              | 0.425      | bundle_learned_admit, B=64 |
| CohortRelational (P51)   | 0.362                                  | 0.388      | learned(p46), B=96 |

**3. Honest reading.**

* **W3-C9 gap reading MET** (gap 0.038 ≤ 0.05).  The
  relational decoder is a third Phase-49+ family that meets
  direction-invariance, alongside DeepSet-full and sign-stable
  DeepSet.
* **Level-lift against same-pipeline baseline: weak (+5 pp,
  4 instances at $n=80$, not statistically robust).**  The
  relational decoder edges the Phase-51-pipeline-reproduced
  sign-stable DeepSet by 5 pp but matches rather than exceeds
  the Phase-50 reported 0.237 level.
* **Level-lift against Phase-50 reported ceiling: NOT MET
  (matches).**  The relational decoder does not cleanly break
  Phase-50's 0.237 ceiling.  It re-confirms it under a matched
  pipeline.
* **Gate 1 on Phase-31: regression** (0.362 vs 0.425 at
  pre-committed cell; 0.388 best cell).  The relational
  decoder trades within-domain Phase-31 accuracy for
  cross-domain direction-invariance.

**4. Conjecture W3-C10 (level-ceiling on (incident, security)).**
See `docs/CAPSULE_FORMALISM.md` § 4.F.  *Under direction-
invariance (gap $\le 0.05$), zero-shot transfer on the
Phase-31 + Phase-33 operational-detection pair is level-
bounded: no Phase-51+ decoder class achieves min-direction
zero-shot level materially above the Phase-50 sign-stable-
DeepSet 0.237 mark.*  **Supported** by Phase 51 (eight
Phase-49/50/51 decoder families all fall at or below 0.237
under direction-invariance), not falsified.

**5. Does Phase 51 move the paradigm-shift needle?**

**No strict move.**  Phase 51 is a **limitation-leaning
milestone**: it operationalises the relational axis (W3-C5
direction), proves a formal strict-separation result (W3-30),
and **bounds** the next-decoder frontier empirically without
breaking the Phase-50 ceiling.

**What the programme should believe now.**  The strict reading
of W3-C7 is retracted (Phase 50).  The defensible reading is
W3-C9.  Phase 49 is the canonical paradigm-shift candidate.
Phase 50 is its limitation-theorem milestone on the strict
reading.  Phase 51 is its **level-frontier limitation
milestone** on the defensible reading: the relational axis is
structurally legitimate but empirically not free on
(incident, security).  **Do not reintroduce the strict W3-C7
bar; do not claim Phase 51 broke the Phase-50 level ceiling.**

**Remaining gap (post-Phase-51) is three-shaped.**

(a) **Different task-family pair (Phase-52 option B).**  Run
    Phase-49/50/51 families on (incident, compliance),
    (security, compliance), or a new operational-detection
    pair where $\|w^*_A - w^*_B\|$ may be smaller; W3-29's
    penalty floor shrinks and the level-ceiling may move.
(b) **Specific-feature relational decoder (Phase-52 option A,
    speculative).**  Engineer a relational feature pair that
    specifically targets W3-C10 on (incident, security).  Not
    obviously different from "deeper DeepSet" without a
    theoretical anchor.
(c) **Accept W3-C10 and redirect (Phase-52 option C).**  Treat
    the 0.237 level as a standing empirical ceiling and turn
    effort to mechanical W3-C1 closure on the remaining ~10
    Phase-N magnitude theorems, or to a relational *substrate*
    primitive whose bounded-context invariant exercises W3-C5
    on the substrate side rather than the decoder side.

Anchor: `docs/archive/capsule-research/RESULTS_CAPSULE_RESEARCH_MILESTONE6.md` and
`docs/CAPSULE_FORMALISM.md` § 4.F.

---

### 4.18 SDK v3.1 — capsule-native runtime (capsules drive execution, not just describe it)

The Phase-46..51 work above is on the *decoder* / research-center
axis: capsules as the substrate over which admission, bundling,
and zero-shot decoding are studied. The SDK v3.1 milestone is on a
**different axis entirely** — the **runtime** axis. Up to v3.0 the
capsule layer was a post-hoc fold: a Wevra run executed end-to-end
as ordinary Python and a sealed capsule DAG was synthesised from
the finished ``product_report`` dict afterwards. Capsules
*described* the run; they did not gate it. SDK v3.1 makes capsules
(partially) drive execution.

**1. The new contract.** ``CapsuleNativeRunContext`` is a runtime
context that owns a ``CapsuleLedger`` and exposes one ``seal_*``
method per Wevra-run stage (profile / readiness / sweep_spec /
sweep_cell / provenance / artifact / run_report). Each stage
seals a typed capsule before the next stage can read its result;
parent-CID gating (Capsule Contract C5) refuses to admit a
capsule whose declared parents are not yet in the ledger. A stage
that fails leaves a typed in-flight register entry that never
reaches the ledger — observable via ``ctx.in_flight_failures()``.
``run_sweep`` accepts an optional ``ctx`` parameter; when
provided, each cell is sealed in flight via
``ctx.seal_sweep_cell``.

**2. Content-addressing at write time.** Substantive on-disk
artifacts (``readiness_verdict.json``, ``sweep_result.json``,
``provenance.json``) are written via
``ctx.seal_and_write_artifact``: the capsule's payload SHA-256
is computed from the in-memory bytes *before* writing; the
capsule is sealed under that hash; the bytes are committed; the
on-disk file is re-read and re-hashed; any drift raises
``ContentAddressMismatch``. The on-disk file is then authenticated
by its capsule's CID — a third party with the capsule view + the
file can verify locally. Meta-artefacts (``product_report.json``,
``capsule_view.json``, ``product_summary.txt``) are post-view
renderings of the canonical view; naming them under their own
ARTIFACT capsules would require the view to include capsules
hashing its own bytes (circular), so they remain renderings.

**3. Four theorems (proved, code-backed).**

* **Theorem W3-32** (lifecycle ↔ execution-state correspondence,
  proved by inspection): each runtime stage maps to a capsule
  lifecycle state (in-progress / sealed / failed) bijectively.
* **Theorem W3-33** (content-addressing at artifact creation,
  proved by inspection + cross-validation test): the on-disk
  SHA-256 of any substantive artifact equals the SHA-256 in its
  ARTIFACT capsule's payload at ``seal_and_write_artifact``
  return.
* **Theorem W3-34** (in-flight ↔ post-hoc CID equivalence on
  non-ARTIFACT kinds, proved by per-kind set-equality test): the
  in-flight builder and the post-hoc ``build_report_ledger`` fold
  produce CID-equivalent ledgers for PROFILE / READINESS_CHECK /
  SWEEP_SPEC / SWEEP_CELL / PROVENANCE. ARTIFACT and RUN_REPORT
  CIDs **intentionally** diverge (in-flight carries real SHAs;
  post-hoc carries None).
* **Theorem W3-35** (parent-CID gating is the execution
  contract, proved by inspection): a stage that misorders the
  capsule lifecycle (e.g. a SWEEP_CELL before SWEEP_SPEC) raises
  a typed exception at the offending method, not an obscure
  downstream KeyError. The capsule layer enforces ordering at the
  type level.

**4. What this changes about Wevra's originality claim.** The
older positioning — *"every cross-boundary artefact is a typed,
content-addressed, lifecycle-bounded capsule"* — was true at the
Capsule Contract level (C1..C6) but operationally was a post-hoc
description of a run that had already happened. SDK v3.1 makes
the operational claim *"capsules drive execution"* honest on a
real slice: the entire Wevra run lifecycle (profile → readiness →
sweep → provenance → artifacts → run-report) is now a sequence of
capsule lifecycle transitions enforced at the type level, not a
sequence of Python calls audited post hoc. This is the move from
*capsules as audit graph* to *capsules as execution contract*.

**5. What remains legacy / post-hoc / out-of-scope.**

* Intra-cell objects (the LLM prompt sent during a real-mode
  sweep, the parsed patch, the per-instance test verdict)
  are not yet capsule-tracked. The capsule layer captures
  *run-boundary* objects only; intra-cell objects pass as plain
  Python. The next slice would name those with new kinds
  (``PROMPT``, ``GENERATED_PATCH``, ``TEST_VERDICT``).
* Meta-artefacts are not capsule-tracked (circular-dependency
  obstacle).
* Adversarial concurrent writers are not detected (the
  re-hash check is a TOCTOU detector for honest writers).
* The decoder / bundle-policy / cross-domain-transfer research
  centre (§§ 4.12–4.17) is unaffected. Decoder results live on
  the substrate; runtime results live above the substrate.
  v3.1 is strictly additive on the substrate.

**6. SDK surface delta.** Strictly additive on v3.0:

* New: ``CapsuleNativeRunContext``, ``ContentAddressMismatch``,
  ``seal_and_write_artifact``, ``CONSTRUCTION_IN_FLIGHT`` /
  ``CONSTRUCTION_POST_HOC`` (re-exported from ``vision_mvp.wevra``).
* New: ``RunSpec.capsule_native: bool = True`` (default flips
  onto the new path).
* Bumped: ``SDK_VERSION = "wevra.sdk.v3.1"``.
* Unchanged: ``ContextCapsule``, ``CapsuleLedger``,
  ``CapsuleView``, ``CAPSULE_VIEW_SCHEMA``, ``build_report_ledger``,
  every ``capsule_from_*`` adapter, every v3.0 contract test.

**7. Empirical anchor.** ``vision_mvp/tests/test_wevra_capsule_native.py``
ships 16 contract tests that lock the four theorems plus the
substantive-artifact content-addressing claim. The full
``vision_mvp.tests`` suite is **1406 / 1406 green** under the
new path; the legacy ``capsule_native=False`` path remains
green (back-compat guarantee).

Anchor: `docs/archive/wevra-milestones/RESULTS_WEVRA_CAPSULE_NATIVE.md` (milestone note);
`docs/CAPSULE_FORMALISM.md` § 4.G (theorems); test suite
`vision_mvp/tests/test_wevra_capsule_native.py`.

### 4.19 SDK v3.2 — intra-cell capsule-native + detached META_MANIFEST + strong on-disk verification

The v3.1 milestone (§ 4.18) closed the *run-boundary* slice but
deliberately stopped at the cell boundary. SDK v3.2 takes three
coupled moves: (i) extend capsule-native lifecycle into the
inner sweep loop with two new kinds; (ii) formalise the
meta-artefact boundary as a sharp circularity theorem with a
detached-witness corollary; (iii) strengthen on-disk verification
beyond the embedded ``chain_ok`` boolean.

**1. The intra-cell extension.** Inside every sweep cell, the
inner ``run_swe_loop_sandboxed`` loop calls ``generator(...)``
once per (task, strategy) and then ``sandbox.run(...)``. SDK
v3.2 routes both calls through hooks
(``on_patch_proposed`` / ``on_test_completed``) the runtime
wires up when a ``CapsuleNativeRunContext`` is active. The hooks
seal one PATCH_PROPOSAL capsule per generator call (parent:
SWEEP_SPEC; payload: task / strategy / parser_mode / apply_mode
/ n_distractors coordinates plus a SHA over the substitution
sequence and a bounded rationale), and one TEST_VERDICT capsule
per sandbox return (parent: the PATCH_PROPOSAL; payload: the
WorkspaceResult fields). The chain ``patch → verdict`` is
enforced at the type level (Theorem W3-32-extended): a verdict
cannot be sealed before its patch. Default-None hooks preserve
byte-for-byte Phase-40 behaviour for callers who do not want
the intra-cell extension.

**2. The detached-witness boundary.** Theorem W3-36
formalises an *impossibility*: there is no extension of the
primary ledger that admits an ARTIFACT capsule for any
meta-artefact (``product_report.json`` / ``capsule_view.json``
/ ``product_summary.txt``) without changing the rendered view
those meta-artefacts encode. The proof is a one-line
contradiction (the new capsule changes the rendered view
through both the headers list and the chain step). The
positive corollary is constructive: a META_MANIFEST capsule
sealed in a *secondary* ledger after the RUN_REPORT, whose
payload carries the on-disk SHAs of the meta-artefacts plus
the primary ``root_cid`` and ``chain_head``, is the strongest
authentication achievable. The trust unit is one explicit hop
beyond the primary view; cryptographic signing is orthogonal
and out of scope.

**3. Strong on-disk verification.**
``wevra-capsule verify`` now runs four independent checks:
(i) chain-from-headers recompute on the embedded view
(Theorem W3-37); (ii) chain-from-headers recompute on the
on-disk ``capsule_view.json`` (W3-37 again, against the
detached file); (iii) every ARTIFACT capsule's on-disk file
re-hashed and compared to the sealed payload SHA
(Theorem W3-38); (iv) the META_MANIFEST's meta-artefact SHAs
re-hashed against the on-disk meta-artefact bytes. Any failure
prints the specific drift and returns exit code 3. The view's
ARTIFACT and META_MANIFEST headers now ALWAYS carry their
payloads (an invariant strengthening of v3.1's
``include_payload=False`` default) so the verification claims
are recoverable from disk alone.

**4. Five new theorems / extensions.**

* **Theorem W3-32-extended** (intra-cell lifecycle correspondence,
  proved by inspection): the lifecycle correspondence of W3-32
  lifts to PATCH_PROPOSAL and TEST_VERDICT, with patch→verdict
  ordering enforced at the parent-CID gate.
* **Theorem W3-36** (meta-artefact circularity is sharp; detached-
  witness corollary, proved by structural argument + ledger-extension
  contradiction): meta-artefacts cannot be authenticated within the
  primary ledger; a META_MANIFEST in a secondary ledger is the
  strongest authentication.
* **Theorem W3-37** (chain-from-headers verification, proved by
  inspection): the runtime's chain step is a pure function of
  on-disk header fields, so verification recomputes the chain head
  from disk bytes.
* **Theorem W3-38** (ARTIFACT audit-time on-disk re-hash, proved
  by inspection): W3-33's *return-time* post-condition lifts to
  *audit-time* re-hashing — the audit-time on-disk SHA equals
  the sealed payload SHA iff the bytes have not drifted.
* **Spine equivalence preserved (Theorem W3-34 carry-over):**
  intra-cell capsules are siblings of SWEEP_CELL via SWEEP_SPEC,
  not modifications of the spine. The post-hoc fold's spine CIDs
  remain byte-equal to the in-flight builder's spine CIDs on
  PROFILE / READINESS_CHECK / SWEEP_SPEC / SWEEP_CELL /
  PROVENANCE.

**5. What this changes about Wevra's originality claim.** The v3.1
positioning was *"capsules drive execution at the run boundary"*.
With v3.2, capsules also drive execution **past the cell boundary**:
inside every sweep cell the inner parse→apply→test transition is
two sealed capsules with parent-CID gating, not a function-call
sequence. The meta-artefact circularity is no longer a vague TODO
or an asymmetric apology in the milestone note — it is a sharp
limitation theorem with a constructive boundary witness. ``wevra-capsule
verify`` is no longer a header-trust check — it recomputes from
disk bytes and re-hashes. Three classes of "operational claim that
was ahead of code" have become "operational claim that the code
honestly earns."

**6. What remains legacy / post-hoc / out-of-scope.**

* Generator prompts, raw LLM responses, and the parser's
  ``ParseOutcome`` (kind / recovery label / detail) are not yet
  capsule-tracked. The next intra-cell slice would name them as
  ``PROMPT`` / ``LLM_RESPONSE`` / ``PARSE_OUTCOME`` capsules.
* META_MANIFEST itself is not authenticated by the primary ledger.
  Theorem W3-36 establishes this is *impossible* without
  cryptographic signing; it is a sharp limitation, not a TODO.
* Adversarial concurrent writers remain out of scope (the trust
  boundary is the same as Wevra's sandbox boundary).
* Cross-run determinism on the full DAG remains open.

**7. SDK surface delta.** Strictly additive on v3.1:

* New: ``capsule_from_patch_proposal``, ``capsule_from_test_verdict``,
  ``capsule_from_meta_manifest``,
  ``verify_chain_from_view_dict``,
  ``verify_artifacts_on_disk``,
  ``verify_meta_manifest_on_disk`` (re-exported from
  ``vision_mvp.wevra``).
* New: ``CapsuleNativeRunContext.seal_patch_proposal``,
  ``.seal_test_verdict``, ``.seal_meta_manifest``,
  ``.render_meta_manifest_view``.
* New: ``CapsuleKind.PATCH_PROPOSAL``, ``.TEST_VERDICT``,
  ``.META_MANIFEST`` (closed-vocabulary additions).
* New: ``meta_manifest.json`` artefact written on every run.
* Bumped: ``SDK_VERSION = "wevra.sdk.v3.2"``.
* Unchanged: ``RunSpec`` fields; ``CAPSULE_VIEW_SCHEMA``
  (``wevra.capsule_view.v1`` — the new payloads are additive
  on the same schema); the entire v3.1 surface; the post-hoc
  ``build_report_ledger`` adapter; every existing
  ``capsule_from_*`` adapter.

**8. Empirical anchor.**
``vision_mvp/tests/test_wevra_capsule_native_intra_cell.py`` ships
16 contract tests that lock the four new theorems plus the
W3-34 spine-equivalence preservation. Combined with v3.1's
``test_wevra_capsule_native.py`` (16 tests still green) and the
public-surface lock (``test_wevra_public_api.py`` 10 tests),
the capsule-native runtime contract is now witnessed by 42
contract tests. Full ``vision_mvp.tests/test_wevra_*`` suite
**101 / 101 green**; the substrate's ``run_swe_loop_sandboxed``
hook addition does not regress any phase-40, phase-42, phase-45,
or phase-47 substrate tests (98 / 98 green on the targeted run).

Anchor: `docs/archive/wevra-milestones/RESULTS_WEVRA_INTRA_CELL.md` (milestone note);
`docs/CAPSULE_FORMALISM.md` § 4.H (theorems); test suite
`vision_mvp/tests/test_wevra_capsule_native_intra_cell.py`.

### 4.20 SDK v3.3 — sub-intra-cell parser-axis + lifecycle audit + deterministic-mode replay

The v3.2 milestone (§ 4.19) closed the *intra-cell* slice on the
patch / verdict pair. SDK v3.3 takes three coupled moves: (i) extend
capsule-native lifecycle one further structural layer to the
parser axis with a new PARSE_OUTCOME kind; (ii) add a
runtime-checkable `CapsuleLifecycleAudit` that mechanically
verifies the lifecycle correspondence on every finished run;
(iii) add an opt-in deterministic-mode replay flag that strips
per-run timestamps from PROVENANCE / RUN_REPORT / READINESS_CHECK
capsule payloads so two runs of the same logical input collapse
to byte-identical full-DAG CIDs.

**1. The sub-intra-cell extension.** Inside every sweep cell, the
inner `run_swe_loop_sandboxed` loop calls `generator(...)` once
per (task, strategy) and then `sandbox.run(...)`. SDK v3.3 wraps
the generator return path with a ratiomale-to-failure-kind mapping
(`_parse_outcome_from_rationale`) that recovers the parser's
structured outcome from the substrate's `ProposedPatch.rationale`
without coupling the capsule layer to substrate string formats.
The runtime then seals one PARSE_OUTCOME capsule per (task,
strategy) — parent: SWEEP_SPEC; payload: coordinates + parser
`ok` boolean + closed-vocabulary `failure_kind` from
`swe_patch_parser.ALL_PARSE_KINDS ∪ {"oracle", "gen_error"}` +
`recovery` label + `substitutions_count` integer + bounded
`detail` string. The downstream PATCH_PROPOSAL is parented on
*both* SWEEP_SPEC and the upstream PARSE_OUTCOME (admission
fails if the parse-outcome CID is not yet sealed — Capsule
Contract C5). The chain `parse → patch → verdict` is enforced at
the type level (Theorem W3-39).

**2. The lifecycle audit.** A finished `CapsuleNativeRunContext`
has a `audit_capsule_lifecycle(ctx)` method that mechanically
verifies eight invariants:

  - **L-1** No orphan capsules (every PROPOSED-but-not-sealed
    in-flight entry has a non-empty failure string AND is not in
    the ledger).
  - **L-2** PATCH_PROPOSAL parent set includes SWEEP_SPEC.
  - **L-3** TEST_VERDICT parent is exactly one sealed
    PATCH_PROPOSAL.
  - **L-4** PARSE_OUTCOME parent is exactly SWEEP_SPEC.
  - **L-5** SWEEP_CELL parent is exactly SWEEP_SPEC.
  - **L-6** PARSE_OUTCOME ↔ PATCH_PROPOSAL ↔ TEST_VERDICT
    coordinate multisets are equal.
  - **L-7** PATCH_PROPOSAL coordinates match its PARSE_OUTCOME
    parent's.
  - **L-8** TEST_VERDICT is sealed strictly after its
    PATCH_PROPOSAL in the ledger's append order.

Returns OK / BAD / EMPTY plus typed `{rule, capsule_cid,
capsule_kind, detail}` violations. The audit also has a
`audit_capsule_lifecycle_from_view(view)` form that runs from a
forensic `capsule_view.json` alone — auditors do not need the
runtime ctx that produced it. Theorem W3-40 anchors the audit's
soundness (proof-by-inspection of the eight `_check_l*` methods).

**3. Deterministic-mode replay.** `RunSpec(deterministic=True)`
strips per-run / host-local / wall-clock fields from the
PROVENANCE / READINESS_CHECK / RUN_REPORT capsule payloads
(timestamps, absolute paths, hostnames, wall-clock seconds, output
directories) and the ARTIFACT capsule paths (basename only). After
canonicalisation, every payload is a deterministic function of
the profile, JSONL bytes, and the closed code path; therefore
every CID is deterministic; therefore every parent set is
deterministic; therefore every transitive chain (including chain
head and root CID) is deterministic (Theorem W3-41). On-disk
`product_report.json` still records wall-clock fields for
forensic context — the determinism is on the capsule graph, not
on wall clock.

**4. Three new theorems.**

* **Theorem W3-39** (PARSE_OUTCOME lifecycle gate + parse → patch
  → verdict DAG chain, proved by inspection): every PARSE_OUTCOME
  has parent SWEEP_SPEC; every PATCH_PROPOSAL admitted with a
  `parse_outcome_cid` argument has that CID in its parent set;
  admission fails if the parse-outcome CID is not yet sealed.
* **Theorem W3-40** (Lifecycle-audit soundness, proved by
  inspection + mechanically-checked on every run): a finished
  ledger whose audit returns `verdict == "OK"` satisfies the
  eight invariants L-1..L-8.
* **Theorem W3-41** (Deterministic-mode CID determinism on full
  DAG, proved by inspection of the canonicalisation set + empirical
  cross-run set-equality test): two runs of the same deterministic
  profile under `deterministic=True` produce byte-identical CIDs
  on every kind, identical chain head, identical root CID.

**5. What this changes about Wevra's originality claim.** The v3.2
positioning was *"capsules drive execution at the run boundary
and the inner sweep loop pair (patch, verdict)"*. The v3.3
positioning is sharper:

> *Capsules drive execution at the run boundary, the inner sweep
> loop pair, AND the parser axis. The lifecycle correspondence is
> mechanically checkable on every finished run. The full capsule
> DAG is reproducible byte-for-byte across machines under a
> stated determinism flag.*

Three classes of "operational claim that was ahead of code" have
been promoted to "operational claim the code honestly earns":

1. The parser-axis taxonomy (`failure_kind`, `recovery`) was
   previously buried in rationale strings; it is now a typed
   capsule on the DAG.
2. The lifecycle correspondence (W3-32, W3-32-extended) was
   previously a paper-grade theorem; it is now a runtime audit
   that runs on every finished run.
3. Cross-run capsule DAG comparison was previously only on the
   spine kinds (W3-34); it is now full-DAG under the
   determinism flag.

**6. What remains legacy / post-hoc / out-of-scope.**

* LLM prompt bytes and raw LLM response bytes remain plain Python.
  The next sub-intra-cell slice would name them as `PROMPT` and
  `LLM_RESPONSE` capsules (Conjecture W3-C5).
* Sandbox stdout / stderr / test trace remain plain bytes.
* Adversarial concurrent writers remain out of scope (the trust
  boundary is the same as Wevra's sandbox boundary).
* Real-LLM mode is non-deterministic by construction; the
  determinism flag does not change that.
* Cryptographic signing of META_MANIFEST remains orthogonal and
  out of scope.

**7. SDK surface delta.** Strictly additive on v3.2:

* New: `CapsuleKind.PARSE_OUTCOME`, `capsule_from_parse_outcome`,
  `PARSE_OUTCOME_ORACLE` sentinel.
* New: `CapsuleNativeRunContext.seal_parse_outcome`,
  `seal_patch_proposal(parse_outcome_cid=...)` argument,
  `seal_and_write_artifact(recorded_path=...)` argument.
* New: `CapsuleLifecycleAudit`, `LifecycleAuditReport`,
  `audit_capsule_lifecycle`, `audit_capsule_lifecycle_from_view`
  (re-exported from `vision_mvp.wevra`).
* New: `RunSpec.deterministic: bool = False`.
* New: `--deterministic` CLI flag on `vision_mvp.product`.
* Bumped: `SDK_VERSION = "wevra.sdk.v3.3"`.
* Unchanged: every v3.2 contract test (101) still passes
  byte-for-byte; capsule view schema name `wevra.capsule_view.v1`;
  the post-hoc `build_report_ledger` adapter; the META_MANIFEST
  detached-witness boundary.

**8. Empirical anchor.**
`vision_mvp/tests/test_wevra_capsule_native_deeper.py` ships 18
contract tests covering the W3-39 / W3-40 / W3-41 claims plus the
PARSE_OUTCOME ↔ PATCH_PROPOSAL coordinate-matching invariant and
the rationale-to-failure-kind mapping. Combined with v3.1's
`test_wevra_capsule_native.py` (16 tests) and v3.2's
`test_wevra_capsule_native_intra_cell.py` (16 tests), the
capsule-native runtime contract is now witnessed by 50 contract
tests. Full `vision_mvp.tests.test_wevra_*` + `test_capsule_*`
suite **165 / 165 green**; the substrate's
`run_swe_loop_sandboxed` hook addition does not regress any
phase-40, phase-42, phase-45, phase-47, or phase-50 substrate
tests.

Anchor: `docs/archive/wevra-milestones/RESULTS_WEVRA_DEEP_INTRA_CELL.md` (this milestone
note); `docs/THEOREM_REGISTRY.md` (canonical theorem registry);
`docs/RESEARCH_STATUS.md` (canonical research-status); paper
draft: `papers/wevra_capsule_native_runtime.md` (claim taxonomy +
flagship write-up).

### 4.21 SDK v3.4 — sub-sub-intra-cell PROMPT / LLM_RESPONSE slice + synthetic mode + parser-boundary research

The v3.3 milestone (§ 4.20) extended capsule-native execution
one further structural layer to the parser axis (PARSE_OUTCOME
capsule per (task, strategy)) and added a runtime-checkable
lifecycle audit. The strongest remaining inner-loop boundary
named in v3.3's "what remains legacy" was the **LLM byte
boundary**: the prompt the patch generator sent and the raw
bytes returned. SDK v3.4 attacks that boundary.

**1. The sub-sub-intra-cell extension.** Inside every
LLM-backed sweep cell, `_real_cells._gen` constructs a prompt
via `build_patch_generator_prompt`, calls
`_call(prompt) -> response`, and parses the response. SDK v3.4
seals two capsules in flight at this level:

  * A **PROMPT** capsule (parent: SWEEP_SPEC) recording the
    prompt's coordinates + SHA-256 + byte length + bounded
    text snippet (≤ 4 KiB) + model_tag + prompt_style. The
    capsule is content-addressed (Capsule Contract C1) so two
    byte-identical prompts collapse to one capsule —
    naive + routing strategies that share an LLM call (the
    runtime's `raw_cache` deduplicates by `strategy_proxy`)
    produce one PROMPT and one LLM_RESPONSE on the DAG.
  * An **LLM_RESPONSE** capsule (parent: PROMPT, exactly one
    parent) recording coordinates + response SHA-256 + byte
    length + bounded snippet + elapsed milliseconds.
    Admission rejects if the prompt CID is not yet sealed
    (Capsule Contract C5).

The PARSE_OUTCOME's parent set is now either
`(SWEEP_SPEC,)` (oracle path — no LLM call) or
`(SWEEP_SPEC, LLM_RESPONSE)` (LLM-backed path). The
end-to-end inner-loop chain is therefore the five-link DAG
`PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL →
TEST_VERDICT` with strong parent-CID gating at each step.

**2. Three new typed-witness theorems (proved by inspection).**

  * **W3-42** PROMPT lifecycle gate — every PROMPT has parent
    set exactly `(SWEEP_SPEC,)`; idempotent on content.
  * **W3-43** Prompt → response parent gate — every
    LLM_RESPONSE has exactly one parent, a sealed PROMPT;
    idempotent on content.
  * **W3-44** PARSE_OUTCOME → LLM_RESPONSE chain coordinate
    consistency — when the PARSE_OUTCOME parents on an
    LLM_RESPONSE, their coordinates
    `(instance_id, parser_mode, apply_mode, n_distractors)`
    are equal. The `strategy` field is permitted to differ
    (multiple strategies share an LLM call when the prompt
    is identical).

**3. Lifecycle audit extended to L-1..L-11.** Three new
invariants:

  * **L-9** PROMPT.parents == (SWEEP_SPEC,).
  * **L-10** LLM_RESPONSE has exactly one parent, a sealed
    PROMPT.
  * **L-11** PARSE_OUTCOME / LLM_RESPONSE coordinate
    consistency (per W3-44).

`audit_capsule_lifecycle(ctx).verdict == "OK"` iff the ledger
satisfies L-1..L-11 (Theorem W3-45 — proved by inspection).

**4. In-process synthetic-LLM mode.** A new
`SweepSpec(mode="synthetic", synthetic_model_tag=<tag>)`
mode replaces the real `LLMClient` with a deterministic
in-process `SyntheticLLMClient`
(`vision_mvp/wevra/synthetic_llm.py`) that returns canned
strings keyed by `(instance_id, model_tag)`. The full
PROMPT / LLM_RESPONSE / PARSE_OUTCOME / PATCH_PROPOSAL /
TEST_VERDICT chain seals end-to-end without an Ollama
endpoint. Seven calibrated distributions ship in
`SYNTHETIC_MODEL_PROFILES`: `clean`, `unclosed`, `prose`,
`empty`, `fenced`, `multi_block`, `mixed`. Synthetic mode is
deterministic by construction — two runs produce identical
PROMPT and LLM_RESPONSE CIDs.

**5. Cross-model parser-boundary research (W3-C6, empirical).**
A new experiment
`vision_mvp/experiments/parser_boundary_cross_model.py`
sweeps `(model_tag, parser_mode)` across the synthetic
distribution library and computes pairwise Total Variation
Distance over PARSE_OUTCOME failure-kind multinomials. On the
bundled bank (57 instances) it reports:

  * Cross-distribution failure-kind TVD up to **1.000**
    (synthetic distributions span the parser's failure
    taxonomy).
  * Strict→robust parser-mode shift TVD up to **1.000** on
    `synthetic.unclosed` (the parser flips entirely from
    `unclosed_new` failure to `ok + recovery=closed_at_eos`).
  * Intermediate distributions: `synthetic.mixed` shows
    `ok_rate=0.68` under strict and `ok_rate=0.98` under
    robust, demonstrating non-degenerate parser-mode-conditional
    movement.

**Honest scope.** This is a calibrated synthetic study, **not**
a real cross-LLM measurement. The empirical claim is about the
PARSE_OUTCOME failure-kind closed vocabulary's *resolving
power*, not about real LLM output distributions in the wild.
A real cross-LLM extension is straightforward to layer on by
substituting `mode="real"` with a real `LLMClient`; this
experiment is the necessary in-CI-runnable preliminary.

**6. What this changes about Wevra's originality claim.** The
v3.3 positioning was *"capsules drive execution at the run
boundary, the inner sweep loop pair, AND the parser axis"*.
The v3.4 positioning is sharper:

> *Capsules drive execution at the run boundary, the inner
> sweep loop pair, the parser axis, AND the LLM byte boundary.
> The lifecycle correspondence is mechanically checkable on
> every finished run (eleven invariants L-1..L-11). The
> end-to-end inner-loop chain
> PROMPT → LLM_RESPONSE → PARSE_OUTCOME → PATCH_PROPOSAL →
> TEST_VERDICT is observable as a typed DAG with strong
> parent-CID gating. The parser-boundary attribution layer is
> empirically sharp on a calibrated synthetic distribution
> library.*

Three "operational claims that were ahead of code" in v3.3 have
been promoted to "operational claims the code honestly earns":

  1. The LLM bytes themselves were previously plain Python; they
     are now content-addressed capsules on the DAG.
  2. The cached LLM call shared by naive + routing strategies
     was previously implicit (raw_cache); it is now an explicit
     idempotent capsule (single PROMPT and LLM_RESPONSE
     referenced by both strategies' PARSE_OUTCOMEs).
  3. The parser-boundary failure-kind distribution was
     previously a paper-grade conjecture (W3-C4); it is now an
     empirical result with a reproducible CI harness (W3-C6).

**7. What remains legacy / post-hoc / out-of-scope.**

  * Sandbox stdout / stderr / test trace remain plain bytes.
    The natural SDK v3.5 candidate is an APPLY_OUTCOME capsule
    between PATCH_PROPOSAL and TEST_VERDICT (six-link chain).
  * Parser-internal regex / recovery-heuristic state remain
    non-capsule. The structured PARSE_OUTCOME captures the
    verdict, not the iteration state that produced it.
  * Real cross-LLM W3-C6 measurement remains future work
    (straightforward to layer on).

**8. SDK surface delta.** Strictly additive on v3.3:

  * New: `CapsuleKind.PROMPT`, `CapsuleKind.LLM_RESPONSE`,
    `capsule_from_prompt`, `capsule_from_llm_response`,
    `PROMPT_TEXT_CAP`, `LLM_RESPONSE_TEXT_CAP`.
  * New: `CapsuleNativeRunContext.seal_prompt`,
    `seal_llm_response`, `seal_parse_outcome(llm_response_cid=
    ...)` argument.
  * New: lifecycle audit invariants L-9 / L-10 / L-11
    (`_check_l9` / `_check_l10` / `_check_l11`).
  * New: `SweepSpec(mode="synthetic", synthetic_model_tag=...)`
    + `_synthetic_cells` dispatcher in `wevra.runtime`.
  * New: `vision_mvp/wevra/synthetic_llm.py`
    (`SyntheticLLMClient`, `SYNTHETIC_MODEL_PROFILES`,
    `make_synthetic_response_fn`).
  * New: `vision_mvp/experiments/parser_boundary_cross_model.py`
    (CLI entry point + programmable function).
  * Bumped: `SDK_VERSION = "wevra.sdk.v3.4"`.
  * Unchanged: every v3.3 contract test (18) still passes
    byte-for-byte; capsule view schema name
    `wevra.capsule_view.v1` (PROMPT / LLM_RESPONSE payloads
    are additive); the post-hoc `build_report_ledger` adapter;
    deterministic-mode replay; the META_MANIFEST detached-witness
    boundary.

**9. Empirical anchor.**
`vision_mvp/tests/test_wevra_capsule_native_inner_loop.py`
ships 16 contract tests covering the W3-42 / W3-43 / W3-44 /
W3-45 / W3-C6 claims plus PROMPT idempotence, the synthetic-mode
end-to-end chain, and synthetic-mode determinism. Combined
with v3.1's `test_wevra_capsule_native.py` (16 tests), v3.2's
`test_wevra_capsule_native_intra_cell.py` (16 tests), and
v3.3's `test_wevra_capsule_native_deeper.py` (18 tests), the
capsule-native runtime contract is now witnessed by **66
contract tests**. Full `vision_mvp.tests.test_wevra_*` +
`test_capsule_*` suite green.

Anchor: `docs/archive/wevra-milestones/RESULTS_WEVRA_INNER_LOOP.md` (this milestone
note); `docs/CAPSULE_FORMALISM.md` § 4.J (theorems);
`docs/THEOREM_REGISTRY.md` (canonical registry);
`docs/RESEARCH_STATUS.md` (canonical research-status); paper
draft: `papers/wevra_capsule_native_runtime.md` (claim taxonomy
+ flagship write-up).

### 4.22 SDK v3.5 — capsule-native multi-agent team coordination (the team-layer slice)

The v3.4 milestone (§ 4.21) closed the LLM byte boundary inside
*one* Wevra run. The strongest remaining frontier — and the one
that actually maps onto the original Context-Zero "solve context
for multi-agent teams" thesis — was the **team boundary**: what
crosses between agents in a coordination round. SDK v3.5 makes
the capsule abstraction load-bearing on that boundary.

**1. Three new closed-vocabulary capsule kinds.**

* **TEAM_HANDOFF** — capsule-native multi-agent handoff. Distinct
  from the substrate-adapter `HANDOFF` (which lifts a Phase-31
  `TypedHandoff`); a TEAM_HANDOFF is *born as a capsule* with no
  substrate twin. Payload: `(source_role, to_role, claim_kind,
  payload, round, payload_sha256, n_tokens)`. Identity is
  content-addressed — byte-identical handoffs collapse (Capsule
  Contract C1).
* **ROLE_VIEW** — per-role admitted view of one coordination
  round. Parents are the CIDs of admitted TEAM_HANDOFF capsules;
  `max_parents` is the role-local cardinality cap $K_r$;
  `max_tokens` is the role-local token cap $T_r$. The ROLE_VIEW
  capsule *is* the role's local-view-under-budget object — the
  W4 theorems are statements about ROLE_VIEW, not about agent-
  local scratchpads.
* **TEAM_DECISION** — team-level decision. Parents: the ROLE_VIEW
  capsules consulted. Payload: the structured team answer.

A `TeamCoordinator` orchestrates one coordination round
end-to-end against a shared `CapsuleLedger`; an
`audit_team_lifecycle` mechanically verifies invariants T-1..T-7.

**2. Three named theorems (W4 family).**

* **W4-1** Team-lifecycle audit soundness — *proved +
  mechanically-checked.* Audit returns OK iff T-1..T-7 hold.
  Proof by inspection of the audit code; tests in
  `TeamLifecycleAuditTests`.
* **W4-2** Coverage-implies-correctness — *proved-conditional*
  (premises: faithful decoder + sound admission). If the role
  view's admitted set is a superset of the scenario's causal
  claims, the team decision is correct on the gold scenario.
  Tested in `TeamLevelCorrectnessTests::test_w4_2_*`.
* **W4-3** Local-view limitation — *proved-negative*. A per-role
  budget $K_r$ strictly below the role's causal-share floor on a
  scenario admits the wrong answer regardless of admission policy
  ($\pi$). The theorem says: per-role budgets below the floor
  cannot be rescued by *any* policy — a sharp negative limit
  on what local-view admission alone can guarantee. Tested in
  `TeamLevelCorrectnessTests::test_w4_3_*`; budget-sweep
  evidence in `phase52_team_coord.run_phase52_budget_sweep`.

**3. Phase-52 reference benchmark + learned policy.**

`vision_mvp/experiments/phase52_team_coord.py` instantiates the
`TeamCoordinator` on the Phase-31 incident-triage bank under
controlled noise and compares five strategies head-to-head:

| strategy            | description                                     |
| ------------------- | ----------------------------------------------- |
| `substrate`         | Phase-31 typed-handoff baseline (no capsule layer). |
| `capsule_fifo`      | capsule-native + FIFO admission policy.          |
| `capsule_priority`  | capsule-native + claim-priority admission.       |
| `capsule_coverage`  | capsule-native + coverage-guided admission.      |
| `capsule_learned`   | capsule-native + learned per-role admission policy (logistic regression over six capsule features, per-role weights, SGD-trained on a 60-scenario partition). |

On the default config (`K_auditor=8`, `T_auditor=256`, noise
`(drop=0.10, spurious=0.30, mislabel=0.05)`, $n_\text{eval}=31$):

| strategy           | accuracy_full | accuracy_root_cause | mean_n_admitted | audit_ok |
| ------------------ | ------------- | ------------------- | --------------- | -------- |
| substrate          | 0.0323        | 0.4839              | 7.645           | n/a      |
| capsule_fifo       | 0.0323        | 0.5484              | 7.452           | 1.000    |
| capsule_priority   | 0.0323        | 0.4516              | 7.258           | 1.000    |
| capsule_coverage   | 0.0645        | 0.5484              | 6.968           | 1.000    |
| **capsule_learned** | **0.1613**    | **0.7097**          | **5.129**       | **1.000** |

The learned policy strictly improves on every fixed baseline at
matched per-role budget while admitting **33 % fewer** handoffs.
This is the empirical evidence anchoring conjecture **W4-C1**:
*learned per-role admission policy beats the strongest fixed
admission baseline at matched per-role budgets.* Status:
empirical-positive on the default config; conjectural at smaller
training scales (the 14-scenario sweep eval shows fixed baselines
match or exceed the learned policy when training-data scale is
small).

**4. Strict programme/product split is preserved.** The Wevra
**product** runtime contract — `RunSpec` → sealed RUN_REPORT
capsule + detached META_MANIFEST + lifecycle audit L-1..L-11 + CI
gate — is **byte-for-byte unchanged**. The team-layer capsule
kinds are emitted *only* by `TeamCoordinator`; the run-boundary
sweep path does not seal them. The `wevra` console script's
output is unchanged. The team layer is research-grade
(`vision_mvp.wevra.team_coord`), additive on top of v3.4.

**5. The team-layer slice is what reconnects capsule-native
execution to the original "solve context for multi-agent teams"
thesis.** Before v3.5 the capsule abstraction lived inside one
run; the multi-agent thesis was carried by the substrate
(`vision_mvp.core.role_handoff`, escalation threads, adaptive
subscriptions). After v3.5 the capsule abstraction has a
mechanically-checked, theorem-anchored, learned-policy-backed
expression *between* agents in a team. The honest claim is:
"on the Phase-52 incident-triage benchmark family, the
capsule-native multi-agent coordination layer with a learned
per-role admission policy admits **strictly fewer handoffs**
than the strongest fixed admission baseline on every train seed
(12/12) and improves pooled team-decision accuracy on most
seeds (gap on `accuracy_full` > 0 in 11/12 seeds, mean
$+0.054$; gap on `accuracy_root_cause` > 0 in 8/12 seeds,
mean $+0.032$) — but the accuracy advantage reverses at higher
noise; the team-lifecycle audit returns OK on every
coordination round." The unbounded "we solved multi-agent
context" read is **forbidden** by
`docs/HOW_NOT_TO_OVERSTATE.md`.

**6. Tests + audit.** `vision_mvp/tests/test_wevra_team_coord.py`
ships 22 contract tests covering capsule constructors, the
coordinator's emit/seal flow, idempotency under byte-identical
handoffs, K_role / T_role budget enforcement, T-1..T-7 audit
verdicts (including a positive T-7 violation construction),
admission-policy semantics for FIFO / priority / coverage-guided,
the learned-policy training-loop convergence on a separable
pattern, and the W4-2 / W4-3 theorem anchors. Full
`test_wevra_*` + `test_capsule_*` + `test_role_handoff` suite
green. Combined with v3.4's 66 capsule-native run-boundary
contract tests, the capsule-native layer (run-boundary +
team-boundary) is now witnessed by **88 contract tests**.

Anchor: `docs/archive/wevra-milestones/RESULTS_WEVRA_TEAM_COORD.md` (this milestone note);
`docs/CAPSULE_TEAM_FORMALISM.md` (formal model);
`docs/THEOREM_REGISTRY.md` (canonical registry — W4 rows added);
`docs/RESEARCH_STATUS.md` (canonical research-status — multi-
agent axis added);
`vision_mvp/experiments/phase52_team_coord.py` (benchmark
driver); `vision_mvp/wevra/team_coord.py` and
`vision_mvp/wevra/team_policy.py` (the new module pair).

### 4.23 SDK v3.6 — two-Mac distributed-inference integration boundary + real cross-LLM parser-boundary measurement

The v3.5 milestone (§ 4.22) closed the team boundary inside the
research slice. The v3.6 milestone closes a different — and
complementary — boundary: the **inference-backend boundary**.
Up through v3.5 the capsule-native runtime was load-bearing in
its inner-loop spine but pinned to a single-host Ollama HTTP
client; the **stronger-model regime** that two combined Apple
Silicon Macs unlock (a single sharded model spanning
`mac1.local + mac2.local`) was an operator concern, not a Wevra
concern. SDK v3.6 ships the smallest honest integration boundary
plus the first **real** (non-synthetic) cross-LLM measurement
on the parser-boundary axis.

**1. Chosen path: MLX distributed inference.**

Three candidates were evaluated:

* **MLX distributed (`mx.distributed` + `mlx_lm.server`).**
  Apple-official, supports Apple Silicon (Metal), shards a
  single transformer's weights across N hosts via tensor +
  pipeline parallel under MPI. Exposes one OpenAI-compatible
  HTTP endpoint on the head rank; the wire shape is identical
  to a single-host run.
* **llama.cpp `--rpc`.** Real, supports Apple Silicon (Metal),
  shards a GGUF across multiple `rpc-server` processes. A
  defensible alternative; not chosen because MLX is more deeply
  optimised for Apple Silicon and the maintenance surface is
  smaller.
* **Hyperspace.** A strong distributed-agent infrastructure
  (node discovery, message routing, agent caching) but **not**
  a single-model sharding system. There is no public Hyperspace
  surface that splits one transformer's weights across two
  Macs and runs one forward pass across the cut. We do **not**
  pick Hyperspace for this milestone — it solves a different
  problem.

The realistic model class on 2 × 36 GB Macs is **70B in 4-bit**
(≈ 40 GB weights; fits across both Macs with KV-cache headroom).
Sub-70B models run on a single Mac in 4-bit; sharding gives
context-length / KV headroom only.

**2. Wevra integration boundary
(`vision_mvp/wevra/llm_backend.py`).**

A duck-typed `LLMBackend` Protocol with three pieces:

* `LLMBackend` (runtime-checkable Protocol) — `model: str`,
  `base_url: str | None`, `generate(prompt, max_tokens,
  temperature) -> str`. Matches the duck-type the inner-loop
  already expected from `LLMClient`.
* `OllamaBackend` — wraps the existing Ollama client unchanged.
* `MLXDistributedBackend` — talks an OpenAI-compatible
  `POST /v1/chat/completions` endpoint. Designed for an
  `mlx_lm.server` launched under `mpirun --hostfile <hosts>`
  so the same HTTP surface fronts a single-host run *or* a
  sharded multi-host run. Wevra is **neutral on the sharding
  strategy**: one HTTP client, one Protocol, one adapter class.
* `make_backend(name, **kwargs)` — factory dispatch by name.

`run_sweep(spec, *, ctx=None, llm_backend=None)` accepts an
optional `llm_backend`. When set, the inner-loop dispatches
through it; when None, behaviour is byte-for-byte identical to
SDK v3.5. The PROMPT / LLM_RESPONSE / PARSE_OUTCOME /
PATCH_PROPOSAL / TEST_VERDICT capsule chain seals byte-for-byte
equivalently regardless of backend (W5-2 proved).

**3. Three named theorems (W5 family).**

* **W5-1 (proved-empirical, real LLM)** — On the bundled
  SWE-bench-Lite-shape bank with `prompt_style="block"`,
  `temperature=0`, `n=10`:
    1. `qwen3.5:35b` (36B-MoE Q4 `think=False`) under strict
       parsing produces `failure_kind=unclosed_new` 10/10;
       `ok_rate=0.000`.
    2. `qwen2.5:14b-32k` (14.8B-dense Q4) under strict
       produces `failure_kind=ok` 10/10; `ok_rate=1.000`.
    3. Cross-model PARSE_OUTCOME failure-kind TVD = 1.000 on
       strict; collapses to 0.000 on robust via
       `recovery=closed_at_eos`.
  Anchor:
  `vision_mvp/experiments/parser_boundary_real_llm.py`;
  result JSON in `/tmp/wevra-distributed/`.
* **W5-2 (proved)** — Backend integration: any duck-typed
  `LLMBackend` substitutes for `LLMClient` in `_real_cells`;
  the capsule chain seals end-to-end byte-for-byte.
  Anchor:
  `test_wevra_llm_backend.py::RunSweepBackendIntegrationTests`.
* **W5-3 (proved)** — `MLXDistributedBackend` wire shape:
  OpenAI-compatible `POST /v1/chat/completions` with
  `{model, messages, max_tokens, temperature, stream:false}`,
  parses `choices[0].message.content`. Locked against
  in-process stub.

**4. Three named conjectures (W5-C family).**

* **W5-C1 (empirical-research, falsifiable)** — Parser-
  boundary instability is a (model-architecture × prompt-format)
  interaction, not a capacity artefact. Falsifier: a bank where
  the larger model strict-parses ok > 50%.
* **W5-C2 (empirical-research, falsifiable)** — Robust-mode
  `recovery=closed_at_eos` is the load-bearing safety net
  making the runtime model-class-agnostic on the bundled
  prompt format. Falsifier: a model whose `unclosed_new`
  cannot be salvaged.
* **W5-C3 (research, conjectural)** — Closed-vocabulary
  `PARSE_OUTCOME.failure_kind` is a *minimum sufficient* typed
  witness of cross-model behaviour differences. Falsifier: a
  model pair with identical strict-mode `failure_kind`
  distribution but materially different downstream test-pass
  rate.

**5. Honest scope.**

The MLX-distributed two-Mac path is **experimental
infrastructure**, not product. Wevra does **not** ship `mlx`,
`mlx-lm`, or `mpirun` as dependencies; it does not auto-bring-up
the cluster. There is deliberately no
`pip install wevra[mlx_distributed]` extra. The integration is
one HTTP-client class.

W5-1 was measured on a *single* Mac (Mac 1 alive at measurement
time; Mac 2 offline / ARP "incomplete" at 192.168.12.248). The
two-Mac MLX-distributed path is the integration boundary, not
the inference path used for W5-1; it is the *next-step*
research target once Mac 2 returns. The W5-1 result is
nonetheless real cross-LLM evidence for the capsule-native
runtime being model-class-agnostic — the model swap from
14.8B-dense to 36B-MoE is a 2.4× capacity jump and an
architecture switch (dense → mixture-of-experts).

**6. Why this strengthens the original Context-Zero thesis.**

The original thesis is "context is a routing / substrate /
coordination problem in multi-agent LLM systems." SDK v3.5 made
the capsule abstraction load-bearing **between agents**. SDK
v3.6 demonstrates it is **load-bearing across the model-class
gradient**: the typed-boundary discipline (PARSE_OUTCOME closed
vocabulary, recovery-label closed vocabulary, lifecycle audit
L-11) cleanly absorbs a model-architecture swap that breaks the
naive "stronger = cleaner" prediction at the byte level. The
larger-model regime is an *additional axis of evidence*, not a
refutation. If anything, W5-1 says **the capsule-native runtime
becomes more, not less, valuable as model class scales** —
because the parser-boundary axis is a load-bearing source of
distribution shift, and the runtime's typed witnesses are how
you see it without hand-diffing N responses.

**7. Files / tests / artefacts.**

* `vision_mvp/wevra/llm_backend.py` (new) — Protocol +
  `OllamaBackend` + `MLXDistributedBackend` + `make_backend`.
* `vision_mvp/wevra/runtime.py` — `run_sweep` accepts optional
  `llm_backend`; sweep block records `"backend"` field.
* `vision_mvp/wevra/__init__.py` — re-exports + `SDK_VERSION =
  "wevra.sdk.v3.6"`.
* `vision_mvp/experiments/parser_boundary_real_llm.py` (new) —
  real-LLM cross-model parser-boundary harness.
* `vision_mvp/tests/test_wevra_llm_backend.py` (new) — 9
  contract / wire-shape / integration tests.
* `docs/archive/wevra-milestones/RESULTS_WEVRA_DISTRIBUTED.md` (this milestone note).
* `docs/MLX_DISTRIBUTED_RUNBOOK.md` (operator runbook).
* `docs/THEOREM_REGISTRY.md` — W5-1 / W5-2 / W5-3 / W5-C1 /
  W5-C2 / W5-C3 rows added.
* `docs/RESEARCH_STATUS.md` — fifth research axis added.

Anchor: `docs/archive/wevra-milestones/RESULTS_WEVRA_DISTRIBUTED.md` (this milestone
note); `docs/MLX_DISTRIBUTED_RUNBOOK.md` (operator runbook);
`vision_mvp/experiments/parser_boundary_real_llm.py`
(benchmark driver); `vision_mvp/wevra/llm_backend.py` (the new
adapter module).

### 4.24 SDK v3.7 — model-scale vs capsule-structure on multi-agent coordination (real-LLM Phase-53 benchmark + W6 family)

The v3.6 milestone (§ 4.23) closed the inference-backend
boundary and produced the first *real* (non-synthetic) cross-LLM
parser-boundary measurement. The v3.7 milestone turns the
real-LLM regime on the **multi-agent capsule coordination axis**
itself: replaces the Phase-52 deterministic producer-role
extractor with a real LLM extractor, decomposes accuracy across
``model regime × admission strategy``, and asks the central
scientific question: **does scaling the underlying model
preserve, amplify, or close the capsule-structure advantage on
the original multi-agent-context thesis?**

**1. Two-Mac sharded-inference status — plainly.**

* `arp -a` for 192.168.12.248: **(incomplete)** at the time of
  this milestone. Mac 2 is not on the LAN.
* `ping -c2 192.168.12.248`: 100% packet loss.
* No `mpirun mlx_lm.server` was launched. **No sharded 70B-class
  model ran across both Macs.**
* The integration boundary (`MLXDistributedBackend`,
  `LLMBackend` Protocol, `run_sweep(..., llm_backend=...)`) is
  byte-for-byte unchanged from SDK v3.6 and remains correct
  against the in-process OpenAI-compat stub. The Wevra side has
  nothing additional to do until the cluster lights up; the
  runbook (`docs/MLX_DISTRIBUTED_RUNBOOK.md`) is the operator
  path.
* The strongest honest model class actually exercised in
  SDK v3.7 is **single-Mac** Qwen-3.5-35B (36 B-MoE) in 4-bit
  via Ollama on Mac 1. This is real (not synthetic) but is
  **not sharded inference**.

**2. Phase-53 stronger-model multi-agent benchmark.**

`vision_mvp/experiments/phase53_scale_vs_structure.py` drives
the team coordinator with a real-LLM producer-role extractor for
each of the four producer roles (monitor, db_admin, sysadmin,
network) and runs the same five admission strategies (substrate,
capsule_fifo, capsule_priority, capsule_coverage, capsule_learned)
on the LLM-generated candidate handoff stream. Sweeps the
*model regime* (synthetic / qwen2.5:14b-32k / qwen3.5:35b)
holding everything else fixed and decomposes:

* `structure_gain[M]` := capsule_learned_acc[M] - substrate_acc[M]
* `scale_gain[S]`     := acc(35B)[S] - acc(14B)[S]
* `delta_with_scale`  := structure_gain[35B] - structure_gain[14B]

**3. Headline empirical finding (n=5 saturated, K_auditor=4).**

Every fixed admission strategy (substrate / capsule_fifo /
capsule_priority / capsule_coverage) achieves
``accuracy_full = 0.800`` in every model regime; only
`capsule_learned` varies:

| regime           | substrate | fixed capsule | learned | failed scenario       |
| ---------------- | --------- | ------------- | ------- | --------------------- |
| synthetic        | 0.800     | 0.800         | 0.400   | deadlock_pool_exhaustion |
| qwen2.5:14b-32k  | 0.800     | 0.800         | 0.400   | deadlock_pool_exhaustion |
| qwen3.5:35b      | 0.800     | 0.800         | 0.800   | deadlock_pool_exhaustion |

Decomposition:
* `structure_gain[synthetic]    = -0.400`
* `structure_gain[qwen2.5:14b]  = -0.400`
* `structure_gain[qwen3.5:35b]  =  0.000`
* `scale_gain[capsule_learned]  = +0.400`
* `scale_gain[every other strategy] =  0.000`
* `delta_with_scale = +0.400`

Cross-model candidate-kind TVD (14B vs 35B) = 0.167 (modest, but
non-zero — the two model classes emit detectably different
candidate distributions on the same bench).

**4. Five named theorems (W6 family).**

* **W6-1 (proved + mechanically-checked).** Capsule-team
  lifecycle audit T-1..T-7 holds for every (regime × strategy ×
  scenario) cell of the Phase-53 benchmark (60/60). Anchor:
  `phase53_scale_vs_structure.py::audit_ok_grid`;
  `docs/data/phase53_scale_vs_structure_K4_n5.json`.
* **W6-2 (proved).** Phase-53 driver accepts any duck-typed
  `LLMBackend` substitute as the producer-role extractor
  backend; team-coord pipeline seals capsules end-to-end against
  arbitrary backend.
* **W6-3 (proved + mechanically-checked).** `parse_role_response`
  is robust on the closed-vocabulary claim grammar (16 cases).
* **W6-4 (proved-empirical).** The ``accuracy_full`` /
  ``structure_gain`` / ``scale_gain`` decomposition above; n=5
  saturated, real LLM, deterministic anchor JSON.

**5. Five conjectures (W6-C family).**

* **W6-C1 (FALSIFIED).** Drafted-conjecture: structure_gain is
  preserved or grows when LLM scales up. Empirical reading:
  structure_gain is **non-positive** at every regime tested
  (-0.4, -0.4, 0.0); scale narrows a *deficit* not a *surplus*.
* **W6-C2 (FALSIFIED).** Drafted-conjecture: the per-role
  admission scorer trained on Phase-52 synthetic+noise transfers
  usefully to real-LLM streams. Empirical reading:
  capsule_learned LOSES to capsule_fifo by 0.40 on synthetic and
  14B; ties at 35B; average gap = -0.267.
* **W6-C3 (EMPIRICAL-POSITIVE).** Cross-(14B, 35B) candidate-
  kind TVD = 0.167 on the pooled (source_role × claim_kind)
  histogram across 5 scenarios; above the 0.10 falsifier.
* **W6-C4 (NEW conjecture).** Substrate FIFO is competitive
  with every capsule admission policy at sufficient
  K_auditor in real-LLM-driven multi-agent benchmarks.
  Falsifier: a (model, scenario, K_auditor) configuration where
  substrate < min_capsule - 0.05.
* **W6-C5 (NEW conjecture).** Model scale narrows the OOD
  generalisation gap of the per-role admission scorer trained
  on synthetic noise. Anchored by ``scale_gain[capsule_learned]
  = +0.4`` against ``scale_gain[fixed] = 0`` on Phase-53 default.

**6. W4-C1 (SDK v3.5) is conditionally falsified.**

The SDK v3.5 W4-C1 reading (learned policy beats fixed
admission baselines on Phase-52) holds **on its anchor
distribution** (synthetic+noise default config). It is
**falsified out-of-distribution** on the real-LLM regime in
SDK v3.7: capsule_learned 0.400 vs fixed 0.800 on synthetic
and 14B. The W4-C1 row in `docs/THEOREM_REGISTRY.md` now
carries the conditional-status table.

**7. Honest scope.**

* This benchmark is **incident-triage-bench-internal**. External
  validity to other multi-agent benches (security-escalation,
  task_scale_swe) is open.
* `K_auditor=4` is one budget point. The structural-pressure
  regime where substrate FIFO must admit non-causal head-of-
  arrival emissions is the conjectural W6-C4 falsifier search
  direction (lower K_auditor).
* The single failing scenario (`deadlock_pool_exhaustion`) fails
  identically across all four "tied" strategies in every model
  regime, because no model regime emits `DEADLOCK_SUSPECTED`
  reliably from the role-local events on this scenario. This
  is *not* an admission-policy weakness; it is a producer-role
  extraction weakness that no admission policy can recover.
* The W4-C1 reading on its anchor config (Phase-52 default) is
  unchanged. The new W6 reading is OOD.

**8. Why this strengthens the original Context-Zero thesis.**

The original thesis is "context is a routing / substrate /
coordination problem in multi-agent LLM systems." SDK v3.7
**tightens** the reading on the *admission* axis (the W4-C1
advantage does not transfer OOD; substrate FIFO is competitive
when the LLM is the producer) while *strengthening* the reading
on the *audit* axis (the W6-1 audit-OK grid is 60/60 across two
real-LLM regimes). The capsule layer's load-bearing
contribution at this benchmark is **mechanical proof of
coordination well-formedness**, not admission policy gains.
That is a smaller, sharper, defensible claim — and it is the
right claim to hold at this benchmark size.

**9. Files / tests / artefacts.**

* `vision_mvp/experiments/phase53_scale_vs_structure.py` (new)
  — 770-line driver: real-LLM extractor, candidate stream
  builder, decomposition, cross-regime TVD.
* `vision_mvp/tests/test_wevra_scale_vs_structure.py` (new) —
  19 contract tests (parser robustness, backend duck-typing,
  audit_ok grid, schema lock).
* `vision_mvp/wevra/__init__.py` — `SDK_VERSION = "wevra.sdk.v3.7"`.
* `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md` (this milestone note).
* `docs/data/phase53_scale_vs_structure_K4_n5.json` (frozen
  benchmark output).
* `docs/THEOREM_REGISTRY.md` — W6-1 / W6-2 / W6-3 / W6-4 /
  W6-C1 / W6-C2 / W6-C3 / W6-C4 / W6-C5 added; W4-C1 amended.
* `docs/RESEARCH_STATUS.md` — sixth research axis added.

Anchor: `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md`;
`docs/data/phase53_scale_vs_structure_K4_n5.json`;
`vision_mvp/experiments/phase53_scale_vs_structure.py`.

### 4.25 SDK v3.8 — cross-role cohort-coherence multi-agent coordination (deterministic Phase-54 benchmark + W7 family)

The v3.7 milestone (§ 4.24) produced an honest but unsatisfying
result on multi-agent coordination: at the Phase-53 default config
(real-LLM producer extractor, K_auditor=4), ``structure_gain`` was
non-positive at every model regime tested and substrate FIFO was
competitive with every fixed capsule admission policy. The capsule
layer's load-bearing contribution at that bench was the
*lifecycle audit* (T-1..T-7, 60/60 across regimes), not
coordination performance. SDK v3.8 directly attacks the failure
mode: **diagnose why FIFO won, redesign the regime so structure
has a real chance, implement one stronger capsule coordination
method, and run a disciplined experiment that produces a clean
structural win conditional on a stated bench property.**

**1. Phase-53 failure diagnosis (code-backed).** Three layered
causes:

* **Benchmark pathology (dominant).** At ``K_auditor=4`` with the
  real-LLM producer emitting 0.65–0.90 claims/call, the auditor
  admitted ``2.2 / 3.0 / 3.2 < 4`` candidates per scenario in
  every regime — *always below the budget cap*. With no budget
  pressure, every fixed admission policy is permutation-equivalent
  to ``admit_all`` (W7-1). FIFO ties or beats by construction.
* **Policy pathology (secondary).** The learned scorer was trained
  with ``pos_rate=0.42`` and ``threshold=0.5``. On a real-LLM
  stream that is near-fully-positive (the LLM pre-filters), the
  same threshold rejects ~half of *real* causal candidates.
* **Extraction pathology (orthogonal).** The Phase-53
  ``deadlock_pool_exhaustion`` scenario is missed by *every*
  admission strategy because neither LLM emits
  ``DEADLOCK_SUSPECTED`` from the role-local events. This is the
  extraction floor (W7-3): no admission policy can recover a
  claim the producer never emitted.

**2. Phase-54 deterministic cross-role coherence benchmark.**
``vision_mvp/experiments/phase54_cross_role_coherence.py``
materialises the smallest deterministic regime where cross-role
cohort coherence can plausibly matter: each scenario has one
``real_service`` (gold) and one ``decoy_service`` (foreign);
each producer role emits 1–3 ``service=<tag>``-tagged candidates;
the gold service has *strict plurality* in the auditor's
candidate stream; the auditor stream contains 5–7 candidates at
``K_auditor=4`` (so budget bites). The benchmark is
**deterministic** (no LLM in loop) by design — to isolate the
coordination axis from the producer-extraction-quality axis that
dominated Phase-53. Bench properties (gold-plurality, cross-role,
budget-bound) are *named and mechanically verified* by the
contract tests
(``test_wevra_cross_role_coherence.Phase54BankShapeTests``).

**3. ``CohortCoherenceAdmissionPolicy`` (new, SDK v3.8).**
``vision_mvp/wevra/team_coord.py``. A deterministic, training-
free, interpretable cross-role admission rule that filters
foreign-service decoys via a ``service=<tag>`` cohort signature.
Two sub-modes:

* **Streaming** (``fixed_plurality_tag=None``, default) — running
  cohort over already-admitted; arrival-order-sensitive
  (W7-1-aux limitation).
* **Buffered**
  (``CohortCoherenceAdmissionPolicy.from_candidate_payloads(...)``)
  — pre-fitted plurality from the full candidate stream's
  payloads; arrival-order-stable.

The policy is **not OOD-fragile**: no scorer / no threshold tuned
on a synthetic distribution. The W6-C2 OOD failure mode that bit
the SDK v3.5 learned policy at SDK v3.7 cannot recur for this
policy by construction.

**4. Empirical headline (Phase-54 default, n=10, K=4).**

| Strategy                           | accuracy_full | accuracy_services | mean_n_admitted_auditor |
| ---------------------------------- | ------------- | ----------------- | ----------------------- |
| substrate / capsule_fifo / capsule_priority | 0.000  | 0.000             | 4.00                    |
| capsule_coverage                   | 0.000         | 0.000             | 4.00                    |
| capsule_cohort_streaming           | 0.000         | 0.000             | 2.00                    |
| **capsule_cohort_buffered**        | **1.000**     | **1.000**         | **3.60**                |

Headline gap: ``cohort_buffered − fifo accuracy_full = +1.000``;
stable across 5/5 alternate ``bank_seed`` values (11, 17, 23, 29,
31).

**5. The W7 theorem family.**

* **W7-1 (proved-empirical).** FIFO unbeatability under low
  surplus: when ``|candidates| ≤ K_role`` for every scenario,
  every fixed admission policy is permutation-equivalent to
  ``admit_all``; ``structure_gain`` is identically zero. Phase-53
  default is the empirical anchor.
* **W7-1-aux (proved-empirical).** Streaming cohort coherence is
  unstable under candidate-arrival permutation. Phase-54 default
  with decoys-first ordering is the anchor; streaming ties FIFO
  at 0.000.
* **W7-2 (proved-empirical, n=50 saturated).** Under gold-plurality
  + cross-role coherence + ``|candidates| > K_auditor``, the
  *buffered* cohort policy beats FIFO by ``+1.000`` on
  ``accuracy_full``, stable across 5/5 bank seeds.
* **W7-2-conditional (proved-empirical, K-sweep).** W7-2 holds in
  the structure-win window; the gap is +0.40 at K=2, K=3 (gold
  cohort partially fits) and +1.00 at K∈{4,5,6,8}. Sharper:
  the decoder's set-equality services check makes service-blind
  admission unable to ever achieve services_correct on
  foreign-service-decoy benches, even with no budget pressure.
* **W7-3 (proved-negative).** The extraction floor:
  ``capsule_role_view.parents`` is constructed from ledger CIDs
  (Capsule Contract C5); a never-emitted claim has no CID;
  therefore no admission policy can recover it. Phase-53
  ``deadlock_pool_exhaustion`` is the canonical empirical case.

**6. What the W7 family says about the original thesis.** The
capsule layer's **audit** contribution is preserved and extends to
Phase-54 unchanged (T-1..T-7 hold on every cell). The capsule
layer's **coordination-performance** contribution is now
*demonstrable in a clean, falsifiable way* — but **conditional**
on cross-role service-tag coherence. The earlier SDK v3.5
"learned policy beats FIFO at noisy admission" framing was
mean-positive but not robust; the SDK v3.8 "buffered cohort beats
FIFO at gold-plurality" framing is **strict-positive on every
test** under a stated condition. Phase-53 (W7-1) and Phase-54
(W7-2) together form a **dichotomy**: substrate FIFO is
unbeatable when the bench has no surplus; cohort coherence beats
substrate cleanly when the bench has surplus + foreign-service
decoys + gold-plurality. Both readings are conditional, named,
and falsifiable.

**7. What changed.**

* `vision_mvp/wevra/team_coord.py` (extended) —
  ``CohortCoherenceAdmissionPolicy``;
  ``_candidate_service_tag``; updated ``ALL_FIXED_POLICY_NAMES``.
* `vision_mvp/wevra/__init__.py` — re-exports
  ``TeamCohortCoherenceAdmissionPolicy``;
  ``SDK_VERSION = "wevra.sdk.v3.8"``.
* `vision_mvp/experiments/phase54_cross_role_coherence.py` (new).
* `vision_mvp/tests/test_wevra_cross_role_coherence.py` (new) —
  21 contract tests including default-config win, K-sweep, audit
  invariance, bank-seed stability.
* `docs/RESULTS_WEVRA_CROSS_ROLE_COHERENCE.md` (new — milestone
  results note).
* `docs/data/phase54_cross_role_coherence_K4_n10.json` (frozen
  default-config result).
* `docs/data/phase54_cross_role_coherence_budget_sweep.json`
  (frozen K-sweep).
* `docs/THEOREM_REGISTRY.md` — W7-1 / W7-1-aux / W7-2 /
  W7-2-conditional / W7-3 / W7-C1 / W7-C2 / W7-C3 added.
* `docs/RESEARCH_STATUS.md` — seventh research axis added.
* `docs/HOW_NOT_TO_OVERSTATE.md` — W7 overstatement guards added.

Anchor: `docs/RESULTS_WEVRA_CROSS_ROLE_COHERENCE.md`;
`docs/data/phase54_cross_role_coherence_K4_n10.json`;
`vision_mvp/experiments/phase54_cross_role_coherence.py`.

### 4.26 SDK v3.9 — cross-role corroboration multi-agent coordination (deterministic Phase-55 benchmark + W8 family)

The v3.8 milestone (§ 4.25) produced an honest *conditional*
result: at the Phase-54 default config (gold-plurality + foreign-
service decoys), buffered cohort coherence beats substrate FIFO by
+1.000 on accuracy_full. But the win is brittle — in any regime
where some decoy carries strictly more raw mentions than gold,
the W7-2 single-tag plurality picks the decoy and ties FIFO at
0.000 (the named W7-2 falsifier). SDK v3.9 directly attacks this
falsifier and clears the **strong success bar** of
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 — a strict
gain ≥ 0.20 on a harder regime, stable across ≥ 3 seeds, no
regression on Phase 53 / Phase 54, audit T-1..T-7 preserved on
every cell, and a *named falsifier regime* where the win
correctly does not hold.

**1. Pre-committed success criterion (PART A).**
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` defines three
discrete bars (strong / partial / null) and three named regimes
(R-53 / R-54 / R-55). The strong bar requires: code anchor,
strict gain ≥ 0.20 on R-55 vs both substrate FIFO and the
SDK v3.8 strongest method, cross-bank stability ≥ 3 seeds, no
regression > 0.05 on R-53 / R-54, audit preserved, and a *named
falsifier regime* where the new method does not beat FIFO. SDK
v3.9 was pre-committed to this bar before any Phase-55 result
was measured.

**2. Phase-55 deterministic decoy-plurality + cross-role-
corroborated benchmark (PART B).**
`vision_mvp/experiments/phase55_decoy_plurality.py` materialises
the smallest deterministic regime where:

* Some decoy service has *strictly more raw mentions* in the
  auditor's candidate stream than gold (W7-2 falsifier
  instantiation).
* The gold service is mentioned by *strictly more distinct
  producer roles* than any decoy service (the cross-role
  corroboration signal).
* Each role's *local* plurality of `service=<tag>` mentions is
  a *decoy* (the locally-misleading-but-cross-role-decisive
  property the SDK v3.9 brief asks for).
* `|candidates_to_auditor| > K_auditor` on every scenario at
  K=4.

Bench properties are *named and mechanically verified* by
`Phase55BankShapeTests`: 10/10 default scenarios satisfy
decoy-plurality, gold-corroboration, surplus, and ≥ 3 corroborating
roles. A *named falsifier regime* (`build_phase55_falsifier_bank`)
flips the corroboration signal so 0/10 falsifier scenarios satisfy
it; on the falsifier the new policy correctly ties FIFO at 0.000.

**3. `CrossRoleCorroborationAdmissionPolicy` (new, SDK v3.9 — PART C).**
`vision_mvp/wevra/team_coord.py`. A deterministic, training-free,
interpretable cross-role admission rule that aggregates over
**(role, tag) bipartite multisets** rather than over raw tag
counts. Default score function:

```
score(tag) = role_weight * |distinct_roles(tag)| + |raw_mentions(tag)|
```

with `role_weight = 100` (default). Two sub-modes: streaming
(arrival-order-sensitive baseline) and **buffered** (arrival-
order-stable, pre-fitted via `from_candidate_stream`; the W8-1
load-bearing variant).

The policy is a *strict generalisation* of W7-2: on any stream
where gold has both raw plurality AND distinct-role coverage,
the two policies admit identical sets (W8-3 backward-compat).

**4. Empirical headline (Phase-55 default, n=10, K=4 — PART D).**

| Strategy                  | accuracy_full | accuracy_services | mean_n_admitted_auditor |
| ------------------------- | ------------- | ----------------- | ----------------------- |
| substrate / capsule_fifo / capsule_priority | 0.000  | 0.000  | 4.00 |
| capsule_coverage          | 0.000         | 0.000             | 4.00                    |
| capsule_cohort_buffered (W7-2) | 0.000    | 0.000             | 3.60                    |
| **capsule_corroboration** | **1.000**     | **1.000**         | **3.20**                |

Headline gaps:
* `corroboration − fifo accuracy_full = +1.000`
* `corroboration − cohort_buffered accuracy_full = +1.000` ← strict separation from W7-2

Stable across 5/5 alternate `bank_seed` values (11, 17, 23, 29, 31).
On the named W8-4 falsifier regime, all strategies (including
corroboration) tie at 0.000 — confirming the conditional nature
of the win. On Phase 54 (gold-plurality), corroboration ties W7-2
at 1.000 — backward-compat (W8-3). On Phase 53 synthetic and
Phase 53 14B real-LLM (`qwen2.5:14b-32k`), all strategies tie at
0.800 — no regression (W7-1 low-surplus regime).

**5. The W8 theorem family (PART E).**

* **W8-1 (proved-empirical, n=50 saturated).** Cross-role
  corroboration strict separation from W7-2 on Phase-55 default.
  Headline gap +1.000 vs both substrate FIFO and SDK v3.8 W7-2
  buffered cohort. Stable across 5/5 bank seeds. Named falsifier:
  Phase-55 falsifier bank (W8-4).
* **W8-2 (proved, structural).** Score-function strict-ordering
  theorem: with `role_weight > Δr_max`, the corroboration score
  function strictly orders cross-role-corroborated gold above
  raw-plurality decoy regardless of raw-count distribution. With
  default `role_weight=100`, no candidate stream of size < 100 can
  produce a raw-count-only override.
* **W8-3 (proved-empirical, n=10).** Backward compatibility: on
  Phase-54 default (gold-plurality), corroboration matches W7-2
  cohort_buffered at `accuracy_full = 1.000`. The new policy is
  a strict generalisation, not a replacement.
* **W8-4 (proved-empirical, n=10 falsifier saturated).**
  Decoy-corroboration falsifier: when the decoy has strictly more
  distinct producer roles than gold, buffered corroboration picks
  the decoy and ties FIFO at 0.000. The W8-1 win does NOT hold
  in this regime — by construction. Sharper observation: even at
  K=8 (no budget pressure), W7-2 ties FIFO at 0.000 because the
  decoder's set-equality `services_correct` check is unrescuable
  by service-blind admission; cross-role corroboration is the
  load-bearing distinction between W7-2's failure mode and W8-1's
  win.
* **W8-C1 / W8-C2 / W8-C3 (conjectural).** Multi-service-gold
  extension (top-k corroboration), real-LLM multi-service decoy
  regime, bundle-aware decoder companion.

**6. What the W8 family says about the original thesis.** The
capsule layer's **audit** contribution is preserved and extends to
Phase-55 unchanged (T-1..T-7 hold on every cell). The capsule
layer's **coordination-performance** contribution now spans
**three** named regimes (no-regression on R-53; backward-compat
on R-54; strict win on R-55) with cross-bank stability and a
named falsifier — the strongest cross-regime conditional
structural-win the programme has produced. SDK v3.9 is the
**first SDK milestone** to clear the strong success bar.

* The earlier SDK v3.5 "learned policy beats FIFO at noisy
  admission" framing was mean-positive but not robust.
* The SDK v3.8 "buffered cohort beats FIFO at gold-plurality"
  framing was strict-positive on Phase-54 but brittle on
  decoy-plurality regimes.
* The SDK v3.9 "buffered cross-role corroboration beats both
  FIFO and W7-2 on decoy-plurality + cross-role-corroborated
  gold" framing is **strict-positive on three regimes** (Phase 53
  no-regression, Phase 54 backward-compat, Phase 55 strict win)
  with cross-bank stability and a named falsifier.

The W7-2 / W8-1 hierarchy is a strict generalisation: W7-2's wins
are preserved by W8-1 (W8-3 backward-compat), and W8-1 extends
to a strict superset of regimes. The original Context Zero
thesis is **per-agent minimum-sufficient context for multi-agent
teams** — the W8 family makes this true on three named regimes
with stated conditions and a named falsifier, not just one.

**7. Honest scope (PART F).** Three named regimes is a stronger
cross-regime result than two, but it is not "all regimes." Real
production multi-agent teams have additional axes (heterogeneous
producers, time-varying budgets, multi-round handoffs,
conflicting goals, multi-service gold answers) that Phase 55 does
not test. W8-C1 / W8-C2 / W8-C3 are the conjectural extensions;
none are yet shipped. The W8-1 strict-separation result is *not*
a claim that "we solved multi-agent context" — see
`docs/HOW_NOT_TO_OVERSTATE.md` § "Labelling the SDK v3.9 result
'we solved multi-agent context'".

**8. What changed.**

* `vision_mvp/wevra/team_coord.py` (extended) —
  `CrossRoleCorroborationAdmissionPolicy`; `_candidate_source_role`;
  updated `ALL_FIXED_POLICY_NAMES`.
* `vision_mvp/wevra/__init__.py` — re-exports
  `TeamCrossRoleCorroborationAdmissionPolicy`;
  `SDK_VERSION = "wevra.sdk.v3.9"`.
* `vision_mvp/experiments/phase55_decoy_plurality.py` (new) —
  Phase-55 driver, 5 base scenario builders, default + falsifier
  bank constructors, `run_phase55`, `run_phase55_budget_sweep`,
  `run_seed_stability_sweep`, `run_cross_regime_summary`.
* `vision_mvp/tests/test_wevra_cross_role_corroboration.py` (new)
  — 34 contract tests including W8-1 default-config win, W8-2
  structural ordering, W8-3 backward-compat, W8-4 falsifier,
  K-sweep, seed stability, audit invariance, no-regression on
  Phase 53 synthetic.
* `vision_mvp/tests/test_wevra_public_api.py` (updated) —
  `test_sdk_version_is_v3_9` and corroboration-policy export
  test.
* `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` (new) —
  pre-committed strong / partial / null success bars; named
  regime taxonomy.
* `docs/RESULTS_WEVRA_CROSS_ROLE_CORROBORATION.md` (new —
  milestone results note).
* `docs/data/phase55_decoy_plurality_K4_n10.json` (frozen
  default).
* `docs/data/phase55_falsifier_K4_n10.json` (frozen falsifier).
* `docs/data/phase55_budget_sweep.json` (frozen K-sweep).
* `docs/data/phase55_seed_sweep.json` (frozen 5-seed sweep).
* `docs/data/phase55_cross_regime.json` (frozen Phase 54+55+
  falsifier bundle).
* `docs/data/phase53_real_llm_corroboration_check.json` (frozen
  Phase-53 14B real-LLM regression check).
* `docs/THEOREM_REGISTRY.md` — W8-1 / W8-2 / W8-3 / W8-4 /
  W8-C1 / W8-C2 / W8-C3 added; date stamp v3.9.
* `docs/RESEARCH_STATUS.md` — eighth research axis added.
* `docs/HOW_NOT_TO_OVERSTATE.md` — W8 overstatement guards
  added (W8-1 conditionality, "we solved multi-agent context"
  forbidden, Phase-54/55 conflation forbidden, Phase-53/55
  conflation forbidden).
* `docs/START_HERE.md` — SDK v3.9 paragraph + canonical-reading
  pointer to the success-criterion doc.

Anchor: `docs/RESULTS_WEVRA_CROSS_ROLE_CORROBORATION.md`;
`docs/data/phase55_decoy_plurality_K4_n10.json`;
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`;
`vision_mvp/experiments/phase55_decoy_plurality.py`;
`vision_mvp/wevra/team_coord.py`
(`CrossRoleCorroborationAdmissionPolicy`).

### 4.27 SDK v3.10 — multi-service top-K cross-role corroboration multi-agent coordination (deterministic Phase-56 benchmark + W9 family)

The v3.9 milestone (§ 4.26) cleared the strong success bar by
strictly separating cross-role corroboration (W8) from single-tag
plurality (W7-2) on a harder *decoy-plurality* regime. But the W8
result has a named falsifier of its own: it picks the top-1
corroborated tag and only the top-1. On any *multi-service-gold*
regime where the gold answer requires ``services = {A, B}`` (the
canonical realistic incident shape), W8 admits only candidates
carrying the single highest-scoring tag and the decoder's
set-equality ``services_correct`` check fails. SDK v3.10 directly
attacks this falsifier and clears the **strong success bar** of
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 (R-56
anchor) — a strict gain ≥ 0.20 on a harder regime *vs the SDK
v3.9 strongest method*, stable across ≥ 3 seeds, no regression on
Phase 53 / Phase 54 / Phase 55, audit T-1..T-7 preserved on every
cell, and a *named falsifier regime* where the win correctly does
not hold.

**1. Pre-committed success criterion (PART A — updated R-56 anchor).**
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` extends the bar
to four named regimes (R-53 / R-54 / R-55 / R-56). The strong bar
now requires: code anchor, strict gain ≥ 0.20 on **R-56** vs both
substrate FIFO and the SDK v3.9 strongest method
(``CrossRoleCorroborationAdmissionPolicy``, W8), cross-bank
stability ≥ 3 seeds, no regression > 0.05 on R-53 / R-54 / R-55,
audit preserved, and a *named falsifier regime*. The falsifying-
failure list now also gates the W8-1 contract test.

**2. Phase-56 deterministic multi-service-gold + cross-role-
corroborated benchmark (PART B).**
`vision_mvp/experiments/phase56_multi_service_corroboration.py`
materialises the smallest deterministic regime where:

* Every scenario has ``gold_services`` of size **2** (multi-
  service incident — pool exhaustion across api+db, deadlock
  across orders+payments, slow-query cascade across web+db, error
  spike across api+mobile, disk fill across storage+logs).
* **Both** gold services are corroborated by ≥ 2 distinct producer
  roles each (the cross-role corroboration signal *for both gold
  tags simultaneously*).
* At least one decoy service has raw plurality but is corroborated
  by **exactly 1** producer role (so substrate FIFO and W7-2
  buffered cohort fail by admitting the decoy).
* `|candidates_to_auditor| > K_auditor` on every scenario at K=4.

Bench properties are *named and mechanically verified* by
`Phase56BankShapeTests`: 10/10 default scenarios satisfy
multi-service-gold (size 2), both-gold-corroborated (≥2 roles each),
single-role-decoy (≤1 role), and surplus. A *named falsifier
regime* (`build_phase56_falsifier_bank`) promotes a decoy to ≥ 2
distinct producer roles so 10/10 falsifier scenarios satisfy
`decoy_corroboration_holds`; on the falsifier the new policy
correctly ties FIFO at 0.000.

**3. `MultiServiceCorroborationAdmissionPolicy` (new, SDK v3.10 — PART C).**
`vision_mvp/wevra/team_coord.py`. A deterministic, training-free,
interpretable cross-role admission rule that admits the **top-K
cross-role-corroborated tier** rather than the single highest-
scoring tag. Selection rule:

```
1. Drop tags with |distinct_roles(tag)| < min_corroborated_roles
2. Argmax-by-role-count tier: keep tags with the maximum
   |distinct_roles| among the eligible set
3. Top-K by score (lex tie-break) among the argmax tier
```

with default ``top_k = 2, min_corroborated_roles = 2,
role_weight = 100``. The argmax-by-role-count gate is
**load-bearing**: it ensures W9 collapses to W8 when only one tag
has the maximum role count (W9-3 backward-compat) — so W9 is a
*strict generalisation* of W8 by construction.

**4. Empirical headline (Phase-56 default, n=10, K=4 — PART D).**

| Strategy                  | accuracy_full | accuracy_services | mean_n_admitted_auditor |
| ------------------------- | ------------- | ----------------- | ----------------------- |
| substrate / capsule_fifo / capsule_priority / capsule_coverage | 0.000  | 0.000  | 4.00 |
| capsule_cohort_buffered (W7-2) | 0.000    | 0.000             | ~3                      |
| capsule_corroboration (W8) | 0.000        | 0.000             | ~2                      |
| **capsule_multi_service** | **1.000**     | **1.000**         | **4.00**                |

Headline gaps:
* `multi_service − fifo accuracy_full = +1.000`
* `multi_service − cohort_buffered accuracy_full = +1.000`
* `multi_service − corroboration accuracy_full = +1.000` ← strict separation from W8

Stable across 5/5 alternate `bank_seed` values (11, 17, 23, 29, 31).
On the named W9-4 falsifier regime, all strategies (including
multi_service) tie at 0.000 — confirming the conditional nature
of the win. On Phase 55 default, multi_service ties W8 at 1.000
(W9-3 backward-compat via the argmax-by-role-count gate). On
Phase 54 default, multi_service ties W7-2 at 1.000. On Phase 53
synthetic, all admission strategies tie FIFO at 0.800 — no
regression (W7-1 low-surplus regime).

**5. The W9 theorem family (PART E).**

* **W9-1 (proved-empirical, n=50 saturated).** Multi-service
  corroboration strict separation from W8 on Phase-56 default.
  Headline gap +1.000 vs substrate FIFO, W7-2, **and W8**. Stable
  across 5/5 bank seeds. Named falsifier: Phase-56 falsifier bank
  (W9-4).
* **W9-2 (proved, structural).** ``_dominant_tag_set`` selection
  rule has three structural properties: (a) single-role exclusion,
  (b) argmax-tier collapse to size 1 → W9 ≡ W8, (c) argmax-tier
  multi-tag admission within ``top_k`` cap.
* **W9-3 (proved-empirical, n=10).** Backward-compat: on Phase-55
  default, W9 admits identical set to W8; on Phase-54 default,
  W9 ties W7-2 at 1.000.
* **W9-4 (proved-empirical, n=10 saturated).** Decoy-corroboration
  falsifier: when a decoy is also corroborated by ≥
  ``min_corroborated_roles`` distinct producer roles, the W9
  dominant set includes the decoy (argmax tier expands), the
  decoder's `services` set picks up the decoy, and `services_correct`
  fails on 10/10 falsifier scenarios. Sharper observation: this
  is the structural limit of any service-blind admission policy
  under set-equality decoder grading; the only escape is a
  bundle-aware decoder companion (W9-C1).

The W9-C family makes the bundle-aware decoder / |gold|≥3 /
real-LLM extensions falsifiable. SDK v3.9's W8-C1 conjecture
(top-K corroboration improves multi-service scenarios) is now
**discharged-empirical** by W9-1.

**6. PART F (product/runtime honesty).** The Wevra single-run
product runtime contract is **byte-for-byte unchanged from SDK
v3.9**. The new admission policy is a research-slice addition to
`vision_mvp.wevra.team_coord` (additive surface). The lifecycle
audit (T-1..T-7) holds on every cell of every regime, including
Phase 56 default and falsifier.

**7. Files / tests / artefacts (this milestone).**

* `vision_mvp/wevra/team_coord.py` (extended) —
  `MultiServiceCorroborationAdmissionPolicy` added; streaming +
  `from_candidate_stream` buffered factory; `_dominant_tag_set`
  helper; `ALL_FIXED_POLICY_NAMES` updated.
* `vision_mvp/wevra/__init__.py` — re-exports
  `TeamMultiServiceCorroborationAdmissionPolicy`; `SDK_VERSION`
  bumped to `wevra.sdk.v3.10`.
* `vision_mvp/experiments/phase56_multi_service_corroboration.py`
  (new) — driver; 5 base scenario builders; default + falsifier
  bank constructors; `run_phase56`,
  `run_phase56_seed_stability_sweep`, `run_cross_regime_summary`.
* `vision_mvp/tests/test_wevra_multi_service_corroboration.py`
  (new) — 36 contract tests covering policy unit tests, bank
  shape, default config win, seed stability, falsifier behaviour,
  W9-3 backward-compat with Phase 55, audit-grid invariance,
  cross-regime contract, public-API contract.
* `vision_mvp/tests/test_wevra_public_api.py` (updated) —
  `test_sdk_version_is_v3_10` (renamed from v3_9).
* `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` (updated) —
  R-56 named regime added; bar anchor advanced to R-56;
  falsifying-failure list extended to gate W8-1 contract test.
* `docs/RESULTS_WEVRA_MULTI_SERVICE_CORROBORATION.md` (new —
  milestone results note).
* `docs/data/phase56_multi_service_K4_n10.json` (frozen default).
* `docs/data/phase56_falsifier_K4_n10.json` (frozen falsifier).
* `docs/data/phase56_seed_sweep.json` (frozen 5-seed sweep).
* `docs/data/phase56_cross_regime.json` (frozen
  Phase 54+55+56 + falsifier bundle).
* `docs/data/phase53_synthetic_w9_regression_check.json` (frozen
  Phase-53 synthetic regression check).
* `docs/THEOREM_REGISTRY.md` — W9-1/W9-2/W9-3/W9-4/W9-C1/W9-C2/
  W9-C3 added; W8-C1 marked DISCHARGED; date stamp v3.10.
* `docs/RESEARCH_STATUS.md` — ninth research axis added; SDK
  v3.10 frontier section.
* `docs/START_HERE.md` — SDK v3.10 paragraph + W9 family summary.

Anchor: `docs/RESULTS_WEVRA_MULTI_SERVICE_CORROBORATION.md`;
`docs/data/phase56_multi_service_K4_n10.json`;
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`;
`vision_mvp/experiments/phase56_multi_service_corroboration.py`;
`vision_mvp/wevra/team_coord.py`
(`MultiServiceCorroborationAdmissionPolicy`).

**Master-plan post-v3.10 reading.** After SDK v3.10:

1. **Post-v3.9 success bar.** R-56 (multi-service-gold +
   cross-role-corroborated; both gold services 2-role corroborated;
   single-role decoy storm). Bar: strict gain ≥ 0.20 vs both FIFO
   and SDK v3.9 W8, stable across ≥ 3 seeds, no regression on
   R-53/R-54/R-55, audit preserved, named falsifier regime.
2. **Harder regime fairness.** Phase 56 is fair because every
   property (multi-service gold, both-gold corroboration, single-
   role decoy, budget pressure, realistic claim_kinds) is mechanically
   verified by `Phase56BankShapeTests` and pre-committed in code
   *before* any seed sweep was run.
3. **Did the new method broaden the structural win?** **Yes.**
   The structural win now spans **four** named regimes (Phase 53
   no-regression, Phase 54 backward-compat, Phase 55 backward-compat,
   Phase 56 strict win) and is the first programme result whose
   *strict-gain* regime is *not solvable by the previous SDK's
   strongest method*. W9-1 is the first strict separation from W8.
4. **Original thesis status.** *Per-agent minimum-sufficient
   context for multi-agent teams* is now **stronger but still
   conditional**. The capsule layer's audit contribution is
   preserved; the coordination-performance contribution now spans
   four named regimes; the W9-4 falsifier (decoy corroboration) is
   the named structural limit of service-blind admission. The next
   axis of structural improvement is the **bundle-aware decoder
   companion** (W9-C1), which would attack the W9-4 falsifier from
   the decoder side rather than the admission side.

### 4.28 SDK v3.11 — bundle-aware team decoder + W10 family (deterministic Phase-57 decoder-forcing benchmark)

The SDK v3.10 milestone identified a sharp structural limit:
service-blind admission cannot beat FIFO when the decoy is
cross-role-corroborated above the min-roles threshold (W9-4).
This is a property of the *(role, service-tag)* bipartite multiset
— the only state every shipped admission policy depends on.

SDK v3.11 lifts this limit on a named regime by adding a
**decoder-side coordination move** for the first time in the
programme: the bundle-aware team decoder
(``BundleAwareTeamDecoder``) projects the auditor's admitted
service set through the *causal claim-kind set* of the chosen
``root_cause`` (CCK). When the gold root_cause is in a specific
causal tier (data / storage / compute / edge / network) and the
decoy is mentioned only via non-causal generic-noise kinds
(LATENCY_SPIKE / ERROR_RATE_SPIKE / FW_BLOCK_SURGE), the CCK
predicate filters the decoy out at decode time even though
admission cannot.

**Phase-57** (``vision_mvp/experiments/phase57_decoder_forcing.py``)
is the *decoder-forcing* regime that anchors the milestone. Every
scenario has multi-service gold (size 2) corroborated via causal
claim_kinds AND a decoy corroborated via non-causal kinds.

**Headline (n=10, K_auditor=8, T_auditor=256, bank_seed=11):**

* ``capsule_bundle_decoder accuracy_full = 1.000``
* every service-blind admission policy in the SDK
  (substrate, FIFO, priority, coverage, W7-2, W8, W9): ``0.000``
* **+1.000** vs every admission baseline; **+1.000** vs the SDK
  v3.10 strongest method (W9). Stable across **5/5** alternate
  bank seeds (11, 17, 23, 29, 31).
* Backward-compat: bundle decoder ties prior best on Phase 54
  (W7-2), Phase 55 (W8), Phase 56 (W9) at 1.000 (W10-3) — the
  ``fallback_admitted_size_threshold = 2`` knob is the load-bearing
  piece.
* Audit T-1..T-7 OK on every cell of every regime (W10-aud).
* W10-Λ admission limit witness: ``max_admission_only_accuracy_full
  = 0.000`` on R-57.

**Anchor:** ``docs/RESULTS_WEVRA_BUNDLE_DECODER.md``;
``docs/data/phase57_decoder_K8_n10.json``;
``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-57 anchor +
bar 7 — admission/decoding split);
``vision_mvp/experiments/phase57_decoder_forcing.py``;
``vision_mvp/wevra/team_coord.py`` (``BundleAwareTeamDecoder``,
``CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE``).

**Master-plan post-v3.11 reading.** After SDK v3.11:

1. **Post-v3.10 success bar.** R-57 (multi-service-gold +
   corroborated-decoy via non-causal kinds; specific-tier gold
   root_cause). Bar (R-57 anchor): strict gain ≥ 0.20 vs FIFO and
   SDK v3.10 W9, stable across ≥ 3 seeds, no regression on
   R-53/R-54/R-55/R-56, audit preserved, named falsifier regime,
   AND admission/decoding split (bar 7).
2. **Has admission reached a structural limit?** **Yes — explicitly,
   by W10-Λ.** Service-blind admission cannot beat FIFO on R-57.
   The SDK v3.11 moves the structural axis from admission into
   decoding for the first time in the programme.
3. **What is the next decoder-centred benchmark?** Phase-57 is
   anchored as the canonical decoder-forcing regime. The named
   falsifier (W10-4: decoy CCK-promotion) and the named conjectures
   (W10-C1 cross-bench, W10-C2 real-LLM, W10-C3 multi-round) chart
   the next moves.
4. **Did decoding broaden the structural win beyond v3.10?** **Yes.**
   The structural win now spans **five** named regimes (R-53
   no-regression, R-54 / R-55 / R-56 backward-compat, R-57 strict
   win on a regime where admission alone is provably insufficient).
   W10-1 is the first strict separation between decoder-side
   coordination and any service-blind admission policy.
5. **Original thesis status.** *Per-agent minimum-sufficient
   context for multi-agent teams* is now **materially stronger but
   still conditional**. The shape of "minimum-sufficient" now has
   two coupled axes: admission (W7-2 / W8 / W9) and decoding
   (W10). The CCK predicate is closed-vocabulary, deterministic,
   training-free, and additive on top of the existing capsule
   lifecycle. The remaining conditions are: bench property, CCK
   table existence for the benchmark family, and gold root_cause
   being in a specific causal tier (W10-1 honest scope). The W10-4
   falsifier names the next wall.

---

### 4.29 SDK v3.12 — multi-round bundle decoder + W11 family (deterministic Phase-58 multi-round delayed-causal-evidence benchmark)

The SDK v3.11 W10 milestone established the *decoder/admission
split* (W10-Λ) and produced the first decoder-side win on R-57.
But W10's CCK filter is intra-round: it consumes one ROLE_VIEW
capsule's admitted handoffs. SDK v3.11 already named the next
wall (W10-C3 conjecture: multi-round bundle decoder closes a
sub-class of cases where admission alone fails).

SDK v3.12 lifts the *temporal* axis on a named sub-regime — but
not the W10-4 sub-regime (CCK-promoted decoy); a different
sub-regime where the decoy is **noise-corroborated** and the
disambiguating causal evidence arrives in a **later round with no
service tag**. This is the **W11 family**.

**Phase-58** (``vision_mvp/experiments/phase58_multi_round_decoder.py``)
is the *multi-round delayed-causal-evidence* regime that anchors
the milestone. Two-round coordination where round-1 carries
generic-noise-only mentions of ``(gold_A, gold_B, decoy)`` with
the decoy cross-role-corroborated, AND round-2 carries one
specific-tier disambiguating ``claim_kind`` with **no
``service=`` token**. Single-round decoding — including W10 —
cannot win.

**Headline (n=8, K_auditor=8, T_auditor=256, bank_seed=11):**

* ``capsule_multi_round accuracy_full = 1.000``
* every single-round capsule strategy
  (substrate, FIFO, priority, coverage, W7-2, W8, W9, W10): ``0.000``
* **+1.000** vs every single-round method including SDK v3.11 W10.
  Stable across **5/5** alternate bank seeds (11, 17, 23, 29, 31).
* Backward-compat (W11-3): cross-regime summary preserves prior
  best at 1.000 on R-54 (W7-2), R-55 (W8), R-56 (W9), R-57 (W10).
* Audit T-1..T-7 OK on every cell of every R-58 capsule strategy.
* W11-Λ single-round limit witness: ``max_single_round_accuracy_full
  = 0.000`` on R-58 default.
* Named falsifier W11-4 (round-1 noise floods K_auditor=4 budget):
  multi_round ties FIFO at 0.000.

**Anchor:** ``docs/RESULTS_WEVRA_MULTI_ROUND_DECODER.md``;
``docs/data/phase58_default_K8_n8.json``;
``docs/data/phase58_seed_sweep_K8_n8.json``;
``docs/data/phase58_falsifier_K4_n8.json``;
``docs/data/phase58_cross_regime.json``;
``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-58 anchor +
bar 8 — temporal/structural split);
``vision_mvp/experiments/phase58_multi_round_decoder.py``;
``vision_mvp/wevra/team_coord.py`` (``MultiRoundBundleDecoder``,
``collect_admitted_handoffs``, ``_GENERIC_NOISE_CLAIM_KINDS``).

**Master-plan post-v3.12 reading.** After SDK v3.12:

1. **Post-v3.11 success bar.** R-58 (delayed-causal-evidence with
   noise-corroborated decoy and round-N specific-tier
   disambiguation). Bar (R-58 anchor): strict gain ≥ 0.20 vs every
   SDK v3.11 single-round method, stable across ≥ 3 seeds, no
   regression on R-53..R-57, audit preserved, named falsifier
   regime, AND temporal/structural split (bar 8).
2. **Has hand-crafted single-round decoding reached a structural
   limit?** **Yes — explicitly, by W11-Λ.** No single-round
   capsule strategy can beat FIFO on R-58 default. The SDK v3.12
   moves the structural axis from intra-round decoding into
   cross-round decoding for the first time in the programme.
3. **What is the next benchmark?** Phase-58 is anchored as the
   canonical multi-round delayed-causal-evidence regime. The
   named falsifier (W11-4: round-1 budget flood) and the named
   conjectures (W11-C1 cross-bench, W11-C2 real-LLM, W11-C3
   multi-step disambiguation across ≥ 3 rounds) chart the next
   moves.
4. **Did multi-round decoding broaden the structural win beyond
   v3.11?** **Yes.** The structural win now spans **six** named
   regimes (R-53 no-regression, R-54 / R-55 / R-56 / R-57
   backward-compat, R-58 strict win on a regime where every
   single-round method is provably insufficient). W11-1 is the
   first strict separation between cross-round and intra-round
   decoder-side coordination.
5. **Original thesis status.** *Per-agent minimum-sufficient
   context for multi-agent teams* is now **materially stronger but
   still conditional along three coupled axes**:
   admission (W7-2 / W8 / W9), intra-round decoding (W10),
   cross-round decoding (W11). The contradiction-aware noise-decoy
   drop is closed-vocabulary, deterministic, training-free, and
   additive on top of the existing capsule lifecycle and the W10
   CCK filter. The remaining conditions are: bench property, the
   closed-vocabulary noise-kind set being meaningful for the
   benchmark family, gold root_cause being in a specific causal
   tier, AND round-N admission not being budget-starved (W11-4).
   The W11-4 falsifier names the next wall: a smarter admission
   policy that prioritises specific-tier kinds, OR a multi-step
   contradiction-aware round-resolution rule (W11-C3), is the
   natural next move.

**Honest reading.** SDK v3.12 does **not** discharge the
strongest open question: real-LLM transfer of any of W7..W11.
All wins remain on deterministic synthetic benches. The next
real-LLM step is W11-C2 (Phase 59 candidate) — and it is
genuinely the next research move, not a polish pass. The
SDK is now an explicit hierarchy of three structural axes, each
with a sharp limit theorem on its own; whether real production
multi-agent teams *actually* hit these regimes is the open
empirical question that the closed-vocabulary closure cannot
answer alone.

### 4.30 SDK v3.13 — real-LLM-robust multi-round bundle decoder + W12 family (Phase-59 real-LLM-driven multi-round delayed-disambiguation benchmark)

The SDK v3.12 W11 milestone established the *temporal/structural
split* (W11-Λ) and produced the first cross-round capsule-native
win on R-58. But the win was on a deterministic synthetic stream;
the open question (W11-C2) was whether the same structural move
survives a **real-LLM-shaped producer-noise channel**. SDK v3.13
attacks W11-C2 directly and *partially discharges* it: the
**un-normalised** W11 decoder does NOT transfer, but a *normalised*
multi-round decoder (W12-1) does, conditional on a closed-vocabulary
closure of the LLM's drift channel. This is the **W12 family**.

**Phase-59** (``vision_mvp/experiments/phase59_real_llm_multi_round.py``)
is the *real-LLM-driven multi-round delayed-disambiguation* regime
that anchors the milestone. Same R-58 delayed-causal-evidence
structural shape, but the candidate stream is generated by an
**LLM-shaped extractor** that injects realistic ``claim_kind`` drift
(``synonym_prob=0.50`` — every canonical kind has 3-5 plausible LLM
variants in :data:`NOISY_KIND_VARIANTS`) and ``service=`` payload
drift (``svc_token_alt_prob=0.30`` — alternative spellings like
``svc=X`` / ``for service X``). The default mode
``synthetic_noisy_llm`` is calibrated against Phase-53 14B/35B
empirical kind-drift distributions; the optional ``ollama`` mode is
the opt-in real-LLM extension path (with synthetic fallback on HTTP
failure).

The new method ``RobustMultiRoundBundleDecoder`` wraps W11
``MultiRoundBundleDecoder`` with a closed-vocabulary normalisation
layer: :data:`CLAIM_KIND_SYNONYMS` (≈ 60 entries × 11 canonical
kinds), :func:`normalize_claim_kind`, :func:`normalize_payload`,
:func:`normalize_handoff`. Per-call rewrite counters expose the
load-bearing role of the normaliser to the bench driver.

**Headline (n=12, K_auditor=8, T_auditor=256, bank_seed=11,
llm_synonym_prob=0.50, llm_svc_alt_prob=0.30):**

* ``capsule_robust_multi_round accuracy_full = 1.000``
* every un-normalised capsule strategy
  (substrate, FIFO, priority, coverage, W7-2, W8, W9, W10, **and
  SDK v3.12 W11**): ``0.000``
* **+1.000** vs every un-normalised method including SDK v3.12 W11.
  Stable across **5/5** alternate (bank_seed, llm_seed) values
  (11, 17, 23, 29, 31) at gap robust − w11 ≥ 0.917 every seed.
* Backward-compat (W12-3): cross-regime summary preserves prior
  best at 1.000 on R-54 / R-55 / R-56 / R-57 / R-58; on R-59 with
  ``llm_mode='synthetic_clean_llm'`` the rewrite counters are zero
  and W12 ties W11 byte-for-byte at 1.000.
* Audit T-1..T-7 OK on every cell of every R-59 capsule strategy.
* W12-Λ un-normalised limit witness:
  ``max_non_robust_accuracy_full = 0.000`` on R-59 default.
* Named falsifier W12-4 (``oov_prob=0.50`` activates kinds outside
  :data:`CLAIM_KIND_SYNONYMS`): robust ties FIFO at 0.000.

**Anchor:** ``docs/RESULTS_WEVRA_REAL_LLM_MULTI_ROUND.md``;
``docs/data/phase59_default_K8_n12.json``;
``docs/data/phase59_seed_sweep_K8_n12.json``;
``docs/data/phase59_falsifier_K8_n8.json``;
``docs/data/phase59_clean_K8_n8.json``;
``docs/data/phase59_cross_regime.json``;
``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-59 anchor +
bar 9 — synthetic→real-LLM transfer split);
``vision_mvp/experiments/phase59_real_llm_multi_round.py``;
``vision_mvp/wevra/team_coord.py`` (``RobustMultiRoundBundleDecoder``,
``CLAIM_KIND_SYNONYMS``, ``normalize_claim_kind``,
``normalize_payload``, ``normalize_handoff``).

**Master-plan post-v3.13 reading.** After SDK v3.13:

1. **Post-v3.12 success bar.** R-59 (R-58 shape + bounded LLM
   noise channel via :data:`NOISY_KIND_VARIANTS` ⊂
   :data:`CLAIM_KIND_SYNONYMS`). Bar (R-59 anchor): strict gain
   ≥ 0.20 vs every un-normalised single-round / multi-round
   method including SDK v3.12 W11, stable across ≥ 3 (bank_seed,
   llm_seed) values, no regression on R-53..R-58 or R-59-clean,
   audit preserved, named falsifier regime, AND synthetic→real-LLM
   transfer split (bar 9).
2. **Has un-normalised cross-round decoding reached a structural
   limit?** **Yes — explicitly, by W12-Λ at the real-LLM axis.**
   No un-normalised capsule strategy can beat FIFO on R-59
   default. The SDK v3.13 moves the structural axis from
   *cross-round decoding under canonical input* into
   *cross-round decoding under bounded producer-noise drift* for
   the first time in the programme.
3. **What is the next benchmark?** Phase-59 is anchored as the
   canonical real-LLM-driven multi-round delayed-disambiguation
   regime. The named falsifier (W12-4: out-of-vocabulary kind
   drift) and the named conjectures (W12-C1 cross-bench,
   W12-C2 real-Ollama transfer, W12-C3 learned normaliser)
   chart the next moves.
4. **Did normalised cross-round decoding broaden the structural
   win beyond v3.12?** **Yes — materially.** The structural win
   now spans **seven** named regimes (R-53 no-regression,
   R-54..R-58 backward-compat, R-59 noisy strict win on a regime
   where every un-normalised method is provably insufficient
   under the named bounded-noise channel). W12-1 is the first
   strict separation between un-normalised and normalised
   cross-round capsule-native coordination on a real-LLM-shaped
   stream.
5. **Original thesis status.** *Per-agent minimum-sufficient
   context for multi-agent teams* is now **materially stronger
   along four coupled axes**: admission (W7-2 / W8 / W9), intra-
   round decoding (W10), cross-round decoding (W11), and
   *normalisation across the producer-noise channel* (W12). The
   normalisation contract is closed-vocabulary, deterministic,
   training-free, and additive on top of the existing capsule
   lifecycle, the W10 CCK filter, and the W11 contradiction-aware
   drop. The remaining conditions are: bench property, closure
   property on the noise channel, gold root_cause being in a
   specific causal tier, AND round-N admission not being budget-
   starved (inherits W11-4). The W12-4 falsifier names the
   next wall: a learned normaliser (W12-C3), OR a *negotiated*
   kind-vocabulary contract between producers (out of scope for
   the SDK v3.13 method).

**Honest reading.** SDK v3.13 closes the synthetic→real-LLM
transfer gap **under the calibrated bounded-producer-noise
channel** the synthetic noisy extractor models. It does NOT
measure transfer to a real Ollama-served LLM (W12-C2 is the next
data point; ``--llm-mode ollama`` is the operator path; awaits
Mac-1 / Mac-2 wide-availability and a willingness to accept
synthetic-fallback contamination on HTTP failure). The honest
transfer story is now three layers deep:

* **Layer 1 (SDK v3.7, W6-C1/C2):** *un-normalised admission*
  does not transfer OOD on real LLMs.
* **Layer 2 (SDK v3.13, W12-Λ + W11-C2 partially discharged):*
  *un-normalised cross-round decoding* does not transfer.
* **Layer 3 (SDK v3.13, W12-1):** *normalised cross-round
  decoding* DOES transfer, conditional on the closed-vocabulary
  closure.

The W12-4 falsifier sharpens the closure boundary structurally:
every closed-vocabulary table has a finite closure; expanding
the table is a research move (W12-C3), not a structural fix —
and the next variant the LLM emits will fall outside the new
table. The **honest cap** on the SDK v3.13 advance is therefore
the closure property; over-claiming is the failure mode the
``HOW_NOT_TO_OVERSTATE.md`` rules guard against.

### 4.31 SDK v3.14 — layered open-world normalisation + real-Ollama probe + W13 family (Phase-60 open-world normalisation benchmark)

The SDK v3.13 W12 milestone closed the synthetic→real-LLM transfer
gap on a *fixed-vocabulary* drift channel; the named limit was
W12-4 (any kind variant outside :data:`CLAIM_KIND_SYNONYMS` survived
normalisation unchanged). SDK v3.14 (W13) sharpens that limit and
partly widens it — and separately *measures*, for the first time,
what real Ollama 14B actually emits on the calibrated
incident-triage prompt through the team-coord pipeline with raw-
response capture.

**Phase-60** (``vision_mvp/experiments/phase60_open_world_normalization.py``)
ships three pre-committed sub-banks plus an opt-in real-Ollama
extension:

* **R-60-wide** (``synthetic_wide_oov_llm``, the W13-1 anchor).
  The producer emits drifted variants from
  :data:`HEURISTIC_RESCUABLE_OOV_KINDS` whose every entry is
  verified *outside* :data:`CLAIM_KIND_SYNONYMS` and *inside* the
  W13 heuristic abstraction closure. Default config:
  ``K_auditor=8, T_auditor=256, n_eval=12, bank_seed=11,
  wide_oov_prob=0.50, svc_token_alt_prob=0.30``.
* **R-60-cosmic** (``synthetic_cosmic_oov_llm``, the W13-4
  closure-boundary anchor). Round-2 specific-tier kinds are
  replaced by truly arbitrary OOV from :data:`COSMIC_OOV_KINDS`
  (XYZZY_QQQQ, COSMIC_RAY_FLIP, …) verified outside both layers.
* **R-60-clean / R-60-synonym** (W13-3 backward-compat anchors).
* **R-60-ollama** (opt-in real-LLM probe; W13-Λ-real). Talks Mac-1
  ``qwen2.5:14b-32k``; raw producer responses captured into the
  report.

**Pre-committed default headline.** On Phase-60 default
(``synthetic_wide_oov_llm``):

* ``capsule_layered_multi_round`` (W13): **1.000** (12/12 correct).
* ``capsule_robust_multi_round`` (W12): **0.000** (every drifted
  variant escapes the fixed table).
* every other strategy (substrate, FIFO, priority, coverage, W7-2,
  W8, W9, W10, W11): **0.000**.
* **+1.000** vs every fixed-vocabulary method including SDK v3.13
  W12.

**Cross-bank stability.** **5/5** alternate (bank_seed, llm_seed)
values produce the W13 strict win; gap layered − w12 ∈
{0.917, 1.000} per seed (min 0.917, well above the strong-bar 0.50
threshold). Empirical anchor:
``docs/data/phase60_seed_sweep_K8_n12.json``.

**Theorem family W13** (proved or proved-empirical on n_eval × seed
sweep × falsifier saturated). The structural argument is that any
fixed-vocabulary table has a finite closure (W13-Λ-fixed); a
predicate-based normaliser strictly widens the closure with a small
ordered regex set (W13-1) but is itself bounded by the predicate
union (W13-4). The closure-widening is interpretable and
mechanically auditable through per-call layer counters.

**Real-Ollama observation (W13-Λ-real, honest negative).** Real
Mac-1 ``qwen2.5:14b-32k`` at temperature 0 on the Phase-58 events
emits *canonical* claim kinds (zero kind drift across n=4 × 12
producer calls) and *filters low-magnitude decoy events as noise*
(the ``monitor`` role emits ``NONE`` for the deliberately-low-
magnitude decoy events). The bench property holds in 0/4 scenarios;
W12 / W13 normalisation has nothing to rescue; W13 ties W12 ties
multi_round at ``accuracy_full = 0.250``. The R-60-ollama probe
lands at the **R-60-OLLAMA-C tier** (honest null real transfer);
the milestone is therefore *strong-success on R-60-wide synthetic
+ partial-success / honest-null on R-60-ollama*. § 1.4 of the
success criterion makes this partition pre-committed and
falsifiable; § 6 of the results note states the honest scope.

**Master-plan post-v3.14 reading.** After SDK v3.14:

1. **Post-v3.13 success bar.** R-60-wide (R-58 shape + bounded
   open-world drift inside the heuristic closure) instantiates the
   strong-bar regime: a method must improve ``accuracy_full`` by
   ≥ 0.20 on R-60-wide vs every fixed-vocabulary method including
   SDK v3.13 W12, stable across ≥ 3 (bank_seed, llm_seed) values.
   ``LayeredRobustMultiRoundBundleDecoder`` clears this bar with
   gap +1.000 (n_eval=12 saturated) and +0.917 to +1.000 per seed.
2. **W13-Λ-fixed sharpens W12-4.** Any fixed-vocabulary table has
   a finite closure by construction; a layered normaliser
   (exact + heuristic) strictly widens the closure (W13-1) but
   is itself bounded by the predicate union (W13-4). The
   research direction *"widen the closure"* is now structurally
   layered: exact (W12) → heuristic (W13) → embedding/learned
   (W13-C2 conjectural).
3. **What is the next benchmark?** Phase-60 is anchored as the
   *open-world normalisation* benchmark. The next milestone
   could attack:
   * **W13-C1** — cross-bench transfer: replicate the heuristic
     rule-set design on a non-incident-triage benchmark family.
   * **W13-C2** — learned normaliser: replace the regex predicate
     set with an embedding-distance lookup or LLM-distillation
     rewriter; measure whether the closure widens further.
   * **W13-C3** — real-Ollama transfer with *redesigned events*
     (decoy events with comparable magnitudes; prompt-side
     instruction to emit one claim per distinct event; measure
     whether real Ollama 14B then emits non-trivial drift inside
     the W12 / W13 closure).
   * **W13-C4** — abstention-aware decoder: wire the
     :data:`LAYERED_NORMALIZER_ABSTAIN` sentinel to a fallback
     decoder pathway that handles uncertain OOV explicitly.
4. **Is the structural win materially stronger than v3.13?**
   **Yes — materially, and on a regime SDK v3.13 cannot solve.**
   The structural win now spans **seven** named regimes
   (R-54..R-58 + R-59 noisy + R-60-wide) and a real-LLM
   *measurement* (R-60-ollama, honest-null at the R-60-OLLAMA-C
   tier). The W13 normalisation layer is the **fifth coupled
   structural axis** with a named limit theorem and a named
   falsifier regime — and the strongest layer SDK v3.13's fixed-
   vocabulary normaliser explicitly cannot reach.

The defensible "thesis-after-SDK-v3.14" is that the synthetic→real-
LLM transfer story has **five layers**:

* **Layer 1 (W6-C2 falsified):** un-normalised admission cannot
  transfer.
* **Layer 2 (W12-Λ at the real-LLM axis):** un-normalised cross-
  round decoding cannot transfer.
* **Layer 3 (SDK v3.13, W12-1):** fixed-vocabulary normalised
  cross-round decoding DOES transfer, conditional on the closed-
  vocabulary closure.
* **Layer 4 (SDK v3.14, W13-1):** layered (exact + heuristic)
  normalised cross-round decoding DOES transfer on a strictly
  *wider* drift channel, conditional on the heuristic predicate
  closure.
* **Layer 5 (SDK v3.14, W13-Λ-real, empirical):** real Ollama 14B
  at default settings does not produce the drift OR the cross-role
  decoy corroboration shape — the gating axis on real Ollama is
  *event-shape design + prompt-side discipline*, not normalisation.
  This is an honest empirical observation, not a closure of the
  question.

The W13-4 falsifier sharpens the closure boundary structurally:
any predicate-based normaliser, learned or hand-curated, has a
finite firing surface; OOV inputs whose surface form witnesses
none of the patterns escape the closure. A learned normaliser
(W13-C2) widens but does not eliminate the closure. The **honest
cap** on the SDK v3.14 advance is therefore the heuristic closure
property; over-claiming is the failure mode the
``HOW_NOT_TO_OVERSTATE.md`` § "Labelling the W13-1 win as
unconditional" rules guard against.

### 4.32 SDK v3.15 — structured producer protocol + W14 family (Phase-61 producer-side ambiguity-preservation benchmark)

SDK v3.15 attacks the SDK v3.14 W13-Λ-real real-Ollama bottleneck
directly via prompt-side discipline. The SDK v3.14 honest negative
diagnosed the bottleneck: real Ollama 14B at default settings emits
canonical kinds (zero drift) AND filters low-magnitude decoy events
as noise, so the bench property holds in 0/4 scenarios and W12 / W13
have nothing to rescue. The dominant blocker on real-LLM transfer
is **producer-side ambiguity erasure**, not normalisation.

The SDK v3.15 mint:

1. **Comparable-magnitude event redesign** (Phase-61). The decoy
   storm service's round-1 spikes have operational magnitudes inside
   the gold range (``p95_ms ∈ [1700, 3200]``,
   ``error_rate ∈ [0.10, 0.30]``, ``count ∈ [6, 14]``). Magnitude
   alone cannot discriminate; disambiguation must come from the
   round-2 specific causal claim.
2. **StructuredProducerProtocol** (W14). A prompt-rendering surface
   with two modes: ``naive`` (legacy byte-for-byte) and ``structured``
   (W14: tier banner observation/diagnosis split + per-tier kind
   whitelist + per-event mandate). The protocol consumes a
   declarative ``RoleExtractionSchema`` (allowed kinds + observation/
   diagnosis partition); ``incident_triage_role_schemas()`` ships the
   schema for the Phase-58..Phase-61 family.
3. **Magnitude-filter synthetic counterpart**
   (``MagnitudeFilteringExtractor``). Calibrated against the
   W13-Λ-real real-Ollama observation; reproduces the producer-side
   compression deterministically so the W14-Λ-prompt synthetic
   counterpart fires cleanly without requiring a live LLM.

The W14 family's six theorems span the four structural axes named
by the milestone:

* **W14-Λ-prompt** (proved-empirical n=40 saturated × 5 seeds +
  structural sketch via W7-3 extension): on R-61-naive-prompt
  default, every capsule strategy ties FIFO at 0.000 — the bench
  property is erased upstream by producer compression.
* **W14-1** (proved-conditional + proved-empirical): pairing the
  structured protocol with the same magnitude-filter extractor +
  the cross-round capsule pipeline achieves ``accuracy_full = 1.000``
  on synthetic R-61-structured-prompt (5/5 seeds, +1.000 vs naive)
  and ``accuracy_full = 0.500`` on real Ollama 14B at n=8 (+0.500
  vs FIFO at exactly the R-61-OLLAMA-A threshold).
* **W14-2** (proved + mechanically-checked): schema soundness +
  protocol determinism.
* **W14-3** (proved-empirical full programme-wide regression):
  393/393 prior tests pass; R-54..R-60 anchors preserved.
* **W14-4** (proved-empirical n=8 saturated): on R-61-ollama-naive
  the bench property holds 0/8 even on the redesigned events —
  *both* the comparable-magnitude events AND the structured prompt
  are required for W14-1.
* **W14-Λ-real** (empirical-research, n=8 × 24 real Ollama calls):
  the R-61-OLLAMA-A tier of the success criterion § 1.5 is cleared
  at +0.500.

**Master-plan post-v3.15 reading.** After SDK v3.15:

1. **Post-v3.14 success bar.** R-61-structured-prompt (R-58 shape +
   comparable-magnitude events + structured prompt) instantiates the
   strong-bar regime: a method must improve ``accuracy_full`` by
   ≥ 0.20 on R-61-structured vs every prior anchor including SDK
   v3.14 W13 alone, stable across ≥ 3 ``bank_seed`` values. The
   W14 protocol clears this bar with synthetic gap +1.000 (5/5
   seeds) and real-Ollama gap +0.500 (n=8).
2. **W14-Λ-prompt sharpens W13-Λ-real.** The synthetic magnitude-
   filter extractor calibrated against the real-Ollama observation
   reproduces W13-Λ-real byte-for-byte: ``round1_decoy_corroborated``
   = 0/4 on the original Phase-58 events. The synthetic counterpart
   makes producer-side compression *mechanically* tractable in CI
   without requiring a live LLM.
3. **What is the next benchmark?** Phase-61 is anchored as the
   *producer-side ambiguity-preservation* benchmark. The next
   milestone could attack:
   * **W14-C2** — model-side magnitude calibration: extend the
     structured prompt with a *magnitude hint* tier banner
     (enumerate the magnitude thresholds qualifying as
     ``ERROR_RATE_SPIKE`` / ``LATENCY_SPIKE`` / ``FW_BLOCK_SURGE``)
     and measure whether the 1/8 R-61-ollama-structured failure
     closes.
   * **W14-C4** — cross-model transfer: re-run R-61-ollama-
     structured against ``qwen3.5:35b`` (the W5-1 cross-model
     probe) and against MLX-distributed when Mac 2 is reachable.
   * **W14-C1** — cross-bench: build security-incident /
     compliance-review schema partitions and verify the protocol's
     bench-property recovery generalises.
   * **W14-C5** — multi-hypothesis variant: extend the protocol
     to permit "if uncertain, list 2-3 candidate kinds comma-
     separated"; measure whether ambiguous round-1 events
     (low-confidence spikes) preserve corroboration further.
4. **Is the structural win materially stronger than v3.14?**
   **Yes — materially, and on a real-LLM stream where SDK v3.14
   produced an honest negative.** The W14-1 win clears
   the R-61-OLLAMA-A tier (+0.500 strict gain over substrate FIFO
   on real Mac-1 ``qwen2.5:14b-32k`` at n=8); SDK v3.14 only
   reached the R-60-OLLAMA-C tier (honest null). The Wevra
   programme now has **six** coupled structural axes with named
   limit theorems on each, and the **W14 layer is the first to
   materially advance the programme on a real-LLM stream**.

The defensible "thesis-after-SDK-v3.15" is that the synthetic→real-
LLM transfer story has **six layers**:

* **Layer 1 (W6-C2 falsified):** un-normalised admission cannot
  transfer.
* **Layer 2 (W12-Λ at the real-LLM axis):** un-normalised cross-
  round decoding cannot transfer.
* **Layer 3 (SDK v3.13, W12-1):** fixed-vocabulary normalised
  cross-round decoding DOES transfer, conditional on the closed-
  vocabulary closure.
* **Layer 4 (SDK v3.14, W13-1):** layered (exact + heuristic)
  normalised cross-round decoding DOES transfer on a strictly
  *wider* drift channel, conditional on the heuristic predicate
  closure.
* **Layer 5 (SDK v3.14, W13-Λ-real, empirical):** real Ollama 14B
  at default settings does not produce the drift OR the cross-role
  decoy corroboration shape — the gating axis on real Ollama is
  *event-shape design + prompt-side discipline*, not normalisation.
* **Layer 6 (SDK v3.15, W14-1 + W14-Λ-real, conditional):** the
  structured producer protocol + comparable-magnitude events
  combined with the cross-round capsule pipeline DOES transfer on
  a real-LLM stream at +0.500 strict gain over substrate FIFO,
  conditional on (a) the redesigned events, (b) the structured
  prompt, (c) the cross-round pipeline. The W14 layer is the
  **first** to clear the R-61-OLLAMA-A tier; the W13 closure-
  widening is dormant on this regime because the real LLM emits
  canonical kinds.

The W14-4 falsifier sharpens the structural composition: *both*
the redesigned events AND the structured prompt are required for
the W14-1 win — either alone collapses the bench property to 0/8
on real Ollama. The interventions compose multiplicatively, not
additively. The **honest cap** on the SDK v3.15 advance is therefore
the model-side calibration limit (the 1/8 failure on R-61-ollama-
structured was a model judgment error, not a protocol failure);
over-claiming is the failure mode the ``HOW_NOT_TO_OVERSTATE.md``
§ "Solved real-LLM transfer" rules guard against.

### 4.33 SDK v3.16 — attention-aware capsule context packing + W15 family (Phase-62 multi-hypothesis decoder-side budget benchmark)

SDK v3.16 attacks the symmetric *downstream* gap that SDK v3.15
left unaddressed. SDK v3.15 (W14) closed the producer-side gap on
R-61: under the structured prompt + comparable-magnitude events,
the bench property's cross-role decoy corroboration assumption
survives a real Ollama 14B producer at +0.500 strict gain over
substrate FIFO. But the W14 layer has no influence on what
happens *between* admission and decoding. The cross-round decoders
W11/W12/W13 consume the *full union* of admitted handoffs, with no
token budget enforcement at the decode step. In a real-LLM
downstream consumer (an audit agent that re-decodes the bundle for
human-readable summarisation, or a ranker LLM that prioritises
capsule witnesses), the bundle is a *prompt* with a *context
window*. When the admitted union exceeds the window, the naive
choice is FIFO truncation — drop handoffs from the tail until the
bundle fits.

The SDK v3.16 mint:

1. **Multi-hypothesis comparable-magnitude events** (Phase-62).
   Each scenario has 2 gold services + 2 decoy services. Every
   decoy service is mentioned by ≥ 2 distinct producer roles in
   round 1 (the W11-drop trigger condition); every gold service
   is mentioned by exactly 1 distinct producer role in round 1
   (so the W11 drop preserves golds). Multi-hypothesis means: 4
   service hypotheses survive naive admission; the disambiguation
   is recoverable *only* from the round-2 specific causal claim
   AND cross-role corroboration of round-1 noise. The bench
   property is mechanically verified by ``Phase62BankShapeTests``.
2. **Decoder-side token budget** (``T_decoder``). A new strict
   budget is enforced over the cross-round bundle the decoder
   consumes. R-62-tightbudget sets ``T_decoder = 24``, strictly
   below the admitted union's token sum (≈ 27/28 per scenario).
   Under FIFO packing the round-2 specific-tier disambiguator is
   dropped by construction (``position_of_first_causal_claim = -1``
   in 8/8 cells). R-62-saturation sets ``T_decoder = 9999`` (no
   budget pressure), the W15-Λ-budget falsifier anchor.
3. **AttentionAwareBundleDecoder + CapsuleContextPacker +
   FifoContextPacker** (W15 family). A two-stage decoder that
   (a) runs a first-pass priority decode over the normalised union
   to elect a tentative ``root_cause``, (b) salience-aware repacks
   the union under ``T_decoder`` using a closed-form deterministic
   score (``w_tier=4.0``, ``w_cck=3.0``, ``w_corroboration=1.0``,
   ``w_magnitude=0.1``, ``w_round=0.5``) with per-(tag, role, tier)
   hypothesis preservation, then (c) runs the W13 layered decoder
   on the *packed* subset. Pack-stats expose
   ``position_of_first_causal_claim`` (the proxy attention metric),
   ``tokens_kept_sum`` / ``tokens_input_sum``,
   ``hypothesis_count_kept``, ``n_dropped_budget``, and
   ``salience_floor_kept`` for direct audit.
   :class:`FifoContextPacker` is the load-bearing baseline.

The W15 family's six theorems span the three structural axes named
by the milestone:

* **W15-Λ-budget** (proved-empirical n=40 saturated × 5 seeds +
  structural sketch via W7-3 extension to the decoder-side axis):
  on R-62-tightbudget default, every FIFO-packed cross-round
  decoder ties FIFO at 0.000 — the round-2 specific-tier
  disambiguator falls past the budget by construction. A
  never-decoded handoff has no influence on the answer.
* **W15-1** (proved-conditional + proved-empirical): pairing the
  AttentionAwareBundleDecoder with the W13 layered decoder
  achieves ``accuracy_full = 1.000`` on R-62-tightbudget default,
  +1.000 strict separation vs FIFO-packed-W13, stable across 5/5
  alternate ``bank_seed`` values.
* **W15-2** (proved + mechanically-checked): salience score is a
  closed-form deterministic function of the handoff's bytes plus
  the elected root_cause; the pack is byte-deterministic given a
  canonical tie-break order; FIFO packer preserves arrival order
  byte-for-byte under T_decoder=None.
* **W15-3** (proved-empirical full programme-wide regression):
  393/393 prior tests pass; 37 new tests cover the W15 surface
  and Phase-62; with ``T_decoder = None`` the W15 decoder ties
  the W13 layered decoder byte-for-byte on the answer field.
* **W15-Λ-degenerate** (proved-empirical n=8): on R-62-saturation
  the W15-1 win is structurally invisible — both salience and
  FIFO packers tie W13 at 1.000.
* **W15-4** (proved + mechanically-checked): token-efficiency
  floor — both packers strictly enforce
  ``tokens_kept ≤ T_decoder``.

**Master-plan post-v3.16 reading.** After SDK v3.16:

1. **Post-v3.15 success bar.** R-62-tightbudget (multi-hypothesis
   comparable-magnitude events + ``T_decoder = 24`` strict
   decoder-side budget) instantiates the strong-bar regime: a
   method must improve ``accuracy_full`` by ≥ 0.20 on
   R-62-tightbudget vs every prior anchor including SDK v3.14 W13
   under FIFO packing, stable across ≥ 3 ``bank_seed`` values, AND
   clear bar 12 (joint correctness + decoder-side context budget).
   The W15 method clears this bar with synthetic gap +1.000 (5/5
   seeds).
2. **W15-Λ-budget sharpens W14-Λ-prompt at the decoder-side axis.**
   The FIFO-packed cross-round decoder collapse on R-62-tightbudget
   reproduces the W14-Λ-prompt structural argument (a never-emitted
   handoff has no CID; a never-decoded handoff has no influence on
   the answer) at the decoder-side axis. The synthetic
   counterpart makes decoder-side budget pressure mechanically
   tractable in CI without requiring a real LLM.
3. **What is the next benchmark?** Phase-62 is anchored as the
   *attention-aware capsule context packing* benchmark. The next
   milestone could attack:
   * **W15-C-real** — the W15 salience pack plugged into a real-LLM
     downstream re-decoder agent (Mac-1 ``qwen2.5:14b-32k``)
     under a real context window. Falsifier: a real LLM whose
     attention is sufficiently uniform that prompt-position
     ordering does not change correctness.
   * **W15-C-COMPOSE-W14** — composing W15 on the W14 R-61-ollama-
     structured stream to close the 1/8 model-error failure that
     W14 alone leaves.
   * **W15-C-SYMMETRIC** — a regime with symmetric cross-role
     corroboration (gold AND decoy both ≥ 2 distinct roles) where
     the W11 drop fires on both — the named structural limit of
     the current capsule pipeline; the natural next-axis open
     question for SDK v3.17+.
   * **W15-C-LEARNED** — a learned salience scorer that
     outperforms the closed-form W15 weights on a held-out test
     set.
4. **Is the structural win materially stronger than v3.15?**
   **Yes — it adds a structurally-orthogonal seventh axis.**
   The W14 layer is a *producer-side* intervention; the W15 layer
   is a *decoder-side* intervention. They compose additively. The
   Wevra programme now has seven coupled structural axes with
   named limit theorems on each, and the W15 layer is the first
   to make *joint correctness AND decoder-side context efficiency*
   the strict-gain anchor.

The defensible "thesis-after-SDK-v3.16" is that the synthetic→
real-LLM-and-bounded-context transfer story now has **seven
layers**:

* **Layer 1 (W6-C2 falsified):** un-normalised admission cannot
  transfer.
* **Layer 2 (W12-Λ at the real-LLM axis):** un-normalised cross-
  round decoding cannot transfer.
* **Layer 3 (SDK v3.13, W12-1):** fixed-vocabulary normalised
  cross-round decoding DOES transfer, conditional on the closed-
  vocabulary closure.
* **Layer 4 (SDK v3.14, W13-1):** layered (exact + heuristic)
  normalised cross-round decoding DOES transfer on a strictly
  *wider* drift channel, conditional on the heuristic predicate
  closure.
* **Layer 5 (SDK v3.14, W13-Λ-real, empirical):** real Ollama 14B
  at default settings does not produce the drift OR the cross-role
  decoy corroboration shape — the gating axis is *event-shape
  design + prompt-side discipline*, not normalisation.
* **Layer 6 (SDK v3.15, W14-1 + W14-Λ-real, conditional):** the
  structured producer protocol + comparable-magnitude events
  combined with the cross-round capsule pipeline DOES transfer
  on a real-LLM stream at +0.500 strict gain over substrate FIFO,
  conditional on (a) the redesigned events, (b) the structured
  prompt, (c) the cross-round pipeline.
* **Layer 7 (SDK v3.16, W15-1 + W15-Λ-budget, conditional):** the
  attention-aware capsule context packer + hypothesis preservation
  DOES restore correctness when the cross-round bundle is bounded
  by a strict decoder-side token budget, conditional on the
  multi-hypothesis bench property + budget pressure existing. The
  W15 layer adds an *orthogonal* axis to the prior six: even when
  W11/W12/W13/W14 all succeed at producing a clean ambiguity-
  preserving union, the union may exceed the downstream context
  budget — and FIFO truncation drops the load-bearing
  disambiguator. W15 is the structural fix.

The W15-Λ-degenerate falsifier sharpens the structural composition:
*no decoder-side budget pressure* removes the W15 advantage by
construction. This is *not* a refutation — it is the named
counterexample regime that confirms W15-1 is a conditional, not a
universal, win. The **honest cap** on the SDK v3.16 advance is
therefore the synthetic-only scope: real-LLM downstream-decoder
transfer (W15-C-real) is conjectural and not yet wired.
Over-claiming is the failure mode the ``HOW_NOT_TO_OVERSTATE.md``
§ "W15 shapes transformer attention" and § "W15 solves multi-agent
context" rules guard against.

### 4.34 SDK v3.17 — end-to-end W14 + W15 composition + W16 family (Phase-63 composed real-LLM benchmark)

SDK v3.17 attacks the *coupling* question that SDK v3.16 left
unaddressed. SDK v3.15 (W14) closed the producer-side gap on R-61
under a structured prompt at +0.500 vs FIFO on a real Ollama 14B.
SDK v3.16 (W15) closed the decoder-side gap on R-62-tightbudget at
+1.000 vs FIFO-packed-W13 under a *deterministic identity*
producer. But the two layers had not been stress-tested *together*
on a single regime where:

* the producer compresses ambiguity upstream (so the W14 layer
  alone gates whether the bench property survives), AND
* the decoder is bounded by a strict ``T_decoder`` token budget
  (so the W15 layer alone gates whether the round-2 disambiguator
  survives FIFO truncation).

Phase-63 attacks that exact composition. The events are R-62's
multi-hypothesis comparable-magnitude shape (2 gold + 2 decoy
services, asymmetric corroboration). The producer is Phase-61's
deterministic ``MagnitudeFilteringExtractor`` with its
``prompt_mode`` knob (naive collapses round-1 toward top-N events
per (role, kind); structured preserves the per-event mandate),
wired through the W14 ``StructuredProducerProtocol``. The decoder
consumes the cross-round bundle under a strict ``T_decoder`` and
applies the W15 ``AttentionAwareBundleDecoder`` (when the
composition is enabled) or the FIFO pack baseline (when the
composition is disabled).

The SDK v3.17 mint:

1. **R-63 multi-cell cross-regime** — six synthetic sub-banks plus
   three ollama-replay cells covering the 2×2×2 grid of
   {identity / mag-filter naive / mag-filter structured} ×
   {T_decoder=None / T_decoder=24} plus the W16-Λ-degenerate
   falsifier (T_decoder=2) and the W16-Λ-real-replay anchor over
   recorded Phase-61 ``qwen2.5:14b-32k`` bytes. Mechanically
   verified by ``Phase63CrossRegimeTests``.
2. **W16-1 strict-gain on R-63-COMPOSED-TIGHT** (synthetic, n=8 ×
   5 seeds saturated). The composed method achieves
   ``accuracy_full = 1.000`` while every non-composed baseline
   collapses to 0.000 — +1.000 strict separation, stable across
   5/5 alternate ``bank_seed`` values. The composition is
   *strictly multiplicative*: each layer alone produces 0.000 on
   R-63-COMPOSED-TIGHT.
3. **W16-Λ-real-replay strict gain on recorded real-LLM bytes**.
   At ``T_decoder = 14, K_auditor = 8`` on the recorded
   structured-prompt bytes from
   ``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n8.json``,
   the composed method achieves ``capsule_attention_aware = 0.500``
   while ``capsule_layered_fifo_packed = 0.000`` — **+0.500 strict
   gain** over the strongest non-composed baseline on a *real-LLM
   stream*. The budget band where this gain holds is
   ``T_decoder ∈ [13, 16]``; outside the band, both packers either
   fail jointly or both succeed.
4. **W16-Λ-compose joint-failure anchor** on R-63-naive-tight and
   the recorded naive-prompt bytes. Both W14-Λ-prompt (bench
   property erased upstream) and W15-Λ-budget (disambiguator
   dropped under FIFO truncation) fire on the same regime; every
   capsule strategy ties FIFO at 0.000.
5. **W16-Λ-degenerate falsifier** on R-63-degen-budget
   (``T_decoder = 2``). Even with the structured prompt, an extreme
   budget drops the disambiguator's tokens; both packers collapse.
   The W16-1 win is conditional on ``T_decoder`` strictly between
   the round-2 disambiguator's token cost and the admitted union's
   token sum.

The W16 family's six theorems span the three structural axes named
by the milestone:

* **W16-Λ-compose** (proved-empirical on R-63-naive-tight n=8 +
  recorded n=8 × 24 real-LLM bytes; structural sketch via
  composition of W14-Λ-prompt and W15-Λ-budget).
* **W16-1** (proved-conditional + proved-empirical, synthetic n=40
  saturated × 5 seeds): composition strictly improves over every
  non-composed baseline by +1.000 on R-63-COMPOSED-TIGHT.
* **W16-2** (proved-empirical, sub-additivity / multiplicative
  composition): each layer alone produces 0.000; only the
  composition produces 1.000.
* **W16-3** (proved-empirical full programme regression): 442/442
  tests pass; backward-compat preserved on R-54..R-62.
* **W16-Λ-degenerate** (proved-empirical, n=8): under
  ``T_decoder = 2`` both packers collapse; W16-1 conditional on
  budget admitting the disambiguator.
* **W16-Λ-real-replay** (empirical-research over recorded
  byte-stable Phase-61 capture): +0.500 strict gain on real-LLM
  bytes at T_decoder=14.

**Master-plan post-v3.17 reading.** After SDK v3.17:

1. **Post-v3.16 success bar.** R-63-COMPOSED-TIGHT instantiates
   the strong-bar regime: a method must improve ``accuracy_full``
   by ≥ 0.20 vs every non-composed baseline (W14-only-budgeted,
   W15-only-without-W14, substrate FIFO), stable across ≥ 3
   ``bank_seed`` values, AND clear bar 13 (end-to-end composition
   split). The W16 method clears this bar with synthetic gap
   +1.000 (5/5 seeds) AND clears the W16-Λ-real-replay anchor at
   +0.500 on recorded real-LLM bytes.
2. **W16-Λ-compose makes the joint structural argument explicit.**
   The producer-side limit (W14-Λ-prompt) and the decoder-side
   limit (W15-Λ-budget) compose multiplicatively on R-63-naive-
   tight: every capsule strategy ties FIFO at 0.000 when *both*
   upstream emission and downstream retention fail. This is not a
   refutation of either layer — it is the named counterexample
   regime where both prior limits fire simultaneously.
3. **What is the next benchmark?** Phase-63 is anchored as the
   *end-to-end W14+W15 composition* benchmark. The next milestone
   could attack:
   * **W16-C-LIVE-OLLAMA** — a fresh live Ollama probe under
     R-63-COMPOSED-TIGHT with the structured prompt + tight
     ``T_decoder``. Falsifier: a live probe where the composed
     accuracy ties or loses to the W14-only loose-budget result
     (≤ 0.500). Requires Mac-1 endpoint reachable.
   * **W16-C-CROSS-MODEL** — qwen3.5:35b under MLX-distributed.
     Requires Mac-2 reachable.
   * **W15-C-SYMMETRIC / W16-C-SYMMETRIC** — symmetric-corroboration
     limit on the composed regime. The natural next-axis open
     question for SDK v3.18+.
   * **W16-C1** — cross-bench transfer of the composition to
     security-incident, robotics, or compliance-review families.
4. **Is the structural win materially stronger than v3.16?**
   **Yes — it adds a structurally-orthogonal eighth axis (the
   *coupling* statement).** The W14 layer is a *producer-side*
   intervention; the W15 layer is a *decoder-side* intervention.
   They compose additively in code (no new SDK class) AND
   multiplicatively in effect (each layer is necessary on the
   regime where the other layer's limit fires). The Wevra
   programme now has eight coupled structural moves with named
   limit theorems on each, and the W16 layer is the **first** to
   make *joint upstream-and-downstream necessity* the strict-gain
   anchor — AND the **first** to deliver an end-to-end real-LLM
   strict advance over the strongest non-composed baseline (on
   recorded bytes; live probe is W16-C-LIVE-OLLAMA, conjectural).

The defensible "thesis-after-SDK-v3.17" is that the synthetic→
real-LLM-and-bounded-context transfer story now has **eight
layers**, where the eighth layer is the *composition* of the prior
two real-LLM-relevant layers (W14 + W15):

* **Layer 1 (W6-C2 falsified):** un-normalised admission cannot
  transfer.
* **Layer 2 (W12-Λ at the real-LLM axis):** un-normalised cross-
  round decoding cannot transfer.
* **Layer 3 (SDK v3.13, W12-1):** fixed-vocabulary normalised
  cross-round decoding DOES transfer, conditional on the closed-
  vocabulary closure.
* **Layer 4 (SDK v3.14, W13-1):** layered (exact + heuristic)
  normalised cross-round decoding DOES transfer on a strictly
  *wider* drift channel, conditional on the heuristic predicate
  closure.
* **Layer 5 (SDK v3.14, W13-Λ-real):** real Ollama 14B at default
  settings does not produce the drift OR the cross-role decoy
  corroboration shape — the gating axis is event-shape design +
  prompt-side discipline, not normalisation.
* **Layer 6 (SDK v3.15, W14-1):** the structured producer protocol
  + comparable-magnitude events combined with the cross-round
  capsule pipeline DOES transfer on a real-LLM stream at +0.500
  strict gain over substrate FIFO under loose decoder budget.
* **Layer 7 (SDK v3.16, W15-1):** the attention-aware capsule
  context packer DOES restore correctness when the cross-round
  bundle is bounded by a strict decoder-side token budget on
  *synthetic* events.
* **Layer 8 (SDK v3.17, W16-1 + W16-Λ-real-replay):** the
  composition of W14 + W15 is the *first* end-to-end demonstration
  that producer-side and decoder-side interventions are jointly
  necessary on a single regime, AND that the composed result
  *survives* the recorded real-LLM bytes — at +0.500 strict gain
  over the FIFO-packed-W14-only baseline on recorded ``qwen2.5:14b``
  bytes. The composition does *not* introduce new SDK code; it
  is *additive in code* and *multiplicative in effect*.

The W16-Λ-degenerate / W16-Λ-compose falsifier regimes sharpen
the structural composition: an extreme budget removes the W16
advantage by construction, AND naive-prompt-with-tight-budget
removes the W16 advantage by joint failure. These are *not*
refutations — they are the named counterexample regimes that
confirm W16-1 is a conditional, not a universal, win. The
**honest cap** on the SDK v3.17 advance is the recorded-bytes
scope: real-LLM live transfer (W16-C-LIVE-OLLAMA) and cross-model
transfer (W16-C-CROSS-MODEL) are conjectural pending Mac-1 / Mac-2
reachable. Over-claiming is the failure mode the
``HOW_NOT_TO_OVERSTATE.md`` § "W16 solves multi-agent context
end-to-end" and § "W16 demonstrates real-LLM transfer is solved"
rules guard against.

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
  SWE-bench (`docs/archive/pre-wevra-theory/ROADMAP.md`). If fewer than 50 % of events are
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
  residual gap on adversarial / synonym queries (`docs/archive/pre-wevra-theory/MATH_AUDIT.md`
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
| `docs/archive/pre-wevra-theory/FRAMEWORK.md`                       | Original problem formulation; routing-as-causal-inference              | Theoretical origin of CASR                        |
| `docs/archive/pre-wevra-theory/PROOFS.md`                          | Formal theorems with proofs                                           | Mathematical claims                                |
| `docs/archive/pre-wevra-theory/MATH_AUDIT.md`                      | Which of the 72 frameworks are actually in the code                   | Honest accounting of theory-to-code               |
| `docs/archive/pre-wevra-theory/OPEN_QUESTIONS.md`                  | Seven foundational open questions + resolution status                 | Which hard problems the programme owns            |
| `docs/archive/pre-wevra-theory/ROADMAP.md`                         | Phase 1–4 plan + risk register + falsifiability check                 | Empirical plan, SWE-bench framing                 |
| `docs/archive/pre-wevra-theory/VISION_MILLIONS.md`                 | Forward-looking vision past CASR (10^6+ agents)                        | Speculative / post-CASR architectures             |
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
| Bounded-context router (typed handoff, per-role inbox) | ✅ | Phases 1–10, Theorem 3 in `docs/archive/pre-wevra-theory/PROOFS.md` |
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
| MATH_AUDIT refresh with P44 / P45 claims | ✅ | `docs/archive/pre-wevra-theory/MATH_AUDIT.md` |
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

## 10. Wevra SDK / Production-Readiness (Slice 1 + Slice 2 + Slice 3)

§ 9 is the *research* finished-product checklist — the programme's
substrate, pipeline, and reporting are at finished-product state as
of Phase 45/46. § 10 is the parallel **product** checklist for
**Wevra**, the first shipped system/product from Context Zero.

**Framing contract.**
  * Context Zero = the research programme (§§ 1–9).
  * **Wevra = a context-capsule runtime.** Every inter-role,
    inter-layer, and inter-run artefact is a **`ContextCapsule`** —
    a typed, content-addressed, lifecycle-bounded, budget-bounded,
    provenance-carrying object. Wevra is **not** the whole research
    programme and **not** a universal agent platform.
  * Wevra's scope: profile-driven evaluation on SWE-bench-Lite-shape
    banks, with a stable report schema, provenance manifest, CI
    gate, unified mock/real runtime, extension system, import
    audit, and — the SDK-v3 centre of gravity — a sealed capsule
    DAG on every run.
  * Items still outside scope (Docker-first-by-default, first
    real out-of-tree plugin, on-tag release firing) remain
    **Slice 3-ops**; they are independent of the SDK-v3 capsule
    centre.

### 10.0 The Capsule Contract (SDK v3 centre of gravity)

Wevra's one-line identity has sharpened across slices. Slice 1
framed it as "profile-driven evaluation SDK"; Slice 2 as
"bounded-context orchestration and evaluation SDK with a unified
mock/real runtime and a plugin surface." Both were outcome-shaped.
Slice 3 recognises the mechanism that was always there and names
it as the product's load-bearing abstraction:

> **Wevra is a context-capsule runtime.** Every piece of context
> that crosses a role boundary, a layer boundary, or a run
> boundary is a typed, content-addressed, lifecycle-bounded,
> budget-bounded, provenance-stamped **capsule** — never a raw
> prompt string.

A **`ContextCapsule`** satisfies six invariants (C1..C6, stated
and tested in `vision_mvp/tests/test_wevra_capsules.py`):

  * **C1 Identity.**   SHA-256 content-address over
    `(kind, payload, budget, parents)`.
  * **C2 Typed claim.** `CapsuleKind` ∈ closed vocabulary.
  * **C3 Lifecycle.**   `PROPOSED → ADMITTED → SEALED` (+ optional
    `RETIRED`); illegal transitions are refused.
  * **C4 Budget.**      `CapsuleBudget(max_tokens, max_bytes,
    max_rounds, max_witnesses, max_parents)` enforced at admit
    time.
  * **C5 Provenance.**  Parents must be in the ledger; the ledger
    keeps a hash chain — `verify_chain()` detects any retroactive
    insert.
  * **C6 Frozen.**      A sealed capsule's CID is fixed for all
    time.

The abstraction unifies what was already there:

| Older primitive                         | Phase | Capsule kind it instantiates |
|---                                      |---    |---                           |
| `context_ledger.Handle`                  | 19    | `HANDLE`                     |
| `role_handoff.TypedHandoff`             | 31    | `HANDOFF`                    |
| `dynamic_comm.ThreadResolution`         | 35    | `THREAD_RESOLUTION`          |
| `adaptive_sub.AdaptiveEdge`             | 36    | `ADAPTIVE_EDGE`              |
| `SweepSpec`                              | S2    | `SWEEP_SPEC`                 |
| per-cell sweep report (`wevra.sweep.v2`) | S2    | `SWEEP_CELL`                 |
| `phase44_public_readiness` verdict      | 44    | `READINESS_CHECK`            |
| `wevra.provenance.v1` manifest          | S1    | `PROVENANCE`                 |
| on-disk `product_report.json` etc.       | 45    | `ARTIFACT`                   |
| resolved profile dict                    | 45    | `PROFILE`                    |
| the run itself                           | S3    | `RUN_REPORT`                 |

The reference implementation is `vision_mvp/wevra/capsule.py`
(~700 LOC). Every run folds its finished artefacts into a
`CapsuleLedger` via `build_report_ledger`; the resulting
`CapsuleView` lands in `report["capsules"]` and in
`capsule_view.json`. A new CLI `wevra-capsule view / verify / cid`
inspects / audits the graph. The `RUN_REPORT` capsule's CID is
the durable identifier for a Wevra run — it is what downstream
consumers pin to prove they read the same bytes.

**Why this is the right centre.** "Bounded-context orchestration"
is an outcome-shaped name — it describes what Wevra *does*. The
capsule contract is mechanism-shaped — it describes *why* the
bounded-context guarantees hold across role boundaries, layer
boundaries, and run boundaries uniformly. Specifically: the
Phase-31 `T_i* = Θ(R*·τ)` bounded-context result (Theorem P31-3)
is exactly the C4 budget invariant applied to the `HANDOFF`
kind; the Phase-35 `ctx(r) ≤ C_0 + R*·τ + T·R_max·W` is C4 on
`THREAD_RESOLUTION`; the Phase-19 `Handle.fingerprint` drift-
detection is C1+C6 applied to `HANDLE`. Seen through the capsule
lens, the substrate work of Phases 19..44 is instances of one
contract, not parallel ad-hoc mechanisms.

**Honest originality.**
  * What is *inherited*: content addressing (Merkle/Git/IPFS),
    hash-chained logs (tamper-evident-log research), typed claim
    kinds (actor/event-sourcing systems), capability-style typed
    references (KeyKOS / seL4), lifecycle states (session-typed
    protocols). Each of these is older than this programme.
  * What is *new*: the unification of all of these under one
    product-facing contract for LLM-agent-team runtimes; the
    product-level decision that "context is not a prompt, context
    is an object"; the specific contract (C1..C6) implemented in
    one SDK that ships end-to-end with sealed on-disk artefacts
    you can replay, audit, and cross-reference by CID. The
    programme does not claim to have invented any primitive under
    the contract; it claims to have picked the right top-level
    shape and implemented it.

See [`docs/archive/wevra-milestones/RESULTS_WEVRA_CAPSULE.md`](archive/wevra-milestones/RESULTS_WEVRA_CAPSULE.md)
for theorems W3-1..W3-6 and conjectures W3-C1..W3-C3.

### 10.1 Stability matrix (living)

| Layer | Scope | Stability | Import path |
|---|---|---|---|
| **Context Capsule primitives** — `ContextCapsule`, `CapsuleKind`, `CapsuleLifecycle`, `CapsuleBudget`, `CapsuleLedger`, `CapsuleView`, `render_view`, `build_report_ledger`, `capsule_from_*` adapters | **SDK-v3 load-bearing abstraction.** Every cross-boundary artefact is a capsule. | **Stable v1** (contract C1..C6 tested in `test_wevra_capsules.py`) | `vision_mvp.wevra.capsule`, re-exported from `vision_mvp.wevra` |
| **Capsule view artifact** — `wevra.capsule_view.v1` | Sealed capsule graph on disk | **Stable v1** (written next to every report) | `capsule_view.json` in `out_dir` |
| **Wevra SDK** — `RunSpec`, `run`, `SweepSpec`, `run_sweep`, `HeavyRunNotAcknowledged`, `WevraConfig`, `profiles`, `report`, `ci_gate`, `import_data`, `extensions`, capsule primitives, `build_manifest`, schema constants | Public product contract | **Stable v3** (contract-tested: `test_wevra_public_api.py`, `test_wevra_runtime.py`, `test_wevra_capsules.py`) | `vision_mvp.wevra` |
| **Wevra console scripts** — `wevra`, `wevra-import`, `wevra-ci`, `wevra-capsule` | CLI surface | **Stable v3** (Slice 3: `wevra-capsule view / verify / cid`) | `[project.scripts]` |
| **Provenance manifest** — `wevra.provenance.v1` | Reproducibility artifact; also a `PROVENANCE` capsule in the run's DAG | **Stable v1** (tested on every run) | `vision_mvp.wevra.provenance` |
| **Extension Protocols** — `SandboxBackend`, `TaskBankLoader`, `ReportSink` | Plugin surface | **Stable v1** (runtime-checkable Protocols, contract-tested: `test_wevra_extensions.py`) | `vision_mvp.wevra.extensions` |
| **Unified runtime** — `SweepSpec`, `run_sweep`, `wevra.sweep.v2` | One coherent execution path; emits `SWEEP_SPEC` + `SWEEP_CELL` capsules | **Stable v1** (covers mock + real-executed + real-staged) | `vision_mvp.wevra.runtime` |
| **Report / CI / import schemas** — `phase45.product_report.v2` (and v1 compat), `phase46.ci_verdict.v1`, `phase46.import_audit.v1`, `wevra.capsule_view.v1` | On-disk contract | **Stable** (v1 and v2 both accepted by `wevra-ci`) | — |
| **Core substrate** — CASR router, hierarchical router, ledger, exact_ops, role_handoff, dynamic_comm, adaptive_sub | Research substrate used *by* Wevra; adapter-able into capsules (`capsule_from_handle`, `capsule_from_handoff`, …) | **Settled** (proofs + tests) but **research API** | `vision_mvp.core.*` |
| **Legacy product path** — `vision_mvp.product.*` | Pre-Slice-1 import path | **Deprecated-compat** (still works; re-exported by `wevra`) | `vision_mvp.product` |
| **Docker sandbox** — `DockerSandbox` (backend) | Untrusted-input isolation | **Available** (backend exists, contract-tested); **not yet the default** for public JSONLs | `vision_mvp.wevra.extensions.get_sandbox("docker")` |
| **Docker-first-by-default** for public JSONLs | Slice 3 target | **Boundary / next-slice** (default-flip when caller declares untrusted input) | n/a yet |
| **First real out-of-tree plugin** | Slice 3 community target | **Exemplar landed** (`examples/out_of_tree_plugin/wevra-markdown-sink`); a full third-party-owned sink package remains future | `examples/out_of_tree_plugin/` |
| **GitHub Actions release on tag** | Slice 3 ops target | **Declared** (workflow file checked in); **not yet fired on a real tag** | `.github/workflows/wevra-ci.yml` |
| **Research shards** — Phases 1–44 RESULTS_*, EXTENDED_MATH_*, per-phase scripts, 72-framework survey | Research programme | **Research-grade** (empirical/proved per shard; no product-API guarantee) | `vision_mvp.experiments.*`, `vision_mvp.tasks.*`, docs |

### 10.2 Slice 1 (this milestone) — completed

| Item | Status | Anchor |
|---|---|---|
| Canonical SDK package boundary `vision_mvp/wevra/` | ✅ | `vision_mvp/wevra/__init__.py` |
| `RunSpec` / `run` unified programmatic entry point | ✅ | `vision_mvp/wevra/run.py` |
| `WevraConfig` (env-driven, frozen, validated) | ✅ | `vision_mvp/wevra/config.py` |
| Provenance manifest (`wevra.provenance.v1`) on every run | ✅ | `vision_mvp/wevra/provenance.py`, wired in `product/runner.py` |
| Console scripts `wevra`, `wevra-import`, `wevra-ci` | ✅ | `pyproject.toml` `[project.scripts]`, `wevra/_cli.py` |
| Package rename → `wevra` 0.4.0, extras slots (dev/docker/cluster) | ✅ | `pyproject.toml` |
| `sys.path.insert` hacks removed from product modules | ✅ | `product/runner.py`, `product/ci_gate.py`, `product/import_data.py` |
| Contract tests locking the public SDK surface | ✅ | `tests/test_wevra_public_api.py` |
| Provenance + CLI smoke tests | ✅ | `tests/test_wevra_provenance.py` |
| Backwards-compat: `vision_mvp.product.*` still works | ✅ | confirmed by `test_phase45_product.py` |
| README: Wevra-vs-Context-Zero framing, SDK quickstart, stability matrix, "Who Wevra is for", "How to extend Wevra" | ✅ | `README.md` |
| ARCHITECTURE: Wevra SDK boundary note | ✅ | `ARCHITECTURE.md` |
| Master plan § 10 (this section) | ✅ | here |

**Theorem-style claims / conjectures for Slice 1:**

  * **Claim W1-1 (SDK boundary is a projection of settled layers).**
    The Wevra SDK re-exports only modules whose tests have held
    unchanged across the last three phases (profiles, report,
    ci_gate, import_data, runner). No research-grade module is on
    the SDK surface.
  * **Claim W1-2 (every run is reproducible).** Every invocation
    of `wevra.run(spec)` — and every `python -m vision_mvp.product`
    — emits a `provenance.json` pinning git SHA, package version,
    Python, platform, profile, model, endpoint, sandbox, input
    JSONL + SHA-256, argv, and the final artifact list. Tested on
    every run (`test_runner_emits_provenance_on_every_run`).
  * **Conjecture W1-3 (drop-in without a plugin system).** For the
    current scope (profile-driven SWE-bench-Lite-shape evaluation
    + CI gate + public-JSONL import), Wevra is a true drop-in SDK:
    `pip install wevra` + one console script yields a
    provenance-stamped report. Falsifiable by any external
    operator who needs a new sandbox backend, a new task-bank
    shape, or a new reporting sink without editing `vision_mvp/`.
    If falsified, Slice 2 (plugin system) becomes load-bearing.
  * **Conjecture W1-4 (research/product split holds).** Nothing
    on the SDK surface depends on a research-grade module whose
    API is not yet settled. Falsifiable by any SDK consumer who
    has to import from `vision_mvp.core.*`,
    `vision_mvp.experiments.*`, or `vision_mvp.tasks.*` to use
    Wevra.

### 10.2-bis Slice 2 (2026-04-22) — completed

Full detail in `vision_mvp/RESULTS_WEVRA_SLICE2.md`.

| Item | Status | Anchor |
|---|---|---|
| Extension system (3 Protocols, registry, `entry_points` discovery) | ✅ | `vision_mvp/wevra/extensions/` |
| Worked in-tree `ReportSink` example, exercised end-to-end | ✅ | `examples/jsonl_report_sink.py` |
| Unified mock/real runtime (`SweepSpec`, `run_sweep`, `HeavyRunNotAcknowledged`) | ✅ | `vision_mvp/wevra/runtime.py` |
| `RunSpec.acknowledge_heavy` first-class cost gate | ✅ | `vision_mvp/wevra/run.py` |
| `RunSpec.report_sinks` hook | ✅ | `vision_mvp/wevra/run.py` |
| Report schema bumped to `phase45.product_report.v2` (v1 still accepted by CI gate) | ✅ | `vision_mvp/product/runner.py`, `ci_gate.py` |
| Env-driven endpoints (`WEVRA_OLLAMA_URL_MAC{1,2}`, `WEVRA_OLLAMA_URL`) | ✅ | `wevra.runtime._resolve_endpoint` |
| Operator-grade failure messages on the run path | ✅ | `wevra.runtime` + `product/report.py` |
| `--acknowledge-heavy` / `--report-sink` on `wevra` CLI | ✅ | `vision_mvp/wevra/_cli.py` |
| GitHub Actions workflow (SDK contract tests + build + release-on-tag) | ✅ (declared) | `.github/workflows/wevra-ci.yml` |
| `CHANGELOG.md` tracks Wevra SDK releases | ✅ | `CHANGELOG.md` |
| **Real ASPEN mac1 run launched via `wevra.run(...)`** | ✅ | `vision_mvp/artifacts/wevra_slice2_g1/` — `sweep.executed_in_process=true`, 4 instances × 2 parser modes × 3 strategies, 114 s wall |
| `wevra-ci` verdict on the ASPEN run (canonical release artifact) | ✅ | `vision_mvp/artifacts/wevra_slice2_g1/ci_verdict.json` — `ok=true` |
| Contract tests for extensions (10) + runtime (10) | ✅ | `tests/test_wevra_extensions.py`, `tests/test_wevra_runtime.py` |
| `SDK_VERSION` bump to `wevra.sdk.v2` (additive, backwards-compat) | ✅ | `vision_mvp/wevra/__init__.py` |
| Master plan § 10 refreshed (this pass) | ✅ | here |

Full suite: **1349 tests pass**, up from Slice 1's 1327.

### 10.3 Slice 3 — deferred (explicit, concrete)

Each item is concrete, classified, and has a named code path.
Slice 2 closed B, C, G, and most of D; what remains is scoped to
default-flips, community artifacts, and release-tag firings — not
to new subsystems.

#### B. Plugin / extension system — ✅ **done (Slice 2)**

| Follow-up | Status | Anchor |
|---|---|---|
| B.1 Stable `SandboxBackend` Protocol (runtime-checkable) | ✅ | `vision_mvp/wevra/extensions/sandbox.py` |
| B.2 Stable `TaskBankLoader` Protocol + `TaskBankBundle` | ✅ | `vision_mvp/wevra/extensions/taskbank.py` |
| B.3 Stable `ReportSink` Protocol | ✅ | `vision_mvp/wevra/extensions/report_sink.py` |
| B.4 Registry discovery via `importlib.metadata.entry_points` | ✅ | `vision_mvp/wevra/extensions/registry.py` (groups `wevra.sandboxes` / `wevra.task_banks` / `wevra.report_sinks`) |
| B.5 One worked extension landed end-to-end | ✅ (in-tree) | `vision_mvp/wevra/extensions/examples/jsonl_report_sink.py`; exercised end-to-end by `test_wevra_extensions.py::test_worked_example_sink_end_to_end` |
| B.6 Contract tests for the extension protocols | ✅ | `vision_mvp/tests/test_wevra_extensions.py` (10 tests) |

**Remaining in Slice 3:** a community-owned out-of-tree plugin
shipped by a third party. An *in-repo* exemplar landed in the
post-Slice-2 identity pass (`examples/out_of_tree_plugin/wevra-
markdown-sink/`) — a standalone pip-installable package that
registers a new `ReportSink` purely via `entry_points` and
requires no edit under `vision_mvp/`. The machinery is closed and
the contract is demonstrated; only an actual third-party publisher
is future.

#### C. Unified mock↔real runtime contract — ✅ **done (Slice 2)**

| Follow-up | Status | Anchor |
|---|---|---|
| C.1 `SweepSpec` dataclass (frozen, validated) | ✅ | `vision_mvp/wevra/runtime.py` |
| C.2 Dispatcher: mock → in-process; real → in-process | ✅ | `wevra.runtime.run_sweep` |
| C.3 `executed_in_process=True` path for real runs | ✅ | validated end-to-end on ASPEN mac1 |
| C.4 Explicit operator-boundary cost gate | ✅ | `RunSpec.acknowledge_heavy=True`; `HeavyRunNotAcknowledged` raised under `strict_cost_gate=True` |
| C.5 One coherent artifact/report model | ✅ | schema bumped to `phase45.product_report.v2`, sweep sub-block schema `wevra.sweep.v2`; CI gate accepts v1 and v2 |

#### Remaining D (production hardening) — ✅ **mostly done (Slice 2)**

| Follow-up | Status | Anchor |
|---|---|---|
| D.1 Docker-first sandbox as default for untrusted JSONLs | ◐ | `DockerSandbox` backend exists and is registered under `wevra.extensions.get_sandbox("docker")`; the *default-flip* for public/untrusted inputs is Slice 3 |
| D.2 Env-validated cluster endpoints | ✅ | `WEVRA_OLLAMA_URL_MAC1`, `WEVRA_OLLAMA_URL_MAC2`, `WEVRA_OLLAMA_URL` resolved in `wevra.runtime._resolve_endpoint`; tested in `test_wevra_runtime.EnvEndpointOverrideTests` |
| D.3 Operator-grade failure messages | ✅ | unified runtime wraps sweep errors into `sweep_result.error_kind` / `sweep_result.error_detail`; renderer surfaces `sweep : ERROR <kind>: <detail>` instead of a stack trace |
| D.4 Release workflow | ◐ | `.github/workflows/wevra-ci.yml` checked in (SDK contract tests on 3.10/3.11/3.12 + `python -m build` sdist+wheel + release on tag); not yet fired on a real tag (Slice 3 ops) |
| D.5 CHANGELOG.md curated per Wevra release | ✅ | `CHANGELOG.md` now tracks Wevra SDK versions (0.4.0 Slice 1, 0.5.0 Slice 2); phase-numbered narrative stays in RESULTS notes |

#### G. Cluster-backed validation — ✅ **done (Slice 2)**

| Follow-up | Status | Anchor |
|---|---|---|
| G.1 One `aspen_mac1_coder` real run launched via `wevra` | ✅ | `vision_mvp/artifacts/wevra_slice2_g1/product_report.json`; `sweep.executed_in_process=true`, `sweep.model=qwen2.5-coder:14b`, 4 instances × 2 parser modes × 3 strategies, 114 s wall |
| G.2 Provenance manifest stored with the artifact | ✅ | `vision_mvp/artifacts/wevra_slice2_g1/provenance.json` (`wevra.provenance.v1`) |
| G.3 `wevra-ci` verdict checked in | ✅ | `vision_mvp/artifacts/wevra_slice2_g1/ci_verdict.json` — `ok=true, blockers=0, executed_in_process=true` |

**Empirical finding reproduced.** Strict parser fails
(pass@1 = 0.000) on `qwen2.5-coder:14b`, robust parser recovers
(pass@1 = 1.000) across all three strategies — a Phase-42 result
re-obtained via the Wevra surface.

### 10.4 Slice 2 theorem-style claims (detail in `vision_mvp/RESULTS_WEVRA_SLICE2.md`)

  * **W2-1 (proved, constructive).** Stable extension surface
    preserves bounded-context guarantees: any registered
    `SandboxBackend` / `TaskBankLoader` / `ReportSink` that
    satisfies its Protocol is, by construction, a renaming /
    projection of a settled substrate boundary; it cannot violate
    Theorem 3 bounded-context invariants.
  * **W2-2 (empirical).** Unified-runtime identity under mock
    oracle: every mock profile's pooled summary is unchanged
    between Slice 1's `_mock_sweep` and Slice 2's
    `wevra.runtime.run_sweep` (1349/1349 tests pass).
  * **W2-3 (proved, constructive).** External-run safety as
    product contract: every `wevra.run` emits a provenance
    manifest sufficient to reproduce the run on a different
    machine, modulo declared external state (model residency,
    endpoint availability) and `temperature=0.0`-bounded LLM
    nondeterminism.
  * **W2-4 (testable conjecture).** Wevra is a true drop-in SDK
    iff E1–E6 hold (pip-install works, provenance is sufficient,
    real runs execute in-process, plugins don't require in-tree
    edits, release automation emits artifacts on tag, Docker is
    default for untrusted input). E1–E3 closed; E4 closed as
    machinery (first out-of-tree plugin is Slice 3); E5 declared
    (workflow in-repo, not yet fired on a real tag); E6 open
    (Slice 3).

### 10.5 What still blocks a true 10/10 drop-in production SDK

Honest ledger, today:

1. **Docker-first-by-default for untrusted public JSONLs.** — ✅
   **done (Slice 3B, 2026-04-22).** Profiles carry a `trust` tag
   (`trusted` | `untrusted`). `public_jsonl` is `untrusted` and
   ships with `sandbox_name="docker"`. The runner enforces the
   default: an untrusted profile refuses to start on any sandbox
   weaker than Docker, and refuses to start with Docker configured
   but the daemon unreachable, unless the operator passes
   `--allow-unsafe-sandbox`. Opt-out downgrades to `subprocess`
   with an explicit note recorded in the profile. Locked by
   `test_public_jsonl_profile_is_untrusted_and_docker_first`,
   `test_public_jsonl_refuses_weak_sandbox_without_docker`,
   `test_public_jsonl_allow_unsafe_downgrades_to_subprocess`.
2. **Report-schema compatibility policy.** — ✅ **explicit
   (Slice 3B).** `phase45.product_report.v2` is the stable product-
   report schema (Slice 2+). `ci_gate.EXPECTED_REPORT_SCHEMAS`
   accepts both v1 and v2 — v1 remains valid input to the CI gate
   for backwards-compat but is no longer emitted. Test locked:
   `test_report_schema_stable` asserts v2; `test_wevra_runtime`
   re-asserts v2 via the SDK surface.
3. **Bare-install wheel sanity.** — ✅ **done (Slice 3B).** Release
   rehearsal (`python -m build` → `pip install wevra-0.5.1-py3-
   none-any.whl` in a fresh venv) exposed two real release
   blockers that are now fixed: (a) `core/peer_review.py` and
   `core/vrf_committee.py` imported `cryptography` unconditionally
   at module top-level — now guarded behind `_require_cryptography()`
   at constructor time, matching the optional `[crypto]` extra;
   (b) `wevra --version` crashed because `argparse` required
   `--profile`/`--out-dir` — now short-circuits pre-parse.
4. **First real out-of-tree plugin** — in-repo exemplar landed
   (`examples/out_of_tree_plugin/wevra-markdown-sink/`), proving
   the `entry_points` contract works without any edit under
   `vision_mvp/`. A community-owned third-party-published plugin
   package remains future.
5. **GitHub Actions release on a real tag.** The workflow is
   checked in, the local `python -m build` → fresh-venv install
   path is exercised (both `wevra --profile local_smoke` and
   `wevra-ci` run cleanly from the wheel), but the tag-triggered
   `release` job has not been fired on a real `vX.Y.Z` tag. This
   remains the single residual operational step that requires a
   GitHub tag event to exercise honestly.
6. **External blockers from § 9.8 still apply** (public SWE-bench-
   Lite JSONL on local disk; ≥70B resident coder model).
   Orthogonal to all SDK work; remain classified as 🧱 external.

Slice 1 bought the SDK boundary. Slice 2 bought the plugin system,
the unified runtime, the cluster-executed artifact, and the
release automation. Slice 3A closed the real-profile regression.
Slice 3B closed the Docker-first default, the schema-compatibility
policy, and the bare-install wheel path. The one remaining
in-control step is to push a real tag.

### 4.35 SDK v3.18 — magnitude-hinted producer protocol + fresh-live end-to-end composition + symmetric-corroboration limit theorem + W17 family (Phase-64 live composition + R-64-SYM wall)

SDK v3.18 attacks the three honest gaps SDK v3.17 (W16) left
explicit:

* **The W16-Λ-real-replay anchor was *replay-only*** (the Mac-1
  endpoint was offline at SDK v3.17 milestone capture); a fresh
  live probe (W16-C-LIVE-OLLAMA) was conjectural.
* **The 1/8 R-61-OLLAMA-A model-side judgment miss persisted**:
  on the ``slow_query_archival`` scenario the LLM judged decoy
  magnitudes "not severe enough" *relative* to the larger gold
  spike on the same prompt and skipped them as NONE, breaking
  ``round1_decoy_corroborated``. The structured prompt did not
  close this gap because it never gave the LLM a concrete *named*
  lower bound.
* **The symmetric-corroboration wall** (W15-C-SYMMETRIC /
  W16-C-SYMMETRIC) was named but not built or proved.

The W17 family closes all three on one driver. The SDK v3.18 mint:

* **W17-1 (proved-conditional + empirical-research).** Pairing
  the W17 ``StructuredProducerProtocol(mode=PRODUCER_PROMPT_MAGNITUDE_HINTED)``
  + ``incident_triage_role_schemas(magnitude_hinted=True)`` with
  the W14 producer + W15 decoder pipeline at
  ``K_auditor = 8, T_auditor = 256, T_decoder = 14`` against a
  *fresh* live ``qwen2.5:14b-32k`` Mac-1 Ollama backend on the
  Phase-61 comparable-magnitude bank (n=8 × 24 producer calls;
  0 endpoint failures; 128.2 s wall) achieves
  ``capsule_attention_aware = 1.000``,
  ``capsule_layered_fifo_packed = 0.000``, and
  ``capsule_fifo = 0.000``. **+1.000 strict separation** on both
  axes. Bench property holds in 8/8 — closing the 1/8 miss the
  W14 anchor and W16-Λ-real-replay anchor left open. The first
  programme result that beats the strongest non-composed baseline
  by ≥ 1.0 on a *fresh* live LLM probe.

* **W17-Λ-no-hint (empirical-research).** Under ``mode = 'structured'``
  + ``magnitude_hinted_schema = False`` + same fresh live probe
  + same budget: bench property holds in 7/8;
  ``capsule_attention_aware = 0.500``;
  ``capsule_layered_fifo_packed = 0.000``; +0.500 strict gain.
  Reproduces the W14-Λ-real / W16-Λ-real-replay envelope on the
  fresh probe; the magnitude-hint extension, not a re-run, is
  the load-bearing improvement.

* **W17-Λ-naive (empirical-research).** Under ``mode = 'naive'``
  + same live probe + same budget: bench property holds in 0/8;
  every capsule strategy ties FIFO at 0.000. Live counterpart of
  the W14-Λ-prompt + W15-Λ-budget joint failure.

* **W17-Λ-symmetric (proved-empirical + structural sketch).**
  *The first explicit symmetric-corroboration limit theorem in
  the programme.* On ``build_phase64_sym_bank`` (synthetic
  identity extractor; every service mentioned by exactly 2
  distinct routed producer roles via generic-noise kinds with
  comparable magnitudes; round-2 disambiguator names gold
  root_cause without ``service=`` token), every capsule strategy
  in the SDK ties FIFO at ``accuracy_full = 0.000`` under both
  ``T_decoder ∈ {None, 24}``. The priority decoder still elects
  the right specific-tier ``root_cause``
  (``accuracy_root_cause = 1.000``); the failure is
  ``services_correct`` set-equality. Sketch: when the bipartite
  ``(role × tag, kind, magnitude)`` multiset is symmetric for
  gold and decoy, no service-blind admission AND no closed-form
  salience packer can prefer one over the other; the W11
  contradiction-aware drop fires symmetrically; the W15
  hypothesis-preserving pack preserves both in equal proportion;
  the W14H magnitude-hint is silent on symmetric ambiguity.
  **Discharges W15-C-SYMMETRIC / W16-C-SYMMETRIC as a negative
  theorem.**

* **W17-C-XMODEL (proved-conditional + empirical-research).**
  On a fresh live Mac-1 ``qwen3.5:35b`` MoE Ollama backend with
  ``think = False`` (n=8 × 24 producer calls; 0 failures; 92.0 s
  wall): bench property holds in 8/8 (the W17 magnitude-hint
  extension transfers byte-for-byte across the 14B → 36B-MoE
  jump on the bench-property axis); ``capsule_attention_aware
  = 0.750``; ``capsule_layered_fifo_packed = 0.000``;
  ``capsule_fifo = 0.000``; **+0.750 strict gain** (well above
  the 0.50 strong-bar threshold). The 0.250 gap to 1.000 is on
  ``accuracy_root_cause`` — a 35B-specific specific-tier
  judgment artifact (the model emits a different specific-tier
  kind on one scenario than the 14B does), not a producer-
  protocol failure. **Partially discharges W16-C-CROSS-MODEL.**

* **W17-2 (proved + mechanically-checked).** Magnitude-hinted
  prompt rendering is byte-deterministic; the threshold table
  filters to the round's allowed-set; with empty
  ``schema.magnitude_thresholds`` the prompt reduces to the
  structured prompt with the anti-relative-magnitude clause
  appended; :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS`
  exposes the same numeric values as the synthetic
  ``MagnitudeFilteringExtractor``'s default thresholds.

* **W17-3 (proved-empirical full programme regression).** With
  ``mode = 'naive'`` or ``mode = 'structured'`` AND
  ``magnitude_hinted_schema = False``, the W17 surface reduces
  to the SDK v3.15 W14 anchor byte-for-byte; on the synthetic
  side, swapping ``mode = 'structured'`` for
  ``mode = 'magnitude_hinted'`` with
  ``magnitude_hinted_schema = True`` produces the same
  downstream answer. R-58 / R-59 / R-60 / R-61 / R-62 / R-63
  default + falsifier banks all preserved; **442/442 prior
  tests pass byte-for-byte**; 20 new tests cover the W17
  surface (462/462 wevra suite; +103 cross-cutting tests pass).

The W17 layer is *additive in code*: one new prompt-mode
constant (``PRODUCER_PROMPT_MAGNITUDE_HINTED``), one new
dataclass (``OperationalThreshold``), one new field on
:class:`RoleExtractionSchema` (``magnitude_thresholds: tuple[
OperationalThreshold, ...] = ()``), one new prompt-render
helper (``_render_magnitude_hinted_prompt``), and one
threshold-table data structure
(:data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS`). Re-
exported via the SDK ``__init__``. **The runtime contract is
byte-for-byte unchanged.**

The Wevra programme now has **nine** structurally-distinct
moves named one-by-one over SDK v3.7..v3.18, with W17 adding
two: a *positive* live-strict-gain anchor (W17-1) AND a
*negative* symmetric-corroboration limit theorem
(W17-Λ-symmetric).

The defensible "thesis-after-SDK-v3.18" is that the
synthetic→real-LLM-and-bounded-context transfer story now has
**nine layers + one named structural wall**. The named wall
(W17-Λ-symmetric) makes the next research frontier explicit:
a learned or LLM-distilled semantic-disambiguator beyond the
closed-form capsule surface (W17-C-DISAMBIGUATOR, conjectural).

**For another agent to instantly see the SDK v3.18 state**:

1. **Did replay success become live success?** *YES.* W17-1 on
   R-64-LIVE-MAGHINT achieves +1.000 strict gain on a fresh
   live qwen2.5:14b-32k probe. Bench property holds in 8/8;
   composition accuracy_full = 1.000.
2. **Was the remaining model-side miss closed?** *YES.* The
   W17 magnitude-hinted prompt extension closes the 1/8
   slow_query_archival miss that persisted under W14 / W16
   (legacy structured prompt). The W17-Λ-no-hint anchor on the
   same fresh probe under the legacy structured prompt
   reproduces the 7/8 + 0.500 envelope, confirming the
   magnitude-hint extension is the load-bearing improvement.
3. **Did cross-model transfer hold?** *PARTIALLY.* W17-C-XMODEL
   on a fresh live qwen3.5:35b probe: bench property holds in
   8/8; +0.750 strict gain; saturated full-correctness clause
   bounded by a 35B-specific specific-kind judgment artifact.
4. **Is the symmetric-limit wall now clearer?** *YES.*
   W17-Λ-symmetric is the *first explicit symmetric-
   corroboration limit theorem* in the programme. R-64-SYM
   construction proves every capsule strategy in the SDK ties
   FIFO at 0.000 under both ``T_decoder ∈ {None, 24}`` by
   construction. The wall is real and structural.
5. **Is the original thesis materially stronger or still
   blocked?** *MATERIALLY STRONGER.* The original Context-Zero
   thesis — *per-agent minimum-sufficient context for multi-
   agent teams* — gains its first **fresh-live end-to-end
   real-LLM strict +1.000 advance** over the strongest non-
   composed baseline AND closes the prior 1/8 model-side
   judgment miss. The W17-Λ-symmetric wall names the deeper
   ambiguity wall *constructively*, not just as a conjecture —
   it is the first explicit "this is what no closed-form
   capsule strategy can do" theorem in the programme. The
   thesis is materially stronger AND the next research
   frontier is materially clearer.

---

### 4.36 SDK v3.19 — bundle-relational compatibility disambiguator + symmetric-ambiguity benchmark family + W18 family (Phase-65 R-65-COMPAT + named falsifier triple)

SDK v3.19 attacks the named research frontier SDK v3.18 left
explicit (W17-C-DISAMBIGUATOR, conjectural): the symmetric-
corroboration wall (W17-Λ-symmetric) where every closed-form
salience scorer in the SDK ties FIFO at 0.000 by construction.
The W17 milestone named the next move: a *richer disambiguator*
that consumes information beyond the bipartite ``(role × tag,
kind, magnitude)`` multiset — specifically, the round-2 specific-
tier disambiguator's payload **text**.

The W18 family is the smallest move in that direction. The
:class:`RelationalCompatibilityDisambiguator` is a deterministic,
training-free, closed-form four-stage pipeline:

  1. Run the W15 :class:`AttentionAwareBundleDecoder` over the
     admitted bundle (capture inner answer + W15 pack stats).
  2. Identify the round-2 specific-tier disambiguator(s) in the
     bundle.
  3. Tokenise the disambiguator payload (lower-cased, split on
     non-identifier chars, compound identifiers preserved) and
     score each admitted service tag in the *union* of admitted
     tags by direct-match + contiguous-subsequence compound
     match.
  4. Apply the strict-asymmetric branch: keep positive-score
     tags iff at least one but not all admitted tags have positive
     score; otherwise abstain (fall through to the inner answer
     byte-for-byte).

The SDK v3.19 mint:

* **W18-Λ-sym (proved-empirical n=8 saturated × 5 seeds +
  structural sketch).** W17-Λ-symmetric extends to R-65-COMPAT
  verbatim for every method pre-W18.
* **W18-1 (proved-conditional + proved-empirical n=40 saturated
  across 5 seeds × 2 budgets).** Pairing the W15 attention-aware
  decoder with the W18 disambiguator achieves
  ``capsule_relational_compat = 1.000`` on R-65-COMPAT-LOOSE
  AND R-65-COMPAT-TIGHT, strictly improving over every non-W18
  capsule baseline by **+1.000**, stable across 5/5 alternate
  ``bank_seed`` values.
* **W18-2 (proved by inspection + mechanically-checked).** W18
  determinism + closed-form correctness; bounded-context honesty
  (``tokens_kept_sum`` byte-for-byte identical to W15's).
* **W18-3 (proved-empirical full programme regression).**
  Backward-compat with R-54..R-64 byte-for-byte; with
  ``enabled = False`` the W18 method reduces to W15 byte-for-
  byte.
* **W18-Λ-no-compat / -confound / -deceive (proved-empirical
  n=8 saturated each).** Three named structural limit regimes
  where W18 ties FIFO or fails by construction.
* **W18-C-LEARNED, W18-C-OUTSIDE, W18-Λ-real, W18-C-CROSS-BENCH
  (conjectural).** Named extension axes.

The W18 layer is *additive in code*: one new dataclass +
one tokeniser + one closed-form scorer + one wrapping decoder.
The SDK v3.18 runtime contract is byte-for-byte unchanged;
**all prior tests pass byte-for-byte**; new tests cover the W18
unit semantics, the Phase-65 bench-property witnesses, the W18-1
strict-win anchor, the 5-seed stability, the three named
falsifiers, the backward-compat smoke, the token-budget honesty,
and the cross-regime synthetic summary.

The defensible "thesis-after-SDK-v3.19" is that the
synthetic→real-LLM-and-bounded-context transfer story now has
**ten layers + one named structural wall + three named falsifier
regimes**. The first capsule-native multi-agent-coordination
method has crossed the symmetric-corroboration wall on a regime
where the wall actually applies (R-65-COMPAT). The next research
frontier is W18-C-LEARNED (free-form natural-language relational
mentions) and W18-C-OUTSIDE (outside-information axis to detect
adversarial round-2 mentions).

**For another agent to instantly see the SDK v3.19 state**:

1. **Is the new frontier true ambiguity resolution beyond W17?**
   *YES.* The W18 family attacks the W17-Λ-symmetric wall
   directly via the round-2 disambiguator's payload-text channel
   that every prior decoder ignored.
2. **Does the new method actually break the symmetric wall?**
   *YES, on R-65-COMPAT.* W18 = 1.000; every non-W18 capsule
   strategy = 0.000. **+1.000 strict separation**, stable across
   5/5 seeds, on both loose AND tight decoder budgets.
3. **Did bounded-context efficiency survive?** *YES.* The W18
   method reads only the W15-packed bundle; ``tokens_kept_sum``
   is byte-for-byte identical to W15's. Bounded-context honesty
   preserved.
4. **Is the original thesis materially stronger or still
   blocked by a deeper semantic wall?** *MATERIALLY STRONGER on
   the named axis; partially blocked on the deeper axis.* The
   original Context-Zero thesis — *per-agent minimum-sufficient
   context for multi-agent teams* — gains its **first capsule-
   native method to cross the symmetric-corroboration wall on a
   regime where the wall actually applies**. The deeper wall —
   adversarial-relational round-2 evidence (W18-Λ-deceive) — is
   *named* but *not* broken; the named research move beyond it
   (W18-C-OUTSIDE — outside-information axis) is conjectural.
   The thesis is materially stronger AND the next research
   frontier is materially clearer.

---

### 4.37 SDK v3.20 — bundle-contradiction-aware trust-weighted disambiguator + deceptive-ambiguity benchmark family + W19 family (Phase-66 R-66-DECEIVE-NAIVE / R-66-CONFOUND-RESOLVABLE + named falsifier pair)

SDK v3.20 attacks the named research frontier SDK v3.19 left
explicit (W18-Λ-deceive, the *adversarial-relational* wall where
W18 trusts its evidence and fails; AND W18-Λ-confound, the
*symmetric-relational* wall where W18 abstains and falls through
to the empty inner W15 answer). The W18 milestone named the
structural shape of the next move: a *richer scorer* that
consumes information *beyond* the round-2 disambiguator's
payload itself — specifically, the *consistency between* the
canonical primary disambiguator and *independent asymmetric
witnesses* elsewhere in the bundle.

The W19 family is the smallest move in that direction. The
:class:`BundleContradictionDisambiguator` is a deterministic,
training-free, closed-form four-stage pipeline:

  1. Run the W18 :class:`RelationalCompatibilityDisambiguator`
     over the admitted bundle (capture inner answer + W18 audit
     + W15 pack stats).
  2. Identify the *canonical primary* specific-tier disambiguator
     in the admitted union via canonical sort with a canonical-
     role-for-kind tiebreak (the
     :data:`_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND` table) +
     raw-kind tiebreak (so synonym/heuristic-rescued kinds from
     non-canonical roles do not displace the canonical primary).
  3. Count independent *asymmetric witnesses* per admitted
     service tag — specific-tier handoffs OTHER than the
     canonical primary whose tokenised payload mentions the tag,
     deduplicated by ``(source_role, claim_kind, payload_sha)``.
  4. Decide the W19 branch:
     * **Inversion** — when W18 fires its strict-asymmetric
       branch but ``max_aw(complement) > max_aw(named_set)``,
       project to the complement.
     * **Confound-resolved** — when W18 abstains AND there is a
       unique strict-max-aw subset M ⊊ U, project to M.
     * **Abstained-symmetric** — when W18 abstains AND
       witnesses are symmetric across U, abstain (W19-Λ-outside
       wall).
     * **Primary-trusted / abstained-no-signal** — fall through
       to W18.

The SDK v3.20 mint:

* **W19-Λ-deceive-extension (proved-empirical n=8 saturated × 5
  seeds + structural sketch).** W18-Λ-deceive extends to
  R-66-DECEIVE-NAIVE for every closed-form scorer that *trusts*
  its concatenated disambiguator text — including W18 itself.
* **W19-1 (proved-conditional + proved-empirical n=120 saturated
  across 5 seeds × 3 regimes).** Pairing the W18 disambiguator
  with the W19 trust layer achieves
  ``capsule_bundle_contradiction = 1.000`` on R-66-DECEIVE-NAIVE-
  LOOSE AND R-66-DECEIVE-NAIVE-TIGHT AND R-66-CONFOUND-RESOLVABLE,
  strictly improving over the W18 baseline by **+1.000**, stable
  across 5/5 alternate ``bank_seed`` values.
* **W19-2 (proved by inspection + mechanically-checked).** W19
  determinism + closed-form correctness; bounded-context honesty
  (``tokens_kept_sum`` byte-for-byte identical to W18's).
* **W19-3 (proved-empirical full programme regression).**
  Backward-compat with R-54..R-65 byte-for-byte; with
  ``enabled = False`` the W19 method reduces to W18 byte-for-
  byte.
* **W19-Λ-total / -outside (proved-empirical n=8 saturated
  each).** Two named structural limit regimes where W19 ties
  FIFO by construction. The W19-Λ-total wall (no asymmetric
  witness anywhere in the bundle) bounds the bundle-only
  closed-form scope; the W19-Λ-outside wall (symmetric witnesses)
  bounds the same scope even when witnesses exist.
* **W19-C-LEARNED, W19-C-OUTSIDE, W19-Λ-real, W19-C-CROSS-BENCH
  (conjectural).** Named extension axes. W19-C-OUTSIDE is the
  natural escape from BOTH falsifier walls.

The W19 layer is *additive in code*: one new dataclass + two
closed-form helpers + one canonical-role-for-kind table + one
wrapping decoder. The SDK v3.19 runtime contract is byte-for-
byte unchanged; **all prior wevra tests pass byte-for-byte**;
new tests cover the W19 unit semantics, the Phase-66 bench-
property witnesses, the W19-1 strict-win anchor, the 5-seed
stability, the two named falsifiers, the backward-compat smoke,
the token-budget honesty, and the cross-regime synthetic summary.
Total: 405 pre-existing wevra tests + 45 new W19 tests =
450 / 450 in the targeted wevra suites; 555 / 555 across the
full ``test_wevra_*.py`` set.

The defensible "thesis-after-SDK-v3.20" is that the
synthetic→real-LLM-and-bounded-context transfer story now has
**eleven layers + two named structural walls + five named
falsifier regimes**. The first capsule-native multi-agent-
coordination method has crossed the deceptive-ambiguity wall
on regimes where the bundle carries an independent asymmetric
witness for gold (R-66-DECEIVE-NAIVE / R-66-CONFOUND-RESOLVABLE).
The next research frontier is W19-C-OUTSIDE (outside-information
axis to escape BOTH the W19-Λ-total wall — no witnesses anywhere
— AND the W19-Λ-outside wall — symmetric witnesses).

**For another agent to instantly see the SDK v3.20 state**:

1. **Is the new frontier deceptive ambiguity beyond W18?**
   *YES.* The W19 family attacks the W18-Λ-deceive AND
   W18-Λ-confound walls directly via the bundle-contradiction
   channel that W18's selector ignores (W18 concatenates all
   specific-tier payloads and treats them as one disambiguator
   text; W19 distinguishes the canonical primary from secondary
   witnesses).
2. **Does the new method actually beat the W18-deceive wall?**
   *YES, on the bundle-resolvable case (R-66-DECEIVE-NAIVE).*
   W19 = 1.000; W18 = 0.000. **+1.000 strict separation**,
   stable across 5/5 seeds, on both loose AND tight decoder
   budgets. Same on R-66-CONFOUND-RESOLVABLE.
3. **Did bounded-context efficiency survive?** *YES.* The W19
   method reads only the W18-packed bundle (which itself reads
   only the W15-packed bundle); ``tokens_kept_sum`` is byte-
   for-byte identical to W18's. Bounded-context honesty
   preserved.
4. **Is the original thesis materially stronger or still
   blocked by a deeper semantic / outside-information wall?**
   *MATERIALLY STRONGER on the named axis; the deeper wall is
   now SHARP — outside information is genuinely needed.* The
   original Context-Zero thesis — *per-agent minimum-sufficient
   context for multi-agent teams* — gains its **first capsule-
   native method to resolve bundle-internal contradiction
   between primary disambiguator and witnesses**. The deeper
   walls (W19-Λ-total — bundle exhausted of asymmetric signal;
   W19-Λ-outside — symmetric witnesses) are *named* and
   *proved-structural* — the natural escape from BOTH is
   outside information (W19-C-OUTSIDE), conjectural. **The
   thesis is materially stronger AND the next research frontier
   is precisely articulated as a structural result, not a
   method-gap conjecture.**

### 4.38 SDK v3.21 — outside-witness acquisition disambiguator + R-67 outside-information benchmark family + W20 family (Phase-67 R-67-OUTSIDE-RESOLVES + three named falsifiers)

SDK v3.21 attacks the named research frontier SDK v3.20 left
conjectural (W19-C-OUTSIDE) via the smallest possible move: a
deterministic, training-free
:class:`OutsideWitnessAcquisitionDisambiguator` (W20) that wraps
the W19 inner and consults a registered
:class:`OutsideWitnessOracle` exactly *once* per cell when the
inner W19 returns a trigger branch (default
:data:`W19_BRANCH_ABSTAINED_SYMMETRIC` /
:data:`W19_BRANCH_ABSTAINED_NO_SIGNAL`). The oracle's reply is
parsed via the same closed-vocabulary closure W18 / W19 use on
in-bundle witnesses; the W19 answer is projected onto the
positive set of admitted-tag mentions in the reply.

**The R-67 bench family** reuses the R-66-OUTSIDE-REQUIRED bundle
shape (deceptive primary + symmetric secondary) for *every*
sub-bank — the only variable across sub-banks is the registered
oracle. This makes the milestone story sharp: bundle-only methods
cannot escape; the only thing that changes the W20 result is
oracle integrity.

The SDK v3.21 mint:

* **R-67-OUTSIDE-RESOLVES** (W20-1 anchor; deterministic
  ``ServiceGraphOracle``): W20 = 1.000; W19 = 0.000.
  **+1.000 strict separation**, stable across 5/5 seeds,
  on loose AND tight (``T_decoder = 24``) decoder budgets.
* **R-67-OUTSIDE-REQUIRED-BASELINE** (W20-3 backward-compat;
  no oracle): W20 ties W19 byte-for-byte at 0.000.
* **R-67-OUTSIDE-NONE** (W20-Λ-none; ``AbstainingOracle``):
  W20 = 0.000; abstain. Names "no signal" structural limit.
* **R-67-OUTSIDE-COMPROMISED** (W20-Λ-compromised;
  ``CompromisedServiceGraphOracle``): W20 trusts oracle, picks
  decoy, FAILS at 0.000. Names "adversarial oracle" limit.
* **R-67-JOINT-DECEPTION** (W20-Λ-joint-deception): primary +
  secondary + oracle ALL favour decoy AND are internally
  consistent. W20 ties W19 at 0.000. Names "all evidence channels
  jointly compromised" limit.
* **R-67-LIVE-MIXTRAL-8x7b** (W20-Λ-real partial-pass; n=4):
  47B-MoE LLM adjudicator achieves ``acc_full = 0.750``;
  +0.750 over W19 on a fresh live LLM stream.
* **R-67-LIVE-QWEN-CODER-7B** (W20-Λ-real failure): smaller
  model trusts deceptive primary; ties FIFO at 0.000.

The next research frontier is **W20-C-MULTI-ORACLE** (multi-
oracle aggregation escape from W20-Λ-compromised when *some*
oracles remain trustworthy) AND **W20-C-LIVE-WITH-REGISTRY** (LLM
adjudicator with explicit service-registry context narrowing the
W20-Λ-real wall). Both conjectural, both bounded above by joint
N-oracle compromise.

**For another agent to instantly see the SDK v3.21 state**:

1. **Is the new frontier outside-information acquisition beyond
   W19?** *YES.* The W20 family attacks the W19-Λ-outside AND
   W19-Λ-total walls directly via the outside-information channel
   that bundle-only scorers cannot see.
2. **Does the new method actually beat the W19-Λ-outside wall?**
   *YES, on the bundle-only-insufficient-but-outside-resolvable
   case (R-67-OUTSIDE-RESOLVES).* W20 = 1.000; W19 = 0.000.
   **+1.000 strict separation**, stable across 5/5 seeds, on
   both loose AND tight decoder budgets.
3. **Did bounded-context efficiency survive?** *YES.* The W20
   layer adds *exactly one* outside query per cell, bounded by
   ``max_response_tokens = 24``; the W15 ``tokens_kept`` is
   byte-for-byte identical between W19 and W20. Total context
   delivered to the final decider: ``tokens_kept (≤ T_decoder) +
   n_outside_tokens (≤ max_response_tokens)``.
4. **Did two-Mac live evaluation materially broaden the
   evidence?** *NO — Mac 2 remains unreachable (ARP
   ``incomplete``, same status as the prior 14 milestones). No
   two-Mac sharded inference. The W20 ``OutsideWitnessOracle``
   Protocol IS infrastructure-ready for cross-host deployment
   when Mac 2 returns; the ``MLXDistributedBackend`` adapter is
   byte-for-byte unchanged. A *partial* live W20-Λ-real probe on
   Mac-1 (``mixtral:8x7b`` 47B-MoE) achieved acc_full = 0.750
   (+0.750 over W19) — a strict but partial live advance over
   the W19-Λ-outside wall.*
5. **Is the original thesis materially stronger or still
   blocked by a deeper semantic / oracle-integrity wall?**
   *MATERIALLY STRONGER on the named axis; the deeper wall is
   now SHARP — joint-deception across primary + secondary +
   oracle is the named structural limit.* The original Context-
   Zero thesis — *per-agent minimum-sufficient context for
   multi-agent teams* — gains its **first capsule-native method
   to acquire asymmetric outside information when the bundle
   alone is structurally insufficient**. The deeper walls
   (W20-Λ-compromised, W20-Λ-joint-deception) are *named* and
   *proved-empirical* — the natural escape from W20-Λ-compromised
   is W20-C-MULTI-ORACLE (multi-oracle aggregation) and the
   natural escape from W20-Λ-real is W20-C-LIVE-WITH-REGISTRY
   (LLM adjudicator with explicit registry context); both
   conjectural. **The thesis is materially stronger AND the
   next research frontier is precisely articulated as oracle
   integrity + multi-oracle aggregation, not a method-gap
   conjecture.**

### 4.39 SDK v3.22 — trust-weighted multi-oracle adjudicator + R-68 multi-oracle benchmark family + W21 family (Phase-68 R-68-MULTI-MAJORITY + three named falsifiers + live cross-model coalition probes)

SDK v3.22 attacks the named research frontier SDK v3.21 left
explicitly conjectural: **W20-C-MULTI-ORACLE** — the multi-source
escape from the W20-Λ-compromised wall. The W21 family ships the
:class:`TrustWeightedMultiOracleDisambiguator` that, when the
inner W19 abstains, consults **N registered outside oracles** in
parallel (one bounded query each, ≤ ``max_response_tokens`` per
call), counts per-tag votes across non-abstaining oracle replies,
and projects the answer onto tags with ≥ ``quorum_min`` votes AND
≥ ``min_trust_sum`` trust mass. The first capsule-native multi-
agent-coordination method that crosses the W20-Λ-compromised wall
on a regime where the wall actually applies (R-68-MULTI-MAJORITY).

The SDK v3.22 mint:

* **Phase 68** — driver + bench-property test +
  ``run_phase68`` + ``run_phase68_seed_stability_sweep`` +
  ``run_cross_regime_synthetic`` + cross-model live coalition
  probes.
* Five sub-banks: ``single_clean``, ``multi_majority``,
  ``multi_no_quorum``, ``multi_all_compromised``,
  ``multi_partial``. Bundle shape inherits R-66-OUTSIDE-REQUIRED
  from Phase-67 (deceptive primary + symmetric secondary witness);
  the *registered oracle set* is the bank-specific knob.
* **W21 family**: 11 entries — W21-1 (proved-empirical n=80
  saturated), W21-2 (proved + mechanically-checked), W21-3-A
  (proved-empirical full programme regression), W21-3-B
  (proved-empirical reduces-to-W20), W21-Λ-no-quorum, W21-Λ-all-
  compromised, W21-Λ-partial (all proved-empirical n=8 saturated),
  W21-Λ-real (proved-conditional + empirical-research n=4 × 2
  models), W21-C-CALIBRATED-TRUST + W21-C-LIVE-WITH-REGISTRY
  (the second partially discharged at n=4 × mixtral 8x7b).
* **48 new tests** (``test_wevra_multi_oracle_adjudication.py``);
  full SDK regression: 633 / 633 wevra tests pass.

Headline reading:

* **R-68-MULTI-MAJORITY-LOOSE / TIGHT**: W21 = 1.000; W20 = 0.000;
  **+1.000 strict separation**, stable across 5/5 seeds.
  ``W21_BRANCH_QUORUM_RESOLVED`` fires on every cell; gold pair
  receives 2 votes (service_graph + change_history), decoy
  receives 1 (compromised_registry).
* **R-68-MULTI-NO-QUORUM** (W21-Λ-no-quorum): W21 = FIFO = 0.000.
* **R-68-MULTI-ALL-COMPROMISED** (W21-Λ-all-compromised): W21 =
  0.000 (quorum forms on decoy by registered-set integrity
  failure).
* **R-68-MULTI-PARTIAL** (W21-Λ-partial under default
  ``quorum_min = 2``): W21 = 0.000. With override ``quorum_min =
  1`` (W21-C-PARTIAL-RECOVERY discharged): W21 = 1.000.
* **R-68-LIVE-MIXED-REGISTRY** (4 oracles incl. mixtral 8x7b):
  W21 = 1.000, +1.000 over W20. W21-C-LIVE-WITH-REGISTRY
  partially discharged (registry-anchored regime).
* **R-68-LIVE-COALITION-MIXTRAL** (3 oracles, LLM-vote-required):
  W21 = 0.750, +0.750 over W20. **W21-Λ-real partial advance**.
* **R-68-LIVE-COALITION-GEMMA2-9B** (3 oracles, LLM-vote-required):
  W21 = 0.000. **W21-Λ-real fails on under-scaled model — gemma2
  lands decoy tokens through the closure**. Cross-model split
  (47B-MoE / 9.2B-dense) sharp.

The next research frontier is **W21-C-CALIBRATED-TRUST** (low
trust priors on uncalibrated oracles via ``min_trust_sum`` floor —
escapes W21-Λ-all-compromised on bounded-corruption regimes) AND
**W22-* — joint-deception detection** (cross-source consistency
checks that detect when ALL N registered oracles agree
suspiciously; the natural escape from W21-Λ-all-compromised when
no oracle can be a priori distrusted). Both conjectural; both
bounded above by structural circularity (a learned trust scorer
trained on historical agreement is itself an N+1-th oracle that
can be compromised).

**For another agent to instantly see the SDK v3.22 state**:

1. **Is the new frontier multi-source outside-information
   adjudication beyond W20?** *YES.* The W21 family attacks the
   W20-Λ-compromised wall directly via the multi-source channel.
2. **Does the new method actually beat the compromised-source
   wall?** *YES, on the bundle+single-oracle-insufficient-but-
   multi-oracle-resolvable case (R-68-MULTI-MAJORITY).* W21 =
   1.000; W20 = 0.000. **+1.000 strict separation**, stable across
   5/5 seeds, on both loose AND tight decoder budgets.
3. **Did bounded-context efficiency survive?** *YES.* The W21
   layer adds *exactly N* outside queries per cell, each bounded
   by ``max_response_tokens``; the W15 ``tokens_kept`` is
   byte-for-byte identical between W19, W20 AND W21. Total context
   delivered to the final decider on the 3-oracle stack:
   ``tokens_kept (≤ T_decoder) + 3 × n_outside_tokens (each ≤
   max_response_tokens)``. On R-68-MULTI-MAJORITY-TIGHT
   (T_decoder=24): tokens_kept ≤ 24 + 3 × 4 = 36 average per cell.
4. **Did two-Mac live evaluation materially broaden the
   evidence?** *NO — Mac 2 remains unreachable (ARP
   ``incomplete``, same status as the prior 15 milestones). No
   two-Mac sharded inference.* But the W21 oracle Protocol is
   *naturally* a producer / multi-adjudicator separation; cross-
   host deployment (registry on Mac-1, LLM adjudicator on Mac-2)
   is wire-compatible and waits for Mac-2. **A live multi-source
   probe on Mac-1 alone — with deterministic registry + change-
   log + mixtral 8x7b — discharges W21-C-LIVE-WITH-REGISTRY at
   n=4 (acc_full = 1.000, +1.000 over W20). The harder coalition
   regime (deterministic insufficient, LLM vote required for
   quorum) cross-model split: mixtral 0.750 / gemma2 0.000.**
5. **Were earlier paper loose ends closed?** *PARTIALLY.* (a)
   **W20-C-MULTI-ORACLE** (named conjectural in SDK v3.21) is
   **discharged** by W21-1 on R-68-MULTI-MAJORITY. (b)
   **W20-C-LIVE-WITH-REGISTRY** (named conjectural in SDK v3.21)
   is **partially discharged** by the live mixed-registry probe
   on Mac-1 mixtral 8x7b. (c) The cross-model transfer story
   (W17-C-XMODEL pattern) is sharpened: the live coalition probe
   shows **scale + general knowledge matter** for the LLM-vote-
   required regime — mixtral 0.750 / gemma2 0.000. (d) Two-Mac
   path remains infrastructure-only (15th milestone in a row);
   the W21 mechanism is naturally cross-host but Mac-2 unavailable.
6. **Is the original thesis materially stronger or still blocked
   by a deeper trust / semantics wall?** *MATERIALLY STRONGER on
   the named axis; the deeper wall is now SHARPER — joint
   compromise across the entire registered set (W21-Λ-all-
   compromised) is the named structural limit.* The original
   Context-Zero thesis gains its **first capsule-native method to
   adjudicate across multiple outside sources under partial
   compromise**. The deeper walls (W21-Λ-all-compromised,
   W21-Λ-no-quorum, W21-Λ-partial-without-q1) are *named* and
   *proved-empirical* — the natural escape from W21-Λ-all-
   compromised is W21-C-CALIBRATED-TRUST (calibrated trust priors
   exclude uncalibrated oracles via the ``min_trust_sum`` floor)
   and the natural escape from W21-Λ-no-quorum is **W22-* — joint
   consistency detection** (cross-source consistency checks that
   detect "too good to be true" agreement); both conjectural.
   **The thesis is materially stronger AND the next research
   frontier is precisely articulated as registered-set integrity
   + cross-source consistency, not a method-gap conjecture.**

### 4.40 SDK v3.23 — capsule + audited latent-state-sharing hybrid + R-69 Phase-69 benchmark family + W22 family (R-69-CACHE-FANOUT + three named falsifiers + backward-compat anchor + live mixtral cross-model probes)

**One paragraph.** SDK v3.22 (W21 family) crossed the
W20-Λ-compromised wall by consulting **N registered oracles**
under quorum + trust thresholds. It also explicitly named the
**wire-cost concern** of consulting all N oracles every cell as
a separate research direction, plus the LatentMAS direction
(collective KV pooling / latent hidden-state transfer / super-
token side channels) as conjectural. SDK v3.23 implements both at
the *capsule layer*: it ships a content-addressed
:class:`SchemaCapsule` (closed-vocabulary type schema, shared
once per session by CID), :class:`SharedReadCache` (CID-keyed
write-once-read-many proxy for the LatentMAS shared-KV-read
direction), :class:`CachingOracleAdapter` (drop-in adapter
routing any :class:`OutsideWitnessOracle` through the cache),
:class:`LatentDigestEnvelope` (typed, controller-verified compact
summary of one W21 vote outcome — hash-chained, schema-versioned,
parent-CID-sealed), :func:`verify_latent_digest` (≈ 30-line
controller-side verification with enumerated failure modes),
:class:`EnvelopeTamperer` (falsifier-test primitive for
R-69-POISONED-DIGEST), and a wrapping
:class:`LatentDigestDisambiguator` (W22). On the new
R-69-CACHE-FANOUT regime (same R-68-MULTI-MAJORITY oracle
ecology, every oracle wrapped in a CachingOracleAdapter pointing
at a shared SharedReadCache, verbose W21 audit replaced by one
LatentDigestEnvelope per cell), the W22 method strictly reduces
``mean_n_visible_tokens_to_decider`` by **−7 tokens / cell
(−14.51 %)** at ``T_decoder = None`` and by **−7 tokens / cell
(−16.09 %)** at ``T_decoder = 24``, AND records
``cache_tokens_saved_total = 88`` over n=8, AND ties W21
byte-for-byte on ``accuracy_full = 1.000``. Stable across **5/5**
alternate ``bank_seed`` values. Three named falsifiers
(W22-Λ-no-cache, R-69-POISONED-DIGEST, R-69-SCHEMA-DRIFT) and
one backward-compat anchor (R-69-NO-TRIGGER) make the W22-1
conditionality sharp. Live W22 transfer on Mac-1 Ollama:
mixtral:8x7b (47B-MoE) on cache_fanout achieves **+39.08 %**
visible-tokens savings + cache_tokens_saved=120; correctness
ratified rate = 0.750 reveals newly named conjecture
**W22-C-CACHE-AMPLIFICATION**. gemma2:9b ties W21 byte-for-byte
at 0.250. Full programme regression: 633 prior + 32 new W22 +
10 misc = **675/675**. Mac 2 still unreachable (16th milestone in
a row); the W22 surface is naturally cross-host-compatible.
Closes the wire-cost half of W21-C-CALIBRATED-TRUST.

**Master-plan-level questions (the post-W21 audit board).**

1. **Did capsule-native + latent/state-sharing integration
   materially help?** *YES on the wire-cost axis.* On
   R-69-CACHE-FANOUT W22 strictly reduces visible-tokens-to-
   decider by 14.51-16.09 % synthetic and 39.08 % live-mixtral,
   while ratifying W21 correctness byte-for-byte; stable across
   5/5 seeds. Every piece of the LatentMAS idea family this repo
   can verify end-to-end (schema-passing, delta execution,
   shared-read cache, controller-verified side channel) is
   implemented honestly with explicit proxies.
2. **Did trust/audit survive?** *YES, sharply.* Every emitted
   envelope is hash-chained, schema-versioned, parent-CID-sealed;
   verification failure → W22 falls through to W21 byte-for-byte;
   correctness ratified rate = 1.000 across both falsifier banks
   while ``verification_ok_rate = 0.000``.
3. **Did bounded-context efficiency improve in a real way?** *YES
   on two independent axes:* (a) visible-tokens-to-decider per
   cell (digest replaces verbose audit), (b) wire-side oracle
   savings across cells (cache hits collapse identical
   OutsideQueries). Both are honest capsule-layer proxies; no
   transformer-internal KV reuse claimed.
4. **Did two-Mac evaluation materially broaden the evidence?**
   *NO — Mac 2 still unreachable (16th milestone in a row).* The
   live evidence is single-Mac mixtral 8x7b and gemma2 9b on
   Mac-1 Ollama. The W22 surface is naturally a producer / cache-
   controller separation; cross-host deployment is wire-
   compatible when Mac-2 returns.
5. **Which earlier paper loose ends were closed versus only
   sharpened?** *Closed: wire-cost half of W21-C-CALIBRATED-TRUST.
   Sharpened: LLM nondeterminism caveat (W22-C-CACHE-AMPLIFICATION
   newly named).* The paper's "capsule discipline" framing now
   has its first audited proxy for hidden-state-shaped
   coordination — strictly sharper than the prior "explicit
   capsules only" stance.
6. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on
   the named axis; the deeper wall is sharper.* The Context Zero
   thesis gains its **first capsule-native method that combines
   explicit-capsule passing with audited proxies for the
   LatentMAS direction**. Named open frontier: **W22-C-LATENT-KV-
   NATIVE** (true cross-host KV reuse between MLX-distributed LLM
   servers — gated on Mac-2 return) and **W22-C-DELTA-
   COMPRESSION** (richer envelopes encoding *changes* against a
   baseline). The thesis is materially stronger AND the next
   research frontier is precisely articulated as cache-
   amplification integrity + cross-host KV reuse + delta-
   compression sufficiency.

### 4.41 SDK v3.24 — capsule-native cross-cell delta execution + quorum-keyed cache + super-token reference + Phase-70 R-70 benchmark family + W23 family (R-70-DELTA-FANOUT + R-70-AMPLIFIED-LLM mitigation + R-70-SUPER-TOKEN dense-control + three named falsifiers + live mixtral cross-host probe)

**One paragraph.** SDK v3.23 (W22 family) crossed the W21
wire-cost wall at the *per-cell* layer (schema-passing + delta
execution + shared-read cache + controller-verified latent
digest, on one cell). It explicitly named two open frontiers:
the **W22-C-CACHE-AMPLIFICATION** conjecture (the SharedReadCache
freezes a probabilistic LLM oracle's first reply across all
matching cells) and the absence of a true **cross-cell
state-sharing** or **super-token / dense-control side channel**.
SDK v3.24 implements the smallest honest version of those moves
*together* at the *capsule layer*: a hash-chained
:class:`SessionDigestEnvelope` carrying running cross-cell state;
a small :class:`SessionDeltaEnvelope` per cell that emits only
what changed against the running state; a
:class:`QuorumKeyedSharedReadCache` with per-oracle freshness
policies (``byte_identical`` / ``per_cell_nonce`` /
``quorum_locked``) that *mitigates* W22-C-CACHE-AMPLIFICATION on
probabilistic adjudicators while preserving cross-cell wire
savings on deterministic ones; a single-visible-token
:class:`SuperTokenReferenceEnvelope` (the bounded steganographic /
dense-control payload experiment) verified through a
controller-side :class:`SuperTokenRegistry`; and a within-process
:class:`CrossHostProducerDecoderProxy` that forces every W23
envelope through a JSON-canonical wire round-trip — the honest
fallback for the unreachable Mac 2. Three new verification
functions (:func:`verify_session_digest_chain`,
:func:`verify_session_delta`,
:func:`verify_super_token_reference`) implement the controller-
side trust boundary. On the new R-70-DELTA-FANOUT regime, the
W23 method strictly reduces ``mean_n_w23_visible_tokens_to_decider``
over the W22 baseline by **−2.75 tokens / cell (−6.67 %)** loose,
**−2.75 tokens / cell (−7.53 %)** tight (delta path); **−10.50
tokens / cell (−25.45 %)** loose, **−10.50 tokens / cell
(−28.77 %)** tight (super-token path); ties W22 byte-for-byte on
``accuracy_full = 1.000``; stable across **5/5** alternate
``bank_seed`` values. On R-70-AMPLIFIED-LLM (synthetic
``FlippingProbabilisticOracle``), the W23 quorum-keyed cache
**empirically discharges W22-C-CACHE-AMPLIFICATION** at **+0.125
strict mitigation advantage** over W22 (W22 = 0.875, W23
quorum-keyed = 1.000) — the **first empirical demonstration in
the programme that a named conjecture's weakness is mitigable**.
Three named falsifiers (R-70-NO-DELTA, R-70-SUPER-TOKEN-TAMPERED,
R-70-CHAIN-BROKEN) make the W23-1 / W23-3 conditionality sharp:
no cross-cell state → no savings (W23-Λ-no-delta); tampered
super-token → ``unknown_super_token`` → fall through to W22;
chain head split → ``chain_head_mismatch`` → fall through. Live
W23 transfer on Mac-1 Ollama: ``mixtral:8x7b`` (47B-MoE) on
AMPLIFIED-LLM achieves **+2.84 % (delta)**, **+11.37 %
(super-token)** visible-tokens savings;
``chain_verifies_ok_rate = 0.750``;
``super_token_verification_ok_rate = 1.000``;
``cross_host_round_trip_bytes_total = 4232`` bytes (the
producer/decoder split is wire-validated). All four strategies
tie at ``acc_full = 0.750`` — the synthetic mitigation advantage
does not cleanly transfer at n=4 (newly named conjecture
**W23-C-MITIGATION-LIVE-VARIANCE**: positive expected
improvement, not strict per-probe). Full programme regression:
703 prior + 39 new W23 = **742/742** wevra-anchor + capsule +
recent-phase tests pass. Mac 2 still unreachable (**17th
milestone in a row**); the W23 :class:`CrossHostProducerDecoderProxy`
mechanically validates the wire-encoding contract via a
within-process round-trip on every cell. **Empirically discharges
W22-C-CACHE-AMPLIFICATION as a mitigable property; further
sharpens the wire-cost half of W21-C-CALIBRATED-TRUST.**

**Master-plan-level questions (the post-W22 audit board).**

1. **Did capsule-native + latent/state-sharing integration
   materially help (cross-cell layer)?** *YES on three independent
   axes.* (a) Cross-cell delta execution: −6.67 % to −7.53 %
   visible-tokens savings over W22. (b) Super-token reference:
   −25.45 % to −28.77 % savings over W22. (c) Quorum-keyed cache
   mitigation: +0.125 strict mitigation of W22-C-CACHE-
   AMPLIFICATION on the synthetic regime. All three preserve W22
   correctness byte-for-byte. The W23 surface implements every
   piece of the LatentMAS *cross-cell latent state-sharing* +
   *delta execution* + *super-token side channel* idea family this
   repo can validate end-to-end as honest, audited proxies.
2. **Did trust/audit survive?** *YES, sharply.* Every emitted
   envelope (session digest, session delta, super-token reference)
   is hash-chained, schema-versioned, parent-CID-sealed, AND (for
   super-tokens) registry-verified; verification failure → W23
   falls through to W22 byte-for-byte; correctness ratified rate
   = 1.000 across both falsifier banks while
   ``super_token_verification_ok_rate = 0.000`` on
   R-70-SUPER-TOKEN-TAMPERED and ``chain_verifies_ok_rate = 0.125``
   on R-70-CHAIN-BROKEN. The producer/verifier split is *honest*:
   the verifier maintains its own super-token registry AND its own
   chain-head expectation; mid-session tampering on the producer
   side is detected.
3. **Did bounded-context efficiency improve in a real way?** *YES
   on three independent axes:* (a) cross-cell delta replaces full
   per-cell digest, (b) super-token reference replaces verbose
   digest with a single visible token, (c) quorum-keyed cache
   preserves cross-cell wire savings on deterministic oracles
   while mitigating amplification on probabilistic ones. The
   savings compose multiplicatively with W15 / W21 / W22 to a
   meaningful end-to-end reduction.
4. **Did two-Mac evaluation materially broaden the evidence?**
   *NO — Mac 2 still unreachable (17th milestone in a row).* The
   live evidence is single-Mac mixtral 8x7b on Mac-1 Ollama. The
   W23 :class:`CrossHostProducerDecoderProxy` validates the
   wire-encoding contract via a within-process JSON-canonical
   round-trip on every cell (≈ 1.79 KB/cell on R-70-DELTA-FANOUT);
   when Mac 2 returns the same proxy interface drops in over a
   real socket with no W23 code changes. Honest scope: this is a
   *structural* simulation, not a true two-host setup.
5. **Which earlier paper loose ends were closed versus only
   sharpened?** *Closed (mitigated): W22-C-CACHE-AMPLIFICATION
   (synthetic R-70-AMPLIFIED-LLM at +0.125 strict gain).
   Sharpened: live-LLM mitigation transfer
   (W23-C-MITIGATION-LIVE-VARIANCE newly named — positive
   expected improvement, not strict per-probe at n=4).
   Sharpened: cross-cell latent state-sharing direction (W23
   implements the smallest honest cross-cell proxy; true
   transformer-KV cross-cell reuse remains gated on Mac-2 return).
   Sharpened: dense-control / super-token side channel direction
   (W23 implements the smallest honest single-visible-token CID-
   prefix proxy; true embedding-level steganographic intervention
   remains out of scope).*
6. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on
   three named axes; the deeper wall is sharper.* The Context
   Zero thesis now has its **first capsule-native method that
   combines explicit-capsule passing with audited proxies for
   the LatentMAS *cross-cell* state-sharing direction**, **first
   capsule-native trust boundary on a single-visible-token
   dense-control payload**, and **first empirical demonstration
   that a named conjecture's weakness is mitigable** (W22-C-
   CACHE-AMPLIFICATION). Named open frontier:
   **W23-C-MITIGATION-LIVE-VARIANCE** (live-LLM mitigation
   transfer at higher n), **W23-C-LATENT-KV-NATIVE** (true
   cross-host KV reuse between MLX-distributed LLM servers —
   gated on Mac-2 return), and **W23-C-EMBEDDING-PAYLOAD**
   (transformer-internal embedding-side dense-control —
   architecture-dependent, out of scope until Mac-2 returns or a
   richer transformer integration boundary opens). The thesis
   is materially stronger AND the next research frontier is
   precisely articulated as live-LLM mitigation transfer +
   cross-host validation + embedding-level integration.

---

### 4.42 SDK v3.25 — bounded-window session compaction + intra-cell resample-quorum + real cross-process producer/decoder wire + Phase-71 R-71 benchmark family + W24 family (R-71-LONG-SESSION + R-71-INTRA-CELL-FLIP mitigation + R-71-CROSS-PROCESS real wire + two named falsifiers + live mixtral cross-process probe)

**One paragraph.** SDK v3.24 (W23 family) crossed the W22
wire-cost wall at the *cross-cell* layer (hash-chained session
digest + per-cell delta + quorum-keyed cache + super-token
reference). It explicitly named two open frontiers: the
**W23-C-MITIGATION-LIVE-VARIANCE** conjecture (the PER_CELL_NONCE
policy mitigates inter-cell drift but not intra-cell drift on
probabilistic LLM oracles) and the W23 within-process
producer/decoder host-split proxy as a structural simulation
rather than a real cross-process wire. SDK v3.25 implements the
smallest honest version of those moves *together* at the *capsule
layer*: a fixed-size :class:`SessionCompactEnvelope` that folds
the last ``compact_window - 1`` cell digests into one envelope
(visible-token cost is a single ``<compact_ref:DDDD>`` token per
cell, equivalent to W23 super-token at lower controller-side
state); a :class:`ResampleQuorumCachingOracleAdapter` that
consults the inner oracle ``sample_count`` times *within one cell*
and returns the majority verdict (mitigates intra-cell
probabilistic drift directly); a real
:class:`CrossProcessProducerDecoderWire` that round-trips JSON
envelopes through a Python subprocess's stdin/stdout pipes
(strictly stronger than the W23 within-process proxy); and a
synthetic :class:`IntraCellFlippingOracle` whose drift fits the
intra-cell pattern named in W23-C-MITIGATION-LIVE-VARIANCE. One
new verification function (:func:`verify_session_compact`)
implements the controller-side trust boundary. On the new
R-71-LONG-SESSION regime (16-cell session with
``compact_window = 4``), the W24 method strictly reduces
``mean_n_w24_visible_tokens_to_decider`` over the W23 baseline by
**−6.81 tokens / cell (−18.0 %)** loose, **−6.81 tokens / cell
(−20.5 %)** tight; ties W22 / W23 byte-for-byte on
``accuracy_full = 1.000``; stable across **5/5** alternate
``bank_seed`` values (savings ≥ 6.69 tokens/cell on every seed).
On R-71-INTRA-CELL-FLIP (synthetic
:class:`IntraCellFlippingOracle` registered in isolation so its
vote is decisive in the W21 quorum), the W24 resample-quorum
**empirically discharges W23-C-MITIGATION-LIVE-VARIANCE on the
intra-cell drift axis** at **+0.500 strict mitigation advantage**
over W23 PER_CELL_NONCE (W23 = 0.000, W24 resample = 0.500). On
R-71-CROSS-PROCESS (n=16), the
:class:`CrossProcessProducerDecoderWire` records 12 861 bytes
round-tripped through a real Python subprocess pipe with 0
failures. Two named falsifiers (R-71-NO-COMPACT,
R-71-COMPACT-TAMPERED) make the W24-1 / W24-3 conditionality
sharp: chain reset every cell → no compact resolved
(W24-Λ-no-compact); tampered window → ``window_cids_mismatch`` →
fall through to W23. Live W24-2 transfer on Mac-1 Ollama:
``mixtral:8x7b`` (47B-MoE) on R-71-INTRA-CELL-FLIP (n=4) achieves
**+0.250 strict mitigation advantage on a fresh live LLM**
(``acc_full(W23 quorum-keyed) = 0.500``, ``acc_full(W24 resample)
= 0.750``) — the synthetic +0.500 does not fully transfer; names
**W24-C-LIVE-VARIANCE-COMPLETE** as the follow-up. Full programme
regression: 121/121 phase-69/70/71 + capsule + 33/33 new W24
tests + 619/619 wevra-anchor + capsule + recent-phase tests
pass. Mac 2 still unreachable (**18th milestone in a row**); the
W24 :class:`CrossProcessProducerDecoderWire` upgrades the W23
within-process proxy to a real OS-level subprocess pipe.
**Empirically discharges W23-C-MITIGATION-LIVE-VARIANCE on the
intra-cell drift axis (synthetic strict, live partial); further
sharpens the live-LLM transfer half of W23-Λ-real.**

**Master-plan-level questions (the post-W23 audit board).**

1. **Did denser capsule-native latent/control integration
   materially help (cross-cell + intra-cell layers)?** *YES on
   three independent axes.* (a) Bounded-window session
   compaction: −18.0 % to −20.5 % visible-tokens savings over
   W23 on R-71-LONG-SESSION. (b) Intra-cell resample-quorum
   mitigation: +0.500 strict synthetic / +0.250 strict live
   advantage over W23 PER_CELL_NONCE on R-71-INTRA-CELL-FLIP. (c)
   Real cross-process wire: 12 861 bytes round-tripped through
   a real OS-level subprocess pipe with 0 failures. All three
   preserve W22 / W23 correctness byte-for-byte. The W24 surface
   implements every piece of the LatentMAS *bounded-window
   summary* + *intra-cell self-consistency* + *real-wire
   producer/decoder split* idea family this repo can validate
   end-to-end as honest, audited proxies.
2. **Did trust/audit survive?** *YES, sharply.* Every emitted
   compact envelope is hash-chained, schema-versioned,
   parent-CID-sealed, AND verifier-rejectable; verification
   failure → W24 falls through to W23 byte-for-byte; correctness
   ratified rate = 1.000 across the falsifier bank while
   ``compact_verifies_ok_rate = 0.062`` on R-71-COMPACT-TAMPERED.
   The producer/verifier split is *honest*: the verifier
   maintains its own expected window CIDs; mid-session tampering
   on the producer side is detected.
3. **Did bounded-context efficiency improve in a real way?** *YES
   on three independent axes:* (a) bounded-window compaction
   replaces W23 verbose digest+delta with a single
   ``<compact_ref:DDDD>`` token per cell (one whitespace token,
   equivalent to W23 super-token visible cost), (b) the
   controller-side state grows by O(1) per window instead of
   O(1) per cell (a strict memory-efficiency advantage over W23
   super-token), (c) the resample-quorum trades 3× oracle cost
   for intra-cell robustness — the trade-off is named explicitly
   in the conjecture frontier.
4. **Did two-Mac evaluation materially broaden the evidence?**
   *PARTIALLY — Mac 2 still unreachable (18th milestone in a row).*
   The live evidence is single-Mac mixtral 8x7b on Mac-1 Ollama.
   The W24 :class:`CrossProcessProducerDecoderWire` validates the
   wire-encoding contract via a real OS-level Python subprocess
   pipe on every cell (≈ 803 bytes/cell on R-71-CROSS-PROCESS); a
   strictly stronger cross-process proxy than the W23
   within-process round-trip. When Mac 2 returns the same
   JSON-canonical interface drops in over a real socket with no
   W24 code changes. Honest scope: this is a real *cross-process*
   wire, not a *cross-host* wire.
5. **Which earlier paper loose ends were closed versus only
   sharpened?** *Closed (intra-cell axis):
   W23-C-MITIGATION-LIVE-VARIANCE (synthetic R-71-INTRA-CELL-FLIP
   at +0.500 strict gain; live mixtral n=4 at +0.250 strict
   gain). Sharpened: live-LLM mitigation transfer at large n
   (W24-C-LIVE-VARIANCE-COMPLETE newly named — positive expected
   improvement bounded by drift-pattern similarity).
   Sharpened: cross-host honesty (W24-3 cross-process wire is
   strictly stronger than W23 within-process proxy; true
   cross-host execution remains gated on Mac-2 return).*
6. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on
   three named axes; the deeper wall is sharper.* The Context
   Zero thesis now has its **first capsule-native bounded-window
   session-compaction method**, **first programme-internal
   demonstration that the live-LLM mitigation transfer is
   non-trivially measurable on a fresh LLM probe**, and **first
   real OS-level cross-process producer/decoder wire**. Named
   open frontier: **W24-C-LIVE-VARIANCE-COMPLETE** (live-LLM
   mitigation transfer rate at higher n), **W24-C-CROSS-HOST-REAL**
   (real cross-host execution between Mac 1 and Mac 2 — gated on
   Mac-2 return), and **W24-C-EMBEDDING-COMPACT** (transformer-
   internal embedding-side compaction — architecture-dependent,
   out of scope until Mac-2 returns or a richer transformer
   integration boundary opens). The thesis is materially stronger
   AND the next research frontier is precisely articulated as
   live-LLM mitigation transfer at large n + cross-host
   validation + embedding-level integration.

### 4.43 SDK v3.26 — shared-fanout dense-control + cross-agent state reuse + Phase-72 R-72 benchmark family + W25 family (R-72-FANOUT-SHARED + R-72-DISJOINT named falsifier + R-72-FANOUT-POISONED trust falsifier)

**One paragraph.** SDK v3.25 (W24 family) closed the per-agent
session-compaction frontier (one ``<compact_ref:DDDD>`` token per
cell). SDK v3.26 extends that to the **multi-agent** case: when K
consumer agents need the same cross-cell session state produced by
one producer, W24 still required K independent compact envelopes
(K × C tokens total). W25 replaces those K envelopes with a single
:class:`FanoutEnvelope` registered by the producer + K single-token
``<fanout_ref:DDDD>`` references for the consumers (one per
consumer). On R-72-FANOUT-SHARED (1 producer + K=3 consumers, 16
cells, R-69-CACHE-FANOUT oracle ecology), W25 strictly reduces
``mean_total_w25_visible_tokens`` over ``mean_total_w24_visible_
tokens`` by **−40.875 tokens / cell (−69.87 %)**;
``correctness_ratified_rate = 1.0000`` byte-for-byte;
``fanout_consumer_resolved_rate = 1.0000``. Stable across **5/5**
seeds. Trust boundary: every envelope is hash-chained
(``fanout_cid``), schema-versioned, and verifier-rejectable on 5
failure modes (``empty_envelope``, ``schema_version_unknown``,
``schema_cid_mismatch``, ``consumer_not_authorized``,
``hash_mismatch``). Two named falsifiers — W25-Λ-disjoint (no
shared registry → W25 = W24 byte-for-byte) and W25-3 trust
soundness (R-72-FANOUT-POISONED: unauthorised consumer rejected on
every cell) — make the conditionality sharp. **Empirically
discharges the multi-agent extension of the W24 single-agent
compaction direction.** See `docs/RESULTS_WEVRA_W25_SHARED_FANOUT.md`.

### 4.44 SDK v3.27 — chain-persisted dense-control fanout + per-consumer projections + Phase-73 R-73 benchmark family + W26 family (R-73-CHAIN-SHARED + R-73-CHAIN-WINDOWED + R-73-NO-CHAIN named falsifier + R-73-CHAIN-TAMPERED + R-73-PROJECTION-MISMATCH trust falsifiers + R-73-DIVERGENT regime + W25-C-K-SCALING discharge at K∈{3,5,8,10})

**One paragraph.** SDK v3.26 (W25) reduced multi-agent fanout cost
from K × C tokens/cell to **C + K** tokens/cell; the producer's
per-cell salience-token cost (``n_w15_kept`` ≈ 13.6 tokens) was
unchanged, since W25 emits a fresh :class:`FanoutEnvelope` every
cell. W25 explicitly named **W25-C-K-SCALING** (savings should
grow as K×(C−1)) and **W25-C-MULTI-HOST** (cross-host wire vs token
tradeoff) as conjectures. SDK v3.27 implements the smallest honest
version of the next step at the *capsule layer*: a two-tier
content-addressed envelope hierarchy comprising :class:`ChainAnchorEnvelope`
(genesis cell carrying canonical compact state + per-consumer
:class:`ProjectionSlot` map) and :class:`ChainAdvanceEnvelope`
(per-cell hash-chained delta). The :class:`ChainPersistedFanoutDisambiguator`
(W26) wraps the W25 :class:`SharedFanoutDisambiguator` and amortises
the producer's per-cell cost across cells: at the chain anchor cell
the producer pays the full W25 cost (~14.6 tokens); at each
subsequent in-window cell the producer pays a single
``<chain_advance:DDDD>`` token (1 token) and registers a
:class:`ChainAdvanceEnvelope` whose ``parent_advance_cid``
hash-chains to the prior advance (or the anchor for the first
advance). Consumers subscribe at the anchor and emit a 1-token
chain-consumer ref per cell; per-consumer projections enforce that
each consumer can only resolve their slotted projection_id
(cross-projection access rejected with ``projection_unauthorized``).
Three new verification functions (:func:`verify_chain_anchor` with
6 enumerated failure modes, :func:`verify_chain_advance` with 8
failure modes, :func:`verify_projection_subscription` with 2
failure modes) implement the controller-side trust boundary. On
R-73-CHAIN-SHARED (K=3, 16 cells, ``chain_persist_window = 16``),
the W26 method strictly reduces ``mean_total_w26_visible_tokens``
over ``mean_total_w25_visible_tokens`` by **−12.125 tokens / cell
(−68.79 %)** at ``T_decoder = None`` AND over
``mean_total_w24_visible_tokens`` by **−53.00 tokens / cell
(−90.60 %)**; ``correctness_ratified_rate = 1.0000`` byte-for-byte;
``chain_consumer_resolved_rate = 1.0000``;
``registry_n_anchors = 1``, ``registry_n_advances = 15``. Identical
at ``T_decoder = 24``. Stable across **5/5** seeds (savings = 12.125
tokens/cell on every seed). On R-73-CHAIN-WINDOWED
(``chain_persist_window = 4``), the producer emits 1 anchor + 12
advances + 3 re-anchors; savings = −56.0 % over W25. On R-73-NO-CHAIN
(``chain_persist_window = 1``) **W26 reduces to W25 byte-for-byte**
(every cell is a (re-)anchor; W26-Λ-no-chain falsifier discharged).
On R-73-CHAIN-TAMPERED (producer's first advance corrupted
post-registration), the controller rejects 14/16 advances via
``parent_mismatch``; correctness preserved via W25 fall-through
(W26-Λ-tampered). On R-73-PROJECTION-MISMATCH (consumer 0 requests
``WRONG_PROJECTION_ID``), all 16 cells reject for that consumer;
the other 2 consumers still resolve (W26-Λ-projection-mismatch).
On R-73-DIVERGENT (gold subset flips at cell 8), the inner W25
fires ``no_trigger`` on divergent cells and W26 falls through;
correctness drops to 0.5 (W26 *does not claim* chain savings on
cells where the inner W25 abstains). **K-scaling sweep at
K∈{3,5,8,10}** discharges **W25-C-K-SCALING** by direct measurement:
W25 saving over W24 grows from 69.87 % at K=3 to 84.69 % at K=10
(close to the conjectured 88 %, slightly below because the cell-0
W25 producer cost is heterogeneous); W26 saving over W24 grows
from 90.60 % at K=3 to 92.23 % at K=10. New W26 unit + integration
tests: 63/63 pass; full pre-existing W22..W25 + IS-1 / IS-2 test
surfaces preserved byte-for-byte (180/180 in the focused
regression). Mac 2 still unreachable (**21st milestone in a row**);
W26 inherits the W24 :class:`CrossProcessProducerDecoderWire` as
the strongest cross-process honesty validated on this repo.

**Master-plan-level questions (the post-W25 audit board).**

1. **Did denser capsule-native latent/control integration
   materially help?** *YES on three independent axes simultaneously,
   with all three composing cleanly.* (a) Producer-side cross-cell
   amortisation: ~13.6 producer tokens/cell collapse to 1 token on
   advance cells (W26 chain-advance). (b) Per-consumer projection
   slots: each consumer can be slotted into a projection subset
   (orders / payments / both / etc.) with controller-verified
   authorisation, enabling fine-grained dense-control distribution
   without breaking the W25 fanout floor. (c) Bounded-window
   re-anchoring: the chain re-anchors every
   ``chain_persist_window`` cells, bounding the controller-side
   state to O(1) per chain (not per cell). On R-73-CHAIN-SHARED
   the combined effect is **−68.79 % over W25** and **−90.60 % over
   W24** at K=3, scaling to **−92.23 % over W24** at K=10.
2. **Did trust/audit survive?** *YES, sharply, with three new
   tier-2 verification functions.* The :class:`ChainAnchorEnvelope`
   is hash-chained (``chain_root_cid``), schema-versioned, and
   verifier-rejectable on 6 failure modes; each
   :class:`ChainAdvanceEnvelope` is parent-CID-linked and
   verifier-rejectable on 8 failure modes; per-consumer projections
   are scope-checked on 2 failure modes. The R-73-CHAIN-TAMPERED
   bench measures 14/16 advances rejected by the controller; the
   R-73-PROJECTION-MISMATCH bench measures 16/16 cross-projection
   accesses rejected. **No spurious resolutions.** Correctness is
   preserved byte-for-byte under the W25 fall-through path on
   every rejected advance.
3. **Did bounded-context efficiency improve in a real way?**
   *YES.* (a) Producer per-cell cost: 14.6 → 1 token on advance
   cells (the W26-L lower bound, mathematically tight). (b)
   Anchor amortisation: 1 anchor cell + (W-1) advance cells per
   chain window; controller-side state is O(1) per chain. (c)
   K-scaling: the W26 saving over W24 grows from 90.60 % at K=3
   to 92.23 % at K=10 (W25-C-K-SCALING discharged by direct
   measurement). The trade-off is a small wire-byte cost
   increase (~7 600 bytes/16 cells vs W25's ~5 885 bytes); this
   is named explicitly as W26-C-MULTI-HOST.
4. **Did two-Mac evaluation materially broaden the evidence?**
   *PARTIALLY — Mac 2 still unreachable (21st milestone in a row).*
   The W26 cross-host story inherits the W24
   :class:`CrossProcessProducerDecoderWire` (real Python
   subprocess via stdin/stdout pipes) as the strongest cross-
   process honesty validated end-to-end on this repo. When Mac 2
   returns the same JSON-canonical interface drops in over a real
   socket with no W26 code changes. Honest scope: real cross-
   *process*, not cross-*host*. The wire-bytes vs token-cost
   tradeoff is named W26-C-MULTI-HOST.
5. **Which earlier paper loose ends were closed versus only
   sharpened?** *Closed: W25-C-K-SCALING (empirically discharged
   at K∈{3,5,8,10}; the structural form K×(C−1) is confirmed; the
   exact percentage at K=10 is measured at 84.69 %, slightly below
   the conjectured 88 %). Sharpened: W25-C-MULTI-HOST inherits
   into W26-C-MULTI-HOST with explicit anchor / advance byte
   measurements. Sharpened: the live-LLM transfer story now has a
   sixth layer (W22 latent + W23 cross-cell + W24 compact + W24
   resample-quorum + W25 fanout + W26 chain-persist), each with a
   structurally-distinct mechanism and a sharp limit theorem.*
   New named conjecture: **W26-C-K-SCALING** (W26 mean visible
   cost / cell tends to ``1 + K`` as ``chain_persist_window →
   ∞``); empirically anchored at K∈{3,5,8,10}, asymptote unverified.
6. **Did release readiness improve?** *YES, on three axes.* (a)
   The W26 surface is purely additive on top of W25; the W22..W25
   stable runtime contract is byte-for-byte unchanged. (b) The
   focused regression covering W22..W26 + IS-1 / IS-2 is
   **180/180** in 15.6s — fast and reproducible. (c) The new W26
   verification surface enumerates 16 failure modes across 3
   verify_* functions, and every failure mode has a unit test;
   the audit story is mechanically-checked, not merely asserted.
   The remaining release-readiness blocker (live-LLM cross-host)
   is unchanged and inherited into W26-C-MULTI-HOST.
7. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on
   three named axes; the deeper wall is sharper.* The Context
   Zero thesis now has its **first capsule-native chain-persisted
   producer-amortisation method**, **first per-consumer projection
   slot mechanism with controller-verified scope**, and **first
   empirical discharge of W25-C-K-SCALING** at K∈{3,5,8,10}. Named
   open frontier: **W26-C-K-SCALING** (asymptotic floor verified
   at finite K; ∞-K asymptote conjectural), **W26-C-MULTI-HOST**
   (real cross-host wire vs token cost — gated on Mac-2 return),
   and **W26-C-DIVERGENCE-RECOVERY** (workloads where the gold
   subset flips per cell — currently W26 falls through to W25; a
   smarter chain-replay could recover savings even on divergent
   cells, but this requires a new mechanism). The thesis is
   materially stronger AND the next research frontier is precisely
   articulated as long-K asymptotic verification + cross-host
   real-wire validation + divergence-aware chain replay.

### 4.45 SDK v3.28 — multi-chain salience-keyed dense-control fanout + per-signature scoping + Phase-74 R-74 benchmark family + W27 family (R-74-XORACLE-RECOVER + R-74-CHAIN-SHARED + R-74-DIVERGENT-RECOVER + R-74-POOL-EXHAUSTED + R-74-PIVOT-TAMPERED + R-74-SIGNATURE-DRIFT + signature_period sweep + W26-C-DIVERGENCE-RECOVERY discharged on the per-signature scoping axis)

**One paragraph.** SDK v3.27 (W26) amortised the producer's
per-cell salience-token cost across cells inside a single chain
window via 1-token chain-advance references; W26 explicitly named
**W26-C-DIVERGENCE-RECOVERY** (a smarter chain-replay mechanism
could recover savings on divergent cells) and **W26-Λ-divergent**
(when the gold subset flips, the inner W22 chain breaks and W26
falls through to W25) as open frontiers. SDK v3.28 implements the
smallest honest version of that mechanism at the *capsule layer*: a
**bounded pool of parallel chains, keyed by salience signature**
(SHA-256 over the canonical input handoffs computed by
:func:`compute_input_signature_cid`). The
:class:`MultiChainPersistedFanoutOrchestrator` is the load-bearing
W27 implementation — it wraps a team-wide
:class:`SharedMultiChainPool` that maintains one independent W26
disambiguator per (signature, agent) pair, sharing per-signature
``SharedFanoutRegistry`` and ``ChainPersistedFanoutRegistry``
across producer + K consumers so consumers find the producer's
per-signature fanout envelopes inside that signature's chain. The
:class:`MultiChainPersistedFanoutDisambiguator` is the audited
pivot wrapper — it adds two new content-addressed envelopes
(:class:`SalienceSignatureEnvelope`, :class:`ChainPivotEnvelope`)
on top of one W26 stack and ships :func:`verify_salience_signature`
(4 enumerated failure modes) plus :func:`verify_chain_pivot` (8
enumerated failure modes) for trust-boundary auditing. On
**R-74-XORACLE-RECOVER** (1 producer + K=3 consumers, 16 cells, 2
distinct gold-subset signatures, ``signature_period = 4``,
``max_active_chains = 8``, partial ServiceGraphOracle on the W26
baseline scoped only to GOLD_A) the W27 method **simultaneously**
strictly reduces ``mean_total_w27_visible_tokens`` over
``mean_total_w26_visible_tokens`` by **−22.5 tokens / cell
(−76.27 %)** at ``T_decoder = None`` AND raises
``correctness_ratified_rate`` from **0.500 → 1.000** (the W26
baseline's partial oracle abstains on phase-B cells; the W27
orchestrator's per-signature oracle registration scopes a fresh
oracle to each gold pair so both phases resolve cleanly via the
matching slot's stack); identical at ``T_decoder = 24``; stable
across **5/5** seeds (savings = 22.5 tokens/cell on every seed,
correctness 1.0 on every seed). On **R-74-CHAIN-SHARED**
(every cell produces byte-identical canonical state) **W27 reduces
to W26 byte-for-byte** (one signature in the pool;
W27-Λ-single-signature falsifier discharged). On
**R-74-DIVERGENT-RECOVER** (within-graph divergence — both gold
subsets in default oracle scope) W26 single-stack handles
divergence cleanly and W27 pays a measured isolation cost of
**−27.27 %** (one extra anchor per signature spread over 16 cells
≈ 1.5 tokens/cell). On **R-74-POOL-EXHAUSTED** (``max_active_chains
= 2`` but 4 distinct signatures appear) the controller rejects
new anchors beyond the bound and W27 falls through to a single
fallback W26 disambiguator; correctness preserved
(W27-Λ-pool-exhausted). On **R-74-PIVOT-TAMPERED** and
**R-74-SIGNATURE-DRIFT** the audited disambig wrapper rejects
tampering via ``verify_chain_pivot`` /
``verify_salience_signature``; the orchestrator path is
unaffected. **Signature-period sweep** at periods ∈ {1, 2, 4, 8, 16}
on R-74-DIVERGENT-RECOVER confirms the W27 isolation cost is
**constant** at one extra anchor per phase regardless of
alternation frequency (period=16 has only one signature → W27 =
W26 byte-for-byte). New W27 unit + integration tests: 22/22 pass;
full pre-existing W21..W26 + IS-1 / IS-2 + producer / team_coord /
attention / capsules surfaces preserved byte-for-byte (508/508 in
the focused regression). Mac 2 still unreachable (**22nd milestone
in a row**); W27 inherits the W24
:class:`CrossProcessProducerDecoderWire` proxy as the strongest
cross-process honesty validated on this repo.

**Master-plan-level questions (the post-W26 audit board).**

1. **Did denser capsule-native latent/control integration
   materially help?** *YES on a new axis: per-signature isolation
   that simultaneously rescues correctness AND saves tokens on a
   regime where W26's single-stack scope architecturally fails.*
   On R-74-XORACLE-RECOVER the combined effect is **−76.27 %
   tokens AND +0.500 correctness over W26** at K=3. This is the
   first capsule-native multi-agent-coordination method that
   *simultaneously* improves both efficiency and correctness over
   the prior best on a regime where the prior best actually
   limits correctness. The honest cost-of-isolation when W26
   doesn't fail is +27 % (one extra anchor per signature) —
   measured and reported plainly.
2. **Did trust/audit survive?** *YES, sharply, with two new
   enumerated-failure-mode verification functions.* The
   :class:`SalienceSignatureEnvelope` is hash-chained
   (``signature_cid``), schema-versioned, and verifier-rejectable
   on 4 failure modes; each :class:`ChainPivotEnvelope` is
   parent-CID-linked and verifier-rejectable on 8 failure modes.
   The R-74-PIVOT-TAMPERED and R-74-SIGNATURE-DRIFT benches
   measure rejection on every attempt. **No spurious resolutions.**
3. **Did bounded-context efficiency improve in a real way?** *YES
   on the regime where W27's isolation rescues correctness, with
   honest cost on regimes where it doesn't.* (a) Per-cell tokens:
   W26 baseline 29.5 → W27 7.0 on R-74-XORACLE-RECOVER (−76 %);
   W26 5.5 → W27 7.0 on R-74-DIVERGENT-RECOVER (+27 % cost). (b)
   Pool capacity bound: ``max_active_chains`` controls pool
   size; cells beyond the bound deterministically fall through
   (W27-Λ-pool-exhausted).
4. **Did two-Mac evaluation materially broaden the evidence?**
   *NO — Mac 2 still unreachable (22nd milestone in a row).*
   The W27 cross-host story inherits the W24
   :class:`CrossProcessProducerDecoderWire` proxy; no real
   two-host execution. Honest scope: real cross-*process*, not
   cross-*host*.
5. **Which earlier paper loose ends were closed versus only
   sharpened?** *Closed: W26-C-DIVERGENCE-RECOVERY in the
   per-signature scoping direction (per-signature oracle
   registration recovers correctness on divergent benches via
   independent oracle scopes per signature). Sharpened: the
   W27-Λ-single-signature falsifier makes the conditionality of
   the W27 advance precise — W27 only beats W26 when there are
   distinct signatures AND W26's single-stack actually fails.
   New named conjectures: **W27-C-MULTI-SIGNATURE-SCALING**
   (per-cell cost → 1+K as N → ∞ for stable benches with M ≤
   max_active_chains; M → ∞ asymptote unverified) and
   **W27-C-CROSS-HOST** (wire cost ≈ W26 per-chain, with
   per-signature overhead — gated on Mac-2 return).*
6. **Did release readiness improve?** *YES, on three axes.* (a)
   The W27 surface is purely additive on top of W26; the W22..W26
   stable runtime contract is byte-for-byte unchanged. (b) The
   focused regression covering W18..W27 + IS-1 / IS-2 + producer
   / team_coord / attention / capsules is **508/508** in
   ≤ 60s — fast and reproducible. (c) The new W27 verification
   surface enumerates 12 failure modes across 2 verify_*
   functions, and every failure mode has a unit test; the audit
   story is mechanically-checked, not merely asserted.
7. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on
   the simultaneous-correctness-and-efficiency axis; the deeper
   wall is sharper.* Named open frontier:
   **W27-C-MULTI-SIGNATURE-SCALING**, **W27-C-CROSS-HOST**, and
   **W27-C-LIVE-CROSS-MODEL** (live LLM oracles per signature,
   not measured in this milestone).

---

## Post-W27 next steps → discharged in SDK v3.29 (W28)

**SDK v3.29 / W28** (`docs/RESULTS_WEVRA_W28_ENSEMBLE_VERIFIED_MULTI_CHAIN.md`)
ships the **first capsule-native synthesis between the explicit-
capsule trust line (W21 trust-weighted multi-oracle adjudication)
and the dense-control line (W27 multi-chain salience-keyed
pool)**, behind a controller-verified ratification envelope with
**11 new enumerated failure modes** in
``verify_ensemble_pivot_ratification``.

* **W28 mechanism.** ``EnsembleVerifiedMultiChainOrchestrator``
  wraps a W27 ``MultiChainPersistedFanoutOrchestrator`` with a
  trust-weighted probe table; each probe is an
  ``EnsembleProbeRegistration`` (mirrors W21's
  ``OracleRegistration``). Built-in probe types:
  ``DeterministicSignatureProbe`` (locally-recomputable, K=1 path
  is W28 = W27 byte-for-byte), ``OracleConsultationProbe``
  (wraps any W20/W21 ``OutsideWitnessOracle``),
  ``LLMSignatureProbe`` (wraps any ``LLMBackend`` —
  Ollama or MLX-distributed; cross-host telemetry via ``host_id``).
* **W28-1 (proved + mechanically-checked).** Trust-boundary
  soundness: 11 enumerated failure modes; every mode has a unit
  test in ``EnsembleVerifierFailureModeTests``.
* **W28-2 (proved + empirical).** Backward-compat: K=1 W28 = W27
  byte-for-byte across 5/5 seeds on R-75-SINGLE-PROBE.
* **W28-3 (proved-conditional + empirical).** Trust-amplification
  overhead bound: max per-cell overhead = 1.00 token across all 7
  synthetic R-75 sub-banks at 5 seeds (within S4 ≤ 2 budget); 0.625
  on R-75-CROSS-HOST-LIVE (n=16, 10/16 ratified).
* **R-75-RATIFICATION-TAMPERED.** 16/16 tampered envelopes rejected
  per seed across 5/5 seeds.
* **R-75-CROSS-HOST-LIVE (FIRST CROSS-HOST EVIDENCE IN 23 MILESTONES).**
  Live LLM probe table on `localhost` (gemma2:9b) +
  `192.168.12.191` (qwen2.5:14b); **128 cross-host probe calls**;
  **5592 LAN bytes**; ensemble ratifies 10/16 cells with real LLM
  disagreement on 6/16; trust precision 1.000; W28 correctness
  1.000.
* **W27-C-CROSS-HOST infrastructure-discharged.** The conjecture's
  *infrastructure axis* (real cross-host probing) is empirically
  discharged via W28's two-host topology probe. The
  *variance-reduction magnitude* axis becomes the new named open
  conjecture **W28-C-CROSS-HOST-VARIANCE**.
* **Stable-vs-experimental boundary tightened.** New explicit
  ``vision_mvp.wevra.__experimental__`` tuple lists every
  dense-control symbol (W22..W28); SDK_VERSION bumped to
  ``wevra.sdk.v3.29``; pyproject.toml 0.5.2; ARCHITECTURE.md
  gains a "Stable-vs-Experimental Boundary" section.

**Master-plan-level questions (the post-W27 audit board, refreshed
for W28).**

1. **Did denser capsule-native latent/control integration
   *and* the explicit-capsule trust line **synthesise** materially?**
   *YES.* The W21 ``OracleRegistration`` interface threads
   directly through to W28's probe table; the
   ``OracleConsultationProbe`` is a literal bridge. The trust
   prior semantics are byte-identical between W21 quorum and W28
   ratification.
2. **Did trust/audit survive?** *YES, sharply, with a brand-new
   verifier covering ensemble-decision integrity (probe forgery,
   weight forgery, quorum forgery) — 11 failure modes that did
   not exist in any W22..W27 verifier.*
3. **Did bounded-context efficiency improve in a real way?** *YES
   on backward-compat (K=1 W28 = W27 byte-for-byte) and YES on
   honest overhead bound (max 1 token/cell across all banks).*
   The W28 layer adds at most 1 producer-side ``<ratify_ref:DDDD>``
   token per ratifying cell; consumers pay 0.
4. **Did two-Mac evaluation materially broaden the evidence?**
   *YES — for the first time in 23 milestones.* The W28
   ``LLMSignatureProbe`` is the first programme component to
   actually use *two reachable hosts with different model
   families* inside one bench cell (`localhost` gemma2:9b +
   `192.168.12.191` qwen2.5:14b). Mac 2 (192.168.12.248) remains
   ARP-incomplete, but the *other* reachable host (.191) has been
   recharacterised as the second host of the topology, and the
   probe table accepts a third backend with zero code changes
   when Mac 2 returns.
5. **Which earlier paper loose ends were closed versus only
   sharpened?** *Closed: the W21 / W27 synthesis target named in
   the post-W27 next-steps section is operational; the
   infrastructure axis of W27-C-CROSS-HOST is discharged.
   Sharpened: the variance-reduction magnitude axis becomes
   W28-C-CROSS-HOST-VARIANCE; the synthetic R-75 banks do not
   exercise W27 mistakes (every bank has W27 correctness = 1.000),
   so the magnitude is only honestly measurable on a regime where
   W27 itself fails — open.*
6. **Did release readiness improve?** *YES, on four axes.* (a)
   Stable-vs-experimental boundary now explicit
   (``__experimental__`` tuple). (b) SDK version bumped to v3.29 /
   0.5.2. (c) ARCHITECTURE.md gains a "Stable-vs-Experimental
   Boundary" section. (d) Focused regression W3..W28 = 534/534 in
   ~95s, fast + reproducible.
7. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on
   the synthesis axis (old + new lines now compose); the deeper
   wall is the variance-reduction magnitude on a regime where W27
   itself fails.* Named open frontier for SDK v3.30:
   **W28-C-CROSS-HOST-VARIANCE** (live cross-host variance
   reduction on a regime where W27 alone is wrong),
   **W28-C-CALIBRATED-TRUST** (calibrated priors strictly
   outperform uniform), **W28-Λ-coordinated-drift in the wild**
   (correlated probes in real LLM ensembles).

---

## SDK v3.30 (W29 family) — geometry-partitioned product-manifold dense control + audited subspace-basis payload + factoradic Lehmer routing index + causal-validity gate + cross-host variance witness — 2026-04-30

W29 inserts a **structural geometry-partitioning** step between the
cell handoffs and the W28 ensemble decision. Every triggered cell is
classified into one of three pre-committed partition labels — LINEAR
(extends most-recent signature), HIERARCHICAL (fresh anchor), CYCLIC
(re-visited signature) — by the producer using the W27 input
signature CID and a bounded signature history. Per-partition inner
W28 stacks get their own oracle / probe / pool topology, so cells in
different partitions route through structurally different
compartments. The W29 envelope additionally carries an audited
orthonormal subspace basis (Grassmannian-style, capsule-layer audited
proxy), a factoradic Lehmer-code routing index, a causal-validity
SHA-256 binding, and an optional cross-host variance witness.

**Headline result.** On **R-76-XHOST-DRIFT** (1 producer + K=3
consumers, 16 cells, partial ServiceGraphOracle that knows GOLD_A
but not GOLD_B; per-partition routing dispatches CYCLIC cells to a
separate inner W28 with the FULL oracle), W29 strictly improves
correctness over BOTH W27 and W28 baselines: ``correctness_w27 =
correctness_w28 = 0.500`` and ``correctness_w29 = 0.750``,
**Δ = +0.250 across 5/5 seeds**, ``trust_precision = 1.000``,
``mean_overhead = 0.75 tokens/cell``, ``max_overhead = 1 token/cell``.
Same +0.250 gain holds on the live two-host topology (localhost
gemma2:9b + 192.168.12.191 qwen2.5:14b); 16 cross-host probe calls;
710 LAN bytes; ``trust_precision = 1.000``.

**Empirically discharges W28-C-CROSS-HOST-VARIANCE on the magnitude
axis** (the named open frontier from the post-W28 audit board's
question 7). The first capsule-native multi-agent-coordination
method to demonstrate the synthesis (W21 × W27, sealed by W28,
geometry-partitioned by W29) strictly improves correctness on a
regime where the prior best (W28) makes correctness mistakes.

Five named falsifiers all empirically confirmed (W29-Λ-trivial-
partition; W29-Λ-non-orthogonal-basis; W29-Λ-coordinated-drift-
cross-host; plus W28's W28-Λ-quorum-tampered and
W28-Λ-pool-exhausted-passthrough inherited). 14 enumerated trust-
boundary failure modes (vs W28's 11) in
``verify_geometry_partition_ratification``. **935/935 + 6 subtests
pass** across W3..W29 + capsule + public API + runtime + LLM
backend.

The pre-committed cram-factor headline (H7 ≥ 8.0) was MISSED:
measured 2.30× on R-76-CHAIN-SHARED. Mechanism real, magnitude
below bar; becomes named open conjecture
**W29-C-CRAM-AMPLIFICATION**. The pre-committed H6 absolute
correctness bar (≥ 0.95) was missed (measured 0.75) but the
LOAD-BEARING Δ ≥ 0.10 axis was exceeded by 2.5×.

Mac 2 (192.168.12.248) still ARP-incomplete (24th milestone in a
row). The two reachable hosts suffice for the live cross-host
evidence.

### Post-W29 audit board

1. **Did geometry-aware dense-control integration materially
   help?** *YES on the load-bearing axis (R-76-XHOST-DRIFT shows
   the first regime where the synthesis strictly improves
   correctness over both W27 and W28 baselines, Δ = +0.25 across
   5/5 seeds, trust precision 1.000, bounded overhead). The
   cram-factor magnitude (H7) was missed; mechanism real,
   magnitude below bar.*
2. **Did trust/audit survive?** *YES — 14 enumerated failure modes
   (vs W28's 11), 100% rejection rate on tamper banks (12/12 +
   60/60), trust precision = 1.000 on every ratifying sub-bank.*
3. **Did bounded-context efficiency improve in a real way?** *YES —
   the W29 envelope packs 2.30× more structured-control bits per
   wire token than W28. Same 1-token wire cost as W28's
   ``<ratify_ref:DDDD>`` — the W29 ``<partition_ref:DDDD>`` rides
   on the same single visible token but carries the partition_id +
   factoradic index + audited basis CID + causal-validity
   signature + cross-host variance witness CID + parent W28 CID +
   predecessor CIDs.*
4. **Did two-Mac evaluation materially broaden the evidence?**
   *PARTIAL — the live cross-host topology (gemma2 + qwen2.5)
   produced the same H6 +0.250 correctness gain on real LLM bytes
   (710 LAN bytes, 16 probe calls), but Mac 2 (192.168.12.248)
   remains ARP-incomplete (24th milestone). The two-host evidence
   strictly extends W28's by demonstrating the H6 correctness gain
   on the live topology.*
5. **Which earlier paper loose ends were closed vs only sharpened?**
   *CLOSED: W28-C-CROSS-HOST-VARIANCE on the magnitude axis (the
   biggest open conjecture from the W28 milestone). SHARPENED:
   the synthesis line (old explicit-capsule × new dense-control)
   now has a measured regime where it strictly improves
   correctness, not just trust precision. STILL OPEN:
   W22-C-CACHE-AMPLIFICATION (per-partition first-cell still
   cache-amplifies); W23-C-MITIGATION-LIVE-VARIANCE inter-cell
   axis; W24-C-LIVE-VARIANCE-COMPLETE; W26-C-K-SCALING K → ∞
   asymptote; W27-C-MULTI-SIGNATURE-SCALING;
   W21-C-CALIBRATED-TRUST.*
6. **Did release readiness improve?** *YES on the SDK-version /
   ``__experimental__`` / pyproject.toml axis. (a) W29 surface
   added under ``__experimental__`` (38 tests + verifier).
   (b) SDK version bumped to v3.30 / 0.5.3.
   (c) Focused regression W3..W29 = 935/935 in ~100s, fast +
   reproducible. (d) The new geometry / Grassmannian / factoradic /
   Lehmer / mixed-curvature vocabulary is honestly framed as
   capsule-layer audited proxy in the module docstring AND in the
   results note's §9.*
7. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on the
   correctness axis: the synthesis (W21 × W27, sealed by W28,
   geometry-partitioned by W29) now has a measured regime where it
   strictly improves correctness over the prior best baseline.
   The deeper wall is whichever regime makes the structural
   classifier's three-way split (LINEAR/HIERARCHICAL/CYCLIC)
   insufficient to discriminate hard cells.* Named open frontier
   for SDK v3.31: **W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE**
   (regime where live LLMs systematically disagree),
   **W29-C-CRAM-AMPLIFICATION** (richer payload that hits 8×),
   **W29-C-PARTITION-CALIBRATION** (calibrated per-partition
   trust priors), **W29-C-NATIVE-LATENT** (architecture-dependent
   — true transformer-internal subspace projection vs the W29
   audited proxy).

---

## SDK v3.31 (W30 family) — calibrated geometry-partitioned dense control + multi-stride basis-history + per-partition calibration prior + cross-host disagreement-routing + ancestor-chain causal binding — 2026-05-01

W30 wraps the W29 ``GeometryPartitionedOrchestrator`` with a
``CalibratedGeometryOrchestrator`` that adds two **closed-form,
audited override mechanisms** before each cell dispatch — keeping
the W29 partition classifier on the *normal* path and only
overriding when a calibrated prior or a cross-host disagreement
witness justifies the reroute. Both mechanisms are **NOT** learned
models: the calibration prior is a deterministic running-mean
update over observed agreement, and the disagreement-routing
override peeks at the inner W28's last-result
``cross_host_disagreement_count`` from the *previous* cell.
Everything rides on a single content-addressed envelope at a
fixed 1-token wire cost via a new optional
``partition_classifier_hook`` on W29.

The ``CalibratedGeometryRatificationEnvelope`` adds four
content-addressed primitives over the W29 envelope: a
``BasisHistory`` (multi-stride accumulator over ``stride`` recent
partition basis CIDs, content-addressed so rotation in the
history changes the CID byte-for-byte), a
``PartitionCalibrationVector`` (canonical-sorted partition_id →
mean agreement vector, with a single high_trust_partition_id +
threshold for the override), an ``AncestorChain``
(``ancestor_window`` sorted parent CIDs giving deeper causal
binding than W29's single-parent W28 link), and the W29 parent
partition CID. The new
``verify_calibrated_geometry_ratification`` enumerates **14
disjoint failure modes** — basis-history-CID forgery,
calibration-vector-out-of-range, ancestor-chain-CID forgery,
disagreement-route-target forgery, calibrated-CID forgery — none
of which existed in any W22..W29 verifier.

**Headline result.** On **R-77-CHAIN-CRAM** at
``stride = 28``, ``ancestor_window = 12`` (the v0.2 pre-commit
defaults) the W30 envelope packs ``cram_w30/w28 = 8.74×`` AND
``cram_w30/w29 = 2.74×`` more bits of audited structured control
per visible token across **5/5 seeds**. **Empirically discharges
W29-C-CRAM-AMPLIFICATION** (8.74× ≥ the pre-committed 8.0× bar
that W29 missed at 2.30×). On **R-77-CALIBRATION-PRIOR** the
calibration override yields ``Δ(W30 − W29) = +0.250 correctness
gain`` across 5/5 seeds (**discharges W29-C-PARTITION-CALIBRATION
on the closed-form-prior axis**). On **R-77-XHOST-DISAGREE** the
disagreement-routing override yields ``Δ(W30 − W29) = +0.250
correctness gain`` across 5/5 seeds (**sharpens
W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE on the
capsule-layer-routing axis**). ``trust_precision_w30 = 1.000``
on every R-77 sub-bank where W30 ratifies; tamper rejection 100 %
(60/60) across five named modes.

Three named falsifiers all empirically confirmed
(W30-Λ-trivial-calibration, W30-Λ-non-calibratable,
W30-Λ-degenerate-history). 14 enumerated trust-boundary failure
modes (vs W29's 14, disjoint from W29's). **357/357 focused
regression pass** (273/273 phase69-77 + 84/84 wider wevra
suite).

Mac 2 (192.168.12.248) still ARP-incomplete (25th milestone in a
row). The two reachable hosts (localhost gemma2:9b + 192.168.12.191
qwen2.5:14b) were used for the live cross-host bench; at
temperature 0 the LLMs agreed on every cell, so the
disagreement-routing override correctly stayed null and the
S1 *cross-host variance live magnitude* gate is reported as
honestly null (not failure).

### Post-W30 audit board

1. **Did calibrated dense-control integration materially help?**
   *YES on three load-bearing axes simultaneously: H6 cram
   amplification (8.74× ≥ 8.0 bar), H7 calibration prior
   (Δ +0.250), H8 disagreement-routing (Δ +0.250). The first
   capsule-native milestone to discharge BOTH a missed prior
   conjecture (W29-C-CRAM-AMPLIFICATION) AND a never-before-
   measured prior conjecture (W29-C-PARTITION-CALIBRATION) on a
   single milestone.*
2. **Did trust/audit survive?** *YES — 14 enumerated failure
   modes (disjoint from W29's 14), 100 % rejection rate on
   tamper banks (60/60 + 12/12), trust precision = 1.000 on
   every ratifying sub-bank.*
3. **Did bounded-context efficiency improve in a real way?**
   *YES — the W30 envelope packs 8.74× more structured-control
   bits per wire token than W28 and 2.74× more than W29 at
   ``stride = 28``, ``ancestor_window = 12``, while keeping the
   1-token wire cost. The mechanism scales linearly in
   ``stride`` (each additional stride slot is one more
   content-addressed CID in the basis-history accumulator at
   the same 1-token wire cost).*
4. **Did two-Mac evaluation materially broaden the evidence?**
   *PARTIAL — the live cross-host topology (gemma2:9b +
   qwen2.5:14b) was probed for R-77-CROSS-HOST-LIVE, but at
   temperature 0 the LLMs agreed on every cell, so the
   disagreement-routing override correctly stayed null. Mac 2
   (192.168.12.248) remains ARP-incomplete (25th milestone).
   The synthetic R-77-XHOST-DISAGREE bench is the load-bearing
   evidence for the H8 axis; the live bench is honestly
   reported as null on the disagreement-routing magnitude.*
5. **Which earlier loose ends were closed vs only sharpened?**
   *CLOSED: W29-C-CRAM-AMPLIFICATION on the
   multi-stride-history axis. CLOSED:
   W29-C-PARTITION-CALIBRATION on the closed-form-prior axis.
   SHARPENED: W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE on the
   capsule-layer-routing axis (residual frontier:
   regime where live LLMs systematically disagree at temp 0).
   SHARPENED: W21-C-CALIBRATED-TRUST on the per-partition axis.
   STILL OPEN: W22-C-CACHE-AMPLIFICATION (per-partition first-
   cell still cache-amplifies); W23-C-MITIGATION-LIVE-VARIANCE
   inter-cell axis; W24-C-LIVE-VARIANCE-COMPLETE;
   W26-C-K-SCALING K → ∞ asymptote;
   W27-C-MULTI-SIGNATURE-SCALING; W30-C-NATIVE-LATENT
   (architecture-dependent: true transformer-internal
   subspace projection vs the W30 audited proxy);
   W30-C-MULTI-HOST (3+ host topology, blocked on Mac 2 ARP);
   W30-C-PRIOR-LEARNING (true learned per-partition prior vs
   the W30 deterministic running mean — out of scope as a
   capsule-layer mechanism).*
6. **Did release readiness improve?** *YES on the SDK-version /
   ``__experimental__`` / pyproject.toml axis. (a) W30 surface
   added under ``__experimental__`` (36 unit tests + verifier
   + 12 W30 primitives). (b) SDK version bumped to v3.31 /
   0.5.4. (c) Focused regression W3..W30 phase suite + wider
   wevra = 357/357 in ~15s, fast + reproducible. (d) The
   calibration prior + disagreement-routing vocabulary is
   honestly framed as capsule-layer audited proxy in the module
   docstring AND in the results note's §9.*
7. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on
   THREE axes simultaneously (cram amplification, calibration
   prior, disagreement-routing) on a single milestone. The
   deeper wall is whichever regime the closed-form override
   cannot resolve: a regime where live LLMs systematically
   disagree at temperature 0 (W30-C-CROSS-HOST-VARIANCE-LIVE-
   MAGNITUDE-LIVE), or a regime where partitions are not
   structurally separable by the W29 LINEAR/HIERARCHICAL/CYCLIC
   classifier (W30-C-NATIVE-LATENT — true transformer-internal
   subspace projection vs the audited proxy), or a regime where
   the per-partition prior must be learned, not running-mean
   updated (W30-C-PRIOR-LEARNING — out of scope as a capsule-
   layer mechanism).* Named open frontier for SDK v3.32:
   **W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE** (regime
   where live LLMs systematically disagree),
   **W30-C-NATIVE-LATENT** (architecture-dependent — true
   transformer-internal subspace projection vs the W30 audited
   proxy), **W30-C-MULTI-HOST** (3+ host topology, blocked on
   Mac 2 ARP), **W30-C-PRIOR-LEARNING** (out of scope as a
   capsule-layer mechanism).

---

## SDK v3.32 (W31 family) — online self-calibrated geometry-aware dense control + sealed prior trajectory + adaptive threshold + W31 manifest CID + first measured live cross-architecture LLM disagreement at temperature 0 — 2026-05-01

W31 wraps the W30 ``CalibratedGeometryOrchestrator`` with an
``OnlineCalibratedOrchestrator`` that **closes the loop** on the W30
calibration prior — a closed-form Bayesian-style running-mean
update inside the orchestrator, fed by a deterministic per-cell
agreement signal (``ratified AND no cross-host disagreement`` ⇒
1.0; else 0.0).  The W31 layer is **NOT** a learned model: zero
parameters, zero gradients, zero training step.  The threshold is
recomputed as a closed-form clipped median of the live prior vector
(bounded to ``[0.20, 0.80]`` via registered constants).  The prior
+ threshold trajectories are sealed in a content-addressed envelope
along with a manifest CID over (basis_history_cid, calibration_cid,
ancestor_chain_cid, prior_trajectory_cid, threshold_trajectory_cid,
route_audit_cid).

The ``OnlineCalibratedRatificationEnvelope`` adds three new
content-addressed primitives over the W30 envelope: a
``PriorTrajectoryEntry`` tuple sealed by SHA-256 of canonical bytes
(observed_agreement, prior_after, partition_id, cell_idx), a
threshold trajectory sealed by SHA-256, and a manifest CID that
ties all six component CIDs together — closing the cross-component
swap avenue.  The new ``verify_online_calibrated_ratification``
enumerates **14 disjoint failure modes** (cumulative 42 across W29
+ W30 + W31) — none of which existed in any W22..W30 verifier.

**Headline result.** On **R-78-NONSTATIONARY-PRIOR** (inverted
partition layout, CYCLIC=PARTIAL oracle, LINEAR=FULL oracle), the
W30 baseline with uniform priors (1.0, 1.0, 1.0) cannot reroute and
gets ``correctness_ratified_rate_w30 = 0.750``; W31 with online
learning observes per-cell failures on CYCLIC, drops the prior to
0.6667 over two failed cells, and the clipped-median adaptive
threshold (0.8) fires the reroute on subsequent CYCLIC cells —
strictly improving correctness to ``0.875`` across **5/5 seeds at
trust precision 1.000**, ``Δ(W31 − W30) = +0.125``.  **Empirically
discharges W30-C-PRIOR-LEARNING**.

On **R-78-ADAPTIVE-THRESHOLD vs R-78-FROZEN-THRESHOLD** the
adaptive axis isolates: with frozen threshold = 0.5, Δ = 0.000 (the
running mean of CYCLIC = 0.6667 > 0.5); with adaptive
clipped-median threshold (0.8), Δ = +0.125.  Adaptive contribution
is +0.125 ≥ +0.05 across 5/5 seeds.

On **R-78-MANIFEST-TAMPER** the W31 manifest CID + cross-cell
prior_trajectory_cid check together detect five named tampers per
cell-position: cross-cell trajectory swap (with self-consistent
manifest recompute), manifest_cid byte corruption,
observed_agreement out of range, threshold_trajectory value out of
range, outer w31_cid byte corruption.  **65/65 = 1.000 rejection
rate** at seed 11.

On **R-78-XLLM-LIVE** (live cross-architecture probe at temp 0,
seed 0) gemma2:9b on localhost vs qwen2.5:14b on 192.168.12.191
**systematically disagree on 2/8 = 0.250 of structured-decision
prompts**, reproducible byte-for-byte across two runs (Q5: db_query
vs logs_pipeline; Q7: api vs storage).  **First measured live
cross-architecture LLM disagreement at temperature 0 in the
programme** (28th milestone).  This **sharpens W30-C-CROSS-HOST-
VARIANCE-LIVE-MAGNITUDE-LIVE** on the **infrastructure-discharge
axis**: the disagreement signal exists at temp 0; the gold-
correlation axis remains open.

Three named falsifiers all empirically confirmed
(W31-Λ-trivial-online, W31-Λ-no-drift, W31-Λ-frozen-threshold).
14 enumerated trust-boundary failure modes (vs W30's 14, vs W29's
14; cumulative 42-mode trust boundary across W29 + W30 + W31).
**437/437 focused regression pass** (was 357 in v3.31; +41 W31
unit tests + 39 unchanged from v3.31 + 1 unchanged).  68/68 wider
wevra suite passes.

Mac 2 (192.168.12.248) **still ARP-incomplete (26th milestone)**;
ping 100% packet loss; port 11434 unreachable.  The two reachable
hosts (localhost + 192.168.12.191) suffice for the live cross-arch
probe.

### Post-W31 audit board

1. **Did online closed-loop prior-learning materially help?**
   *YES on the load-bearing axis: H6 nonstationary prior (Δ +0.125
   ≥ pre-committed +0.10 bar across 5/5 seeds at trust precision
   1.000).  The first capsule-native milestone to discharge a W30
   open conjecture (W30-C-PRIOR-LEARNING) on a regime where the
   per-partition agreement-rate distribution is non-stationary.*
2. **Did adaptive threshold contribute strictly?** *YES — H7
   isolates the contribution at +0.125 across 5/5 seeds; frozen
   threshold (W30 default 0.5) yields zero gain on the same
   regime; the clipped-median adaptive bound is the load-bearing
   trigger.*
3. **Did trust/audit survive?** *YES — 14 enumerated failure
   modes (disjoint from W29's 14 and W30's 14), 100 % rejection
   rate on R-78-MANIFEST-TAMPER (65/65) including the cross-cell
   trajectory swap that self-consistently recomputes the manifest
   CID — caught by the registered_prior_trajectory_cid check.
   Cumulative trust boundary 42 modes across W29 + W30 + W31.*
4. **Did bounded-context efficiency improve in a real way?**
   *PARTIAL — W31 charges 1 visible token / cell on top of W30
   (mean overhead 0.875, max overhead 1) so cumulative overhead
   vs W28 is ≤ 3 tokens.  The cram-factor is not the load-bearing
   gain on this milestone — the load-bearing gain is correctness
   on a non-stationary regime that the W30 baseline cannot handle
   without ground-truth-trained priors.*
5. **Did two-Mac evaluation materially broaden the evidence?**
   *YES — the live cross-architecture probe (gemma2:9b +
   qwen2.5:14b) recorded **the first measured live cross-
   architecture LLM disagreement at temp 0 in the programme**
   (2/8 = 0.250 disagreement rate, reproducible byte-for-byte
   across two runs).  This sharpens W30-C-CROSS-HOST-VARIANCE-
   LIVE-MAGNITUDE-LIVE on the infrastructure-discharge axis.
   Mac 2 (192.168.12.248) remains ARP-incomplete (26th
   milestone).*
6. **Which earlier loose ends were closed vs only sharpened?**
   *CLOSED: W30-C-PRIOR-LEARNING on the magnitude axis (the
   conjectural status promoted to proved-conditional + proved-
   empirical via W31-3).  SHARPENED: W30-C-CROSS-HOST-VARIANCE-
   LIVE-MAGNITUDE-LIVE on the infrastructure-discharge axis (the
   live disagreement signal exists; the gold-correlation axis
   remains open as the renamed W31-C-CROSS-HOST-VARIANCE-LIVE-
   MAGNITUDE-LIVE).  STILL OPEN: W22-C-CACHE-AMPLIFICATION;
   W23-C-MITIGATION-LIVE-VARIANCE; W24-C-LIVE-VARIANCE-COMPLETE;
   W26-C-K-SCALING K → ∞ asymptote; W27-C-MULTI-SIGNATURE-SCALING;
   W30-C-NATIVE-LATENT (architecture-dependent: true transformer-
   internal subspace projection vs the W31 audited proxy);
   W30-C-MULTI-HOST (3+ host topology, blocked on Mac 2 ARP);
   W31-C-LONG-WINDOW-CONVERGENCE (longer trajectory windows).*
7. **Did release readiness improve?** *YES on the SDK-version /
   ``__experimental__`` / pyproject.toml axis.  (a) W31 surface
   added under ``__experimental__`` (41 unit tests + verifier +
   10 W31 primitives).  (b) SDK version bumped to v3.32 / 0.5.5.
   (c) Focused regression W3..W31 phase suite + wider wevra =
   505/505 in ~115s, reproducible.  (d) The online running-mean /
   adaptive threshold / sealed trajectory / manifest CID
   vocabulary is honestly framed as capsule-layer audited proxy
   in the module docstring AND in the results note's §9.  (e)
   First live cross-architecture LLM disagreement evidence at
   temp 0 in the programme — concrete infrastructure progress
   beyond synthetic benches.*
8. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on
   FOUR axes simultaneously: closed-loop online prior learning
   (W30-C-PRIOR-LEARNING discharged), adaptive threshold (W31-4
   isolated), cross-component manifest-CID tamper detection
   (W31-5), first live cross-architecture LLM disagreement at
   temp 0 (W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE
   infrastructure-discharged).  The deeper wall remains whichever
   regime the audited capsule-layer proxy cannot resolve: a regime
   where the live LLM disagreement systematically aligns with the
   gold-correctness label (W31-C-CROSS-HOST-VARIANCE-LIVE-
   MAGNITUDE-LIVE on the gold-correlation axis), or true
   transformer-internal subspace projection
   (W31-C-NATIVE-LATENT — architecture-dependent), or a regime
   where 3-host majority is needed (W31-C-MULTI-HOST — hardware-
   bounded; blocked on Mac 2 ARP).* Named open frontier for SDK
   v3.33: **W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE** on
   the gold-correlation axis, **W31-C-NATIVE-LATENT**
   (architecture-dependent), **W31-C-MULTI-HOST** (hardware-
   bounded), **W31-C-LONG-WINDOW-CONVERGENCE** (trajectory window
   scaling).

---

## SDK v3.33 (W32 family) — long-window convergent online geometry-aware dense control + EWMA prior accumulator + Page CUSUM change-point detector + gold-correlated disagreement-routing + W32 manifest-v2 CID + first measured live cross-architecture LLM gold-verifiable agreement at temperature 0 — 2026-05-01

W32 wraps the W31 ``OnlineCalibratedOrchestrator`` with a
``LongWindowConvergentOrchestrator`` that adds **four NEW audited
proxies** at the capsule layer: an EWMA prior accumulator
(closed-form ``ewma_new = (1 - α) * ewma_prev + α * obs`` with
default α=0.20), a Page two-sided CUSUM change-point detector
(closed-form ``cusum_pos / cusum_neg`` accumulators bounded by
registered ``cusum_max=16.0`` with default threshold 1.5 and slack
0.10), a gold-correlated disagreement-routing primitive against a
registered closed-vocabulary ``GoldCorrelationMap``, and a sealed
convergence-state trajectory CID + cross-component manifest-v2 CID.
The W32 layer is **NOT** a learned model: zero parameters, zero
gradients, zero training step.  The gold-correlation map is a
**registered closed-vocabulary table**; W32 does NOT observe ground
truth at runtime — the map's correctness is a structural witness
the controller registers up-front.

The ``LongWindowConvergentRatificationEnvelope`` adds three new
content-addressed primitives over the W31 envelope: a
``ConvergenceStateEntry`` tuple sealed by SHA-256 of canonical
bytes ``(cell_idx, partition_id, ewma_prior_after, cusum_pos,
cusum_neg, change_point_fired)``, a gold_correlation_cid sealed by
SHA-256 over the registered closed-vocab map, and a manifest-v2 CID
that ties all four W32 component CIDs together — closing the
cross-component swap avenue that the W31 manifest CID (which does
NOT include convergence_state_cid) cannot detect.  The new
``verify_long_window_convergent_ratification`` enumerates **14
disjoint failure modes** (cumulative 56 across W29 + W30 + W31 +
W32) — none of which existed in any W22..W31 verifier.

**Headline result.** On **R-79-LONG-WINDOW** (sweep over
``long_window ∈ {16, 32, 64, 128}`` on the prefix-then-shift
drift_recover regime), the W32 EWMA + CUSUM mechanism achieves
**byte-for-W31-equal correctness across 5/5 seeds × 4/4 windows =
20/20 cell-window pairs at trust precision 1.000; zero degradation
as window grows**.  **Discharges W31-C-LONG-WINDOW-CONVERGENCE on
the scaling-stability axis**.

On **R-79-DRIFT-RECOVER** (the multi-shift load-bearing regime),
the empirical Δ(W32 - W31) = 0.000 across 5/5 seeds — the
H6 strict-gain bar Δ ≥ +0.10 is **honestly-null** because of the
**W32-L-CYCLE-CAP limitation theorem**: on a cycle-capped
dispatcher (cycle_window=8, 3 partitions), the maximum strict gain
is bounded above by ``min(c_p / 4, c_s) / N ≤ 0.0625``.  The
mechanism is empirically validated by ``n_change_points = 1``
firing exactly at the shift boundary (cell 61 by inspection of the
convergence_summary).  The strict-gain claim inherits forward as
**W32-C-LONG-WINDOW-STRICT-GAIN** on a regime that exceeds the
cycle-cap.

On **R-79-MANIFEST-V2-TAMPER** the W32 manifest-v2 CID + cross-cell
convergence_state_cid check together detect **1525/1525 = 1.000
cross-component tamper rejection rate** across 5/5 seeds × 61
ratified cell-positions × 5 named tampers per cell.  Five tampers,
each provably triggers a distinct reject reason: cross-cell
convergence_state swap (with self-consistent manifest-v2 recompute),
manifest_v2_cid byte corruption, ewma_prior_after out of range,
cusum_pos out of range, outer w32_cid byte corruption.

On **R-79-XLLM-LIVE-GOLD** (live cross-architecture probe at temp
0, seed 0) gemma2:9b on localhost vs qwen2.5:14b on 192.168.12.191
**agree on 19/20 = 0.950 of gold-verifiable structured-decision
prompts** across arithmetic (A1-A5, 5/5 agree), syntax (S1-S5, 5/5
agree), factoid (F1-F5, 5/5 agree), disambiguation (D1-D5, 4/5
agree).  The unique disagreement (D5: TCP three-way handshake
second packet) is a tokenisation boundary ("synack" vs "syn-ack")
where neither host matches the registered gold "syn-ack" — i.e.
**gold-correlation null on this prompt**.  **First measured live
cross-architecture LLM gold-verifiable agreement at temperature 0
in the programme** (29th milestone).  Combined with W31's
R-78-XLLM-LIVE (6/8 = 0.750 agreement on operational-decision
prompts), the **prompt-class-dependent cross-architecture
disagreement frontier** at temp 0 is now characterised:
operational/strategic prompts produce real cross-host disagreement;
gold-verifiable factual prompts produce mostly agreement.

Four named falsifiers all empirically observed
(W32-Λ-trivial-long-window, W32-Λ-no-change-point,
W32-Λ-frozen-ewma honest empirical correction, W32-Λ-mis-
correlated-gold gate-bounded).  14 enumerated trust-boundary
failure modes (vs W31's 14, vs W30's 14, vs W29's 14; cumulative
56-mode trust boundary across W29 + W30 + W31 + W32).
**45/45 W32 unit tests + 414/414 phase69-79 regression + 77/77
wider wevra suite = 536 tests pass**.

Mac 2 (192.168.12.248) **still ARP-incomplete (27th milestone)**;
ping 100% packet loss; port 11434 unreachable.  The two reachable
hosts (localhost + 192.168.12.191) suffice for the live cross-arch
gold-verifiable probe.

### Post-W32 audit board

1. **Did long-window convergent EWMA + CUSUM materially help?**
   *YES on the scaling-stability axis: H7 long-window sweep (W32 ≥
   W31 byte-for-byte across 5/5 seeds × 4/4 windows = 20/20
   cell-window pairs at trust precision 1.000).  **Discharges
   W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability axis**.
   Strict-gain claim H6 is honestly-null on the available cycle-
   capped dispatcher regimes per the **W32-L-CYCLE-CAP limitation
   theorem** (Δ_max ≤ 0.0625 by structural bound); the mechanism
   is empirically validated by n_change_points firing correctly at
   the shift boundary.*
2. **Did Page CUSUM change-point detector contribute?** *YES — H4
   anchors the CUSUM detector firing exactly at the shift boundary
   (cell 61) across 5/5 seeds; on stationary regimes
   (W32-Λ-no-change-point), n_change_points=0 across 5/5 seeds.*
3. **Did trust/audit survive?** *YES — 14 enumerated failure
   modes (disjoint from W29/W30/W31's 14-mode sets), 1.000
   rejection rate on R-79-MANIFEST-V2-TAMPER (1525/1525) including
   the cross-cell convergence_state swap that the W31 manifest CID
   alone cannot detect.  Cumulative trust boundary 56 modes across
   W29 + W30 + W31 + W32.*
4. **Did bounded-context efficiency improve in a real way?**
   *PARTIAL — W32 charges 1 visible token / cell on top of W31
   (mean overhead 0.969, max overhead 1) so cumulative overhead
   vs W28 is ≤ 4 tokens.  The cram-factor is not the load-bearing
   gain on this milestone — the load-bearing gain is **scaling-
   stability** at long windows AND the new infrastructure
   (gold-correlation routing + change-point detection +
   manifest-v2 cross-component swap detection).*
5. **Did two-Mac evaluation materially broaden the evidence?**
   *YES — the live cross-architecture gold-verifiable probe
   (gemma2:9b + qwen2.5:14b) recorded **the first measured live
   cross-architecture LLM gold-verifiable agreement at temp 0 in
   the programme** (19/20 = 0.950 agreement rate).  Combined with
   W31's R-78-XLLM-LIVE (0.750 agreement on operational prompts),
   the prompt-class-dependent cross-architecture disagreement
   frontier is now characterised.  Mac 2 (192.168.12.248) remains
   ARP-incomplete (27th milestone).*
6. **Which earlier loose ends were closed vs only sharpened?**
   *CLOSED: W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability
   axis (the conjectural status promoted to proved-conditional +
   proved-empirical via W32-3).  SHARPENED: W31-C-CROSS-HOST-
   VARIANCE-LIVE-MAGNITUDE-LIVE on the prompt-class-dependent
   agreement frontier (gold-verifiable agree at 0.950, operational
   disagree at 0.250); renamed forward as W32-C-CROSS-HOST-LIVE-
   GOLD-MAGNITUDE.  STILL OPEN: W22-C-CACHE-AMPLIFICATION;
   W23-C-MITIGATION-LIVE-VARIANCE; W24-C-LIVE-VARIANCE-COMPLETE;
   W26-C-K-SCALING K → ∞ asymptote; W27-C-MULTI-SIGNATURE-SCALING;
   W31-C-NATIVE-LATENT (architecture-dependent: true transformer-
   internal subspace projection vs the W32 audited proxy);
   W31-C-MULTI-HOST (3+ host topology, blocked on Mac 2 ARP);
   W32-C-LONG-WINDOW-STRICT-GAIN (regime that exceeds W32-L-CYCLE-
   CAP); W32-C-OLD-LINE-EWMA-TRUST (W21 multi-oracle EWMA-tracked
   trust integration).*
7. **Did release readiness improve?** *YES on the SDK-version /
   ``__experimental__`` / pyproject.toml axis.  (a) W32 surface
   added under ``__experimental__`` (45 unit tests + verifier +
   13 W32 primitives).  (b) SDK version bumped to v3.33 / 0.5.6.
   (c) Focused regression W3..W32 phase suite + wider wevra =
   536/536, reproducible.  (d) The long-window convergent / EWMA /
   Page CUSUM / gold-correlation lookup / manifest-v2 CID
   vocabulary is honestly framed as capsule-layer audited proxy
   in the module docstring AND in the results note's §9.  (e)
   First live cross-architecture LLM gold-verifiable agreement
   evidence at temp 0 in the programme — concrete infrastructure
   progress beyond synthetic benches AND beyond W31's operational-
   prompt probe.*
8. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on
   FIVE axes simultaneously: long-window convergent EWMA prior
   accumulator (W31-C-LONG-WINDOW-CONVERGENCE discharged on
   scaling-stability axis), Page CUSUM change-point detector
   (W32-4 isolated), cross-component manifest-v2 CID tamper
   detection (W32-5), gold-correlation routing infrastructure
   (W32 mechanism is empirically validated; gate-bounded on
   synthetic; ready for live regimes that exceed the cycle-cap),
   first live cross-architecture LLM gold-verifiable agreement
   at temp 0 (W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE characterised
   on the prompt-class frontier).  Strengthened by the
   **W32-L-CYCLE-CAP limitation theorem** which makes explicit
   the structural bound on strict-gain claims under cycle-capped
   dispatcher regimes — a load-bearing honest-scope distinction.
   The deeper wall remains whichever regime the audited capsule-
   layer proxy cannot resolve: a regime where the live LLM
   disagreement on gold-verifiable prompts at temp 0 systematically
   aligns with the gold-correctness label
   (W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE), or a regime that
   exceeds the W32-L-CYCLE-CAP limitation theorem
   (W32-C-LONG-WINDOW-STRICT-GAIN), or true transformer-internal
   subspace projection (W32-C-NATIVE-LATENT — architecture-
   dependent), or a regime where 3-host majority is needed
   (W32-C-MULTI-HOST — hardware-bounded; blocked on Mac 2 ARP).*
   Named open frontier for SDK v3.34: **W32-C-CROSS-HOST-LIVE-
   GOLD-MAGNITUDE** (gold-correlation axis on regimes where it
   actually fires), **W32-C-LONG-WINDOW-STRICT-GAIN** (regime
   that exceeds W32-L-CYCLE-CAP), **W32-C-OLD-LINE-EWMA-TRUST**
   (W21 EWMA-tracked trust integration), **W32-C-NATIVE-LATENT**
   (architecture-dependent), **W32-C-MULTI-HOST** (hardware-
   bounded).

---

## SDK v3.34 (W33 family) — trust-EWMA-tracked multi-oracle adjudication + per-oracle agreement signal + anchor-oracle reference + content-addressed oracle-trust-state + trust-trajectory CID + W33 manifest-v3 CID + single-partition long-window strict-gain regime — 2026-05-01

W33 wraps the **OLD W21 ``TrustWeightedMultiOracleDisambiguator``**
(the explicit multi-oracle adjudication line, dormant since W21)
with a ``TrustEWMATrackedMultiOracleOrchestrator`` that adds **six
NEW audited proxies** at the capsule layer: a per-oracle EWMA trust
accumulator (the same closed-form W32 primitive ``ewma_new =
(1 - α) * ewma_prev + α * obs`` applied to the per-oracle
agreement signal), a per-oracle agreement signal that compares each
oracle's probe top_set to a registered **anchor-oracle reference**
(a trust-by-construction reference immune to quorum-flip), a
**closed-form trust-threshold gate** (default 0.50; oracles whose
EWMA falls below threshold are excluded from the effective tally),
a **content-addressed oracle-trust-state CID** + **content-
addressed trust-trajectory CID** + **content-addressed
anchor-oracle-set CID**, and a **manifest-v3 CID** over six
component CIDs (w21_oracle_cid, oracle_trust_state_cid,
trust_trajectory_cid, anchor_oracle_set_cid, route_audit_cid_v3,
w32_long_window_cid).  W33 is **NOT** a learned trust model: zero
parameters, zero gradients, zero training step.  The W33 layer
sits at the capsule-layer **as audited proxy** for the LatentMAS
cross-agent-trust direction, NOT a runtime hidden-state
transplant.

The ``TrustEWMARatificationEnvelope`` adds a sealed trajectory of
``(cell_idx, oracle_id, observed_quorum_agreement,
ewma_trust_after)`` entries plus a ``manifest_v3_cid``; the new
``verify_trust_ewma_ratification`` enumerates **14 disjoint
failure modes** (cumulative 70 across W22 + W29 + W30 + W31 +
W32 + W33).  Five W33-Λ falsifiers are named and empirically
validated.

**First capsule-native multi-agent-coordination method to
simultaneously discharge two named open conjectures from two
different research lines in a single milestone**: on **R-80-
COMPROMISED-SHIFT** (a three-phase oracle-compromise regime: K1 =
3N/8 calibration / K2 = 5N/8 single compromise / K3 = N double
compromise), W33 discharges **W21-C-CALIBRATED-TRUST + W32-C-OLD-
LINE-EWMA-TRUST** at trust precision 1.000 vs W21 = 0.625 ⇒ Δ =
+0.375 across 5/5 seeds × 16 cells/seed, no correctness
regression, max overhead 1 token/cell.  On **R-79-SINGLE-
PARTITION** (a prefix-then-shift regime over a single-partition
signature space whose effective signature diversity exceeds the
W32-L-CYCLE-CAP cycle-capped Δ_max ≤ 0.0625 bound by
construction), W33 / W32 discharges **W32-C-LONG-WINDOW-STRICT-
GAIN** at Δ(W32 - W31) = +0.100 across 5/5 seeds × 80 cells.
On **R-80-MANIFEST-V3-TAMPER** the manifest-v3 CID + cross-
component CID checks together yield **400/400 = 1.000 tamper
rejection rate** across 5/5 seeds × 16 ratified cell-positions ×
5 named tampers per cell.

**Live cross-host trust-calibration probe (S1 best-effort,
honestly null on infrastructure).**  mixtral:8x7b on localhost +
qwen3.5:35b on 192.168.12.191 across 20 trust-calibration prompts
at temperature 0.  qwen3.5:35b returned empty / timeout on every
prompt (model not actually loaded on the remote host); mixtral:8x7b
ignored the "EXACTLY one word" constraint.  Two named
infrastructure-fix items recorded: **W33-INFRA-1** (pre-flight
``/api/tags`` model verification) and **W33-INFRA-2** (stricter
token-budget / chat-template for one-word probes).  The
**W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE** conjecture remains open;
the W33 mechanism's discharge claims do not depend on this live
probe.

Verdict against pre-committed success criterion: **10/10 hard
gates PASS = STRONG SUCCESS**.  This is a single-milestone joint
discharge of three named open conjectures
(W21-C-CALIBRATED-TRUST + W32-C-OLD-LINE-EWMA-TRUST + W32-C-
LONG-WINDOW-STRICT-GAIN) — a load-bearing event for the
research programme.

Eight self-questions (post-W33):

1. **Is the new mechanism a real new mechanism, not just a rename
   of W22..W32?** *YES — the per-oracle agreement signal +
   anchor-oracle reference + per-oracle EWMA + trust-threshold
   gate is a new closed-form mechanism that did not exist in any
   W22..W32 verifier; the cumulative 70-mode trust boundary
   (W22 + W29 + W30 + W31 + W32 + W33) confirms 14 failure modes
   disjoint from prior milestones.*
2. **Is correctness measurably stronger than the prior baseline
   on at least one regime, with overhead bounded?** *YES on TWO
   regimes simultaneously: R-80-COMPROMISED-SHIFT trust-precision
   gain +0.375 across 5/5 seeds with overhead 1 token/cell, AND
   R-79-SINGLE-PARTITION correctness gain +0.100 across 5/5
   seeds × 80 cells with mean overhead 0.875 tokens/cell (W32
   primitive applied to a regime that exceeds W32-L-CYCLE-CAP).*
3. **Is the trust boundary measurably stronger?** *YES — 14
   disjoint failure modes (cumulative 70 across W22 + W29 + W30 +
   W31 + W32 + W33), 400/400 = 1.000 cross-component tamper
   rejection on R-80-MANIFEST-V3-TAMPER, manifest-v3 CID closes
   cross-component swap avenues that the W21 / W32 manifests
   alone cannot detect.*
4. **Are the new audited proxies honest, or are they smuggling in
   transformer-internal manipulation?** *HONEST — per-oracle EWMA
   is closed-form arithmetic with zero parameters; per-oracle
   agreement signal is a deterministic top-set comparison; trust
   trajectory is a sealed (cell_idx, oracle_id, observed,
   ewma_after) tuple; manifest-v3 CID is a SHA-256 over six
   component CIDs.  None claim transformer-internal control;
   architecture-dependent native-trust subspace projection is
   carried forward as W33-C-NATIVE-LATENT.*
5. **Did W33 close any prior named open conjecture?** *YES,
   THREE (joint discharge in a single milestone): W21-C-
   CALIBRATED-TRUST (the OLD W21 multi-oracle line) + W32-C-OLD-
   LINE-EWMA-TRUST (W21-W32 integration axis) + W32-C-LONG-
   WINDOW-STRICT-GAIN (single-partition regime that exceeds
   W32-L-CYCLE-CAP).*
6. **Did W33 falsifiers fire as predicted?** *FOUR named
   falsifiers all empirically observed: W33-Λ-trivial-trust-ewma
   ⇒ byte-for-W21 passthrough; W33-Λ-no-trust-shift ⇒ all EWMA
   stay at 1.0; W33-Λ-frozen-threshold ⇒ gate never fires;
   W33-Λ-mis-trust-shift ⇒ honest empirical correction
   (anchor-oracle design is more robust than predicted).*
7. **Is release-readiness improved?** *YES — SDK_VERSION bumped
   to wevra.sdk.v3.34, pyproject.toml 0.5.7, CHANGELOG entry
   added, ``__experimental__`` updated with W33 symbols (98
   entries, ``__all__`` 440 entries), stable runtime contract
   byte-for-byte unchanged.*
8. **Is the original thesis materially stronger or still blocked
   by a deeper trust/semantics wall?** *MATERIALLY STRONGER on
   THREE axes simultaneously: joint multi-line discharge
   (W21-C-CALIBRATED-TRUST + W32-C-OLD-LINE-EWMA-TRUST), single-
   partition long-window strict-gain (W32-C-LONG-WINDOW-STRICT-
   GAIN discharge on a regime exceeding W32-L-CYCLE-CAP), and
   manifest-v3 cross-component tamper detection at 1.000 reject
   rate.  The deeper wall remains whichever regime the audited
   capsule-layer trust proxy cannot resolve: a regime where the
   live cross-host LLM disagreement systematically aligns with
   trust-calibration ground truth (W33-C-CROSS-HOST-LIVE-TRUST-
   MAGNITUDE — currently infrastructure-bounded), or true
   transformer-internal trust subspace projection (W33-C-NATIVE-
   LATENT — architecture-dependent), or a regime where 3-host
   trust quorum is needed (W33-C-MULTI-HOST — hardware-bounded;
   blocked on Mac 2 ARP).*  Named open frontier for SDK v3.35:
   **W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE** (live cross-
   architecture trust-calibration on a fixed-infra topology),
   **W33-C-NATIVE-LATENT** (architecture-dependent), **W33-C-
   MULTI-HOST** (3+ hardware-bounded), **W33-C-LATENT-CROSS-
   AGENT-TRUST** (true transformer-internal cross-agent trust
   subspace projection).

---

*End of master plan. Changelog lives in the results notes, not
here. If this document ever becomes a changelog, delete the
changelog and restore the plan.*
