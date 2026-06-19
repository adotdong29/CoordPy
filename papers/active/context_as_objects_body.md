## 1. Introduction

Context is the central systems problem in multi-agent LLM workflows.
The default engineering response is prompt-centric: gather more text,
compress it, summarize it, and deliver the result to the next model
call. That framing makes context look like a window-management
problem. In practice it is several distinct problems at once:

1. A producer must preserve relevant ambiguity rather than collapse
   it upstream.
2. A bounded local view must admit the right evidence under budget.
3. A decoder must interpret admitted evidence jointly rather than
   item-by-item.
4. Evidence that arrives in different rounds may need to be decoded
   together.
5. Producer outputs may drift lexically or structurally away from the
   canonical surface expected by downstream logic.
6. Even after admission and decoding, a downstream consumer may still
   receive the wrong subset if the bundle is packed naively under a
   strict token budget.

Most agent systems blur these subproblems inside raw prompts, message
logs, and free-form tool traces. As a result, failures become hard to
classify. A wrong answer may come from upstream producer collapse,
lexical drift, insufficient admission, missing temporal composition,
context-window truncation, or genuine semantic ambiguity. When these
are not separated, the diagnosis tends to be vague: "the model was
confused", "the prompt was weak", "the retriever missed something",
or "the context was too large".

This paper advances a different thesis:

> **Context in multi-agent systems should be treated as an object, not**
> **as untyped text.**

The right unit of context is a typed, immutable, provenance-carrying
object whose lifecycle and budget are explicit. We call these objects
**capsules**. A capsule is not just an envelope around bytes. It is
part of a contract: it has a closed-vocabulary kind, a content-derived
identity, declared parents, explicit budgets, and a lifecycle state.
Once that contract is in place, the context problem becomes
decomposable, auditable, and falsifiable.

### 1.1 Core claim

The strongest current claim of the Context Zero programme is not that
all multi-agent context problems are solved. It is that the context
problem can be turned from a vague prompt-engineering complaint into a
stack of explicit, benchmarked subproblems:

- producer-side ambiguity preservation,
- normalization of producer drift,
- admission under role-local budget,
- intra-round bundle decoding,
- cross-round bundle decoding,
- and downstream bundle packing under bounded decoder context.

Each of these axes can be isolated, attacked, and bounded. The paper's
real contribution is this decomposition plus the evidence that it is
load-bearing.

### 1.2 Why CoordPy matters

The codebase contribution is **CoordPy**, the first product produced by
the Context Zero programme. CoordPy is not the whole programme and it is
not claimed to be a universal agent platform. Its scientific value is
that it makes the thesis executable:

- capsules are load-bearing at runtime rather than reconstructed after
  the fact;
- team coordination is implemented through typed handoffs and bounded
  role views rather than prompt blobs;
- the benchmark family is wired directly to the same runtime objects;
- every positive result is paired with a named falsifier or limit;
- and many invariants are mechanically checked on every run.

The code is not merely an implementation appendix to the theory. It
is the evidence for the theory.

### 1.3 Main contributions

This paper makes six main contributions.

1. **Capsule-native execution.** We show that a runtime can use typed,
   content-addressed objects as its execution contract all the way to
   the LLM byte boundary.
2. **Capsule-native team coordination.** We extend the same contract
   from single-run execution to between-agent coordination.
3. **A benchmark ladder for multi-agent context.** We define a
   sequence of named regimes, R-53 through R-64, each created to
   expose a specific structural failure mode.
4. **Positive and negative theorem/result pairs.** Each method class
   is paired with its own named win and its own named falsifier or
   limit.
5. **A decomposition of the context problem into structural axes.**
   This is what makes the programme cumulative rather than a sequence
   of disconnected benchmark wins.
6. **The first fresh live end-to-end strict real-LLM win under
   bounded context.** SDK v3.18 provides the first result in the
   programme where producer-side and decoder-side interventions are
   both load-bearing and jointly beat the strongest weaker baseline on
   a fresh live real-LLM stream.

### 1.4 What this paper does not claim

This paper does **not** claim that multi-agent context has been
solved universally. The strongest current results remain conditional
on named benchmark properties, structured forms of drift, and bounded
semantic surfaces. Most importantly, the programme now has a named
**symmetric ambiguity wall**: once corroboration, magnitude, round
structure, normalization, and packing are all symmetric, the current
closed-form method family fails by construction.

That negative theorem is not a weakness of the paper. It is evidence
that the programme is now precise enough to identify what remains
unsolved.

### 1.5 Paper roadmap

The paper is organized as follows. Sections 2 through 5 define the
setting, the capsule contract, the runtime, and the team-level object
model. Section 6 introduces the eight structural axes that emerged
from the programme. Sections 7 through 9 describe the benchmark
ladder, the method families, and the evaluation discipline. Section 10
reports the main empirical and theorem-level results. Sections 11
through 17 synthesize those results into the strongest current thesis,
explain what has and has not been solved, position the work relative
to adjacent literatures, and state the current limitations plainly.
The appendices then summarize the claim taxonomy, map milestones to
paper contributions, and list the remaining paper-production steps for
a venue submission pass.

## 2. Setting and Problem Definition

We consider a multi-agent team with roles
$R = \{r_1, \ldots, r_m\}$ collaborating over one or more rounds.
Each role emits evidence items about a shared task. Evidence may be
causal, generic, misleading, redundant, or lexically drifted. One or
more deciding roles operate under bounded budgets and must produce a
team-level decision.

The paper focuses on the following failure modes:

- **Producer collapse:** the producer suppresses alternative
  hypotheses too early.
- **Producer drift:** the producer emits semantically useful but
  lexically drifted kinds or payloads.
- **Admission overload:** the deciding role cannot admit everything.
- **Single-round insufficiency:** round-local evidence is not enough.
- **Decoder-budget failure:** the right evidence is admitted but
  cannot fit into the final decoder context.
- **Symmetric ambiguity:** even with preserved evidence and bounded
  packing, gold and decoy remain indistinguishable under the current
  feature surface.

The programme's claim is that these must be modeled separately. A
single large prompt hides the distinction.

### 2.1 Formal decomposition

Let a scenario unfold over rounds $t = 1, \dots, T$. At each round,
producer role $r_i$ observes a multiset of raw events
$E_{i,t} = \{e_{i,t,1}, \dots, e_{i,t,n}\}$. A producer protocol
$P_i$ maps those events into a multiset of emitted handoff payloads
$H_{i,t}$. A normalizer $N$ maps emitted payloads into a canonical
surface. An admission policy $\pi_r$ selects a bounded subset
$A_{r,t} \subseteq \bigcup_i N(H_{i,t})$ for a deciding role $r$. A
decoder $D$ maps either a single-round view $A_{r,t}$ or a union of
multiple views $\bigcup_t A_{r,t}$ to an answer $\hat{y}$.

The decomposition studied in this paper is therefore:

- **Producer problem:** which informative hypotheses survive into
  $H_{i,t}$?
- **Normalization problem:** which emitted payloads are mapped
  correctly into the canonical surface?
- **Admission problem:** which canonicalized items survive under the
  role-local budget?
- **Decoding problem:** how should the surviving bundle be
  interpreted jointly?
- **Packing problem:** if a downstream consumer has its own strict
  token budget, which subset and ordering of the decoded bundle should
  be kept?

The standard prompt-centric stack typically fuses all of these into
one lossy transformation. The capsule-native view keeps them distinct
and therefore falsifiable.

### 2.2 What counts as "context" here

The paper uses "context" in a narrower and stronger sense than most
prompt-engineering discussions. Context here is not "all bytes seen by
the system." It is the typed, bounded, provenance-aware information
that a role or decoder is permitted to condition on at a particular
decision point. That distinction matters because:

- it is possible for the system to have seen information that a given
  role should not read directly;
- it is possible for preserving *too much* context to be harmful,
  because it destroys the downstream token budget or drowns decisive
  evidence;
- and it is possible for "better context" to mean **less text** but
  **more structure**.

## 3. Capsule Contract

The shared object model is the **capsule contract**. Each capsule
satisfies six invariants:

1. **Identity (C1).** The capsule identifier is a SHA-256 hash of a
   canonicalized representation of kind, payload, budget, and parent
   set.
2. **Typed kind (C2).** Every capsule belongs to a closed vocabulary.
3. **Lifecycle (C3).** Capsules move through
   `PROPOSED -> ADMITTED -> SEALED [-> RETIRED]`.
4. **Budget (C4).** Capsules carry explicit limits such as token
   count, byte size, witness count, round count, or parent count.
5. **Provenance (C5).** Capsules form a DAG stored in a hash-chained
   ledger.
6. **Immutability (C6).** Sealed capsules are frozen.

None of these ingredients is new in isolation. The novelty is that
the same contract is used all the way from runtime execution to team
coordination to evaluation.

### 3.1 Why the contract matters for science

The contract is what lets the programme attach real negative theorems
to system behavior. Several of the strongest limits in the paper
depend directly on the contract:

- if a producer never emits a handoff, no downstream method can
  recover it because the missing evidence has no CID and never enters
  the ledger;
- if a decoder consumes only a bounded packed subset, then dropped
  capsules are not just "ignored information" but a typed excluded
  set;
- if a benchmark property fails, the failure is visible at the object
  level rather than inferred heuristically from text.

Without the contract, many of the paper's strongest negative results
would be soft observations rather than crisp limits.

### 3.2 Scope and cost of the closed vocabulary

The closed vocabulary is simultaneously a strength and a cost.

It is a strength because:

- it makes audit possible,
- it makes theorem statements concrete,
- it makes normalization and decoding surfaces explicit,
- and it prevents silent drift in the meaning of kinds.

It is a cost because:

- every table has a finite closure,
- domain transfer is not automatic,
- and eventually the system hits a wall where richer semantics are
  required.

The later layers, especially W12, W13, and W17, are best read as
systematic explorations of that tradeoff.

## 4. Capsule-Native Execution

The first stage of the programme made capsules load-bearing inside one
CoordPy run. The execution spine includes:

`PROFILE -> READINESS_CHECK -> SWEEP_SPEC -> SWEEP_CELL -> PROVENANCE -> ARTIFACT -> RUN_REPORT`

and then extends inward:

`PROMPT -> LLM_RESPONSE -> PARSE_OUTCOME -> PATCH_PROPOSAL -> TEST_VERDICT`

Key established runtime results include:

- lifecycle/execution correspondence for the run spine and inner loop;
- content-addressing at write time for substantive artifacts;
- deterministic DAG replay;
- mechanical lifecycle audit;
- and a sharp impossibility theorem for authenticating
  meta-artifacts inside the primary ledger, together with a detached
  witness construction.

### 4.1 Runtime as scientific instrument

The runtime should not be understood only as engineering. It is also
the measurement instrument for the research programme. Because every
load-bearing artifact is a typed capsule:

- the exact producer prompt can be referenced and hashed,
- the exact model output can be referenced and hashed,
- parser outcomes become typed witnesses,
- and later benchmark runs can be compared at the object level rather
  than only at the answer level.

This is part of why the programme can draw stronger conclusions than a
typical benchmark-only paper. The runtime is not external to the
claim; it is what makes the claim inspectable.

### 4.2 Why deterministic replay matters

Several later arguments rely on the distinction between:

- a fresh live run,
- a byte-stable replay of a live run,
- and a synthetic or synthetic-real-shaped generator.

Deterministic replay at the capsule DAG level is what lets the paper
state these distinctions precisely. W16 can make a replay-based
composition claim without pretending it is a fresh live claim, and
W17 can then say exactly what changed when the result moved from
replay to live.

## 5. Team-Level Capsule Coordination

At the team layer, the core coordination objects are:

- `TEAM_HANDOFF`: a typed handoff from one role to another;
- `ROLE_VIEW`: a bounded admitted view for one role in one
  coordination step;
- `TEAM_DECISION`: the team-level decision.

The team layer introduces mechanically checked lifecycle invariants
T-1..T-7. It also enables the key separation used throughout the
paper:

- what the producer emits,
- what the deciding role admits,
- what the decoder reads,
- and what the final answer asserts

can all be benchmarked independently.

### 5.1 Local views as bounded-context objects

The `ROLE_VIEW` object is where the programme's bounded-context thesis
becomes operational. A role view is not just a slice of the ledger. It
is the exact set of capsules the role is allowed to condition on,
subject to:

- parent constraints,
- target-role constraints,
- cardinality limits,
- token limits,
- and later, decoder-side packing limits on downstream consumption.

In other words, the role view is the first explicit formal object in
the codebase that stands in for "the minimum sufficient context for
this role at this time." The W15 layer sharpens that further by
introducing the distinction between admitted role view and decoder-fed
packed subset.

### 5.2 Why team-level audit matters

The programme's strongest multi-agent claims are not just about
accuracy. They are also about **well-formedness**. A team-level answer
is only meaningful if the coordination round itself is structurally
sound. The T-1..T-7 audit makes that explicit. This is why almost
every benchmark table in the programme reports audit preservation
alongside correctness.

## 6. Structural Axes of the Context Problem

The best way to understand the programme is as a ladder of structural
axes. Each later axis only became visible because the earlier ones
were isolated first.

### 6.1 Axis 1: Admission under budget

Question: what should a bounded role-local view admit?

Main results:

- **W7-2 / R-54.** Buffered cohort coherence beats FIFO when the gold
  tag has strict plurality.
- **W8-1 / R-55.** Cross-role corroboration beats single-tag
  plurality when decoys have raw plurality but gold has stronger
  distinct-role support.
- **W9-1 / R-56.** Multi-service corroboration beats single-tag
  corroboration on a multi-service gold regime.

Named limits:

- **W7-1.** FIFO is unbeatable when the producer already emits fewer
  candidates than the budget cap.
- **W9-4.** Service-blind admission fails when the decoy is also
  sufficiently corroborated.

Interpretation:

Admission alone can solve some hard context problems, but only when
the distinguishing structure is already visible at the level of what
gets admitted.

### 6.2 Axis 2: Intra-round bundle decoding

Question: once a single-round bundle is admitted, can bundle-aware
decoding solve a regime admission cannot solve?

Main result:

- **W10-1 / R-57.** Bundle-aware decoding strictly beats every
  admission policy on a decoder-forcing regime.

Named limit:

- **W10-Λ.** Admission alone is structurally insufficient on R-57.

Interpretation:

This is the first point where the programme proves that the meaning of
the **bundle** matters, not just the membership of the admitted set.

### 6.3 Axis 3: Cross-round decoding

Question: can evidence from multiple rounds be decoded jointly when
early evidence carries service inventory and later evidence carries
causal specificity?

Main result:

- **W11-1 / R-58.** Multi-round bundle decoding beats every
  single-round method.

Named limits:

- **W11-Λ.** Single-round methods are structurally insufficient.
- **W11-4.** Round-level budget starvation is a sharp falsifier.

Interpretation:

The programme now has a precise example of a context problem that is
not about retrieving the right item but about carrying the right
information across time.

### 6.4 Axis 4: Fixed-vocabulary normalization

Question: if producer drift stays inside a known synonym closure, can
normalization restore the cross-round decoder?

Main result:

- **W12-1 / R-59.** Fixed-table normalization plus multi-round
  decoding closes the synthetic-to-real-shaped gap.

Named limit:

- **W12-4.** Out-of-vocabulary drift defeats the fixed table.

Interpretation:

This is the first place where the programme converts a familiar LLM
"robustness" issue into a named structural axis with both a strict
positive result and a strict falsifier.

### 6.5 Axis 5: Layered open-world normalization

Question: can a layered heuristic normalizer widen the closure beyond
exact lookup while preserving backward compatibility?

Main result:

- **W13-1 / R-60-wide.** Layered normalization strictly beats the
  fixed table on a synthetic open-world regime.

Named limits:

- **W13-4.** Cosmic OOV is a sharp finite-closure wall.
- **W13-Λ-real.** On the first real-Ollama probe, the model emits
  canonical kinds and filters away the intended ambiguity upstream, so
  normalization becomes structurally invisible.

Interpretation:

W13 matters because it teaches the programme not to chase the wrong
next move. If the real producer is already canonicalizing or
compressing aggressively, then downstream normalization is not the
bottleneck.

### 6.6 Axis 6: Producer-side ambiguity preservation

Question: if the real producer compresses away ambiguity, can prompt
and protocol design preserve the hard event shape?

Main result:

- **W14-1 / R-61.** Structured producer protocol yields the first
  real-LLM strict gain on the cross-round stack.

Named limit:

- **W14-Λ-prompt.** If the producer does not emit the necessary
  ambiguous evidence, no downstream capsule method can recover it.

Interpretation:

The W14 layer is the first time the programme attacks the bottleneck
before the capsule coordination pipeline even starts.

### 6.7 Axis 7: Decoder-side bounded-context packing

Question: even if the right evidence is emitted and admitted, can
naive decoder-side packing destroy the advantage under a strict token
budget?

Main result:

- **W15-1 / R-62-tightbudget.** Attention-aware capsule context
  packing strictly beats FIFO packing under bounded `T_decoder`.

Named limits:

- **W15-Λ-budget.** FIFO packing ties FIFO when the decisive
  disambiguator falls past the token budget.
- **W15-Λ-degenerate.** If the token budget is effectively
  unbounded, the W15 gain disappears.

Interpretation:

This is one of the programme's most important conceptual moves
because it turns "minimum sufficient context" from a slogan into a
measurable object.

### 6.8 Axis 8: End-to-end composition

Question: can producer-side ambiguity preservation and decoder-side
packing be jointly necessary on the same regime?

Main results:

- **W16-1 / R-63 synthetic.** W14+W15 composition yields the first
  strict end-to-end composition gain.
- **W16-Λ-real-replay.** The same composition yields a strict gain on
  recorded real-LLM bytes.
- **W17-1 / R-64 live.** Magnitude-hinted producer protocol plus W15
  packing yields the first fresh live end-to-end strict +1.000
  advance over the strongest non-composed baseline.

Named limits:

- **W16-Λ-compose.** When producer collapse and decoder-budget
  pressure both fire, neither layer alone is sufficient.
- **W17-Λ-no-hint.** The legacy structured producer prompt preserves
  only a 7/8 + 0.500 envelope on the live axis.
- **W17-Λ-naive.** Naive prompt plus tight decoder budget yields
  total joint failure.

Interpretation:

This is the strongest evidence so far for the original Context Zero
thesis: the context problem is neither purely upstream nor purely
downstream.

### 6.9 The new wall: symmetric ambiguity

The strongest current negative theorem is:

- **W17-Λ-symmetric.** When gold and decoy are symmetrically
  corroborated under comparable magnitudes, every current
  capsule-native strategy in the SDK ties FIFO at 0.000, even though
  the root-cause label itself can still be correct.

This wall names the next frontier: richer semantic disambiguation
beyond the current closed-form feature surface.

## 7. Benchmark Ladder: R-53 to R-64

The benchmark family is cumulative. Each regime was built because the
previous one exposed a specific failure mode.

| Regime | Purpose | Main winner | Main limit |
| --- | --- | --- | --- |
| R-53 | Show when FIFO is unbeatable | none | low-surplus ceiling |
| R-54 | Gold-plurality admission regime | W7 | streaming instability |
| R-55 | Cross-role corroboration beats plurality | W8 | corroborated decoy |
| R-56 | Multi-service corroboration | W9 | corroborated multi-service decoy |
| R-57 | Decoder-forcing | W10 | admission ceiling |
| R-58 | Delayed-causal-evidence across rounds | W11 | single-round ceiling |
| R-59 | Synthetic-real-shaped producer drift | W12 | fixed-table OOV wall |
| R-60 | Open-world drift + first real-Ollama probe | W13 | closure wall; upstream erasure |
| R-61 | Producer-side ambiguity preservation | W14 | producer compression |
| R-62 | Decoder-side bounded context packing | W15 | FIFO packing wall |
| R-63 | End-to-end composition | W16 | joint producer+decoder failure |
| R-64 | Fresh live composition + symmetric wall | W17 | symmetric ambiguity |

### 7.1 Benchmark design principles

The benchmark ladder obeys four design rules.

1. **Each regime is motivated by the previous regime's strongest
   failure mode.**
2. **Each regime has a named bench property.**
3. **Each regime has at least one named falsifier.**
4. **Each new regime preserves or checks earlier anchors.**

This is what keeps the programme from turning into arbitrary benchmark
shopping.

### 7.2 Regime families in plain language

The ladder can also be read narratively:

- **R-53** asks when the producer is already so clean that structure
  cannot help.
- **R-54/R-55/R-56** ask when better admission rescues bounded local
  context.
- **R-57** asks when decoding matters more than admission.
- **R-58** asks when time matters.
- **R-59/R-60** ask when producer drift becomes the bottleneck.
- **R-61** asks what happens when the producer erases ambiguity
  itself.
- **R-62** asks what happens when the decoder cannot afford to read
  the full admitted union.
- **R-63** asks whether those two difficulties interact.
- **R-64** asks what happens when every current asymmetry is removed.

This narrative is the actual research arc of the programme.

### 7.3 Why the ladder is not benchmark shopping

A fair concern for any long-running benchmark programme is that later
regimes may simply encode the method designer's preferred answer. The
paper tries to make that accusation difficult in four ways.

First, each new regime is motivated by a concrete failure mode exposed
by the previous strongest method. The regime does not appear from a
blank slate. It is a response to an identified ceiling. Second, later
regimes preserve earlier anchors rather than replacing them. A method
that wins only on the new regime but regresses badly on earlier ones
does not count as a clean programme advance. Third, each regime has a
named falsifier. The falsifier is not rhetorical. It is an executable
counter-regime in which the new method is expected to tie or fail.
Fourth, the success bars are pre-committed in the repository rather
than post-hoc descriptions written after a favorable run.

This design does not make the ladder bias-free in some impossible
absolute sense. It does, however, convert the relevant bias questions
into visible artifacts: reviewers can inspect which failure motivated
which regime, whether the stated falsifier really binds, and whether
the new method preserves earlier anchors. That is substantially more
scientifically useful than a single monolithic benchmark with no
internal causal story.

## 8. System Design and Method Families

### 8.1 Runtime layer

The runtime contribution is the capsule-native execution contract.
This layer supplies:

- immutable typed objects,
- a hash-chained provenance surface,
- deterministic replay,
- lifecycle audit,
- and content-addressed artifacts.

By itself, this does not solve multi-agent context, but it makes
later claims auditable.

### 8.2 Team coordination layer

The team layer contributes typed handoffs, role views, and team
decisions. The deciding role never sees an arbitrary prompt blob; it
sees a bounded, typed object set.

#### 8.2.1 Why this differs from ordinary message passing

At first glance, typed handoffs may look like ordinary structured
messages. The difference is that capsules are:

- content-addressed,
- lifecycle-bounded,
- provenance-linked,
- budgeted,
- and manipulated by the same runtime contract that governs the rest
  of the system.

That is what allows the programme to make cross-layer claims ordinary
message schemas typically do not support.

### 8.3 Admission methods

The admission family evolves from:

- FIFO and fixed priority,
- to buffered cohort coherence,
- to cross-role corroboration,
- to multi-service corroboration.

Each method is small, deterministic, and interpretable. Each also has
its own named limit. The programme does not hide the fact that
admission is only one part of the story.

#### 8.3.1 Why the admission ladder matters

The admission-side ladder is sometimes misread as a sequence of
slightly fancier heuristics. That is not the right reading. Each step
isolates a different informational pattern:

- **plurality**,
- **cross-role corroboration**,
- **multi-service corroboration**.

Those are different structural features of the evidence stream. W7,
W8, and W9 matter because they show exactly which kinds of structure
admission can exploit before decoding becomes necessary.

### 8.4 Decoder methods

The decoder family evolves from:

- single-round priority decoding,
- to bundle-aware intra-round decoding,
- to contradiction-aware cross-round decoding.

The decisive step is that the decoder operates on bundles of capsules
rather than on isolated items. This is where the argument starts to
look like a real context solution rather than a better filter.

#### 8.4.1 From item scoring to bundle interpretation

The decoder-side move changes the scientific object of study. The
question is no longer "which items are best?" but "what does this
bundle mean jointly?" That distinction is what allows the programme to
state W10-Λ and W11-Λ as ceilings on admission or single-round
reasoning rather than as failures of one heuristic.

### 8.5 Normalization methods

Normalization evolves from:

- no normalization,
- to fixed closed-vocabulary synonym tables,
- to layered heuristic abstraction rules.

The key scientific point is not that a table got larger. It is that
the programme turned closure and closure-failure into named research
objects with positive and negative results.

#### 8.5.1 Why fixed vs layered normalization is a research axis

Normalization is often treated as preprocessing. In this paper it is a
first-class research axis because:

- its closure can be measured,
- its failure modes can be named,
- it interacts with decoder correctness directly,
- and its real-LLM relevance depends on what the producer actually
  emits.

The W13-Λ-real observation is especially important because it shows
that improving a downstream normalizer may be scientifically
irrelevant if the producer is not generating the ambiguity one hoped
to normalize.

### 8.6 Producer-side protocol

The W14/W17 producer protocol family is the first layer that acts
before the capsule pipeline receives evidence. Its purpose is to stop
the model from erasing the hard event shape too early.

W14 contributes:

- observation/diagnosis separation,
- per-tier kind whitelists,
- one-claim-per-event discipline.

W17 adds:

- explicit operational threshold tables,
- anti-relative-magnitude instructions,
- and a fresh live comparison showing that these additions, not mere
  rerunning, close the remaining 1/8 miss.

#### 8.6.1 Protocol design as scientific intervention

The producer protocol is not merely "better prompting." In the
programme it functions as a controlled intervention on the producer's
emission distribution. That distinction matters:

- a good benchmark should not hard-code the answer into the prompt;
- but it may legitimately constrain the producer to emit one claim per
  event, or to separate observation from diagnosis, if that is exactly
  the phenomenon under study.

W14 and W17 are therefore best understood as *protocol* results, not
prompt-aesthetics results.

### 8.7 Decoder-side packing

The W15 packer turns bounded-context efficiency into an explicit axis
rather than a vague hope. The packer does not claim to manipulate
transformer attention weights directly. Instead it optimizes
prompt-facing evidence order and retention under budget using a
closed-form salience score plus hypothesis preservation.

#### 8.7.1 Why token efficiency is not cosmetic

Many systems can improve accuracy by reading more. That is not the bar
in this paper. The W15 layer matters precisely because it creates a
regime where:

- the right evidence exists,
- the system has already admitted it,
- but a bounded downstream consumer still fails unless the bundle is
  packed well.

That is the first place where "minimum sufficient context" becomes a
measurable systems property rather than a slogan.

### 8.8 End-to-end composition

W16 and W17 are the first layers that prove two different parts of
the system must work together on the same cell. This is the strongest
evidence so far for the original Context Zero thesis.

#### 8.8.1 Why composition is the real milestone

The W16/W17 composition results are important because they rule out a
common failure mode of layered research programmes: each layer works
on its own benchmark but they do not matter together. W16-Λ-compose
and W17-1 show the opposite:

- the layers can fail jointly,
- the layers can help jointly,
- and the resulting strict gains are larger than the gains of the
  isolated pieces.

That is the closest the programme has come so far to an actual
end-to-end context solution story.

### 8.9 What remains outside the current family

The paper is explicit that the current methods remain largely
closed-form. The next frontier likely requires:

- richer semantic compatibility scoring,
- learned or distilled disambiguation,
- or domain transfer beyond incident-style coordination.

That is not an omission; it is the precise consequence of
W17-Λ-symmetric.

## 9. Experimental Methodology

### 9.1 Evaluation discipline

The programme uses pre-committed success bars in
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`. A milestone is not
counted as a real advance unless it clears a named bar with:

- a code anchor,
- a benchmark property,
- a named falsifier,
- non-regression on earlier anchors,
- and audit preservation.

This discipline exists because earlier conjectures in the programme
were overstated and later retracted. The paper benefits from that
history rather than hiding it.

#### 9.1.1 Why the retractions matter

The earlier retractions are not embarrassing details to omit. They are
what justify the later discipline. Several later success bars exist
specifically because earlier claims were too easy to state and too
weak to survive replication.

### 9.2 Metrics

The key metrics are:

- `accuracy_full`,
- `accuracy_services`,
- `accuracy_root_cause`,
- audit pass rate,
- admitted capsule counts,
- token/context usage,
- ambiguity survival rate,
- decoy corroboration rate,
- normalization/OOV statistics,
- and cross-seed or cross-bank stability.

For W15-W17, the programme also uses bundle-facing proxy metrics such
as:

- `position_of_first_causal_claim`,
- `tokens_kept_sum`,
- `hypothesis_count_kept`,
- and `n_dropped_budget`.

These are not raw transformer attention weights. They are explicit,
auditable proxies for prompt-facing evidence placement under budget.

#### 9.2.1 Accuracy alone is not enough

The later layers reject a pure-accuracy notion of success. A method
that matches accuracy only by reading materially more context or by
relying on an effectively unbounded decoder budget does not count as a
strong advance in the programme's sense. This is why W15 and later
success bars include decoder-side token budgets explicitly.

### 9.3 Real-LLM evaluation

The real-LLM work in the paper is deliberately conservative. When a
probe is live, it is called live. When it is recorded replay, it is
called replay. When it is synthetic-real-shaped, it is called that.
This matters because the live-vs-replay distinction is one of the
paper's key honesty constraints.

#### 9.3.1 Replay is still scientifically useful

Replay is not a rhetorical fallback. It serves a specific scientific
role:

- it isolates downstream method changes while holding producer bytes
  constant;
- it enables exact comparison between composed and non-composed
  decoders;
- and it provides a bridge between synthetic and live evidence.

The paper therefore treats replay as a legitimate evidence class, but
never as a substitute for fresh live claims.

### 9.4 Reproducibility and artifact discipline

The programme's reproducibility stance is intentionally stronger than
"the code runs on our machine." Each milestone writes durable JSON
artifacts into `docs/data/`, keeps explicit result notes in `docs/`,
and ties named theorem/result families to code anchors and bench
properties. This paper benefits directly from that discipline because
its strongest claims are not reconstructed from memory. They are read
off of preserved benchmark families, CLI entrypoints, and checked
artifacts.

The practical reproducibility story has four components:

1. **Deterministic synthetic anchors.** Synthetic regimes are designed
   to be rerunnable in CI and under seed sweeps.
2. **Byte-stable replay where live endpoints are not the point under
   study.** Replay is used only when holding producer bytes fixed is
   the scientifically relevant move.
3. **Explicit live capture when the claim is about live behavior.**
   The v3.18 claims are careful about this distinction.
4. **A public theory registry and success-bar registry.** These
   registries ensure that claims cannot quietly change type between
   conjecture, empirical observation, and stronger theorem-style
   statement.

No reproducibility regime is perfect. Real model endpoints can change;
local hardware can differ; and some live probes necessarily depend on
service availability. But the paper's main factual claims are all
backed either by deterministic anchors, stable replay artifacts, or
explicit live captures whose status is recorded rather than blurred.

## 10. Main Results

Before discussing the layers one by one, Table 1 summarizes the
programme's strongest current reading.

| Layer | Representative regime | Strongest method | Strongest result | Named limit |
| --- | --- | --- | --- | --- |
| Admission | R-55 / R-56 | W8 / W9 | strict gains over FIFO when corroboration structure exists | corroborated decoy |
| Intra-round decoding | R-57 | W10 | strict gain where admission alone is insufficient | admission ceiling |
| Cross-round decoding | R-58 | W11 | strict gain when delayed evidence matters | single-round ceiling |
| Fixed normalization | R-59 | W12 | synthetic-real-shaped transfer under bounded closure | fixed OOV wall |
| Layered normalization | R-60 | W13 | wider synthetic open-world closure | cosmic-OOV wall |
| Producer protocol | R-61 / R-64 | W14 / W17 | live ambiguity-preservation gain | producer compression |
| Decoder packing | R-62 | W15 | bounded-context strict gain under tight decoder budget | FIFO packing wall |
| Composition | R-63 / R-64 | W16 / W17 | live end-to-end strict gain over strongest non-composed baseline | symmetric ambiguity |

### 10.1 Runtime layer

The runtime story is already stronger than a standard agent harness:

- capsules govern execution to the LLM byte boundary,
- artifacts are content-addressed at creation time,
- lifecycle audit is mechanical,
- deterministic replay exists,
- and meta-artifact authentication has a sharp impossibility theorem
  plus a constructive workaround.

This gives the team results a trustworthy substrate.

#### 10.1.1 Why this is more than infrastructure

Without this runtime, many later failures would remain hard to
classify. In a conventional agent stack, the difference between
"producer never emitted decoy" and "downstream decoder dropped decoy"
can be surprisingly difficult to pin down. In CoordPy, those are
different object-level states.

### 10.2 Admission layer

The admission results establish that structure can beat FIFO, but only
in the right regimes:

- W7 wins under gold plurality,
- W8 wins under cross-role corroboration,
- W9 wins under multi-service corroboration.

The admission layer is therefore real but limited.

#### 10.2.1 What the admission ladder actually proves

The admission ladder proves that bounded-context coordination already
contains real structure before learned semantics or bundle
interpretation appear. The later decoder wins do not erase W7-W9;
they explain where their limits are.

### 10.3 Decoder layer

The decoder results establish that downstream bundle interpretation is
not reducible to better admission:

- W10 crosses the admission ceiling,
- W11 crosses the single-round ceiling.

This is where the programme first demonstrates that the meaning of the
bundle, not just its membership, matters.

#### 10.3.1 Why bundle semantics changed the programme

Before W10, the strongest story was "choose the right evidence."
After W10 and W11, the story becomes "choose the right evidence, then
interpret it jointly, possibly across time." That is much closer to
what researchers informally mean when they say that an agent team has
a context problem.

### 10.4 Normalization layer

The normalization results show that real or real-shaped producer drift
must be handled explicitly:

- W12 wins under bounded fixed-table closure,
- W13 widens the closure,
- W13-Λ-real shows that normalization is not always the active
  bottleneck on real producers.

#### 10.4.1 The hidden value of the null real result

The W13-Λ-real observation is one of the most important negative
results in the programme because it prevents a wasted research path.
It shows that the right next move after W13 was not "add more
synonyms." It was "fix the producer-side event shape." That kind of
negative result is exactly what a serious paper should include.

### 10.5 Producer-side protocol layer

The W14/W17 story is one of the paper's central contributions.

W14 shows that:

- ambiguity can be erased upstream,
- no downstream method can recover missing emitted evidence,
- and structured prompts can restore the needed event shape.

W17 then shows:

- the remaining model-side relative-magnitude miss can be closed,
- the fresh live gain can be doubled from +0.500 to +1.000,
- and the result transfers partially across model class.

#### 10.5.1 Why W17 is more than a prompt tweak

The W17 layer is not just a slightly better prompt. It closes a
specific model-side loophole: the model was comparing event magnitudes
relatively instead of against the operational thresholds that define
the benchmark property. The contribution is therefore structural: it
changes which events qualify to become capsules in the first place.

### 10.6 Decoder-side packing layer

W15 shows that even when the right evidence exists in the admitted
union, a bounded decoder context can still destroy the win. This is
one of the strongest parts of the paper because it makes the phrase
"minimum sufficient context" measurable rather than philosophical.

#### 10.6.1 Why the packing result is central to the original goal

The original goal of the programme was never just "better
coordination." It was **bounded** coordination: per-agent minimum
sufficient context. W15 is therefore central because it is the first
place where the programme measures not only whether the right answer
is obtained but whether the answer survives a strict decoder context
limit.

### 10.7 Composition layer

W16 and W17 together provide the strongest end-to-end story in the
programme:

- producer-side preservation matters,
- decoder-side packing matters,
- both can be jointly necessary,
- and together they yield the first fresh live strict win.

#### 10.7.1 Why W16 and W17 are the paper's practical center

If the paper had to pick one main result for a broad audience, it
would be W17: fresh live producer-side ambiguity preservation plus
decoder-side packing yields the first strict +1.000 gain over the
strongest weaker baseline on a live model stream.

### 10.8 Symmetric ambiguity wall

The new negative theorem W17-Λ-symmetric is as important as the fresh
live win. It shows that the current method family still depends on
asymmetry in the evidence pattern. Once that asymmetry disappears, the
closed-form capsule-native methods in the SDK stop being sufficient.

That is not a weakness of the paper. It is the cleanest statement yet
of what remains unsolved.

#### 10.8.1 Why the symmetric wall is scientifically valuable

The symmetric wall gives the next paper its problem statement. Without
it, the programme could keep stacking conditional wins without ever
identifying the deeper semantic bottleneck. W17-Λ-symmetric stops that
drift. It says, in effect: the next step is not another packing tweak;
it is a richer ambiguity-resolution method.

## 11. Result-by-Result Synthesis

It is useful to summarize the cumulative story as a progression of
questions and answers.

### 11.1 Can context be made into a runtime object?

Yes. The W3 runtime results show that capsules can govern execution,
not just describe it after the fact.

### 11.2 Can the same object model govern team coordination?

Yes. The W4 layer shows that typed handoffs and role views can be
made load-bearing between agents.

### 11.3 Can structure beat FIFO at all?

Yes, but conditionally. W7-W9 prove several admission-side wins with
named limits.

### 11.4 Can decoding beat admission?

Yes. W10 proves the first decoder-side strict separation from every
admission policy.

### 11.5 Can time matter structurally?

Yes. W11 proves that some regimes require cross-round reasoning.

### 11.6 Can synthetic drift results transfer?

Partly. W12 and W13 show that transfer is possible under bounded or
layered closure assumptions, but W13-Λ-real shows that real producer
behavior can make normalization irrelevant by collapsing ambiguity
upstream.

### 11.7 Can real models be made to preserve the hard event shape?

Yes. W14 and W17 show that prompt/protocol design can recover the
necessary ambiguity on real models, and that operational threshold
language can close the remaining model-side relative-magnitude miss.

### 11.8 Does bounded decoder context really matter?

Yes. W15 shows that downstream context packing is not cosmetic; it can
be the difference between perfect success and total failure.

### 11.9 Do the producer and decoder layers compose?

Yes. W16 and W17 show the first end-to-end composition results, first
on replayed bytes and then on a fresh live real-LLM probe.

### 11.10 What remains unsolved?

The symmetric wall. W17-Λ-symmetric shows that the current closed-form
surface cannot resolve true symmetry.

## 12. What Has Actually Been Solved?

The programme has **not** solved all of multi-agent context. What it
has solved is the decomposition and several substantial subclasses:

1. it solved several admission-sensitive subclasses;
2. it solved decoder-forcing subclasses that admission cannot solve;
3. it solved cross-round subclasses that single-round decoding cannot
   solve;
4. it solved bounded producer-drift subclasses under explicit
   normalization assumptions;
5. it solved a producer-side ambiguity-preservation subclass on a real
   model;
6. it solved a decoder-budget subclass where bounded context itself is
   the main difficulty;
7. and it solved the first end-to-end live composition subclass where
   both producer-side and decoder-side layers are jointly necessary.

That is a substantial research result. It is also still conditional.

### 12.1 Why this still counts as progress toward "solving context"

The phrase "solve context" can be misleading if read as a universal
claim. In this paper, progress means something more disciplined:
showing that more of the difficulty is now structurally understood,
measurable, and addressable. By that standard, the programme has made
substantial progress. It has turned multiple previously blurred
failure modes into explicit theorem/result pairs and has demonstrated
end-to-end live improvement on at least one real model stream.

## 13. The Strongest Current Thesis After SDK v3.18

The strongest current thesis is:

> **Multi-agent context becomes tractable when the system is designed**
> **as a layered capsule-native coordination pipeline in which**
> **producer-side ambiguity preservation, normalization, admission,**
> **bundle-aware decoding, cross-round decoding, and bounded-context**
> **packing are each explicit, audited, and benchmarked.**

This is much stronger than "better prompts help" and much more
precise than "context windows are too small."

It is also stronger than the runtime-only claim. The runtime matters
because it lets the programme audit the evidence flow. But the real
scientific content is the layered decomposition above.

### 13.1 The shortest honest thesis statement

If the entire programme had to be reduced to one sentence, it would
be:

> Multi-agent context is not a single bottleneck but a layered object
> pipeline; solving it means making each layer explicit, bounded, and
> empirically accountable.

## 14. Why the Symmetric Wall Matters

The symmetric wall is not a side note. It is the next real frontier.

The current methods all rely on some asymmetry:

- a gold plurality,
- a corroboration asymmetry,
- a causal-tier asymmetry,
- a round asymmetry,
- a producer-side preservation asymmetry,
- or a packing asymmetry.

W17-Λ-symmetric shows that once the evidence becomes fully symmetric
under the current feature surface, the closed-form strategy family in
the SDK can no longer separate gold from decoy. That means the next
class of methods must add some richer semantic or learned
disambiguation capability.

This is exactly where the next paper or next major section of the
programme should go.

### 14.1 What the next method probably looks like

The symmetric wall strongly suggests that the next successful method
will have to go beyond the current closed-form surface. Likely
directions include:

- learned compatibility scoring across bundles,
- distilled semantic disambiguators,
- or richer narrative-level hypothesis scoring.

The current paper does not attempt to solve that wall prematurely. It
earns the right to ask the question precisely.

## 15. Limitations

This paper has several important limitations.

1. **The strongest positive results remain benchmark-conditional.**
   That is by design, but it matters.
2. **The current family is still domain-specific.** Most strong
   results are in incident-style coordination regimes.
3. **The current semantic surface is still largely closed-form.**
   The next wall likely requires richer learned semantics.
4. **Cross-model transfer is only partial.** The 35B result is a real
   positive but not yet saturation.
5. **The product boundary is narrower than the research surface.**
   Many of the strongest methods remain research-grade and opt-in.

These are not weaknesses to hide. They are the reason the paper is
credible.

### 15.1 Why the limitations section is part of the contribution

One of the programme's distinguishing features is that limitations are
named as part of the method story, not left to reviewer inference.
This has two benefits:

- it makes the positive claims stronger, because they are bounded;
- and it turns the next research agenda into a direct continuation of
  the current one.

### 15.2 Internal validity threats

The main internal-validity question is whether later regimes merely
mirror the engineering intuitions of the current method family. The
paper's answer is incomplete but concrete: named falsifiers, anchor
preservation, and explicit ceilings reduce that risk, but they do not
eliminate it. Reviewers should still inspect whether a given regime
overfits a specific closed-form scoring rule or prompt protocol.

Another internal-validity issue is the mixture of evidence classes.
Some claims are purely deterministic synthetic statements; others are
live empirical observations against model endpoints. The paper tries to
avoid category mistakes by naming these evidence classes explicitly.
Still, a reader should not treat a synthetic stability result and a
live endpoint result as equally robust in the same sense.

### 15.3 External validity threats

The programme currently lives in a particular family of tasks:
incident-style, multi-service, causally entangled coordination. That
family is broad enough to expose real context failures but narrow
enough that some of the current semantic surfaces are meaningful. It
remains an open question how much of the W7-W17 ladder transfers to:

- product-planning teams,
- long-horizon software agents,
- scientific assistants,
- negotiation or dialogue-heavy teams,
- or workflows where the relevant evidence is not naturally organized
  as service-root-cause hypotheses.

This is why the paper presents the current thesis as a programme
result, not a universal theorem about all multi-agent cognition.

### 15.4 Why the remaining wall matters more than another incremental win

The symmetric-corroboration wall is not just the next benchmark. It is
the point at which the current feature surface no longer carries enough
information to decide correctly. That matters because it distinguishes
between two different futures for the programme.

In one future, a modestly richer but still largely interpretable
semantic compatibility surface breaks the wall and preserves bounded
context efficiency. In the other, symmetry can only be broken by
methods that effectively reintroduce large opaque model calls over
nearly the whole bundle, at which point the object model still helps
auditing but no longer supplies the main disambiguation power. The
paper does not know yet which future is correct, but it now states the
choice sharply enough to study.

## 16. Related Work and Positioning

The programme sits at the intersection of several literatures:

- content-addressed and tamper-evident object systems,
- event-sourcing and provenance-aware execution,
- exact-memory and bounded-context systems,
- multi-agent coordination and blackboard-style architectures,
- retrieval and memory systems for LLMs,
- prompt/protocol design for structured extraction,
- and evaluation/runtime harnesses for LLM systems.

The distinct contribution here is not merely that CoordPy has a ledger
or that it uses typed objects. It is that the paper uses one object
model to unify:

- runtime execution,
- team coordination,
- theorem/limit statements,
- and benchmarked empirical advances.

The final submission version should include a proper bibliography and
explicit positioning against adjacent systems, memory papers, and
multi-agent reasoning papers. This draft intentionally avoids
inventing loose citations without a proper reference pass.

### 16.1 Memory, retrieval, and long-context systems

Many current systems papers frame the context problem as retrieval or
memory management: decide which past messages, tool outputs, or
documents to place in front of the next model call. That literature is
highly relevant here, but the present paper differs in two important
ways.

First, it does not assume that "the right context" already exists as a
stable set of textual units waiting to be ranked. The producer may have
collapsed useful ambiguity before retrieval even begins. Second, the
paper separates admission, decoding, and packing into different axes.
This makes it possible to say that one system failed because the right
object never survived the producer, while another failed because the
right object survived but fell out of the bounded decoder bundle.

### 16.2 Multi-agent prompting and role specialization

There is a growing literature on prompt-based multi-agent systems in
which different agents critique, debate, verify, or specialize. The
paper is aligned with that literature in spirit but differs in method.
Most prompt-centric multi-agent work takes the message exchange itself
as the central object. Here, the message exchange is downstream of a
typed object model with explicit lifecycle and provenance. That shift
is what lets the paper connect protocol design, normalizer design,
decoder design, and bounded-context packing inside one cumulative
programme.

### 16.3 Formal methods, provenance, and systems audit

The capsule contract also places the paper near provenance-aware and
auditable systems work. However, ordinary provenance systems typically
stop at traceability: which component produced which artifact? The
present paper uses provenance as part of a tighter execution contract.
The same typed object surface that makes audit possible also serves as
the unit of coordination and the unit of scientific evaluation. That is
why the paper is not only a systems-audit paper, not only a runtime
paper, and not only an agent paper.

### 16.4 How this paper differs from a runtime paper

A systems reader may initially see CoordPy as a runtime paper with
benchmark appendices. That is not the right reading. The runtime is
necessary, but the main scientific object is the decomposition of
context across the benchmark ladder. The runtime is what makes the
decomposition executable.

### 16.5 How this paper differs from a prompting paper

A reader coming from prompt engineering may initially see W14 or W17
as prompt wins. That is also not the right reading. The producer
protocol layers matter because they sit inside a broader object-level
stack with named downstream limits. Their meaning comes from that
stack, not from prompt craft alone.

## 17. Discussion: What Would Count as Truly Solving Context?

In this repo, a serious claim that context is solved would require at
least:

1. strong results across several benchmark families rather than one;
2. fresh live real-LLM wins, not replay only;
3. explicit bounded-context efficiency, not just accuracy;
4. robustness beyond hand-built asymmetries;
5. and a convincing story for what happens at the symmetric wall.

The programme is not there yet. But it is now far beyond the stage of
"interesting intuition." It has:

- an executable runtime contract,
- a growing theorem registry,
- multiple strict positive separations,
- multiple named negative theorems,
- and the first fresh live end-to-end win.

That is enough to support a serious publication.

### 17.1 What a skeptical reviewer should believe

After reading the paper, a skeptical reviewer should at least accept
the following:

1. The programme has an unusually explicit object-level runtime and
   coordination surface.
2. The benchmark ladder is cumulative and not arbitrary.
3. Several strong positive results are real, not cherry-picked.
4. Several strong negative results are also real, and they sharpen the
   frontier rather than weakening the paper.
5. The live end-to-end result is strong enough to justify a main paper
   even though the overall thesis remains conditional.

## 18. Conclusion

This paper has one central message:

> **Context in multi-agent LLM systems is not primarily a prompt-size**
> **problem. It is an object-level coordination problem.**

Capsules provide the object model. CoordPy provides the executable
runtime. The benchmark ladder R-53 through R-64 turns the context
problem into a sequence of explicit, falsifiable subproblems.

The strongest current result is not "we solved context." It is:

- admission helps and has a named ceiling,
- decoding helps beyond that and has a named ceiling,
- cross-round reasoning helps beyond that and has a named ceiling,
- normalization helps beyond that and has a named ceiling,
- producer-side ambiguity preservation helps beyond that,
- decoder-side bounded-context packing helps beyond that,
- producer-decoder composition helps beyond that,
- and symmetric ambiguity is the next named wall.

That decomposition is the real scientific contribution. It turns
"context" from a vague complaint into a structured research programme
with executable evidence, honest limits, and a clear next frontier.

### 18.1 Final take-away

The deepest contribution of the paper is not any single +1.000 result.
It is the fact that the programme can now say, with code and tests
behind it, **which layer failed**:

- the producer,
- the normalizer,
- the admission policy,
- the single-round decoder,
- the cross-round decoder,
- the bounded-context packer,
- or the semantic surface itself.

That is what "context as objects" finally buys.

## Appendix A. Claim Taxonomy

The programme uses an explicit taxonomy:

- **proved**
- **proved-conditional**
- **mechanically-checked**
- **empirical**
- **conjectural**
- **retracted**

This taxonomy is essential. Many agent-system papers blur these
statuses; this programme explicitly does not.

## Appendix B. Milestone-to-Paper Map

The current paper incorporates the following layers:

- **W3 family:** capsule contract and capsule-native runtime
- **W4 family:** team-level capsule coordination
- **W7-W9 families:** admission-side coordination ladder
- **W10 family:** intra-round bundle decoding
- **W11 family:** cross-round bundle decoding
- **W12 family:** fixed-vocabulary normalization under bounded
  producer drift
- **W13 family:** layered open-world normalization and real-Ollama
  null result
- **W14 family:** producer-side ambiguity preservation
- **W15 family:** decoder-side bounded-context packing
- **W16 family:** end-to-end producer+decoder composition
- **W17 family:** fresh live composition, magnitude-hinted protocol,
  cross-model live transfer, and symmetric-corroboration wall

## Appendix C. Submission Pass Still Needed

This Markdown manuscript is now substantially beyond a skeleton, but a
submission-quality camera-ready pass would still need production work
of the normal kind:

1. a BibTeX-backed bibliography and in-text citation layer;
2. a figure set with final captions and cross-references;
3. compact venue-shaped tables distilled from the full benchmark
   artifacts;
4. theorem statements rewritten into the exact style of the target
   venue;
5. a reproducibility appendix listing exact scripts, flags, seeds, and
   artifact paths;
6. a pruning pass that moves some implementation detail from the main
   body into appendices.

Those are paper-production tasks, not missing scientific content.

## Appendix D. Recommended figure and table set

The current manuscript would benefit most from the following figures
and tables:

1. **Figure 1: capsule-native runtime overview.**
   Show the path from setup to sealed artifacts to the LLM byte
   boundary.
2. **Figure 2: structural-axis ladder.**
   Show the eight axes and the benchmark that isolates each one.
3. **Figure 3: benchmark ladder timeline.**
   Present R-53 through R-64 as a cumulative sequence of ceilings and
   crossings.
4. **Figure 4: W14-W15-W16-W17 composition diagram.**
   Show producer protocol, normalization, admission, decoding, packing,
   and the bounded decoder input bundle.
5. **Figure 5: symmetric wall.**
   Show why gold and decoy become indistinguishable under the current
   feature surface.
6. **Table 1: regime summary.**
   Keep the ladder summary from Section 7.
7. **Table 2: strongest result by layer.**
   Keep the synthesis table from Section 10.
8. **Table 3: live real-LLM composition results.**
   Present the v3.18 qwen2.5 and qwen3.5 comparisons compactly.

## References

Bibliography intentionally omitted from this Markdown draft.
Replace this section in the final submission version with a real
BibTeX-backed bibliography after the venue-targeting pass.
