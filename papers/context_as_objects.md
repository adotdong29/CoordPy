# Context as Objects: Capsule-Native Coordination for Multi-Agent Teams

> Main paper draft for the Context Zero programme.
> Status: updated through SDK v3.18, 2026-04-27.
> Scope: this is the primary paper draft for the programme's
> multi-agent context thesis. It is not a milestone note and is not
> intended to preserve every intermediate step. It is intended to
> state the strongest current research argument, the main system
> design, the benchmark ladder, the positive results, the negative
> theorems, and the open walls honestly.

## Abstract

Multi-agent LLM systems usually treat context as text: prompts, JSON
blobs, tool traces, and free-form summaries passed between roles.
That design conflates several distinct problems: preserving
ambiguity, normalizing producer drift, admitting evidence under
budget, decoding evidence jointly, and carrying information across
rounds. We argue that this is the wrong abstraction. The unit of
context should be a typed, immutable, lifecycle-bounded object with
explicit budget and provenance.

We implement this view in **Wevra**, a capsule-native runtime and
research harness produced by the **Context Zero** programme. A
capsule is a content-addressed object satisfying six invariants:
identity, type, lifecycle, budget, provenance, and immutability. We
first show that capsules can be made load-bearing in execution rather
than merely post-hoc metadata: the runtime seals capsules from run
setup to the LLM byte boundary and audits them mechanically. We then
extend the same contract to multi-agent coordination, where agents
exchange typed handoff capsules rather than raw messages.

The main scientific contribution of the paper is a decomposition of
the multi-agent context problem into eight coupled structural axes:
(1) admission under budget, (2) intra-round bundle decoding,
(3) cross-round bundle decoding, (4) fixed-vocabulary normalization
of producer drift, (5) layered open-world normalization,
(6) producer-side ambiguity preservation, (7) decoder-side
context packing under bounded token budgets, and (8) end-to-end
producer-plus-decoder composition. For each axis we provide positive
results, named falsifiers, and limit theorems. Across SDK v3.8-v3.18
we build a benchmark ladder R-54 through R-64 that isolates these
axes progressively. Admission alone wins in some regimes but hits a
named ceiling. Bundle-aware decoding crosses that ceiling.
Cross-round decoding crosses a further temporal ceiling.
Normalization crosses producer-drift ceilings. Producer-side
protocol design closes the first real-LLM upstream-erasure gap.
Decoder-side context packing closes a bounded downstream token-budget
gap. End-to-end composition then yields the first fresh live
real-LLM strict +1.000 advance over the strongest non-composed
baseline. Finally, we prove a new negative theorem:
**symmetric corroboration remains a hard wall** for every current
closed-form capsule-native method in the programme.

The paper does **not** claim that multi-agent context is solved in an
unqualified sense. The strongest honest claim is narrower and more
useful: multi-agent context becomes tractable when evidence is
represented as typed objects and when the runtime explicitly
separates producer-side ambiguity preservation, normalization,
admission, intra-round decoding, cross-round decoding, and
decoder-side bounded-context packing. The next open problem is no
longer "can capsules help?" It is "what richer semantics or learned
disambiguators are required once corroboration, magnitude, round
structure, and bounded packing are all exhausted?"

## 1. Introduction

Context is the central systems problem in multi-agent LLM workflows.
The default engineering response is prompt-centric: gather more text,
compress it, summarize it, and deliver a large prompt to the next
model call. That framing makes context look like a window-management
problem. In practice it is several problems at once:

1. A producer must preserve relevant ambiguity rather than compress
   it away upstream.
2. A bounded local view must admit the right evidence under strict
   budget.
3. A decoder must interpret admitted evidence jointly rather than
   item-by-item.
4. Evidence arriving in different rounds must sometimes be combined.
5. Lexically drifted producer outputs must be mapped back into a
   common semantic surface.
6. The final downstream consumer must receive the minimum sufficient
   bundle, not simply the earliest or largest bundle.

Most current systems fuse these problems into raw strings and ad hoc
JSON objects. That makes failures hard to classify. A wrong answer
may come from missing evidence, excess evidence, local ambiguity,
temporal ambiguity, lexical drift, context packing, or upstream
producer compression. Without a sharper abstraction, those failures
are typically diagnosed in vague language: "the model was confused,"
"the prompt was weak," or "the context was too large."

This paper advances a different thesis:

> **Context in multi-agent systems should be treated as an object,**
> **not as untyped text.**

The right unit of context is a typed, immutable, provenance-carrying
object whose lifecycle and budget are explicit. We call these objects
**capsules**. A capsule is not merely an envelope around bytes. It is
part of a contract: it has a closed-vocabulary kind, a content-derived
identity, declared parents, explicit budgets, and a lifecycle state.
Once that contract is in place, the context problem becomes
decomposable and falsifiable.

### 1.1 The programme-level claim

The strongest current claim of the Context Zero programme is not that
all multi-agent context problems are solved. It is that the context
problem can be turned from a vague prompt-engineering complaint into a
stack of explicit, testable subproblems:

- producer-side ambiguity preservation,
- normalization of producer drift,
- admission under role-local budget,
- intra-round bundle decoding,
- cross-round bundle decoding,
- and downstream bundle packing under bounded decoder context.

This decomposition is the paper's real scientific contribution. The
benchmark results matter because they progressively validate and bound
each layer.

### 1.2 Why Wevra matters

The codebase contribution is **Wevra**, the first product produced by
the Context Zero programme. Wevra is not the whole programme and it is
not claimed to be a universal multi-agent platform. Its research value
is that it makes the thesis executable:

- capsules are load-bearing at runtime, not just after the fact;
- team coordination is implemented through typed handoffs and role
  views rather than prompt blobs;
- the benchmark family is wired directly to the same runtime objects;
- every positive result is paired with a named falsifier;
- and many invariants are mechanically checked on every run.

### 1.3 Main contributions

This paper makes six core contributions.

1. **Capsule-native execution.** We show that a runtime can use typed,
   content-addressed objects as the execution contract itself,
   extending to the LLM byte boundary.
2. **Capsule-native team coordination.** We extend the same contract
   from single-run execution to between-agent coordination.
3. **A benchmark ladder for multi-agent context.** We define a
   sequence of regimes, R-53 through R-64, each created to expose a
   specific structural failure mode.
4. **Positive and negative theorem/result pairs.** Each new method
   arrives with both a winning regime and a named falsifier or limit.
5. **A decomposition of the context problem into structural axes.**
   This is what makes the programme cumulative rather than a sequence
   of unrelated benchmarks.
6. **The first live end-to-end strict real-LLM win under bounded
   context.** SDK v3.18 provides the first fresh live result where
   producer-side and decoder-side interventions are both load-bearing
   and strictly beat the strongest weaker baseline.

### 1.4 What this paper does not claim

This paper does **not** claim that multi-agent context has been
solved universally. The strongest current results are still
conditional on named benchmark properties, closed-vocabulary or
closed-form surfaces, and bounded types of producer noise. Most
importantly, the programme now has a named **symmetric ambiguity**
wall: once corroboration, magnitude, round structure, and packing are
all symmetric, the current method family fails by construction.

That negative theorem is not a weakness of the paper. It is evidence
that the programme is now precise enough to identify what remains
unsolved.

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
  semantic surface.

The programme's claim is that these must be modeled separately. A
single large prompt hides the distinction.

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

## 4. Capsule-Native Execution

The first stage of the programme made capsules load-bearing inside one
Wevra run. The execution spine includes:

`PROFILE -> READINESS_CHECK -> SWEEP_SPEC -> SWEEP_CELL -> PROVENANCE -> ARTIFACT -> RUN_REPORT`

and then extends inward:

`PROMPT -> LLM_RESPONSE -> PARSE_OUTCOME -> PATCH_PROPOSAL -> TEST_VERDICT`

The runtime contribution of the programme is already nontrivial:

- lifecycle/execution correspondence for the run spine and inner loop;
- content-addressing at write time for substantive artifacts;
- deterministic DAG replay;
- mechanical lifecycle audit;
- a sharp impossibility theorem for authenticating meta-artifacts
  inside the primary ledger, with a constructive detached witness.

These runtime results matter for the team story because they turn
every later result into something auditable rather than anecdotal.

## 5. Team-Level Capsule Coordination

At the team layer, the core coordination objects are:

- `TEAM_HANDOFF`: a typed handoff from one role to another;
- `ROLE_VIEW`: a bounded admitted view for one role in one
  coordination step;
- `TEAM_DECISION`: the team-level decision.

The team layer introduces mechanically checked lifecycle invariants
T-1..T-7. It also enables the key decomposition used in the rest of
the paper:

- what the producer emits,
- what the deciding role admits,
- what the decoder reads,
- and what the final answer asserts

can all be separated and benchmarked independently.

## 6. A Structural Decomposition of the Context Problem

The best way to understand the programme is as a ladder of structural
axes. Each later axis only became visible because the earlier ones
were isolated first.

### 6.1 Axis 1: Admission under budget

Question: what should a bounded role-local view admit?

Main results:

- **W7-2 / R-54.** Buffered cohort coherence beats FIFO when the gold
  tag has strict plurality.
- **W8-1 / R-55.** Cross-role corroboration beats raw plurality when
  decoys have plurality but gold has stronger distinct-role support.
- **W9-1 / R-56.** Multi-service corroboration beats single-tag
  corroboration in a multi-service gold regime.

Named limits:

- **W7-1.** FIFO is unbeatable when the producer already emits fewer
  candidates than the budget cap.
- **W9-4.** Service-blind admission fails when the decoy is also
  sufficiently corroborated.

### 6.2 Axis 2: Intra-round bundle decoding

Question: once a single-round bundle is admitted, can bundle-aware
decoding solve a regime admission cannot solve?

Main result:

- **W10-1 / R-57.** Bundle-aware decoding strictly beats every
  admission policy on a decoder-forcing regime.

Named limit:

- **W10-Λ.** Admission alone is structurally insufficient on R-57.

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

### 6.4 Axis 4: Fixed-vocabulary normalization

Question: if producer drift stays inside a known synonym closure, can
normalization restore the cross-round decoder?

Main result:

- **W12-1 / R-59.** Fixed-table normalization plus multi-round
  decoding closes the synthetic-to-real-shaped gap.

Named limit:

- **W12-4.** Out-of-vocabulary drift defeats the fixed table.

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

### 6.6 Axis 6: Producer-side ambiguity preservation

Question: if the real producer compresses away ambiguity, can prompt
and protocol design preserve the hard event shape?

Main result:

- **W14-1 / R-61.** Structured producer protocol yields the first
  real-LLM strict gain on the cross-round stack.

Named limit:

- **W14-Λ-prompt.** If the producer does not emit the necessary
  ambiguous evidence, no downstream capsule method can recover it.

### 6.7 Axis 7: Decoder-side context packing

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

### 6.9 The new wall: symmetric ambiguity

The strongest current negative theorem is:

- **W17-Λ-symmetric.** When gold and decoy are symmetrically
  corroborated under comparable magnitudes, every current
  capsule-native strategy in the SDK ties FIFO at 0.000, even though
  the root-cause label itself can still be correct.

This wall names the next frontier: richer semantic disambiguation.

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

The ladder is itself part of the scientific argument. The later
regimes do not supersede the earlier ones; they explain exactly why
the earlier wins and failures happened.

## 8. System Design and Methods

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
decisions. The key design decision is that the deciding role never
sees an arbitrary prompt blob; it sees a bounded, typed object set.

### 8.3 Admission methods

The admission family evolves from:

- FIFO and fixed priority,
- to buffered cohort coherence,
- to cross-role corroboration,
- to multi-service corroboration.

Each method is simple and deterministic, but each also has a sharp
named limit. This is important: the programme does not hide the fact
that admission is only one part of the story.

### 8.4 Decoder methods

The decoder family evolves from:

- single-round priority decoding,
- to bundle-aware intra-round decoding,
- to contradiction-aware cross-round decoding.

The decisive step is that the decoder operates on bundles of capsules
rather than individual items. That is where the argument begins to
look like a true context solution rather than a better filter.

### 8.5 Normalization methods

Normalization evolves from:

- no normalization,
- to fixed closed-vocabulary synonym tables,
- to layered heuristic abstraction rules.

The key scientific point is not that the table got bigger. It is that
the programme turned fixed closure and open-world closure into named
research objects with sharp limits.

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

### 8.7 Decoder-side packing

The W15 packer is important because it turns bounded-context
efficiency into an explicit axis rather than a vague hope. The packer
does not claim to manipulate transformer attention weights directly.
Instead it optimizes prompt-facing evidence order and retention under
budget using a closed-form salience score plus hypothesis
preservation.

### 8.8 End-to-end composition

W16 and W17 are the first layers that prove two different parts of
the system must work together on the same cell. This is the strongest
evidence so far for the original Context Zero thesis.

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
discipline rather than hiding it.

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

### 9.3 Real-LLM evaluation

The real-LLM work in this paper is deliberately conservative. When a
probe is live, it is called live. When it is recorded replay, it is
called replay. When it is synthetic-real-shaped, it is called that.
This matters because the main paper should make the live-vs-replay
distinction impossible to miss.

## 10. Main Results

### 10.1 Runtime layer

The runtime story is already stronger than a standard agent harness:

- capsules govern execution to the LLM byte boundary,
- artifacts are content-addressed at creation time,
- lifecycle audit is mechanical,
- deterministic replay exists,
- and meta-artifact authentication has a sharp impossibility theorem
  plus a constructive workaround.

This gives the team results a trustworthy substrate.

### 10.2 Admission layer

The admission results establish that structure can beat FIFO, but only
in the right regimes:

- W7 wins under gold plurality.
- W8 wins under cross-role corroboration.
- W9 wins under multi-service corroboration.

The admission layer is therefore real but limited.

### 10.3 Decoder layer

The decoder results establish that downstream bundle interpretation is
not reducible to better admission:

- W10 crosses the admission ceiling.
- W11 crosses the single-round ceiling.

This is where the programme first demonstrates that the meaning of the
bundle, not just its membership, matters.

### 10.4 Normalization layer

The normalization results show that real or real-shaped producer drift
must be handled explicitly:

- W12 wins under bounded fixed-table closure.
- W13 widens the closure.
- W13-Λ-real shows that normalization is not always the active
  bottleneck on real producers.

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

### 10.6 Decoder-side packing layer

W15 shows that even when the right evidence exists in the admitted
union, a bounded decoder context can still destroy the win. This is
one of the strongest parts of the paper because it makes the phrase
"minimum sufficient context" measurable rather than philosophical.

### 10.7 Composition layer

W16 and W17 together provide the strongest end-to-end story in the
programme:

- producer-side preservation matters,
- decoder-side packing matters,
- both can be jointly necessary,
- and together they yield the first fresh live strict win.

### 10.8 Symmetric ambiguity wall

The new negative theorem W17-Λ-symmetric is as important as the fresh
live win. It shows that the current method family still depends on
asymmetry in the evidence pattern. Once that asymmetry disappears, the
closed-form capsule-native methods in the SDK stop being sufficient.

That is not a failure of the paper. It is the cleanest statement yet
of what remains unsolved.

## 11. Synthesis: What Has Actually Been Solved?

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

## 12. The Strongest Current Thesis After SDK v3.18

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

## 13. Why the Symmetric Wall Matters

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

This is exactly where a future paper or follow-on section should go.

## 14. Limitations

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

## 15. Related Work and Positioning

The programme sits at the intersection of several literatures:

- content-addressed and tamper-evident object systems,
- event-sourcing and provenance-aware execution,
- exact-memory and bounded-context systems,
- multi-agent coordination and blackboard-style architectures,
- retrieval/memory systems for LLMs,
- and evaluation/runtime harnesses for LLM systems.

The distinct contribution here is not merely that Wevra has a ledger
or that it uses typed objects. It is that the paper uses one object
model to unify:

- runtime execution,
- team coordination,
- theorem/limit statements,
- and benchmarked empirical advances.

The final submission version should include a real bibliography and
explicit positioning against adjacent systems and multi-agent
reasoning papers. This draft intentionally avoids inventing loose
citations without a proper reference pass.

## 16. Discussion: What Would Count as Truly Solving Context?

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

## 17. Conclusion

This paper has one central message:

> **Context in multi-agent LLM systems is not primarily a prompt-size**
> **problem. It is an object-level coordination problem.**

Capsules provide the object model. Wevra provides the executable
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

Before submission, the following still need a real paper-production
pass:

1. proper bibliography and citation integration;
2. figure set:
   - runtime/capsule system diagram,
   - benchmark ladder diagram,
   - layered-axis diagram,
   - live-composition diagram,
   - symmetric-wall diagram;
3. compact result tables prepared for the target venue;
4. pruning or relocation of low-signal implementation detail into an
   appendix;
5. a venue-specific abstract/intro tightening pass.

## References

Bibliography intentionally omitted from this Markdown draft.
Replace this section in the final submission version with a real
BibTeX-backed bibliography after the venue-targeting pass.
