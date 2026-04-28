# Context as Objects: Capsule-Native Coordination for Multi-Agent Teams

> Working main-paper draft for the Context Zero programme.
> Status: SDK v3.17, 2026-04-27 (this draft was last fully revised
> at SDK v3.14; SDK v3.15 / v3.16 / v3.17 add the W14 producer
> protocol, W15 decoder context-packing, and W16 end-to-end
> composition layers as additional structural axes — see § 8 for
> the synthesis-after-v3.17 reading and the canonical milestone
> notes
> ``docs/RESULTS_WEVRA_PRODUCER_AMBIGUITY.md`` (W14),
> ``docs/RESULTS_WEVRA_ATTENTION_AWARE.md`` (W15), and
> ``docs/RESULTS_WEVRA_COMPOSED_REAL_LLM.md`` (W16) for the
> evidence anchors).
> Scope: this is the main paper draft, not a milestone note. It
> synthesizes the programme's current runtime, theory, and benchmark
> results into one publication-shaped argument.

## Abstract

Multi-agent LLM systems usually treat context as untyped text passed
between prompts, tools, and roles. That design makes three different
problems collapse into one: what evidence is preserved by the
producer, what evidence is admitted under budget, and how admitted
evidence is jointly interpreted. We argue that this is the wrong
abstraction. The unit of context should be a typed, immutable,
lifecycle-bounded object with explicit budget and provenance.

We implement this view in **Wevra**, a capsule-native runtime and
research harness produced by the **Context Zero** programme. A
capsule is a content-addressed object satisfying six invariants:
identity, type, lifecycle, budget, provenance, and immutability. We
first show that capsules can be made load-bearing in execution rather
than merely post-hoc metadata: the runtime seals capsules from run
setup through the LLM byte boundary and audits them mechanically. We
then extend the same contract to multi-agent coordination, where
agents exchange typed handoff capsules rather than raw messages.

The paper's main contribution is a decomposition of the multi-agent
context problem into eight structural axes (after SDK v3.17): (1)
admission under budget, (2) intra-round decoding, (3) cross-round
decoding, (4) fixed-table normalization of producer drift, (5)
layered open-world normalization, (6) producer-side ambiguity
preservation under structured prompts (W14, SDK v3.15), (7)
decoder-side capsule context packing under bounded T_decoder (W15,
SDK v3.16), and (8) end-to-end composition where producer-side and
decoder-side interventions are jointly necessary on the same
regime (W16, SDK v3.17). For each axis we provide both positive and
negative results: named regimes where the axis is sufficient, and
sharp falsifiers where it is not. Across SDK v3.8-v3.17 we build a
progressive benchmark family (R-54 through R-63) that isolates these
axes. We show that admission alone can win in some regimes but has a
named structural limit; bundle-aware decoding crosses that ceiling;
cross-round reasoning crosses a further temporal limit; layered
normalization crosses a fixed-vocabulary limit under synthetic
open-world drift; the structured producer protocol closes the
real-LLM upstream-erasure gap on R-61 at +0.500 vs FIFO on
recorded ``qwen2.5:14b-32k`` bytes; the attention-aware capsule
context packer closes the decoder-side budget gap on
R-62-tightbudget at +1.000 vs FIFO-pack on synthetic events; and
the W14+W15 composition is the **first end-to-end real-LLM strict
advance over the strongest non-composed baseline** on R-63 (W16-1
synthetic +1.000 strict gain; W16-Λ-real-replay +0.500 strict
gain on recorded ``qwen2.5:14b`` bytes at ``T_decoder = 14``).

The strongest negative result remains that a real Ollama producer
can collapse the benchmark's ambiguity upstream, making downstream
reasoning invisible (W14-Λ-prompt). The strongest *coupling* result
is W16-Λ-compose: when the producer collapses upstream AND the
decoder is budget-bounded simultaneously, both named limits fire
jointly and no capsule strategy in the SDK clears the bar.

The paper does **not** claim that multi-agent context is solved in an
unqualified sense. The honest thesis is narrower and stronger:
multi-agent context becomes tractable when evidence is represented as
typed objects AND when the runtime explicitly separates producer-side
ambiguity preservation, normalization, admission, intra-round
decoding, cross-round decoding, decoder-side context packing, AND
the composition of producer- and decoder-side interventions on
regimes where both fire. The open problem is no longer
"can capsules help?"; it is "under what real producer conditions do
these axes remain load-bearing under tight context windows, and
which axes can be replaced or learned away?"

## 1. Introduction

Context is the central systems problem in multi-agent LLM workflows.
Most current systems treat context as a prompt-engineering issue:
gather more text, compress it, summarize it, and send the result to
the next model call. That framing hides the fact that a team of
reasoners faces several distinct subproblems:

1. A producer must preserve relevant ambiguity rather than collapse
   it too early.
2. A bounded local view must admit the right pieces of evidence under
   budget.
3. A downstream decision rule must interpret admitted evidence
   jointly rather than item-by-item.
4. The system must remain auditable: when it is wrong, the failure
   should have a typed witness.

In the default prompt-centric design, these steps are fused into raw
strings and ad hoc JSON objects. The result is hard to reason about,
hard to verify, and hard to falsify. The same "context problem" can
mean producer drift in one benchmark, budget pressure in another, and
cross-round delayed evidence in a third.

This paper advances a different claim:

> **Context should be treated as an object, not as a blob of text.**
> In multi-agent systems, the right unit of context is a typed,
> immutable, provenance-carrying object whose lifecycle and budget are
> explicit. The context problem is then the composition of several
> object-level subproblems, not a monolithic prompt problem.

We call these objects **capsules**. A capsule carries a closed
vocabulary kind, a canonical content-derived identity, explicit
budgets, declared parents, and a lifecycle state. Capsules were
initially introduced in this programme as a formal unification layer.
The work summarized here shows that they can do more than name a
formalism:

- they can serve as a runtime execution contract inside one Wevra run;
- they can serve as the coordination object between agents;
- and they can support a sequence of benchmarked, falsifiable
  advances on the multi-agent context problem.

### 1.1 What this paper contributes

This paper makes six main contributions.

1. **Capsule-native execution.** We describe a runtime in which
   cross-boundary artifacts are sealed as typed capsules during
   execution rather than reconstructed afterward. The runtime now
   reaches the LLM byte boundary and supports mechanical lifecycle
   audit.
2. **Team-level capsule coordination.** We extend capsules from
   single-run provenance to between-agent coordination via typed
   handoffs, role-local views, and team decisions.
3. **A benchmark family for multi-agent context.** We define a
   sequence of named regimes, R-53 through R-60, each designed to
   isolate a specific structural question: when FIFO is unbeatable,
   when corroboration matters, when decoding matters, when
   cross-round reasoning matters, and when normalization matters.
4. **Paired positive and negative results.** For each coordination
   axis we give not just a win but also a named falsifier or limit.
   This includes admission ceilings, decoder ceilings, temporal
   ceilings, fixed-vocabulary normalization ceilings, and an honest
   real-LLM null result.
5. **A stronger decomposition of the context problem.** The paper
   argues that "solving context" is not one thing. It is the joint
   management of producer ambiguity, normalization, admission,
   decoding, and cross-round state.
6. **A publication-ready research object.** Rather than a chain of
   milestone notes, this draft presents the programme's current
   thesis in paper form with explicit claims, evidence classes, and
   non-claims.

### 1.2 What this paper does not claim

This paper does **not** claim that the programme has fully solved
multi-agent context in the strongest possible sense. The current
positive results are still conditional on named bench properties and
closed-vocabulary assumptions. The paper also does not claim that the
Wevra product runtime is a universal multi-agent platform. Wevra is
the first product of the Context Zero programme, and the research
layer remains ahead of the product boundary.

The right current claim is stronger than "we have a neat runtime" and
weaker than "context is solved everywhere":

> **Capsule-native coordination is now a theorem-backed, benchmarked,
> and progressively broadened method family for solving bounded forms
> of multi-agent context.**

## 2. Problem Setting

We consider a multi-agent team with roles
$R = \{r_1, \ldots, r_m\}$ collaborating on a task that unfolds over
one or more rounds. Each role emits evidence items. Some evidence is
causal, some generic, some misleading, and some redundant. An
auditing or deciding role has a bounded local budget and must produce
a team-level decision.

The main failure modes we care about are:

- **over-admission:** too much irrelevant evidence survives budget;
- **under-admission:** relevant evidence is dropped;
- **local ambiguity:** each role's evidence looks plausible on its
  own but is misleading globally;
- **temporal ambiguity:** early evidence is non-diagnostic and later
  evidence disambiguates it;
- **producer drift:** the producer emits semantically useful but
  lexically drifted outputs;
- **producer collapse:** the producer compresses away ambiguity
  before the downstream system ever sees it.

The programme's thesis is that these should be modeled separately.
Prompt-only systems tend to confound them.

## 3. Capsule Contract

The capsule contract is the shared object model used throughout the
paper. A capsule satisfies six invariants:

1. **Identity (C1).** The capsule identifier is a SHA-256 hash of a
   canonicalized representation of its kind, payload, budget, and
   parent set.
2. **Typed kind (C2).** Every capsule kind belongs to a closed
   vocabulary.
3. **Lifecycle (C3).** Capsules move through explicit states
   `PROPOSED -> ADMITTED -> SEALED [-> RETIRED]`.
4. **Budget (C4).** Capsules carry explicit limits such as token
   count, byte size, witness count, round count, or parent count.
5. **Provenance (C5).** Capsules form an acyclic DAG and are stored
   in a hash-chained ledger.
6. **Frozen state (C6).** Sealed capsules are immutable.

These invariants are not novel separately; what is novel is using the
same contract from runtime execution through team-level coordination
and through the benchmark family in which the scientific claims are
made.

## 4. System Overview: Wevra as a Capsule-Native Runtime

The Wevra runtime is the implementation substrate for the research.
At the single-run level, the runtime seals a chain of capsules
covering:

`PROFILE -> READINESS_CHECK -> SWEEP_SPEC -> SWEEP_CELL -> PROVENANCE -> ARTIFACT -> RUN_REPORT`

and then extends the capsule-native slice inward:

`PROMPT -> LLM_RESPONSE -> PARSE_OUTCOME -> PATCH_PROPOSAL -> TEST_VERDICT`

Key runtime results already established include:

- lifecycle/execution correspondence (W3-32, W3-39, W3-42..W3-45);
- content-addressing at artifact creation time (W3-33);
- mechanical lifecycle audit (W3-40, W3-45);
- deterministic replay at the full DAG level (W3-41);
- a sharp impossibility result for authenticating meta-artifacts
  inside the primary ledger, plus a detached witness construction
  (W3-36).

Those results matter here for two reasons. First, they give the team
layer a trustworthy execution substrate. Second, they provide typed
failure witnesses instead of opaque logs.

## 5. Team-Level Coordination Model

At the team layer, the main coordination objects are:

- `TEAM_HANDOFF`: a typed handoff from one role to another;
- `ROLE_VIEW`: the admitted local view for a role in one
  coordination step;
- `TEAM_DECISION`: the team-level decision capsule.

Lifecycle audit at the team level checks invariants T-1..T-7,
including sealing, parent-type constraints, budget caps, and routing
consistency. This layer gives us a clean separation between:

- what producers emit,
- what local views admit,
- what decoders infer,
- and what the final decision says.

That separation is what makes the rest of the paper possible.

## 6. An Eight-Axis Decomposition of the Context Problem

The strongest current contribution of the programme is not any one
benchmark. It is the decomposition below.

> *Reading note (SDK v3.17).* The original paper draft (SDK v3.14)
> named five structural axes; SDK v3.15 adds a sixth (W14
> producer-side ambiguity preservation), SDK v3.16 adds a seventh
> (W15 decoder-side capsule context packing), and SDK v3.17 adds
> an eighth (W16 end-to-end W14+W15 composition; this last is a
> *coupling* statement, not a new mechanism). The five-axis
> sub-decomposition (sections 6.1–6.5 below) is preserved
> verbatim as the SDK v3.14 reading; sections 6.6–6.8 below are
> the SDK v3.15-v3.17 additions; section 6.9 is the
> "thesis-after-v3.17" synthesis. The runtime contract remains
> byte-for-byte unchanged across all eight axes.

### 6.1 Axis 1: Admission under budget

Question: given a stream of candidate handoffs, which ones should a
bounded local view admit?

Positive results:

- W7-2: buffered cohort coherence strictly beats FIFO on a
  gold-plurality regime.
- W8-1: cross-role corroboration strictly beats single-tag plurality
  on a decoy-plurality regime.
- W9-1: multi-service corroboration strictly beats single-service
  corroboration on a multi-service-gold regime.

Negative results:

- W7-1: when the producer emits fewer candidates than the budget cap,
  FIFO is unbeatable by construction.
- W9-4: service-blind admission has a named structural limit when a
  decoy is also sufficiently corroborated.

### 6.2 Axis 2: Intra-round decoding

Question: once evidence is admitted in a single round, can decoding
over the admitted bundle resolve ambiguities admission cannot?

Positive result:

- W10-1: bundle-aware team decoding strictly beats every admission
  baseline on R-57.

Negative result:

- W10-Λ: service-blind admission is structurally insufficient on
  R-57.

### 6.3 Axis 3: Cross-round decoding

Question: when early evidence carries service inventory and later
evidence carries disambiguating causal structure, can decoding over
the union of rounds close a gap no single-round method can close?

Positive result:

- W11-1: multi-round bundle decoding strictly beats every single-
  round method on R-58.

Negative results:

- W11-Λ: single-round strategies are structurally insufficient on
  R-58.
- W11-4: round-level budget starvation is a sharp falsifier.

### 6.4 Axis 4: Fixed-vocabulary normalization

Question: if producers drift lexically but stay within a known
synonym closure, can a closed-vocabulary normalizer recover the
cross-round win?

Positive result:

- W12-1: fixed-table normalization plus cross-round decoding closes
  the synthetic-to-real-shaped gap on R-59.

Negative result:

- W12-4: finite closure is a sharp limit; OOV drift defeats the
  fixed table.

### 6.5 Axis 5: Layered open-world normalization

Question: can a layered normalizer widen the fixed-table closure
beyond exact lookup while preserving backward compatibility?

Positive result:

- W13-1: layered heuristic normalization strictly beats fixed-table
  normalization on R-60-wide.

Negative results:

- W13-4: cosmic OOV is a sharp closure boundary.
- W13-Λ-real: on the first real-Ollama probe, the bottleneck is not
  normalization but the producer collapsing ambiguity upstream.

### 6.6 Axis 6: Producer-side ambiguity preservation (SDK v3.15)

Question: when the real LLM producer compresses round-1 toward a
single best diagnosis (the W13-Λ-real observation), can a prompt-side
intervention restore the bench property's structural ingredients
*before* admission rather than after the fact?

Positive results:

- W14-1: pairing the ``StructuredProducerProtocol`` (per-event
  mandate + observation/diagnosis tier banner) with the cross-round
  capsule pipeline restores the bench property and achieves
  ``accuracy_full = 1.000`` on R-61-structured-prompt synthetic
  (vs 0.000 under the naive prompt; +1.000 strict separation,
  stable across 5/5 seeds).
- W14-Λ-real: on real Mac-1 ``qwen2.5:14b-32k`` at temperature 0
  on the redesigned comparable-magnitude events under the
  structured prompt, the bench property holds in 7/8 scenarios
  (vs 0/8 under naive); the cross-round pipeline achieves
  ``accuracy_full = 0.500`` (vs 0.000 under naive) — **+0.500
  strict gain over substrate FIFO on a real-LLM stream** at
  exactly the R-61-OLLAMA-A tier threshold.

Negative results:

- W14-Λ-prompt: under producer-side compression, every downstream
  layer (admission, decoder, normaliser) ties FIFO at 0.000.
- W14-4: both the comparable-magnitude event redesign AND the
  structured prompt are required; either alone is insufficient.

### 6.7 Axis 7: Decoder-side capsule context packing (SDK v3.16)

Question: when the cross-round bundle is bounded by a strict
``T_decoder`` token budget (e.g. an LLM context window), can
salience-aware capsule context packing keep the round-2 specific-
tier disambiguator that FIFO truncation would drop?

Positive result:

- W15-1: pairing the ``AttentionAwareBundleDecoder`` (closed-form
  salience: tier + CCK + corroboration + magnitude + round; with
  per-(tag, role, tier) hypothesis preservation) with the W13
  layered decoder achieves ``accuracy_full = 1.000`` on
  R-62-tightbudget while every FIFO-packed cross-round decoder
  collapses to 0.000 — +1.000 strict separation, stable across
  5/5 seeds. The packer puts the round-2 specific claim at rank 0
  of the kept bundle in 8/8 cells (the proxy attention metric).

Negative results:

- W15-Λ-budget: under FIFO arrival-order packing, ``T_decoder``
  below the union token sum drops the round-2 disambiguator by
  construction; extends the W7-3 extraction floor to the
  decoder-side axis.
- W15-Λ-degenerate: under no budget pressure, salience reordering
  is a no-op on the answer field.

### 6.8 Axis 8: End-to-end W14 + W15 composition (SDK v3.17)

Question: do the W14 producer-side and W15 decoder-side
interventions compose on a single regime where both are
independently load-bearing? And does the composition deliver a
strict end-to-end real-LLM gain over the strongest non-composed
baseline?

Positive results:

- W16-1: pairing the W14 ``StructuredProducerProtocol`` with the
  W15 ``AttentionAwareBundleDecoder`` simultaneously achieves
  ``accuracy_full = 1.000`` on R-63-COMPOSED-TIGHT (mag-filter
  structured prompt + ``T_decoder = 24``) while every non-composed
  baseline collapses to 0.000 — **+1.000 strict separation,
  stable across 5/5 seeds**. The composition is *additive in
  code* (no new SDK class) and *multiplicative in effect* (each
  layer is necessary on the regime where the other layer's limit
  fires).
- W16-Λ-real-replay: on the recorded Phase-61 ``qwen2.5:14b-32k``
  bytes (n=8 × 24 producer calls, byte-stable) at
  ``T_decoder = 14``, the composed pipeline achieves
  ``capsule_attention_aware = 0.500`` while
  ``capsule_layered_fifo_packed = 0.000`` — **+0.500 strict gain
  over the FIFO-packed-W14-only baseline on a real-LLM stream**.
  The first end-to-end real-LLM strict advance over the strongest
  non-composed baseline in the programme.
- W16-2: each layer alone produces 0.000 on R-63-COMPOSED-TIGHT
  (W14-only-budgeted FIFO-pack: 0.000; W15-only-without-W14
  attention-aware over the naive-prompt stream: 0.000); only the
  composition produces 1.000. The composition is sub-additive at
  the failure-mode level — the named limits compose
  multiplicatively.

Negative results:

- W16-Λ-compose: on R-63-naive-tight (mag-filter naive prompt +
  ``T_decoder = 24``), W14-Λ-prompt and W15-Λ-budget fire
  simultaneously; every capsule strategy ties FIFO at 0.000.
- W16-Λ-degenerate: under ``T_decoder = 2`` (below the round-2
  disambiguator's token cost), both packers collapse; the W16-1
  win is conditional on a budget that admits *some* of the union
  but not *all* of it.

### 6.9 Synthesis after SDK v3.17

The eight axes collectively support a sharper thesis than the
five-axis reading of SDK v3.14:

1. **Producer-side ambiguity preservation** (W14, axis 6) is
   load-bearing on real-LLM regimes where the model's natural
   compression tendency erases the bench property upstream.
2. **Decoder-side capsule context packing** (W15, axis 7) is
   load-bearing when the auditor's downstream consumer is bounded
   by a strict token budget.
3. **The composition** (W16, axis 8) is load-bearing on regimes
   where both axes 6 and 7 fire simultaneously — the regime that
   most resembles a *production* multi-agent system with both a
   real LLM producer and a context-window-bounded downstream
   audit agent.
4. The *runtime contract is unchanged* across all eight axes: each
   axis is a research-grade SDK extension on the team-coordination
   surface, opt-in only.

The honest reading after SDK v3.17 is that multi-agent context
becomes tractable when the runtime separates these eight axes
*and* couples them on the regimes where the coupling is required.
The W16 result is the first end-to-end real-LLM evidence that
the composition holds in practice, not just in theory.

## 7. Benchmark Family

The benchmark family is intentionally cumulative. Each regime was
designed only after the prior regime exposed a specific failure mode.

| Regime | Main question | Winning axis | Named limit |
| --- | --- | --- | --- |
| R-53 | Is FIFO already unbeatable under low surplus? | None; fixed policies tie | W7-1 / W6 reading |
| R-54 | Does buffered coherence beat FIFO when gold has strict plurality? | Admission | Streaming instability |
| R-55 | Does cross-role corroboration beat raw plurality? | Admission | Corroborated-decoy falsifier |
| R-56 | Does multi-service corroboration beat single-tag corroboration? | Admission | W9-4 |
| R-57 | Can decoding beat every admission policy? | Intra-round decoding | W10-4 |
| R-58 | Can cross-round reasoning beat every single-round method? | Cross-round decoding | W11-4 |
| R-59 | Does synthetic-to-real-shaped producer drift require normalization? | Fixed-table normalization | W12-4 |
| R-60 | Is fixed-table normalization enough under open-world drift? | Layered normalization | W13-4 / W13-Λ-real |

This progression is one of the paper's main strengths. The later
benchmarks do not overwrite the earlier ones; they explain them.

## 8. Experimental Methodology

### 8.1 Evaluation discipline

The programme uses pre-committed benchmark properties, named
falsifiers, and cross-regime evaluation. A result is considered
material only when:

- the benchmark property is mechanically checked;
- the method clears a pre-committed success bar;
- at least one falsifier regime is named and confirmed;
- regression on earlier anchor regimes is bounded or absent;
- lifecycle audit passes on every evaluated cell.

This is encoded explicitly in
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`.

### 8.2 Metrics

The main metrics are:

- `accuracy_full`: exact task correctness;
- `accuracy_services`: correctness of the service set where
  applicable;
- `accuracy_root_cause`: correctness of the root-cause label;
- audit pass rate;
- budget usage / admitted count;
- cross-seed or cross-bank stability;
- normalization statistics and OOV counts for the later regimes.

### 8.3 Deterministic vs real-LLM settings

Most of the strongest positive results so far are on deterministic
synthetic or synthetic-real-shaped regimes. Real-model evidence is
used carefully and honestly:

- R-53 provides real-LLM evidence for producer effects in a lower-
  surplus team setting.
- W5-1 provides real cross-model parser-boundary evidence on
  single-run parsing.
- R-60-ollama is the first real multi-round probe for the current
  line of work, and it is currently a null result for the intended
  benchmark property.

## 9. Main Results

### 9.1 Runtime results

The runtime side of the story is already stronger than a typical
agent harness:

- execution is capsule-native through the LLM byte boundary;
- artifacts are content-addressed at write time;
- lifecycle correctness is mechanically audited;
- deterministic replay exists at the DAG level;
- meta-artifact authentication has a sharp impossibility theorem and
  a constructive workaround.

This matters because it turns research claims into executable,
auditable artifacts rather than post-hoc narratives.

### 9.2 Admission results

The admission story improves in steps:

- **R-54 / W7-2:** buffered cohort coherence wins where FIFO and
  single-pass streaming fail.
- **R-55 / W8-1:** cross-role corroboration wins where raw plurality
  fails.
- **R-56 / W9-1:** multi-service corroboration wins where
  single-service corroboration fails.

The admission-side thesis is therefore real but conditional:
admission can solve some hard context problems, but not all.

### 9.3 Decoder results

The decoder story begins only once admission has hit a named wall.

- **R-57 / W10-1:** bundle-aware decoding beats every admission-only
  policy.
- **R-58 / W11-1:** multi-round decoding beats every single-round
  method.

This is the first point where the programme can say something much
stronger than "better filtering helps." The downstream meaning of a
bundle matters, and sometimes later evidence must reinterpret earlier
evidence.

### 9.4 Normalization results

The transfer story then adds another layer:

- **R-59 / W12-1:** fixed-table normalization is sufficient under a
  bounded synthetic-real-shaped noise channel.
- **R-60-wide / W13-1:** layered heuristic normalization strictly
  widens the closure beyond fixed-table normalization.
- **R-60-cosmic / W13-4:** there is no free lunch; open-world
  normalization has a sharp closure boundary.

### 9.5 Real-Ollama result

The first real-Ollama probe is an honest negative:

- the model emits canonical kinds;
- it filters low-magnitude decoy events;
- the intended ambiguity does not survive production;
- therefore the downstream normalization advantage is structurally
  invisible on that probe.

This is one of the most important results in the paper, precisely
because it prevents the wrong next move. The issue is not "add more
synonyms." The issue is that producer-side ambiguity preservation has
become the next dominant bottleneck.

## 10. Theorem-Level Synthesis

The strongest current programme-level synthesis can be stated as
follows.

### 10.1 Positive claim

For a family of multi-agent coordination tasks with:

- typed role-local evidence,
- bounded role-local budgets,
- delayed or distributed disambiguating information,
- and producer drift that remains inside a controlled closure,

capsule-native coordination provides a stack of constructive methods
that progressively solve broader subclasses of the context problem:
admission, intra-round decoding, cross-round decoding, and
normalization.

### 10.2 Negative claim

No single axis is sufficient in general:

- FIFO can be unbeatable when surplus is absent.
- Admission-only policies fail under corroborated-decoy ambiguity.
- Single-round decoders fail under delayed evidence.
- Fixed-table normalizers fail under open-world drift.
- Even layered normalizers can become irrelevant if the producer
  collapses the ambiguity upstream.

### 10.3 Strongest current thesis

The strongest defensible current thesis is therefore:

> **Solving multi-agent context requires an explicitly layered
> architecture in which producer ambiguity preservation,
> normalization, admission, intra-round decoding, and cross-round
> decoding are all first-class and separately testable.**

This is materially stronger than anything the programme could claim at
SDK v3.7, v3.8, or even v3.10.

## 11. Why This Matters Beyond the Benchmarks

The broader significance of the work is not that one benchmark went
from 0.0 to 1.0. It is that the programme has made the context
problem *structurable*.

Without this decomposition, failures in multi-agent LLM systems tend
to be diagnosed vaguely:

- "the model was confused,"
- "the prompt was weak,"
- "the retriever missed something,"
- "the context window was too small."

With the capsule-native framework, the failure can often be stated
more sharply:

- the producer failed to preserve ambiguity;
- the admitted view lacked sufficient corroboration;
- the decoder was single-round when the regime required cross-round
  reasoning;
- the normalizer's closure was too narrow;
- or the benchmark property did not survive the real producer.

That is what makes this line of work publishable. It is not just an
engineering artifact; it is a progressively sharpened theory of where
context failure comes from in team-based LLM systems.

## 12. Limitations

The paper has several important limitations.

1. **Deterministic benches still dominate the strongest wins.** The
   strongest positive results remain synthetic or synthetic-real-
   shaped.
2. **The real-Ollama story is still early.** The current real probe
   is informative but small.
3. **The semantics are still closed-world.** Even the layered
   normalizer is hand-engineered and finite.
4. **The current tasks are narrow incident-style coordination tasks.**
   Cross-domain transfer remains an open problem at the team layer.
5. **The product boundary remains narrower than the research claim.**
   The research layer is ahead of what should be marketed as stable
   SDK surface.

These are not flaws to hide; they are the next publication agenda.

## 13. Related Work

This work sits at the intersection of several literatures:

- content-addressed and Merkle-style object systems;
- event-sourcing and tamper-evident ledgers;
- bounded-context and exact-memory systems for agents;
- multi-agent coordination and role-specialized reasoning;
- retrieval and memory systems for LLMs;
- structured or tool-mediated prompting;
- and programmatic evaluation harnesses for LLM-based software tasks.

What distinguishes this programme from a generic memory or agent
framework is the combination of:

- a single capsule contract across runtime and team coordination,
- a theorem/benchmark discipline with named falsifiers,
- and a decomposition of context into the five structural axes above.

The paper should eventually include a conventional bibliography and
positioning against adjacent systems papers, memory papers, and
multi-agent reasoning papers. This draft leaves those citations to
the final submission pass rather than fabricating them loosely here.

## 14. Discussion: What Would Count as "Solving Context"?

The phrase is easy to misuse. In this repo, a serious claim would
need at least the following:

1. wins that survive multiple benchmark families rather than one
   custom regime;
2. wins that survive real producers, not only synthetic ones;
3. explicit evidence that producer-side ambiguity preservation is
   addressed rather than ignored;
4. robustness to some open-world drift without hand-curated closure
   for every case;
5. and a bounded, honest statement of where the solution does not
   apply.

The programme is not there yet. But it is also no longer at the
"interesting intuition" stage. It now has:

- runtime theorems,
- coordination theorems,
- multiple structural ceilings,
- multiple constructive separations,
- and the first real evidence that producer-side behavior can erase
  downstream context difficulty.

That is enough to justify a main paper.

## 15. Conclusion

This paper has one central message: **context in multi-agent LLM
systems is best understood as an object-level coordination problem,
not a prompt-length problem.** Capsules provide the object model.
Wevra provides the executable runtime. The benchmark family R-53
through R-60 provides a falsifiable research program rather than a
single story.

The strongest current result is not "we solved context." It is this:

- admission helps in some regimes and has a named ceiling;
- decoding helps beyond that and has a named ceiling;
- cross-round reasoning helps beyond that and has a named ceiling;
- normalization helps beyond that and has a named ceiling;
- and real producers can move the bottleneck upstream.

That decomposition is the real scientific contribution. It turns
"context" from a vague complaint into a sequence of explicit,
researchable, and partially solved subproblems.

## Appendix A. Current Claim Taxonomy

The programme now uses an explicit taxonomy:

- **proved**: proof or proof by inspection;
- **proved-conditional**: proof under stated assumptions;
- **mechanically-checked**: checked by runtime audit or contract
  tests;
- **empirical**: measured on a published bench and seed;
- **conjectural**: stated with a named falsifier;
- **retracted**: earlier reading withdrawn.

This taxonomy is not cosmetic. It is one of the main reasons the
paper can make strong claims without collapsing into hype.

## Appendix B. Recommended Figure / Table Package for Submission

This Markdown draft is text-first. A submission version should likely
include:

1. a system diagram showing the capsule-native runtime and team layer;
2. a five-axis diagram showing producer -> normalization -> admission
   -> intra-round decoding -> cross-round decoding;
3. one summary table for R-53 through R-60;
4. one theorem/limit table;
5. one real-vs-synthetic transfer table centered on W12 and W13.

## References

Bibliography intentionally omitted from this draft file.
Before submission, replace this section with a real BibTeX-backed
bibliography covering:

- content-addressed storage / Merkle DAG systems,
- event-sourcing / tamper-evident logging,
- bounded-context / exact-memory agent systems,
- multi-agent coordination and blackboard-style architectures,
- retrieval-augmented generation and memory for LLMs,
- software-engineering agent benchmarks and orchestration systems.
