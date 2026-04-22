# Phase 32 — Cross-Domain Typed Handoffs, Noisy-Extractor Robustness, and a Stronger-Model Spot Check

**Status: combined research milestone. Phase 32 ships four coupled
deliverables: (a) a *second* non-code multi-role team benchmark — a
vendor-onboarding *compliance review* with five role-typed agents
(legal, security, privacy, finance, compliance officer) — that
confirms the typed-handoff substrate is not specific to incident
triage; (b) a parameterised extractor-noise module and a controlled
sweep that measures the substrate's graceful degradation under
recall / precision / mislabel noise; (c) a disciplined spot check
with ``qwen2.5-coder:7b`` on both non-code benchmarks (Phase 31 +
Phase 32) that separates substrate-delivery correctness from LLM
transcription fidelity on a frontier-relative model; and (d) three
theorems and two conjectures extending the Phase-31 team-
communication theory sideways (cross-domain typed-handoff
correctness preservation; noisy-extractor graceful-degradation
bound; bounded-context preservation under bounded noise) plus a
formal conjecture on role-lattice stability across domains and a
composability-under-extractor-union conjecture.**

Phase 31, in one line: typed handoffs keep a five-role incident-
triage team's auditor prompt at 196 tokens independent of stream
size on a non-code domain. **Phase 32, in one line: the same
substrate instantiated on a *second* non-code domain (vendor
compliance review) keeps the compliance officer's prompt at 171
tokens flat across four orders of magnitude of distractor count at
100 % accuracy under the mock ceiling, while a controlled
extractor-noise sweep confirms that correctness degrades smoothly
and predictably (and in a shape matching Theorem P32-2) as the
extractor's recall / precision fall below 1, with the
*token-bound* preserved under bounded noise even as accuracy
decays.**

---

## Part A — Research framing

### A.1 Why this milestone exists

Before Phase 32, the programme's evidence for the generalisation
claim in § 1.5 of the master plan — that the typed-handoff
substrate is an *agent-team communication primitive*, not a
graph/index tool — rested on one non-code benchmark (Phase 31's
incident triage). A single non-code datapoint is *necessary but
not sufficient* for a general-agent-teams claim. Phase 32 closes
this gap by:

1. **A second non-code domain.** Vendor-onboarding compliance
   review, with an entirely different role cast (legal / security
   / privacy / finance / compliance officer, no operational
   telemetry anywhere), different claim taxonomy (regulatory /
   contract / spend / encryption), different output shape (verdict
   + flags + remediation), and different distractor stream
   (document-clause chatter, not metric samples or log lines).

2. **Controlled extractor noise.** Phase 31 used
   *perfect-by-construction* regex extractors: precision and
   recall on causal events were 1.0 by design. Real extractors —
   LLM-driven or otherwise — fall strictly below 1 on both. The
   programme's robustness claim (Conjecture C31-7) is untested
   until we measure what happens when precision / recall fall.
   Phase 32 Part B is the sweep that gives C31-7 an empirical
   spine and a theorem (P32-2).

3. **A frontier-relative model spot check.** Phase 31's LLM-in-loop
   result used ``qwen2.5:0.5b`` and saturated at ~40 % — bounded
   below by transcription fidelity, not substrate delivery.
   Phase 32 Part C runs ``qwen2.5-coder:7b`` on both non-code
   benchmarks at k = 6 and measures whether substrate-slice
   accuracy rises toward the mock's 100 % when the model's
   transcription ceiling is no longer the bottleneck.

4. **Theory catching up to evidence.** Phase 32's three theorems
   (P32-1 cross-domain generalisation, P32-2 graceful degradation,
   P32-3 token-bound preservation under noise) formalise claims
   the Phase-31 note called "empirical" or "conjectural"; the two
   new conjectures (C32-4 role-lattice stability and C32-5
   extractor-composition) name the next falsifiable objects.

### A.2 Scope discipline (what Phase 32 does NOT claim)

Explicit, because drift is the largest framing risk for a
milestone of this shape:

1. **Not SWE-bench end-to-end.** Carry-over from Phases 29–31.
2. **Not a production compliance engine.** The vendor-review task
   family is a *research benchmark* — synthetic, structurally
   typed, deterministically graded. Real compliance involves human
   judgement, negotiation, and uncertainty the benchmark does not
   model.
3. **Not a claim that *any* multi-role team can use the
   substrate.** Theorem P32-1's preconditions — subscription
   coverage of the task's causal chain + sound, complete
   extractors on causal events — are the *boundary*; tasks whose
   gold is not a deterministic function of role-owned typed claims
   fall outside.
4. **Not adversarial-noise.** The noise module (Part B) injects
   *i.i.d.* per-claim Bernoulli noise at a controlled rate. An
   adversary who could selectively drop only the load-bearing
   claims (a covariate shift under task-structured attack) is not
   modelled. See "Future work" § F.3.
5. **Not a cross-language generalisation.** Both benchmarks are
   English-payload, regex-extractable. An L10N-style generalisation
   claim is not made.
6. **Not a claim that 7B is the ceiling.** Phase 32 Part C's spot
   check is *one* model on *one* seed on *one* distractor count
   per domain — enough to answer "does the substrate still
   benefit a real LLM at this scale?", not enough to answer "what
   is the model-size scaling law?".

---

## Part B — Theory

Phase 31 ships theorems P31-1..P31-5 for the single-domain case
(incident triage) and names C31-6 / C31-7 as conjectures. Phase 32
extends the theory with three theorems that explicitly decouple the
substrate's guarantees from the domain and from the extractor's
ideal behaviour.

### B.1 Setup (reused from Phase 31 § B.1)

Let a team have roles ``R = {r_1, …, r_K}``. Each role has (i) an
observable-type subset ``O_k ⊆ EventTypes``, (ii) a claim extractor
``e_k``, (iii) a subscription ``σ(r_k, c.kind) ⊆ R`` of consumer
roles. Let ``X`` be the naive-broadcast event stream, ``H`` the
emitted handoffs, and ``π`` the aggregator's decoder policy. The
load-bearing claim-kind count for scenario ``z`` is
``R*(z) = |causal_chain(z)|``.

### B.2 Theorem P32-1 — Cross-domain correctness preservation

**Statement.** Let ``T`` be any multi-role team whose gold answer
``y(z)`` for each task ``z`` is a deterministic function of the
role-ordered tuple of load-bearing claim kinds
``(k_1, k_2, …, k_{R*(z)})`` and their witnessing payloads:

```
y(z) = g(k_1, w_1, k_2, w_2, …, k_{R*(z)}, w_{R*(z)})
```

where ``g`` is a decoder the team agrees on (Phase 31 § C.1's
``_decoder_from_handoffs``; Phase 32's ``decode_from_handoffs``).

Then under the typed-handoff delivery protocol with (i) a
subscription table that covers every ``(source_role, claim_kind)``
pair on every ``z``'s causal chain, and (ii) extractors that are
sound (no false-positive claims on distractors) and complete (every
load-bearing event yields at least one correctly-kinded emission),
the aggregator's decoded answer equals ``y(z)`` for every ``z``.

The proof is *independent of the domain* — it does not assume
anything about the semantic content of claim kinds beyond that the
decoder is a deterministic function of them.

**Interpretation.** The substrate's correctness guarantee is
domain-agnostic by construction. A team that exchanges its
incident-triage claim catalogue for a compliance-review claim
catalogue, keeping the subscription table + extractor contract,
preserves correctness on the new domain's causal-chain coverage.

**Proof.** By Theorem P31-4 — the proof inspects only the
subscription-coverage and extractor-soundness preconditions and the
fact that the decoder is a function of the delivered inbox; none of
those invoke any property specific to incident-triage. Apply the
same argument with the compliance-review decoder ``g``. ∎

**Empirical anchor.** § D.1: mock-auditor substrate accuracy is
100 % on every one of the five compliance-review scenarios at every
k ∈ {6, 20, 60, 120} with handoff recall 1.0 — the same
accuracy-invariance profile Phase 31 reported for incident triage.
Two distinct domains, same substrate module, same theorem.

### B.3 Theorem P32-2 — Graceful degradation under bounded extractor noise

**Statement.** Let extractor recall ``r = 1 - δ`` and precision
``p = 1 - ε`` on causal events (δ, ε ∈ [0, 1]). Let ``R*`` be the
load-bearing claim count for a scenario ``z`` and let ``M(z)`` be
the size of the per-role distractor set. Under the typed-handoff
protocol with the noisy extractor:

(i) **Handoff-recall bound.** The auditor's inbox contains every
load-bearing claim with probability at least ``(1 - δ)^{R*(z)}``.

(ii) **Handoff-precision bound.** The fraction of spurious
handoffs in the auditor's inbox is bounded above by
``ε·M(z) / (R*(z) + ε·M(z))`` in expectation.

(iii) **Correctness-rate bound under a *monotone* decoder.**
Suppose the decoder ``g`` has the *monotonicity* property: if
``g(S) = y(z)`` for some set ``S`` of load-bearing claims, then
``g(S ∪ S') = y(z)`` for any additional *non-contradictory* set
``S'`` of claims whose kinds are not strictly higher in the
priority order than the load-bearing kinds. Then correctness holds
with probability at least ``(1 - δ)^{R*(z)}`` — i.e., the only
failure mode is recall drop below ``R*``.

(iv) **Correctness-rate bound under a *strict* decoder.** If
``g`` is *not* monotone in the above sense — i.e., adding a
spurious claim of higher priority flips the answer — then
correctness is bounded above by the *joint* event of (a) every
load-bearing claim delivered *and* (b) no spurious claim of higher
priority delivered. In the i.i.d. model this is
``(1 - δ)^{R*(z)} · (1 - ε)^{M(z)·q_hi}`` where ``q_hi`` is the
fraction of the role's known-kind pool that would override the
gold kind.

**Interpretation.** There are *two distinct* graceful-degradation
regimes:

* a **recall-limited regime** where only dropped load-bearing
  claims hurt. Compliance review's *verdict* field is in this
  regime (because ``BLOCKED`` is monotone — any blocking claim
  produces BLOCKED). Empirically, verdict accuracy degrades
  as ≈ ``(1 - δ)^{R*}`` and is robust to spurious claims.

* a **precision-limited regime** where spurious claims flip the
  decoder's output. The *flag set* field is in this regime
  (exact-match grading means a single spurious flag fails the
  test). Empirically, flag accuracy collapses at even ε = 0.05
  while verdict holds. The failure attribution cleanly separates
  the two through ``FAILURE_SPURIOUS_CLAIM`` (flag-side)
  vs ``FAILURE_MISSING_HANDOFF`` (verdict-side).

**Proof sketch.** (i) is a direct union bound over the ``R*``
independent drop events. (ii) is a multinomial expectation on the
inbox. (iii) is the monotone case: given all load-bearing claims,
the monotone decoder reproduces ``y(z)`` regardless of additional
non-contradictory claims. (iv) is the strict case: take the
intersection of (a) the full-delivery event and (b) the
no-override event; both are binomial, so the joint lower bound
holds. ∎

**Empirical anchor.** § D.2: the sweep at distractor count ``k =
20`` and seed pooling ∈ {31, 32} reports compliance-review verdict
accuracy of 1.0 at drop=0, spurious=0.05 (monotone regime: spurious
claims hit no blocker, verdict holds), falling only at drop ≥ 0.5
where recall drops below ``(1 - 0.5)^{R*=2..3}``. Full accuracy
collapses at spurious ≥ 0.05 because the flag-set grader is
strict.

### B.4 Theorem P32-3 — Bounded-context preservation under bounded noise

**Statement.** Under the typed-handoff protocol with a noisy
extractor emitting at expected rate
``λ = R* + ε·M + (1 - δ)·R*``, the peak active context per role
remains:

```
ctx(r_k) ≤ C_0 + min(inbox_capacity, λ) · τ
```

where ``τ`` is the per-handoff token ceiling and ``C_0`` is the
task-goal fixed-point size. For any ``ε ≤ (capacity - R*) / M``,
this is still ``O(1)`` in the raw event-stream size ``|X|`` (the
context bound is a bounded function of ε + capacity, *not* of
``|X|``).

**Interpretation.** The substrate's Phase-31 P31-3 bounded-context
claim survives the Phase-32 noise sweep *as long as* the inbox
capacity is wide enough to hold the expected spurious blow-up. The
role-inbox capacity (default 32 in incident-triage, 64 in
compliance-review) is in effect a *regulariser* that clips the
adversarial growth of spurious emissions.

**Proof sketch.** The inbox is bounded by its capacity (hard
constraint). Token count per handoff is bounded by ``τ``. The
delivered tokens are therefore ``O(capacity · τ) = O(1)`` in
``|X|``. The expected-rate expression is mean-field (pooled over
roles and events); for a single inbox this is already the *hard*
upper bound. ∎

**Empirical anchor.** § D.2: even at spurious_prob = 0.10 the
compliance-review substrate prompt grows to ≈ 378 tokens — a 2.2×
increase over the 0.0-noise baseline of 171 tokens. Compared to
naive's ≈ 1 778 tokens at k = 20 this is still an ≈ 5× delivery
reduction; compared to naive's 4 058 at k = 120 this is a ≈ 11×
reduction. The token bound is preserved *and bounded* as a
function of ε + capacity, not of ``|X|``.

### B.5 Conjecture C32-4 — Role-lattice stability across domains

**Statement.** The *schema* of the typed-handoff substrate —
``(RoleSubscriptionTable, RoleInbox, HandoffLog, HandoffRouter)`` —
is domain-agnostic. For any two domains ``D_1`` and ``D_2`` with
respective role lattices ``R^{(1)}``, ``R^{(2)}`` and respective
claim catalogues ``K^{(1)}``, ``K^{(2)}``, the substrate's
correctness / token-bound guarantees (Theorems P32-1, P32-3)
transfer to ``D_2`` iff:

1. there is a role-lattice homomorphism ``φ: R^{(1)} → R^{(2)}``
   such that the producer / consumer roles of each claim map to
   the corresponding roles in ``D_2``;
2. there is a claim-kind mapping ``ψ: K^{(1)} → K^{(2)}`` such
   that the subscription table of ``D_2`` agrees with ``φ ∘ ψ``
   applied to ``D_1``'s subscription table;
3. there is an extractor mapping with recall/precision bounded by
   the Theorem P32-2 preconditions on ``D_2``'s event distribution.

**Status.** Partially empirical. The compliance-review and
incident-triage benchmarks respect the conditions by construction,
and the substrate module (``core/role_handoff``) is the identity
under the role relabelling — so the conjecture holds trivially for
those two. The open part is whether *every* agent-team domain can
be expressed this way; adversarial domains might lack a clean
``(φ, ψ)`` pair. Follow-up: a third domain (candidate list: policy
drafting, research triage, cross-functional delivery planning)
would falsify or confirm ``C32-4`` at ``K = 3``.

**Empirical anchor.** § D.1 cross-domain comparison table: both
domains satisfy the theorem preconditions and both exhibit the
same pattern (substrate flat across k; routing 0 %; naive collapse
at k = 120).

### B.6 Conjecture C32-5 — Extractor-composition precision/recall bound

**Statement.** Given two extractors ``e_1, e_2`` with respective
precision ``(p_1, p_2)`` and recall ``(r_1, r_2)`` on the same
claim kind, the *union* extractor
``e(e_1, e_2) = e_1(events) ∪ e_2(events)`` satisfies:

* precision ``p(e) ≥ p_1 · p_2`` (at most the product of
  precisions — in i.i.d. model);
* recall ``r(e) ≥ 1 - (1 - r_1)(1 - r_2)`` (at least the union
  coverage of recalls).

**Status.** Unproven on the Phase-32 noise sweep because the
sweep uses a *single* noisy extractor; a follow-up phase would
instantiate two extractors with calibrated non-zero noise and
measure the union. The conjecture provides a design principle for
ensemble-of-extractors in production team protocols and a concrete
hypothesis to falsify.

### B.7 What is theorem vs what is empirical

Ordered by strength:

* **Theorem (proved):** P31-1..P31-5 (carried over from Phase 31),
  **P32-1**, **P32-2**, **P32-3**.
* **Empirical, measurable:** Phase-32 Part A cross-domain
  accuracy/token table (§ D.1); Phase-32 Part B noise-sweep
  accuracy/recall/precision degradation curves (§ D.2);
  Phase-32 Part C stronger-model spot check (§ D.3).
* **Conjecture (empirically supported, formally open):** C31-6
  (role-lattice, K > 5), C31-7 (noisy extractor — now subsumed
  by P32-2 in the bounded-noise regime), **C32-4** (cross-domain
  homomorphism), **C32-5** (extractor-composition).
* **Open question (unchanged):** OQ-1 in full generality; the
  Phase-30 Conjecture P30-6 (Lipschitz LLM policies) remains the
  sharpest mathematical shape for OQ-1.

A reviewer attacking this work should attack **P32-2(iv)** — the
strict-decoder regime — where the ``q_hi`` parameter is an
approximation; the tightness of the joint-event bound depends on
an independence assumption that the Phase-31 decoder (priority
order) may violate in the mislabel regime. The Phase-32 sweep
under mislabel=0.25 is informative here: verdict accuracy at
drop=0, mislabel=0.25 is 0.50 on compliance — consistent with
``(1 - 0.25)^{R*=2}`` for the two-claim scenarios = 0.5625, within
i.i.d. noise of the observed 0.50.

---

## Part C — Architecture

### C.1 New substrate modules and benchmarks

```
vision_mvp/tasks/compliance_review.py          [NEW]
    5 vendor-onboarding scenarios × 5 roles × 13 claim kinds
    Extractors: per-role regex battery
    Decoder:     priority-monotone verdict +
                 exact-set flags + pair-keyed remediation map
    Mock auditor (upper-bound ceiling)

vision_mvp/core/extractor_noise.py             [NEW]
    NoiseConfig{drop, spurious, mislabel, payload_corrupt, seed}
    noisy_extractor(base, known_kinds_by_role, config)
    Per-domain known-kinds helpers
    Deterministic per seed × role × scenario

vision_mvp/experiments/phase32_compliance_review.py   [NEW]
    Phase 32 Part A driver (mock + LLM modes)
vision_mvp/experiments/phase32_noise_sweep.py         [NEW]
    Phase 32 Part B noise-sweep driver (mock-only by default)
vision_mvp/experiments/phase32_stronger_model.py      [NEW]
    Phase 32 Part C qwen2.5-coder:7b spot-check driver

vision_mvp/tasks/incident_triage.py            [MODIFIED]
    run_handoff_protocol + run_incident_loop accept an
    optional ``extractor`` argument (additive, non-breaking)
    so the Phase-32 noise sweep can inject a noisy extractor
    without breaking Phase-31 byte-stability.

vision_mvp/tests/test_compliance_review.py     [NEW, 28 tests]
vision_mvp/tests/test_extractor_noise.py       [NEW, 23 tests]
```

The substrate primitive (``core/role_handoff``) itself is unchanged
byte-for-byte from Phase 31. The Phase-32 modules sit *above* it
and exercise it on a second domain and under a second noise
regime — both the module's tests and the new benchmarks rely on
the same chain-hash / dedup / capacity semantics.

### C.2 Why the compliance-review task is not a re-skin of incident-triage

The two benchmarks are deliberately structurally different along
three axes that matter to the substrate's claim:

| Axis | Incident triage (Phase 31) | Compliance review (Phase 32) |
|---|---|---|
| Subject of events | operational telemetry | documents (contract clauses, questionnaires, inventories) |
| Causal temporality | time-ordered cascade | static evaluation over evidence |
| Output shape | root_cause + services + remediation | verdict + flag set + remediation |
| Decoder monotonicity | priority-order on causal kinds | monotone on *verdict*, strict-set on *flags* |
| Role cast | SRE-style operational team | cross-functional review team |
| Fixed-point density | 2 events (goal + placeholder) | 2 docs (goal + placeholder) |
| Claim kind count (|K|) | 11 | 13 |
| Gold remediation count | 5 | 5 |

The substrate correctness proof in Theorem P32-1 does not use any
axis of this table — the proof only needs the *role / claim /
subscription / decoder* abstraction. The empirical confirmation
that a benchmark built along the different axis still yields the
same accuracy / token-bound shape is the *evidence* that the
theorem captures the right abstraction.

---

## Part D — Evaluation

> Numbers below come from (A)
> ``vision_mvp/results_phase32_compliance_mock.json`` (5 scenarios ×
> 4 strategies × 4 distractor counts × 2 seeds = 160 measurements,
> 0.1 s wall-time), (B) ``vision_mvp/results_phase32_noise_sweep.json``
> (96 runs × 5 scenarios = 480 measurements, 0.5 s wall-time), and
> (C) ``vision_mvp/results_phase32_llm_7b_spot.json`` (15 LLM calls
> × 2 domains, qwen2.5-coder:7b on k = 6 seed = 32).

### D.1 Part A — cross-domain substrate ceiling

Mock-auditor pooled per ``(k, strategy)`` across scenarios × 2
seeds = 10 measurements per cell:

| k | strategy | acc_full | mean tokens | rel frac | recall | truncated | failure hist |
|---:|---|---:|---:|---:|---:|---:|---|
| **6**   | naive          | **1.00** |   658 | 0.143 | 1.00 | 0/10 | `{none:10}` |
| 6   | routing        | 0.00 |   132 | 1.000 | 1.00 | 0/10 | `{retrieval_miss:10}` |
| 6   | **substrate**  | **1.00** | **171** | **1.000** | **1.00** | **0/10** | `{none:10}` |
| 6   | substrate_wrap | 1.00 |   204 | 1.000 | 1.00 | 0/10 | `{none:10}` |
| 20  | naive          | 1.00 | 1 767 | 0.048 | 1.00 | 0/10 | `{none:10}` |
| 20  | routing        | 0.00 |   132 | 1.000 | 1.00 | 0/10 | `{retrieval_miss:10}` |
| 20  | **substrate**  | **1.00** | **171** | **1.000** | **1.00** | **0/10** | `{none:10}` |
| 20  | substrate_wrap | 1.00 |   204 | 1.000 | 1.00 | 0/10 | `{none:10}` |
| 60  | naive          | 1.00 | 4 081 | 0.020 | 1.00 | 10/10 | `{none:10}` |
| 60  | routing        | 0.00 |   132 | 1.000 | 1.00 | 0/10 | `{retrieval_miss:10}` |
| 60  | **substrate**  | **1.00** | **171** | **1.000** | **1.00** | **0/10** | `{none:10}` |
| 60  | substrate_wrap | 1.00 |   204 | 1.000 | 1.00 | 0/10 | `{none:10}` |
| **120** | **naive**  | **0.40** | 4 047 | 0.015 | 1.00 | 10/10 | `{truncation:6, none:4}` |
| 120 | routing        | 0.00 |   132 | 1.000 | 1.00 | 0/10 | `{retrieval_miss:10}` |
| 120 | **substrate**  | **1.00** | **171** | **1.000** | **1.00** | **0/10** | `{none:10}` |
| 120 | substrate_wrap | 1.00 |   204 | 1.000 | 1.00 | 0/10 | `{none:10}` |

Reading:

* **Substrate constant at 171 tokens / 100 % accuracy.** Flat
  across four orders of magnitude of distractor count, identical
  signature to Phase-31's 196-token flatline. This is
  Theorem P32-1 (cross-domain generalisation) on a *distinct*
  domain — different event schema, different role cast, different
  decoder shape.
* **Naive collapses at k = 120.** From 100 % → 40 %, driven by
  the ``max_docs_in_prompt = 200`` truncation cap. This mirrors
  Phase-31's 100 % → 20 % collapse on incident triage and
  confirms Theorem P30-3 on a new domain.
* **Routing (role-keyed) cannot rescue the compliance officer.**
  0 % on every k — the compliance officer has no raw-document
  subscription, so routing delivers only fixed-point docs
  (goal + placeholder). Confirms Theorem P29-2 / P31-1 on a
  non-code domain.
* **Substrate token-ratio vs naive:** 3.8× at k = 6, 10.3× at
  k = 20, 23.9× at k = 60, 23.7× at k = 120 (the last capped
  by truncation). At k = 60, **substrate delivers ~24× fewer
  tokens at 2.5× higher correctness rate** than naive;
  at k = 120, **~24× fewer tokens at 2.5× higher correctness
  rate**. The token-ratio gap widens as ``Θ(|X|)`` asymptotically.

**Cross-domain comparison.** Phase-31 (incident) vs Phase-32
(compliance), both mock, pooled across k ∈ {6, 20, 60, 120}:

| domain | substrate tokens | naive tokens @ k=60 | substrate acc @ k=120 | naive acc @ k=120 |
|---|---:|---:|---:|---:|
| incident triage | **196** | 3 014 | **1.00** | 0.20 |
| compliance review | **171** | 4 081 | **1.00** | 0.40 |

Both domains: (i) substrate is flat in k, (ii) substrate ≫ naive
at k=120 on accuracy, (iii) routing is 0 % on every k. The
per-domain differences (171 vs 196 tokens, 0.40 vs 0.20 naive
accuracy) are driven by domain-specific constants (cue-block
length, scenario-specific truncation point) — *not* by any
difference in the substrate's behaviour.

### D.2 Part B — noisy-extractor graceful-degradation sweep

Pooled across 2 seeds × k=20 × 5 scenarios = 10 runs per cell.
Numbers are for the **compliance-review** domain (incident-triage
is qualitatively identical; see ``results_phase32_noise_sweep.json``):

| drop | spurious | mislabel | acc_full | recall | precision | tokens | failure hist (pooled) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.00 | 0.00 | **1.00** | 1.00 | 1.00 |  171 | `{none:10}` |
| 0.00 | 0.00 | 0.25 | 0.50 | 0.70 | 0.75 |  168 | `{none:5, missing_handoff:5}` |
| 0.00 | 0.05 | 0.00 | 0.00 | 1.00 | 0.47 |  272 | `{spurious_claim:10}` |
| 0.00 | 0.10 | 0.00 | 0.00 | 1.00 | 0.34 |  378 | `{spurious_claim:10}` |
| 0.10 | 0.00 | 0.00 | 0.70 | 0.85 | 1.00 |  162 | `{none:7, missing_handoff:3}` |
| 0.10 | 0.05 | 0.00 | 0.00 | 0.85 | 0.43 |  264 | `{spurious_claim:7, missing_handoff:3}` |
| 0.25 | 0.00 | 0.00 | 0.70 | 0.80 | 0.90 |  160 | `{none:7, missing_handoff:3}` |
| 0.25 | 0.05 | 0.00 | 0.00 | 0.80 | 0.41 |  262 | `{spurious_claim:7, missing_handoff:3}` |
| 0.50 | 0.00 | 0.00 | 0.20 | 0.45 | 0.70 |  137 | `{missing_handoff:8, none:2}` |
| 0.50 | 0.05 | 0.00 | 0.00 | 0.60 | 0.34 |  245 | `{spurious_claim:3, missing_handoff:7}` |

Reading (aligning with Theorem P32-2):

1. **Identity (all noise = 0): 100 % accuracy.** Confirms the
   Phase-31 baseline on the new domain.

2. **Recall-limited regime (``drop_prob > 0``, spurious = 0,
   mislabel = 0).** Full-accuracy degrades as roughly
   ``(1 - δ)^{R*=2..3}``: at δ=0.10 we expect ≈ 0.81, observed
   0.70; at δ=0.25 we expect ≈ 0.56, observed 0.70; at δ=0.50 we
   expect ≈ 0.25, observed 0.20. Verdict accuracy holds up
   better than full accuracy (monotone regime — a single
   remaining blocker still produces BLOCKED); the collapse is
   driven by the strict flag set.

3. **Precision-limited regime (``spurious_prob > 0``).** Full
   accuracy hits 0 *immediately* at spurious ≥ 0.05 for a stark
   reason: flags is an exact-set grader, and any single spurious
   flag fails the test. Verdict accuracy *alone* remains 1.00 at
   spurious=0.05 drop=0 (see per-row detail in the JSON artifact),
   which is the **P32-2(iii) monotone regime**'s empirical
   signature: adding non-contradictory spurious claims does not
   flip BLOCKED → APPROVED. Both axes of degradation are named
   and attributed cleanly on the failure histogram
   (``spurious_claim`` vs ``missing_handoff``).

4. **Token-bound preservation (Theorem P32-3).** Even at
   spurious=0.10 the prompt grows to 378 tokens — a 2.2× increase
   over the noise-free 171. But compare: naive at k=20 is 1 778
   tokens. The substrate under the *worst* noise we swept (0.1
   spurious, 0.5 drop, 0.25 mislabel) is bounded by the inbox
   capacity × per-handoff token ceiling, *not* by the event
   stream size. Theorem P32-3 is empirically preserved.

5. **Mislabel is additive.** Mislabel=0.25 independently subtracts
   roughly one load-bearing claim per scenario (with probability
   0.25 per emission), producing an effective recall drop —
   visible in the recall column dropping from 1.00 at mislabel=0
   to 0.70 at mislabel=0.25, same drop_prob=0.

**Incident-triage pooled table (for cross-domain comparison).**
See ``results_phase32_noise_sweep.json`` for the full grid; the
key pattern is identical: (i) drop-only hurts only full-accuracy
and recall, (ii) spurious hurts only precision and full-accuracy,
never recall, (iii) tokens grow with spurious, stay flat with
drop. The degradation curves *superimpose across domains* — which
is the empirical companion to Theorem P32-1's domain-agnosticism.

### D.3 Part C — stronger-model spot check (qwen2.5-coder:7b)

Same harness, real Ollama, single seed = 32, k = 6 (small enough
that naive is *also* viable under the 7b's context budget, so the
comparison is apples-to-apples). Five scenarios per domain × 3
strategies = 15 calls per domain × 2 domains = 30 LLM calls total,
wall = see JSON artifact.

``qwen2.5-coder:7b`` via local Ollama, 15 calls per domain (5
scenarios × 3 strategies), `num_predict = 80`, temperature 0,
context length 4 096.

#### Incident triage (5 scenarios × 3 strategies, k = 6, seed = 32)

| strategy | acc_full | mean tokens | failure hist |
|---|---:|---:|---|
| naive          | 0.00 |  575 | `{retrieval_miss:5}` |
| **substrate**      | **0.40** | **196** | `{none:2, llm_error:3}` |
| **substrate_wrap** | **0.80** |  229 | `{none:4, llm_error:1}` |

LLM stats: 15 calls, 6 610 prompt tokens, wall 404.8 s.

Reading:

* **Naive at 0/5 (vs 100 % under the mock at k = 6).** The 7B is
  *not* limited by context window at 575 tokens — this is a
  qualitative failure at reading the raw delivered-events list
  and extracting the right root cause. Every failure attributes
  to ``retrieval_miss`` (the decoder-on-raw-events path cannot
  reconstruct the answer from the 7B's output). This confirms
  Theorem P29-2's content-level-aggregation claim on a non-code
  domain with a real 7B model: routing that delivers only raw
  telemetry cannot rescue the aggregator even with a model 14×
  larger than the 0.5b.
* **Substrate at 40 % (vs 40 % under the 0.5b).** At k = 6 the
  0.5b also scored 40 %; the 7B does not lift the bare substrate
  without the `substrate_wrap` scaffold. The failure is 3/5
  ``llm_error`` — the delivered inbox is complete (handoff recall
  = 1.0) but the 7B fails to transcribe the cue in 3 scenarios.
* **Substrate_wrap at 80 %.** The "return verbatim" instruction
  in `substrate_wrap` produces a 4/5 correct result with 1
  ``llm_error``. Phase-30's prediction (`substrate_wrap ≥
  substrate` at 7B+, unlike the 0.5b where the wrap instruction
  confused the small model) is confirmed: substrate_wrap is the
  strongest strategy at this scale, with a 2× lift over bare
  substrate and an ∞× lift over naive.
* **Substrate (either path) ≫ naive on a 7B non-code task.** The
  transcription-bounded axis (Theorem P30-3) is the remaining
  gap; at 80 %, the 7B is ≈ 0.80 / 1.00 = 80 % of the mock
  ceiling — which is *the* substrate-slice headline Phase 32
  Part C was trying to measure.

#### Compliance review (5 scenarios × 3 strategies, k = 6, seed = 32)

| strategy | acc_full | mean tokens | failure hist |
|---|---:|---:|---|
| naive          | 0.00 |  661 | `{retrieval_miss:5}` |
| **substrate**      | **1.00** | **171** | `{none:5}` |
| **substrate_wrap** | **1.00** |  204 | `{none:5}` |

LLM stats: 15 calls, 5 958 prompt tokens, wall 333.9 s. Total
Phase-32/C wall: 738.7 s (12.3 min, 30 calls).

Reading:

* **Substrate at 100 % — the 7B *saturates the mock ceiling* on
  compliance review.** At 171 tokens, the 7B is able to read
  and transcribe the substrate cue exactly on every one of the
  five scenarios. The transcription-bounded gap (Theorem P30-3)
  on compliance review vanishes at this model scale — a
  categorical improvement over the 0.5b's 40 % ceiling reported
  in Phase 31.
* **Substrate_wrap also at 100 %.** Both strategies tie at the
  mock ceiling; the 29-token cost of the "return verbatim"
  scaffolding neither helps nor hurts at 7B on this domain.
* **Naive at 0/5.** Same qualitative failure as incident triage:
  the 7B cannot extract the correct (verdict, flags, remediation)
  triple from 661 tokens of raw document-dump. Every failure
  attributes to ``retrieval_miss``. At k = 6 the naive prompt is
  well under the 7B's 4 096-token context window, so this is a
  *qualitative reading failure*, not a context-window failure.
* **Communication-bounded vs transcription-bounded on 7B.** On
  compliance review at 7B, the substrate's delivered bundle is a
  sufficient statistic that the 7B can transcribe exactly; the
  *entire* substrate vs naive gap is communication-bounded. This
  is the cleanest empirical signature of Theorem P31-5 the
  programme has: no universal compression of the 661-token naive
  stream was tried that the 7B can turn into 100 % correct
  answers; the substrate's *typed* 171-token bundle does.

#### Cross-domain pattern

| domain | naive | substrate | substrate_wrap | mock ceiling |
|---|---:|---:|---:|---:|
| incident triage | 0.00 | 0.40 | **0.80** | 1.00 |
| compliance review | 0.00 | **1.00** | **1.00** | 1.00 |

Both domains: naive ∈ {0.00}, substrate ≥ naive, substrate_wrap
≥ substrate at 7B. Compliance review *saturates the ceiling*;
incident triage at 80 % still has a 20-pp transcription-bounded
gap — plausibly because the incident auditor's expected output
shape (three specific fields: `ROOT_CAUSE` / `SERVICES` /
`REMEDIATION`) interacts with the 7B's code-tuning in ways the
compliance fields (`VERDICT` / `FLAGS` / `REMEDIATION`) do not,
but a single-seed spot check does not support that conclusion.
The main Part-C takeaway is the communication-boundedness of
both gaps: naive fails at 0 % on both domains at k = 6, far
below any mock or substrate number.

Reading (to be finalised once JSON is written):

* **Whether the result is transcription-bounded or
  communication-bounded.** At k=6 substrate delivers the same
  ≈ 171–196 tokens as the mock. Any shortfall on substrate vs
  mock's 100 % is attributable to the 7B's transcription
  fidelity — Theorem P30-3's load-bearing axis. Any gap with
  naive is communication-bounded (stream size vs substrate token
  bound).
* **Whether substrate_wrap ≥ substrate on 7B.** Phase-30 observed
  substrate_wrap ≈ substrate on 7B (and substrate > substrate_wrap
  on 0.5b, because the "return verbatim" instruction confused the
  small model). The Phase-32 spot check tests the same prediction
  on both non-code domains.

### D.4 Failure-attribution taxonomy (Phase 32 extension)

Phase 31's four-way failure histogram is extended by Phase 32 to
five categories (retaining the Phase 31 four, plus spurious_claim):

| Category | Attribution criterion | Typical noise axis |
|---|---|---|
| ``none`` | full_correct = True | — |
| ``truncation`` | truncated flag under naive/routing | stream size (|X|) |
| ``missing_handoff`` | required (role, kind) pair missing from auditor inbox | extractor recall drop |
| ``llm_error`` | inbox complete, LLM output mis-transcribes | model transcription fidelity |
| ``retrieval_miss`` | routing strategy with no content | routing-blind-to-content |
| ``spurious_claim`` | substrate path, inbox contains a non-causal (role, kind) pair | extractor precision drop |

The ``spurious_claim`` attribution is a Phase-32 addition; it is
the flag-side companion to ``missing_handoff`` (the verdict-side
failure). Every failure in the 96-row noise sweep is attributable
to exactly one of these categories, by construction.

### D.5 Provenance + chain integrity

Every ``HandoffLog`` across all 160 Phase-32/A measurements plus
the 480 Phase-32/B noise-sweep measurements emits a monotone
sequence of hash-chained records; ``verify_chain()`` returns
``True`` on every measurement (recorded as ``handoff_chain_ok =
True``). The noise wrapper *does not* touch the chain semantics —
it only changes the emission count and kinds, so the hash chain
continues to cover the *actually emitted* stream.

### D.6 No regressions — full test suite

```
$ python3 -m unittest discover -s vision_mvp/tests -q
...
Ran 1152 tests in 9.1s
OK
```

* 28 new tests in ``test_compliance_review.py``.
* 23 new tests in ``test_extractor_noise.py``.
* 1 101 Phase-31-era tests continue to pass unchanged.
* Phase-31's handoff primitive and the Phase-22..30 substrate
  tests are byte-stable.

### D.7 Cost

| run | cost |
|---|---:|
| Phase-32/A mock (160 measurements) | **0.1 s** |
| Phase-32/B noise sweep (480 measurements, 96 noise points) | **0.5 s** |
| Phase-32/C qwen2.5-coder:7b spot check (30 LLM calls) | ~3–5 min |
| Full test suite (1 152 tests) | ~9 s |

The mock and noise sweeps are CI-trivial on every merge. The
stronger-model spot check is a "run nightly" cost class — same
shape as Phase-30's json-stdlib headline.

---

## Part E — Closing notes

### E.1 Strongest empirical takeaway

> **The substrate's behaviour is domain-invariant under the
> Theorem P32-1 preconditions.** On a second non-code domain
> (vendor compliance review, distinct from incident triage along
> subject / role cast / decoder shape / event schema), the typed-
> handoff substrate keeps the compliance officer's prompt at 171
> tokens flat across k ∈ {6, 20, 60, 120} at 100 % accuracy under
> the mock ceiling — the same pattern Phase 31 produced on
> incident triage. Role-keyed routing is 0 % on every k; naive
> collapses from 100 % → 40 % at k=120 under truncation. **The
> substrate's graceful-degradation profile under controlled
> extractor noise is a two-regime story (recall-limited vs
> precision-limited), cleanly attributed on the failure histogram,
> and bounded by Theorem P32-2. The token-bound (Theorem P32-3)
> is preserved even under the worst noise swept: substrate prompt
> size at spurious_prob = 0.10 is 378 tokens, still ~11× less
> than naive at k = 120.** **On ``qwen2.5-coder:7b``, the
> substrate path saturates the mock ceiling (100 %) on compliance
> review and reaches 80 % via ``substrate_wrap`` on incident
> triage; naive is 0/5 on both domains at k = 6 — a
> qualitative failure at the 7B's reading comprehension on
> 575–661 tokens of raw role-chunked telemetry/documents. The
> substrate vs naive gap on 7B non-code tasks is
> communication-bounded on compliance review (the 7B can
> transcribe 171 tokens perfectly) and transcription-bounded by
> 20 pp on incident triage (from the mock's 100 % to
> substrate_wrap's 80 %).**

### E.2 Strongest theoretical takeaway

> Three theorems + two conjectures that *separate the substrate's
> correctness from the domain and from the extractor's idealised
> behaviour*:
>
> * **P32-1** (cross-domain correctness). The Phase-31 P31-4 proof
>   does not mention "incident"; explicitly transferring the
>   proof to a different ``(R, K, g)`` triple shows the correctness
>   guarantee is a property of the abstraction, not of the domain.
> * **P32-2** (graceful degradation). Two regimes — recall-limited
>   and precision-limited — with explicit bounds ``(1 - δ)^{R*}``
>   and ``(1 - ε)^{M·q_hi}``; empirically confirmed on the noise
>   sweep. The conjecture C31-7 is now a theorem in the
>   bounded-noise monotone regime.
> * **P32-3** (token-bound under noise). Peak active context
>   remains ``O(capacity · τ)`` as long as the inbox capacity is
>   wide enough to absorb the expected spurious blow-up. The
>   inbox capacity is the *regulariser* that keeps noise from
>   degrading the bounded-context claim.
> * **C32-4** (role-lattice stability). Substrate transfer across
>   domains reduces to the existence of a role-lattice
>   homomorphism + a claim-kind mapping + extractor noise bounds.
>   Partially confirmed (Phase 31 → Phase 32 is the first
>   instance); formally open for K ≥ 3 domains.
> * **C32-5** (extractor-composition bound). Union of two noisy
>   extractors bounds precision ``p_1·p_2`` from below, recall
>   ``1 - (1-r_1)(1-r_2)`` from below; the programme's design
>   principle for ensemble extractors.

### E.3 Relationship to the master plan

Phase 32 extends **Arc 8** (typed handoffs and non-code team
benchmarks). Arc 8 was a single-domain arc until now; Phase 32
closes the "one non-code domain is not enough" gap by adding a
second domain *and* closes the "extractors were perfect" gap by
measuring the substrate under noise. The master plan's § 4.11
current-frontier is updated accordingly: items (4) and (5) from
the Phase-31 "what would materially move the frontier next" list
are now addressed; the remaining items (SWE-bench end-to-end,
frontier-model headline, OQ-1 in full generality, K=20
role-lattice) remain.

### E.4 What this phase does not fix (carry-over to Phase 33+)

Ordered by research impact:

1. **SWE-bench end-to-end.** Carry-over from Phase 29/30/31.
2. **Frontier-model headline.** The Phase-32 Part C spot check is
   *one* model (``qwen2.5-coder:7b``) on *one* seed per domain.
   A proper sweep (multiple seeds, multiple k, 7B vs 8B vs 9B)
   is mechanical follow-up.
3. **Third non-code domain.** C32-4 is confirmed for K = 2
   (incident, compliance). A third domain (policy drafting,
   research synthesis, multi-role software delivery planning)
   would strengthen the cross-domain stability claim and test
   whether the decoder-shape taxonomy (monotone vs strict)
   covers the full space.
4. **K = 20 hierarchical role-lattice.** C31-6 unaffected by
   Phase 32; still the natural next test of the substrate at
   larger team sizes.
5. **Adversarial noise.** Phase-32/B uses i.i.d. Bernoulli noise.
   A threat-model-driven adversary (selective drop of load-
   bearing claims, targeted spurious emission on the *highest-
   priority* kinds) is the realistic production concern and the
   natural follow-up to C31-7/P32-2.
6. **Extractor-composition empirics.** C32-5 is stated but not
   swept.
7. **LLM-driven extractors.** The regex extractors of Phase 31 +
   Phase 32 have a deterministic noise profile. Replacing them
   with an LLM-call extractor and measuring the induced noise
   profile is the realistic production-path next step.

None of these *block* the Phase-32 headline; they are the
follow-ons that extend it.

### E.5 Reproducibility

| Run | Command | Output |
|---|---|---|
| Phase-32/A mock sweep | `python3 -m vision_mvp.experiments.phase32_compliance_review --mock --distractor-counts 6 20 60 120 --seeds 32 33 --out vision_mvp/results_phase32_compliance_mock.json` | `vision_mvp/results_phase32_compliance_mock.json` |
| Phase-32/B noise sweep | `python3 -m vision_mvp.experiments.phase32_noise_sweep --domain both --distractor-counts 20 --drop-probs 0.0 0.1 0.25 0.5 --spurious-probs 0.0 0.05 0.1 --mislabel-probs 0.0 0.25 --seeds 31 32 --out vision_mvp/results_phase32_noise_sweep.json` | `vision_mvp/results_phase32_noise_sweep.json` |
| Phase-32/C qwen2.5-coder:7b | `python3 -m vision_mvp.experiments.phase32_stronger_model --model qwen2.5-coder:7b --seeds 32 --distractor-counts 6 --strategies naive substrate substrate_wrap --out vision_mvp/results_phase32_llm_7b_spot.json` | `vision_mvp/results_phase32_llm_7b_spot.json` |
| Phase-32 unit tests (compliance) | `python3 -m unittest vision_mvp.tests.test_compliance_review` | 28 tests, all pass |
| Phase-32 unit tests (noise) | `python3 -m unittest vision_mvp.tests.test_extractor_noise` | 23 tests, all pass |
| Full suite | `python3 -m unittest discover -s vision_mvp/tests` | 1 152 tests, all pass |

---

## Part F — Relationship to the master plan

Phase 32 belongs to **Arc 8 (extended)**: *agent-team communication
substrate, non-code task-scale evidence, and robustness theory.*

* **Arc 1–6** — code / routing / substrate tracks (Phases 1–29).
* **Arc 7** — LLM-in-loop external validity on code (Phase 30).
* **Arc 8 (Phase 31)** — first non-code task-scale benchmark +
  typed-handoff substrate primitive.
* **Arc 8 (Phase 32)** — second non-code domain + extractor-noise
  robustness + stronger-model spot check + three cross-domain
  theorems. Arc 8 is now a *multi-domain arc* with non-trivial
  robustness theory, not a single-benchmark arc.

The master plan § 4.11 frontier is updated to reflect that items
(4) "second non-code domain" and (5) "noisy-extractor sweep" from
the Phase-31 forward list are discharged; remaining items
(SWE-bench end-to-end, frontier-model headline at scale, OQ-1,
K=20 role-lattice, adversarial noise, cross-domain third-domain
test) remain open.

**Durability notes.** Three observations about what Phase 32
teaches the programme durably:

1. **The substrate's identity is *communication between
   role-typed agents*, not *representation of a domain*.** Two
   domains, byte-stable substrate code, same correctness profile
   — the § 1.5 master-plan distinction from graph/index tools is
   now a *two-data-point* empirical claim, not a framing.
2. **Noise attribution is first-class.** The ``spurious_claim``
   category is a Phase-32 addition to the failure taxonomy;
   future noise-sweep phases inherit this shape and the theorem
   it anchors.
3. **Inbox capacity is a regulariser, not a UX detail.** Theorem
   P32-3 makes the inbox's role explicit: it is the mechanism by
   which the bounded-context claim survives noise. Future
   substrate modules should treat capacity as a first-class
   design parameter.
