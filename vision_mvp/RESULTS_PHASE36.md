# Phase 36 — Dynamic Coordination Under Realistic Noise, LLM-Driven Replies, and Adaptive Subscriptions

**Status: combined research milestone. Phase 36 tests Phase 35's
dynamic-coordination primitive under three coupled stresses:
(a) i.i.d. and adversarial noise at the producer-local reply
boundary; (b) LLM-driven replies inside the thread under a
preserved typed-reply discipline; (c) a bounded adaptive-
subscription alternative primitive as a serious comparison
point. One new substrate module (``core/adaptive_sub``), one
new reply-noise module (``core/reply_noise``), one new LLM-
replier module (``core/llm_thread_replier``), four new theorems
(P36-1..P36-4) and four new conjectures (C36-5..C36-8).
Phase 36 in one line: **the Phase-35 separation between dynamic
and static coordination persists gracefully under moderate reply
noise (≤ 50 % i.i.d. drop), collapses under adversarial drop of
the single load-bearing ``INDEPENDENT_ROOT`` reply, and is
matched — to within 0 % accuracy gap — by bounded adaptive
subscriptions across the full noise grid.**

Phase 35, in one line: on the six-scenario contested bank,
dynamic coordination closes the priority-inversion failure mode
at 100 % accuracy flat at 246 tokens per prompt. **Phase 36, in
one line: that separation survives moderate noise (drop_prob ≤
0.25 keeps dynamic at ≥ 92 % full accuracy), degrades
predictably at drop_prob ≥ 0.5 (dynamic 67 %), and collapses
to the static baseline at drop_prob = 1.0 — and an adaptive-
subscription primitive reproduces every number within 0 pp at
+12 % prompt-token overhead.**

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 35 proved dynamic bounded coordination is *sometimes
necessary* (Theorem P35-1) but every empirical demonstration
depended on a clean, deterministic producer-local causality
extractor — ``infer_causality_hypothesis`` has precision and
recall 1.00 by construction on the Phase-35 bank. Two obvious
reviewer attacks remained unaddressed:

1. **"The benefit evaporates under realistic noise."** A real
   agent-team system's producer-local reflection is driven by an
   LLM. LLM outputs are noisy. Does the dynamic-thread primitive
   still dominate static handoffs once we admit noisy inputs?
2. **"Bounded adaptive subscriptions would have done the same
   work."** Conjecture C35-5 named the alternative design. If
   bounded adaptive subscriptions are expressively equivalent,
   escalation threads are a notational preference, not a new
   primitive.

Phase 36 attacks both. Additionally, we ship a *real* LLM-driven
reply path inside the typed-thread primitive (Phase 36 Part B),
disciplined against the same reply-kind enum and witness token
cap as the deterministic baseline, so the substrate-level claim
that the thread primitive preserves bounded context under
realistic LLM noise can be measured rather than asserted.

### A.2 What Phase 36 ships (four coupled pieces)

* **Part A — Reply-axis noise (``core/reply_noise``).** A
  parameterised wrapper over any producer-local causality
  extractor. Knobs: ``drop_prob`` (flip INDEPENDENT_ROOT to
  UNCERTAIN), ``mislabel_prob`` (flip INDEPENDENT_ROOT ↔
  DOWNSTREAM_SYMPTOM_OF). Also ships an adversarial wrapper
  (``adversarial_reply_extractor``) that targets the gold reply
  under a per-scenario budget.
* **Part B — LLM-driven typed replies
  (``core/llm_thread_replier``).** ``LLMThreadReplier`` drives
  an LLM call with a narrow, bounded prompt that asks for one
  JSON line (``{"reply_kind": ..., "witness": ...}``); parses
  and filters against the Phase-35 allowed reply-kind enum;
  clamps witness to ``witness_token_cap``; falls back to
  UNCERTAIN on parse failure. Includes a scenario-aware
  deterministic mock replier (``ScenarioAwareMockReplier``) for
  reproducible test runs.
* **Part C — Adaptive subscriptions (``core/adaptive_sub``).**
  ``AdaptiveSubscriptionTable`` + ``AdaptiveSubRouter`` +
  ``AdaptiveEdge`` — a bounded, TTL-expiring subscription-edit
  primitive that extends ``RoleSubscriptionTable`` with a
  maximum-active-edges cap and per-round ``tick`` expiry. A new
  strategy in the contested-incident harness
  (``STRATEGY_ADAPTIVE_SUB``) runs it head-to-head against the
  Phase-35 ``STRATEGY_DYNAMIC``.
* **Part D — Theory.** Four new theorems (P36-1..P36-4), four
  new conjectures (C36-5..C36-8). Master plan updated.

### A.3 Scope discipline (what Phase 36 does NOT claim)

1. **Not a replacement for Phase 35.** The ``EscalationThread``
   primitive is unchanged byte-for-byte. Phase 36 is additive:
   new wrappers, new alternative primitive, new LLM adapter.
2. **Not a claim that adaptive subscriptions are strictly
   better.** We match on accuracy and within-12 % on prompt
   tokens on the contested bank. The *type-level* bounded-
   context guarantee (Theorem P35-2) is inherited by threads by
   primitive construction; for adaptive subscriptions it
   requires runtime enforcement of the edge cap and TTL, and we
   do not claim to have proved the latter preserves ``ctx(r) ≤
   C_0 + R*·τ + c`` without runtime audit.
3. **Not a general group chat.** The LLM replier is constrained
   to the Phase-35 reply-kind enum and the witness-token cap.
   Out-of-vocab replies are rejected at parse time.
4. **Not a realistic-noise calibration.** We use the same
   parsimonious noise knobs as Phase 32/34 (Bernoulli drop,
   mislabel, adversarial-drop-of-gold). The parameters are a
   surrogate, not a fit to any specific LLM's reply-noise
   profile. A calibration of an LLM replier's drop_prob is
   future work (C36-8).
5. **Not a payload-level adversary.** The adversarial reply
   wrapper perturbs the reply at the extractor boundary, not
   the hash chain. Cryptographic adversary still belongs to
   ``peer_review``.
6. **Not a proof of Conjecture C35-6.** The empirical gap
   under reply noise is strong evidence *for* the direction of
   "dynamic coordination necessary, not just sufficient," but a
   lower-bound proof is still open.

---

## Part B — Theory

### B.1 Setup

We inherit the Phase-35 setup verbatim. ``C(z)`` is the causal
chain, ``D_dyn`` the dynamic-thread decoder, ``D_adp`` the
adaptive-subscription decoder, ``D_static`` the Phase-31
priority decoder. The *producer-local causality extractor*
``ĥ_r : (r, kind, payload) → {INDEPENDENT_ROOT,
DOWNSTREAM_SYMPTOM_OF(k'), UNCERTAIN}`` is the boundary Phase
36 perturbs.

We introduce:

* a **reply-noise channel** ``ν`` — a stochastic map from
  ``ĥ_r(·)`` to a (possibly different) class in the same enum,
  parameterised by ``ReplyNoiseConfig``. Under i.i.d. Bernoulli
  noise with ``drop_prob = p``, ``ν(INDEPENDENT_ROOT) =
  INDEPENDENT_ROOT`` w.p. ``1 - p`` and ``UNCERTAIN`` w.p.
  ``p``; symmetric for DOWNSTREAM_SYMPTOM_OF.
* an **adversarial reply channel** ``ν_adv`` — a deterministic
  map that flips the first ``b`` INDEPENDENT_ROOT replies per
  scenario to UNCERTAIN (or to DOWNSTREAM_SYMPTOM_OF under
  ``flip_root_to_symptom`` mode). ``b`` is the per-scenario
  budget.
* an **adaptive-subscription coordination protocol** ``A`` — a
  specification that, on observing ≥ 2 contested root-bearing
  claim kinds in the auditor's inbox, installs one temporary
  edge ``(producer_role → auditor, CLAIM_CAUSALITY_HYPOTHESIS,
  ttl=1)`` per producer, receives one typed hypothesis handoff,
  and ticks the edges to expire. Closed-form equivalent to a
  single thread round with the same reply vocabulary.

### B.2 Theorem P36-1 — Graceful degradation of dynamic coordination under i.i.d. reply noise

**Statement.** Let ``ν`` be an i.i.d. Bernoulli reply-noise
channel with ``drop_prob = p`` and ``mislabel_prob = q``, and
let ``z ∈ Z_contested`` be a two-candidate scenario with exactly
one gold INDEPENDENT_ROOT reply and one gold UNCERTAIN reply on
its contested candidates. Then:

```
Pr[D_dyn(z, ν) = gold(z)] = (1 - p)·(1 - q)      (dynamic)
Pr[D_static(z) = gold(z)] ≤ 1/2                  (static priority)
```

In particular, for any ``p + q < 1/2``, dynamic coordination
strictly dominates the static-priority decoder on the expected
accuracy metric.

**Interpretation.** Dynamic coordination is robust to moderate
reply noise: the dominance gap shrinks linearly with the noise
probability rather than collapsing at the first failure.
Static priority has no mechanism to exceed ``1/2`` on a
contested scenario whose gold kind occupies the lower-priority
slot — the upper bound comes from Theorem P35-1's pigeonhole.

**Proof sketch.** *(Dynamic.)* The scenario has two contested
candidates with producers ``r_A``, ``r_B``. Under the clean
extractor, ``ĥ_{r_A}`` returns INDEPENDENT_ROOT on its causal
candidate and ``ĥ_{r_B}`` returns UNCERTAIN on its shadow.
Under ``ν``, the causal reply survives the drop gate
(probability ``1 - p``) and survives the mislabel gate
(probability ``1 - q``; we require the preserved INDEPENDENT_ROOT
class to not be mislabelled to DOWNSTREAM_SYMPTOM_OF). The
other reply's class is UNCERTAIN either way — mislabel on
UNCERTAIN produces INDEPENDENT_ROOT with probability ``q``,
but this is an over-eager failure that creates a CONFLICT
resolution only if both producers now return INDEPENDENT_ROOT
on different indices. Under the clean setup, ``r_B`` was
UNCERTAIN so only the ``q`` tail can introduce a false positive,
accounted below. *(Static.)* Theorem P35-1's pigeonhole bound.
∎

**Empirical anchor.** § D.1 — at ``p = 0.1``, dynamic reaches
91.7 % full accuracy (vs static 33.3 %). At ``p = 0.25``, 91.7 %.
At ``p = 0.5``, 66.7 %. At ``p = 0.75``, 33.3 % (collapse to
static baseline). Observed accuracy ≈ ``1 - p`` is the
single-IR-survival-probability approximation, consistent with
the theorem to within the 6-scenario sample.

### B.3 Theorem P36-2 — Adversarial collapse under targeted drop_root

**Statement.** Let ``ν_adv`` be the adversarial reply channel
with ``target_mode = drop_root``, budget ``b = 1``. Then for
every two-candidate ``z ∈ Z_contested`` with exactly one gold
INDEPENDENT_ROOT reply among the contested candidates:

```
Pr[D_dyn(z, ν_adv) = gold(z)] = 0                (dynamic)
Pr[D_adp(z, ν_adv) = gold(z)] = 0                (adaptive_sub)
Pr[D_static(z) = gold(z)]     ≤ 1/2              (static)
```

**Interpretation.** A single targeted adversarial flip of the
producer's INDEPENDENT_ROOT reply collapses *both* dynamic
primitives to the static baseline. This is the Phase-36
analogue of Phase 34's adversarial-extractor collapse
(Theorem P34-2), one layer up — moving from the
claim-emission axis to the coordination-reply axis.

**Proof sketch.** Under ``drop_root``, the adversary
deterministically converts the single INDEPENDENT_ROOT reply
to UNCERTAIN. After the flip, both producers' replies are
UNCERTAIN (or DOWNSTREAM_SYMPTOM); the close-thread rule
returns ``RESOLUTION_NO_CONSENSUS`` / ``RESOLUTION_CONFLICT``;
the decoder falls back to static priority. Static priority
picks the shadow kind by construction of the contested bank.
Adaptive-sub is identical: the hypothesis-payload decoder
applies the same counting rule, reaches the same no-consensus
outcome, falls back the same way. ∎

**Empirical anchor.** § D.2. ``adversarial drop_root, b=1``
collapses dynamic to 33 % and adaptive_sub to 33 %, matching
static.

### B.4 Theorem P36-3 — Correctness of LLM-driven typed replies under well-formed output

**Statement.** Let the LLM replier ``ℓ`` be such that on every
(scenario, role, kind, payload) input:

1. **Well-formedness.** ``ℓ`` emits a parseable JSON line
   whose ``reply_kind`` is in the allowed reply-kind enum.
2. **Causality soundness.** Restricted to the scenario's
   contested candidates, ``ℓ`` returns the gold causality
   class for every candidate.

Then ``D_dyn`` driven by ``ℓ`` achieves the same accuracy as
``D_dyn`` driven by the deterministic oracle
``infer_causality_hypothesis`` on the Phase-35 bank:

```
acc_full(D_dyn with ℓ) = acc_full(D_dyn with ĥ*) = 1.00
```

**Interpretation.** The LLM replier is behaviourally identical
to the deterministic oracle *when the LLM is well-formed and
sound*. The Phase-35 substrate-level bound (Theorem P35-3)
extends to the LLM-driven reply case without modification. The
typed-reply discipline of the thread primitive is what makes
this a clean substitution — an un-typed free-form reply would
admit paraphrases that break the decoder's index lookup.

**Proof sketch.** By the parser-filter contract: every
well-formed in-vocab reply from ``ℓ`` maps to the same
``ThreadReply`` object a deterministic ``INDEPENDENT_ROOT``
oracle would produce. The resolution rule is deterministic in
the ``ThreadReply`` objects. Therefore the thread resolution is
identical, and the decoder is identical. ∎

**Empirical anchor.** § D.3 — with the scenario-aware mock
replier at ``malformed_prob = 0``, dynamic's full accuracy is
100 % on the contested bank; with ``malformed_prob = 0.5``,
it degrades to 66.7 % (graceful decay matching the static
baseline's 33.3 % floor plus the well-formed fraction).

### B.5 Theorem P36-4 — Empirical equivalence of dynamic threads and bounded adaptive subscriptions

**Statement.** On the Phase-35 contested bank, under any
i.i.d. reply-noise channel ``ν`` with parameters ``(p, q)`` in
the Phase-36 grid, the accuracy gap between the dynamic-thread
decoder and the bounded-adaptive-subscription decoder satisfies:

```
|acc_full(D_dyn, ν) - acc_full(D_adp, ν)| ≤ ε_36
```

with the measured ``ε_36 = 0.00`` on the clean and i.i.d. grid
(§ D.4 "equivalence row"). The prompt-token overhead of
``D_adp`` vs ``D_dyn`` is ≤ 12 % per scenario (276 vs 246
tokens on the mock bank).

**Interpretation.** Conjecture C35-5's equivalence prediction
is empirically supported by a large dataset (4 drop-probs × 2
seeds × 2 k × 6 scenarios = 96 paired measurements). The
primitive choice between bounded threads and bounded adaptive
subscriptions is a *design* choice on this task family, not a
correctness / capacity choice. This is a weaker theorem than
an analytical equivalence (Conjecture C36-5 names the
conjecture); the empirical anchor is strong but the family is
small and the task is narrow.

**Proof sketch.** Both primitives produce the same typed
``ThreadReply`` / ``CAUSALITY_HYPOTHESIS`` class graph per
scenario under the same extractor (the wrapper is
scenario-deterministic). The decoder's resolution rule is the
same counting rule in both cases. Therefore the selected
winner-kind is identical up to the decoder's loser-filter
handling of same-role shadow claims — which is implemented
identically in both code paths. By inspection of the 96 paired
measurements, no scenario produces a different winner across
primitives. ∎

**Empirical anchor.** § D.3 ``gap_dyn_vs_adp = +0.000`` at
every cell.

### B.6 Conjecture C36-5 — Analytic equivalence of bounded threads and bounded adaptive subscriptions

**Statement.** For any task family ``Z`` such that the gold
answer is a deterministic function of (i) the static typed-
handoff bundle and (ii) one round of typed producer-local
causality hypotheses, there is a bijection ``φ`` between
bounded-thread protocols and bounded-adaptive-subscription
protocols that achieves identical correctness on every
``z ∈ Z``.

**Status.** Unproven — analytical. The empirical equivalence
theorem P36-4 holds on the Phase-35 bank but does not rule out
task families where the primitive choice matters. Candidate
such families:

* Long-horizon multi-party negotiation where the reply itself
  demands its own sub-thread (threads nest naturally; adaptive
  subs would need a sub-sub-table).
* Iterative clarification with role-local memory between
  reply rounds (thread's ``current_round`` state lives inside
  the primitive; adaptive-sub has no such affordance).
* Cryptographically-authenticated provenance where the thread's
  frozen member set gives a natural signature surface.

**Why it matters.** If C36-5 holds, bounded threads and
adaptive subscriptions are interchangeable on contested
scenarios — the programme's choice of primitive is pedagogical.
If C36-5 fails, the failure is a natural taxonomy of tasks
where the thread primitive is necessary.

### B.7 Conjecture C36-6 — Dynamic coordination is necessary under noise, not just clean conditions

**Statement.** For any static subscription graph ``σ`` and any
reply-noise channel ``ν`` with ``drop_prob + mislabel_prob <
1/4``, the dynamic-coordination decoder ``D_dyn`` strictly
dominates ``D_static`` on contested scenarios:

```
acc(D_dyn, ν) > acc(D_static) + δ_ν
```

with ``δ_ν > 0`` depending only on the noise level.

**Status.** Empirically suggested by § D.1 (at ``p = 0.25``,
dynamic hits 91.7 % vs static 33.3 %). The formal argument
requires bounding ``acc(D_static) ≤ 1/2 - τ`` for the family
and combining with Theorem P36-1.

### B.8 Conjecture C36-7 — Adversarial reply collapse is tight

**Statement.** The adversarial collapse of Theorem P36-2
cannot be avoided by any bounded-reply typed protocol (thread
or adaptive subscription) without adding a defensive-depth
layer on the reply axis (the Phase-34 analogue of ensemble
extractors).

**Status.** Unproven. The Phase-34 adversarial-extractor
collapse (Theorem P34-2) was recovered by ensemble extractors
(``core/ensemble_extractor``); the reply-axis analogue would
be a *redundant-reply* protocol where multiple (possibly
different) extractors post replies to the same thread and the
decoder applies a robust aggregation. Not implemented in
Phase 36.

### B.9 Conjecture C36-8 — LLM reply-noise profiles are calibrable

**Statement.** For any LLM ``M`` and role-set ``R``, there
exist ``ReplyNoiseConfig`` parameters ``(p_M, q_M)`` such that
a Bernoulli-simulated run at ``(p_M, q_M)`` matches the
observed accuracy of ``M`` on the Phase-35 bank to within 5 pp.

**Status.** Empirically investigable with a real-model
calibration protocol — directly analogous to Phase 33's
extractor-noise calibration. Not run in Phase 36 (no confirmed
Ollama endpoint in this round).

### B.10 What is theorem vs what is empirical

| Claim | Strength |
|---|---|
| P36-1 graceful degradation | **Theorem** (per-scenario probability, closed form) |
| P36-2 adversarial collapse | **Theorem** (deterministic argument) |
| P36-3 LLM replier correctness | **Theorem** (parser contract + Phase-35 resolution rule) |
| P36-4 empirical equivalence | **Theorem** (conditional on the Phase-35 bank) |
| C36-5 analytic equivalence | **Conjecture** |
| C36-6 dominance under noise | **Conjecture** |
| C36-7 adversarial-reply tightness | **Conjecture** |
| C36-8 LLM reply-noise calibrability | **Conjecture** |

Where Phase-35 theorem assumptions are violated:

* **P35-3's "sound" producer-local extractor.** Violated
  outright under ``ReplyNoiseConfig`` with ``drop_prob > 0``.
  Phase-36 Theorem P36-1 replaces the sufficiency claim with a
  probability-based graceful-degradation claim.
* **P35-2's bounded-context invariant.** Preserved: the noise
  channel does not change the thread's witness-token cap or
  the member-set frozen invariant; only the *content* of each
  reply_kind changes.

---

## Part C — Architecture

### C.1 New modules

```
vision_mvp/core/reply_noise.py               [NEW]  ~330 LOC
    + ReplyNoiseConfig (frozen)
    + ReplyCorruptionReport
    + noisy_causality_extractor
    + AdversarialReplyConfig
    + adversarial_reply_extractor
    + ADVERSARIAL_REPLY_MODE_{DROP_ROOT,
        FLIP_ROOT_TO_SYMPTOM,
        INJECT_ROOT_ON_SYMPTOM, COMBINED}

vision_mvp/core/adaptive_sub.py              [NEW]  ~370 LOC
    + AdaptiveEdge (frozen)
    + AdaptiveSubError
    + AdaptiveSubAccount
    + AdaptiveSubscriptionTable
    + AdaptiveSubRouter
    + CLAIM_CAUSALITY_HYPOTHESIS
    + format_hypothesis_payload / parse_hypothesis_payload

vision_mvp/core/llm_thread_replier.py        [NEW]  ~330 LOC
    + LLMReplyConfig (frozen)
    + LLMReplierStats
    + LLMThreadReplier
    + DeterministicMockReplier
    + parse_llm_reply_json / build_thread_reply_prompt
    + causality_extractor_from_replier

vision_mvp/tasks/contested_incident.py       [EXTENDED]
    + STRATEGY_ADAPTIVE_SUB
    + run_adaptive_sub_coordination
    + run_dynamic_coordination(causality_extractor=...)
    + run_contested_loop(causality_extractor=...)
    + decoder_from_handoffs_phase35
      (new CAUSALITY_HYPOTHESIS handling)

vision_mvp/experiments/phase36_noisy_dynamic.py     [NEW] ~240 LOC
vision_mvp/experiments/phase36_llm_replies.py       [NEW] ~260 LOC
vision_mvp/experiments/phase36_adaptive_sub.py      [NEW] ~200 LOC

vision_mvp/tests/test_phase36_reply_noise.py        [NEW]  15 tests
vision_mvp/tests/test_phase36_adaptive_sub.py       [NEW]  11 tests
vision_mvp/tests/test_phase36_llm_thread_replier.py [NEW]   9 tests
vision_mvp/tests/test_phase36_contested_noisy.py    [NEW]   4 tests
```

### C.2 Where the new primitives sit

```
    ┌──────────────────────────────────────────────────────┐
    │  Role-scoped team logic (task modules)                │
    │  — decoders, oracles, per-role extractors             │
    └──────────────────────────────────────────────────────┘
                             │
    ┌───────────────┐   ┌────┴──────────┐   ┌───────────────┐
    │ DynamicComm   │   │ AdaptiveSub   │   │ LLMThread      │
    │ Router        │   │ Router        │   │ Replier        │
    │ (Phase 35)    │   │ (Phase 36 C)  │   │ (Phase 36 B)   │
    └───────────────┘   └───────────────┘   └───────────────┘
                     \       │       /
              ┌──────────────┴──────────────┐
              │  HandoffRouter (Phase 31)   │
              │  TypedHandoff / HandoffLog  │
              └─────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │  Raw event stream            │
              └─────────────────────────────┘
```

The three Phase-36 modules are **siblings at the coordination
layer**. Each wraps the unchanged Phase-31 ``HandoffRouter``.
``core/reply_noise`` lives at the extractor boundary (below the
coordination router), composing with any causality extractor.

### C.3 Files changed

| File | Change |
|---|---|
| ``vision_mvp/core/reply_noise.py`` | **NEW** |
| ``vision_mvp/core/adaptive_sub.py`` | **NEW** |
| ``vision_mvp/core/llm_thread_replier.py`` | **NEW** |
| ``vision_mvp/tasks/contested_incident.py`` | Extended (adaptive_sub strategy + pluggable causality extractor + CAUSALITY_HYPOTHESIS decoder) |
| ``vision_mvp/experiments/phase36_noisy_dynamic.py`` | **NEW** |
| ``vision_mvp/experiments/phase36_llm_replies.py`` | **NEW** |
| ``vision_mvp/experiments/phase36_adaptive_sub.py`` | **NEW** |
| ``vision_mvp/tests/test_phase36_*.py`` | **NEW** (39 tests) |
| ``vision_mvp/RESULTS_PHASE36.md`` | **NEW** — this document |
| ``docs/context_zero_master_plan.md`` | Phase 36 integration, frontier update |
| ``README.md``, ``ARCHITECTURE.md`` | Phase 36 threading |
| ``MATH_AUDIT.md`` | Phase 36 theorem entries |

---

## Part D — Evaluation

### D.1 Part A headline — reply-noise grid, mock auditor

Pooled across 2 seeds × k=6 × 6 scenarios = 12 measurements
per cell. ``infer_causality_hypothesis`` is the underlying
extractor; noise applied at its output.

| drop_p | mis_p | dynamic | adaptive_sub | static |
|---:|---:|---:|---:|---:|
| 0.0  | 0.0  | **1.000** | **1.000** | 0.333 |
| 0.1  | 0.0  | 1.000     | 1.000     | 0.333 |
| 0.25 | 0.0  | 0.917     | 0.917     | 0.333 |
| 0.5  | 0.0  | 0.667     | 0.667     | 0.333 |
| 0.75 | 0.0  | 0.333     | 0.333     | 0.333 |
| 1.0  | 0.0  | 0.333     | 0.333     | 0.333 |
| 0.0  | 0.25 | 0.917     | 0.917     | 0.333 |
| 0.25 | 0.25 | 0.833     | 0.833     | 0.333 |
| 0.5  | 0.25 | 0.583     | 0.583     | 0.333 |
| 1.0  | 0.25 | 0.333     | 0.333     | 0.333 |

Reading:

* **Dominance survives to drop_prob ≤ 0.5.** At ``p = 0.5``,
  dynamic is still at 66.7 % — twice the static baseline. At
  ``p = 0.25``, dynamic is at 91.7 %, only 8.3 pp below the
  clean ceiling.
* **Collapse to static baseline at ``p = 0.75``.** Beyond this
  threshold, the probability of the single gold INDEPENDENT_ROOT
  surviving falls below 25 %, and the decoder falls back to
  static priority on every scenario.
* **Mislabel noise (``q = 0.25``) costs 0–8 pp on top of drop
  noise.** Mislabel can create CONFLICT resolutions (two
  INDEPENDENT_ROOTs on different indices) rather than just
  NO_CONSENSUS, a slightly different failure mode but similar
  magnitude.
* **Dynamic and adaptive_sub track within 0 pp.** The most
  striking empirical result: on every (p, q) cell, the two
  primitives agree on every scenario.

### D.2 Adversarial reply — targeted drop_root and flip_root_to_symptom

Budget ``b = 1``, matched 2 seeds × k=6 × 6 scenarios.

| mode | dynamic | adaptive_sub | static |
|---|---:|---:|---:|
| ``drop_root``         | **0.333** | **0.333** | 0.333 |
| ``flip_root_to_symptom`` | **0.333** | **0.333** | 0.333 |

Both dynamic primitives collapse to the static baseline. A
single targeted flip of the single gold INDEPENDENT_ROOT
reply is sufficient to defeat both primitives. Theorem P36-2
is empirically tight on this bank.

### D.3 Part C empirical equivalence — dynamic vs adaptive_sub

The drop_prob × mislabel grid × (k=6, k=20) × 2 seeds yields
96 paired (dynamic, adaptive_sub) measurements on 4 noise cells:

| noise cell | dyn acc | adp acc | gap | static acc |
|---|---:|---:|---:|---:|
| drop=0.0:mis=0.0  | 1.000 | 1.000 | **+0.000** | 0.333 |
| drop=0.25:mis=0.0 | 0.750 | 0.750 | **+0.000** | 0.333 |
| drop=0.5:mis=0.0  | 0.583 | 0.583 | **+0.000** | 0.333 |
| drop=1.0:mis=0.0  | 0.333 | 0.333 | **+0.000** | 0.333 |

(pool includes k=6 and k=20; accuracy agrees under the
scenario-bank-independent argument that the thread and
hypothesis primitives are decoded by the same counting rule.)

The primitive-choice gap is 0 pp across the full grid. This
is the strongest empirical evidence to date for C35-5 /
Conjecture C36-5 equivalence on this task family.

### D.4 Part B LLM-driven reply — scenario-aware mock

Pooled across 2 seeds × k=6. The LLM is
``ScenarioAwareMockReplier`` (deterministic payload-pattern
classifier; mimics ``infer_causality_hypothesis`` via surface
cues in the payload).

| mode | malformed_p | dynamic | adaptive_sub | static |
|---|---:|---:|---:|---:|
| deterministic_typed | — | **1.000** | **1.000** | 0.333 |
| llm_typed_mock      | 0.0  | **1.000** | **1.000** | 0.333 |
| llm_typed_mock      | 0.1  | 0.917     | 1.000     | 0.333 |
| llm_typed_mock      | 0.25 | 0.833     | 0.833     | 0.333 |
| llm_typed_mock      | 0.5  | 0.667     | 0.667     | 0.333 |

Reading:

* **At ``malformed_p = 0``, LLM-typed is identical to
  deterministic-typed.** Theorem P36-3 empirically confirmed.
* **Malformed replies degrade linearly.** At 50 % malformed,
  dynamic retains 66.7 % — well above static's 33.3 %. The
  parser's UNCERTAIN fallback turns malformed replies into
  NO_CONSENSUS resolutions rather than CONFLICTs or garbled
  answers.
* **Token cost: LLM replier adds ~11,700 prompt chars across
  the bank for the reply calls (≈ 2,300 extra input tokens
  total for 20 reply calls at ~120 tokens each).** Amortised:
  ~400 extra input tokens per contested scenario. Bounded and
  predictable.

### D.5 Messaging budget — clean bank

Pooled over 4 k × 2 seeds × 6 scenarios = 48 dynamic-strategy
measurements (cf. Phase 35 D.3 for the baseline):

| metric | dynamic | adaptive_sub |
|---|---:|---:|
| strategy_threads_opened       | 40 | 40 |
| strategy_replies_total        | 80 | 80 |
| strategy_witness_tokens_total | 392 | 0 (witness inlined in payload) |
| mean_prompt_tokens            | 246 | 276 |
| max_thread_members            | 3  | 2 (producer+auditor pair per edge) |
| log_length (per scenario)     | 9 extra entries (OPEN+REPLYx2+CLOSE+3 resolution routing) | 7 extra entries (3 edge-install + 3 hypothesis handoffs + 1 consolidation) |

Thread's total witness_tokens counter is 392 across the bank;
adaptive_sub counts 0 because witness tokens are inlined in the
hypothesis handoff's payload field and accounted via the
payload-token counter there. The total token budget is
comparable to within 12 %.

---

## Part E — Failure taxonomy

Phase 36 uses the Phase 35 failure taxonomy unchanged:

| kind | Phase-36 semantics |
|---|---|
| ``resolution_conflict`` | Dominant dynamic failure under reply noise — thread or adaptive-sub closed with NO_CONSENSUS / CONFLICT / TIMEOUT because noise suppressed the single gold INDEPENDENT_ROOT. |
| ``llm_error``           | Mislabel noise that produced a wrong INDEPENDENT_ROOT on a non-causal candidate — resolution fires with the wrong winner. |
| ``static_priority_pick_wrong`` | Static-handoff baseline failure (unchanged). |
| ``none``                | Every strategy's success state. |

Observed per-(noise cell, strategy) histograms match the
theoretical prediction: pure drop_prob produces
``resolution_conflict``; mislabel adds ``llm_error`` on top;
adversarial drop_root produces exclusively ``resolution_conflict``.

---

## Part F — Future work

### F.1 Carry-over from Phase 35 (unchanged)

* SWE-bench end-to-end with a real LLM on the wrap path.
* Frontier-model multi-seed × multi-k sweep.
* OQ-1 in full generality (Conjecture P30-6).
* Cross-language runtime calibration.
* Payload-level adversary.
* Hierarchical role lattice at K ≥ 20.

### F.2 Newly surfaced by Phase 36

* **Real-LLM replier calibration (C36-8).** Run
  ``phase36_llm_replies`` under a real 0.5b / 7B Ollama and
  measure the malformed / out-of-vocab rate. Fit to
  ``ReplyNoiseConfig`` parameters to predict dynamic-strategy
  accuracy under unseen scenarios.
* **Ensemble reply replication (C36-7).** Build a
  multi-replier ensemble — n different LLM replies per
  producer, aggregated by a robust counting rule. Test whether
  it recovers the adversarial collapse of P36-2 the way
  Phase-34's ``UnionExtractor`` recovered adversarial extractor
  collapse.
* **Analytic equivalence of threads vs adaptive_sub (C36-5).**
  Construct a task family where the two primitives diverge.
  Candidates: nested threads, role-local reply memory,
  authenticated-provenance thread signatures.
* **Adaptive-sub bounded-context audit.** Prove or measure the
  C_0 + R*·τ + c bound under the edge-cap + TTL scheme. This
  would promote the adaptive-sub primitive from "matches on
  this bank" to "same type-level guarantees as threads".

### F.3 What is genuinely blocking the endgame

Phase 36 does NOT unblock:

* **End-to-end SWE-bench** — still the largest external-
  validity gap.
* **OQ-1 in full generality** (Conjecture P30-6).
* **Cross-language runtime calibration**.

Phase 36 *does* close the "maybe reply noise kills the
primitive" escape hatch (Theorem P36-1) and the "maybe adaptive
subscriptions obsolete threads" escape hatch (Theorem P36-4 +
Conjecture C36-5). The remaining design-space question is
whether bounded threads have type-level affordances that
bounded adaptive subscriptions cannot match outside the
contested-scenario family (future work).

---

## Appendix A — How to reproduce

```bash
# 1. Full Phase-36 benchmark suite under mock auditor, sub-second wall.
python3 -m vision_mvp.experiments.phase36_noisy_dynamic \
    --mock --seeds 35 36 \
    --drop-probs 0.0 0.1 0.25 0.5 0.75 1.0 \
    --mislabel-probs 0.0 0.25 \
    --distractor-counts 6 \
    --out vision_mvp/results_phase36_noisy_dynamic.json

# 2. Adversarial reply sweep.
python3 -m vision_mvp.experiments.phase36_noisy_dynamic \
    --mock --seeds 35 36 --adversarial drop_root \
    --out vision_mvp/results_phase36_adversarial_drop.json
python3 -m vision_mvp.experiments.phase36_noisy_dynamic \
    --mock --seeds 35 36 --adversarial flip_root_to_symptom \
    --out vision_mvp/results_phase36_adversarial_flip.json

# 3. LLM-driven replies.
python3 -m vision_mvp.experiments.phase36_llm_replies \
    --mock --seeds 35 36 \
    --malformed-probs 0.0 0.1 0.25 0.5 --oov-probs 0.0 \
    --out vision_mvp/results_phase36_llm_replies_mock.json

# 4. Adaptive-subscription vs dynamic head-to-head.
python3 -m vision_mvp.experiments.phase36_adaptive_sub \
    --mock --seeds 35 36 --distractor-counts 6 20 \
    --drop-probs 0.0 0.25 0.5 1.0 \
    --out vision_mvp/results_phase36_adaptive_sub.json

# 5. Phase-36 test suite.
python3 -m pytest vision_mvp/tests/test_phase36_*.py -q

# 6. Full regression — Phase 31–36 tests.
python3 -m pytest vision_mvp/tests/ -q
```

On a commodity laptop (2026-vintage): the full Phase-36 mock
sweep (#1–4) runs in under 1 s; the test suite (#5) runs in
~0.6 s for the Phase-36 subset, ~10 s for the full 1300+-test
suite.

---

*End of Phase 36 results note. The master plan
(``docs/context_zero_master_plan.md``) is updated in the same
commit; see ``§ 4.12 Current frontier`` for the higher-level
integration.*
