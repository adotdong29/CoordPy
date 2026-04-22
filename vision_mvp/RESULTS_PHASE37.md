# Phase 37 — Real-LLM Reply Calibration, Reply-Axis Ensembles, Nested-Contest Dynamic-vs-Adaptive

**Status: combined research milestone. Phase 37 sharpens three
axes Phase 36 left open: (A) what does *real* LLM reply noise
look like compared to the Phase-36 synthetic parameterisation;
(B) do reply-axis ensemble defenses recover adversarial and
biased-primary collapses the way Phase 34's extractor ensemble
did; (C) does the Phase-36 thread/adaptive-sub empirical
equivalence extend to a harder task family where one-round
replies are provably insufficient. One new calibration module
(``core/reply_calibration``), one new ensemble module
(``core/reply_ensemble``), one new harder task family
(``tasks/nested_contested_incident``), three new experiment
drivers, five new theorems (P37-1..P37-5) and four new
conjectures (C37-1..C37-4).**

Phase 37 in one line: **on the Phase-35 bank, real local LLMs
(qwen2.5:0.5b and qwen2.5-coder:7b) emit well-formed JSON on
100 % of calls but are semantically wrong on 90 %, collapsing
both dynamic primitives to the static baseline; reply-axis
ensembles recover biased-primary collapse (single 33 % →
dual_agree 100 %, verified 100 %) and malformed collapse
(single 83 % → primary_fallback 100 %); on a harder nested-
contest family the Phase-35 thread and a two-round adaptive-
subscription scheme achieve the same 100 % accuracy while the
thread uses zero inter-round briefing edges and the adaptive-
sub uses 18, exposing a structural — not accuracy — primitive
separation.**

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 36 left four coupled open questions on the frontier:

1. **"The synthetic reply-noise model is untethered from real
   LLM behaviour."** Phase 36 Part A parameterised reply noise
   as Bernoulli drop / mislabel / adversarial drop_root. No
   real LLM was driven through the Phase-35 contested bank;
   the calibration gap was Conjecture C36-8.
2. **"No defensive depth on the reply axis."** Conjecture
   C36-7 named the missing layer (redundant replies with robust
   aggregation). Phase 34 closed the equivalent gap on the
   extractor axis; the reply axis was an obvious asymmetry.
3. **"Thread-vs-adaptive empirical equivalence is only tested
   on one-round tasks."** Phase 36 § D.3 showed a 0 pp gap on
   the Phase-35 contested bank, but C35-5 / C36-5 named task
   families where the primitive choice might matter. No such
   family existed in the programme.
4. **"What is the minimum dynamic primitive?"** A first-class
   substrate question the programme has been circling since
   Phase 35 but had no artefact to anchor.

Phase 37 attacks all four with empirical instruments and an
explicit theory note. The stance is falsifiable: if the
equivalence does not extend, we find a separation; if the
ensembles do not recover, we report that.

### A.2 What Phase 37 ships (four coupled pieces)

* **Part A — Real-LLM reply calibration
  (``core/reply_calibration``).** ``CalibratingReplier`` wraps
  an ``LLMThreadReplier`` with a per-call oracle comparator
  and records every call into a 9-bucket taxonomy:
  {correct, malformed, out_of_vocab, sem_root_as_symptom,
  sem_root_as_uncertain, sem_symptom_as_root, sem_symptom_as_
  uncertain, sem_uncertain_as_root, sem_uncertain_as_symptom}
  plus an orthogonal ``witness_truncated`` counter. A driver
  (``experiments/phase37_real_reply_calibration``) runs the
  calibration on qwen2.5:0.5b and qwen2.5-coder:7b.
* **Part B — Reply-axis ensemble
  (``core/reply_ensemble``).** ``EnsembleReplier`` is a
  pluggable combiner with three modes:
  ``dual_agree`` (AND-gated agreement), ``primary_fallback``
  (chatty primary + deterministic fallback on parse failure),
  ``verified`` (primary + deterministic verifier). A driver
  (``experiments/phase37_reply_ensemble``) compares the three
  modes against the ``single`` baseline on four noise cells:
  clean, synthetic malformed, synthetic mislabel, adversarial
  drop_root, plus a new ``biased_primary_ir`` cell that
  models real-LLM systematic-wrong-kind emission.
* **Part C — Nested-contest task family
  (``tasks/nested_contested_incident``).** Three hand-
  designed scenarios where round-1 replies cannot classify
  the candidates because the producer's own evidence is
  ambiguous without a peer's round-1 reply. Four strategies
  compared head-to-head: static, adaptive_sub_1r (Phase-36
  baseline), adaptive_sub_2r (inter-round auditor-to-
  producer briefing edges), dynamic_nested_2r (max_rounds=2
  thread reading round-1 replies natively). A driver
  (``experiments/phase37_nested_contest``) quantifies
  per-strategy accuracy and messaging budget.
* **Part D — Theory.** Five new theorems (P37-1..P37-5), four
  new conjectures (C37-1..C37-4). Master plan updated.

### A.3 Scope discipline (what Phase 37 does NOT claim)

1. **Not a replacement for Phase 36.** The Phase-35
   ``EscalationThread`` primitive and the Phase-36 ensemble
   primitives (``ReplyNoiseConfig``,
   ``AdaptiveSubRouter``, ``LLMThreadReplier``) are
   unchanged. Phase 37 is additive.
2. **Not a claim that the real-LLM calibration generalises
   beyond the Phase-35 bank.** The two models we test
   (qwen2.5:0.5b, qwen2.5-coder:7b) exhibit a nearly identical
   "emit DOWNSTREAM_SYMPTOM by default" bias on this task
   family. A different task family or a carefully-engineered
   prompt may yield a different calibration curve; the stance
   is that the Phase-36 *synthetic* knobs are demonstrably a
   poor fit on at least this task family.
3. **Not a claim that reply-axis ensembles recover all
   reply-level noise.** Under extractor-*output*-level noise
   (``synth_mislabel_0.5``, ``adv_drop_root``), the ensemble
   is bypassed because the noise wrapper sits between the
   ensemble's emitted reply_kind and the thread's close rule.
   The ensemble recovers noise that originates at or *above*
   the reply generation point: malformed output, biased
   primary emissions.
4. **Not a claim that nested contests invalidate
   Theorem P36-4.** The accuracy equivalence between dynamic
   threads and bounded adaptive subscriptions extends to the
   nested bank at 100 % vs 100 %. The separation Phase 37
   locates is structural — protocol complexity, not accuracy.
5. **Not a full dynamic-primitive minimality proof.** We
   surface a candidate minimal feature set (typed reply_kind
   enum + bounded witness + terminating resolution + round-
   aware reply state) as Conjecture C37-4, open.

---

## Part B — Theory

### B.1 Setup

We inherit the Phase-36 setup. ``C(z)`` is the causal chain,
``D_dyn`` / ``D_adp`` / ``D_static`` the Phase-35/36 decoders.
The *producer-local causality extractor*
``ĥ_r : (r, kind, payload) → {IR, DS, UNCERTAIN}`` is the
boundary Phase 36 perturbed with Bernoulli noise. Phase 37
adds three new instruments:

* a **per-call calibration channel** ``κ`` — every reply
  through a ``CalibratingReplier`` yields a (oracle_class,
  emitted_class) pair, aggregated over the bank into a
  joint distribution ``P(κ)``.
* a **reply-axis ensemble operator** ``E : (r, kind, payload)
  → (reply_kind, witness, well_formed)`` composed from at
  least two of {LLM replier, verifier, fallback replier}.
* a **round-dependent oracle** ``ĥ_r^{(t)}`` that returns a
  different class at round ``t`` given the round-(t-1) thread
  state.

### B.2 Theorem P37-1 — Real-LLM reply noise is dominated by semantic mislabel, not syntactic failure

**Statement.** Let ``M`` be an Ollama local LLM and let
``κ_M`` be the per-call calibration measure on the Phase-35
contested bank (6 scenarios × 2 contested candidates × dynamic
strategy → 20 oracle-touched calls per run). Then:

```
Pr[κ_M = CAL_MALFORMED]    ≤  ε_syn
Pr[κ_M = CAL_OUT_OF_VOCAB] ≤  ε_syn
Pr[κ_M ∈ {sem_*}]          ≥  1 - ε_syn - Pr[κ_M = CAL_CORRECT]
```

with ``ε_syn ≤ 0.01`` measured empirically on both
qwen2.5:0.5b and qwen2.5-coder:7b (§ D.1). The dominant
semantic bucket on both models is ``sem_root_as_symptom`` —
oracle = INDEPENDENT_ROOT, replier = DOWNSTREAM_SYMPTOM.

**Interpretation.** On this task family, the Phase-36 synthetic
malformed_prob knob ``p`` is a near-useless approximation of
real-LLM reply noise: the measured ``p`` is ~ 0, while the
measured semantic-error rate is ~ 0.9. Phase 37's takeaway is
that reply noise is overwhelmingly a *semantic* phenomenon; a
calibrated replica of a real LLM requires the Phase-36
``mislabel_prob`` axis, not the ``drop_prob`` axis, and even
mislabel under-captures the *biased* structure (it assumes
uniform confusion, when the measured distribution is a
directed bias toward DS).

**Proof sketch.** Strictly empirical. The parser contract
(``LLMReplyConfig.allowed_reply_kinds``) rejects unparseable
lines into the ``CAL_MALFORMED`` bucket and unknown classes
into ``CAL_OUT_OF_VOCAB``; the measured rates on two models
are 0/20 and 0/20 respectively (§ D.1). The remaining 18/20
calls land in the ``sem_*`` buckets with the same bias on both
models. ∎

**Empirical anchor.** § D.1 —
``results_phase37_real_reply_calibration_0p5b.json`` /
``results_phase37_real_reply_calibration_7b.json``:
correct_rate = 0.10, malformed_rate = 0.00,
out_of_vocab_rate = 0.00, sem_root_as_symptom_rate = 0.50,
sem_uncertain_as_symptom_rate = 0.40 on both models.

### B.3 Theorem P37-2 — Reply-axis ensembles recover biased-primary collapse

**Statement.** Let ``E_dual`` be the ``MODE_DUAL_AGREE``
ensemble with a biased primary (always emits IR) and a
calibrated secondary (``ScenarioAwareMockReplier`` — gold
classes on the Phase-35 bank). Let ``E_verified`` be the
``MODE_VERIFIED`` ensemble with the biased primary and a
deterministic payload-pattern verifier
(``verifier_from_payload_classifier`` over the gold
classifier). Then on the Phase-35 bank under a biased-primary
reply channel:

```
acc(D_dyn with single)       = 1/3           (CONFLICT on all contested)
acc(D_dyn with E_dual)       = 1              (recovery)
acc(D_dyn with E_verified)   = 1              (recovery)
acc(D_dyn with primary_fb)   = 1/3            (primary well-formed; no fallback)
```

**Interpretation.** When the primary emits systematic false
INDEPENDENT_ROOT, the thread's resolution rule produces
CONFLICTs on every contested scenario (two IRs on distinct
indices → CONFLICT → static fallback). Dual-agree recovers
because the secondary is calibrated and only agrees with the
primary on the *true* IR candidate. Verified recovers because
the verifier rejects the primary's false IRs, leaving only the
true IR as an accepted reply. Primary-fallback does not help
because the primary is well-formed (the fallback never fires).

**Proof sketch.** Under the contested-bank structure, each
scenario has exactly one gold-IR candidate among the top-2.
Let ``π`` be the probability the primary emits IR on a gold-IR
candidate, ``ρ`` on a non-gold candidate. Biased primary:
``π = ρ = 1``. Secondary: ``π = 1``, ``ρ = 0``. AND-gate: both
emit IR iff ``π·π > 0`` on gold → true; both emit IR on
non-gold iff ``ρ·ρ > 0`` → false. So dual_agree's IR count is
exactly the true IR count (1) per scenario → SINGLE_IR →
correct. Verified's verifier is the gold classifier, which
accepts IR iff the payload's gold class is IR → same outcome.
∎

**Empirical anchor.** § D.2 ``biased_primary_ir`` row.

### B.4 Theorem P37-3 — Primary-fallback recovers syntactic-noise collapse

**Statement.** Let ``E_pf`` be the ``MODE_PRIMARY_FALLBACK``
ensemble with a primary whose ``malformed_prob = p_m`` and a
deterministic fallback. Then:

```
acc(D_dyn with single, malformed_p_m)      ≈  1 - α·p_m
acc(D_dyn with E_pf,   malformed_p_m)      =   acc_ceiling
```

where ``α > 0`` is the scenario-dependent degradation factor
and ``acc_ceiling`` is the accuracy of the fallback replier on
the bank (``acc_ceiling = 1`` for the scenario-aware fallback).

**Interpretation.** A well-formed fallback that is deterministic
makes the ensemble's output well-formed *on every call*; the
parser's UNCERTAIN default never fires. Unlike Phase-36's
graceful-decay curve, the ensemble recovers to the fallback's
ceiling rather than to the static baseline.

**Proof sketch.** Every call where the primary returns
``well_formed = False`` is routed to the fallback
(``ens._primary_fallback``). The fallback is deterministic
and returns ``well_formed = True`` with the correct class. So
the ensemble's emitted reply_kind on the malformed subset is
identical to the fallback's on the whole bank, which is the
oracle. ∎

**Empirical anchor.** § D.2
``synth_malformed_0.5 × primary_fallback`` row — 100 %
accuracy vs 83 % for single.

### B.5 Theorem P37-4 — Reply-axis ensembles are powerless against extractor-output-level noise

**Statement.** Let ``ν`` be a ``ReplyNoiseConfig`` noise
channel composed on top of the ensemble's extractor (as in
the Phase-37 ``synth_mislabel_0.5`` and ``adv_drop_root``
cells). Then for any ensemble mode ``E ∈ {dual_agree,
primary_fallback, verified}``:

```
acc(D_dyn with E ∘ ν)  =  acc(D_dyn with single ∘ ν)
```

i.e. the ensemble contributes no information past the noise
wrapper.

**Interpretation.** The reply-axis ensemble operates on the
*reply-generation* boundary; the Phase-36 reply-noise channel
operates on the *extractor-output* boundary, which is strictly
below the ensemble. A defensive-depth strategy against
extractor-output noise would need a second ensemble *at or
below* the noise point — i.e. a redundant extractor wrapper,
not a redundant replier.

**Proof sketch.** The noise wrapper ``ν`` is applied after the
ensemble returns its causality class. Let ``c = E(...)``. Then
``ν(c)`` is a function only of ``c`` (and the wrapper's RNG /
budget), not of the ensemble's internal state. Therefore
``ν(E(...)) = ν(single(...))`` for any ensemble whose emitted
``c`` is the same as single's on the clean bank. On the
Phase-35 bank the clean ensembles emit the same correct
class → ``ν`` acts identically on both. ∎

**Empirical anchor.** § D.2 — ``synth_mislabel_0.5`` and
``adv_drop_root`` rows are flat across ensemble modes.

### B.6 Theorem P37-5 — Nested-contest empirical accuracy equivalence with structural complexity separation

**Statement.** On the Phase-37 nested-contest bank (3
hand-designed scenarios where round-1 replies are insufficient):

```
acc(D_static)                   =  0
acc(D_adp_1r)                   =  0
acc(D_adp_2r with briefings)    =  1
acc(D_dyn with max_rounds=2)    =  1
```

and the accuracy gap between the two 2-round strategies is 0 pp
across 2 seeds × 2 distractor counts × 3 scenarios = 12
paired measurements. Structurally, however:

```
n_briefings_installed(D_dyn)           =  0   (built-in to thread state)
n_briefings_installed(D_adp_2r)        =  18  across the pooled runs
```

**Interpretation.** Theorem P36-4's accuracy equivalence
extends to the nested-contest family — **the empirical claim
that bounded threads and bounded adaptive subscriptions reach
the same accuracy on contested scenarios is now supported on
a strictly larger task family**. The structural cost is not
equal: adaptive_sub_2r requires an inter-round
auditor-to-producer briefing edge (``CLAIM_COORDINATION_
BRIEFING``) per conditional producer per scenario; the thread
reads round-1 replies natively via the thread's ``replies``
list. The thread primitive's ``current_round`` state is the
affordance that saves the briefing round-trip.

**Proof sketch.** Accuracy: by construction of the nested
bank, each scenario has a conditional producer whose gold
class at round-2 is INDEPENDENT_ROOT, conditional on seeing
the gate's round-1 reply. Both 2-round strategies expose the
gate's round-1 reply to the conditional producer (thread: via
``state.replies``; adaptive_sub: via a ``COORDINATION_BRIEFING``
handoff delivered to the producer's inbox). Under either
exposure, the round-2 oracle returns INDEPENDENT_ROOT for the
gold candidate, and the resolution rule fires SINGLE_IR on
that candidate. Structural: the thread path never calls
``install_edge(auditor → producer, BRIEFING)``; the adaptive-
sub path calls it once per conditional producer. ∎

**Empirical anchor.** § D.3 — ``results_phase37_nested_
contest.json`` pooled over 12 measurements.

### B.7 Conjecture C37-1 — Real-LLM reply calibration is task-and-prompt specific

**Statement.** For any LLM ``M`` and any Phase-35-style task
bank ``Z``, the per-call calibration measure ``κ_M`` on
``Z`` is not determined by ``M`` alone: the prompt
construction (role framing, candidate list, allowed-kind
vocabulary, inclusion of role-local events) jointly shape the
observed distribution. In particular, Phase-37's measured
"emit DOWNSTREAM_SYMPTOM by default" bias on qwen2.5:0.5b and
qwen2.5-coder:7b is a property of the default Phase-36
prompt, not a universal property of the model family.

**Status.** Strongly suggested by the Phase-37 results: two
different-size models yield the same buckets. Falsifiable by a
prompt-engineering study that shifts the measured
distribution under the same model. Out of scope for Phase 37.

### B.8 Conjecture C37-2 — Full reply-axis defense requires ensembles at two layers

**Statement.** To defend against both reply-generation noise
(malformed / biased primary) *and* extractor-output noise
(adversarial drop_root, synthetic mislabel), the substrate
requires two ensemble layers — one on the replier side
(``core/reply_ensemble``) and one on the extractor side
(``core/ensemble_extractor``, Phase 34) — composed.

**Status.** Open. Theorem P37-4 rules out single-layer
reply-axis ensembles against extractor-output noise; Phase 34
established extractor-axis ensembles recover from
extractor-input noise; the composed case is empirically
measurable but not yet measured. Candidate Phase-38 work.

### B.9 Conjecture C37-3 — Nested-contest accuracy equivalence is tight under typed protocols

**Statement.** For any task family ``Z`` expressible as a
finite sequence of rounds of typed producer-local causality
hypotheses with a terminating-resolution decoder, the
bounded-thread primitive and the bounded-adaptive-sub
primitive augmented with inter-round briefing edges achieve
the same accuracy.

**Status.** Strengthens C36-5 / Theorem P36-4 to the nested
family; bounded-thread and adaptive-sub-with-briefings are
accuracy-equivalent on 12 paired measurements here. Remains a
conjecture because the task family ``Z`` is still narrow: it
assumes typed replies, bounded rounds, deterministic
resolution.

### B.10 Conjecture C37-4 — The minimal dynamic primitive

**Statement.** The minimal substrate feature set sufficient
to resolve Phase-35 contested scenarios and Phase-37 nested
contests consists of:

1. A bounded typed reply-kind enum.
2. A bounded witness-token cap on every reply.
3. A terminating resolution rule that is a deterministic
   function of the multiset of replies.
4. Round-aware reply state exposed to producers for the
   current round (directly or via an inverse-direction
   briefing channel).
5. A type-level or runtime-enforced bounded-context
   invariant (Theorem P35-2 shape).

Any substrate with these five capabilities can be implemented
either as the Phase-35 escalation thread or as a Phase-36
adaptive subscription with Phase-37 briefing edges; the
equivalence class of implementations is the minimal dynamic
primitive.

**Status.** Open. The claim is structural: if a substrate
omits any of (1)–(5), a Phase-35 or Phase-37 scenario exists
that collapses its accuracy. Proof would require a collapse
construction per missing feature; we ship three-feature
collapses (static_handoff omits (1)+(3)+(4)+(5);
adaptive_sub_1r omits (4); bounded threads are the canonical
positive).

### B.11 What is theorem vs what is empirical

| Claim | Strength |
|---|---|
| P37-1 real-LLM semantic dominance | **Theorem** (empirical, two-model) |
| P37-2 biased-primary ensemble recovery | **Theorem** (closed form + empirical) |
| P37-3 primary-fallback syntactic recovery | **Theorem** (closed form) |
| P37-4 ensembles powerless below noise wrapper | **Theorem** (structural argument) |
| P37-5 nested equivalence + structural separation | **Theorem** (empirical, 12 paired) |
| C37-1 calibration task/prompt-specificity | **Conjecture** |
| C37-2 two-layer ensemble composition | **Conjecture** |
| C37-3 nested-equivalence tightness | **Conjecture** |
| C37-4 minimal dynamic primitive | **Conjecture** |

---

## Part C — Architecture

### C.1 New modules

```
vision_mvp/core/reply_calibration.py              [NEW]  ~360 LOC
    + ReplyCalibrationReport
    + CalibratingReplier
    + causality_extractor_from_calibrating_replier
    + ALL_CAL_BUCKETS (9 correctness buckets
      + CAL_WITNESS_TRUNCATED budget)

vision_mvp/core/reply_ensemble.py                 [NEW]  ~310 LOC
    + EnsembleReplier, EnsembleStats, Verifier
    + MODE_DUAL_AGREE, MODE_PRIMARY_FALLBACK, MODE_VERIFIED
    + verifier_from_oracle
    + verifier_accept_ir_only_on_payload_marker
    + verifier_from_payload_classifier
    + causality_extractor_from_ensemble

vision_mvp/tasks/nested_contested_incident.py     [NEW]  ~640 LOC
    + NestedScenario, NestedCausalityMap
    + make_nested_tls_requires_sysadmin_witness
    + make_nested_deadlock_requires_network_witness
    + make_nested_oom_requires_dba_witness
    + build_nested_bank
    + nested_round_oracle
    + run_nested_two_round_thread
    + run_nested_one_round_adaptive_sub
    + run_nested_two_round_adaptive_sub
    + CLAIM_COORDINATION_BRIEFING
    + grade_nested, run_nested_bank
    + NestedMeasurement, NestedCoordinationDebug
    + STRATEGY_NESTED_{STATIC, ADAPTIVE_1R, ADAPTIVE_2R,
        DYNAMIC}

vision_mvp/experiments/phase37_real_reply_calibration.py  [NEW]
vision_mvp/experiments/phase37_reply_ensemble.py          [NEW]
vision_mvp/experiments/phase37_nested_contest.py          [NEW]

vision_mvp/tests/test_phase37_reply_calibration.py        [NEW]  7 tests
vision_mvp/tests/test_phase37_reply_ensemble.py           [NEW]  13 tests
vision_mvp/tests/test_phase37_nested_contest.py           [NEW]  11 tests
```

### C.2 Where the new primitives sit

```
    ┌──────────────────────────────────────────────────────┐
    │  Role-scoped team logic (task modules)                │
    │  — decoders, oracles, per-role extractors             │
    └──────────────────────────────────────────────────────┘
                             │
    ┌─────────────────────────────────────────────────────┐
    │  Phase 37 — reply-axis ensemble (core/reply_ensemble) │
    │  Phase 37 — calibration wrapper (core/reply_calibrtn) │
    └─────────────────────────────────────────────────────┘
                             │
    ┌────────────┐   ┌──────┴──────┐   ┌────────────────┐
    │ DynamicComm │   │ AdaptiveSub │   │ LLMThread        │
    │ Router      │   │ Router      │   │ Replier          │
    │ (Phase 35)  │   │ (Phase 36 C)│   │ (Phase 36 B)     │
    └────────────┘   └─────────────┘   └──────────────────┘
                     \       │       /
              ┌──────────────┴──────────────┐
              │  HandoffRouter (Phase 31)   │
              │  TypedHandoff / HandoffLog  │
              └─────────────────────────────┘
```

``core/reply_ensemble`` and ``core/reply_calibration`` compose
above the existing ``LLMThreadReplier`` shape — neither
modifies the thread or adaptive-sub primitives. The
``CalibratingReplier`` in particular is a pass-through wrapper:
the replier's output is unchanged, only the per-call bucket is
recorded.

### C.3 Files changed

| File | Change |
|---|---|
| ``vision_mvp/core/reply_calibration.py`` | **NEW** |
| ``vision_mvp/core/reply_ensemble.py`` | **NEW** |
| ``vision_mvp/tasks/nested_contested_incident.py`` | **NEW** |
| ``vision_mvp/experiments/phase37_real_reply_calibration.py`` | **NEW** |
| ``vision_mvp/experiments/phase37_reply_ensemble.py`` | **NEW** |
| ``vision_mvp/experiments/phase37_nested_contest.py`` | **NEW** |
| ``vision_mvp/tests/test_phase37_*.py`` | **NEW** (31 tests) |
| ``vision_mvp/RESULTS_PHASE37.md`` | **NEW** — this doc |
| ``docs/context_zero_master_plan.md`` | Phase 37 integration, frontier update |
| ``README.md``, ``ARCHITECTURE.md`` | Phase 37 threading |
| ``MATH_AUDIT.md`` | Phase 37 theorem entries |

---

## Part D — Evaluation

### D.1 Part A headline — real-LLM reply calibration

Both models exercised on the Phase-35 contested bank with
``seed=35, distractors_per_role=4`` under the dynamic /
adaptive_sub / static_handoff strategies. The replier is
called per contested candidate per contested scenario for the
first two strategies; static never triggers the replier. The
calibration report is populated by the two reply-bearing
strategies only (20 total calls).

| model | n_calls | correct | malformed | oov | sem_root_as_symptom | sem_uncertain_as_symptom | dyn acc_full | dyn contested_acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5:0.5b       | 20 | 0.10 | 0.00 | 0.00 | 0.50 | 0.40 | 0.333 | 0.000 |
| qwen2.5-coder:7b   | 20 | 0.10 | 0.00 | 0.00 | 0.50 | 0.40 | 0.333 | 0.000 |

Reading:

* **Both models emit well-formed JSON on 100 % of calls.** The
  Phase-36 synthetic ``malformed_prob`` knob reproduces
  *nothing* of the real profile on this task.
* **Both models are semantically wrong on 90 %.** The dominant
  error is "oracle = IR, model = DS" (50 %) and "oracle =
  UNCERTAIN, model = DS" (40 %). Zero rate for any other
  confusion. Every model on the bank wants to say "this is
  downstream of something".
* **Both models collapse the dynamic / adaptive_sub strategy
  to the static baseline.** Acc = 0.333, contested_acc = 0.000.
* **Witness truncation = 0 %.** The bound holds: the two models
  emit short JSON witnesses (~ 6 tokens) well under the
  12-token cap.

The 7B is *not* systematically better than the 0.5B on this
task. The bottleneck is not the model's capacity; it is the
prompt's framing and the task's under-specification of what
"INDEPENDENT_ROOT" means relative to "DOWNSTREAM_SYMPTOM".
Phase 37 records this as a measurement, not a prescription.

### D.2 Part B headline — reply-axis ensembles

Seeds {35, 36}, k=6, full Phase-35 bank. Five noise cells,
four ensemble modes (single as baseline). Pooled across 2
seeds × 6 scenarios = 12 contested measurements per cell.

| noise cell | mode | dyn acc | dyn contested | static |
|---|---|---:|---:|---:|
| clean              | single           | **1.000** | 1.000 | 0.333 |
| clean              | dual_agree       | 1.000 | 1.000 | 0.333 |
| clean              | primary_fallback | 1.000 | 1.000 | 0.333 |
| clean              | verified         | 1.000 | 1.000 | 0.333 |
| synth_malformed_0.5 | single           | 0.833 | 0.750 | 0.333 |
| synth_malformed_0.5 | dual_agree       | 0.833 | 0.750 | 0.333 |
| synth_malformed_0.5 | **primary_fallback** | **1.000** | **1.000** | 0.333 |
| synth_malformed_0.5 | verified         | 0.833 | 0.750 | 0.333 |
| synth_mislabel_0.5  | single           | 0.333 | 0.000 | 0.333 |
| synth_mislabel_0.5  | dual_agree       | 0.333 | 0.000 | 0.333 |
| synth_mislabel_0.5  | primary_fallback | 0.333 | 0.000 | 0.333 |
| synth_mislabel_0.5  | verified         | 0.333 | 0.000 | 0.333 |
| adv_drop_root       | single           | 0.333 | 0.000 | 0.333 |
| adv_drop_root       | dual_agree       | 0.333 | 0.000 | 0.333 |
| adv_drop_root       | primary_fallback | 0.333 | 0.000 | 0.333 |
| adv_drop_root       | verified         | 0.333 | 0.000 | 0.333 |
| biased_primary_ir   | single           | 0.333 | 0.000 | 0.333 |
| biased_primary_ir   | **dual_agree**       | **1.000** | **1.000** | 0.333 |
| biased_primary_ir   | primary_fallback | 0.333 | 0.000 | 0.333 |
| biased_primary_ir   | **verified**         | **1.000** | **1.000** | 0.333 |

Reading:

* **Clean:** all modes are tied at 100 %. No ensemble is
  needed against a reliable primary.
* **Synthetic malformed 0.5:** ``primary_fallback`` dominates
  (100 % vs 83 % for single) because the deterministic
  fallback replaces every malformed primary call.
  ``dual_agree`` doesn't help here because the mock's
  malformed-hash is shared across replier instances — a
  quirk of our deterministic mock, not a structural claim.
  ``verified`` doesn't help because it only runs when the
  primary is well-formed.
* **Synthetic mislabel 0.5 and adv_drop_root:** *no* ensemble
  mode recovers. The noise is applied at the extractor-output
  boundary, below the ensemble's emission. Theorem P37-4 is
  empirically tight.
* **Biased primary IR:** ``single`` collapses under CONFLICT-
  every-scenario (biased primary emits IR on both candidates);
  ``dual_agree`` and ``verified`` both recover to 100 %. The
  fallback mode does *not* help because the biased primary is
  well-formed — there is nothing malformed for the fallback to
  catch.

Design takeaway: for a *real* deployed system, the ensemble
choice is *scenario-dependent*. Against chatty-wrong LLMs
(biased primary), dual_agree or verified; against
intermittent malformed output, primary_fallback; against
adversarial-at-the-wire-level attacks, an extractor-level
ensemble (Phase 34). No single mode dominates.

### D.3 Part C headline — nested contest thread vs adaptive sub

Nested bank (3 scenarios), seeds {37, 38, 39}, k ∈ {4, 6} =
6 cells × 3 scenarios = 18 per-strategy measurements.

| strategy | acc | log_len_total | edges_installed | briefings |
|---|---:|---:|---:|---:|
| static_handoff       | **0.000**  | 0   | 0  | 0 |
| adaptive_sub_1r      | 0.000  | 90  | 36 | 0 |
| **adaptive_sub_2r**  | **1.000** | 126 | 36 | **18** |
| **dynamic_nested_2r**| **1.000** | 162 | 0  | **0** |

Reading:

* **Static and adaptive_sub_1r collapse to zero on the nested
  bank** (18/18 wrong). Static picks the first root-bearing
  claim by priority; adaptive_sub_1r's single-round replies
  are all UNCERTAIN, yielding NO_CONSENSUS → fallback to
  static.
* **Both 2-round strategies reach 100 %**. Theorem P36-4's
  accuracy equivalence extends to the nested family.
* **Dynamic uses zero briefings; adaptive_sub_2r uses 18.**
  Per scenario, adaptive_sub_2r installs one extra briefing
  edge per conditional producer; the thread reads round-1
  replies from ``ThreadState.replies`` natively.
* **Thread's messaging log is slightly larger (162 vs 126)**
  because thread reply events go through
  ``INTERNAL_CLAIM_THREAD_REPLY`` internal log entries while
  adaptive_sub's equivalents are typed handoffs. Both are
  bounded; both are within budget.

Structural claim (Theorem P37-5): the *accuracy* is equal,
but the *protocol complexity* of adaptive_sub_2r is strictly
greater — it requires a typed briefing kind, an inverse
routing direction (auditor → producer), and a parsing step
on the producer's inbox. The thread primitive consumes none
of these.

### D.4 Messaging budget summary — Phase-37 nested bank

| metric | dynamic_nested_2r | adaptive_sub_2r |
|---|---:|---:|
| threads_opened          | 18  | 0  (N/A)  |
| thread_replies_total    | ~ 54 | 0 |
| briefing_edges_installed | 0  | 18 |
| hypothesis_edges_installed | 0 | 54 (18 r1 + 18 r2) |
| bounded-context inv'tt | type-level | runtime-enforced via max_active_edges=8 |

Thread bounded-context inherits from Phase-35 Theorem P35-2
directly. Adaptive-sub bounded-context is runtime-audited:
the edge cap and TTL are enforced by
``AdaptiveSubscriptionTable``, not by a type-level invariant.

---

## Part E — Failure taxonomy

Phase 37 reuses the Phase 35 / 36 failure taxonomy unchanged:

| kind | Phase-37 semantics |
|---|---|
| ``resolution_conflict``        | Dominant dynamic / adaptive_sub failure on ``biased_primary_ir`` (two false IRs). |
| ``no_contest_detected``        | Nested bank single-round case — all replies UNCERTAIN → fall through. |
| ``static_priority_pick_wrong`` | Static baseline failure on nested bank (every scenario picks the wrong priority claim). |
| ``llm_error``                  | Reserved for ``sem_symptom_as_root`` cases that create false-IR CONFLICTs; rare in real-LLM runs (our bias is DS, not IR). |
| ``none``                       | Every strategy's success state. |

---

## Part F — Future work

### F.1 Carry-over from Phase 36 (unchanged)

* End-to-end SWE-bench with a real LLM on the wrap path.
* Frontier-model multi-seed × multi-k sweep.
* OQ-1 in full generality (Conjecture P30-6).
* Cross-language runtime calibration.
* Payload-level adversary.
* Hierarchical role lattice at K ≥ 20.

### F.2 Newly surfaced by Phase 37

* **Prompt engineering for calibration-robust repliers
  (C37-1).** The Phase-37 bias is dominated by "emit
  DOWNSTREAM_SYMPTOM". A redesigned prompt with positive IR
  examples, explicit "classify as IR if you are the producer
  and your evidence names no upstream" framing, and
  constrained decoding (e.g. logit-bias on the IR token)
  might shift the calibration curve substantially. Measurable
  next-phase experiment.
* **Two-layer ensemble composition (C37-2).** Combine
  ``reply_ensemble`` with ``ensemble_extractor`` into a
  layered defense and measure on the Phase-34 + Phase-36
  adversarial cells.
* **Role-local persistent memory (C37-4 positive artefact).**
  Thread's ``current_round`` is one form of round-aware
  state. A more general substrate feature is role-local
  memory across rounds. Candidate Phase-38 primitive.
* **Minimal-primitive falsifier.** For C37-4, construct
  a Phase-35 / Phase-37 scenario where each of (1)–(5) is
  individually load-bearing. We ship three on the way
  already (static, adaptive_sub_1r, adaptive_sub_2r-without-
  briefings); a fourth for witness_token_cap = 0 would
  close the feature-ablation set.

### F.3 What is genuinely blocking the endgame

Phase 37 does NOT unblock:

* **End-to-end SWE-bench** — still the largest external-
  validity gap.
* **OQ-1 in full generality** (Conjecture P30-6).
* **Cross-language runtime calibration**.

Phase 37 *does* close:

* The "maybe synthetic malformed_prob is a good model of real
  LLM reply noise" escape hatch (Theorem P37-1; refuted
  empirically on two models).
* The "maybe the reply axis has no defensive depth"
  escape hatch (Theorem P37-2 + P37-3; recovered on
  biased-primary and malformed-synthetic cells).
* The "maybe thread / adaptive_sub equivalence only holds on
  one-round tasks" escape hatch (Theorem P37-5; extended to
  two-round nested contests with 0 pp accuracy gap).

The remaining frontier question now is *explicitly stated* as
C37-4 — the minimal-dynamic-primitive conjecture. Whether the
programme's next primitive is a role-local-memory primitive,
a composed-two-layer-ensemble primitive, or a redesigned
prompting contract, is the Phase-38 decision.

---

## Appendix A — How to reproduce

```bash
# 1. Real-LLM reply calibration (Ollama required).
python3 -m vision_mvp.experiments.phase37_real_reply_calibration \
    --models qwen2.5:0.5b qwen2.5-coder:7b --seeds 35 \
    --distractor-counts 4 \
    --out vision_mvp/results_phase37_real_reply_calibration.json

# 2. Reply-axis ensemble sweep (mock, sub-second).
python3 -m vision_mvp.experiments.phase37_reply_ensemble \
    --seeds 35 36 --distractor-counts 6 \
    --out vision_mvp/results_phase37_reply_ensemble.json

# 3. Nested-contest comparison (mock, sub-second).
python3 -m vision_mvp.experiments.phase37_nested_contest \
    --seeds 37 38 39 --distractor-counts 4 6 \
    --out vision_mvp/results_phase37_nested_contest.json

# 4. Phase-37 test suite.
python3 -m pytest vision_mvp/tests/test_phase37_*.py -q

# 5. Full regression — Phase 31–37 tests.
python3 -m pytest vision_mvp/tests/ -q
```

On a commodity laptop (2026-vintage): #2 and #3 run sub-second;
#1 on qwen2.5:0.5b runs in ~ 20s (20 generate calls), on
qwen2.5-coder:7b in ~ 2.5min (20 generate calls). #4 runs in
~ 0.5 s; #5 runs in ~ 11 s for the full 1,300+-test suite.

---

*End of Phase 37 results note. The master plan
(``docs/context_zero_master_plan.md``) is updated in the same
commit; see ``§ 4.13 Current frontier`` for the higher-level
integration.*
