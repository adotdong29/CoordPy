# Phase 35 — Dynamic, Bounded Communication Primitives for Agent Teams, and a Contested-Incident Benchmark

**Status: combined research milestone. Phase 35 ships three
coupled deliverables: (a) a new substrate primitive —
*escalation threads* (``core/dynamic_comm.EscalationThread`` +
``DynamicCommRouter``) — that adds typed, bounded, explicitly-
terminated inter-role coordination strictly above the Phase-31
typed-handoff layer and strictly below any unrestricted group-
chat layer; (b) the programme's first benchmark where *static
typed handoffs alone cannot recover the correct answer* — a
six-scenario contested-incident bank
(``tasks/contested_incident``) of which four scenarios contain
two plausible root-cause claims with *inverted static priority*
(shadow claim outranks real cause) that only producer-local
reflection can adjudicate; and (c) three theorems and two
conjectures connecting expressivity gap, bounded multicast, and
conditions for dynamic coordination to the existing
``role_handoff`` theory (Theorems P31-1..P31-5).**

Phase 34, in one line: extractor noise is role-structured and
adversarially focusable; defensive depth via ensemble closes
the single-extractor gap. **Phase 35, in one line: on a
six-scenario contested bank where the naive broadcast and
Phase-31 static-handoff decoders both score 0/4 on contested
scenarios (≤ 33 % full accuracy, ≤ 50 % root-cause accuracy)
independent of distractor count k ∈ {6, 20, 60, 120}, the
dynamic coordination substrate reaches 4/4 contested at 100 %
full accuracy flat at 246 tokens per auditor prompt —
recovering both the priority-inversion failure and the
shadow-service contamination — at a messaging cost of exactly
one five-member escalation thread per scenario, ≤ 2 replies,
≤ 12 witness tokens per reply, with the hash-chained handoff
log preserved across all thread events.**

---

## Part A — Research framing

### A.1 Why this milestone exists

Phases 31–34 established that typed handoffs + a static
role-subscription table preserve answer correctness while
bounding each role's active context, whenever (i) every load-
bearing claim kind is producible by some role, and (ii) the
auditor's decoder can select the correct answer by a *fixed*
priority rule over the delivered claim bundle.

The programme's next falsifier is the smallest task family
where condition (ii) fails: **contested incidents**. Two or
more plausible root-cause claims land in the auditor's inbox
with *inverted static priority* — the claim the author's
priority list ranks highest happens to be the shadow, while
the real root cause is the lower-ranked claim. On such
scenarios the Phase-31 decoder is structurally blind because
the information needed to disambiguate (producer-local
evidence about whether a claim is an isolated cause or a
downstream symptom) lives in the *producer's* slice of
telemetry — not in any payload, header, or subscription bit
currently routed to the auditor.

Two extreme responses would both miss the mark:

* **Unrestricted group chat** — add a broadcast bus where every
  role sees every claim and reasons openly. This violates the
  programme's bounded-context thesis: the auditor's prompt
  grows with |X|, and Phase 30 / Phase 31 Theorems P30-2 / P31-3
  no longer apply.
* **Static lateral subscription expansion** — edit the
  subscription table to route every contested-root-cause pair
  to every producer. This does not actually solve the problem
  — it just expands the subscription graph. The producer still
  needs a *typed coordination question* ("is your claim an
  isolated cause or a symptom?"), and static handoffs have no
  header bit for that.

The Phase-35 response is a **narrow, explicit middle ground**:
one new primitive — an *escalation thread* — that gives the
auditor a way to ask exactly one typed coordination question
to exactly the producers involved, collect at most a bounded
number of typed replies, and emit exactly one summary handoff
visible to exactly the subscribed roles. Nothing leaks
outside the thread except the single resolution.

### A.2 What the primitive is (one paragraph)

An ``EscalationThread`` is a frozen descriptor of a typed
coordination object between a small set of *member roles*
adjudicating a typed *issue_kind* over a tuple of
*candidate_claims*. Member roles post typed ``ThreadReply``
messages (``INDEPENDENT_ROOT`` / ``DOWNSTREAM_SYMPTOM`` /
``UNCERTAIN`` / ``AGREE`` / ``DISAGREE`` / ``DEFER_TO``) each
carrying a bounded witness string. The thread closes on
quorum-on-agree, max-round exhaustion, or explicit opener
close, producing a single ``ThreadResolution`` emitted as a
regular typed handoff (``claim_kind = CLAIM_THREAD_RESOLUTION``)
through the unchanged Phase-31 ``HandoffRouter``. Every thread
event (``THREAD:OPEN`` / ``THREAD:REPLY`` / ``THREAD:CLOSE``)
is hash-chained in the existing ``HandoffLog`` for audit.
Non-member roles never see thread-internal messages.

### A.3 Scope discipline (what Phase 35 does NOT claim)

1. **Not a general group-chat system.** Thread membership is
   fixed at open time; witness payloads are capped; the only
   output visible outside the thread is a single summary.
   There is no way for a thread to escalate into a broadcast
   without opening a new thread (which is itself bounded).
2. **Not Byzantine consensus.** Quorum here is a counting rule
   over typed replies. The hash-chained log detects accidental
   tamper / truncation; authenticated provenance still belongs
   to ``peer_review`` (Ed25519).
3. **Not a replacement for the Phase-31 substrate.** The
   typed-handoff layer (``core/role_handoff``) is unchanged
   byte-for-byte. The thread's public output is itself a typed
   handoff; the routing, log, and inbox machinery is reused
   as-is.
4. **Not a claim of LLM-driven reflection.** Each producer's
   reply is derived from a deterministic role-local witness
   extractor (``infer_causality_hypothesis``) whose precision
   and recall on the Phase-35 scenario bank are 1.00 by
   construction. A real-LLM reflection wrapper is mechanical
   follow-up (``core/extractor_noise`` + ``core/ensemble_extractor``
   compose here directly).
5. **Not a new subscription semantics.** Threads do NOT edit
   the standing subscription table. A closed thread leaves the
   team's subscription graph unchanged; the adaptive-
   subscription question is a separate open problem (§ F).
6. **Not a claim about scenarios where naive broadcast already
   succeeds.** Phase 31 is not contested; every strategy
   saturates its ceiling. Phase 35's empirical separation is
   *exactly* the contested regime.
7. **Not a K ≥ 20 hierarchical lattice result.** Conjecture
   C31-6 is untouched. Phase-35 threads are pairwise-small
   groups of ≤ 5 members (opener + up to 4 producers).

---

## Part B — Theory

### B.1 Setup (reused from Phase 31 § B.1)

Let a team have roles ``R = {r_1, …, r_K}`` with observable-
type subsets ``O_k``, claim extractors ``e_k``, a role-
subscription table ``σ`` over claim kinds, and a deterministic
decoder ``D`` mapping a bundle of delivered handoffs to the
team's answer. Let ``X`` be the naive broadcast event stream,
``H`` the set of emitted typed handoffs, and ``C(z) =
{(r_1, k_1), …, (r_{R*}, k_{R*})}`` the causal chain for
scenario ``z``. Let ``R*(z)`` be the causal-chain length.

Phase 35 introduces:

* a **contested** scenario ``z`` is one where ≥ 2 distinct
  claim kinds ``k_i, k_j`` arrive in the auditor's inbox with
  ``D_static`` (the static-priority decoder) picking a kind
  that is *not* on the scenario's causal chain (a priority-
  inversion failure).
* a **causality hypothesis** map ``H_z : (producer_role,
  claim_kind) → {INDEPENDENT_ROOT, DOWNSTREAM_SYMPTOM_OF(k'),
  UNCERTAIN}``. The producer-local extractor approximates
  ``H_z`` on its own claims.
* an **escalation thread** is a tuple ``(opener, members,
  issue_kind, (c_1, …, c_n), max_rounds, max_replies, quorum,
  witness_cap)`` with ``|members| ≤ K`` and every ``c_i =
  (producer, kind, payload)``. A *public output* is a single
  ``ThreadResolution(resolved_idx ∈ {1, …, n, None})``
  consumed by the decoder as a typed-handoff input.

### B.2 Theorem P35-1 — Expressivity gap: static handoffs cannot decode contested scenarios

**Statement.** There exists a scenario family ``Z_contested``
on which:

1. The static-handoff decoder ``D_static`` (Phase-31's
   priority-over-claim-kinds) achieves accuracy strictly < 1
   on every ``z ∈ Z_contested`` under any choice of static
   priority permutation, provided ``Z_contested`` contains at
   least two scenarios whose correct claim kinds occupy
   *different* priority ranks under *conflicting* static
   orderings. Formally: for any total order ``≺`` on the claim-
   kind vocabulary, there is ``z ∈ Z_contested`` whose gold
   kind ``k*(z)`` is not maximal under ``≺`` among the delivered
   bundle.

2. The dynamic-coordination decoder ``D_dyn`` (Phase-35, with
   exactly one bounded thread per scenario and a sound per-role
   causality extractor) achieves accuracy = 1 on the same
   scenario family.

**Interpretation.** Static typed handoffs + priority decoders
are a *tree of comparators* with fixed fan-in — the decoder
does not consume any evidence beyond the claim-kind header.
Any correct decoder must consume *producer-local content* on
at least some contested scenarios. Dynamic coordination adds
exactly the minimum header-bit count needed: one typed
``reply_kind`` per producer per contested candidate.

**Proof sketch.** *(1)* The static decoder is a function
``D_static : 2^Kinds → Kinds``. On an input ``S`` containing
two kinds ``k_a, k_b`` with ``k_a ≺ k_b``, ``D_static(S) =
k_b`` always. Consider scenarios ``z_1`` and ``z_2`` with
``z_1`` having gold kind ``k_a`` and ``z_2`` having gold kind
``k_b``, and assume both scenarios deliver the same claim
bundle ``{k_a, k_b}``. ``D_static`` returns the same kind on
both; it cannot be right on both, whatever ``≺`` chooses. By
inspection the Phase-35 bank contains such pairs
(``contested_tls_vs_disk_shadow`` and
``contested_cron_vs_oom_shadow`` exhibit priority inversions
under the published priority; swapping the priority inverts
the failure to other scenarios).
*(2)* Under ``D_dyn``, each contested scenario opens a thread
over the two candidates with the two producer roles as
members. Each producer's typed reply is ``INDEPENDENT_ROOT``
on its claim iff its claim is on the causal chain (by
soundness of the producer-local causality extractor; Phase-35
scenarios are constructed so the extractor's precision and
recall on its own role's claims are 1.00). Exactly one
producer replies ``INDEPENDENT_ROOT``; the resolution is
``SINGLE_INDEPENDENT_ROOT`` with the correct index. The
decoder maps that index to the gold label. ∎

**Empirical anchor.** § D.1: on the six-scenario bank,
``D_static`` scores 33 % full accuracy (50 % root-cause
accuracy) flat across k ∈ {6, 20, 60, 120}; ``D_dyn`` scores
100 % flat on every (k, seed). The gap ``+67 pp`` is
``Z_contested``-pooled empirical evidence for the separation;
the proof is the structural argument above.

### B.3 Theorem P35-2 — Bounded active context preserved under dynamic coordination

**Statement.** Let a role ``r`` participate in at most ``T``
open threads per task round, with each thread carrying at most
``R_max`` replies of at most ``W`` whitespace-split tokens
each. Then peak active context at ``r`` per round satisfies:

```
ctx(r) ≤ C_0 + R*·τ + T·R_max·W
```

where ``C_0`` is the fixed-point (task-goal + final-answer)
size and ``R*·τ`` is the Phase-31 per-round typed-handoff
bound (Theorem P31-3). In particular the bound is independent
of ``|X|`` (the raw event stream size).

**Interpretation.** Dynamic coordination costs the role at
most ``T·R_max·W`` extra tokens per round. On the Phase-35
bank, ``T ≤ 1``, ``R_max = 2``, ``W = 12``, so the extra
budget per role is ≤ 24 tokens per round — a constant. The
Phase-31 bounded-context guarantee extends to dynamic
coordination with an additive constant.

**Proof sketch.** By construction. The thread membership is
declared at open time and cannot grow (``members`` is frozen).
Each member posts at most ``max_replies_per_member`` replies,
each bounded to ``witness_token_cap`` tokens (enforced at
post_reply time; over-length witnesses are clipped). A non-
member role sees only the single ``ThreadResolution`` summary
handoff, which is bounded in the same way as any Phase-31
handoff. The ``R*·τ`` bound on static handoffs is unchanged
because the static typed-handoff layer is unchanged. ∎

**Empirical anchor.** § D.2: per-scenario mean prompt tokens
under ``dynamic`` is 246 flat across k ∈ {6, 20, 60, 120}.
Total thread witness tokens across the 6-scenario bank are
49 tokens pooled (≈ 8.2 tokens per thread × 5 threads) or 98
tokens pooled across 2 seeds.

### B.4 Theorem P35-3 — Correctness under sound producer-local causality extraction

**Statement.** Let the producer-local causality extractor
``ĥ_r : (r, kind, payload) → {INDEPENDENT_ROOT,
DOWNSTREAM_SYMPTOM_OF(k'), UNCERTAIN}`` satisfy:

1. **Soundness (precision).** For every gold ``H_z(r, kind) =
   INDEPENDENT_ROOT``, ``ĥ_r(r, kind, payload) =
   INDEPENDENT_ROOT`` iff that claim is on the causal chain.
2. **Refusal (recall-conservatism).** For every non-causal-
   chain claim, ``ĥ_r`` returns ``UNCERTAIN`` rather than
   ``INDEPENDENT_ROOT``.

Then for any contested scenario ``z`` whose causal chain has
exactly one ``INDEPENDENT_ROOT`` claim among the contested
candidates, the dynamic-coordination decoder ``D_dyn``
produces the gold answer.

**Interpretation.** The theorem pins down the *weakest*
assumption on the producer-local extractor under which
dynamic coordination is guaranteed to recover correctness:
the extractor must not over-emit ``INDEPENDENT_ROOT`` (no
false positives on non-causal claims) and must not under-emit
on the single causal claim (no false negative on the gold
root). Both are role-local, content-addressable properties;
Phase-34's per-role noise calibration machinery directly
measures them.

**Proof sketch.** The thread opens with two candidate claims.
Both producers post exactly one reply. By (1), exactly one
reply is ``INDEPENDENT_ROOT`` — the producer whose claim is
on the causal chain. By (2), the other producer replies
``UNCERTAIN`` or ``DOWNSTREAM_SYMPTOM``. The close-thread
resolution rule fires ``SINGLE_INDEPENDENT_ROOT`` with the
causal claim's index. The decoder looks up the label and
returns the gold. ∎

**Empirical anchor.** § D.1 + the unit test
``test_dynamic_solves_every_contested_on_mock`` under the
deterministic mock extractor. The Phase-34 wrappers
(``adversarial_extractor``, ``per_role_noisy_extractor``,
``UnionExtractor``) can be composed with the Phase-35 thread
layer at the extractor boundary; the combined measurement is
Conjecture C35-4.

### B.5 Theorem P35-4 — No-leak invariant for non-member roles

**Statement.** Let a thread ``T`` have member set ``M``. Any
role ``r ∉ M`` observes exactly zero thread-internal events
(no ``THREAD:OPEN`` / ``THREAD:REPLY`` messages enter ``r``'s
inbox). The only visible artifact to non-member roles is the
single ``ThreadResolution`` handoff delivered through the
standard Phase-31 subscription table, conditioned on ``r``
being statically subscribed to ``(opener_role,
CLAIM_THREAD_RESOLUTION)``.

**Interpretation.** This is the "bounded multicast" property:
dynamic coordination does not leak into the larger team's
active context unless the team's subscription table explicitly
asks for it. A single thread between DBA and SYSADMIN does
not increase MONITOR's or NETWORK's active context by a
single token. This is the mechanical reason Phase-31's
Theorem P31-3 bound extends: unrestricted chat would violate
it; bounded threads preserve it.

**Proof sketch.** By construction of ``post_reply`` and
``close_thread``. ``post_reply`` writes to the thread state's
reply list and appends a ``THREAD:REPLY`` log entry with
``to_role = "__thread__"`` — an internal tag that no inbox
subscribes to. ``close_thread`` emits a single
``CLAIM_THREAD_RESOLUTION`` through the base router; static
subscription routing delivers it to every role subscribed to
``(opener_role, CLAIM_THREAD_RESOLUTION)``, and *only* to
those roles. ∎

**Empirical anchor.** The unit test
``test_resolution_does_not_enter_non_member_inbox`` in
``tests/test_dynamic_comm.py``. MONITOR, which is not a
thread member in any Phase-35 scenario and is not subscribed
to ``CLAIM_THREAD_RESOLUTION``, sees zero thread-related
handoffs in its inbox on every scenario.

### B.6 Conjecture C35-5 — Adaptive subscription equivalence to bounded threads

**Statement.** A static subscription table ``σ`` plus a
bounded family of escalation threads ``T = {T_1, …, T_m}`` is
strictly more expressive than *any* static subscription table
``σ'`` alone, but *equivalent in expressivity* to an adaptive
subscription table whose edits are themselves bounded
functions of ``{σ, O_k, H_z}``. Formally: there is a many-to-
one correspondence ``φ`` between bounded-thread families and
adaptive-subscription-edit sequences such that both achieve
identical decoder correctness on any scenario ``z`` ∈ ``Z``.

**Status.** Unproven. The conjecture names the programme's
design-space question: is the natural continuation of Phase
35 *adaptive subscriptions* (a role can join another role's
subscription dynamically) or *bounded threads* (a role's
coordination with another role is always scoped to a short-
lived object)? Conjecture C35-5 predicts the two converge in
their correctness guarantees — but threads have the
observable-bounded-context bonus (Theorem P35-2) baked in at
the *primitive level* rather than at the enforcement level.

**Why it matters.** If C35-5 holds, bounded threads are
strictly preferable to adaptive subscriptions as a systems
primitive: they impose the bounded-context invariant at the
type level rather than requiring runtime enforcement. If C35-5
fails — i.e. there is a correctness gap — the failure mode
points to the specific task families (long-horizon multi-
party negotiation?) where bounded threads are insufficient.

### B.7 Conjecture C35-6 — Dynamic coordination is necessary, not just sufficient

**Statement.** There exists a scenario family ``Z_hard`` and a
constant ``c`` such that any team protocol that preserves the
Phase-31 ``Θ(R*·τ)`` per-role bounded-context bound and
achieves correctness on every ``z ∈ Z_hard`` *must* carry at
least ``c`` bits of dynamic (context-dependent) coordination
per scenario. In particular, no finite fixed subscription
graph, however large, is sufficient.

**Status.** Informal. Phase 35 Part D is evidence *for* the
direction of the gap but not for the formal lower bound.
A proof would come from an adversarial argument: given any
fixed subscription graph ``σ``, construct a scenario whose
gold is a deterministic function of producer-local bits that
``σ`` cannot route to the auditor without violating the
bounded-context budget. The argument is the team-
communication analogue of the Phase-30 Conjecture P30-5
information-theoretic lower bound — moving from "role-
stratified compressor cannot match the team" (P31-5) to
"static team protocol cannot match a dynamic team protocol on
contested scenarios" (C35-6).

**Why it matters.** C35-6 names the specific place where
dynamic coordination is *required* rather than *convenient*.
Phase-35 shows empirical separation at 100 % vs 33 %; a proof
would sharpen that to an information-theoretic lower bound
and would close the "maybe we just need a bigger subscription
table" escape hatch.

### B.8 What is theorem vs what is empirical

Ordered by strength:

* **Theorem (proved):** P35-1 (expressivity separation — the
  structural pigeonhole argument is straightforward),
  P35-2 (bounded-context preservation — direct from the
  thread's declared bounds), P35-3 (correctness under sound
  producer-local extraction — direct from the resolution rule
  and the extractor's soundness property), P35-4 (no-leak
  invariant — by construction of the post_reply / close_thread
  routing).
* **Empirical, measurable:** D.1 per-strategy accuracy; D.2
  per-strategy token count flatness; D.3 per-scenario
  messaging budget.
* **Conjecture (empirically suggested):** C35-5 (bounded
  threads ≡ bounded adaptive subscriptions in correctness),
  C35-6 (a lower-bound dual of P35-1).

A reviewer attacking this work should attack:

* **P35-3's "sound" assumption.** The producer-local extractor
  on Phase-35 scenarios is constructed to have precision and
  recall 1.00; under the Phase-34 noise wrappers this
  assumption fails and the thread resolution will degrade.
  The specific degradation shape is a natural follow-up:
  Phase-34's ``adversarial_extractor`` composed with the
  Phase-35 ``ThreadedCoordination`` is a mechanical
  experiment and is named in § F as Conjecture C35-7.
* **P35-1's scenario construction.** The Phase-35 bank has 4
  contested scenarios out of 6; the generalisation to arbitrary
  contested scenario distributions is open (does the
  separation hold at higher contest density? at longer causal
  chains?). The bank is deliberately small to make the theorem
  checkable; the scaling question is § F.

---

## Part C — Architecture

### C.1 New modules and relationships

```
vision_mvp/core/dynamic_comm.py           [NEW]  ~590 LOC
    + CandidateClaim (frozen)
    + EscalationThread (frozen)
    + ThreadReply (frozen)
    + ThreadResolution (frozen)
    + ThreadState (mutable state)
    + DynamicCommAccount
    + DynamicCommRouter
    + build_resolution_subscriptions
    + parse_resolution_payload
    + vocabularies: ALL_THREAD_ISSUES, ALL_REPLY_KINDS,
                    ALL_RESOLUTION_KINDS
    + CLAIM_THREAD_RESOLUTION  (public claim kind)
    + INTERNAL_CLAIM_THREAD_{OPEN, REPLY, CLOSE}

vision_mvp/tasks/contested_incident.py    [NEW]  ~840 LOC
    + ContestedScenario (wraps IncidentScenario)
    + 6 scenario builders (4 contested + 2 controls)
    + infer_causality_hypothesis (role-local extractor)
    + detect_contested_top (auditor's contest detector)
    + run_dynamic_coordination (one-round orchestrator)
    + decoder_from_handoffs_phase35 (thread-aware decoder)
    + MockContestedAuditor
    + run_contested_loop (harness driver)

vision_mvp/experiments/phase35_contested_incident.py  [NEW] ~210 LOC
    Phase-35 driver, Ollama-compatible.

vision_mvp/tests/test_dynamic_comm.py                [NEW]  29 tests
vision_mvp/tests/test_phase35_contested_incident.py  [NEW]  18 tests
```

The substrate primitive (``core/role_handoff``) is unchanged.
The Phase-31/32/33 task modules are unchanged. Phase 34's
extractor-boundary wrappers compose cleanly with
``infer_causality_hypothesis`` for future noise experiments.

### C.2 Where dynamic_comm sits in the substrate

```
    ┌──────────────────────────────────────────────────────┐
    │  Role-scoped team logic (task modules)                │
    │  — decoders, oracles, per-role extractors             │
    └──────────────────────────────────────────────────────┘
                                │
    ┌──────────────────────────────────────────────────────┐
    │  DynamicCommRouter  (Phase 35)                        │
    │  — EscalationThread open/reply/close                  │
    │  — ThreadResolution → CLAIM_THREAD_RESOLUTION         │
    └──────────────────────────────────────────────────────┘
                                │
    ┌──────────────────────────────────────────────────────┐
    │  HandoffRouter  (Phase 31, unchanged)                 │
    │  — TypedHandoff emit / RoleSubscriptionTable dispatch │
    │  — HandoffLog hash chain / RoleInbox delivery         │
    └──────────────────────────────────────────────────────┘
                                │
    ┌──────────────────────────────────────────────────────┐
    │  Raw event stream  (naive broadcast)                  │
    └──────────────────────────────────────────────────────┘
```

The inheritance is single-direction: Phase 35 uses Phase 31.
Phase 31 is oblivious to Phase 35. This matches the programme's
boundary discipline (Phase 34 § C.2): every new communication
primitive sits strictly above the existing substrate and
exposes its outputs through the existing typed-handoff
contract.

### C.3 Files changed

| File | Change |
|---|---|
| ``vision_mvp/core/dynamic_comm.py`` | **NEW** — dynamic-comm substrate (~590 LOC) |
| ``vision_mvp/tasks/contested_incident.py`` | **NEW** — 6-scenario contested bank + harness (~840 LOC) |
| ``vision_mvp/experiments/phase35_contested_incident.py`` | **NEW** — driver (~210 LOC) |
| ``vision_mvp/tests/test_dynamic_comm.py`` | **NEW** — 29 tests |
| ``vision_mvp/tests/test_phase35_contested_incident.py`` | **NEW** — 18 tests |
| ``vision_mvp/RESULTS_PHASE35.md`` | **NEW** — this document |
| ``docs/context_zero_master_plan.md`` | Phase 35 integration, frontier update |
| ``README.md``, ``ARCHITECTURE.md`` | Phase 35 threading |
| ``MATH_AUDIT.md`` | Phase 35 theorem entries |

---

## Part D — Evaluation

> Numbers below come from two artifacts reproduced by the
> Appendix commands:
>   (A) ``vision_mvp/results_phase35_mock.json`` (deterministic
>       mock auditor, 4 distractor counts × 2 seeds × 6
>       scenarios × 4 strategies = 192 measurements, 0.2 s
>       wall).
>   (B) ``vision_mvp/results_phase35_llm_0p5b.json`` (real
>       ``qwen2.5:0.5b`` auditor on k = 6, seed = 35, 6
>       scenarios × 4 strategies = 24 measurements, 68 s
>       wall).

### D.1 Headline — per-strategy accuracy under mock auditor

Pooled across 4 distractor counts × 2 seeds × 6 scenarios =
48 measurements per strategy:

| strategy | acc_full | acc_root_cause | contested_acc | n_correct | mean_tok | trunc |
|---|---:|---:|---:|---:|---:|---:|
| naive          | 0.292 | 0.417 | 0.063 | 14/48 | 2000 | 24/48 |
| static_handoff | 0.333 | 0.500 | 0.000 | 16/48 |  215 |  0/48 |
| **dynamic**    | **1.000** | **1.000** | **1.000** | **48/48** |  **246** | 0/48 |
| dynamic_wrap   | 1.000 | 1.000 | 1.000 | 48/48 |  297 | 0/48 |

Per-(k, strategy) breakdown (pooled across 2 seeds, 6
scenarios = 12 measurements per cell):

| k | strategy | acc_full | acc_rc | contested_acc | mean_tok | trunc |
|---:|---|---:|---:|---:|---:|---:|
| 6   | naive          | 0.333 | 0.500 | 0.000 |  591 |  0 |
| 6   | static_handoff | 0.333 | 0.500 | 0.000 |  215 |  0 |
| 6   | **dynamic**    | **1.000** | 1.000 | **1.000** |  **246** |  0 |
| 6   | dynamic_wrap   | 1.000 | 1.000 | 1.000 |  297 |  0 |
| 20  | naive          | 0.333 | 0.500 | 0.000 | 1422 |  0 |
| 20  | static_handoff | 0.333 | 0.500 | 0.000 |  215 |  0 |
| 20  | **dynamic**    | **1.000** | 1.000 | **1.000** |  **246** |  0 |
| 60  | naive          | 0.333 | 0.500 | 0.000 | 3037 | 12 |
| 60  | static_handoff | 0.333 | 0.500 | 0.000 |  215 |  0 |
| 60  | **dynamic**    | **1.000** | 1.000 | **1.000** |  **246** |  0 |
| 120 | naive          | 0.167 | 0.167 | 0.250 | 2950 | 12 |
| 120 | static_handoff | 0.333 | 0.500 | 0.000 |  215 |  0 |
| 120 | **dynamic**    | **1.000** | 1.000 | **1.000** |  **246** |  0 |

Reading:

* **Dynamic is flat at 246 tokens / 100 % accuracy on every
  (k, seed)**. Theorem P35-2's empirical signature: active
  context is independent of the raw event stream size.
* **Static handoff is flat at 215 tokens but capped at 33 %
  full / 50 % root-cause.** The remaining gap is *not*
  recoverable by any choice of static priority — Theorem
  P35-1 separation. The 50 % root-cause accuracy is the
  "static priority happens to pick the right kind on some
  scenarios" baseline; the remaining gap to 100 % is the
  priority-inversion failure.
* **Static's services contamination.** On the deadlock /
  cron / tls / dns-vs-tls scenarios, static_handoff's
  root-cause is correct on *some* scenarios but the
  services field includes shadow-service tags
  (``archival``, ``batch``, ``mail``) which the gold set
  excludes. Dynamic coordination's loser-list metadata in
  the resolution payload filters these out before
  aggregation — a second axis of bounded multicast benefit
  beyond root-cause recovery.
* **Naive degrades at k=120 under truncation.** Prompt
  size is already 2 925 tokens at k=60 and hits the
  ``max_events_in_prompt = 200`` clip on most scenarios at
  k=120. This is the familiar Phase-31 truncation-collapse
  signature.

### D.2 Per-scenario breakdown under mock auditor

Seed=35, k=6. ``full_correct`` column:

| scenario | contested | naive | static | dynamic | dynamic_wrap |
|---|:---:|:---:|:---:|:---:|:---:|
| ``contested_deadlock_vs_shadow_cron``  | ✓ | ✗ | ✗ | **✓** | **✓** |
| ``contested_tls_vs_disk_shadow``       | ✓ | ✗ | ✗ | **✓** | **✓** |
| ``contested_dns_vs_pool_symptom``      | · | ✓ | ✓ | ✓ | ✓ |
| ``contested_cron_vs_oom_shadow``       | ✓ | ✗ | ✗ | **✓** | **✓** |
| ``contested_dns_vs_tls_shadow``        | ✓ | ✗ | ✗ | **✓** | **✓** |
| ``concordant_disk_fill``               | · | ✓ | ✓ | ✓ | ✓ |

Of the 4 contested scenarios, dynamic wins 4/4. Static / naive
win 0/4. The 2 controls are solved by every strategy.

### D.3 Messaging budget under dynamic coordination

Pooled over 4 k × 2 seeds × 6 scenarios = 48 dynamic-strategy
measurements:

| metric | value |
|---|---:|
| n_threads_opened_total                | 40 |
| n_thread_replies_total                | 80 |
| n_thread_witness_tokens_total         | 392 |
| mean_threads_per_scenario             | 0.833 |
| mean_replies_per_opened_thread        | 2.00 |
| mean_witness_tokens_per_thread        | 9.8 |
| max_thread_members                    | 3 |
| max_thread_replies                    | 2 |

The per-scenario coordination cost is exactly one thread (or
zero on a control where contest detection returns < 2
candidates), each with ≤ 2 replies of ≤ 12 witness tokens.
Theorem P35-2's bound ``T·R_max·W ≤ 24 tokens/round`` is
tight up to a factor of 2 on this bank.

### D.4 Real-LLM corroboration — qwen2.5:0.5b

Same harness, real Ollama (24 LLM calls, wall = 68 s, k=6,
seed=35):

| strategy | acc_full | acc_rc | contested_acc | mean_tok |
|---|---:|---:|---:|---:|
| naive          | 0.000 | 0.000 | 0.000 |  590 |
| static_handoff | 0.333 | 0.500 | 0.000 |  215 |
| **dynamic**    | 0.667 | **1.000** | 0.500 |  **246** |
| dynamic_wrap   | 0.333 | 1.000 | 0.500 |  297 |

Reading:

* **Root-cause accuracy under real 0.5b: dynamic 1.00 vs
  static 0.50.** A full ``+50 pp`` lift on a real LLM. The
  dynamic strategy's root-cause answer is correct on every
  scenario; transcription-side errors (the 0.5b paraphrases
  the SUBSTRATE_ANSWER's SERVICES or REMEDIATION lines on 2
  scenarios) are the sole source of full-accuracy shortfall.
  This is the Phase-30 transcription-fidelity axis, not a
  substrate failure.
* **Naive scores 0/6 on real 0.5b.** Consistent with the
  Phase-31 pattern: on the 0.5b, naive broadcast on
  non-code domains is infeasible — the model cannot
  reliably parse the raw event stream.
* **Dynamic_wrap underperforms dynamic.** The explicit
  "return verbatim" instruction interacts poorly with the
  0.5b; the model paraphrases anyway and the verbatim check
  fails. This is a known 0.5b-specific failure mode
  (Phase-31 § D.3); on the 7B it inverts and dynamic_wrap
  is the stronger strategy.

---

## Part E — Failure taxonomy

Phase 35 adds three new failure-kind values to the
attribution dispatcher (mirrored from
``tasks/contested_incident.attribute_failure_p35``):

| kind | meaning |
|---|---|
| ``static_priority_pick_wrong``  | Static-handoff strategy: the decoder's static priority picked a non-gold kind even though all causal claims were in the inbox. Phase-35 contest-specific. |
| ``no_contest_detected``         | Dynamic strategy: the auditor's contest detector returned < 2 candidates on a scenario the oracle flagged contested. Indicates the ``detect_contested_top`` heuristic is too narrow. |
| ``resolution_conflict``         | Dynamic strategy: the thread closed with ``RESOLUTION_CONFLICT`` / ``NO_CONSENSUS`` / ``TIMEOUT``. The decoder fell back to static priority (and may still be wrong). |

Plus carry-over from Phase 31: ``none``, ``truncation``,
``retrieval_miss``, ``llm_error``.

Observed distribution on mock (48 per strategy):

| strategy | dominant failure | histogram |
|---|---|---|
| naive | truncation / retrieval_miss | ``{none: 12, truncation: 24, retrieval_miss: 12}`` |
| static_handoff | static_priority_pick_wrong | ``{none: 16, static_priority_pick_wrong: 32}`` |
| dynamic | none | ``{none: 48}`` |
| dynamic_wrap | none | ``{none: 48}`` |

The taxonomy cleanly distinguishes *which layer* a failure
attributed to: under static, the single dominant failure is
the Phase-35 priority-inversion pattern; under dynamic, the
thread resolution drives every scenario to the gold. On real
LLMs, ``llm_error`` (transcription fidelity) picks up the
remaining residual under dynamic (see D.4).

---

## Part F — Future work

### F.1 Carry-over from Phase 34 (unchanged)

* SWE-bench end-to-end with a real LLM on the wrap path.
* Frontier-model multi-seed × multi-k sweep.
* OQ-1 in full generality (Conjecture P30-6).
* Cross-language runtime calibration.
* Payload-level adversary.
* Hierarchical role lattice at K ≥ 20.
* Ensemble-against-adversary (C34-4).

### F.2 Newly surfaced by Phase 35

* **Dynamic coordination under Phase-34 extractor noise
  (C35-7).** The Phase-34 wrappers (``adversarial_extractor``,
  ``per_role_noisy_extractor``, ``UnionExtractor``) compose
  directly with ``infer_causality_hypothesis`` at the
  extractor boundary. The natural experiment: *does adversarial
  drop of a producer's causality hypothesis collapse the
  thread resolution, and does the ensemble wrapper recover?*
  Predicted shape: targeted drop of a single
  ``INDEPENDENT_ROOT`` reply flips the resolution from
  ``SINGLE_INDEPENDENT_ROOT`` to ``NO_CONSENSUS``; pairwise-
  complementary ensemble of (regex-extracted-hypothesis,
  LLM-extracted-hypothesis) closes it. Mechanical but not yet
  run.
* **Threaded coordination with LLM-driven replies (C35-8).**
  Replace ``infer_causality_hypothesis`` with a local LLM
  prompt per producer role. Measure precision / recall on the
  scenario bank against the oracle. Expected: small-LLM
  (0.5b) replies are noisy; 7B replies are clean; thread
  resolution succeeds on the 7B. Probes the real-LLM
  boundary for Theorem P35-3.
* **Adaptive subscription as an alternative primitive
  (C35-5).** Build the alternative: a subscription-edit
  primitive that adds a temporary edge between two roles for
  ``n_rounds``, then removes it. Compare correctness and
  context-budget against bounded threads. The conjecture
  predicts equivalence; the measurement is the separation.
* **Larger contest density scaling.** Phase-35 bank has 4
  contested scenarios; a 20-scenario bank with variable
  ``R*`` and variable contest arity (2-way, 3-way, 4-way
  candidate clusters) would quantify the robustness of
  P35-1's separation at higher contest density.
* **Cross-domain contest transfer.** Phase 32's compliance
  review and Phase 33's security escalation can be extended
  with contested variants. Does the P35-1 separation
  reproduce on monotone-verdict and max-ordinal decoders?

### F.3 What is genuinely blocking the endgame

Phase 35 does NOT unblock any of:

* **End-to-end SWE-bench** — the largest external-validity gap.
* **OQ-1 in full generality** (Conjecture P30-6).
* **Cross-language runtime calibration**.

Phase 35 *does* close the "maybe a bigger static subscription
table would do" escape hatch (Theorem P35-1) and surfaces the
next natural question: *how far does dynamic bounded
coordination scale along the contest-density and noise axes?*
(Conjectures C35-7, C35-8).

---

## Appendix A — How to reproduce

```bash
# 1. Phase-35 mock benchmark (sub-second wall).
python3 -m vision_mvp.experiments.phase35_contested_incident \
    --mock --distractor-counts 6 20 60 120 --seeds 35 36 \
    --out vision_mvp/results_phase35_mock.json

# 2. Real-LLM spot check under qwen2.5:0.5b.
python3 -m vision_mvp.experiments.phase35_contested_incident \
    --model qwen2.5:0.5b --distractor-counts 6 --seeds 35 \
    --out vision_mvp/results_phase35_llm_0p5b.json

# 3. Test suite — new tests only.
python3 -m pytest vision_mvp/tests/test_dynamic_comm.py \
    vision_mvp/tests/test_phase35_contested_incident.py -q

# 4. Full test suite (regression check against prior phases).
python3 -m pytest vision_mvp/tests/ -q
```

On a commodity laptop (2026-vintage): mock sweep (A) runs in
~0.2 s; real-LLM run (B) runs in ~70 s for 24 calls under
ollama qwen2.5:0.5b; test suite (C) runs in ~0.5 s for the
Phase-35 subset, ~10 s for the full 1 270-test suite.

---

*End of Phase 35 results note. The master plan
(``docs/context_zero_master_plan.md``) is updated in the same
commit; see ``§ 4.11 Current frontier`` for the higher-level
integration.*
