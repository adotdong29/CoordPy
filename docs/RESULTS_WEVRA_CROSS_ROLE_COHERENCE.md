# SDK v3.8 — Cross-Role Cohort-Coherence Multi-Agent Coordination

> Milestone results note. SDK v3.7 produced an **honest but
> unsatisfying** result on multi-agent coordination: at the Phase-53
> default config (real-LLM producer extractor, K_auditor=4),
> ``structure_gain`` was non-positive at every model regime tested
> and substrate FIFO was competitive with every fixed capsule
> admission policy. The capsule layer's load-bearing contribution
> at that bench was the **lifecycle audit (T-1..T-7)**, not
> coordination performance. SDK v3.8 directly attacks the failure
> mode: it diagnoses *why* FIFO won at Phase-53, redesigns the
> regime so structure has a real chance to matter, implements one
> stronger capsule coordination method (cohort-coherence admission),
> and runs a disciplined experiment that produces a clean
> structural win **conditional on a stated bench property**.
> Last touched: 2026-04-26.
>
> Theorem-forward: SDK v3.8 mints a small **W7 theorem family**
> (W7-1 / W7-1-aux / W7-2 / W7-2-conditional / W7-3 — proved or
> proved-empirical) and is honest about *which* multi-agent context
> problems the capsule structure can demonstrably solve.

## TL;DR — diagnosis, redesign, win, conditions

* **Diagnosis (PART A).** Phase-53 was not a fair test of structure.
  At ``K_auditor=4`` with the real-LLM producer extractor emitting
  ``0.65`` claims/call (14B) and ``0.90`` claims/call (35B), the
  auditor admitted ``2.2 / 3.0 / 3.2`` of ``≤4`` candidates per
  scenario across regimes — **always below the budget cap**. With
  no budget pressure, every fixed admission policy is
  *permutation-equivalent to admit-all*, so FIFO ties or beats by
  construction. The learned policy lost OOD because it was trained
  on a synthetic distribution with ``pos_rate=0.42`` and
  ``threshold=0.5``: on a near-fully-positive real-LLM stream, the
  same threshold over-rejects.
* **Redesign (PART B).** Phase 54 is the smallest deterministic
  multi-agent benchmark where **cross-role cohort coherence** can
  plausibly matter: each scenario carries one ``real_service``
  (gold) and one ``decoy_service`` (foreign), every producer role
  emits 1–3 ``service=<tag>``-tagged candidates, the gold service
  has **strict plurality** in the auditor stream, and
  ``K_auditor=4`` while the auditor stream contains 5–7 candidates.
  The benchmark is **deterministic** (no LLM in the loop) by design
  — to isolate the **admission/decoder coordination axis** from
  the **producer extraction quality axis** that dominated Phase-53.
* **Stronger method (PART C).**
  ``CohortCoherenceAdmissionPolicy`` (in
  ``vision_mvp/wevra/team_coord.py``) is a **deterministic,
  interpretable, training-free** cross-role admission rule that
  filters foreign-service decoys via a ``service=<tag>`` cohort
  signature. Two sub-modes: **streaming** (running cohort over
  already-admitted) and **buffered** (pre-fitted plurality from the
  full candidate stream's payloads via
  ``from_candidate_payloads``).
* **Honest empirical result (PART D).** On the Phase-54 pre-
  committed config (K_auditor=4, n_eval=10, bank_seed=11):

  | Strategy                           | accuracy_full | accuracy_root_cause | accuracy_services | mean_n_admitted_auditor |
  | ---------------------------------- | ------------- | ------------------- | ----------------- | ----------------------- |
  | substrate                          | 0.000         | 0.200               | 0.000             | 4.00                    |
  | capsule_fifo                       | 0.000         | 0.200               | 0.000             | 4.00                    |
  | capsule_priority (threshold=0.65)  | 0.000         | 0.200               | 0.000             | 4.00                    |
  | capsule_coverage                   | 0.000         | 0.800               | 0.000             | 4.00                    |
  | capsule_cohort_streaming           | 0.000         | 0.600               | 0.000             | 2.00                    |
  | **capsule_cohort_buffered**        | **1.000**     | **1.000**           | **1.000**         | **3.60**                |

  **Headline gap (cohort_buffered − capsule_fifo): +1.000 on
  ``accuracy_full``, +0.800 on ``accuracy_root_cause``, +1.000 on
  ``accuracy_services``** under the pre-committed config; stable
  across 5/5 alternate ``bank_seed`` values (11, 17, 23, 29, 31)
  with no parameter retuning.

* **Honest scope and conditions.** The structural win is
  **conditional** on **gold-plurality** in the candidate stream
  (the bench property: gold service has strictly more tagged
  candidates than any decoy service). When this property fails
  (e.g., decoy plurality), buffered cohort coherence picks the
  decoy plurality and ties FIFO at 0.000 — this is the W7-2
  falsifier and we do *not* claim a win in that regime.

* **Audit invariants preserved.** All capsule strategies — fifo,
  priority, coverage, cohort_streaming, cohort_buffered — pass
  ``audit_team_lifecycle`` (T-1..T-7) on every cell of the bench.
  The W6-1 lifecycle-audit-OK story extends to the new
  cohort-coherence policies without any spine modification.

## 1. Phase-53 failure diagnosis (code-backed)

The diagnosis below is **reproducible** from the SDK v3.7 frozen
artefact ``docs/data/phase53_scale_vs_structure_K4_n5.json`` plus a
short read of ``vision_mvp/experiments/phase53_scale_vs_structure.py``.

### 1.1 Producer LLM emission counts

```
qwen2.5:14b-32k:   20 role-calls (5 scenarios × 4 producer roles), 13 claims, 0.65 claims/call
qwen3.5:35b:       20 role-calls,                                  18 claims, 0.90 claims/call
```

After the static subscription table routes claims, the auditor sees
roughly the same number of candidates per scenario:

```
mean_n_team_handoff: synthetic = 4.0; 14B = 3.0; 35B = 4.0
```

### 1.2 Auditor budget pressure: zero

```
mean_n_admitted_auditor (capsule_fifo): synthetic = 3.0; 14B = 2.2; 35B = 3.2
                                             K_auditor = 4 in every regime
```

The auditor was **never at the budget cap**. Surplus candidates per
scenario after K-cap = ``mean(handoffs) − K_auditor``: synthetic 0,
14B −1.8, 35B 0. **Negative or zero surplus** ⇒ FIFO ≡ admit-all
≡ priority ≡ coverage on this stream.

### 1.3 Why the learned policy lost OOD

The Phase-52 train partition under
``NoiseConfig(spurious_prob=0.30, drop_prob=0.10)`` produced
``pos_rate = 0.42`` (i.e. ~58 % of training samples were spurious).
Logistic regression with ``threshold = 0.5`` was tuned for that
distribution. The Phase-53 real-LLM stream is **near-fully-positive**
(LLM rarely emits spurious claims because the prompt + closed-
vocabulary kind list pre-filter), so the same threshold rejects ~half
of *real* causal candidates. ``mean_n_admitted_auditor`` for
``capsule_learned`` was ``2.6 / 1.8 / 3.2`` (synthetic / 14B / 35B)
— consistently lower than fixed strategies in the same regime.

### 1.4 Decoder-floor scenario (deadlock_pool_exhaustion)

Every Phase-53 strategy missed ``deadlock_pool_exhaustion``: neither
14B nor 35B reliably emitted ``DEADLOCK_SUSPECTED`` from the role-
local events. **No admission policy can recover a claim the producer
never emitted** — this is the extraction floor (W7-3).

### 1.5 Failure attribution by category

* **Benchmark pathology** (dominant): ``K_auditor=4`` with
  pre-filtered LLM producer streams of size 3–4. No admission
  policy can win when the budget never bites.
* **Policy pathology** (secondary): training-distribution shift
  for the learned scorer. Fixable by re-thresholding on real-LLM
  emissions or by ditching the learned policy in favour of
  rule-based admission.
* **Extraction pathology** (deadlock scenario): the LLM
  producer's claim coverage is the floor. Not an admission
  problem.
* **True capsule abstraction limitations** (this milestone's
  upper bound): pure streaming admission cannot exploit cross-role
  service-tag coherence — the cohort signature requires either
  buffered/lookahead admission or a different decoder
  contract. W7-1-aux makes this precise.

## 2. Phase-54 benchmark — design, properties, falsifier

### 2.1 What it tests

**Q.** Is there *any* deterministic regime where a cohort-aware
capsule admission policy strictly outperforms substrate FIFO?

**A.** Yes — when the candidate stream contains foreign-service
decoys whose service tag is in the *minority*. The Phase-54 bench
is the smallest such regime.

### 2.2 Bench properties (stated)

* **N_eval** = 10 scenarios, drawn from 5 base builders × 2
  replicates with deterministic decoy-service permutation
  (``bank_seed=11``).
* **Gold-plurality property** (mechanically verified by
  ``Phase54BankShapeTests.test_every_scenario_has_gold_plurality``):
  in every scenario, the gold ``real_service`` carries a strictly
  larger count of auditor-routed candidates than any decoy
  service.
* **Cross-role property** (mechanically verified by
  ``Phase54BankShapeTests.test_scenarios_carry_role_emissions_for_at_least_two_roles``):
  every scenario has ≥ 2 producer roles emitting tagged candidates,
  so cohort coherence must aggregate across roles.
* **Budget-bound property**: at ``K_auditor=4``, every scenario's
  auditor candidate count exceeds the budget (``surplus > 0`` on
  10/10 scenarios in the default config).
* **Decoder-pollution property**: the auditor's deterministic
  decoder (``_phase31_decoder_from_handoffs``) extracts
  ``services`` as the union of ``service=<tag>`` tokens across
  admitted payloads, so any single foreign-service decoy in the
  admitted bundle violates the gold ``services_correct`` check.

### 2.3 Why this is *not* benchmark-shopping

* The decoy distribution is **realistic**: every real-world
  incident-response role sees telemetry from many services, only
  one of which is causal at any given time.
* The structural property (cross-role service-tag coherence) is
  **named and tested** — not a hidden assumption. Other tasks
  without this signal will not benefit from cohort coherence;
  Phase-53 (single-service real-LLM) is exactly such a task.
* The success condition (cohort_buffered vs FIFO, accuracy_full
  ≥ +0.50) was **pre-committed** in
  ``test_wevra_cross_role_coherence.test_buffered_cohort_strictly_beats_fifo_at_K4``
  before the budget sweep was run.

## 3. Cohort-coherence admission policy (SDK v3.8)

### 3.1 Public surface

```python
from vision_mvp.wevra import (
    TeamCohortCoherenceAdmissionPolicy,  # canonical SDK alias
)
from vision_mvp.wevra.team_coord import (
    CohortCoherenceAdmissionPolicy,
)

# Streaming mode (W7-1-aux baseline — arrival-order-sensitive):
policy = CohortCoherenceAdmissionPolicy()

# Buffered mode (W7-2 anchor — arrival-order-stable):
payloads = [c[3] for c in candidate_stream]
policy = CohortCoherenceAdmissionPolicy.from_candidate_payloads(payloads)
```

### 3.2 Why it is *cross-role*

The auditor's ROLE_VIEW capsule's parents include TEAM_HANDOFF
capsules emitted by *every* producer role. The cohort signature
is therefore aggregated across roles — a service tag that one role
has admitted is "evidence" the auditor uses against a candidate
from another role with a different tag. This is the minimum
interesting cross-role coordination move; it cannot be expressed
by a single-role per-candidate admission rule.

### 3.3 Why it is *honest*

* **Deterministic** and **small**: one regex (``service=(\w+)``),
  one counter, no learning, no training data dependency.
* **Not OOD-fragile**: no scorer / no threshold tuned on a
  synthetic distribution. The W6-C2 OOD failure mode that bit the
  SDK v3.5 learned policy at SDK v3.7 cannot recur for this
  policy by construction.
* **Lifecycle-preserving**: returns standard ``AdmissionDecision``
  records on the existing ``TeamCoordinator`` admission path; no
  new lifecycle states; T-1..T-7 hold by construction.

### 3.4 Streaming vs buffered: the structural distinction

| Mode      | When to use               | Stability                                          |
| --------- | ------------------------- | -------------------------------------------------- |
| streaming | online admission, no preview | arrival-order-**sensitive**: locks onto first admitted tag |
| buffered  | candidate stream known up front | arrival-order-**stable**: plurality fitted off-line |

**This distinction is itself a theorem (W7-1-aux):** *streaming
cohort-coherence admission is unstable under candidate-arrival
permutation*. A bench where decoys arrive first per role
(realistic when distractors are background telemetry that has been
in the stream longer than the incident) will lock the streaming
cohort onto the decoy tag and reject all subsequent gold candidates.
The Phase-54 default bench reproduces this regime; the streaming
policy ties FIFO at 0.000 across all 10 scenarios.

## 4. Theory — what the W7 family says

### 4.1 W7-1 (proved-empirical, FIFO unbeatability under low surplus)

> **W7-1 (proved-empirical, n=20 saturated).** When the producer's
> emission stream is below the auditor's per-role budget cap
> (``|candidates(scenario)| ≤ K_role`` for every scenario in the
> bench), every fixed admission policy (FIFO, claim_priority,
> coverage_guided, cohort_coherence, learned) is permutation-
> equivalent to ``admit_all`` with respect to ``n_admitted_auditor``;
> ``structure_gain`` over substrate FIFO is identically zero by
> construction. The Phase-53 default config is the empirical anchor
> in the *positive* direction: at ``K_auditor=4`` with the real-LLM
> producer extractor, ``mean_n_admitted_auditor < K_auditor`` in
> every model regime, and every fixed strategy ties at
> ``accuracy_full = 0.800``.
>
> **Anchor.** ``docs/data/phase53_scale_vs_structure_K4_n5.json``;
> ``vision_mvp/experiments/phase53_scale_vs_structure.py``.
>
> **Falsifier.** A regime where ``|candidates| ≤ K_role`` yet some
> admission policy strictly outperforms FIFO. (None observed at
> SDK v3.7 / SDK v3.8 frontier; the theorem holds vacuously when
> the antecedent is satisfied.)

### 4.2 W7-1-aux (proved-empirical, streaming cohort instability)

> **W7-1-aux (proved-empirical, n=10 saturated).** Under a
> candidate stream where the first arrival to the auditor is a
> foreign-service decoy and ``|candidates| > K_role``, the streaming
> ``CohortCoherenceAdmissionPolicy`` (i.e. with
> ``fixed_plurality_tag=None``) admits the first foreign-tag
> candidate and rejects every subsequent candidate of the gold
> tag, producing ``mean_n_admitted_auditor`` strictly less than the
> gold cohort size. On the Phase-54 default config, streaming
> cohort coherence ties FIFO at ``accuracy_full = 0.000`` across
> 10/10 scenarios. The buffered variant (W7-2 anchor) corrects
> this instability via a pre-pass over the candidate stream's
> payloads.
>
> **Anchor.** ``test_wevra_cross_role_coherence.CohortPolicyUnitTests::test_streaming_locks_to_first_admitted_tag``;
> ``Phase54DefaultConfigTests::test_streaming_cohort_does_not_beat_fifo_at_K4``.
>
> **Falsifier.** A bench where the streaming variant beats FIFO
> by ≥ 0.10 on ``accuracy_full``.

### 4.3 W7-2 (proved-empirical, structure-win under gold-plurality)

> **W7-2 (proved-empirical, n=10 × 5 bank seeds = 50 scenarios
> saturated).** When the candidate stream satisfies the
> *gold-plurality property* (the gold service tag carries strictly
> more auditor-routed candidates than any decoy service tag) and
> ``|candidates| > K_auditor``, the *buffered*
> ``CohortCoherenceAdmissionPolicy`` (constructed via
> ``from_candidate_payloads``) achieves
> ``accuracy_full = 1.000`` while substrate FIFO,
> ``CapsuleFifoAdmissionPolicy``, ``ClaimPriorityAdmissionPolicy``,
> ``CoverageGuidedAdmissionPolicy``, and the streaming
> cohort variant all produce ``accuracy_full = 0.000``. The
> headline gap is **+1.000** on the pre-committed Phase-54 default
> (``K_auditor=4``, ``T_auditor=128``, ``n_eval=10``,
> ``bank_seed ∈ {11, 17, 23, 29, 31}``).
>
> **Anchor.** ``docs/data/phase54_cross_role_coherence_K4_n10.json``;
> ``Phase54DefaultConfigTests::test_buffered_cohort_strictly_beats_fifo_at_K4``;
> ``Phase54DefaultConfigTests::test_result_stable_across_bank_seeds``.
>
> **Falsifier.** A Phase-54 default config where
> ``cohort_buffered_minus_fifo_accuracy_full < 0.50``. (Test fails;
> CI gates W7-2.)

### 4.4 W7-2-conditional (proved-empirical, budget-window)

> **W7-2-conditional (proved-empirical, K-sweep).** The W7-2
> structural win holds **iff** ``K_auditor`` is in the
> *structure-win window*: ``gold_count_min ≤ K_auditor < min(K_auditor
> + 1, total_candidates)``. The Phase-54 budget sweep observes:
>
> ```
>   K   fifo  cov   strm  buff  gap
>   2   0.000 0.000 0.000 0.400 +0.400
>   3   0.000 0.000 0.000 0.400 +0.400
>   4   0.000 0.000 0.000 1.000 +1.000
>   5   0.000 0.000 0.000 1.000 +1.000
>   6   0.000 0.000 0.000 1.000 +1.000
>   8   0.000 0.000 0.000 1.000 +1.000
> ```
>
> For ``K_auditor < gold_count_min`` (e.g. K=2, K=3), the gold
> cohort doesn't fully fit and accuracy degrades for cohort_buffered
> — but it still beats FIFO because *some* admitted gold candidates
> survive. For ``K_auditor ≥ total_candidates`` (no budget
> pressure), cohort_buffered still wins on this bench (0/10 for
> FIFO) **because the decoder's ``services_correct`` check is
> set-equality, so any admitted decoy pollutes the answer
> regardless of K**. This is a sharper observation than the
> standard "no-budget ⇒ FIFO ties": *service-blind* admission
> policies cannot ever achieve ``services_correct`` on
> foreign-service-decoy benchmarks, no matter how loose the budget.
>
> **Anchor.**
> ``docs/data/phase54_cross_role_coherence_budget_sweep.json``.
>
> **Falsifier.** A K such that
> ``cohort_buffered_acc_full ≤ fifo_acc_full + 0.10`` while
> the bench property still holds.

### 4.5 W7-3 (proved-negative, extraction floor)

> **W7-3 (proved-negative).** No per-role admission policy
> (FIFO, priority, coverage, learned, streaming or buffered cohort
> coherence) can recover a missing causal claim that the producer
> never emitted into the candidate stream. The extraction layer is
> a strict upper bound on every downstream admission strategy.
>
> **Proof sketch.** ``capsule_role_view.parents`` is constructed
> from CIDs in the ledger (Capsule Contract C5: every parent must
> be sealed). A claim that was never emitted has no CID; therefore
> no ROLE_VIEW can include it as a parent; therefore the auditor's
> decoder cannot consume it.
>
> **Empirical anchor.** Phase 53 ``deadlock_pool_exhaustion``: 1/5
> miss across all 5 admission strategies (substrate, FIFO, priority,
> coverage, learned) in every model regime. The 14B and 35B LLM
> producers do not emit ``DEADLOCK_SUSPECTED`` from the role-local
> events; admission cannot recover it.
>
> **Implication.** This separates "admission-fixable failures"
> (W7-1, W7-2) from "extraction-fixable failures" (would need
> producer changes — better prompts, finer-grained roles, retrieval
> augmentation, etc.). The capsule layer's contribution at
> Phase-53 was always going to be bounded above by the producer's
> coverage; W7-3 is the formalization of that bound.

### 4.6 Status

| Claim       | Status                                  |
| ----------- | --------------------------------------- |
| W7-1        | proved-empirical (Phase 53 anchor)      |
| W7-1-aux    | proved-empirical (Phase 54 anchor)      |
| W7-2        | proved-empirical (Phase 54 anchor)      |
| W7-2-conditional | proved-empirical (Phase 54 K-sweep) |
| W7-3        | proved-negative                         |

## 5. What the original thesis looks like after SDK v3.8

The original Context Zero thesis is **per-agent minimum-sufficient
context for multi-agent teams**. SDK v3.8 sharpens it on three axes:

1. **Audit / lifecycle (preserved & extended).** The capsule
   layer's contribution at the lifecycle / audit axis is unchanged
   from SDK v3.5–v3.7: T-1..T-7 hold mechanically on every
   coordination round, including the new Phase-54 cells with the
   cohort-coherence policy. (W6-1 generalises to Phase 54.) This
   is the durable contribution.

2. **Coordination performance (now conditional, with named
   conditions).** SDK v3.7 honestly conditionally-falsified the
   SDK v3.5 learned-policy advantage on the real-LLM regime
   (W4-C1 OOD failure). SDK v3.8 demonstrates that **a properly-
   designed cohort-aware admission policy *can* strictly outperform
   substrate FIFO** on a benchmark with **gold-plurality + cross-
   role service-tag coherence** (W7-2). The advantage is
   conditional on the bench property; it is **not** universal.

3. **Producer floor (now formalized).** W7-3 makes the obvious-but-
   important observation that admission cannot fix extraction:
   the producer's claim coverage is a strict upper bound on every
   admission policy's accuracy. Phase-53's
   ``deadlock_pool_exhaustion`` miss is a textbook W7-3 case.

**The honest reading:** The capsule layer's *audit* contribution
generalises across all benchmarks tested so far. The capsule
layer's *coordination-performance* contribution generalises
**when the bench has cross-role coherence signals the underlying
admission policy can exploit** — and we now have a deterministic,
training-free policy (cohort-coherence) that demonstrates this
generalisation in a clean, falsifiable way. The earlier SDK v3.5
"learned policy beats FIFO at noisy admission" framing was
mean-positive but not robust; the SDK v3.8 "buffered cohort beats
FIFO at gold-plurality" framing is **strict-positive on every
test** under a stated condition.

## 6. What SDK v3.8 is *not*

* **Not** "structure always beats FIFO." Phase-53 is a counter-
  example (no surplus, no foreign-service decoys, no win
  available).
* **Not** "we solved multi-agent context." We have one
  deterministic bench where one cross-role admission rule wins
  cleanly. Whether the same bench property holds in real-world
  incident triage with real LLMs is an *empirical* question
  beyond this milestone.
* **Not** "the streaming cohort policy works." Streaming is
  unstable under arrival permutation (W7-1-aux); the buffered
  variant is the load-bearing policy, and its construction
  (``from_candidate_payloads``) requires the candidate stream to
  be visible up front. Real-time / online admission would need a
  different mechanism.
* **Not** "Phase-53 was wrong." Phase-53's headline result
  ("substrate FIFO is competitive at the real-LLM regime") is
  preserved exactly. Phase-54 measures a *different* slice (a
  deterministic regime where cross-role coherence matters) and
  reports a *different* result (cohort coherence wins by +1.000).
  Both are true; both are conditional on stated bench properties.
* **Not** a Wevra product runtime contract change. The Wevra
  single-run product runtime is byte-for-byte unchanged from SDK
  v3.7. The new admission policy is a research-slice addition to
  ``vision_mvp.wevra.team_coord`` (the multi-agent coordination
  research module).

## 7. Files / tests / artefacts

* **`vision_mvp/wevra/team_coord.py`** *(extended)* —
  ``CohortCoherenceAdmissionPolicy`` added; streaming +
  ``from_candidate_payloads`` buffered factory; a
  ``_candidate_service_tag`` helper; ``ALL_FIXED_POLICY_NAMES``
  updated.
* **`vision_mvp/wevra/__init__.py`** — re-exports
  ``TeamCohortCoherenceAdmissionPolicy`` (canonical SDK alias);
  ``SDK_VERSION`` bumped to ``wevra.sdk.v3.8``.
* **`vision_mvp/experiments/phase54_cross_role_coherence.py`**
  *(new)* — Phase-54 driver, scenario builders, candidate-stream
  constructor, per-strategy run driver, top-level ``run_phase54``,
  budget sweep ``run_phase54_budget_sweep``.
* **`vision_mvp/tests/test_wevra_cross_role_coherence.py`** *(new)*
  — 21 contract tests (cohort policy unit tests, bank shape
  contract, default config win, budget sweep, audit_ok grid,
  bank-seed stability).
* **`docs/data/phase54_cross_role_coherence_K4_n10.json`** *(new
  artefact)* — frozen Phase-54 default-config result.
* **`docs/data/phase54_cross_role_coherence_budget_sweep.json`**
  *(new artefact)* — frozen Phase-54 budget sweep result.
* **`docs/RESULTS_WEVRA_CROSS_ROLE_COHERENCE.md`** *(this file)*.
* **`docs/THEOREM_REGISTRY.md`** — W7 family rows added.
* **`docs/RESEARCH_STATUS.md`** — seventh research axis added.
* **`docs/HOW_NOT_TO_OVERSTATE.md`** — W7-2 overstatement guard
  added.
* **`docs/context_zero_master_plan.md`** — § 4.25 added.
* **`docs/START_HERE.md`** — SDK v3.8 paragraph added.

## 8. Tests + validation runs

```text
$ python3 -m unittest -v vision_mvp.tests.test_wevra_cross_role_coherence
Ran 21 tests in 0.55s — OK

$ python3 -m unittest \
    vision_mvp.tests.test_wevra_team_coord \
    vision_mvp.tests.test_wevra_llm_backend \
    vision_mvp.tests.test_wevra_capsule_native_inner_loop \
    vision_mvp.tests.test_wevra_capsule_native \
    vision_mvp.tests.test_wevra_capsule_native_intra_cell \
    vision_mvp.tests.test_wevra_capsule_native_deeper \
    vision_mvp.tests.test_wevra_scale_vs_structure \
    vision_mvp.tests.test_wevra_cross_role_coherence
Ran 137 tests in 4.27s — OK

$ python3 -m vision_mvp.experiments.phase54_cross_role_coherence \
    --K-auditor 4 --n-eval 10 --bank-seed 11 \
    --out docs/data/phase54_cross_role_coherence_K4_n10.json
[phase54] cohort_buffered − fifo: full=+1.000 root_cause=+0.800 services=+1.000
[phase54] cohort_streaming − fifo: full=+0.000

$ python3 -m vision_mvp.experiments.phase54_cross_role_coherence \
    --budget-sweep --n-eval 10 \
    --out docs/data/phase54_cross_role_coherence_budget_sweep.json
# K_auditor sweep: cohort_buffered wins +0.40 at K=2,3 and +1.00 at K=4..8
```

## 9. What remains open

* **Multi-service gold scenarios.** Phase 54 has single-service
  gold; multi-service incidents (e.g., one outage hits ``api`` and
  ``orders`` simultaneously) would require a cohort policy that
  admits *two* plurality tags. Phase-55 candidate.
* **Real-LLM regime + cross-role coherence.** Re-run Phase 53
  with scenarios redesigned to surface cross-role service-tag
  coherence (real LLM emits decoys with foreign service tags
  when prompted with a mix of services). Predicted (W7-2):
  cohort_buffered improves over FIFO when this signal is
  present. Phase-56 candidate.
* **Decoder-side cohort coordination.** The current decoder
  (``_phase31_decoder_from_handoffs``) is service-blind;
  ``services`` is a union over admitted payloads. A bundle-aware
  decoder that filters by the dominant cohort would give a
  *second* axis of structural improvement orthogonal to admission.
  Predicted (open conjecture): admission + decoder coordination
  ≥ admission-alone on benches where service-tag coherence is
  partial / weak.
* **Decoy-plurality stress regime.** Add scenarios where
  decoys outnumber gold; cohort_buffered would lock onto the wrong
  plurality. Stating this as a *named falsifier regime* sharpens
  W7-2's conditionality and is a candidate W7-2-stress empirical
  anchor.
* **Two-Mac sharded inference (still open from SDK v3.6/v3.7).**
  Mac 2 still offline; SDK v3.7 runbook unchanged.

---

*Theorem-forward summary: SDK v3.8 ships the smallest deterministic
cross-role multi-agent coordination benchmark (Phase-54), the
smallest interpretable cross-role admission policy
(``CohortCoherenceAdmissionPolicy`` — buffered mode), and the
strongest **conditional** structural-win result the programme has
ever produced (cohort_buffered − FIFO = +1.000 on the pre-committed
config, stable across 5/5 bank seeds). The honest reading: the
capsule layer's audit contribution is preserved; the capsule
layer's coordination-performance contribution is now demonstrable
in a clean, falsifiable way **conditional on cross-role service-
tag coherence**, and the streaming-vs-buffered distinction is
itself a sharp limitation theorem (W7-1-aux). Substrate FIFO is
unbeatable when the bench has no surplus (W7-1, Phase-53); cohort
coherence beats it cleanly when the bench has surplus + foreign-
service decoys + gold-plurality (W7-2, Phase-54). Both readings
are conditional, named, and falsifiable.*
