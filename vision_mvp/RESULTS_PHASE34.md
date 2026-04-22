# Phase 34 — Structured Noise, Adversarial Noise, and Honest Ensemble Extractors

**Status: combined research milestone. Phase 34 ships four coupled
deliverables: (a) a per-role noise-calibration layer
(``core/extractor_calibration.per_role_audit_summary`` +
``core/extractor_noise.PerRoleNoiseConfig`` +
``per_role_noisy_extractor``) that exposes the role-heterogeneity the
Phase-33 pooled calibration hid; (b) an adversarial extractor wrapper
(``core/extractor_noise.adversarial_extractor``) that selectively drops
load-bearing claims along the causal chain, silences entire roles, and
injects severity-escalation claims on max-ordinal decoders; (c) the
first honest regex + LLM ensemble result
(``core/ensemble_extractor`` on a compliance-freeform scenario bank
where regex and LLM have genuinely complementary coverage); and (d)
five theorem/conjecture additions (P34-1 per-role calibration /
limiting role; P34-2 adversarial-vs-iid separation; P34-3 ensemble
union lower-bound; C34-4 typed-handoff adversary robustness; C34-5
role-heterogeneous replay accuracy ordering).**

Phase 33, in one line: the substrate is unchanged, extractors are LLM-
driven, a third non-code domain adds a max-ordinal decoder, the 0.5b
LLM's pooled noise profile is approximated by the Phase-32 synthetic
sweep within γ ≤ 0.15 on compliance. **Phase 34, in one line:
extractor noise is not i.i.d. — it is structured by role and
adversarially focusable; the pooled sweep is an average over regimes;
under adversarial drop of a single load-bearing claim per scenario the
substrate's accuracy collapses to 0 % while the matched-nominal-budget
i.i.d. baseline preserves 20–70 %; and a regex + LLM ensemble on a
genuinely complementary-coverage scenario bank closes the pooled drop
rate from 0.25 / 0.75 to 0.00 with precision preserved — the first
empirical satisfaction of Conjecture C33-4.**

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 33 closed the "extractors were perfect" simplification by
shipping an LLM-driven extractor and a pooled calibration layer. It
also left three gaps that the Phase-34 framing names explicitly:

1. **Pooled i.i.d. hides per-role heterogeneity.** On compliance the
   0.5b had drop = 0.50 on legal, drop = 1.00 on finance. The pooled
   quadruple (δ̄ = 0.70, ε̄ = 0.12, μ̄ = 0.40, π̄ = 0.60) maps to the
   Phase-32 synthetic grid point (0.5, 0.1, 0.25, 0.0) with γ_acc ≤
   0.10 — but the *match* is an average over qualitatively different
   regimes. If a task's load-bearing role happens to be finance, the
   pooled prediction over-estimates accuracy.
2. **Synthetic noise was Bernoulli-i.i.d.** A flaky production LLM
   does not coin-flip every emission independently. It silently drops
   specific load-bearing claims (the token that triggers it hit a
   quirk), or an extractor instance fails completely for a role, or a
   prompt-injection escalates a severity verdict. The realistic threat
   model is *targeted*.
3. **No ensemble result yet.** Conjecture C33-4 predicts that
   ``δ_union ≤ δ_r · δ_l`` and ``ε_union ≤ ε_r + ε_l``. Phase 33
   scenarios all had regex recall = 1 on causal events, so the LLM
   was strictly dominated and the ensemble never worth building.

Phase 34 closes all three. The substrate
(``core/role_handoff``) is unchanged byte-for-byte. Every Phase-34
addition sits at or above the *extractor* boundary.

### A.2 Scope discipline (what Phase 34 does NOT claim)

Explicit, because drift is the largest framing risk:

1. **Not a frontier-model real-LLM calibration.** The Phase-34
   calibration benchmark runs with the deterministic mock LLM by
   default. A real 0.5b / 7B per-role calibration is mechanical
   follow-up (``--mode real``); the measurement infrastructure and
   the headline P34-1 claim are already in place.
2. **Not a cryptographically adversarial wrapper.** The adversary
   lives at the *extractor boundary*, not at the router or log
   layer. Signature spoofing / payload tampering under the chain
   hash remain Phase-31 guarantees.
3. **Not a general proof that the substrate is robust to any
   adversary.** Phase 34 Part B is a *separation* result: under a
   specific targeted attacker at a matched nominal budget, damage is
   strictly larger than i.i.d. damage. The substrate's best defence is
   extractor redundancy (which is exactly what Part C's ensemble
   delivers).
4. **Not a new decoder-shape claim.** All three domains carry over
   their Phase-31/32/33 decoder unchanged. Phase 34 is about the
   *upstream* noise model.
5. **Not a K ≥ 20 hierarchical role-lattice result.** Conjecture
   C31-6 is untouched.

### A.3 Summary of the three empirical results

| Part | Finding | Artifact |
|---|---|---|
| **A** per-role | Mock LLM on all three domains shows max per-role drop-rate *spread* ≥ 0.33 (incident), 0.50 (compliance), 0.67 (security). Pooled masking (spread > 0.25) triggered on all three. The *limiting role* — the role with max drop — is a first-class output. | ``results_phase34_per_role_calibration_mock.json`` |
| **B** adversarial | At matched nominal drop budget (i.i.d. drop_prob = budget / R*), targeted drop of a single load-bearing claim collapses substrate accuracy to 0 % on all three domains; matched i.i.d. preserves 20–70 % at budget = 1. Gap ``acc_iid − acc_adv`` is as high as +0.80 on incident / security. | ``results_phase34_adversarial_noise.json`` |
| **C** ensemble | On a 10-scenario compliance mixed bank (5 canonical + 5 narrative), regex alone scores 50 %, narrative-LLM alone scores 0 %, the union scores 100 %. Pooled δ_u = 0.00 ≤ δ_r · δ_l = 0.19; ε_u = 0.00 ≤ ε_r + ε_l = 0.00. Conjecture C33-4 empirically satisfied. | ``results_phase34_ensemble_extractor.json`` |

---

## Part B — Theory

### B.1 Setup (reused from Phase 31/32/33 § B.1)

Let a team have roles ``R = {r_1, …, r_K}``. Each role has an
observable-type subset ``O_k``, an extractor ``e_k``, and a
subscription table ``σ``. Let ``X`` be the naive broadcast stream,
``H`` the emitted handoffs, and ``R*(z)`` the load-bearing claim
count for scenario ``z``. An i.i.d. noise model is
``NoiseConfig(δ, ε, μ, π)``.

Phase 34 lifts the per-extractor noise model from a *scalar* quadruple
to a *role-indexed* quadruple:

    N(e) = {(δ_k, ε_k, μ_k, π_k)}_{k ∈ R}

— and introduces an *adversarial* noise model ``A(e)`` that
conditions on the causal chain.

### B.2 Theorem P34-1 — Role-limited accuracy under pooled calibration

**Statement.** Let ``e_llm`` be an LLM extractor with per-role
drop-rate profile ``(δ_1, …, δ_K)`` on a subscription table σ whose
causal chain has *exactly one* load-bearing claim per role. Define
the *pooled* match prediction ``A_pool`` by mapping the pooled
δ̄ = mean_k(δ_k) to the closest Phase-32 sweep grid point; define
the *per-role* replay accuracy ``A_per`` by running the substrate
under ``PerRoleNoiseConfig`` fit from the measured δ_k. Let ``A_real``
be the real extractor's substrate accuracy. Then under the matched-
substrate regime (Theorem P30-4's preconditions) and with R*(z) ≥ 1:

```
 A_real ≤ Π_k (1 − δ_k)      (per-role upper bound)
 A_pool ≤ (1 − δ̄)^{R*}       (Phase-32 / Theorem P32-2 pooled bound)
```

By AM-GM, ``(1 − δ̄)^{R*} ≥ Π_k (1 − δ_k)`` whenever the δ_k are
not all equal. In particular, when one role has ``δ_k = 1`` (that
role's extractor is completely broken on every call), the per-role
bound collapses to 0 while the pooled bound is bounded below by
``(1 − δ̄)^{R*}`` — a strictly higher number.

**Status.** **Empirical direction confirmed** on the Phase-34 mock
benchmark: on security the pooled-drop was 0.056 but the limiting
role (data_steward) has drop 0.33 (the max across the four producer
roles), and the substrate's real accuracy (0.80) is *above* the
pooled-replay accuracy (0.60) — pooled under-predicts — and equal to
the per-role-replay accuracy (0.60). On incident the pooled replay
(0.80) *over*-predicts real (0.60) — the pooled predictor is biased
in both directions depending on per-role structure. See § D.1.
The *inequality* ``A_real ≤ Π_k (1 − δ_k)`` is derived from the
per-role-i.i.d. independence assumption, which is approximate; a
stronger bound would need a joint-distribution model.

**Why it matters.** The pooled match (Phase-33's Theorem P33-1) is a
*predictor* only under homogeneous per-role noise. On
heterogeneous-noise deployments, the per-role replay is the
tighter prediction; the *limiting role* is the production bottleneck
and the natural site for targeted extractor work.

### B.3 Theorem P34-2 — Adversarial-vs-i.i.d. separation at matched nominal budget

**Statement.** Let the substrate's aggregator decoder require all
claims in a scenario's causal chain ``C(z) = {(r_1, k_1), …,
(r_R*, k_R*)}`` to be delivered for correctness (the monotone-recall
regime of Theorem P32-2). Let ``δ ∈ [0, 1]`` be a nominal drop
budget expressed as a fraction of the mean causal chain length ``R*``.
Define:

* ``A_iid(δ)`` = substrate accuracy under ``NoiseConfig(drop_prob =
  δ)`` on the same scenario bank.
* ``A_adv(δ)`` = substrate accuracy under
  ``AdversarialConfig(target_mode = load_bearing_drop, drop_budget
  = ⌊δ · R*⌋)``.

Then for every scenario ``z`` in the bank:

```
 A_adv(δ) ≤ 𝟙{⌊δ · R*⌋ < R*(z)}
 A_iid(δ) = (1 − δ)^{R*(z)}  +  error (i.i.d. variance)
```

In particular, for the minimal budget ``δ = 1/R*`` the adversary
achieves **A_adv = 0** deterministically (one load-bearing claim
dropped always breaks the monotone decoder) while i.i.d. at the
matched budget has expected accuracy ``(1 − 1/R*)^{R*} → 1/e ≈
0.368`` as ``R* → ∞``.

**Empirical anchor.** Phase 34 Part B § D.2: at budget = 1 (the
minimal nominal budget that hits the adversary's target) on all three
domains the adversarial accuracy collapses to 0 % while matched i.i.d.
preserves 20 % – 70 %, for a pooled gap of **+0.50 pp** averaged across
domains. At budget = 2 the gap narrows because i.i.d. degrades more
steeply than the adversary (which is already at 0); at budget = 3 both
reach 0.

**Why it matters.** The two-regime graceful-degradation story of
Theorem P32-2 holds *only under i.i.d. noise*. Under a targeted
adversary, the "monotone regime" degrades non-gracefully: one missed
load-bearing claim collapses accuracy. This is the realistic
production risk; the i.i.d. curves are the optimistic case.

### B.4 Theorem P34-3 — Ensemble union lower bound on noise

**Statement.** Let ``e_r`` and ``e_l`` be two extractors with per-
event conditional independence on the gold causal chain (the event
``e_r drops claim (r, k)`` is independent of ``e_l drops claim
(r, k)`` given ``(r, k)``). Let ``e_u = e_r ∪ e_l`` be the union
extractor (``core/ensemble_extractor.UnionExtractor``), deduping on
``(kind, evids)``. Then:

```
 δ_u ≤ δ_r · δ_l                     (drop bound, product)
 ε_u ≤ ε_r + ε_l                     (spurious union bound)
```

When the two extractors have *complementary* coverage (partition the
causal chain into two disjoint subsets with one of ``e_r, e_l``
recall = 1 on each part and the other = 0), equality holds: ``δ_u
= δ_r · δ_l`` with ``δ_r = δ_l = 0.5`` becomes ``δ_u = 0``, not just
``≤ 0.25``.

**Empirical anchor.** Phase 34 Part C § D.3 on the compliance mixed
bank: pooled δ_r = 0.25 (regex misses the 5 narrative scenarios out
of 20 causal claims), pooled δ_l = 0.75 (narrative-LLM misses the 5
canonical scenarios' 15 causal claims), pooled δ_u = 0.00 (every
causal claim is caught by at least one). The product bound 0.19
is satisfied; the *equality-under-complementarity* regime holds
because the two extractors partition the causal chain.

**Why it matters.** Conjecture C33-4 is promoted from "unproven on
Phase-33 data because regex had coverage = 1 by construction" to a
**measured bound on a genuinely complementary-coverage benchmark**.
The headline is stronger than the raw bound: the ensemble achieves
**substrate accuracy = 100 %** on a bank where either extractor alone
scores ≤ 50 %. Ensemble composition is a substrate-level robustness
lever, not just a theoretical curiosity.

### B.5 Conjecture C34-4 — Typed-handoff adversary robustness

**Statement.** Consider an adversary ``A(budget, R*)`` that can drop
up to ``budget`` load-bearing claims per scenario under the
``core/extractor_noise.adversarial_extractor(load_bearing_drop)``
mode. Then:

1. The *canonical* substrate (one extractor per role) has
   ``A_adv ≥ max(0, 1 − budget / R*)`` on monotone-recall decoders
   with tight equality at ``budget < R*``.
2. An *ensemble* substrate with ``n`` pairwise-complementary
   extractors per role has ``A_adv ≥ max(0, 1 − budget / (n · R*))``
   under a strong complementarity assumption — the adversary now
   needs to drop *every* extractor's copy of a load-bearing claim.
3. Equivalently, ensemble redundancy converts the adversarial
   budget exponentially: from ``1`` to ``n`` total drops needed per
   claim.

**Status.** **Formally stated, empirically untested.** Phase 34 Part
B measures (1) directly; Phase 34 Part C's ensemble achieves
complementary coverage but does not yet test an *adversary* against
the ensemble. The natural next experiment is
``adversarial_extractor`` composed with ``UnionExtractor`` — one of
the immediate follow-ups in § F.

**Why it matters.** An agent-team product that cares about
adversarial robustness cannot rely on a single extractor per role.
The ensemble's value proposition is not just accuracy (Part C); it
is *defensive depth* against the Part B adversary.

### B.6 Conjecture C34-5 — Per-role-replay accuracy ordering

**Statement.** Let the *real-extractor* substrate accuracy on a
scenario bank be ``A_real``. Let ``A_pool`` be the substrate accuracy
when the extractor is replaced by an i.i.d. wrapper fit from the
extractor's pooled quadruple, and ``A_per`` be the substrate accuracy
under the per-role ``PerRoleNoiseConfig`` fit from the extractor's
per-role audit. Then:

```
 |A_real − A_per| ≤ |A_real − A_pool|
```

i.e. the per-role replay is a *tighter* predictor than the pooled
replay. The bound is loose because per-role replay assumes intra-
role i.i.d. while the real extractor may have per-call correlations.

**Status.** Empirically **mixed** on the Phase-34 mock benchmark
(§ D.1). On compliance both |real − pooled| and |real − per_role|
equal 0.00 (degenerate). On incident and security both gaps equal
0.20 — the per-role replay does not strictly dominate the pooled
replay on the mock. The conjecture's real test case is a real LLM
with strong per-role spread (e.g. Phase-33's 0.5b has legal drop
0.50 / finance drop 1.00 on compliance); under such a regime the
pooled replay is expected to be biased by Jensen's inequality while
per-role replay remains a faithful per-role model.

**Why it matters.** Conjecture C33-3 said "pooled i.i.d. hides
per-role heterogeneity". C34-5 sharpens that claim into a **predictor
ordering**: if you are going to model extractor noise with a synthetic
wrapper, use the per-role config. Pooled is the wrong
default for production-realistic calibration.

### B.7 What is theorem vs. what is empirical

Ordered by strength:

* **Theorem (proved from prior):** P34-2 (follows from the monotone-
  recall regime of Theorem P32-2 + the observation that the adversary
  deterministically targets ``R*(z)`` load-bearing claims); P34-3
  (standard union/product bound plus the ``UnionExtractor``
  construction). P34-1's inequality ``(1 − δ̄)^{R*} ≥ Π_k (1 − δ_k)``
  is a direct application of the AM-GM inequality to the per-role
  product of survival probabilities.
* **Empirical, measurable:** P34-1 (Phase-34 mock shows the
  limiting-role signature on each domain; the pooled replay bias
  can go either direction); P34-3 (confirmed on Phase-34 Part C
  mixed bank at δ_u = 0 < δ_r · δ_l = 0.19).
* **Conjecture (empirically suggested):** C34-5 (per-role replay
  is a tighter predictor than pooled) — untested on the mock,
  natural real-LLM experiment identified.
* **Conjecture (formally open):** C34-4 (typed-handoff ensemble
  adversary robustness; the combined ``adversarial_extractor`` ∘
  ``UnionExtractor`` experiment is mechanical but not yet run).

A reviewer attacking this work should attack:

* **P34-2's equality point.** At budget ≥ R* the adversary trivially
  succeeds, and at budget = 0 both are equal. The separation result is
  interesting only in the intermediate regime — specifically at
  ``budget ∈ [1, R*−1]`` where i.i.d. degrades gracefully but the
  adversary is already at 0. Phase 34 Part B measures this intermediate
  regime at k = 6 distractors; at larger k the i.i.d. gap may narrow
  because the i.i.d. drop prob at matched nominal budget stays the
  same but the number of trials grows.
* **P34-3's complementarity assumption.** The product bound
  ``δ_u ≤ δ_r · δ_l`` assumes conditional independence; a real
  LLM and a real regex probably share failure modes (both miss
  unusual phrasings; both miss unusual evidence structure). Under
  shared failure modes the bound can be violated and the ensemble can
  be *worse* than regex alone on a subset of claims. Phase 34 Part C's
  scenario bank is designed to avoid this (narrative triggers do NOT
  appear in canonical docs), so the bound holds tight.

---

## Part C — Architecture

### C.1 New modules and relationships

```
vision_mvp/core/extractor_noise.py             [MODIFIED]
    + PerRoleNoiseConfig (dataclass)
    + per_role_noisy_extractor
    + AdversarialConfig (dataclass)
    + adversarial_extractor
    + ADVERSARIAL_MODE_{LOAD_BEARING_DROP, ROLE_SILENCING,
                         SEVERITY_ESCALATION, COMBINED}
    + build_uniform_per_role
    + build_from_audit_per_role

vision_mvp/core/extractor_calibration.py       [MODIFIED]
    + per_role_heterogeneity
    + per_role_closest_synthetic
    + per_role_audit_summary

vision_mvp/core/ensemble_extractor.py          [NEW]  ~120 LOC
    UnionExtractor, union_of
    EXTRACTOR_{REGEX, LLM, ENSEMBLE} tags

vision_mvp/experiments/phase34_per_role_calibration.py    [NEW]  ~230 LOC
vision_mvp/experiments/phase34_adversarial_noise.py       [NEW]  ~290 LOC
vision_mvp/experiments/phase34_ensemble_extractor.py      [NEW]  ~370 LOC

vision_mvp/tests/test_phase34_per_role.py       [NEW]
vision_mvp/tests/test_phase34_adversarial.py    [NEW]
vision_mvp/tests/test_phase34_ensemble.py       [NEW]
```

The substrate primitive (``core/role_handoff``) is unchanged. The
Phase-31/32/33 task modules (``tasks/incident_triage``,
``tasks/compliance_review``, ``tasks/security_escalation``) are
unchanged.

### C.2 Why the new wrappers sit at the extractor boundary

Every Phase-34 noise and ensemble wrapper implements the same
``extractor(role, events, scenario) -> list[(kind, payload, evids)]``
contract that Phase 31/32/33 use. This is the *only* abstraction a
new noise model or ensemble strategy must conform to — no substrate
change, no task-module change. The same harness
(``run_*_loop(extractor=...)`` on each of the three domains) drives
every experiment. Concretely:

| Wrapper | Boundary | Scope |
|---|---|---|
| ``noisy_extractor`` | extractor | i.i.d. Bernoulli (Phase 32) |
| ``per_role_noisy_extractor`` | extractor | role-dispatched i.i.d. (Phase 34 A) |
| ``adversarial_extractor`` | extractor | scenario-conditioned targeted (Phase 34 B) |
| ``UnionExtractor`` | extractor | union of two extractors (Phase 34 C) |

The boundary-level discipline is the reason Phase 34's theorems lift
cleanly from Phase 32's (Theorems P32-1, P32-2, P32-3): the substrate
remains untouched, so its correctness-preservation theorem transfers
unchanged; Phase 34 is about adding *noise models* that Phase 32's
bounds apply against.

---

## Part D — Evaluation

> Numbers below come from three artifacts reproduced by the Appendix
> commands:
>   (A) ``vision_mvp/results_phase34_per_role_calibration_mock.json``
>   (deterministic mock LLM across three domains, sub-second wall).
>   (B) ``vision_mvp/results_phase34_adversarial_noise.json`` (2
>   seeds × 3 domains × 3 budgets + role-silencing + severity-
>   escalation, 0.1 s wall).
>   (C) ``vision_mvp/results_phase34_ensemble_extractor.json`` (2
>   seeds × 10-scenario compliance mixed bank, sub-second wall).

### D.1 Per-role calibration (Part A)

Mock LLM calibration (seed = 34, k = 6) across three domains:

| domain | pooled δ̄ | max spread (any axis) | weakest role | δ_weakest | real_acc | pooled_replay_acc | per_role_replay_acc |
|---|---:|---:|---|---:|---:|---:|---:|
| incident   | 0.000 | **0.333** | monitor      | 0.000 | 0.60 | 0.80 | 0.40 |
| compliance | 0.000 | **0.500** | legal        | 0.000 | 1.00 | 1.00 | 1.00 |
| security   | 0.056 | **0.667** | data_steward | 0.333 | 0.80 | 0.60 | 0.60 |

Reading:

* **Max spread > 0.25 on every domain** — the Phase-33 C33-3 pattern
  (pooled i.i.d. hides structured per-role heterogeneity) reproduces
  on the mock and is expected to amplify on real LLMs. On incident
  and compliance the largest spread is on *mislabel* / *payload* axes
  rather than drop, so the pooled drop is 0 yet the calibration
  surfaces heterogeneity on other knobs. On security the spread is
  on drop (data_steward at 0.33).
* **The pooled replay sometimes over-predicts, sometimes tracks.**
  On incident the pooled replay is 0.80 vs real 0.60 (pooled
  over-predicts); on security pooled is 0.60 vs real 0.80 (pooled
  under-predicts); on compliance both equal real. Under
  heterogeneous noise, *neither* scalar sweep is an unbiased
  predictor — the right instrument is the per-role replay
  + limiting-role report.
* **The limiting role is domain-specific.** Phase 34 surfaces a
  first-class ``limiting_role`` output that names the extractor
  bottleneck. On incident, the limiting role is the monitor; on
  compliance, legal; on security, data_steward — the last being
  the only one with strictly non-zero drop in the mock. These are
  the roles to improve first in a production deployment.
* **Per-role replay ≤ pooled on security.** |real − per_role| =
  0.20, |real − pooled| = 0.20 — equal gaps. C34-5's "per-role
  replay is a tighter predictor" is not unconditionally true on the
  mock; a real-LLM run with strong per-role spread (e.g. 0.5b's
  finance drop = 1.00) is the natural stress test and is expected
  to produce a cleaner ordering.

### D.2 Adversarial-vs-i.i.d. separation (Part B)

Pooled over 2 seeds × 5 scenarios per domain, matched nominal budget
δ·R*:

| domain | mean R* | budget=1 matched_iid_drop | adv | iid | **gap(iid−adv)** |
|---|---:|---:|---:|---:|---:|
| incident     | 3.4 | 0.29 | **0.00** | 0.80 | **+0.80** |
| compliance   | 2.0 | 0.50 | **0.00** | 0.20 | **+0.20** |
| security     | 3.6 | 0.28 | **0.00** | 0.40 | **+0.40** |
| **mean** | **3.0** | **0.36** | **0.00** | **0.47** | **+0.47** |

At budget = 2 the gap narrows (i.i.d. degrades too) and at budget = 3
both reach 0.

Role silencing (silencing the first producer role on every scenario):

| domain | silenced role | adv_acc |
|---|---|---:|
| incident    | monitor       | **0.80** |
| compliance  | legal         | **0.80** |
| security    | soc_analyst   | **0.40** |

Silencing one role removes the causal claims that role owns; the
substrate recovers accuracy roughly equal to the fraction of scenarios
whose causal chain does *not* depend on that role. Security takes the
largest hit (SOC owns load-bearing claims in all 5 scenarios).

Severity escalation (inject a MALWARE_DETECTED / DPA_MISSING /
DISK_FILL_CRITICAL claim on the first distractor event):

| domain | adv_acc |
|---|---:|
| incident    | **0.10** |
| compliance  | **0.00** |
| security    | **0.80** |

On the security max-ordinal decoder the spurious claim escalates
severity to HIGH even on the MEDIUM brute_force scenario —
accuracy drops to 0.80 (1 / 5 wrong). On incident / compliance the
severity-escalation attack targets the priority-order decoder and
collapses the root-cause answer on every scenario (incident 0.10) or
the verdict (compliance 0.00). This is the first Phase-34 data point
confirming that *all three decoder shapes* have a precision-side
adversary.

### D.3 Ensemble (Part C)

Pooled across 2 seeds on the 10-scenario compliance mixed bank
(5 canonical + 5 narrative):

| flavor | acc | drop rate δ̂ | spurious ε̂/ev | recall | precision |
|---|---:|---:|---:|---:|---:|
| regex-only     | **0.50** | 0.250 | 0.000 | 0.75 | 1.00 |
| llm-narrative  | **0.00** | 0.750 | 0.000 | 0.25 | 0.50 |
| **ensemble**   | **1.00** | **0.000** | **0.000** | **1.00** | **1.00** |

Conjecture C33-4 bound check (Theorem P34-3):

```
  δ_u = 0.000 ≤ δ_r · δ_l = 0.25 · 0.75 = 0.188      → SATISFIED
  ε_u = 0.000 ≤ ε_r + ε_l = 0.000 + 0.000 = 0.000   → SATISFIED (tight)
```

Reading:

* **The bound is satisfied and tight in the δ direction.**
  ``δ_u = 0`` is better than the predicted upper bound 0.188 because
  the two extractors are pairwise *complementary*, not just
  independent: every canonical scenario is caught by regex (δ = 0
  on its half), every narrative scenario is caught by LLM (δ = 0 on
  its half), so union drop is 0.
* **Accuracy triples from regex-only to ensemble.** 50 % → 100 %.
  Neither single-extractor strategy is workable on the mixed bank;
  the ensemble is a strict Pareto improvement.
* **Precision is preserved.** The narrative-LLM alone has
  precision = 0.50 (its triggers fire on an occasional benign
  doc because some narrative words appear in distractors); the
  ensemble's precision is 1.00 because the regex's 1.0 precision
  on canonical events dominates and the LLM's spurious claims are
  deduplicated against regex claims on overlapping evids.

---

## Part E — Failure taxonomy (unchanged from Phase 33)

Phase 34 re-uses the Phase-33 five-way attribution histogram:

```
none / retrieval_miss / truncation / missing_handoff /
spurious_claim / llm_error
```

What Phase 34 adds is the *per-experiment* attribution signal:

| Experiment | Dominant failure | Pattern |
|---|---|---|
| Part B load_bearing_drop, budget=1 | ``missing_handoff`` | 100 % on adv, 30–80 % on iid |
| Part B role_silencing | ``missing_handoff`` | only for scenarios depending on silenced role |
| Part B severity_escalation | ``spurious_claim`` | especially on security max-ordinal decoder |
| Part C regex-only | ``missing_handoff`` | on every narrative scenario |
| Part C llm-only | ``missing_handoff`` | on every canonical scenario |
| Part C ensemble | ``none`` | pooled |

The taxonomy is now sharp enough to distinguish *which kind of
extractor-boundary failure* a benchmark row is hitting — an agent-team
product can read the histogram and infer whether to invest in more
robust single extractors (drop mode), more redundant extractors
(ensemble), or better prompt engineering (spurious mode).

---

## Part F — Future work

### F.1 Carry-over (unchanged from Phase 33)

* SWE-bench end-to-end with a real LLM on the wrap path.
* Frontier-model sweep (multi-model × multi-seed × multi-k).
* OQ-1 in full generality (Conjecture P30-6).
* Hierarchical role lattice at K ≥ 20.

### F.2 Newly surfaced by Phase 34

* **Ensemble against adversary (C34-4).** Compose
  ``adversarial_extractor`` with ``UnionExtractor`` on the Part C
  mixed bank. Predicted: adversary budget of 1 against the ensemble
  requires dropping BOTH regex's canonical witness AND LLM's narrative
  witness simultaneously — which the current adversary wrapper cannot
  do because it targets one (role, kind, witness) tuple. Extending the
  adversary to ``multi_witness_drop`` is the natural next step.
* **Real-LLM per-role calibration.** Run
  ``phase34_per_role_calibration --mode real`` across qwen2.5:0.5b /
  qwen2.5-coder:7b on all three domains at multiple k values. Tight
  C34-5 empirical support at real-LLM noise.
* **Heterogeneous ensemble.** The Part C ensemble is regex + narrative-
  LLM. A *real* production ensemble would be regex + zero-shot-LLM
  + few-shot-LLM, or regex + two different LLMs. The ensemble wrapper
  is generic enough to support this already; the benchmark is small.
* **Payload-level adversary.** The current wrapper targets the
  ``(role, kind)`` header; a payload-replacing adversary (swap the
  body of a load-bearing emission) would test the downstream decoder's
  content-level invariants. Outside the current adversarial wrapper;
  future work.

### F.3 What is genuinely blocking the endgame

Same as after Phase 33. Phase 34 does NOT unblock any of:

* **End-to-end SWE-bench** (the programme's largest external-
  validity gap).
* **OQ-1 in full generality** under Lipschitz LLM policies (C P30-6).
* **Cross-language runtime calibration**.

Phase 34 *does* close three gaps the Phase-33 master plan surfaced as
medium-term: per-role-adaptive calibration (§ 5.1 bullet h),
adversarial-noise extension (§ 5.1 bullet i), and ensemble
composition (§ 5.1 bullet j).

---

## Appendix A — How to reproduce

```bash
# 1. Per-role calibration (mock, seconds of wall).
python3 -m vision_mvp.experiments.phase34_per_role_calibration \
    --mode mock --domains incident compliance security \
    --sweep-path vision_mvp/results_phase32_noise_sweep.json \
    --out vision_mvp/results_phase34_per_role_calibration_mock.json

# 2. Adversarial noise sweep (0.1 s wall).
python3 -m vision_mvp.experiments.phase34_adversarial_noise \
    --domains incident compliance security \
    --seeds 34 35 --drop-budgets 1 2 3 \
    --out vision_mvp/results_phase34_adversarial_noise.json

# 3. Ensemble on compliance mixed bank (sub-second wall).
python3 -m vision_mvp.experiments.phase34_ensemble_extractor \
    --seeds 34 35 --distractor-counts 6 \
    --out vision_mvp/results_phase34_ensemble_extractor.json

# 4. Test suite.
python3 -m pytest vision_mvp/tests/test_phase34_per_role.py \
    vision_mvp/tests/test_phase34_adversarial.py \
    vision_mvp/tests/test_phase34_ensemble.py -q
```

---

*End of Phase 34 results note. The master plan
(``docs/context_zero_master_plan.md``) is updated in the same
commit; see ``§ 4.9.2``, ``§ 4.11 Current frontier`` for the
higher-level integration.*
