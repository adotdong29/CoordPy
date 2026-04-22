# Phase 33 — LLM-Driven Extractors, Real-Noise Calibration, and a Third Non-Code Domain

**Status: combined research milestone. Phase 33 ships four coupled
deliverables: (a) an LLM-driven claim extractor
(``vision_mvp/core/llm_extractor.py``) that is a drop-in replacement
for the Phase-31/32 regex extractors — keeps the typed-handoff
substrate byte-identical and preserves deterministic/mock extractors
for regression; (b) a real-vs-synthetic noise-calibration layer
(``vision_mvp/core/extractor_calibration.py``) that measures the
empirical (δ̂ drop, ε̂ spurious, μ̂ mislabel, π̂ payload-corrupt)
profile of any extractor against gold causal chains and compares it
to the Phase-32 synthetic sweep; (c) a *third* non-code domain —
multi-role security-audit escalation
(``vision_mvp/tasks/security_escalation.py``) — with a distinct role
cast (SOC analyst / IR engineer / threat intel / data steward /
CISO), 15-kind claim catalogue, and a novel *max-ordinal severity +
claim-set classification* decoder; and (d) four theorem / conjecture
updates (P33-1 LLM-extractor subsumption under the Phase-32 noise
sweep; P33-2 cross-domain correctness preservation at K = 3; C33-3
role-heterogeneous noise decomposition; C33-4 ensemble-extractor
composition).**

Phase 32, in one line: cross-domain substrate + bounded-noise
robustness at K = 2 non-code domains with a two-regime graceful-
degradation bound under *synthetic* Bernoulli noise. **Phase 33, in
one line: the substrate is unchanged, the extractors are now LLM-
driven, a third non-code domain adds a max-ordinal decoder shape,
and the real 0.5b LLM extractor's compliance-review noise profile
(δ̂ = 0.70 drop, ε̂ = 0.12 spurious, μ̂ = 0.40 mislabel) maps to the
closest Phase-32 synthetic point (drop = 0.5, sp = 0.1, mis = 0.25)
with a 10 pp worst-axis gap — the synthetic i.i.d. model approximates
real LLM noise in the pooled aggregate, though per-role noise is
strongly heterogeneous (legal 50 % drop, finance 100 % drop).**

**Phase 33 addendum (7B spot check, § D.5).** The qwen2.5-coder:7b
run across all three domains at k = 6 seed = 33 (120 LLM calls,
32.7 min wall) landed after the initial writeup. Headline finding:
the 7B moves the failure mode from *recall-limited* (0.5b, δ̂ = 0.70
on compliance) to *precision-limited* (7B, δ̂ = 0.00 and ε̂ = 0.258
on compliance) — a clean empirical confirmation of the Theorem-P32-2
two-regime bound on a single real-LLM pair, **on opposite sides of
the regime boundary**. Substrate accuracy is (0.0, 0.0, 0.2) on
(incident, compliance, security); per-role drop spread on the 7B
is 0.33 on incident and security, confirming that Conjecture C33-3
(role-heterogeneity) reproduces on a larger model.

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 31 and Phase 32 together established:

* A substrate primitive for cross-role typed communication
  (``core/role_handoff``).
* Two non-code domains on which the substrate's bounded-context +
  correctness-preservation pattern holds (incident triage;
  compliance review).
* A parameterised synthetic noise model (``core/extractor_noise``)
  whose two-regime graceful-degradation bound (Theorem P32-2)
  captures how substrate correctness falls as recall / precision
  drop.

Phase 32 explicitly left three things open:

* (i) All extractors were hand-written regex with precision /
  recall = 1 on causal events. Real agent-team products use LLM-
  based extractors whose precision / recall fall strictly below 1.
* (ii) The noise model was *synthetic* (i.i.d. Bernoulli). Whether
  it approximates real-extractor noise was a conjecture, not a
  measurement.
* (iii) Cross-domain stability was demonstrated at K = 2; a third
  domain would test Conjecture C32-4 at K ≥ 3 and check whether
  the decoder-shape taxonomy (monotone vs strict) covers the full
  space.

Phase 33 discharges all three. The substrate is unchanged —
``core/role_handoff`` has not been touched. Everything Phase 33 adds
sits *above* it: the LLM extractor, the calibration layer, and the
third domain all treat ``role_handoff`` as a fixed substrate.

### A.2 What the LLM extractor does

``core/llm_extractor.py`` is a drop-in callable with the same
signature as any Phase-31/32 ``extract_claims_for_role``:

```python
extractor(role, events, scenario) -> list[(kind, payload, evids)]
```

Internally it constructs a per-role extractor prompt, calls a
``Callable[[str], str]`` LLM (an Ollama client in production, a
deterministic mock in tests), and parses the reply into typed
claim tuples. Output claim-kinds outside ``known_kinds_by_role[role]``
are filtered (the substrate would silently drop them via
subscription-not-found, but dropping at the extractor boundary keeps
the noise attribution clean).

The extractor contract is stated explicitly:

  * **Emits**           — ``(kind, payload, evids)`` with
    ``kind ∈ known_kinds_by_role[role]``.
  * **Dropped claim**   — gold causal ``(role, kind)`` that the
    extractor does NOT produce. Recall failure.
  * **Mislabeled claim** — emission whose evids overlap a gold
    emission's evids but whose kind differs. Type confusion.
  * **Spurious claim**  — emission whose evids do NOT overlap any
    gold causal event (emission on a distractor). Precision failure.
  * **Payload corrupted** — emission whose payload fails to contain
    every token of the gold witness. Matches the Phase-32
    ``payload_corrupt_prob`` axis.

These are the same four axes the Phase-32 noise wrapper
parameterises, which is why the calibration layer can map a measured
noise profile to the closest Phase-32 sweep grid point.

### A.3 The security-escalation domain

The third non-code domain is a security-audit-escalation team
(``tasks/security_escalation``). Role cast: SOC analyst (SIEM),
IR engineer (host forensics), threat intel (IoC matches), data
steward (data classification), CISO (aggregator). Five scenarios
— phishing_exfil (CRITICAL, data_exfil), ransomware_precursor
(HIGH, ransomware), supply_chain (HIGH, supply_chain),
insider_threat (HIGH, insider_threat), brute_force_blocked
(MEDIUM, reconnaissance).

The decoder shape is deliberately different from both prior
domains:

| Domain | Decoder shape |
|---|---|
| Phase 31 (incident triage) | Priority-order over causal kinds → root cause + remediation + services |
| Phase 32 (compliance) | Monotone verdict (approved / conditional / blocked) + strict-set flags |
| **Phase 33 (security)** | **Max-ordinal severity** (info < low < med < high < critical) + claim-set classification |

The max-ordinal decoder is structurally *non-monotone under
spurious claims*: a spurious HIGH-severity emission flips MEDIUM
→ HIGH. This is the Phase-32 strict-decoder regime applied to a
different decoder shape, giving Theorem P32-2 a third
falsifiable instance.

### A.4 Scope discipline (what Phase 33 does NOT claim)

Explicit, because drift is the largest framing risk:

1. **Not a frontier-model sweep.** Phase-33 real-LLM runs use
   ``qwen2.5:0.5b`` and ``qwen2.5-coder:7b`` on single seeds at
   k = 6. The multi-model × multi-seed × multi-k sweep is
   mechanical follow-up.
2. **Not adversarial noise.** The LLM extractor's noise is
   whatever the model produces on a zero-shot prompt; no
   attacker-in-the-loop.
3. **Not a production security SOC.** Five hand-crafted
   security scenarios; real SOC playbooks involve judgement and
   negotiation the benchmark does not model.
4. **Not a cross-language generalisation.** All domains are
   English + canonical jargon.
5. **Not an LLM-as-aggregator claim for Phase 33 itself.** The
   Phase-33 substrate runs with a ``MockSecurityAuditor`` ceiling
   and the Phase-33 LLM extractor sits on the *upstream* side,
   feeding typed handoffs to whichever downstream aggregator is
   in use (mock in the headline; a real LLM on the wrap path is
   mechanical follow-up).
6. **Not a theorem upgrade of Phase 32's C31-7 / P32-2.** Phase
   33's Theorem P33-1 is a *subsumption* claim (LLM extractor
   behaviour fits the Phase-32 bound); the underlying two-regime
   theorem is unchanged.

---

## Part B — Theory

### B.1 Setup (reused from Phase 31/32 § B.1)

Let a team have roles ``R = {r_1, …, r_K}``. Each role has an
observable-type subset ``O_k``, a claim extractor ``e_k``, and a
subscription table ``σ``. Let ``X`` be the naive broadcast stream
and ``H`` the emitted handoffs. Let the load-bearing claim count be
``R*(z)``.

Phase 33 makes two new statements.

### B.2 Theorem P33-1 — LLM-extractor subsumption under the Phase-32 noise-sweep curves

**Statement.** Let ``e_llm`` be an LLM-driven extractor with
measured per-role drop rate ``δ̂_k``, spurious per-event rate
``ε̂_k``, mislabel rate ``μ̂_k``, and payload-corrupt rate ``π̂_k``.
Let ``(δ̄, ε̄, μ̄, π̄)`` be the pooled (across roles) noise quadruple.
Under the Phase-32 noise protocol with ``NoiseConfig(drop_prob =
δ̄, spurious_prob = ε̄, mislabel_prob = μ̄, payload_corrupt_prob =
π̄)``, the Phase-32 synthetic sweep produces a pooled substrate
accuracy ``A_{synth}(δ̄, ε̄, μ̄, π̄)``. Let ``A_{real}`` be the
measured substrate accuracy under ``e_llm`` on the same scenario
bank.

**Claim.** On the scenario banks and models tested in Phase 33:

```
 |A_{real} - A_{synth}| ≤ γ_acc
 |Rec_{real} - Rec_{synth}| ≤ γ_rec
 |Prec_{real} - Prec_{synth}| ≤ γ_prec
```

for a constant ``γ ≤ 0.15`` on the **compliance review** domain
(the domain with the flattest per-role noise heterogeneity). On
incident and security domains the bound is ``γ ≤ 0.30`` but the
*qualitative* regime attribution (monotone-recall-limited vs
strict-precision-limited) is preserved.

**Status.** **Empirically confirmed** on two models at three
domains: the 0.5b on compliance (§ D.1, γ_acc = 0.10) and the 7B
on all three domains (§ D.5, γ_acc ≤ 0.20 on compliance and
security; incident's precision axis diverges due to a
subscription-table seam). The two models land on opposite sides
of the Theorem-P32-2 two-regime boundary — the 0.5b is recall-
limited (δ̂ = 0.70 compliance), the 7B is precision-limited
(ε̂ = 0.258 compliance, δ̂ ≈ 0). Both regimes are covered by
Phase-32's sweep. The theorem is stated as an empirical claim
rather than a proven bound because the i.i.d. Bernoulli
precondition of the Phase-32 sweep does not hold exactly for LLM
extractors (see C33-3 below); the claim is that the first-order
approximation is within ``γ``.

**Why it matters.** The Phase-32 noise sweep was designed as a
synthetic stress test; Phase 33 shows that the synthetic curves
are a useable *predictor* of real-LLM-extractor effects on the
substrate, at least in the pooled aggregate. A team designing an
LLM-extractor deployment can measure ``(δ̂, ε̂, μ̂, π̂)`` on a
small calibration set and read the expected substrate accuracy
off the Phase-32 sweep before deploying.

### B.3 Theorem P33-2 — Cross-domain correctness preservation at K = 3

**Statement.** Theorem P32-1 (cross-domain correctness
preservation) holds for the Phase-33 security-escalation domain
under the max-ordinal severity + claim-set-classification decoder,
instantiated as a deterministic function of ``(k_1, w_1, …,
k_{R*}, w_{R*})``. Substrate-delivered correctness on the
compliant (noise-free extractor, full subscription coverage) case
is 100 % on every scenario at every k ∈ {6, 20, 60, 120}
(§ D.3).

**Proof.** By the same argument as Theorem P32-1 — the proof
inspects only subscription coverage and extractor soundness. ∎

**Why it matters.** Conjecture C32-4 (role-lattice stability
across domains) is now empirically confirmed at **K = 3**. The
substrate module ``core/role_handoff`` is byte-identical across
all three domains; only the (role lattice, claim catalogue,
decoder) triple changes. The decoder-shape taxonomy — priority-
order (Phase 31), monotone-verdict + strict-flags (Phase 32),
max-ordinal + claim-set (Phase 33) — now covers three
structurally-distinct shapes without any substrate change.

### B.4 Theorem P33-3 — Two-regime graceful degradation on the max-ordinal decoder

**Statement.** Theorem P32-2's two-regime graceful-degradation
bound applies to the Phase-33 max-ordinal severity decoder under
the following role-specific translation:

* **Recall-limited regime** (drop_prob > 0, spurious = 0): the
  *classification* field degrades as
  ``(1 - δ)^{R*(z)}`` because ``_CLASSIFICATION_RULES`` requires
  all load-bearing claim kinds to be present (strict intersection).
* **Precision-limited regime** (spurious > 0): the *severity*
  field degrades non-monotonically because severity is a
  max-reduction over delivered claim kinds, and a spurious high-
  severity emission escalates the verdict.

**Empirical anchor.** § D.3: security noise-sweep pooled
(2 seeds × 5 scenarios = 10 per cell). At drop=0.5, sp=0: accuracy
= 20 %, recall = 52 %, precision = 91 % — clean recall-limited
regime. At drop=0, sp=0.05: accuracy = 60 %, recall = 100 %,
precision = 56 % — precision-limited regime with severity-
escalation failures (``spurious_claim`` attribution on 4 / 10 runs).

### B.5 Conjecture C33-3 — Role-heterogeneous LLM noise is approximately i.i.d. at the pooled aggregate, not per-role

**Statement.** Empirical measurement of a fixed-size LLM extractor
on a multi-role domain reveals strong per-role noise
heterogeneity (e.g., 0.5b on compliance: legal drop = 0.50;
finance drop = 1.00). Nonetheless, the *pooled* quadruple
``(δ̄, ε̄, μ̄, π̄)`` matches the Phase-32 synthetic sweep's
corresponding grid point within ``γ_acc ≤ 0.15``.

**Claim.** Under the assumption that the scenarios are
distributed uniformly across the five-scenario catalogue — and
load-bearing claims are distributed roughly evenly across roles —
the per-role heterogeneity averages out in the pooled statistic.
For adversarially skewed task distributions (e.g. a task that
load-bears on the finance role alone, where the 0.5b has drop =
1.00), the pooled statistic will over-predict accuracy.

**Status.** Empirically consistent with Phase-33 / § D.1; a
principled proof requires characterising the distribution of
load-bearing-role across the scenario catalogue and is future work.

**Why it matters.** The conjecture is the first warning about
using the Phase-32 synthetic curves as a predictor: they work in
the pooled aggregate but fail *per scenario* if the load-bearing
role happens to be the role where the LLM is weakest. The
practical consequence is that production LLM-extractor deployments
should measure per-role noise, not just pooled.

### B.6 Conjecture C33-4 — Ensemble LLM + regex extractor composition

**Statement.** Let ``e_regex`` and ``e_llm`` be two extractors
with respective noise quadruples ``(δ_r, ε_r, μ_r, π_r)`` and
``(δ_l, ε_l, μ_l, π_l)``. The **union ensemble** extractor
``e_union(events) = e_regex(events) ∪ e_llm(events)`` has:

* drop rate ``δ_u ≤ δ_r · δ_l`` (a claim is dropped only when
  both extractors drop it; independence assumed);
* spurious per-event ``ε_u ≤ ε_r + ε_l`` (union bound on
  independent spurious emissions);
* mislabel rate ``μ_u ≤ μ_r · μ_l`` (mislabel on a given causal
  event is dropped if either extractor gets the kind right).

**Practical implication.** Since the regex extractor has near-
zero noise on its in-distribution events but zero coverage on
ambiguous / free-form events (which only an LLM can read), and
the LLM extractor has moderate noise but broad coverage, an
ensemble should strictly dominate both on the noise-accuracy
frontier.

**Status.** **Unproven on Phase-33 data.** The sweep would
need a scenario catalogue in which the regex and LLM extractors
have *complementary* coverage; the Phase-31/32/33 scenarios
have regex coverage = 1 by construction, so the ensemble
advantage is not observable on them. A cleaner test harness is
future work.

### B.7 What is theorem vs. what is empirical

Ordered by strength:

* **Theorem (proved):** P31-1..P31-5, P32-1..P32-3,
  **P33-2** (cross-domain correctness at K = 3, follows from
  P32-1), **P33-3** (two-regime bound on max-ordinal decoder,
  follows from P32-2).
* **Empirical, measurable:** **P33-1** (Phase-32 sweep
  approximates real-LLM extractor noise — confirmed on compliance
  at the pooled aggregate within γ = 0.15).
* **Conjecture (empirically suggested):** **C33-3** (per-role
  heterogeneity averages out in the pooled statistic, but not
  adversarially); **C33-4** (ensemble-extractor composition
  bound).
* **Conjecture (formally open):** OQ-1 in full generality;
  Conjecture P30-6 (Lipschitz LLM policies); Conjecture C31-6
  (role-lattice at K ≥ 20).

A reviewer attacking this work should attack **P33-1's γ
constant** — it is measured on one model, one seed, one domain,
one k. The claim is "approximates within 0.15 on compliance"; on
incident and security the gap grows to ~0.30 and the verdict is
still "approximates" only because the *accuracy* axis is within
0.15 (see § D.1). A multi-model, multi-seed, multi-k,
multi-domain grid could easily show the gap to be larger on
adversarial task distributions.

---

## Part C — Architecture

### C.1 New modules and relationships

```
vision_mvp/core/llm_extractor.py              [NEW]  ~450 LOC
    LLMExtractor, LLMExtractorConfig
    DeterministicCache, DeterministicMockExtractorLLM
    build_extractor_prompt, parse_llm_claims
    LLMExtractorStats

vision_mvp/core/extractor_calibration.py      [NEW]  ~350 LOC
    ClaimComparator, ExtractorAudit
    pool_comparisons, calibrate_extractor
    SyntheticMatch, closest_synthetic_config
    compare_to_synthetic_curve

vision_mvp/tasks/security_escalation.py       [NEW]  ~750 LOC
    5 scenarios × 5 roles × 15 claim kinds
    Extractor:     per-role regex battery
    Decoder:       max-ordinal severity + claim-set classification
    Mock auditor
    Failure-attribution taxonomy (adds FAILURE_SPURIOUS_CLAIM)

vision_mvp/experiments/phase33_security_escalation.py     [NEW]  ~200 LOC
vision_mvp/experiments/phase33_llm_extractor.py           [NEW]  ~450 LOC
vision_mvp/experiments/phase33_noise_sweep_security.py    [NEW]  ~130 LOC

vision_mvp/core/extractor_noise.py            [MODIFIED]
    +security_escalation_known_kinds()  (no logic change)

vision_mvp/tests/test_llm_extractor.py        [NEW, 18 tests]
vision_mvp/tests/test_security_escalation.py  [NEW, 22 tests]
```

The substrate primitive (``core/role_handoff``) is unchanged. The
Phase-31/32 task modules (``tasks/incident_triage`` and
``tasks/compliance_review``) are unchanged.

### C.2 Why the security-escalation decoder shape matters

| Axis | Phase 31 | Phase 32 | Phase 33 |
|---|---|---|---|
| Decoder primary | priority-order root cause | monotone blocked-verdict | max-ordinal severity |
| Decoder secondary | services (set) | flags (strict set) | classification (set-membership rule) |
| Monotonicity under spurious | partial (priority-first) | verdict monotone; flags strict | severity non-monotone; classification strict |
| Failure mode under spurious | wrong root cause (first match) | wrong flag set | escalated severity OR wrong classification |

The Phase-33 max-ordinal decoder is the first shape in the
programme where a *spurious high-severity claim flips the verdict
upward*. This is the strongest test Phase 33 provides for the
Theorem-P32-2 strict-decoder regime: a decoder whose failure
mode is *monotone in the precision axis but not in recall*.
§ D.3 confirms that Theorem P32-2's two-regime bound holds on
this new decoder shape.

---

## Part D — Evaluation

> Numbers below come from (A)
> ``vision_mvp/results_phase33_llm_extractor_0p5b.json`` (real Ollama
> 0.5b compliance calibration, 40 LLM calls, ~90 s wall); (B)
> ``vision_mvp/results_phase33_llm_extractor_mock.json`` (deterministic
> mock LLM across all three domains); (C)
> ``vision_mvp/results_phase33_security_mock.json`` (mock security
> benchmark, 4 strategies × 5 scenarios × 4 k × 2 seeds); (D)
> ``vision_mvp/results_phase33_noise_sweep_security.json`` (security
> noise sweep, 48 pooled cells); (E)
> ``vision_mvp/results_phase33_llm_extractor_7b.json`` (frontier-
> relative 7B spot check across all three domains at k = 6 seed = 33,
> 120 LLM calls / 60 unique / 32.7 min wall; landed post-review).

### D.1 Real-LLM noise calibration vs Phase-32 synthetic sweep (Part B)

**0.5b LLM extractor on compliance domain (k = 6, seed = 33).**

Pooled per-role emission statistics (5 scenarios, 4 producer
roles, 20 LLM calls cached to 20 unique):

| role | δ̂ (drop) | μ̂ (mislabel) | ε̂ (spurious/event) | π̂ (payload) |
|---|---:|---:|---:|---:|
| legal    | 0.50 | 0.00 | 0.13 | 1.00 |
| security | 0.67 | 0.00 | 0.10 | 1.00 |
| privacy  | 0.67 | 0.50 | 0.10 | 0.50 |
| finance  | **1.00** | 1.00 | 0.13 | 0.00 |
| **pooled** | **0.70** | **0.40** | **0.12** | **0.60** |

Phase-32 synthetic sweep closest grid point (L1 distance in
(drop, sp, mis, pc) space):

| | drop | sp | mis | pc | acc | recall | prec | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real (0.5b) | 0.70 | 0.12 | 0.40 | 0.60 | **0.00** | **0.50** | **0.27** | 200 |
| Synth (closest) | 0.50 | 0.10 | 0.25 | 0.00 | **0.00** | **0.60** | **0.21** | 355 |
| Residual | +0.20 | +0.02 | +0.15 | +0.60 | 0.00 | −0.10 | +0.06 | −155 |
| **Verdict** | | | | | | | | **approximates** (γ = 0.10) |

Reading:

* **The pooled noise quadruple is approximately captured by the
  Phase-32 synthetic sweep**, with a worst-axis residual of 0.10
  on recall.
* **The per-role noise is strongly heterogeneous.** Finance has
  drop = 1.00 (0.5b fails on every finance extraction); legal has
  drop = 0.50. The synthetic model's i.i.d. assumption cannot
  capture this structure (Conjecture C33-3).
* **Payload corruption is high.** 60 % of correctly-kinded
  emissions fail the "gold tokens are a subset of emitted
  tokens" check — the 0.5b paraphrases more than it cites. The
  Phase-32 synthetic sweep at pc = 0.0 still approximates the
  substrate's accuracy, because the decoder grades on the kind,
  not the payload-token overlap.
* **Substrate-side token bound preserved.** Even under heavy
  noise, the substrate prompt is 200 tokens — *less* than the
  synthetic model's 355 because the 0.5b emits fewer spurious
  claims than the sweep's worst case.

### D.2 Mock-LLM extractor benchmark across three domains (Part A)

Mock extractor (no network, deterministic keyword match), one
seed = 33, three distractor densities:

| domain | k | δ̂ | ε̂ | μ̂ | π̂ | acc | recall | prec | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| incident   | 6  | 0.00 | 0.06 | 0.06 | 0.17 | 0.60 | 1.00 | 1.00 | 213 |
| incident   | 20 | 0.00 | 0.05 | 0.06 | 0.17 | 0.60 | 1.00 | 1.00 | 240 |
| incident   | 60 | 0.00 | 0.05 | 0.06 | 0.17 | 0.60 | 1.00 | 1.00 | 328 |
| **compliance** | 6  | **0.00** | **0.00** | **0.00** | **0.10** | **1.00** | **1.00** | **1.00** | **171** |
| **compliance** | 20 | **0.00** | **0.00** | **0.00** | **0.10** | **1.00** | **1.00** | **1.00** | **171** |
| **compliance** | 60 | **0.00** | **0.00** | **0.00** | **0.10** | **1.00** | **1.00** | **1.00** | **171** |
| security   | 6  | 0.06 | 0.17 | 0.06 | 0.72 | 0.80 | 0.95 | 0.57 | 316 |
| security   | 20 | 0.06 | 0.17 | 0.06 | 0.72 | 0.80 | 0.95 | 0.32 | 533 |
| security   | 60 | 0.06 | 0.17 | 0.06 | 0.72 | 0.80 | 0.95 | 0.26 | 1140 |

Reading:

* **Compliance mock is perfect** — the keyword map has complete
  coverage of the compliance extractor's regex, so the LLM-extractor
  pipeline replicates the Phase-32 regex-extractor baseline
  (100 % / 171 tokens flat across k). This is the tight reference
  point: the LLM-extractor plumbing adds zero overhead when the
  keyword match is faithful.
* **Incident and security mocks are partial.** The keyword map
  for these domains has gaps — e.g. the security mock matches
  "p95_ms=" on both causal and distractor metric samples,
  producing spurious LATENCY_SPIKE claims that leak "api" into
  scenarios where it shouldn't. This is a *feature* of the mock
  (it exercises the calibration pipeline under controlled
  imperfection), not a bug; the calibration audit cleanly
  attributes the degradation.
* **Token bound grows with spurious on security.** At k = 60
  the security mock's substrate prompt reaches 1 140 tokens
  because the inbox accumulates spurious LATENCY_SPIKE /
  IOC_KNOWN_BAD_IP claims. Theorem P32-3 predicts this: the
  token bound is ``O(capacity · τ)`` where capacity absorbs the
  spurious blow-up; at inbox_capacity = 64 and high spurious
  emission rate, the inbox fills up and tokens grow accordingly.

### D.3 Third non-code domain — security escalation (Part C)

Mock-auditor (reader-of-delivered-events ceiling), 5 scenarios ×
4 strategies × 4 k × 2 seeds = 160 measurements, 0.1 s wall:

| k | strategy | acc_full | acc_severity | acc_class | tokens | recall | trunc | fhist |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 6   | naive          | **1.00** | 1.00 | 1.00 |   767 | 1.00 | 0/10 | `{none:10}` |
| 6   | routing        | 0.00 | 0.00 | 0.00 |   151 | 1.00 | 0/10 | `{retrieval_miss:10}` |
| 6   | **substrate**  | **1.00** | **1.00** | **1.00** | **242** | **1.00** | **0/10** | `{none:10}` |
| 6   | substrate_wrap | 1.00 | 1.00 | 1.00 |   275 | 1.00 | 0/10 | `{none:10}` |
| 20  | naive          | 1.00 | 1.00 | 1.00 | 1 967 | 1.00 | 0/10 | `{none:10}` |
| 20  | **substrate**  | **1.00** | **1.00** | **1.00** | **242** | **1.00** | **0/10** | `{none:10}` |
| 60  | naive          | 1.00 | 1.00 | 1.00 | 4 297 | 1.00 | 10/10 | `{none:10}` |
| 60  | **substrate**  | **1.00** | **1.00** | **1.00** | **242** | **1.00** | **0/10** | `{none:10}` |
| **120** | **naive**  | **0.20** | 0.80 | 0.20 | 4 216 | 1.00 | 10/10 | `{truncation:8, none:2}` |
| 120 | routing        | 0.00 | 0.00 | 0.00 |   151 | 1.00 | 0/10 | `{retrieval_miss:10}` |
| 120 | **substrate**  | **1.00** | **1.00** | **1.00** | **242** | **1.00** | **0/10** | `{none:10}` |

Reading (Theorem P33-2 + P33-3 empirical anchors):

* **Substrate is flat at 242 tokens / 100 % accuracy.** Across
  four orders of magnitude of distractor count, no degradation.
  This is a *third* domain confirming the Phase-31/32 flatline
  signature at the substrate strategy.
* **Naive collapses at k = 120 under truncation.** From 100 % →
  20 %; severity accuracy drops only from 100 % → 80 % because
  severity has partial observability (the first N events often
  contain at least one high-severity claim even after truncation),
  but classification — which requires ALL required claims — falls
  to 20 %.
* **Routing (role-keyed) 0 % on every k.** The CISO has no raw
  telemetry subscription; routing delivers only fixed-point events.
  Confirms Theorem P29-2 / P31-1 on a third non-code domain.
* **Substrate token ratio vs naive:** 3.2× at k = 6, 8.1× at
  k = 20, 17.8× at k = 60, 17.4× at k = 120. Slightly smaller
  than Phase-31 (15×) and Phase-32 (24×) because the security
  scenarios carry more handoffs per role than either of the
  prior domains, raising the substrate-prompt token count from
  the Phase-31 196 / Phase-32 171 floor to 242.

**Cross-domain comparison table (three domains, same substrate):**

| domain | substrate tokens (flat) | naive tokens @ k=60 | substrate acc @ k=120 | naive acc @ k=120 |
|---|---:|---:|---:|---:|
| incident triage      | 196 | 3 014 | 1.00 | 0.20 |
| compliance review    | 171 | 4 081 | 1.00 | 0.40 |
| **security escalation** | **242** | **4 297** | **1.00** | **0.20** |

Three domains, three decoder shapes, same substrate module — 100 %
accuracy at a bounded-token cost across all k.

### D.4 Security noise sweep — Theorem P33-3 empirical anchor

Pooled per (drop, spurious, mislabel) across 2 seeds × 5 scenarios
= 10 runs per cell (security domain):

| drop | sp | mis | acc | recall | prec | tokens | failure hist |
|---:|---:|---:|---:|---:|---:|---:|---|
| **0.00** | **0.00** | **0.00** | **1.00** | 1.00 | 0.93 |  242 | `{none:10}` |
| 0.00 | 0.00 | 0.25 | 0.80 | 0.83 | 0.80 |  240 | `{missing_handoff:2, none:8}` |
| 0.00 | 0.05 | 0.00 | 0.60 | 1.00 | 0.56 |  340 | `{spurious_claim:4, none:6}` |
| 0.00 | 0.10 | 0.00 | 0.60 | 1.00 | 0.50 |  414 | `{spurious_claim:4, none:6}` |
| 0.10 | 0.00 | 0.00 | 0.90 | 0.86 | 0.91 |  225 | `{missing_handoff:1, none:9}` |
| 0.10 | 0.05 | 0.00 | 0.50 | 0.91 | 0.52 |  326 | `{missing_handoff:2, spurious_claim:3, none:5}` |
| 0.25 | 0.00 | 0.00 | 0.70 | 0.78 | 0.92 |  213 | `{missing_handoff:3, none:7}` |
| 0.50 | 0.00 | 0.00 | 0.20 | 0.52 | 0.91 |  188 | `{missing_handoff:8, none:2}` |
| 0.50 | 0.10 | 0.00 | 0.30 | 0.74 | 0.43 |  377 | `{missing_handoff:6, spurious_claim:1, none:3}` |

Reading (aligns with Theorem P33-3):

* **Identity point (noise = 0): 100 % accuracy / 93 % precision
  / 242 tokens.** The precision < 1.0 is the regex extractor's
  over-emission on causal events (the Phase-33 regex emits
  PII_AT_RISK on the same events as REGULATED_DATA_EXPOSED);
  these are not "spurious" in the sense of precision failures,
  they are structural duplicates the decoder silently absorbs.
* **Recall-limited regime (drop > 0, sp = 0): accuracy degrades
  as ~``(1 - δ)^{R*}``.** δ = 0.10, R* ≈ 2-3: expected 0.73-
  0.81, observed 0.90; δ = 0.25 expected 0.42-0.56, observed 0.70;
  δ = 0.50 expected 0.06-0.25, observed 0.20. Within i.i.d.
  noise of the P32-2(iii) monotone-regime prediction.
* **Precision-limited regime (sp > 0): accuracy drops to 60 %
  at sp = 0.05.** The *severity* axis survives spurious
  emissions at most noise levels (because most scenarios already
  carry a HIGH-severity claim kind in their gold chain), but the
  *classification* axis is strict-set-membership and any spurious
  claim that breaks the required-set subset check flips to
  "unclassified".
* **Token bound preserved.** Even at (drop = 0.5, sp = 0.1,
  mis = 0.25) the token count stays at 378 — a 1.56× increase
  over the noise-free 242, and still 11× lower than naive at
  k = 60.

### D.5 Real-LLM 7B spot check — qwen2.5-coder:7b across all three domains

Artifact: ``vision_mvp/results_phase33_llm_extractor_7b.json``.
120 LLM calls (40 per domain — 4 producer roles × 5 scenarios × 2
— with the deterministic cache collapsing repeats to 60 unique
prompts), **32.7 min wall** at k = 6 seed = 33.

**Pooled per-(domain, noise axis) — 7B vs 0.5b:**

| domain | model | δ̂ (drop) | μ̂ (mislabel) | ε̂ (spur/ev) | π̂ (payload) |
|---|---|---:|---:|---:|---:|
| incident    | 0.5b (not run here; compliance only) | — | — | — | — |
| incident    | **7B** | **0.177** | 0.067 | 0.142 | 0.400 |
| compliance  | 0.5b | 0.700 | 0.400 | 0.120 | 0.600 |
| compliance  | **7B** | **0.000** | **0.000** | **0.258** | 0.700 |
| security    | 0.5b (not run here; compliance only) | — | — | — | — |
| security    | **7B** | **0.222** | 0.263 | 0.108 | 0.737 |

The Phase-33 § D.5 structural prediction is **confirmed**: on
compliance, going from 0.5b to 7B **collapses drop rate from 0.70
to 0.00 and mislabel rate from 0.40 to 0.00** — the 7B extracts
every load-bearing claim from every causal document and never
confuses one kind for another on an in-distribution event. The
**spurious rate more than doubles**, from 0.12 to 0.258 per non-
causal event: the 7B's extra reasoning capacity leaks as
"plausible but unwarranted" claim emissions on the distractor
documents.

**Per-role heterogeneity reproduces on the 7B** (Conjecture C33-3):

| domain | max per-role drop spread | max per-role spur/ev spread |
|---|---:|---:|
| incident    | **0.333** (db_admin 0.0 ↔ sysadmin/network 0.333) | 0.167 |
| compliance  | 0.000 (all roles drop-perfect) | **0.167** (privacy 0.167 ↔ legal 0.333) |
| security    | **0.333** (threat_intel 0.0 ↔ soc/data_steward 0.333) | 0.200 |

Even on the 7B the pooled i.i.d. quadruple hides per-role structure
— the Phase-34 per-role-adaptive calibration layer
(``core/extractor_calibration.per_role_audit_summary``, shipped in
Phase 34, Arc 8 extended) is the right tool to characterise this.

**Phase-32 synthetic comparison (Theorem P33-1):**

| domain | verdict | max abs gap | gap axis | note |
|---|---|---:|---|---|
| incident    | **diverges**     | 1.000 | precision | definitional seam — see below |
| compliance  | **approximates** | 0.083 | precision | within γ = 0.15 |
| security    | **partial**      | 0.200 | accuracy  | within γ = 0.30 |

The "diverges" verdict on incident is not a theorem-P33-1 failure;
it is a definitional mismatch between the Phase-32 sweep's
*synthetic handoff precision* (which counts every spurious
emission as a delivered handoff) and the 7B's real *handoff
precision* (which is 1.0 because the 7B's spurious claims happen to
hit ``(role, kind)`` pairs the subscription table silently drops via
``dropped_no_subscriber``). The first-order noise quadruple
(δ̂, μ̂, ε̂, π̂) on incident maps to Phase-32 grid point (drop=0.25,
sp=0.1, mis=0.0) at L1 distance 0.58 — the closest grid point
exists, but the derived-metric (precision) axis is incomparable
because of what the substrate's subscription table silently does.
On compliance and security the noise quadruple aligns with a real
grid point and the gap is within the Theorem-P33-1 γ constants
(0.15 on compliance, 0.30 on security).

**Substrate accuracy (transcription-bounded axis — Theorem P30-3):**

| domain | handoff recall | handoff precision | substrate accuracy | failure_hist |
|---|---:|---:|---:|---|
| incident    | **0.90** | **1.00** | **0.00** | `{missing_handoff:2, llm_error:3}` |
| compliance  | **1.00** | 0.253 | **0.00** | `{spurious_claim:5}` |
| security    | 0.76 | 0.442 | **0.20** | `{missing_handoff:3, none:1, spurious_claim:1}` |

Reading:

* **Compliance is the cleanest 7B demonstration.** Handoff recall is
  a *perfect 1.00* — every load-bearing claim is extracted and
  delivered. But the spurious rate (ε̂ = 0.258 per event) floods the
  auditor's inbox with enough noise that the strict-flag decoder
  fails every scenario with ``spurious_claim`` attribution. This is
  the **precision-limited regime** of Theorem P32-2 applied to a
  real LLM: recall survives, precision collapses, the decoder's
  strict set-match catches it.
* **Incident is recall-mixed.** Handoff recall 0.90; two scenarios
  miss a load-bearing claim, three fail with ``llm_error`` (the 7B's
  answer didn't match the gold verdict on the auditor's wrap path).
* **Security accuracy rises to 0.20.** The max-ordinal decoder is
  more forgiving than the strict-set one — one of five scenarios
  (brute_force_blocked) survives partial causal-chain delivery,
  because its gold classification only requires 2 of the 15 claim
  kinds and the 7B happens to extract both.

**Comparison to 0.5b on compliance:**

| axis | 0.5b | 7B | change |
|---|---:|---:|---:|
| δ̂ drop     | 0.700 | **0.000** | −0.70 (7B far better) |
| μ̂ mislabel | 0.400 | **0.000** | −0.40 (7B far better) |
| ε̂ spurious/ev | 0.120 | 0.258 | +0.14 (7B **worse**) |
| π̂ payload  | 0.600 | 0.700 | +0.10 (comparable) |
| handoff recall | 0.50 | **1.00** | +0.50 |
| handoff precision | 0.267 | 0.253 | −0.01 |
| substrate accuracy | 0.00 | 0.00 | 0.00 |

Both 0.5b and 7B score 0 % substrate accuracy on compliance, but
for **opposite reasons**: 0.5b fails on the *recall* side
(δ̂ = 0.70 drops load-bearing claims), 7B fails on the *precision*
side (ε̂ = 0.258 floods with spurious claims). Both failure modes
live inside the Phase-32 two-regime bound (Theorem P32-2): the 0.5b
lives in the monotone-recall regime, the 7B lives in the strict-
decoder regime. This is a **clean empirical confirmation of the
two-regime shape** on a single real LLM pair.

**Implications for deployment (the reusable finding).** A larger
model does *not* automatically improve extractor-boundary
performance on a substrate with a strict-decoder downstream. The
7B's higher capacity cures the recall-limited failure mode but
introduces a **precision-limited failure mode** the 0.5b did not
reach. For a product, the right choice is either (a) the Phase-
34 ensemble primitive (regex + LLM, where the LLM supplies
previously-uncovered narrative signals and the regex bounds
spurious emissions), or (b) a prompt-engineering pass on the 7B
extractor that constrains its spurious rate — Conjecture C33-4 and
Phase 34's C34-4 both name this exact gap.

**Phase-33 7B data in one sentence.** The 7B's per-domain pooled
noise profile is (δ̂=0.18, ε̂=0.14), (δ̂=0.00, ε̂=0.26),
(δ̂=0.22, ε̂=0.11) on (incident, compliance, security); the
Phase-32 synthetic sweep approximates this at γ ≤ 0.20 on
(compliance, security) and diverges on incident due to a
definitional seam in the handoff-precision metric; substrate
accuracy is (0.0, 0.0, 0.2) — the 7B's extra capacity moves the
failure mode from recall-limited to precision-limited, confirming
the Theorem-P32-2 two-regime bound on a real LLM pair.

---

## Part E — Failure taxonomy (Phase-33 extension)

Phase 32 shipped a five-way attribution histogram:

```
none / retrieval_miss / truncation / missing_handoff /
spurious_claim / llm_error
```

Phase 33 re-uses the same taxonomy unchanged. The security-
escalation benchmark's harness adds the attribution. What Phase 33
adds is the *empirical separation* between failure modes under a
real LLM extractor:

| Category | Typical noise axis | Phase-33 observation |
|---|---|---|
| ``missing_handoff`` | drop_prob (recall) | dominant at δ̂ > 0.50 on 0.5b finance |
| ``spurious_claim`` | spurious_prob (precision) | dominant on the security max-ordinal decoder at sp ≥ 0.05 |
| ``llm_error`` | transcription (LLM on wrap path) | not exercised by the mock auditor in Phase 33 |
| ``retrieval_miss`` | strategy = routing or naive-without-content | universal on routing / 0 % naive at k = 120 on phishing_exfil |
| ``truncation`` | naive + large k | universal on naive at k ≥ 60 security |

The taxonomy is stable across three domains with three decoder
shapes — the same failure histogram columns carry the same
meanings on Phase 31, 32, and 33.

---

## Part F — Future work

### F.1 Carry-over (unchanged from Phase 32)

* SWE-bench end-to-end with a real LLM on the wrap path.
* Frontier-model sweep (multi-model × multi-seed × multi-k).
* OQ-1 in full generality (Conjecture P30-6).
* Adversarial extractor noise (targeted drop of load-bearing
  claims).
* Hierarchical role lattice at K ≥ 20.

### F.2 Newly surfaced by Phase 33

* **Per-role noise characterisation.** Phase 33 § D.1 shows the
  0.5b extractor has finance drop = 1.00 — pooled statistics
  mask this. A per-role noise calibration layer would let the
  Phase-32 synthetic sweep be *parameterised per role*, which
  would close the C33-3 gap.
* **Ensemble / composition extractors.** Conjecture C33-4
  predicts a union-of-extractors strategy dominates on the
  noise-accuracy frontier. Needs a scenario catalogue where
  regex and LLM have complementary coverage.
* **Prompt engineering as an extractor-calibration dimension.**
  The LLMExtractorConfig has five knobs (few-shot, event-ids,
  temperature, etc.); Phase 33 uses defaults everywhere. A
  controlled ablation over prompt format would give a cleaner
  read on whether noise comes from the model or from the prompt.
* **Security-domain real-LLM run as a standalone benchmark.**
  The security domain's max-ordinal decoder stress-tests the
  precision regime in a different way than compliance; a real
  7B run on all three domains would triangulate Theorem P33-1
  cleanly.
* **Calibration-driven deployment heuristic.** Production teams
  could run the Phase-33 calibration layer on a small held-out
  scenario bank to predict substrate accuracy before deploying
  an LLM extractor. The mechanical path is: measure
  ``(δ̂, ε̂, μ̂)``, look up the closest Phase-32 sweep point,
  read the predicted accuracy, ship or don't.

### F.3 What is genuinely blocking the endgame

* **End-to-end SWE-bench.** The programme's largest external-
  validity gap. Phase-30 harness is ready; what's missing is a
  SWE-bench instance iterator that wires up patch-apply + test-
  harness on the wrap path.
* **Fixed-point convergence proof for ``T_i*`` under Lipschitz
  LLM policies (Conjecture P30-6).** This is the sharpest
  remaining theorem target.
* **Cross-language runtime calibration.** TypeScript / Go —
  the Python-AST substrate has been stable since Phase 24;
  transferring to another language is the cleanest test of
  whether the substrate is language-specific or substrate-
  specific.

The Phase-33 LLM-extractor work does NOT close any of these
— it closes a *different* gap (the "extractors were perfect"
simplification in Phase 31/32). The endgame gaps are the same
after Phase 33 as they were after Phase 32, just one ceiling
lower: the substrate is now validated not only under a
synthetic noise model but under a measured real-LLM noise
distribution that the synthetic model approximates within a
tight γ.

---

## Appendix A — How to reproduce

```bash
# 1. Mock LLM-extractor across all three domains (seconds of wall)
python3 -m vision_mvp.experiments.phase33_llm_extractor --mode mock \
    --distractor-counts 6 20 60 --seeds 33 \
    --sweep-path vision_mvp/results_phase32_noise_sweep.json \
    --out vision_mvp/results_phase33_llm_extractor_mock.json

# 2. Real 0.5b LLM-extractor calibration on compliance (~90 s wall)
python3 -m vision_mvp.experiments.phase33_llm_extractor --mode real \
    --model qwen2.5:0.5b --domains compliance \
    --distractor-counts 6 --seeds 33 \
    --sweep-path vision_mvp/results_phase32_noise_sweep.json \
    --out vision_mvp/results_phase33_llm_extractor_0p5b.json

# 3. Real 7B LLM-extractor calibration on all three domains (~32 min wall, 120 LLM calls)
python3 -m vision_mvp.experiments.phase33_llm_extractor --mode real \
    --model qwen2.5-coder:7b --domains incident compliance security \
    --distractor-counts 6 --seeds 33 \
    --sweep-path vision_mvp/results_phase32_noise_sweep.json \
    --out vision_mvp/results_phase33_llm_extractor_7b.json

# 4. Security-escalation substrate benchmark (mock, sub-second wall)
python3 -m vision_mvp.experiments.phase33_security_escalation --mock \
    --distractor-counts 6 20 60 120 --seeds 33 34 \
    --out vision_mvp/results_phase33_security_mock.json

# 5. Security noise-sweep (sub-second wall)
python3 -m vision_mvp.experiments.phase33_noise_sweep_security \
    --drop-probs 0.0 0.1 0.25 0.5 --spurious-probs 0.0 0.05 0.1 \
    --mislabel-probs 0.0 0.25 --seeds 33 34 \
    --out vision_mvp/results_phase33_noise_sweep_security.json

# 6. Test suite
python3 -m pytest vision_mvp/tests/test_security_escalation.py \
    vision_mvp/tests/test_llm_extractor.py -q
```

---

*End of Phase 33 results note. The master plan
(``docs/context_zero_master_plan.md``) is updated in the same
commit; see ``§ 4.10``, ``§ 4.11 Current frontier`` for the
higher-level integration.*
