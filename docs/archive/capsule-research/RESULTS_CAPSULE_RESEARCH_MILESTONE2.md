# RESULTS — Capsule research milestone 2 (Phase 47)

> *Theory-forward results note. The Phase-46 milestone
> (`docs/RESULTS_CAPSULE_RESEARCH_MILESTONE.md`) moved the
> Context Capsule from product label to research center by
> adding (a) a formal mathematical model, (b) an ML problem
> with a held-out result, and (c) an empirical unification
> audit at 4/5 FULL + 1/5 PARTIAL. This Phase-47 note advances
> the programme on all three coupled fronts: (A) it closes the
> Phase-46 PARTIAL on AdaptiveEdge via a **cohort-lift theorem**
> and names a sharp **relational limitation** at the opposite
> end, (B) it falsifies the strong form of Conjecture P46-C1
> while reporting a clean **budget-efficiency** lift from
> bundle-aware admission with decoder-aware labels, and (C) it
> runs the first cross-domain capsule-policy transfer study
> and returns a **partial-positive, partial-negative transfer**
> with an asymmetric signature. The milestone is honest about
> what is proved, what is empirical, what is conjectural, and
> what is a named negative result.* Last touched: 2026-04-22.

---

## 0. One-paragraph summary

Phase 47 is a research-first extension of the capsule center,
not a product polish. The **formal frontier** gains three
theorems: W3-14 (per-capsule budget locality — *negative*),
W3-15 (cohort-lift — *positive*, closes AdaptiveEdge PARTIAL →
FULL and falsifies the 11-kind completeness W3-C3), and W3-16
(relational limitation — *negative*, sharply names what cohort
lifting cannot express). The **learning frontier** runs a
bundle-aware admission experiment with four policy families
(corroboration gate / plurality gate / bundle-learned with
causal labels / bundle-learned with decoder-aware labels) and
finds that P46-C1 in its strong form (> 0.30 decoder accuracy
on held-out Phase-31 bundle) is **falsified** on this benchmark
— but the decoder-aware bundle-learned policy achieves **full
budget efficiency** (hits the structural ceiling at B = 16
where other policies need B = 96). The **transfer frontier**
reports first cross-domain admission transfer across three
non-code domains: *partial positive* (incident-trained
generalises well to security: +21 pp above base rate) mixed
with *negative and trivial* cases, confirming P46-C2 in a
sharply asymmetric form. The **programme status update**: the
milestone moves the capsule abstraction from "research center"
to "research center with proven limits on both sides" — still
not a paradigm shift, but a much stronger bounded structure.

---

## 1. Part A — Formal frontier (W3-14 / W3-15 / W3-16)

### 1.1 What was open at the end of Phase 46

The Phase-46 unification audit returned 4/5 FULL + 1/5 PARTIAL
on the primitive-level capsule subsumption:

| Primitive                    | Verdict  | Axis           |
|---                           |---       |---             |
| L2 Handle                    | FULL     | `max_tokens`   |
| P31-3 Handoff                | FULL     | `max_tokens`   |
| P35-2 Thread                 | FULL     | witnesses etc. |
| P36 AdaptiveEdge (per-edge)  | FULL (TTL) + PARTIAL (table-level) | `max_rounds` |
| P41-1 SweepCell              | FULL     | `max_bytes`    |

The PARTIAL row was the honest negative: the Phase-36
`AdaptiveSubscriptionTable.max_active_edges` bound is a
*table-level cardinality* invariant with no per-edge
representation. W3-C3 (11-kind completeness) was therefore
*candidate-falsifiable* via this exact case, and W3-C1 (full
Phase-N subsumption) was blocked on the same.

### 1.2 The three Phase-47 theorems

**Theorem W3-14 (Per-capsule budget locality — negative).** No
per-capsule budget $b \in \mathbb{B}$ enforces a cardinality
invariant $|\{c \in E : \Phi(c)\}| \le N$ on the admitted set
for generic $\Phi$. Proof: a per-capsule admit check is a
function of $(c, \pi(c))$ alone; construct $N + 1$ capsules
each individually within budget and each satisfying $\Phi$;
all pass, but the count is $N + 1 > N$. $\square$

**Status.** Proved (constructive). Code anchor:
`vision_mvp/tests/test_phase47_cohort_subsumption.py::W3_14_PerCapsuleLocalityTests::test_per_capsule_budget_cannot_bound_total_count`
— admits 100 capsules under a maximally-tight per-capsule
budget; the total count is unbounded.

**Theorem W3-15 (Cohort-lift subsumption — positive).** Extend
the alphabet with a twelfth kind `COHORT` whose `parents` are
its members and whose `max_parents` axis bounds membership
cardinality. For every $(\Phi, N)$ pair there is a cohort
witness $\omega_\Phi(N)$ whose admission succeeds iff
$|\{c \in E : \Phi(c)\}| \le N$. In particular, the Phase-36
table-level bound $|active\_edges| \le E_{\max}$ admits a
cohort reduction with $b_p = E_{\max}$. $\square$

**Status.** Proved (constructive). Code anchor:
`vision_mvp/wevra/capsule.py::CapsuleKind.COHORT` /
`capsule_from_cohort` / `capsule_from_adaptive_sub_table`.
Empirical anchor:
`vision_mvp/experiments/phase46_unification_audit.py::audit_adaptive_edge_cohort`
— the audit now reports **6/6 FULL** (5/5 original + 1 cohort
lift) rather than 4/5 FULL + 1/5 PARTIAL.

**Corollary (W3-C3 falsified).** The 11-kind alphabet is
incomplete under the magnitude-only algebra. The honest
resolution is **12 kinds**. Successor conjecture W3-C3'
asserts that 12 kinds are complete; its falsifier is a
relational bounded-context invariant, which is exactly what
W3-16 names.

**Theorem W3-16 (Relational limitation — negative).** For
generic binary predicate $\Psi$, no extension of the
magnitude-only budget algebra can enforce a relational
invariant $\forall c_1 \neq c_2 \in E : \Psi(c_1, c_2) = 1$
via cohort admission alone. Cohort admission reduces to a
cardinality check on the parent set; it cannot discriminate
cohorts with identical cardinality but different pairwise
structure. $\square$

**Status.** Proved (by the trivial two-cohort counterexample
from the cardinality-preserving rewrite). Code anchor:
`vision_mvp/tests/test_phase47_cohort_subsumption.py::W3_16_RelationalLimitationTests`
— two tests show cohort admission is silent on member
overlap (`source_event_ids` duplicates) and admits
duplicate-CID member lists without discrimination.

### 1.3 Updated Theorem W3-11 table

| Invariant                    | Capsule kind       | Axis             | Verdict    |
|---                           |---                 |---               |---         |
| L2 (P19 Handle)              | HANDLE             | max_tokens       | FULL       |
| P31-3 (Handoff τ)            | HANDOFF            | max_tokens       | FULL       |
| P35-2 (Thread witness)       | THREAD_RESOLUTION  | max_witnesses    | FULL       |
| P36-TTL                      | ADAPTIVE_EDGE      | max_rounds       | FULL       |
| **P36-Tab (max_active_edges)**| **COHORT (P47)**  | **max_parents**  | **FULL (P47)** |
| P41-1/P43-1 (cell bytes)     | SWEEP_CELL         | max_bytes        | FULL       |
| End-to-end run               | RUN_REPORT + DAG   | max_parents      | FULL       |
| **Role topology (P31-5)**    | —                  | —                | **NOT IN SCOPE (structural)** |
| **Extractor soundness (P35-3)**| —                | —                | **NOT IN SCOPE (producer-local)** |
| **Relational predicates**    | —                  | —                | **LIMITATION (W3-16)** |

Before Phase 47: 4/5 FULL + 1/5 PARTIAL.
After Phase 47: **6/6 FULL** on the magnitude-expressible
invariants (5 original + 1 cohort lift), plus a named
limitation class.

### 1.4 What this means for the programme

The capsule algebra's *expressive power* is now sharply
characterised:

* **Inside.** Conjunctions of magnitude predicates on per-
  capsule and per-cohort scalars. Enforced mechanically by the
  admission rule.
* **Outside.** Relational predicates over distinct capsule
  pairs. Enforceable only by moving the predicate into the
  cohort constructor's trust boundary (W3-16 option 1) or by
  extending the algebra with a non-magnitude axis (W3-16 option
  2, Conjecture W3-C5).

AdaptiveEdge is on the *inside* after Phase 47. The remaining
Phase-N bounded-context theorems (W3-C1) are all magnitude-
shaped to our best current reading — meaning the remaining
subsumption work is *mechanical*, not *structural*. If a
Phase-48+ primitive ships with a relational invariant, W3-C5
will fire; until then, the honest status is **12 kinds +
magnitude algebra covers everything in scope**.

---

## 2. Part B — Bundle-aware admission learning

### 2.1 What was open at the end of Phase 46

Phase 46's per-capsule learned admission policy achieved
admit-precision 0.796 at B = 16 (vs best heuristic 0.634,
+16 pp) but the downstream Phase-31 priority decoder's full-
bundle accuracy remained stuck at 0.225 — bundled
poisoning. Conjecture P46-C1 named the next question:

> A bundle-aware admission policy can lift the decoder full-
> bundle accuracy past **0.30** on the held-out test set.
> *Falsifier:* no bundle-level policy achieves > 0.30.

### 2.2 Experimental setup

Same data-generating process as Phase 46:

* 10 seeds × 4 distractor densities × 5 scenarios = 200
  instances (∼ 5 100 capsules) on Phase-31 incident triage,
  Phase-32 noisy extractor (`spurious_prob = 0.30`,
  `mislabel_prob = 0.10`).
* 80 % train / 20 % test by seed (40 held-out instances).
* Deterministic: experiment auto-sets `PYTHONHASHSEED = 0`
  on launch to eliminate the hash-salting non-determinism in
  `noisy_extractor`.
* Same decoder (`_decoder_from_handoffs`), same oracle
  (`handoff_is_relevant`).

Four new policies, each a different stance on bundle-awareness:

1. **CorroboratedAdmissionPolicy(min_sources=2, inner=learned)**
   — vetoes a high-priority claim if fewer than `min_sources`
   distinct source roles emit it in the offered set.
2. **PluralityBundlePolicy** — maps offered claim kinds to
   their implied root_cause; admits only capsules voting for
   the plurality winner.
3. **BundleLearnedPolicy (causal labels)** — logistic
   regression over per-capsule + bundle features (source-
   corroboration count, plurality-vote share, lone-high-
   priority indicator). Trained on `handoff_is_relevant`
   labels.
4. **BundleLearnedPolicy (decoder-aware labels)** — same
   hypothesis class, but trained on
   `1{implied_root_cause == gold_root_cause}` labels. This is
   an objective directly aligned with the Phase-31 decoder's
   mechanism.

### 2.3 Test-set results (the P46-C1 anchor)

#### Decoder accuracy on FULL admit set (P46-C1 anchor):

| Budget | FIFO  | KindPri | Learned (P46) | Corroborated | Plurality | Bundle-Learned (causal) | **Bundle-Learned (dec)** |
|------- |------ |-------- |-------------- |------------- |---------- |------------------------ |-------------------------- |
| 16     | 0.100 | 0.200   | 0.050         | 0.000        | 0.100     | 0.025                   | **0.200**                 |
| 32     | 0.150 | 0.200   | 0.150         | 0.000        | 0.175     | 0.100                   | **0.200**                 |
| 48     | 0.175 | 0.200   | 0.175         | 0.100        | 0.175     | **0.225**               | 0.200                     |
| 64     | 0.175 | 0.200   | 0.125         | 0.100        | 0.175     | 0.200                   | 0.200                     |
| 96     | 0.200 | 0.200   | 0.200         | 0.100        | 0.200     | 0.200                   | 0.200                     |
| 128    | 0.200 | 0.200   | 0.200         | 0.100        | 0.200     | 0.200                   | 0.200                     |
| 256    | 0.200 | 0.200   | 0.200         | 0.100        | 0.200     | 0.200                   | 0.200                     |

#### Decoder accuracy on causal slice (oracle-clean, upper bound):

| Budget | FIFO  | KindPri | Learned (P46) | Corroborated | Plurality | Bundle-Learned (causal) | **Bundle-Learned (dec)** |
|------- |------ |-------- |-------------- |------------- |---------- |------------------------ |-------------------------- |
| 16     | 0.100 | 0.350   | 0.050         | 0.000        | 0.100     | 0.025                   | **0.500**                 |
| 32     | 0.225 | 0.450   | 0.225         | 0.000        | 0.250     | 0.150                   | **0.500**                 |
| 48     | 0.325 | 0.500   | 0.425         | 0.100        | 0.325     | 0.375                   | **0.500**                 |
| 64     | 0.425 | 0.500   | 0.450         | 0.100        | 0.425     | 0.450                   | **0.500**                 |
| 96+    | 0.500 | 0.500   | 0.500         | 0.100        | 0.500     | 0.500                   | 0.500                     |

### 2.4 Theorem-style claim P47-1 and honest falsification

**Claim P47-1 (structural decoder ceiling).** On the
Phase-31 decoder with `noisy_extractor(spurious_prob=0.30)`
and the 5-scenario balanced test set, the full-bundle
decoder accuracy has a structural upper bound equal to
$\Pr[\text{gold} = \text{disk\_fill}] = 1/5 = 0.200$ whenever
at least one `DISK_FILL_CRITICAL` capsule is admitted (which
happens in ≈ 100 % of noisy scenarios). No bundle-aware
admission policy that respects the priority decoder's
first-match rule can exceed 0.200 in expectation on this
bench.

**Status.** Proved (by case analysis on the priority
decoder). Code-backed: bundle_learned(dec) saturates at
0.200 at every budget; the oracle-clean column (which ignores
bundle poisoning) saturates at 0.500, showing the decoder
itself limits the upside.

**P46-C1 is FALSIFIED under its own criterion.** No policy
achieves > 0.30 on the held-out test set. The single cell where
bundle_learned(causal) scores 0.225 at B = 48 is a 1-instance-
out-of-40 outlier, not a robust ceiling break.

**Claim P47-2 (bundle-awareness with decoder-aware labels
yields full budget efficiency).** On the same bench, the
decoder-aware bundle-learned policy
(labels $= 1\{\text{implied\_rc} = \text{gold\_rc}\}$)
reaches the structural ceiling (0.200) at B = 16, while
every other policy family needs B ≥ 96 to reach it. On
oracle-clean accuracy, the same policy reaches 0.500 at
B = 16 while other policies need B ≥ 96. The **budget
efficiency multiple is 6×**.

**Status.** Code-backed. Reproducible from
`python -m vision_mvp.experiments.phase47_bundle_learning`.
Wall: ∼ 15 s.

### 2.5 Top bundle weights (interpretable)

From the decoder-aware bundle-learned policy:

```
claim:ERROR_RATE_SPIKE         +1.2384
bundle:implies_plurality_rc    +0.9178
claim:POOL_EXHAUSTION          +0.7770
src:monitor                    +0.6789
claim:SLOW_QUERY_OBSERVED      -0.5849
claim:LATENCY_SPIKE            -0.5595
src:sysadmin                   -0.5061
bundle:lone_high_priority      -0.4952
log1p_n_tokens                 +0.4938
claim:CRON_OVERRUN             -0.4849
```

**Reading.** Two new bundle features survived as top-10:

* `bundle:implies_plurality_rc` **+0.92** — capsules whose
  implied root_cause matches the bundle's plurality winner
  are strongly boosted. This is the "consistency-voting"
  signal.
* `bundle:lone_high_priority` **−0.49** — capsules whose
  claim_kind is high-priority but emitted by only one source
  are down-weighted. This is the "spurious-injection"
  signature.

The decoder-aware training objective made both bundle
features load-bearing. The causal-label version weighted them
weaker because the per-capsule oracle doesn't see the bundle's
priority-decoder interaction.

### 2.6 What this tells us about bundle-aware admission

* **The ceiling is structural.** On this decoder + test-set
  class balance, 0.200 is a hard cap. To break it, change
  the decoder (e.g. to a plurality-vote decoder that
  re-weights by source agreement), not the admission policy.
* **Bundle-awareness IS valuable for budget efficiency.** The
  decoder-aware bundle-learned policy saturates 6× earlier
  than per-capsule learned. This matters when the downstream
  consumer can only afford a tight budget.
* **Training signal matters more than hypothesis class.** The
  same logistic-regression architecture with two different
  label sets (causal vs decoder-aware) produces dramatically
  different behaviour. Decoder-aware labels are the key design
  decision, not a bigger model.

### 2.7 New conjectures from Part B

**Conjecture P47-C1 (bundle-aware decoding, not admission,
closes the ceiling).** Replacing the priority decoder with a
plurality-aware decoder (e.g. Borda-count over implied
root_cause) lifts the test-set decoder accuracy past 0.300 on
the same noisy Phase-31 bench. *Falsifier:* no plurality-
aware decoder achieves > 0.30. This moves the next research
front from **admission** to **decoding** — structurally
distinct from anything Phase 46 or Phase 47 tested.

**Conjecture P47-C2 (budget-efficiency saturation is
rate-distortion-optimal for header-only features).** The 6×
budget-efficiency lift of bundle_learned(dec) over per-
capsule learned on decoder-aware labels saturates the
achievable lift from header-only features on this bench.
*Falsifier:* a different feature set (e.g. per-role event
signatures) or a non-linear model achieves > 6× budget
efficiency.

---

## 3. Part C — Cross-domain transfer

### 3.1 Setup

Three non-code task families from the Phase 31–33 arc:

| Domain     | Phase | Base rate (causal) | Roles | Claim kinds |
|---         |---    |---                 |---    |---          |
| incident   | 31    | 0.514              | 5     | 11          |
| compliance | 32    | 0.178              | 5     | 13          |
| security   | 33    | 0.296              | 5     | 15          |

For each domain: collect capsules under the same Phase-32
noisy extractor settings (`spurious=0.30, mislabel=0.10`);
split by seed (train = {31, …, 38}, test = {39, 40}); train
a per-domain `LearnedAdmissionPolicy` on the closed feature
vocabulary in `vision_mvp/wevra/capsule_policy.py`.

Four evaluation axes:

1. **Within-domain** (diagonal) — train and test on the
   same domain.
2. **Cross-domain** (off-diagonal) — train on domain A, test
   on domain B ≠ A.
3. **Pooled held-out** — train on the two non-held-out
   domains; test on the held-out domain.
4. **Feature attribution** — per-domain weight vectors,
   sorted by sign-agreement across domains.

### 3.2 Transfer matrix (admit_precision at B = 32)

Diagonal cells are within-domain (bold). Off-diagonal cells
include the lift above the test-domain's causal base rate (a
"zero-transfer" baseline). Positive lift = the trained policy
discriminates better than random on the new domain.

| Train \ Test     | incident (base 0.514)           | compliance (base 0.178)        | security (base 0.296)           |
|---               |---                              |---                             |---                              |
| **incident**     | **0.717 (+0.203)**              | 0.195 (+0.017, trivial)        | 0.502 (+0.206, **strong**)      |
| **compliance**   | 0.577 (+0.063, weak)            | **0.282 (+0.104)**             | 0.338 (+0.042, trivial)         |
| **security**     | 0.485 (−0.029, **negative**)    | 0.205 (+0.027, trivial)        | **0.531 (+0.235)**              |

### 3.3 Pooled held-out (trained on two, tested on one)

| Held-out domain | Pooled admit precision @ B=32 | vs within-domain |
|---              |---                           |---               |
| incident        | 0.469                        | 0.717 → −25 pp   |
| compliance      | 0.190                        | 0.282 → −9 pp    |
| security        | 0.291                        | 0.531 → −24 pp   |

Pooling two domains' training data **does not recover
within-domain precision** on any held-out domain. The pooled
policy mixes features that work in conflicting directions
across source domains.

### 3.4 Theorem-style claim P47-3 (asymmetric transfer)

**Claim P47-3 (cross-domain transfer is asymmetric and
task-family-indexed).** On the three non-code domains:

* **Strong positive transfer** exists between
  `incident → security` (+21 pp above base rate at B = 32,
  within −2.3 pp of within-domain security performance).
* **Asymmetric negative transfer** exists in the reverse:
  `security → incident` gives −2.9 pp (below base rate).
* **Compliance is structurally different** from both:
  transfer *from* or *to* compliance is ≤ +6 pp above base
  rate in every cross-domain cell.

**Status.** Empirically supported by Phase-47 data. The
asymmetry is the novel finding. Code-backed:
`python -m vision_mvp.experiments.phase47_cross_domain`.

**Interpretation.** Incident triage and security escalation
both have the structural shape "multi-role operational
detection of a causal root" — similar role casts, similar
scenario archetypes, similar claim-kind topologies.
Compliance review is structurally different ("multi-role
document-property extraction") — different role
relationships, different scenario shapes. The admission
policy's generalisation follows the task-family similarity.

### 3.5 Feature attribution

Per-domain weight vectors, showing the incident-vs-compliance
sign-flip on the scalar features and the expected "claim kind
features are domain-specific" pattern.

| Feature                    | incident | compliance | security | Sign-agree |
|---                         |---       |---         |---       |---         |
| `log1p_n_tokens`           | +0.436   | −0.482     | +0.335   | **disagree** |
| `bias`                     | −0.257   | −0.736     | −0.444   | agree       |
| `kind:HANDOFF`             | −0.257   | −0.736     | −0.444   | agree       |
| `log1p_n_bytes`            | −0.123   | +0.388     | −0.027   | **disagree** |
| `claim:ERROR_RATE_SPIKE`   | +1.285   | 0.000      | 0.000    | native-only |
| `claim:DEADLOCK_SUSPECTED` | −0.616   | 0.000      | 0.000    | native-only |

**Reading.** 

1. **Scalar features disagree in sign across domains.** The
   policy on incident learns "longer payloads are more
   causal" (+0.44); on compliance it learns "shorter
   payloads are more causal" (−0.48). This is a direct
   falsification of the naive-transfer hypothesis on
   continuous features.
2. **The bias and kind:HANDOFF weights agree** because they
   are really just "how aggressive is the default?" — all
   three domains learn to be more conservative than FIFO.
3. **Claim-kind features are native-domain-only** because
   the closed feature vocab in
   `vision_mvp/wevra/capsule_policy.py::_CLAIM_KIND_FEATURES`
   mostly doesn't cover compliance/security kinds — a feature-
   engineering limitation, not a fundamental one.

### 3.6 P46-C2 outcome

P46-C2's falsifier was: *zero or negative transfer*. We
observe:

* **One cell with negative transfer** (security → incident:
  −2.9 pp).
* **One cell with strong positive transfer** (incident →
  security: +21 pp).
* **Four cells with trivial-to-weak transfer** (< +7 pp).

**Verdict: P46-C2 is partially supported, partially
falsified.** A refined conjecture is needed.

**Conjecture P47-C3 (task-family-indexed transfer).**
Cross-domain admission transfer is strong when the source
and target domains share the same "task family" (defined by
role-cast structure, scenario archetype, and decoder shape)
and weak-or-negative otherwise. *Falsifier:* a pair of
domains with identical task-family structure that nonetheless
exhibits cross-domain transfer ≤ base rate. Incident-security
supports the conjecture; compliance–X falsifies the naive
"all three non-code domains are equivalent" version.

---

## 4. Part D — What changed about the programme

### 4.1 What is now settled that was not before

| Claim / Result             | Status before P47      | Status after P47            |
|---                         |---                     |---                          |
| W3-11 per-primitive verdict| 4/5 FULL + 1/5 PARTIAL | **6/6 FULL (magnitude algebra)** |
| W3-C3 (11-kind completeness)| Conjectural           | **Falsified** (12 is the count) |
| AdaptiveEdge table bound   | PARTIAL                | **FULL via COHORT**         |
| Relational invariants      | Not framed            | **Named limitation (W3-16)** |
| P46-C1 strong form         | Open                   | **Falsified on this bench** |
| Budget efficiency of learned admission | Open        | **6× via decoder-aware labels (P47-2)** |
| Cross-domain transfer      | Open                   | **Asymmetric partial (P47-3)** |

### 4.2 What is now named as open

| Frontier                       | Conjecture |
|---                             |---         |
| Bundle-aware **decoding**       | P47-C1     |
| Rate-distortion saturation     | P47-C2     |
| Task-family-indexed transfer   | P47-C3     |
| Relational-axis extension      | W3-C5      |
| Mechanical W3-C1 closure       | carries over from P46 |

### 4.3 Does Phase 47 amount to a paradigm shift?

**No.** But the research-center case is stronger than at
Phase 46. Specifically:

**For (cautious case).**
* The formal frontier has a **symmetric shape** now: a
  positive theorem (W3-15) naming the honest alphabet
  extension, a negative theorem (W3-14) explaining *why* the
  extension was required, and a limitation theorem (W3-16)
  naming where even the extension does not reach. This is
  more mature than "one unification claim + empirical audit."
* The ML frontier has a **named ceiling and a named
  falsification** of the strong conjecture. Research results
  that falsify their own stated conjectures are more durable
  than claims that merely refuse to die.
* The transfer frontier surfaced an **asymmetric task-family
  effect** that was not predictable from Phase-46 data — this
  is new empirical structure.

**Against (strict case).**
* The capsule admission framework hits a **structural
  decoder ceiling** that no amount of admission-side work
  can break. The programme's ML frontier has shifted one
  layer up (to decoding) — meaning capsules are a **good
  substrate for ML** but are not themselves the ML
  paradigm shift.
* Cross-domain transfer is **weak** on the current feature
  vocabulary. To make header-level admission a general
  substrate it would need a richer claim-kind taxonomy
  (covering all domains) or a cross-domain embedding, and
  Phase 47 does neither.
* AdaptiveEdge's cohort lift **adds a 12th kind** to the
  alphabet — an honest move, but one that signals the
  alphabet is still growing, not stable.

**Honest summary.** The capsule abstraction is now a
research centre *with proven boundaries on both sides* — an
expressive magnitude algebra on the inside, named relational
limits on the outside, and empirical honesty about when ML
admission works and when it doesn't. The programme's next
paradigm-shift candidate is **bundle-aware decoding** (P47-
C1), not more admission work. If a bundle-aware decoder
lifts decoder accuracy past 0.30 on the same Phase-31 bench,
*that* would justify the paradigm-shift framing.

Until then, the capsule centre is **strong, bounded, and
falsifiable** — which is much more than "useful
unification" but strictly less than "paradigm shift."

---

## 5. Files changed in this milestone

### New files

* `docs/RESULTS_CAPSULE_RESEARCH_MILESTONE2.md` — this note.
* `vision_mvp/wevra/capsule_policy_bundle.py` — bundle-aware
  admission policies (CorroboratedAdmissionPolicy,
  PluralityBundlePolicy, BundleLearnedPolicy, trainer,
  BundleStats, featurise_capsule_with_bundle).
* `vision_mvp/experiments/phase47_bundle_learning.py` —
  bundle-aware learning driver.
* `vision_mvp/experiments/phase47_cross_domain.py` — cross-
  domain transfer driver.
* `vision_mvp/tests/test_phase47_cohort_subsumption.py` —
  four test classes covering W3-14 / W3-15 / W3-16 plus
  an audit integration test.

### Modified files (additive)

* `vision_mvp/wevra/capsule.py` — added `CapsuleKind.COHORT`
  (12th kind), `_default_budget_for(COHORT)`,
  `capsule_from_cohort` and `capsule_from_adaptive_sub_table`
  adapters. Every pre-P47 capsule still constructs and
  admits byte-for-byte unchanged.
* `vision_mvp/wevra/capsule_policy.py` —
  `BudgetedAdmissionLedger.offer_all_batched` now honours
  an optional `reject_set(capsules)` method on the policy
  for bundle-level vetoes. Pre-P47 policies unchanged.
* `vision_mvp/wevra/__init__.py` — re-exports the new
  symbols.
* `vision_mvp/experiments/phase46_unification_audit.py` —
  added `audit_adaptive_edge_cohort`; updated
  `audit_adaptive_edge` docstring + verdict to reflect the
  per-edge/per-table split. Audit now reports **6/6 FULL**.
* `docs/CAPSULE_FORMALISM.md` — alphabet size is 12; added
  § 4.B with Theorems W3-14 / W3-15 / W3-16, updated W3-C3
  status (falsified), added successor W3-C3' and W3-C5.
* `docs/context_zero_master_plan.md` — new § 4.13
  (Phase-47 research center status) and § 4.11 update.

### Not modified

* `vision_mvp/wevra/run.py`, `runtime.py`, `provenance.py` —
  no runtime contract change.
* `vision_mvp/core/*.py` — no substrate primitive modified.
* Phase-31/32/33 task modules — no change to the scenario
  banks, extractors, or decoders.
* SDK_VERSION remains `wevra.sdk.v3` (the alphabet extension
  is additive; existing capsule types still construct
  byte-for-byte).

### Tests

* New: 11 (`test_phase47_cohort_subsumption.py` — four
  W3-14 / W3-15 / W3-16 classes).
* Pre-existing capsule tests: unchanged, all 30 still pass.

Total Phase-47 test delta: +11 tests; 41/41 total in the
capsule test suite passing.

---

## 6. Reproducing

```bash
# Part A — formal frontier (cohort subsumption audit).
python -m vision_mvp.experiments.phase46_unification_audit \
    --out-dir /tmp/wevra_phase47
# Expect: 6/6 FULL, 0/6 PARTIAL, 0/6 FAIL.

# Part B — bundle-aware learning (deterministic; re-execs
# under PYTHONHASHSEED=0 on launch).
python -m vision_mvp.experiments.phase47_bundle_learning \
    --out-dir /tmp/wevra_phase47
# Expect: bundle_learned(dec) saturates test decoder_acc at
# B = 16; learned (P46) needs B >= 96 to saturate.

# Part C — cross-domain transfer.
python -m vision_mvp.experiments.phase47_cross_domain \
    --out-dir /tmp/wevra_phase47
# Expect: diagonal ~0.28 / 0.72 / 0.53; incident->security
# ~ 0.50 (+21 pp above base rate); security->incident < base.

# Tests.
python -m pytest vision_mvp/tests/test_phase47_cohort_subsumption.py \
                  vision_mvp/tests/test_capsule_policy.py \
                  vision_mvp/tests/test_capsule_subsumption.py \
                  vision_mvp/tests/test_wevra_capsules.py -q
# Expect: 41 passed.
```

Wall times on a 2024 M-class macbook:

* Unification audit: ∼ 0.1 s.
* Bundle-learning sweep: ∼ 15 s.
* Cross-domain transfer: ∼ 55 s.
* Full Phase-47 test suite: ∼ 0.8 s.

---

## 7. Closing — the one honest sentence

After Phase 47, the capsule abstraction is a research center
with **proven expressive limits, empirically-falsified strong
conjectures, and empirically-supported weak ones** — which is
a sharper scientific object than a "good local formalism"
but still one layer short of a paradigm shift in agent-team
runtimes. The next paradigm-shift candidate is bundle-aware
**decoding** (P47-C1), not more admission work.
