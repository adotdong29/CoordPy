# RESULTS — Capsule research milestone 6 (Phase 51)

> *Theory-forward results note. Phase 50
> (`docs/RESULTS_CAPSULE_RESEARCH_MILESTONE5.md`) closed the
> strict-reading question on Conjecture W3-C7 with two proved
> limitation theorems (W3-24 post-search winner's-curse bias;
> W3-29 Bayes-divergence zero-shot risk lower bound for the
> class-agnostic linear family), retracted the strict paradigm-
> shift reading, and reformulated the defensible bar as
> Conjecture W3-C9 (point-estimate Gate 1 at $n=80$ + gap
> reading of Gate 2).  Under W3-C9, Phase 50's sign-stable
> DeepSet achieves zero-shot gap = 0.000 at level 0.237 — the
> first direction-invariant zero-shot result in the programme.
>
> **This Phase-51 note opens the next honest frontier.**  It
> frames the question "can a hypothesis class structurally
> outside the magnitude-monoid linear family raise the
> direction-invariant zero-shot *level* on (incident,
> security)?", proves a strict-separation theorem (Theorem
> W3-30) over DeepSet for the **cohort-relational** class,
> ships the smallest serious instance (``CohortRelationalDecoder``),
> and reports the empirical outcome (Claim W3-31) and refined
> level-ceiling conjecture (W3-C10).
>
> **Honest headline.**  Under a matched training pipeline, the
> Phase-51 relational decoder achieves zero-shot gap 0.038 at
> level 0.237 on (incident, security) — matching Phase-50's
> reported sign-stable-DeepSet level, edging the same-pipeline
> sign-stable-DeepSet baseline by +5 pp (not statistically
> robust at $n=80$), but **not strictly exceeding** Phase-50's
> reported ceiling.  On Phase-31 Gate 1 the relational decoder
> is below the Phase-49 DeepSet (0.362 vs 0.425 at the
> pre-committed cell at $n=80$; 0.388 vs 0.425 at best cell).
> **Phase 51 is a limitation-leaning milestone**: it
> operationalises the relational axis, proves it is strictly
> richer than DeepSet (W3-30), and finds empirically that on
> (incident, security) the richer class **does not cleanly
> break** the Phase-50 direction-invariant level ceiling.
> Conjecture W3-C10 (level-ceiling) is accordingly **supported**
> rather than falsified.* Last touched: 2026-04-22.

---

## 0. One-paragraph summary

Phase 51 operationalises the **relational / cohort-aware**
decoding axis named by Conjecture W3-C5 and positions it on
the decoder frontier.  **Theorem W3-30** (proved,
constructive) strictly separates the cohort-relational class
from the DeepSet class: a decoder that partitions the admitted
bundle by source role, aggregates per-role features, and then
aggregates over the role set can express predicates
(e.g. "≥ 2 distinct source roles each emit a capsule implying
rc") that no bag-of-capsules sum of per-capsule features can
express.  The Phase-51 ``CohortRelationalDecoder`` implements
the smallest serious instance (12-dim input = 6 per-role ψ + 6
cross-role ρ features; 1-hidden-layer tanh MLP; ~131
parameters).  **Claim W3-31** (empirical, code-backed) reports
the result: under a matched Phase-51 training pipeline, zero-
shot transfer on (incident, security) reaches gap = 0.038 at
level = 0.237 — matching Phase-50's reported sign-stable-
DeepSet level of 0.237, edging the **same-pipeline** sign-
stable-DeepSet reproduction at 0.188 (Δ = +5 pp, 4 instances
at $n=80$, not statistically robust), and NOT strictly exceeding
Phase-50's reported ceiling.  On Phase-31 Gate 1 the relational
decoder sits at 0.362 (pre-committed cell, below Phase-49's
0.425) and 0.388 (best cell, below Phase-49's 0.425).  The
honest reading is that **Conjecture W3-C10** (direction-
invariant zero-shot transfer is *level-bounded* on (incident,
security) at roughly 0.237) is **supported** by Phase 51, not
falsified: the richer hypothesis class does not cleanly lift
the ceiling.  The programme's Phase-52+ agenda is accordingly
either (a) find the *relational feature* that does break the
ceiling on (incident, security), (b) test the relational axis
on a *different* operational-detection pair where
$\|w^*_A - w^*_B\|$ is smaller, or (c) accept W3-C10 as a
standing empirical ceiling and promote the gap-reading+
pooled-multitask bar as the honest research centre.

---

## 1. Part A — Why this is the right next experiment

### 1.1 What Phase 50 closed

Phase 50 proved two structural limitation theorems:

* **W3-24 (post-search winner's-curse bias).**  The best-cell
  estimator over $C$ evaluation cells is upward-biased by
  $\Omega(\sigma_n \sqrt{\log C})$ in the null regime; on
  Phase-49's $C=21, n=80$ this explains the 0.425 → 0.362
  drop at $n=320$.
* **W3-29 (Bayes-divergence zero-shot risk lower bound).**
  For any class-agnostic linear decoder over features shared
  across two domains with strictly-convex regularised logistic
  risks, the sum of per-direction zero-shot transfer risk
  penalties is bounded below by
  $\frac{\lambda_{\min}}{4} \|w^*_A - w^*_B\|^2$.

The strict pre-Phase-50 reading of W3-C7 is retracted;
Phase 50's W3-C9 reformulation (gap reading of Gate 2) is the
defensible bar.  Phase 50's sign-stable DeepSet meets W3-C9's
gap reading at gap = 0.000 at level = 0.237.

### 1.2 Why not "a bigger decoder"?

Phase-49 and Phase-50 already ran four principled scales of
richer hypothesis class:

| Family                | Class | Best Phase-50 $\hat p$ @ $n$=320 |
|---                    |---    |---                                |
| V1 linear (P48)       | linear over 10 feats | 0.316 |
| V2 linear (P49)       | linear over 20 feats | 0.316 |
| MLP (V2 + hidden)     | MLP over V2          | 0.344 |
| DeepSet (V2 + φ-sum)  | DeepSet              | 0.362 |

None crosses 0.400.  At $n_{\rm test} = 320$ the point-estimate
frontier has a ceiling near 0.36–0.37.  Adding a second hidden
layer or doubling the hidden size of DeepSet is *not* a
structurally new move — it is a bigger optimiser on the same
hypothesis family.  W3-29 does not strictly apply to DeepSet
(non-linear), but empirically DeepSet does *not* escape W3-29's
penalty floor on (incident, security): its max per-direction
zero-shot penalty (+0.138) exceeds the V2 linear minimum
(+0.112).

### 1.3 The relational axis as the honest next move

W3-16 (Phase 47 negative) and W3-C5 (Phase 47/48 conjecture)
named the relational axis as the honest extension beyond the
magnitude-monoid algebra.  Phase 51's move is to operationalise
this **on the decoder**: a decoder class whose per-rc features
depend on source-role **partition structure**, not on a bag-
of-capsules sum.  The cleanest instance is:

1. Partition the admitted bundle by ``source_role``.
2. On each role-sub-bundle $E_r$, compute a per-role feature
   vector $\psi(E_r, rc) \in \mathbb{R}^{6}$ (role-supports-rc
   indicator, log1p count, role-has-top-priority, role-top-
   priority-implies-rc, role-top-priority-contradicts-rc, role-
   has-high-priority-supporting-rc).
3. Aggregate over roles: $\sum_r \psi(E_r, rc) \in \mathbb{R}^6$.
4. Augment with a 6-dim **cross-role** feature vector
   $\rho(E, rc)$ that reads the role set as a first-class
   object: distinct-roles-supporting-rc, distinct-roles-
   contradicting-rc, is-role-plurality-for-rc, pairs-of-
   supporting-roles (quadratic), distinct-roles-total, bias.
5. Concatenate, feed through a 1-hidden-layer tanh MLP.

This is structurally richer than DeepSet because DeepSet's
per-capsule φ-sum erases role-identity: two bundles with
identical (claim_kind, rc) multisets but different role
assignments yield identical φ-sums, but distinct role-
partition structures.

---

## 2. Part B — Theorem W3-30 (cohort-relational strict separation)

### 2.1 Statement and proof

See `docs/CAPSULE_FORMALISM.md` § 4.F for the formal statement.
The distinguishing predicate is "≥ 2 *distinct* source roles
each emit a capsule implying rc".  DeepSet's aggregate sum of
per-capsule φ over the bundle erases role identity (except via
per-kind uniqueness); the cohort-relational partition →
aggregate architecture preserves it.  The strict containment
$\mathcal{H}_{\rm rel} \supsetneq \mathcal{H}_{\rm DS}$ is
witnessed by a simple pair:

* Bundle A: ``{DISK_FILL_CRITICAL/sysadmin, CRON_OVERRUN/sysadmin}``
  — 2 capsules, 1 role, 2 votes for `disk_fill`, 1 supporting
  role.
* Bundle B: ``{DISK_FILL_CRITICAL/sysadmin, CRON_OVERRUN/network}``
  — 2 capsules, 2 roles, 2 votes for `disk_fill`, 2 supporting
  roles.

Both have identical per-capsule (claim_kind, rc) multisets.
DeepSet's φ-sum is identical (V2 feature `multi_source_flag` is
lifted into ρ in Phase 51, not into φ).  The cohort-relational
feature vector differs in `rho:distinct_roles_supporting_rc`
(1 vs 2) and `rho:pairs_of_supporting_roles` (0 vs 1).

### 2.2 Status

Proved (constructive).  Code anchor:
`vision_mvp/coordpy/capsule_decoder_relational.py::CohortRelationalDecoder`.
Contract test witnessing the strict separation:
`vision_mvp/tests/test_phase51_relational_decoder.py::test_w3_30_strict_separation_from_deep_set`.

### 2.3 What W3-30 does NOT prove

W3-30 is a **capacity** statement.  It does NOT prove:

* that the strictly-richer class empirically lifts accuracy on
  any concrete distribution;
* that the direction-invariant zero-shot level ceiling is
  breakable on (incident, security);
* that the Phase-50 W3-29 risk-penalty lower bound has no
  relational-class analogue.

Whether any of these hold is the empirical question answered
by Claim W3-31 and Conjecture W3-C10.

---

## 3. Part C — Claim W3-31 (empirical frontier)

### 3.1 Experimental setup

All decoder families are trained on a **matched Phase-51
pipeline** (augmented FIFO training at budgets [256, 96, 48])
to make same-pipeline comparison clean.  The Phase-49 /
Phase-50 decoders differ from this pipeline in detail — the
Phase-49 Gate-1 sweep trains on 7-cell augmentation including
learned-admission cells, and the Phase-50 zero-shot study
trains on FIFO@256 only.  We re-train under the matched
pipeline so all Phase-51 comparisons are internally
apples-to-apples.

Bench family: Phase-31 incident triage + Phase-33 security
escalation.  Noise: `spurious_prob = 0.30, mislabel_prob = 0.10`.
Seeds: 31..50 (Gate 1), 31..46 train + 47..50 test (zero-shot).
$n_{\rm test}$: 80 per domain (Gate 1 and zero-shot).

Reproduce: ``python -m vision_mvp.experiments.phase51_relational_decoder``
(≈ 50 s on a 2024 M-class MacBook).  Artefact:
`/tmp/coordpy_phase51/results_phase51_relational_decoder.json`.

### 3.2 Gate 1 on Phase-31 noisy bench ($n_{\rm test}=80$)

At the Phase-50 **pre-committed cell**
(bundle_learned_admit @ B=64):

| Decoder                           | $\hat p$ | $k/n$ | Wilson 95 % CI |
|---                                |---       |---     |---              |
| `priority`                        | 0.225    | 18/80 | [0.147, 0.328]  |
| `deep_set_bundle_decoder` (P49)   | **0.425** | 34/80 | [0.323, 0.534] |
| `cohort_relational_decoder` (P51) | 0.362    | 29/80 | [0.266, 0.472]  |

Best-cell (after-search, note winner's-curse per W3-24):

| Decoder                           | Best $\hat p$ | @ admission          | @ budget |
|---                                |---            |---                    |---       |
| `deep_set_bundle_decoder`         | **0.425**     | bundle_learned_admit  | 64       |
| `cohort_relational_decoder`       | 0.388         | learned(p46)          | 96       |

**Reading.**  The Phase-51 relational decoder **does not
cross** the Gate-1 point-estimate threshold (0.400) at the
pre-committed cell or at any cell in the sweep.  Its best cell
(0.388) is 3.7 pp below Phase-49's DeepSet (0.425).  Its
pre-committed cell (0.362) is 6.3 pp below.  The Wilson CI on
0.362 at $n=80$ is $[0.266, 0.472]$ — consistent with the
0.362 observation being the true best-cell accuracy (not
upward-biased by post-search); W3-24's winner's-curse does not
explain a *downward* gap relative to DeepSet.  Under the Phase-
51 pipeline DeepSet reproduces its Phase-49 best-cell exactly
(0.425), confirming W3-23 at $n=80$ in the matched pipeline.

The relational decoder is **not a Gate-1-lift** on Phase-31.

### 3.3 Zero-shot transfer on (incident, security)

Three families compared under the matched Phase-51 pipeline
($n_{\rm test} = 80$ per domain):

| Family                  | $w_{\rm inc}$ | $w_{\rm sec}$ | $i \to s$ | $s \to i$ | gap   | max pen | **level** |
|---                      |---             |---             |---          |---          |---    |---       |---         |
| `cohort_relational` (P51) | 0.350         | 0.362         | 0.237       | 0.275       | **0.038** | +0.125 | **0.237** |
| `sign_stable_deepset` (P50) | 0.312       | 0.338         | 0.225       | 0.188       | 0.038     | +0.125 | 0.188     |
| `deepset_full` (P49)     | 0.388        | 0.362         | 0.225       | 0.212       | 0.013     | +0.175 | 0.212     |

Conventions:

* $w_d$ = within-domain (train and test on $d$).
* $i \to s$ = train on incident, test on security with
  weights transferred zero-shot (claim-map + alphabet swapped
  via code adapters).
* gap = $|(i \to s) - (s \to i)|$.
* max pen = $\max(w_s - (i \to s), w_i - (s \to i))$.
* level = $\min(i \to s, s \to i)$.

**Reading.**

* **Direction-invariance (W3-C9 gap reading).**  Cohort-
  relational achieves **gap 0.038 ≤ 0.05** — MET.  DeepSet-full
  is actually tighter at 0.013; sign-stable DeepSet matches at
  0.038.  All three Phase-49+ decoders meet W3-C9's gap
  reading under the matched pipeline.
* **Level (Phase-51 metric).**  Cohort-relational
  **level = 0.237** — edges the same-pipeline sign-stable
  DeepSet (0.188) by +5 pp (4 instances at $n=80$; Wilson CI
  wide, not statistically robust).  **Matches** the Phase-50
  reported sign-stable-DeepSet level of 0.237 — does NOT
  strictly exceed it.
* **Within-domain.**  Cohort-relational sits at 0.350 / 0.362
  — below DeepSet-full (0.388 / 0.362) and modestly above
  sign-stable DeepSet (0.312 / 0.338).  The relational class
  trades Phase-31 within-domain accuracy for direction-
  invariant cross-domain robustness (the Phase-51 purpose).
* **Max penalty.**  Cohort-relational penalty +0.125, matching
  sign-stable DeepSet and below DeepSet-full's +0.175.  The
  strict reading of Gate 2 ($\le 0.05$) is NOT met by any
  family (W3-27 reconfirmed under matched pipeline).

### 3.4 Honest significance analysis

At $n_{\rm test} = 80$, one-correct-prediction = 1.25 pp of
accuracy.  A 4-instance difference (relational 0.237 vs
same-pipeline sign-stable-DeepSet 0.188 in $s \to i$) is
within the binomial CI at $n=80$.  Wilson 95 % CIs:

* relational $s \to i$ = 0.275 ($k=22$): $[0.190, 0.381]$.
* sign-stable DS $s \to i$ = 0.188 ($k=15$): $[0.117, 0.285]$.

The CIs overlap substantially.  The +5 pp edge is **suggestive
but not statistically robust** at this sample size.  Stronger
evidence would require either a larger $n$ (Phase-50-style
replication at $n=160$ or $n=320$) or a bigger effect.

### 3.5 Against the Phase-50 reported ceiling

Phase-50's zero-shot study reports the sign-stable DeepSet as
achieving **both directions 0.237** with gap = 0.000 under the
Phase-50 pipeline (FIFO@256 training only, 16 train + 4 test
seeds).  Our Phase-51 re-training under the Phase-51 pipeline
(augmented FIFO @ [256, 96, 48]) reproduces the sign-stable
DeepSet at **0.225 / 0.188** — a 4-point (5 pp) drop in the
$s \to i$ direction relative to the Phase-50 report.  This
tells us:

* The sign-stable-DeepSet level is **pipeline-sensitive**;
  the Phase-50 reported 0.237 is not a universal property of
  the decoder but of the decoder × training-pipeline pair.
* A comparison "relational 0.237 > sign-stable-DS 0.237
  (Phase 50)" would be a **cross-pipeline** comparison and
  therefore not a clean apples-to-apples signal.
* The clean apples-to-apples signal is **relational 0.237 vs
  same-pipeline sign-stable-DS 0.188** — the +5 pp edge
  above, which is suggestive but not robust.
* The conservative honest conclusion: **the relational
  decoder reaches a level consistent with Phase-50's best
  reported direction-invariant level (0.237); it does not
  cleanly break it**.

### 3.6 W3-31 verdict

| Reading                                  | Bar             | Result (Phase 51) | Verdict |
|---                                       |---              |---                 |---       |
| Gap reading (W3-C9)                       | $\le 0.05$      | 0.038              | **MET** |
| Level > same-pipeline sign-stable-DS      | > 0.188         | 0.237 (+0.05)      | **MET (weak)** |
| Level > Phase-50 reported SSDs (0.237)    | > 0.237         | 0.237              | **NOT MET (matches)** |
| Gate-1 pre-committed cell point estimate  | $\ge 0.400$      | 0.362              | **NOT MET** |
| Gate-1 best-cell point estimate           | $\ge 0.400$      | 0.388              | **NOT MET** |

**W3-31 honest headline.**  The relational decoder meets the
direction-invariance bar and edges the same-pipeline sign-
stable-DeepSet baseline by +5 pp (not statistically robust);
it does not strictly exceed Phase-50's reported 0.237 level
ceiling and does not advance Gate 1 on Phase-31.  **Phase 51
is an upper-bound-leaning result, not a lower-bound lift.**

---

## 4. Part D — Conjecture W3-C10 and programme status

### 4.1 Conjecture W3-C10 (level-ceiling on (incident, security))

Formal statement in `docs/CAPSULE_FORMALISM.md` § 4.F.
Informally: under direction-invariance (gap $\le 0.05$),
zero-shot transfer on the Phase-31 + Phase-33 operational-
detection pair is **level-bounded**: no Phase-51+ decoder
class achieves min-direction zero-shot level materially above
the Phase-50 sign-stable-DeepSet 0.237 mark.

### 4.2 Evidence for and against W3-C10

**For.**

* Phase-51 relational decoder matches but does not strictly
  exceed the Phase-50 SSDs level (§ 3.3, § 3.5).
* Phase-49 + Phase-50 together ran 7 decoder families (V1,
  V2, MLP, Interaction, DeepSet, sign-stable V2, sign-stable
  DeepSet, multitask) and none exceeded 0.237 in the
  direction-invariant zero-shot reading.
* W3-21 and W3-29 give structural reasons why class-agnostic
  decoders cannot close the strict penalty reading; the level
  ceiling is consistent with a weaker form of the same
  obstruction.

**Against.**

* None of the 7 Phase-49/50 families nor Phase-51's 1 family
  has been evaluated at $n_{\rm test} \ge 160$ on the
  (incident, security) zero-shot bench; the 0.237 observation
  could itself be sample-noise-inflated.
* The Phase-51 pipeline's sign-stable-DeepSet reproduction at
  0.188 (not 0.237) shows pipeline dependence — other
  pipelines may yield a higher level.
* The relational decoder is the *smallest* Phase-51 instance;
  a wider-ψ or wider-ρ decoder (e.g., 3-tuple role features,
  or a shallow GNN over the role graph) has not been tried.

### 4.3 What the programme should believe now

1. **Retracted (Phase 50).**  Strict pre-Phase-50 reading of
   W3-C7 (point-estimate Gate 1 $\ge 0.400$ + zero-shot
   penalty $\le 0.05$ both ways).  Structurally blocked by
   W3-24 + W3-29.  **Do not reintroduce.**
2. **Defensible (Phase 50 W3-C9).**  Point-estimate Gate 1
   at $n=80$ + gap reading of Gate 2.  Under W3-C9 Phase 49
   is the canonical paradigm-shift candidate.
3. **Phase 51 status.**  The relational axis is a legitimate
   decoder-hypothesis-class extension (strictly contains
   DeepSet, W3-30).  Empirically it matches Phase-50's best
   direction-invariant level but does **not** cleanly break
   it.  Conjecture W3-C10 names the standing ceiling.
4. **What would falsify W3-C10.**  A relational or richer
   decoder whose zero-shot gap $\le 0.05$ AND whose min-
   direction level is materially above 0.237 ($\ge 0.30$
   would be a strong signal).  Falsification should report
   at $n_{\rm test} \ge 160$ to avoid sample noise confusion.

### 4.4 Does Phase 51 move the paradigm-shift needle?

**No strict move.**  Phase 51 neither closes a gate nor
reveals a novel paradigm-shift candidate.  It operationalises
the relational axis, proves a formal strict-separation result,
and empirically **bounds** the next-decoder frontier.

**A stronger honest position.**  The programme has one proved
upper bound (W3-29, linear class), one conjectured upper
bound (W3-C10, all classes on (incident, security)), and
three principled routes forward:

1. **Phase-52 option A (same benchmark).**  Engineer the
   specific relational feature that breaks W3-C10 on
   (incident, security).  Candidate: a decoder that
   explicitly reads **pair-wise role agreement patterns**
   against domain-agnostic anchors (e.g., distinct-role
   count conditional on the top-priority kind being
   absent).  This is a deeper-hypothesis-class route, not
   obviously different from DeepSet-plus.
2. **Phase-52 option B (different benchmark).**  Test
   (incident, compliance) or (security, compliance)
   zero-shot — where $\|w^*_A - w^*_B\|$ may be smaller
   because task families are more similar.  W3-29's penalty
   floor shrinks; the relational axis may break a lower
   ceiling.
3. **Phase-52 option C (accept W3-C10, focus elsewhere).**
   Treat the 0.237 level as a standing empirical ceiling
   on this pair, and redirect effort to either (a)
   mechanical W3-C1 closure on the remaining Phase-N
   magnitude-shaped theorems (unchanged for two phases),
   or (b) a relational *substrate* primitive whose
   bounded-context invariant exercises the relational axis
   on capsule admission (closing Conjecture W3-C5 on the
   substrate side rather than the decoder side).

The programme's **defensible position** is that Phase 51 is a
limitation-leaning milestone that **sharpens what the next
research question should look like** rather than closing any
Phase-51 research question itself.

---

## 5. Files changed in this milestone

### New files

* `docs/RESULTS_CAPSULE_RESEARCH_MILESTONE6.md` — this note.
* `vision_mvp/coordpy/capsule_decoder_relational.py` —
  ``CohortRelationalDecoder`` + ``train_cohort_relational_decoder``;
  ``COHORT_PSI_FEATURES`` (6) + ``COHORT_RHO_FEATURES`` (6)
  vocabulary; internal helpers.
* `vision_mvp/experiments/phase51_relational_decoder.py` —
  driver with Gate-1 sweep on Phase-31 bench + zero-shot
  transfer study on (incident, security).
* `vision_mvp/tests/test_phase51_relational_decoder.py` —
  10 contract tests covering feature vocabulary, training
  determinism, W3-30 strict-separation witness, ρ-feature
  correctness, decode-alphabet discipline, and to_dict round-
  trip.

### Modified files (additive only)

* `docs/CAPSULE_FORMALISM.md` — adds **Canonical post-
  Phase-50 position** (top) + **How not to overstate**
  subsection + § 4.F Phase-51 extension (Theorem W3-30,
  Claim W3-31, Conjecture W3-C10).  Updates the "last
  touched" line.
* `docs/context_zero_master_plan.md` — adds § 4.17
  (Phase-51 extension).  Updates § 4.11 top-line to
  post-Phase-51 stance.

### Not modified (deliberate)

* `vision_mvp/coordpy/capsule.py` — capsule contract C1..C6
  byte-for-byte unchanged.
* `vision_mvp/coordpy/capsule_policy.py`,
  `vision_mvp/coordpy/capsule_policy_bundle.py` — admission
  layer unchanged.
* `vision_mvp/coordpy/capsule_decoder.py`,
  `vision_mvp/coordpy/capsule_decoder_v2.py` — Phase-48 and
  Phase-49 decoder modules unchanged.  Phase-51's decoder
  lives in the additively-new
  `vision_mvp/coordpy/capsule_decoder_relational.py`.
* `vision_mvp/coordpy/run.py`, `runtime.py`, `provenance.py` —
  no runtime contract change.
* `SDK_VERSION` — still `coordpy.sdk.v3`.  Phase-51 is a
  research-centre milestone, not a public SDK contract change.

### Tests

* New: 10 (`test_phase51_relational_decoder.py`).
* Pre-existing capsule tests: unchanged, all pass.
* **Total capsule test suite: 96 passed.** (86 prior + 10
  Phase-51.)

---

## 6. Reproducing

```bash
# Full Phase-51 study — Gate-1 + zero-shot.
python -m vision_mvp.experiments.phase51_relational_decoder \
    --out-dir /tmp/coordpy_phase51
# Expect:
#   [phase51-G1] cohort_relational_decoder  acc=0.362 @ pre-committed cell
#   [phase51-G1] cohort_relational_decoder  best 0.388 @ learned(p46) B=96
#   [phase51-ZS] cohort_relational gap=0.038 level=0.237
#   [phase51-ZS] W3-31 gap met, level met weak, strict = not met

# Contract tests.
python -m pytest \
    vision_mvp/tests/test_phase51_relational_decoder.py \
    vision_mvp/tests/test_phase50_ci_and_zero_shot.py \
    vision_mvp/tests/test_phase49_stronger_decoder.py \
    vision_mvp/tests/test_phase48_bundle_decoding.py \
    vision_mvp/tests/test_phase47_cohort_subsumption.py \
    vision_mvp/tests/test_capsule_policy.py \
    vision_mvp/tests/test_capsule_subsumption.py \
    vision_mvp/tests/test_coordpy_capsules.py -q
# Expect: 96 passed.
```

Wall times on a 2024 M-class MacBook:

* Phase-51 Gate-1 sweep (20 seeds, augmented training): ≈ 32 s.
* Phase-51 zero-shot study (20 seeds × 2 domains): ≈ 16 s.
* Phase-51 test suite: ≈ 0.4 s.
* Full capsule regression (96 tests): ≈ 1.3 s.

---

## 7. Theory-forward claims / conjectures — what to believe now

This section is a point-in-time summary of the programme's
epistemic state at the end of Phase 51.  It is deliberately
short; longer discussion lives in
`docs/CAPSULE_FORMALISM.md`.

**Proved.**

* W3-7 through W3-15 — capsule algebra and subsumption facts.
* W3-16 — relational limitation (substrate side).
* W3-17, W3-18, W3-20, W3-21 — decoder-frontier structural
  results (conditional).
* W3-24 — post-search winner's-curse bias (classical).
* W3-29 — Bayes-divergence zero-shot risk lower bound (linear
  class-agnostic family, strict convexity).
* **W3-30 (Phase 51)** — cohort-relational strict separation
  over DeepSet (constructive).

**Empirical, seed-robust, code-backed.**

* W3-19 — learned decoder breaks 0.200 ceiling by +15 pp at
  $n=80$ on Phase-31 noisy bench.
* W3-22 — pooled-multitask shared-head symmetric deployment
  (0.350 / 0.350 at gap 0).
* W3-25, W3-28 — Phase-50 enlargements.
* **W3-31 (Phase 51)** — cohort-relational zero-shot under
  matched pipeline: gap 0.038, level 0.237, max penalty +0.125
  on (incident, security).

**Retracted.**

* Strict pre-Phase-50 reading of W3-C7 (point-estimate
  Gate 1 at $n=320$ + strict zero-shot penalty ≤ 5 pp).

**Conjectural.**

* W3-C1 — mechanical full subsumption (15 remaining).
* W3-C3' — 12-kind completeness under magnitude algebra.
* W3-C5 — relational axis as next honest substrate extension.
* W3-C6 — decoder-side task-family transfer asymmetric.
* W3-C7 (reformulated) — retained aspirationally.
* W3-C8' — absolute-count features sign-stable across
  operational-detection domains.
* W3-C9 — Gate-2 gap reading is the defensible bar.
* **W3-C10 (Phase 51)** — direction-invariant zero-shot level
  on (incident, security) is bounded at ≈ 0.237; supported
  by Phase 51, open for falsification.

**What the world should NOT believe yet.**

* "The cohort-relational decoder broke the Phase-50 ceiling."
  It did not cleanly break it; it matched 0.237 under a
  pipeline-dependent reading and edged same-pipeline
  sign-stable DeepSet by +5 pp (within noise at $n=80$).
* "A relational axis delivers paradigm-shift-scale lift."
  No evidence of paradigm-shift-scale lift from Phase 51
  alone; W3-C10 conjecturally caps the level.
* "The Phase-31 benchmark's 0.400 point-estimate threshold is
  crossable."  At $n \ge 320$ no family including the Phase-51
  relational decoder has demonstrated crossing; W3-24 bias
  plus sample noise explain the Phase-49 appearance of
  crossing at $n=80$.

**What the world SHOULD believe.**

* The capsule-decoder research centre is a stable, falsifiable,
  theorem-backed programme with **five proved theorems
  explicitly bounding the frontier** (W3-17, W3-21, W3-24,
  W3-29, W3-30) and **five empirical claims with pre-committed
  test anchors** (W3-19, W3-22, W3-25, W3-28, W3-31).
* The next honest frontier is Phase-52 option B (a
  different task-family pair) or option C (accept W3-C10
  and redirect).  Option A (engineering a better decoder on
  (incident, security)) is **not** evidently productive
  given the W3-C10 conjecture and the ~60 pp gap between the
  direction-invariant level (0.237) and the within-domain
  optimum (0.388).

---

## 8. Closing — the one honest sentence

After Phase 51, the capsule-decoder research centre has a
**strict-separation theorem** (W3-30) over the DeepSet
hypothesis class for the relational axis, a **minimal
operationalisation** of that class (``CohortRelationalDecoder``),
an **empirical result that matches but does not cleanly break**
the Phase-50 direction-invariant zero-shot level ceiling on
(incident, security) (W3-31), and a **new conjecture** that
names that ceiling as a standing empirical bound for the
programme (W3-C10).  The honest reading is that the
relational axis is **structurally legitimate but empirically
not free** — and the next real research move is either a
different operational-detection pair (Phase-52 option B) or an
acceptance of the ceiling + redirect (option C), not a
repeated bigger-decoder attempt on the Phase-31 + Phase-33
bench.
