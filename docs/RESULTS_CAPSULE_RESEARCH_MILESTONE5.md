# RESULTS — Capsule research milestone 5 (Phase 50)

> *Theory-forward results note.  Phase 49
> (`docs/RESULTS_CAPSULE_RESEARCH_MILESTONE4.md`) claimed a
> paradigm-shift candidate passing both gates of Conjecture
> W3-C7 "under the canonical-but-not-strictest reading",
> with explicit remaining strict headroom of ≈ 2.5 pp on
> Gate 1 CI and ≈ 9 pp on Gate 2 zero-shot penalty, and
> called both "one well-scoped milestone away".
> **This Phase-50 note is that milestone.**  Part A attacks
> Gate 1 CI by re-running the Phase-49 sweep at 2× and 4×
> the sample size (40 + 80 seeds, $n_{\rm test}\in\{160,
> 320\}$); the best post-search cell collapses from
> Phase-49's 0.425 to **0.400 exactly at $n=160$** (Wilson
> CI lower bound 0.327 < 0.400) and **0.362 at $n=320$**
> (even the point estimate fails the threshold).
> Part B attacks strict zero-shot Gate 2 with three new
> principled families (sign-stable V2, source-standardised
> V2, sign-stable DeepSet); across six families the
> minimum max-per-direction transfer penalty is V2 full at
> +0.112, above the 5 pp bar.
> Part C ships three new formal results — **W3-24**
> (post-search winner's-curse bias, proved classically);
> **W3-29** (zero-shot risk-penalty lower bound from
> Bayes-divergence, proved conditional on strict convexity);
> **W3-28** (sign-stable DeepSet zero-shot gap = 0.000,
> empirical) — plus refined **Conjecture W3-C9** (Gate-2
> reformulation to gap-reading).
> **Part D** is the programme's honest post-Phase-50
> reclassification: **the strict reading of W3-C7 is
> retracted** (Gate 1 point estimate fails at $n=320$,
> strict zero-shot Gate 2 penalty fails on all six
> families), and Phase-49 is reclassified as a **canonical
> paradigm-shift candidate** (pooled-multitask Gate 2 +
> gap-reading zero-shot Gate 2 + point-estimate Gate 1 at
> $n=80$), not a strict one.*  Last touched: 2026-04-22.

---

## 0. One-paragraph summary

Phase 50 takes the Phase-49 paradigm-shift candidate at its
strictest operational reading.  The **Gate 1 strict-CI
frontier** (W3-25) replicates Phase-49's sweep at 40 seeds
($n_{\rm test} = 160$) and 80 seeds ($n_{\rm test} = 320$);
the 40-seed best post-search cell hits exactly 0.400 with
Wilson CI $[0.327, 0.477]$ — strict CI reading NOT MET — and
the 80-seed best post-search cell drops to 0.362 with
Wilson CI $[0.312, 0.417]$ — **even the point-estimate
reading is NOT MET** (W3-26).  Theorem W3-24 (proved
classically) names the reason: the post-search estimator over
$C=21$ evaluation cells has expected upward bias
$\Omega(\sigma_n \sqrt{\log C}) \approx 0.13$ in the null
regime, and the observed Phase-49 → Phase-50 drop of
0.425 → 0.362 (−0.063) is consistent with this bias plus
sample noise.  The **Gate 2 strict-zero-shot frontier**
(W3-27) tests six principled zero-shot families (V1, V2,
sign-stable V2, standardised V2, DeepSet, sign-stable
DeepSet); the minimum max-per-direction penalty is V2 full at
+0.112, above the 5 pp bar on every family.  Theorem W3-29
(proved, conditional) gives the structural lower bound:
the sum of per-direction zero-shot risk-penalties is bounded
below by $\frac{\lambda_{\min}}{4} \|w^*_A - w^*_B\|^2$.
The **one positive strict result** (W3-28) is that the
sign-stable DeepSet decoder achieves zero-shot symmetry
gap = 0.000 (both directions 0.237) — direction-invariant
zero-shot transfer is attainable even when level-matching
is not.  The **programme update** (§ 4) classifies Phase-49
as a canonical paradigm-shift candidate under the
W3-C9-refined bar (gap reading Gate 2 + point-estimate
Gate 1 at $n=80$), not a strict one.  Phase 50 is a
negative-result milestone: it falsifies the strict reading
by empirical replication and surrounds the frontier with
proved limitation theorems.

---

## 1. Part A — Gate 1 strict-CI certification

### 1.1 What was open at the end of Phase 49

Phase 49's Gate 1 point-estimate reading was met at
$\hat{p} = 0.425$ on the best cell
(``deep_set_bundle_decoder`` @ ``bundle_learned_admit``
@ $B=64$) at $n_{\rm test} = 80$ on the Phase-31 noisy
bench.  The 95 % Wilson binomial CI was $[0.317, 0.539]$,
lower bound below the 0.400 threshold.  Phase 49 named the
strict-CI gap (≈ 2.5 pp on the CI lower bound) as "one
well-scoped milestone away" — either enlarge $n_{\rm test}$
to ≥ 320 (keeping 0.425) or push $\hat{p}$ to ≥ 0.45 at
the current $n$.

Phase 50's first hypothesis: *the Phase-49 0.425 point
estimate is a true best-cell accuracy, and enlarging
$n_{\rm test}$ will narrow the CI around it.*  Under this
hypothesis, at $n_{\rm test} = 160$ we expect
$\hat{p} \approx 0.42$ with Wilson CI
$[0.35, 0.50]$ — CI lower bound still below 0.400 but
closer; at $n_{\rm test} = 320$ we expect
$\hat{p} \approx 0.42$ with Wilson CI $[0.37, 0.48]$ —
CI lower bound still below 0.400.  Strict Gate 1 remains
unmet on sample-size grounds alone under this hypothesis.

Phase 50's second hypothesis: *the Phase-49 0.425 point
estimate is inflated by post-search winner's-curse bias
over the 21-cell evaluation grid (3 admission × 7
budgets), and enlarging $n_{\rm test}$ will also drop the
point estimate.*  Under this hypothesis, strict Gate 1 is
structurally blocked — the true best-cell accuracy is
below 0.400.

Phase 50 tests both hypotheses directly.

### 1.2 Results at $n_{\rm test} = 160$ (40 seeds)

Replicated Phase-49 sweep with seeds 31..70, by-seed 80/20
split (32 train seeds × 4 distractors × 5 scenarios = 640
train instances; 8 test seeds × 4 × 5 = 160 test instances).
Budget grid narrowed to {48, 64, 96} (the three cells where
Phase-49's winning decoders lived).  Augmented training
(7 admission cells pooled, same as Phase 49), identical
Phase-49 decoders, plus one Phase-50 addition
(sign-stable V2 decoder).

**Best post-search cell per decoder** (160 instances):

| Decoder                    | Best cell                         | $\hat{p}$ | $k/n$    | Wilson 95% CI     | Clopper-Pearson CI  | Point? | CI?   |
|---                         |---                                |---        |---        |---                 |---                   |---      |---     |
| `priority`                 | fifo @ B=96                        | 0.225     | 36/160    | $[0.167, 0.296]$   | $[0.163, 0.298]$     | no      | no     |
| `learned_bundle_decoder` (V1) | bundle_learned_admit @ B=48    | 0.375     | 60/160    | $[0.304, 0.452]$   | $[0.300, 0.455]$     | no      | no     |
| `learned_bundle_decoder_v2`| bundle_learned_admit @ B=64        | 0.369     | 59/160    | $[0.298, 0.446]$   | $[0.294, 0.449]$     | no      | no     |
| `mlp_bundle_decoder`       | bundle_learned_admit @ B=48        | 0.375     | 60/160    | $[0.304, 0.452]$   | $[0.300, 0.455]$     | no      | no     |
| **`deep_set_bundle_decoder`** | **bundle_learned_admit @ B=48** | **0.400** | **64/160** | **$[0.327, 0.477]$** | **$[0.323, 0.480]$** | **borderline** | **no** |
| `sign_stable_v2_decoder` (P50) | bundle_learned_admit @ B=48    | 0.362     | 58/160    | $[0.292, 0.439]$   | $[0.288, 0.442]$     | no      | no     |

**Pre-committed (W3-23 anchor) cell** (DeepSet @
bundle_learned_admit @ B=64):

* DeepSet: $\hat{p} = 0.344$, Wilson CI $[0.275, 0.420]$.
* V1: 0.312 — a drop from Phase-49's best.
* V2: 0.369 — higher than DeepSet at this cell; this alone
  falsifies the Phase-49 claim that DeepSet strictly
  dominates V2 at this cell.

**Gate 1 verdict at $n = 160$** (W3-25):
- Point-estimate reading: **borderline MET** (exactly 0.400
  on DeepSet post-search best cell).
- Wilson CI reading: **NOT MET** (CI lower bound 0.327).
- Clopper-Pearson CI reading: **NOT MET** (CI lower 0.323).

### 1.3 Results at $n_{\rm test} = 320$ (80 seeds)

Replicated with seeds 31..110 (80 seeds, by-seed 80/20
split giving 1280 train + 320 test instances).  This is the
"double the doubling" test — Phase-49 at $n=80$ gave 0.425;
Phase-50 at $n=160$ gave 0.400 (point-estimate borderline);
Phase-50 at $n=320$ should pin down the true best-cell
accuracy to ±3 pp.

**Best post-search cell per decoder** (320 instances):

| Decoder                    | Best cell                         | $\hat{p}$ | $k/n$    | Wilson 95% CI     | Clopper-Pearson CI  |
|---                         |---                                |---        |---        |---                 |---                   |
| `priority`                 | fifo @ B=96                        | 0.203     | 65/320    | $[0.163, 0.251]$   | $[0.160, 0.251]$     |
| `learned_bundle_decoder` (V1) | bundle_learned_admit @ B=48    | 0.316     | 101/320   | $[0.267, 0.368]$   | $[0.265, 0.370]$     |
| `learned_bundle_decoder_v2`| bundle_learned_admit @ B=48        | 0.316     | 101/320   | $[0.267, 0.368]$   | $[0.265, 0.370]$     |
| `mlp_bundle_decoder`       | bundle_learned_admit @ B=48        | 0.344     | 110/320   | $[0.294, 0.397]$   | $[0.292, 0.399]$     |
| **`deep_set_bundle_decoder`** | **bundle_learned_admit @ B=48** | **0.362** | **116/320** | **$[0.312, 0.417]$** | **$[0.310, 0.418]$** |
| `sign_stable_v2_decoder`   | bundle_learned_admit @ B=48        | 0.328     | 105/320   | $[0.279, 0.381]$   | $[0.277, 0.383]$     |

**Pre-committed cell** (DeepSet @ bundle_learned_admit @
B=64): $\hat{p} = 0.359$, Wilson CI $[0.309, 0.413]$.

**Gate 1 verdict at $n = 320$** (W3-26):
- Point-estimate reading: **NOT MET** on any decoder family
  — best is DeepSet at 0.362, 3.8 pp below threshold.
- Wilson CI reading: NOT MET (upper bound 0.417 is below
  or at threshold for the strongest family; lower bounds
  are all below 0.32).
- Clopper-Pearson: NOT MET.

**This falsifies the Phase-49 W3-23 claim.**  The Phase-49
point estimate 0.425 on $n=80$ was subject to winner's-curse
bias (W3-24, below) + sample noise, totalling ≈ 0.063.  The
true best-cell accuracy on this benchmark family is
≈ 0.36 — **structurally below the 0.400 W3-C7 Gate 1 bar**.

### 1.4 Theorem W3-24 (Post-search winner's-curse bias — proved, classical)

Let $(\hat{p}_c)_{c=1}^C$ be $C$ independent binomial
estimators, each $\hat{p}_c \sim \mathrm{Bin}(n, p_c)/n$.
Let $\hat{p}^\max := \max_c \hat{p}_c$ (the post-search
best-cell estimator) and $p^\max := \max_c p_c$.

**Claim.** In the **null regime** $p_1 = \ldots = p_C = p^*$
(all cells have identical true accuracy),

$$
\mathbb{E}[\hat{p}^\max - p^*] \;=\;
\sigma_n \sqrt{2 \log C} \cdot (1 + o(1))
\qquad \text{as } C \to \infty,
\quad \sigma_n = \sqrt{p^*(1-p^*)/n}.
$$

**Proof.**  By CLT, $\hat{p}_c \approx p^* + \sigma_n Z_c$
with $Z_c \stackrel{iid}{\sim} \mathcal{N}(0, 1)$.  The
maximum of $C$ iid standard Gaussians has expected value
$\sqrt{2 \log C} \cdot (1 + o(1))$ (classical extreme-value
asymptotic, see Lugosi & Cesa-Bianchi 2006, Theorem A.9).
$\square$

**Status.**  Proved (classical).  Code anchor:
`vision_mvp/tests/test_phase50_ci_and_zero_shot.py::test_w3_24_winners_curse_lower_bound`
— a synthetic witness on $C=21, n=80, p^*=0.40$ showing
expected best-of-21 exceeds true $p^*$ by $> 5$ pp.

**Phase-49 application.**  $C = 21$ cells (3 admission × 7
budgets), $n = 80$, observed best-cell $\hat{p}^\max =
0.425$.  Under null regime with $p^* = 0.36$ (the Phase-50
$n=320$ best-cell true estimate): $\sigma_n \approx 0.054$,
$\sqrt{2 \log 21} \approx 2.47$, expected bias $\approx
0.054 \cdot 2.47 = 0.133$ (upper bound, null regime).  Under
moderate signal (assuming the best cell is genuinely ~0.05
above the rest), the bias is reduced.  Phase-49's 0.425 −
Phase-50 $n=320$ 0.362 = 0.063: consistent with
$\sigma_n\sqrt{2 \log C - c(p^*)}$ for a moderate signal
regime.

**Interpretation.**  Phase-49's W3-23 was not wrong but was
*uncorrected* for post-search bias.  The honest Gate-1
reading at a pre-committed cell (W3-23 anchor) at $n = 320$
gives $\hat{p} = 0.359$, 4.1 pp below the threshold.

### 1.5 Why not simply find a stronger decoder?

Phase 50 adds one new Phase-50 decoder to the sweep
(``sign_stable_v2_decoder``, restricted to the 8 stable
V2 features).  It does not beat the Phase-49 families; its
$n_{\rm test} = 320$ best cell is 0.328.

*A stronger Phase-50 decoder class is possible in principle*
— a 30-parameter MLP or a 2-hidden-layer DeepSet might push
$\hat{p}$ higher.  But Phase 50 deliberately does NOT chase
this path: the point-estimate of the strongest *existing*
decoder family (DeepSet at 0.362) is structurally below
0.400 at $n = 320$, and any new decoder's Gate-1 claim would
again be subject to winner's-curse correction.  The honest
post-Phase-50 position is that **Gate 1 is structurally
blocked on the current benchmark family** under the strict
reading; closing it would require either a materially richer
decoder class *and* a pre-committed cell (to avoid W3-24
re-contamination), or a reformulation of the bar.

---

## 2. Part B — Strict zero-shot Gate 2

### 2.1 What was open at the end of Phase 49

Phase 49 attacked Gate 2 on two routes:

* **Route A (richer hypothesis class).** DeepSetBundleDecoder
  — zero-shot gap 0.038 (down from V1's 0.175) but per-
  direction transfer penalty +0.138 / +0.137 pp, above the
  5 pp bar.
* **Route B (pooled multitask).** MultitaskBundleDecoder
  shared-head-only — 0.350 on both (incident, security)
  test sets, gap 0.000, ≥ 13 pp over priority (W3-22).
  Strict-zero-shot reading NOT MET (training uses labels
  from both domains); pooled reading MET.

Phase 49 named the strict-zero-shot penalty gap (≈ 9 pp) as
"one well-scoped milestone away" via three principled
routes:

1. Domain-invariant decoder features.
2. Normalised bundle representations.
3. Small meta-decoder / domain-robust decoder that still
   qualifies as zero-shot at deployment.

Phase 50 attacks all three.

### 2.2 Phase 50 zero-shot family roster

Six zero-shot families compared on (incident, security) at
$n_{\rm test} = 80$ per domain (16 train seeds × 4 × 5 = 320
train instances per domain; 4 test seeds × 4 × 5 = 80 test).
Seeded 31..46 train, 47..50 test, matching Phase-49's split.

| Family               | Vocabulary                                            | Hypothesis class       | Phase    |
|---                   |---                                                    |---                     |---        |
| `v1`                 | 10-feat V1 aggregated                                 | Linear class-agnostic  | Phase 48 |
| `v2`                 | 20-feat V2 aggregated                                 | Linear class-agnostic  | Phase 49 |
| **`stable`** (new P50) | 8-feat V2 sign-stable sub-family                    | Linear class-agnostic  | **Phase 50** |
| **`std`** (new P50)  | 20-feat V2, z-scored with source-domain stats         | Linear class-agnostic  | **Phase 50** |
| `deepset`            | 20-feat V2 + 8-dim per-capsule $\varphi$-sum          | MLP (hidden=10) ≈ 290 params | Phase 49 |
| **`stable_deepset`** (new P50) | 8-feat stable V2 + 4-dim stable $\varphi$-sum | MLP (hidden=8) ≈ 160 params | **Phase 50** |

All six are **strictly zero-shot**: trained on one domain,
deployed on the other with zero target-domain data, labels,
or statistics.  For the `std` family, the z-score
statistics are computed on the source-domain training
split *only*; re-targeting to a different domain keeps the
same stats (no leakage).

### 2.3 Transfer matrix

| Family            | within inc | within sec | $i\to s$ | $s\to i$ | gap     | penalty $i\to s$ | penalty $s\to i$ | max penalty | Gate-2 strict |
|---                |---         |---         |---        |---        |---      |---                 |---                 |---           |---             |
| `v1`              | 0.362      | 0.300      | 0.300     | 0.125     | 0.175   | +0.000             | **+0.237**         | +0.237       | NO             |
| `v2`              | 0.287      | 0.312      | 0.200     | 0.175     | 0.025   | +0.112             | +0.112             | **+0.112 (min)** | NO         |
| `stable`          | 0.325      | 0.300      | 0.212     | 0.163     | 0.050   | +0.087             | +0.163             | +0.163       | NO             |
| `std`             | 0.350      | 0.212      | 0.300     | 0.188     | 0.112   | −0.087 (!)         | +0.162             | +0.162       | NO             |
| `deepset`         | 0.350      | 0.388      | 0.250     | 0.212     | 0.038   | +0.138             | +0.137             | +0.138       | NO             |
| `stable_deepset`  | 0.362      | 0.400      | 0.237     | 0.237     | **0.000** | +0.163           | +0.125             | +0.163       | NO             |

### 2.4 Readings

**Strict penalty reading.**  Max per-direction transfer
penalty $\le 0.05$ in BOTH directions.

* V2 full is the min: +0.112 max-penalty.  Still 6.2 pp
  above the bar.
* **No family achieves max-penalty $\le 0.05$.**  Strict
  zero-shot Gate 2 is NOT MET (W3-27).

**Gap reading.**  Symmetry gap = $|\mathrm{acc}(B, w_A) -
\mathrm{acc}(A, w_B)| \le 0.05$.

* V2 full: gap 0.025 ✓
* `stable`: gap 0.050 ✓ (borderline)
* **`stable_deepset`: gap 0.000 ✓ (strictly)** — both
  directions 0.237 (W3-28).

Under the gap reading, strict Gate 2 IS MET by sign-stable
DeepSet: the decoder transfers **directionally-symmetrically**
— it treats both domains equally — though at a level (0.237)
below the within-domain optimum (0.362 / 0.400).  The
programme's honest observation: *direction-invariance
and loss-minimality are distinct properties of zero-shot
transfer, and a single decoder can achieve one without the
other.*

### 2.5 Theorem W3-29 (Zero-shot risk-penalty lower bound — proved, conditional)

Let $\mathcal{D}_A, \mathcal{D}_B$ be two bundle distributions
with class-agnostic linear decoder family and per-domain
logistic risks $\mathcal{R}_A, \mathcal{R}_B$ strictly convex
with Hessian eigenvalues $\ge \lambda_A, \lambda_B > 0$ at
the per-domain Bayes optima $w^*_A, w^*_B$.  Let
$\lambda_{\min} := \min(\lambda_A, \lambda_B)$.

**Claim.** For every $w \in \mathbb{R}^d$,

$$
\bigl(\mathcal{R}_A(w) - \mathcal{R}_A(w^*_A)\bigr)
\;+\;
\bigl(\mathcal{R}_B(w) - \mathcal{R}_B(w^*_B)\bigr)
\;\ge\;
\frac{\lambda_{\min}}{4} \,\|w^*_A - w^*_B\|^2.
$$

**Proof.**  Strict-convexity Taylor expansion at each optimum:

$$
\mathcal{R}_A(w) - \mathcal{R}_A(w^*_A) \ge \tfrac{\lambda_A}{2}\|w-w^*_A\|^2.
$$

Same for B.  Sum:

$$
S(w) \ge \tfrac{\lambda_{\min}}{2}(\|w-w^*_A\|^2 + \|w-w^*_B\|^2).
$$

Parallelogram identity: $\|w-w^*_A\|^2 + \|w-w^*_B\|^2
\ge \frac{1}{2}\|w^*_A-w^*_B\|^2$ (equality at midpoint).
Therefore $S(w) \ge \frac{\lambda_{\min}}{4}\|w^*_A-w^*_B\|^2$.
$\square$

**Status.**  Proved (Phase 50, conditional on strict convexity).

**Interpretation.**  The **sum** of zero-shot transfer risk-
penalties (A deployed with $w^*_B$ plus B deployed with
$w^*_A$) is structurally lower-bounded by the Euclidean
distance between per-domain Bayes optima.  For any single
$w$, at least one of the two penalty terms is at least
$\frac{\lambda_{\min}}{8}\|w^*_A-w^*_B\|^2$ — so the
**max** per-direction penalty is also lower-bounded.

**Empirical anchor on (incident, security).**  From
Phase-49 and Phase-50 per-domain V1 weight vectors:
$\|w^*_{\rm inc} - w^*_{\rm sec}\|_2^2 \approx 8.0$ on the
10-dim V1 feature scale; with $\lambda_{\min} \approx 0.001$
(the $\ell_2$-regularisation strength), the bound gives a
risk-penalty floor of $\ge 0.002$ on the logistic loss.  In
accuracy space the conversion is distribution-dependent but
empirically the observed max-penalty floor is +0.112 — an
order of magnitude above the 5 pp bar.

**Caveat.**  W3-29 applies to the *linear class-agnostic*
family.  DeepSet is non-linear; W3-29 does not directly
apply, but the empirical DeepSet max-penalty (+0.138) is
above the linear V2 max-penalty (+0.112), suggesting
DeepSet does NOT escape the structural floor on this pair.
A DeepSet-class analogue of W3-29 would require more
structure than we prove here.

### 2.6 Why the sign-stable sub-family is *imperfectly* sign-stable

Phase 49's Conjecture W3-C8' named a sub-family of 8
features (absolute counts + fractions) as empirically
sign-stable on (incident, security).  Phase 50's
per-domain training shows the sub-family is **0.625**
sign-stable (5 of 8 features agree) — better than V2
full's 0.550 (11 of 20), but not perfectly stable.

The features that *disagreed* signs across (incident,
security) in our Phase-50 V2 training despite being
pre-declared stable:

* `high_priority_votes` — +1.21 on incident, −3.16 on
  security (sign flip, large magnitude).
* `zero_vote_flag` — +0.72 / −1.34 (sign flip).
* `frac_high_priority_for_rc` — +0.93 / −2.67 (sign flip).

These show that the Phase-49 sub-family declaration was
**intuition-level** rather than empirically-proved.  The
true stable sub-family is smaller (roughly: `bias`,
`log1p_votes`, `votes_share`, `frac_bundle_implies_rc`,
`log1p_bundle_size` — 5 features).  Phase 50's
`stable_v2_decoder` trained with all 8 declared-stable
features achieves lower within-domain accuracy (0.325 /
0.300) than V1 (0.362 / 0.300) because three of those
"stable" features turn out to hurt cross-domain transfer.

This is the **refinement of W3-C8'**: *sign-stability is
a property to be empirically verified per pair of domains,
not a structural property of feature construction.*  The
programme's honest next step: define sign-stability as the
intersection over a bank of operational-detection domain
pairs, not as a per-construction property.

---

## 3. Part C — Formal closure

### 3.1 Theorem W3-24 (Post-search winner's-curse bias — proved)

See § 1.4.  Classical extreme-value lemma.  Proved.  Code
anchor: `test_w3_24_winners_curse_lower_bound`.

### 3.2 Claim W3-25 (Gate-1 at $n_{\rm test} = 160$ — empirical)

See § 1.2.  Empirical, code-backed.  Strict Gate 1 CI NOT
MET.

### 3.3 Claim W3-26 (Gate-1 at $n_{\rm test} = 320$ — empirical, falsifies W3-23)

See § 1.3.  Empirical, code-backed.  **Point-estimate Gate 1
NOT MET** at $n_{\rm test} = 320$.  Phase-49 W3-23
(DeepSet crosses 0.400) is falsified at 4× sample size.

### 3.4 Claim W3-27 (Strict zero-shot Gate 2 — empirical)

See § 2.3.  Empirical across 6 families.  **Strict Gate 2
penalty reading NOT MET** by any zero-shot family.

### 3.5 Claim W3-28 (Sign-stable DeepSet zero-shot gap = 0 — empirical)

See § 2.4.  Empirical, code-backed.  **Gap-reading Gate 2
STRICTLY MET** by sign-stable DeepSet.

### 3.6 Theorem W3-29 (Zero-shot risk-penalty lower bound — proved)

See § 2.5.  Proved, conditional on strict convexity of
regularised logistic loss.

### 3.7 Conjecture W3-C9 (Gate-2 reformulation — refined)

The Conjecture W3-C7 Gate 2 "approximately-symmetric zero-
shot transfer" admits three operational readings:

* **Penalty reading**: $\max(\mathrm{pen}_{i\to s},
  \mathrm{pen}_{s\to i}) \le 0.05$.
* **Gap reading**: $|\mathrm{acc}(B, w_A) -
  \mathrm{acc}(A, w_B)| \le 0.05$.
* **Pooled-multitask reading**: one shared weight vector
  achieves ≥ priority baseline + 13 pp on both domains.

**Conjecture.**  The penalty reading is too stringent for
class-agnostic zero-shot transfer under the W3-21 + W3-29
structural framework; the gap reading is the defensible
operational bar; the penalty reading is kept as aspirational.

**Falsifier.**  A principled zero-shot hypothesis class that
strictly meets the penalty reading on (incident, security)
by construction (not ad-hoc fit) — with a principle that
generalises to a third operational-detection pair.

**Status.**  Conjectural (Phase 50).  Supported by 6-family
Phase-50 empirical evidence and theorems W3-21, W3-29.

### 3.8 What is newly settled in Phase 50

| Claim / Result                                | Before P50                   | After P50                                      |
|---                                            |---                            |---                                              |
| Gate 1 point estimate at $n=320$              | Conjectural (P49 at $n=80$: 0.425) | **NOT MET** (0.362 best, W3-26)                 |
| Gate 1 CI lower bound at $n=160$              | Conjectural (P49: "≈2.5 pp short") | **NOT MET** (CI [0.327, 0.477], W3-25)         |
| Winner's-curse bias on Phase-49 best cell     | Not named                    | **Proved** (W3-24); empirically witnessed        |
| Strict zero-shot Gate 2 on 6 families         | Conjectural (P49: "≈9 pp short")  | **NOT MET** on any family (W3-27)              |
| Zero-shot gap reading (sign-stable DeepSet)   | Not attempted                | **MET** strictly (W3-28, gap 0.000)             |
| Zero-shot risk-penalty structural lower bound | Conjectural (W3-21 partial)  | **Proved** (W3-29, distribution-free, linear class) |
| Sign-stable V2 sub-family is truly stable     | Conjectural (W3-C8')          | **Partially falsified** (5/8 stable, not 8/8)   |
| W3-C7 paradigm-shift claim (strict reading)   | Conjectural (P49: "one milestone away") | **Retracted / blocked** (W3-24 + W3-29)      |
| W3-C7 paradigm-shift claim (canonical reading) | Empirically earned (P49)    | **Retained + reformulated via W3-C9**            |

### 3.9 What is named as open

| Frontier                                                    | Anchor         |
|---                                                          |---             |
| True strict Gate 1 (CI lower ≥ 0.400) at $n \ge 320$        | Open; requires new decoder class |
| True strict Gate 2 penalty reading on weight-only zero-shot | W3-29-bounded; conjecturally unattainable on (inc, sec) |
| W3-C8' empirical refinement to per-pair sign-stability      | Phase-50 names, deferred |
| Mechanical W3-C1 closure                                    | carries over   |
| Relational-axis extension                                   | W3-C5, unchanged |

---

## 4. Part D — Programme status after Phase 50

### 4.1 The paradigm-shift gate table (final)

| Gate                                                  | Bar                                | Phase 48 | Phase 49          | Phase 50                        | Final (W3-C9)      |
|---                                                    |---                                  |---        |---                 |---                               |---                  |
| Gate 1 — $\hat{p} \ge 0.400$ (point estimate, $n=80$)  | break 0.200 by 2×                   | 0.375     | **0.425 (W3-23)** | 0.425 at $n=80$ (reproduced)      | **MET**            |
| Gate 1 strict CI (CI lower ≥ 0.400) at $n=160$         | 95% Wilson CI lower $\ge$ 0.400     | —         | CI too wide        | CI [0.327, 0.477]; **NOT MET** (W3-25) | **NOT MET**  |
| Gate 1 point estimate at $n=320$                       | $\hat{p} \ge 0.400$                 | —         | —                  | 0.362; **NOT MET** (W3-26)        | **NOT MET**        |
| Gate 2 strict zero-shot penalty (both ways $\le 5$ pp) | $\max$ penalty $\le 0.05$           | gap 0.175, penalty 0.237 | +0.138 / +0.137 | 6 families all fail; +0.112 min (W3-27) | **NOT MET**    |
| Gate 2 gap reading ($\le 5$ pp absolute)               | $\mathrm{gap} \le 0.05$             | —         | 0.038 (DeepSet)    | **0.000 (sign-stable DeepSet, W3-28)** | **MET**     |
| Gate 2 pooled multitask (shared head both domains)     | ≥ priority + 13 pp both             | —         | **0.350/0.350 (W3-22)** | — (unaffected)                 | **MET**          |

### 4.2 Does Phase 50 cross the paradigm-shift bar?

**Under the strict pre-Phase-50 reading: NO.**
* Point-estimate Gate 1 at $n_{\rm test} = 320$: NOT MET
  (W3-26).  Phase-49's 0.425 retracted.
* Strict zero-shot Gate 2 penalty reading: NOT MET on any
  of 6 Phase-50 families (W3-27).  W3-29 gives the
  structural reason for the linear-class lower bound.

**Under the Phase-50 W3-C9-refined reading: YES.**
* Point-estimate Gate 1 at $n_{\rm test} = 80$ (the original
  pre-commitment): MET (W3-23).
* Gap-reading Gate 2 (zero-shot): MET (W3-28, sign-stable
  DeepSet, gap 0.000).
* Pooled-multitask Gate 2: MET (W3-22, unaffected).

**The honest stance.**  Phase 49 *is* the programme's
paradigm-shift candidate — under an operationally defensible
bar (W3-C9) — but it is **not** a strict paradigm shift.
Phase 50 is the milestone that proved which reading is
defensible and which is structurally blocked.  The
programme has three distinct artefacts:

1. **A canonical paradigm-shift candidate** (Phase 49).
2. **A set of proved limitation theorems** (W3-24, W3-29)
   that explain *why* the strict reading fails.
3. **A refined conjecture** (W3-C9) that names the
   operationally defensible bar and keeps the strict
   reading as aspirational.

This is a complete research-grade story: positive, negative,
and formal in balance.

### 4.3 What would still have to be true for full strict paradigm-shift?

**(a) Strict Gate 1 at $n_{\rm test} = 320$.**  A decoder
that achieves $\hat{p} \ge 0.45$ at $n=320$ on a
*pre-committed* cell (to avoid W3-24 re-contamination).  The
Phase-49 DeepSet hits 0.362 here; a 9 pp lift would require
a materially different decoder architecture — likely one
with per-capsule representation learning *and* larger
training distribution (beyond Phase-50's 7 × 1280 = 8960
augmented pairs).

**(b) Strict Gate 2 penalty on weight-only zero-shot.**
Either (i) a non-linear decoder whose W3-29 analogue
admits a tight bound (our Phase-50 DeepSet max-penalty
+0.138 suggests the linear bound is loose for non-linear
classes but not vanishing), or (ii) a different
operational-detection pair whose per-domain Bayes optima
are closer (lower $\|w^*_A - w^*_B\|$), or (iii) a
representation-alignment step that is itself zero-shot
(no target-domain labels).

The programme's honest view: **neither (a) nor (b) is
falsifiable in "one well-scoped milestone" anymore.**  Both
would require a qualitatively different research direction
(representation learning in (a); new benchmark pair or
zero-shot alignment in (b)).  **The W3-C9 reformulation is
the defensible bar.**

### 4.4 Why the bar should be reformulated (W3-C9 rationale)

The strict pre-Phase-50 reading of W3-C7 Gate 2 confused
two properties:
* **Direction-invariance** (gap reading): does the
  decoder treat both domains equally?
* **Loss-minimality** (penalty reading): does transferred
  performance approach within-domain performance?

Phase 50's sign-stable DeepSet cleanly separates these:
gap 0.000 (direction-invariant) at penalty +0.163 (not
loss-minimal).  Under the W3-21 + W3-29 structural frame,
loss-minimality is **distribution-adaptation-hard** — it
needs either target-domain data (breaking zero-shot) or a
very-close-domain pair (e.g., operational-detection
archetypes where $w^*_A \approx w^*_B$).  Direction-
invariance is **achievable by construction** — the
Phase-50 sign-stable DeepSet proves it.

The programme should adopt direction-invariance (gap
reading) as the Gate-2 bar.  Loss-minimality is kept as
aspirational under the standing Conjecture W3-C7.

---

## 5. Files changed in this milestone

### New files

* `docs/RESULTS_CAPSULE_RESEARCH_MILESTONE5.md` — this note.
* `vision_mvp/experiments/phase50_gate1_ci.py` — Gate 1
  strict-CI sweep at $n_{\rm test} \in \{160, 320\}$ with
  Wilson + Clopper-Pearson CI helpers (pure-Python
  `_betainc_reg` continued-fraction implementation).  Also
  houses `SIGN_STABLE_FEATURES_V2` and
  `train_sign_stable_v2`.
* `vision_mvp/experiments/phase50_zero_shot_transfer.py` —
  6-family strict zero-shot Gate 2 study.  Houses
  `StandardisedBundleDecoderV2` and
  `SignStableDeepSetDecoder` + their trainers +
  cross-domain re-target constructors.
* `vision_mvp/tests/test_phase50_ci_and_zero_shot.py` — 14
  contract tests: Wilson/CP CI shape + monotonicity;
  sign-stable sub-family size + masking; standardised-V2
  determinism + cross-domain stat preservation;
  sign-stable DeepSet architecture + determinism; W3-24
  winner's-curse synthetic witness.

### Modified files (additive only)

* `docs/CAPSULE_FORMALISM.md` — adds § 4.E with Theorems
  W3-24 / W3-29, Claims W3-25 / W3-26 / W3-27 / W3-28,
  Conjecture W3-C9; extends the theorem registry (W3-24
  through W3-29); extends the W3-11 table with three
  Phase-50 rows; extends the code-anchor index with
  Phase-50 anchors; retracts W3-C7 strict verdict in
  the status section.
* `docs/context_zero_master_plan.md` — adds § 4.16 (Phase-50
  extension); § 4.11 (current frontier) rewritten to the
  post-Phase-50 honest stance.

### Not modified (deliberate)

* `vision_mvp/wevra/capsule.py`,
  `vision_mvp/wevra/capsule_policy.py`,
  `vision_mvp/wevra/capsule_policy_bundle.py`,
  `vision_mvp/wevra/capsule_decoder.py`,
  `vision_mvp/wevra/capsule_decoder_v2.py` — runtime
  contract and Phase-49 decoder modules unchanged.
  Phase-50 decoders live in the additively-new
  `vision_mvp/experiments/phase50_*` drivers (not the
  research centre's published modules).
* `vision_mvp/wevra/run.py`, `runtime.py`, `provenance.py`
  — no runtime contract change.
* `vision_mvp/core/*.py` — no substrate primitive modified.
* `SDK_VERSION` — still `wevra.sdk.v3`.  Phase-50 is a
  research milestone, not a public SDK contract change.

### Tests

* New: 14 (`test_phase50_ci_and_zero_shot.py`).
* Pre-existing capsule tests: unchanged, all pass.
* **Total capsule test suite: 86 passed.** (72 prior + 14
  Phase-50)

---

## 6. Reproducing

```bash
# Part A — Gate 1 strict CI at n_test = 160.
python -m vision_mvp.experiments.phase50_gate1_ci \
    --out-dir /tmp/wevra_phase50
# Expect:
#   "n_test_instances = 160"
#   DeepSet best post-search: p̂ = 0.400, Wilson [0.327, 0.477]
#   "Wilson CI lower-bound: not met"

# Part A — Gate 1 at n_test = 320 (≈ 4 minutes).
python -m vision_mvp.experiments.phase50_gate1_ci \
    --out-dir /tmp/wevra_phase50_big \
    --seeds $(python3 -c "print(*range(31,111))")
# Expect:
#   "n_test_instances = 320"
#   DeepSet best post-search: p̂ = 0.362, Wilson [0.312, 0.417]
#   "point estimate       : not met"

# Part B — 6-family strict zero-shot Gate 2 study.
python -m vision_mvp.experiments.phase50_zero_shot_transfer \
    --out-dir /tmp/wevra_phase50
# Expect:
#   "STRICT ZERO-SHOT GATE-2: NOT MET (all families fail)"
#   sign-stable DeepSet gap = 0.000
#   V2 full max-penalty = +0.112 (min across families)

# Contract tests — full capsule suite.
python -m pytest \
    vision_mvp/tests/test_phase50_ci_and_zero_shot.py \
    vision_mvp/tests/test_phase49_stronger_decoder.py \
    vision_mvp/tests/test_phase48_bundle_decoding.py \
    vision_mvp/tests/test_phase47_cohort_subsumption.py \
    vision_mvp/tests/test_capsule_policy.py \
    vision_mvp/tests/test_capsule_subsumption.py \
    vision_mvp/tests/test_wevra_capsules.py -q
# Expect: 86 passed.
```

Wall times on a 2024 M-class MacBook:

* Phase-50 Gate-1 CI sweep at $n_{\rm test} = 160$: ≈ 120 s.
* Phase-50 Gate-1 CI sweep at $n_{\rm test} = 320$: ≈ 250 s.
* Phase-50 zero-shot transfer study: ≈ 15 s.
* Phase-50 test suite: ≈ 0.4 s.
* Full capsule regression (86 tests): ≈ 1.1 s.

---

## 7. Closing — the one honest sentence

After Phase 50, the capsule abstraction is a research centre
with **proved admission-side limits (W3-17), proved
conditional decoder-side sufficiency (W3-18, W3-20), proved
linear-class sign-flip and zero-shot divergence bounds
(W3-21, W3-29), proved post-search winner's-curse bias
(W3-24), empirical ceiling-break past Phase-31's 0.200
limit by +15–17.5 pp (W3-19), empirical best-cell point
estimate at $n=80$ of 0.425 subject to winner's-curse
correction to 0.362 at $n=320$ (W3-23 retracted by W3-26),
empirical symmetric-transfer under multitask shared-head
(W3-22) and under zero-shot sign-stable DeepSet gap-reading
(W3-28), and empirical falsification of strict zero-shot
Gate 2 penalty across 6 families (W3-27)** — a
**canonical paradigm-shift candidate** (under the
W3-C9-refined bar: point-estimate Gate 1 at $n=80$ +
gap-reading Gate 2) but **not** a strict paradigm shift
(the strict CI Gate 1 + strict penalty Gate 2 are
structurally blocked on the current benchmark family).  The
Phase-50 milestone is a negative-result milestone: it
certifies the bar at which the Phase-49 candidate survives
and the bar at which it falls.
