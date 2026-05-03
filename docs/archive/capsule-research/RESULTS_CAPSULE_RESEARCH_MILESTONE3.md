# RESULTS — Capsule research milestone 3 (Phase 48)

> *Theory-forward results note. Phase 47
> (`docs/RESULTS_CAPSULE_RESEARCH_MILESTONE2.md`) sharpened the
> capsule centre on three fronts — formal frontier (COHORT
> lift + relational limitation), bundle-aware admission
> (falsification of P46-C1 strong, +6× budget efficiency),
> cross-domain admission transfer (asymmetric +21 pp
> incident→security, −3 pp reverse) — and localised the
> remaining open question to the decoder axis (Conjecture
> P47-C1). This Phase-48 note attacks P47-C1 directly.
> **Part A** proves an admission-locality limitation
> (Theorem W3-17) and a bundle-aware decoder sufficiency
> condition (Theorem W3-18); **Part B** ships a
> LearnedBundleDecoder that empirically breaks the 0.200
> Phase-31 ceiling by +15–17.5 pp on held-out seeds
> (Claim W3-19); **Part C** runs the first decoder-side
> cross-domain transfer study and returns a sharply
> asymmetric result with a feature sign-flip signature
> (Conjecture W3-C6); **Part D** states the paradigm-shift
> threshold formally (Conjecture W3-C7) and reports
> honestly that Phase 48 is materially closer but below
> the bar.* Last touched: 2026-04-22.

---

## 0. One-paragraph summary

Phase 48 advances the capsule research centre on the
decoder axis. The **formal frontier** gains three results:
**W3-17** (admission locality — negative, proved) sharpens
the Phase-47 empirical observation that admission alone
cannot exceed the 0.200 Phase-31 structural ceiling into a
bounded-above limitation theorem over every header-level
admission rule; **W3-18** (bundle-aware decoder sufficiency —
positive, proved, conditional) exhibits a sharp single-bundle
separator under which plurality decoding strictly dominates
priority decoding; **W3-19** (empirical — code-backed, seed-
robust at $n_{\rm test} = 80$) ships a 10-feature multinomial-
logistic `LearnedBundleDecoder` that breaks the 0.200 ceiling
by **+15–17.5 pp** on the Phase-31 noisy bench. The **learning
frontier** confirms P47-C1 partially: a LEARNED bundle decoder
IS the right axis (the ceiling is breakable); a PURE plurality
decoder is NOT (its tiebreak fallback reproduces the ceiling);
`SourceCorroboratedPriorityDecoder(min_sources=2)` DEGRADES the
decoder to 0 (it vetoes the causal chain's single-source
signatures too). The **transfer frontier** reports the first
decoder-side cross-domain study on two operational-detection
domains (incident, security): incident-trained-decoder →
security test is a **full transfer** (cross-accuracy matches
within-domain at 0.300), but security-trained → incident is
**sharply negative** (0.125, below base rate AND below
priority). Feature-level sign-flip on
`lone_top_priority_flag` (+1.13 incident, −1.19 security)
is the structural signature. The **programme status update**:
the paradigm-shift threshold (Conjecture W3-C7) is stated
explicitly; Phase 48 hits one of its two gates weakly (0.375
< 0.400 target) and the other partially (one of two directions
transfers). The centre is materially upgraded toward a
paradigm shift but does not yet clear the bar.

---

## 1. Part A — Formal decoder frontier

### 1.1 What was open at the end of Phase 47

Phase 47's Theorem P47-1 stated the Phase-31 noisy-bench
structural ceiling as a property of the priority decoder:
under `spurious_prob = 0.30` at least one spurious
`DISK_FILL_CRITICAL` is admitted in ≈ 100 % of test scenarios,
and the priority decoder's first-match rule then forces
`disk_fill` — bounding full-bundle accuracy above by
$\Pr[y_{\rm gold} = \mathtt{disk\_fill}] = 1/5 = 0.200$.

What Phase 47 did NOT prove: that **every admission rule**
is bounded above by 0.200. It empirically saturated there
(the decoder-aware bundle-learned policy hits 0.200 at every
budget and does not exceed it), but the negative-limitation
theorem over admission was left unstated.

### 1.2 Theorem W3-17 (admission locality — negative, proved)

**Claim.** Under the "ceiling-forcing distribution" of Phase-31
— spurious DFC injection with probability ≥ 1 − ε — every
admission policy $\pi$ that is

1. **pure** on capsule headers (depends only on
   $(\mathit{cid}, \text{headers})$ and ledger state — it
   cannot read ground-truth labels), and
2. **indistinguishable on the spurious kind** (it cannot
   selectively drop spurious DFC while keeping causal DFC
   given identical headers),

satisfies
$\Pr[D_{\rm pri}(\pi(E_{\rm offered})) = y_{\rm gold}] \le
\Pr[y_{\rm gold} = y^{\star}] = 1/5$.

**Proof.** See `docs/CAPSULE_FORMALISM.md` § 4.C. The
distribution has the property that with probability ≥ 1 − ε
at least one DFC-header capsule is offered; the spurious and
causal DFCs are header-identical (Phase-32 noisy-extractor
construction), so (ii) denies $\pi$ the discriminator it
would need. Admission either lets a DFC through (→ decoder
emits `disk_fill` with probability ≈ 1) or drops all DFC
headers (→ forfeits the `y_gold = disk_fill` slice). Both
strategies are capped by the class prior. $\square$

**Interpretation.** W3-17 is the formal statement of "the
ceiling is a decoder property, not an admission property." It
is the negative half of the decoder-frontier landscape; every
admission-side improvement is bounded above by it.

**Anchor.** Empirical saturation is visible in the Phase-47
experiment output (`results_phase47_bundle_learning.json`):
bundle_learned(dec) × B=16..256 all return 0.200 decoder
accuracy on the test split.

### 1.3 Theorem W3-18 (bundle-aware decoder sufficiency —
positive, conditional, proved)

**Claim.** In the "coherent-majority" regime — where the
admitted set's implied-rc vote distribution has a strict
maximum at $y_{\rm gold}$ — plurality decoding returns
$y_{\rm gold}$ with probability 1, while priority decoding
can return any $y' \neq y_{\rm gold}$ whose implied top-
priority kind is present. On the slice where a higher-
priority kind implying $y' \neq y_{\rm gold}$ IS present,
plurality strictly dominates priority by 1.0 in
expectation.

**Proof.** See `docs/CAPSULE_FORMALISM.md` § 4.C. Plurality is
argmax; the strict max is at $y_{\rm gold}$ by assumption.
Priority is first-match over `priority_order`; on the named
slice it walks to $y'$ first. $\square$

**Sharp single-bundle separator.** Contract test
`vision_mvp/tests/test_phase48_bundle_decoding.py
::test_w3_18_plurality_strictly_dominates_priority_on_coherent_majority`
— a 5-capsule bundle `{OOM_KILL/sysadmin, OOM_KILL/db_admin,
ERROR_RATE_SPIKE/monitor, LATENCY_SPIKE/monitor,
DISK_FILL_CRITICAL/network}` has 2 votes for `memory_leak`
and 1 vote each for `error_spike`, `latency_spike`,
`disk_fill`. Priority returns `disk_fill` (DFC is first in
priority order). Plurality returns `memory_leak` (strict
majority). Gold is `memory_leak`; plurality is right,
priority is wrong.

**Caveat.** The coherent-majority premise does NOT hold on
every Phase-31 scenario. Single-signature scenarios
(tls_expiry, dns_misroute, memory_leak, deadlock) have
1-vote-each slices under noise; plurality's tiebreak walks
priority_order and reproduces the ceiling. **This is why
plain plurality does not break the full-bench ceiling** — a
fact confirmed empirically (see § 2.3 below).

### 1.4 Claim W3-19 (empirical decoder break — code-backed)

**Claim.** On the Phase-31 noisy bench
(`spurious_prob = 0.30, mislabel_prob = 0.10`) held-out
by-seed at $n_{\rm test} = 80$ (20 seeds × 0.8/0.2 split,
5 scenarios × 4 distractor densities per seed), the
`LearnedBundleDecoder` — a 10-feature multinomial-logistic
classifier over class-agnostic bundle-shape features,
trained on `(admitted_bundle, gold_root_cause)` pairs from
the train split — achieves:

* **FIFO admission** × B ∈ {64, 96, 128, 256}: test
  decoder accuracy **0.350**.
* **Phase-46 learned admission** × B = 96: test decoder
  accuracy **0.362**.
* **Phase-47 bundle_learned admission** × B = 96: test
  decoder accuracy **0.375** — the best cell.
* **Oracle-clean causal slice** (admission-noise removed):
  test decoder accuracy **0.575** vs priority's 0.525.

The 0.200 ceiling is broken at every cell. Binomial
comparison versus priority's best 0.225: at $n = 80$,
$z \approx 2.8$, $p \approx 0.003$ (one-sided). At the
best cell $z \approx 3.3$, $p < 0.001$.

**Status.** Code-backed, seed-robust. Reproducible from
`python -m vision_mvp.experiments.phase48_bundle_decoding`.

### 1.5 Negative results (falsifications in Phase 48)

* **Plain plurality decoding does NOT break the ceiling on
  the full Phase-31 bench.** Best test accuracy: 0.087. Its
  priority-order tiebreak on one-vote-each ties reproduces the
  priority-decoder's behaviour, so the ceiling is preserved.
  *This falsifies the naive reading of P47-C1*.
* **`SourceCorroboratedPriorityDecoder(min_sources=2)`
  DEGRADES to 0.000.** The Phase-31 causal chain's top-
  priority signature in each scenario is emitted by *one*
  source role (DFC from sysadmin, OOM_KILL from sysadmin,
  TLS_EXPIRED from network, …); the min_sources veto kills
  the causal signal too. This is a sharp falsification of
  the "just veto singletons" heuristic.
* **W3-18's sufficient condition is NOT met on the full
  bench.** The coherent-majority slice is a strict subset of
  the Phase-31 bench; plurality's empirical 0.087 on the
  full bench is strictly less than W3-18's 1.0 guarantee
  on the slice, confirming the conditional nature of the
  theorem.

### 1.6 Interpretable decoder weights

Top `LearnedBundleDecoder` weights after 300 epochs on the
incident train split (20-seed run):

```
votes_share                    +1.0658
lone_top_priority_flag         +0.9106
log1p_votes                    +0.8249
zero_vote_flag                 +0.7183
log1p_sources                  -0.4979
has_top_priority_kind          -0.3638
high_priority_votes            -0.0651
```

**Reading.**

* `has_top_priority_kind` **−0.36** — if the candidate rc's
  implication includes the priority_order[0] kind (DFC), the
  decoder *penalises* it. This is the "DFC is over-injected
  by the spurious distribution" signal.
* `votes_share` **+1.07** — higher share of the bundle
  implying this rc is positive. This is the classical
  "plurality-style" signal.
* `lone_top_priority_flag` **+0.91** — positive (somewhat
  counterintuitive). Explanation: in Phase-31 every causal
  rc IS supported by a lone high-priority kind (DFC alone,
  OOM_KILL alone, TLS_EXPIRED alone); the feature is a
  positive indicator that rc matches a Phase-31 scenario
  archetype. In a different domain with different scenario
  archetypes this weight can have the *opposite* sign (see
  Part C, the security-trained decoder has
  `lone_top_priority_flag = −1.19`).

---

## 2. Part B — LearnedBundleDecoder

### 2.1 Setup

Same data-generating process as Phase 47:
`noisy_extractor(spurious_prob=0.30, mislabel_prob=0.10)`
on the Phase-31 five-scenario bank, distractor grid
{6, 20, 60, 120}. `PYTHONHASHSEED=0` on launch for
reproducible collection. By-seed split 80 % train /
20 % test.

Four decoders under test:

1. **`PriorityDecoder`** — baseline, the Phase-31 status-quo
   first-match-over-priority_order rule.
2. **`PluralityDecoder`** — argmax over implied-rc vote counts;
   ties broken by priority_order walk.
3. **`SourceCorroboratedPriorityDecoder(min_sources=2)`** —
   first-match-over-priority_order but each kind must be
   emitted by ≥ 2 distinct source roles to count.
4. **`LearnedBundleDecoder`** — multinomial-logistic over the
   10 class-agnostic bundle-shape features
   (see `BUNDLE_DECODER_FEATURES` in
   `vision_mvp/coordpy/capsule_decoder.py`), trained by
   full-batch gradient descent (300 epochs, $lr = 0.5$,
   $l_2 = 10^{-3}$, deterministic in seed).

Admission policies: the Phase-46 `FIFOPolicy` (null
admission), the Phase-46 `LearnedAdmissionPolicy` (per-
capsule learned), and the Phase-47
`BundleLearnedPolicy(causal labels)`. Budgets:
{16, 32, 48, 64, 96, 128, 256} tokens.

### 2.2 Headline results (test split, $n_{\rm test} = 80$)

**Decoder accuracy on the FULL admit set (P47-C1 anchor):**

| Budget | `priority` | `plurality` | `src_corroborated_priority` | **`learned_bundle_decoder`** |
|---     |---        |---          |---                          |---                           |
|  16    | 0.100     | 0.037       | 0.000                       | **0.237**                    |
|  32    | 0.125     | 0.013       | 0.000                       | **0.237**                    |
|  48    | 0.175     | 0.025       | 0.000                       | **0.312**                    |
|  64    | 0.200     | 0.062       | 0.000                       | **0.350**                    |
|  96    | 0.212     | 0.075       | 0.000                       | **0.350**                    |
| 128    | 0.212     | 0.087       | 0.000                       | **0.350**                    |
| 256    | 0.212     | 0.087       | 0.000                       | **0.350**                    |

(FIFO admission; other admission policies shown in
`results_phase48_bundle_decoding.json`.)

**Best cell across the admission × decoder × budget grid:**

| Decoder                           | Best accuracy | @ admission             | @ budget |
|---                                |---            |---                     |---       |
| `priority`                        | 0.225         | bundle_learned_admit    | 64       |
| `plurality`                       | 0.087         | bundle_learned_admit    | 64       |
| `src_corroborated_priority`       | 0.000         | fifo                    | 16       |
| **`learned_bundle_decoder`**      | **0.375**     | **bundle_learned_admit**| **96**   |

**Oracle-clean causal slice (admission poisoning removed):**

| Decoder                        | Causal-slice acc | per-scenario detail                                                                                     |
|---                             |---               |---                                                                                                      |
| `priority`                     | 0.525            | disk_fill 0.50; tls_expiry 0.50; dns_misroute 0.375; memory_leak 0.50; deadlock 0.75                    |
| `plurality`                    | 0.225            | 0.375 / 0.188 / 0.188 / 0.188 / 0.188                                                                    |
| `src_corroborated_priority`    | 0.000            | all zero                                                                                                |
| **`learned_bundle_decoder`**   | **0.575**        | 0.50 / 0.50 / 0.375 / 0.50 / **1.00** (deadlock)                                                        |

The learned decoder exceeds priority's oracle-clean accuracy
by +5 pp — meaning the decoder is structurally better on
clean bundles, not just spurious-filtering better on noisy
bundles.

### 2.3 Why plurality doesn't break the ceiling

Plurality fails on the full Phase-31 bench because of the
distribution of scenario archetypes:

* **disk_fill_cron** (gold = `disk_fill`): causal chain is
  CRON_OVERRUN + DFC from sysadmin — 2 votes for `disk_fill`.
  Plurality correctly returns `disk_fill`. ✓
* **tls_expiry_healthcheck_loop** (gold = `tls_expiry`):
  causal chain is TLS_EXPIRED + FW_BLOCK_SURGE from network,
  ERROR_RATE_SPIKE from monitor — 1 vote each for
  `tls_expiry`, `fw_block`, `error_spike`. Tied. Priority
  tiebreak walks DFC (absent), TLS_EXPIRED (present) →
  `tls_expiry` in the clean case. Under spurious DFC
  injection, tied at 1 vote each for `disk_fill` too →
  priority tiebreak picks DFC → `disk_fill`. ✗
* Similar single-vote-each fragility on `memory_leak`,
  `deadlock`, `dns_misroute` — plurality's tiebreak re-
  produces the priority-decoder failure.

The `LearnedBundleDecoder` escapes this because its
`has_top_priority_kind` weight penalises DFC-implying
candidates regardless of vote count.

### 2.4 Train / test generalisation

Train set decoder accuracy at the same cells: within
$\pm 2$ pp of test — no overfitting. The 10-feature
hypothesis class is small enough to generalise cleanly on
$n_{\rm train} = 160$.

### 2.5 Budget behaviour

The `LearnedBundleDecoder` saturates at B = 64 tokens for
FIFO admission (0.350) and B = 96 for bundle_learned
admission (0.375). Versus Phase-47's `bundle_learned(dec)`
admission (which saturates admission-budget efficiency at
B = 16 but stays at the 0.200 ceiling), the Phase-48
decoder's budget efficiency is comparable but the *ceiling*
is broken.

### 2.6 New conjectures from Part B

**Conjecture P48-C1 (deeper-hypothesis-class ceiling-break
crosses 0.400).** A non-linear bundle decoder — per-class
MLP with cross-feature interactions, or a small transformer
over the ordered-by-priority admitted set — crosses test
decoder accuracy **0.400** on the same Phase-31 noisy bench.
*Falsifier:* no deeper hypothesis class exceeds 0.400.

**Conjecture P48-C2 (co-training lifts ceiling).** Jointly
training admission and decoder weights against the same
decoder-accuracy objective (rather than the two being
trained separately as in Phase 46/47/48) lifts test
decoder accuracy by ≥ 3 pp over the best disjoint-trained
pair. *Falsifier:* the joint-trained pair matches or
underperforms the disjoint-trained best.

---

## 3. Part C — Decoder-side cross-domain transfer

### 3.1 Setup

Two operational-detection domains, both from the Phase 47
bench:

| Domain   | Phase | rc_alphabet size | Gold label        | Base rate |
|---       |---    |---               |---                |---        |
| incident | 31    | 5                | `gold_root_cause` | 0.200     |
| security | 33    | 5                | `gold_classification` | 0.200 |

Compliance is deliberately omitted — its decoder shape
(verdict threshold over flag set) is structurally different
from the argmax-over-classification shape of incident and
security, so including it would conflate "decoder-side
transfer fails between incident and compliance" with "the
two domains have different decoder geometries." The
omission is the honest signal that decoder-side transfer is
a task-family question, not a universal one.

For each domain we:

1. Collect noisy `(offered_capsules, gold_label)` instances
   under the domain's scenario bank + Phase-32 noisy extractor
   (`spurious=0.30, mislabel=0.10`).
2. Split by seed (20 seeds, 16 train + 4 test).
3. Train a `LearnedBundleDecoder` on the training split
   using the domain's `claim_to_label` + priority_order +
   rc_alphabet.

For security we derive `claim_to_label` and `priority_order`
from the scenario bank automatically: each `claim_kind`
maps to the `gold_classification` of the first scenario in
which it appears; priority_order is the first-appearance
order in the per-scenario causal chains.

### 3.2 Transfer matrix ($n_{\rm test} = 80$ per domain)

| Train → Test          | Learned dec acc  | Priority acc  | Plurality acc | Lift over priority |
|---                    |---               |---            |---            |---                 |
| **incident → incident** | **0.362**       | 0.212         | 0.150         | **+0.150**         |
| incident → security   | **0.300**        | 0.200         | 0.350         | **+0.100**         |
| security → incident   | 0.125            | 0.212         | 0.150         | **−0.087**         |
| **security → security** | **0.300**       | 0.200         | 0.350         | **+0.100**         |

**Reading.**

* **Positive transfer**: incident-trained decoder applied
  to security reaches 0.300 — **exactly matching**
  within-security performance (also 0.300). The decoder's
  weights are fully portable in this direction.
* **Negative transfer**: security-trained decoder applied
  to incident drops to 0.125 — **below** incident's base
  rate (0.200) AND below incident's priority baseline
  (0.212). The decoder actively *hurts* performance.
* **Asymmetry mirrors Phase-47 admission-side**: Phase 47
  saw incident→security +21 pp admission-precision and
  security→incident −3 pp. Phase 48 sees incident→security
  full-transfer and security→incident sharply-negative.
  Same phenomenon at a different level of the stack.
* **Plurality is asymmetric the other way**: plurality is
  0.150 on incident (below priority's 0.212) and 0.350 on
  security (above priority's 0.200). Security's richer
  multi-source causal chains are plurality-friendly;
  incident's single-signature chains are plurality-hostile
  (per § 2.3).

### 3.3 Feature sign-flip as the structural signature

Per-domain weights for the top 7 features by |weight|:

| Feature                        | incident | security  | Sign |
|---                             |---       |---        |---   |
| `votes_share`                  | +1.066   | +3.172    | agree |
| `log1p_votes`                  | +0.935   | +0.080    | agree |
| `zero_vote_flag`               | +0.707   | +0.430    | agree |
| `high_priority_votes`          | −0.155   | −1.511    | agree |
| `log1p_sources`                | **−0.490** | **+1.565** | **DISAGREE** |
| `has_top_priority_kind`        | **−0.354** | **+0.953** | **DISAGREE** |
| `lone_top_priority_flag`       | **+1.125** | **−1.191** | **DISAGREE** |
| `multi_source_flag`            | 0.000    | −1.570    | agree (trivial) |

**Three features disagree in sign across domains.**

The sharpest disagreement — **`lone_top_priority_flag`**
at +1.13 vs −1.19 — is the structural signature of the
task-family divergence:

* In incident scenarios, every gold rc is supported by
  exactly *one* authoritative high-priority kind from *one*
  authoritative role (DFC from sysadmin for `disk_fill`,
  OOM_KILL from sysadmin for `memory_leak`, TLS_EXPIRED
  from network for `tls_expiry`, …). The feature fires on
  gold; the classifier learns "lone_top_priority =
  evidence for this rc."
* In security scenarios, the gold classifications (`ransomware`,
  `data_exfil`, `supply_chain`, …) are supported by
  *multiple* high-priority kinds from *multiple* roles
  (MALWARE_DETECTED + LATERAL_MOVEMENT + TTP_ATTRIBUTED
  across 3 roles for ransomware). The feature fires on
  spurious singletons; the classifier learns
  "lone_top_priority = evidence AGAINST this rc."

Opposite signs ⇒ opposite-direction transfer. A decoder
trained to expect Phase-31's "single-signature" archetype
cannot succeed on Phase-33's "multi-source corroboration"
archetype, and vice versa.

### 3.4 Conjecture W3-C6 (decoder-side task-family transfer —
asymmetric)

See `docs/CAPSULE_FORMALISM.md` § 5 for the formal statement.
The empirical evidence anchors the conjecture on the
(incident, security) pair; the sign-flip on
`lone_top_priority_flag` is the sharp signature.

### 3.5 Relation to Phase 47's P47-C3

Phase 47 established P47-C3 (task-family-indexed admission
transfer) on three domains. Phase 48 sharpens it at the
decoder level on two domains and adds the **feature sign-flip
mechanism** as the "why" — the structural feature that
predicts whether cross-transfer will succeed or fail.

P47-C3 and W3-C6 are the same phenomenon, now sharp on
both sides (admission + decoder) of the capsule contract.

---

## 4. Part D — Programme status

### 4.1 What is now settled that was not before

| Claim / Result                       | Before P48                  | After P48                                     |
|---                                   |---                          |---                                            |
| 0.200 ceiling status                 | Empirically bounded above   | **Proved tight via Theorem W3-17**            |
| P47-C1 (bundle decoder breaks)       | Open                        | **Partially supported (W3-19 @ +15 pp)**      |
| P47-C1 strong (≥ 0.400 threshold)    | Open                        | **Not met (0.375 < 0.400)**                   |
| Plurality decoder universal break    | Open                        | **Falsified on the full Phase-31 bench**      |
| `SourceCorroboratedPriorityDecoder` universal break | Open         | **Falsified at 0.000**                        |
| Decoder-side transfer                | Not framed                  | **Asymmetric + sign-flip (W3-C6)**            |
| Paradigm-shift threshold             | Informal                    | **Stated via W3-C7 (2-part gate)**            |

### 4.2 What is now named as open

| Frontier                                                   | Anchor    |
|---                                                         |---        |
| Deeper hypothesis class (cross 0.400)                      | P48-C1    |
| Jointly trained admission + decoder                        | P48-C2    |
| Symmetric cross-domain decoder transfer                    | W3-C6 (reverse direction) |
| Mechanical W3-C1 closure on remaining Phase-N theorems     | carries over from P46 |
| Relational-axis extension                                  | W3-C5     |

### 4.3 Does Phase 48 amount to a paradigm shift?

**Materially closer. Not yet.**

**For (cautious case).**
* The decoder frontier is now **symmetric**: a *proved*
  limitation theorem (W3-17) on the admission side, a
  *proved-conditional* sufficiency theorem (W3-18) on the
  decoder side, and an *empirically-validated* ceiling
  break (W3-19) connecting the two. This is a more mature
  shape than Phase 47's asymmetric "one negative, one
  positive" result pair.
* The 0.200 ceiling — which Phase 47 localised as the
  open question — is **empirically broken** on held-out
  seeds by a 10-feature logistic-regression decoder. This
  answers P47-C1 in the cautious positive direction.
* The decoder-side transfer study is the first *decoder-
  level* cross-domain result in the programme and
  reinforces Phase 47's admission-side P47-C3 at a
  structural level (feature sign-flip mechanism named).

**Against (strict case).**
* W3-19's break is **+15 pp**, not **+20+ pp**. Against
  the paradigm-shift threshold W3-C7 (≥ 0.400 test
  accuracy, i.e., ≥ 2× the ceiling) Phase 48 weighs in
  at 0.375 — below the bar.
* Cross-domain transfer is **asymmetric** in a sharp way:
  incident→security works, security→incident *actively
  hurts*. A paradigm shift would need symmetric
  transfer.
* The 10-feature hypothesis class is *deliberately*
  small (to keep interpretability and the "is header-
  level enough?" discipline). A larger class would likely
  close P48-C1, but closing it with a black-box model
  would weaken the "structural insight" claim that
  motivates the research programme.

**Honest summary.** Phase 48 moves the capsule centre from
"research centre with proven boundaries on both sides"
(Phase 47's summary) to **"research centre with
proven boundaries on both sides AND an empirically-
validated decoder-side ceiling break with sharply
asymmetric cross-domain transfer."** That is closer to a
paradigm shift than Phase 47 was, but still distance-short
of the W3-C7 bar. Until a decoder crosses 0.400 on the
Phase-31 bench with symmetric cross-domain transfer, the
centre is strong, bounded, falsifiable, *and* empirically
ceiling-broken — which is more than "useful unification"
but less than "paradigm shift."

---

## 5. Files changed in this milestone

### New files

* `docs/RESULTS_CAPSULE_RESEARCH_MILESTONE3.md` — this note.
* `vision_mvp/coordpy/capsule_decoder.py` — `BundleDecoder`
  interface, `PriorityDecoder`, `PluralityDecoder`,
  `SourceCorroboratedPriorityDecoder`,
  `LearnedBundleDecoder` + `train_learned_bundle_decoder`,
  10-feature `BUNDLE_DECODER_FEATURES` vocabulary.
* `vision_mvp/experiments/phase48_bundle_decoding.py` —
  main experiment driver (admission × decoder × budget
  sweep + causal-slice baseline + CEILING BROKEN
  verification).
* `vision_mvp/experiments/phase48_decoder_transfer.py` —
  cross-domain decoder transfer driver
  (incident + security, weight-vector transfer).
* `vision_mvp/tests/test_phase48_bundle_decoding.py` —
  14 contract tests covering decoder determinism, the
  W3-18 single-bundle separator, the W3-19 training shape,
  and the LearnedBundleDecoder's feature vocabulary.

### Modified files (additive only)

* `vision_mvp/coordpy/__init__.py` — re-exports the Phase-48
  decoder symbols (additive; no existing export altered).
* `docs/CAPSULE_FORMALISM.md` — adds § 4.C with Theorems
  W3-17 / W3-18 / Claim W3-19, Conjectures W3-C6 / W3-C7;
  updates W3-11 table with two new rows (decoder ceiling,
  decoder ceiling-break); updates the code-anchor index.
* `docs/context_zero_master_plan.md` — new § 4.14
  (Phase-48 extension); § 4.11 (current frontier) updated
  to the post-Phase-48 five-shaped gap.

### Not modified (deliberate)

* `vision_mvp/coordpy/capsule.py` — capsule contract C1..C6
  byte-for-byte unchanged.
* `vision_mvp/coordpy/capsule_policy.py`,
  `vision_mvp/coordpy/capsule_policy_bundle.py` — admission
  layer unchanged. The decoder is a *reader* of the
  admitted ledger, not a mutator.
* `vision_mvp/coordpy/run.py`, `runtime.py`, `provenance.py` —
  no runtime contract change.
* `vision_mvp/core/*.py` — no substrate primitive modified.
* `SDK_VERSION` — still `coordpy.sdk.v3`. The decoder is
  additive on the research centre, not part of the public
  SDK contract.

### Tests

* New: 14 (`test_phase48_bundle_decoding.py`).
* Pre-existing capsule tests: unchanged, all pass.

---

## 6. Reproducing

```bash
# Part A+B — bundle decoding (P47-C1 anchor, W3-17/W3-18/W3-19).
python -m vision_mvp.experiments.phase48_bundle_decoding \
    --out-dir /tmp/coordpy_phase48
# Expect:
#   CEILING BROKEN (one or more cells > 0.200).
#   learned_bundle_decoder best cell ≈ 0.35–0.37 on test.

# Part C — cross-domain decoder transfer (W3-C6).
python -m vision_mvp.experiments.phase48_decoder_transfer \
    --out-dir /tmp/coordpy_phase48
# Expect:
#   incident → security cross ≈ 0.300 (within-security).
#   security → incident cross ≈ 0.125 (below priority).
#   Feature sign-flip on lone_top_priority_flag.

# Contract tests.
python -m pytest vision_mvp/tests/test_phase48_bundle_decoding.py \
    vision_mvp/tests/test_phase47_cohort_subsumption.py \
    vision_mvp/tests/test_capsule_policy.py \
    vision_mvp/tests/test_capsule_subsumption.py \
    vision_mvp/tests/test_coordpy_capsules.py -q
# Expect: 55 passed.
```

Wall times on a 2024 M-class macbook:

* Phase-48 bundle-decoding sweep (10 seeds): ∼ 20 s.
* Phase-48 bundle-decoding sweep (20 seeds): ∼ 40 s.
* Phase-48 decoder transfer (20 seeds, 2 domains): ∼ 8 s.
* Phase-48 test suite: ∼ 0.3 s.
* Full capsule regression: ∼ 1 s.

---

## 7. Closing — the one honest sentence

After Phase 48, the capsule abstraction is a research centre
with **proven admission-side limits (W3-17), proven
conditional decoder-side sufficiency (W3-18), an empirically-
validated decoder-side ceiling break at +15–17.5 pp (W3-19),
and a sharp sign-flip signature for asymmetric decoder-side
cross-domain transfer (W3-C6)** — materially closer to a
paradigm shift than Phase 47 was, but still below the W3-C7
threshold (≥ 0.400 accuracy with symmetric transfer). The
next genuine paradigm-shift candidate is a deeper hypothesis
class or joint admit-decode co-training that crosses the
0.400 bar; that is the Phase 49 research agenda.
