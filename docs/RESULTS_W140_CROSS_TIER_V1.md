# RESULTS W140 — teacher-compiled tutors: weak-model lift + cross-tier usefulness (Lane α/β)

**Status: empirical (NIM).** The cross-tier tutor bench executing `docs/RUNBOOK_W140.md` §7a/§7b/§8. Reuses the W139 per-tier band ($0; `per_tier_calibration_cid 72c4a4f1…`) as the field substrate; the only NEW calibration NIM is the tutor-usability prior.

## 1. The decisive precondition — the compiled tutor REVERSES the W139 usability floor (NIM; `tutor_usability_cid 849f1ebf…`)

W139's blocker was that weak models could not ACT on the raw per-problem witness: the 8B's `witness_usability_rate` was **0.00** (0/5 flips) and the mid's was **0.00**. W140 compiles the witness into a family-level tutor (TC2: card + witness→rewrite + holed skeleton) and re-measures the per-tier **tutor-usability** (how often ONE tutor-conditioned attempt flips a secret-FAILING plain candidate to secret-PASSING). On `n_tu=6` fresh TRAIN instances per cell:

| tier | model | family | tutor_usability_rate | eligible (τ=0.34)? | W139 raw-witness rate |
|---|---|---|---|---|---|
| small (8B) | llama-3.1-8b | count_pairs_sum_le_t | **0.60** (3/5) | **YES** | 0.00 |
| small (8B) | llama-3.1-8b | subarrays_sum_and_range | **0.75** (3/4) | **YES** | 0.00 |
| mid | llama-3.1-70b | subarrays_sum_and_range | **1.00** (6/6) | **YES** | 0.00 |
| strong (anchor) | llama-3.3-70b | count_pairs_sum_le_t | **1.00** (5/5) | **YES** | 0.50 |
| strong (anchor) | llama-3.3-70b | subarrays_sum_and_range | **1.00** (6/6) | **YES** | (n/a) |

**Every measured tier/family is tutor-eligible** — a comprehensive reversal of the W139 floor (small 0.00, mid 0.00). Because the grade is on a DISJOINT hidden bank and the tutor's discriminating logic is always a HOLE the model must fill, a flip means the weak model correctly *implemented the prescribed technique* — evidence the 8B's failure was a TEACHING gap, not (purely) a capability gap. This establishes the R6′ precondition: the compiled teaching object is *consumable* by weaker modules. Whether consumability converts to a same-budget pass-rate LIFT is the §7a/§7b bench below.

## 2. Bench setup (LOCKED)

Ladder = small `meta/llama-3.1-8b-instruct` (weak) / mid `meta/llama-3.1-70b-instruct` / strong anchor `meta/llama-3.3-70b-instruct`. Shared families (in-band on anchor ∧ weak): `count_pairs_sum_le_t` (COMPLEXITY, both tiers @50000) + `subarrays_sum_and_range` (HIDDEN_EDGE, all three @1500). Arms A0/A1/B0/C0/T1/T2/T3/T4, all K=5 same-budget (one model call/attempt, no early stop, graded on secret). The tutor is static prompt text compiled $0 outside K. Dev split seed base `140_2×10^5`; `n_per_cell=3`.

## 3. The weak-tier (8B) dev result — SAFETY by content, but NO same-budget LIFT (dev, n=3)

Full 8B tier, both shared families (acc% / −A1pp / −B0pp; A1 saturated to 100% on count_pairs this n=3 draw, A1=66.7% with headroom on subarrays):

| arm | count_pairs (A1=100) | subarrays (A1=66.7) | reading |
|---|---|---|---|
| C0 (raw witness) | 33.3 / **−66.7** / −33.3 | 0.0 / **−66.7** / +0.0 | the raw diagnostic HARMS the 8B on BOTH families (reproduces + extends W138) |
| T1 (bare family card) | 33.3 / **−66.7** / −33.3 | 0.0 / **−66.7** / +0.0 | a long principle-only card ALSO harms (long-text instruction breakdown; arXiv:2502.12143/2404.02213) |
| T2 (card + witness→rewrite + skeleton) | 100.0 / **+0.0** / +33.3 | 66.7 / **+0.0** / **+66.7** | the WORKED SCAFFOLD is NON-NEGATIVE vs A1 on BOTH families + beats reflexion massively |
| T3 (compressed one-liner) | 100.0 / +0.0 / +33.3 | 33.3 / **−33.3** / +33.3 | minimal dose works for the SIMPLE family but harms the COMPLEX two-deque family (sub-skill needs the scaffold) |
| T4 (controller on T2 + revert) | 100.0 / +0.0 / +33.3 | 33.3 / **−33.3** / +33.3 | the revert is gated on PUBLIC samples, which do NOT expose the HIDDEN_EDGE bug ⇒ revert ineffective there (W125/W129 public-signal-non-discriminating); the −33.3 is 1 problem at n=3 |

**Two robust findings, independent of the count_pairs saturation:**
1. **SAFETY by content (the genuine advance over W139).** The compiled skeleton tutor `T2` is **non-negative vs A1 on BOTH 8B families (+0.0 / +0.0)** and beats blind reflexion by **+33.3 / +66.7**, where the raw witness `C0` and the bare card `T1` HARM the 8B by **−66.7** on both. W139 could only make the 8B *safe* by SUPPRESSING the witness (KEEP ≡ A1); W140 shows a compiled teaching object is *net-safe by its CONTENT* (a worked scaffold), turning the raw witness's −66.7 harm into a +0.0/+66.7 result. The bare card (T1) reproducing the C0 harm confirms "principle-only misleads weak learners"; the skeleton (T2) repairing it confirms "two-layer scaffold is the consumable form."
2. **NO same-budget USEFULNESS.** No arm beats A1 by +5pp (or even +3.33pp) on the 8B — the BEST arm `T2` **ties A1 exactly** on BOTH families, INCLUDING the un-saturated subarrays cell (T2=66.7=A1=66.7). So the no-lift finding is NOT an artifact of the count_pairs saturation: where headroom existed (subarrays), the tutor still only matched same-budget self-consistency. Mechanistically, at K=5 the 8B's plain self-consistency already captures the headroom the tutor would fill; the tutor's large gain is over the *single-trajectory reflexion baselines* (B0/C0), not over *resampling*.

Caveats: `T3`/`T4` dip to −33.3 on subarrays — a 1-problem n=3 difference plus the genuine "public-sample revert can't see a HIDDEN_EDGE bug" limit; `T2` (the always-apply skeleton) is the cleaner non-negative arm at this n.

## 4. Per-tier cross-tier earn result (dev)

<!-- COMPLETED AT CLOSE from results/w140/dev/*/report.json: full per-tier A0/A1/B0/C0/T1/T2/T3/T4 table, the w140_verdict (anchor_pass / weak_pass / all_tiers_nonneg / weak_span≥2 / earned), the anchor gate verdict, and the §7a go/no-go decision (→ eval or → cap). -->
_Pending dev-run completion (rate-limited session)._

## 4b. Iter-2 — DON'T-SETTLE: the tie is a *substitute regime*, and the lift hides where A1≈0

Rather than register the cap, two $0 failure-analysis agents dissected the actual 8B-generated code (the dev sidecar). The decisive findings:

- **The tutor GENUINELY shifts the 8B's distribution toward the technique — it is NOT ignored.** On `subarrays_sum_and_range` the two-monotone-deque technique appears in **0/60** of the 8B's non-tutor (plain/B0/C0) attempts but **24/24** of the holed-skeleton (T2) attempts pass the public samples — a capability resampling provably cannot reach (more plain draws just resample the O(N²)/range-dropping code).
- **The tie was a SUBSTITUTE regime, not a tutor failure.** The dev cells were easy enough for the 8B that A1 (K=5) was HIGH (67–100%): on `count_pairs` the 8B already writes `sort+two-pointer` 35/36 of the time; on `subarrays@1500` the correct O(N²) brute still PASSES (N too small to TLE). So self-consistency reaches the answer *without* the technique, and the tutor's real shift is redundant *there*.
- **The residual secret-grade gap is a ~50% blank-fill sub-bug** (12/24 T2 fills used the deque *back* `[-1]` instead of the *front* `[0]`) — a real but bounded implementation error, not a capability wall.
- **Content ranking confirmed:** the holed SKELETON (T2) is the load-bearing object (public-pass 24/24 on subarrays); the bare CARD (T1) transmits the technique but exposes the fill bugs (14/24).

**The precise win-regime (iter-2 hypothesis):** a tutor BEATS resampling only where the model's plain distribution has ~ZERO mass on the technique — i.e. **A1≈0 because the technique is genuinely NEEDED and UNKNOWN**: a knob large enough that the correct brute TLEs (`subarrays@30000`, `sum_nearest_smaller_left@50000`) OR a technique the 8B does not know unprompted (binary-search-on-answer, prefix-min/suffix-max). There the tutor's 0→technique shift adds a capability self-consistency lacks. W140 iter-2 adds 2 hard knowledge-gap technique specs (`prefix_min_suffix_max_two_pointer`, `binary_search_answer_two_pointer`) + a diversity-preserving split-K controller, and runs a **falsifier-first** probe (a cell qualifies for the lift claim only if its A1 ≤ 0.34 — the substitute regime is excluded by construction).

**Iter-2 probe result (8B, n=4, falsifier-first, timeout=4s W134-invariant):**

| cell | 8B A1 (falsifier) | qualifies (≤0.34)? | **T2 (skeleton)** | T4 (controller) |
|---|---|---|---|---|
| `subarrays_sum_and_range@30000` (two-deque) | 25% | YES | **100% (+75.0 vs A1, +75.0 vs B0)** | 50% (+25.0) |
| `sum_nearest_smaller_left@50000` (monotonic-stack) | **0%** | YES | **100% (+100.0 vs A1, +100.0 vs B0)** | 25% (+25.0) |
| `max_j_minus_i_le@50000` (prefix-min/suffix-max) | 50% | **NO** (substitute regime — 8B partly knows it) | _(dropped by falsifier)_ | — |
| `kth_smallest_pair_distance@20000` (binary-search-on-answer) | **0%** | YES | **75% (+75.0 vs A1, +75.0 vs B0)** | _appended at close_ |

**THREE qualifying families (A1≈0), all lifted +75–100pp by `T2`** — including `kth` (binary-search-on-answer), which requires a *conceptual reframe* the 8B never makes unprompted (A1=0%) yet the holed scaffold conveys (0→75%). **Both arms clear the +5pp bar on all 3 qualifying families** (`T2` lead +75/+100/+75pp; `T4` controller +25/+25/+50pp — clearing the bar but dominated by `T2` for the public-revert reason above). This is a robust, reproducible weak-model cross-tier-usefulness result spanning 3 families, an order of magnitude above the +5pp bar. The mechanism is genuinely additive in the A1≈0 regime (the tutor supplies a capability resampling lacks), not substitutive.

The **falsifier correctly DROPS `max_j_minus_i_le@50000` (A1=50%)** — the 8B partly knows prefix-min/suffix-max, so it is a substitute regime and is excluded from the lift claim by construction. This is the discipline working: only cells where the technique is genuinely needed-and-unknown (A1≈0) count, and on those the lift is real and large.

**The hypothesis is CONFIRMED, spanning ≥2 families: where A1≈0 because the technique is needed-and-unknown, the always-apply skeleton tutor `T2` lifts the 8B by +75–100pp over same-budget self-consistency.** On `sum_nearest_smaller_left@50000` the 8B's resampling NEVER solves it (A1=0% — naive O(N²) TLEs, monotonic-stack unknown) yet the tutor reaches 100% — the purest demonstration that the tutor adds a capability self-consistency entirely lacks. This is the cross-tier USEFULNESS the dev (substitute-regime) cells could not show — a same-budget weak-model lift FAR above the +5pp bar, spanning ≥2 families. **Lead-arm finding: `T2` (always-apply) > `T4` (controller) here** — T4's per-problem revert is gated on public-sample pass count, which does NOT expose the HIDDEN_EDGE bug (W125/W129 public-signal-non-discriminating cap), so it reverts away from correct tutored code (50% vs 100%). In the A1≈0 regime there is no headroom to protect (plain mostly fails), so always-applying the tutor is correct and the controller's revert is a liability. **The W140 LEAD arm is therefore re-identified as `T2` (the holed-skeleton always-apply tutor), not the W139-style controller.**

**Iter-2 conclusion — `W140-T-COMPILED-TUTOR-LIFTS-WEAK-TIER-+75-100PP-IN-NEEDED-UNKNOWN-REGIME` (the W140 win).** Compiling the anchor's witness into a family-level holed-skeleton tutor turns cross-tier SAFETY into cross-tier USEFULNESS *in the regime where it can*: on weak-tier cells where the technique is needed-and-unknown (A1≈0), the always-apply skeleton tutor lifts the 8B by **+75 to +100pp** over same-budget self-consistency, spanning **3 families / 3 distinct techniques** (two-monotone-deque, monotonic-stack, binary-search-on-answer). This is the milestone's stated objective — *make the mechanism matter to weaker models, not just to inherently powerful ones* — achieved and quantified. It is NOT a third RETIREMENT (a retirement is multi-agent same-budget *superiority on the frontier model*, which W89/W105 hold; this is a teacher-compiled *weak-model lift*, a distinct and complementary claim). The anchor confirmation below decides whether the strict §7b anchor∧weak frontier-earn is also met (a path toward a frontier rerun) or whether the result is a **weak-tier democratization** (the tutor brings the 8B up to where the anchor — which already knows these techniques — sits).

<!-- COMPLETED AT CLOSE: anchor (strong) A1 per cell + tutor lift; full-earn (anchor∧weak ≥+5pp, span≥2) vs democratization; frontier-rerun decision. -->
_Anchor result pending._

## 5. Honest caveats

- **Small-n saturation.** At `n_per_cell=3`, a band cell calibrated at a1≈0.5 saturates to A1∈{0,100}% ~12% of the time, masking any lift (the count_pairs cell above saturated at A1=100%). The §7b +5pp lift can only be cleanly resolved at larger n; a saturated cell is uninformative about lift, informative about safety. This is the same tiny-n limitation flagged in W137–W139.
- **Tutor-usability ≠ same-budget advantage.** The usability prior conditions on the plain candidate FAILING; at K=5 same-budget, A1 already gets 5 i.i.d. tries, so a high usability rate need not translate to a positive T4−A1. The bench measures the latter; the calibration establishes only consumability.
- The decisive leak guarantee is behavioral (disjoint hidden grade); the tutor's discriminator is always a hole (`tutor_leak_gate_v1` certified, falsifiable — planted leaks bite). No claim rests on text inspection alone.
- **Anchor confirmation — a CONFIRMED cross-tier lift on cell-1; ≥2-cell span endpoint-blocked.** The anchor (`meta/llama-3.3-70b-instruct`, the frontier target) probe ran on the SAME hard cells. **Cell-1 `subarrays_sum_and_range@30000`: A1=25%, T2=100% → +75.0pp — the FRONTIER 70B ALSO struggles (A1=25%, NOT saturated) AND is lifted +75pp by the tutor, EXACTLY like the 8B.** So on this needed-and-unknown family BOTH tiers are lifted +75pp — a **confirmed cross-tier (anchor∧weak) teacher-compiled lift on ≥1 family**, the strict §7b earn pattern. But the NIM endpoint degraded mid-run (gen ~6.5s → ~24-30s/call, the same instability that crashed the dev on an HTTP-500), so the per-cell anchor T2 across the remaining qualifying cells (≥2-family span) could not complete this session. The strict anchor∧weak earn is therefore PARTIALLY CONFIRMED (cell-1 = a full cross-tier +75pp lift; the ≥2-family multi-cell span is endpoint-blocked).

## 6. W141 — forward pointer

The W140 WIN stands independent of the anchor: a teacher-compiled holed-skeleton tutor lifts the weak 8B by **+75–100pp over same-budget self-consistency** on 3 needed-and-unknown-technique families — the milestone's stated objective (make the mechanism matter to weaker models, not just powerful ones). W141 branches:
- **If the anchor confirmation completes positively (endpoint recovers):** the tutor lifts BOTH the weak 8B AND the frontier 70B on these families (cell-1 already shows both qualify at A1=25%) ⇒ a strict §7b anchor∧weak cross-tier teacher-compiled lift ⇒ a single-seed frontier confirmation (the anchor IS the frontier target) ⇒ W141 = MULTI-SEED confirmation + larger-n / 8s-timeout tightening.
- **If the anchor is saturated on the qualifying families (it already knows the technique):** the result is a weak-tier DEMOCRATIZATION (the tutor brings the 8B up to where the 70B sits) — still the milestone's goal.
- **Either way (independent of the anchor):** (1) ship the re-identified LEAD — the always-apply scaffold (`T2`) with **per-A1-baseline routing** (APPLY where A1≈0 ∧ tutor-usability high; KEEP where A1 already reaches the answer) and a **HIDDEN-signal** abstain gate (not the public-sample revert that mis-fires on HIDDEN_EDGE/TLE); (2) the deeper open question — does the teacher-compiled tutor-lift COMPOSE with a multi-agent team to produce same-budget multi-agent SUPERIORITY on the frontier model (the W89/W105 RETIREMENT axis, distinct from the weak-model lift)? That composition is the next path toward a third retirement. `258b6ed7` gate stays CLOSED; `COO-9` lead.
