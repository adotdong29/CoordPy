# RESULTS W142 — moderate-`p` family screen + high-K discover reliability + retirement attempt

**Thesis.** W141 earned the first clean no-oracle resistant SAME-BUDGET superiority on ONE family (`count_pairs_sum_le_t`, fair `p=0.08`, +70.6pp at K=4 over no-oracle verified-selection) but was short of the §7b ≥3-family / ≥2-mode span — blocked by the frontier 70B's BIMODAL per-technique generation. W142 attacks that blocker directly: **(α)** screen a wide resistant-by-construction family slate for moderate-`p` extractable supply under a fair neutral baseline; **(β)** raise discover-K, hold amortize-K, and attempt a family-level retirement only if ≥3 families honestly admit; **(γ)** primary-source research + architecture-requirements (R7′) + the stronger-model gate. RUNBOOK_W142 §1–§13 LOCKED before any spend.

## 1. The mechanism (built, $0-validated, explicit-import only)
- `coordpy.moderate_p_family_slate_v1` — the screen slate: 3 NEW families (`count_pairs_product_le_t` sort+two-pointer; `count_triples_sum_lt_t` O(N³)→O(N²); `count_subarrays_range_le_l` two-deque counting) + the extractable existing families (`count_pairs_sum_le_t`, `count_pairs_absdiff_le_d`, `subarrays_sum_and_range`, `sum_nearest_smaller_left`) + 2 NON-extractable negative controls (`count_subarrays_sum_divisible_k` prefix-hash, `kth_smallest_pair_distance` BSoA). `slate_cid 9b8cf758…`.
- `coordpy.moderate_p_family_screen_v1` — the $0 gates (G1 parser-neutral / G2 exact-oracle discriminating / G3 gated-accumulator extractable / G4 novelty) + fair-`p` over a neutral-prompt bank + Wilson + the moderate-`p` band rule.
- `coordpy.discover_amortize_accounting_v1` — the explicit equal-total-family budget identity + the `(1−p)^{K_a}` per-member superiority.
- Drivers: `scripts/run_w142_family_screen_v1.py` (Lane α), `scripts/run_w142_retirement_pilot_v1.py` (Lane β; wires discover-K ≠ amortize-K — the W141 gap). `coordpy/__init__.py` untouched; `0.5.20` unchanged.

## 2. The DECISIVE $0 result — extractability narrows the veins (machine-checked, no NIM)
The W141 AST extractor only cleanly holes a SINGLE printed accumulator updated by `acc += <expr>` gated by an `if` (pattern a) or a shrink-`while` (pattern b). The W142 G3 gate = `compile_tutor_from_winner_v1(ref).compiled ∧ n_pred_holes ≥ 1` (a GATED accumulator — the technique's DECISION is the hole, making it a teaching object, not a contribution-fill). Run on every candidate's canonical reference at $0 (validation `FAILS: NONE`):

| family | vein | G1 | G2 | G3 | G4 | n_pred | verdict |
|---|---|---|---|---|---|---|---|
| count_pairs_sum_le_t | sort+two-pointer | ✓ | ✓ | ✓ | ✓ | 1 | extractable (W141 win) |
| count_pairs_absdiff_le_d | sort+two-pointer | ✓ | ✓ | ✓ | ✓ | 1 | extractable |
| count_pairs_product_le_t (NEW) | sort+two-pointer | ✓ | ✓ | ✓ | ✓ | 1 | extractable |
| count_triples_sum_lt_t (NEW) | sort+two-pointer | ✓ | ✓ | ✓ | ✓ | 1 | extractable |
| count_subarrays_range_le_l (NEW) | two-deque | ✓ | ✓ | ✓ | ✓ | 1 | extractable |
| subarrays_sum_and_range | two-deque | ✓ | ✓ | ✓ | ✓ | 1 | extractable |
| sum_nearest_smaller_left | monotonic-stack | ✓ | ✓ | ✓ | ✓ | 1 | extractable (W141 p=0.67 high) |
| **count_subarrays_sum_divisible_k** | prefix-hash | ✓ | ✓ | **✗** | ✓ | **0** | **REJECTED — technique in dict-maintenance, no gating predicate** |
| **kth_smallest_pair_distance** | binary-search-on-answer | ✓ | ✓ | **✗** | ✓ | **0** | **REJECTED — printed answer is a reassignment, not an accumulator** |

**The teaching-object representation is itself a binding constraint on which families self-tutoring can amortize** (architecture finding, ARCHITECTURE_REQUIREMENTS_W142_V4 §4): the discover-then-amortize head (R7′) and the teaching-compiler head (R6′) are COUPLED — a family is amortizable only if its technique is expressible as a gated-accumulator hole. The prefix-hash and BSoA veins are rejected at $0, before any NIM — exactly the de-risk prediction.

## 3. The fair-`p` band rule + the same-budget budget theory (LOCKED)
- **Fair neutral baseline (FNB):** the W141-v4 self-consistency prompt + a 2nd neutral phrasing (FormatSpread arXiv:2310.11324); names NO technique, NO efficiency/time/size cue, NO data structure. `p̂ = passes-on-hidden / K` (a hidden pass ⇒ correct AND efficient by construction). Median over the bank; band screen uses the median.
- **Admission:** `p̂ ∈ [0.10, 0.50]` AND Wilson-95% excludes 0 and 1 (IRT peak-Fisher-information band; metabench arXiv:2407.12844 / Fluid arXiv:2509.11106), on top of the $0 G1–G4.
- **Same-budget retirement claim (equal total family budget `G`):** at `G = M·K_a`, ST's per-member superiority over no-oracle verified-selection B0 is `(1−p)^{K_a}` — retirement-grade for `p ≲ 0.5` at `K_a=4` (p=0.10→66pp, 0.25→32pp, 0.40→13pp, 0.50→6.25pp), and ZERO at the bimodal extremes. Raising discover-K is a one-time family cost `K_d/M → 0`, reported separately, never folded into a per-problem claim.

## 4. Screen results (Lane α) — fair `p` per family on the frontier anchor — **SUCCESS (3 families, 3 veins, 2 modes)**
Fair neutral prompt (canonical FNB), K=12, `meta/llama-3.3-70b-instruct`, 7 parallel per-family runs (`slate_cid 9b8cf758`):

| family | vein | mode | fair `p̂` (k/12) | Wilson-95% | in-band | ADMITTED |
|---|---|---|---|---|---|---|
| **count_pairs_sum_le_t** | sort+two-pointer | COMPLEXITY_BLIND | **0.333** (4/12) | [0.138, 0.609] | ✓ | **YES** |
| **subarrays_sum_and_range** | two-deque | HIDDEN_EDGE | **0.167** (2/12) | [0.047, 0.448] | ✓ | **YES** |
| **sum_nearest_smaller_left** | monotonic-stack | COMPLEXITY_BLIND | **0.500** (6/12) | [0.254, 0.746] | ✓ | **YES** (top edge) |
| count_pairs_absdiff_le_d | sort+two-pointer | — | 0.000 (0/12) | [0, 0.242] | ✗ | no — 0-supply |
| count_pairs_product_le_t (NEW) | sort+two-pointer | — | 0.000 (0/12) | [0, 0.242] | ✗ | no — 0-supply |
| count_triples_sum_lt_t (NEW) | sort+two-pointer | — | 0.000 (0/12) | [0, 0.242] | ✗ | no — 0-supply |
| count_subarrays_range_le_l (NEW) | two-deque | — | 0.000 (0/12) | [0, 0.242] | ✗ | no — 0-supply |

**Lane α SUCCESS:** 3 admitted families, **3 distinct technique veins** (sort+two-pointer / two-deque / monotonic-stack), **2 modes** (COMPLEXITY_BLIND + HIDDEN_EDGE), 36 anchor instances. Both the ≥3-family AND the ≥2-mode §7b span are reachable.

**Sharp finding — the band is a THIN FIXED set, not expandable.** All 4 new/sibling families added to the slate (`count_pairs_absdiff`, `count_pairs_product`, `count_triples_sum`, `count_subarrays_range_le_l`) came back at **`p=0.000` (0/12)** — the frontier 70B is CLUELESS on them (the bimodal wall's `p≈0` half). The 3 admitted families are exactly the 3 the W138 headroom-band calibration found. So the moderate-`p` band EXISTS and spans ≥3 veins / 2 modes, but it cannot be widened by adding new technique surfaces — a precise empirical confirmation that the bimodal-generation structure is real (architecture finding R7′: the per-family supply estimator can detect the band but cannot manufacture it; ARCHITECTURE_REQUIREMENTS_W142_V4 §3-4).

Per-member same-budget superiority theory `(1−p)^{K_a=4}`: count_pairs_sum +19.8pp, subarrays_sum_and_range +48.2pp (strong), sum_nearest +6.25pp (borderline — at `p=0.5` verified-selection nearly saturates at K=4). The aggregate retirement margin is carried by the two low-`p` families; the high-edge NSL is a marginal contributor.

**LOCKED Lane β slate (pre-registered before any β spend):** families = {count_pairs_sum_le_t, subarrays_sum_and_range, sum_nearest_smaller_left}; `K_amortize=4`, `M=4` members; arms A1/B0/ST + NEG (alien wrong-vein scaffold, built $0); earn = ST−B0 ≥ +5pp aggregated ∧ span ≥3 fam OR ≥2 modes ∧ NEG no-lift ∧ no-oracle audit ∧ equal-per-member-budget (discovery `K_d` reported as the amortized one-time cost, `K_d/M → 0`).

**Discover-reliability (the Lane β finding, runbook §6 — discover-K raised as pre-authorized).** At `K_d=12`, the no-oracle DISCOVER ABSTAINED on BOTH low-`p` families (`count_pairs_sum` and `subarrays_sum_and_range`: `no_correct_efficient_candidate`) while it succeeded on the high-edge `sum_nearest` (p=0.5, `discovered=True, winner_passes_secret=True`). The n=12 fair-`p` estimates are NOISY — the screen's count_pairs_sum 4/12 and a discover 0/12 are both consistent with a true `p≈0.15–0.20`, where `(1−p)^12` leaves ~11% of K_d=12 discover runs empty. ⇒ **the low-`p` families — exactly where the amortization win `(1−p)^{K_a}` is LARGEST — are the HARDEST to discover, so high-K discover reliability is the binding lever** (the W142 thesis). `K_discover` was raised to **24** for the two low-`p` families (P(≥1 winner)≈0.95–0.99) and held at 12 for `sum_nearest`; reported per-family as the one-time amortized discovery cost. The earn comparison (ST vs B0) is unchanged — both at equal per-member `K_a=4`.

## 5. Lane β — discover-reliability + family-level retirement attempt — **RETIREMENT NOT EARNED (ST−B0 = +0.0pp)**
Parallel single-family pilots on the 3 admitted families, `K_amortize=4`, `M=4`, alien-vein NEG, `meta/llama-3.3-70b-instruct`:

| family | mode | K_d | discovered | winner_passes_secret | A1 | B0 | ST | /M | p (members) | q | NEG (ST−B0) | (1−p)^4 theory |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| count_pairs_sum_le_t | COMPLEXITY | 48 | **False** | None | 3 | 3 | 3 | 4 | 0.31 | 0.00 | −2 | +22pp |
| subarrays_sum_and_range | HIDDEN_EDGE | 24 | **False** | False | 2 | 1 | 1 | 4 | 0.19 | 0.00 | −1 | +44pp |
| sum_nearest_smaller_left | COMPLEXITY | 12 | True | True | 4 | 4 | 4 | 4 | 0.50 | 1.00 | +0 | +6pp |
| **AGGREGATE** | | | | | **9** | **8** | **8** | **12** | | | all ≤0 | |

**ST−B0 = +0.0pp; 0 families with ST>B0; span FAILS; RETIREMENT NOT EARNED.** NEG no-lift holds (alien-scaffold lifts all ≤0 ✓). **W89 (+5.56) + W105 (+7.00) STAND as the only two retirements.**

**The three admitted families each failed for a DISTINCT, machine-diagnosed reason (the W142 sharp finding):**
1. **`sum_nearest` (high-edge p=0.5) — B0 SATURATION.** DISCOVER succeeded end-to-end (`discovered=True`, scaffold compiled n_holes=2, `q=1.00`), but no-oracle verified-selection B0 already solves 4/4 at K=4 (`1−0.5^4=0.94`/member) ⇒ ST ties B0, +0pp. The high-`p` half of the bimodal wall: where discovery is easy, the baseline saturates.
2. **`count_pairs_sum` (low-`p`) — DISCOVER ABSTAIN via S1 false-negatives.** At K_d=48: 47/48 pass public, 27/48 are *efficient* but **DISAGREE with the self-brute** (S1 rejects), and the 12 that *agree* are the slow naive (not efficient) ⇒ **0 correct+efficient winners committed**. The screen found 4/12 secret-passing on the same instance, so S1 is **false-negative-rejecting genuinely-correct+efficient candidates** — the no-oracle verifier's S1 reliability depends on the SELF-BRUTE being correct AND counting-convention-matched; a buggy/convention-mismatched self-brute false-rejects correct candidates. The W141 "0 FP/FN" held on samples where the self-brute was correct; at low-`p` discovery here it did not.
3. **`subarrays_sum_and_range` (2nd mode, HIDDEN_EDGE) — VERIFIER FALSE-POSITIVE + non-extractable.** Committed 6 "winners" (consensus 1.0) that are the sum-only naive (agree with the self-brute on the non-binding small bank, efficient) but **fail secret** (`winner_passes_secret=False`) and have **no holeable accumulator** (`extract:no_accumulator_update`) ⇒ discarded (non-negativity held). The W125 looks-right-fails-hidden limit on the multi-constraint family: the S1 adversarial bank did not generate range-binding inputs to catch the naive.

**The W142 result (a sharp, honest cap that CONFIRMS+SHARPENS W141).** Lane α proved the moderate-`p` band EXISTS and spans ≥3 veins / 2 modes; Lane β proves the **discover-then-amortize CONVERSION is bottlenecked because win-size and discovery-difficulty are COUPLED via `p`**: the high-`p` edge gives easy discovery but a saturated baseline (no headroom), and the low-`p` edge gives a large theoretical win (`(1−p)^4`) but unreliable no-oracle DISCOVERY — the verifier's S1 is brittle exactly where it matters (self-brute correctness/convention at low `p`; constraint-coverage of the adversarial bank for the multi-constraint 2nd mode). Raising K_d cannot fix a verifier false-negative or a generation distribution with ~0 correct+efficient mass. ⇒ `W142-L-DISCOVER-THEN-AMORTIZE-CONVERSION-IS-VERIFIER-RELIABILITY-AND-BASELINE-SATURATION-BOUND`.

**W143 targets (the precise, machine-checkable blockers):** (a) a no-oracle verifier robust at low `p` — validate the self-brute beyond public samples (cross-check two independent self-brutes; reject on internal disagreement) and guarantee the S1 adversarial bank COVERS each stated constraint (so the multi-constraint naive is caught); (b) a discovery mechanism beyond i.i.d. resampling for the low-correct-rate regime (W128 role-diverse search to lift the correct+efficient generation rate); (c) a moderate-`p` family strictly INSIDE `(0.1, 0.4)` on BOTH edges (avoid B0-saturation at 0.5 AND the sub-0.1 discovery wall). These are mechanism upgrades, NOT post-hoc baseline tuning — W142 registers the honest cap; the upgrades earn their own milestone.

## 6. Honest framing
A retirement requires same-budget multi-agent superiority on the FRONTIER spanning ≥3 families / ≥2 modes with the no-oracle audit holding, NEG (fake-scaffold) failing, and the fair-neutral baseline unmodified. The bimodal-generation wall is a real structural prediction; the moderate-`p` ∧ extractable ∧ span-distinct intersection may be THIN. A `<3`-family screen result is a legitimate, registered supply cap (the mechanism stands as a de-risked first; the retirement is gated by the model's intrinsic bimodality, not the coordination machinery). W89 (+5.56) + W105 (+7.00) STAND unless a clean retirement-grade earn is registered. NIM endpoint ~62s/call this session (full 1536-token program gens); the screen is budgeted accordingly.
