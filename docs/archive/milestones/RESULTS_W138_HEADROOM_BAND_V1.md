# RESULTS W138 — headroom-band battlefield search + cross-scale mechanism validation

**Status: Lane α COMPLETE (band landed); Lane β IN PROGRESS (§7a dev).** Executes `docs/RUNBOOK_W138.md` (locked before any NIM). `ultracode` OFF. `coordpy.__version__ == "0.5.20"`; `coordpy/__init__.py` untouched; no PyPI. W89+W105 remain the only two retirements unless a clean MULTI-SEED `PASS_MECHANISM_DRIVEN`.

## 0. The reframe that drives W138

W137's repaired parser-neutral field came back BIMODAL at 70B (every template A1∈{0,1}; 1/17 survivor) ⇒ $0 frontier. But the W137 dev mechanism bench showed the M1 family-routed witness controller **already scored B−A1 = +33.33pp** and passed BOTH MLB sub-gates on the surviving complexity cells — it failed **only** `SPAN_FAIL(modes=1<2, families=1<3)`. **W137 was a SPAN failure, not a mechanism failure.** Two confounds narrowed the field:
1. **Measurement:** A1 was measured at `n_a1=1` (a single any-of-K draw on ONE instance) ⇒ a1_rate∈{0,1} BY CONSTRUCTION (`count_pairs_absdiff_le_d` was culled `HC4_DEAD` on a coin-flip the bench then rescued at +33pp).
2. **Construction:** the discrete templates were single-difficulty-point, so per-template A1 was bimodal.

W138 fixes both: **A1 measured as a population RATE over n_a1=8 instances** + **continuous hardness-knob (hidden N) sweep** + a **functional family** designed to populate a 2nd mode (IRT peak-information at p≈0.5 — metabench arXiv:2407.12844; continuous-knob smooth difficulty — CLRS-Text arXiv:2406.04229 / FuncBenchGen arXiv:2509.26553).

## 1. Lane α — headroom-band battlefield search (COMPLETE)

**Slate (`coordpy.headroom_band_slate_v3`, `slate_fingerprint_cid = 8676f283…`):** 9 knob-parameterized families (7 complexity / 2 functional) × {20000, 50000} (complexity) and {4000, 30000} (functional) = 18 candidate cells. Build self-test PASS: every cell HC1-parser-neutral + HC2 exact-oracle-discriminating; deterministic; the W136 I/O-confound fixture, the W137 bimodality detector, and the fake-different discipline all BITE.

**Calibration (`coordpy.headroom_band_calibration_v2`, ladder = `meta/llama-3.1-8b-instruct` + `meta/llama-3.3-70b-instruct`; n_a0=5 both tiers + n_a1=8 K=5 at the strong anchor; `calibration_cid = ff87cb60…`; 180 NIM calls):** band-admission HC3 ∧ HB3 (intermediate a1_rate with Wilson-95% excluding 0,1) ∧ HB4 (cross-scale discrimination).

| family | mode | best knob | strong a0 | **strong a1 (rate/8)** | small a0 | Wilson | verdict |
|---|---|---|---|---|---|---|---|
| count_pairs_sum_le_t | COMPLEXITY | 20000 | 0.00 | **0.60** | 0.20 | [0.23,0.88] | **ADMIT** |
| subarrays_sum_and_range | HIDDEN_EDGE | 30000 | 0.00 | **0.40** | 0.00 | [0.12,0.77] | **ADMIT** |
| sum_nearest_smaller_left | COMPLEXITY | 50000 | 0.40 | **0.20** | 0.00 | [0.04,0.62] | **ADMIT** |
| count_inversions / longest_bounded_subarray / mod_then_maxsub | — | both | ≥0.80 | 1.00 | 1.00 | — | cull HC3_SATURATED |
| count_pairs_absdiff_le_d / count_subarrays_sum_le_s / max_j_minus_i_le | COMPLEXITY | both | 0.00 | 0.00 | 0.00 | [0.00,0.43] | cull HB3_DEAD (baseline) |

**Result: 5/18 cells admitted ⇒ 3 surviving FAMILIES across 2 surviving MODES (COMPLEXITY_BLIND + HIDDEN_EDGE_STATE_MISS).** The band GO/NO-GO (`≥3 families OR ≥2 modes`) = **TRUE on both criteria** — the construction succeeded where W137 had 1 family / 1 mode. The n_a1=8 fix is decisive: `count_pairs_sum_le_t@20000` reads a1=0.60 here vs A1=1 (saturated) in W137. Cross-scale discrimination holds (e.g. nearest_smaller strong a1=0.20 > small a0=0.00).

**Note on the a1=0 "dead" complexity cells:** `absdiff`/`subarrays_sum_le_s`/`max_j_minus_i` are baseline-dead (a1=0) but the complexity witness is known to RESCUE such cells (W137 absdiff: a1=0 → +33pp). HB3's `a1>0` floor is a conservative proxy for "not capability-dead"; the dev bench's `non_saturated` mode can include them (the §7a/§7b gate counts only ACTUAL rescues, so this widens the rescue surface without weakening the locked thresholds).

## 2. Lane β — cross-scale mechanism validation (IN PROGRESS)

Arms A0/A1/B0/C0(complexity witness)/N0(counterexample witness)/X1(family-routed controller, LEAD)/X2(negative control), K=5 same-budget, scored via the verbatim W108 evaluator + `repaired_field_mechanism_bench_v1.evaluate_gate_v1`.

**$0 witness-firing de-risk (before dev NIM):** the COMPLEXITY witness fires correctly on the O(N²) naive at the band knob (kind=COMPLEXITY at N=50000). The COUNTEREXAMPLE witness (N0/EW1) does **NOT** fire on the functional families — `find_counterexample_witness_v1` searches only *small* probe inputs (≤ token cap), while the functional families are benign on small inputs (so the naive passes public) and only fail on large ones. **This mechanistically explains W133's EW1 +0** and bounds the 2nd (non-complexity) mode to firing/capability limits at 70B (architecture requirements R3/R7/R9). The win path is therefore **the proven complexity witness rescuing across ≥3 distinct complexity families** (the §7b "complexity-only" exclusion bites only on a *single* family).

**§7a dev — the mechanism journey (3 NIM runs, all on disjoint dev seed 138_2xx):**
1. **3 admitted families (n=3):** A1=44.4, B0=66.7, **X1=77.8 (X1−A1=+33.33, X1−B0=+11.11)** — both clear the §7b +5pp bar — but `SPAN_FAIL` (1 family). Per-problem: blind reflexion B0 already solves the EASY complexity families (lifts a1=0.2→1.0 via K feedback attempts), so the oracle witness has no edge over reflexion there; its edge is only where reflexion's blind feedback fails. **B0 itself beats A1 by +22.2pp = the W89/W105 reflexion mechanism reproduced on a resistant-by-construction band (R10).**
2. **`non_saturated` 6 families incl. the a1=0 complexity families (n=2):** A1=25, B0=25 (blind reflexion FAILS the hard a1=0 families). **The pure complexity witness C0 = +33.33pp vs A1; the X1 controller only +16.67 — DOMINATED by C0** because its counterexample-routing is dead weight (the 2nd mode does not fire). ⇒ the operative lead is C0 (pre-registered).
3. **C0 as lead, complexity-only 5 families (n=3) — §7a PASS:** A0=0.0, A1=26.67, B0=33.33, **C0=80.00 ⇒ C0−A1 = +53.33pp, C0−B0 = +46.67pp**, MLB1=0.87, MLB2=0.77, all_structural, no_net_regression, **rescues span ALL 5 complexity families** (absdiff/count_pairs_sum/count_subarrays_sum/max_j_minus_i/nearest_smaller). **GATE PASS.**

So on the constructed multi-family complexity band, the same-budget oracle complexity witness beats BOTH self-consistency (+53pp) AND blind reflexion (+47pp) across 5 distinct algorithmic families — converting W133/W137's "real-but-single-family" witness into a multi-family §7a pass. The 2nd (counterexample) mode is honestly DEAD (R3/R7/R9), so the claim is multi-FAMILY single-MODE (complexity), which the locked span rule (`≥3 families OR ≥2 modes`) admits.

**§7b eval (held-out 138_3xx slice, n=4, 20 problems):** on the frontier anchor (`meta/llama-3.3-70b-instruct`) the margins + span HOLD on held-out data: **C0=65.0, A1=25.0, B0=40.0 ⇒ C0−A1 = +40.00pp, C0−B0 = +25.00pp, rescues span ≥3 families** (absdiff/count_pairs_sum/count_subarrays_sum), all structural, no regression. **BUT the locked 8B two-tier check FAILS: at `meta/llama-3.1-8b-instruct` the witness HURTS (A1=35.0 → C0=10.0, C0−A1 = −25.00pp)** ⇒ `TWO_TIER_SAME_SIGN_FAIL`.

This is a clean, important finding: **the complexity-witness gain is CAPABILITY-GATED.** It requires a generator able to *correctly implement* the efficient algorithm when prompted; the 8B cannot, so the witness pushes it toward broken efficient code and *lowers* its pass-rate below its own self-consistency baseline (literature-confirmed: small models need verifiers stronger than themselves — arXiv:2404.17140; self-correction can hurt — arXiv:2310.01798; architecture requirements R2/R7). The witness is a real mechanism, but its benefit is conditional on generator capability.

**Stronger-dev tier (`meta/llama-3.1-70b-instruct`, same held-out slice):** A0=50, A1=70, B0=80, **C0=80 ⇒ C0−A1 = +10.00pp, C0−B0 = +0.00pp** (1 family rescued). llama-3.1-70b is near-SATURATED on this anchor-calibrated band (A1=70%), so blind reflexion already maxes it (B0=80) and the witness merely ties reflexion. ⇒ **the witness-over-REFLEXION edge does NOT reproduce off the anchor.**

### §7b VERDICT — NOT EARNED (cross-tier robustness fails)

| tier | A1 | B0 | C0 | C0−A1 | C0−B0 |
|---|---|---|---|---|---|
| **llama-3.3-70b (frontier anchor / calibration point)** | 25 | 40 | 65 | **+40** | **+25** |
| llama-3.1-70b (capable dev proxy) | 70 | 80 | 80 | +10 | **+0** |
| llama-3.1-8b (locked small tier) | 35 | — | 10 | **−25** | — |

The held-out anchor margins + span PASS (conditions 1-4), but the **locked 8B two-tier check FAILS** (condition 5): the witness gain is **(a) capability-gated** (the 8B is *hurt* −25pp — it cannot implement the efficient algorithm the witness prescribes, so it writes broken code below its own self-consistency baseline) and **(b) anchor-specific over reflexion** (llama-3.1-70b ties B0 at +0). The locked small tier failed, and the characterization proxy *weakens* (not strengthens) the robustness case. Per the discipline ("do not count a close/weak edge as a win"), **the §7b two-tier robustness condition is NOT met ⇒ NO frontier rerun (§7c not run) ⇒ NO 3rd retirement.** `W89 (+5.56) + W105 (+7.00) STAND as the only two retirements.`

## 5. Outcome — what W138 establishes (and what it does not)

**ESTABLISHED (REVISES W137):**
- A parser-neutral, resistant-by-construction `0<A1<1` headroom band **IS constructible** at 70B — W137's "no template has 0<A1<1 / no mechanism-headroom band" cap was an **n_a1=1 measurement artifact + a knob-pinning artifact**, not a property of the field. Measuring A1 as a population rate (n_a1=8) + sweeping the continuous N knob lands 3-5 complexity families + 1 hidden_edge family in the band.
- On the **frontier anchor** (the W105 retirement model), the exact-oracle **complexity witness beats BOTH self-consistency (+40pp held-out) AND blind reflexion (+25pp) across ≥3 distinct complexity families** — the FIRST time the W133/W137 complexity witness clears the §7a/§7b SPAN gate (multi-family). §7a dev = +53/+47, all 5 families.
- **Blind reflexion (B0) itself beats self-consistency (A1) on the resistant band** (+22pp first run) = the W89/W105 mechanism reproduced on a resistant-by-construction field (R10).

**NOT ESTABLISHED (the honest boundaries):**
- The witness gain is **NOT robust cross-tier** — capability-gated (8B hurt) and anchor-specific over reflexion (llama-3.1-70b ties B0). ⇒ NO frontier earn, NO 3rd retirement.
- The **2nd (counterexample/non-complexity) mode is firing/capability-bound** — `find_counterexample_witness_v1` searches only small probe inputs, which functional families pass; this mechanistically explains W133's EW1 +0 (R3/R7/R9).

**Carry-forward:** W138 REVISES `W137-L` (the field is not inherently no-headroom) and ADDS the capability-gate finding. W89+W105 STAND. Stronger-model gate CLOSED (`258b6ed7` invariant). No version bump; no PyPI; `coordpy/__init__.py` untouched; `COO-9` lead.

**W139** = recalibrate the band PER-MODEL (so each tier is tested at its own p≈0.5 point) + a capability-MATCHED controller (abstain/KEEP for weak generators so the witness never hurts; R5) to recover cross-tier robustness — the band exists, the witness earns at the anchor, the open problem is making the gain hold across the model ladder. The architecture requirements (R1-R12) name what a coordination-native net must add (internal verifier matched to generator capability R2; in-band difficulty estimator R4; capability-aware allocation R5).

## 3. Lane γ — research + architecture-requirements + frontier gate

- Primary-source grounding wired into the gates (see RUNBOOK §9): IRT zero-information/metabench filter; continuous-knob CLRS-Text/FuncBenchGen; oracle-grounded counterexample/test feedback; off-band-inert self-consistency.
- `docs/ARCHITECTURE_REQUIREMENTS_W138_V1.md` — R1–R12 the eventual coordination-native architecture must satisfy to survive the W120–W137 chain (W138 tests the triad R1/R2/R3-R9 under the band condition R4/R10/R12).
- Stronger-model gate re-derived: `NO_CERTIFIABLE_STRONGER_MODEL`, decision CID `258b6ed7` invariant, {KNOWN:1, UNKNOWN:4}, gate CLOSED. Frontier target stays `meta/llama-3.3-70b-instruct`.

## 4. Carry-forward & boundary

W123–W137 caps STAND unless new evidence changes them. W137's bimodality cap is REVISED by W138's Lane α: the repaired field is NOT inherently no-headroom — a real `0<A1<1` band spanning ≥2 modes / ≥3 families exists once A1 is measured as a rate and the hardness knob is swept. No version bump; no PyPI; `coordpy/__init__.py` untouched. `COO-9` lead.
