# RESULTS W139 — per-tier band recalibration + capability-matched witness compiler + cross-tier mechanism validation

**Status: Lane α COMPLETE; Lane β §7a dev FAIL (SPAN) ⇒ $0 eval, $0 frontier; NO third retirement.** Executes `docs/RUNBOOK_W139.md` (locked before any NIM). `ultracode` OFF. `coordpy.__version__ == "0.5.20"`; `coordpy/__init__.py` untouched; no PyPI. W89+W105 remain the only two retirements.

## 0. The reframe that drives W139

W138 ESTABLISHED a parser-neutral, resistant-by-construction `0<A1<1` band at 70B, and that on the frontier anchor the exact-oracle complexity witness beats self-consistency (+40pp held-out) AND blind reflexion (+25pp) across ≥3 complexity families. But §7b cross-tier robustness FAILED on two named modes: (1) **capability gate** — the witness HURTS the 8B (C0−A1 = −25; it writes broken efficient code below its own self-consistency baseline); (2) **per-ability band** — `llama-3.1-70b` was tested on the ANCHOR-calibrated band where it is near-saturated, so reflexion already maxed it and the witness only tied (+0). W139 attacks both: **(α) per-MODEL band recalibration** (each tier at its own p≈0.5 — Fluid Benchmarking arXiv:2509.11106) and **(β) a capability-MATCHED controller** (route APPLY-witness vs KEEP-self-consistency by *measured* generator capability, so the witness never hurts a weak generator — R2′/R5′; arXiv:2404.17140) + a **large-probe counterexample witness** reviving the W138-dead 2nd mode.

## 1. Lane α — per-tier band recalibration (COMPLETE)

`coordpy.per_tier_band_calibration_v1` measures A1-as-a-RATE at **every** tier (the W138/W137 engine measured A1 only at the anchor). Ladder V2 = `llama-3.1-8b` (small) / `llama-3.1-70b` (mid) / `llama-3.3-70b` (strong anchor). Calibration (4 cells × 3 tiers, n_cal=4, K=4; `per_tier_calibration_cid = 72c4a4f1…`; `slate_fingerprint_cid = 6ad6771b…`; 221 NIM):

| cell | mode | small a1 | mid a1 | strong a1 |
|---|---|---|---|---|
| count_pairs_sum_le_t@20000 | COMPLEXITY | 0.75 | **1.00 (saturated)** | 0.75 |
| count_pairs_sum_le_t@50000 | COMPLEXITY | 0.50 | **1.00 (saturated)** | 0.50 |
| subarrays_sum_and_range@1500 | HIDDEN_EDGE | 0.50 | 0.75 | 0.25 |
| subarrays_sum_and_range@4000 | HIDDEN_EDGE | 0.00 | 1.00 | 0.25 |

**The decisive Lane-α finding:** the **mid tier (`llama-3.1-70b`) is SATURATED on the complexity family (a1=1.00 at both knobs) — it is STRONGER than the anchor (`llama-3.3-70b`) on this field**, so it has *no complexity headroom*. This ROOT-CAUSES the W138 "mid +0": the mid had nothing to gain because it already solves the complexity cells. Per-tier bands: small → {count_pairs@50000, subarrays@1500}; mid → {subarrays@1500 only}; strong → {count_pairs@50000, subarrays@1500}. **Witness-usability (the capability prior):** strong **0.50 → ELIGIBLE**; small **0.0 → ineligible**; mid **0.0 → ineligible** (measured on its HIDDEN_EDGE band cell; see §2 caveat). So only the anchor is witness-eligible ⇒ the controller will KEEP on small + mid.

## 2. Lane β — capability-matched cross-tier mechanism validation (§7a dev)

Dev bench (`scripts/run_w139_cross_tier_bench_v1.py`, n_per_cell=2, diagnostics at every tier; 210 NIM; per-tier slices from the 139_2xx seeds). Arms K=5 same-budget. **Cm** = capability-matched controller (LEAD). **C0** = blind-apply complexity witness (W138's arm). **Nb** = large-probe counterexample (2nd-mode revival).

| tier | elig | n | A0 | A1 | B0 | **Cm** | Cm−A1 | Cm−B0 | C0(blind) | Nb(largeCE) |
|---|---|---|---|---|---|---|---|---|---|---|
| **small (8B)** | False | 4 | 25 | 25 | **0** | 25 | **+0** | +25 | 25 | 25 |
| **mid (3.1-70B)** | False | 2 | 0 | 50 | 100 | 100 | +50* | +0 | 50 | 0 |
| **strong (anchor)** | True | 4 | 25 | 25 | 75 | **100** | **+75** | **+25** | 50 | 75 |

*the mid +50 is a KEEP-tier SAMPLING ARTIFACT — see §3.

**What the controller DID (real, by-design):**
- **8B KEEP protects against harm — including reflexion's.** Blind reflexion B0 *hurts* the 8B (A1=25 → B0=**0**: its reject-bit drives the 8B into worse code). Cm KEEPs (all `KEEP_PLAIN` actions) ⇒ Cm ≡ A1 = 25 (**Cm−A1 = +0, non-negative**; Cm−B0 = +25). This eliminates BOTH the W138 witness harm (−25) AND blind reflexion's harm. Validates R2′/R5′ (a verifier/critic the generator cannot act on must be suppressed, not applied).
- **Anchor APPLY + 2nd-mode revival.** At the anchor Cm = 100% (vs A1=25, B0=75). The controller correctly routes per problem: complexity cells → witness fires only when needed (`PLAIN_NO_WITNESS` when the model's code is already correct), HIDDEN_EDGE cells → `WITNESS_APPLY` via the **large-probe counterexample** (`Nb = 75%` at the anchor, vs the W138 small-probe N0 = 0 — the dead 2nd mode is revived live, also shown $0 in the build self-test).

**§7a DEV GATE — FAIL (`SPAN_FAIL`).** Lead Cm at the anchor: Cm−A1 = **+75.0**, Cm−B0 = **+25.0** (both clear the +3.33 margin), all_structural, no_net_regression. BUT rescues over blind reflexion (Cm passes ∧ B0 fails) span **1 mode / 1 family** (`COMPLEXITY_BLIND` / `count_pairs_sum_le_t`) < the required ≥2 modes OR ≥3 families ⇒ **`SPAN_FAIL`**. Per the locked discipline: **$0 eval, $0 frontier, NO third retirement.** The structural cause is the same one W138's first dev run noted: **blind reflexion B0 is already strong on this band** (it passed BOTH anchor HIDDEN_EDGE problems), so the witness's incremental rescues over B0 concentrate where B0 fails (the complexity side) — they do not span ≥2 modes.

## 3. §7b / cross-tier VERDICT — NOT EARNED

The bench's `two_tier_same_sign` flag computed TRUE (anchor +75 positive; mid +50 "positive"; all non-negative), but this is **an artifact and did not gate-in an earn** (the §7a SPAN gate failed first):
- **The mid Cm−A1 = +50 is KEEP-tier SAMPLING NOISE.** The mid is witness-INELIGIBLE, so Cm KEEPs (all `KEEP_PLAIN` ≡ plain self-consistency). Cm and A1 are then the SAME policy with DIFFERENT i.i.d. draws; at n=2, Cm passed 2/2 where A1 passed 1/2 — pure variance, not a mechanism gain. The mid's `Nb = 0` confirms it has no real witness benefit. Corrected for the artifact, the mid's true delta is ~0 (KEEP).
- **No robust SECOND positive tier exists on this ladder.** Only the anchor (strong) is genuinely positive. The structural reason (the real W139 result): **no second tier has BOTH (a) mechanism headroom AND (b) witness-usability** — the 3.1-70B is saturated on complexity (no headroom; it is *stronger* than the anchor here) and the 8B is capability-bound (no usability). A capability-matched controller, however good, can only earn at the single tier that has both.

⇒ **`W139-L-CAPABILITY-MATCHED-NON-NEGATIVITY-ACHIEVED-BUT-WITNESS-GAIN-STAYS-ANCHOR-CONCENTRATED`.** W139 ELIMINATES the W138 harm (non-negativity on all tiers, incl. protection from reflexion harm) and REVIVES the 2nd mode (large-probe), but the gain stays anchor-concentrated; cross-tier robustness in the strong sense (positive on ≥2 tiers with span) is NOT earned. **`W89 (+5.56) + W105 (+7.00)` STAND as the only two retirements.**

## 4. Lane γ — research + architecture-requirements + frontier gate

- Primary sources wired into the gates (RUNBOOK §9): **Fluid Benchmarking arXiv:2509.11106** (per-ability bands — the central Lane-α construction; it predicted the mid-saturation finding), arXiv:2404.17140 (small models need a strong verifier — predicts the 8B harm; supports KEEP-vs-APPLY), arXiv:2310.01798 (self-correction can degrade), arXiv:2505.13553 + arXiv:2408.13745 (public/trial-test pass as the commit signal), arXiv:2408.03314 (difficulty-conditioned allocation), metabench arXiv:2407.12844 / tinyBenchmarks arXiv:2402.14992 (IRT). **Honest flag:** the capability-gate ("bigger models repair more") is OUR W138/W139 empirical finding, NOT literature-backed.
- `docs/ARCHITECTURE_REQUIREMENTS_W139_V2.md` — refines R2′ (verifier matched to generator capability), R4′ (per-ability difficulty — the mid-saturation is the textbook case), R5′ (capability-aware allocation — the KEEP that achieves non-negativity).
- Stronger-model gate re-derived: `NO_CERTIFIABLE_STRONGER_MODEL`, decision CID `258b6ed7` invariant, {KNOWN:1, UNKNOWN:4}, CLOSED. Frontier target stays `meta/llama-3.3-70b-instruct`. Not launched ($0 — §7a failed).

## 5. Outcome — what W139 establishes (and what it does not)

**ESTABLISHED (REVISES/EXTENDS W138):**
- **The capability-matched controller achieves NON-NEGATIVITY on every tier** — the W138 8B −25 harm is eliminated, AND the controller protects the 8B from blind reflexion's harm (B0 25→0). Validates R2′/R5′: route the verifier ON only where the generator can use it; KEEP elsewhere.
- **Per-tier recalibration root-causes the W138 mid +0**: `llama-3.1-70b` is SATURATED on the complexity family (a1=1.00) — *stronger than the anchor* — so it had no headroom for any mechanism (R4′: the informative band is per-ability).
- **The large-probe counterexample REVIVES the W138-dead 2nd mode** (Nb=75% at the anchor; W138 N0=0 was a `SMALL_PROBE_TOKEN_CAP=400` firing artifact — also proven $0).

**NOT ESTABLISHED (honest boundaries):**
- **§7a SPAN_FAIL** ⇒ no frontier earn, no third retirement. Blind reflexion is strong enough on this band that the witness's rescues over it span only 1 mode.
- **No robust 2nd positive tier** — the apparent mid +50 is KEEP-tier sampling noise (n=2); no tier on this ladder has both headroom AND witness-usability.
- **Tiny n (2–4 per tier)** — these are noisy DEV signals, not robust measurements (HOW_NOT_TO_OVERSTATE).

## 6. Carry-forward & boundary

W123–W138 caps STAND. W138's "capability-gated + anchor-specific" cap is REFINED by W139: the harm is *fixable* (capability-matched KEEP → non-negativity), but the *gain* stays anchor-concentrated for a structural reason (no second tier with both headroom and usability). **W140** = a DENSER / different model ladder (a tier between 8B and 70B with both headroom and usability), or a mechanism that does not depend on the single-tier headroom∧usability coincidence, or a primary-KNOWN stronger model when the `258b6ed7` gate opens. No version bump; no PyPI; `coordpy/__init__.py` untouched. `COO-9` lead.
