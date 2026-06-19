# W122 — milestone summary (matched-family multi-seed closure + mechanism probe + stronger-model gate)

Milestone W122 / `COO-9` (sibling). `ultracode` OFF. No version bump; no PyPI;
`coordpy/__init__.py` untouched. Graph refreshed at START (HEAD `6b05ccd2`) and END.

## One line

W122 attacked the one live caveat on the matched-family ICPC contrast — single seed each
side — by running the pre-committed paired-seed closure (seed 120002, then the earned
tiebreaker seed 120003) on BOTH the W120 resistant and the W121 exposed 30-slices under a
SYMMETRIC rule locked before any NIM. **FINAL 3-seed outcome:
`AMBIGUOUS_THIRD_PAIRED_SEED_EARNED` (B4) — closure NOT achieved.** 3-seed means RESISTANT
+4.44 pp (in the 3.34–5.00 gap, out of band) / EXPOSED +8.89 pp (out of band); neither field
clean (no `PASS_MECHANISM_DRIVEN` seed on either side, any seed) ⇒ B1/B2/B3 all off ⇒ B4.
Seed 120003 spiked **+10.00 pp on BOTH fields** (`PASS_NON_MECHANISM_DRIVEN`), so the 2-seed
"resistant null vs exposed popped" asymmetry **dissolved** — the matched contrast is
unresolvable at n=30 (±10 pp variance from ~3 rescues). No 4th seed (rule caps at three). In
parallel, the strongest non-reflexion mechanism (M3) was KILLED NIM-free (ICPC secret-grading
denies its differentiator ⇒ ceiling mechanism-robust) and the stronger-model gate was
confirmed STRUCTURALLY CLOSED. Both $0 NIM. **W89 + W105 remain the only two retirements; W122
adds none.**

## Three lanes

* **Lane α (matched-family multi-seed closure — main empirical lane):** reused the EXACT
  W120 resistant 30-slice (CID `01bf9ef8…`) + W121 exposed 30-slice (CID `32d15db5…`), both
  re-derived NIM-free == provenance == pilot guard; ran Maverick × BOTH × seeds 120002 +
  120003, 1×30×K5/field/seed = **1320 NIM calls total** (660/seed). Per-seed: RESISTANT
  120002 +3.33 FAIL (8/9; MLB-2 8.7%) → 120003 **+10.00 PASS_NON_MECHANISM_DRIVEN** (9/9;
  MLB-2 16.7%; 3 rescues / 0 regr; merkle `adf55ff9…`); EXPOSED 120002 **+13.33
  PASS_NON_MECHANISM_DRIVEN** (9/9; MLB-2 28.6%; 4 rescues / 0 regr; A1 dipped to 20%) →
  120003 **+10.00 PASS_NON_MECHANISM_DRIVEN** (9/9; MLB-2 18.5%; 3 rescues / 0 regr; merkle
  `d88d025f…`). 3-seed means RESISTANT **+4.44** (gap, out of band) / EXPOSED **+8.89** (out
  of band); neither all-clean per seed ⇒ **B4 `AMBIGUOUS_THIRD_PAIRED_SEED_EARNED`**
  (code-emitted; `caveat_closed=false`). Closure NOT achieved: the resistant field also spiked
  (+10.00 on 120003), so n=30 is too noisy to resolve the contrast. The symmetric closure rule
  + ambiguity band were LOCKED in `docs/RUNBOOK_W122.md` before any NIM; the driver was fixed
  first so the 3-seed aggregate gathers ALL prior seeds keyed by seed (NIM-free verified). No
  4th seed.
* **Lane β (same-family different mechanism — M3):** NIM-free audit
  (`audit_icpc_mechanism_signal_v1`) over the real 660-call W120+W121 sidecars (240
  reflexion turns) ⇒ `m3_exclusive_signal_fraction = 0.000` < 0.33 ⇒ `KILL_M3_LANE_NIM_FREE`.
  On official ICPC the hidden oracle is SECRET token-diff (no expected value revealed), so
  M3's expected/actual differentiator is structurally absent; the same-family ceiling is
  **mechanism-robust**, not merely reflexion-specific. $0 NIM.
* **Lane γ (stronger-model gate / graphify / truth):** re-checked primary cutoffs LIVE
  (incl. direct DeepSeek-V4 card PDF re-fetch = NO cutoff); the resistant field
  (2024-11..2025-11) is anchored to Maverick Aug-2024, so any stronger model (later/UNKNOWN
  cutoff) would find the field EXPOSED, not resistant ⇒ NOT resistant-certifiable. Gate does
  NOT open; Maverick stays the unique matched-family target; `{KNOWN:1, UNKNOWN:4}`;
  decision CID `258b6ed7…` invariant. $0 NIM.

## The 2×3 (after W122 seed 120003) — FINAL, 3-seed

| field | 120001 | 120002 | 120003 | 3-seed mean | null band? |
|---|---|---|---|---|---|
| RESISTANT (W120 family) | +0.00 FAIL | +3.33 FAIL | **+10.00 PASS_NON_MECH** | **+4.44** | ❌ out (3.34–5.00 gap) |
| EXPOSED (W121 family) | +3.33 FAIL | **+13.33 PASS_NON_MECH** | **+10.00 PASS_NON_MECH** | **+8.89** | ❌ out of band |

The retirement-grade margins (+5.56 / +7.00) appear ONLY on HumanEval-family (easy) code. On
official ICPC BOTH fields are now out of band but BOTH are non-mechanism-driven (no
`PASS_MECHANISM_DRIVEN` seed anywhere; MLB-2 < 33% on every non-FAIL seed): each shows
occasional large rescue-concentrated spikes (resistant +0/+3.33/+10; exposed +3.33/+13.33/+10).
The 2-seed "resistant null vs exposed popped" asymmetry **dissolved** at the 3rd seed. The
matched control is **NOT resolved at n=30** — the per-field B−A1 estimator is too noisy
(±10 pp from ~3 rescues) ⇒ **B4**; W123 = larger n PER FIELD.

## Entitlement after W122 (FINAL, 3-seed)

* **IS** (unchanged): TWO confirmed retirements — W89 + W105, Llama-3.3-70B @ 70B
  contamination-EXPOSED-HumanEval-family.
* **NEW (3-seed B4):** the literal single-seed caveat is retired (3 seeds each side) but
  **replaced by a small-n-variance limitation** — the matched contrast is unresolvable at
  n=30 (3-seed means resistant +4.44 / exposed +8.89, both out of band, neither clean ⇒ B4).
  The ceiling is **mechanism-robust** (M3 cannot earn an ICPC probe) and the stronger-model
  gate is **structurally closed**. The 3-seed data **further undercuts** a clean contamination
  read: contamination-RESISTANT code spiked +10.00 just like exposed code ⇒ the spikes are
  sampling variance, not exposure.
* **IS NOT:** a third retirement (W122 adds none), a clean win on either field, a closure of
  the contrast, contamination proven OR refuted, OR a "resistant stays null" result (it did
  not — resistant spiked +10.00 on seed 120003). Both +10/+13 spikes are
  `PASS_NON_MECHANISM_DRIVEN` (rescue-concentrated; not mechanism wins). Still NOT cross-class
  / MBPP-family / cross-modal / "context solved".

## Files / tests / discipline

* New: `coordpy/coordpy_icpc_paired_seed_closure_v1.py`;
  `scripts/run_w122_paired_seed_pilot.py`, `scripts/run_w122_mechanism_signal_audit_v1.py`;
  `tests/test_w122_paired_seed_closure_v1.py` (17 tests); `docs/RUNBOOK_W122.md`,
  `docs/RESULTS_W122_PAIRED_SEED_CLOSURE_V1.md`, `docs/RESULTS_W122_MECHANISM_AUDIT_V1.md`,
  this summary, `docs/CONTAMINATION_CONTROL_FRAMING_W122_V1.md`,
  `docs/FRONTIER_RELEVANCE_AUDIT_W122_V1.md`; `results/w122/`. Fixed a W121 doc CID typo
  (`f7cdc917`→`01bf9ef8`).
* Tests: 17 W122 + reuse-chain regression (W121/W120/W114) pass. 32nd consecutive
  preflight/earn-discipline validation. Explicit-import-only; `coordpy/__init__.py`
  untouched; no version bump; no PyPI.
* `COO-9` stays the lead path (an ambiguous-then-tiebreaker result does not force a
  code-line change).
* **W123** = the 3-seed aggregate is **B4** ⇒ register the residual ambiguity; W123 = accept
  the bounded HumanEval-family ceiling OR escalate to a larger n PER FIELD (≥100/field, not
  more seeds at n=30; NO 4th seed), OR a primary-KNOWN stronger-than-Maverick model on BOTH
  battlefields if one opens. `COO-9` stays lead.

## FINAL 3-SEED OUTCOME — `AMBIGUOUS_THIRD_PAIRED_SEED_EARNED` (B4)

Seed 120003 (660 NIM calls) produced **+10.00 pp on BOTH fields** (`PASS_NON_MECHANISM_DRIVEN`;
resistant 9/9, MLB-2 16.7%, merkle `adf55ff9…`; exposed 9/9, MLB-2 18.5%, merkle `d88d025f…`;
3 net rescues / 0 regressions each). 3-seed means: RESISTANT **+4.44** (in the 3.34–5.00 gap,
out of the null band) / EXPOSED **+8.89** (out of band); `resistant_in_null_band=false`,
`exposed_in_null_band=false`, both `shows_margin=false`, `all_seeds_clean_pass=false` ⇒
code-emitted branch **`AMBIGUOUS_THIRD_PAIRED_SEED_EARNED` (B4)**, `caveat_closed=false`. This
is B4 AFTER the 3rd seed (the rule caps at three; **no 4th**). Closure NOT achieved: the
resistant field also spiked, so the contrast is unresolvable at n=30. **W89 + W105 STAND; W122
adds no retirement.** W123 = accept the bounded claim or escalate to larger n PER FIELD.
Verdict artifact:
`results/w122/paired_seed/w122_paired_seed_120003_20260531T184401Z/paired_seed_closure_verdict.json`.
