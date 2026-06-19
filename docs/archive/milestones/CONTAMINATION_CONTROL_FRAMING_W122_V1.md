# Contamination-control framing — W122 (matched-family multi-seed closure)

## What W122 tested

W121 ran the matched within-family within-model exposure control (same official ICPC
package family, same Maverick, same grader/evaluator/K, opposite cutoff sides) and found
EXPOSED +3.33 pp vs RESISTANT +0.00 pp — both within the ±3.34 pp null band, both FAIL ⇒
the matched-family exposure flip did NOT reproduce the retirement-grade HumanEval-family
margins ⇒ contamination-confound **WEAKENED**, difficulty/family-ease implicated. The one
live caveat: **single seed each side (120001)**.

**W122 ran the pre-committed paired-seed closure (seed 120002, then the earned tiebreaker seed
120003) on BOTH fields under a symmetric closure rule locked before any NIM.** The FINAL
3-seed aggregate is **`AMBIGUOUS_THIRD_PAIRED_SEED_EARNED` (B4)** — closure NOT achieved: at
the 3rd seed the RESISTANT field also spiked (+10.00 pp `PASS_NON_MECHANISM_DRIVEN`), so the
2-seed "resistant null vs exposed popped" asymmetry dissolved and the matched contrast is
revealed to be unresolvable at n=30. This is the targeted, mechanically-stopped attack on the
one named caveat, honest about an ambiguous outcome rather than forcing a closure
(`docs/FRONTIER_RELEVANCE_AUDIT_W122_V1.md`). No 4th seed.

## The multi-seed within-family within-model contrast (FINAL, 3 seeds)

| | EXPOSED (W121 family) | RESISTANT (W120 family) |
|---|---|---|
| model / family / grader / evaluator | Maverick / official ICPC / secret-case oracle / W108 gates | same |
| cutoff side | ≤ 2024-08-31 (EXPOSED) | > 2024-08-31 (RESISTANT) |
| seed 120001 B−A1 | +3.33 pp (FAIL) | +0.00 pp (FAIL) |
| seed 120002 B−A1 | **+13.33 pp (PASS_NON_MECHANISM_DRIVEN)** | +3.33 pp (FAIL) |
| seed 120003 B−A1 | **+10.00 pp (PASS_NON_MECHANISM_DRIVEN)** | **+10.00 pp (PASS_NON_MECHANISM_DRIVEN)** |
| **3-seed mean B−A1** | **+8.89 pp** (out of band) | **+4.44 pp** (out of band, in the 3.34–5.00 gap) |
| MLB-2 rescue (s1/s2/s3) | 25.0% / 28.6% / 18.5% (all FAIL <33%) | 8.0% / 8.7% / 16.7% (all FAIL) |
| A0 (s1/s2/s3) | 6.67% / 6.67% / 6.67% | 20.00% / 16.67% / 23.33% |

At the 3rd seed BOTH fields produced a +10.00 pp `PASS_NON_MECHANISM_DRIVEN` spike (3 rescues
/ 0 regressions each). The exposed 3-seed mean is +8.89 (out of band); the resistant 3-seed
mean rose to +4.44 (also out of band, in the 3.34–5.00 gap). Neither field is clean (no
`PASS_MECHANISM_DRIVEN` seed anywhere) ⇒ **`AMBIGUOUS_THIRD_PAIRED_SEED_EARNED` (B4)**. The
resistant field is **NOT** reliably null — it spikes just like the exposed field.

## The two hypotheses, at 3 seeds

* **H_contamination** (margins ride on exposure): predicts EXPOSED ICPC should margin while
  RESISTANT ICPC should not. **Further undercut at 3 seeds.** The exposed mean (+8.89) is
  still above the resistant mean (+4.44), so a faint exposure-CONSISTENT ordering remains —
  BUT the RESISTANT field ALSO produced a +10.00 pp spike (seed 120003), so
  contamination-RESISTANT code margins just like exposed code on the lucky seed. If exposure
  drove the spikes, resistant code should not spike; it did. The spikes are
  non-mechanism-driven (MLB-2 < 33% everywhere) and seed-dependent ⇒ sampling variance, not a
  clean exposure effect.
* **H_difficulty/family-ease + small-n variance** (margins ride on HumanEval-family
  ease/structure; ICPC B−A1 is noisy at n=30): predicts BOTH ICPC fields fail to show a clean
  mechanism margin regardless of exposure, with large rescue-driven swings at small n.
  **Matched at 3 seeds**: neither field has a `PASS_MECHANISM_DRIVEN` seed; both swing ±10 pp
  across seeds on ~3 rescues.

W122-at-3-seeds is therefore **genuinely ambiguous (B4)**: it neither confirms a clean
exposure margin (both fields spike, non-mechanism-driven) nor establishes a clean resistant
null (resistant spiked too). The honest finding is that the matched contrast is **unresolvable
at n=30** — the per-field estimator is dominated by rescue-concentration variance. The 3rd
seed answered the 2-seed question ("was +13.33 a fluke or a real exposed tendency?"): it was a
**seed-specific rescue spike** — the resistant field reproduced the same +10.00 spike, so it
is not an exposed-specific tendency.

## The honest residual (what W122-at-3-seeds does NOT license)

* **NOT a closure / NOT "contamination weakened, multi-seed".** The 3-seed aggregate is
  ambiguous (B4), not B1. It would be wrong to claim the matched-family null survived
  multi-seed — neither field stayed in the null band (resistant +4.44, exposed +8.89).
* **NOT a third retirement / NOT a clean margin on either field.** Every non-FAIL seed is
  `PASS_NON_MECHANISM_DRIVEN` (MLB-2 < 33%; rescue-concentrated) — the W109/W112
  exposed-control signature, now seen on BOTH fields, NOT a mechanism win. **W89 + W105 remain
  the only two retirements.** W122 adds none.
* **NOT "contamination re-strengthened".** B2 explicitly did NOT fire (it requires a clean
  exposed margin AND a resistant null; resistant is +4.44, not null). The exposed > resistant
  ordering is real but faint, non-mechanism-driven, and undercut by the resistant field's own
  +10.00 spike — calling the confound "strengthened" would over-read seed-dependent noise.
* **NOT a "resistant stays null" result.** The resistant field spiked +10.00 on seed 120003;
  the bounded-ceiling reading of W120/W121 ("resistant ICPC is null") does NOT hold across
  seeds at n=30. What holds is the weaker, robust claim: no CLEAN mechanism margin on either
  field, any seed.
* **The 3rd seed was the pre-committed tiebreaker, not seed-chasing**, and is the LAST seed:
  the rule caps at three (no 4th). B4-after-the-3rd-seed routes W123 to larger n PER FIELD.
* **ICPC ≠ HumanEval family.** W122 controls family WITHIN ICPC; the inference to the
  HumanEval-family retirements is still a cross-family extrapolation, observational.
* **Mechanism-robust still holds.** The ICPC ceiling is not a reflexion-prompt artifact: the
  strongest non-reflexion mechanism (M3) cannot even earn a probe on ICPC (secret-grading
  denies its differentiator; `RESULTS_W122_MECHANISM_AUDIT_V1.md`).

## Status (FINAL, 3-seed)

Contamination-confound: **AMBIGUOUS at 3 seeds (B4) — unresolved, and a clean contamination
read is further undercut.** Both fields landed out of band but BOTH on rescue-concentrated
non-mechanism-driven spikes (resistant 3-seed mean +4.44; exposed +8.89; no
`PASS_MECHANISM_DRIVEN` seed anywhere). The resistant field's own +10.00 spike (seed 120003)
shows contamination-RESISTANT code margins just like exposed code, so the spikes are sampling
variance at n=30, not exposure — the matched contrast cannot be resolved at this n. The W121
single-seed "weakened" read is NOT multi-seed-confirmed (the exposed field did not stay flat),
and contamination is NOT established (the resistant field did not stay null); the faint
exposed > resistant ordering (+8.89 > +4.44) is exposure-CONSISTENT but non-mechanism-driven
and noisy. The bounded ceiling is **mechanism-robust** (M3 cannot earn an ICPC probe) and the
stronger-model gate is **structurally closed**. **W89 + W105 STAND; W122 adds no retirement.**
W123 = accept the bounded claim or escalate to larger n PER FIELD (≥100/field; no 4th n=30
seed).
