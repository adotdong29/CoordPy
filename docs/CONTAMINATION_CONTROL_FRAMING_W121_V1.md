# Contamination-control framing — W121 (matched EXPOSED ICPC control)

## What W121 tested

The contamination confound asks: *do the W89/W105 same-budget multi-agent reflexion
retirements (+5.56 / +7.00 pp on HumanEval-family) reflect a real capability gain, or do
they ride on benchmark contamination (the model has seen HumanEval-family solutions)?*
W110–W120 read the resistant-code nulls as STRENGTHENING the contamination reading — but
that read was **confounded**: the exposed retirements are on EASY HumanEval-family code
while the resistant nulls are on HARD ICPC / LiveCodeBench code, so "exposed vs resistant"
across the programme was entangled with "easy vs hard" and "HumanEval-family vs other".

**W121 removes the confound the only clean way: it holds the family + difficulty FIXED
(official ICPC, same regional series) and flips ONLY exposure.** It runs the SAME model
(Maverick), SAME grader, SAME evaluator, SAME K/budget on a matched EXPOSED ICPC
battlefield (pre-Aug-2024 editions of the SAME regionals W120 used) and contrasts it with
the W120 resistant ICPC result.

## The clean within-family within-model contrast

| | EXPOSED (W121) | RESISTANT (W120) |
|---|---|---|
| model | Maverick | Maverick |
| family / grader / evaluator | official ICPC / secret-case oracle / W108 gates | same |
| cutoff side | ≤ 2024-08-31 (EXPOSED) | > 2024-08-31 (RESISTANT) |
| A0 pass@1 | **6.67 %** (2/30) | 20.00 % (6/30) |
| A1 pass@1 | 26.67 % (8/30) | 23.33 % (7/30) |
| B pass@1 | 30.00 % (9/30) | 23.33 % (7/30) |
| **B − A1** | **+3.33 pp** (FAIL) | **+0.00 pp** (FAIL) |
| MLB-1 invocation | 93.33 % PASS | 83.33 % PASS |
| MLB-2 rescue | 25.00 % FAIL | 8.00 % FAIL |
| Phase-2 gates | 8/9 | 6/9 |

**The matched-family exposure flip did NOT reproduce the retirement-grade margins.** On
the SAME official ICPC family at comparable difficulty, flipping only exposure moved B−A1
from +0.00 (resistant, FAIL) to **+3.33 pp (exposed, FAIL)** — both within the
pre-committed ±3.34 pp null band, both below the mechanism-load-bearing floor (MLB-2 <
33 %). The +5.56 / +7.00 pp clean-PASS HumanEval-family margins were **not** reproduced by
exposure within ICPC.

## The two hypotheses, and which W121 favors

* **H_contamination** (margins ride on exposure): predicts EXPOSED ICPC should margin
  (like W89/W105) and RESISTANT ICPC should not. **Prediction FAILED** — exposed ICPC is
  +3.33 pp (FAIL), not a clean margin.
* **H_difficulty/family-ease** (margins ride on HumanEval-family ease/structure, which
  affords mechanism headroom): predicts BOTH exposed and resistant ICPC should fail (ICPC
  is hard regardless of exposure). **Prediction MATCHED** — both sides FAIL (+0.00 / +3.33).

W121 therefore **WEAKENS the strong contamination reading** and **implicates
difficulty/family-ease** as the driver of the W89/W105 margins. The "exposed-vs-resistant
dissociation is confounded with benchmark family/difficulty" loophole is now addressed in
the ICPC setting: with family + difficulty held fixed, exposure produced at most a
sub-floor nudge.

## The honest residual (what W121 does NOT license)

* **NOT "contamination refuted."** There is a faint, contamination-CONSISTENT gradient:
  exposed B−A1 +3.33 > resistant +0.00; exposed reflexion-rescue 25 % > resistant 8 %.
  Direction matches H_contamination; magnitude (one net problem: +rsamistake +isbnconversion
  −icouldhavewon; sub-floor MLB-2) does not. So contamination is not excluded as a MINOR
  additive contributor — it is demoted from "the dominant driver" to "at most a small one."
* **NOT a clean exposed win / not a third retirement.** The exposed pilot FAILED its gates
  (8/9; MLB-2 25 % < 33 %). W121 adds NO retirement; **W89 (+5.56) + W105 (+7.00) remain
  the only two.**
* **Single seed each side.** +3.33 pp is one net problem at n=30, one seed (120001). A
  paired seed is NOT earned (the result is on the null side of the pre-committed band, and
  chasing a one-problem edge is the W106 margin-cap anti-pattern) — but a future paired
  seed on BOTH battlefields is the natural next tightening (W122).
* **Difficulty comparability is empirically supported, not assumed.** Exposed A0 (6.67 %)
  is if anything LOWER than resistant A0 (20.00 %), so the exposed slice is not easier —
  the +3.33 is not a difficulty/ease artifact, and the matched-difficulty design held.
* **ICPC ≠ HumanEval family.** W121 controls family WITHIN ICPC; the inference to the
  HumanEval-family retirements (that their margin is ease/structure, not exposure) is a
  cross-family extrapolation, still observational.

## Status

Contamination-confound: **WEAKENED** (first matched-family within-model exposure control;
exposure within ICPC did not reproduce the margin) — **NOT refuted** (faint sub-floor
exposure-consistent gradient remains; single seed). The bounded ceiling **HARDENS** and is
re-pointed: the two retirements are best read as **HumanEval-family-(ease/structure)-
specific at 70B**, with exposure a possible minor contributor rather than the dominant
driver. W89 + W105 STAND; resistant-and-exposed ICPC superiority is 0 clean across FIVE
ICPC-family cells now (W120 resistant +0.00; W121 exposed +3.33; both FAIL) plus W108/W110/
W113.
