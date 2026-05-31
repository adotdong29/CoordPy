# RESULTS — W122: matched-family multi-seed CLOSURE on official ICPC (paired seed on BOTH fields)

Milestone W122 / `COO-9`. `ultracode` OFF. No version bump; no PyPI; `coordpy/__init__.py`
untouched. Graph refreshed at start (HEAD `6b05ccd2`) and end (see milestone summary).

## Headline

W120 (RESISTANT ICPC, +0.00 pp FAIL) and W121 (matched EXPOSED ICPC, +3.33 pp FAIL), both
single-seed (120001), landed within the ±3.34 pp null band ⇒ W121 read `CONFOUND_WEAKENS`.
The ONE remaining live caveat was **single seed each side**. W122 ran the pre-committed
paired-seed closure on BOTH the EXACT W120 resistant 30-slice and the EXACT W121 exposed
30-slice (only the seed changed), under a SYMMETRIC closure rule locked before any NIM.
**FINAL 3-seed outcome (code-emitted): `AMBIGUOUS_THIRD_PAIRED_SEED_EARNED` (B4) — closure
NOT achieved.** Seed 120002 was ambiguous (resistant null +1.67 / exposed out-of-band +8.33),
earning the pre-committed 3rd paired seed (120003). Seed 120003 then produced **+10.00 pp on
the RESISTANT field** and **+10.00 pp on the EXPOSED field** (BOTH `PASS_NON_MECHANISM_DRIVEN`),
pushing the **3-seed means to RESISTANT +4.44 pp (in the 3.34–5.00 gap, OUT of the null band)
and EXPOSED +8.89 pp (out of band)**. Neither field is clean (no `PASS_MECHANISM_DRIVEN` seed
on either side, any seed), so B1/B2/B3 are all off ⇒ **B4**. The single-seed caveat is **NOT
closed by a clean resolution**: three seeds did not collapse the contrast — instead the 2-seed
"resistant null vs exposed popped" asymmetry **dissolved** (the RESISTANT field also spiked
+10.00), revealing that at n=30 the per-field B−A1 is too noisy (±10 pp from a few rescues) to
resolve. No 4th seed (the rule caps at three). **W89 + W105 remain the only two retirements;
W122 adds none.**

* **Lane α (paired-seed closure): 3-seed AMBIGUOUS (B4) — closure NOT achieved.** 3-seed
  means RESISTANT +4.44 (gap, out of band) / EXPOSED +8.89 (out of band); neither field clean
  ⇒ B1/B2/B3 all off ⇒ `AMBIGUOUS_THIRD_PAIRED_SEED_EARNED`. Seed 120003 spiked +10.00 on
  BOTH fields (`PASS_NON_MECHANISM_DRIVEN`), so the resistant field is **not** reliably null —
  the contrast is unresolvable at n=30. No 4th seed; W123 = larger n PER FIELD or accept the
  bounded claim.
* **Lane β (different mechanism): KILLED NIM-free.** See `RESULTS_W122_MECHANISM_AUDIT_V1.md`
  — the ICPC secret-grading regime structurally denies M3 its differentiator ⇒
  `m3_exclusive_signal_fraction = 0.000` < 0.33 ⇒ the same-family ceiling is
  **mechanism-robust**, not merely reflexion-specific. $0 NIM.
* **Lane γ (stronger model): gate STRUCTURALLY CLOSED.** No reachable stronger-than-Maverick
  model has a primary-KNOWN cutoff ≤ ~Aug-2024, so none can be RESISTANT-certified on the
  matched battlefields. Maverick stays the unique matched-family target. $0 NIM.

## Lane α — the paired-seed result (Maverick × BOTH fields, seed 120002)

Runbook `docs/RUNBOOK_W122.md` LOCKED before any NIM (built from HEAD `6b05ccd2`). Both
30-slices CID-guarded and re-derived NIM-free == provenance == pilot guard
(resistant `01bf9ef8…`, exposed `32d15db5…`). The ONLY change vs W120/W121 is the seed.
Maverick × each 30-slice, seed 120002, 1×30×K=5/field = **660 NIM calls** (resistant wall
4778 s + exposed wall 3942 s ≈ 145 min). Grader = official secret cases (token-diff oracle;
NO LLM judge); reflexion feedback = public samples + judge verdict + stderr only.

### Per-seed (verbatim W108 evaluator)

| field | seed | A0 | A1 | B | B−A1 | MLB-1 | MLB-2 | gates | verdict |
|---|---|---|---|---|---|---|---|---|---|
| RESISTANT | 120001 (W120) | 20.00 | 23.33 | 23.33 | **+0.00** | 83.3% P | 8.0% F | 6/9 | FAIL |
| RESISTANT | 120002 (W122) | 16.67 | 26.67 | 30.00 | **+3.33** | 76.7% P | 8.7% F | 8/9 | FAIL |
| EXPOSED | 120001 (W121) | 6.67 | 26.67 | 30.00 | **+3.33** | 93.3% P | 25.0% F | 8/9 | FAIL |
| EXPOSED | 120002 (W122) | 6.67 | 20.00 | 33.33 | **+13.33** | 93.3% P | 28.6% F | 9/9 | **PASS_NON_MECHANISM_DRIVEN** |

* RESISTANT seed 120002 (merkle `7259545f…`): net +1 problem — rescue `averagesubstringvalue`;
  0 regressions.
* EXPOSED seed 120002 (merkle `3537e510…`): net **+4** problems — rescues `pawnshop`,
  `rsamistake`, `champernownecount`, `icouldhavewon`; **0 regressions**. A1 dipped to 20.00%
  this seed (sampling variance; it was 26.67% on 120001), so the +13.33 pp margin is partly
  a low-A1-baseline effect. MLB-2 = 28.6% (4/14) < 33% ⇒ **non-mechanism-driven** (the
  margin is rescue-concentrated, not a high rescue rate) — the W109/W112 exposed-control
  signature (a margin without a load-bearing mechanism), here on the EXPOSED side.

### 2-seed aggregate (the closure verdict)

| field | seeds | margins (pp) | **2-seed mean** | in ±3.34 null band? | clean ≥ +5.00 margin? |
|---|---|---|---|---|---|
| RESISTANT | 120001, 120002 | +0.00, +3.33 | **+1.67** | ✅ yes | no |
| EXPOSED | 120001, 120002 | +3.33, +13.33 | **+8.33** | ❌ no | no — not all-clean per seed |

Through the pre-committed symmetric closure rule (`interpret_paired_closure_v1`,
`docs/RUNBOOK_W122.md` § 2): resistant is null (B1-side) but exposed mean is OUT of the null
band, so B1 (both-null closure) does NOT fire. The exposed mean ≥ +5.00 BUT the field is
**not all-clean per seed** (seed 120001 was FAIL; seed 120002 was `PASS_NON_MECHANISM_DRIVEN`,
not `PASS_MECHANISM_DRIVEN`), so B2 (clean exposed margin re-strengthens contamination) does
**not** fire either (the rule requires every exposed seed `PASS_MECHANISM_DRIVEN`). Resistant
shows no clean margin, so B3 does not fire. ⇒ **`AMBIGUOUS_THIRD_PAIRED_SEED_EARNED`** (B4);
`caveat_closed = False`; `third_seed_earned = True`.

**This is the rule working as designed.** It refused to (a) declare the caveat closed on a
field that popped out of band, and (b) count a rescue-concentrated `PASS_NON_MECHANISM_DRIVEN`
spike as a contamination re-strengthening. It flagged genuine ambiguity and earned the
pre-committed tiebreaker.

## The earned 3rd paired seed (120003)

Per `docs/RUNBOOK_W122.md` § 2 B4 / § 6, the 3rd paired seed is EARNED (and ONLY the 3rd —
no 4th). Seed 120003 was LAUNCHED on BOTH the resistant and exposed 30-slices (same model,
slices, grader, evaluator, K; only the seed changed), 660 NIM calls. **The driver was
corrected first** so the closure step aggregates ALL prior seeds keyed by seed
({120001, 120002, 120003} per field), not just {120001, 120003} (verified NIM-free: the
collector reproduces the 2-seed B4 verdict byte-for-byte from disk). The 3-seed aggregate
re-runs the SAME symmetric rule.

### FINAL 3-SEED VERDICT — `AMBIGUOUS_THIRD_PAIRED_SEED_EARNED` (B4); closure NOT achieved

Seed 120003 ran on BOTH fields (same model/slices/grader/evaluator/K; only the seed changed;
660 NIM calls). Per-seed (verbatim W108 evaluator):

| field | seed | A0 | A1 | B | B−A1 | gates | MLB-2 | verdict |
|---|---|---|---|---|---|---|---|---|
| RESISTANT | 120003 | 23.33 | 23.33 | 33.33 | **+10.00** | 9/9 | 16.7% | `PASS_NON_MECHANISM_DRIVEN` |
| EXPOSED | 120003 | 6.67 | 16.67 | 26.67 | **+10.00** | 9/9 | 18.5% | `PASS_NON_MECHANISM_DRIVEN` |

* RESISTANT 120003 (bench merkle `adf55ff9…`): net **+3** — rescues `letterballoons`,
  `bigand`, `conveyorbeltsushi`; **0 regressions**. MLB-2 16.7% < 33% ⇒ non-mechanism-driven.
* EXPOSED 120003 (bench merkle `d88d025f…`): net **+3** — rescues `blueberrywaffle`,
  `champernownecount`, `icouldhavewon`; **0 regressions**. MLB-2 18.5% < 33% ⇒
  non-mechanism-driven. (`icouldhavewon` was a *regression* on exposed seed 120001 and
  `champernownecount` a *rescue* on seed 120002 — the same problems flip rescue↔regression
  across seeds: the signature of sampling variance, not a stable mechanism effect.)

**3-seed aggregate (code-emitted, `interpret_paired_closure_v1`):**

| field | seeds | margins (pp) | **3-seed mean** | null band (±3.34)? | clean ≥+5 margin? |
|---|---|---|---|---|---|
| RESISTANT | 120001, 120002, 120003 | +0.00, +3.33, +10.00 | **+4.44** | ❌ no (in the 3.34–5.00 gap) | no — not all-clean per seed |
| EXPOSED | 120001, 120002, 120003 | +3.33, +13.33, +10.00 | **+8.89** | ❌ no | no — not all-clean per seed |

`resistant_in_null_band = false`, `exposed_in_null_band = false`, both `shows_margin = false`,
`all_seeds_clean_pass = false` on BOTH fields. Through the precedence B3 > B2 > B1 > B4: **B3**
needs a clean resistant ≥+5 margin (resistant mean +4.44 < 5 AND not all-clean) — off; **B2**
needs a clean exposed margin AND a resistant null (resistant +4.44 is NOT null) — off; **B1**
needs both means in the null band (neither is) — off ⇒ **`AMBIGUOUS_THIRD_PAIRED_SEED_EARNED`
(B4)**; `caveat_closed = false`. The resistant 3-seed mean +4.44 lands in the (3.34, 5.00)
gap — a 3-seed lattice point (multiples of 1.11 pp) that, unlike 2-seed means (multiples of
1.67 pp), *can* fall in the gap.

**This is `B4` AFTER the 3rd seed — a terminal state, not a request for a 4th.** The
pre-committed rule (`RUNBOOK_W122` §2/§6) buys exactly ONE tiebreaker seed and **forbids a
4th**; the code's `third_seed_earned = true` is the generic B4 flag, but §8 governs the
post-3rd-seed case (`w123_fire_condition_v1`): *register the residual ambiguity; W123 = accept
the bounded claim or escalate to a larger n PER FIELD (not more seeds at n=30).* No 4th seed
was or will be bought.

**What the 3rd seed revealed.** At 2 seeds the picture looked like "resistant null vs exposed
popped". The 3rd seed **dissolved that asymmetry**: the RESISTANT field also produced a +10.00
pp `PASS_NON_MECHANISM_DRIVEN` seed, so resistant is **not** reliably null across seeds
(+0.00, +3.33, +10.00; mean +4.44) and both fields now carry the SAME signature — occasional
large, rescue-concentrated, non-mechanism-driven B−A1 spikes (MLB-2 < 33% on every non-FAIL
seed; **no `PASS_MECHANISM_DRIVEN` seed on either field, any seed**). At n=30 with K=5 a single
seed's B−A1 swings ±10 pp on ~3 rescues, so the n=30 estimator is simply too noisy to resolve a
few-pp matched-family contrast. **Closure is NOT achieved; the honest finding is small-n
irresolvability, and W123 must escalate n PER FIELD.**

**Secondary continuity label (do NOT over-read).** The reused W121 single-axis interpreter
emits `multiseed_exposed_vs_resistant_outcome = EXPOSED_MARGIN_VS_RESISTANT_NULL_DIFFICULTY_LOOPHOLE_CLOSED`
**only because** it checks one thing — exposed mean ≥ +5 (8.89 ≥ 5) — and *assumes the
resistant side is null*. At 3 seeds that premise is **false** (resistant +4.44 is itself out of
band), so the "loophole closed / exposure dissociation" reading does **NOT** hold. The
authoritative verdict is the PRIMARY symmetric rule: **B4 AMBIGUOUS**. The faint exposed >
resistant ordering (+8.89 > +4.44) is exposure-CONSISTENT but non-mechanism-driven, noisy, and
undercut by the resistant field's own +10.00 spike.

**Net.** Single-seed caveat: the literal objection is retired (3 seeds each side) but
**replaced by a small-n-variance limitation** — the contrast is genuinely unresolvable at n=30
(B4). No third retirement (**W89 + W105 STAND, the only two**). Contamination: still WEAKENED /
unresolved, and the 3-seed data **further undercuts** a clean contamination read
(contamination-RESISTANT code spikes +10.00 just like exposed code ⇒ the spikes are sampling
variance, not exposure). Difficulty comparability held across all three seeds (exposed A0
6.67% ≤ resistant A0 20.00% mean). Lane β (M3 mechanism-robust kill) and Lane γ (stronger-model
gate closed) unchanged. `COO-9` stays lead. Artifacts:
`results/w122/paired_seed/w122_paired_seed_120003_20260531T184401Z/paired_seed_closure_verdict.json`.

## What the 2-seed STEP showed (interim — SUPERSEDED by the FINAL 3-SEED VERDICT above)

> **Interim read, retained for the record.** At 2 seeds the contrast looked like "resistant
> null vs exposed popped". The 3rd seed (above) **dissolved that asymmetry** — the resistant
> field also spiked +10.00 pp — so the bullets below describe the 2-seed snapshot only; the
> authoritative outcome is the 3-seed **B4** (resistant mean +4.44, exposed mean +8.89, both
> out of band, neither clean) recorded above.

* **The single-seed caveat is NOT closed at 2 seeds.** On the resistant field the null held
  (mean +1.67). On the EXPOSED field a second seed produced a large, rescue-concentrated,
  non-mechanism-driven spike (+13.33), so the exposed side is now bimodal across seeds
  (+3.33, +13.33; mean +8.33). The fields no longer look alike at 2 seeds ⇒ genuinely
  ambiguous ⇒ the 3rd seed was earned (it then ALSO landed B4 — see the FINAL verdict above).
* **No third retirement.** W89 (+5.56) + W105 (+7.00), Llama-3.3-70B contamination-EXPOSED
  HumanEval-family, **remain the only two**. W122 adds none. The exposed +13.33 is
  `PASS_NON_MECHANISM_DRIVEN` (MLB-2 < 33%, rescue-concentrated, low-A1-baseline) — the
  W109/W112 exposed-control signature, NOT a clean mechanism win and NOT a retirement.
* **The exposed spike was exposure-consistent, NOT contamination-proven.** At 2 seeds the
  exposed side popping up (while resistant was still null AT 2 SEEDS) was in the direction
  H_contamination predicts — but it was single-seed, non-mechanism-driven, and the resistant
  field's own second seed also rose to +3.33, so the picture was noisy. The 3rd paired seed
  was exactly the right instrument to disambiguate "exposed genuinely margins" from "one
  rescue-concentrated sampling spike" — and it answered "sampling spike": the resistant field
  ALSO spiked +10.00 (see the FINAL verdict above).
* **Difficulty comparability held, both seeds.** Exposed A0 (6.67% both seeds) ≤ resistant
  A0 (20.00% / 16.67%) ⇒ the exposed slice is not easier; the matched-difficulty design held.

## W123 (resolved by the actual branch)

The 3-seed aggregate is **B4** ⇒ the pre-committed post-3rd-seed branch fires (`RUNBOOK_W122`
§8, `w123_fire_condition_v1`): **register the residual ambiguity; W123 = accept the bounded
claim OR escalate to a larger n PER FIELD (NOT more seeds at n=30; NO 4th seed)**; OR a
reachable primary-KNOWN stronger-than-Maverick model on BOTH matched ICPC battlefields if one
opens. The natural reading, given the small-n-variance finding, is to accept the bounded
HumanEval-family ceiling (W89 + W105) and, only if the matched contrast is worth re-attacking,
escalate to ≥100 problems PER FIELD rather than buy more n=30 seeds. `COO-9` stays lead (a
B4-ambiguous result does not force a code-line move).

Artifacts: `results/w122/paired_seed/w122_paired_seed_120002_20260531T155909Z/`
(`paired_seed_closure_verdict.json` [2-seed] + per-field `*_reflexion_bench_report.json` +
`*_reflexion_calls.jsonl` sidecars with per-call prompt/response SHAs + `provenance.json`);
the seed-120003 run dir (3-seed); `results/w122/mechanism_audit/mechanism_audit_verdict.json`.
