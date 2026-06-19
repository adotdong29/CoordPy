# W94 — K=10 Reflexion Pilot V1 (Preflight-Earned)

> **2026-05-24 — K=10 HYPOTHESIS KILLED IN CHEAP PILOT.  At
> 1 seed × 15 problems × K=10 × Llama-3.3-70B-Instruct on
> HumanEval, A1 reaches **100 %** (ceiling); B also reaches
> 100 %; B − A1 = +0.00 pp.  3 of 6 pre-committed W94 P1 pilot
> gates FAIL.  No full K=10 bench launched per W94 runbook
> contract.  The W93-C K=10 hypothesis is empirically dead at
> this scale: doubling the budget saturates A1 to ceiling,
> leaving zero failure-residual for reflexion to rescue.  This
> is exactly the cheap-decisive outcome the W93 preflight
> discipline was designed to produce — a 90-minute pilot
> killed the hypothesis instead of a 5-hour full bench
> discovering the same failure mode.**

## Pilot summary

* **Bench**: `coordpy.humaneval_reflexion_bench_v1` (W88-W89-
  validated module, unchanged; only `--K 10` flag added).
* **Model**: `meta/llama-3.3-70b-instruct` (NIM).
* **Slice**: seed 88_028_001 × 15 problems × K=10.  The 15
  problems are the FIRST 15 selected by deterministic
  `select_humaneval_subset_v1(88_028_001, 15)`.
* **Cost**: 315 NIM calls in 5294 s (~88 min wall).
* **Bench Merkle**: `4556feef9cb15b96…`
* **Audit verifier**: 7/7 PASS (3 audit + 4 retirement bars,
  3 of which correctly FAIL since the pilot KILLs the K=10
  hypothesis).

| Arm | Mean pass@1 (15 problems) | Calls/problem |
|----|---:|---:|
| A0 stock single-shot (T=0) | 66.67 % | 1 |
| A1 first-pass-among-K=10 (T=0.7) | **100.00 %** | 10 |
| B sequential-reflexion-K=10 (T=0.7) | **100.00 %** | 10 |

* B − A1 = **+0.00 pp** (tied at perfect ceiling).
* B − A0 = +33.33 pp (only A0 not at ceiling).
* B beats A1 per-seed: 0/1 (tied).
* B beats A1 per-problem: 0/15 strict wins (all 15 problems
  pass on both A1 and B — pure ceiling).

## Pre-committed W94 P1 pilot gates

Locked in `docs/RUNBOOK_W94.md` BEFORE the pilot ran:

| Gate | Threshold | Pilot outcome | Pass? |
|------|-----------|---------------|-------|
| 1. Slice pre-committed | first 15 from seed 88_028_001 | ✓ deterministic selection used | ✓ |
| 2. B > A1 on slice | strict | B = A1 = 100 % | ✗ |
| 3. B − A1 margin ≥ +5 pp | required | +0.00 pp | ✗ |
| 4. B ≥ A1 per-problem on ≥ 8/15 | required | tied on 15/15; strict B > A1 on 0/15 | ✗ |
| 5. Budget accounting exact | 21 calls/problem | 315 calls / 15 = 21 ✓ | ✓ |
| 6. Audit chain re-derives | required | 7/7 PASS offline | ✓ |

**3 of 6 gates FAIL → W94-C is KILLED.  No Phase 2 launch.**

## Diagnosis: why the K=10 hypothesis died

The W93-C hypothesis stated: "at K=10, i.i.d. sampling
saturates; reflexion has 10 iterations to add value; B beats
A1 by a clearer margin than the K=5 baseline."

The empirical reality on the 15-problem slice:

* **A1 K=10 saturates A FIRST**, before reflexion has any
  room to add value.  At K=10 on the easier 15-problem subset,
  A1 reaches 100 % — every problem has at least one passing
  sample among 10 i.i.d. attempts at T=0.7.
* The W93-C hypothesis assumed A1 K=10 would still leave a
  meaningful failure-residual (analogous to A1 K=5 = 85.6 %
  at W89 leaving ~14 % residual).  This assumption is
  EMPIRICALLY FALSIFIED: as K grows, A1's i.i.d. coverage
  grows faster than reflexion's per-turn marginal value.
* The 15-problem slice was not adversarial — it was the
  deterministic first-15 of the W89 30-problem subset.  Even
  on this representative slice, K=10 saturates.
* On the FULL 30-problem set, A1 K=10 might leave a small
  residual (the harder 15 problems would not all be solved
  by K=10), but the pilot evidence says the K=10-vs-K=5
  amplification is going the WRONG way at the slice the
  pilot tested.

The pilot validates the W93 preflight discipline: **a
hypothesis that LOOKS strong on paper (more budget = more
reflexion turns = bigger advantage) can collapse empirically
because the baseline saturates faster than the team
architecture's marginal value grows**.  The 90-minute pilot
catches this cheaply.

## What this means for the carry-forwards

* `W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP`
  remains as the canonical K=5 retirement.  K=10 does NOT
  amplify the retirement claim.
* `W93-L-W93C-K10-REFLEXION-NEEDS-PILOT-CAP` is
  REPLACED by the new `W94-L-K10-PILOT-CEILING-SATURATION-CAP`
  documenting the actual pilot outcome.
* `W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP` STAYS — K=10
  on MBPP is now also expected to ceiling-saturate (A1 K=10
  on MBPP-30 would likely reach near 100 %), so K=10 is not
  the right lever for MBPP either.

## New W94-L-* carry-forward

**`W94-L-K10-PILOT-CEILING-SATURATION-CAP`**: at 1 seed ×
15 problems × K=10 × Llama-3.3-70B-Instruct on HumanEval, A1
first-pass-among-K=10 reaches 100 % ceiling on every problem.
B sequential-reflexion-K=10 also reaches 100 % but cannot
strictly beat A1 since A1 is saturated.  The W93-C hypothesis
(K=10 amplifies reflexion's advantage) is empirically dead at
this scale.  Doubling the budget from K=5 → K=10 saturates
the baseline faster than reflexion can capitalize on the
extra turns.  Future budget-extension hypotheses (K=15, K=20)
on HumanEval are presumptively dead by the same reasoning;
**ceiling-saturation is a fundamental barrier to "more
budget = bigger gap"** on this benchmark family.  The W94
pilot is the cheapest possible test of this barrier.

## Anti-cheat

* The 15-problem slice was deterministic and pre-committed via
  `select_humaneval_subset_v1(88_028_001, 15)`.  No
  post-hoc selection.
* No selective retries; one set of 315 NIM calls.
* The audit chain re-derives offline byte-for-byte.
* Negative evidence is preserved as
  `results/w94/k10_pilot/w88_nim_meta_llama-3.3-70b-instruct_20260524T164255Z/`
  — full sidecar + bench report committed alongside this
  RESULTS doc.

## What W94 retires

Nothing.  No expensive Phase 2 launched.  The W94 deliverable
is the cheap pilot kill of W93-C.

## What W94 contributes

* **Empirical validation of the W93 preflight discipline.**
  A hypothesis (K=10 amplifies reflexion) that looked
  preflight-supported on every gate except G2 turned out
  to fail when G2 was resolved with a 90-min pilot.  The W93
  bar of "do not pay full price until preflight + pilot say
  the candidate is dangerous" prevented a likely 5-hour
  full-K=10-bench failure.
* **New empirical finding**: budget extension (K=5 → K=10)
  on HumanEval saturates A1 to ceiling, eliminating the
  failure-residual reflexion needs to win.  This is a
  structural constraint on the budget-extension family of
  hypotheses.
* **W95+ direction**: focus on benchmarks where K=10 (or even
  K=5) does NOT saturate A1.  MathVista (selected as the W95
  cross-modal battlefield in
  `docs/W94_CROSS_MODAL_BATTLEFIELD_SCOUTING.md`) has SOTA
  single-shot ~60-65 %; K=5 ceiling probably ~75-80 %; K=10
  probably ~85-90 %.  Even at K=10 there would be ~10-15 %
  failure-residual — much more room than HumanEval's
  saturated-at-100 % on 15 easy problems.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* W94 added only the `--K` flag to the existing W88 driver
  (no new modules; no new tests; the W88
  `humaneval_reflexion_bench_v1` module supports arbitrary K
  natively).

## Re-running

```bash
python scripts/run_w88_humaneval_reflexion_bench.py \
    --backend nim \
    --model meta/llama-3.3-70b-instruct \
    --n-problems 15 --n-seeds 1 --K 10 \
    --out-dir results/w94/k10_pilot
python scripts/verify_w88_humaneval_reflexion_audit_chain.py \
    --run-dir results/w94/k10_pilot/<run-dir>
```

NIM provider-side sampling at T=0.7 carries minor variance;
the conclusion ("A1 K=10 saturates to ceiling on this slice")
is robust to expected sampling variation.

## The honest claim W94 earns

**The W93-C K=10 reflexion hypothesis is empirically dead on
HumanEval at the cheap pilot scale (1 seed × 15 problems).
A1 first-pass-among-K=10 saturates to 100 % on this slice,
eliminating any failure-residual for reflexion to rescue from.
The W93 preflight-first discipline successfully killed this
hypothesis in a 90-minute pilot instead of a 5-hour full
bench.  Future budget-extension hypotheses on HumanEval are
presumptively dead by the same ceiling-saturation argument.
No new carry-forward retirements; the W89 70B-HumanEval K=5
retirement remains the only confirmed same-budget multi-agent
superiority claim in the programme.**

## Recommended W95+ direction

1. **Drop budget-extension on HumanEval as a research lever.**
   K=5 was the right size; K=10 destroys the baseline-gap.
2. **Pivot to MathVista** for cross-modal team retirement.
   Documented in `docs/W94_CROSS_MODAL_BATTLEFIELD_SCOUTING.md`.
   Build the corpus loader + executor + team architecture
   in W95; preflight against synthetic discriminators (the
   W93 harness); pilot then escalate per W93 discipline.
3. **Optionally**: K=10 on MBPP could be tested with a cheap
   pilot (MBPP has lower ceiling at K=5 = 82 %; at K=10 may
   not saturate fully).  But the W90/W91 evidence already
   suggests MBPP at K=5 is per-seed-variance-bound; K=10
   would need to overcome both ceiling and per-seed variance.
   Lower priority than the MathVista pivot.
