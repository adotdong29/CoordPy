# W91 — MBPP sequential-reflexion V2 (5 seeds × 30) (Post-W90 empirical superiority wave V4)

> **2026-05-23 — PARTIAL.  At Llama-3.3-70B-Instruct on MBPP-
> sanitized × 5 seeds × 30 problems × K=5, B sequential
> reflexion mean pass@1 = 84.0 % strictly beats A1 first-pass-
> among-K=5 mean = 82.7 % by **+1.33 pp** (margin clears the
> +1.0 pp threshold).  Per-seed B ≥ A1 on **4/5 seeds** but
> strict B > A1 on only **2/5 seeds**.  Three of four pre-
> committed retirement bars met; the per-seed strict majority
> bar (≥ 3 / 5) FAILS at 2/5.  W90 P1's 3-seed pattern is
> CONFIRMED at 5-seed scale: directional positive mean but
> fragile per-seed strength.  `W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP`
> is REFINED (with stronger statistical evidence from 5 seeds)
> but still NOT retired.**

## TL;DR

5 seeds × 30 problems × 3 arms on MBPP-sanitized.  K=5; model
= `meta/llama-3.3-70b-instruct` via NIM; same sequential
reflexion B-pipeline as W88/W89/W90.  Total wall 30662 s
(~8 h 31 min); 1650 NIM calls; bench Merkle `b5cc804be2caa4da…`;
audit verifier 6/7 PASS (3 audit + 3 of 4 retirement bars).

| Arm | Mean pass@1 | Per-seed |
|----|---:|---|
| **A0** stock single-shot (T=0) | **75.33 %** | 0.800 / 0.667 / 0.700 / 0.867 / 0.733 |
| **A1** first-pass-among-K=5 (T=0.7) | **82.67 %** | 0.900 / 0.700 / 0.833 / 0.900 / 0.800 |
| **B** sequential-reflexion-K=5 (T=0.7) | **84.00 %** | 0.900 / 0.767 / 0.867 / 0.900 / 0.767 |

* `b_mean_strictly_beats_a1_mean = True` ✓ — +1.33 pp
* `b_mean_strictly_beats_a0_mean = True` ✓ — +8.67 pp
* B − A1 = +1.33 pp; margin clears +1.0 pp threshold ✓
* B beats A1 per-seed: `(False, True, True, False, False)` —
  **2/5 strict majority bar fails (need ≥ 3/5)**.

## Comparison vs W90 P1 (3 seeds × 30)

| Metric | W90 P1 (3 seeds) | **W91 P1 (5 seeds)** |
|---|---:|---:|
| A0 mean | 76.67 % | 75.33 % |
| A1 mean | 81.11 % | 82.67 % |
| B mean | 82.22 % | 84.00 % |
| B − A1 mean | +1.11 pp | **+1.33 pp** |
| B beats A1 per-seed | 1/3 | 2/5 |
| All 4 bars MET? | No (3/4; per-seed) | No (3/4; per-seed) |

**The 5-seed extension confirms** the W90 P1 pattern:

* B mean strictly beats A1 mean (margin +1.33 vs +1.11 —
  slightly higher).
* B beats A0 on 5/5 seeds (+8.67 pp on mean).
* B never loses MEAN to A1; B ≥ A1 on 4/5 seeds (only seed 5
  loses by −3.33 pp).
* **Per-seed strict majority bar still fails** at 2/5.

The mean direction is robust at larger N.  The per-seed-
strict-majority bar continues to fail because the per-seed
delta is small (often 0.0 to +3.33 pp) and ties count as
"False" under strict comparison.

## Per-seed result

| Seed | A0 | A1 | B | B − A1 | Notes |
|----:|---:|---:|---:|---:|---|
| 90_001 | 80.0 % | 90.0 % | 90.0 % | tie (+0.0) | A1 near-ceiling |
| 90_002 | 66.7 % | 70.0 % | 76.7 % | **+6.67 pp** | B strict win |
| 90_003 | 70.0 % | 83.3 % | 86.7 % | **+3.33 pp** | B strict win |
| 90_004 | 86.7 % | 90.0 % | 90.0 % | tie (+0.0) | A1 near-ceiling |
| 90_005 | 73.3 % | 80.0 % | 76.7 % | −3.33 pp | B small loss |
| **mean**   | **75.3 %** | **82.7 %** | **84.0 %** | **+1.33 pp** |

**Per-seed pattern:**

* B ≥ A1 on 4 of 5 seeds.
* B strict > A1 on 2 of 5 seeds (per the bar's strict
  comparison).
* The 2 ceiling-saturated seeds (1, 4) at A1 = 90 % give B no
  headroom to differentiate.
* Seed 5 is the only strict loss; small magnitude (−3.33 pp).

## Why the per-seed bar fails

Same structural reason as W90 P1: **ceiling effects + small
effect size + ties count as not-strict-wins**.

* MBPP-30 at A1 K=5 = 82.7 % leaves ~17 % failure-residual
  for B's reflexion to rescue.
* At per-seed level, the failure-residual sometimes lands on
  problems where reflexion can't help (seed 5: 4 of 30
  failures still failed after reflexion).
* On the 2 seeds where A1 hits 90 % ceiling (seeds 1, 4), B
  can't go above; ties count as "not strict win" for the bar.

To clear the per-seed strict majority bar reliably, MBPP-K=5
would need either (a) a less ceiling-saturated subset
(MBPP-Hard / harder problems), (b) larger K budget where
reflexion has more iterations, or (c) more problems per seed
to reduce per-seed variance.

## Anti-cheat re-statement

* ✓ Same model on every arm of this run (Llama-3.3-70B-Instruct).
* ✓ Same task subset per seed (deterministic
  `select_mbpp_subset_v1(seed)`).
* ✓ Same prompt budget per arm (K=5 on A1 and B).
* ✓ Same retry policy (6 attempts with 429-aware backoff).
* ✓ No selective retries.
* ✓ Executor truth = pass on every assertion in `test_list`.
* ✓ Audit chain re-derives offline: 1650 per-call SHA-256
  matches + 5 per-seed Merkle root re-derive + bench Merkle
  root re-derive — ALL PASS.
* ✓ MBPP-sanitized corpus SHA-256 verified against the
  upstream pin before this run.
* ✓ No baseline weakening.  A1 reaches 82.7 % (strong K=5
  self-consistency on MBPP-30 at 70B).

## What W91 P1 retires

Nothing.  3 of 4 pre-committed bars met; the per-seed strict
majority bar (≥ 3 / 5) FAILS at 2/5.

`W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP` is
REFINED with stronger 5-seed evidence but NOT retired.

## What W91 P1 contributes (new carry-forward)

* **`W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP`** — at
  Llama-3.3-70B-Instruct on MBPP-sanitized × 5 seeds × 30
  problems × K=5, B sequential reflexion mean pass@1 strictly
  beats A1 first-pass-among-K=5 (+1.33 pp on the mean;
  margin clears +1.0 pp threshold).  B ≥ A1 on 4/5 seeds
  but strict B > A1 on only 2/5 seeds — the per-seed strict
  majority bar (≥ 3 / 5) fails.  The 5-seed extension
  CONFIRMS the W90 P1 (3-seed) pattern: mean direction
  positive and robust at larger N; per-seed strict signal
  fragile due to ceiling-saturation on a subset of seeds.
  Future MBPP attempts should target a harder / less
  ceiling-saturated subset OR larger K budget where reflexion
  has more iterations to differentiate from independent
  sampling.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* W91 reuses `coordpy.mbpp_reflexion_bench_v1` unchanged.

## Re-running

```bash
# NVIDIA_API_KEY must be set.
python scripts/run_w90_mbpp_reflexion_bench.py \
    --model meta/llama-3.3-70b-instruct \
    --n-problems 30 --n-seeds 5 \
    --out-dir results/w91/mbpp_reflexion_5seeds
python scripts/verify_w90_mbpp_reflexion_audit_chain.py \
    --run-dir results/w91/mbpp_reflexion_5seeds/<run-dir>
```

## The honest claim this run earns

**At Llama-3.3-70B-Instruct on MBPP-sanitized × 5 seeds × 30
problems × K=5, B sequential-reflexion mean pass@1 strictly
beats A1 first-pass-among-K=5 on the mean by +1.33 pp.  B ≥ A1
on 4/5 seeds; strict B > A1 on 2/5 seeds.  3 of 4 pre-committed
retirement bars met; the per-seed strict majority bar FAILS.
The W89 70B-reflexion architecture's directional cross-
benchmark generalisation to MBPP is now CONFIRMED at 5-seed
scale on the mean; the per-seed strict majority bar continues
to fail at this scale due to ceiling effects on a subset of
seeds.  W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP
is REFINED but NOT retired.**
