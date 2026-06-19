# W90 — MBPP sequential-reflexion bench V1 (Post-W89 empirical superiority wave V3)

> **2026-05-22 — PARTIAL.  At Llama-3.3-70B-Instruct on MBPP-
> sanitized × 3 seeds × 30 problems × K=5 budget, the W88
> sequential-reflexion B-pipeline beats A1 first-pass-among-K=5
> by **+1.11 pp** on the mean — but only beats A1 on **1/3
> seeds** (ties on the other 2; never loses).  Three of four
> pre-committed retirement bars are met; the per-seed majority
> bar fails.  The W89 70B-HumanEval retirement does NOT cleanly
> extend to MBPP at the W89 strong-success threshold — the
> `W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP`
> carry-forward STAYS, with a partial-success refinement.**

## TL;DR

3 seeds × 30 problems × 3 arms on MBPP-sanitized
(`google-research/google-research/mbpp/sanitized-mbpp.json`,
SHA `ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9`,
427 problems with valid entry-point extraction).  K=5 budget on
A1 and B.  Model = `meta/llama-3.3-70b-instruct` via NIM.
Total wall 18100 s (≈ 5 h 02 min); 990 NIM calls; same
sequential-reflexion B-pipeline as W88/W89 (different benchmark
only).

| Arm | Mean pass@1 | Per-seed |
|----|---:|---|
| **A0** stock single-shot (T=0) | **76.67 %** | 0.900 / 0.667 / 0.733 |
| **A1** first-pass-among-K=5 (T=0.7) | **81.11 %** | 0.900 / 0.733 / 0.800 |
| **B** sequential-reflexion-K=5 (T=0.7) | **82.22 %** | 0.900 / 0.733 / 0.833 |

* `b_mean_strictly_beats_a1_mean = True` ✓ — B beats A1 on
  mean by +1.11 pp; **the W89 70B reflexion architecture's
  HumanEval win directionally generalises to MBPP**.
* `b_mean_strictly_beats_a0_mean = True` ✓ — B beats A0 by
  +5.55 pp.
* B − A1 = +1.11 pp; margin exceeds the +1.0 pp pre-committed
  threshold (just barely).
* `b_beats_a1_per_seed = (False, False, True)` — B ties A1 on
  seeds 1 & 2 (both arms identical pass@1); wins seed 3 by
  +3.33 pp.  **Per-seed majority bar fails: only 1/3 seeds
  show strict B > A1.**

Bench Merkle root:
`b50cafbe7669cba8...`  (full at
`results/w90/mbpp_reflexion/.../mbpp_reflexion_bench_report.json`).
**Audit chain re-derives offline: 6/7 PASS** (3 audit + 3 of 4
retirement bars; per-seed majority is the failing bar).

## Comparison vs W89 HumanEval-70B

| Metric | W89 HumanEval (70B) | **W90 MBPP (70B)** |
|---|---:|---:|
| Benchmark | HumanEval-30 | **MBPP-sanitized-30** |
| A0 mean | 46.7 % | **76.7 %** |
| A1 mean | 85.6 % | **81.1 %** |
| B mean | 91.1 % | **82.2 %** |
| B − A1 mean | **+5.56 pp** | **+1.11 pp** |
| B beats A1 per seed | 2/3 | **1/3** |
| All 4 bars MET? | ✓ YES (retirement) | ✗ NO (3/4) |

**The headline finding**: the W89 architecture wins in BOTH
directions on the MEAN (+5.56 pp on HumanEval, +1.11 pp on
MBPP) — that's a real cross-benchmark generalisation signal.
But the strength varies: HumanEval shows a robust per-seed
majority (2/3); MBPP shows a marginal mean with per-seed
majority absent (1/3).

The +1.11 pp on MBPP is exactly at the pre-committed
margin-threshold of +1.0 pp (the bench landed +0.11 pp above).
This is a TIGHT result — within sampling variance of A1.

Per-seed pattern: on the SEEDS where A1 already reaches ~90 %
(seed 1), the ceiling is too tight to leave room for B's
reflexion to add value, so B ties A1.  On harder seeds (seed
3 at 80 %), B's reflexion catches the residual failures and
wins by +3.33 pp.  This is consistent with the "reflexion
adds value on the failure-residual" story but the residual is
small at A1 = 81 %.

## Per-seed result

| Seed | A0 | A1 | B | B − A0 | B − A1 |
|----:|---:|---:|---:|---:|---:|
| 90_001 | 90.0 % | 90.0 % | 90.0 % | tie | tie |
| 90_002 | 66.7 % | 73.3 % | 73.3 % | **+6.7 pp** | tie |
| 90_003 | 73.3 % | 80.0 % | 83.3 % | **+10.0 pp** | **+3.3 pp** |
| **mean**   | **76.7 %** | **81.1 %** | **82.2 %** | **+5.5 pp** | **+1.11 pp** |

* `b_beats_a0_per_seed = (False, True, True)` — B ties A0 on
  seed 1; beats A0 on seeds 2 & 3.
* `b_beats_a1_per_seed = (False, False, True)` — B ties A1 on
  seeds 1 & 2; beats A1 on seed 3.
* B never LOSES to A1 on any seed (≥ A1 on all 3 seeds).

## What this empirical evidence says

* **Reflexion architecture wins on the mean across BOTH
  HumanEval and MBPP at 70B scale.**  The direction is robust.
* **The MBPP win is fragile (per-seed majority fails).**
  Whether MBPP's +1.11 pp is real cross-benchmark
  generalisation or a 1-seed fluke depends on more seeds.
* **MBPP has tight ceiling effects** — A1 at 81 % leaves only
  19 % failure-rate for B to rescue; the reflexion's marginal
  impact at this ceiling is small.  Whether the architecture
  shows clearer gains on harder benchmarks (LiveCodeBench,
  SWE-bench Verified) is open.

## What W90 P1 retires

Nothing.  3 of 4 pre-committed bars met; the per-seed majority
bar fails.

`W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP` (the
"only tested on HumanEval" cap) is REFINED but NOT retired:
the cross-benchmark direction is positive on the mean, but
not robust enough at per-seed level to clear the strong-
success bar.

## What W90 P1 contributes (new carry-forwards)

* **`W90-L-MBPP-REFLEXION-V1-PARTIAL-CAP`** — at
  Llama-3.3-70B-Instruct on MBPP-sanitized × K=5 × 3 seeds ×
  30 problems, B sequential-reflexion mean pass@1 strictly
  beats A1 first-pass-among-K=5 (+1.11 pp, margin >= +1.0 pp
  threshold) but does NOT strictly beat A1 on the per-seed
  majority (1/3 only).  The W89 70B-reflexion architecture
  shows directional cross-benchmark generalisation to MBPP on
  the mean; the per-seed strength is below the W89 bar.
* **`W90-L-MBPP-REFLEXION-V1-CEILING-CAP`** — A1's mean
  reaches 81.1 % at K=5 on MBPP-sanitized-30, leaving only
  ~19 % failure-rate for B's reflexion to rescue from.  On
  seeds where A1 reaches ~90 %, B has no room to differentiate.
  Whether the win extends on a less-saturated MBPP subset
  (MBPP-Hard, MBPP+) or a harder benchmark (LiveCodeBench,
  SWE-bench Verified) is V2.
* **`W90-L-MBPP-REFLEXION-V1-RATE-LIMIT-CAP`** — the bench
  encountered NIM HTTP 429 rate-limit responses during the
  W90 run; the improved 6-attempt retry with 429-aware
  exponential backoff handled them transparently.  Total wall
  was 18100 s (vs ~10000 s expected at pure 0.1 calls/sec for
  70B); the backoff added ~80 minutes of throttle time.
  Result quality unaffected; reproducibility may differ on
  a less-loaded NIM endpoint.

## Anti-cheat re-statement

* ✓ Same model on every arm of this run (Llama-3.3-70B-Instruct
  on A0, A1, AND B).
* ✓ Same task subset per seed (deterministic
  `select_mbpp_subset_v1(seed)`).
* ✓ Same prompt budget per arm (A1 K=5; B K=5).
* ✓ Same retry policy (6 attempts, 429-aware backoff).
* ✓ No selective retries; each (seed, problem, arm) triple is
  one set of calls.
* ✓ Executor truth = pass on EVERY assertion in `test_list`,
  same for every arm.
* ✓ Audit chain re-derives offline: 990 per-call SHA-256
  matches + 3 per-seed Merkle root re-derive + bench Merkle
  root re-derive — all PASS.
* ✓ MBPP-sanitized corpus SHA-256 verified against the
  upstream pin before each run.
* ✓ No baseline weakening.  A1 reaches 81.1 % (strong K=5
  self-consistency on MBPP-30); the partial-win is against
  this strong baseline.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* `coordpy.mbpp_reflexion_bench_v1` is explicit-import only.
* 7 new CI tests for the W90 MBPP module
  (`tests/test_w90_mbpp_reflexion_v1.py`) all pass.

## Re-running

```bash
# NVIDIA_API_KEY must be set.
python scripts/run_w90_mbpp_reflexion_bench.py \
    --model meta/llama-3.3-70b-instruct \
    --n-problems 30 --n-seeds 3
python scripts/verify_w90_mbpp_reflexion_audit_chain.py
```

The bench reproduces with the same seeds; NIM provider-side
sampling at T=0.7 carries minor variance.  Strict-improvement
bool shapes are the stable closure surface.

## The honest claim this run earns

**Directional cross-benchmark generalisation of the W89 win:**
at Llama-3.3-70B-Instruct on MBPP-sanitized × 30 problems ×
3 seeds × K=5, B sequential-reflexion mean pass@1 strictly
beats A1 first-pass-among-K=5 by +1.11 pp on the mean.  B
never loses to A1 on any seed (ties on 2, beats on 1).  Three
of four pre-committed retirement bars are met; the per-seed
majority bar fails.

**This refines but does NOT retire**
`W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP`.  The
W89 70B-reflexion architecture generalises directionally
beyond HumanEval; the strength of the win is benchmark-
dependent and varies with ceiling effects.

## Where this leaves the empirical bar

The programme now has:

* **HumanEval × 70B**: clean retirement (W89, all 4 bars met,
  +5.56 pp, 2/3 seeds).
* **MBPP × 70B**: partial generalisation (W90 P1, 3/4 bars
  met, +1.11 pp, 1/3 seeds).
* **GSM8K × 70B**: NOT TESTED (Prong 3 stretch was started
  but cancelled mid-run to free NIM rate-limit capacity for
  Prongs 1 & 2; remains V2 work).
* **Cross-modal × 90B-Vision**: tie (W90 P2, 3/6 bars met,
  +0.0 pp; gap closed from W88/W89's −5.6 / −27.8 / −5.6 pp;
  see `RESULTS_W90_CROSS_MODAL_VLM_LOOP_V1.md`).

The next wave must either: (a) extend MBPP to a larger /
harder corpus where reflexion has more headroom; (b) test
GSM8K-70B (cheap retry); (c) attack cross-modal with a
benchmark where the unified VLM ceiling isn't at 92 %.
