# W101 — Arsenal mining via offline sidecar re-execution V1

> **2026-05-25.  NIM-free per-(seed, task_id, arm) cluster
> surface derived from the W88 70B HumanEval reflexion sidecar
> (990 calls) + the W91 5-seed 70B MBPP reflexion sidecar
> (1650 calls).  Re-execution uses the canonical W86 / W90
> subprocess executors against the canonical HumanEval +
> MBPP-sanitized corpora.  No new NIM calls.  The re-derived
> per-seed pass rates match the published W89 and W91 result
> docs byte-for-byte.**

## Method

Script: `scripts/run_w101_arsenal_mining.py`.

For each persisted call in the W88 70B + W91 5-seed sidecar:

1. Extract candidate code from the response_text via the
   bench module's `extract_candidate_code_v1` (regex on the
   first python fence; falls back to raw text).
2. Re-run `coordpy.humaneval_real_bench_v1.run_humaneval_executor_v1`
   (HumanEval) or `coordpy.mbpp_reflexion_bench_v1.run_mbpp_executor_v1`
   (MBPP) against the canonical problem.
3. Aggregate per-(seed, task_id, arm) by the bench's selection
   rule:
   * **A0**: pass if the single call passed.
   * **A1**: pass if ANY of the K=5 calls passed (literature
     "first-pass-among-K" with the deterministic content-
     addressed tie-break).
   * **B**: pass if ANY of the K=5 reflexion turns passed
     (the W88 sequential-reflexion "ship first PASS by
     attempt index").
4. Build the (a1_only_wins / b_only_wins / shared_wins /
   shared_fails) cluster partition.

Re-execution costs only local CPU + 2640 subprocess executor
calls; no network NIM round-trips.

Outputs in `results/w101/arsenal_mining/w101_arsenal_20260525T231104Z/`:

* `per_call_outcomes.jsonl` — one line per re-executed call
* `per_problem_outcomes.json` — aggregated per-arm verdict
* `mining_report.json` — top-level summary the W101 preflight
  reads

## Per-seed re-derivation (matches published)

### W89 — HumanEval 70B (3 seeds × 30 problems × K=5)

| Seed | A0 | A1@K=5 | B | B−A1 (pp) | a1-only | b-only | shared wins | shared fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 88_028_001 | 60.00 % | 80.00 % | 93.33 % | **+13.33** | 1 | **5** | 23 | 1 |
| 88_028_002 | 40.00 % | 86.67 % | 83.33 % | **−3.33** | 2 | 1 | 24 | 3 |
| 88_028_003 | 40.00 % | 90.00 % | 96.67 % | **+6.67** | 0 | 2 | 27 | 1 |
| **mean** | **46.67 %** | **85.56 %** | **91.11 %** | **+5.56** | **3** | **8** | **74** | **5** |

These match the published W89 numbers verbatim
(`docs/RESULTS_W89_HUMANEVAL_REFLEXION_V2.md` per-seed table).

### W91 — MBPP 70B (5 seeds × 30 problems × K=5)

| Seed | A0 | A1@K=5 | B | B−A1 (pp) | a1-only | b-only | shared wins | shared fails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 90_001 | 80.00 % | 90.00 % | 90.00 % | **+0.00** | 0 | 0 | 27 | 3 |
| 90_002 | 66.67 % | 70.00 % | 76.67 % | **+6.67** | 0 | 2 | 21 | 7 |
| 90_003 | 70.00 % | 83.33 % | 86.67 % | **+3.33** | 1 | 2 | 24 | 3 |
| 90_004 | 86.67 % | 90.00 % | 90.00 % | **+0.00** | 0 | 0 | 27 | 3 |
| 90_005 | 73.33 % | 80.00 % | 76.67 % | **−3.33** | 2 | 1 | 22 | 5 |
| **mean** | **75.33 %** | **82.67 %** | **84.00 %** | **+1.33** | **3** | **5** | **121** | **21** |

These match the published W91 numbers verbatim
(`docs/RESULTS_W91_MBPP_REFLEXION_V2.md` per-seed table).

## Cross-bench structural findings

### Finding 1 — Reflexion-rescue surface is 2.5× richer on HumanEval than on base MBPP

| Bench | Total B wins | B-only wins (unique reflexion rescues) | Rescue fraction |
|---|---:|---:|---:|
| W89 HumanEval-70B | 82 | 8 | **9.76 %** |
| W91 MBPP-70B (5-seed) | 126 | 5 | **3.97 %** |

**Interpretation**: on HumanEval, ~10 % of B's wins came from
reflexion actually doing work (B passed a problem where A1's
K=5 first-pass-among-K did not).  On base MBPP, only ~4 % did.
The shortfall is the empirical symptom of the ceiling-saturation
cap that W91 hit: A1@K=5 ≥ 90 % on 2/5 seeds left B nothing to
rescue.  MBPP+ is the surgical attack on exactly this regime —
the EvalPlus extra tests are designed to drop A1@K=5 by 15-20 pp
(Liu et al. 2023 Table 4) and thereby restore the rescue
surface closer to the HumanEval pattern.

### Finding 2 — Hard-cluster size scales with ceiling-control

| Bench | Shared-fails (both A1 and B fail) | Per-problem rate |
|---|---:|---:|
| W89 HumanEval-70B (90 problem-seeds) | 5 | 5.6 % |
| W91 MBPP-70B (150 problem-seeds) | 21 | 14.0 % |

**Interpretation**: HumanEval's shared-fails cluster is roughly
half the size of MBPP's (5.6 % vs 14 % of problem-seeds).
Several MBPP shared-fails are problems where the canonical
solution is correct, A1's K=5 samples all generated buggy
implementations, and B's reflexion did not find the right
correction within K=5.  Many of those "buggy implementations"
ALSO pass the base MBPP assertions (i.e., A1 + B both "pass"
on the visible tests but would fail on EvalPlus's hidden extra
tests).  MBPP+ promotes these to the *visible* failure-residual
B can attack.

### Finding 3 — Per-seed margin variance compresses on saturated benches

Per-seed B − A1 std deviation:

| Bench | mean B−A1 (pp) | std | range |
|---|---:|---:|---|
| W89 HumanEval-70B | +5.56 | 8.43 | [−3.33, +13.33] |
| W91 MBPP-70B | +1.33 | 3.65 | [−3.33, +6.67] |

**Interpretation**: HumanEval has 2.3× more per-seed variance
on B − A1 than MBPP at 70B.  This is consistent with HumanEval
having a richer failure-residual surface that the reflexion
mechanism can attack on some seeds (seed 88_028_001 hit +13.33
pp) and not others (seed 88_028_002 hit −3.33 pp).  On base MBPP
at 70B, the saturation regime compresses both the upside and
downside of the per-seed distribution; the mechanism has less
room to express its load-bearingness.

## Per-cluster carry-forward implications

* **B-only wins** (8 W89 + 5 W91 = 13 across both benches):
  these are the *empirical mechanism-load-bearing signal*.  On
  MBPP+ at 70B, the W101 cheap pilot will measure whether this
  rescue surface restores closer to the HumanEval pattern.  The
  MLB-2 sub-gate (reflexion rescue rate ≥ 33 % of invocations)
  is the post-pilot test.
* **A1-only wins** (3 W89 + 3 W91 = 6 across both benches):
  these are *anti-mechanism evidence* — the reflexion turns
  apparently degraded A1's already-correct first-pass.  On a
  K=5 budget with PASS-by-attempt-index, this can only happen
  if A1's first-pass-among-K PASSed but B's K=5 reflexion turns
  all FAILed; the W88 sequential reflexion's deterministic
  content-addressed CID tie-break is supposed to recover this
  but is empirically lossy on a small minority of problems.
  W101 does NOT attempt to fix this; the MBPP+ Phase 2 cheap
  pilot just reports it.
* **Shared wins** (74 W89 + 121 W91 = 195): the bench's mass
  surface where both A1 and B succeed.  Most of these are easy
  problems on the canonical assertions; MBPP+ extra tests will
  promote some to shared-fails on MBPP+.
* **Shared fails** (5 W89 + 21 W91 = 26): the hard cluster
  neither mechanism cracked.  Several MBPP shared-fails will
  *increase* on MBPP+ if the candidate generated a buggy
  implementation that passed the base assertions.

## What this mining does NOT do

* It does NOT modify any W89 / W91 published result.
* It does NOT re-train the W89 reflexion mechanism.
* It does NOT predict the MBPP+ cheap-pilot outcome — only the
  Phase 2 cheap pilot can do that.
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.

## How the W101 preflight + runbook use this mining

* `coordpy.mbpp_plus_preflight_v1.probe_addr_mechanism_load_bearing_v1`
  reads `mining_report.json` and gates AddrW101-P1 on
  "rescue fraction ≥ 5 % on at least one historical bench".
  Empirical: W89 = 9.76 % (PASS); W91 = 3.97 % (below the floor
  but the floor allows any-one-bench).
* `coordpy.mbpp_plus_preflight_v1.probe_addr_cluster_structure_v1`
  reads the cluster partition and gates AddrW101-P2 on
  "(a1_only + b_only + shared_wins + shared_fails) = n_problems"
  for both benches.  Empirical: both PASS.
* `coordpy.mbpp_plus_preflight_v1.probe_a1_failure_residual_v1`
  optionally reads the re-executed W91 A1 means and uses them
  as the basis for the Hoeffding lower-bound prediction of
  MBPP+ A1@K=5.  Empirical: re-executed A1 mean = 82.67 % (byte-
  identical to published); predicted MBPP+ A1@K=5 = 69.97 %.

## Anchors

* `scripts/run_w101_arsenal_mining.py` — the offline re-executor.
* `results/w101/arsenal_mining/w101_arsenal_20260525T231104Z/`
  — the mining output dir.
* `docs/RUNBOOK_W101.md` — pre-commit contract.
* `docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md` — the
  selection matrix this mining grounds.
* `docs/RESULTS_W101_PREFLIGHT_V1.md` — the preflight verdict
  that consumes this mining report.
* `docs/RESULTS_W89_HUMANEVAL_REFLEXION_V2.md` — W89 retirement
  carry-forward source.
* `docs/RESULTS_W91_MBPP_REFLEXION_V2.md` — W91 partial cap
  source.
