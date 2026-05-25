# W101 — MBPP+ NIM-free preflight verdict V1

> **2026-05-25.  6 of 8 probes PASS; 2 DEFERRED on operator
> step (MBPP+ corpus fetch + SHA pin).  All 4 substantive
> probes (P3 / P4 / AddrW101-P1..P4) PASS.  The cheap NIM pilot
> is NOT YET earned; the W101 runbook's pre-committed gating
> requires P1 + P2 PASS at re-run after corpus fetch.  No
> infrastructure is built that violates the W93 preflight-first
> discipline; the deliverable is the empirically-grounded
> verdict + the conditional cheap-pilot driver.**

## Inputs

| Field | Value |
|---|---|
| Candidate mechanism | B (W89 sequential reflexion on MBPP+ at K=5) |
| Target model | `meta/llama-3.3-70b-instruct` |
| Arsenal-mining report | `results/w101/arsenal_mining/w101_arsenal_20260525T231104Z/mining_report.json` |
| MBPP+ cache | Absent (operator step pending) |
| Preflight script | `scripts/run_w101_mbpp_plus_preflight.py` |
| Preflight module | `coordpy.mbpp_plus_preflight_v1` |
| Verdict cid | `4ccc007baeb54a4f34139feb1ed7bc14363e330589e2a9db46177d6dd83f7a28` |
| Output dir | `results/w101/mbpp_plus_preflight/w101_mbpp_plus_preflight_20260525T232015Z/` |

## Per-probe verdicts

### P1 — corpus integrity: **DEFERRED**

* Reason: MBPP+ cache absent at `~/.cache/coordpy/mbpp-plus.jsonl.gz`.
* Operator step: fetch the EvalPlus MBPP+ release artifact +
  record the actual SHA-256 in `MBPP_PLUS_TRUSTED_SHA256_OVERRIDE`
  env var OR update `MBPP_PLUS_EXPECTED_SHA256_V020` in
  `coordpy/mbpp_plus_loader_v1.py`.
* Documented in `docs/RUNBOOK_W101.md` § "Phase 2 — conditional".

### P2 — executor self-test on canonical solutions: **DEFERRED**

* Reason: same gate as P1; the executor self-test runs only
  after MBPP+ corpus loads.
* Re-runs automatically once the cache + SHA pin are in place.

### P3 — A1@K=5 failure-residual estimate: **PASS**

| Field | Value |
|---|---:|
| W91 5-seed 70B MBPP A1 mean | 82.67 % |
| W91 re-executed A1 mean (arsenal mining) | 82.67 % (byte-identical) |
| Published EvalPlus drop — min (Hoeffding lower bound) | 12.7 pp |
| Published EvalPlus drop — mean | 14.78 pp |
| Published EvalPlus drop — max | 18.5 pp |
| **Predicted MBPP+ A1@K=5 (conservative)** | **69.97 %** |
| Predicted MBPP+ A1@K=5 (central) | 67.89 % |
| Predicted MBPP+ A1@K=5 (optimistic) | 64.17 % |
| A1 saturation threshold (gate 2) | 90.00 % |
| **Saturation margin** | **20.03 pp** |
| Phase 2 margin floor (+5 pp gate 4) | 5.0 pp |
| **Headroom for B − A1 ≥ +5 pp** | 25.03 pp |

Verdict: PASS.  Predicted A1@K=5 is structurally below the 90 %
saturation gate AND leaves enough headroom for B to clear the
+5 pp Phase 2 bar.

### P4 — decomposition argument: **PASS**

* 1727-char structural argument written:
  * MBPP+ extra tests (≈35× more hidden tests per problem) catch
    wrong-edge-case implementations, off-by-one bugs, corner-case
    type-coercion failures, and wrong-input-validation patterns
    — exactly the failure classes the W89 sequential-reflexion
    B-pipeline reads from the subprocess executor's stderr tail.
  * Under A1's i.i.d. sampling, each of K=5 samples sees only
    the function signature + sample assertion; EvalPlus extra
    tests are HIDDEN, so candidates that "work on the sample
    assertion but fail on edge cases" are statistically common.
  * Under B's sequential reflexion, the executor surfaces the
    FIRST extra-test failure to the next attempt's prompt via
    the stderr tail (`plus#N: AssertionError: ...`), which is
    structurally informative about the bug class.
  * The W89 retirement at +5.56 pp on HumanEval-70B is the
    explicit empirical precedent.

### AddrW101-P1 — mechanism-load-bearing prior: **PASS**

* Reflexion rescue fractions across W89 + W91:
  * W89 HumanEval-70B: **9.76 %** (8 / 82 B wins were unique
    reflexion-on-A1-failure rescues).
  * W91 MBPP-70B (5-seed): **3.97 %** (5 / 126 B wins).
* Threshold: ≥ 5 % rescue fraction on at least one historical
  bench (pre-committed BEFORE empirical extraction; the W91
  rescue fraction is itself the empirical symptom of the
  ceiling-saturation cap MBPP+ is designed to relieve).
* W89 clears with margin; AddrW101-P1 PASSes.

### AddrW101-P2 — per-problem cluster structure: **PASS**

* Cluster partition (a1_only / b_only / shared_wins /
  shared_fails) sums correctly:
  * W89: 3 + 8 + 74 + 5 = 90 (= 3 seeds × 30 problems).
  * W91: 3 + 5 + 121 + 21 = 150 (= 5 seeds × 30 problems).
* Both well-formed.

### AddrW101-P3 — cross-bench failure-residual stability: **PASS**

* Predicted MBPP+ A1@K=5 = 69.97 % (same as P3).
* Saturation margin = 20.03 pp.
* Threshold: margin ≥ 10 pp.
* AddrW101-P3 PASSes with margin.

### AddrW101-P4 — anti-pattern guard: **PASS**

* No `bounded_window` / `compaction` / `summary` /
  `context_pruning` tokens in `coordpy/mbpp_plus_reflexion_bench_v1.py`.
* W101 bench module imports only:
  * `coordpy.mbpp_plus_executor_v1` (loader of EvalPlus extra
    tests + executor)
  * `coordpy.mbpp_plus_loader_v1` (corpus + SHA verification)
  * `coordpy.mbpp_reflexion_bench_v1.extract_candidate_code_v1`
    (the existing W90 regex code extractor; reused unchanged)
* AddrW101-P4 PASSes.

## Summary

| Probe | Verdict |
|---|---|
| P1 corpus integrity | DEFERRED (operator step) |
| P2 executor self-test | DEFERRED (operator step) |
| P3 A1@K=5 failure-residual | PASS |
| P4 decomposition argument | PASS |
| AddrW101-P1 mechanism-load-bearing prior | PASS |
| AddrW101-P2 per-problem cluster structure | PASS |
| AddrW101-P3 cross-bench failure-residual stability | PASS |
| AddrW101-P4 anti-pattern guard | PASS |

**6 of 8 PASS; 2 DEFERRED; overall: cheap pilot NOT yet earned
(needs P1 + P2 PASS at re-run).**

## Decision applied per the pre-committed runbook (DECISION_LOGIC)

Per `docs/RUNBOOK_W101.md` § "Cheap NIM pilot — decision logic":

* Branch 1 (P1 + P2 both PASS → pilot ENTITLED): **NOT YET
  TRIGGERED**.  Operator must fetch MBPP+ + record SHA pin +
  re-run preflight.
* No carry-forward added by this preflight verdict (the 2
  DEFERRED probes are infrastructure conditionals, NOT
  empirical caps).
* The W101 milestone closes with the preflight infrastructure
  + arsenal mining + bench module + cheap-pilot driver script
  built and unit-tested; the NIM cheap pilot is the *next*
  operator-authorised step.

## Operator playbook to advance past DEFERRED

```bash
# 1) Fetch the EvalPlus MBPP+ release artifact.
mkdir -p ~/.cache/coordpy
curl -L -o ~/.cache/coordpy/mbpp-plus.jsonl.gz \
    https://github.com/evalplus/evalplus/releases/download/v0.2.0/MbppPlus-v0.2.0.jsonl.gz

# 2) Record the SHA-256 of the fetched artifact.
sha256sum ~/.cache/coordpy/mbpp-plus.jsonl.gz
export MBPP_PLUS_TRUSTED_SHA256_OVERRIDE=<sha_from_above>
# OR update coordpy/mbpp_plus_loader_v1.py MBPP_PLUS_EXPECTED_SHA256_V020 (preferred for audit trail).

# 3) Re-run W101 preflight WITH the executor self-test.
python scripts/run_w101_mbpp_plus_preflight.py

# 4) If preflight PASSes all 8 probes, launch the cheap pilot.
export NVIDIA_API_KEY=...
python scripts/run_w101_mbpp_plus_pilot.py \
    --model meta/llama-3.3-70b-instruct \
    --n-problems 30 --seed 101001

# 5) Evaluate against the 9 Phase 2 gates + MLB-1 + MLB-2;
#    write docs/RESULTS_W101_MBPP_PLUS_PHASE2_70B_V1.md.
```

## What this preflight does NOT do

* It does NOT spend any NIM budget.
* It does NOT predict the MBPP+ cheap-pilot outcome with high
  confidence — only the cheap pilot can do that.
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.
* It does NOT modify `coordpy/__init__.py`.
* It does NOT introduce any anti-pattern primitive (per
  AddrW101-P4).

## Anchors

* `scripts/run_w101_mbpp_plus_preflight.py` — preflight runner.
* `coordpy/mbpp_plus_preflight_v1.py` — preflight probes.
* `results/w101/mbpp_plus_preflight/w101_mbpp_plus_preflight_20260525T232015Z/verdict.json`
  — empirical verdict JSON.
* `docs/RUNBOOK_W101.md` — pre-commit contract this preflight
  satisfies.
* `docs/RESULTS_W101_ARSENAL_MINING_V1.md` — the sidecar mining
  the AddrW101 probes consume.
* `docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md` — the
  battlefield selection this preflight grounds.
