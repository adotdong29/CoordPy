# W102 — MBPP+ V2 cheap pilot Phase 2 70B V1

> **2026-05-25.  Cheap MBPP+ V2 pilot verdict at Llama-3.3-70B-Instruct on 1 seed × 30 problems × K=5 = 330 NIM calls.  Pre-committed W95 9-gate Phase 2 + W101 MLB-1 + MLB-2 sub-gate evaluation.  Slice locked at seed 101_001 BEFORE any NIM call.**

## Inputs

| Field | Value |
|---|---|
| Candidate mechanism | B (W89 sequential reflexion on MBPP+ V2 at K=5) |
| Target model | `meta/llama-3.3-70b-instruct` |
| Slice CID | `64aab030153728a11e4a39d2533ce3178fd2bd452ccb415528ef7434322a743d` |
| Bench Merkle root | `8737306d5b40d024a47936bb0a894d80d0deed5d43cac1bc2e1a4597b8f61656` |
| Pilot wall | 4742.2 s |
| Seed | 101_001 |
| n_problems | 30 |

## Headline numbers

| Arm | Pass rate (n / N) |
|---|---:|
| A0 | 73.33% (22 / 30) |
| A1 @ K=5 | 83.33% (25 / 30) |
| B (sequential reflexion K=5) | 76.67% (23 / 30) |
| **B − A1** | **-6.67 pp** |
| **B − A0** | **+3.33 pp** |

## Per-problem cluster surface

| Cluster | Count |
|---|---:|
| a1_only_wins (B regression) | 3 |
| b_only_wins (B rescue) | 1 |
| shared_wins | 22 |
| shared_fails (hard cluster) | 4 |

### Per-problem outcomes (slice seed 101_001)

| idx | task_id | A0 | A1@K=5 | B | B first-pass attempt | Cluster |
|---:|---|---|---|---|---:|---|
| 0 | Mbpp/142 | FAIL | FAIL | PASS | 1 | b_only (rescue) |
| 1 | Mbpp/255 | FAIL | FAIL | FAIL | — | shared_fail |
| 2 | Mbpp/88 | PASS | PASS | PASS | 0 | shared_win |
| 3 | Mbpp/748 | FAIL | FAIL | FAIL | — | shared_fail |
| 4 | Mbpp/109 | FAIL | PASS | FAIL | — | **a1_only (B regression)** |
| 5 | Mbpp/395 | PASS | PASS | PASS | 0 | shared_win |
| 6 | Mbpp/297 | PASS | PASS | PASS | 0 | shared_win |
| 7 | Mbpp/271 | FAIL | PASS | PASS | 0 | shared_win |
| 8 | Mbpp/389 | PASS | PASS | PASS | 0 | shared_win |
| 9 | Mbpp/282 | PASS | PASS | PASS | 0 | shared_win |
| 10 | Mbpp/242 | PASS | PASS | PASS | 0 | shared_win |
| 11 | Mbpp/741 | FAIL | FAIL | FAIL | — | shared_fail |
| 12 | Mbpp/471 | PASS | PASS | PASS | 3 | shared_win (reflexion-invoked, rescued via turn 3) |
| 13–16, 18–20, 22, 23, 25–29 | (various) | PASS | PASS | PASS | 0 | shared_win (16 problems) |
| 17 | Mbpp/424 | FAIL | PASS | FAIL | — | **a1_only (B regression)** |
| 21 | Mbpp/445 | PASS | PASS | FAIL | — | **a1_only (B regression)** |
| 24 | Mbpp/431 | FAIL | FAIL | FAIL | — | shared_fail |

### B mechanism-load-bearing diagnosis

Of B's 23 PASSes:
* **21** PASSed on attempt index 0 (initial solver call; reflexion never invoked).
* **2** PASSed via a reflexion turn (problem 0 at attempt 1; problem 12 at attempt 3).

Of the 9 problems where B attempt 0 FAILed:
* **2** were rescued by a later reflexion turn (problems 0 + 12).
* **7** stayed FAILed despite K=4 reflexion attempts (problems 1, 3, 4, 11, 17, 21, 24).
* Of those 7, **3** are **a1_only_wins** — i.i.d. K=5 sampling found a passing candidate but the cumulative-history-conditioned reflexion chain never did.  This is the W97 / W98 / W99 RealWorldQA-B0 structural pattern: conditioning on prior failures biases the next sample distribution in a way that hurts on problems where success requires sampling diversity.

**MLB sub-gate FAIL is the load-bearing diagnosis**: reflexion was invoked on only 9 / 30 problems (MLB-1 30 % < 33 % floor) AND of those, only 2 / 9 ended up PASSing (MLB-2 22 % < 33 % floor).  On this slice + seed, the W89 sequential-reflexion mechanism is *not* the structural advantage the W102 arsenal-mining prior (B − A1 = +5.28 pp on W91 historical responses re-graded against MBPP+ V2) suggested it would be on fresh sampling.

## Phase 2 gates

| # | Gate | Verdict |
|---|---|---|
| 1 | Slice pre-committed (seed 101_001; 30 problems) | PASS |
| 2 | A1 @ K=5 < 90 % | PASS |
| 3 | B > A1 | FAIL |
| 4 | B − A1 ≥ +5 pp | FAIL |
| 5 | B > A0 by ≥ +5 pp | FAIL |
| 6 | Per-problem majority B ≥ A1 (≥ 16/30; observed 27/30) | PASS |
| 7 | Budget exact (1 + 5 + 5 = 11 calls / problem) | PASS |
| 8 | Audit chain re-derives offline | PASS |
| 9 | Executor stays clean (canonical solutions PASS on slice problems) | PASS |

## MLB sub-gates (mechanism-load-bearingness)

| Sub-gate | Threshold | Observed | Verdict |
|---|---:|---:|---|
| **MLB-1** reflexion-cycle invocation rate | ≥ 33 % | 30.00 % (9/30) | FAIL |
| **MLB-2** reflexion rescue rate | ≥ 33 % | 22.22 % (2/9) | FAIL |

## Verdict

**6 of 9 Phase 2 gates PASS.**  MLB sub-gates: MLB-1 = FAIL, MLB-2 = FAIL.

**Verdict label: `FAIL`**.

### Decision applied per the pre-committed W102 RUNBOOK

* Cheap pilot FAILs → carry-forward `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`.
* **W103 = HumanEval+ cheap pilot** using the W102-built backup-lane infrastructure (preflight 7/7 PASS; verdict cid 4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4).
* `COO-9` remains the lead path — EvalPlus-family attack on the W91 base-MBPP cap is still the right direction; the question shifts to which EvalPlus benchmark is the right battlefield.

## Honest framing

* This is a 1-seed × 30-problem cheap pilot AT THE Phase 2 SIZE.  It is NOT retirement evidence — that requires W104+ Phase 3 multi-seed (3 seeds × 100 problems × K=5).
* The W89 70B HumanEval K=5 multi-seed retirement remains the only confirmed multi-seed same-budget multi-agent superiority retirement in the programme.
* The W102 arsenal-mining cross-bench cluster surface (see `docs/RESULTS_W102_ARSENAL_MINING_V1.md`) predicted B − A1 ≈ +5.28 pp on MBPP+ V2 when re-grading the W91 70B response set; this empirical pilot tests whether the prediction holds on a NEW seed with fresh K=5 sampling.  **The prediction did NOT hold**: the empirical margin is −6.67 pp, a 11.95 pp swing below the cross-bench prior.  The cross-bench mining was an upper bound (re-graded historical responses produced under different sampling conditions); the fresh-sampling pilot is the ground truth.
* **What this means for the W101 V1 silent-degeneration finding**: V1's silent degeneration would have produced a misleading "MBPP+" PASS by silently scoring against base-MBPP only.  V2 correctly scores against the EvalPlus extra-test surface AND honestly reports the empirical FAIL.  V2 is doing exactly what preflight P5 + P6 were designed to ensure: not hiding bad news.
* **What this means for the W91 cap**: `W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP` is NOT lifted by W102.  The W101 / W102 hypothesis was that MBPP+'s extra-test surface would lift the W91 cap by relieving the ceiling-saturation regime; on the W102 slice + fresh seed, that hypothesis FAILs.  Either the seed 101_001 / 30-problem slice is an unlucky draw OR the structural problem is deeper than a ceiling-pressure issue (the MLB-2 = 22 % rescue rate suggests the reflexion mechanism's load-bearingness on this benchmark family is genuinely weaker than the W89 HumanEval rescue rate of 47 %).
* **What this entitles**: W103 = HumanEval+ cheap pilot via the W102-built backup-lane infrastructure.  HumanEval+ has the larger historical mechanism load-bearing margin (W89 retirement +5.56 pp; W88 rescue fraction 9.76 % vs W91's 3.97 %); it is the natural next attack.  COO-9 stays the lead path.

## Anchors

* `results/w102/mbpp_plus_v2_pilot/w102_mbpp_plus_v2_pilot_meta_llama-3.3-70b-instruct_20260526T001306Z/mbpp_plus_v2_reflexion_bench_report.json` — bench report.
* `results/w102/mbpp_plus_v2_pilot/w102_mbpp_plus_v2_pilot_meta_llama-3.3-70b-instruct_20260526T001306Z/mbpp_plus_v2_reflexion_calls.jsonl` — per-call sidecar.
* `docs/RUNBOOK_W102.md` — pre-commit contract.
* `docs/RESULTS_W102_MBPP_PLUS_LOADER_V2_FIX_V1.md` — V2 schema-fix doc.
* `docs/RESULTS_W102_ARSENAL_MINING_V1.md` — cross-bench mining (empirical priors).
* `docs/RESULTS_W102_MILESTONE_SUMMARY_V1.md` — milestone summary.
