# W103 — HumanEval+ cheap pilot Phase 2 70B V1

> **2026-05-25.  Cheap HumanEval+ pilot verdict at Llama-3.3-70B-Instruct on 1 seed × 30 problems × K=5 = 330 NIM calls.  Slice helper-anchored (`coordpy.code_slice_selector_v1` consumption; COO-14 downstream).  Pre-committed W95 9-gate Phase 2 + W101 MLB-1 + MLB-2 sub-gate evaluation.  Pilot CONDITIONAL on the W102 HumanEval+ preflight re-confirmation (7/7 PASS, verdict cid `4f57a2cf...`) + helper-anchored slice (CID `c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2`).**

## Inputs (provenance)

| Field | Value |
|---|---|
| Candidate mechanism | B (W89 sequential reflexion on HumanEval+ at K=5) |
| Target model | `meta/llama-3.3-70b-instruct` |
| HumanEval+ corpus SHA-256 | `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492` |
| Preflight verdict cid (re-confirmed W103) | `4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4` |
| Helper proposal CID (humaneval_plus) | `a5b3a2c15c4e3a0c3f33a47ed80334b759065b72daf76e2818a230d6a7256327` |
| Helper proposal CID (humaneval top-up) | `b7325b9646009a4a3fd71442cc55d3fd7c72a44690f6b6878ee5fb6d9ffcf607` |
| Mining report CID | `63465b7777ed05bda48f9bb02edd4aaa30e6c78315829f7e57b3e80973664dd6` |
| Slice CID (helper-priority order) | `c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2` |
| Slice CID (bench iteration order) | `d5364a2f5a6ab3d6febe69b99d8424f75a54ad6f1dbde9e5e8e2d7e62c9e3052` |
| Bench Merkle root | `68f4a9669f1bd03e6b3cb393a436e4f04aca034a0bad9c4b5ea8a002faabfd6d` |
| Pilot wall | 7424.3 s |
| Seed (candidate sampling RNG) | 103001 |
| n_problems | 30 |
| Arsenal-mining prior (RECORDED; NOT a Phase 2 gate input) | B−A1 = +5.56 pp; rescue = 9.21 % (W102 cross-bench extension) |

## Headline numbers

| Arm | Pass rate (n / N) |
|---|---:|
| A0 | 50.00% (15 / 30) |
| A1 @ K=5 | 50.00% (15 / 30) |
| B (sequential reflexion K=5) | 70.00% (21 / 30) |
| **B − A1** | **+20.00 pp** |
| **B − A0** | **+20.00 pp** |

## Per-problem cluster surface

| Cluster | Count |
|---|---:|
| a1_only_wins (B regression) | 1 |
| b_only_wins (B rescue) | 7 |
| shared_wins | 14 |
| shared_fails (hard cluster) | 8 |

## Phase 2 gates

| # | Gate | Verdict |
|---|---|---|
| 1 | Slice pre-committed (helper-anchored; 30 problems; CID locked BEFORE NIM call) | PASS |
| 2 | A1 @ K=5 < 90 % | PASS |
| 3 | B > A1 | PASS |
| 4 | B − A1 ≥ +5 pp | PASS |
| 5 | B > A0 by ≥ +5 pp | PASS |
| 6 | Per-problem majority B ≥ A1 (≥ 16/30; observed 29/30) | PASS |
| 7 | Budget exact (1 + 5 + 5 = 11 calls / problem) | PASS |
| 8 | Audit chain re-derives offline | PASS |
| 9 | Executor stays clean | PASS |

## MLB sub-gates (mechanism-load-bearingness)

| Sub-gate | Threshold | Observed | Verdict |
|---|---:|---:|---|
| **MLB-1** reflexion-cycle invocation rate | ≥ 33 % | 56.67 % (17/30) | PASS |
| **MLB-2** reflexion rescue rate | ≥ 33 % | 47.06 % (8/17) | PASS |

## Verdict

**9 of 9 Phase 2 gates PASS.**  MLB sub-gates: MLB-1 = PASS, MLB-2 = PASS.

**Verdict label: `PASS_MECHANISM_DRIVEN`**.

### Decision applied per the pre-committed W103 RUNBOOK

* Cheap pilot PASSes 9/9 + MLB sub-gates clearing → **Branch A** of the W103 RUNBOOK § Planning lane.
* **W104 = HumanEval+ cross-scale confirmation** at a SECOND model class (per the W96-C / W100 cross-scale discipline).  W104 RUNBOOK locks the exact target model.
* W105+ = HumanEval+ Phase 3 retirement bench (3 seeds × 100 problems × K=5) IF W104 cross-scale PASSes.
* Carry-forward registered: `W103-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-70B-PASS` (Phase 2 cheap-pilot single-seed PASS; NOT a retirement).
* `COO-9` remains the lead path — the W89 mechanism extends to a SECOND EvalPlus benchmark family at the cheap-pilot scale.
* Programme entitlement: the W89 sequential-reflexion mechanism extends to a SECOND published code benchmark family (HumanEval+ EvalPlus-hardened) at the cheap-pilot scale.  This is stronger than W102 alone (which FAILed on MBPP+ V2) but weaker than a multi-seed Phase 3 retirement.

## Honest framing

* This is a 1-seed × 30-problem cheap pilot AT THE Phase 2 SIZE.  It is NOT retirement evidence — that requires W105+ Phase 3 multi-seed (3 seeds × 100 problems × K=5).
* The W89 70B HumanEval K=5 multi-seed retirement remains the only confirmed multi-seed same-budget multi-agent superiority retirement in the programme.
* The W102 arsenal-mining HumanEval+ re-grade prior predicted B − A1 ≈ +5.56 pp on W88 historical responses re-graded against HumanEval+ extra tests.  This pilot is the fresh-K=5 sampling ground truth.  Per the W102 anti-pattern carry-forward, the prior is RECORDED but is NOT a Phase 2 gate input.
* The slice is helper-anchored (first NIM-spending pilot to consume `coordpy.code_slice_selector_v1` as a load-bearing input).  COO-14 transitions from "shipped helper with worked example" to "shipped helper as load-bearing pilot input".

## Anchors

* `results/w103/humaneval_plus_pilot/w103_humaneval_plus_pilot_meta_llama-3.3-70b-instruct_20260526T022037Z/humaneval_plus_reflexion_bench_report.json` — bench report.
* `results/w103/humaneval_plus_pilot/w103_humaneval_plus_pilot_meta_llama-3.3-70b-instruct_20260526T022037Z/humaneval_plus_reflexion_calls.jsonl` — per-call sidecar.
* `results/w103/humaneval_plus_pilot/w103_humaneval_plus_pilot_meta_llama-3.3-70b-instruct_20260526T022037Z/provenance.json` — explicit provenance fields.
* `docs/RUNBOOK_W103.md` — pre-commit contract.
* `docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md` — slice attestation.
* `docs/RESULTS_W103_HUMANEVAL_PLUS_PREFLIGHT_RECONFIRM_V1.md` — preflight re-confirmation.
* `docs/RESULTS_W103_MILESTONE_SUMMARY_V1.md` — milestone summary.
* `docs/FRONTIER_RELEVANCE_AUDIT_W103_V1.md` — frontier audit supplement.
