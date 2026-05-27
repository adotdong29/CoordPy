# W104 — HumanEval+ cross-scale Phase 2 pilot V1

> **2026-05-26.  Cross-scale HumanEval+ Phase 2 cheap pilot verdict at `meta/llama-3.1-70b-instruct` on the W103 helper-anchored 30-problem slice (BYTE-FOR-BYTE reuse; slice CID `c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2`) × K=5 = 330 NIM calls.  Cross-scale form actually achieved: cross-generation (Llama-3.1 vs Llama-3.3 at 70B).**

> **Reachability event**: pre-locked primary target `meta/llama-3.1-405b-instruct` returned HTTP 404 on the reachability smoke probe; the pre-locked backup `meta/llama-3.1-70b-instruct` was applied per the W104 RUNBOOK § Target-model selection rule.  The cross-scale shape achieved is therefore cross-generation at the same parameter scale (Llama 3.1 vs Llama 3.3 at 70B), NOT cross-scale-UP.  This is a weaker form of cross-scale than the primary target would have produced; the verdict reads against the W89 / W103 same-scale base-HumanEval template, NOT against the W96-A / W100 cross-scale-UP precedent.

## Inputs (provenance)

| Field | Value |
|---|---|
| Candidate mechanism | B (W89 sequential reflexion on HumanEval+ at K=5) |
| Target model (used) | `meta/llama-3.1-70b-instruct` |
| Pre-locked primary target | `meta/llama-3.1-405b-instruct` (unreachable (HTTP 404)) |
| Pre-locked backup target | `meta/llama-3.1-70b-instruct` (used) |
| Cross-scale form achieved | cross-generation (Llama-3.1 vs Llama-3.3 at 70B) |
| HumanEval+ corpus SHA-256 | `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492` |
| Slice CID (helper-priority order; W103 reused) | `c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2` |
| Slice CID (bench iteration order; W103 reused) | `d5364a2f5a6ab3d6febe69b99d8424f75a54ad6f1dbde9e5e8e2d7e62c9e3052` |
| Preflight verdict cid (W102/W103 re-used) | `4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4` |
| Helper proposal CID (humaneval_plus; W103 reused) | `a5b3a2c15c4e3a0c3f33a47ed80334b759065b72daf76e2818a230d6a7256327` |
| Mining report CID | `63465b7777ed05bda48f9bb02edd4aaa30e6c78315829f7e57b3e80973664dd6` |
| Bench Merkle root | `1a3d93aa5a2119338a9dca94e4d23e9275d921631b9b712c4f13f3e1e99d0171` |
| Pilot wall | 7506.9 s |
| Seed (candidate sampling RNG) | 104001 |
| n_problems | 30 |
| Target-selection-rule version | `coordpy.w104_target_selection_rule.v1` |
| Arsenal-mining prior (RECORDED; NOT a Phase 2 gate input) | B−A1 = +5.56 pp; rescue = 9.21 % (W102 cross-bench extension) |
| W103 70B empirical anchor (RECORDED; NOT a Phase 2 gate input) | B−A1 = +20.00 pp; MLB-2 = 47.06 % |

## Headline numbers

| Arm | Pass rate (n / N) |
|---|---:|
| A0 | 46.67% (14 / 30) |
| A1 @ K=5 | 53.33% (16 / 30) |
| B (sequential reflexion K=5) | 63.33% (19 / 30) |
| **B − A1** | **+10.00 pp** |
| **B − A0** | **+16.67 pp** |

## Per-problem cluster surface

| Cluster | Count |
|---|---:|
| a1_only_wins (B regression) | 0 |
| b_only_wins (B rescue) | 3 |
| shared_wins | 16 |
| shared_fails (hard cluster) | 11 |

## Phase 2 gates

| # | Gate | Verdict |
|---|---|---|
| 1 | Slice pre-committed (W103 byte-equal reuse; 30 problems; CID locked BEFORE NIM call) | PASS |
| 2 | A1 @ K=5 < 90 % | PASS |
| 3 | B > A1 | PASS |
| 4 | B − A1 ≥ +5 pp | PASS |
| 5 | B > A0 by ≥ +5 pp | PASS |
| 6 | Per-problem majority B ≥ A1 (≥ 16/30; observed 30/30) | PASS |
| 7 | Budget exact (1 + 5 + 5 = 11 calls / problem) | PASS |
| 8 | Audit chain re-derives offline | PASS |
| 9 | Executor stays clean | PASS |

## MLB sub-gates (mechanism-load-bearingness)

| Sub-gate | Threshold | Observed | Verdict |
|---|---:|---:|---|
| **MLB-1** reflexion-cycle invocation rate | ≥ 33 % | 56.67 % (17/30) | PASS |
| **MLB-2** reflexion rescue rate | ≥ 33 % | 35.29 % (6/17) | PASS |

## Cross-scale comparator block (vs W103 70B)

| Field | Scale A (W103 70B) | Scale B (W104 meta/llama-3.1-70b-instruct) |
|---|---|---|
| Model | `meta/llama-3.3-70b-instruct` | `meta/llama-3.1-70b-instruct` |
| Bench Merkle | `68f4a9669f1bd03e...` | `1a3d93aa5a211933...` |
| MLB-1 invocation | 56.67% | 56.67% |
| MLB-2 rescue | 47.06% | 35.29% |
| B − A1 (pp) | +20.00 | +10.00 |

* **Cross-scale shift on B − A1**: -10.00 pp
* **Cross-scale shift on MLB-2**: -11.77 pp

### Cluster-shift aggregate

| Shift | Count |
|---|---:|
| `stayed` | 8 |
| `improved` | 11 |
| `regressed` | 11 |
| `flipped` | 0 |

## Verdict

**9 of 9 Phase 2 gates PASS.**  MLB sub-gates: MLB-1 = PASS, MLB-2 = PASS.

**Verdict label: `PASS_MECHANISM_DRIVEN`**.

### Decision applied per the pre-committed W104 RUNBOOK

* Cheap pilot PASSes 9/9 + MLB sub-gates clearing → **Branch A** of the W104 RUNBOOK § Planning lane.
* Carry-forward registered: `W104-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-405B-PASS` (single-seed cross-scale cheap-pilot PASS at `meta/llama-3.1-70b-instruct`; NOT a multi-scale retirement).
* **W105 = HumanEval+ Phase 3 retirement bench** (3 seeds × 100 problems × K=5 × 2 scales = 6 600 NIM calls) is ENTITLED.  Slice pack pre-built in `data/w105/phase3_slice_pack/<RUN>/slice_pack.json` (pack CID `8be55f3bf1650df3...`).
* `COO-9` remains the lead path.  Programme entitlement: the W89 sequential-reflexion mechanism extends to HumanEval+ across TWO model classes (cross-generation (Llama-3.1 vs Llama-3.3 at 70B)) at Phase 2 cheap-pilot quality.  Retirement-grade generalisation still requires W105 Phase 3 multi-seed.

## Honest framing

* **Backup-target reality**: pre-locked primary `meta/llama-3.1-405b-instruct` was unreachable (HTTP 404 on NIM); the pre-locked backup `meta/llama-3.1-70b-instruct` was used.  The cross-scale form actually achieved is cross-generation (Llama-3.1 vs Llama-3.3 at 70B) — WEAKER than the primary target's cross-scale-UP form would have been.
* Honest framing: this is NOT a 70B → 405B cross-scale test.  It is a 70B-Llama-3.3 → 70B-Llama-3.1 cross-generation test at the same parameter scale.
* This is a 1-seed × 30-problem cheap pilot AT THE Phase 2 SIZE.  It is NOT retirement evidence — retirement requires W105 Phase 3 multi-seed (3 seeds × 100 problems × K=5 × 2 scales).
* The W89 70B HumanEval K=5 multi-seed retirement remains the only confirmed multi-seed same-budget multi-agent superiority retirement in the programme.
* The slice is BYTE-EQUAL to W103 (same 30 task_ids in the same bench-iteration order).  Cross-scale comparator catches mix-ups at write time.
* The W104 arsenal-mining prior + W103 70B empirical anchor are RECORDED in provenance but are NOT Phase 2 gate inputs.  Per the W102 anti-pattern carry-forward, cross-bench / cross-scale priors are UPPER BOUNDS only; fresh-K=5 sampling at the cross-scale target is the ground truth.

## Anchors

* `results/w104/humaneval_plus_cross_scale_pilot/w104_humaneval_plus_cross_scale_pilot_meta_llama-3.1-70b-instruct_20260526T215829Z/humaneval_plus_reflexion_bench_report.json` — bench report.
* `results/w104/humaneval_plus_cross_scale_pilot/w104_humaneval_plus_cross_scale_pilot_meta_llama-3.1-70b-instruct_20260526T215829Z/humaneval_plus_reflexion_calls.jsonl` — per-call sidecar.
* `results/w104/humaneval_plus_cross_scale_pilot/w104_humaneval_plus_cross_scale_pilot_meta_llama-3.1-70b-instruct_20260526T215829Z/provenance.json` — explicit provenance fields.
* `results/w104/humaneval_plus_cross_scale_pilot/w104_humaneval_plus_cross_scale_pilot_meta_llama-3.1-70b-instruct_20260526T215829Z/cross_scale_comparator_report.json` — cross-scale comparator JSON.
* `results/w104/humaneval_plus_cross_scale_pilot/w104_humaneval_plus_cross_scale_pilot_meta_llama-3.1-70b-instruct_20260526T215829Z/cross_scale_comparator_report.md` — comparator markdown.
* `docs/RUNBOOK_W104.md` — pre-commit contract.
* `docs/RESULTS_W104_CROSS_SCALE_COMPARATOR_V1.md` — cross-scale comparator narrative.
* `docs/RESULTS_W104_HELPER_W105_PLANNING_V1.md` — W105 Phase 3 slice pack + Branch C dispatch.
* `docs/RESULTS_W104_MILESTONE_SUMMARY_V1.md` — milestone summary.
* `docs/FRONTIER_RELEVANCE_AUDIT_W104_V1.md` — frontier audit supplement.
