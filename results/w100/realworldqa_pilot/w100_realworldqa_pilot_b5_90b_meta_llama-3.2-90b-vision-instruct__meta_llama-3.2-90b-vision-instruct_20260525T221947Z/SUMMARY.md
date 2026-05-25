# W100 RealWorldQA B5 Phase 2 90B pilot — w100_realworldqa_pilot_b5_90b_meta_llama-3.2-90b-vision-instruct__meta_llama-3.2-90b-vision-instruct_20260525T221947Z

Candidate: `B5`  
Scale: `90b`  
Total wall: 557s  
VLM model: `meta/llama-3.2-90b-vision-instruct`  
Text/solver model: `meta/llama-3.2-90b-vision-instruct`  
Parquet shard SHAs: `['0ed8b55558692309', '7dcb3ac3483362ca']`  
Corpus Merkle root: `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`  
Bench Merkle root: `0b18c0c1ed0cf2bc1d76d2135d8b864494f0aac873c767a04cd3b1a5002c9993`  
Seed Merkle root:  `dbd4c1a1c929e878e1d864a96692e816f01ed35416a81d9feb86e28aeef4dd84`  
Total NIM calls: text=102 vlm=228  

## AddrW100 NIM-free pre-flight probes

* **AddrW100_B5_P4_cross_scale_route_mass**: PASS — Question type distribution on slice = {'multi_choice_letter': 18, 'numeric': 4, 'yes_no': 6, 'short_text': 2}; route distribution = {'vlm_team_b0': 18, 'a1_vlm_k5': 12}; expected = {'vlm_team_b0': 18, 'a1_vlm_k5': 12}; expected_match=True; on-disk match: 30/30 per-pid routes equal W99 11B; mismatches=[]

## Per-arm pass rates

* A0_text:           46.67 %
* A1_vlm K=5:        80.00 %
* B (B5):              83.33 %  (B − A1 = +3.33 pp; B − A0 = +36.67 pp)

Question type distribution: `{'multi_choice_letter': 18, 'numeric': 4, 'yes_no': 6, 'short_text': 2}`
Route distribution (B5): `{'vlm_team_b0': 18, 'a1_vlm_k5': 12}`

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_realworldqa_subset_v1; 30 pids pre-committed.
* **2_a1_lt_90pct**: PASS — A1@K=5 = 80.00% < 90 %? (PASS)
* **3_b_strictly_beats_a1**: PASS — B = 83.33% vs A1 = 80.00%; B > A1? True
* **4_margin_b_over_a1_ge_5pp**: FAIL — B − A1 = +3.33 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: PASS — B − A0 = +36.67 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B ≥ A1 on 29/30 problems (threshold ≥ 16)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11; OK)
* **8_audit_chain_present**: PASS — bench_merkle=0b18c0c1ed0cf2bc…, seed_merkle=dbd4c1a1c929e878…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_realworldqa_answer_v1 with identical semantics; offline verifier re-checks.

## Structural verdict: `PHASE_2_FAIL`

## Cross-scale comparison vs W99 11B (same slice, same candidate)

* matched problems: 30
* new wins at 90B vs 11B: 0
* new losses at 90B vs 11B: 5
* both pass: 25
* neither pass: 0
* new loss pids: ['rwqa_test_000155', 'rwqa_test_000223', 'rwqa_test_000246', 'rwqa_test_000430', 'rwqa_test_000533']

## Overall verdict: `FAIL — W100 B5 90B Phase 2 KILLED; routing-ceiling does not generalize cross-scale on this slice`
