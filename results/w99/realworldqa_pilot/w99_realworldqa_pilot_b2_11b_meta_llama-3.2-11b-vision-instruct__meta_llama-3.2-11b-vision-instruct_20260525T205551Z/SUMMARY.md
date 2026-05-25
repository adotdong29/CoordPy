# W99 RealWorldQA B2 Phase 2 pilot — w99_realworldqa_pilot_b2_11b_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260525T205551Z

Candidate: `B2`  
Total wall: 577s  
VLM model: `meta/llama-3.2-11b-vision-instruct`  
Text/solver model: `meta/llama-3.2-11b-vision-instruct`  
Parquet shard SHAs: `['0ed8b55558692309', '7dcb3ac3483362ca']`  
Corpus Merkle root: `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`  
Bench Merkle root: `8ad8369ada4c44aff4c031b5b9d149f3bffc406f1a493f01acf1020a879224bf`  
Seed Merkle root:  `b5c67404706a22d6563ee7069c6ae83ddb81db517cf02a7f90aaf253edd0f799`  
Total NIM calls: text=147 vlm=183  

## Per-arm pass rates

* A0_text:           36.67 %
* A1_vlm K=5:        93.33 %
* B (B2):              100.00 %  (B − A1 = +6.67 pp; B − A0 = +63.33 pp)

Question type distribution: `{'multi_choice_letter': 18, 'numeric': 4, 'yes_no': 6, 'short_text': 2}`
Final VLM invocations (B2): 3  
Final VLM rescues (B2): 3  

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_realworldqa_subset_v1; 30 pids pre-committed.
* **2_a1_lt_90pct**: FAIL — A1@K=5 = 93.33% < 90 %? (FAIL — saturated)
* **3_b_strictly_beats_a1**: PASS — B = 100.00% vs A1 = 93.33%; B > A1? True
* **4_margin_b_over_a1_ge_5pp**: PASS — B − A1 = +6.67 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: PASS — B − A0 = +63.33 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B ≥ A1 on 30/30 problems (threshold ≥ 16)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11; OK)
* **8_audit_chain_present**: PASS — bench_merkle=8ad8369ada4c44af…, seed_merkle=b5c67404706a22d6…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_realworldqa_answer_v1 with identical semantics; offline verifier re-checks.

## Structural verdict: `STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP`

## Direct comparison vs W97 D2-B0 (same slice)

* B2 rescues vs W97 D2-B0: 5 pids (pids: ['rwqa_test_000135', 'rwqa_test_000403', 'rwqa_test_000555', 'rwqa_test_000615', 'rwqa_test_000718'])
* B2 regressions vs W97 D2-B0: 0 pids (pids: [])

## Direct comparison vs W98 B1 (same slice)

* B2 rescues vs W98 B1: 6 pids (pids: ['rwqa_test_000013', 'rwqa_test_000155', 'rwqa_test_000204', 'rwqa_test_000225', 'rwqa_test_000615', 'rwqa_test_000713'])
* B2 regressions vs W98 B1: 0 pids (pids: [])

## Overall verdict: `STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP — consider 90B with written justification`
