# W97 RealWorldQA Phase 2 pilot — w97_realworldqa_pilot_11b_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260525T182409Z

Total wall: 584s  
VLM model: `meta/llama-3.2-11b-vision-instruct`  
Text/solver model: `meta/llama-3.2-11b-vision-instruct`  
Parquet shard SHAs: `['0ed8b55558692309', '7dcb3ac3483362ca']`  
Corpus Merkle root: `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`  
Bench Merkle root: `c2454c7fee69fec7e5f80efc0dcec9a82b2c2f839186e1d92f0cf4877bb0e234`  
Seed Merkle root:  `96c3337354c6522e46c2ddba1ecb6d415d71e4a89de60fa91597b76c7e8efb55`  
Total NIM calls: text=150 vlm=180  

## Per-arm pass rates

* A0_text:      36.67 %
* A1_vlm K=5:   90.00 %
* B_vlm_team:   83.33 %  (B − A1 = -6.67 pp; B − A0 = +46.67 pp)

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_realworldqa_subset_v1; 30 pids pre-committed.
* **2_a1_lt_90pct**: FAIL — A1@K=5 = 90.00% < 90 %? (FAIL — saturated)
* **3_b_strictly_beats_a1**: FAIL — B = 83.33% vs A1 = 90.00%; B > A1? False
* **4_margin_b_over_a1_ge_5pp**: FAIL — B − A1 = -6.67 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: PASS — B − A0 = +46.67 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B ≥ A1 on 25/30 problems (threshold ≥ 16)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11; OK)
* **8_audit_chain_present**: PASS — bench_merkle=c2454c7fee69fec7…, seed_merkle=96c3337354c6522e…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_realworldqa_answer_v1 with identical semantics; offline verifier re-checks.

## Overall verdict: `FAIL — W97 D2-B0 Phase 2 KILLED at this scale; document W97-L-* carry-forward`
