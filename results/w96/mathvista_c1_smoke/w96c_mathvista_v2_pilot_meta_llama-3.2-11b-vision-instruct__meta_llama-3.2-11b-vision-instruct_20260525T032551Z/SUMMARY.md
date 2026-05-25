# W96-C V2 MathVista Phase 2 pilot — w96c_mathvista_v2_pilot_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260525T032551Z

Total wall: 25s  
VLM model: `meta/llama-3.2-11b-vision-instruct`  
Text/solver model: `meta/llama-3.2-11b-vision-instruct`  
Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`  
Corpus Merkle root: `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`  
Bench Merkle root: `b7408b15b22eae4d26265e51f7845390d15a40eeaa095adbe6d57544148aa8ee`  
Seed Merkle root:  `59722efaefbab02e926bfdfc65f2b9487e5eff822ec20c97986ebfe0329b867e`  
Total NIM calls: text=4 vlm=7  

## Per-arm pass rates

* A0_text:           100.00 %
* A1_vlm K=5:        0.00 %
* B_vlm_team_v2:     100.00 %  (B_v2 − A1 = +100.00 pp; B_v2 − A0 = +0.00 pp)

## V2 verifier-rescue accounting

* Text-only PASS (W95-B0-style win): [1] / 1 (per-seed)
* Verifier-rescue (text-only FAIL → VLM-verifier PASS): [0] / 1 (per-seed)

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_mathvista_subset_v1; 1 pids pre-committed.
* **2_a1_lt_90pct**: PASS — A1@K=5 = 0.00% < 90 %? (PASS)
* **3_b_strictly_beats_a1**: PASS — B_v2 = 100.00% vs A1 = 0.00%; B_v2 > A1? True
* **4_margin_b_over_a1_ge_5pp**: PASS — B_v2 − A1 = +100.00 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: FAIL — B_v2 − A0 = +0.00 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B_v2 ≥ A1 on 1/1 problems (threshold ≥ 1)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B_v2 = 11 calls (expected=11)
* **8_audit_chain_present**: PASS — bench_merkle=b7408b15b22eae4d…, seed_merkle=59722efaefbab02e…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_answer_v1.

## Overall verdict: `FAIL — Phase 2 KILLED`
