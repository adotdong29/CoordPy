# W95 MathVista Phase 2 pilot — w95_mathvista_pilot_meta_llama-3.2-90b-vision-instruct__meta_llama-3.2-90b-vision-instruct_20260524T230158Z

Total wall: 1274s  
VLM model: `meta/llama-3.2-90b-vision-instruct`  
Text/solver model: `meta/llama-3.2-90b-vision-instruct`  
Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`  
Corpus Merkle root: `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`  
Bench Merkle root: `0946b88c4e288f359bff0fd8c0cfe606aae01b5bbef56d9b3e8e004f7ea60ef9`  
Seed Merkle root:  `f97ae5eba02b8d5ef2a5304b942c31539c7e5c838d7de359b58b40b39808113d`  
Total NIM calls: text=150 vlm=180  

## Per-arm pass rates

* A0_text:      33.33 %
* A1_vlm K=5:   63.33 %
* B_vlm_team:   73.33 %  (B − A1 = +10.00 pp; B − A0 = +40.00 pp)

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_mathvista_subset_v1; 30 pids pre-committed.
* **2_a1_lt_90pct**: PASS — A1@K=5 = 63.33% < 90 %? (PASS)
* **3_b_strictly_beats_a1**: PASS — B = 73.33% vs A1 = 63.33%; B > A1? True
* **4_margin_b_over_a1_ge_5pp**: PASS — B − A1 = +10.00 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: PASS — B − A0 = +40.00 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B ≥ A1 on 27/30 problems (threshold ≥ 16)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11; OK)
* **8_audit_chain_present**: PASS — bench_merkle=0946b88c4e288f35…, seed_merkle=f97ae5eba02b8d5e…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_answer_v1 with identical semantics; offline verifier re-checks.

## Overall verdict: `PASS — Phase 3 entitled`
