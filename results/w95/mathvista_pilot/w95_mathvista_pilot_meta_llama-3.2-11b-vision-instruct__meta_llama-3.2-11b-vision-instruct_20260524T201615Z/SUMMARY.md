# W95 MathVista Phase 2 pilot — w95_mathvista_pilot_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260524T201615Z

Total wall: 440s  
VLM model: `meta/llama-3.2-11b-vision-instruct`  
Text/solver model: `meta/llama-3.2-11b-vision-instruct`  
Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`  
Corpus Merkle root: `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`  
Bench Merkle root: `4f76bcd4ba605d1689103033dc1ef315befad4c521ff4851ca72d23832c11d50`  
Seed Merkle root:  `c697377f3dff8595efea176a486471ce1d79866545a2f231c8499e434b87b412`  
Total NIM calls: text=150 vlm=180  

## Per-arm pass rates

* A0_text:      36.67 %
* A1_vlm K=5:   66.67 %
* B_vlm_team:   76.67 %  (B − A1 = +10.00 pp; B − A0 = +40.00 pp)

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_mathvista_subset_v1; 30 pids pre-committed.
* **2_a1_lt_90pct**: PASS — A1@K=5 = 66.67% < 90 %? (PASS)
* **3_b_strictly_beats_a1**: PASS — B = 76.67% vs A1 = 66.67%; B > A1? True
* **4_margin_b_over_a1_ge_5pp**: PASS — B − A1 = +10.00 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: PASS — B − A0 = +40.00 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B ≥ A1 on 27/30 problems (threshold ≥ 16)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11; OK)
* **8_audit_chain_present**: PASS — bench_merkle=4f76bcd4ba605d16…, seed_merkle=c697377f3dff8595…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_answer_v1 with identical semantics; offline verifier re-checks.

## Overall verdict: `PASS — Phase 3 entitled`
