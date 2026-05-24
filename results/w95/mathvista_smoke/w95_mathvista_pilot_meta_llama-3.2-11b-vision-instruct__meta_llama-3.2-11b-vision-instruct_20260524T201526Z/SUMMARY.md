# W95 MathVista Phase 2 pilot — w95_mathvista_pilot_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260524T201526Z

Total wall: 28s  
VLM model: `meta/llama-3.2-11b-vision-instruct`  
Text/solver model: `meta/llama-3.2-11b-vision-instruct`  
Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`  
Corpus Merkle root: `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`  
Bench Merkle root: `826c7b0f39b66bf03db31d5e0ef772336a5ffcbdb74f7b27fe3c87ebab16294c`  
Seed Merkle root:  `de74585830f342da82cd110a3395e17fcc6624df666e674692de624fa7c5beaa`  
Total NIM calls: text=5 vlm=6  

## Per-arm pass rates

* A0_text:      100.00 %
* A1_vlm K=5:   100.00 %
* B_vlm_team:   100.00 %  (B − A1 = +0.00 pp; B − A0 = +0.00 pp)

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_mathvista_subset_v1; 1 pids pre-committed.
* **2_a1_lt_90pct**: FAIL — A1@K=5 = 100.00% < 90 %? (FAIL — saturated)
* **3_b_strictly_beats_a1**: FAIL — B = 100.00% vs A1 = 100.00%; B > A1? False
* **4_margin_b_over_a1_ge_5pp**: FAIL — B − A1 = +0.00 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: FAIL — B − A0 = +0.00 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B ≥ A1 on 1/1 problems (threshold ≥ 1)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11; OK)
* **8_audit_chain_present**: PASS — bench_merkle=826c7b0f39b66bf0…, seed_merkle=de74585830f342da…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_answer_v1 with identical semantics; offline verifier re-checks.

## Overall verdict: `FAIL — Phase 2 KILLED`
