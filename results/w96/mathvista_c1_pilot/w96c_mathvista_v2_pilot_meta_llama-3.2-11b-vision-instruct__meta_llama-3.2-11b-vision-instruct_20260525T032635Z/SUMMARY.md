# W96-C V2 MathVista Phase 2 pilot — w96c_mathvista_v2_pilot_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260525T032635Z

Total wall: 448s  
VLM model: `meta/llama-3.2-11b-vision-instruct`  
Text/solver model: `meta/llama-3.2-11b-vision-instruct`  
Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`  
Corpus Merkle root: `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`  
Bench Merkle root: `748212dc21e745951bf617622a6079a53271a22819dd5509d5fa77c9800f0277`  
Seed Merkle root:  `482040f1890ac87e8ed1b56460fba37bfd5489dc43ddf6c1d5ed91cc77037933`  
Total NIM calls: text=120 vlm=210  

## Per-arm pass rates

* A0_text:           36.67 %
* A1_vlm K=5:        63.33 %
* B_vlm_team_v2:     63.33 %  (B_v2 − A1 = +0.00 pp; B_v2 − A0 = +26.67 pp)

## V2 verifier-rescue accounting

* Text-only PASS (W95-B0-style win): [19] / 30 (per-seed)
* Verifier-rescue (text-only FAIL → VLM-verifier PASS): [0] / 30 (per-seed)

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_mathvista_subset_v1; 30 pids pre-committed.
* **2_a1_lt_90pct**: PASS — A1@K=5 = 63.33% < 90 %? (PASS)
* **3_b_strictly_beats_a1**: FAIL — B_v2 = 63.33% vs A1 = 63.33%; B_v2 > A1? False
* **4_margin_b_over_a1_ge_5pp**: FAIL — B_v2 − A1 = +0.00 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: PASS — B_v2 − A0 = +26.67 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B_v2 ≥ A1 on 26/30 problems (threshold ≥ 16)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B_v2 = 11 calls (expected=11)
* **8_audit_chain_present**: PASS — bench_merkle=748212dc21e74595…, seed_merkle=482040f1890ac87e…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_answer_v1.

## Overall verdict: `FAIL — Phase 2 KILLED`
