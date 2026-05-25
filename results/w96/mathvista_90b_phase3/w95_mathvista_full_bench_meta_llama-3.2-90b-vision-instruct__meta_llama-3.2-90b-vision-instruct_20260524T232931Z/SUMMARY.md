# W95 MathVista Phase 3 retirement bench — w95_mathvista_full_bench_meta_llama-3.2-90b-vision-instruct__meta_llama-3.2-90b-vision-instruct_20260524T232931Z

Total wall: 11794s  
VLM model: `meta/llama-3.2-90b-vision-instruct`  
Text/solver model: `meta/llama-3.2-90b-vision-instruct`  
Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`  
Corpus Merkle root: `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`  
Bench Merkle root: `899c213a2755b26c6caae0e0d88c1922770d5a552dbbc98b2ba27e62a9bc2c52`  
Seed Merkle root:  `d224410c57ed6843021214c0cb6656e823840204884fd11e996a8bcd9d6d106e`  
Total NIM calls: text=1500 vlm=1800  

## Per-arm pass rates

* A0_text:      28.00 %
* A1_vlm K=5:   71.33 %
* B_vlm_team:   66.33 %  (B − A1 = -5.00 pp; B − A0 = +38.33 pp)

## Pre-committed Phase 3 retirement bars (W88 6-bar shape)

* **1_b_strictly_beats_a0_mean**: PASS — B mean = 66.33% > A0 mean = 28.00%? True
* **2_b_strictly_beats_a1_mean**: FAIL — B mean = 66.33% > A1 mean = 71.33%? False
* **3_margin_b_over_a0_ge_5pp**: PASS — B − A0 = +38.33 pp (threshold ≥ +5 pp)
* **4_margin_b_over_a1_ge_5pp**: FAIL — B − A1 = -5.00 pp (threshold ≥ +5 pp)
* **5_b_beats_a0_per_seed_majority**: PASS — B > A0 on 3/3 seeds (threshold ≥ 2)
* **6_b_beats_a1_per_seed_majority**: FAIL — B > A1 on 1/3 seeds (threshold ≥ 2)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11)
* **8_audit_chain_present**: PASS — bench_merkle=899c213a2755b26c…, all 3 seed Merkle roots present
* **9_slices_pre_committed_per_seed**: PASS — 3 pre-committed slices recorded ([100, 100, 100] pids)

## Overall verdict: `FAIL — retirement bars NOT all met; document negative as W95-L-* carry-forward`
