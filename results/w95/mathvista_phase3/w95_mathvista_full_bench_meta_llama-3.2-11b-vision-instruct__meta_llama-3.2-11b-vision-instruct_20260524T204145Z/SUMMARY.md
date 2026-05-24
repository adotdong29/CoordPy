# W95 MathVista Phase 3 retirement bench — w95_mathvista_full_bench_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260524T204145Z

Total wall: 5864s  
VLM model: `meta/llama-3.2-11b-vision-instruct`  
Text/solver model: `meta/llama-3.2-11b-vision-instruct`  
Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`  
Corpus Merkle root: `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`  
Bench Merkle root: `2257c4991e0d07c85da6f70e372b82874d8ca3b38521da029cc8734d67cf2fb7`  
Seed Merkle root:  `ccbc4e6bfa845e8c14b59b370206f4acff0dd978bcf60c1f7c0c3eac6e3a88a9`  
Total NIM calls: text=1500 vlm=1800  

## Per-arm pass rates

* A0_text:      30.33 %
* A1_vlm K=5:   67.67 %
* B_vlm_team:   71.33 %  (B − A1 = +3.67 pp; B − A0 = +41.00 pp)

## Pre-committed Phase 3 retirement bars (W88 6-bar shape)

* **1_b_strictly_beats_a0_mean**: PASS — B mean = 71.33% > A0 mean = 30.33%? True
* **2_b_strictly_beats_a1_mean**: PASS — B mean = 71.33% > A1 mean = 67.67%? True
* **3_margin_b_over_a0_ge_5pp**: PASS — B − A0 = +41.00 pp (threshold ≥ +5 pp)
* **4_margin_b_over_a1_ge_5pp**: FAIL — B − A1 = +3.67 pp (threshold ≥ +5 pp)
* **5_b_beats_a0_per_seed_majority**: PASS — B > A0 on 3/3 seeds (threshold ≥ 2)
* **6_b_beats_a1_per_seed_majority**: PASS — B > A1 on 2/3 seeds (threshold ≥ 2)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11)
* **8_audit_chain_present**: PASS — bench_merkle=2257c4991e0d07c8…, all 3 seed Merkle roots present
* **9_slices_pre_committed_per_seed**: PASS — 3 pre-committed slices recorded ([100, 100, 100] pids)

## Overall verdict: `FAIL — retirement bars NOT all met; document negative as W95-L-* carry-forward`
