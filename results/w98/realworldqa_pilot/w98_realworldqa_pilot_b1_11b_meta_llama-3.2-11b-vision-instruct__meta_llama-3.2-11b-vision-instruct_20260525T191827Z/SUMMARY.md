# W98 RealWorldQA B1 Phase 2 pilot — w98_realworldqa_pilot_b1_11b_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260525T191827Z

Total wall: 30s  
VLM model: `meta/llama-3.2-11b-vision-instruct`  
Text/solver model: `meta/llama-3.2-11b-vision-instruct`  
Parquet shard SHAs: `['0ed8b55558692309', '7dcb3ac3483362ca']`  
Corpus Merkle root: `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`  
Bench Merkle root: `c286b96ff1ddcf3ad08fb5d457e2440cbba9e4e3336b710bbbdcaf9131be3b74`  
Seed Merkle root:  `0c345cfa9dcd93521cc0211bfe70e99b26f08926ca54c0e108201a9ef41c8d0e`  
Total NIM calls: text=5 vlm=6  

## Per-arm pass rates

* A0_text:           0.00 %
* A1_vlm K=5:        0.00 %
* B_vlm_team_v2 (B1): 0.00 %  (B − A1 = +0.00 pp; B − A0 = +0.00 pp)

Question type distribution: `{'multi_choice_letter': 1}`

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_realworldqa_subset_v1; 1 pids pre-committed.
* **2_a1_lt_90pct**: PASS — A1@K=5 = 0.00% < 90 %? (PASS)
* **3_b_strictly_beats_a1**: FAIL — B = 0.00% vs A1 = 0.00%; B > A1? False
* **4_margin_b_over_a1_ge_5pp**: FAIL — B − A1 = +0.00 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: FAIL — B − A0 = +0.00 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B ≥ A1 on 1/1 problems (threshold ≥ 1)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11; OK)
* **8_audit_chain_present**: PASS — bench_merkle=c286b96ff1ddcf3a…, seed_merkle=0c345cfa9dcd9352…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_realworldqa_answer_v1 with identical semantics; offline verifier re-checks.

## Structural verdict: `PHASE_2_FAIL`

## Direct comparison vs W97 D2-B0 (same slice)

* W98 B1 rescues vs W97 D2-B0: 0 pids (pids: [])
* W98 B1 regressions vs W97 D2-B0: 1 pids (pids: ['rwqa_test_000155'])
## Overall verdict: `FAIL — W98 B1 Phase 2 KILLED at this scale; document W98-L-* carry-forward`
