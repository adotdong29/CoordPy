# W98 RealWorldQA B1 Phase 2 pilot — w98_realworldqa_pilot_b1_11b_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260525T191938Z

Total wall: 951s  
VLM model: `meta/llama-3.2-11b-vision-instruct`  
Text/solver model: `meta/llama-3.2-11b-vision-instruct`  
Parquet shard SHAs: `['0ed8b55558692309', '7dcb3ac3483362ca']`  
Corpus Merkle root: `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`  
Bench Merkle root: `dbc807a059bdb9cd6f51078b6b73ba8605861061c6d126f5ec0fa8e33051836f`  
Seed Merkle root:  `669ccedc5c181e74a3af50d42f9f223f3392e210e9bb80f9f7df4ea2f77f6bf8`  
Total NIM calls: text=150 vlm=180  

## Per-arm pass rates

* A0_text:           36.67 %
* A1_vlm K=5:        86.67 %
* B_vlm_team_v2 (B1): 80.00 %  (B − A1 = -6.67 pp; B − A0 = +43.33 pp)

Question type distribution: `{'multi_choice_letter': 18, 'numeric': 4, 'yes_no': 6, 'short_text': 2}`

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_realworldqa_subset_v1; 30 pids pre-committed.
* **2_a1_lt_90pct**: PASS — A1@K=5 = 86.67% < 90 %? (PASS)
* **3_b_strictly_beats_a1**: FAIL — B = 80.00% vs A1 = 86.67%; B > A1? False
* **4_margin_b_over_a1_ge_5pp**: FAIL — B − A1 = -6.67 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: PASS — B − A0 = +43.33 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B ≥ A1 on 27/30 problems (threshold ≥ 16)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11; OK)
* **8_audit_chain_present**: PASS — bench_merkle=dbc807a059bdb9cd…, seed_merkle=669ccedc5c181e74…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_realworldqa_answer_v1 with identical semantics; offline verifier re-checks.

## Structural verdict: `PHASE_2_FAIL`

## Direct comparison vs W97 D2-B0 (same slice)

* W98 B1 rescues vs W97 D2-B0: 4 pids (pids: ['rwqa_test_000135', 'rwqa_test_000403', 'rwqa_test_000555', 'rwqa_test_000718'])
* W98 B1 regressions vs W97 D2-B0: 5 pids (pids: ['rwqa_test_000013', 'rwqa_test_000155', 'rwqa_test_000204', 'rwqa_test_000225', 'rwqa_test_000713'])
## Overall verdict: `FAIL — W98 B1 Phase 2 KILLED at this scale; document W98-L-* carry-forward`
