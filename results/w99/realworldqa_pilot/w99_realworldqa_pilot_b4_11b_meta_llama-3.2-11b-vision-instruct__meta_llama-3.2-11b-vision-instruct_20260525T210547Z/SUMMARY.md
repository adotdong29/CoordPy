# W99 RealWorldQA B4 Phase 2 pilot — w99_realworldqa_pilot_b4_11b_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260525T210547Z

Candidate: `B4`  
Total wall: 648s  
VLM model: `meta/llama-3.2-11b-vision-instruct`  
Text/solver model: `meta/llama-3.2-11b-vision-instruct`  
Parquet shard SHAs: `['0ed8b55558692309', '7dcb3ac3483362ca']`  
Corpus Merkle root: `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`  
Bench Merkle root: `5a5a0731ecb197d4b1125b868993b23b277faaf84f1ee00dc17d0ba55ce5f25e`  
Seed Merkle root:  `104f4600990f8e610056431170f1cb46615c63dece38655bf8a8e05b1fc738d9`  
Total NIM calls: text=150 vlm=180  

## Per-arm pass rates

* A0_text:           36.67 %
* A1_vlm K=5:        93.33 %
* B (B4):              76.67 %  (B − A1 = -16.67 pp; B − A0 = +40.00 pp)

Question type distribution: `{'multi_choice_letter': 18, 'numeric': 4, 'yes_no': 6, 'short_text': 2}`

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_realworldqa_subset_v1; 30 pids pre-committed.
* **2_a1_lt_90pct**: FAIL — A1@K=5 = 93.33% < 90 %? (FAIL — saturated)
* **3_b_strictly_beats_a1**: FAIL — B = 76.67% vs A1 = 93.33%; B > A1? False
* **4_margin_b_over_a1_ge_5pp**: FAIL — B − A1 = -16.67 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: PASS — B − A0 = +40.00 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B ≥ A1 on 24/30 problems (threshold ≥ 16)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11; OK)
* **8_audit_chain_present**: PASS — bench_merkle=5a5a0731ecb197d4…, seed_merkle=104f4600990f8e61…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_realworldqa_answer_v1 with identical semantics; offline verifier re-checks.

## Structural verdict: `PHASE_2_FAIL`

## Direct comparison vs W97 D2-B0 (same slice)

* B4 rescues vs W97 D2-B0: 4 pids (pids: ['rwqa_test_000135', 'rwqa_test_000403', 'rwqa_test_000555', 'rwqa_test_000718'])
* B4 regressions vs W97 D2-B0: 6 pids (pids: ['rwqa_test_000013', 'rwqa_test_000076', 'rwqa_test_000204', 'rwqa_test_000438', 'rwqa_test_000441', 'rwqa_test_000713'])

## Direct comparison vs W98 B1 (same slice)

* B4 rescues vs W98 B1: 2 pids (pids: ['rwqa_test_000155', 'rwqa_test_000225'])
* B4 regressions vs W98 B1: 3 pids (pids: ['rwqa_test_000076', 'rwqa_test_000438', 'rwqa_test_000441'])

## Overall verdict: `FAIL — W99 B4 Phase 2 KILLED at this scale; document W99-L-* carry-forward`
