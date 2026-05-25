# W100 RealWorldQA B2 Phase 2 90B pilot — w100_realworldqa_pilot_b2_90b_meta_llama-3.2-90b-vision-instruct__meta_llama-3.2-90b-vision-instruct_20260525T220904Z

Candidate: `B2`  
Scale: `90b`  
Total wall: 611s  
VLM model: `meta/llama-3.2-90b-vision-instruct`  
Text/solver model: `meta/llama-3.2-90b-vision-instruct`  
Parquet shard SHAs: `['0ed8b55558692309', '7dcb3ac3483362ca']`  
Corpus Merkle root: `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`  
Bench Merkle root: `e1f9493efb284cf230104ab3d2e0805b7bebfdf41547c46502534022fd17e859`  
Seed Merkle root:  `8bc346bcb57029fa4d04c4d4185adcfe277097b657e737f83a6ab59f4d9acc72`  
Total NIM calls: text=141 vlm=189  

## AddrW100 NIM-free pre-flight probes

* **AddrW100_B2_P5_cross_scale_rescue_prior**: PASS — W99 B2 11B per-problem: n=30; B2 PASS=30; final-VLM invoked=3; final-VLM rescued=3.  W96-D 90B residual=20.51 pp; expected unique-A1-rescues at 90B (by residual) ≥ 6 (threshold ≥ 3 for rescue-prior stability)

## Per-arm pass rates

* A0_text:           46.67 %
* A1_vlm K=5:        76.67 %
* B (B2):              73.33 %  (B − A1 = -3.33 pp; B − A0 = +26.67 pp)

Question type distribution: `{'multi_choice_letter': 18, 'numeric': 4, 'yes_no': 6, 'short_text': 2}`
Final VLM invocations (B2): 9  
Final VLM rescues (B2): 1  

## Pre-committed Phase 2 gates

* **1_slice_pre_committed**: PASS — slice taken by select_realworldqa_subset_v1; 30 pids pre-committed.
* **2_a1_lt_90pct**: PASS — A1@K=5 = 76.67% < 90 %? (PASS)
* **3_b_strictly_beats_a1**: FAIL — B = 73.33% vs A1 = 76.67%; B > A1? False
* **4_margin_b_over_a1_ge_5pp**: FAIL — B − A1 = -3.33 pp (threshold ≥ +5 pp)
* **5_b_over_a0_ge_5pp**: PASS — B − A0 = +26.67 pp (image must be load-bearing; threshold ≥ +5 pp)
* **6_per_problem_b_ge_a1_majority**: PASS — B ≥ A1 on 27/30 problems (threshold ≥ 16)
* **7_budget_accounting_exact**: PASS — Each problem uses 1 A0 + 5 A1 + 5 B = 11 calls (expected=11; OK)
* **8_audit_chain_present**: PASS — bench_merkle=e1f9493efb284cf2…, seed_merkle=8bc346bcb57029fa…
* **9_executor_stays_clean**: PASS — Executor invariants intact: every arm routes through evaluate_realworldqa_answer_v1 with identical semantics; offline verifier re-checks.

## Mechanism-load-bearingness sub-gates (B2 only)

* **MLB_1_invocation_rate_le_50pct**: PASS — Final-VLM invoked on 9/30 = 30.00% of problems (threshold ≤ 50 %)
* **MLB_2_rescue_rate_ge_33pct**: FAIL — Final-VLM rescued 1/9 = 11.11% of invocations (threshold ≥ 33.33 %)

## Structural verdict: `PHASE_2_FAIL`

## Cross-scale comparison vs W99 11B (same slice, same candidate)

* matched problems: 30
* new wins at 90B vs 11B: 0
* new losses at 90B vs 11B: 8
* both pass: 22
* neither pass: 0
* new loss pids: ['rwqa_test_000013', 'rwqa_test_000111', 'rwqa_test_000155', 'rwqa_test_000223', 'rwqa_test_000533', 'rwqa_test_000615', 'rwqa_test_000713', 'rwqa_test_000718']

## Overall verdict: `FAIL — W100 B2 90B Phase 2 KILLED; promote COO-9 per W100 code-pivot contingency; document W100-L-* carry-forward`
