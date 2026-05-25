# W98 RealWorldQA preflight — w98_preflight_11b_meta_llama-3.2-11b-vision-instruct_20260525T191413Z

Candidate model: `meta/llama-3.2-11b-vision-instruct`  
Corpus Merkle root: `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`  
Verdict cid: `a5d958506249292c3d88623c73001d819fefdae174fe43c3583d936c67c7bc9e`  

## Composite preflight (W96-D P1..P4)

* B1 composite overall: `PASS`
  * P1_corpus_integrity: `PASS` — parquet_total_bytes=678342154 (range_ok=True); n_problems=765 in [700,800] (ok=True); n_with_image=765; n_with_answer=765
  * P2_executor_self_test: `PASS` — executor self-test on gold: 765/765 = 100.00% (threshold=98.00%)
  * P3_a1_failure_residual: `PASS` — published single-shot RealWorldQA for meta/llama-3.2-11b-vision-instruct = 50.00%; estimated A1@K=5 = 73.44%; estimated residual = 26.56 pp (ceiling = 80.00%); pass = True
  * P4_decomposition_argument: `PASS` — decomp argument len=1607 (threshold=200; ok=True); multimodal-completeness in sample of 500 = 100.0% (threshold=95.0%; ok=True)
* B2 composite overall: `PASS`
  * P1_corpus_integrity: `PASS` — parquet_total_bytes=678342154 (range_ok=True); n_problems=765 in [700,800] (ok=True); n_with_image=765; n_with_answer=765
  * P2_executor_self_test: `PASS` — executor self-test on gold: 765/765 = 100.00% (threshold=98.00%)
  * P3_a1_failure_residual: `PASS` — published single-shot RealWorldQA for meta/llama-3.2-11b-vision-instruct = 50.00%; estimated A1@K=5 = 73.44%; estimated residual = 26.56 pp (ceiling = 80.00%); pass = True
  * P4_decomposition_argument: `PASS` — decomp argument len=1354 (threshold=200; ok=True); multimodal-completeness in sample of 500 = 100.0% (threshold=95.0%; ok=True)

## W98 addressability probes

* **AddrP1_typed_prompt_yes_no_recovery_rate** (B1): `PASS` — 3/5 W97 unique-A1-rescues have answer in prose-extractable form (threshold 3/5)
* **AddrP2_schema_coverage_of_failure_cluster** (B1): `PASS` — schema contains all required primitives
* **AddrP3_direct_vision_rescue_prior** (B2): `PASS` — A1 rescues 5/5 W97 unique-A1-rescues by re-seeing the image; B2 final-turn VLM has equivalent visual access on the failure cluster
* **AddrP4_short_circuit_preserves_wins** (B1+B2): `PASS` — V2 short-circuit ok=True; V3 short-circuit ok=True
* **AddrP5_budget_exact** (B1+B2): `PASS` — V2: 1 reader + 4 text-solver = 5 = K(5). V3: 1 reader + 3 text-solver + 1 final-vlm-or-pad = 5 = K(5).
* **AddrP6_question_type_parser_correctness** (B1): `PASS` — parser correct on 29/30 = 96.7% (threshold 90%)
* **AddrP7_b2_final_vlm_invocation_share** (B2): `PASS` — W97 D2-B0 FAILed on 5/30 = 16.7% of slice; upper bound on B2 final-VLM invocation share (threshold ≤ 30%)

## Verdicts

* B1 overall: `PASS`  (composite=True; addr=True)
* B2 overall: `PASS`  (composite=True; addr=True)

## Decision: `B1`

BOTH SURVIVED; cross-candidate decision: promote B1 (typed scene-graph + question-typed solver) to the cheap NIM pilot.  B2 (direct-vision final-turn) is deferred to W99 only if B1 pilot PASSes Phase 2 at both scales and B2's distinct mechanism remains plausibly load-bearing.

