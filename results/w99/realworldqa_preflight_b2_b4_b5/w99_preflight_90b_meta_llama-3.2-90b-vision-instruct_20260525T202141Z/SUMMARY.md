# W99 RealWorldQA preflight — w99_preflight_90b_meta_llama-3.2-90b-vision-instruct_20260525T202141Z

Candidate model: `meta/llama-3.2-90b-vision-instruct`  
Corpus Merkle root: `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`  
Verdict cid: `0bacd989850008b5416eaa6c0f4b1bacd88968a03f53d9853f5c2c454af6e907`  

## Composite preflight (W96-D P1..P4)

* B2 composite overall: `PASS`
  * P1_corpus_integrity: `PASS` — parquet_total_bytes=678342154 (range_ok=True); n_problems=765 in [700,800] (ok=True); n_with_image=765; n_with_answer=765
  * P2_executor_self_test: `PASS` — executor self-test on gold: 765/765 = 100.00% (threshold=98.00%)
  * P3_a1_failure_residual: `PASS` — published single-shot RealWorldQA for meta/llama-3.2-90b-vision-instruct = 60.00%; estimated A1@K=5 = 79.49%; estimated residual = 20.51 pp (ceiling = 80.00%); pass = True
  * P4_decomposition_argument: `PASS` — decomp argument len=1698 (threshold=200; ok=True); multimodal-completeness in sample of 500 = 100.0% (threshold=95.0%; ok=True)
* B4 composite overall: `PASS`
  * P1_corpus_integrity: `PASS` — parquet_total_bytes=678342154 (range_ok=True); n_problems=765 in [700,800] (ok=True); n_with_image=765; n_with_answer=765
  * P2_executor_self_test: `PASS` — executor self-test on gold: 765/765 = 100.00% (threshold=98.00%)
  * P3_a1_failure_residual: `PASS` — published single-shot RealWorldQA for meta/llama-3.2-90b-vision-instruct = 60.00%; estimated A1@K=5 = 79.49%; estimated residual = 20.51 pp (ceiling = 80.00%); pass = True
  * P4_decomposition_argument: `PASS` — decomp argument len=1314 (threshold=200; ok=True); multimodal-completeness in sample of 500 = 100.0% (threshold=95.0%; ok=True)
* B5 composite overall: `PASS`
  * P1_corpus_integrity: `PASS` — parquet_total_bytes=678342154 (range_ok=True); n_problems=765 in [700,800] (ok=True); n_with_image=765; n_with_answer=765
  * P2_executor_self_test: `PASS` — executor self-test on gold: 765/765 = 100.00% (threshold=98.00%)
  * P3_a1_failure_residual: `PASS` — published single-shot RealWorldQA for meta/llama-3.2-90b-vision-instruct = 60.00%; estimated A1@K=5 = 79.49%; estimated residual = 20.51 pp (ceiling = 80.00%); pass = True
  * P4_decomposition_argument: `PASS` — decomp argument len=1234 (threshold=200; ok=True); multimodal-completeness in sample of 500 = 100.0% (threshold=95.0%; ok=True)

## W99 addressability probes

* **AddrW99_B2_P1_nim_free_upper_bound** (B2): `PASS` — W97 conf-table: both_pass=22, unique_b=3, unique_a1=5, neither=0.  B2 NIM-free: best=100.00%, realistic=96.67%, conservative=90.00%.  A1@K=5 (W97)=90.00%.  Threshold: realistic ≥ A1 + 5 pp
* **AddrW99_B2_P2_short_circuit_static** (B2): `PASS` — V3 short-circuit=True, padding=True, final_vlm=True
* **AddrW99_B2_P3_final_vlm_rescue_prior** (B2): `PASS` — A1 K=5 rescues 5/5 W97 unique-A1-rescues by re-seeing the image; B2 final-turn VLM has equivalent visual access on the same cluster
* **AddrW99_B2_P4_budget_exact** (B2): `PASS` — B2 K=5
* **AddrW99_B4_P1_schema_primitives_retained** (B4): `PASS` — schema retains all W98 yes/no-fix primitives
* **AddrW99_B4_P2_hint_field_removed** (B4): `PASS` — reader prompt mentions direct_answer_hint 1× (only the explicit removal admonition; threshold ≤ 1); solver template has hint = False
* **AddrW99_B4_P3_budget_exact** (B4): `PASS` — B4 K=5
* **AddrW99_B5_P1_oracle_simulation** (B5): `PASS` — NIM-free oracle: B5=30/30 = 100.00%; A1@K=5 (W97) = 90.00%; margin = +10.00 pp (threshold ≥ +5 pp).  Routing: 18 multi-choice → D2-B0 (PASS 18/18); 12 non-mc → A1 K=5 (PASS 12/12)
* **AddrW99_B5_P2_parser_correctness** (B5): `PASS` — parser correct on 29/30 = 96.7% (threshold ≥ 90%)
* **AddrW99_B5_P3_budget_exact** (B5): `PASS` — B5 K=5 on either route

## Verdicts

* B2 overall: `PASS`  (composite=True; addr=True)
* B4 overall: `PASS`  (composite=True; addr=True)
* B5 overall: `PASS`  (composite=True; addr=True)

## Decision

Survivors: B2, B4, B5.  Per the W99 brief's expensive-run discipline (multiple cheap tries allowed where each earns it), the ranked promotion order is by NIM-free expected lift (descending): B5 (+10.00 pp), B2 (+6.67 pp), B4 (NIM-required).  Up to 2 winners may be promoted to cheap NIM pilots; B4 is gated on B2 or B5 PASS at the same scale since its prediction has no NIM-free oracle.

