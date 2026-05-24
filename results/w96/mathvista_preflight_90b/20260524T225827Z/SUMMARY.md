# W95 MathVista preflight composite verdict — 20260524T225827Z

Candidate: `W96-A-90B`  
Candidate model: `meta/llama-3.2-90b-vision-instruct`  
Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`  
Corpus Merkle root: `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`  
Problem count: 1000  

## W95 MathVista cheap probes

* **P1_corpus_integrity**: PASS — parquet_bytes=141568126 (range_ok=True); n_problems=1000/1000 (ok=True); n_with_image=1000; n_with_answer=1000; n_bad_answer_type=0; n_bad_question_type=0
* **P2_executor_self_test**: PASS — executor self-test on gold: 1000/1000 = 100.00% (threshold=98.00%)
* **P3_a1_failure_residual**: PASS — published single-shot for meta/llama-3.2-90b-vision-instruct = 49.00%; estimated A1@K=5 = 72.77%; estimated residual = 27.23 pp (ceiling = 80.00%); pass = True
* **P4_decomposition_argument**: PASS — decomp argument len=1086 (threshold=200; ok=True); geo/chart-style share in sample of 200 = 57.0% (threshold=20.0%; ok=True)

## W93 5-gate harness

* **G1_hypothesis_written**: PASS — hypothesis length 1102 chars; present
* **G2_sidecar_evidence**: PASS — P1_corpus_integrity=PASS, P2_executor_self_test=PASS, P3_a1_failure_residual=PASS, P4_decomposition_argument=PASS
* **G3_adversarial_ablation**: PASS — structural ablation: removing the vlm_reader stage collapses W95-B0 to A1 unified-VLM, by construction (no separate extraction representation, no executor-diagnostics reflexion turns); the ablation is load-bearing for the candidate's identity.
* **G4_budget_accounting**: PASS — candidate uses 5 model calls/problem; A1 uses 5 → matches
* **G5_benchmark_justification**: PASS — benchmark='MathVista-testmini'; is_humaneval_visual=False; justification len=816

## Composite verdict: `PASS`

Proceed to W95 Phase 2 (cheap NIM pilot) per the pre-committed gates in `docs/RUNBOOK_W95.md`.
