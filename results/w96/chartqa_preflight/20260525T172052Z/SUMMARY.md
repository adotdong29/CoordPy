# W96-D ChartQA preflight — 20260525T172052Z

Candidate model: `meta/llama-3.2-11b-vision-instruct`  
Parquet URL:     `https://huggingface.co/datasets/lmms-lab/ChartQA/resolve/main/data/test-00000-of-00001.parquet`  
Parquet SHA-256: `165263505f2998aba65d819b44be832edecd92d676fee2c030645f784cd55d06`  
Parquet bytes:   `72610993`  
Corpus n:        `2500`  
Corpus Merkle:   `e8d0942411e6dd4e70ca9a5a8c0843b6dfd27a6d489d85920faf7dbc9d10a9c9`  
Decomposition argument: 1263 chars
P3 ceiling (max A1@K=5): 80.00%

## ChartQA composite verdict (P1..P4)

- overall: `FAIL`
- verdict_cid: `e16ab7f53136a852d4d7835a7857037a104ffd7ca8cb5eaff367a60f61377db9`
- P1_corpus_integrity: PASS — parquet_bytes=72610993 (range_ok=True); n_problems=2500 in [2000,3000] (ok=True); n_with_image=2500; n_with_labels=2500
- P2_executor_self_test: PASS — executor self-test on gold: 2500/2500 = 100.00% (threshold=98.00%)
- P3_a1_failure_residual: FAIL — published single-shot ChartQA for meta/llama-3.2-11b-vision-instruct = 83.40%; estimated A1@K=5 = 91.69%; estimated residual = 8.31 pp (ceiling = 80.00%); pass = False
- P4_decomposition_argument: PASS — decomp argument len=1263 (threshold=200; ok=True); human-split share in sample of 500 = 100.0% (threshold=20.0%; ok=True)

## W93 5-gate composite verdict (G1..G5)

- overall: `PASS`
- verdict_cid: `c35ebfda7fee351526dcadae2f39e237d7c7c072ff340020b893c0008635ad97`
- G1_hypothesis_written: PASS — hypothesis length 669 chars; present
- G2_sidecar_evidence: PASS — W95-B0 architecture has empirical same-budget +3.67 pp Phase 3 (MathVista 11B) and +10 pp Phase 2 (MathVista 11B / 90B) cross-modal evidence at K=5 byte-exact.  ChartQA has explicit recoverable chart structure (axes, legend, data values) better matched to W95-B0's vlm_reader + text_solver decomposition than MathVista's diverse figure / geometry / table / chart mix.
- G3_adversarial_ablation: PASS — Removing the vlm_chart_reader step from D1-B0 collapses the team to a text-only solver K=5 (no image access), which is structurally equivalent to A0_text K=5 — expected to fail vs A1_vlm K=5 by ≥ +10 pp.  D1-B0's hypothesised advantage thus relies entirely on the chart-extraction step being load-bearing, which is the load-bearing structural feature.
- G4_budget_accounting: PASS — candidate uses 5 model calls/problem; A1 uses 5 → matches
- G5_benchmark_justification: PASS — benchmark='ChartQA-test'; is_humaneval_visual=False; justification len=569

## Overall: `FAIL`

**Per the W96-D runbook cross-battlefield pivot rule: D1 (ChartQA) is killed at this scale. Pivot to D2 (RealWorldQA) per `docs/RUNBOOK_W96D.md`.**
