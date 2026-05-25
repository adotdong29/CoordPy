# W96-D RealWorldQA preflight — 20260525T172641Z

Candidate model: `meta/llama-3.2-90b-vision-instruct`  
Parquet URLs:    `https://huggingface.co/datasets/lmms-lab/RealWorldQA/resolve/main/data/test-00000-of-00002.parquet, https://huggingface.co/datasets/lmms-lab/RealWorldQA/resolve/main/data/test-00001-of-00002.parquet`  
Shard SHA-256:   `0ed8b555586923099bd5d6ba5dd8b656b403ccfc418881facd237a6d6fe64952, 7dcb3ac3483362ca082cd4cddd0ab1389e9a276f1aefd4603fde0a2ce6bc74d0`  
Total bytes:     `678342154`  
Corpus n:        `765`  
Corpus Merkle:   `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`  
Decomposition argument: 1083 chars
P3 ceiling (max A1@K=5): 80.00%

## RealWorldQA composite verdict (P1..P4)

- overall: `PASS`
- verdict_cid: `1e38d04b97c69d8257d6eff6b30992d5ed57bbac06c1a648178b464951ebe4af`
- P1_corpus_integrity: PASS — parquet_total_bytes=678342154 (range_ok=True); n_problems=765 in [700,800] (ok=True); n_with_image=765; n_with_answer=765
- P2_executor_self_test: PASS — executor self-test on gold: 765/765 = 100.00% (threshold=98.00%)
- P3_a1_failure_residual: PASS — published single-shot RealWorldQA for meta/llama-3.2-90b-vision-instruct = 60.00%; estimated A1@K=5 = 79.49%; estimated residual = 20.51 pp (ceiling = 80.00%); pass = True
- P4_decomposition_argument: PASS — decomp argument len=1083 (threshold=200; ok=True); multimodal-completeness in sample of 500 = 100.0% (threshold=95.0%; ok=True)

## W93 5-gate composite verdict (G1..G5)

- overall: `PASS`
- verdict_cid: `adfe2d135056a2968de83126aee3d18d34fe537764844bb0ae544416322ff540`
- G1_hypothesis_written: PASS — hypothesis length 692 chars; present
- G2_sidecar_evidence: PASS — W95-B0 architecture has empirical same-budget +3.67 pp Phase 3 (MathVista 11B) and +10 pp Phase 2 (MathVista 11B / 90B) cross-modal evidence at K=5 byte-exact.  RealWorldQA preserves the multimodal-decomposition structural feature (perception → text reasoning) the W95-B0 architecture was built around; the ChartQA preflight FAIL on saturation grounds is benchmark-specific, not architecture-specific.
- G3_adversarial_ablation: PASS — Removing the vlm_scene_reader step from D2-B0 collapses the team to a text-only reasoner K=5 (no image access), which is structurally equivalent to A0_text K=5 — expected to fail vs A1_vlm K=5 because RealWorldQA answers are not derivable from the question alone without scene perception.  D2-B0's hypothesised advantage thus relies entirely on the scene-extraction step being load-bearing.
- G4_budget_accounting: PASS — candidate uses 5 model calls/problem; A1 uses 5 → matches
- G5_benchmark_justification: PASS — benchmark='RealWorldQA-test'; is_humaneval_visual=False; justification len=647

## Overall: `PASS`
