# W96-C MathVista C1 preflight — 20260525T032522Z

Candidate model: `meta/llama-3.2-11b-vision-instruct`  
Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`  
Corpus Merkle:   `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`  
Decomposition argument: 1362 chars

## W95 composite verdict

- overall: `PASS`
- verdict_cid: `29345745177e59587f722cb2e7fa02b9696f125e47ab222e632f76739cd10e3c`
- P1_corpus_integrity: PASS — parquet_bytes=141568126 (range_ok=True); n_problems=1000/1000 (ok=True); n_with_image=1000; n_with_answer=1000; n_bad_answer_type=0; n_bad_question_type=0
- P2_executor_self_test: PASS — executor self-test on gold: 1000/1000 = 100.00% (threshold=98.00%)
- P3_a1_failure_residual: PASS — published single-shot for meta/llama-3.2-11b-vision-instruct = 33.00%; estimated A1@K=5 = 59.75%; estimated residual = 40.25 pp (ceiling = 80.00%); pass = True
- P4_decomposition_argument: PASS — decomp argument len=1362 (threshold=200; ok=True); geo/chart-style share in sample of 200 = 57.0% (threshold=20.0%; ok=True)

## W96-C Q4 (turn-4 upper bound)

- PASS — observed 21.79% vs threshold 50.0% (90 of 413 W95-B0 passes)

## W96-C Q5 (A1-only rescue pool)

- PASS — observed 13.0% vs threshold 5.0% (78 of 600 (seed,problem) pairs)

## Overall: `PASS`
