# W95 — MathVista cheap-probe preflight V1

> **2026-05-24 — PREFLIGHT PASSES.  Four W95 cheap probes
> (corpus integrity, executor self-test on gold, A1 failure-
> residual estimate, decomposition argument) ALL pass against
> the canonical `AI4Math/MathVista` testmini parquet
> (SHA-256 `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`,
> 141 568 126 bytes, 1 000 problems, corpus Merkle root
> `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`).
> The W93 5-gate harness also PASSES the W95-B0 candidate
> (`vlm_reader + math_solver + executor-guided reflexion`) at
> `meta/llama-3.2-11b-vision-instruct`.  Composite verdict:
> **PASS** — Phase 2 (cheap NIM pilot) is preflight-earned per
> `docs/RUNBOOK_W95.md`.  No NIM spend yet; W95 has not earned a
> full bench.**

## Composite verdict (run `20260524T193937Z`)

* **Composite passes**: True.
* **MathVista preflight verdict CID**:
  `3602cf5d5611b89992bda38df60196f9ab9093c2ccebb8d9fd8239aba4f8f039`.
* **W93 harness verdict CID**:
  `27ca04283abd6b4820852bb2bdff02d655005eb285c36921b8a3fc8b63c52b57`.
* **Composite path**:
  `results/w95/mathvista_preflight/20260524T193937Z/composite_preflight_verdict.json`.

## W95 cheap-probe outcomes

| Probe | Verdict | Summary |
|---|---|---|
| P1 corpus integrity | PASS | parquet=141 568 126 B; 1000/1000 problems; every problem has image + answer + valid answer/question type. |
| P2 executor self-test on gold | PASS | gold-as-prediction → 1000/1000 = 100.00 % under the W95 executor. By-rule: 17 multi_choice_letter, 523 multi_choice_text, 458 numeric_tolerance, 2 text_exact (0 failures). |
| P3 A1 failure-residual estimate | PASS | published single-shot Llama-3.2-11B-Vision = 33.00 %; estimated A1@K=5 = 59.75 %; residual = 40.25 pp (ceiling 80 %). |
| P4 decomposition argument | PASS | written argument = 1086 chars (threshold 200); 57.0 % of the 200-problem sample carries geometry / chart / figure / table tags (threshold 20 %). |

## W93 5-gate harness outcomes

| Gate | Verdict | Summary |
|---|---|---|
| G1 hypothesis written | PASS | 1102 chars; names how W95-B0 differs from A1. |
| G2 sidecar evidence | PASS | composes the 4 MathVista probes (all PASS). |
| G3 adversarial ablation | PASS | structural: removing `vlm_reader` stage collapses W95-B0 to A1 unified VLM by construction. |
| G4 budget accounting | PASS | 5 model calls/problem = K=5. |
| G5 benchmark justification | PASS | 816-char argument citing W94 scouting note and the ~3-4× larger residual vs HumanEval-Visual K=5. |

## Corpus identity (anchored for downstream audit)

* Parquet URL: `https://huggingface.co/datasets/AI4Math/MathVista/resolve/main/data/testmini-00000-of-00001-725687bf7a18d64b.parquet`
* Parquet SHA-256: `373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d`
* Parquet bytes: 141 568 126
* Problem count: 1 000
* Corpus Merkle root (sha256 over sorted per-pid leaves):
  `dea27472fc12e697b1bb708d62dd4072662dcc7edd36bf89c9a9a3c6946101d5`

These four values are the W95 anti-cheat anchor.  Any future W95
bench run MUST verify the parquet against this SHA before
running a single NIM call; mismatches refuse to proceed.

## What W95 cheap probes prove

1. **MathVista is reproducible from a single canonical bytes
   payload.**  The parquet hashes to a stable SHA; the loader
   decodes deterministically into 1 000 typed capsules with
   non-empty images and valid answer schemas.
2. **The executor is honest.**  Feeding each problem's gold
   answer back through `evaluate_answer_v1` scores 1000/1000.
   No silent baseline-suppression mechanism is hiding in the
   executor.  Every arm will be judged under the same truth
   function.
3. **A1@K=5 leaves real headroom on Llama-3.2-Vision.**  Under
   the documented `single_shot → A1@K=5` estimator (correlation
   = 0.5), Llama-3.2-11B-Vision sits at ~60 % A1@K=5 with a
   40 pp failure-residual; even the larger 90 B model would
   sit at ~73 % with 27 pp residual.  This is 3–5× the residual
   HumanEval-Visual K=5 left for B to rescue (W88–W92: 8–12 pp).
4. **The W95-B0 candidate has structural surface area to work
   on.**  At least 20 % of MathVista testmini's problems are
   geometry / chart / scientific / table-style, where the
   vision-extract → math-solve decomposition is *cognitively*
   distinct from a single unified VLM forward.  The cheap
   sample puts that share at ~57 % — well above the 20 %
   threshold.

## What W95 cheap probes do NOT prove

* That the W95-B0 candidate will actually win a NIM pilot.  All
  the cheap evidence says is "not presumptively hostile".
* That MathVista is the *best* battlefield.  W94 picked it on
  argument; W95 preflight confirms the argument's premises are
  factual but not the conclusion.
* That Llama-3.2-Vision is the right model.  The published
  single-shot SOTA is a proxy; the real NIM pilot may show
  different per-seed behaviour.
* That the answer-match executor is the *only* clean evaluator.
  It is *a* clean evaluator that mirrors the canonical MathVista
  upstream.

## What is now entitled

Per `docs/RUNBOOK_W95.md` Phase 2 contract:

* **W95-B0 has earned a cheap NIM pilot** (1 seed × 30 problems
  × K=5 × Llama-3.2-11B-Vision-Instruct; budget ~330 NIM calls,
  ~30-60 min wall) under the pre-committed Phase 2 pilot gates.
* W95-B0 has NOT earned a full bench.  Phase 3 launches IFF the
  Phase 2 pilot gates ALL pass.

## What is NOT entitled

* No claim of multi-agent cross-modal team superiority on
  MathVista.  W95 has not run a single NIM call against
  MathVista yet.
* No retirement of any existing W88–W92 cross-modal
  carry-forward.  Those remain.
* No claim that this is the second confirmed same-budget
  multi-agent superiority retirement.  W89 70B-HumanEval-K=5
  remains the only such retirement.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* W95 modules (`coordpy.mathvista_loader_v1`,
  `coordpy.mathvista_executor_v1`,
  `coordpy.mathvista_preflight_v1`) are explicit-import only.

## Re-running

```bash
python scripts/run_w95_mathvista_preflight.py \
  --cache-dir ~/.cache/coordpy/mathvista \
  --out-dir results/w95/mathvista_preflight \
  --candidate-model meta/llama-3.2-11b-vision-instruct \
  --candidate-id W95-B0 \
  --expected-parquet-sha256 \
    373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d
```

Subsequent runs reuse the cached parquet and SHA-verify it before
re-running the probes; mismatches refuse to proceed.

## What W95 retires

Nothing.  No NIM run was launched.  The W95 deliverable so far
is the corpus + executor + preflight infrastructure and the
preflight-earned entitlement to a cheap pilot.

## The honest claim W95 earns at this point

**The MathVista testmini battlefield has passed all four
documented W95 cheap probes against the canonical parquet
SHA-anchor, and the W95-B0 candidate has passed all five W93
preflight gates.  The composite preflight verdict is PASS, which
under the `docs/RUNBOOK_W95.md` contract entitles W95-B0 to a
cheap NIM pilot (1 seed × 30 problems × K=5 × Llama-3.2-11B-
Vision-Instruct; ~330 NIM calls).  No NIM spend has occurred
yet.  W95 has not retired any carry-forward and is not entitled
to a stronger claim than that the cheap probes did not kill the
candidate.**
