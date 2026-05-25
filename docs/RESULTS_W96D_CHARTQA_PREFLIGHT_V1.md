# W96-D — ChartQA cheap preflight V1 (DECISIVE FAIL; D1 killed)

> **2026-05-25 — D1 (ChartQA) preflight FAILed composite at BOTH
> 11B and 90B on the canonical ``lmms-lab/ChartQA`` test split
> (2500 problems; SHA-anchored parquet
> `165263505f2998aba65d819b44be832edecd92d676fee2c030645f784cd55d06`).
> The W95 P3 saturation cap (A1@K=5 ≤ 80 %) is breached at both
> scales by published-SOTA estimate alone — ChartQA is
> presumptively saturated for the unified-VLM K=5 baseline; the
> residual a team could rescue is < 9 pp at either scale, far
> below the +20 pp floor.  Per the W96-D runbook's cross-
> battlefield pivot rule, D1 is killed cheaply and W96-D
> advances to D2 (RealWorldQA).  No NIM spend.**

## Configuration

| Field | Value |
|---|---|
| Battlefield | ChartQA test (lmms-lab/ChartQA) |
| Parquet URL | `https://huggingface.co/datasets/lmms-lab/ChartQA/resolve/main/data/test-00000-of-00001.parquet` |
| Parquet SHA-256 | `165263505f2998aba65d819b44be832edecd92d676fee2c030645f784cd55d06` |
| Parquet bytes | `72,610,993` |
| n_problems | `2500` (1250 human_test + 1250 augmented_test) |
| Corpus Merkle root | `e8d0942411e6dd4e70ca9a5a8c0843b6dfd27a6d489d85920faf7dbc9d10a9c9` |
| Candidate VLM family | `meta/llama-3.2-{11b,90b}-vision-instruct` |
| Decomposition argument | 1263 chars; D1-B0 (W95-B0 chart-port) |
| P3 ceiling | 80.00 % (W95 default; A1@K=5 ≤ 80 % required) |

## Probe verdicts

### 11B (`meta/llama-3.2-11b-vision-instruct`)

| Probe | Verdict | Detail |
|---|---|---|
| P1 corpus integrity | **PASS** | parquet 72.6 MB in [50, 800] MB range; 2500 problems in [2000, 3000]; n_with_image=2500; n_with_labels=2500 |
| P2 executor self-test | **PASS** | 2500/2500 = 100.00 % gold-as-prediction (threshold 98 %) |
| P3 A1 saturation | **FAIL** | published single-shot 83.40 % → A1@K=5 estimate **91.69 %** → residual **8.31 pp** (ceiling 80 %; FAIL by 11.69 pp) |
| P4 decomposition argument | **PASS** | 1263 chars (threshold 200); 100 % human-split share in first 500 rows |
| W93 G1 hypothesis | PASS | 669 chars |
| W93 G2 sidecar evidence | PASS | W95-B0 cross-benchmark prior |
| W93 G3 adversarial ablation | PASS | reader-removal coherence |
| W93 G4 budget accounting | PASS | 5 model calls / problem; matches K=5 |
| W93 G5 benchmark justification | PASS | 569 chars; non-HumanEval-Visual |
| **Overall composite** | **FAIL** | ChartQA P3 saturation is the load-bearing failure |

Run dir: `results/w96/chartqa_preflight/20260525T172052Z/`
Composite verdict CID: `e16ab7f53136a852d4d7835a7857037a104ffd7ca8cb5eaff367a60f61377db9`

### 90B (`meta/llama-3.2-90b-vision-instruct`)

| Probe | Verdict | Detail |
|---|---|---|
| P1 corpus integrity | **PASS** | identical to 11B |
| P2 executor self-test | **PASS** | identical to 11B |
| P3 A1 saturation | **FAIL** | published single-shot 85.50 % → A1@K=5 estimate **92.75 %** → residual **7.25 pp** (ceiling 80 %; FAIL by 12.75 pp) |
| P4 decomposition argument | **PASS** | identical to 11B |
| W93 G1..G5 | PASS (all 5) | identical hypothesis / evidence / ablation / budget / justification |
| **Overall composite** | **FAIL** | ChartQA P3 saturation worsens slightly at 90B (residual −1.06 pp vs 11B) |

Run dir: `results/w96/chartqa_preflight/20260525T172112Z/`
Composite verdict CID: `ff0f833bfe864d1a6bbefb6129ed07f0901f23a03784418f01ada296c1e0d8f8`

## Cross-scale interpretation

The cross-scale residual narrows from 8.31 pp at 11B to 7.25 pp at
90B (a −1.06 pp residual loss), consistent with the H2-saturation
pattern the W96-A Phase 3 evidence confirmed on MathVista:
stronger VLMs absorb more of the cheap-pilot residual into the
unified-VLM K=5 forward.  ChartQA is empirically a worse
battlefield for the W95-B0-derived team decomposition at this
candidate VLM family — at BOTH scales the unified VLM is too
strong for a +5 pp team-superiority margin to be structurally
available at K=5 byte-exact.

The W93 5-gate composite PASSes at both scales — the W96-D
architectural commitment (D1-B0) is hypothesis-coherent, with a
load-bearing reader stage, exact K=5 budget, and non-HumanEval-
Visual justification.  What fails is **the battlefield**, not
the architecture.

## Mechanism inference

The empirical signal at preflight is that **ChartQA's unified
A1@K=5 ceiling is presumptively too high** for the W95-B0
decomposition to clear the +5 pp Phase 2 bar:

* Even under the optimistic correlation-blended estimator
  (correlation=0.5; W95 default), A1@K=5 ≥ 91 % at both scales.
* The +5 pp Phase 2 bar therefore requires B ≥ 96 %, which is
  beyond the +20 pp residual floor.
* The single-shot SOTA narrows from MathVista (33 → 49 %) to
  ChartQA (83 → 86 %) — i.e., the W95 → W96-A
  retirement-margin narrowing pattern is structurally amplified
  at this benchmark.

The published SOTA values used by P3 are anchored anti-cheat
constants:

* Llama-3.2-11B-Vision-Instruct ChartQA test = 83.4 %
  (Meta release notes, Sep 2024).
* Llama-3.2-90B-Vision-Instruct ChartQA test = 85.5 %
  (Meta release notes).

Even if these values are 2-3 pp pessimistic (a normal range for
third-party evals), the resulting A1@K=5 estimate would still
exceed 88 % at both scales, leaving < 12 pp residual — still
short of the +20 pp floor.

## What the W96-D D1 line empirically rules out

* **ChartQA test (canonical split) is NOT a viable cross-modal
  battlefield for the W95-B0 architecture at K=5 byte-exact with
  Llama-3.2-Vision-Instruct.**  Adds carry-forward
  `W96-L-CHARTQA-PREFLIGHT-D1-B0-P3-SATURATION-CAP`.
* The W96-D D1-B0 architecture is hypothesis-coherent and passes
  the W93 5-gate composite at both scales; the FAIL is purely
  benchmark-level (A1@K=5 saturation), not architecture-level.

## What is NOT yet entitled

* No NIM smoke test was performed against the 11B / 90B endpoint
  for D1.  Total NIM spend on W96-D D1 = **zero**.
* No new carry-forward retirement.
* W95 / W96-A / W96-C MathVista carry-forwards are NOT retired.

## Next moves per the W96-D runbook

1. **Pivot to D2 (RealWorldQA)** — covered in
   `docs/RESULTS_W96D_REALWORLDQA_PREFLIGHT_V1.md`.
2. If D2 also fails preflight, advance to W96-C C2
   (tool-augmented solver) or further battlefield scouting per
   the Linear-recommended ordering.

## Discipline validation

This is the W93 → W94 → W95 → W96-A → W96-C → W96-D preflight-
first + cross-scale discipline working as designed:

* W93: 3 candidates killed by cheap preflight (no NIM spend).
* W94: K=10 cross-modal candidate killed in 90-min cheap pilot;
  MathVista selected as W95 battlefield.
* W95: Phase 2 single-seed +10 pp narrowed to Phase 3 multi-seed
  +3.67 pp (NOT retirement).
* W96-A: Phase 2 single-seed +10 pp at 90B narrowed to Phase 3
  multi-seed −5.00 pp (DECISIVE NEGATIVE).
* W96-C C1: Phase 2 cross-scale ambiguous (11B FAIL +0.00 pp;
  90B PASS +13.33 pp but mechanism not load-bearing);
  Phase 3 not entitled.
* **W96-D D1: preflight composite FAIL at both 11B and 90B by
  P3 saturation (residual 8.31 / 7.25 pp << 20 pp floor); 0 NIM
  spend; battlefield killed cheaply.**

## Anti-cheat (carry-forward from W88–W96-C)

* Parquet SHA-256 anchored at preflight time (recorded above).
* No selective retries.
* Executor truth = `evaluate_chartqa_answer_v1` for every arm
  (none run yet; would carry into the pilot).
* No LLM judge anywhere.
* Published-SOTA values used by P3 are anchored anti-cheat
  constants in `coordpy.chartqa_preflight_v1.CHARTQA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL`;
  updates are explicit code changes.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New modules `coordpy.chartqa_loader_v1`,
  `coordpy.chartqa_executor_v1`,
  `coordpy.chartqa_preflight_v1` remain explicit-import only.

## Carry-forwards added

* `W96-L-CHARTQA-PREFLIGHT-D1-B0-P3-SATURATION-CAP` — the
  ChartQA test split (lmms-lab/ChartQA, 2500 problems) is
  presumptively saturated for Llama-3.2-Vision-Instruct K=5 at
  both 11B and 90B; estimated A1@K=5 residual = 8.31 / 7.25 pp
  << the W95 +20 pp floor; cross-modal team superiority at K=5
  byte-exact is not structurally available on this benchmark
  for the W95-B0-derived architecture.

## Carry-forwards retired

None.

## Re-running

```bash
.venv/bin/python scripts/run_w96d_chartqa_preflight.py \
  --candidate-model meta/llama-3.2-11b-vision-instruct

.venv/bin/python scripts/run_w96d_chartqa_preflight.py \
  --candidate-model meta/llama-3.2-90b-vision-instruct
```

## The honest claim W96-D D1 preflight earns

**On the canonical `lmms-lab/ChartQA` test split (2500
problems; parquet SHA-anchored), the W96-D D1-B0 (W95-B0-derived
chart-extraction team) preflight FAILs the W95 P3 saturation
ceiling at both Llama-3.2-11B-Vision-Instruct (A1@K=5 estimate
91.69 %; residual 8.31 pp) and Llama-3.2-90B-Vision-Instruct
(A1@K=5 estimate 92.75 %; residual 7.25 pp) under the
correlation-blended estimator at K=5 (correlation=0.5).  The
W93 5-gate composite PASSes at both scales (the architecture is
hypothesis-coherent), but P3's saturation cap is breached by
> 11 pp at either scale.  Per the W96-D runbook's cross-
battlefield pivot rule, D1 is killed cheaply and W96-D advances
to D2 (RealWorldQA).  No NIM spend on D1.  Adds carry-forward
`W96-L-CHARTQA-PREFLIGHT-D1-B0-P3-SATURATION-CAP`.**
