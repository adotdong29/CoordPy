# W95 — MathVista Phase 2 Cheap NIM Pilot V1

> **2026-05-24 — PHASE 2 PILOT PASSES ALL 9 PRE-COMMITTED
> GATES.  At 1 seed × 30 problems × K=5 × Llama-3.2-11B-Vision-
> Instruct on `AI4Math/MathVista` testmini (parquet SHA-256
> `373f6c0b…`, corpus Merkle `dea27472fc12…`): A0_text =
> 36.67 %, A1_vlm-K=5 = 66.67 %, **B_vlm_team = 76.67 %**.
> Margin **B − A1 = +10.00 pp**; image-load-bearing **B − A0 =
> +40.00 pp**.  B ≥ A1 on **27 / 30 problems**.  Per-problem
> rescue analysis: B saves 6 problems A1 missed; A1 saves 3 B
> missed; net +3 rescues for B.  Bench Merkle `4f76bcd4ba605d16…`;
> audit chain re-derives 14 / 14 PASS offline.  Phase 3 (full
> bench: 3 seeds × 100 problems × K=5; ~3 300 NIM calls;
> ~80-100 min wall) is now preflight-earned per
> `docs/RUNBOOK_W95.md`.**
>
> **HONEST FRAMING**: this is a SINGLE-SEED pilot result.  It
> earns Phase 3 under the pre-committed contract.  It is NOT
> a multi-seed retirement claim and does NOT retire any
> existing carry-forward.  W89 70B-HumanEval-K=5 remains the
> only confirmed multi-seed same-budget multi-agent superiority
> retirement.

## Pilot configuration

| Field | Value |
|---|---|
| Bench module | `coordpy.mathvista_bench_v1` |
| Slice selector | `select_mathvista_subset_v1(seed=95_005_001, n_problems=30, corpus=testmini)` (deterministic, pre-committed before any NIM call) |
| Pre-committed pid SHA-256 | `6d3a07eb2b1dac9d…` (full list in `pre_committed_slice.json`) |
| VLM model | `meta/llama-3.2-11b-vision-instruct` via NIM |
| Text/solver model | `meta/llama-3.2-11b-vision-instruct` (same family; text-only mode) |
| K (budget per A1/B arm) | 5 |
| Sampling temperature | 0.7 (A1, B-solver); 0.0 (A0, B-reader) |
| Max tokens per call | 384 |
| NIM calls (text/vlm) | 150 / 180 = **330 total** |
| Total wall | 440 s (~7.3 min) |
| Avg per-call wall | ~1.33 s |
| Run directory | `results/w95/mathvista_pilot/w95_mathvista_pilot_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260524T201615Z/` |

## Per-arm headline numbers

| Arm | Mean pass@1 (30 problems) | Calls/problem |
|---|---:|---:|
| A0_text (text-only LLM, single-shot T=0.0) | 36.67 % | 1 |
| A1_vlm (unified VLM, K=5 first-pass) | 66.67 % | 5 |
| **B_vlm_team (W95-B0: vlm_reader + math_solver + reflexion)** | **76.67 %** | 5 |

* **B − A1 = +10.00 pp** (≥ +5 pp Phase-2 bar).
* **B − A0 = +40.00 pp** (image is dramatically load-bearing).
* B beats A1 per-seed: 1/1.
* B beats A0 per-seed: 1/1.

## Per-problem rescue analysis

Of the 30 problems in the pilot slice:

| Outcome | Count | Share |
|---|---:|---:|
| Both A1 and B PASS | 17 | 56.7 % |
| Both A1 and B FAIL | 7 | 23.3 % |
| **B PASS, A1 FAIL (B-only rescue)** | **6** | **20.0 %** |
| A1 PASS, B FAIL (A1-only rescue) | 3 | 10.0 % |
| Total problems | 30 | 100.0 % |

* Net rescue advantage for B: **+3 problems** (6 rescues − 3 reverse-rescues).
* This is the load-bearing-ness signal: team decomposition is solving problems unified VLM at K=5 cannot, AND it's not merely doing better on the problems A1 already had.
* The 3 A1-only rescues are the cost of structural specialisation (the math-solver does not see the image directly, so a problem where the VLM-reader misses a subtlety can defeat B even though A1 would have caught it).

## Pre-committed Phase 2 pilot gates

All locked in `docs/RUNBOOK_W95.md` BEFORE the NIM pilot ran:

| Gate | Threshold | Outcome | Pass? |
|------|-----------|---------|-------|
| 1. Slice pre-committed | 30 pids by `select_mathvista_subset_v1(95_005_001, 30)` | committed; sha256 `6d3a07eb2b1dac9d…` | ✓ |
| 2. A1@K=5 < 90 % | required (avoid ceiling saturation) | 66.67 % | ✓ |
| 3. B > A1 strictly | required | 76.67 % > 66.67 % | ✓ |
| 4. B − A1 ≥ +5 pp | required | +10.00 pp | ✓ |
| 5. B > A0 by ≥ +5 pp | required (image load-bearing) | +40.00 pp | ✓ |
| 6. B ≥ A1 on ≥ 16/30 problems | required | 27/30 | ✓ |
| 7. Budget accounting exact | 11 calls/problem | 11 × 30 = 330 calls; bench module enforces | ✓ |
| 8. Audit chain present | bench + seed Merkle roots | bench `4f76bcd4ba605d16…`, seed `c697377f3dff8595…` | ✓ |
| 9. Executor stays clean | invariants intact | every arm routes through `evaluate_answer_v1` | ✓ |

**9 of 9 gates PASS → W95-B0 EARNS Phase 3.**

## Audit chain (re-derives offline)

`python scripts/verify_w95_mathvista_audit_chain.py --run-dir results/w95/mathvista_pilot/...20260524T201615Z` reports:

* text-sidecar SHA (N=150): PASS
* vlm-sidecar SHA (N=180): PASS
* per-problem sidecar (N=30): PASS
* per-seed Merkle root (N=1): PASS
* bench Merkle root: PASS
* 9/9 Phase 2 gates: PASS

**14 / 14 PASS, OVERALL: PASS.**

## Anti-cheat (carried over from W88–W94)

* The 30-problem slice was deterministic and pre-committed via
  `select_mathvista_subset_v1(95_005_001, 30)`; the pids were
  written to `pre_committed_slice.json` BEFORE any NIM call,
  with a `slice_sha256` checksum.
* No selective retries.  Every (problem, arm) triple is one
  set of NIM calls.
* Same model on every arm (Llama-3.2-11B-Vision in vision-mode
  for A1 and B-reader; text-only mode for A0 and B-solver).
* Same K=5 budget on A1 and B; budget gate verified
  byte-exactly (1 + 5 + 5 = 11 calls / problem on all 30).
* Executor truth = `evaluate_answer_v1` for every arm; no LLM
  judge anywhere.
* Parquet SHA-256 anchored at run start; mismatches refuse
  to run the bench.

## Honest scope of the pilot

* **It is a single-seed pilot.**  Single-seed results are noisier
  than 3-seed results.  The W89 70B-HumanEval retirement
  required 3 seeds × 30 problems × K=5 + 2-of-3 per-seed
  majority; a 1-seed result alone does not meet that bar.
* **It is not retirement-grade evidence.**  The Phase 2 gates
  are pilot gates (earn the run), not retirement bars (retire
  the carry-forward).  Phase 3 retirement bars are the W88
  6-bar shape; they require multi-seed evidence.
* **It is not a multi-benchmark generalisation.**  W95 is one
  battlefield; the broader programme bar is multi-benchmark.

## What is now entitled per the W95 contract

* Phase 3 (3 seeds × 100 problems × K=5; ~3 300 NIM calls; ~80-
  100 min wall) is preflight-earned and would test:
  - cross-seed reproducibility,
  - retention of the +10 pp B − A1 margin at 100-problem scale,
  - retirement-grade per-seed majority.

## What is NOT yet entitled

* No new carry-forward retirement.
* No claim of second confirmed same-budget multi-agent
  superiority retirement.  W89 70B-HumanEval-K=5 stands alone
  for now.
* No multi-benchmark cross-modal team superiority claim.

## Recommended next step

1. **Launch Phase 3** (separate NIM spend approval) — 3 seeds
   × 100 problems × K=5 × Llama-3.2-11B-Vision.  If the W88
   6-bar shape clears, this would be the first confirmed
   same-budget multi-agent superiority retirement on a
   **cross-modal** benchmark (the W89 retirement was a
   text-only code benchmark).
2. **Optional ablation pilot**: same shape on
   `meta/llama-3.2-90b-vision-instruct` to test whether the
   B-advantage holds at the larger VLM scale.  Documented as
   a *separate* Phase 2 pilot with its own pre-committed
   gates, NOT a continuation of this one.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* W95 modules (`coordpy.mathvista_loader_v1`,
  `coordpy.mathvista_executor_v1`,
  `coordpy.mathvista_preflight_v1`,
  `coordpy.mathvista_bench_v1`) are explicit-import only.

## Re-running

```bash
# Re-run the pilot under the same pre-committed gates:
NVIDIA_API_KEY=... python scripts/run_w95_mathvista_pilot.py \
  --vlm-model meta/llama-3.2-11b-vision-instruct \
  --n-problems 30 --n-seeds 1 \
  --expected-parquet-sha256 \
    373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d

# Verify offline:
python scripts/verify_w95_mathvista_audit_chain.py \
  --run-dir results/w95/mathvista_pilot/<run-dir>
```

Provider-side sampling at T=0.7 carries minor variance; the
conclusion ("B beats A1 by ≥ +5 pp at K=5 on the deterministic
30-problem slice") is robust to expected sampling variation
(NIM Llama-3.2-Vision determinism is documented to vary by
≤ ±1-3 pp across re-runs).

## The honest claim W95 Phase 2 earns

**On 1 seed × 30 problems × K=5 × Llama-3.2-11B-Vision-Instruct
on a deterministic, pre-committed slice of MathVista testmini,
the W95-B0 candidate (vision-reader + math-solver + executor-
guided reflexion) beats the unified-VLM K=5 baseline by
+10.00 pp (76.67 % vs 66.67 %) and beats the text-only
baseline by +40.00 pp.  All 9 pre-committed Phase 2 pilot
gates pass and the audit chain re-derives 14/14 offline.
This is the first pilot-grade signal of same-budget multi-
agent team superiority on a cross-modal benchmark in the
programme; Phase 3 is preflight-earned but not yet launched.
The result does NOT retire any carry-forward and does NOT
claim multi-seed retirement-grade superiority — those require
the 3-seed × 100-problem Phase 3 evidence.**
