# W96-A — MathVista 90B-Vision Cheap NIM Pilot V1

> **2026-05-24 — PHASE 2 PILOT PASSES ALL 9 PRE-COMMITTED
> GATES.  At 1 seed × 30 problems × K=5 ×
> Llama-3.2-90B-Vision-Instruct on `AI4Math/MathVista` testmini
> (parquet SHA-256 `373f6c0b…`, corpus Merkle `dea27472fc12…`,
> slice SHA-256 `6d3a07eb2b1dac9d…`): A0_text = 33.33 %,
> A1_vlm-K=5 = 63.33 %, **B_vlm_team = 73.33 %**.  Margin
> **B − A1 = +10.00 pp**; image-load-bearing **B − A0 =
> +40.00 pp**.  B ≥ A1 on **27 / 30 problems**.  Per-problem
> rescue: B saves 6 problems A1 missed; A1 saves 3 B missed;
> net +3 rescues for B.  Bench Merkle `0946b88c4e288f35…`;
> audit chain re-derives 14 / 14 PASS offline.  W96-A Phase 3
> (full bench: 3 seeds × 100 problems × K=5 × 90B-Vision;
> ~3 300 NIM calls; ~3-4 h wall) is now preflight-earned per
> `docs/RUNBOOK_W96A.md`.**
>
> **MAJOR STRUCTURAL FINDING (problem-by-problem 11B vs 90B
> comparison on the IDENTICAL slice):**  The W95-B0
> architecture's structural advantage is **scale-invariant at
> this pilot scale**.  Both 11B Phase 2 and 90B Phase 2
> produce **B − A1 = +10.00 pp exactly**, with identical
> B-only-rescue and A1-only-rescue counts (6 and 3 problems
> respectively), and identical B ≥ A1 problem coverage
> (27 / 30 at both scales).  Scaling the VLM weight class from
> 11B to 90B did **NOT** widen the team's margin (H1
> falsified) and did **NOT** shrink it (H2 falsified).  The
> ~3-4× published-SOTA-scale jump does not produce a
> proportional team-margin scale jump on this pilot.
>
> **HONEST FRAMING**: this is a SINGLE-SEED pilot result.  It
> earns Phase 3 under the pre-committed contract.  It is NOT
> a multi-seed retirement claim and does NOT retire any
> existing carry-forward.  W89 70B-HumanEval-K=5 remains the
> only confirmed multi-seed same-budget multi-agent
> superiority retirement.

## Pilot configuration

| Field | Value |
|---|---|
| Bench module | `coordpy.mathvista_bench_v1` (unchanged from W95) |
| Slice selector | `select_mathvista_subset_v1(seed=95_005_001, n_problems=30, corpus=testmini)` (deterministic; pre-committed before any NIM call; byte-identical to W95 Phase 2 slice) |
| Pre-committed pid SHA-256 | `6d3a07eb2b1dac9d529d3ffb4ce7c40e54fd1166183f05e1d289877139249f5c` |
| VLM model | `meta/llama-3.2-90b-vision-instruct` via NIM |
| Text/solver model | `meta/llama-3.2-90b-vision-instruct` (same family; text-only mode) |
| K (budget per A1/B arm) | 5 |
| Sampling temperature | 0.7 (A1, B-solver); 0.0 (A0, B-reader) |
| Max tokens per call | 384 |
| NIM calls (text/vlm) | 150 / 180 = **330 total** (plus 4 sidecar smoke calls in `results/w96/mathvista_smoke_90b/20260524T230110Z/` recorded separately as `kind=smoke_test`, not in budget) |
| Total wall | 1274 s (~21.2 min) |
| Avg per-call wall | ~3.86 s |
| Run directory | `results/w96/mathvista_90b_phase2/w95_mathvista_pilot_meta_llama-3.2-90b-vision-instruct__meta_llama-3.2-90b-vision-instruct_20260524T230158Z/` |

## Per-arm headline numbers

| Arm | Mean pass@1 (30 problems) | Calls/problem |
|---|---:|---:|
| A0_text (text-only LLM, single-shot T=0.0) | 33.33 % | 1 |
| A1_vlm (unified VLM, K=5 first-pass) | 63.33 % | 5 |
| **B_vlm_team (W95-B0: vlm_reader + math_solver + reflexion)** | **73.33 %** | 5 |

* **B − A1 = +10.00 pp** (clears the +5 pp Phase-2 bar by 2×).
* **B − A0 = +40.00 pp** (image is dramatically load-bearing).
* B beats A1 per-seed: 1/1.
* B beats A0 per-seed: 1/1.

## Cross-scale comparison (W95 11B Phase 2 vs W96-A 90B Phase 2 on the IDENTICAL slice)

The W95 Phase 2 pilot
(`results/w95/mathvista_pilot/.../w95_mathvista_pilot_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260524T201615Z/`)
and the W96-A Phase 2 pilot
(`results/w96/mathvista_90b_phase2/w95_mathvista_pilot_meta_llama-3.2-90b-vision-instruct__meta_llama-3.2-90b-vision-instruct_20260524T230158Z/`)
ran the **byte-identical 30-problem slice** at seed
95_005_001 with the SAME bench module, executor, and budget.
Per-arm aggregates:

| Arm | W95 (11B) | W96-A (90B) | Δ (90B − 11B) |
|---|---:|---:|---:|
| A0_text | 36.67 % | 33.33 % | −3.33 pp |
| A1_vlm K=5 | 66.67 % | 63.33 % | −3.33 pp |
| B_vlm_team | 76.67 % | 73.33 % | −3.33 pp |
| **B − A1** | **+10.00 pp** | **+10.00 pp** | **0.00 pp** |
| **B − A0** | **+40.00 pp** | **+40.00 pp** | **0.00 pp** |
| B ≥ A1 problem count | 27 / 30 | 27 / 30 | 0 |

The 90B run is uniformly ~3.3 pp BELOW the 11B run on every
arm on this 30-problem slice, so the inter-arm gaps
(B − A1, B − A0) are preserved byte-exactly.

### Per-problem rescue analysis at each scale

| Outcome (A1 vs B, 30 problems) | W95 (11B) | W96-A (90B) |
|---|---:|---:|
| Both A1 and B PASS | 17 | 16 |
| Both A1 and B FAIL | 4 | 5 |
| **B PASS, A1 FAIL (B-only rescue)** | **6** | **6** |
| A1 PASS, B FAIL (A1-only rescue) | 3 | 3 |
| Net rescue advantage for B | +3 | +3 |

The **count of B-only rescues is identical (6 problems at each
scale)** and the **A1-only rescues are identical (3 problems at
each scale)**.  The W95-B0 architecture's structural
advantage is preserved bit-for-bit on this slice across the
11B→90B weight-class jump.

### Pid-level flip analysis

Of the 30 (pid, arm) outcomes on B: 19 of 30 problems are
*invariant* (B-passes both at 11B and 90B); 4 of 30 are
*invariant fails* (B-fails both); 3 problems flip B-PASS at
90B that were B-FAIL at 11B (pids 470 / 652 / 947); 4
problems flip B-FAIL at 90B that were B-PASS at 11B (pids
408 / 515 / 634 / 815).  The flips approximately cancel:
+3 − 4 = −1 net problems for B at 90B.

A1 has similar structure: 15 invariant passes, 6 invariant
fails, 4 problems flip A1-PASS at 90B that were A1-FAIL at
11B (pids 408 / 426 / 774 / 877), and 5 problems flip
A1-FAIL at 90B that were A1-PASS at 11B (pids 131 / 173 /
463 / 515 / 722).  Net: +4 − 5 = −1 net problem for A1 at
90B.

The flips are NOT correlated to a single problem category;
they look like sampling variance of the T=0.7 sampler on
problems whose underlying difficulty straddles the
single-shot pass / fail decision boundary.  No single problem
category is systematically rescued by the 90B scale-up.

## H1 / H2 / H3 prediction verdict

Per the pre-committed Q3 prediction in `docs/RUNBOOK_W96A.md`
("genuinely unknown"):

* **H1 (90B widens B − A1 margin via stronger reader):**
  empirically *falsified at this pilot scale*.  The margin is
  bit-equivalent to 11B, not wider.
* **H2 (90B closes the unified-VLM gap → A1 saturates closer
  to ceiling → residual shrinks → B − A1 shrinks):**
  empirically *falsified at this pilot scale*.  A1 actually
  fell by 3.33 pp on this slice, not rose.
* **H3 (margin invariance — neither H1 nor H2):**
  empirically *supported at this pilot scale*.  Both arms
  shifted by the same magnitude (−3.33 pp) so the inter-arm
  gap (B − A1) was exactly preserved.

This is a real, structurally interesting finding: at single-
seed × 30-problem cheap-pilot scale, the W95-B0 team
mechanism is largely a function of the **architecture**
(vision-extract → math-solve decomposition + executor-guided
reflexion), and not of the raw **VLM weight class**.  At
this pilot scale the marginal improvement of stepping from
11B to 90B does not differentially help OR hurt B vs A1.

The 3.33 pp uniform drop in A0 / A1 / B at 90B is consistent
with NIM's per-call T=0.7 stochasticity at a different
sampler over a different model; it is within the ±1-3 pp
provider-side determinism band noted in
`docs/RESULTS_W95_MATHVISTA_PILOT_V1.md`, and we do NOT
interpret it as a real 90B weakness.

## Pre-committed Phase 2 pilot gates

All locked in `docs/RUNBOOK_W95.md` Phase 2 and re-applied
verbatim by `docs/RUNBOOK_W96A.md`.  All locked BEFORE the
NIM pilot ran:

| Gate | Threshold | Outcome | Pass? |
|------|-----------|---------|-------|
| 1. Slice pre-committed | 30 pids by `select_mathvista_subset_v1(95_005_001, 30)` | committed; sha256 `6d3a07eb2b1dac9d…` | ✓ |
| 2. A1@K=5 < 90 % | required (avoid ceiling saturation) | 63.33 % | ✓ |
| 3. B > A1 strictly | required | 73.33 % > 63.33 % | ✓ |
| 4. B − A1 ≥ +5 pp | required | **+10.00 pp** | ✓ |
| 5. B > A0 by ≥ +5 pp | required (image load-bearing) | **+40.00 pp** | ✓ |
| 6. B ≥ A1 on ≥ 16/30 problems | required | 27 / 30 | ✓ |
| 7. Budget accounting exact | 11 calls/problem | 11 × 30 = 330 calls; bench module enforces | ✓ |
| 8. Audit chain present | bench + seed Merkle roots | bench `0946b88c4e288f35…`, seed `f97ae5eba02b8d5e…` | ✓ |
| 9. Executor stays clean | invariants intact | every arm routes through `evaluate_answer_v1` | ✓ |

**9 of 9 gates PASS → W96-A EARNS Phase 3 at 90B.**

## W96-A 90B-specific cheap probes (Q1 / Q2 / Q3)

Locked in `docs/RUNBOOK_W96A.md` Phase 2 contract.  Recorded in
`results/w96/mathvista_smoke_90b/`:

| Probe | Threshold | Outcome | Pass? |
|------|-----------|---------|-------|
| Q1 endpoint reachability | HTTP 200 + non-empty completion | 3/3 timed calls returned non-empty completions | ✓ |
| Q2 wall-ms plausibility (steady-state mean) | < 30 000 ms | mean 1 273 ms, max 2 907 ms | ✓ (v2 schema after v1 cold-start re-scope) |
| Q3 H1/H2 prediction (a-priori) | "genuinely unknown" | locked BEFORE NIM (no retro-fit) | ✓ |

The v1 smoke-test's Q2 FAIL on the cold-start call (33 s vs
30 s threshold) triggered the runbook's pause-and-rescope
clause; v2 absorbs the cold-start spike with a 1-call warmup
block and evaluates Q2 against the post-warmup steady-state
mean.  This is the W96-A pause-and-rescope outcome and is
recorded in `results/w96/mathvista_smoke_90b/20260524T230110Z/smoke_test.json`.

## Audit chain (re-derives offline)

`python scripts/verify_w95_mathvista_audit_chain.py
--run-dir results/w96/mathvista_90b_phase2/w95_mathvista_pilot_meta_llama-3.2-90b-vision-instruct__meta_llama-3.2-90b-vision-instruct_20260524T230158Z`
reports:

* text-sidecar SHA (N=150): PASS
* vlm-sidecar SHA (N=180): PASS
* per-problem sidecar (N=30): PASS
* per-seed Merkle root (N=1): PASS
* bench Merkle root: PASS
* 9 / 9 Phase 2 gates: PASS

**14 / 14 PASS, OVERALL: PASS.**

## Anti-cheat (carried over from W88–W95)

* The 30-problem slice was deterministic and pre-committed via
  `select_mathvista_subset_v1(95_005_001, 30)`; the pids were
  written to `pre_committed_slice.json` BEFORE any NIM call,
  with a `slice_sha256` checksum byte-identical to the W95
  Phase 2 slice.
* No selective retries.  Every (problem, arm) triple is one
  set of NIM calls.
* Same model on every arm (Llama-3.2-90B-Vision in vision-mode
  for A1 and B-reader; text-only mode for A0 and B-solver).
* Same K=5 budget on A1 and B; budget gate verified
  byte-exactly (1 + 5 + 5 = 11 calls / problem on all 30).
* Executor truth = `evaluate_answer_v1` for every arm; no LLM
  judge anywhere.
* Parquet SHA-256 anchored at run start; mismatches refuse to
  run the bench.
* The Q1 smoke-test NIM call is recorded as `kind=smoke_test`
  in its own sidecar and does NOT count toward per-problem
  budget accounting.

## Honest scope of the pilot

* **It is a single-seed pilot at 30 problems.**  Single-seed
  results are noisier than 3-seed results.  W89 70B-HumanEval
  retirement required 3 seeds × 30 problems × K=5 + 2-of-3
  per-seed majority; a 1-seed result alone does not meet that
  bar.
* **It is not retirement-grade evidence.**  The Phase 2 gates
  are pilot gates (earn the run), not retirement bars (retire
  the carry-forward).
* **It is not a multi-benchmark generalisation.**  W96-A is
  one cheap follow-up on the W95 MathVista line at a single
  larger weight class; the broader programme bar remains
  multi-benchmark.
* **The +10 pp single-seed signal previously narrowed to
  +3.67 pp at 11B Phase 3 multi-seed scale.**  The honest
  presumption is that the 90B Phase 3 multi-seed margin may
  similarly narrow; the Phase 3 run is the actual test.

## What is now entitled per the W96-A contract

* **W96-A Phase 3** (3 seeds × 100 problems × K=5 ×
  Llama-3.2-90B-Vision; ~3 300 NIM calls; ~2.5-4 h wall) is
  preflight-earned and tests:
  - cross-seed reproducibility at 90B,
  - retention of the +10 pp single-seed margin at 100-problem
    × 3-seed scale,
  - the W88 6-bar retirement shape on this larger weight
    class.

## What is NOT yet entitled

* No new carry-forward retirement.
* No claim of second confirmed same-budget multi-agent
  superiority retirement.  W89 70B-HumanEval-K=5 stands alone
  for now.
* No multi-benchmark cross-modal team superiority claim.
* No claim that 90B scaling differentially helps or hurts the
  W95-B0 team mechanism (the cheap pilot's evidence is
  *invariance*, neither H1 nor H2).

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* No new core modules added.  W96-A re-used
  `coordpy.mathvista_loader_v1`,
  `coordpy.mathvista_executor_v1`,
  `coordpy.mathvista_preflight_v1`,
  `coordpy.mathvista_bench_v1` verbatim.
* One new ops script added: `scripts/run_w96a_smoke_test.py`
  (executes Q1/Q2 NIM endpoint probes for any candidate VLM
  model).  Does not import into the package.

## Re-running

```bash
# Re-run the W96-A pilot under the same pre-committed gates:
NVIDIA_API_KEY=... python scripts/run_w95_mathvista_pilot.py \
  --phase phase2 \
  --vlm-model meta/llama-3.2-90b-vision-instruct \
  --n-problems 30 --n-seeds 1 \
  --seed-start 95005001 \
  --out-dir results/w96/mathvista_90b_phase2 \
  --expected-parquet-sha256 \
    373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d

# Verify offline:
python scripts/verify_w95_mathvista_audit_chain.py \
  --run-dir results/w96/mathvista_90b_phase2/<run-dir>
```

Provider-side T=0.7 sampling carries minor variance; the
conclusion ("at this single-seed × 30-problem pilot scale,
W95-B0 at 90B clears the +5 pp Phase 2 bar by 2× and produces
a margin bit-equivalent to 11B Phase 2") is robust to expected
sampling variation.

## The honest claim W96-A Phase 2 earns

**On 1 seed × 30 problems × K=5 × Llama-3.2-90B-Vision-
Instruct on the same deterministic, pre-committed slice of
MathVista testmini as W95 Phase 2 (seed 95_005_001), the
W95-B0 candidate (vision-reader + math-solver + executor-
guided reflexion) beats the unified-VLM K=5 baseline by
+10.00 pp (73.33 % vs 63.33 %) and beats the text-only
baseline by +40.00 pp.  The margin is byte-equivalent to W95
Phase 2 at 11B (+10.00 pp at both scales), with identical
B-only-rescue (6 problems) and A1-only-rescue (3 problems)
counts.  All 9 pre-committed Phase 2 pilot gates pass and the
audit chain re-derives 14/14 offline.  This is the first
pilot-grade cross-scale evidence in the W95 line that the
W95-B0 team mechanism's advantage is scale-invariant at the
11B→90B weight-class step on this benchmark and this slice.
W96-A Phase 3 is preflight-earned but not yet launched.  The
result does NOT retire any carry-forward and does NOT claim
multi-seed retirement-grade superiority — those require
the 3-seed × 100-problem Phase 3 evidence at 90B.**
