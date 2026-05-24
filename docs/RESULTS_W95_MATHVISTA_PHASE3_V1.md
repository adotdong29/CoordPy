# W95 — MathVista Phase 3 Retirement-Grade Bench V1

> **2026-05-24 — PHASE 3 FALLS SHORT OF RETIREMENT BY 1.33 PP
> ON THE MARGIN BAR.  At 3 seeds × 100 problems × K=5 ×
> Llama-3.2-11B-Vision-Instruct on `AI4Math/MathVista` testmini:
> A0_text = 30.33 %, A1_vlm-K=5 = 67.67 %, **B_vlm_team =
> 71.33 %**.  Cross-seed margin **B − A1 = +3.67 pp** (below
> the pre-committed +5 pp retirement bar by 1.33 pp); B − A0 =
> +41.00 pp.  Per-seed: 2 / 3 seeds have B > A1, 3 / 3 have B
> > A0; per-seed B-A1 deltas are +6.00 / +10.00 / **−5.00** pp.
> B ≥ A1 on **267 / 300** problems = 89 % aggregate.  5 of the
> 6 pre-committed W88 retirement bars PASS; bar 4 (margin ≥
> +5 pp on B − A1) FAILS at +3.67 pp.  Audit chain re-derives
> 13 / 14 PASS offline (only bar-4 fails by margin; all five
> audit-chain re-derivations PASS).**
>
> **W95-B0 is EMPIRICALLY POSITIVE on MathVista (B beats A1 on
> the mean and on 2/3 seeds) but NOT retirement-grade under
> the W88 6-bar shape.**  This is exactly the pre-commit
> discipline working as designed: the single-seed +10.00 pp
> Phase 2 pilot signal narrowed to +3.67 pp under multi-seed
> evaluation.  Adds carry-forward
> `W95-L-MATHVISTA-RETIREMENT-MARGIN-CAP` documenting the
> narrow miss.

## Bench configuration

| Field | Value |
|---|---|
| Bench module | `coordpy.mathvista_bench_v1` |
| Phase | Phase 3 (full retirement bench) |
| Seeds (pre-committed) | 95_005_001 / 95_005_002 / 95_005_003 |
| Problems / seed | 100 (deterministic per-seed slice via `select_mathvista_subset_v1`) |
| Slice pid SHAs | `1569e2fa65e35cd6…` / `34d14d7496e98e1c…` / `c74e78b87709ad46…` |
| VLM model | `meta/llama-3.2-11b-vision-instruct` via NIM |
| Text/solver model | same (text-only mode) |
| K (budget per A1/B arm) | 5 |
| Sampling temperature | 0.7 (A1, B-solver); 0.0 (A0, B-reader) |
| Max tokens / call | 384 |
| NIM calls (text/vlm) | 1500 / 1800 = **3300 total** |
| Total wall | 5864 s (~97.7 min) |
| Run directory | `results/w95/mathvista_phase3/w95_mathvista_full_bench_..._20260524T204145Z/` |

## Per-arm headline numbers

| Arm | Cross-seed mean pass@1 | Calls/problem | Total calls |
|---|---:|---:|---:|
| A0_text (text-only LLM, single-shot T=0.0) | 30.33 % | 1 | 300 |
| A1_vlm (unified VLM, K=5 first-pass) | 67.67 % | 5 | 1500 |
| **B_vlm_team (W95-B0)** | **71.33 %** | 5 | 1500 |

* B − A1 mean = **+3.67 pp** (below +5 pp retirement-bar margin).
* B − A0 mean = **+41.00 pp**.

## Per-seed breakdown

| Seed | A0 | A1 | B | B − A1 | B − A0 |
|---|---:|---:|---:|---:|---:|
| 95_005_001 | 32.00 % | 67.00 % | 73.00 % | **+6.00 pp** | +41.00 pp |
| 95_005_002 | 31.00 % | 64.00 % | 74.00 % | **+10.00 pp** | +43.00 pp |
| 95_005_003 | 28.00 % | 72.00 % | 67.00 % | **−5.00 pp** | +39.00 pp |
| **Mean** | **30.33 %** | **67.67 %** | **71.33 %** | **+3.67 pp** | **+41.00 pp** |

The third seed reversed the signal (B fell to 67 % while A1 stayed at 72 %); this single seed's −5 pp delta dragged the cross-seed margin below the +5 pp bar.

## Per-problem rescue analysis (aggregate)

Of 300 total (seed, problem) pairs:

| Outcome | Count | Share |
|---|---:|---:|
| Both A1 and B PASS | 170 | 56.7 % |
| Both A1 and B FAIL | 53 | 17.7 % |
| **B PASS, A1 FAIL (B-only rescue)** | **44** | **14.7 %** |
| A1 PASS, B FAIL (A1-only rescue) | 33 | 11.0 % |
| Total | 300 | 100 % |

* Net rescue advantage for B: **+11 problems** across the 300.
* B ≥ A1 on **267 / 300 = 89.0 %** of (seed, problem) pairs.

## Pre-committed W95 Phase 3 retirement bars (W88 6-bar shape)

Locked in `docs/RUNBOOK_W95.md` BEFORE the NIM run:

| Bar | Threshold | Outcome | Pass? |
|------|-----------|---------|-------|
| 1. b_mean strictly beats a0_mean | required | 71.33 % > 30.33 % | ✓ |
| 2. b_mean strictly beats a1_mean | required | 71.33 % > 67.67 % | ✓ |
| 3. b_mean − a0_mean ≥ +5 pp | required | **+41.00 pp** | ✓ |
| 4. b_mean − a1_mean ≥ +5 pp | required | **+3.67 pp** | **✗** (misses by 1.33 pp) |
| 5. B > A0 on > half seeds | required | 3 / 3 | ✓ |
| 6. B > A1 on > half seeds | required | 2 / 3 | ✓ |
| 7. budget accounting exact | invariant | 1 + 5 + 5 = 11 calls/problem on every problem | ✓ |
| 8. audit chain present | invariant | bench Merkle `2257c4991e0d07c8…`; all 3 seed Merkle roots present | ✓ |
| 9. slices pre-committed per seed | invariant | all 3 slices recorded with sha256 BEFORE NIM | ✓ |

**5 of 6 retirement bars PASS; bar 4 FAILS at +3.67 pp.**  Under
the W88 6-bar shape, retirement requires ALL 6.

## Audit chain (re-derives offline)

`python scripts/verify_w95_mathvista_audit_chain.py --run-dir
results/w95/mathvista_phase3/w95_mathvista_full_bench_..._20260524T204145Z`
reports:

* text-sidecar SHA (N=1500): PASS
* vlm-sidecar SHA (N=1800): PASS
* per-problem sidecar (N=300): PASS
* per-seed Merkle root (N=3): PASS
* bench Merkle root: PASS
* W95 Phase 3 retirement bars: 8 / 9 PASS, 1 FAIL (bar 4 by margin)

**13 / 14 PASS, OVERALL: FAIL** (the bench is audit-clean; the
single FAIL is the retirement-margin bar, not an audit defect).

## What this means

### Empirical positive signal

* **W95-B0 beats A1 unified-VLM K=5 on the mean** (71.33 % vs
  67.67 %) by +3.67 pp at 3-seed × 100-problem scale on
  Llama-3.2-11B-Vision.
* **On 2 of 3 seeds**, B > A1 by ≥ +6 pp.
* **Image is dramatically load-bearing**: B − A0 = +41.00 pp at
  3-seed × 100-problem scale.
* **A1 unified VLM does not ceiling-saturate** at 67.67 % — there
  is genuine failure-residual that B exploits.
* B ≥ A1 on 89 % of (seed, problem) pairs.

### Pre-commit discipline negative

* The pre-committed +5 pp margin bar from `docs/RUNBOOK_W95.md`
  is empirically falsified at +3.67 pp.  Under the W88 6-bar
  shape, the W95-B0 candidate at this scale and model does NOT
  meet retirement-grade evidence.
* The third seed reversed (B − A1 = −5.00 pp), shrinking the
  cross-seed mean.

### Discipline validation

This is the W93 → W94 → W95 preflight-first discipline working
as designed:

* W95 Phase 2 (single seed × 30 problems) showed +10.00 pp
  margin.
* W95 Phase 3 (3 seeds × 100 problems) narrowed that to
  +3.67 pp — a **2.7× reduction** when moving from a single-seed
  cheap pilot to multi-seed retirement scale.
* The +5 pp pre-committed margin bar would have been TRIVIALLY
  cleared by Phase 2's single-seed +10 pp.  But the bar's
  purpose is exactly to gate against variance, and Phase 3
  showed the variance was real.

## Honest scope

* **W95-B0 IS positive on MathVista at multi-seed scale**, but
  *not retirement-grade* under the pre-committed W88 6-bar
  shape.
* **W95 Phase 3 does NOT retire any carry-forward.**  No new
  positive retirement claim is licensed.
* **W95-L-MATHVISTA-RETIREMENT-MARGIN-CAP** documents the
  narrow miss.
* **W89 70B-HumanEval-K=5 remains the only confirmed multi-
  seed same-budget multi-agent superiority retirement.**

## What might unlock retirement

The narrow miss suggests several plausible W96+ directions; each
would require its own pre-committed runbook + preflight:

1. **Larger VLM** (Llama-3.2-90B-Vision).  Bigger model may show
   sharper team-advantage; OR may saturate A1 closer to ceiling
   and shrink the residual.  Cheap pilot first.
2. **Larger sample** (5 seeds × 100, or 3 seeds × 300).
   Variance scaling: the third-seed −5 pp may be outlier.
3. **Architecture refinement**: stronger reflexion prompts;
   verifier turn; tool-augmented math solver (numerical
   evaluation).  Each would need its own preflight gate that
   shows the change is load-bearing on the residual.
4. **A different battlefield** if Phase 3 evidence indicates
   MathVista is structurally near-ceiling at +5 pp.

None of these is a Phase 4 of W95 by default; each is a fresh
candidate that must clear preflight.

## Anti-cheat (carried over from W88–W94)

* The 3 × 100 slices were deterministic and pre-committed via
  `select_mathvista_subset_v1(seed, 100, corpus)`.  All three
  pid lists were SHA-256-hashed and written to
  `pre_committed_slice.json` BEFORE any NIM call.
* Same model (Llama-3.2-11B-Vision-Instruct) on every arm
  (vision mode for A1 / B-reader; text-only mode for A0 /
  B-solver).
* Same K=5 budget on A1 and B; budget gate enforced
  byte-exactly (1 + 5 + 5 = 11 calls / problem on all 300).
* Executor truth = `evaluate_answer_v1` for every arm; no LLM
  judge anywhere.
* Parquet SHA-256 anchored at run start.
* No selective retries.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* All W95 modules are explicit-import only.

## Re-running

```bash
NVIDIA_API_KEY=... python scripts/run_w95_mathvista_pilot.py \
  --phase phase3 \
  --vlm-model meta/llama-3.2-11b-vision-instruct \
  --n-problems 100 --n-seeds 3 \
  --out-dir results/w95/mathvista_phase3 \
  --expected-parquet-sha256 \
    373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d
```

## The honest claim W95 Phase 3 earns

**On 3 seeds × 100 problems × K=5 × Llama-3.2-11B-Vision-
Instruct on a deterministic, pre-committed slice of MathVista
testmini, the W95-B0 candidate (vision-reader + math-solver +
executor-guided reflexion) beats the unified-VLM K=5 baseline
by +3.67 pp (71.33 % vs 67.67 %) on the cross-seed mean and on
2/3 seeds, and beats the text-only baseline by +41.00 pp; B ≥
A1 on 267/300 problems = 89 %.  Five of the six pre-committed
W88 retirement bars PASS; the +5 pp margin bar on B − A1
narrowly fails at +3.67 pp.  Under the W88 6-bar shape, this
is NOT a retirement; the W93/W94/W95 preflight-first discipline
correctly converted a single-seed +10 pp Phase 2 pilot into a
multi-seed +3.67 pp finding that the strict retirement
threshold does not license.  W95-B0 stands as a documented
positive directional signal that fails the strict retirement
threshold by 1.33 pp; W89 70B-HumanEval-K=5 remains the only
confirmed multi-seed same-budget multi-agent superiority
retirement.  Adds carry-forward
`W95-L-MATHVISTA-RETIREMENT-MARGIN-CAP`.**
