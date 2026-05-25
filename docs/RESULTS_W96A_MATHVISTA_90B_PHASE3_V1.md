# W96-A — MathVista 90B-Vision Phase 3 Retirement-Grade Bench V1

> **2026-05-24 — PHASE 3 AT 90B DECISIVELY FAILS RETIREMENT
> WITH A NEGATIVE TEAM MARGIN.  At 3 seeds × 100 problems ×
> K=5 × Llama-3.2-90B-Vision-Instruct on `AI4Math/MathVista`
> testmini (the SAME 3 deterministic slices as W95 Phase 3 at
> 11B, byte-identical): A0_text = 28.00 %, A1_vlm-K=5 =
> 71.33 %, **B_vlm_team = 66.33 %**.  Cross-seed margin
> **B − A1 = −5.00 pp** (B LOSES to A1 by 5 pp at retirement
> scale).  Per-seed B − A1 = +4.00 / −7.00 / −12.00 pp; B
> beats A1 on **1 of 3** seeds.  B − A0 = +38.33 pp (image is
> still load-bearing).  **3 of 6 pre-committed W88 retirement
> bars FAIL** (bars 2, 4, 6); bars 1, 3, 5, 7, 8, 9 PASS.
> Audit chain re-derives 11 / 14 PASS offline (only the 3
> retirement-margin bars fail; all 5 audit-chain re-derivations
> PASS).**
>
> **DECISIVE EMPIRICAL FINDING:** scaling the VLM weight class
> from Llama-3.2-11B-Vision to Llama-3.2-90B-Vision **HURTS**
> the W95-B0 team's relative advantage on MathVista at K=5 by
> a cross-scale shift of **−8.67 pp** on B − A1 (from +3.67 pp
> at 11B to −5.00 pp at 90B).  The H2-saturation hypothesis
> (90B closes the unified-VLM gap → A1 saturates closer to
> ceiling → residual shrinks → B − A1 collapses or reverses)
> is empirically supported at multi-seed retirement scale.
> Adds carry-forward
> `W96-L-MATHVISTA-90B-RETIREMENT-MARGIN-CAP` documenting that
> *scaling the VLM does NOT retire the cross-modal carry-
> forward*.  The W95-B0 architecture has an empirical ceiling
> on MathVista; the next research lever must be architectural
> (W96-C verifier turn / tool-augmented solver per `COO-19`),
> not weight-scale.

## Bench configuration

| Field | Value |
|---|---|
| Bench module | `coordpy.mathvista_bench_v1` |
| Phase | Phase 3 (full retirement bench) |
| Seeds (pre-committed) | 95_005_001 / 95_005_002 / 95_005_003 — **byte-identical to W95 Phase 3 at 11B** |
| Problems / seed | 100 (deterministic per-seed slice via `select_mathvista_subset_v1`) |
| Slice pid SHAs | `1569e2fa65e35cd6…` / `34d14d7496e98e1c…` / `c74e78b87709ad46…` (**byte-equal to W95 Phase 3 slices**) |
| VLM model | `meta/llama-3.2-90b-vision-instruct` via NIM |
| Text/solver model | same (text-only mode) |
| K (budget per A1/B arm) | 5 |
| Sampling temperature | 0.7 (A1, B-solver); 0.0 (A0, B-reader) |
| Max tokens / call | 384 |
| NIM calls (text/vlm) | 1500 / 1800 = **3300 total** |
| Total wall | 11 794 s (~196.6 min ≈ 3.28 h) |
| Run directory | `results/w96/mathvista_90b_phase3/w95_mathvista_full_bench_..._20260524T232931Z/` |
| Bench Merkle root | `899c213a2755b26c6caae0e0d88c1922770d5a552dbbc98b2ba27e62a9bc2c52` |
| Per-seed Merkle roots | `d224410c…` / `72edf36a…` / `d0a6f330…` |

## Per-arm headline numbers

| Arm | Cross-seed mean pass@1 | Calls/problem | Total calls |
|---|---:|---:|---:|
| A0_text (text-only LLM, single-shot T=0.0) | 28.00 % | 1 | 300 |
| A1_vlm (unified VLM, K=5 first-pass) | 71.33 % | 5 | 1500 |
| **B_vlm_team (W95-B0)** | **66.33 %** | 5 | 1500 |

* B − A1 mean = **−5.00 pp** (clearly below +5 pp bar; B *loses*).
* B − A0 mean = **+38.33 pp** (image still dramatically load-bearing).

## Per-seed breakdown

| Seed | A0 | A1 | B | B − A1 | B − A0 |
|---|---:|---:|---:|---:|---:|
| 95_005_001 | 33.00 % | 70.00 % | 74.00 % | **+4.00 pp** | +41.00 pp |
| 95_005_002 | 33.00 % | 70.00 % | 63.00 % | **−7.00 pp** | +30.00 pp |
| 95_005_003 | 18.00 % | 74.00 % | 62.00 % | **−12.00 pp** | +44.00 pp |
| **Mean** | **28.00 %** | **71.33 %** | **66.33 %** | **−5.00 pp** | **+38.33 pp** |

Only seed 95_005_001 (the same seed identity as the Phase 2 +10 pp pilot) gives B a positive margin at multi-seed scale; the other two seeds reverse.

## Per-problem rescue analysis (aggregate)

Of 300 total (seed, problem) pairs:

| Outcome | Count | Share |
|---|---:|---:|
| Both A1 and B PASS | 169 | 56.3 % |
| Both A1 and B FAIL | 56 | 18.7 % |
| **B PASS, A1 FAIL (B-only rescue)** | **30** | **10.0 %** |
| **A1 PASS, B FAIL (A1-only rescue)** | **45** | **15.0 %** |
| Total | 300 | 100 % |

* Net rescue advantage for A1: **+15 problems** across the 300.
* B ≥ A1 on **255 / 300 = 85.0 %** of (seed, problem) pairs.

This is **opposite-signed** to W95 11B Phase 3, where B had +11 net rescues; at 90B, A1 now has +15 net rescues.

## Cross-scale comparison: W95 11B vs W96-A 90B at Phase 3 retirement scale

The W95 Phase 3 bench
(`results/w95/mathvista_phase3/w95_mathvista_full_bench_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260524T204145Z/`)
and the W96-A Phase 3 bench
(`results/w96/mathvista_90b_phase3/w95_mathvista_full_bench_meta_llama-3.2-90b-vision-instruct__meta_llama-3.2-90b-vision-instruct_20260524T232931Z/`)
ran the **byte-identical 3 × 100-problem slice set** (seeds
95_005_001 / 95_005_002 / 95_005_003) with the SAME bench
module, executor, and budget.

### Cross-seed aggregates

| Arm | 11B (W95) | 90B (W96-A) | Δ (90B − 11B) |
|---|---:|---:|---:|
| A0_text | 30.33 % | 28.00 % | −2.33 pp |
| A1_vlm K=5 | 67.67 % | **71.33 %** | **+3.67 pp** |
| B_vlm_team | 71.33 % | 66.33 % | **−5.00 pp** |
| **B − A1** | **+3.67 pp** | **−5.00 pp** | **−8.67 pp** |
| B − A0 | +41.00 pp | +38.33 pp | −2.67 pp |
| B-only rescues (total) | 44 | 30 | −14 |
| A1-only rescues (total) | 33 | 45 | +12 |
| Net B-A1 rescue advantage | +11 | −15 | **−26** |
| B ≥ A1 problem count | 267 / 300 (89.0 %) | 255 / 300 (85.0 %) | −12 |

### Per-seed B − A1 (problem-level fair comparison)

| Seed | 11B B−A1 | 90B B−A1 | Δ |
|---|---:|---:|---:|
| 95_005_001 | +6.00 pp | +4.00 pp | −2.00 pp |
| 95_005_002 | +10.00 pp | −7.00 pp | **−17.00 pp** |
| 95_005_003 | −5.00 pp | −12.00 pp | −7.00 pp |

The 11B → 90B cross-scale shift on B − A1 is **uniformly
negative** across all 3 seeds and is dramatic on seed 2 (a
−17 pp swing); seed 3 was already the 11B outlier (−5 pp)
and gets even more negative at 90B (−12 pp).

### Pid-level interpretation

A1 at 90B gains 3.67 pp (67.67 % → 71.33 %) — exactly the
H2 prediction direction.  B at 90B **loses** 5.00 pp on the
same problems.  These shifts are not symmetric: B is hurt
*more* than A1 is helped.  The likely mechanism: at 90B, the
unified-VLM forward absorbs enough of the "vision + math"
joint signal in a single auto-regressive pass that the
explicit `vlm_reader → math_solver` decomposition's
intermediate text representation becomes a LIABILITY (the
solver's text-only forward can't see the image directly, so
problems where 90B's monolithic forward would have succeeded
under A1 now FAIL under B because the vlm_reader stage drops
some salient detail that the unified forward would have
preserved).  This is the *opposite* of the structural
advantage the team mechanism enjoyed at 11B, where the
unified forward was weaker and the explicit decomposition
recovered failures it could not.

## H1 / H2 / H3 verdict (post-Phase 3)

Per the W96-A pre-committed Q3 prediction in `docs/RUNBOOK_W96A.md`
("genuinely unknown") and the Phase 3 pre-commit prediction
("H3-A marginally more likely than H3-B; H3-A is 90B replicates
the W95 narrowing"):

* **H1 (90B widens B − A1 margin):** decisively
  *falsified at multi-seed retirement scale*.
* **H3-A (90B replicates the W95 narrowing to ~+3-4 pp):**
  *under-counted the magnitude of narrowing*.  90B narrows
  WORSE than 11B did, going from +3.67 pp to −5.00 pp (a 90B
  worse-than-11B shift of −8.67 pp on the cross-seed mean).
* **H3-B (90B preserves the margin and retires):** decisively
  *falsified*.
* **H2 (90B saturates A1 closer to ceiling → residual shrinks →
  B's edge collapses or reverses):** *empirically supported
  at multi-seed retirement scale*.  This is the dominant
  reading of the W96-A Phase 3 evidence.

The **architectural-invariance reading from Phase 2 (both 11B
and 90B give +10 pp single-seed × 30) was a 30-problem
sampling artefact**: at the same retirement scale (3 × 100),
the 11B and 90B benches diverge by 8.67 pp on B − A1.  The
W96-A Phase 2 to Phase 3 narrowing is **WORSE** than the W95
Phase 2 to Phase 3 narrowing:

* W95 11B: pilot +10 pp → retirement +3.67 pp (−6.33 pp
  narrowing).
* W96-A 90B: pilot +10 pp → retirement **−5.00 pp** (−15.00 pp
  narrowing — over 2× the 11B narrowing).

The single-seed pilot at 90B was *more* misleading than the
single-seed pilot at 11B because the 30-problem slice
happened to over-weight problems where B's structural edge
survived 90B's stronger A1.  At 3-seed × 100 problems, the
edge is gone.

## Pre-committed W96-A Phase 3 retirement bars (W88 6-bar shape)

Locked in `docs/RUNBOOK_W96A.md` Phase 3 section BEFORE the
NIM run (which itself was locked in `docs/RUNBOOK_W95.md`
Phase 3 verbatim):

| Bar | Threshold | Outcome | Pass? |
|------|-----------|---------|-------|
| 1. b_mean strictly beats a0_mean | required | 66.33 % > 28.00 % | ✓ |
| 2. b_mean strictly beats a1_mean | required | 66.33 % > 71.33 % ? **NO** | **✗** |
| 3. b_mean − a0_mean ≥ +5 pp | required | **+38.33 pp** | ✓ |
| 4. b_mean − a1_mean ≥ +5 pp | required | **−5.00 pp** | **✗** (misses by 10.00 pp) |
| 5. B > A0 on > half seeds | required | 3 / 3 | ✓ |
| 6. B > A1 on > half seeds | required | 1 / 3 | **✗** |
| 7. budget accounting exact | invariant | 1 + 5 + 5 = 11 calls/problem on all 300 | ✓ |
| 8. audit chain present | invariant | bench Merkle `899c213a2755b26c…`; all 3 seed Merkle roots present | ✓ |
| 9. slices pre-committed per seed | invariant | all 3 slices SHA-anchored BEFORE NIM | ✓ |

**3 of 6 retirement bars FAIL; 6 of 9 audit-extended bars PASS.**

Retirement requires ALL of bars 1..6 to PASS.  Bars 2, 4, 6
fail decisively.  No retirement claim is licensed.

## Audit chain (re-derives offline)

`python scripts/verify_w95_mathvista_audit_chain.py
--run-dir results/w96/mathvista_90b_phase3/w95_mathvista_full_bench_meta_llama-3.2-90b-vision-instruct__meta_llama-3.2-90b-vision-instruct_20260524T232931Z`
reports:

* text-sidecar SHA (N=1500): PASS
* vlm-sidecar SHA (N=1800): PASS
* per-problem sidecar (N=300): PASS
* per-seed Merkle root (N=3): PASS
* bench Merkle root: PASS
* W96-A Phase 3 retirement bars: 6 / 9 PASS, 3 FAIL (bars 2,
  4, 6 by margin/sign)

**11 / 14 PASS, OVERALL: FAIL** (the bench is audit-clean;
the 3 FAILs are retirement-bar misses, not audit defects).

## What this means

### Decisive empirical negative

* **W95-B0 at 90B LOSES to A1 unified-VLM K=5 on the mean**
  (66.33 % vs 71.33 %) by −5.00 pp at 3-seed × 100-problem
  scale on Llama-3.2-90B-Vision.
* B beats A1 on only 1 of 3 seeds.
* The 11B → 90B cross-scale shift on B − A1 is uniformly
  negative across all 3 seeds.

### What the W96-A line empirically rules out

* **Scaling the VLM weight class is NOT the answer.**  Going
  from 11B-Vision to 90B-Vision at the same K=5 with the same
  W95-B0 architecture on the same deterministic 3 × 100
  slice makes B's relative position *worse*, not better.
* The earlier Phase 2 architectural-invariance reading was a
  pilot-scale sampling artefact.  At retirement scale, the
  invariance disintegrates: 11B and 90B diverge by 8.67 pp on
  B − A1.

### Why scaling backfires here (mechanism inference)

The W95-B0 team's advantage at 11B was structurally tied to
the unified VLM's failure-residual: A1 at 11B got 67.67 % at
K=5, leaving ~32 pp residual that the explicit `vlm_reader
→ math_solver` decomposition could exploit.  At 90B, A1
climbs to 71.33 % at K=5 — only a 3.67 pp gain in absolute
terms, but it cuts into the *easiest* part of the residual
(problems where the unified-VLM had a near-miss that K=5
sampling could clear).  The remaining residual at 90B is
dominated by problems where:

1. The vision step itself is too hard (no number of math-
   solver retries can recover).
2. The math step's representation needs the *image* present,
   not just a text extraction (the W95-B0 architecture's
   structural weakness — the math_solver does NOT see the
   image).

At 90B, the second class of problems grows in relative size
(A1's stronger monolithic forward now succeeds on problems
where the explicit decomposition's text extraction is
lossy).  Hence the A1-only rescue count *rises* from 33 to
45 (+12 problems) and the B-only rescue count *falls* from
44 to 30 (−14 problems).

### Discipline validation

This is the W93 → W94 → W95 → W96 preflight-first discipline
working as designed:

* W95 Phase 3 narrowed +10 pp pilot → +3.67 pp retirement.
* W96-A Phase 3 narrowed +10 pp pilot → **−5.00 pp**
  retirement.
* The +5 pp pre-committed retirement bar correctly
  distinguished a single-seed pilot signal from multi-seed
  retirement evidence — *twice on the same line*.

### What is NOT yet entitled (vs what some readers might wish)

* No new carry-forward retirement.  In particular, the W95
  carry-forward `W95-L-MATHVISTA-RETIREMENT-MARGIN-CAP` is
  NOT just preserved — it is now **augmented** by
  `W96-L-MATHVISTA-90B-RETIREMENT-MARGIN-CAP`, which adds
  the specific finding that scaling the VLM weight class does
  not close the margin.
* W89 70B-HumanEval-K=5 remains the only confirmed
  multi-seed same-budget multi-agent superiority retirement.

## What might still unlock retirement (W96 next move)

The cross-scale evidence empirically rules out the
"scale up the VLM" lever.  Two architectural levers remain
viable per the Linear-recommended ordering:

1. **W96-C (`COO-19`) — architecture refinement.**  Replace
   the math_solver's text-only forward with either:
   * a *VLM-verifier* turn that re-reads the image with the
     candidate answer in context and accepts/rejects (the C1
     variant), giving the team a second chance to see the
     image directly;
   * a *tool-augmented solver* that can perform numeric
     arithmetic or symbolic algebra explicitly (the C2
     variant), addressing the W95-B0 weakness where the
     math step's reasoning would benefit from a Python
     evaluator.

   Both require their own pre-committed runbook + preflight +
   cheap pilot per the W93 discipline.  Each addresses the
   *specific* mechanism that W96-A Phase 3 empirically
   identified as the cap.

2. **W96-B (`COO-18`) — wider-sample retirement attempt at
   11B.**  Now *de-prioritised* by the W96-A evidence: the
   90B Phase 3 ran 3 fresh seeds × 100 problems and B − A1
   was uniformly negative across all 3 seeds at the larger
   weight class.  A wider 11B sample is no longer plausibly
   the cheapest path to retirement; it would only confirm
   what is already known about the 11B narrowing pattern.

3. **W96-D (`COO-20`) — pivot to ChartQA or RealWorldQA.**
   Remains the documented backup if MathVista is now
   considered structurally capped at the W95-B0 architecture.

The Linear backlog should be re-prioritised: `COO-19` → top
of W96 queue; `COO-18` → de-prioritised; `COO-17` → DONE
with negative verdict; `COO-20` remains as backup.

## Anti-cheat (carried over from W88–W95)

* The 3 × 100 slices were deterministic and pre-committed via
  `select_mathvista_subset_v1(seed, 100, corpus)`.  All three
  pid lists were SHA-256-hashed and written to
  `pre_committed_slice.json` BEFORE any Phase 3 NIM call.
  Slice SHAs are byte-equal to W95 Phase 3.
* Same model (Llama-3.2-90B-Vision-Instruct) on every arm
  (vision mode for A1 / B-reader; text-only mode for A0 /
  B-solver).
* Same K=5 budget on A1 and B; budget gate enforced
  byte-exactly (1 + 5 + 5 = 11 calls / problem on all 300).
* Executor truth = `evaluate_answer_v1` for every arm; no
  LLM judge anywhere.
* Parquet SHA-256 anchored at run start.
* No selective retries.
* Q3 pre-Phase-3 prediction was locked in `docs/RUNBOOK_W96A.md`
  BEFORE the Phase 3 NIM call ran; the actual outcome's
  agreement with H3-A direction (and exceeding the predicted
  narrowing) is consistent with the pre-commit prediction.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* No new core modules added.  W96-A Phase 3 re-used
  `coordpy.mathvista_bench_v1` and the pilot script verbatim.

## Re-running

```bash
NVIDIA_API_KEY=... python scripts/run_w95_mathvista_pilot.py \
  --phase phase3 \
  --vlm-model meta/llama-3.2-90b-vision-instruct \
  --n-problems 100 --n-seeds 3 \
  --seed-start 95005001 \
  --out-dir results/w96/mathvista_90b_phase3 \
  --expected-parquet-sha256 \
    373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d

# Verify offline:
python scripts/verify_w95_mathvista_audit_chain.py \
  --run-dir results/w96/mathvista_90b_phase3/<run-dir>
```

## The honest claim W96-A Phase 3 earns

**On 3 seeds × 100 problems × K=5 × Llama-3.2-90B-Vision-
Instruct on the IDENTICAL 3-seed pre-committed slice set as
W95 Phase 3 at 11B, the W95-B0 candidate (vision-reader +
math-solver + executor-guided reflexion) is BEATEN by the
unified-VLM K=5 baseline by −5.00 pp on the cross-seed mean
(66.33 % vs 71.33 %) and on 2 of 3 seeds; it beats the
text-only baseline by +38.33 pp on the mean and 3/3 seeds.
The cross-scale shift on B − A1 is uniformly negative across
all 3 seeds (−2 / −17 / −7 pp) and the cross-seed mean
shifts from +3.67 pp at 11B to −5.00 pp at 90B (an 8.67 pp
swing AGAINST the team).  Six of nine pre-committed
audit-extended bars PASS; three retirement-margin bars (2,
4, 6) FAIL decisively.  Under the W88 6-bar shape, this is
NOT a retirement; under the *cross-scale interpretation* it
is the FIRST decisive empirical disconfirmation in the W95
line that scaling the VLM weight class can close the
+5 pp margin gap.  Adds carry-forward
`W96-L-MATHVISTA-90B-RETIREMENT-MARGIN-CAP`.  W89
70B-HumanEval-K=5 remains the only confirmed multi-seed
same-budget multi-agent superiority retirement.  The W96
next move per the Linear-recommended ordering becomes
`COO-19` (architecture refinement) rather than further
weight-scale or sample-scale exploration.**
