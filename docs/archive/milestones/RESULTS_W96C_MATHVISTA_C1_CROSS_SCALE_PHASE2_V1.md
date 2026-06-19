# W96-C — MathVista C1 (VLM-Verifier-Final-Turn) cross-scale Phase 2 V1 (AMBIGUOUS)

> **2026-05-24 — CROSS-SCALE PHASE 2 IS AMBIGUOUS, NOT
> RETIREMENT-GRADE.  At 1 seed × 30 problems × K=5 on
> `AI4Math/MathVista` testmini's pre-committed seed-95_005_001
> slice:**
>
> | Scale | A0 | A1 K=5 | **B_v2** | **B_v2 − A1** | Verifier rescues |
> |---|---:|---:|---:|---:|---:|
> | **11B-Vision** | 36.67 % | 63.33 % | 63.33 % | **+0.00 pp** | **0 / 11 = 0.0 %** |
> | **90B-Vision** | 36.67 % | 66.67 % | 80.00 % | **+13.33 pp** | **1 / 7 = 14.3 %** |
>
> **The W96-C C1 candidate collapses at 11B (+0 pp; FAIL Phase
> 2) but PASSes at 90B (+13.33 pp).  This is exactly the
> "candidate only looks good at one scale" pattern the W96-C
> runbook's cross-scale rule (locked POST-W96-A) treats as a
> warning, not a green light.  The verifier-rescue mechanism
> (the load-bearing architectural addition C1 was designed
> around) fires on only 0 / 11 text-only failures at 11B and
> 1 / 7 at 90B — the verifier is NOT load-bearing in either
> case.  The 90B Phase 2 PASS is therefore most plausibly
> driven by sampling variance on the 3-turn text-only chain
> + single-seed slice-luck on a +6.67 pp B-arm delta vs W96-A
> V1 on the identical slice (V1 +10 pp pilot → V1 Phase 3 −5 pp
> retirement bench).  C1 is NOT entitled to Phase 3 at any
> scale.**

## Bench configurations

### 11B-Vision

| Field | Value |
|---|---|
| Bench module | `coordpy.mathvista_bench_v2` (W96-C C1 V2) |
| VLM model | `meta/llama-3.2-11b-vision-instruct` via NIM |
| Phase | Phase 2 cheap pilot (1 seed × 30 problems) |
| Total wall | 448 s (~7.5 min) |
| Bench Merkle | `748212dc21e745951bf617622a6079a53271a22819dd5509d5fa77c9800f0277` |
| Run dir | `results/w96/mathvista_c1_pilot/w96c_mathvista_v2_pilot_meta_llama-3.2-11b-vision-instruct__..._20260525T032635Z/` |

### 90B-Vision

| Field | Value |
|---|---|
| Bench module | `coordpy.mathvista_bench_v2` (W96-C C1 V2) |
| VLM model | `meta/llama-3.2-90b-vision-instruct` via NIM |
| Phase | Phase 2 cheap pilot (1 seed × 30 problems) |
| Total wall | 1269 s (~21.2 min) |
| Bench Merkle | `e7fb5290cd9d439f2666e641e0d7e670be062aeb28ea1fc38266d870251076ba` |
| Run dir | `results/w96/mathvista_c1_pilot/w96c_mathvista_v2_pilot_meta_llama-3.2-90b-vision-instruct__..._20260525T033556Z/` |

## Cross-architecture, cross-scale comparison (same 30-problem slice)

| Arm | W95 V1 11B | W96-C V2 11B | Δ V2 − V1 11B | W96-A V1 90B | W96-C V2 90B | Δ V2 − V1 90B |
|---|---:|---:|---:|---:|---:|---:|
| A0_text | 36.67 % | 36.67 % | 0.00 pp | 33.33 % | 36.67 % | +3.34 pp (variance) |
| A1_vlm K=5 | 66.67 % | 63.33 % | −3.34 pp (variance) | 63.33 % | 66.67 % | +3.34 pp (variance) |
| **B_team** | **76.67 %** | **63.33 %** | **−13.34 pp** | **73.33 %** | **80.00 %** | **+6.67 pp** |
| **B − A1** | **+10.00 pp** | **+0.00 pp** | **−10.00 pp** | **+10.00 pp** | **+13.33 pp** | **+3.33 pp** |

At 11B, V2 LOST 13.34 pp on the B arm vs V1 on the same slice (V1's 4th solver turn was load-bearing on ≥ 4 problems V2 cannot reach).  At 90B, V2 GAINED 6.67 pp on the B arm vs V1 on the same slice (V2's 3-turn text-only chain happened to sample better; the verifier rescued 1 problem).

## Verifier-rescue accounting (the C1 mechanism)

| Scale | text-only PASS (W95-B0-style) | verifier-rescue (text-only all FAIL → verifier PASS) | Verifier called | Verifier rescue rate |
|---|---:|---:|---:|---:|
| 11B | 19 / 30 | **0 / 30** | 11 / 30 | **0 / 11 = 0.0 %** |
| 90B | 23 / 30 | **1 / 30** | 7 / 30 | **1 / 7 = 14.3 %** |

The vlm_verifier_final turn (V2's load-bearing architectural
addition over V1) rescued **0 problems at 11B and 1 problem at
90B**.  At 11B, the verifier had 11 chances to recover and got
none.  At 90B, the verifier had 7 chances and got 1.  The
rescue rate is structurally too low to drive a meaningful
margin gain on a 30-problem slice.

## Per-problem rescue analysis (V2 vs A1)

### 11B

| Outcome | Count |
|---|---:|
| Both A1 and B_v2 PASS | 15 |
| Both A1 and B_v2 FAIL | 7 |
| B-only rescue (B_v2 PASS, A1 FAIL) | 4 |
| A1-only rescue (A1 PASS, B_v2 FAIL) | 4 |

### 90B

| Outcome | Count |
|---|---:|
| Both A1 and B_v2 PASS | 19 |
| Both A1 and B_v2 FAIL | 5 |
| B-only rescue (B_v2 PASS, A1 FAIL) | 5 |
| A1-only rescue (A1 PASS, B_v2 FAIL) | 1 |

At 90B, V2 has 5 B-only rescues (V2 PASS, A1 FAIL) and 1
A1-only rescue (A1 PASS, V2 FAIL).  Net rescue advantage =
+4 problems.  Of these 5 B-only rescues, 4 came from text-
only and 1 from the verifier rescue.  Net B-A1 = +4 problems
= +13.33 pp.

At 11B, V2 has 4 B-only rescues and 4 A1-only rescues.  Net
rescue advantage = 0 problems.  All 4 B-only rescues came
from text-only.  Net B-A1 = 0 problems = +0.00 pp.

## Pre-committed Phase 2 gates (cross-scale)

### 11B (V2 Phase 2)

| Gate | Outcome | Pass? |
|------|---------|-------|
| 1. Slice pre-committed | 30 pids, SHA recorded | ✓ |
| 2. A1 < 90 % | 63.33 % | ✓ |
| 3. B_v2 strictly > A1 | 63.33 % > 63.33 %? **NO** | **✗** |
| 4. B_v2 − A1 ≥ +5 pp | **+0.00 pp** | **✗** (misses by 5 pp) |
| 5. B_v2 − A0 ≥ +5 pp | +26.67 pp | ✓ |
| 6. Per-problem B_v2 ≥ A1 majority | 26 / 30 | ✓ |
| 7. Budget exact | 1 + 5 + 5 = 11 | ✓ |
| 8. Audit chain present | both present | ✓ |
| 9. Executor clean | invariants | ✓ |

**7 of 9 gates PASS; gates 3 and 4 FAIL at 11B.**

### 90B (V2 Phase 2)

| Gate | Outcome | Pass? |
|------|---------|-------|
| 1. Slice pre-committed | 30 pids, SHA recorded | ✓ |
| 2. A1 < 90 % | 66.67 % | ✓ |
| 3. B_v2 strictly > A1 | 80.00 % > 66.67 % | ✓ |
| 4. B_v2 − A1 ≥ +5 pp | **+13.33 pp** | ✓ |
| 5. B_v2 − A0 ≥ +5 pp | +43.33 pp | ✓ |
| 6. Per-problem B_v2 ≥ A1 majority | 29 / 30 | ✓ |
| 7. Budget exact | 1 + 5 + 5 = 11 | ✓ |
| 8. Audit chain present | both present | ✓ |
| 9. Executor clean | invariants | ✓ |

**9 of 9 gates PASS at 90B.**

## H1 / H2 cross-scale verdict (post-Phase 2)

Per the W96-C C1 pre-pilot prediction in `docs/RUNBOOK_W96C.md`
("slightly H1-leaning at 11B; slightly H2-leaning at 90B"):

* **H1 (C1 wins cross-scale)**: **partially supported, but
  with critical mechanism-level caveats.**  90B Phase 2 PASS
  at +13.33 pp meets the formal +5 pp bar.  11B Phase 2 FAILs
  at +0.00 pp.  The cross-scale invariance required by the
  runbook (PASS at both scales) is not met.
* **H2 (C1 ties or loses cross-scale)**: **partially supported.**
  11B is a clear FAIL; 90B is a formal PASS but the verifier-
  rescue mechanism (the architectural addition C1 is built
  around) is NOT load-bearing (1 / 7 = 14.3 % rescue rate at
  90B; 0 / 11 = 0.0 % at 11B).

The **architectural-invariance reading from the W96-A Phase 2
pattern** (V1 +10 pp at both 11B and 90B on the same slice)
does not extend to V2: V2 swings from −10 pp vs V1 at 11B to
+6.67 pp vs V1 at 90B on the identical slice.  The +6.67 pp
delta at 90B is well within single-seed sampling variance on
the 3-turn solver chain (estimated ±5–7 pp at K=5 / 30
problems / T=0.7).

## Mechanism inference (cross-scale)

The empirical signal is that **the verifier-rescue mechanism
is not driving the cross-scale result**:

* At 11B: the verifier rescued 0 / 11 text-only failures.
  The architecture FAILed.
* At 90B: the verifier rescued 1 / 7 text-only failures.
  The architecture PASSed by +13.33 pp.

For the verifier to be the load-bearing driver of the 90B PASS,
the verifier's contribution would need to be a major share of
the +13.33 pp margin.  Verifier rescues = 1 problem = +3.33 pp;
the remaining +10 pp of the V2 − A1 margin at 90B comes from
text-only solver passes (23 / 30) exceeding A1 K=5 passes
(20 / 30) by 3 problems = +10 pp.

Those 3 extra text-only passes at 90B are not attributable to
the C1 architectural change (the text-only chain is V1-style
unchanged except for one fewer reflexion turn).  They are
attributable to:

1. **Single-seed sampling variance** on the 3-turn solver
   chain at T=0.7 (≤ 30 problems × 3 turns; SE ≈ ±5 pp).
2. **Slice-luck**: this specific 30-problem deterministic
   slice happens to favour V2's 3-turn chain at 90B over A1's
   K=5 i.i.d. sampling.  W96-A V1 at 90B on the same slice
   got +10 pp at single-seed Phase 2 then **−5 pp** at 3-seed
   × 100 Phase 3 (a −15 pp narrowing); V2's +13 pp is in the
   same ballpark single-seed.

The W96-A lesson is direct: a +10 pp Phase 2 PASS on a
30-problem slice does NOT predict a +5 pp Phase 3 retirement
at 3-seed × 100.  V2's +13 pp Phase 2 PASS at 90B is
structurally subject to the same risk.

## Cross-scale rule outcome (per `docs/RUNBOOK_W96C.md`)

The runbook locks the Phase 3 entitlement rule:

> Either Phase 2 PASS at *both* 11B and 90B (strongest case);
> or Phase 2 PASS at 90B only, with a written justification
> for why 11B Phase 2 PASS is not required.

V2's outcome: 11B FAIL + 90B PASS.

The runbook's "90B-only" exception was reserved for "the W96-A
negative was at 90B; if C1 fixes 90B specifically, that is
sufficient for the next Phase 3 at 90B."  The honest reading
of the empirical evidence is that C1 does NOT specifically
fix 90B: the verifier mechanism is not load-bearing in either
case.  The 90B PASS is most plausibly sampling-variance-
driven on the 3-turn solver chain + single-seed slice luck.

Per the user's stronger guidance (locked in the milestone
pre-commit): *"if a candidate only looks good at one scale
and collapses at the other, treat that as a warning, not a
green light."*  This is exactly the scale-dependent-collapse
pattern.  **C1 is therefore NOT entitled to Phase 3 at any
scale.**

## Discipline validation

This is the W93 → W94 → W95 → W96-A → W96-C preflight + cross-
scale discipline working as designed:

* W95 Phase 2 single-seed +10 pp → Phase 3 multi-seed +3.67 pp
  (narrowed by −6.33 pp).
* W96-A Phase 2 single-seed +10 pp → Phase 3 multi-seed
  −5.00 pp (narrowed by −15.00 pp).
* W96-C C1 Phase 2 cross-scale: 11B FAIL + 90B PASS at
  +13.33 pp → cross-scale rule blocks Phase 3 escalation.

The +5 pp pre-committed retirement bar + the cross-scale rule
correctly identified that the V2 architecture's 90B PASS is
not robust enough to justify a Phase 3 run.

## What this means

### Decisive empirical findings

* **The vlm_verifier_final mechanism is NOT load-bearing** at
  either 11B (rescue rate 0 / 11 = 0.0 %) or 90B (rescue rate
  1 / 7 = 14.3 %).
* **The V2 architecture's net B-arm performance vs V1 on the
  same slice is scale-dependent**: V2 LOSES 13.34 pp at 11B,
  GAINS 6.67 pp at 90B.  Net cross-scale: ambiguous.
* **The 90B +13.33 pp V2 − A1 margin is most plausibly
  sampling-variance-driven** on a 30-problem single-seed
  slice with T=0.7 solver sampling.

### What the W96-C C1 line empirically rules out

* **The VLM-Verifier-Final-Turn architecture, at the K=5
  budget with one VLM verifier call at T=0.0, is NOT a
  reliable mechanism for closing the cross-modal team
  superiority margin on MathVista at the W95-B0-derived
  decomposition shape.**  At 11B, the verifier doesn't fire
  enough; at 90B, the verifier doesn't fire enough either.
  The architectural addition (1 VLM call replacing 1 text
  call) costs more in lost text-only solver capacity than it
  gains in image-grounded rescue capacity at this budget
  shape.

### What is NOT yet entitled

* No new carry-forward retirement.
* The W95 `W95-L-MATHVISTA-RETIREMENT-MARGIN-CAP` and W96-A
  `W96-L-MATHVISTA-90B-RETIREMENT-MARGIN-CAP` are NOT retired;
  they are joined by two new W96-C carry-forwards (below).

## What might still unlock retirement (next moves)

Per the Linear-recommended ordering and the arsenal-mining
inventory in `docs/RESULTS_W96C_ARSENAL_MINING_V1.md`:

1. **W96-D (`COO-20`) — battlefield pivot to ChartQA or
   RealWorldQA.**  MathVista's structural ceiling on the
   W95-B0 decomposition may simply be empirical fact: V1 caps
   at +3.67 pp (Phase 3); V2 caps at +0 pp / +13 pp single-
   seed (Phase 2 cross-scale); the V1→V2→Phase-3 progression
   would likely narrow to ≤ +3 pp by analogy with W96-A.  A
   different benchmark with a different ceiling profile
   (ChartQA, RealWorldQA) may produce a cleaner ≥ +5 pp
   signal.

2. **W96-C C2 — tool-augmented solver.**  Addresses the
   *arithmetic* failure mode rather than the *image-access*
   failure mode.  The V2 evidence shows that the image-access
   mechanism (verifier) is not load-bearing; tool augmentation
   of the math step might recover problems where the
   extraction is correct but the arithmetic is hard.  Cost:
   needs tool budget accounting at K=5 (a tool call is not a
   model call but takes wall time; the tool substrate from
   W84 has the discipline for this).

3. **W96-C V3 (verifier-in-loop, not just final).**  Spend the
   budget on *multiple* verifier calls interleaved with
   solver turns, rather than one final verifier turn.  This
   trades MORE solver budget for MORE verifier budget; at
   K=5 the trade-off is tight (e.g., 1 reader + 1 solver + 1
   verifier + 1 solver + 1 verifier).  Higher complexity, more
   verifier chances; needs its own preflight + pilot per the
   W93 discipline.  V3 is **not** implemented in this
   milestone; it is documented as a future candidate.

The Linear backlog should be re-prioritised: `COO-19` (C1)
moves to Done with the cross-scale ambiguous verdict; `COO-20`
(W96-D) is elevated to High as the next-best line; C2 / V3
remain Medium / Low as architectural backups.

## Anti-cheat (carried over from W88–W96-A)

* Both 11B and 90B Phase 2 ran on the **byte-identical
  pre-committed slice** (seed 95_005_001; 30 pids; same as
  W95 Phase 2 and W96-A Phase 2).
* Same VLM model on every arm at each scale (vision mode for
  A1 / B-reader / B-verifier; text-only mode for A0 / B-solver).
* Same K=5 budget byte-exactly (verifier always runs on every
  problem, even when text-only short-circuits, to keep the
  per-problem call count fixed).
* Executor truth = `evaluate_answer_v1` for every arm; no
  LLM judge.
* Parquet SHA-256 anchored at run start (`373f6c0b…`) at both
  scales.
* No selective retries.
* Pre-commit decision to run 90B Phase 2 on cross-scale
  informational grounds was recorded in
  `docs/RESULTS_W96C_MATHVISTA_C1_11B_PHASE2_V1.md` BEFORE
  any 90B NIM call ran; the 90B PASS does not retro-fit the
  cross-scale rule.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* `coordpy.mathvista_bench_v2` remains explicit-import only.

## Carry-forwards added

* `W96-L-MATHVISTA-V2-C1-VERIFIER-FINAL-11B-PHASE2-CAP` —
  C1 (VLM-Verifier-Final-Turn at K=5 byte-exact) FAILed the
  +5 pp Phase 2 margin bar at 11B-Vision on the byte-identical
  W95 Phase 2 slice; verifier rescue rate 0 / 11 = 0.0 % on
  text-only-failed problems.  V2 LOST 13.34 pp on the B arm
  vs V1 on the same slice (V1's 4th solver turn rescued 4
  problems V2 cannot reach).
* `W96-L-MATHVISTA-V2-C1-VERIFIER-FINAL-90B-PHASE2-SINGLE-
  SEED-NON-MECHANISM-DRIVEN-CAP` — C1 PASSed the +5 pp Phase
  2 bar at 90B at +13.33 pp single-seed, but the verifier
  rescue rate (1 / 7 = 14.3 %) is too low to drive the margin;
  the 90B PASS is most plausibly sampling-variance-driven on
  the 3-turn solver chain + single-seed slice luck.  Per the
  cross-scale rule, this does NOT license Phase 3 escalation.

## Carry-forwards retired

None.

## Re-running

```bash
# 11B
NVIDIA_API_KEY=... python scripts/run_w96c_mathvista_pilot.py \
  --vlm-model meta/llama-3.2-11b-vision-instruct \
  --n-problems 30 --n-seeds 1 \
  --out-dir results/w96/mathvista_c1_pilot

# 90B
NVIDIA_API_KEY=... python scripts/run_w96c_mathvista_pilot.py \
  --vlm-model meta/llama-3.2-90b-vision-instruct \
  --n-problems 30 --n-seeds 1 \
  --out-dir results/w96/mathvista_c1_pilot
```

## The honest claim W96-C C1 cross-scale Phase 2 earns

**On 1 seed × 30 problems × K=5 × Llama-3.2-{11B, 90B}-Vision-
Instruct on the byte-identical W95 / W96-A Phase 2 deterministic
slice, the W96-C C1 (VLM-Verifier-Final-Turn) candidate produces
a CROSS-SCALE-AMBIGUOUS result: B_v2 − A1 = +0.00 pp at 11B
(FAIL +5 pp bar; gates 3 and 4 FAIL) and +13.33 pp at 90B (PASS
+5 pp bar; all 9 gates PASS).  The architectural addition's
load-bearing mechanism (the VLM-verifier rescuing text-only
failures) fires on 0 / 11 = 0.0 % of text-only failures at 11B
and 1 / 7 = 14.3 % at 90B — too rare in either case to drive a
meaningful margin gain.  The 90B PASS is therefore most
plausibly sampling-variance + slice-luck on the 3-turn text-
only solver chain.  V2 LOST 13.34 pp on the B arm vs V1 at
11B on the identical slice; V2 GAINED 6.67 pp vs V1 at 90B on
the identical slice — a scale-dependent collapse pattern the
runbook's cross-scale rule explicitly treats as a warning,
not a green light.  Per the W96-C cross-scale rule + the
W96-A lesson (single-seed +10 pp can narrow to −5 pp at
multi-seed Phase 3), C1 is **NOT entitled to Phase 3 at any
scale in this milestone**.  Adds carry-forwards
`W96-L-MATHVISTA-V2-C1-VERIFIER-FINAL-11B-PHASE2-CAP` and
`W96-L-MATHVISTA-V2-C1-VERIFIER-FINAL-90B-PHASE2-SINGLE-SEED-
NON-MECHANISM-DRIVEN-CAP`.  W89 70B-HumanEval-K=5 remains the
only confirmed multi-seed same-budget multi-agent superiority
retirement.  The next move per the post-W96-C Linear ordering
is `COO-20` (W96-D battlefield pivot to ChartQA or
RealWorldQA), with `COO-19` C2 (tool-augmented solver) and a
documented V3 (verifier-in-loop) as architectural backups.**
