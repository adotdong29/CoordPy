# W96-C — MathVista C1 (VLM-Verifier-Final-Turn) 11B Phase 2 V1 (NEGATIVE)

> **2026-05-24 — PHASE 2 AT 11B-VISION DECISIVELY FAILS THE
> PRE-COMMITTED MARGIN BAR.  At 1 seed × 30 problems × K=5 ×
> Llama-3.2-11B-Vision-Instruct on `AI4Math/MathVista` testmini
> (the SAME deterministic 30-problem slice as W95 Phase 2 and
> W96-A Phase 2, byte-identical): A0_text = 36.67 %, A1_vlm-K=5
> = 63.33 %, B_vlm_team_v2 = 63.33 %.  Margin
> **B_v2 − A1 = +0.00 pp** vs the +5 pp Phase 2 retirement bar.
> Gates 3 and 4 (B > A1; margin ≥ +5 pp) FAIL; gates 1, 2, 5,
> 6, 7, 8, 9 PASS (7 of 9).  Per-problem majority: B_v2 ≥ A1
> on 26 / 30 problems (87 %).  **Verifier rescues = 0 / 11
> text-only failures** — the load-bearing mechanism C1 was
> designed to introduce did not fire on a single problem on
> this slice.**

## Bench configuration

| Field | Value |
|---|---|
| Bench module | `coordpy.mathvista_bench_v2` (W96-C C1 V2) |
| Phase | Phase 2 cheap pilot |
| Seed | 95_005_001 — **byte-identical to W95 Phase 2 + W96-A Phase 2 at 11B** |
| Problems | 30 (deterministic via `select_mathvista_subset_v1`) |
| Slice pid SHA | byte-equal to W95 / W96-A Phase 2 |
| VLM model | `meta/llama-3.2-11b-vision-instruct` via NIM |
| Text/solver model | same (text-only mode) |
| K (budget per A1/B arm) | 5 |
| Sampling temperature | 0.7 (A1, B-solver); 0.0 (A0, B-reader, B-verifier) |
| Max tokens / call | 384 |
| NIM calls (text/vlm) | 120 / 210 = **330 total** |
| Total wall | 448 s (~7.5 min) |
| Run directory | `results/w96/mathvista_c1_pilot/w96c_mathvista_v2_pilot_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260525T032635Z/` |
| Bench Merkle root | `748212dc21e745951bf617622a6079a53271a22819dd5509d5fa77c9800f0277` |
| Seed Merkle root | `482040f1890ac87e8ed1b56460fba37bfd5489dc43ddf6c1d5ed91cc77037933` |

## Per-arm headline numbers

| Arm | Pass@1 on 30-problem slice | Calls/problem |
|---|---:|---:|
| A0_text (text-only, T=0.0) | 36.67 % | 1 |
| A1_vlm K=5 (T=0.7) | 63.33 % | 5 |
| **B_vlm_team_v2 (C1)** | **63.33 %** | 5 |

* **B_v2 − A1 = +0.00 pp** (FAIL; the +5 pp Phase 2 margin bar misses by 5 pp)
* B_v2 − A0 = +26.67 pp (image is still load-bearing for B — but no more than for A1)

## Cross-architecture comparison vs W95-B0 V1 on the same slice

W95 Phase 2 at 11B (V1, `mathvista_bench_v1.B_vlm_team`) on the
*exact same 30-problem slice* (seed 95_005_001):

| Arm | W95 V1 | W96-C V2 | Δ (V2 − V1) |
|---|---:|---:|---:|
| A0_text | 36.67 % | 36.67 % | 0.00 pp |
| A1_vlm K=5 | 66.67 % | 63.33 % | −3.33 pp (sampling variance; both well under 90 % ceiling) |
| B_team | **76.67 %** | 63.33 % | **−13.33 pp** |
| **B − A1** | **+10.00 pp** | **+0.00 pp** | **−10.00 pp** |

**V2 LOST 13.33 pp on the B arm vs V1.**  The vlm_verifier_final
turn (V2's net architectural addition) DID NOT recover the
4 W95-B0 problems that V1's 4th text-solver turn rescued; the
verifier shipped 0 of those rescues.  V2 also failed to recover
any of the 4 A1-only-rescue problems on this slice that the V1
chain also missed.

## Per-problem rescue analysis (V2 only, this slice)

| Outcome (V2 vs A1) | Count | Share |
|---|---:|---:|
| Both A1 and B_v2 PASS | 15 | 50.0 % |
| Both A1 and B_v2 FAIL | 7 | 23.3 % |
| **B_v2 PASS, A1 FAIL (B-only rescue)** | **4** | **13.3 %** |
| **A1 PASS, B_v2 FAIL (A1-only rescue)** | **4** | **13.3 %** |
| Total | 30 | 100 % |

* B-only rescue count: 4 — **all came from text-only solver** (the verifier rescued 0 problems).
* A1-only rescue count: 4 — **the verifier failed to recover any of these**.

## Verifier-rescue accounting (V2-specific instrumentation)

| Metric | Value |
|---|---|
| Text-only PASS (W95-B0-style win) | 19 / 30 (B_v2 shipped a text-only-solver answer) |
| Verifier-rescue (text-only all FAIL → verifier PASS) | **0 / 30** |
| Verifier called (text-only all FAIL → verifier ran) | 11 / 30 (text-only failed; verifier always runs to keep K=5 byte-exact) |
| **Verifier rescue rate** | **0 / 11 = 0.0 %** of text-only-failed problems |

This is the strongest empirical signal in the bench: on the
11 problems where the W95-B0-style text-only chain failed, the
VLM-Verifier-Final shot **also failed on every one of them**.
The verifier-as-VLM has no marginal recovery capability beyond
what A1's K=5 sampling already achieves.

## A1-only-rescue problems the verifier missed (this slice)

The 4 problems where A1 K=5 succeeded but V2 failed (the
verifier's ideal rescue target):

| pid | question_type | answer_type | gold |
|---|---|---|---|
| 470 | multi_choice | text | PDE-Refiner |
| 626 | free_form | integer | 1 |
| 676 | free_form | integer | 0 |
| 947 | multi_choice | text | 115° |

Mix of multi-choice text + free-form numeric — the verifier
failed across both answer-type classes.  The failure is not
localised to one MathVista question structure.

## Pre-committed W96-C V2 Phase 2 gates

| Gate | Threshold | Outcome | Pass? |
|------|-----------|---------|-------|
| 1. Slice pre-committed | 30 pids SHA-anchored | 30 pids, SHA recorded | ✓ |
| 2. A1 < 90 % | required | 63.33 % | ✓ |
| 3. B_v2 strictly > A1 | required | 63.33 % > 63.33 % ? **NO** | **✗** |
| 4. B_v2 − A1 ≥ +5 pp | required | **+0.00 pp** | **✗** (misses by 5 pp) |
| 5. B_v2 − A0 ≥ +5 pp | required | +26.67 pp | ✓ |
| 6. Per-problem B_v2 ≥ A1 majority | ≥ 16 / 30 | 26 / 30 | ✓ |
| 7. Budget exact | 1 + 5 + 5 = 11 | exact | ✓ |
| 8. Audit chain present | bench + seed Merkle | both present | ✓ |
| 9. Executor stays clean | invariants | clean | ✓ |

**2 of 9 pre-committed Phase 2 gates FAIL.**  Phase 2 is KILLED
at 11B per the runbook.

## H1 / H2 verdict (post-11B Phase 2)

Per the W96-C C1 pre-pilot prediction in `docs/RUNBOOK_W96C.md`
("slightly H1-leaning at 11B; the cheap pilot decides"):

* **H1 (C1 wins)**: **falsified at 11B Phase 2.**  V2 ties A1
  on the mean and on the per-problem majority cannot rescue
  the +5 pp margin.
* **H2 (C1 ties or loses)**: **empirically supported.**  V2's
  vlm_verifier_final adds zero load-bearing capability beyond
  A1's K=5 sampling on this slice; the cost of removing one
  text-only solver turn dominates.

The W95-B0 V1's 4th solver turn was empirically load-bearing
at 11B (≥ 4 problems on this 30-problem slice; matches Q4's
21.79 % upper-bound estimate scaled to the slice size).  V2's
choice to spend that budget on the verifier did NOT pay off.

## Mechanism inference

The V2 architecture replaces V1's 4th text-only solver turn
with a VLM-Verifier-Final call.  The verifier sees the image,
the structured extraction, and the prior text-only candidates +
executor verdicts; it produces a single final answer.

Why the verifier failed on this slice:

1. **The verifier is the SAME VLM as A1.**  When A1 K=5 fails
   on a problem, the verifier's single shot at T=0.0 is
   structurally a 6th independent VLM sample (just with prior
   context).  At 11B, the verifier's marginal recovery rate
   on A1's residual is ≤ 1 / 5 (or worse) — too low to clear
   the +5 pp margin on a 30-problem slice.
2. **The extracted facts may bias the verifier.**  The
   verifier's prompt includes the vlm_reader's structured
   extraction; if that extraction was wrong (the W96-A failure
   mode), the verifier may anchor on the wrong extraction
   rather than re-read the image fresh.
3. **The text-only candidates may bias the verifier.**  The
   verifier sees 3 failed candidates; if all 3 picked the
   same wrong answer because the extraction was lossy, the
   verifier may anchor on the consensus wrong answer rather
   than propose a structurally new one.
4. **Single-shot at T=0.0 is a strict constraint.**  The
   verifier has 1 shot at T=0.0 while A1 has 5 shots at T=0.7.
   Even if the verifier were a *stronger* VLM (which it is
   not), its single-shot ceiling is structurally lower than
   A1's K=5 ceiling.

## Decision logic (per `docs/RUNBOOK_W96C.md` cross-scale rule)

The runbook says:

> If Phase 2 FAILS at the *first* tested scale (11B), the 90B
> Phase 2 is **NOT** automatically run.  The author writes the
> W96-C negative result doc and decides whether the 90B Phase 2
> is informative enough to justify the additional ~330 NIM
> call spend; this is recorded as a pre-commit decision before
> any 90B NIM call.

**Pre-commit decision (locked 2026-05-24 BEFORE the 90B NIM
call):**  The 90B Phase 2 is run *anyway* on cross-scale
informational grounds:

1. The W96-A failure mode is at 90B specifically (B_v2 − A1
   = +3.67 → −5.00 pp from 11B to 90B at multi-seed retirement
   scale on V1).  C1 was designed to address the 90B-specific
   mechanism (A1 climbs into the residual on lossy-extraction
   problems); the 11B result tells us little about whether C1
   helps at 90B specifically.
2. The 11B verifier-rescue rate of 0 / 11 may be a sample-size
   artefact.  At 90B, the unified VLM is stronger and the
   verifier-as-VLM may have a higher per-A1-only-problem
   rescue rate.
3. Cost is modest (~330 NIM calls, ~45–60 min wall).  The
   90B confirmation either kills C1 at both scales (clean
   final negative for the W96-C C1 line) or surfaces a
   genuine scale-dependent positive (which would itself be
   important enough to investigate further — though Phase 3
   would still NOT be launched in this milestone per the
   cross-scale rule).

This decision is anchored here BEFORE the 90B Phase 2 NIM
call begins; the 90B result is the actual discriminator.

## Anti-cheat (carried over from W88–W96-A)

* Slice is byte-identical to the pre-committed W95 Phase 2 /
  W96-A Phase 2 30-problem slice (seed 95_005_001).
* Same VLM model on every arm.
* Same K=5 budget byte-exactly (verifier always runs to keep
  the per-problem call count fixed).
* Executor truth = `evaluate_answer_v1` for every arm.  No
  LLM judge.
* Parquet SHA-256 anchored at run start (`373f6c0b…`).
* No selective retries.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* `coordpy.mathvista_bench_v2` remains explicit-import only.

## Carry-forwards added (provisional, pending 90B confirmation)

* `W96-L-MATHVISTA-V2-C1-VERIFIER-FINAL-11B-PHASE2-CAP` —
  C1 (VLM-Verifier-Final-Turn at K=5 byte-exact) failed the
  +5 pp Phase 2 margin bar at 11B-Vision on the byte-identical
  W95 Phase 2 slice; verifier rescue rate was 0 / 11 = 0.0 %
  on text-only-failed problems.

## The honest claim W96-C C1 11B Phase 2 earns

**On 1 seed × 30 problems × K=5 × Llama-3.2-11B-Vision-Instruct
on the byte-identical W95 / W96-A Phase 2 deterministic slice,
the W96-C C1 (VLM-Verifier-Final-Turn) candidate (W95-B0
vlm_reader + 3 text-only solver / reflexion turns + 1
vlm_verifier_final turn) TIES the unified-VLM K=5 baseline on
the mean (63.33 % vs 63.33 %) and shows a per-problem majority
(26 / 30 ≥ A1).  The vlm_verifier_final turn rescued 0 of
11 text-only failures (0.0 % rescue rate) — the load-bearing
mechanism C1 was designed to introduce did not fire on a
single problem on this slice.  V2 LOST 13.33 pp on the B arm
versus W95-B0 V1 on the identical slice (V1: 76.67 %; V2:
63.33 %), confirming that removing one text-only solver turn
dominated the cost-benefit at 11B.  2 of 9 pre-committed Phase
2 gates FAIL (gates 3 and 4: B > A1; margin ≥ +5 pp).  Adds
provisional carry-forward
`W96-L-MATHVISTA-V2-C1-VERIFIER-FINAL-11B-PHASE2-CAP`.  W89
70B-HumanEval-K=5 remains the only confirmed multi-seed
same-budget multi-agent superiority retirement.  90B Phase 2
is running for cross-scale confirmation per the runbook's
"strong reason" exception (the W96-A failure mode is at 90B
specifically; the 11B result tells us about 11B but not
necessarily about 90B).  No Phase 3 launch is licensed in any
case.**
