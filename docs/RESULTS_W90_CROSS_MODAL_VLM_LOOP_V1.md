# W90 — Cross-modal VLM-in-loop bench V1 (Post-W89 empirical superiority wave V3)

> **2026-05-22 — PARTIAL.  The W90 VLM-in-loop architecture
> CLOSES the cross-modal gap from W88/W89's −5.6 / −27.8 / −5.6 pp
> to **+0.0 pp** — the best result so far across the cross-modal
> programme.  However, B_vlm_loop only **TIES** A1_vlm (both at
> 91.7 % mean pass@1), with B winning 1/3 seeds.  The pre-
> committed retirement bars require strict superiority and a
> per-seed majority; neither is met.  Carry-forward
> `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP` STAYS.**

## TL;DR

3 seeds × 12 problems × 3 arms on the HumanEval-Visual corpus
(`strip_mode=doctest_only`).  K=5 budget on A1_vlm and B_vlm_loop.
VLM = `meta/llama-3.2-90b-vision-instruct`; text-LM (A0_text
floor only) = `meta/llama-3.1-8b-instruct`; total wall 3766 s
(≈ 1 h 03 min); 36 text-LM calls + 360 VLM calls.

| Arm | Mean pass@1 | Per-seed |
|----|---:|---|
| **A0_text** (text-only, no image) | **75.00 %** | 0.667 / 0.833 / 0.750 |
| **A1_vlm** (single-agent VLM, K=5 first-pass) | **91.67 %** | 0.833 / 1.000 / 0.917 |
| **B_vlm_loop** (same VLM, K=5 sequential reflexion) | **91.67 %** | 0.917 / 0.917 / 0.917 |

* `b_vlm_loop_mean_strictly_beats_a0_text_mean = True` ✓ —
  +16.67 pp; image is load-bearing.
* `b_vlm_loop_mean_strictly_beats_a1_vlm_mean = False` ✗ —
  B and A1 TIE at 91.67 % mean.  Per-seed: B beats A1 on 1/3
  (seed 2 +8.3 pp), loses on seed 1 (−8.3 pp), ties on seed 3.

Bench Merkle root:
`cb8b4250ef141ba5...`  (full at
`results/w90/cross_modal_vlm_loop/.../cross_modal_vlm_loop_bench_report.json`).
**Audit chain re-derives offline: 4/4 PASS** (36 text-LM + 360
VLM SHA-256s, 3 per-seed Merkle, bench Merkle).

## Cumulative cross-modal evidence (W88 → W89 → W90)

| Run | VLM | Code/Text LM | Strip mode | Arm shape | A1 | B | B−A1 |
|---|---|---|---|---|---:|---:|---:|
| W88 V1 | 11B-Vision | 8B | doctest_only | VLM-extract+code-LM | 86.1 % | 80.6 % | −5.56 pp |
| W89 P2 | 90B-Vision | 8B | all_docstring | VLM-extract+code-LM | 86.1 % | 58.3 % | −27.78 pp |
| W89 P3 | 90B-Vision | 70B | doctest_only | VLM-extract+code-LM | 91.7 % | 86.1 % | −5.56 pp |
| **W90 P2** | **90B-Vision** | (n/a; text-only A0 floor uses 8B) | doctest_only | **VLM-in-loop reflexion** | **91.7 %** | **91.7 %** | **+0.00 pp** |

**The W90 VLM-in-loop is the BEST cross-modal architecture in the
programme so far.**  The gap to A1_vlm has tightened from
−5.6 / −27.8 / −5.6 pp under the various W88/W89 split
configurations to **exactly 0** under VLM-in-loop.  But "best
so far" is not "strict superiority"; tie ≠ retirement.

## Strategic finding

The W88/W89 evidence showed the VLM-extract + code-LM-generate
split is structurally falsified at three model scales.  W90 P2
tests the simplest possible structural replacement: **don't
extract; keep the VLM in the loop**.  Result: the cross-modal
substrate's image-context-every-turn property IS load-bearing
relative to the W88/W89 split (gap closes ~5.6 pp), but at K=5
on this corpus with this VLM, multi-turn reflexion adds zero
marginal value over independent K=5 sampling.

Two plausible reasons (not yet disambiguated):

1. **Ceiling effect.**  A1_vlm at 91.7 % leaves only ~8.3 % of
   problems for B's reflexion to "rescue".  Even a 60 %
   rescue rate would add only ~5 pp — close to the +5 pp
   retirement-margin threshold but not above it.
2. **Reflexion doesn't transfer to multimodal at this scale.**
   The W89 P1 finding that 70B sequential reflexion beats
   K=5 sampling on HumanEval (+5.56 pp) does NOT replicate at
   90B-Vision on HumanEval-Visual.  The image stays the same
   across turns; only stderr changes.  Plain stderr feedback
   may give the VLM less actionable signal than it does in
   the text-only case.

The W90 evidence does not let us pick (1) vs (2) — both
predict tie-or-narrow-loss at K=5.

## Per-seed result

| Seed | A0_text | A1_vlm | B_vlm_loop | B−A0_text | B−A1_vlm |
|----:|---:|---:|---:|---:|---:|
| 90_046_001 | 66.7 % | 83.3 % | 91.7 % | **+25.0 pp** | **+8.3 pp** |
| 90_046_002 | 83.3 % | 100.0 % | 91.7 % | **+8.3 pp** | −8.3 pp |
| 90_046_003 | 75.0 % | 91.7 % | 91.7 % | **+16.7 pp** | tie |
| **mean**   | **75.0 %** | **91.7 %** | **91.7 %** | **+16.67 pp** | **+0.00 pp** |

* `b_vlm_loop_beats_a0_text_per_seed = (True, True, True)`.
* `b_vlm_loop_beats_a1_vlm_per_seed = (True, False, False)`.

Seed-2 is the only seed where A1_vlm reached 100 % — B
couldn't beat it (can't go above 100).  Seed 1 is B's best
relative result (+8.3 pp).  The +0 pp mean is the honest
zero — there's a small per-seed signal but it averages out
across the 3 seeds tested.

## Arm shape

### A0_text — text-only, no image (1 model call at T=0)

```
stripped_prompt -> text-LM(T=0) -> code -> executor -> pass/fail
```

The "no-image floor".  Same as W88 / W89's A0_text arm.  Uses
Llama-3.1-8B-Instruct for honest comparability with W88's
A0_text and to keep the floor honest (a stronger 70B-text A0
would close the gap to A1_vlm artificially without showing
that the image carries the load).

### A1_vlm — single-agent VLM, K=5 first-pass (5 model calls at T=0.7)

```
stripped_prompt + image -> VLM(T=0.7) x K=5 independent
                       -> first PASS by executor; else first
```

Same as W88's A1_vlm but with 90B-Vision instead of 11B-Vision.

### B_vlm_loop — SAME VLM, K=5 sequential reflexion (5 model calls at T=0.7)

```
Call 0: stripped_prompt + image -> VLM(T=0.7) -> code -> exec
Call k (k≥1):
    stripped_prompt + image  # SAME image every turn
    + cumulative history of (cand_i, executor_stderr_i for i<k)
    -> VLM(T=0.7) -> code -> exec
Ship first PASS by attempt index; else lex-smallest CID
```

NO text-only extraction handoff.  The image is in context on
EVERY turn.  Same model on every turn; only the prompt structure
distinguishes turn 0 (initial solver) from turn k > 0
(reflexion-driven refiner).  Total K=5 = exactly A1_vlm's
budget.

## What W90 P2 retires

Nothing yet.  3 of 6 retirement bars passed (image-load-bearing
direction); 3 of 6 failed (team-organisation direction).

`W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP` stays —
even though VLM-in-loop is structurally different from the
W88 split, the retirement bars require strict superiority over
A1_vlm, which W90 P2 does not achieve.

## What W90 P2 contributes

* **The strongest cross-modal architecture in the programme so
  far.**  W90 P2 closes the B−A1_vlm gap to exactly zero
  (+0.00 pp), beating the W88/W89 split configurations by 5.6
  to 27.8 pp.
* **Empirical evidence that the W88/W89 split's failure mode
  was the text-only extraction handoff.**  Removing the
  handoff (keeping image in context every turn) clearly helps —
  but at K=5 with the 91.7 %-K=5 ceiling, the multi-turn
  reflexion can't squeeze out additional wins.
* **A new honest carry-forward:**
  `W90-L-CROSS-MODAL-VLM-LOOP-V1-K5-TIE-CAP` — at K=5 budget
  on Llama-3.2-90B-Vision-Instruct on HumanEval-Visual N=12 ×
  3 seeds, VLM-in-loop sequential reflexion TIES first-pass-
  among-K=5 at 91.7 % mean.  The architecture is the strongest
  cross-modal team shape tested in the programme; strict
  superiority at this scale requires either a different
  benchmark (less saturated ceiling), a different reflexion
  signal (e.g., test generation + verification), or a larger
  budget.

## Anti-cheat re-statement

* ✓ Same VLM model on A1_vlm AND every turn of B_vlm_loop
  (Llama-3.2-90B-Vision-Instruct).
* ✓ Same task subset per seed across arms (the W88
  `select_cross_modal_subset_v1(seed)` discipline preserved).
* ✓ Same prompt budget per arm (A1_vlm K=5; B_vlm_loop K=5).
* ✓ Same retry policy (6 attempts with 429-aware backoff).
* ✓ No selective retries; each (seed, problem, arm) triple is
  exactly one set of calls.
* ✓ Executor truth = full `problem.test` block, same for every
  arm.
* ✓ Audit chain re-derives offline (4/4 PASS via
  `scripts/verify_w90_cross_modal_vlm_loop_audit_chain.py`).
* ✓ No baseline weakening.  A1_vlm at 91.7 % is the strongest
  K=5 self-consistency we've ever measured on this corpus.

## Honest carry-forward limitations

* **`W90-L-CROSS-MODAL-VLM-LOOP-V1-K5-TIE-CAP`** — described
  above.  The headline W90 P2 finding.
* **`W90-L-CROSS-MODAL-VLM-LOOP-V1-K5-CEILING-CAP`** — A1_vlm
  K=5 reaches 91.7 % mean pass@1 on this corpus; the ceiling
  is too close to leave room for reflexion to add a
  measurable margin.  A larger K (10 / 20) or a harder corpus
  may give reflexion room to differentiate.  V2.
* **`W90-L-CROSS-MODAL-VLM-LOOP-V1-90B-VISION-SCALE-CAP`** —
  V1 ran at Llama-3.2-90B-Vision-Instruct.  Whether a stronger
  multimodal model (GPT-4o, Claude 4 Vision, Gemini 2.5 Vision)
  exhibits a different reflexion-to-sampling differential is
  V2.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* `coordpy.cross_modal_vlm_loop_bench_v1` is explicit-import
  only.
* 5 new CI tests for the W90 cross-modal VLM-loop module
  (`tests/test_w90_cross_modal_vlm_loop_v1.py`) all pass.

## Re-running

```bash
# NVIDIA_API_KEY must be set.
python scripts/run_w90_cross_modal_vlm_loop_bench.py \
    --vlm-model meta/llama-3.2-90b-vision-instruct \
    --text-model meta/llama-3.1-8b-instruct \
    --n-problems 12 --n-seeds 3 \
    --strip-mode doctest_only
python scripts/verify_w90_cross_modal_vlm_loop_audit_chain.py
```

The bench reproduces with the same seeds; NIM provider-side
sampling at T=0.7 carries minor variance.  Strict-improvement
bool shapes are the stable closure surface.

## Where this leaves the cross-modal carry-forward

* `W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP` — **stays**
  (substrate-level claim; not addressed by this prong).
* `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP` —
  **stays**.  The W90 P2 architecture is BETTER than the W88
  split (gap closes to 0 from −5.6 pp), but does not strictly
  beat A1_vlm.  The W88 split's empirical failure stands;
  W90 P2 demonstrates a stronger architecture that ties (not
  beats) the same-budget unified VLM baseline.

* **The cross-modal carry-forward is now FOUR structural
  attempts deep.**  Future attempts must explore: larger K
  budgets where independent sampling saturates; deeper
  substrate-level cross-modal injection; benchmarks where the
  unified VLM is genuinely weak.
