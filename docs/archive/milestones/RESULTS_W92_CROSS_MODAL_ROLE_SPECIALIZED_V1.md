# W92 — Cross-modal role-specialized V1 (Post-W91 empirical superiority wave V5)

> **2026-05-24 — DECISIVE NEGATIVE.  W92 introduces a NEW
> multi-agent cross-modal architecture with three role-
> specialized models (VLM-Planner, Code-Implementer (×3),
> VLM-Verifier).  Across 7 seeds × 12 problems × K=5 × Llama-
> 3.2-90B-Vision + Llama-3.3-70B-Instruct × HumanEval-Visual
> all_docstring, B_role_spec 77.4 % loses to A1_vlm 88.1 % by
> **−10.71 pp**; B beats A1_vlm on **0 of 7 seeds**.  This is
> the THIRD independent cross-modal architecture in this
> programme to lose to the unified-VLM K=5 baseline.  At this
> point the cumulative evidence is decisive: cross-modal team
> architectures at K=5 on HumanEval-Visual lose to unified VLM
> across split, VLM-in-loop, AND role-specialized
> configurations.  `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`
> STAYS — and the next retirement attempt must change the
> BENCHMARK, not just the architecture.**

## TL;DR

7 seeds × 12 problems × 3 arms on HumanEval-Visual
`all_docstring`.  K=5 budget on A1_vlm and B_role_spec.  VLM =
`meta/llama-3.2-90b-vision-instruct`; text-LM (A0_text floor +
B's Implementer) = `meta/llama-3.3-70b-instruct`.  Total wall
17080 s (~4 h 45 min); 336 text-LM calls + 588 VLM calls; bench
Merkle `c511df459a88ba2e…`; audit verifier 4/4 audit PASS.

| Arm | Mean pass@1 | Per-seed |
|----|---:|---|
| **A0_text** (text-only, no image, 70B) | **53.57 %** | 0.417 / 0.583 / 0.667 / 0.417 / 0.583 / 0.500 / 0.583 |
| **A1_vlm** (single-agent VLM, K=5 first-pass) | **88.10 %** | 0.750 / 0.833 / 1.000 / 0.833 / 0.917 / 1.000 / 0.833 |
| **B_role_spec** (Planner + Implementer×3 + Verifier) | **77.38 %** | 0.583 / 0.750 / 0.917 / 0.750 / 0.833 / 0.833 / 0.750 |

* `b_role_spec_mean_strictly_beats_a0_text_mean = True` ✓ —
  +23.81 pp; image strongly load-bearing.
* `b_role_spec_mean_strictly_beats_a1_vlm_mean = False` ✗ —
  −10.71 pp; LOSES to unified VLM.
* B beats A1_vlm per-seed: `(False, False, False, False, False,
  False, False)` — **0 of 7 seeds** (every per-seed delta
  negative, range −8.33 to −16.67 pp).

## Architecture (the W92 pivot)

```
Turn 0 (T=0):  VLM-Planner     (image, prompt)            → Plan
Turn 1 (T=0.7): Code-Implementer-v1 (prompt, Plan)        → code_v1
                                                          + executor
Turn 2 (T=0):  VLM-Verifier    (image, prompt, code_v1,
                                stderr_v1)                → Critique
Turn 3 (T=0.7): Code-Implementer-v2 (prompt, Plan, code_v1,
                                     stderr_v1, Critique) → code_v2
                                                          + executor
Turn 4 (T=0.7): Code-Implementer-v3 (prompt, Plan, all history,
                                     Critique)            → code_v3
                                                          + executor

Selection: first PASS among (code_v1, code_v2, code_v3); else
lex-smallest CID.
```

**Call mix**: 2× VLM (Planner, Verifier) + 3× code-LM
(Implementer ×3) = 5 model calls, exactly matching A1_vlm's
K=5 unified-VLM budget.  Note B uses LESS total compute than
A1_vlm (the code-LM is smaller than the VLM); a B win would
have been doubly informative.  But B LOSES.

## Why this architecture failed

Three structural reasons emerge from the empirical pattern:

1. **The Implementer has no image access.**  Even with a
   detailed Plan from the VLM-Planner, the Code-Implementer is
   a text-only model that cannot verify or recover from
   Planner mis-extractions.  When the Planner misreads even
   one I/O example, all 3 Implementer attempts inherit that
   error.
2. **3 i.i.d. VLM samples > 3 conditioned code-LM samples.**
   A1_vlm gets 5 independent shots from a 90B VLM with full
   image access on every shot.  B has 3 code-LM shots all
   conditioned on the same Plan + Critique.  Conditioned
   sampling reduces diversity; at K=5 budget the diversity
   gain dominates.
3. **The Verifier's role is bottlenecked by the Implementer's
   ceiling.**  The Critique can identify bugs but the
   Implementer can only act on those identifications within
   its text-only context.  When the bug is a vision-extraction
   error, the Implementer cannot independently verify the
   plan against the image.

## Cumulative cross-modal evidence (W88 → W92)

| Run | VLM | Code-LM | Strip mode | Architecture | Seeds × Probs | A1 | B | B−A1 | B>A1 seeds |
|---|---|---|---|---|---|---:|---:|---:|---:|
| W88 V1 | 11B-V | 8B | doctest_only | split | 3×12 | 86.1 % | 80.6 % | −5.56 pp | 0/3 |
| W89 P2 | 90B-V | 8B | all_docstring | split | 3×12 | 86.1 % | 58.3 % | −27.78 pp | 0/3 |
| W89 P3 | 90B-V | 70B | doctest_only | split | 3×12 | 91.7 % | 86.1 % | −5.56 pp | 0/3 |
| W90 P2 | 90B-V | (8B fl) | doctest_only | VLM-in-loop | 3×12 | 91.7 % | 91.7 % | +0.00 pp | 1/3 |
| W91 P2 | 90B-V | (8B fl) | all_docstring | VLM-in-loop | 3×12 | 83.3 % | 86.1 % | +2.78 pp | 2/3 |
| W91 P2b | 90B-V | (8B fl) | all_docstring | VLM-in-loop | 7×12 | 84.5 % | 77.4 % | −7.14 pp | 2/7 |
| **W92** | **90B-V** | **70B** | **all_docstring** | **role-specialized** | **7×12** | **88.1 %** | **77.4 %** | **−10.71 pp** | **0/7** |

**Three independent architectures, decisively negative**:

* SPLIT (VLM-extract + code-LM-generate): −5.6 to −27.8 pp.
* VLM-IN-LOOP (single-VLM, image every turn): +0.0 (tie) to
  −7.1 pp at 7-seed scale.
* ROLE-SPECIALIZED (VLM-Planner + Code-Implementer + VLM-
  Verifier): **−10.7 pp on 7 seeds, 0/7 per-seed strict
  losses**.

**Image is load-bearing across all 7 configurations**
(B − A0_text = +13.9 / +16.7 / +52.8 / +16.7 / +41.7 / +34.5 /
**+23.8** pp; all > +5 pp threshold).

## What W92 actually shows

* **The cross-modal team architecture family (at K=5 on
  HumanEval-Visual with Llama-3.2-{11B, 90B}-Vision-Instruct
  vs unified-VLM K=5 baseline) is empirically a wrong
  battlefield.**  Across three architecturally-distinct
  attempts spanning model scales, strip modes, and seed
  counts, the team architecture loses to unified VLM.
* **Role specialization is no better than VLM-in-loop on this
  benchmark.**  Both shapes hit ~77.4 % mean at 7-seed scale;
  unified VLM at K=5 hits ~84-88 %.
* **The image-load-bearing property is robust** at 7
  configurations.

## What W92 retires

Nothing.  3/6 retirement bars met (image direction); 3/6 fail
(team-organisation direction).

`W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP` STAYS,
with even stronger negative evidence at 7-seed scale.

## What W92 contributes (new W92-L-* carry-forwards)

* **`W92-L-CROSS-MODAL-ROLE-SPECIALIZED-V1-DECISIVE-NEGATIVE-CAP`**
  — At 7 seeds × 12 problems × Llama-3.2-90B-Vision + Llama-3.3-
  70B-Instruct × HumanEval-Visual all_docstring × K=5, the
  VLM-Planner + Code-Implementer-×3 + VLM-Verifier role-
  specialized architecture LOSES to unified-VLM K=5 by
  −10.71 pp on the mean; B wins 0 of 7 seeds.  Worse than the
  W91 P2b VLM-in-loop result (−7.14 pp).  Adding the Verifier
  + Implementer specialization did NOT help and apparently
  hurt slightly.  Decisive negative against the role-
  specialization hypothesis at this budget on this benchmark.
* **`W92-L-CROSS-MODAL-HUMANEVAL-VISUAL-WRONG-BATTLEFIELD-CAP`**
  — Three architecturally-distinct cross-modal team attempts
  (split, VLM-in-loop, role-specialized) at K=5 on HumanEval-
  Visual + Llama-3.2-{11B, 90B}-Vision-Instruct ALL lose to
  unified-VLM K=5.  The cumulative empirical evidence is
  decisive: HumanEval-Visual at K=5 against the unified VLM
  baseline is the wrong battlefield for proving cross-modal
  team superiority.  Future retirement attempts must choose a
  benchmark where the unified VLM K=5 baseline does NOT
  already approach ceiling, or move to a substrate-level
  cross-modal injection architecture.
* **`W92-L-CROSS-MODAL-ROLE-SPEC-V1-IMPLEMENTER-TEXT-ONLY-CAP`**
  — The Code-Implementer has no direct image access; it only
  sees the Planner's plan + Verifier's critique.  Whether a
  vision-grounded Code-Implementer (i.e., the Implementer is
  also a VLM with image access) changes the picture is V2.

## Anti-cheat re-statement

* ✓ Same VLM model (90B-Vision) on A1_vlm AND on B's Planner +
  Verifier turns.
* ✓ Same code-LM model (70B Llama-3.3-Instruct) on A0_text +
  B's Implementer turns.
* ✓ Same K=5 model-call budget per arm (A1_vlm 5× VLM; B 2×
  VLM + 3× code-LM — B uses LESS total compute and still
  loses).
* ✓ Same task subset per seed across arms.
* ✓ Same retry policy (6 attempts, 429-aware backoff).
* ✓ No selective retries.
* ✓ No baseline weakening.  A1_vlm at 88.1 % is a strong K=5
  self-consistency at 90B-Vision on all_docstring.
* ✓ Audit chain re-derives offline (4/4 PASS).

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New module `coordpy.cross_modal_role_specialized_bench_v1`
  is explicit-import only.
* 6 new CI tests for the W92 role-specialized module
  (`tests/test_w92_cross_modal_role_specialized_v1.py`) all
  pass.

## Re-running

```bash
# NVIDIA_API_KEY must be set.
python scripts/run_w92_cross_modal_role_specialized_bench.py \
    --vlm-model meta/llama-3.2-90b-vision-instruct \
    --text-model meta/llama-3.3-70b-instruct \
    --n-problems 12 --n-seeds 7 \
    --strip-mode all_docstring
python scripts/verify_w92_cross_modal_role_specialized_audit_chain.py
```

NIM provider-side sampling at T=0.7 carries variance; per-seed
deltas may differ on a fresh re-run.  The conclusion ("B_role_spec
LOSES to A1_vlm at K=5 on HumanEval-Visual all_docstring") is the
stable claim and is robust to expected sampling variance at 7
seeds.

## The honest claim this run earns

**At 7 seeds × 12 problems × Llama-3.2-90B-Vision + Llama-3.3-
70B-Instruct × HumanEval-Visual all_docstring × K=5, the
role-specialized cross-modal team (VLM-Planner + Code-
Implementer-×3 + VLM-Verifier) LOSES to the unified-VLM K=5
baseline by 10.71 pp on the mean; B wins 0 of 7 seeds.  This
is the third independent cross-modal architecture in this
programme to lose to unified VLM, and the cumulative
evidence at this point is decisive: HumanEval-Visual at K=5
against unified-VLM K=5 is the wrong battlefield for proving
cross-modal team superiority.  The W88 cross-modal carry-
forward stays with the strongest negative evidence yet from
W92 (worse than W91 P2b's −7.14 pp); future retirement
attempts must move to a benchmark where the unified VLM K=5
baseline does NOT already approach ceiling, or to a
substrate-level cross-modal injection architecture.**

## Strategic implication for W93+

The cumulative W88 → W92 evidence forces a hard pivot:

1. **Either**: choose a multimodal benchmark where the unified
   VLM K=5 baseline is genuinely weaker (not saturating
   90 %+).  Candidates: MathVista (math + vision), ChartQA,
   DocVQA, MMVet, SEED-Bench, RealWorldQA.  The cross-modal
   team architecture needs more failure-residual to rescue.
2. **Or**: develop a substrate-level cross-modal injection
   architecture (the W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP
   direction) that doesn't compete with the unified VLM at
   K=5 sampling — instead, intervenes at the hidden-state
   layer to provide complementary signal that pure sampling
   can't access.

Continuing to attack the current battlefield is now positively
falsified by 3 independent architectures.
