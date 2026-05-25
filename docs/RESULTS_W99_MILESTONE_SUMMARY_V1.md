# W99 — RealWorldQA candidate tournament milestone summary V1

> 2026-05-25.  Multi-candidate tournament closing the
> post-W98 RealWorldQA arc.  Slate: B2 (direct-vision final-
> turn answerer; structural frontier lead) + B4 (typed schema
> WITHOUT ``direct_answer_hint``; minimal repair of W98 B1) +
> B5 (deterministic NIM-free question-type router / switch
> baseline).  All three preflight-earned at both 11B and 90B.
> All three promoted to 1-seed × 30-problem × K=5 cheap NIM
> pilots at 11B per the W99 brief's "multiple cheap tries
> allowed when multiple candidates earn it" rule.
>
> The per-candidate verdicts and the milestone-level
> implication are recorded below.  No version bump.  No PyPI
> publish.

## Inputs

| Field | Value |
|---|---|
| Battlefield | RealWorldQA test (``lmms-lab/RealWorldQA``) |
| Slice | seed=96_504_002; n=30; same as W97 / W98 |
| Candidate model | ``meta/llama-3.2-11b-vision-instruct`` |
| Sampling | T=0.7 (A1, B-solver); T=0.0 (A0, B-reader, B2-final-VLM) |
| K | 5 (byte-exact on A1 and every B route) |
| W97 baseline (same slice) | A0=36.67 / A1=90.00 / D2-B0=83.33 (W97 pilot) |
| W98 baseline (same slice) | A0=36.67 / A1=86.67 / B1=80.00 (W98 pilot) |
| W99 preflight verdict | ALL THREE PASS at both 11B and 90B (no NIM spent) |

## Per-candidate pilot verdicts (Phase 2 at 11B)

| Candidate | Mechanism | A0 | A1@K=5 | B | B−A1 | Gates PASS | Structural verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **B5** (switch baseline) | Route multi-choice → W97 D2-B0; else → A1 K=5 | 36.67 % | 93.33 % | **100.00 %** | **+6.67 pp** | 8 / 9 | ``STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP`` (gate 2 only fail; per-problem 30/30; oracle-prediction-match) |
| **B2** (direct-vision final-turn) | Reader + 3 text-solver + final-VLM-on-fail | 36.67 % | 93.33 % | **100.00 %** | **+6.67 pp** | 8 / 9 | ``STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP`` (gate 2 only fail; per-problem 30/30; final-VLM 3/3 rescue) |
| **B4** (typed schema sans hint) | Typed scene-graph (no ``direct_answer_hint``) + question-typed solver | 36.67 % | 93.33 % | **76.67 %** | **−16.67 pp** | 5 / 9 | ``PHASE_2_FAIL`` (gates 2/3/4 all FAIL; **WORSE than W98 B1**; hint-removal hypothesis REFUTED) |

(Per-candidate result docs:
``docs/RESULTS_W99_REALWORLDQA_B5_PHASE2_V1.md`` (done);
``docs/RESULTS_W99_REALWORLDQA_B2_PHASE2_V1.md`` (pending);
``docs/RESULTS_W99_REALWORLDQA_B4_PHASE2_V1.md`` (pending).)

## NIM-free-vs-empirical alignment

| Candidate | NIM-free prediction | Empirical | Alignment |
|---|---|---|---|
| **B5** | 30 / 30 oracle (+10.00 pp vs W97 A1) | 30 / 30 (+6.67 pp vs W99 A1) | EXACT match on per-problem PASS pattern; +6.67 pp margin instead of +10.00 because A1 saturated higher under fresh sampling. |
| **B2** | realistic +6.67 pp (W97 conf-table; 80 % final-VLM rescue) | 30 / 30 (+6.67 pp vs W99 A1); final-VLM 3 / 3 rescue | EXACT margin match; final-VLM rescue rate EXCEEDED realistic (100 % vs 80 %). |
| **B4** | reasoning-only ~ 26-29 / 30 (+0 to +6 pp) | 23 / 30 (−16.67 pp vs W99 A1) | REFUTED: 3-6 pp WORSE than best-case prediction; hint-removal hypothesis fails empirically. |

## Cross-candidate per-problem matrix (slot, to be filled)

(After B2 + B4 land we will publish a 30-row table of
``pid × {A0, A1, D2-B0_W97, B1_W98, B2, B4, B5}`` pass
flags to expose the empirical mechanism overlap.)

## Honest framing of the milestone

* **What W99 has earned so far (B5 PASS)**: a deterministic
  NIM-free question-type router cleanly clears the +5 pp
  Phase 2 bar (structural verdict) on the W97 slice at 11B.
  This proves the per-question routing ceiling, not multi-
  agent context superiority.  B5 stays classified baseline-
  only.
* **What B2 PASS would earn** (if its pilot clears): the
  image-at-decision-boundary mechanism would join the active
  frontier.  Cross-scale 90B Phase 2 entitled.
* **What B4 PASS would earn** (if its pilot clears): the
  typed-schema-sans-hint mechanism would be the simplest
  frontier repair of W98 B1.  Cross-scale 90B Phase 2
  entitled.
* **What W99 has NOT earned**: multi-agent context
  superiority on RealWorldQA.  Even with B5 + B2 + B4 all
  PASSing, that claim requires Phase 3 retirement-grade
  multi-seed multi-problem evidence which is OUT OF SCOPE
  this milestone.

## COO-9 promotion decision (resolved)

**B2 PASSed Phase 2 at 11B.**  Therefore:

* The structural mechanism (image at decision boundary) DOES
  clear the +5 pp Phase 2 bar on RealWorldQA at 11B.
* **COO-9 (second code benchmark) is NOT promoted to lead**.
  It stays at High priority but the cross-modal RealWorldQA
  arc has a genuine W100 follow-up (cross-scale 90B Phase 2
  for B2; if both scales clear, then Phase 3 retirement
  planning).
* The typed-extract-then-text-reason sub-family (D2-B0 +
  W98 B1 + W99 B4) IS empirically capped at 11B; future work
  on that sub-family should NOT continue to chase the same
  family without a structural fix (B2-style image at
  decision boundary).
* B5 (switch baseline) PASS confirms the per-question
  ceiling is high enough, but B5 stays baseline-only ceiling
  reference.

## Carry-forwards (final)

### Added (this milestone)

* ``W99-L-REALWORLDQA-B5-SWITCH-BASELINE-PASS-11B-SLICE-SATURATION-CAP``
* ``W99-L-REALWORLDQA-B5-SWITCH-IS-BASELINE-NOT-FRONTIER-CAP``
* ``W99-L-REALWORLDQA-B2-DIRECT-VISION-FINAL-TURN-PHASE2-11B-STRUCTURAL-PASS-SLICE-SATURATION-CAP``
* ``W99-L-REALWORLDQA-W95-B0-FAMILY-NOT-STRUCTURALLY-CAPPED-AT-IMAGE-AT-DECISION-BOUNDARY-CAP``
* ``W99-L-REALWORLDQA-B4-TYPED-SCHEMA-WITHOUT-HINT-PHASE2-11B-FAIL-CAP``
* ``W99-L-REALWORLDQA-TYPED-EXTRACT-THEN-REASON-SUBFAMILY-EMPIRICALLY-CAPPED-AT-11B-CAP``

### Retired

**None.**  W89 70B-HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.

### Frontier-audit reclassifications

* **Active frontier (promoted)**: B2 (direct-vision final-
  turn answerer) — confirmed structural frontier mechanism
  on RealWorldQA at 11B.
* **Active frontier (added; reasoning-only NIM-free):
  retired**: B4 (typed schema sans hint) — empirically
  refuted; moves to dead-direction.
* **Baseline-only ceiling**: B5 (question-type router) —
  confirmed switch-baseline behaviour; +6.67 pp ceiling on
  this slice; classification unchanged.
* **Dead direction (newly classified)**: typed-extract-then-
  text-reason sub-family of W95-B0 (D2-B0 + W98 B1 + W99 B4
  all capped at ≤ −6.67 pp; specifically W99 B4 worsened to
  −16.67 pp).

## Discipline status

Preflight-first + cross-scale + multi-candidate tournament
discipline validated NINE consecutive times (W93 / W94 / W95
/ W96-A / W96-C / W96-D / W97 / W98 / **W99**).

## Stable boundary preservation

* ``coordpy.__version__`` unchanged at ``0.5.20``.
* ``coordpy.SDK_VERSION`` unchanged at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* New modules ``coordpy.realworldqa_bench_v4`` (B4) and
  ``coordpy.realworldqa_bench_v5`` (B5) explicit-import only.
  ``coordpy.realworldqa_bench_v3`` (B2; built W98) unchanged.

## Cross-references

* ``docs/RUNBOOK_W99.md`` — pre-commit contract.
* ``docs/FRONTIER_RELEVANCE_AUDIT_W99_V1.md`` — frontier vs
  baseline-only vs dead vs anti-pattern.
* ``docs/RESULTS_W99_ARSENAL_MINING_V1.md`` — per-failure
  mining + candidate-slate justification.
* ``docs/RESULTS_W99_PREFLIGHT_V1.md`` — preflight verdict
  (all three PASS at both scales).
* ``docs/RESULTS_W99_REALWORLDQA_B5_PHASE2_V1.md`` — B5
  pilot verdict (STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP).
* ``docs/RESULTS_W99_REALWORLDQA_B2_PHASE2_V1.md`` — B2
  pilot verdict (pending).
* ``docs/RESULTS_W99_REALWORLDQA_B4_PHASE2_V1.md`` — B4
  pilot verdict (pending).
* ``linear_github_mapping.json`` (W99 entry to be appended
  after final pilot lands).
