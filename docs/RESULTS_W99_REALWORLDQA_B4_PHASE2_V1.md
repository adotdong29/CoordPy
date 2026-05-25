# W99 — RealWorldQA B4 Phase 2 cheap pilot V1 (FAIL at 11B; refutes the hint-removal-only repair hypothesis)

> **2026-05-25 — On the W96-D D2 preflight-earned slice
> (seed 96_504_002; 30 problems × K=5) of
> ``lmms-lab/RealWorldQA`` test at Llama-3.2-11B-Vision-
> Instruct, W99 B4 (D2-B4: typed scene-graph + question-typed
> solver, WITHOUT the ``direct_answer_hint`` field) Phase 2
> FAILs 4 of 9 pre-committed gates.  **B4 = 76.67 %; B4 − A1
> = −16.67 pp; structurally WORSE than W98 B1 (80.00 % / −6.67
> pp) and W97 D2-B0 (83.33 % / −6.67 pp) on the same slice.**
> The hint-removal hypothesis is empirically REFUTED: removing
> ``direct_answer_hint`` made the typed-schema family worse,
> not better.  Cross-scale 90B Phase 2 is NOT entitled.

## Configuration

| Field | Value |
|---|---|
| Battlefield | RealWorldQA test (``lmms-lab/RealWorldQA``) |
| Parquet shard SHA-256 (shard 0) | ``0ed8b555586923099bd5d6ba5dd8b656b403ccfc418881facd237a6d6fe64952`` |
| Parquet shard SHA-256 (shard 1) | ``7dcb3ac3483362ca082cd4cddd0ab1389e9a276f1aefd4603fde0a2ce6bc74d0`` |
| Corpus n_problems | 765 |
| Corpus Merkle root | ``dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`` |
| Slice seed | ``96_504_002`` (SAME as W97 / W98 / W99 B2 / W99 B5) |
| Slice n_problems | 30 |
| Slice SHA-256 | ``f53c71c2d355ac55…`` |
| VLM model | ``meta/llama-3.2-11b-vision-instruct`` |
| Text/solver model | (same VLM in text-only mode) |
| Temperature | 0.7 (A1, B4-solver); 0.0 (A0, B4-reader) |
| K | 5 |
| Calls per problem | 1 A0 + 5 A1 + 5 B = 11 |
| Total NIM calls | text = 150, vlm = 180, sum = 330 |
| Run wall | 648 s (~11 min) |
| Bench Merkle root | ``5a5a0731ecb197d4b1125b868993b23b277faaf84f1ee00dc17d0ba55ce5f25e`` |
| Seed Merkle root | ``104f4600990f8e610056431170f1cb46615c63dece38655bf8a8e05b1fc738d9`` |

## Per-arm pass rates

| Arm | Pass rate | Diff vs A0 | Diff vs A1 | Same-slice W97 D2-B0 | Same-slice W98 B1 | Same-slice W99 B2 | Same-slice W99 B5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0_text | **36.67 %** | — | — | 36.67 % | 36.67 % | 36.67 % | 36.67 % |
| A1_vlm K=5 | **93.33 %** | +56.67 pp | — | 90.00 % | 86.67 % | 93.33 % | 93.33 % |
| B4 (typed sans hint) | **76.67 %** | +40.00 pp | **−16.67 pp** | 83.33 % | 80.00 % | 100.00 % | 100.00 % |

## Pre-committed Phase 2 gates (W95 9-gate shape)

| Gate | Verdict | Detail |
|---|---|---|
| 1 — slice pre-committed | **PASS** | 30 pids; slice SHA matches W97 / W98 / B2 / B5 exactly |
| 2 — A1 < 90 % | **FAIL** | A1@K=5 = **93.33 %** (saturated) |
| 3 — B strictly beats A1 | **FAIL** | B4 (76.67 %) < A1 (93.33 %) |
| 4 — Margin B − A1 ≥ +5 pp | **FAIL** | B4 − A1 = **−16.67 pp** (deficit of 21.67 pp below threshold) |
| 5 — Margin B − A0 ≥ +5 pp | **PASS** | B4 − A0 = +40.00 pp |
| 6 — Per-problem B ≥ A1 on ≥ 16 / 30 | **PASS** | B4 ≥ A1 on **24 / 30** problems (80 %; majority bar cleared but margin is per-problem net negative) |
| 7 — Budget accounting exact | **PASS** | 1 + 5 + 5 = 11 calls/problem |
| 8 — Audit chain present | **PASS** | bench + seed Merkle roots recorded |
| 9 — Executor stays clean | **PASS** | Every arm routes through ``evaluate_realworldqa_answer_v1`` |

**5 of 9 gates PASS; gates 2/3/4 FAIL.**

## Structural verdict

Per ``docs/RUNBOOK_W99.md`` decision logic:

* Gate 2 FAIL: A1 saturation (93.33 %) ✓
* Gate 3 FAIL: B4 does NOT strictly beat A1 ✗
* Gate 4 FAIL: B4 − A1 = −16.67 pp (not just below +5pp; LARGELY negative) ✗

Option A (structural positive despite slice saturation) does
NOT apply because gate 3 + gate 4 both fail.  **Structural
verdict: ``PHASE_2_FAIL``.**

## Per-problem failures (B4 FAIL set)

| pid | Q-type | Gold | A1 | W97 D2-B0 | W98 B1 | W99 B2 | W99 B5 | W99 B4 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| ``000013`` | multi_choice_letter | B | FAIL | PASS | FAIL | PASS | PASS | **FAIL** | B4 lost K=4 reflexion-cycling that D2-B0 had even though B4 still has 4 solver turns; typed schema rigidity is the culprit. |
| ``000076`` | multi_choice_letter | A | PASS | PASS | PASS | PASS | PASS | **FAIL** | New regression unique to B4; W97 D2-B0 and W98 B1 both PASSed. Schema-sans-hint introduced a *new* failure mode. |
| ``000204`` | multi_choice_letter | C | PASS | PASS | FAIL | PASS | PASS | **FAIL** | B4 also fails this one (W98 B1 also failed; structural multi-choice issue persists). |
| ``000438`` | numeric | 1 | PASS | PASS | PASS | PASS | PASS | **FAIL** | New regression unique to B4. |
| ``000441`` | multi_choice_letter | C | FAIL | PASS | PASS | PASS | PASS | **FAIL** | B4 LOST the W97 D2-B0 unique-B-rescue.  W98 B1 + W99 B2 + B5 all kept it. |
| ``000615`` | yes_no | No | PASS | FAIL | FAIL | PASS | PASS | **FAIL** | B4 fails the residual viewer-pov problem; only B2 + B5 recovered it (via image access). |
| ``000713`` | numeric | 2 | PASS | PASS | FAIL | PASS | PASS | **FAIL** | W98 B1 also failed this; B4 inherits the same failure mode. |

## Direct comparison vs prior candidates on same slice

* **vs W97 D2-B0** (free-text extraction): B4 LOSES on ``000076``, ``000441``, ``000615`` (new losses vs D2-B0).  B4 RECOVERS only ``000403``, ``000555``, ``000718`` (3 of 5 W97 unique-A1-rescues).  Net B4 − D2-B0 = +3 − 4 = **−1 problem worse**.  Wait, recompute: W97 D2-B0 = 25 PASS; B4 = 23 PASS.  B4 is 2 problems worse than D2-B0 on this slice.
* **vs W98 B1** (typed schema WITH hint): B4 LOSES on ``000076``, ``000441``, ``000438``.  B4 RECOVERS ``000403``, ``000555``, ``000718`` (same 4 yes/no rescues as B1) MINUS ``000615`` which B1 also fails.  W98 B1 = 24 PASS; B4 = 23 PASS.  **B4 is 1 problem worse than B1.**

## Honest reading

### What B4's FAIL refutes

* **The hint-removal-only repair hypothesis is REFUTED**:
  removing ``direct_answer_hint`` does NOT recover W98 B1's
  multi-choice regressions.  It introduces *new* multi-choice
  failures (``000076``, ``000441``) that even W97 D2-B0 (no
  hint at all) PASSed.
* **The W95-B0-derived "typed schema" path has a structural
  cap that is not specifically about the hint field**: the
  rigidity of the typed JSON schema itself (forcing JSON
  output; specific field set) appears to constrain the
  solver's reasoning in ways that are not fully captured by
  the single-field-removal hypothesis.

### What B4's FAIL does NOT prove

* B4 does NOT prove the W95-B0 family as a WHOLE is capped:
  B2 (structurally distinct — image at decision boundary)
  PASSed on the same slice.  The cap is on the *typed
  extract-then-text-reason* sub-family, not the full family.
* B4 does NOT prove the typed schema is useless: it still
  recovered 4 of 5 W97 unique-A1-rescues (same yes/no
  recovery as B1).  The schema primitives ARE load-bearing
  on yes/no perception; the failure is on multi-choice
  reflexion-cycling.

### What B4's FAIL DOES warrant

* Adding a NEW carry-forward ``W99-L-REALWORLDQA-B4-TYPED-SCHEMA-WITHOUT-HINT-PHASE2-11B-FAIL-CAP``.
* The typed-schema family (W95-B0-derived extract-then-
  text-reason without image-at-decision-boundary) is
  empirically capped on RealWorldQA at 11B through THREE
  distinct mechanisms now:
    * W97 D2-B0 (free-text): −6.67 pp
    * W98 B1 (typed WITH hint): −6.67 pp
    * W99 B4 (typed WITHOUT hint): **−16.67 pp** (worst of
      the three)
* The ONLY repair path that worked is B2 (image at decision
  boundary on the failure cluster).  This sharpens the
  frontier audit: the W95-B0 family must be augmented with
  direct visual grounding to escape the cap.

## Cross-scale rule

Per ``docs/RUNBOOK_W99.md`` carry-over:

* Phase 2 FAIL ⇒ 90B Phase 2 NOT auto-launched for B4.
* Cross-scale 90B for B4 is NOT entitled.

## Anti-cheat (all carry-forward held)

* Both parquet shards SHA-anchored at pilot start.
* Slice pre-committed BEFORE any NIM call.
* Same VLM model on every arm.
* Same K=5 byte-exact budget.
* Executor = ``evaluate_realworldqa_answer_v1``.
* No selective retries; no LLM judge.
* Per-call sidecars + per-seed Merkle + bench Merkle.

## Stable boundary preservation

* ``coordpy.__version__`` unchanged at ``0.5.20``.
* ``coordpy.SDK_VERSION`` unchanged at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* ``coordpy.realworldqa_bench_v4`` explicit-import only.

## Carry-forwards

### Added

* **``W99-L-REALWORLDQA-B4-TYPED-SCHEMA-WITHOUT-HINT-PHASE2-11B-FAIL-CAP``**
  — B4 (typed scene-graph + question-typed solver with
  ``direct_answer_hint`` REMOVED; K=5 byte-exact) FAILs
  Phase 2 at 11B with B − A1 = −16.67 pp on the 96_504_002 /
  30-problem slice — WORSE than W98 B1 (typed WITH hint;
  −6.67 pp) by 10.00 pp.  Removing the hint did NOT recover
  W98 B1's multi-choice regressions; it introduced NEW
  multi-choice / numeric failures that even W97 D2-B0 (no
  hint at all) PASSed.  The hint-removal-only repair
  hypothesis is empirically REFUTED.
* **``W99-L-REALWORLDQA-TYPED-EXTRACT-THEN-REASON-SUBFAMILY-EMPIRICALLY-CAPPED-AT-11B-CAP``**
  — the typed-extract-then-text-reason sub-family of the
  W95-B0 architecture (D2-B0 free-text + D2-B1 typed with
  hint + D2-B4 typed without hint) is now empirically
  capped at ≤ +5 pp Phase 2 on RealWorldQA at 11B through
  THREE distinct mechanisms.  The ONLY repair path that
  cleared Phase 2 was B2's image-at-decision-boundary
  mechanism, which is structurally NOT in the typed-extract-
  then-text-reason sub-family.

### Retired

**None.**  W89 70B-HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.

## The honest claim W99 B4 Phase 2 (11B) earns

**On the W96-D-earned 96_504_002 / 30-problem slice of
``lmms-lab/RealWorldQA`` test at ``meta/llama-3.2-11b-
vision-instruct``, the W99 B4 candidate (typed scene-graph
extraction + question-typed solver WITH ``direct_answer_hint``
field REMOVED; K=5 byte-exact) FAILs Phase 2: B4 = 76.67 %
(23 / 30) vs A1 = 93.33 %; B4 − A1 = **−16.67 pp**.  4 of 9
pre-committed gates FAIL (gate 2 A1 saturated; gate 3 B not
> A1; gate 4 margin; gate 5/6/7/8/9 PASS).  Structural
verdict: ``PHASE_2_FAIL``.  The hint-removal hypothesis is
empirically REFUTED: B4 is 10.00 pp WORSE than W98 B1 (typed
WITH hint), introducing NEW multi-choice failures
(``000076`` lost to D2-B0 + ``000441`` lost the W97 D2-B0
unique-B-rescue) AND failing to recover the residual
viewer-pov problem (``000615``).  The typed-extract-then-
text-reason sub-family of the W95-B0 architecture is now
empirically capped at ≤ +5 pp Phase 2 on RealWorldQA at 11B
through THREE distinct mechanisms (D2-B0 free-text at −6.67
pp; D2-B1 typed-with-hint at −6.67 pp; D2-B4 typed-without-
hint at −16.67 pp).  Only B2 (image-at-decision-boundary)
cleared Phase 2 in W99.  Cross-scale 90B Phase 2 NOT
entitled for B4.  Adds two carry-forwards.  No retirements.
Discipline validation #9 (preflight-first + cross-scale +
multi-candidate tournament).**
