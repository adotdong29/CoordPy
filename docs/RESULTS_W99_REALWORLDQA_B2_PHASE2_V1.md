# W99 — RealWorldQA B2 Phase 2 cheap pilot V1 (STRUCTURALLY POSITIVE at 11B with slice-saturation cap)

> **2026-05-25 — On the W96-D D2 preflight-earned slice
> (seed 96_504_002; 30 problems × K=5) of
> ``lmms-lab/RealWorldQA`` test at Llama-3.2-11B-Vision-
> Instruct, W99 B2 (D2-B2: direct-vision final-turn answerer
> = free-text scene reader + 3 text-solver turns with
> reflexion + 1 final-VLM-answerer on text-solver FAIL)
> achieves **PERFECT 30 / 30 = 100.00 % pass rate**.  8 of 9
> pre-committed Phase 2 gates PASS; gate 2 (A1 < 90 %) FAILs
> because A1@K=5 = 93.33 % on this run (same slice saturation
> as B5).  Per ``docs/RUNBOOK_W99.md`` **Option A** (locked
> BEFORE this pilot), the verdict is **STRUCTURALLY POSITIVE
> despite slice-saturation artefact** since B − A1 = +6.67 pp
> > +5 pp threshold AND per-problem B ≥ A1 on 30 / 30 = 100 %.
> The final-VLM answerer was invoked on only 3 / 30 problems
> (text-solver chain short-circuited on the other 27) and
> **rescued 3 / 3 = 100.00 %** of its invocations.  Cross-
> scale 90B Phase 2 is entitled with written justification.
>
> **B2 is the STRUCTURAL frontier mechanism** (image at
> decision boundary on the failure cluster).  Unlike B5
> (switch baseline), B2's PASS is a **frontier-relevant**
> result: keeping the image alive at the final turn IS load-
> bearing on the W97 failure cluster.  But the bar for the
> claim "multi-agent context superiority is solved on
> RealWorldQA" remains higher than a 1-seed × 30-problem
> Phase 2 — Phase 3 retirement evidence is OUT OF SCOPE this
> milestone.

## Configuration

| Field | Value |
|---|---|
| Battlefield | RealWorldQA test (``lmms-lab/RealWorldQA``) |
| Parquet shard SHA-256 (shard 0) | ``0ed8b555586923099bd5d6ba5dd8b656b403ccfc418881facd237a6d6fe64952`` |
| Parquet shard SHA-256 (shard 1) | ``7dcb3ac3483362ca082cd4cddd0ab1389e9a276f1aefd4603fde0a2ce6bc74d0`` |
| Corpus n_problems | 765 |
| Corpus Merkle root | ``dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`` |
| Slice seed | ``96_504_002`` (SAME as W97 / W98 / W99 B5) |
| Slice n_problems | 30 |
| Slice SHA-256 | ``f53c71c2d355ac55…`` |
| VLM model | ``meta/llama-3.2-11b-vision-instruct`` |
| Text/solver model | (same VLM in text-only mode) |
| Temperature | 0.7 (A1, B2-text-solver); 0.0 (A0, B2-reader, B2-final-VLM) |
| K | 5 |
| Calls per problem | 1 A0 + 5 A1 + 5 B = 11 |
| Total NIM calls | text = 147, vlm = 183, sum = 330 |
| Run wall | 577 s (~10 min) |
| Bench Merkle root | ``8ad8369ada4c44aff4c031b5b9d149f3bffc406f1a493f01acf1020a879224bf`` |
| Seed Merkle root | ``b5c67404706a22d6563ee7069c6ae83ddb81db517cf02a7f90aaf253edd0f799`` |

## Per-arm pass rates

| Arm | Pass rate | Diff vs A0 | Diff vs A1 | Same-slice W97 D2-B0 | Same-slice W98 B1 | Same-slice W99 B5 |
|---|---:|---:|---:|---:|---:|---:|
| A0_text | **36.67 %** | — | — | 36.67 % | 36.67 % | 36.67 % |
| A1_vlm K=5 | **93.33 %** | +56.67 pp | — | 90.00 % | 86.67 % | 93.33 % |
| B2 (direct-vision final-turn) | **100.00 %** | +63.33 pp | **+6.67 pp** | 83.33 % | 80.00 % | 100.00 % |

## Pre-committed Phase 2 gates (W95 9-gate shape)

| Gate | Verdict | Detail |
|---|---|---|
| 1 — slice pre-committed | **PASS** | 30 pids; slice SHA matches W97 / W98 / B5 exactly |
| 2 — A1 < 90 % | **FAIL** | A1@K=5 = **93.33 %** (saturated; same slice saturation as B5) |
| 3 — B strictly beats A1 | **PASS** | B2 (100.00 %) > A1 (93.33 %) |
| 4 — Margin B − A1 ≥ +5 pp | **PASS** | B2 − A1 = **+6.67 pp** ≥ +5 pp |
| 5 — Margin B − A0 ≥ +5 pp | **PASS** | B2 − A0 = **+63.33 pp** |
| 6 — Per-problem B ≥ A1 on ≥ 16 / 30 | **PASS** | B2 ≥ A1 on **30 / 30** problems (100 %) |
| 7 — Budget accounting exact | **PASS** | 1 + 5 + 5 = 11 calls/problem |
| 8 — Audit chain present | **PASS** | bench + seed Merkle roots recorded |
| 9 — Executor stays clean | **PASS** | Every arm routes through ``evaluate_realworldqa_answer_v1`` |

**8 of 9 gates PASS; gate 2 FAILs on A1 saturation.**

## Final-VLM invocation summary

| Statistic | Value |
|---|---|
| Final VLM invocations | **3 / 30 = 10.00 %** (well below AddrW99-B2-P4's ≤ 30 % ceiling) |
| Final VLM rescues | **3 / 3 = 100.00 %** |
| Text-solver short-circuit count | **27 / 30 = 90.00 %** |

### Per-problem final-VLM detail

* ``rwqa_test_000135`` (Yes; "Are there any stop signs?") — text-solver FAILed 3 turns; final-VLM saw image + extraction + 3 prior FAIL candidates → PASS.
* ``rwqa_test_000555`` (No; "are the cars facing left?") — text-solver FAILed 3 turns; final-VLM → PASS.
* ``rwqa_test_000615`` (No; "Is the traffic light green for us?") — text-solver FAILed 3 turns; final-VLM → PASS.  **This is the W97 viewer-pov problem that NEITHER W97 D2-B0 NOR W98 B1 could solve.**  B2's final-VLM's direct image access recovered it.

The final-VLM mechanism worked exactly as predicted: it
fired only when the text-solver chain FAILed (3 problems
that were W97 unique-A1-rescues), and on each invocation it
PASSed with full image access.

## Structural verdict

Per ``docs/RUNBOOK_W99.md`` **Option A** (pre-committed
BEFORE any NIM call):

* Gate 2 FAIL (A1@K=5 = 93.33 % ≥ 90 %) ✓
* B − A1 = +6.67 pp > +5 pp ✓
* Per-problem B ≥ A1 on 30 / 30 ≥ 16 / 30 ✓

**Structural verdict: ``STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP``.**

## Direct comparison vs W97 D2-B0 (same slice)

B2 PASSed every problem W97 D2-B0 FAILed:

* ✅ ``rwqa_test_000135`` (Yes; "stop signs") — final-VLM PASS
* ✅ ``rwqa_test_000403`` (No; "Is the light green?") — text-solver PASS (sampling)
* ✅ ``rwqa_test_000555`` (No; "cars facing left") — final-VLM PASS
* ✅ ``rwqa_test_000615`` (No; "traffic light green for us") — **final-VLM PASS** (W97 D2-B0 + W98 B1 both FAIL)
* ✅ ``rwqa_test_000718`` (Yes; depth) — text-solver PASS (sampling)

B2 regressed NO problem W97 D2-B0 won.

## Direct comparison vs W98 B1 (same slice)

B2 PASSed every problem W98 B1 FAILed (same set as B5 + 1
extra):

* ✅ ``rwqa_test_000013`` (B; vehicle direction) — text-solver PASS
* ✅ ``rwqa_test_000155`` (B; gun direction) — text-solver PASS
* ✅ ``rwqa_test_000204`` (C; how many lanes) — text-solver PASS
* ✅ ``rwqa_test_000225`` (A; speed limit) — text-solver PASS
* ✅ ``rwqa_test_000615`` (residual viewer-pov yes/no) — final-VLM PASS
* ✅ ``rwqa_test_000713`` (2; how many cars) — text-solver PASS

B2 regressed NO problem W98 B1 won.

## NIM-free prediction vs empirical result

| Prediction | Empirical | Match |
|---|---|---|
| B2 realistic prediction (+6.67 pp; 80 % final-VLM rescue) | **+6.67 pp** | EXACT margin |
| Final VLM invocation count ≤ 30 % (W97 D2-B0 FAIL = 16.7 %) | 10.00 % | BETTER than upper bound |
| Final VLM rescue rate ~ 80 % | 100 % | EXCEEDED |
| Per-problem B ≥ A1 majority | 30 / 30 = 100 % | PERFECT |

The realistic NIM-free prediction (+6.67 pp) matched the
empirical margin exactly.  The final-VLM rescue rate
EXCEEDED the realistic prediction (100 % vs 80 %).

## Honest reading

### What B2 PASS proves

* **The image-at-decision-boundary mechanism IS load-
  bearing on the W97 failure cluster**.  The final-VLM
  recovered 3 / 3 of the text-solver-FAIL problems, including
  the residual ``000615`` viewer-pov problem that NEITHER
  W97 D2-B0 NOR W98 B1 recovered.
* **B2 is a STRUCTURAL frontier mechanism**: a committed
  answerer with full image access at the decision boundary,
  invoked only on the failure cluster.  Mechanistically
  distinct from W96-C C1 verifier (binary agree/disagree;
  empirically refuted on MathVista).
* **The W95-B0 family CAN be repaired by a structural fix**
  on RealWorldQA at 11B.  The cap at −6.67 pp is NOT
  structural at the family level; it was structural at the
  *no-image-at-decision-boundary* level.
* **The final-VLM invocation rate (10 %) is low**: most
  problems are solved by the text-solver chain.  The
  final-VLM is a *targeted rescue*, not a generic re-answer
  loop.  This is exactly the "image at decision boundary
  where load-bearing" structure the W99 brief endorsed.

### What B2 PASS does NOT prove

* **NOT multi-agent context superiority retirement.**  This
  is 1-seed × 30-problem Phase 2; retirement requires
  multi-seed multi-problem Phase 3.
* **NOT slice-independent.**  A1 saturated at 93.33 % on
  this slice; cross-slice variance is a follow-up question.
* **NOT a complete cross-modal solution**: 90B and other
  cross-modal benchmarks are not tested by this milestone.

### What B2 PASS DOES warrant

* **Cross-scale 90B Phase 2 entitled with written
  justification** (Option C of cross-scale rule; B − A1 =
  +6.67 pp at 11B with per-problem 30 / 30).
* **Promotion of B2 from preflight-earned to active
  frontier confirmed** in the W99 frontier audit.
* **The cross-modal RealWorldQA arc has a frontier-relevant
  W100 follow-up** (90B Phase 2; potentially Phase 3 if
  multiple scales clear).
* **W95-B0 family REPAIR** confirmed via the B2 mechanism.

## Cross-scale rule application

Per ``docs/RUNBOOK_W99.md``:

* Structural verdict ``STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP``
  licenses 90B Phase 2 with written justification.
* 90B A1@K=5 estimated 79.49 % per W96-D preflight — likely
  NOT saturated.  So 90B Phase 2 should have a cleaner
  gate-2 outcome.
* **90B Phase 2 for B2 is entitled and recommended as a
  W100 follow-up.**

## Anti-cheat (all carry-forward held)

* Both parquet shards SHA-anchored at pilot start.
* Slice pre-committed BEFORE any NIM call.
* Same VLM model on every arm (A0 / A1 / B2-reader / B2-
  text-solver-text / B2-final-VLM).
* Same K=5 byte-exact budget.  Text-solver short-circuit
  pads with text-solver retries on same prompt to maintain
  budget parity.
* Executor truth = ``evaluate_realworldqa_answer_v1``.
* No selective retries; no LLM judge.
* Per-call sidecars + per-seed Merkle + bench Merkle
  written.

## Stable boundary preservation

* ``coordpy.__version__`` unchanged at ``0.5.20``.
* ``coordpy.SDK_VERSION`` unchanged at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* ``coordpy.realworldqa_bench_v3`` (B2, built W98) explicit-
  import only.

## Carry-forwards

### Added

* **``W99-L-REALWORLDQA-B2-DIRECT-VISION-FINAL-TURN-PHASE2-11B-STRUCTURAL-PASS-SLICE-SATURATION-CAP``**
  — B2 (free-text scene reader + 3 text-solver + 1 final-
  VLM-on-fail; K=5 byte-exact) achieves PERFECT 100 % PASS
  on the 96_504_002 / 30-problem slice at 11B; B2 − A1 =
  +6.67 pp; per-problem 30 / 30 majority.  Gate 2 FAILs on
  slice saturation (A1 = 93.33 %).  Final-VLM invoked on
  3 / 30 problems and rescued 3 / 3.  STRUCTURAL frontier
  mechanism confirmed.  Cross-scale 90B Phase 2 entitled
  with written justification.
* **``W99-L-REALWORLDQA-W95-B0-FAMILY-NOT-STRUCTURALLY-CAPPED-AT-IMAGE-AT-DECISION-BOUNDARY-CAP``**
  — the W95-B0 family's cap at −6.67 pp via D2-B0 (W97) +
  D2-B1 (W98) was at the *no-image-at-decision-boundary*
  level, not the architecture-family level.  B2 (image at
  decision boundary on text-solver FAIL) clears the +5 pp
  Phase 2 bar.

### Retired

**None.**  W89 70B-HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.

## The honest claim W99 B2 Phase 2 (11B) earns

**On the W96-D-earned 96_504_002 / 30-problem slice of
``lmms-lab/RealWorldQA`` test at ``meta/llama-3.2-11b-
vision-instruct``, the W99 B2 candidate (direct-vision
final-turn answerer = free-text scene reader at T=0.0 + 3
text-solver turns with executor-guided reflexion at T=0.7 +
1 final-VLM answerer at T=0.0 invoked only when all text-
solver turns FAIL; K=5 byte-exact) achieves PERFECT 100 %
pass rate (30 / 30) and B − A1 = +6.67 pp.  8 of 9 pre-
committed Phase 2 gates PASS; gate 2 FAILs on A1 saturation
(93.33 % on this slice).  Per the pre-committed Option A in
``docs/RUNBOOK_W99.md``, the structural verdict is
``STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP``.  The
final-VLM mechanism is empirically load-bearing on the W97
failure cluster: invoked on 3 / 30 = 10.00 % of problems
(well below the 30 % AddrW99-B2-P4 ceiling) and rescued
3 / 3 = 100.00 % of invocations — INCLUDING the residual
``000615`` viewer-pov problem that NEITHER W97 D2-B0 NOR
W98 B1 could solve.  B2 PASSes are evidence that the image-
at-decision-boundary mechanism IS load-bearing as the W99
brief predicted; multi-agent context superiority remains
UN-CLAIMED unless a Phase 3 multi-seed multi-problem bench
clears.  Cross-scale 90B Phase 2 is entitled with written
justification (90B A1 estimated 79.49 % per W96-D preflight
should clear gate 2 cleanly).  B2 is promoted from
preflight-earned to active frontier confirmed in the W99
frontier audit.  W95-B0 family REPAIR via B2 mechanism
confirmed: the cap at −6.67 pp was at the *no-image-at-
decision-boundary* level, not the architecture-family
level.  Adds two carry-forwards.  Discipline validation #9
(preflight-first + cross-scale + multi-candidate
tournament).**
