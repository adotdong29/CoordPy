# W100 — RealWorldQA B2 Phase 2 cross-scale 90B cheap pilot V1 (FAIL — clean cross-scale collapse)

> **2026-05-25 — On the W96-D-earned 96_504_002 / 30-problem
> slice of ``lmms-lab/RealWorldQA`` test at
> ``meta/llama-3.2-90b-vision-instruct``, W100 B2 (direct-vision
> final-turn answerer; UNCHANGED from W98 / W99) achieves
> **22 / 30 = 73.33 % pass rate**.  A1 @ K=5 = 76.67 %; A0 =
> 46.67 %.  **B2 − A1 = −3.33 pp; gates 3 + 4 FAIL.**
> Mechanism-load-bearingness sub-gate **MLB-2 FAILS** (final-
> VLM rescue rate = 1 / 9 = 11.11 %; threshold ≥ 33.33 %).
> Structural verdict ``PHASE_2_FAIL``.  The image-at-decision-
> boundary mechanism is **NOT load-bearing at 90B** even though
> it WAS load-bearing at 11B.  This is a clean **cross-scale
> collapse** — same pattern as W96-C C1 verifier (PASS at 90B
> but variance-driven; mechanism not load-bearing).
>
> Per ``docs/RUNBOOK_W100.md`` Part H (code-pivot contingency,
> pre-committed BEFORE this pilot), the verdict triggers:
>
> 1. ``COO-9`` (second code benchmark) PROMOTED to lead path.
> 2. The W99 W95-B0-family REPAIR claim is restricted to the
>    11B regime; cross-scale generalisation is NOT earned.
> 3. The cross-modal RealWorldQA arc is **frozen at 11B** in
>    the W100 frontier audit.
> 4. Phase 3 retirement bench NOT launched.
>
> The W99 11B B2 PASS carry-forward
> ``W99-L-REALWORLDQA-B2-DIRECT-VISION-FINAL-TURN-PHASE2-11B-STRUCTURAL-PASS-SLICE-SATURATION-CAP``
> STANDS unchanged; the 90B FAIL adds a *cross-scale-bound*
> qualifier carry-forward
> ``W100-L-REALWORLDQA-B2-DIRECT-VISION-FINAL-TURN-PHASE2-90B-CAP``
> but does NOT erase the 11B truth.

## Configuration

| Field | Value |
|---|---|
| Battlefield | RealWorldQA test (``lmms-lab/RealWorldQA``) |
| Parquet shard SHA-256 (shard 0) | ``0ed8b555586923099bd5d6ba5dd8b656b403ccfc418881facd237a6d6fe64952`` |
| Parquet shard SHA-256 (shard 1) | ``7dcb3ac3483362ca082cd4cddd0ab1389e9a276f1aefd4603fde0a2ce6bc74d0`` |
| Corpus n_problems | 765 |
| Corpus Merkle root | ``dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab`` |
| Slice seed | ``96_504_002`` (SAME as W97 / W98 / W99) |
| Slice n_problems | 30 |
| Slice SHA-256 | ``f53c71c2d355ac55…`` |
| VLM model | ``meta/llama-3.2-90b-vision-instruct`` |
| Text/solver model | (same VLM in text-only mode) |
| Temperature | 0.7 (A1, B2-text-solver); 0.0 (A0, B2-reader, B2-final-VLM) |
| K | 5 |
| Calls per problem | 1 A0 + 5 A1 + 5 B = 11 |
| Total NIM calls | text = 141, vlm = 189, sum = 330 |
| Run wall | 611 s (~10 min) |
| Bench Merkle root | ``e1f9493efb284cf230104ab3d2e0805b7bebfdf41547c46502534022fd17e859`` |
| Seed Merkle root | ``8bc346bcb57029fa4d04c4d4185adcfe277097b657e737f83a6ab59f4d9acc72`` |

## Per-arm pass rates

| Arm | 90B pass rate | Diff vs A0 | Diff vs A1 | W99 11B (same slice) |
|---|---:|---:|---:|---:|
| A0_text | **46.67 %** (14 / 30) | — | — | 36.67 % (10 / 30) |
| A1_vlm K=5 | **76.67 %** (23 / 30) | +30.00 pp | — | 93.33 % (28 / 30) |
| B2 (direct-vision final-turn) | **73.33 %** (22 / 30) | +26.67 pp | **−3.33 pp** | 100.00 % (30 / 30) |

Cross-scale shift on B − A1: **−10.00 pp** (from W99-11B
+6.67 pp to W100-90B −3.33 pp).  This is steeper than W96-C
C1's 11B → 90B shift (+0.00 → +13.33 pp; opposite sign and
non-mechanism-driven) and similar in magnitude to W96-A's MathVista
cross-scale shift on B − A1 (W95-B0 11B Phase 3 +3.67 pp → W96-A
90B Phase 3 −5.00 pp = −8.67 pp).

## AddrW100 NIM-free pre-flight probes (PASSed before NIM call)

* **AddrW100-B2-P5 — cross-scale rescue-prior stability**: PASS.
  W99 11B B2 per-problem: n = 30; B2 PASS = 30; final-VLM
  invoked = 3; final-VLM rescued = 3.  W96-D 90B residual
  headroom 20.51 pp ⇒ expected unique-A1-rescues at 90B ≥ 6
  (threshold ≥ 3).
  *Interpretation*: the probe correctly predicted that 90B has
  more A1-only rescue room than 11B (residual 20.51 pp vs
  6.67 pp).  What the probe could NOT predict NIM-free is
  whether B2's text-solver chain + final-VLM could exploit that
  extra room.  The empirical answer is **no**.

## Pre-committed Phase 2 gates (W95 9-gate shape; byte-identical to W99)

| Gate | Verdict | Detail |
|---|---|---|
| 1 — slice pre-committed | **PASS** | 30 pids; slice SHA matches W97 / W98 / W99 exactly |
| 2 — A1 < 90 % | **PASS** | A1 @ K=5 = **76.67 %** (cleanly NOT saturated; matches W96-D residual prediction of ~ 79.49 %) |
| 3 — B strictly beats A1 | **FAIL** | B2 (73.33 %) < A1 (76.67 %) |
| 4 — Margin B − A1 ≥ +5 pp | **FAIL** | B2 − A1 = **−3.33 pp** |
| 5 — Margin B − A0 ≥ +5 pp | **PASS** | B2 − A0 = +26.67 pp (image is still load-bearing vs text-only floor) |
| 6 — Per-problem B ≥ A1 on ≥ 16 / 30 | **PASS** | B2 ≥ A1 on **27 / 30** problems (90 %) |
| 7 — Budget accounting exact | **PASS** | 1 + 5 + 5 = 11 calls / problem |
| 8 — Audit chain present | **PASS** | bench + seed Merkle roots recorded |
| 9 — Executor stays clean | **PASS** | Every arm routes through ``evaluate_realworldqa_answer_v1`` |

**7 of 9 gates PASS; gates 3 + 4 FAIL.**

Gate 2 PASSes cleanly at 90B — the W99 slice-saturation cap
does NOT apply at 90B (A1 = 76.67 % is well below 90 %).  Option A
of ``RUNBOOK_W99`` (treat B − A1 as discriminator under
saturation) is therefore NOT applicable; the −3.33 pp margin is
a *clean* fail.

## Mechanism-load-bearingness sub-gates (B2; new W100)

| Sub-gate | Verdict | Detail |
|---|---|---|
| MLB-1 — Final-VLM invocation rate ≤ 50 % | **PASS** | 9 / 30 = **30.00 %** of problems (3x the 11B rate of 10.00 %; still below 50 % ceiling) |
| MLB-2 — Final-VLM rescue rate ≥ 33 % | **FAIL** | 1 / 9 = **11.11 %** (vs 11B's 3 / 3 = 100.00 %) |

The MLB-2 FAIL is the key mechanism-level finding.  At 11B, the
final-VLM was a *targeted rescue* — invoked rarely and PASSing
when invoked.  At 90B, the final-VLM is invoked 3x more often
AND PASSes 9x less often per invocation.  **The mechanism is no
longer load-bearing.**

## Structural verdict

Per ``docs/RUNBOOK_W100.md`` decision logic:

* Gates 3 + 4 FAIL with A1 NOT saturated ⇒ no Option-A relief.
* MLB-2 FAIL ⇒ even if gate 4 had narrowly cleared, the
  mechanism would be classified non-load-bearing per the
  W96-C C1 precedent.

**Structural verdict: ``PHASE_2_FAIL`` (clean cross-scale collapse).**

## Cross-scale per-problem mining vs W99 11B (same slice; same candidate)

22 / 30 both-pass; **0 new wins at 90B vs 11B; 8 new losses; 0
neither-pass**.

### The 8 problems B2 LOST at 90B that B2 WON at 11B

| pid | qt | A0 90B | A1 90B | Final-VLM 11B inv / res | Final-VLM 90B inv / res |
|---|---|:---:|:---:|:---:|:---:|
| `rwqa_test_000013` | multi_choice_letter | PASS | FAIL | not invoked / — | INVOKED / FAIL |
| `rwqa_test_000111` | multi_choice_letter | FAIL | PASS | not invoked / — | INVOKED / FAIL |
| `rwqa_test_000155` | multi_choice_letter | FAIL | FAIL | not invoked / — | INVOKED / FAIL |
| `rwqa_test_000223` | numeric | PASS | FAIL | not invoked / — | INVOKED / FAIL |
| `rwqa_test_000533` | multi_choice_letter | PASS | FAIL | not invoked / — | INVOKED / FAIL |
| `rwqa_test_000615` | yes_no | PASS | PASS | **invoked / rescued** | INVOKED / FAIL |
| `rwqa_test_000713` | numeric | PASS | PASS | not invoked / — | INVOKED / FAIL |
| `rwqa_test_000718` | yes_no | FAIL | FAIL | not invoked / — | INVOKED / FAIL |

Patterns:

* **All 8 regressions fall through to the final-VLM at 90B.**
  Text-solver chain failed on every one of them.  Compare: at
  11B, 7 of these 8 were solved by text-solver short-circuit
  (no final-VLM invocation); only `000615` needed the final-VLM
  (and the 11B final-VLM rescued it).
* **The final-VLM rescues only 1 / 9 at 90B.**  The one rescue
  was `000246` (a multi-choice problem A1 also PASSed).  On the
  8 regression problems, the final-VLM FAILed every one.
* **A1 covers 3 / 8 of B2's losses** (`000111`, `000615`,
  `000713`).  So A1 alone would have done strictly better than
  B2 on those three.  On the other 5 losses, neither B2 nor A1
  PASSes — these are slice-cell limits both arms hit.
* **No new B2 wins vs 11B.**  The 22 / 30 both-pass set is
  exactly the W99 11B PASS set minus the 8 regression pids;
  B2 produced no 90B-specific gains.

### Question-type breakdown of the 8 regressions

* multi-choice (5): `000013`, `000111`, `000155`, `000533`,
  `000113`-class (none) — i.e., the multi-choice-extraction
  cluster that D2-B0 + B5 explicitly target.
* numeric (2): `000223`, `000713`.
* yes_no (2): `000615`, `000718`.

The regression cluster is **not concentrated in a single
question type**; B2's degradation at 90B is broad, not specific
to one failure surface.  This is consistent with the
mechanism-collapse interpretation: the final-VLM at 90B is
less reliable across the board, not just on a narrow cluster.

## Honest reading

### What B2 90B FAIL proves

* **The image-at-decision-boundary mechanism does NOT
  generalize cross-scale on RealWorldQA**.  The W99 11B PASS
  was real but slice-and-scale-bound.
* **The text-solver chain at 90B is LESS reliable than at
  11B** in the rescue-pool sense: more problems fall through
  to the final-VLM (3 / 30 → 9 / 30).  Counter-intuitively, a
  stronger underlying VLM produces more text-solver dropouts
  (likely because A1 at 90B is more confident on more cells
  and the text-solver template, designed for 11B's failure
  pool, doesn't gracefully scale up).
* **The final-VLM at 90B is LESS effective per invocation**
  (1 / 9 = 11.11 % vs 11B's 3 / 3 = 100 %).  The mechanism is
  empirically non-load-bearing at 90B.
* **This is the W96-C C1 cross-scale-collapse pattern**: a
  mechanism that looks load-bearing at one scale produces
  variance-driven or anti-load-bearing behaviour at the other.
  Both cases pre-committed mechanism sub-gates that caught the
  collapse cleanly.

### What B2 90B FAIL does NOT erase

* **The W99 11B B2 PASS carry-forward stands**.
  ``W99-L-REALWORLDQA-B2-DIRECT-VISION-FINAL-TURN-PHASE2-11B-STRUCTURAL-PASS-SLICE-SATURATION-CAP``
  is unchanged.  The 11B structural PASS (100 % on 30 / 30;
  final-VLM rescued 3 / 3 of the W97 unique-A1-rescue cluster)
  is real — it just doesn't generalize cross-scale.
* The W95-B0 family REPAIR (W99 finding that image-at-decision-
  boundary clears the +5 pp Phase 2 bar at 11B) is unchanged
  AT 11B.  The qualifier "AT 11B" is now load-bearing.

### What B2 90B FAIL DOES warrant

* **`COO-9` (second code benchmark) PROMOTED to lead path**
  per ``docs/RUNBOOK_W100.md`` Part H.
* The cross-modal RealWorldQA arc is **frozen at 11B** in the
  W100 frontier audit.  Future RealWorldQA work requires a
  structurally new mechanism, not another tweak to the W95-B0
  family.
* Phase 3 retirement bench NOT launched.  W100 closes Phase
  2 cross-scale confirmation with FAIL.

## Cross-scale rule application

Per ``docs/RUNBOOK_W100.md``:

* Cross-scale confirmation at 90B FAILed.
* Phase 3 NOT entitled (requires BOTH 11B and 90B Phase 2 PASS;
  90B FAILed).
* No Option-A relief (A1 not saturated at 76.67 %).
* MLB-2 FAIL classifies this as a mechanism-non-load-bearing
  FAIL regardless of gate-4 margin.

## Anti-cheat (all carry-forward held)

* Both parquet shards SHA-anchored at pilot start.
* Slice pre-committed BEFORE any NIM call (seed 96_504_002 +
  30 pids; slice SHA = same as W97 / W98 / W99).
* Same VLM model on every arm (A0 / A1 / B2-reader / B2-text-
  solver-text / B2-final-VLM all use
  ``meta/llama-3.2-90b-vision-instruct``).
* Same K=5 byte-exact budget on A1 and B2.  Text-solver short-
  circuit pads with text-solver retries on same prompt to
  maintain budget parity.
* Executor truth = ``evaluate_realworldqa_answer_v1``.
* No selective retries; no LLM judge.
* Per-call sidecars + per-seed Merkle + bench Merkle written.

## Stable boundary preservation

* ``coordpy.__version__`` unchanged at ``0.5.20``.
* ``coordpy.SDK_VERSION`` unchanged at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* ``coordpy.realworldqa_bench_v3`` (B2; built W98) unchanged.

## Carry-forwards

### Added

* **``W100-L-REALWORLDQA-B2-DIRECT-VISION-FINAL-TURN-PHASE2-90B-CAP``**
  — B2 (direct-vision final-turn answerer; UNCHANGED from
  W98 / W99) FAILs cross-scale confirmation at 90B on the
  96_504_002 / 30-problem slice.  A0 = 46.67 %, A1 @ K=5 =
  76.67 %, **B2 = 73.33 %; B2 − A1 = −3.33 pp**.  7 / 9 gates
  PASS (gates 3 + 4 FAIL with A1 NOT saturated; no Option-A
  relief).  Mechanism-load-bearingness sub-gate MLB-2 FAILs
  (final-VLM rescue rate 1 / 9 = 11.11 %; threshold ≥ 33.33 %);
  MLB-1 PASSes (invocation rate 30 %; threshold ≤ 50 %).
  Per-problem mining: 22 / 30 both-pass with 11B; 0 new wins;
  8 new losses spanning multi-choice (5) + numeric (2) +
  yes_no (2).  Cross-scale collapse pattern matches W96-C C1
  (single-scale mechanism not load-bearing).
* **``W100-L-REALWORLDQA-B2-CROSS-SCALE-COLLAPSE-MECHANISM-NON-LOAD-BEARING-AT-90B-CAP``**
  — The image-at-decision-boundary mechanism is empirically
  non-load-bearing at 90B even though it WAS load-bearing at
  11B.  Future work on RealWorldQA with this mechanism is
  restricted to the 11B regime; cross-scale generalisation
  requires a structurally new mechanism.

### Retired

**None.**  W89 70B-HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.  The
W99 11B B2 carry-forward STANDS unchanged.

## Discipline status

Preflight-first + cross-scale + multi-candidate-tournament-
then-confirm discipline validated TEN consecutive times: W93 /
W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 / **W100**.

W100 is the second case where the discipline caught a 90B
cross-scale collapse cleanly via pre-committed mechanism sub-
gates (the first was W96-C C1 verifier; W100 B2 is the second
empirically distinct collapse pattern at the same scale).  The
W93 preflight-first + W96-C cross-scale rule + W100 MLB sub-
gates together form a three-layer defence against false
retirement-level claims.

## Next move

Per ``docs/RUNBOOK_W100.md`` Part H (code-pivot contingency,
pre-committed BEFORE this pilot):

1. **PROMOTE ``COO-9`` to lead path.**  Cross-modal RealWorldQA
   arc is structurally restricted to the 11B regime; the
   programme needs a different battlefield to make further
   structural progress on multi-agent context.
2. **W100 stops at the cross-scale-confirmation deliverable.**
   Phase 3 retirement bench NOT launched.  B5 90B Phase 2 still
   runs as the **ceiling-reference** sub-deliverable per the
   user's W100 brief Part F.
3. **W101 (forward-looking)**: ``COO-9`` runbook + corpus
   selection from {MBPP+, HumanEval+, APPS, LiveCodeBench,
   SWE-bench-lite}.  Pre-commit selection criteria + preflight
   composite BEFORE any expensive run, per the W93 discipline.

## The honest claim W100 B2 90B Phase 2 (FAIL) earns

**On the 96_504_002 / 30-problem slice of ``lmms-lab/RealWorldQA``
test at ``meta/llama-3.2-90b-vision-instruct``, the W100 B2
candidate (direct-vision final-turn answerer; UNCHANGED from
W99) achieves 73.33 % pass rate (22 / 30) and B2 − A1 = −3.33 pp
with A1 cleanly NOT saturated at 76.67 %.  Gates 3 + 4 FAIL with
no Option-A relief.  Mechanism-load-bearingness sub-gate MLB-2
FAILs (final-VLM rescue rate 1 / 9 = 11.11 %; vs 11B's
3 / 3 = 100 %); MLB-1 PASSes.  Per-problem mining: 22 / 30 both-
pass with 11B; 0 new wins at 90B; 8 new losses spanning multi-
choice (5) + numeric (2) + yes_no (2), all falling through to
the final-VLM which then fails 8 / 8 of them.  The image-at-
decision-boundary mechanism is empirically NON-LOAD-BEARING AT
90B even though it WAS load-bearing at 11B.  This is the W96-C
C1 cross-scale-collapse pattern at a structurally distinct
mechanism.  Per ``docs/RUNBOOK_W100.md`` Part H code-pivot
contingency (pre-committed BEFORE this pilot), the verdict
triggers PROMOTION OF ``COO-9`` (second code benchmark) TO LEAD
PATH and FREEZES the cross-modal RealWorldQA arc at 11B in the
W100 frontier audit.  The W99 11B B2 PASS carry-forward
``W99-L-REALWORLDQA-B2-DIRECT-VISION-FINAL-TURN-PHASE2-11B-STRUCTURAL-PASS-SLICE-SATURATION-CAP``
STANDS unchanged; the 90B FAIL adds two new W100 carry-forwards
but does NOT erase the 11B truth.  No retirement.  No version
bump.  Discipline validation #10 (preflight-first + cross-scale
+ MLB sub-gate + multi-candidate-tournament-then-confirm).**
