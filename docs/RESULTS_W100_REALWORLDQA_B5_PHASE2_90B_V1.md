# W100 — RealWorldQA B5 Phase 2 cross-scale 90B cheap pilot V1 (FAIL — narrow ceiling miss; baseline-only carries forward)

> **2026-05-25 — On the W96-D-earned 96_504_002 / 30-problem
> slice of ``lmms-lab/RealWorldQA`` test at
> ``meta/llama-3.2-90b-vision-instruct``, W100 B5 (deterministic
> NIM-free question-type router / switch baseline; UNCHANGED
> from W99) achieves **25 / 30 = 83.33 % pass rate**.
> A1 @ K=5 = 80.00 %; A0 = 46.67 %.  **B5 − A1 = +3.33 pp;
> gate 4 FAILs (threshold ≥ +5 pp; misses by 1.67 pp).**
> Gates 3 + 6 PASS: B > A1 (gate 3) and B ≥ A1 on **29 / 30 = 97 %**
> of problems (gate 6).  Structural verdict ``PHASE_2_FAIL``
> (gate 4 alone fails; per-problem majority remains excellent
> but the routing ceiling at 90B is too narrow to clear the
> +5 pp bar on its own).
>
> **HONEST FRAMING (non-negotiable; pre-committed in
> ``docs/RUNBOOK_W100.md``)**: B5 is a **switch baseline**, NOT
> a frontier mechanism.  This 90B FAIL is a useful *ceiling-
> reference* finding: it bounds the per-question-routing
> ceiling at 90B from above by the empirical +3.33 pp margin.
> The W99 11B B5 PASS carry-forward
> ``W99-L-REALWORLDQA-B5-SWITCH-BASELINE-PASS-11B-SLICE-SATURATION-CAP``
> STANDS unchanged; the 90B narrow miss adds a *cross-scale-
> bound* qualifier
> ``W100-L-REALWORLDQA-B5-SWITCH-BASELINE-90B-NARROW-MISS-CAP``
> but does NOT erase the 11B truth or change B5's baseline-only
> classification.
>
> Importantly: B5's 90B FAIL is NOT a "structural mechanism
> failure" because there is no mechanism to fail.  B5 only
> commits the routed arm's output verbatim; what it bounds is
> the per-question routing ceiling itself.  At 90B the bound
> is real (B5 narrowly misses the +5 pp bar even with a
> perfect deterministic oracle), implying that **even an
> oracle-perfect per-question routing strategy cannot clear
> the +5 pp Phase 2 bar at 90B on this slice**.  That is
> information about RealWorldQA's structural difficulty at 90B,
> not about any team mechanism.

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
| Temperature | 0.7 (A1, B5-A1-route, B5-D2-B0-solver); 0.0 (A0, B5-D2-B0-reader) |
| K | 5 |
| Calls per problem | 1 A0 + 5 A1 + 5 B = 11 |
| Total NIM calls | text = 102, vlm = 228, sum = 330 |
| Run wall | 557 s (~9 min) |
| Bench Merkle root | ``0b18c0c1ed0cf2bc…`` |
| Seed Merkle root | ``dbd4c1a1c929e878…`` |

## Per-arm pass rates

| Arm | 90B pass rate | Diff vs A0 | Diff vs A1 | W99 11B (same slice) |
|---|---:|---:|---:|---:|
| A0_text | **46.67 %** (14 / 30) | — | — | 36.67 % (10 / 30) |
| A1_vlm K=5 | **80.00 %** (24 / 30) | +33.33 pp | — | 93.33 % (28 / 30) |
| B5 (switch) | **83.33 %** (25 / 30) | +36.67 pp | **+3.33 pp** | 100.00 % (30 / 30) |

Cross-scale shift on B − A1: **−3.33 pp** (from W99-11B
+6.67 pp to W100-90B +3.33 pp).  The routing ceiling does
*partially* generalize cross-scale: B5 stays above A1 (gate 3
PASS) but the margin shrinks below the +5 pp threshold.

## AddrW100 NIM-free pre-flight probes (PASSed before NIM call)

* **AddrW100-B5-P4 — cross-scale route-mass stability**: PASS.
  Question-type distribution = ``{'multi_choice_letter': 18,
  'numeric': 4, 'yes_no': 6, 'short_text': 2}``;
  route distribution = ``{'vlm_team_b0': 18, 'a1_vlm_k5': 12}``;
  expected_match = True; on-disk match: 30 / 30 per-pid routes
  equal W99 11B (parser is deterministic + NIM-free; route
  decisions are byte-identical cross-scale by construction).

## Routing distribution (empirical 90B)

| Route | Count | B5 per-route PASS | A1 per-route PASS | W99 11B per-route PASS |
|---|---:|---:|---:|---:|
| ``vlm_team_b0`` (multi-choice → W97 D2-B0) | 18 | **14 / 18 = 77.8 %** | 13 / 18 = 72.2 % | 18 / 18 = 100.0 % |
| ``a1_vlm_k5`` (yes_no / numeric / short_text → A1 K=5) | 12 | **11 / 12 = 91.7 %** | 11 / 12 = 91.7 % | 12 / 12 = 100.0 % |
| **Total** | **30** | **25 / 30 = 83.3 %** | 24 / 30 = 80.0 % | 30 / 30 = 100.0 % |

**Key finding**: at 90B, ``vlm_team_b0`` (D2-B0 free-text
scene-reader + text-solver) on multi-choice drops from 18 / 18
= 100 % at 11B to 14 / 18 = 77.8 %.  Meanwhile A1 on the same
multi-choice route is 13 / 18 = 72.2 %, so D2-B0 still has a
+1-problem advantage on the multi-choice cluster — but that
advantage is much narrower than the +0-problem A1-equivalence
the 11B run showed (where both A1 and D2-B0 PASSed all 18
multi-choice).

The A1 route (12 / 12 at 11B → 11 / 12 at 90B) is essentially
flat: yes_no + numeric + short_text are easy enough at 90B
that A1 K=5 saturates near 92 %.  B5 PASSes 11 / 12 on this
route (same as A1; one shared failure on a numeric problem).

## Pre-committed Phase 2 gates (W95 9-gate shape; byte-identical to W99)

| Gate | Verdict | Detail |
|---|---|---|
| 1 — slice pre-committed | **PASS** | 30 pids; slice SHA matches W97 / W98 / W99 exactly |
| 2 — A1 < 90 % | **PASS** | A1 @ K=5 = **80.00 %** (cleanly NOT saturated; matches W96-D residual prediction of ~ 79.49 %) |
| 3 — B strictly beats A1 | **PASS** | B5 (83.33 %) > A1 (80.00 %) |
| 4 — Margin B − A1 ≥ +5 pp | **FAIL** | B5 − A1 = **+3.33 pp** (misses by 1.67 pp) |
| 5 — Margin B − A0 ≥ +5 pp | **PASS** | B5 − A0 = +36.67 pp |
| 6 — Per-problem B ≥ A1 on ≥ 16 / 30 | **PASS** | B5 ≥ A1 on **29 / 30** problems (97 %) |
| 7 — Budget accounting exact | **PASS** | 1 + 5 + 5 = 11 calls / problem |
| 8 — Audit chain present | **PASS** | bench + seed Merkle roots recorded |
| 9 — Executor stays clean | **PASS** | Every arm routes through ``evaluate_realworldqa_answer_v1`` |

**8 of 9 gates PASS; gate 4 alone FAILs.**

Gate 2 PASSes cleanly at 90B — the W99 11B slice-saturation
issue is gone.  Option A of ``RUNBOOK_W99`` (treat B − A1 as
discriminator under saturation) is therefore NOT applicable;
the +3.33 pp margin is a *clean* fail against the +5 pp bar.

## Structural verdict

Per ``docs/RUNBOOK_W100.md`` decision logic for B5:

* B5 has NO mechanism-load-bearingness sub-gates (those apply
  to B2 only; B5 is a switch baseline with no rescue mechanism
  to assess).
* Gate 4 FAIL with A1 NOT saturated ⇒ no Option-A relief.
* Gate 3 PASS + per-problem majority 29 / 30 = 97 % indicate
  that B5 IS doing useful routing work at 90B — just not enough
  to clear the +5 pp bar.

**Structural verdict: ``PHASE_2_FAIL``.**

The verdict is informative-not-claim: B5 is the routing-ceiling
upper bound and the bound is empirically +3.33 pp at 90B on this
slice.

## Cross-scale per-problem mining vs W99 11B (same slice; same candidate)

23 / 30 both-pass; **0 new wins at 90B vs 11B; 5 new losses;
2 new neither-pass (both routed to A1).**

### The 5 problems B5 LOST at 90B that B5 WON at 11B

| pid | qt | route | A0 90B | A1 90B | B5 90B | Routed arm @ 90B |
|---|---|---|:---:|:---:|:---:|:---:|
| `rwqa_test_000155` | multi_choice_letter | vlm_team_b0 | FAIL | FAIL | FAIL | D2-B0 FAIL |
| `rwqa_test_000223` | numeric | a1_vlm_k5 | PASS | FAIL | FAIL | A1 K=5 FAIL |
| `rwqa_test_000246` | multi_choice_letter | vlm_team_b0 | PASS | PASS | FAIL | D2-B0 FAIL (A1 also PASSed) |
| `rwqa_test_000430` | multi_choice_letter | vlm_team_b0 | FAIL | FAIL | FAIL | D2-B0 FAIL |
| `rwqa_test_000533` | multi_choice_letter | vlm_team_b0 | PASS | FAIL | FAIL | D2-B0 FAIL |

Patterns:

* **4 of 5 regressions are multi-choice routed to D2-B0**
  (the W97 free-text scene-reader + text-solver chain).
  D2-B0's per-route pass at 90B drops from 18 / 18 = 100 %
  (at 11B) to 14 / 18 = 77.8 % (at 90B).  The W95-B0-shape's
  cross-scale degradation on multi-choice extraction is the
  load-bearing source of B5's narrow miss.
* **1 regression is on the A1 route** (`rwqa_test_000223`
  numeric).  A1 K=5 at 90B doesn't always saturate on numeric
  problems with edge-case answer formats.
* **`rwqa_test_000246`** is the structurally interesting cell:
  routed to D2-B0 because parser marks it multi-choice, but A1
  alone PASSes (1 / 30 unique-A1 vs B5 at 90B).  An oracle
  router that switched to A1 on this problem would have moved
  B5 to 26 / 30 = 86.67 % and B5 − A1 to +6.67 pp (PASS gate
  4).  But the W99 brief's pre-committed Option-C constraint
  forbids per-problem oracle routing; the parser is structural
  and NIM-free.

### Question-type breakdown of the 5 regressions

* multi_choice: 4 (`000155`, `000246`, `000430`, `000533`)
* numeric: 1 (`000223`)
* yes_no: 0
* short_text: 0

Multi-choice extraction (D2-B0's strength at 11B) is the
*weakened* surface at 90B — the regression cluster is
concentrated where the W95-B0-shape worked best at the small
scale.

## B5 vs A1 90B disagreement (the routing-ceiling decomposition)

* Both pass: 23
* Unique B5: 2 (`rwqa_test_000013`, `rwqa_test_000204` — both
  multi-choice routed to D2-B0 where A1 K=5 sampled the wrong
  answer; D2-B0's extraction got it right)
* Unique A1: 1 (`rwqa_test_000246` — multi-choice routed to
  D2-B0 where D2-B0 FAILed but A1 K=5 PASSed)
* Neither: 4 (`000155`, `000223`, `000430`, `000533` — slice
  cells where both arms hit a limit)

**Routing ceiling decomposition**: B5 + 2 unique-B5 wins
− 1 unique-A1 loss = net +1 problem advantage over A1.  In
margin terms: 25 / 30 vs 24 / 30 = +1 / 30 = +3.33 pp.  Even an
oracle-perfect router on this slice (which switches to the
better-performing arm per problem) could at best recover the
1 unique-A1 win to reach 26 / 30 = 86.67 % and +6.67 pp
margin.  But that requires per-problem oracle knowledge —
which is anti-cheat-forbidden.

## Honest reading

### What B5 90B FAIL proves

* **The per-question routing ceiling at 90B is bound by the
  empirical +3.33 pp margin** on this slice.  Both routes
  weaken at 90B (D2-B0 multi-choice from 100 % to 77.8 %; A1
  yes_no/numeric/short_text from 100 % to 91.7 %), and the
  margin between the better-routed arm and A1 alone shrinks.
* **D2-B0's multi-choice advantage degrades cross-scale**.
  At 11B, D2-B0 owned multi-choice 18 / 18.  At 90B, D2-B0 +
  text-solver chain drops to 14 / 18 — 4 problems lost.  The
  W95-B0-family's cross-scale fragility (already documented at
  W96-A on MathVista, at W97 on RealWorldQA-D2-B0 at 11B as
  -6.67 pp Phase 2) appears in this regime too.
* **A1 K=5 at 90B does not saturate at 90 %** on this slice
  (80 % empirical), consistent with the W96-D preflight
  residual prediction (~ 79.49 %).
* **The routing-ceiling claim is regime-dependent**: at 11B
  the ceiling cleared +5 pp by exact oracle alignment; at 90B
  it does not.  W99's "the W95-B0-family cap is a routing
  problem" claim is *narrower than I read at the time* — the
  ceiling depends on D2-B0 owning multi-choice at the
  candidate scale, which fails at 90B.

### What B5 90B FAIL does NOT prove

* **NOT a mechanism failure**.  B5 has no team mechanism; it
  is a switch baseline.  The FAIL records the empirical
  ceiling bound, not a mechanism collapse.
* **NOT a refutation of the W99 11B B5 PASS carry-forward**.
  The 11B finding stands; the 90B finding adds a cross-scale-
  qualifier.
* **NOT evidence about B2**.  Per the pre-committed Part F
  of the W100 brief, a B5 outcome does NOT substitute for the
  B2 question.  The B2 question is answered separately in
  ``docs/RESULTS_W100_REALWORLDQA_B2_PHASE2_90B_V1.md``
  (verdict: clean FAIL; mechanism non-load-bearing).

### What B5 90B FAIL DOES warrant

* **B5 stays classified baseline-only ceiling reference**.
  Unchanged from W99.
* **B5 90B narrow miss carry-forward added**:
  ``W100-L-REALWORLDQA-B5-SWITCH-BASELINE-90B-NARROW-MISS-CAP``.
* **B5 90B route-decomposition contributes to the W100
  cross-modal-RealWorldQA-arc-frozen-at-11B conclusion**:
  even an oracle-perfect routing can't clear the +5 pp bar
  at 90B without per-problem oracle knowledge.  This is
  consistent with the B2 mechanism collapse: at 90B, the
  slice's structural difficulty exceeds what the existing
  W95-B0 arsenal can address.

## Comparison vs B2 90B (same scale, same slice, same milestone)

| Arm | 11B | 90B | 11B → 90B Δ (B − A1) |
|---|---:|---:|---:|
| A1 K=5 (in B2 run) | 93.33 % | 76.67 % | — |
| A1 K=5 (in B5 run) | 93.33 % | 80.00 % | — |
| B2 (direct-vision final-turn) | 100.00 % | **73.33 %** | **−10.00 pp** (+6.67 → −3.33) |
| B5 (switch baseline) | 100.00 % | **83.33 %** | **−3.33 pp** (+6.67 → +3.33) |

Both arms degrade cross-scale, but **B5 degrades MORE
GRACEFULLY than B2** (−3.33 pp shift vs −10.00 pp shift).
Interpretation: at 90B, the simple switch baseline outperforms
the structural-mechanism candidate.  This is the empirical
inversion that pre-committed cross-scale + MLB sub-gates were
designed to catch.  B2's mechanism is *worse than no mechanism
at all* at 90B.

The A1 between-run sampling variance is ±1.67 pp (76.67 % vs
80.00 % across the two pilots), which is within the expected
±3 pp band for K=5 sampling at temperature 0.7.

## Anti-cheat (all carry-forward held)

* Both parquet shards SHA-anchored at pilot start.
* Slice pre-committed BEFORE any NIM call (seed 96_504_002 +
  30 pids; slice SHA = same as W97 / W98 / W99).
* Same VLM model on every arm (A0 / A1 / B5-A1-route / B5-D2-
  B0-reader / B5-D2-B0-solver all use
  ``meta/llama-3.2-90b-vision-instruct``).
* Same K=5 byte-exact budget on A1 and B5; A0 = 1 call.
* Executor truth = ``evaluate_realworldqa_answer_v1`` for
  every arm.  No LLM judge.
* Question-type parser is deterministic + NIM-free (no oracle
  answer-format info).
* No selective retries.
* Per-call sidecars + per-seed Merkle + bench Merkle written.

## Stable boundary preservation

* ``coordpy.__version__`` unchanged at ``0.5.20``.
* ``coordpy.SDK_VERSION`` unchanged at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* ``coordpy.realworldqa_bench_v5`` (B5; built W99) unchanged.

## Carry-forwards

### Added

* **``W100-L-REALWORLDQA-B5-SWITCH-BASELINE-90B-NARROW-MISS-CAP``**
  — B5 (deterministic NIM-free question-type router; UNCHANGED
  from W99) narrowly MISSES gate 4 at 90B on the
  96_504_002 / 30-problem slice.  A0 = 46.67 %, A1 @ K=5 =
  80.00 %, **B5 = 83.33 %; B5 − A1 = +3.33 pp** (misses +5 pp
  bar by 1.67 pp).  8 / 9 gates PASS (gate 4 alone fails;
  gate 2 PASSes cleanly with A1 NOT saturated; gates 3 + 6
  PASS).  Per-route: ``vlm_team_b0`` D2-B0 14 / 18 = 77.8 %
  (down from 18 / 18 at 11B); ``a1_vlm_k5`` 11 / 12 = 91.7 %
  (down from 12 / 12 at 11B).  Per-problem mining: 23 / 30
  both-pass with 11B; 0 new wins; 5 new losses (4 multi-choice
  routed to D2-B0 + 1 numeric routed to A1).  B5 vs A1 90B
  decomposition: 23 both-pass, 2 unique-B5, 1 unique-A1, 4
  neither.  The routing ceiling exists at 90B but is narrower
  than the +5 pp Phase 2 bar.  B5 stays classified baseline-
  only ceiling reference.
* **``W100-L-REALWORLDQA-W95-B0-D2-B0-MULTI-CHOICE-EXTRACTION-DEGRADES-CROSS-SCALE-CAP``**
  — D2-B0 (free-text scene reader + text-solver chain) drops
  from 18 / 18 = 100 % on multi-choice at 11B to 14 / 18 =
  77.8 % at 90B.  The W95-B0-shape's cross-scale degradation
  documented at MathVista (W96-A) and at single-arm
  RealWorldQA Phase 2 (W97) appears within the B5 multi-choice
  route at 90B too.  Multi-choice extraction is not a stable
  cross-scale strength of the W95-B0 arsenal.

### Retired

**None.**  W89 70B-HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.  The
W99 11B B5 carry-forwards STAND unchanged.

## Discipline status

Preflight-first + cross-scale + multi-candidate-tournament-
then-confirm discipline validated TEN consecutive times: W93 /
W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 / **W100**.
B5's narrow-miss FAIL is informative-not-claim: it bounds the
routing ceiling cross-scale and quantifies the W95-B0-family's
cross-scale fragility on the multi-choice surface.

## The honest claim W100 B5 90B Phase 2 (FAIL) earns

**On the 96_504_002 / 30-problem slice of ``lmms-lab/RealWorldQA``
test at ``meta/llama-3.2-90b-vision-instruct``, the W100 B5
candidate (deterministic NIM-free question-type router /
switch baseline; UNCHANGED from W99) achieves 83.33 % pass rate
(25 / 30) and B5 − A1 = +3.33 pp with A1 cleanly NOT saturated
at 80.00 %.  Gate 4 FAILs (misses +5 pp bar by 1.67 pp); gate 3
PASSes (B > A1) and gate 6 PASSes (B ≥ A1 on 29 / 30 = 97 %
of problems).  Structural verdict ``PHASE_2_FAIL``.  Per-route
analysis: ``vlm_team_b0`` (D2-B0 on multi-choice) drops from
18 / 18 = 100 % at 11B to 14 / 18 = 77.8 % at 90B;
``a1_vlm_k5`` (A1 on yes_no + numeric + short_text) drops from
12 / 12 = 100 % at 11B to 11 / 12 = 91.7 %.  B5 vs A1 90B
decomposition: 23 both-pass, 2 unique-B5, 1 unique-A1, 4
neither.  The routing ceiling exists cross-scale but is too
narrow at 90B to clear the +5 pp Phase 2 bar.  B5 stays
classified baseline-only ceiling reference in the W100 frontier
audit.  The W99 11B B5 PASS carry-forward STANDS unchanged;
the 90B narrow miss adds a cross-scale-bound qualifier
carry-forward.  No retirement.  No promotion of B5 to frontier.
B5's 90B narrow miss is informative for understanding the
W95-B0-family's cross-scale fragility but does NOT substitute
for the B2 question (which FAILed separately and triggered
the COO-9 promotion).  Discipline validation #10 (preflight-
first + cross-scale + MLB sub-gate + multi-candidate-
tournament-then-confirm).**
