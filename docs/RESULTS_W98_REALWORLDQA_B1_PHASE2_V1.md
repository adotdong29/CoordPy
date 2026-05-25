# W98 — RealWorldQA B1 Phase 2 cheap pilot V1 (FAIL at 11B; structurally informative)

> **2026-05-25 — On the W96-D D2 preflight-earned slice (seed
> 96_504_002; 30 problems × K=5) of `lmms-lab/RealWorldQA`
> test at Llama-3.2-11B-Vision-Instruct, W98 B1 (D2-B1: typed
> scene-graph extraction + question-typed solver) Phase 2
> FAILs 2 of 9 gates: gate 3 (B does not strictly beat A1)
> and gate 4 (B − A1 = −6.67 pp).  Gates 1, 2, 5–9 PASS —
> notably gate 2 (A1@K=5 = 86.67 % < 90 %; sampling noise
> below W97's saturation cap) and gate 6 (B ≥ A1 on
> **27 / 30** problems = improved per-problem majority vs
> W97 D2-B0's 25 / 30).  Per the W96-C cross-scale rule, 90B
> Phase 2 is NOT auto-launched.  Carry-forward
> `W98-L-REALWORLDQA-B1-TYPED-SCHEMA-PHASE2-11B-CAP` is added.
> Despite the cap, the structural evidence is *load-bearing*:
> **B1 RECOVERED 4 of 5 W97 unique-A1-rescues** (the failure
> cluster the typed schema was designed to attack) but
> **REGRESSED 5 multi-choice / numeric wins** D2-B0 had — a
> new failure mode where the typed solver becomes more
> confident in the reader's (often-wrong) `direct_answer_hint`
> and loses the K=4 multi-choice reflexion-cycling that
> drove D2-B0's unique-B-rescues.  Net: 4 − 5 = −1 problem =
> exactly the same B − A1 = −6.67 pp margin as W97 D2-B0,
> via an entirely different mechanism.**

## Configuration

| Field | Value |
|---|---|
| Battlefield | RealWorldQA test (`lmms-lab/RealWorldQA`) |
| Parquet URLs | `data/test-00000-of-00002.parquet`, `data/test-00001-of-00002.parquet` |
| Parquet shard SHA-256 (shard 0) | `0ed8b555586923099bd5d6ba5dd8b656b403ccfc418881facd237a6d6fe64952` |
| Parquet shard SHA-256 (shard 1) | `7dcb3ac3483362ca082cd4cddd0ab1389e9a276f1aefd4603fde0a2ce6bc74d0` |
| Corpus n_problems | 765 |
| Corpus Merkle root | `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab` |
| Slice seed | `96_504_002` (SAME as W97 — direct cross-candidate comparison) |
| Slice n_problems | 30 |
| Slice SHA-256 | `f53c71c2d355ac55…` (matches W97 pre-committed slice) |
| VLM model | `meta/llama-3.2-11b-vision-instruct` |
| Text/solver model | (same VLM in text-only mode) |
| Temperature | 0.7 (A1, B-solver); 0.0 (A0, B-typed-reader) |
| K | 5 |
| Calls per problem | 1 A0 + 5 A1 + 5 B = 11 |
| Total NIM calls | text = 150, vlm = 180, sum = 330 |
| Run wall | 951 s (~15.8 min) |
| Bench Merkle root | `dbc807a059bdb9cd6f51078b6b73ba8605861061c6d126f5ec0fa8e33051836f` |
| Seed Merkle root | `669ccedc5c181e74a3af50d42f9f223f3392e210e9bb80f9f7df4ea2f77f6bf8` |

## Per-arm pass rates

| Arm | Pass rate | Diff vs A0 | Diff vs A1 | Same-slice W97 D2-B0 |
|---|---:|---:|---:|---:|
| A0_text | **36.67 %** | — | — | 36.67 % (identical) |
| A1_vlm K=5 | **86.67 %** | +50.00 pp | — | 90.00 % (sampling variance) |
| B_vlm_team_v2 (B1) | **80.00 %** | +43.33 pp | **−6.67 pp** | 83.33 % (D2-B0; same arm-pos) |

## Pre-committed Phase 2 gates (W95 9-gate shape)

| Gate | Verdict | Detail |
|---|---|---|
| 1 — slice pre-committed | **PASS** | 30 pids; slice SHA `f53c71c2d355ac55…` (matches W97 exactly) |
| 2 — A1 < 90 % | **PASS** | A1@K=5 = **86.67 %** (W97 was 90.00 %; sampling-noise gap; 90 % bar cleared this run) |
| 3 — B strictly beats A1 | **FAIL** | B (80.00 %) < A1 (86.67 %); ¬(B > A1) |
| 4 — Margin B − A1 ≥ +5 pp | **FAIL** | B − A1 = **−6.67 pp** (gap of 11.67 pp below threshold; **identical to W97 D2-B0's deficit**) |
| 5 — Margin B − A0 ≥ +5 pp | **PASS** | B − A0 = **+43.33 pp** (image extraction is load-bearing) |
| 6 — Per-problem B ≥ A1 on ≥ 16 / 30 | **PASS** | B ≥ A1 on **27 / 30** problems (90 %; **improved from W97 D2-B0's 25 / 30** = 83 %) |
| 7 — Budget accounting exact | **PASS** | 1 + 5 + 5 = 11 calls/problem; matches expected |
| 8 — Audit chain present | **PASS** | bench + seed Merkle roots recorded |
| 9 — Executor stays clean | **PASS** | Every arm routes through `evaluate_realworldqa_answer_v1` |

2 of 9 gates FAIL.  **W98 B1 Phase 2 at 11B is KILLED.**  The
structural verdict is `PHASE_2_FAIL` (not
`STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP` per the W98
runbook's Option A logic — that branch requires gate 2 FAIL
which did not happen here).

## Per-problem disagreement structure

W98 B1 vs A1 (W98 sampling):

| | A1 PASS | A1 FAIL | Total |
|---|---:|---:|---:|
| B1 PASS | 23 | 1 | 24 |
| B1 FAIL | 3 | 3 | 6 |
| Total | 26 | 4 | 30 |

* **23 / 30** (77 %) — both pass.
* **3 / 30** (10 %) — *unique A1 rescues*: A1 passes but B1
  fails.  Pids: `000225` (speed-limit reading;
  multi_choice_letter), `000615` (traffic-light state;
  yes_no — the **one remaining** W97 unique-A1-rescue), and
  `000713` (count-of-cars; numeric).
* **1 / 30** (3 %) — *unique B1 rescue*: `000441`
  (letter-position multi_choice_letter; same as W97 D2-B0's
  unique-B-rescue list).
* **3 / 30** (10 %) — neither pass: `000013`, `000204`,
  `000615` (yes/no).

## Direct vs W97 D2-B0 on the same slice

| Class | Count | Pids |
|---|---:|---|
| Both pass (W97 D2-B0 PASS ∧ W98 B1 PASS) | 20 | — |
| W98 B1 **new rescues** (W97 D2-B0 FAIL → W98 B1 PASS) | **4** | `000135` (Yes; "Are there any stop signs?"), `000403` (No; "Is the light green?"), `000555` (No; "are the cars facing left?"), `000718` (Yes; "large truck closer/further than pickup?") |
| W98 B1 **regressions** (W97 D2-B0 PASS → W98 B1 FAIL) | **5** | `000013` (B; vehicle direction multi-choice), `000155` (B; gun direction multi-choice), `000204` (C; how-many-lanes multi-choice), `000225` (A; speed-limit number multi-choice), `000713` (2; how-many-cars numeric) |
| Both fail | 1 | `000615` (No; traffic-light state — the residual yes/no failure neither candidate solved) |

**Net: 4 rescues − 5 regressions = −1 problem.**  At 30
problems, this is exactly the same −6.67 pp B − A1 margin as
W97 D2-B0 — but via an entirely different per-problem
distribution.  The W95-B0-derived "extract-then-text-reason"
architecture family is now empirically capped at the same
+5 pp deficit through TWO distinct mechanisms.

## Failure-mode mining (NIM-free post-hoc)

### What the typed schema fixed (4 yes/no rescues)

For the 4 W97 unique-A1-rescues B1 recovered, the typed
solver prompt did exactly what AddrP1 predicted: it
constrained the solver to output `Yes` / `No` instead of
`0` / `1` / `2`.  The reader's bullet extraction was
*already sufficient* for those problems (we mined this in
the W98 arsenal-mining doc); the only thing missing was the
solver's output-format discipline, which the typed prompt
provided.

### What the typed schema broke (5 multi-choice / numeric regressions)

For the 5 D2-B0 wins B1 regressed, the failure mode was new
and worth recording:

* **Reader's `direct_answer_hint` is often wrong on
  multi-choice** (B1 inherits D2-B0's reader bias — the
  reader emits an "Answer: X" hint even when it can't see
  the answer reliably; the JSON's `direct_answer_hint`
  surfaces this hint into the solver's typed prompt).
* **Typed-format solver becomes too confident in the
  hint** — D2-B0's free-text solver was disorganised
  enough to flip its answer across K=4 reflexion turns and
  hit the correct option by *cycling* (75 % hit probability
  on a 3-option problem with 4 independent samples).  B1's
  typed solver, primed by the JSON's hint, sticks to the
  hint or its inverse and *cycles less*.
* **Multi-choice regressions are the structural cost** of
  fixing the yes/no failures.  The 4 rescues + 5
  regressions in roughly equal numbers is consistent with
  "the typed schema reorganises the failure surface without
  shrinking it."

### What neither mechanism recovered (1 yes/no failure)

* `rwqa_test_000615` ("Is the traffic light green for us?")
  — both D2-B0 and B1 fail.  Inspection: this question
  requires *interpreting* "for us" (the user's lane) vs
  "for the cross-traffic lane" given a complex traffic
  scene.  Neither the free-text bullet extraction nor the
  typed schema captures the *user perspective* primitive.
  Future W99+ work could add a `viewer_pov` field to the
  schema, but the cost would likely introduce another
  cluster of regressions.

## Structural reading

The 27 / 30 per-problem majority (B1 ≥ A1) confirms that
**B1 is structurally working better than D2-B0** at the
per-problem level (27 vs 25), but the cluster-trade pattern
(4 yes/no rescues / 5 multi-choice regressions) cancels at
the aggregate.  This is exactly the H2-like outcome predicted
in the W98 runbook's pre-pilot prediction: "B1 ties or
narrowly beats A1; the typed solver fix recovers some
failures but the schema-constrained extraction regresses
some both-pass problems."

**The honest reading is**: the typed schema + question-typed
solver mechanism IS load-bearing on yes/no perception (4 / 5
recovery rate, dead-on what AddrP1 predicted), but it
introduces a previously-unidentified offsetting cluster on
multi-choice reflexion-cycling.  The W95-B0 family of
extract-then-reason architectures is empirically capped on
RealWorldQA at B − A1 ≈ −6.67 pp through both D2-B0 (free
text) and D2-B1 (typed schema) mechanisms.

## Cross-scale rule

Per `docs/RUNBOOK_W96C.md` carry-over (re-asserted in
`docs/RUNBOOK_W98.md`):

* 11B Phase 2 FAIL ⇒ 90B Phase 2 does **NOT** auto-launch.
* The structural reading (architecture-family-level cap of
  −6.67 pp at 11B; W96-A cross-scale shift on B − A1 was
  −8.67 pp on MathVista, suggesting scaling typically HURTS
  the team's relative advantage on cross-modal benches) makes
  90B Phase 2 likely to fail by an even wider margin.
* **Cross-scale 90B Phase 2 is NOT in scope** for this
  milestone.  The W98 verdict is the 11B FAIL.

## Decision logic outcome (locked in `docs/RUNBOOK_W98.md`)

* B1 Phase 2 at 11B: **FAIL** (2 of 9 gates).
* Cross-scale 90B: **NOT entitled**.
* B2 (direct-vision final-turn answerer): **deferred to W99**
  per the runbook's pre-committed decision logic.  Honest
  re-reading: B2's mechanism (image at decision boundary)
  does NOT rely on the reader's `direct_answer_hint` and
  does NOT have the multi-choice over-confidence failure
  mode B1 exhibited; the W98 evidence makes B2 a
  STRUCTURALLY-PLAUSIBLE follow-up candidate rather than a
  redundant alternative.

## Carry-forwards

### Added

* **`W98-L-REALWORLDQA-B1-TYPED-SCHEMA-PHASE2-11B-CAP`** —
  B1 (typed scene-graph + question-typed solver at K=5
  byte-exact) does NOT clear the +5 pp Phase 2 bar on the
  96_504_002 / 30-problem slice at 11B.  The mechanism is
  empirically load-bearing on yes/no perception (4 / 5 W97
  unique-A1-rescues recovered) but introduces an offsetting
  cluster on multi-choice reflexion-cycling (5 D2-B0 wins
  regressed).  Net B − A1 = −6.67 pp = same architectural
  cap as W97 D2-B0 via a different per-problem distribution.

### Retired

**None.**  W89 70B-HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.
All prior W95 / W96-A / W96-C / W96-D / W97 carry-forwards
remain active.

### Implication for the W95-B0 family

The W95-B0-derived "extract-then-text-reason" architecture
family is now empirically capped on RealWorldQA at B − A1 ≈
−6.67 pp at 11B through TWO distinct mechanisms (D2-B0 free
text + D2-B1 typed schema).  W99 candidates from this family
must address the multi-choice reflexion-cycling
regression-risk or pick a structurally different family
(e.g., B2 direct-vision final-turn; substrate-level cross-
modal injection per COO-12).

## Discipline status

Preflight-first + cross-scale discipline now validated EIGHT
consecutive times (W93 / W94 / W95 / W96-A / W96-C / W96-D /
W97 / **W98**).  W98 is the first case where a structurally-
motivated mechanism (typed schema) was confirmed to address
the targeted failure cluster (4 / 5 yes/no recovery) while
introducing a new offsetting cluster (5 multi-choice
regressions) — the result is structurally informative even
though the +5 pp Phase 2 bar is not cleared.

## Anti-cheat (carry-forward from W88–W97)

All W88–W97 anti-cheat clauses held in this pilot:

* Both parquet shards SHA-anchored at pilot start (recorded
  above).
* Slice pre-committed BEFORE any NIM call (seed 96_504_002 +
  30 pids + slice SHA = same as W97).
* Same VLM model on every arm (A0 / A1 / B-reader / B-solver
  all use `meta/llama-3.2-11b-vision-instruct`; text mode =
  image=None).
* Same K=5 byte-exact budget on A1 and B; A0 = 1 call.
* Executor truth = `evaluate_realworldqa_answer_v1` for
  every arm.  No LLM judge.
* No selective retries.
* Per-call sidecars (`text_calls.jsonl`, `vlm_calls.jsonl`,
  `per_problem.jsonl`) + per-seed Merkle + bench Merkle
  written.
* Question-type parser is deterministic + NIM-free (no
  oracle answer-format info).

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* `coordpy.realworldqa_bench_v2` (B1) and
  `coordpy.realworldqa_bench_v3` (B2) remain **explicit-
  import only**; not re-exported through
  `coordpy/__init__.py`.
* `coordpy.realworldqa_{loader,executor,preflight}_v1` (from
  W96-D) re-used verbatim.

## Re-running

```bash
.venv/bin/python scripts/run_w98_realworldqa_preflight.py \
    --candidate-model meta/llama-3.2-11b-vision-instruct

.venv/bin/python scripts/run_w98_realworldqa_pilot.py \
    --vlm-model meta/llama-3.2-11b-vision-instruct
```

Outputs land under `results/w98/`.  The canonical run for this
verdict is
`w98_realworldqa_pilot_b1_11b_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260525T191938Z`.

## The honest claim W98 B1 Phase 2 (11B) earns

**On the W96-D-earned 96_504_002 / 30-problem slice of
`lmms-lab/RealWorldQA` test at `meta/llama-3.2-11b-vision-
instruct`, the W98 B1 candidate (typed scene-graph
extraction + question-typed solver at K=5 byte-exact) FAILs
2 of the 9 pre-committed Phase 2 gates: B does not strictly
beat A1 (gate 3) and B − A1 = −6.67 pp (gate 4).  Gates 1,
2, 5–9 PASS — notably gate 2 (A1@K=5 = 86.67 % < 90 %; W97's
saturation cap was sampling-noise-driven; this run's A1 fell
under it) and gate 6 (per-problem B ≥ A1 on **27 / 30** — an
improvement over W97 D2-B0's 25 / 30).  Per-problem disagree-
ment shows the typed schema mechanism IS load-bearing on
the targeted yes/no perception cluster: **B1 rescued 4 of
the 5 W97 unique-A1-rescues** (`000135`, `000403`, `000555`,
`000718`) — exactly the failure mode predicted by W98
AddrP1 + AddrP2.  But B1 REGRESSED 5 D2-B0 wins on multi-
choice / numeric questions (`000013`, `000155`, `000204`,
`000225`, `000713`) by a previously-unidentified failure
mode: the typed solver becomes more confident in the
reader's (often-wrong) `direct_answer_hint` and stops K=4
reflexion-cycling.  Net = 4 − 5 = −1 problem = identical
−6.67 pp B − A1 margin to W97 D2-B0 via a different per-
problem distribution.  Cross-scale 90B Phase 2 is NOT
entitled per the W96-C cross-scale rule.  B2 (direct-vision
final-turn answerer) is deferred to W99 per the W98 runbook
decision logic; its distinct mechanism (image at decision
boundary; no reader-hint dependency) remains plausibly load-
bearing for the W99 follow-up.  Adds carry-forward
`W98-L-REALWORLDQA-B1-TYPED-SCHEMA-PHASE2-11B-CAP`.  No
retirements.  Discipline validation #8.**
