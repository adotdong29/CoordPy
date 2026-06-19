# W97 — RealWorldQA D2-B0 Phase 2 cheap pilot V1 (FAIL at 11B; informative)

> **2026-05-25 — On the W96-D D2 preflight-earned slice (seed
> 96_504_002; 30 problems × K=5) of `lmms-lab/RealWorldQA` test
> at Llama-3.2-11B-Vision-Instruct, D2-B0 (W95-B0 scene-port)
> Phase 2 FAILs 3 of 9 gates: gate 2 (A1 saturated at 90%),
> gate 3 (B does not strictly beat A1), gate 4 (B − A1 = −6.67
> pp).  Gates 1 and 5–9 PASS.  Per the W96-C cross-scale rule,
> 90B Phase 2 is NOT auto-launched.  Carry-forward
> `W97-L-REALWORLDQA-D2-B0-PHASE2-11B-CAP` is added.  Despite
> the cap, the structural evidence is *informative*: B − A0 =
> +46.67 pp (image extraction is real signal); B ≥ A1 on 25/30
> problems (83 %); and the 5 unique-A1-rescues are exactly the
> *vision-bound yes/no perception* class the D2-B1 scene-graph
> design was sketched to attack.**

## Configuration

| Field | Value |
|---|---|
| Battlefield | RealWorldQA test (`lmms-lab/RealWorldQA`) |
| Parquet URLs | `data/test-00000-of-00002.parquet`, `data/test-00001-of-00002.parquet` |
| Parquet shard SHA-256 (shard 0) | `0ed8b555586923099bd5d6ba5dd8b656b403ccfc418881facd237a6d6fe64952` |
| Parquet shard SHA-256 (shard 1) | `7dcb3ac3483362ca082cd4cddd0ab1389e9a276f1aefd4603fde0a2ce6bc74d0` |
| Total parquet bytes | `678,342,154` |
| Corpus n_problems | `765` |
| Corpus Merkle root | `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab` |
| Slice seed | `96_504_002` |
| Slice n_problems | `30` |
| Slice SHA-256 | `f53c71c2d355ac55…` (full SHA in `pre_committed_slice.json`) |
| VLM model | `meta/llama-3.2-11b-vision-instruct` |
| Text/solver model | (same VLM in text-only mode) |
| Temperature | `0.7` (A1, B); `0.0` (A0, B-scene-reader) |
| K | `5` |
| Calls per problem | `1 A0 + 5 A1 + 5 B = 11` |
| Total NIM calls | text=`150`, vlm=`180`, sum=`330` |
| Run wall | `584 s` (~9.7 min) |
| Bench Merkle root | `c2454c7fee69fec7e5f80efc0dcec9a82b2c2f839186e1d92f0cf4877bb0e234` |
| Seed Merkle root | `96c3337354c6522e46c2ddba1ecb6d415d71e4a89de60fa91597b76c7e8efb55` |

## Per-arm pass rates

| Arm | Pass rate | Diff vs A0 | Diff vs A1 |
|---|---:|---:|---:|
| A0_text  | **36.67 %** | — | — |
| A1_vlm K=5 | **90.00 %** | +53.33 pp | — |
| B_vlm_team | **83.33 %** | +46.67 pp | **−6.67 pp** |

## Pre-committed Phase 2 gates (W95 9-gate shape)

| Gate | Verdict | Detail |
|---|---|---|
| 1 — slice pre-committed | **PASS** | 30 pids; slice SHA `f53c71c2d355ac55…` |
| 2 — A1 < 90 % | **FAIL** | A1@K=5 = **90.00 %** = ceiling; slice happened to be saturation-prone at 11B |
| 3 — B strictly beats A1 | **FAIL** | B (83.33 %) < A1 (90.00 %); ¬(B > A1) |
| 4 — Margin B − A1 ≥ +5 pp | **FAIL** | B − A1 = **−6.67 pp** (gap of 11.67 pp below threshold) |
| 5 — Margin B − A0 ≥ +5 pp | **PASS** | B − A0 = **+46.67 pp** (image extraction is load-bearing) |
| 6 — Per-problem B ≥ A1 on ≥ 16 / 30 | **PASS** | B ≥ A1 on **25 / 30** problems (83 %) |
| 7 — Budget accounting exact | **PASS** | 1 + 5 + 5 = 11 calls/problem; matches expected |
| 8 — Audit chain present | **PASS** | bench + seed Merkle roots recorded |
| 9 — Executor stays clean | **PASS** | Every arm routes through `evaluate_realworldqa_answer_v1` |

3 of 9 gates FAIL.  **W97 D2-B0 Phase 2 at 11B is KILLED.**

## Per-problem disagreement structure

The full 30-problem confusion table:

| | A1 PASS | A1 FAIL | Total |
|---|---:|---:|---:|
| B PASS | 22 | 3 | 25 |
| B FAIL | 5 | 0 | 5 |
| Total | 27 | 3 | 30 |

* **22 / 30** (73 %) — both pass; team architecture and unified
  VLM are interchangeable on these.
* **5 / 30** (17 %) — *unique A1 rescues*: A1 passes but B fails.
  This is the cost of structural specialization (the text-LM
  cannot re-see the image after the reader call).
* **3 / 30** (10 %) — *unique B rescues*: B passes but A1 fails.
  This is the team's structural advantage when extraction
  succeeds.
* **0 / 30** (0 %) — neither passes; A1 K=5 saturates this
  particular slice perfectly.

## Failure-mode mining (NIM-free post-hoc)

### Unique A1 rescues (5) — the structural weakness D2-B0 exposes

All five are *vision-bound yes/no or color/state perception*
where bullet-list extraction may drop the discriminating
signal:

| pid | Gold | Question (head) |
|---|---|---|
| `rwqa_test_000135` | `Yes` | "Are there any stop signs?" |
| `rwqa_test_000403` | `No` | "Is the light green?" |
| `rwqa_test_000555` | `No` | "are the cars facing left?" |
| `rwqa_test_000615` | `No` | "Is the traffic light green for us?" |
| `rwqa_test_000718` | `Yes` | "Is the large truck that's closest to us further from the camera than the pickup …" |

Pattern: **existence detection** (stop signs), **small-object
color/state** (traffic light state), **orientation** (cars
facing direction), and **depth ordering with conditional**
(truck closer / further than pickup).  A free-text bullet list
that says "objects: car, truck, pickup, traffic_light" can
easily lose the *state* (is the light green?), the *count* (do
any stop signs exist?), or the *depth ordering* (which is
closer?).  These are precisely the features a structured
scene-graph schema would mandate.

### Unique B rescues (3) — the structural advantage D2-B0 captures

All three are *multi-choice with extractable spatial /
positional answers* where structured extraction → text-LM
reasoning beats IID VLM sampling:

| pid | Gold | Question (head) |
|---|---|---|
| `rwqa_test_000013` | `B` | "Which direction is the vehicle directly in front of us traveling? A. Straight B. …" |
| `rwqa_test_000155` | `B` | "Which direction is the gun facing? A. left B. right" |
| `rwqa_test_000441` | `C` | "Where is the letter c relative in the entire word in this image?" |

These exercise *position*, *direction*, and *relative
positioning* — the cleanest "team decomposition wins" subset
of RealWorldQA.

## Structural reading

The 25 / 30 per-problem majority (B ≥ A1) confirms that **D2-B0
is structurally working**.  The −6.67 pp margin shortfall comes
from a *thin tail*: 5 unique-A1-rescues exceed 3 unique-B-
rescues by net −2 problems.  At 30 problems, net −2 = −6.67 pp.
At a larger sample size the variance would narrow; but the
structural asymmetry (vision-bound yes/no questions favour A1)
is the dominating signal.

This is the **predicted H2 outcome** from the W97 runbook's
pre-pilot prediction: "RealWorldQA's spatial / identification /
sign-reading subsets reward preserved visual signal more than
MathVista did, and the W95-B0 shape discards the image after
the reader call."

## Cross-scale rule

Per `docs/RUNBOOK_W96C.md` carry-over (re-asserted in
`docs/RUNBOOK_W97.md`):

* 11B Phase 2 FAIL ⇒ 90B Phase 2 does **NOT** auto-launch.
* The structural reading (A1 already at 90 % saturation at 11B
  on this slice; W96-D 90B preflight residual was only 20.51 pp)
  predicts that 90B's A1@K=5 on the same slice is likely ≥ 90 %,
  almost certainly failing gate 2 again — and B's relative
  advantage tends to *shrink* with VLM scale (W96-A's
  cross-scale shift = −8.67 pp on MathVista).
* **Cross-scale 90B Phase 2 is NOT in scope** for this
  milestone.  The W97 verdict is the 11B FAIL.

## Carry-forwards

### Added

* `W97-L-REALWORLDQA-D2-B0-PHASE2-11B-CAP` — D2-B0 (W95-B0
  scene-port at K=5 byte-exact) does NOT clear the +5 pp Phase
  2 bar on the 96_504_002 / 30-problem slice at 11B.  Structural
  signal exists (B − A0 = +46.67 pp; B ≥ A1 on 25 / 30) but the
  vision-bound yes/no / state / depth question subset drives a
  net −2 problem deficit.  Estimated 90B follow-on is
  structurally pre-empted by A1 saturation.

### Retired

**None.**  W89 70B-HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.  All
prior W95 / W96-A / W96-C / W96-D carry-forwards remain active.

## Discipline status

Preflight-first + cross-scale discipline now validated SEVEN
consecutive times (W93 / W94 / W95 / W96-A / W96-C / W96-D /
**W97**).  W97 is the second case where a preflight-earned
pilot at the cheap scale FAILed by structural asymmetry the
preflight could not catch (the first was W96-C C1 11B which
also FAILed cleanly).  In both cases the cross-scale rule
correctly blocked 90B escalation.

## Next-move options

Honest options for the next milestone, in order of structural
strength:

1. **W97-B (D2-B1 structured scene-graph extraction)** —
   highest expected leverage.  Replace the bullet-list
   extraction with the W97 arsenal-mining schema (objects with
   explicit position/depth/color tags; explicit existence
   flags; explicit state primitives; explicit depth ordering).
   Run its own preflight + cheap pilot.  Direct attack on the
   5 unique-A1-rescue failure cluster.
2. **W97-C (Question-typed routing)** — cheaper attack: classify
   each question as multi-choice / yes-no / numeric / free-form,
   and route multi-choice problems to D2-B0 (where it wins) and
   yes-no / vision-state problems to A1 (where unified VLM
   wins).  Same K=5 budget; structurally a "best-of-two"
   selector.  Caveat: routing decision logic must itself be
   anti-cheat (no oracle question-type info).
3. **`COO-9` (second code benchmark)** — promote, since the
   cross-modal lead is now formally three-benchmarks-capped
   (MathVista narrow miss; ChartQA saturated; RealWorldQA
   below margin).
4. **`COO-12` (substrate-level cross-modal injection)** — the
   hard alternative.  Requires hardware-local VLM and patch-
   embedding access.

The recommendation is **W97-B** as the immediate next move
because (a) the failure-mode mining specifies exactly the
schema features that would attack the unique-A1-rescue cluster,
(b) the D2-B1 design is already documented in
`docs/RESULTS_W97_ARSENAL_MINING_V1.md`, and (c) the cost is
another ~330 NIM calls per scale — the same cheap-pilot
envelope as W97-A.

## Anti-cheat (carry-forward from W88–W96-D)

All W88–W96-D anti-cheat clauses held in this pilot:

* Both parquet shards SHA-anchored at pilot start (recorded
  above).
* Slice pre-committed BEFORE any NIM call (seed 96_504_002 + 30
  pids + slice_sha256).
* Same VLM model on every arm (A0 / A1 / B all use
  `meta/llama-3.2-11b-vision-instruct`; text mode = image=None).
* Same K=5 byte-exact budget on A1 and B; A0 = 1 call.
* Executor truth = `evaluate_realworldqa_answer_v1` for every
  arm.  No LLM judge.
* No selective retries.
* Per-call sidecars (`text_calls.jsonl`, `vlm_calls.jsonl`,
  `per_problem.jsonl`) + per-seed Merkle + bench Merkle
  written.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New module `coordpy.realworldqa_bench_v1` is **explicit-import
  only**; not re-exported through `coordpy/__init__.py`.
* `coordpy.realworldqa_{loader,executor,preflight}_v1` (from
  W96-D) re-used verbatim.

## Re-running

```bash
.venv/bin/python scripts/run_w97_realworldqa_smoke.py \
  --candidate-model meta/llama-3.2-11b-vision-instruct

.venv/bin/python scripts/run_w97_realworldqa_pilot.py \
  --vlm-model meta/llama-3.2-11b-vision-instruct
```

Outputs land under `results/w97/realworldqa_pilot/`.  The
canonical run for this verdict is
`w97_realworldqa_pilot_11b_meta_llama-3.2-11b-vision-instruct__meta_llama-3.2-11b-vision-instruct_20260525T182409Z`.

## The honest claim W97 D2-B0 Phase 2 (11B) earns

**On the W96-D-earned 96_504_002 / 30-problem slice of
`lmms-lab/RealWorldQA` test at `meta/llama-3.2-11b-vision-
instruct`, the D2-B0 (W95-B0 scene-port at K=5 byte-exact)
candidate FAILs 3 of the 9 pre-committed Phase 2 gates: A1
saturates at 90 % (gate 2), B does not strictly beat A1 (gate
3), and B − A1 = −6.67 pp (gate 4).  The other 6 gates PASS,
including B − A0 = +46.67 pp (gate 5: image extraction is real
signal) and B ≥ A1 on 25 / 30 problems (gate 6).  Per-problem
disagreement analysis shows the −6.67 pp deficit is driven by
5 unique-A1-rescues on vision-bound yes/no / state / depth
questions vs 3 unique-B-rescues on multi-choice / spatial
questions — exactly the failure mode the D2-B1 structured
scene-graph design was sketched to attack.  Cross-scale 90B
Phase 2 is NOT entitled per the W96-C cross-scale rule; the
W96-D D2 preflight estimated A1@K=5 ≈ 79.49 % at 90B on the
full corpus, which on the saturation-prone slice would likely
exceed 90 % and fail gate 2 again.  Adds carry-forward
`W97-L-REALWORLDQA-D2-B0-PHASE2-11B-CAP`.  No retirements.
Discipline validation #7.**
