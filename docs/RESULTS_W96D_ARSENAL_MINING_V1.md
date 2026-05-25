# W96-D arsenal-mining inventory (pre-pilot, NIM-free)

> 2026-05-25.  Documentation-only inventory of reusable
> mechanisms in the CoordPy repo that could attack the W96-D
> battlefield (ChartQA primary, RealWorldQA backup) at K=5
> byte-exact budget.  No NIM calls.  Goal: ensure W96-D is
> hypothesis- and arsenal-driven, not vibes-driven; explicitly
> record what was considered and rejected so the W96-D candidate
> selection can be audited.

## Why this pass differs from W96-C arsenal mining

The W96-C arsenal-mining pass
(`docs/RESULTS_W96C_ARSENAL_MINING_V1.md`) inventoried mechanisms
that could attack the W96-A-identified failure mode on MathVista:
*math_solver / reflexion blindness to the image at 90B-Vision*.
W96-D is not an architecture refinement at MathVista; it is a
**battlefield pivot**.  The arsenal pass therefore reorients:
which mechanisms in the repo are load-bearing for the structural
features of ChartQA / RealWorldQA?

## Battlefield-specific structural features

### ChartQA (D1 lead)

ChartQA problems pair a chart image with a numeric or short-text
question.  Charts have explicit, recoverable structure:

* axes (x/y labels, units, scale, log-vs-linear);
* legend (color → series mapping);
* data values (tick labels, bar heights, line points, scatter
  coordinates);
* annotations (title, sub-title, captions, footnotes);
* derived relations (max/min, sum, percentage of total, trend).

The unified VLM K=5 baseline has to do all of this in one
forward pass per sample.  A team-decomposed pipeline can dedicate
one full VLM call to chart extraction (no question-coupling
pressure) and one full LLM call to math (no perception
pressure).

### RealWorldQA (D2 backup)

RealWorldQA problems pair a real-world driving / scene image with
a multi-choice or short-form question about spatial relations,
counting, or object identification.  Less explicit recoverable
structure than ChartQA; more entanglement between perception and
reasoning.  Team decomposition is plausibly less advantaged here
than on charts.

## Mechanism candidates surfaced by the mining pass

| Mechanism | Module / vintage | ChartQA leverage | RealWorldQA leverage | Selected for W96-D? |
|---|---|---|---|---|
| **VLM-Chart/Scene-Reader → Text-Solver** (W95-B0 port) | `coordpy.mathvista_bench_v1` (W95) | Direct port; B-reader extracts chart → bullet-list table; B-solver reads table → answer. Same K=5 byte-exact. | Direct port; B-reader extracts scene → bullet-list; B-solver reads bullets → answer. | **YES — D1-B0 lead candidate (ChartQA)**; **YES — D2-B0 backup (RealWorldQA)** |
| **Structured-Table-Extraction** (new W96-D mechanism on top of W95-B0) | derives from W95-B0 but with chart-specific extraction schema | High — charts ARE tables; lossless extraction is structurally feasible. The B-reader can emit explicit `axis_x_label`, `axis_y_label`, `series:[…]`, `data:[(x,y)…]` instead of a free-text bullet list. | Low — scenes are not tables; structured schema does not fit. | **YES — D1-B1 (ChartQA-only candidate)** if D1-B0 leaves residual |
| **VLM-Verifier-Final-Turn** (W96-C C1) | `coordpy.mathvista_bench_v2` | Considered — but **already empirically refuted on MathVista** at K=5 byte-exact (rescue rate 0/11 at 11B; 1/7 at 90B). | Considered — same evidence applies. | **NO — refuted by W96-C; documented but not selected as W96-D lead** |
| **Tool-Augmented Solver** (W96-C C2) | `coordpy.tool_call_substrate_v1` (W84) | Considered — relevant for numeric chart questions where extraction is correct but arithmetic over the extracted table is hard.  Requires explicit K=5 budget accounting (tool calls are not model calls but consume wall time). | Considered — less relevant; RealWorldQA is not arithmetic-heavy. | **DEFER — D1-B2 architectural backup (ChartQA-only)**; only attempt if D1-B0 passes preflight but Phase 2 narrows |
| **VLM-in-loop (image every turn)** | `coordpy.cross_modal_vlm_loop_bench_v1` (W90) | Considered — keeps image in every solver turn; trades budget against extraction depth. | Considered — keeps image in every turn; spatially-grounded reasoning may benefit. | **NO — empirically refuted at HumanEval-Visual; W96-C arsenal pass already deferred this; not selected as lead** |
| **VLM-Planner + Code-Implementer + VLM-Verifier** (W92) | `coordpy.cross_modal_role_specialized_bench_v1` (W92) | Considered — three-role split would dedicate two VLM calls (Planner + Verifier) at the cost of fewer solver attempts. | Considered — same. | **NO — W92 lost by −10.71 pp on HumanEval-Visual; same K=5 structural shape; not selected** |
| **Multi-modal patch payload** | `coordpy.multi_modal_payload_v1`, `coordpy.vision_substrate_v1` (W87) | Considered — would let solver re-index into chart without re-encoding. | Considered — would let solver re-index into scene without re-encoding. | **INFEASIBLE — requires VLM hidden-state access; NIM HTTPS API does not expose patch-level embeddings.  Same constraint as W96-C C3.** |
| **Adversarial consensus / repair** | `coordpy.adversarial_consensus_repair_v1` (W81) | Considered — useful only with parallel candidate solvers. | Considered — same. | **NO — V2's solver chain is sequential; no parallel branches at K=5** |
| **Failure-cluster mining (W93)** | `coordpy.failure_cluster_miner_v1` | Mining ChartQA / RealWorldQA failures has no prior W95-like sidecars to mine yet — would mine W95 / W96-A / W96-C MathVista sidecars for cross-benchmark hints. | Same. | **PARTIAL — used after a Phase 2 pilot lands; not used in preflight** |
| **Composed long-horizon multi-agent recovery** | `coordpy.composed_long_horizon_multi_agent_recovery_v1` (W83) | Considered — provides substrate-restore + consensus + integrity composition. | Considered — same. | **NO — not load-bearing for a 5-call K-budget cross-modal pilot; valuable for substrate-level work, not the W96-D shape** |
| **Multi-modal payload carry-state** | `coordpy.composed_multimodal_pipeline_v1` (W87) | Considered — composed substrate-level pipeline. | Considered — same. | **NO — substrate-level; W96-D is API-only NIM HTTPS** |

## Selected lead architecture — D1-B0 (ChartQA port of W95-B0)

D1-B0 reuses, **verbatim where possible** from W95-B0:

* `extract_candidate_answer_v1` shape — line-anchored final-answer
  extraction (same anti-cheat extractor for every arm).
* W95 capsule shapes (`ArmCallCapsuleV1`, `ArmOutcomeCapsuleV1`,
  `SeedReportV1`, `BenchReportV1`) — rename to ChartQA-specific
  names but keep field-for-field schema parity so the audit chain
  reuses W95's verifier without modification.
* W95 selection rule (first PASS short-circuits, else last
  candidate).
* W95 anti-cheat clauses (same VLM family on A1 and B-reader;
  same text-LM on A0 and B-solver; same K=5 byte-exact budget;
  same deterministic slice).

D1-B0 adapts:

* **B-reader prompt** → chart-specific extraction schema
  (axis labels with units; legend mapping; data values as
  `(x_label, value)` pairs or table cells; title).
* **B-solver prompt** → chart-table-aware (use the extracted
  table as ground truth; do not invent values).
* **A0 / A1 prompts** → chart-specific instruction to emit only
  the final answer (number or short string), no prose.

D1-B0 does NOT add new mechanisms over W95-B0; it is the most
conservative possible port and the cleanest reference.  The
arsenal-mining pass surfaces D1-B1 (structured-table extraction)
as the cheapest meaningful refinement to try IF the preflight
passes AND D1-B0 leaves residual at Phase 2.

## Conditional secondary candidate — D1-B1 (ChartQA structured-table extraction)

D1-B1 is **not implemented in this milestone**.  It is
documented here so the W96-D preflight + Phase 2 evidence can be
read with the right context if/when this becomes the next move.
The hypothesis: the W95-B0 "bullet list" extraction is lossy on
charts because models compress real chart structure into
free-form prose.  An explicit JSON / table schema for the
extraction (`{"axes": {"x": {...}, "y": {...}}, "series":
[{"label": "...", "points": [(x, y), ...]}], "annotations":
[...]}`) would force lossless extraction and let the solver
treat the chart as a queryable table.

D1-B1 stays in the K=5 byte-exact budget (1 VLM reader + 4 text
solver, identical to W95-B0).  Differences vs D1-B0 are purely
in prompt shape — no new modules.

## Rejected candidates and why

### Why not D1-C-loop (VLM-in-loop on ChartQA)

The W90 VLM-in-loop pattern keeps the image in every solver
turn.  On HumanEval-Visual K=5, W90 LOST by −1..−7 pp.  W93's
failure-cluster diagnosis showed the pattern is closer to A1
i.i.d. K=5 than a structurally distinct team.  On ChartQA, the
unified A1 K=5 baseline is already strong (single-shot ~83-85 %
for Llama-3.2-Vision-Instruct on the full test set per Meta's
release notes); the team needs explicit decomposition advantage,
not more i.i.d. sampling.  D1-C-loop is not selected.

### Why not D1-three-role (W92 port)

W92 LOST by −10.71 pp on HumanEval-Visual at K=5 byte-exact.
The structural shape (2 VLM + 3 code-LM) burns budget on a
second vision-grounded call that did not pay for itself.  On
ChartQA, the same shape would burn budget on a Verifier call at
T=0 (the W96-C C1 pattern at K=5 byte-exact failed to rescue at
both 11B and 90B on MathVista).  D1-three-role is not selected.

### Why not D1-substrate (multi-modal patch payload)

Requires VLM hidden-state access at per-patch granularity.  The
NIM HTTPS API does not expose patch embeddings; loading
Llama-3.2-Vision weights locally is out of scope for the W96-D
session (same constraint as W96-C C3).  This stays as a
documented future direction in `COO-12` (substrate-level
cross-modal injection as the hard alternative).

## Cheap-probe predictions before the W96-D preflight runs

The W96-D preflight (`scripts/run_w96d_chartqa_preflight.py`)
runs the W95 4-probe composite + W93 5-gate composite (re-used
verbatim with ChartQA-specific decomposition argument).  The
single new battlefield-specific decision is **P3's saturation
ceiling**.  The W95 default is 80 % (i.e., A1@K=5 must be
≤ 80 %).  Published Llama-3.2-Vision-Instruct single-shot scores
on the ChartQA test set:

| Model | ChartQA test (single-shot) | A1@K=5 estimate (corr=0.5) | P3 verdict at 80 % ceiling |
|---|---:|---:|---|
| Llama-3.2-11B-Vision-Instruct | ~83.4 % | ~91.7 % | **likely FAIL** |
| Llama-3.2-90B-Vision-Instruct | ~85.5 % | ~92.8 % | **likely FAIL** |

This is a high-prior cap.  ChartQA is already saturated on the
unified-VLM K=5 baseline at the published single-shot scores;
the structural residual a team could rescue may be too small to
clear the +5 pp Phase 2 bar even with a perfect extraction.

If P3 fails ChartQA at both scales, the W96-D battlefield pivots
to **RealWorldQA (D2)** per the pre-committed runbook contract.
The cross-battlefield discipline is the deliverable: we either
prove cheaply that ChartQA has room, or we kill ChartQA cheaply
and pivot.

## Honest framing

This is an *arsenal mining inventory*, not a benchmark result.
The selected D1-B0 candidate has not been validated empirically;
the W96-D preflight + (if earned) Phase 2 pilot is what will
decide.  The arsenal-mining pass is a discipline mechanism (the
user's instruction "use more of the arsenal, not less"),
recorded so the W96-D candidate selection can be audited
against the alternatives that were considered and rejected.
The Linear `COO-20` issue identified ChartQA and RealWorldQA as
the two battlefield candidates; the arsenal mining confirms D1
(ChartQA) as the lead with D1-B0 as the cheapest meaningful
candidate, and explicitly defers D1-B1 (structured-table
extraction) and D1-B2 (tool-augmented solver) as architectural
refinements that only earn the next move if D1-B0's preflight
+ Phase 2 evidence justifies them.
