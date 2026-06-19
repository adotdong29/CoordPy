# W96-D — Battlefield pivot to ChartQA (D1) or RealWorldQA (D2) — runbook

> **Pre-commit contract for W96-D, locked 2026-05-25 BEFORE any
> NIM call for the W96-D candidate.**  W95 (MathVista, 11B) /
> W96-A (MathVista, 90B) / W96-C C1 (MathVista, 11B + 90B) all
> produced ≤ +3.67 pp retirement-grade or cross-scale-ambiguous
> results on the W95-B0-derived decomposition.  Three benches'
> worth of evidence (W95 Phase 3 +3.67 pp, W96-A Phase 3 −5.00 pp,
> W96-C C1 cross-scale 11B +0.00 pp / 90B +13.33 pp ambiguous)
> confirm that the MathVista battlefield + W95-B0 decomposition
> is empirically capped.  W96-D pivots to a different
> battlefield with a (presumed) different ceiling profile.
>
> **The W96-D deliverable in this milestone is the preflight
> verdict — earn or kill the next expensive run by cheap evidence,
> per the W93/W94/W95/W96-A/W96-C discipline.  No expensive run
> is in scope for this milestone unless the preflight + cheap
> pilot earn it.**
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.

## Linear

* `COO-20` (W96-D): Cross-modal battlefield pivot to ChartQA or
  RealWorldQA — High priority post-W96-C.
* Parent: `COO-6` (post-W96-A empirical frontier backlog).
* This runbook covers **D1 (ChartQA) as the lead** and **D2
  (RealWorldQA) as the backup**.  The mid-runbook pivot rule
  records when D1 dies cheaply and D2 takes over.

## Arsenal-mining anchor

`docs/RESULTS_W96D_ARSENAL_MINING_V1.md` records the mining
inventory.  Selected lead candidate is **D1-B0** — a conservative
port of the W95-B0 architecture (vlm_reader + math_solver +
executor-guided reflexion) to ChartQA, with chart-specific prompt
adaptations and no new mechanism over W95-B0.  Architectural
refinements (D1-B1 structured-table extraction; D1-B2
tool-augmented solver) are deferred to W96-D Phase 2 evidence and
documented but **not implemented in this milestone**.

## Hypothesis (locked 2026-05-25)

W95-B0's architectural cap at MathVista may be benchmark-specific:
the math_solver / reflexion chain is text-only, and at 90B-Vision
the unified A1 K=5 climbs into the residual on problems whose
extraction is lossy.  ChartQA has explicit recoverable chart
structure (axes, legend, data values, annotations); a chart-
extraction reader can produce a structurally-complete bullet list
that the text-solver can treat as ground truth.

Two structurally distinct outcomes are plausible:

* **H1 (D1-B0 wins on ChartQA):** the chart-specific decomposition
  closes the +5 pp bar at 11B / 90B at K=5 byte-exact.  Earns
  Phase 3 retirement bench (3 seeds × N problems).
* **H2 (ChartQA is saturated; D1 dies in preflight):** the
  published single-shot SOTA for Llama-3.2-Vision-Instruct on
  ChartQA test is ~83-85 %; A1@K=5 estimated ~91-93 %; the
  residual a team could rescue is < 10 pp; the +5 pp bar is
  out of reach.  W96-D pivots to **D2 (RealWorldQA)**.

**Pre-pilot prediction (locked 2026-05-25 BEFORE NIM):** "H2 is
slightly more likely a priori — Meta's release notes report ChartQA
test scores in the 83-86 % range for both 11B and 90B Vision-
Instruct, putting A1@K=5 deep into ceiling territory.  The cheap
preflight is the discriminator; W96-D may end as a documented
battlefield-pivot to D2 (RealWorldQA) with no NIM spend on D1
at all."

## Baselines (locked 2026-05-25)

Identical W95-shape (A0 / A1 / B) on whichever battlefield is
active.

* **A0** — text-only mode (image=None) of the VLM at T=0.0, K=1.
* **A1** — unified VLM mode at T=0.7, K=5.
* **B (D1-B0 for ChartQA; D2-B0 for RealWorldQA)** —
  `vlm_chart_reader + text_solver + executor-guided reflexion`
  at total K=5 byte-exact: 1 VLM reader (T=0.0) + 4 text solver
  turns (T=temperature, executor-guided reflexion).

D2-B0 differs from D1-B0 only in prompt content (scene
description vs chart extraction).  The architectural shape is
identical.

## Pre-committed W96-D preflight gates (NIM-free)

A W96-D NIM pilot launches IFF ALL of these hold for the chosen
battlefield.  All re-use the W95 / W93 preflight harness without
modification; only the decomposition argument and the candidate
benchmark text are W96-D-specific.

### W95 cheap probes (mandatory, NIM-free)

Re-run via the W96-D preflight script with the chosen-battlefield
loader + executor:

1. **P1 — corpus integrity**: parquet SHA-anchored; correct
   number of problems for the chosen split; every problem
   carries non-empty image + non-empty gold answer + valid
   schema fields.
2. **P2 — executor self-test on gold**: ≥ 98 % gold-as-prediction
   pass rate under the W96-D executor.  Below 98 % means the
   executor silently penalises every arm.
3. **P3 — A1 saturation / failure-residual estimate**: estimated
   A1@K=5 from published single-shot SOTA must leave ≥ 20 pp
   residual.  Default ceiling 80 %.
4. **P4 — decomposition argument**: written structural argument
   for B's decomposition (≥ 200 chars) AND corpus has enough
   problems where extraction + solve is plausibly distinct
   from unified VLM.

### W93 5-gate composite (mandatory)

The W93 harness's G1..G5 gates are re-run with the W96-D-
specific hypothesis, evidence check, ablation check, budget
accounting, and benchmark justification.  All 5 gates must pass
for composite verdict.

### W96-D-specific cheap probes

5. **R1 — battlefield SOTA mining**: the preflight records the
   published SOTA for the candidate VLM family on the chosen
   battlefield in an evidence sidecar.  Used by P3.
6. **R2 — A1-only residual upper bound**: estimated as
   `1.0 − A1@K=5_estimate`.  Provides an a-priori ceiling on
   what B can rescue at all.  Recorded for honesty; does not
   gate the pilot directly (P3's PASS implies this is ≥ 20 pp).

## Cross-battlefield pivot rule (locked 2026-05-25)

The W96-D battlefield order is **D1 first (ChartQA), D2 second
(RealWorldQA)**.

* If D1 preflight **PASSES composite** at the chosen scale →
  D1 is preflight-earned for a NIM smoke test + Phase 2 cheap
  pilot.  Do NOT touch D2 in this milestone — record D2 as
  "untested; D1 earned the next move".
* If D1 preflight **FAILS composite** at the chosen scale →
  document the D1 cap (`W96-L-CHARTQA-PREFLIGHT-*-CAP`); do NOT
  spend any NIM on D1; pivot to D2 cheaply.
* If D2 preflight **PASSES composite** at the chosen scale →
  D2 is preflight-earned for a NIM smoke test + Phase 2 cheap
  pilot.
* If both D1 and D2 preflight **FAIL composite** → W96-D
  produces TWO carry-forwards
  (`W96-L-CHARTQA-PREFLIGHT-*-CAP` +
  `W96-L-REALWORLDQA-PREFLIGHT-*-CAP`); the milestone deliverable
  is the documented double-pivot; W96 advances to either W96-D2
  (further battlefield scouting) or W96-C C2 (tool-augmented
  solver) per the Linear-recommended ordering.

The default candidate model for the D1 / D2 preflight is
`meta/llama-3.2-11b-vision-instruct` (cheaper scale).  The 90B
preflight is run only if (a) 11B preflight PASSes and we are
preparing the cross-scale entitlement check, OR (b) the W96-A
lesson explicitly requires it.

## NIM smoke test (only if a preflight composite PASSes)

1. One 1-token POST against the candidate model returns HTTP 200
   with a non-empty completion.  Recorded as the W96-D run
   sidecar's `kind=smoke_test` entry.
2. Optional dry-run of the B arm on 1 problem with a real NIM
   call shape (1 A0 + 5 A1 + 5 B = 11 NIM calls).  Confirms the
   B wiring works against the live endpoint.

**No NIM smoke test is in scope unless the preflight composite
PASSes for the chosen battlefield.**

## Phase 2 — cheap NIM pilot (only if NIM smoke PASSes)

### Pilot shape (locked 2026-05-25 BEFORE NIM)

* **Bench**: `coordpy.{chartqa,realworldqa}_bench_v1` (new
  modules per battlefield; not implemented in this milestone
  unless preflight earns them).
* **Model**: `meta/llama-3.2-{11b,90b}-vision-instruct` via NIM.
* **Slice**: **1 seed × 30 problems** — same shape as W95 / W96-A /
  W96-C Phase 2; the slice seed identity for W96-D is
  pre-committed in the loader as **96_504_001** (D1 default)
  and **96_504_002** (D2 default).
* **Budget**: K=5 calls per problem on A1 and B; A0 = 1 call.
* **Expected cost**: 30 × 11 = 330 NIM calls; ~20-30 min wall
  at 11B; ~60-90 min wall at 90B.

### Pre-committed Phase 2 pilot gates (W95 9-gate shape)

Phase 2 escalates to a Phase 3 pre-commit IFF ALL of these hold.
Gate texts are byte-identical to W95/W96-A/W96-C Phase 2; only
the arm name and slice seed change.

1. **Slice pre-committed**: 30 problems by the W96-D loader's
   deterministic slice with seed 96_504_001 (D1) or 96_504_002
   (D2) BEFORE any NIM call.  Slice SHA recorded.
2. **A1 < 90 %**: A1@K=5 pass rate on the 30-problem slice must
   stay below 90 %.
3. **B > A1**: `b_pass_rate > a1_pass_rate`.
4. **Margin ≥ +5 pp**: `b_pass_rate − a1_pass_rate ≥ 5.0 pp`.
5. **B > A0 by ≥ +5 pp**: image is load-bearing in B.
6. **Per-problem majority**: B ≥ A1 on ≥ 16 of 30 problems.
7. **Budget accounting exact**: 1 + 5 + 5 = 11 calls per
   problem.
8. **Audit chain re-derives**: per-call sidecars + per-seed
   Merkle + bench Merkle re-derive offline.
9. **Executor stays clean**: P2 re-run on the 30 slice problems
   at end-of-run → 100 % pass.

If any gate fails, **W96-D Phase 2 at the candidate scale is
KILLED.**  Cross-scale rule (next section) decides whether the
candidate is killed overall or escalates to the other scale.

## Cross-scale rule (W96-C carry-over, locked 2026-05-25)

W96-A taught that a +10 pp single-seed Phase 2 pilot can flip to
−5 pp at multi-seed retirement scale.  W96-C taught that a
candidate may PASS at one scale and FAIL at the other.  The
W96-D cross-scale rule (re-asserted from `docs/RUNBOOK_W96C.md`):

* **Phase 3 entitled** IFF Phase 2 PASS at BOTH 11B and 90B, OR
  Phase 2 PASS at 90B alone with a written justification.
* A Phase 2 PASS at 11B alone is **NOT sufficient** for Phase 3
  at 90B; the candidate gets a cross-scale Phase 2 at 90B first.
* A Phase 2 FAIL at 11B does **NOT auto-launch** a 90B Phase 2;
  the author pre-commits the 90B-Phase-2 decision in writing
  before any 90B NIM call.

## Phase 3 — full bench (NOT in this milestone)

If Phase 2 earns it, Phase 3 will require its own runbook
section locked BEFORE any Phase 3 NIM call, mirroring W96-A
Phase 3 in `docs/RUNBOOK_W96A.md`:

* 3 seeds × 100 problems × K=5.
* W88 6-bar retirement shape (same as W95 / W96-A).
* Audit-chain re-derivation requirement.

**No Phase 3 will be launched in this milestone.**  W96-D's job
is to earn or kill the next expensive run; the Phase 3 launch
decision and runbook are explicitly out of scope until Phase 2
evidence exists at both scales (or at 90B with written
justification).

## Anti-cheat (carry-forward from W88–W96-C)

All W88–W96-C anti-cheat clauses carry forward verbatim.
W96-D-specific additions:

* **Slice is deterministic** by the W96-D loader's
  `select_chartqa_subset_v1` / `select_realworldqa_subset_v1`
  function; the slice seed is pre-committed in this runbook
  (96_504_001 for D1; 96_504_002 for D2).
* **Same VLM model on every arm** — A1 / B-reader use the
  candidate VLM in vision mode; A0 / B-solver use the same
  candidate VLM in text-only mode (image=None).  No cross-family
  mixing.
* **Same K=5 byte-exact budget** on A1 and B; A0 = 1 call.
* **Executor truth** = `evaluate_chartqa_answer_v1` /
  `evaluate_realworldqa_answer_v1` for every arm.  No LLM judge.
* **Parquet / dataset SHA anchored** at preflight time.
* **No selective retries.**

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New modules `coordpy.chartqa_loader_v1`,
  `coordpy.chartqa_executor_v1`, `coordpy.chartqa_preflight_v1`,
  `coordpy.realworldqa_loader_v1`,
  `coordpy.realworldqa_executor_v1`,
  `coordpy.realworldqa_preflight_v1` are **explicit-import only**.
  They do NOT enter the `coordpy/__init__.py` public surface
  (advanced work, not promoted to default API).
* New scripts `scripts/run_w96d_chartqa_preflight.py` and (if
  earned) `scripts/run_w96d_realworldqa_preflight.py` are
  independent of W95 / W96-A / W96-C scripts; nothing pre-existing
  is modified.

## Operational plan

1. **(W96-D NIM-free arsenal mining)** — DONE; see
   `docs/RESULTS_W96D_ARSENAL_MINING_V1.md`.
2. **(W96-D D1 NIM-free preflight at 11B)**
   a. Run
      `scripts/run_w96d_chartqa_preflight.py
      --candidate-model meta/llama-3.2-11b-vision-instruct`
      and record the verdict to
      `results/w96/chartqa_preflight_11b/<RUN_ID>/`.
   b. Verify composite PASS or document the cap.
3. **(W96-D D1 NIM-free preflight at 90B — only if 11B PASS)**
   a. Re-run with `--candidate-model
      meta/llama-3.2-90b-vision-instruct`.
4. **(W96-D D2 NIM-free preflight — only if D1 FAILS)**
   a. Run
      `scripts/run_w96d_realworldqa_preflight.py
      --candidate-model meta/llama-3.2-11b-vision-instruct`.
   b. Verify composite PASS or document the cap.
5. **(W96-D NIM smoke test — only if any preflight PASS)**
   a. One 1-token POST against the candidate model.
6. **(W96-D Phase 2 cheap pilot — only if smoke PASS; OUT OF
   SCOPE for this milestone unless explicitly earned)**
   a. Launch the 1-seed × 30-problem pilot.
   b. Apply the 9 Phase 2 gates.
7. **(Linear ↔ GitHub sync)**
   a. Update `COO-20` with the preflight verdict + pivot
      decision.
   b. Update `COO-6` summary if the verdict materially changes
      the W96 next move.
   c. Append a `W96-D` entry to `linear_github_mapping.json`
      and run `scripts/sync_linear_github_v1.py validate`.

## Honest framing

W96-D's job in this milestone is to **earn or kill the next
expensive run on a new battlefield by cheap evidence**.  A
preflight PASS at one scale licences a single NIM smoke test +
1-seed × 30-problem Phase 2 pilot; nothing more.  A preflight
FAIL at both scales for both battlefields earns a documented
double-pivot carry-forward and a Linear-recommended advance to
W96-C C2 (tool-augmented solver, the architectural backup on
MathVista) or further battlefield scouting.  Either way, the
W93/W94/W95/W96-A/W96-C discipline holds: no expensive run is
purchased on vibes.
