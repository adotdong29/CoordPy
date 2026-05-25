# W97 — RealWorldQA D2-B0 Phase 2 cheap pilot — runbook

> **Pre-commit contract for W97, locked 2026-05-25 BEFORE any
> NIM call for the W97 candidate.**
>
> W96-D D2 (RealWorldQA) preflight PASSed all 9 composite gates
> at both 11B-Vision (residual 26.56 pp) and 90B-Vision
> (residual 20.51 pp).  Per the W96-D runbook, D2 is
> preflight-earned for a NIM smoke test + 1-seed × 30-problem
> Phase 2 cheap pilot.  W97 launches that cheap pilot under the
> same 9-gate Phase 2 shape as W95 / W96-A / W96-C — gate texts
> are byte-identical to those runbooks; only the arm name and
> slice seed change.
>
> **The W97 deliverable in this milestone is the D2-B0 cheap
> pilot verdict — earn or kill the next expensive run by cheap
> evidence, per the W93/W94/W95/W96-A/W96-C/W96-D discipline.
> No Phase 3 retirement bench is in scope for this milestone
> unless Phase 2 PASS at BOTH 11B AND 90B (the W96-C cross-scale
> rule).**
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.

## Linear

* New issue **`COO-21`** (W97-A): RealWorldQA D2-B0 Phase 2
  cheap pilot at Llama-3.2-11B-Vision (cross-scale 90B
  conditional on 11B PASS).  Parent: `COO-6`.  High priority.
* Related: `COO-20` (W96-D parent; closed).
* This runbook covers **D2-B0 at 11B as the lead** and **D2-B0
  at 90B as the conditional cross-scale step**.  D2-B1
  (structured scene-graph) is documented as the architectural
  refinement in `docs/RESULTS_W97_ARSENAL_MINING_V1.md` but is
  NOT in scope for this milestone.

## Arsenal-mining anchor

`docs/RESULTS_W97_ARSENAL_MINING_V1.md` records the extended
mining inventory for RealWorldQA.  Selected lead candidate is
**D2-B0** — a conservative port of the W95-B0 architecture
(vlm_scene_reader + text_solver + executor-guided reflexion) to
RealWorldQA, with scene-specific prompts and no new mechanism
over W95-B0.  Architectural refinements (D2-B1 structured
scene-graph extraction) are deferred to a future W97-B
milestone and documented but **not implemented in this
milestone**.

## Frontier-relevance audit anchor

`docs/FRONTIER_RELEVANCE_AUDIT_W97_V1.md` classifies every
in-repo mechanism as active-frontier / baseline-only /
historical-artifact / dead-direction / anti-pattern.  W97 D2-B0
is sourced from the active frontier arsenal (W95-B0 scene-port +
W93 preflight + W96-D loader/executor/preflight) only.  No
bounded-window / compaction / summary mechanism is allowed in
the lead path.

## Hypothesis (locked 2026-05-25)

W95-B0's empirical signature on MathVista is +3.67 pp Phase 3
(B − A1; sub-retirement).  The W96-A 90B Phase 3 showed scaling
the VLM HURTS at MathVista.  The W96-C C1 verifier-final
refinement was not load-bearing.  W96-D's D1 (ChartQA) was
saturated.  The remaining route to cross-modal team superiority
at K=5 byte-exact is **a different benchmark distribution where
W95-B0's extract-then-reason shape lands cleanly**.

RealWorldQA's structural profile (per the W97 arsenal mining):

* **Counting / spatial-relations subset** — extraction is
  load-bearing.  Free-text bullet extraction may be lossy on
  spatial primitives.  D2-B0 is *structurally vulnerable* here.
* **Multi-choice subset** — the answer set is explicit.  The
  text solver does less work; A1 K=5 sampling has more even
  return.
* **Identification / sign-reading subset** — vision-dominant.
  Text solver cannot recover signal the extraction drops.  D2-B0
  is *structurally weakest* here.
* **Action / activity subset** — entanglement-dominated.  D2-B0
  expected to lose vs A1 K=5.

Two structurally distinct outcomes are plausible:

* **H1 (D2-B0 narrowly clears Phase 2):** the extraction
  schema lands well enough on the counting + multi-choice
  subset to clear +5 pp.  Earns cross-scale 90B Phase 2.
* **H2 (D2-B0 narrowly fails Phase 2):** extraction loss on
  spatial / identification subset dominates; B narrowly loses
  or ties A1.  Documents `W97-L-REALWORLDQA-D2-B0-PHASE2-CAP`;
  licences a D2-B1 preflight as the W97-B follow-up move.

**Pre-pilot prediction (locked 2026-05-25 BEFORE NIM):** "H2 is
the more likely outcome a priori — RealWorldQA's question
distribution rewards preserved visual signal more than
MathVista did, and the W95-B0 shape discards the image after
the reader call.  The cheap pilot is the discriminator; W97 may
end as a documented D2-B0 cap with no expensive 90B / Phase 3
spend at all."  The smoke run sidecar records this prediction
hash before any NIM call.

## Baselines (locked 2026-05-25)

Identical W95-shape (A0 / A1 / B) on RealWorldQA.

* **A0** — text-only mode (image=None) of the VLM at T=0.0, K=1.
* **A1** — unified VLM mode at T=0.7, K=5.
* **B (D2-B0)** — `vlm_scene_reader + text_solver +
  executor-guided reflexion` at total K=5 byte-exact: 1 VLM
  reader (T=0.0) + 4 text solver turns (T=temperature,
  executor-guided reflexion).

D2-B0's architectural shape is **byte-equivalent to W95-B0**;
only prompt content changes (scene description vs math
extraction).

## Pre-committed W97 Phase 2 pilot gates (W95 9-gate shape)

Phase 2 escalates to a cross-scale 90B Phase 2 (and then,
conditionally, to a Phase 3 pre-commit) IFF ALL of these hold.
Gate texts are byte-identical to W95 / W96-A / W96-C Phase 2;
only the arm name and slice seed change.

1. **Slice pre-committed**: 30 problems by the W96-D
   RealWorldQA loader's deterministic slice with **seed
   96_504_002** BEFORE any NIM call.  Slice SHA recorded.
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

If any gate fails at 11B, **W97 D2-B0 Phase 2 at 11B is
KILLED.**  The cross-scale rule (next section) decides whether
the candidate is killed overall or escalates to a separate 90B
Phase 2.

## Cross-scale rule (W96-C carry-over, locked 2026-05-25)

* **Cross-scale 90B Phase 2 entitled** IFF 11B Phase 2 PASS
  (all 9 gates) — OR with written justification if 11B FAILS
  by a narrow margin.
* **Phase 3 entitled** IFF Phase 2 PASS at BOTH 11B and 90B,
  OR Phase 2 PASS at 90B alone with a written justification.
* A Phase 2 PASS at 11B alone is **NOT sufficient** for Phase 3
  at 90B; the candidate gets a cross-scale Phase 2 at 90B first.
* A Phase 2 FAIL at 11B does **NOT auto-launch** a 90B Phase 2;
  the author pre-commits the 90B-Phase-2 decision in writing
  before any 90B NIM call.

## Phase 3 — full bench (NOT in this milestone)

If both Phase 2 stages clear, Phase 3 will require its own
runbook section locked BEFORE any Phase 3 NIM call, mirroring
W96-A Phase 3 in `docs/RUNBOOK_W96A.md`:

* 3 seeds × 100 problems × K=5.
* W88 6-bar retirement shape (same as W95 / W96-A).
* Audit-chain re-derivation requirement.

**No Phase 3 will be launched in this milestone.**  W97's job
is to earn or kill the next expensive run by cheap evidence;
the Phase 3 launch decision and runbook are explicitly out of
scope until Phase 2 evidence exists at both scales (or at 90B
with written justification).

## Pilot shape (locked 2026-05-25 BEFORE NIM)

* **Bench**: `coordpy.realworldqa_bench_v1` (new in W97;
  explicit-import only; mirrors `coordpy.mathvista_bench_v1`).
* **Model**: `meta/llama-3.2-11b-vision-instruct` (default) or
  `meta/llama-3.2-90b-vision-instruct` (cross-scale conditional).
  Same VLM family on A1 and B-reader; same text-LM (same model
  in text mode) on A0 and B-solver.
* **Slice**: **1 seed × 30 problems** via
  `select_realworldqa_subset_v1(seed=96_504_002,
  n_problems=30, corpus=test)`.  Same slice seed as the W96-D
  D2 pre-commit; SHA recorded before any NIM call.
* **Budget**: K=5 calls per problem on A1 and B; A0 = 1 call.
* **Expected cost**: 30 × 11 = 330 NIM calls at one scale;
  ~20-30 min wall at 11B; ~60-90 min wall at 90B.
  Cross-scale (both 11B + 90B) = 660 calls total.

## NIM smoke test (mandatory before pilot)

1. One 1-token POST against the candidate model returns HTTP
   200 with a non-empty completion.  Recorded as the W97 run
   sidecar's `kind=smoke_test` entry.
2. One 1-problem dry-run on the W97 D2-B0 wiring (1 A0 + 5 A1
   + 5 B = 11 NIM calls) at 11B.  Confirms the B-reader / text-
   solver / reflexion wiring works against the live endpoint
   before the 330-call pilot.
3. Smoke run dir: `results/w97/realworldqa_smoke_11b/<RUN_ID>/`.

## Anti-cheat (carry-forward from W88–W96-D)

All W88–W96-D anti-cheat clauses carry forward verbatim.  W97-
specific additions:

* **Slice is deterministic** by the W96-D loader's
  `select_realworldqa_subset_v1` function with seed
  **96_504_002**; the slice pids are SHA-anchored at run start.
* **Same VLM model on every arm** — A1 / B-reader use the
  candidate VLM in vision mode; A0 / B-solver use the same
  candidate VLM in text-only mode (image=None).  No cross-family
  mixing.
* **Same K=5 byte-exact budget** on A1 and B; A0 = 1 call.
* **Executor truth** = `evaluate_realworldqa_answer_v1` for
  every arm.  No LLM judge.
* **Parquet shards SHA-anchored** (both shards) at pilot start;
  mismatches refuse to run the bench.
* **No selective retries.**
* **Prompts are content-addressed** (per-call sidecar records
  prompt SHA + image SHA + response SHA).

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New modules `coordpy.realworldqa_bench_v1` is **explicit-import
  only**.  It does NOT enter the `coordpy/__init__.py` public
  surface (advanced work, not promoted to default API).
* New scripts `scripts/run_w97_realworldqa_smoke.py` and
  `scripts/run_w97_realworldqa_pilot.py` are independent of W95
  / W96-A / W96-C / W96-D scripts; nothing pre-existing is
  modified.

## Operational plan

1. **(W97 NIM-free arsenal mining + frontier-relevance audit)** —
   DONE; see `docs/RESULTS_W97_ARSENAL_MINING_V1.md` and
   `docs/FRONTIER_RELEVANCE_AUDIT_W97_V1.md`.
2. **(W97 D2-B0 bench module + tests)** —
   `coordpy/realworldqa_bench_v1.py` + `tests/test_w97_realworldqa_v1.py`.
3. **(W97 NIM smoke test at 11B)**
   a. Run
      `scripts/run_w97_realworldqa_smoke.py --candidate-model
      meta/llama-3.2-11b-vision-instruct` and record the verdict
      to `results/w97/realworldqa_smoke_11b/<RUN_ID>/`.
   b. Confirm 1-token POST PASSes and 1-problem dry-run runs
      end-to-end without errors.
4. **(W97 D2-B0 Phase 2 at 11B — only if smoke PASS)**
   a. Run
      `scripts/run_w97_realworldqa_pilot.py --candidate-model
      meta/llama-3.2-11b-vision-instruct`.
   b. Apply the 9 Phase 2 gates.
   c. Output: `results/w97/realworldqa_pilot_11b/<RUN_ID>/`
      with per-call sidecars, per-seed Merkle, bench Merkle,
      `phase2_gates.json`, `SUMMARY.md`.
5. **(W97 D2-B0 Phase 2 at 90B — conditional)**
   a. Only if 11B Phase 2 PASSes all 9 gates OR with explicit
      written justification.
6. **(W97 Phase 3 launch — OUT OF SCOPE)**
   a. Requires both Phase 2 PASS.
   b. Phase 3 runbook to be written separately if earned.
7. **(Linear ↔ GitHub sync)**
   a. Create `COO-21` (W97-A: D2-B0 Phase 2 cheap pilot) under
      `COO-6`.
   b. Update `COO-20` with the W97 follow-up note.
   c. Update `COO-6` summary with W97 verdict.
   d. Append a `W97` entry to `linear_github_mapping.json` and
      run `scripts/sync_linear_github_v1.py validate`.

## Honest framing

W97's job in this milestone is to **earn or kill the next
expensive run on RealWorldQA by cheap evidence**.  A Phase 2
PASS at 11B licences a cross-scale Phase 2 at 90B; nothing
more.  A Phase 2 FAIL at 11B documents
`W97-L-REALWORLDQA-D2-B0-PHASE2-CAP` and licences a documented
W97-B follow-up on D2-B1 (structured scene-graph extraction)
or a different battlefield.  Either way, the
W93/W94/W95/W96-A/W96-C/W96-D discipline holds: **no expensive
run is purchased on vibes; bounded / compaction / summary
mechanisms are NOT promoted to lead-path status; the lead is
the preflight-earned arsenal-driven candidate D2-B0.**
