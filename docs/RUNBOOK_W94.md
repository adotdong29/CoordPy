# W94 — Preflight-Earned K=10 Pilot + Cross-Modal Battlefield Pivot (runbook)

> **Pre-commit contract for W94, locked 2026-05-24 BEFORE any
> pilot run.**  W93's preflight harness killed all 3 candidate
> architectures on cheap evidence; only W93-C (K=10 reflexion)
> failed solely on G2 (no K=10 evidence yet in W88–W92
> sidecars).  W94 resolves G2 with the smallest-decisive
> pilot, then escalates only if earned.  Cross-modal in W94
> is scouting-only; no expensive bench on HumanEval-Visual.

## Phase 1 — K=10 pilot (resolves G2 for W93-C)

### Pilot shape

* **Bench**: `coordpy.humaneval_reflexion_bench_v1` (the W89-
  validated module).  Same A0 / A1 / B arms as W89; only K
  changes.
* **Model**: `meta/llama-3.3-70b-instruct` via NIM (same as
  W89 retirement).
* **Budget**: K=10 model calls per problem on A1 and B.
  A1 = K=10 first-pass-among-K; B = K=10 sequential reflexion.
* **Slice**: **1 seed (88_028_001) × 15 problems**.  The 15
  problems are the FIRST 15 problems selected by the
  deterministic `select_humaneval_subset_v1(seed=88_028_001,
  n_problems=15)` — pre-committed BEFORE results are viewed.
  These 15 are a STRICT SUBSET of the W89 / W91 P1 30-problem
  subset; no cherry-picking.
* **Expected cost**: 15 × 21 = 315 NIM calls; ~52 min wall.

### Pre-committed pilot gates (W94 P1 bars)

W94-P1 launches a FULL K=10 bench (Phase 2) IFF ALL of these
hold AFTER the pilot completes:

1. **Slice pre-committed**: the 15 problems are taken by
   `select_humaneval_subset_v1(88_028_001, n_problems=15)`
   BEFORE any K=10 NIM call.  ✓ Built into the driver.
2. **B > A1 on the slice**: `b_pass_at_1 > a1_pass_at_1` for
   seed 88_028_001 on the 15-problem subset.
3. **Margin ≥ +5 pp**: `b_pass_at_1 − a1_pass_at_1 ≥ 5.0 pp`.
   (At small N=15 the per-problem variance is high; +5 pp is
   the W93 retirement-bar margin and is the right cheap
   discriminator.)
4. **Per-problem majority**: B's per-problem outcomes vs A1's
   per-problem outcomes — B ≥ A1 on at least **8 of 15
   problems**.  (>50 % per-problem.)
5. **Budget accounting exact**: per-call sidecar shows
   exactly 1 A0 call + 10 A1 calls + 10 B calls per problem
   = 21 model calls/problem on every problem.  No branching
   that exceeds K=10.
6. **Audit chain re-derives**: standard W88-style audit
   (SHA-256 + per-seed Merkle + bench Merkle re-derive
   offline; verifier reports PASS).

If any gate fails, **W94-C is KILLED** — no full K=10 bench
launched; the negative evidence is preserved and the K=10
hypothesis is retired with the W94-L-K10-PILOT-FAILED-CAP
carry-forward.

### Reasoning behind the bars

* **Why margin ≥ +5 pp at the pilot?**  W89 K=5 showed +5.56
  pp at 30 problems × 3 seeds.  The K=10 hypothesis predicts
  the gap GROWS (more reflexion turns; diminishing i.i.d.
  diversity).  If at K=10 the gap doesn't even reach +5 pp on
  a single-seed pilot, the hypothesis is weak even before
  variance considerations.  A pilot result of +1-4 pp is
  ambiguous and would predict the full bench is a coin flip;
  per W94 discipline, ambiguous = KILL.
* **Why per-problem majority ≥ 8/15?**  At 15 problems, B ≥
  A1 on 8 (53 %) is a reasonable per-instance majority bar.
  Per-seed majority bar from earlier waves (≥ 2/3 or ≥ 3/5)
  generalises to per-problem when seeds collapse to one.

## Phase 2 — Conditional full K=10 escalation (only if Phase 1 passes)

### Full bench shape (if earned)

* **Bench**: same `coordpy.humaneval_reflexion_bench_v1`.
* **Model**: same Llama-3.3-70B-Instruct.
* **Budget**: K=10 on A1 and B.
* **Scope**: 3 seeds (88_028_001 / 88_028_002 / 88_028_003)
  × 30 problems × K=10.  Same seed identity as W89 K=5 for
  direct comparability.
* **Expected cost**: 3 × 30 × 21 = 1890 NIM calls; ~5.25
  hours wall.

### Pre-committed full-bench retirement bars

Same 4 bars as W89 K=5, applied at K=10:

1. `b_mean_strictly_beats_a1_mean = True`
2. `b_mean − a1_mean ≥ +1.0 pp` (cross-seed average margin)
3. `b_mean_strictly_beats_a0_mean = True`
4. B beats A1 on more than half the seeds (≥ 2 of 3).

If all 4 met:
* **NEW carry-forward retirement**:
  `W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP` is
  REFINED (the K=5 result was the prior bar; K=10 establishes
  a stronger budget-extended claim).  Additionally, a NEW
  W94-T-HUMANEVAL-REFLEXION-K10-* claim establishes the
  K=10 superiority.
* The MBPP per-seed-majority failure
  (`W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP`) is NOT directly
  addressed by W94 but the K=10 architecture suggests a path
  for W95 (run MBPP at K=10).

If fewer bars met: new
`W94-L-HUMANEVAL-REFLEXION-K10-*-CAP` records the additional
negative evidence; the K=10 hypothesis dies at scale.

## Phase 3 — Cross-modal scouting (no expensive bench)

W93 evidence: HumanEval-Visual K=5 vs unified-VLM K=5 is
empirically the WRONG battlefield (3 architectures, 7
configurations, all negative on team-organisation).  W94
cross-modal work is **scouting only**:

* Document candidate benchmark families (MathVista, MMVet,
  ChartQA, DocVQA, SEED-Bench, RealWorldQA) using PUBLISHED
  SOTA VLM scores.
* For each: estimate the unified-VLM K=5 ceiling.  A
  benchmark where SOTA single-shot is < 70 % has room for B
  to win; a benchmark where SOTA single-shot > 90 % is
  another presumptively hostile battlefield.
* Select 1-2 candidate battlefields for W95+ exploration.
* Document the selection rationale.

W94 does NOT launch any cross-modal NIM run.  Cross-modal
deliverable is a battlefield-selection note.

## Anti-cheat (carry-forward from W88–W93)

All W88–W93 anti-cheat clauses carry forward.  W94-specific:

* Pilot slice (15 problems) is taken by deterministic
  `select_humaneval_subset_v1(88_028_001, n_problems=15)`.
  The driver writes the chosen task_ids to the sidecar BEFORE
  the K=10 calls run.  No cherry-picking.
* If the pilot fails any gate, W94 does NOT silently re-run.
  The negative evidence is committed.
* The full K=10 bench (Phase 2) is launched only if all 6
  pilot gates pass.  If launched, retirement bars are the
  W89 4-bar shape.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* W94 adds a `--K` flag to
  `scripts/run_w88_humaneval_reflexion_bench.py` (small
  parameter-only change; module unchanged).
* W94's new docs are explicit non-code surfaces.

## Operational plan

1. Add `--K` flag + `--n-problems` flag to the existing
   driver (small edit; ~5 min).
2. Smoke test K=10 with 1 problem (~3 min).
3. Launch K=10 pilot (1 seed × 15 problems × K=10) in
   background (~52 min wall).
4. Wait for completion; verify audit chain.
5. Apply W94 P1 pilot gates.  Decide: pass → Phase 2;
   fail → kill.
6. Cross-modal scouting writeup (parallel; no NIM calls).
7. Document results + commit.
8. Conditional: launch full K=10 bench if Phase 1 passed.
9. Final commit + push approval.
