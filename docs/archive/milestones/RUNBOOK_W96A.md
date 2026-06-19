# W96-A — MathVista cheap pilot at Llama-3.2-90B-Vision (runbook)

> **Pre-commit contract for W96-A, locked 2026-05-24 BEFORE any
> NIM call against the 90B-Vision endpoint.**  W95 Phase 3 narrowly
> failed the +5 pp retirement bar on Llama-3.2-11B-Vision-Instruct
> (B − A1 = +3.67 pp at 3 seeds × 100 problems).  W96-A asks the
> cheapest possible follow-up question: does scaling the VLM
> family from 11B to 90B-Vision widen or shrink the B-team's
> margin?  This runbook is a strict W95-shape extension; the
> bench module, executor, and gate evaluator are reused without
> modification.
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.

## Linear

* `COO-17` (W96-A): MathVista Phase 2 cheap pilot at
  Llama-3.2-90B-Vision (this runbook).
* Parent: `COO-6` (post-W95 empirical frontier backlog).

## Hypothesis (locked 2026-05-24)

Llama-3.2-90B-Vision is ~3-4× the published single-shot
MathVista testmini score of the 11B variant (~49 % vs ~33 %).
Two structurally distinct outcomes are plausible:

* **H1 (positive):** the better VLM produces a *better*
  structured extraction in the B-reader role, which the
  math-solver can exploit; the team gap (B − A1) **widens**
  because the residual on B (vision-extract failure modes) is
  smaller AND the residual on A1 (unified-VLM monolithic
  attention) is still real.
* **H2 (negative):** the better VLM closes the unified-VLM gap
  by absorbing the math-solver step into a single forward;
  A1 saturates closer to its ceiling and the residual that B
  exploited at 11B disappears.  Under H2, B − A1 **shrinks**
  or reverses.

W96-A's cheapest discriminator between H1 and H2 is exactly a
W95-shape Phase 2 pilot at 90B with the same K=5 budget, same
deterministic 30-problem slice (seed 95_005_001), and same
executor / preflight / audit-chain machinery, evaluated
against the same 9 W95 Phase 2 gates.

We do NOT claim that 90B will retire any carry-forward.  W96-A
is a same-shape direct comparison whose only job is to decide
which of H1 / H2 the evidence supports.

## Baselines (locked 2026-05-24)

Identical to W95 Phase 2 except that the VLM model identifier
is `meta/llama-3.2-90b-vision-instruct` instead of
`meta/llama-3.2-11b-vision-instruct`.  Same arms (A0 / A1 / B0),
same K, same executor, same retry policy, same temperatures.

* **A0** — text-only mode (image=None) of
  `meta/llama-3.2-90b-vision-instruct` at T=0.0, K=1.
* **A1** — unified VLM mode of
  `meta/llama-3.2-90b-vision-instruct` at T=0.7, K=5.
* **B0** — `vlm_reader + math_solver + executor-guided
  reflexion`; same 1 + 1 + ≤ 3 = 5 calls; final reflexion turn
  pads budget to byte-exact 5 on A1 and B.

The B0 architecture is the W95-B0 candidate; reused
unmodified.  The W96-A delta is *only* the VLM weight class.

## Pre-committed W96-A NIM-free preflight gates

A W96-A NIM pilot launches IFF ALL of these hold.  All re-use
the existing W95 preflight harness without modification.

### W95 cheap probes (mandatory, NIM-free)

Re-run `scripts/run_w95_mathvista_preflight.py
--candidate-model meta/llama-3.2-90b-vision-instruct` and
require composite PASS:

1. **P1 — corpus integrity**: same parquet SHA, same 1000
   problems; identical to W95.
2. **P2 — executor self-test on gold**: ≥ 98 % gold-as-prediction
   pass rate; identical to W95.
3. **P3 — A1 failure-residual estimate (90B)**: the
   `MATHVISTA_PUBLISHED_SOTA_SINGLE_SHOT_BY_MODEL` table
   already carries `llama-3.2-90b-vision-instruct = 49.00 %`.
   At K=5 / correlation=0.5 this gives an estimated A1@K=5 of
   ~72.78 %, leaving ~27.22 pp residual — above the 20 pp
   floor.  P3 must PASS for 90B; if it fails, W96-A is killed
   in preflight without spending a single NIM call.
4. **P4 — decomposition argument**: same W95-B0 architecture →
   same written decomposition argument; corpus geometry/chart
   share is unchanged.

### W93 5-gate harness (mandatory)

Re-run via the same preflight script; the hypothesis text,
ablation argument, and benchmark justification are W95-B0's
text (the architecture is identical, only the VLM weight class
changed).  All 5 gates must PASS for the composite verdict to
PASS.

### W96-A 90B-specific cheap probes (mandatory)

These are additional probes specific to running on 90B.  They
are recorded but do not require new code paths; the existing
preflight harness suffices.

5. **Q1 — NIM endpoint reachability for 90B**: a single
   1-token POST against
   `meta/llama-3.2-90b-vision-instruct` returns HTTP 200 with
   a non-empty completion.  This is the *only* NIM call
   permitted before the formal pilot launches; it is recorded
   to the W96-A run sidecar as a smoke-test call.  If it
   fails, W96-A is parked until the endpoint recovers.
6. **Q2 — 90B per-call cost-of-wall plausibility**: the smoke
   call's wall-ms is recorded; if > 30 s for a single 1-token
   text completion, the pilot is paused and re-scoped (the 11B
   pilot's per-call wall was 5-15 s for K=5).
7. **Q3 — H1/H2 a-priori commitment**: the runbook records
   pre-pilot the *one-line prediction* of which outcome the
   author expects (H1 widen, H2 shrink, or genuinely unknown).
   This is a memory-of-claim mechanism; it does NOT gate the
   pilot but is anchored in the run sidecar so the post-hoc
   verdict cannot retro-fit the author's prior.

**Pre-pilot prediction (locked 2026-05-24 BEFORE NIM):**
"Genuinely unknown.  The W95 Phase 3 evidence is consistent
with either H1 (89 % B ≥ A1 problem coverage suggests B-mechanism
is real and should benefit from a stronger reader) or H2
(67.67 % A1 mean already shows headroom shrinking at 11B; 90B
plausibly closes more of that headroom in the unified forward).
The cheap pilot decides."

## Phase 2 — cheap NIM pilot (only if all G/P/Q gates pass)

### Pilot shape (locked 2026-05-24)

* **Bench**: `coordpy.mathvista_bench_v1` (unchanged from W95).
* **Model**: `meta/llama-3.2-90b-vision-instruct` via NIM.
* **Slice**: **1 seed (95_005_001) × 30 problems** —
  *byte-identical* to the W95 Phase 2 slice so that 11B vs 90B
  is directly comparable problem-by-problem.
* **Budget**: K=5 calls per problem on A1 and B; A0 = 1 call.
* **Expected cost**: 30 × 11 = 330 NIM calls; ~45-120 min
  wall (90B is slower per-call than 11B; the COO-17 risk note
  estimates 1.5-2× the 11B wall).

### Pre-committed Phase 2 pilot gates (W95 P1 bars, re-applied verbatim)

W96-A Phase 2 escalates to a FULL bench (W96-A Phase 3 at 90B,
or W96-B wider sample at 11B) IFF ALL of these hold.  These
are the same 9 gates locked in `docs/RUNBOOK_W95.md` Phase 2;
they are reused without modification by the pilot script.

1. **Slice pre-committed**: 30 problems by
   `select_mathvista_subset_v1(95_005_001, 30)` BEFORE any
   MathVista NIM call.  Slice SHA recorded.
2. **A1 < 90 %**: A1@K=5 pass rate on the 30-problem slice must
   stay below 90 %; if A1 ≥ 90 % at 90B, the same
   ceiling-saturation pathology as HumanEval-Visual K=5 has
   appeared on MathVista at the larger weight class — W96-A
   is killed and a `W96-L-MATHVISTA-90B-CEILING-CAP`
   carry-forward documents the saturation.
3. **B > A1**: `b_pass_rate > a1_pass_rate` on the slice.
4. **Margin ≥ +5 pp**: `b_pass_rate − a1_pass_rate ≥ 5.0 pp`.
   This is the strict W93/W95 retirement-margin bar; the
   cheap discriminator between "real signal at 90B" and
   "variance".
5. **B > A0 by ≥ +5 pp**: image is load-bearing in B (otherwise
   B is a glorified text-only solver, especially relevant at
   90B where the text-only A0 mode is itself stronger).
6. **Per-problem majority**: B ≥ A1 on ≥ 16 of 30 problems
   (> 53 %).
7. **Budget accounting exact**: per-call sidecar shows exactly
   1 A0 call + 5 A1 calls + 5 B calls = 11 model calls /
   problem on every problem.  No branching that exceeds K=5.
8. **Audit chain re-derives**: standard W88-style audit
   (SHA-256 + per-seed Merkle + bench Merkle re-derive offline
   via `scripts/verify_w95_mathvista_audit_chain.py`).
9. **Executor stays clean**: end-of-run P2 re-run on the 30
   slice problems → 100 % pass.  Drift means the bench truth
   function changed mid-run.

If any gate fails, **W96-A Phase 2 is KILLED** — no expensive
run earns the next slot.  Negative evidence is preserved as a
JSON sidecar + `docs/RESULTS_W96A_MATHVISTA_90B_PILOT_V1.md`
entry, and a `W96-L-MATHVISTA-90B-PILOT-*-CAP` carry-forward
is registered.  W96 then advances to `COO-19` (W96-C
architecture refinement) per the Linear-recommended ordering.

## Phase 3 — full bench (only if W96-A Phase 2 earns it)

### Phase 2 outcome (locked 2026-05-24)

**Phase 2 PASSED all 9 pre-committed gates** with **B − A1 =
+10.00 pp** at single-seed × 30 problems at 90B
(`docs/RESULTS_W96A_MATHVISTA_90B_PILOT_V1.md`).  The
+10.00 pp margin is byte-equivalent to W95 Phase 2 at 11B,
with identical B-only-rescue and A1-only-rescue counts.  The
+10 pp margin is NOT borderline (2× the +5 pp Phase 2 bar);
under this runbook's pre-commit, Phase 3 at 90B is the
default next step.

### Phase 3 shape (locked 2026-05-24 BEFORE any Phase 3 NIM call)

Identical W88 6-bar retirement shape to W95 Phase 3, only the
VLM weight class changes.

* **Bench**: same `coordpy.mathvista_bench_v1`.
* **Model**: `meta/llama-3.2-90b-vision-instruct` via NIM.
* **Budget**: K=5, exactly as Phase 2.
* **Scope**: 3 seeds × 100 problems × K=5.  Seeds
  pre-committed BEFORE any Phase 3 NIM call: **95_005_001 /
  95_005_002 / 95_005_003** — the *exact same seed identities*
  as W95 Phase 3 at 11B, so the 11B vs 90B comparison stays
  problem-level fair across both Phase 2 (single seed) and
  Phase 3 (3-seed).
* **Expected cost**: 3 × 100 × 11 = 3300 NIM calls; expected
  wall ~2.5-4 h (90B per-call wall observed in Phase 2 is
  ~3.86 s avg; scaling linearly to 3300 calls is ~3.5 h).

### Pre-committed Phase 3 retirement bars (W88 6-bar shape)

Locked verbatim from `docs/RUNBOOK_W95.md` Phase 3 (reused
without modification by the pilot script's `--phase phase3`
mode in `_evaluate_phase3_retirement_bars()`).  All thresholds
apply to the 3-seed cross-seed aggregates:

1. `b_mean strictly beats a0_mean`
2. `b_mean strictly beats a1_mean`
3. `b_mean − a0_mean ≥ +5.0 pp`
4. `b_mean − a1_mean ≥ +5.0 pp`
5. B beats A0 on > half seeds (≥ 2 of 3).
6. B beats A1 on > half seeds (≥ 2 of 3).
7. Budget accounting exact (1 + 5 + 5 = 11 calls / problem on
   every problem on every seed).
8. Audit chain present (per-seed Merkle roots + bench Merkle
   root re-derive offline).
9. Slices pre-committed per seed (all 3 pid lists SHA-256
   hashed BEFORE NIM).

If all 6 retirement bars (1..6) PASS:

* **NEW carry-forward retirement** — `W96-T-MATHVISTA-VLM-
  TEAM-K5-90B-SUPERIORITY` is registered as the FIRST confirmed
  cross-modal same-budget multi-agent superiority retirement.
* This would be the SECOND confirmed same-budget multi-agent
  superiority retirement overall (the first being W89
  HumanEval-70B-Reflexion-K=5).

If fewer bars pass:

* `W96-L-MATHVISTA-90B-RETIREMENT-MARGIN-CAP` records the
  additional negative evidence.
* W96 advances to `COO-19` (architecture refinement / verifier
  turn / tool-augmented solver) per the Linear-recommended
  ordering, since further scaling has been empirically shown
  to not help at 90B.

### Anti-cheat additions for Phase 3

* The 3 seed identities (95_005_001 / 95_005_002 / 95_005_003)
  are pre-committed in this runbook BEFORE the Phase 3 NIM
  call.  This is the SAME seed list W95 Phase 3 used at 11B,
  so the 11B vs 90B Phase 3 comparison stays problem-level
  fair on every seed.
* Same model on every arm (90B Vision in vision-mode for A1 /
  B-reader; text-only mode for A0 / B-solver).
* Same K=5 budget; budget gate enforced byte-exactly.
* Executor truth = `evaluate_answer_v1` for every arm; no LLM
  judge.
* Parquet SHA-256 anchored at run start.
* Same NIM HTTPS path; 429-aware backoff carries over.

### Phase 3 launch command (locked 2026-05-24)

```
NVIDIA_API_KEY=... python scripts/run_w95_mathvista_pilot.py \
  --phase phase3 \
  --vlm-model meta/llama-3.2-90b-vision-instruct \
  --n-problems 100 --n-seeds 3 \
  --seed-start 95005001 \
  --out-dir results/w96/mathvista_90b_phase3 \
  --expected-parquet-sha256 \
    373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d
```

### Q3 a-priori Phase 3 prediction (locked BEFORE Phase 3 NIM)

The Phase 2 cross-scale evidence (11B and 90B both produce
+10.00 pp margin at single-seed × 30) is *suggestive* but
not conclusive that 90B Phase 3 will replicate the W95 Phase
3 narrowing (single-seed +10 pp → multi-seed +3.67 pp).  Two
distinct outcomes are plausible at Phase 3:

* **H3-A (90B replicates the narrowing):** the multi-seed
  margin lands near +3-4 pp, similarly missing the +5 pp
  retirement bar.  This would re-confirm `W95-L-MATHVISTA-
  RETIREMENT-MARGIN-CAP` as a structural cap on the W95-B0
  architecture at this benchmark; W96-A would NOT retire the
  carry-forward and W96 would advance to W96-C architecture
  refinement.
* **H3-B (90B does NOT narrow):** the structural mechanism is
  robust enough at 90B that the third-seed reversal seen at
  11B (-5 pp on seed 95_005_003) does not appear at 90B,
  and the multi-seed margin stays at +5-10 pp.  This would
  retire the cross-modal carry-forward and become the SECOND
  confirmed multi-seed same-budget multi-agent superiority
  retirement.

**Locked pre-Phase-3 prediction:**  H3-A is the marginally
more likely outcome (the architectural-invariance evidence
from Phase 2 cuts both ways — robust margin at single-seed
AND robust narrowing at multi-seed), but the prior is weak
and the evidence is the actual Phase 3 run.

## Anti-cheat (carry-forward from W88–W95)

All W88–W95 anti-cheat clauses carry forward verbatim.  W96-A-
specific additions:

* Slice is the byte-identical pre-committed
  `select_mathvista_subset_v1(95_005_001, 30)` slice from W95
  Phase 2; we are testing the SAME problems on a stronger
  model, not a fresh slice.  This makes the 11B vs 90B
  comparison problem-level fair.
* No selective retries; one set of NIM calls per (seed,
  problem, arm) triple.
* Executor truth = `evaluate_answer_v1` for every arm; no LLM
  judge anywhere.
* The MathVista testmini parquet SHA-256 is anchored at
  W96-A run start; mismatches refuse to run the pilot.
* The Q1 smoke-test NIM call is recorded but does NOT count
  toward the pilot's per-problem budget accounting; it is a
  separate sidecar entry with `kind=smoke_test`.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* No new modules added.  W96-A re-uses
  `coordpy.mathvista_loader_v1`,
  `coordpy.mathvista_executor_v1`,
  `coordpy.mathvista_preflight_v1`,
  `coordpy.mathvista_bench_v1` verbatim.
* Pilot script `scripts/run_w95_mathvista_pilot.py` is
  re-used unmodified (`--vlm-model
  meta/llama-3.2-90b-vision-instruct`).
* Preflight script
  `scripts/run_w95_mathvista_preflight.py` is re-used
  unmodified (`--candidate-model
  meta/llama-3.2-90b-vision-instruct`).

## Operational plan

1. (W96-A NIM-free preflight)
   a. Re-run
      `scripts/run_w95_mathvista_preflight.py
      --candidate-model meta/llama-3.2-90b-vision-instruct`
      and record the verdict to
      `results/w96/mathvista_preflight_90b/<RUN_ID>/`.
   b. Verify composite PASS.
   c. Q1 smoke-test: one 1-token POST against the 90B
      endpoint; record reachability + wall-ms.
2. (W96-A Phase 2 cheap NIM pilot — only if 1 passed)
   a. Launch the 1-seed × 30-problem pilot via
      `scripts/run_w95_mathvista_pilot.py --phase phase2
      --vlm-model meta/llama-3.2-90b-vision-instruct
      --n-problems 30 --n-seeds 1 --out-dir
      results/w96/mathvista_90b_phase2`.
   b. Apply the 9 W95 Phase 2 gates; pass → escalate decision
      (Phase 3 at 90B vs W96-B wider 11B).  Fail → kill.
3. (Audit chain re-derivation)
   a. Run
      `scripts/verify_w95_mathvista_audit_chain.py
      --run-dir <pilot-run-dir>`.
4. (Linear ↔ GitHub sync)
   a. Update `COO-17` with Phase 2 verdict + run-dir CID.
   b. Update `COO-6` summary if the verdict materially changes
      the W96 next move.
   c. Append a `W96-A` entry to `linear_github_mapping.json`
      and run `scripts/sync_linear_github_v1.py`.

## Honest framing

W96-A is a 1-seed × 30-problem cheap pilot.  It cannot retire
any carry-forward by itself; it can only earn or kill the
next expensive run.  The W93/W94/W95 preflight-first
discipline says: a +10 pp pilot does NOT license a retirement
claim, and a -5 pp pilot does kill the line cheaply.  W96-A
is the next cheap probe on the W95 line, nothing more.
