# W95 — MathVista cross-modal team-superiority programme (runbook)

> **Pre-commit contract for W95, locked 2026-05-24 BEFORE any
> NIM call on MathVista.**  W94 retired HumanEval-Visual K=5 as a
> serious cross-modal battlefield (3 architectures × 7
> configurations, all negative) and selected MathVista testmini
> as the W95 battlefield per
> `docs/W94_CROSS_MODAL_BATTLEFIELD_SCOUTING.md`.  W95 builds the
> MathVista line preflight-first under the W93 discipline: cheap
> probes earn (or kill) the expensive run BEFORE a single NIM
> dollar is spent.
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`.

## Hypothesis (locked 2026-05-24)

Multi-modal math reasoning on MathVista testmini decomposes
naturally into a **vision-reader** stage (extract numerical,
geometric, and tabular content from the image into structured
text) and a **math-solver** stage (compute the answer from the
extraction + question text).  A multi-agent team that explicitly
separates these stages can:

1. Spend its per-problem budget on different *kinds* of model
   calls (vision-extract, math-solve, verifier) rather than on
   independent unified-VLM samples;
2. Preserve the image as cross-modal evidence across turns
   instead of folding it once into a single-shot VLM forward;
3. Exploit MathVista's lower unified-VLM ceiling
   (published single-shot ~33 % for Llama-3.2-11B-Vision and
   ~49 % for Llama-3.2-90B-Vision) — a much larger failure
   residual than HumanEval-Visual K=5's 8–12 %.

We do NOT claim that any specific architecture will retire any
specific carry-forward.  This is the W95 *line of attack*; the
W93 preflight discipline + the W95 cheap probes earn (or kill)
specific candidates.

## Baselines (locked 2026-05-24)

For every (seed, problem, arm) triple the SAME provider,
SAME budget K, SAME retry policy, and SAME executor
(`coordpy.mathvista_executor_v1.evaluate_answer_v1`) are used.

* **A0** — text-only LLM, single-shot at T=0.0.  Sees ONLY the
  MathVista `query` (question + multi-choice options), no image
  access.  Floor for "image is load-bearing".
* **A1** — unified VLM (e.g., Llama-3.2-11B-Vision-Instruct or
  Llama-3.2-90B-Vision-Instruct), K=5 INDEPENDENT samples at
  T=0.7, ship the first prediction that the W95 executor scores
  PASS.  Same as W88/W90's A1_vlm shape adapted to MathVista's
  answer-match semantics.
* **B** — multi-agent team (specific architecture chosen per
  candidate; default candidate B0 below).  K=5 model calls total,
  no branching above K.

The **default W95 candidate B0** is the
``vlm_reader + math_solver`` team:

  1. VLM-Vision-Reader at T=0.0 (1 call) — extracts a structured
     text bullet list of numerical / geometric / tabular facts
     from the image, conditioned on the question.
  2. Math-Solver at T=0.7 (1 call) — generates a candidate
     answer from the question + VLM extraction (no image).
  3. Executor verifies via `evaluate_answer_v1`; if FAIL, the
     solver does up to 3 sequential reflexion turns conditioned
     on (prior candidate, executor verdict, executor diagnostics)
     history (≤ 3 calls).
  4. Total = 1 + 1 + ≤ 3 = ≤ 5 calls; padded to exactly 5 by a
     final reflexion turn so budget parity with A1 is byte-exact.

The B0 candidate architecture is RECORDED here but is NOT
permitted to earn an expensive run until it passes W95 preflight
gates AND W93's 5-gate harness.

## Pre-committed preflight gates (locked BEFORE any NIM call)

A W95 candidate launches a NIM pilot (`Phase 2`) IFF ALL of
these hold:

### W93 5-gate harness (mandatory)

1. **G1 — hypothesis written** for the candidate, ≥ 50 chars,
   names how it differs from A1.
2. **G2 — sidecar evidence** — either prior W88-W92 evidence
   (none exists for MathVista) OR a cheap NIM-free probe verdict
   from `coordpy.mathvista_preflight_v1.run_mathvista_preflight_v1`
   with `overall_passes = True`.
3. **G3 — adversarial ablation** — removing the team's key
   structural feature reduces its hypothesised advantage on a
   cheap preflight estimate (or a documented design-time
   argument).
4. **G4 — budget accounting** — exactly K calls per problem on
   every arm.
5. **G5 — benchmark justification** — written, ≥ 50 chars,
   names why MathVista is a better battlefield than
   HumanEval-Visual K=5.

### W95 MathVista-specific cheap probes (mandatory)

These run with NO NIM calls.

1. **P1 — corpus integrity**: testmini parquet hashes to a known
   SHA, decodes to 1000 problems, every problem has an image +
   answer + valid answer/question type.
2. **P2 — executor self-test on gold**: feeding each problem's
   gold answer through `evaluate_answer_v1` must score ≥ 98 %.
   Else the executor silently penalises every arm equally and
   the bench truth function is broken.
3. **P3 — A1 failure-residual estimate**: the candidate VLM's
   published single-shot SOTA on MathVista testmini, projected
   to A1@K=5, must leave ≥ 20 pp residual (estimated A1@K=5
   pass rate ≤ 80 %).  Documented presumption; the real pilot
   may revise.
4. **P4 — decomposition argument**: ≥ 200-char written
   structural argument for B AND the corpus has ≥ 20 % geometry
   / figure / chart / scientific problems in the sample.

If any G or P gate fails, the candidate is KILLED in preflight.
Negative evidence is preserved as a JSON sidecar +
`docs/RESULTS_W95_MATHVISTA_PREFLIGHT_V1.md` entry.

## Phase 2 — Cheap NIM pilot (only if all G/P gates pass)

### Pilot shape (locked 2026-05-24)

* **Bench**: a new `coordpy.mathvista_bench_v1` module (built in a
  follow-up commit after Phase 1's preflight verdict).
* **Model**: `meta/llama-3.2-11b-vision-instruct` via NIM (the
  cheaper VLM; if it fails honestly, escalate to
  `meta/llama-3.2-90b-vision-instruct` in a separate pilot).
* **Slice**: **1 seed (95_005_001) × 30 problems**, deterministic
  via `select_mathvista_subset_v1(seed=95_005_001,
  n_problems=30, corpus=...)`.  Pre-committed BEFORE results
  viewed.  The 30 problems span the 4 main MathVista
  task categories (geometry / math word / figure / table) by
  the deterministic selection.
* **Budget**: K=5 calls per problem on A1 and B; A0 = 1 call.
* **Expected cost**: 30 × 11 = 330 NIM calls; ~30-60 min wall.

### Pre-committed Phase 2 pilot gates (W95 P1 bars)

W95 Phase 2 escalates to a FULL bench (Phase 3) IFF ALL hold:

1. **Slice pre-committed**: 30 problems by
   `select_mathvista_subset_v1(95_005_001, 30)` BEFORE any
   MathVista NIM call.
2. **A1 < 90 %**: A1@K=5 pass rate on the 30-problem slice must
   stay below 90 %; if A1 already ≥ 90 %, MathVista has the same
   HumanEval-Visual ceiling-saturation pathology and W95 is
   killed.
3. **B > A1**: `b_pass_rate > a1_pass_rate` on the slice.
4. **Margin ≥ +5 pp**: `b_pass_rate − a1_pass_rate ≥ 5.0 pp`.
   Same W93 retirement-margin bar; the cheap discriminator
   between "real signal" and "variance".
5. **B > A0 by ≥ +5 pp**: image is load-bearing in B (otherwise
   B is a glorified text-only solver).
6. **Per-problem majority**: B ≥ A1 on ≥ 16 of 30 problems
   (> 53 %).
7. **Budget accounting exact**: per-call sidecar shows exactly
   1 A0 call + 5 A1 calls + 5 B calls = 11 model calls/problem
   on every problem.  No branching that exceeds K=5.
8. **Audit chain re-derives**: standard W88-style audit
   (SHA-256 + per-seed Merkle + bench Merkle re-derive offline;
   verifier reports PASS).
9. **Executor stays clean**: at the end of the bench, re-run P2
   on the slice (gold → executor → 100 % pass).  Drift means
   the bench truth function changed mid-run.

If any gate fails, **W95 Phase 2 is KILLED** — no full bench
launched; the negative evidence is preserved and the candidate
is retired with a `W95-L-MATHVISTA-PILOT-*-CAP` carry-forward.

## Phase 3 — Full bench (only if Phase 2 earns it)

### Full bench shape (locked 2026-05-24)

* **Bench**: same `coordpy.mathvista_bench_v1`.
* **Model**: same as Phase 2.
* **Budget**: same K=5.
* **Scope**: 3 seeds (95_005_001 / 95_005_002 / 95_005_003) ×
  100 problems × K=5.  Same seed identity as Phase 2 + two
  fresh seeds for cross-seed evidence.
* **Expected cost**: 3 × 100 × 11 = 3300 NIM calls;
  ~5-9 hours wall.

### Pre-committed full-bench retirement bars

Same W88 6-bar shape adapted to MathVista's answer-match:

1. `b_mean_strictly_beats_a0_mean = True`
2. `b_mean_strictly_beats_a1_mean = True`
3. `b_mean − a0_mean ≥ +5.0 pp`
4. `b_mean − a1_mean ≥ +5.0 pp`
5. B beats A0 on > half seeds (≥ 2 of 3).
6. B beats A1 on > half seeds (≥ 2 of 3).

If all 6 met:

* **NEW carry-forward retirement** — `W88-L-CROSS-MODAL-CODE-V1-
  SPLIT-NOT-LOAD-BEARING-CAP` is not directly retired (it is a
  benchmark-specific finding on HumanEval-Visual), but a NEW
  carry-forward `W95-T-MATHVISTA-VLM-TEAM-K5-SUPERIORITY` is
  registered as a same-budget multi-agent superiority claim on
  MathVista.
* **Retirement claim**: this would be the SECOND confirmed
  same-budget multi-agent superiority retirement (the first
  being W89 HumanEval-70B-Reflexion-K=5).

If fewer bars met: new
`W95-L-MATHVISTA-PILOT-*-CAP` records the additional negative
evidence; the MathVista line is retired or moved to the
ChartQA / RealWorldQA backups per `docs/W94_CROSS_MODAL_BATTLEFIELD_SCOUTING.md`.

## Anti-cheat (carry-forward from W88–W94)

All W88–W94 anti-cheat clauses carry forward.  W95-specific:

* Slice is taken by deterministic
  `select_mathvista_subset_v1(seed, n_problems, corpus)`.  The
  bench writes the chosen pids to the sidecar BEFORE the NIM
  calls run.  No cherry-picking.
* No selective retries; one set of NIM calls per (seed, problem,
  arm) triple.
* Executor truth = `evaluate_answer_v1` for every arm.
* The MathVista testmini parquet SHA-256 is anchored at bench
  start; mismatches refuse to run the bench.
* No LLM-judge anywhere; the executor is deterministic.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* `coordpy.mathvista_loader_v1`, `coordpy.mathvista_executor_v1`,
  `coordpy.mathvista_preflight_v1` are explicit-import only.

## Operational plan

1. (Phase 1 / W95 cheap probes — preflight)
   a. Build the loader / executor / preflight modules.  ✓
   b. Author this runbook.  ✓
   c. Fetch the canonical testmini parquet to
      `~/.cache/coordpy/mathvista/`; SHA-anchor.
   d. Run all four W95 probes via
      `scripts/run_w95_mathvista_preflight.py`.
   e. Commit results to `results/w95/mathvista_preflight/`.
   f. Write `docs/RESULTS_W95_MATHVISTA_PREFLIGHT_V1.md`.
2. (Phase 2 / cheap NIM pilot — only if 1 passed)
   a. Build the bench module `coordpy.mathvista_bench_v1`.
   b. Smoke-test on 1 problem.
   c. Launch the 30-problem pilot in background.
   d. Apply pilot gates; pass → Phase 3; fail → kill.
3. (Phase 3 / full bench — only if 2 passed)
   a. Launch the 3 × 100 × K=5 full bench in background.
   b. Apply retirement bars; declare or refuse retirement.

## Honest framing

The W95 deliverable is whatever the preflight + pilot produce,
not a pre-decided retirement.  W94 already demonstrated that
preflight + cheap pilots can KILL hypotheses that look strong on
paper.  W95 may do the same.  The W95 contribution is the
infrastructure to ask the question cheaply.
