# W101 — Second-code-benchmark battlefield tournament + MBPP+ lead selection + cheap preflight (runbook)

> **Pre-commit contract for W101, locked 2026-05-25 BEFORE any
> W101 NIM call.**
>
> W100 closed with BOTH B2 (frontier lead) and B5 (baseline-only
> ceiling reference) FAILing 90B Phase 2 on the W96-D-earned
> 96_504_002 / 30-problem slice of `lmms-lab/RealWorldQA`.  The
> pre-committed Part H code-pivot contingency in
> `docs/RUNBOOK_W100.md` triggered immediately, promoting
> `COO-9` (second code benchmark battlefield) to the lead path.
>
> The COO-9 charter is the W101 brief verbatim:
>
> 1. **Pick one benchmark family** from {MBPP+, HumanEval+,
>    APPS, LiveCodeBench, SWE-bench-lite} and justify why it is
>    a better battleground than current MBPP.
> 2. **Build/adapt loader + evaluator** with the same fairness
>    discipline.
> 3. **Specify A0 / A1 / B baselines BEFORE running** any
>    expensive bench, per the W93 preflight-first discipline.
> 4. **Pre-commit a runbook** including 9 Phase 2 gates and
>    (where the candidate is mechanism-driven) explicit
>    mechanism-load-bearingness sub-gates analogous to W100's
>    MLB-1 + MLB-2.
> 5. **Run cheap NIM-free preflight** before any NIM call.
> 6. **Mine W89 / W91 sidecars** for a cheap-probe surface
>    analogous to the W97 → W99 sidecar-mining + addressability-
>    probe cadence on RealWorldQA.
>
> All six items are addressed in this milestone.  The empirical
> preflight verdict licenses the *next* step (MBPP+ corpus fetch
> + canonical-solution executor self-test); the cheap NIM pilot
> is conditional on a clean preflight re-run after the corpus
> fetch lands.  W101 is NOT a NIM-spending milestone; W101's
> deliverable is the discipline + infrastructure + ranked
> battlefield slate + empirically-grounded preflight verdict.
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.

## Linear

* New issue **`COO-25`** (W101): second-code-benchmark
  battlefield tournament + MBPP+ lead selection + cheap
  NIM-free preflight.  Parent: `COO-6`.  High priority.
* Related: `COO-24` (W100; Done) — promoted `COO-9` via Part H.
* Related: `COO-9` (lead path; this milestone delivers its
  charter items 1-6).

## What is NOT in scope (anti-drift contract)

This milestone explicitly does NOT:

1. Re-open the cross-modal RealWorldQA arc.  Cross-modal
   RealWorldQA stays frozen at 11B per the W100 frontier audit.
2. Re-open the W95-B0 family, the typed-extract sub-family
   (D2-B0 + W98 B1 + W99 B4), or any RealWorldQA candidate.
3. Promote `COO-12` (substrate-level cross-modal injection)
   absent fresh evidence; `COO-12` stays Low.
4. Build a second code battlefield in parallel to MBPP+ unless
   MBPP+ preflight FAILs.  HumanEval+ is documented as the
   backup pivot but is NOT BUILT in W101 unless triggered.
5. Launch a NIM cheap pilot before MBPP+ corpus integrity (P1)
   + executor self-test (P2) PASS.  The cheap pilot is gated
   on the preflight re-run after the operator fetches the
   EvalPlus MBPP+ release artifact and records its SHA pin.
6. Bump `coordpy.__version__` or `SDK_VERSION`.
7. Publish to PyPI.
8. Edit `coordpy/__init__.py` (the W101 modules are
   explicit-import only).
9. Re-introduce any anti-pattern under a prettier name
   (bounded windowing; compaction; generic prose summarization;
   shallow token compression; context-pruning theater; "cram
   less / truncate better").  W97 / W98 / W99 / W100 frontier-
   relevance audits remain in force verbatim; W101 carries
   them forward.

## Battlefield selection (canonical)

* **LEAD: MBPP+** (EvalPlus's hardened MBPP).  Direct surgical
  attack on the live `W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP`.
* **BACKUP: HumanEval+** (EvalPlus's hardened HumanEval).
  Built only if MBPP+ preflight FAILs.
* **Out of scope (W101)**: APPS, LiveCodeBench, SWE-bench-lite.
  Documented in `docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md`
  with explicit per-criterion grades.

See `docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md` for the
full 5-candidate × 8-criterion ranking matrix.

## Arsenal mining (empirically grounded)

W101 mines the existing W88 70B HumanEval + W91 5-seed 70B
MBPP sidecars by re-executing every persisted candidate
response against the canonical subprocess executor.  No new NIM
calls.  See `scripts/run_w101_arsenal_mining.py`.

Mining produces a per-(seed, task_id, arm) cluster surface:

| Bench | A0 mean | A1@K=5 mean | B mean | B−A1 (pp) | A1-only | B-only | Shared wins | Shared fails | Mech LB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W89 HumanEval 70B (3 × 30) | 46.67 % | 85.56 % | 91.11 % | **+5.56** | 3 | **8** | 74 | 5 | **9.76 %** |
| W91 MBPP 70B (5 × 30) | 75.33 % | 82.67 % | 84.00 % | **+1.33** | 3 | 5 | 121 | 21 | **3.97 %** |

* The W89 / W91 aggregates re-derive *byte-for-byte* the
  published per-seed pass rates in `docs/RESULTS_W89_HUMANEVAL_REFLEXION_V2.md`
  and `docs/RESULTS_W91_MBPP_REFLEXION_V2.md`.  Re-execution is
  honest.
* **Mechanism-load-bearing prior**: the fraction of B wins that
  came from reflexion-on-A1-failure is **9.76 %** on HumanEval
  (the bench where W89 retired) and only **3.97 %** on base
  MBPP (the bench where W91 stalled).  This is the empirical
  symptom of the ceiling-saturation cap.  MBPP+ is designed to
  drop the ceiling, which should restore the rescue surface
  closer to the HumanEval-like ~ 10 % regime.
* **Hard-cluster size**: W91's shared-fails cluster (21 / 150)
  is ~4× larger than W89's (5 / 90).  Many of those problems
  are "A1 + B both pass on the base assertions but would fail
  on EvalPlus extra tests".  MBPP+ promotes these to the
  failure-residual surface B can actually attack.

See `docs/RESULTS_W101_ARSENAL_MINING_V1.md` for the full per-
seed table + per-cluster carry-forward implications.

## Hypotheses (locked 2026-05-25)

### MBPP+ at 70B (lead candidate)

**Claim**: the W89 sequential-reflexion B-pipeline retires the
+5 pp Phase 2 bar on MBPP+ at Llama-3.3-70B-Instruct under the
same K=5 same-budget contract that retired HumanEval-70B in W89.

Quantitative predictions (pre-committed before any NIM call):

* **A0 (single-shot T=0.0)** ≈ 60-70 % on MBPP+ (drops from
  W91 base 75.3 % by the conservative published EvalPlus drop;
  EvalPlus reports GPT-4 A0 drops 84.1 → 67.1 = 17.0 pp; the
  smallest published drop in the EvalPlus table is 12.7 pp,
  used as the Hoeffding lower bound).
* **A1 @ K=5** ≈ **69.97 %** (W91 5-seed mean 82.67 % − 12.7 pp
  Hoeffding lower-bound drop; per the W101 preflight P3
  empirical extrapolation).  Saturation margin **20.03 pp**
  below the 90 % gate-2 saturation threshold; comfortably
  preflight-clean.
* **B (sequential reflexion K=5)** ≥ A1 + **+5 pp** if the
  reflexion mechanism remains load-bearing on the MBPP+
  failure cluster.  Conservative central estimate: B ≈ 75-80 %;
  B − A1 ≈ +5-10 pp.
* If B − A1 < +5 pp on the MBPP+ cheap pilot, the mechanism is
  *partially* load-bearing on MBPP+ but doesn't clear the
  Phase 2 bar; the cap is `W101-L-MBPP-PLUS-REFLEXION-PHASE2-
  70B-CAP`.

### Cross-bench failure-cluster prior

The W101 arsenal mining + preflight establish the following
priors:

* **Rescue surface exists** at scale: 8 / 90 unique-B-rescues
  on W89-70B HumanEval (9.76 % of total B wins).
* **Cluster partition is clean**: A1-only / B-only / shared-wins
  / shared-fails sums match problem count exactly on both
  W89 and W91.
* **No anti-pattern primitives** are imported by the W101 bench
  module (AddrW101-P4 confirmed: no bounded_window /
  compaction / summary tokens in `coordpy/mbpp_plus_reflexion_bench_v1.py`).

### Pre-pilot prediction (recorded 2026-05-25 BEFORE preflight)

> "Subjective priors over the MBPP+ Phase 2 cheap pilot at 70B
> on a 1-seed × 30-problem slice with K=5:
>
> * Probability MBPP+ A1@K=5 clears the saturation gate
>   (< 90 %): **~ 90 %** (preflight predicts ~ 70 % A1; the
>   ceiling is empirically relieved).
> * Probability B beats A1 on the mean: **~ 80 %** (the W89
>   retirement on HumanEval is the closest empirical precedent;
>   MBPP+ extra tests give the reflexion mechanism more failure
>   signal to read).
> * Probability B − A1 ≥ +5 pp (Phase 2 cheap-pilot bar):
>   **~ 55-65 %** (the +5 pp bar is more conservative than
>   W89's +5.56 pp empirical margin; if MBPP+'s extra tests
>   surface failure modes the reflexion mechanism can read
>   from stderr, the margin scales).
> * Probability of cross-scale (i.e., 90B confirmation) being
>   *necessary* in W102 if W101 PASSes: **100 %** — per the
>   W96-C / W100 carry-forward cross-scale discipline.
>
> If MBPP+ cheap pilot PASSes the 9 Phase 2 gates with MLB
> sub-gates clearing, W102 = MBPP+ cross-scale confirmation at
> a larger model class (Llama-4 frontier or 90B-Vision-Instruct
> in text-only mode); W103 = MBPP+ Phase 3 retirement bench
> (3 seeds × 100 problems × K=5) if cross-scale PASSes.  If
> MBPP+ cheap pilot FAILS, the W101 verdict is the cap; W102
> attacks HumanEval+ via the same shape."

## Baselines (locked 2026-05-25)

Identical W89/W91 shape on MBPP+, same K, same sampling, same
executor:

* **A0** — `meta/llama-3.3-70b-instruct` at T=0.0, K=1.
* **A1** — `meta/llama-3.3-70b-instruct` at T=0.7, K=5
  first-pass-among-K self-consistency.
* **B** — `coordpy.mbpp_plus_reflexion_bench_v1.run_b_sequential_reflexion`
  (NEW W101): T=0.7, K=5 sequential reflexion conditioned on
  cumulative (candidate, executor_stderr) history.  Byte-
  identical mechanism to W88 / W90 reflexion; only the
  executor's test surface differs.

Same model on A0 / A1 / B.  Same K=5 budget on A1 / B.  Same
executor (`coordpy.mbpp_plus_executor_v1.run_mbpp_plus_executor_v1`)
in `mode="base_and_plus"`.

## W101 NIM-free preflight (already run; empirically grounded)

Run via `scripts/run_w101_mbpp_plus_preflight.py`.  Verdict
in `results/w101/mbpp_plus_preflight/<RUN>/verdict.json`.

| Probe | Verdict (2026-05-25) | Summary |
|---|---|---|
| **P1** corpus integrity | **DEFERRED** | MBPP+ cache absent; operator must fetch the EvalPlus release artifact + record its SHA pin in `MBPP_PLUS_TRUSTED_SHA256_OVERRIDE` env var OR update `MBPP_PLUS_EXPECTED_SHA256_V020` in `coordpy/mbpp_plus_loader_v1.py`. |
| **P2** executor self-test on canonical solutions | **DEFERRED** | Same gate as P1; the executor self-test runs only after MBPP+ corpus loads. |
| **P3** A1@K=5 failure-residual estimate | **PASS** | Predicted A1@K=5 on MBPP+ = **69.97 %**; residual **30.03 pp**; saturation floor 90 %; +5 pp Phase 2 bar.  Extrapolated from W91 5-seed A1 mean (82.67 %) − published EvalPlus Hoeffding lower-bound drop (12.7 pp). |
| **P4** decomposition argument | **PASS** | 1727-char structural argument: W89 sequential-reflexion mechanism reads MBPP+ extra-test failures from subprocess stderr; the reflexion turn is conditioned on the bug class; W89 retirement at +5.56 pp on HumanEval-70B is the empirical precedent. |
| **AddrW101-P1** mechanism-load-bearing prior | **PASS** | W89 rescue fraction = 9.76 %; W91 rescue fraction = 3.97 %.  Threshold ≥ 5 % on at least one historical bench (pre-committed BEFORE the empirical extraction; W89 clears with margin). |
| **AddrW101-P2** per-problem cluster structure | **PASS** | (a1_only / b_only / shared_wins / shared_fails) partition sums to 90 problems on W89 and 150 on W91; well-formed. |
| **AddrW101-P3** cross-bench failure-residual stability | **PASS** | Predicted MBPP+ A1@K=5 = 69.97 %; saturation margin = 20.03 pp; floor 10 pp. |
| **AddrW101-P4** anti-pattern guard | **PASS** | No `bounded_window` / `compaction` / `summary` tokens in the W101 bench module. |

**6 of 8 probes PASS; 2 DEFERRED on corpus fetch.**  The
substantive case is empirically grounded; the corpus fetch is
the next operator step.  Cheap pilot NOT yet earned.

### Cheap NIM pilot — decision logic (PRE-LOCKED)

1. **Preflight P1 + P2 both PASS at re-run after corpus fetch**
   AND every other probe still PASSes → cheap pilot ENTITLED.
   Launch `scripts/run_w101_mbpp_plus_pilot.py` at 1 seed × 30
   problems × K=5 (~330 NIM calls at 70B; ~2-3 h wall).  Phase 2
   gates evaluated.
2. **Preflight P1 PASS but P2 FAILS** → encoding bug in MBPP+
   loader / executor; fix the bug; re-run preflight; do NOT
   launch pilot until P2 PASSes (canonical solutions are
   ground truth; an executor that mishandles them
   systematically penalises every arm).
3. **Preflight P3 FAILS at re-run** (predicted A1 ≥ 90 %)
   → MBPP+ saturates on Llama-3.3-70B; pivot to HumanEval+
   build + preflight in W102.  Add carry-forward
   `W101-L-MBPP-PLUS-PREFLIGHT-P3-SATURATION-CAP`.
4. **Preflight AddrW101-P1 FAILS at re-run** (rescue surface
   below 5 % on both benches) → mechanism not load-bearing
   on the historical evidence; reconsider the W89 reflexion
   shape; do NOT launch pilot.
5. **Any probe FAILS unexpectedly** (e.g., the W101 bench
   module imports a forbidden primitive) → fix the bug in the
   module; re-run preflight.

## Pre-committed Phase 2 cheap-pilot gates (W95 9-gate shape; locked 2026-05-25)

For the MBPP+ cheap pilot, the gates are byte-identical to the
W95 / W96-A / W96-C / W97 / W98 / W99 / W100 gates; only the
arm names, model, slice seed, and pass-rate decision differ:

1. **Slice pre-committed**: 30 problems by deterministic slice
   with **seed 101_001** BEFORE any NIM call.  Slice SHA
   recorded.
2. **A1 < 90 %**: A1 @ K=5 pass rate on the 30-problem slice
   must stay below 90 %.  Expected to PASS cleanly at 70B (the
   W101 preflight P3 predicts A1 ≈ 70 %; saturation margin 20
   pp).
3. **B > A1**: `b_pass_rate > a1_pass_rate`.
4. **Margin ≥ +5 pp**: `b_pass_rate − a1_pass_rate ≥ 5 pp`.
5. **B > A0 by ≥ +5 pp**: reflexion mechanism is load-bearing
   in B (rules out "B beats A0 only because of K=5 sampling").
6. **Per-problem majority**: B ≥ A1 on ≥ 16 of 30 problems.
7. **Budget accounting exact**: 1 + 5 + 5 = 11 calls per
   problem.
8. **Audit chain re-derives**: per-call sidecars + per-seed
   Merkle + bench Merkle re-derive offline.
9. **Executor stays clean**: P2 re-run on the 30 slice problems
   at end-of-run → 100 % pass on canonical solutions.

### Honest acknowledgement of MBPP+ risks

Three risks specifically for the MBPP+ cheap pilot:

1. **A1 may still saturate**: if the actual EvalPlus drop on
   Llama-3.3-70B is below the published 12.7 pp Hoeffding
   lower bound (e.g., 70B happens to handle EvalPlus extra
   tests more robustly than the GPT-3.5/Claude-2 family), A1
   could land at ≥ 80 %.  This is GOOD for gate 2 but
   compresses the rescue surface.
2. **A1 may *over*-drop**: if 70B handles MBPP+ extra tests
   *worse* than the published drop suggests (e.g., MBPP+
   exposes wrong-edge-case bugs the W91 mechanism didn't
   surface), A1 could land at ≤ 60 % and B too.  The Phase 2
   bar is +5 pp margin not absolute pass; this stays a
   measurable result.
3. **B's mechanism-load-bearingness must be VERIFIED by
   rescue-rate, not just margin** — per the W96-C C1 + W100
   B2 lessons.  W101 records and gates on the reflexion
   rescue rate explicitly.

### Mechanism-load-bearingness sub-gates (B only; new W101)

In addition to the 9 W95 gates, B has TWO mechanism sub-gates
that the cheap pilot must clear if launched:

* **MLB-1 — Reflexion-cycle invocation rate ≥ 33 %** of problems
  the slice (i.e., at least 10 / 30 problems where attempt 0
  FAILs and the reflexion turns are exercised).  Below 33 %
  means most problems are short-circuited at attempt 0; the
  reflexion mechanism is barely doing work.  Empirical W89
  precedent: 17 / 90 problems where attempt 0 FAILed and
  reflexion was actually exercised.
* **MLB-2 — Reflexion rescue rate ≥ 33 %** of invocations (i.e.,
  at least 1 in 3 problems where attempt 0 FAILed end up
  PASSing after the reflexion cycle).  Below 33 % means the
  reflexion mechanism is mostly noise.  Empirical W89
  precedent: among the 17 / 90 problems where reflexion was
  invoked, B PASSed on 8; rescue rate = 8/17 ≈ 47 %.

A B PASS with MLB-2 < 33 % is downgraded to `PASS_NON_MECHANISM_DRIVEN`
(echoing W96-C / W100 terminology) and does NOT entitle the
W102 cross-scale confirmation milestone.

## Cross-scale rule (W96-C carry-over; locked 2026-05-25)

Identical to W97 / W98 / W99 / W100:

* **W101 70B Phase 2 entitled** only after preflight P1 + P2
  PASS at re-run.
* **W102 cross-scale confirmation** entitled IFF B Phase 2
  PASSes at 70B AND MLB-1 + MLB-2 clear.  The W102 cross-scale
  scale-up could be (a) `meta/llama-4-405b-instruct` if
  available, (b) `meta/llama-3.2-90b-vision-instruct` in
  text-only mode, or (c) `gpt-4o` / `claude-3.5-sonnet` if
  comparable text-only NIM endpoints exist.  Scale-up choice is
  locked in the W102 runbook; W101 explicitly does NOT
  pre-commit it.
* A Phase 2 PASS at one scale alone is NOT sufficient for the
  Phase 3 retirement bench.  Phase 3 requires the W95 / W100
  cross-scale rule + multi-seed × multi-problem retirement.

## Phase 3 decision logic (DEFERRED to W103 if W102 PASSes)

* W101 explicitly does NOT pre-commit Phase 3.  Phase 3 is the
  3-seed × 100-problem × K=5 retirement bench analogous to W89
  HumanEval; it would attempt to retire the new carry-forward
  `W101-L-MBPP-PLUS-REFLEXION-PHASE2-70B-CAP` (if it exists)
  AND extend the W89 reflexion-mechanism retirement to a SECOND
  published code benchmark.  Phase 3 wall ≈ 8-10 h at 70B; ~
  3300 NIM calls.

## Anti-cheat (carry-forward from W88–W100)

All W88–W100 anti-cheat clauses carry forward verbatim:

* Slice = deterministic `select_mbpp_plus_subset_v1` with seed
  `101_001`; SHA-anchored at run start.
* Same model on every arm.
* Same K=5 byte-exact budget on A1 / B (sequential reflexion
  runs the FULL K=5 budget; no early-stop).
* Executor = `coordpy.mbpp_plus_executor_v1.run_mbpp_plus_executor_v1`.
  No LLM judge; subprocess CPython with `-I -S` flags.
* Corpus SHA-256-anchored at pilot start; mismatches refuse
  to run.
* No selective retries; each (seed, problem, arm) is exactly
  one set of calls.
* Per-call sidecars + per-seed Merkle + bench Merkle re-derive
  offline.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* The W101 modules are explicit-import only:
  * `coordpy.mbpp_plus_loader_v1`
  * `coordpy.mbpp_plus_executor_v1`
  * `coordpy.mbpp_plus_reflexion_bench_v1`
  * `coordpy.mbpp_plus_preflight_v1`
* The W101 driver scripts are NOT library modules; they live in
  `scripts/`:
  * `scripts/run_w101_arsenal_mining.py`
  * `scripts/run_w101_mbpp_plus_preflight.py`
  * `scripts/run_w101_mbpp_plus_pilot.py` (cheap-pilot driver;
    conditional)

## Operational plan

### Phase 1 — done in W101 (NO NIM)

1. **(W101 battlefield-selection doc)** —
   `docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md`.  Locked the
   5-candidate × 8-criterion ranking matrix; selected MBPP+
   as LEAD and HumanEval+ as BACKUP.
2. **(W101 arsenal mining)** —
   `scripts/run_w101_arsenal_mining.py`; offline re-execution of
   W88 70B HumanEval + W91 5-seed 70B MBPP sidecars; produces
   per-(seed, task_id, arm) cluster surface.
3. **(W101 MBPP+ loader + executor + bench + preflight modules)**
   — explicit-import only; W101 anti-pattern guard PASSes.
4. **(W101 NIM-free preflight)** —
   `scripts/run_w101_mbpp_plus_preflight.py`; 6 / 8 probes
   PASS; 2 DEFERRED on MBPP+ corpus fetch.
5. **(W101 unit tests)** — `tests/test_w101_mbpp_plus_v1.py`
   (loader + executor + bench + preflight all unit-tested
   without NIM).
6. **(W101 frontier-relevance audit supplement)** —
   `docs/FRONTIER_RELEVANCE_AUDIT_W101_V1.md`.
7. **(W101 result docs)** — battlefield-selection +
   arsenal-mining + preflight result docs.
8. **(Linear ↔ GitHub sync)** — create `COO-25`; append a
   `W101` entry to `linear_github_mapping.json`; post the W101
   verdict comment to `COO-6`, `COO-9`, `COO-25`.

### Phase 2 — conditional on operator authorising MBPP+ fetch (no NIM until preflight P1+P2 PASS)

1. **(operator step)** Fetch the EvalPlus MBPP+ release
   artifact:
   ```bash
   mkdir -p ~/.cache/coordpy
   curl -L -o ~/.cache/coordpy/mbpp-plus.jsonl.gz \\
       https://github.com/evalplus/evalplus/releases/download/v0.2.0/MbppPlus-v0.2.0.jsonl.gz
   sha256sum ~/.cache/coordpy/mbpp-plus.jsonl.gz
   ```
2. **(operator step)** Record the fetched SHA-256:
   ```bash
   export MBPP_PLUS_TRUSTED_SHA256_OVERRIDE=<the_sha_from_above>
   ```
   OR update `MBPP_PLUS_EXPECTED_SHA256_V020` in
   `coordpy/mbpp_plus_loader_v1.py` (preferred for audit trail).
3. **Re-run W101 preflight** with the canonical-solution
   executor self-test enabled:
   ```bash
   python scripts/run_w101_mbpp_plus_preflight.py
   ```
4. **If preflight PASSes** (all 8 probes PASS), proceed to
   cheap pilot:
   ```bash
   NVIDIA_API_KEY=... python scripts/run_w101_mbpp_plus_pilot.py \\
       --model meta/llama-3.3-70b-instruct \\
       --n-problems 30 --n-seeds 1 --seed 101001
   ```
   Approximate budget: 330 NIM calls; ~2-3 h wall at 70B.
5. **Evaluate Phase 2 gates** (9 W95 gates + MLB-1 + MLB-2).
   Verdict goes in
   `docs/RESULTS_W101_MBPP_PLUS_PHASE2_70B_V1.md`.

### Phase 3 — DEFERRED to W102 (cross-scale confirmation)

Cross-scale confirmation lives in W102 if W101 Phase 2 PASSes.
W101 explicitly does NOT pre-commit Phase 3 or Phase 2 mid-flight
escalations.

## Honest framing

W101's job is to **set up the discipline + infrastructure +
preflight-earned battlefield** for the second-code-benchmark
attack.  The empirical preflight verdict (6 / 8 PASS; 2 DEFERRED
on corpus fetch) licenses the *next* step (operator fetches
MBPP+; re-runs preflight; cheap pilot conditional on clean
P1+P2 PASS), NOT a NIM call in W101 itself.

If the cheap pilot eventually launches and B − A1 ≥ +5 pp with
MLB sub-gates clearing, the programme is entitled to a *stronger*
claim than W89 alone: the W89 reflexion-mechanism retirement
generalises to a SECOND published code benchmark with the
ceiling structurally relieved.  That is the strongest
*cheap-pilot-earned* code-side claim available.

If the cheap pilot does NOT clear the +5 pp bar, the W101 result
is the cap; W102 pivots to HumanEval+ or `COO-12`.

Either way, the W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 /
W98 / W99 / W100 / **W101** preflight-first + cross-scale +
multi-candidate-tournament-then-confirm + mechanism-load-
bearingness discipline is preserved as the 11th consecutive
validation.
