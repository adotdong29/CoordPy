# W102 — MBPP+ lead + HumanEval+ backup + COO-14 helper (runbook)

> **Pre-commit contract for W102, locked 2026-05-25 BEFORE any
> W102 NIM call and BEFORE any V2 code is built.**
>
> W101 closed with a 6/8 PASS preflight verdict on MBPP+ but
> with 2 DEFERRED gates (P1 corpus integrity + P2 executor
> self-test) waiting on operator fetch of the EvalPlus MBPP+
> release artifact.  The cheap NIM pilot was NOT YET earned.
>
> W102 advances THREE lanes in the same milestone:
>
> 1. **Lead lane (MBPP+)** — resolve the W101 deferred gates;
>    fix the W101 loader schema bug discovered during the W102
>    fetch attempt; re-run preflight under the corrected V2
>    infrastructure; if 8/8 PASS, the cheap pilot is genuinely
>    earned and launched; the next cross-scale path is
>    pre-committed.
> 2. **Backup lane (HumanEval+)** — full loader + executor +
>    reflexion bench + preflight infrastructure built in
>    parallel.  HumanEval+ is the second EvalPlus battlefield;
>    if MBPP+ FAILs unexpectedly, W103 attacks HumanEval+
>    immediately.
> 3. **Helper lane (COO-14)** — code-side slice-selection +
>    candidate-ranking helper built on top of the W101
>    arsenal-mining infrastructure; reused by every future
>    code-side milestone.
>
> All three lanes deliver real working infrastructure +
> unit-tested + Linear-synced.  The cheap pilot is conditional
> on lead-lane preflight clearing 8/8; the backup and helper
> lanes ship regardless.
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.

## Linear

* New issue **`COO-26`** (W102): MBPP+ V2 lead-lane execution +
  HumanEval+ backup-lane build + COO-14 helper-lane delivery.
  Parent: `COO-6`.  High priority.
* Related: `COO-25` (W101; Done) — infrastructure + preflight.
* Related: `COO-9` (lead path; W102 delivers the next step of
  its charter).
* Related: `COO-14` (helper) — W102 delivers the explicit
  Definition-of-Done deliverable for COO-14.
* Related: `COO-6` (hub) — comment posted on milestone close.

## What is NOT in scope (anti-drift contract)

This milestone explicitly does NOT:

1. Re-open the cross-modal RealWorldQA arc.  Cross-modal
   RealWorldQA stays frozen at 11B per the W100 frontier
   audit; W101 + W102 carry this verbatim.
2. Re-open the W95-B0 family, the typed-extract sub-family
   (D2-B0 + W98 B1 + W99 B4), or any RealWorldQA candidate.
3. Promote `COO-12` (substrate-level cross-modal injection)
   absent fresh evidence; `COO-12` stays Low.
4. Build APPS / LiveCodeBench / SWE-bench-lite infrastructure.
   The W101 battlefield-selection matrix locked these out of
   scope; W102 inherits that decision verbatim.
5. Bump `coordpy.__version__` or `SDK_VERSION`.
6. Publish to PyPI.
7. Edit `coordpy/__init__.py`.  All new W102 modules are
   explicit-import only.
8. Re-introduce any anti-pattern under a prettier name
   (bounded windowing; compaction; generic prose summarization;
   shallow token compression; context-pruning theater; "cram
   less / truncate better").  W97–W101 frontier-relevance
   audits remain in force verbatim; W102 carries them forward.
9. Launch a Phase 3 retirement bench.  Phase 3 is W103+ if
   W102 cheap pilot AND cross-scale confirmation both PASS.

## Critical W102 finding (locked BEFORE any V2 build)

While performing the W101 deferred operator step (fetch the
EvalPlus MBPP+ release artifact + record its SHA-256 pin), the
W102 fetch path discovered:

* The W101-pinned URL
  `https://github.com/evalplus/evalplus/releases/download/v0.2.0/MbppPlus-v0.2.0.jsonl.gz`
  returns HTTP 404.  The EvalPlus GitHub releases contain
  only model-output zip files, not the dataset.
* The canonical dataset distribution is on Hugging Face:
  * MBPP+ — `https://huggingface.co/datasets/evalplus/mbppplus/resolve/main/data/test-00000-of-00001-d5781c9c51e02795.parquet`
    (378 rows; LFS SHA-256 oid
    `dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`).
  * HumanEval+ — `https://huggingface.co/datasets/evalplus/humanevalplus/resolve/main/test.jsonl`
    (164 rows; LFS SHA-256 oid
    `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492`).
* The fetched MBPP+ parquet schema is `{task_id, code, prompt,
  source_file, test_imports, test_list, test}`.  `test_list`
  carries 3-ish BASE assertions (identical structure to
  base-MBPP-sanitized).  `test` is a single Python program
  that defines `inputs` + `results` arrays and iterates calling
  the entry-point function with each input — this is where
  the EvalPlus hidden tests live.
* The W101 V1 loader expects parallel `plus_input` /
  `plus_output` arrays.  **Those fields do not exist in the
  actual EvalPlus dataset.**  Against real data, V1's
  `build_plus_assertions` returns `[]`, and the executor
  silently passes any candidate that satisfies the 3 base
  assertions alone.  The V1 cheap pilot, had it been launched,
  would have silently degenerated to a base-MBPP pilot — the
  exact SATURATED-CEILING regime W101 was designed to attack.

**This is a serious infrastructure bug discovered honestly via
the W102 fetch step.  W102 MUST fix it BEFORE any NIM spend.**

The fix is V2 infrastructure that consumes the actual
EvalPlus schema:

* `coordpy.mbpp_plus_loader_v2.py` — reads the HF parquet (or
  the same parquet from a local SHA-pinned cache); exposes
  `MbppPlusProblemV2` carrying `task_id`, `prompt`,
  `canonical_code`, `base_test_list`, `extra_test_program`,
  `entry_point`.
* `coordpy.mbpp_plus_executor_v2.py` — runs a candidate against
  EITHER base assertions, OR the extra `test` program (the
  iterative `inputs`/`results` loop), OR both, in a fresh
  CPython subprocess.  Modes: `base_only` / `plus_only` /
  `base_and_plus`.
* `coordpy.mbpp_plus_reflexion_bench_v2.py` — wires V2 loader +
  V2 executor with the W89 sequential-reflexion mechanism;
  byte-identical to V1 mechanism shape; only the executor
  surface differs.
* `coordpy.mbpp_plus_preflight_v2.py` — extends V1 preflight
  with two NEW probes:
  * **P5 — extra-test-surface integrity**: ≥ 95 % of rows
    must carry a non-empty `test` program with at least one
    detectable iteration line.  Catches the V1 silent-
    degradation failure mode directly.
  * **P6 — V1-vs-V2 canonical-solution agreement**: V2
    executor in `plus_only` mode must pass the canonical
    `code` on ≥ 95 % of rows (sanity).

V1 stays in-repo as a documented anti-pattern + historical
artifact + explicit-import-only.  V1's failure mode is the
template for W102's new P5 + P6 probes; the V1 → V2 contrast
is documented in
`docs/RESULTS_W102_MBPP_PLUS_LOADER_V2_FIX_V1.md`.

## Lead lane — MBPP+ V2 (CONDITIONAL on preflight)

### Decision logic (PRE-LOCKED before V2 build)

1. **V2 infrastructure builds + unit-tested + Linear-synced**
   regardless of preflight outcome.  This is the W102
   lead-lane discipline deliverable.
2. **V2 preflight runs (NIM-free)**:
   * `P1` — corpus integrity (MBPP+ cache present + parquet
     SHA matches pin).
   * `P2` — executor self-test on canonical solutions (V2
     executor in `base_and_plus` mode → ≥ 98 % pass on
     canonical `code` field).
   * `P3` / `P4` — unchanged from V1.
   * `P5` (NEW) — extra-test-surface integrity (≥ 95 % rows
     carry non-empty `test` program).
   * `P6` (NEW) — V1-vs-V2 canonical-solution agreement (≥ 95 %
     parity on `code` field under `plus_only` mode).
   * `AddrW101-P1..P4` — unchanged from V1.
3. **Cheap pilot ENTITLED iff all 10 probes PASS** (P1+P2+P3+
   P4+P5+P6 + AddrW101-P1..P4).  Strictly stricter than W101
   (which allowed 1 DEFERRED).  W102 has no DEFERREDs once
   corpus is fetched + SHA pin recorded.
4. **Cheap pilot launches under V2 bench** (1 seed × 30
   problems × K=5 ≈ 330 NIM calls at 70B; ~2-3 h wall).
   Same model on every arm; same K=5 budget; deterministic
   slice with seed 101_001; W101 9-gate Phase 2 shape +
   MLB-1 + MLB-2 sub-gates evaluated.
5. **Cross-scale W103 path pre-committed (this runbook)**:
   * If cheap pilot PASSes (B − A1 ≥ +5 pp AND MLB sub-gates
     clear) → W103 is cross-scale confirmation at a SECOND
     model class.  The 70B retirement of `W91-L-MBPP-
     REFLEXION-V2-5SEED-PARTIAL-CAP` requires 3 seeds × 100
     problems at the same regime, which is a Phase 3 W104+
     milestone — NOT W103.  W103 first confirms the cross-
     scale lift IN THE CHEAP PILOT BUDGET.
   * If cheap pilot PASSes B − A1 ≥ +5 pp BUT MLB-2 FAILs
     (verifier-style non-mechanism-driven pass) → mark as
     `PASS_NON_MECHANISM_DRIVEN`; W103 does NOT cross-scale.
     Pivot to HumanEval+ in W103.
   * If cheap pilot FAILs (B − A1 < +5 pp) → record cap
     `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`; W103
     pivots to HumanEval+ via the backup lane built in W102.

### Phase 2 cheap-pilot gates (W95 9-gate shape; locked verbatim from W101)

1. **Slice pre-committed**: 30 problems by deterministic slice
   with **seed 101_001** BEFORE any NIM call.  Slice SHA
   recorded.
2. **A1 < 90 %**: A1 @ K=5 pass rate on the 30-problem slice
   must stay below 90 %.
3. **B > A1**: `b_pass_rate > a1_pass_rate`.
4. **Margin ≥ +5 pp**: `b_pass_rate − a1_pass_rate ≥ 5 pp`.
5. **B > A0 by ≥ +5 pp**: reflexion mechanism is load-bearing.
6. **Per-problem majority**: B ≥ A1 on ≥ 16 of 30 problems.
7. **Budget accounting exact**: 1 + 5 + 5 = 11 calls per
   problem.
8. **Audit chain re-derives**: per-call sidecars + per-seed
   Merkle + bench Merkle re-derive offline.
9. **Executor stays clean**: P2 re-run on the 30 slice problems
   at end-of-run → 100 % pass on canonical solutions.

### Mechanism-load-bearingness sub-gates (B only; locked verbatim from W101)

* **MLB-1 — Reflexion-cycle invocation rate ≥ 33 %** of
  problems on the slice.
* **MLB-2 — Reflexion rescue rate ≥ 33 %** of invocations.

A B PASS with MLB-2 < 33 % is downgraded to
`PASS_NON_MECHANISM_DRIVEN` (echoing W96-C / W100 / W101
precedent).

## Backup lane — HumanEval+ infrastructure (UNCONDITIONAL)

Built regardless of MBPP+ cheap pilot outcome.  The backup
lane delivers:

* `coordpy.humaneval_plus_loader_v1.py` — SHA-pinned HF
  release loader for `evalplus/humanevalplus/test.jsonl`
  (164 rows; LFS oid
  `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492`).
* `coordpy.humaneval_plus_executor_v1.py` — subprocess CPython
  executor that runs candidate + the row's `test` field
  (which defines `check(candidate)`) and calls
  `check(<entry_point>)`.  Mirrors W86 HumanEval executor
  shape verbatim; the only difference is the `test` field's
  content (EvalPlus hidden tests vs base HumanEval `check`).
* `coordpy.humaneval_plus_reflexion_bench_v1.py` — wires the
  loader + executor with the W89 sequential-reflexion
  mechanism; byte-identical mechanism shape; A0 / A1 / B arms
  identical to W88.
* `coordpy.humaneval_plus_preflight_v1.py` — same W93 5-gate
  + AddrW102-Hplus-P1..P4 shape as the W101 MBPP+ preflight,
  re-purposed for HumanEval+.

### A0 / A1 / B definitions (pre-committed; identical to W88 verbatim)

* **A0** — `meta/llama-3.3-70b-instruct` at T=0.0, K=1.
* **A1** — `meta/llama-3.3-70b-instruct` at T=0.7, K=5
  first-pass-among-K self-consistency.
* **B** — sequential-reflexion-K=5 at T=0.7, each turn
  conditioned on cumulative (candidate, executor_stderr)
  history.

Same model on every arm; same K=5 budget; deterministic slice
with seed 102_001 (HumanEval+ uses a fresh seed namespace to
preserve W88 / W89 audit isolation).

### HumanEval+ battlefield-specific ranking

Per the W101 battlefield-selection matrix
(`docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md`), HumanEval+
ranks:

| Criterion | Grade | Notes |
|---|---|---|
| C1 ceiling pressure | B | published HumanEval+ drop ~14 pp from base; W89 A1 mean ≈ 85.56 % drops to ≈ 71-72 % on HumanEval+. |
| C2 executor cleanness | A | subprocess Python + `check()` block; no LLM judge. |
| C3 decomposition fit (W89 shape) | A | byte-identical port. |
| C4 per-problem failure surface | B | at A1 ≈ 72 % on 30 problems → expected unique-A1-failures ≈ 8-9. |
| C5 reproducibility | A | single JSONL.  SHA-pinned via HF LFS oid. |
| C6 W93 preflight compatibility | A | A1@K=5 residual computable from W88 70B HumanEval sidecar re-executed against HumanEval+ extra `check()` (W102 helper-lane builds this).  Same shape as MBPP+ P3. |
| C7 MBPP-ceiling-trap avoidance | A | EvalPlus relieves HumanEval ceiling. |
| C8 cheap-pilot budget | A | same 330 calls × 70B as W89 / W91 / W101. |

**Overall**: BACKUP (one B-grade short of LEAD on per-problem
failure surface size — HumanEval+ failure surface is smaller
than MBPP+ because base HumanEval is already harder than base
MBPP at 70B).  MBPP+ stays LEAD; HumanEval+ stays BACKUP.

### Cheap probes (NIM-free) ready in W102

* Loader integrity P1 + executor self-test P2 against the
  cached HumanEval+ JSONL.
* P3 residual estimate (W88 70B HumanEval A1=85.56 % − 14 pp
  published EvalPlus drop = predicted HumanEval+ A1 ≈ 71.6 %;
  saturation margin 18.4 pp).
* P4 decomposition argument (same structural argument as
  MBPP+ but with HumanEval-as-base prior).
* P5 extra-test-surface integrity (same probe shape as the
  W102 MBPP+ V2 P5).
* AddrW102-Hplus-P1..P4 (same shape as W101 AddrW101-P1..P4
  but with W88 70B HumanEval sidecar replacing W91 MBPP).

HumanEval+ preflight is **built + run** in W102 but the cheap
NIM pilot for HumanEval+ is **NOT launched in W102** — it is
the W103 fallback if MBPP+ fails.  This preserves
expensive-run discipline.

## Helper lane — COO-14 code-side slice-selection + candidate-ranking helper

W102 ships `coordpy.code_slice_selector_v1.py` (NEW; explicit-
import only).  Implements the four-item COO-14 Definition of
Done verbatim:

1. **Rank candidate directions cheaply** using committed
   W88–W93 + W101 evidence: given an arsenal-mining report
   (per `scripts/run_w101_arsenal_mining.py`), produces a
   ranked candidate-direction table per bench, ordered by
   {rescue_fraction, hard_cluster_size, mean_b_minus_a1_pp,
   per-seed margin variance}.
2. **Select failure-cluster slices for pilots**: given a
   per-(seed, task_id, arm) cluster surface, returns the
   top-K problems by {unique-B-rescue, hard-cluster, A1-only}
   subject to a cheap-pilot budget (default n_problems=30;
   K=5 cycles; ≤ 660 NIM calls).
3. **"What exact problems should the next cheap pilot
   attack?"**: explicit API
   `propose_cheap_pilot_slice(bench, n_problems)` returns the
   (seed, task_id) tuples + a justification string per
   problem (which cluster it came from + why it earns the
   slot).
4. **Feed output into runbooks before expensive runs**:
   serialises to a runbook-ready Markdown table via
   `format_slice_proposal_markdown(...)`; W103+ runbooks can
   `include` this table verbatim.

The helper is **stand-alone** (no NIM; no expensive bench;
no model loading) and is **unit-tested** end-to-end on
synthetic + the real W101 arsenal-mining report.

### Helper-lane deliverable (locked)

* `coordpy.code_slice_selector_v1.py` (new module; explicit-
  import only).
* `scripts/run_w102_code_slice_proposal.py` (driver that
  produces the W103+ pilot-slice proposal from the W101
  arsenal-mining report).
* `tests/test_w102_code_slice_selector_v1.py` (≥ 8 tests).
* `docs/RESULTS_W102_CODE_SLICE_SELECTOR_V1.md` (helper-lane
  result doc; documents the API + a worked example).

## W102 arsenal mining (extension of W101)

`scripts/run_w102_arsenal_mining.py` extends
`scripts/run_w101_arsenal_mining.py` with TWO new offline
cluster surfaces:

1. **W88 70B HumanEval candidates re-executed against
   HumanEval+ `check()`** — cross-bench failure-residual
   surface for the backup lane preflight P3.  990 candidate
   responses × 1 subprocess executor call each.  No NIM.
2. **W91 5-seed 70B MBPP candidates re-executed against
   MBPP+ V2 `test` program** — cross-bench failure-residual
   surface for the lead lane preflight P3 / P6.  1650
   candidate responses × 1 subprocess executor call each.
   No NIM.

Output:
`results/w102/arsenal_mining/<RUN>/mining_report.json` —
extends the W101 report shape with `humaneval_plus` and
`mbpp_plus_v2` blocks alongside the W101 `humaneval` and
`mbpp` blocks.  The helper-lane slice selector consumes this
extended report.

## Pre-pilot prediction (recorded 2026-05-25 BEFORE V2 preflight)

> "Subjective priors over the MBPP+ V2 Phase 2 cheap pilot at
> 70B on a 1-seed × 30-problem slice with K=5, conditional on
> V2 preflight PASSing 10/10:
>
> * Probability A1@K=5 clears the saturation gate (< 90 %):
>   **~ 88 %** (preflight predicts ~ 70 % A1; the V2 extra-
>   test surface is the literature's hardened MBPP+ exact
>   distribution, not a parallel-array approximation).
> * Probability B beats A1 on the mean: **~ 75 %** (the W89
>   retirement on HumanEval is the closest empirical
>   precedent; MBPP+ V2 extra tests give the reflexion
>   mechanism more failure signal to read).
> * Probability B − A1 ≥ +5 pp: **~ 50-60 %** (somewhat lower
>   than the W101 prior because the V1 → V2 schema correction
>   may reveal the W91 base-MBPP cap was less ceiling-driven
>   and more mechanism-distribution-driven than the W101
>   battlefield argument estimated).
> * Probability the W101 silent-degradation bug was the
>   binding cause of the W91 base-MBPP cap: **~ 20 %** (most
>   of the W91 cap was true ceiling saturation; the V1 bug
>   would only show up when running against the actual
>   EvalPlus extra tests; W91 ran against base-MBPP-sanitized).
> * Probability MLB-2 sub-gate clears (rescue rate ≥ 33 %):
>   **~ 65 %** (the W89 precedent was 47 % rescue rate; W101
>   arsenal mining showed W91 base-MBPP had only 28 %, which
>   the MBPP+ extra-test surface should raise back).
> * Probability of cross-scale (W103) being *necessary* if
>   W102 PASSes: **100 %** — per the W96-C / W100 cross-scale
>   discipline.
>
> If MBPP+ V2 cheap pilot PASSes the 9 Phase 2 gates with MLB
> sub-gates clearing, W103 = MBPP+ V2 cross-scale confirmation
> at the next model class; W104 = MBPP+ V2 Phase 3 retirement
> bench (3 seeds × 100 problems × K=5) if W103 cross-scale
> PASSes.  If MBPP+ V2 cheap pilot FAILS, the W102 verdict
> is the cap; W103 attacks HumanEval+ using the backup-lane
> infrastructure W102 already ships."

## Cross-scale rule (W96-C carry-over; locked 2026-05-25)

Identical to W97 / W98 / W99 / W100 / W101:

* **W102 70B Phase 2 entitled** only after V2 preflight
  PASSes 10/10.
* **W103 cross-scale confirmation** entitled IFF B Phase 2
  PASSes at 70B AND MLB-1 + MLB-2 both clear.  Scale-up
  choice is W103 runbook's job; W102 explicitly does NOT
  pre-commit it.
* A Phase 2 PASS at one scale alone is NOT sufficient for
  the Phase 3 retirement bench.

## Anti-cheat (carry-forward from W88–W101 verbatim)

All anti-cheat clauses carry forward verbatim:

* Slice = deterministic `select_mbpp_plus_subset_v2` with seed
  `101_001`; SHA-anchored at run start.
* Same model on every arm.
* Same K=5 byte-exact budget on A1 / B (sequential reflexion
  runs the FULL K=5 budget; no early-stop).
* Executor = `coordpy.mbpp_plus_executor_v2.run_mbpp_plus_executor_v2`.
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
* W102 modules are explicit-import only:
  * `coordpy.mbpp_plus_loader_v2`
  * `coordpy.mbpp_plus_executor_v2`
  * `coordpy.mbpp_plus_reflexion_bench_v2`
  * `coordpy.mbpp_plus_preflight_v2`
  * `coordpy.humaneval_plus_loader_v1`
  * `coordpy.humaneval_plus_executor_v1`
  * `coordpy.humaneval_plus_reflexion_bench_v1`
  * `coordpy.humaneval_plus_preflight_v1`
  * `coordpy.code_slice_selector_v1`
* W102 driver scripts live in `scripts/`:
  * `scripts/run_w102_arsenal_mining.py`
  * `scripts/run_w102_mbpp_plus_v2_preflight.py`
  * `scripts/run_w102_humaneval_plus_preflight.py`
  * `scripts/run_w102_mbpp_plus_v2_pilot.py` (cheap-pilot
    driver; conditional on V2 preflight 10/10 PASS)
  * `scripts/run_w102_code_slice_proposal.py` (helper-lane
    driver)

## Operational plan

### Phase 1 — done in W102 (NO NIM)

1. **(W102 lead-lane V2 build)** — loader V2 + executor V2 +
   reflexion bench V2 + preflight V2 modules; unit-tested.
2. **(W102 backup-lane build)** — HumanEval+ loader +
   executor + bench + preflight modules; unit-tested.
3. **(W102 helper-lane build)** — `code_slice_selector_v1`
   module + driver + result doc; unit-tested.
4. **(W102 arsenal mining)** —
   `scripts/run_w102_arsenal_mining.py`; extends W101 mining
   with HumanEval+ and MBPP+ V2 cross-bench cluster surfaces.
5. **(W102 lead-lane V2 preflight)** —
   `scripts/run_w102_mbpp_plus_v2_preflight.py`; expected
   to PASS 10/10 if the V2 schema is correct.
6. **(W102 backup-lane preflight)** —
   `scripts/run_w102_humaneval_plus_preflight.py`; expected
   to PASS 8/8 cleanly (HumanEval+ schema is well-understood
   from the W88 retirement template).
7. **(W102 frontier-relevance audit supplement)** —
   `docs/FRONTIER_RELEVANCE_AUDIT_W102_V1.md`.
8. **(W102 result docs)** — V2 fix doc + arsenal mining doc +
   preflight docs + helper-lane doc + milestone summary.
9. **(Linear ↔ GitHub sync)** — create `COO-26`; append a
   `W102` entry to `linear_github_mapping.json`; post the
   W102 verdict comment to `COO-6`, `COO-9`, `COO-14`,
   `COO-25`, `COO-26`.

### Phase 2 — conditional on V2 preflight 10/10 PASS

1. **Launch cheap MBPP+ V2 pilot** (1 seed × 30 problems ×
   K=5 ≈ 330 NIM calls at 70B; ~2-3 h wall):
   ```bash
   NVIDIA_API_KEY=... python scripts/run_w102_mbpp_plus_v2_pilot.py \
       --model meta/llama-3.3-70b-instruct \
       --n-problems 30 --seed 101001
   ```
2. **Evaluate Phase 2 gates** (9 W95 gates + MLB-1 + MLB-2).
   Verdict goes in
   `docs/RESULTS_W102_MBPP_PLUS_V2_PHASE2_70B_V1.md`.
3. **Decision applied per the pre-locked W102 decision
   logic** (this runbook §"Lead lane — MBPP+ V2 (CONDITIONAL
   on preflight)").

### Phase 3 — DEFERRED to W103+ (cross-scale + retirement)

W102 explicitly does NOT pre-commit Phase 3 or Phase 2 mid-
flight escalations.

## Honest framing

W102's job is to:

1. **Honestly fix the W101 silent-degradation bug** in the
   MBPP+ loader + executor before any NIM spend.  This is
   exactly the W93 preflight-first discipline in action.
2. **Resolve the W101 deferred gates** with the real
   EvalPlus corpus + SHA pin recorded.
3. **Build the backup lane (HumanEval+)** so W103 has a real
   alternative if MBPP+ fails.
4. **Ship the helper lane (COO-14)** so future code-side
   milestones are cheaper + sharper.
5. **Launch the cheap pilot ONLY IF** the V2 preflight
   genuinely earns it.  No buying long runs from hope.

If the cheap pilot PASSes, the programme is entitled to the
*stronger* claim that the W89 reflexion mechanism extends to
a SECOND published code benchmark family (MBPP+ EvalPlus-
hardened) at the cheap-pilot scale.  Retirement-grade
generalisation requires W103 cross-scale + W104+ Phase 3
multi-seed.  W102 alone is NOT a multi-benchmark same-budget
retirement.

If the cheap pilot FAILS, the W102 verdict is the cap; W103
attacks HumanEval+ via the backup-lane infrastructure already
shipped.  Either outcome preserves the W93–W101 preflight-
first + cross-scale + multi-candidate-tournament-then-confirm
+ mechanism-load-bearingness discipline as the 12th
consecutive validation.
