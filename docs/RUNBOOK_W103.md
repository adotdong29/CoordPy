# W103 — HumanEval+ lead pilot + helper-consumption + code-line hardening (runbook)

> **Pre-commit contract for W103, locked 2026-05-25 BEFORE any
> W103 NIM call and BEFORE the W103 pilot driver is written.**
>
> W102 closed with the MBPP+ V2 cheap pilot FAIL
> (B − A1 = −6.67 pp; MLB-1 30 % FAIL; MLB-2 22.22 % FAIL) and
> the HumanEval+ backup-lane infrastructure preflight-PASSed
> 7 / 7 (verdict cid
> `4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4`).
> Per branch 3 of the pre-committed W102 RUNBOOK decision logic,
> **W103 lead path = HumanEval+ cheap pilot using the W102-built
> backup-lane infrastructure**.  `COO-9` remains the lead path
> (EvalPlus-family attack on the W91 base-MBPP cap is still the
> right direction; the question shifts to which EvalPlus
> benchmark is the right battlefield).
>
> W103 is NOT a one-step milestone.  It advances THREE lanes in
> the same milestone:
>
> 1. **Lead lane (HumanEval+)** — re-confirm the W102 preflight;
>    consume `coordpy.code_slice_selector_v1` as a real input,
>    not an afterthought; launch the HumanEval+ cheap pilot at
>    70B if preflight stays 7 / 7 + the slice is helper-anchored;
>    evaluate the pre-committed 9 Phase 2 gates + MLB-1 +
>    MLB-2 mechanism-load-bearingness sub-gates; apply the
>    pre-committed decision branch.
> 2. **Hardening lane (code-line discipline)** — codify the two
>    W102 lessons (silent-degeneration-via-schema-assumption +
>    cross-bench-arsenal-mining-priors-as-cheap-pilot-earning-
>    evidence) into durable guardrails so neither failure mode
>    can quietly recur.
> 3. **Planning lane (next-step escalation or fallback)** —
>    pre-commit the immediate W104 next move per pilot outcome:
>    cross-scale at 90B if PASS, mechanism-load-bearingness
>    refresh and code-line ranking if FAIL.
>
> The pilot is CONDITIONAL on the preflight re-confirmation AND
> on the slice being helper-anchored.  The hardening lane and
> planning lane ship UNCONDITIONALLY.
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.  Advanced work remains explicit-import
> only.

## Linear

* New issue **`COO-27`** (W103): HumanEval+ lead pilot at 70B +
  code-line hardening + W104 escalation/fallback pre-commit.
  Parent: `COO-6`.  High priority.
* Related: `COO-9` (lead path) — remains at High; W103 advances
  the EvalPlus-family attack on the next benchmark.
* Related: `COO-14` (helper) — first real downstream consumption
  of `coordpy.code_slice_selector_v1` in a pilot.
* Related: `COO-26` (W102; Done) — backup-lane infrastructure
  W103 consumes.
* Related: `COO-25` (W101; Done) — battlefield-selection matrix
  W103 inherits.

## What is NOT in scope (anti-drift contract)

W103 explicitly does NOT:

1. Re-open the cross-modal RealWorldQA arc.  RealWorldQA stays
   frozen at 11B per the W100 frontier audit.  W101 / W102 /
   W103 carry this verbatim.
2. Re-open the W95-B0 family, the typed-extract sub-family, or
   any RealWorldQA candidate.
3. Re-attempt MBPP+ V2 at 70B.  W102 FAIL is a cap; re-running
   on a fresh seed would be hope-driven, not evidence-earned.
4. Promote `COO-12` (substrate-level cross-modal injection)
   absent fresh evidence; `COO-12` stays Low.
5. Build APPS / LiveCodeBench / SWE-bench-lite infrastructure.
   The W101 battlefield-selection matrix locked these out of
   scope; W103 inherits that decision verbatim.
6. Bump `coordpy.__version__` or `SDK_VERSION`.
7. Publish to PyPI.
8. Edit `coordpy/__init__.py`.  Any new W103 modules are
   explicit-import only.
9. Re-introduce any anti-pattern under a prettier name
   (bounded windowing; compaction; generic prose summarization;
   shallow token compression; context-pruning theater; "cram
   less / truncate better").  The W97 – W102 frontier-relevance
   audits stay in force verbatim.
10. Launch Phase 3 (3 seeds × 100 problems × K=5) at W103.
    Phase 3 is W104+ only if W103 cheap pilot AND W104 cross-
    scale confirmation BOTH PASS.
11. Re-instate `coordpy.mbpp_plus_loader_v1` / `..._executor_v1`
    / `..._reflexion_bench_v1` as live infrastructure.  These
    are demoted to historical artifact + anti-pattern (silent-
    degeneration failure mode) per the W102 audit.
12. Re-rank candidate benches against fresh-K=5 cheap-pilot
    evidence WITHOUT honouring the W102 lesson that arsenal-
    mining priors are an UPPER BOUND on what the mechanism
    could produce on the new surface.  The +5 pp Phase 2 bar
    is earned by fresh-K=5 sampling only.

## Operational state (cheap evidence in hand BEFORE W103 starts)

| Field | Value |
|---|---|
| `coordpy.__version__` | `0.5.20` |
| `coordpy.SDK_VERSION` | `coordpy.sdk.v3.43` |
| W102 HumanEval+ preflight | **7 / 7 PASS** (`results/w102/humaneval_plus_preflight/w102_humaneval_plus_preflight_20260526T000500Z/verdict.json`; verdict cid `4f57a2cf...`) |
| W103 HumanEval+ preflight re-confirm | **7 / 7 PASS** (`results/w102/humaneval_plus_preflight/w102_humaneval_plus_preflight_20260526T015421Z/verdict.json`; verdict cid `4f57a2cf...` — identical) |
| HumanEval+ corpus SHA-256 (LFS oid) | `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492` |
| Predicted HumanEval+ A1@K=5 | 72.86 % (W88 70B HumanEval A1 mean 85.56 % − 12.7 pp Hoeffding lower bound) |
| Saturation margin (A1 < 90 %) | 17.14 pp |
| Cross-bench rescue prior (W89 base HumanEval) | 9.76 % rescue fraction; +5.56 pp B − A1; **the only confirmed multi-seed same-budget multi-agent superiority retirement** |
| W102 arsenal-mining HumanEval+ re-grade prior | +5.56 pp B − A1; 9.21 % rescue fraction on W88 responses re-graded against HumanEval+ extra-test surface |
| W103 helper-proposal humaneval_plus CID | `a5b3a2c15c4e3a0c3f33a47ed80334b759065b72daf76e2818a230d6a7256327` (28 unique task_ids over 4 clusters: 7 b_only_wins + 12 shared_fails + 2 a1_only_wins + 9 shared_wins; top-up of 2 to reach 30 from base humaneval helper-proposal) |
| W103 helper-proposal humaneval CID (top-up source) | `b7325b9646009a4a3fd71442cc55d3fd7c72a44690f6b6878ee5fb6d9ffcf607` |

## Critical W103 anti-pattern carry-forwards from W102

1. **`coordpy.mbpp_plus_loader_v1` is an anti-pattern**, not a
   loader.  The V1 schema (parallel `plus_input` / `plus_output`
   arrays) does NOT exist in the real EvalPlus release.  Against
   real data, V1 silently emits 0 plus-assertions; cheap-pilot
   would silently degenerate to base-MBPP.  W102 P5 + P6 probes
   are the structural defence; W103 EXTENDS this defence to
   HumanEval+ via a parallel anti-pattern test (see Hardening
   lane below).
2. **Cross-bench arsenal-mining priors are an UPPER BOUND, not
   a Phase 2 earning signal.**  W102 re-graded W91 responses
   showed B − A1 = +5.28 pp on the EvalPlus MBPP+ V2 extra-test
   surface; the fresh-K=5 cheap pilot at seed 101_001 produced
   B − A1 = −6.67 pp — an 11.95 pp swing below the prior.
   W103 records the W102 HumanEval+ re-grade prior (+5.56 pp;
   9.21 % rescue) as a PREFLIGHT input only — it earns the
   AddrW102-Hplus-W89-Rescue probe and the P3 saturation-margin
   check.  It does NOT earn the +5 pp Phase 2 margin bar.
   Fresh-K=5 sampling is the ground truth.
3. **MLB-2 rescue rate varies by benchmark family at 70B.**
   W102 MBPP+ V2 produced MLB-2 = 22.22 % (well below the W89
   HumanEval-family 47 % retirement template).  W103's
   HumanEval+ pilot reads against the W89 template; if MLB-2
   collapses similarly the verdict is `FAIL` even if margin
   gates pass.

## Lead lane — HumanEval+ cheap pilot at 70B (CONDITIONAL on preflight)

### Decision logic (pre-locked BEFORE driver is written)

1. **Pilot driver builds + unit-tested + Linear-synced**
   regardless of preflight outcome.  The driver mirrors
   `scripts/run_w102_mbpp_plus_v2_pilot.py` shape verbatim,
   adapted to the HumanEval+ bench surface.
2. **Preflight re-confirms 7 / 7 PASS** (achieved above;
   verdict cid `4f57a2cf...`).  Strictly the same verdict cid
   as the W102 run (deterministic; same corpus SHA + cache
   path).
3. **Slice is helper-anchored** (NEW W103 contract; this is the
   COO-14 downstream-consumption deliverable):
   * The pilot driver loads `proposals.json` produced by
     `scripts/run_w102_code_slice_proposal.py`.
   * Extracts `humaneval_plus` proposal entries in priority
     order (b_only_wins → shared_fails → a1_only_wins →
     shared_wins; the helper's locked composite-priority).
   * De-duplicates on `task_id` (some task_ids appear in
     multiple historical seeds).
   * Tops up to 30 problems using the base `humaneval`
     helper-proposal task_ids that are NOT already in the set.
   * Verifies every selected `task_id` is present in the
     SHA-pinned HumanEval+ corpus.
   * Computes a slice CID from the FINAL 30-tuple ordered by
     helper priority.
   * Refuses to run if any task_id is missing from the corpus
     or if the cluster mix degenerates to all `shared_wins`
     (the structural-defence guard).
4. **Cheap pilot launches** (1 seed × 30 problems × K=5 = 330
   NIM calls at `meta/llama-3.3-70b-instruct`; ~1-2 h wall
   subject to NIM throttling).  Single-seed slice with
   `--seed 103001` (preserves audit-chain isolation from
   W88 / W89 / W102 namespaces).
5. **Phase 2 9-gate + MLB-1 + MLB-2 sub-gates** evaluated per
   the W95-9-gate locked shape (carry-forward verbatim from
   W101 / W102).
6. **Decision branch** (pre-locked here; applied at verdict):
   * **Branch A — full PASS** (9 / 9 gates + MLB-1 + MLB-2 both
     clear): retire `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-
     CAP` is NOT possible (MBPP+ V2 is a different bench), but
     register
     `W103-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-70B-PASS`
     (1-seed cheap pilot at 70B).  W104 = HumanEval+
     cross-scale confirmation at a SECOND model class
     (provisionally `meta/llama-3.3-90b-instruct`-class; W103
     RUNBOOK does NOT lock the exact target model — that's
     W104's pre-commit work).  Phase 3 retirement bench
     (3 seeds × 100 problems × K=5) is W105+ and requires
     W103 PASS + W104 cross-scale PASS.
   * **Branch B — PASS with MLB-2 FAIL** (9 / 9 gates clear
     but reflexion rescue rate < 33 %): downgrade to
     `PASS_NON_MECHANISM_DRIVEN` (mirrors W96-C / W100
     precedent); W104 explores mechanism variations on
     HumanEval+ rather than cross-scale.  Add carry-forward
     `W103-L-HUMANEVAL-PLUS-MECHANISM-LOAD-BEARINGNESS-WEAK-
     AT-70B-CAP`.
   * **Branch C — pilot FAIL** (margin bar FAIL or per-problem-
     majority FAIL): add carry-forward
     `W103-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-70B-CAP`.
     This would be a MATERIALLY surprising outcome — the W89
     retirement on base HumanEval at +5.56 pp + the W102
     arsenal-mining +5.56 pp / 9.21 % rescue prior both predict
     a positive margin.  A FAIL here means cross-bench
     generalisation of the W89 mechanism is structurally
     bounded at the base-HumanEval template, with EvalPlus
     extras not helping.  W104 then refreshes the code-line
     ranking honestly (the W101 battlefield matrix is the
     starting point; the remaining candidates are APPS /
     LiveCodeBench / SWE-bench-lite — each of which the W101
     matrix ranked out of scope and would need its own
     justification).

### Phase 2 cheap-pilot gates (W95 9-gate shape; verbatim from W101 / W102)

1. **Slice pre-committed**: 30 problems by deterministic helper-
   anchored selection (above) BEFORE any NIM call.  Slice CID
   recorded at run start.
2. **A1 < 90 %**: A1 @ K=5 pass rate on the 30-problem slice
   must stay below 90 %.  Predicted HumanEval+ A1@K=5 = 72.86 %
   on the full corpus; on the helper-anchored slice (which is
   weighted toward historically-hard problems) A1 is expected
   to be LOWER, not higher — saturation risk is structurally
   small.
3. **B > A1**: `b_pass_rate > a1_pass_rate`.
4. **Margin ≥ +5 pp**: `b_pass_rate − a1_pass_rate ≥ 5 pp`.
5. **B > A0 by ≥ +5 pp**: reflexion mechanism is load-bearing.
6. **Per-problem majority**: B ≥ A1 on ≥ 16 of 30 problems.
7. **Budget accounting exact**: 1 + 5 + 5 = 11 calls per
   problem; 330 calls total.
8. **Audit chain re-derives**: per-call sidecars + per-seed
   Merkle + bench Merkle re-derive offline.
9. **Executor stays clean**: HumanEval+ canonical-solution
   self-test re-run at end-of-run → 100 % pass on the 30 slice
   problems.

### Mechanism-load-bearingness sub-gates (B only; W100 / W101 / W102 carry-forward verbatim)

* **MLB-1 — Reflexion-cycle invocation rate ≥ 33 %** of
  problems on the slice (≥ 10 / 30 problems where attempt 0
  FAILs and reflexion is exercised).
* **MLB-2 — Reflexion rescue rate ≥ 33 %** of MLB-1 numerator
  (≥ 1 in 3 reflexion-exercised problems end up PASSing).

A B PASS with MLB-2 < 33 % is downgraded to
`PASS_NON_MECHANISM_DRIVEN` (Branch B).

### Anti-cheat (verbatim from W88 – W102)

* Slice = helper-anchored from `proposals.json`; final task_id
  list SHA-anchored at run start.
* Same model on every arm.
* Same K=5 byte-exact budget on A1 / B (sequential reflexion
  runs the FULL K=5 budget; no early-stop).
* Executor = `coordpy.humaneval_plus_executor_v1.run_humaneval_
  plus_executor_v1`.  No LLM judge; subprocess CPython.
* Corpus SHA-256-anchored at pilot start; mismatches refuse to
  run.
* No selective retries; each (seed, problem, arm) is exactly
  one set of calls.
* Per-call sidecars + per-seed Merkle + bench Merkle re-derive
  offline.

## Hardening lane — code-line discipline (UNCONDITIONAL)

W102 surfaced two W101-level failure modes the programme can
NEVER step on quietly again:

1. **Silent-degeneration via schema assumption** — the V1 MBPP+
   loader assumed parallel `plus_input` / `plus_output` arrays;
   real data has neither.  W102 P5 + P6 probes catch this for
   MBPP+; W103 extends the structural defence to HumanEval+
   via:
   * A new unit test
     `test_w103_humaneval_plus_loader_no_silent_degeneration`
     that constructs a synthetic HumanEval+ row WITHOUT a
     `check(candidate)` block and asserts the executor's
     canonical-solution pass rate collapses to ≤ 5 % on that
     synthetic corpus (i.e., the silent-degradation failure
     mode is detectable in unit tests; would have been caught
     PRE-NIM if the V1 MBPP+ loader had this guard).
   * Confirmation that
     `coordpy.humaneval_plus_preflight_v1.probe_humaneval_plus_
     extra_test_surface_v1` already checks `def check(` is
     present in 95 %+ of rows (it does; this codifies it).
2. **Cross-bench arsenal-mining priors as cheap-pilot earning
   evidence** — W102 re-graded W91 responses showed +5.28 pp
   B − A1 on MBPP+ V2 extra tests; the fresh-K=5 cheap pilot
   at seed 101_001 produced −6.67 pp.  W103 codifies this
   anti-pattern via:
   * A new unit test
     `test_w103_arsenal_mining_prior_is_not_earning_evidence`
     that asserts the helper's composite-score `> 0` does NOT
     by itself license a pilot — the W93 preflight 5-gate +
     AddrW10X-Pn probes are the authoritative earning surface.
   * Provenance fields added to the pilot driver and verdict
     doc: `corpus_sha`, `helper_proposal_cid`,
     `mining_report_cid`, `preflight_verdict_cid`,
     `slice_cid`, `arsenal_mining_prior_b_minus_a1_pp`
     (RECORDED but explicitly NOT a gate input).

These two hardening additions are durable guardrails — they
prevent re-occurrence by making the failure mode visible at
test-collection time + at verdict-write time, NOT only at
post-pilot autopsy time.

### Hardening-lane deliverable (locked)

* `tests/test_w103_code_line_discipline_v1.py` (new test file;
  ≥ 6 tests covering silent-degeneration synthetic-row guard +
  arsenal-mining-prior-is-not-earning-evidence assertion +
  provenance-fields-present check + slice-helper-consumption
  validation + corpus-SHA-pin guard + helper-anti-pattern-
  guard sanity).
* `scripts/run_w103_humaneval_plus_pilot.py` — pilot driver
  WITH provenance fields written into the bench report.
* `docs/RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md` —
  verdict doc WITH the provenance fields verbatim.

## Planning lane — W104 pre-commit (UNCONDITIONAL)

The W104 next step is pre-committed here so the milestone
boundary doesn't drift on outcome.

### Branch A — W103 PASS_MECHANISM_DRIVEN (full PASS)

W104 = HumanEval+ cross-scale confirmation at a second
model-class.  Provisionally `meta/llama-3.1-90b-instruct` or
the next available 405B-class NIM endpoint.  W104 RUNBOOK
locks the exact target + preflight + slice rule (likely the
SAME helper-anchored slice as W103 to maximise per-problem
re-derivation power).  Phase 3 retirement bench (3 seeds ×
100 problems × K=5 on cross-scale) is W105+ work.

### Branch B — W103 PASS_NON_MECHANISM_DRIVEN

W104 = HumanEval+ mechanism-variation slate (parallel B-style
variants: B1 = enforced-reflexion-on-attempt-0 (no early-pass),
B2 = test-aware decomposition reader+solver on the EvalPlus
extra-test surface, B3 = sidecar-driven failure-cluster
targeting).  Pre-commit cheap-pilot earning rule: at least one
B-variant must lift MLB-2 ≥ 33 % AND keep margin ≥ +5 pp.

### Branch C — W103 FAIL

W104 = code-line ranking refresh.  The pre-committed candidate
set:

| Candidate | W101 matrix verdict | W103-FAIL re-eval triggers |
|---|---|---|
| **APPS** | Out of scope (C-grade infra cost) | Re-evaluate ONLY if HumanEval+ FAILs structurally similar to MBPP+ V2 (margin < 0 pp + MLB-2 < 33 %); otherwise APPS remains out of scope (infra burden does not justify the next attempt). |
| **LiveCodeBench** | Out of scope (time-anchored harness complexity) | Same condition as APPS; preferred over APPS if the answer is "mechanism is fine, ceiling pressure is wrong"; LiveCodeBench has lower per-problem ceiling at 70B. |
| **SWE-bench-lite** | Out of scope (F-grade decomposition fit) | STAYS out of scope unconditionally — the W89 reflexion mechanism does not have the structural shape to attack SWE-bench-lite's repo-level failure surface. |
| **W104 dispatch decision** | — | If HumanEval+ FAIL is mechanism-distribution (MLB-2 < 33 % + margin < +5 pp), W104 = LiveCodeBench preflight (NIM-free).  If HumanEval+ FAIL is ceiling-pressure (A1 > 90 %), W104 = APPS preflight (NIM-free).  If HumanEval+ FAIL is per-seed sampling (MLB-2 ≥ 33 % but margin < +5 pp), W104 = HumanEval+ multi-seed cheap confirmation at 70B (3 seeds × 30 problems × K=5 on the same helper-anchored slice). |

The dispatch decision is recorded in
`docs/RESULTS_W103_MILESTONE_SUMMARY_V1.md` post-pilot per the
applied branch.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* No new `coordpy.*` modules in W103.  The lead lane reuses
  W102 backup-lane modules verbatim.  The hardening lane lives
  in tests + driver + result docs.
* New W103 artefacts:
  * `scripts/run_w103_humaneval_plus_pilot.py` (lead-lane
    pilot driver; consumes W102 HumanEval+ infrastructure +
    `code_slice_selector_v1` helper).
  * `tests/test_w103_code_line_discipline_v1.py` (hardening-
    lane tests).
  * `docs/RUNBOOK_W103.md` (this file).
  * `docs/RESULTS_W103_HUMANEVAL_PLUS_PREFLIGHT_RECONFIRM_V1.md`
    (pre-pilot reconfirmation doc).
  * `docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md` (slice-
    selection helper-consumption attestation; the W103 COO-14
    downstream-consumption deliverable).
  * `docs/RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md` (pilot
    verdict doc; populated post-pilot).
  * `docs/RESULTS_W103_MILESTONE_SUMMARY_V1.md` (milestone
    summary; populated at close).
  * `docs/FRONTIER_RELEVANCE_AUDIT_W103_V1.md` (frontier audit
    supplement; 13th preflight-discipline validation).

## Cross-scale rule (W96-C carry-over; verbatim from W102)

* **W103 70B Phase 2 entitled** only after W102 HumanEval+
  preflight re-confirms 7 / 7 PASS (achieved 2026-05-25
  verdict cid `4f57a2cf...`).
* **W104 cross-scale confirmation** entitled IFF B Phase 2
  PASSes at 70B AND MLB-1 + MLB-2 both clear.  Scale-up
  choice is W104 runbook's job; W103 explicitly does NOT
  pre-commit the exact W104 target model.
* A Phase 2 PASS at one scale alone is NOT sufficient for the
  Phase 3 retirement bench.

## Operational plan

### Phase 1 — done in W103 (NO NIM)

1. **(W103 hardening lane)** —
   `tests/test_w103_code_line_discipline_v1.py`; ≥ 6 tests;
   all PASS.
2. **(W103 lead-lane driver)** —
   `scripts/run_w103_humaneval_plus_pilot.py`; helper-anchored
   slice; provenance fields recorded; refuses unpinned
   operation per W93 discipline.
3. **(W103 helper-consumption attestation)** —
   `docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md` documents the
   slice rule + final 30 task_ids + slice CID + helper-
   proposal CID + base-humaneval top-up entries.
4. **(W103 preflight reconfirmation)** —
   `docs/RESULTS_W103_HUMANEVAL_PLUS_PREFLIGHT_RECONFIRM_V1.md`
   records the 2026-05-25 fresh-run 7 / 7 PASS verdict cid.
5. **(W103 frontier-relevance audit supplement)** —
   `docs/FRONTIER_RELEVANCE_AUDIT_W103_V1.md`; 13th
   consecutive preflight-discipline validation.
6. **(Linear ↔ GitHub sync)** — create `COO-27`; append a
   `W103` entry to `linear_github_mapping.json`; post W103
   verdict comments to `COO-6`, `COO-9`, `COO-14`, `COO-26`,
   `COO-27`.

### Phase 2 — conditional on preflight reconfirmation + helper-anchored slice (achieved)

1. **Launch cheap HumanEval+ pilot** at 70B:
   ```bash
   NVIDIA_API_KEY=... python scripts/run_w103_humaneval_plus_pilot.py \
       --model meta/llama-3.3-70b-instruct \
       --slice-proposal-json results/w103/code_slice_proposals/latest_run/proposals.json \
       --n-problems 30 --seed 103001
   ```
2. **Evaluate Phase 2 gates** (9 W95 gates + MLB-1 + MLB-2).
   Verdict goes in
   `docs/RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md`.
3. **Apply pre-locked decision branch** (A / B / C above).

### Phase 3 — DEFERRED to W104+ (cross-scale + retirement)

W103 explicitly does NOT pre-commit Phase 3 or Phase 2 mid-
flight escalations.  W104 is pre-committed in the Planning
lane § above by the applicable branch.

## Pre-pilot prediction (recorded 2026-05-25 BEFORE W103 pilot)

> "Subjective priors over the HumanEval+ Phase 2 cheap pilot at
> 70B on a helper-anchored 30-problem slice with K=5,
> conditional on preflight re-confirmation:
>
> * Probability A1@K=5 clears the saturation gate (< 90 %):
>   **~ 95 %** (predicted A1 = 72.86 % on the full corpus;
>   helper-anchored slice skews toward historically-harder
>   problems so A1 is expected to be ~ 60 – 70 % on this slice).
> * Probability B beats A1 on the mean: **~ 80 %** (W89 base-
>   HumanEval at +5.56 pp is the closest precedent and is on
>   the same benchmark family; HumanEval+ extra tests give the
>   reflexion mechanism MORE failure signal per problem; helper
>   anchoring further concentrates the slice on the rescue
>   surface).
> * Probability B − A1 ≥ +5 pp: **~ 55-65 %** (lower than the
>   probability-of-direction because the W102 lesson is exactly
>   that fresh-K=5 sampling can swing many pp below an arsenal-
>   mining prior; the W102 swing was -11.95 pp; even half of
>   that on HumanEval+ would land at -0.5 pp which is FAIL).
> * Probability MLB-2 sub-gate clears (rescue rate ≥ 33 %):
>   **~ 70 %** (the W89 retirement was 47 % rescue rate on
>   HumanEval; HumanEval+ has more failure surface per problem
>   so the rescue surface should be at least as large; W102
>   showed MBPP+ V2 collapsed to 22.22 % rescue on the MBPP
>   family at 70B; HumanEval-family is structurally different
>   and the W89 template is on this family).
> * Probability the W103 verdict is a stronger claim than
>   currently licensed: **~ 50 %** — IF the pilot PASSes 9/9
>   + MLB-1 + MLB-2 at 70B AND cross-scale confirmation
>   (W104) PASSes, the programme would be entitled to claim
>   that the W89 sequential-reflexion mechanism extends to a
>   SECOND published code benchmark family (HumanEval+ EvalPlus-
>   hardened) at the cheap-pilot scale.  Retirement-grade
>   generalisation requires W104 + W105 multi-seed.  W103 alone
>   is NOT a multi-benchmark same-budget retirement.
>
> If W103 cheap pilot FAILs, the W103 verdict caps the W89
> mechanism at the base-HumanEval template (or at the per-seed-
> sampling variance ceiling).  The code-line ranking refresh
> in Branch C is the pre-committed fallback."

## Honest framing

W103's job is to:

1. **Honestly consume the COO-14 helper** as a real downstream
   input.  The W102 helper-lane produced a worked example with
   no real-pilot consumption; W103 makes it a load-bearing
   input.
2. **Honestly extend the W102 hardening discipline** to the
   HumanEval+ line so the W102 silent-degradation + arsenal-
   mining-prior anti-patterns cannot quietly recur on the next
   benchmark.
3. **Honestly pre-commit W104** by outcome so milestone
   boundaries don't drift on result.
4. **Launch the cheap pilot ONLY IF** preflight re-confirms +
   the slice is helper-anchored.  No buying long runs from
   hope.

If W103 PASSes, the programme is entitled to the *stronger*
claim that the W89 reflexion mechanism extends to a SECOND
published code benchmark family (HumanEval+ EvalPlus-hardened)
at the cheap-pilot scale.  Retirement-grade generalisation
requires W104 cross-scale + W105+ Phase 3 multi-seed.  W103
alone is NOT a multi-benchmark same-budget retirement.

If W103 FAILs, the W103 verdict is the cap on the W89 mechanism
cross-bench generalisation at the cheap-pilot scale, and the
code-line ranking refresh in Branch C is the pre-committed
fallback.  Either outcome preserves the W93 – W102 preflight-
first + cross-scale + multi-candidate-tournament-then-confirm
+ mechanism-load-bearingness + silent-degradation-anti-pattern-
guard + arsenal-mining-prior-anti-pattern-guard discipline as
the 13th consecutive validation.
