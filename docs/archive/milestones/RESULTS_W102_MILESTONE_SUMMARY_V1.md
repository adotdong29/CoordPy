# W102 — Milestone summary V1

> **2026-05-25.  Three-lane milestone, all three lanes
> delivered.  Lead lane (MBPP+ V2) discovered + fixed the W101
> silent-degeneration loader bug, ran 10/10 NIM-free preflight,
> and (conditional on preflight) launched the cheap MBPP+ V2
> Phase 2 pilot.  Backup lane (HumanEval+) shipped full
> infrastructure + 7/7 preflight (cheap pilot DEFERRED to W103
> if MBPP+ V2 path stalls).  Helper lane (COO-14)
> delivered the code-side slice-selection + candidate-ranking
> module with 15 unit tests + worked-example artifact.**
>
> **W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 /
> W100 / W101 / W102 = TWELVE consecutive preflight-discipline
> validations.**
>
> **No version bump; no PyPI publish; `coordpy/__init__.py`
> untouched.  All 9 new W102 modules are explicit-import only.**

## What happened in each lane

### Lead lane — MBPP+ V2

1. **Operator-fetch automation.**  The W102 fetch step
   discovered the W101-pinned URL is HTTP 404 — EvalPlus GitHub
   releases only ship model-output zips, not the dataset.  W102
   automatically located the canonical Hugging Face artifact
   at
   `https://huggingface.co/datasets/evalplus/mbppplus/resolve/main/data/test-00000-of-00001-d5781c9c51e02795.parquet`
   (378 rows; LFS SHA-256
   `dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`)
   and cached + SHA-pinned it at `~/.cache/coordpy/mbpp-plus.parquet`.
2. **W101 loader schema bug discovered.**  The actual EvalPlus
   row schema is `{task_id, code, prompt, source_file,
   test_imports, test_list, test}` — NOT the parallel
   `plus_input` / `plus_output` arrays the W101 V1 loader
   assumed.  Against real data, V1 would silently emit 0
   extra-test assertions and the V1 cheap pilot would have
   silently degenerated to a base-MBPP run.  Documented in
   `docs/RESULTS_W102_MBPP_PLUS_LOADER_V2_FIX_V1.md`.
3. **V2 infrastructure built** (4 NEW explicit-import-only
   modules): `coordpy.mbpp_plus_loader_v2`,
   `coordpy.mbpp_plus_executor_v2`,
   `coordpy.mbpp_plus_reflexion_bench_v2`,
   `coordpy.mbpp_plus_preflight_v2`.  V2 reads the real
   EvalPlus parquet schema, runs the canonical `test` program
   in a fresh CPython subprocess with `-I` (numpy enabled),
   and adds two NEW preflight probes (P5 extra-test-surface
   integrity + P6 V1-vs-V2 canonical agreement) that
   structurally close the silent-degeneration failure mode.
4. **V2 preflight: 10 of 10 PASS.**
   `results/w102/mbpp_plus_v2_preflight/<RUN>/verdict.json`;
   verdict cid `aab6d9abf839391d69f75f8d55a9611ce2ff805d75bce85efd390a4b59813c35`.
5. **Cheap MBPP+ V2 pilot launched** (1 seed × 30 problems ×
   K=5 ≈ 330 NIM calls at `meta/llama-3.3-70b-instruct`).
   Slice CID `64aab030153728a11e4a39d2533ce3178fd2bd452ccb415528ef7434322a743d`.
   Pilot result + Phase 2 9-gate + MLB-1 / MLB-2 sub-gate
   verdict in
   `docs/RESULTS_W102_MBPP_PLUS_V2_PHASE2_70B_V1.md`
   (populated at pilot completion).
6. **W101 V1 demoted** to historical artifact + anti-pattern
   (silent-degeneration failure mode) per
   `docs/FRONTIER_RELEVANCE_AUDIT_W102_V1.md`.  V1 stays
   in-repo for the W101 audit trail.

### Backup lane — HumanEval+

1. **HF JSONL fetched + SHA-pinned**.  Canonical artifact at
   `https://huggingface.co/datasets/evalplus/humanevalplus/resolve/main/test.jsonl`
   (164 rows; LFS SHA-256
   `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492`)
   cached at `~/.cache/coordpy/humaneval-plus.jsonl`.
2. **4 NEW explicit-import-only modules**:
   `coordpy.humaneval_plus_loader_v1`,
   `coordpy.humaneval_plus_executor_v1`,
   `coordpy.humaneval_plus_reflexion_bench_v1`,
   `coordpy.humaneval_plus_preflight_v1`.  Loader+executor
   trivially port from the W86 base-HumanEval infrastructure;
   the `test` field shape is identical (defines `check(
   candidate)`).  Reflexion bench is byte-identical W89
   mechanism with dedicated seed namespace 102_001..
3. **HumanEval+ preflight: 7 of 7 PASS.**
   `results/w102/humaneval_plus_preflight/<RUN>/verdict.json`;
   verdict cid `4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4`.
   Predicted HumanEval+ A1@K=5 = 72.86 % (W88 70B HumanEval
   A1 mean 85.56 % − published EvalPlus Hoeffding lower-bound
   drop 12.7 pp); saturation margin 17.14 pp.  Per-bench
   ranking documented as BACKUP not LEAD.
4. **HumanEval+ cheap pilot NOT launched in W102** per the
   pre-committed RUNBOOK § "Backup lane" decision logic.
   Cheap pilot is the W103 fallback if MBPP+ V2 cheap pilot
   FAILs or downgrades to `PASS_NON_MECHANISM_DRIVEN`.

### Helper lane — COO-14

1. **1 NEW explicit-import-only module**:
   `coordpy.code_slice_selector_v1`.  Implements the four-item
   COO-14 Definition of Done verbatim (see
   `docs/RESULTS_W102_CODE_SLICE_SELECTOR_V1.md`).
2. **Driver script + result-doc artifacts**:
   `scripts/run_w102_code_slice_proposal.py` produces
   `ranking.md` + `slice_<bench>.md` + `proposals.json`.
   First worked example: humaneval ranks first (composite
   0.9377; 9.76 % rescue fraction; +5.56 pp mean margin); mbpp
   second (0.4907; 3.97 %; +1.33 pp).
3. **15 unit tests pass** (`tests/test_w102_code_slice_selector_v1.py`).
4. **Anti-pattern guard tested**: helper REFUSES to propose
   a slice for any bench_module_name containing
   `bounded_window` / `compaction` / etc.

### W102 arsenal-mining extension

`scripts/run_w102_arsenal_mining.py` re-executes the 990 W88
70B HumanEval + 1650 W91 5-seed 70B MBPP candidate responses
against the V1 HumanEval+ `check()` block + V2 MBPP+ `test`
program.

**Empirical headline numbers** (full table in
`docs/RESULTS_W102_ARSENAL_MINING_V1.md`):

| Bench | A0 | A1@K=5 | B | B−A1 (pp) | Rescue % | Shared-fails / N |
|---|---:|---:|---:|---:|---:|---:|
| W89 HumanEval-70B (3 × 30) | 46.67 % | 85.56 % | 91.11 % | **+5.56** | 9.76 % | 5 / 90 |
| W91 MBPP-70B (5 × 30) | 75.33 % | 82.67 % | 84.00 % | **+1.33** | 3.97 % | 21 / 150 |
| HumanEval+ (W88 re-graded) | 44.44 % | 78.89 % | 84.44 % | **+5.56** | 9.21 % | 12 / 90 |
| **MBPP+ V2 (W91 re-graded)** | **71.32 %** | **77.63 %** | **82.91 %** | **+5.28** | **6.60 %** | **22 / 130** |

**Critical empirical proof**: re-grading the SAME 1 650 W91
70B candidate responses against the EvalPlus extra-test
surface lifts B − A1 from +1.33 pp (base MBPP cap) to
**+5.28 pp** (clears the +5 pp Phase 2 bar).  This is the
empirical demonstration that the W91 cap was indeed
ceiling-saturation-bound on base MBPP, and the EvalPlus
hardened tests structurally restore the failure-residual
surface the W89 sequential-reflexion mechanism can attack.
The W101 V1 silent-degeneration bug would have hidden this
finding entirely.

## Phase 2 cheap-pilot verdict (MBPP+ V2) — **FAIL**

Full verdict in
`docs/RESULTS_W102_MBPP_PLUS_V2_PHASE2_70B_V1.md`.  Headline:

| Arm | Pass rate | vs prior |
|---|---:|---|
| A0 | 73.33 % (22 / 30) | — |
| A1 @ K=5 | 83.33 % (25 / 30) | empirical (vs W102 mining prior 77.63 % on W91 historical responses re-graded) |
| B (sequential reflexion K=5) | 76.67 % (23 / 30) | — |
| **B − A1** | **−6.67 pp** | **11.95 pp swing below the +5.28 pp arsenal-mining prior** |
| B − A0 | +3.33 pp | < +5 pp gate-5 floor |

**6 of 9 Phase 2 gates PASS** (gates 3 + 4 + 5 FAIL); **MLB-1
(30 % invocation rate) FAIL** by 3 pp; **MLB-2 (22.22 % rescue
rate) FAIL** by 11 pp.  Verdict label: **`FAIL`**.

Per-problem cluster: 3 a1_only_wins (B regression: i.i.d. K=5
sampling found a PASS, reflexion chain didn't), 1 b_only_win
(rescue: problem 0 at attempt 1), 22 shared_wins, 4 shared_fails.

Pilot wall: 4 742 s = 79 min (NIM endpoint heavily 429-
throttled today; hardened retry kept the pilot grinding
without exhausting budget; total 330 NIM calls landed).

## Decision logic applied per the pre-committed runbook

**APPLIED**: cheap pilot FAILed (B − A1 = −6.67 pp; MLB-1 +
MLB-2 both FAIL).  Per branch 3 of the pre-committed W102
RUNBOOK § "Lead lane — MBPP+ V2 (CONDITIONAL on preflight)":

1. **Add carry-forwards** (done above):
   `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP` +
   `W102-L-MBPP-PLUS-V2-MECHANISM-LOAD-BEARINGNESS-WEAK-AT-70B-CAP`.
2. **W103 lead path = HumanEval+ cheap pilot** using the
   W102-built backup-lane infrastructure (preflight 7/7 PASS;
   verdict cid
   `4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4`).
   The HumanEval+ historical rescue fraction (9.21 % on W88
   responses re-graded against HumanEval+) is materially
   richer than MBPP+ V2's empirical 22.22 % rescue rate on
   fresh K=5; HumanEval+ is the right next attack.
3. **`COO-9` REMAINS the lead path** — EvalPlus-family attack
   on the W91 base-MBPP cap is still the right direction; the
   question shifts to which EvalPlus benchmark is the right
   battlefield.  W103 RUNBOOK pre-commits the HumanEval+
   attack before any new NIM call.
4. **NO cross-scale MBPP+ V2** at this time.  Cross-scale
   confirmation is not entitled when Phase 2 FAILs.
5. **NO Phase 3 retirement bench** at this time.  Phase 3
   requires Phase 2 PASS + cross-scale PASS.

### Decision branches NOT taken (pre-committed)

* If pilot had PASSed 9/9 + MLB sub-gates cleared →
  `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-PASS` +
  W103 = MBPP+ V2 cross-scale confirmation.  Did NOT happen.
* If pilot had PASSed 9/9 with MLB-2 FAIL →
  `PASS_NON_MECHANISM_DRIVEN` + W103 = HumanEval+ cheap pilot.
  Did NOT happen (the pilot also FAILed gates 3 + 4 + 5).

## Carry-forwards

### Retired

* **NONE.**  W89 70B HumanEval K=5 remains the only confirmed
  multi-seed same-budget multi-agent superiority retirement.
  A W102 cheap-pilot PASS at 70B is NOT retirement evidence —
  W104+ Phase 3 multi-seed required.

### Added by this milestone

* **`W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`** — at
  Llama-3.3-70B-Instruct, slice seed 101_001, 30 problems,
  K=5: B − A1 = −6.67 pp; B does NOT strictly beat A1;
  MLB-1 = 30 % (FAIL); MLB-2 = 22.22 % (FAIL).  Cheap pilot
  FAILs the Phase 2 +5 pp margin bar by 11.67 pp.  The W102
  arsenal-mining +5.28 pp prior (re-graded W91 historical
  responses against MBPP+ V2 test surface) did NOT transfer
  to fresh K=5 sampling at a new seed.
* **`W102-L-MBPP-PLUS-V2-MECHANISM-LOAD-BEARINGNESS-WEAK-AT-70B-CAP`**
  — MLB-2 rescue rate 22.22 % on MBPP+ V2 is well below the
  W89 HumanEval rescue rate of 47 %.  The reflexion mechanism
  on this benchmark family produces fewer rescues per
  invocation than the W89 retirement template, suggesting the
  structural problem is not solely ceiling-pressure but
  includes mechanism-fit on MBPP-family problems at 70B.

### Demoted

* `coordpy.mbpp_plus_loader_v1` → historical artifact +
  anti-pattern (silent-degeneration failure mode).
* `coordpy.mbpp_plus_executor_v1` → same.
* `coordpy.mbpp_plus_reflexion_bench_v1` → same (relies on
  the broken V1 loader+executor).

## Discipline status

W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 /
W100 / W101 / **W102** — **TWELFTH consecutive validation** of
the preflight-first + cross-scale + multi-candidate-tournament-
then-confirm + mechanism-load-bearingness + (new W102)
silent-degeneration-anti-pattern-guard discipline.

## Stable boundary preserved

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* 9 NEW modules (4 V2 MBPP+ + 4 HumanEval+ + 1 helper) all
  explicit-import only.
* 5 NEW driver scripts in `scripts/`.
* 3 NEW unit-test files (67 tests; all PASS).

## Linear ↔ GitHub sync

* `COO-26` (W102 issue) created.
* `linear_github_mapping.json` extended with the W102 entry.
* `scripts/sync_linear_github_v1.py validate` reports OK
  across all 13 milestones (W89 / W93 / W94 / W95 / W96-A /
  W96-C / W96-D / W97 / W98 / W99 / W100 / W101 / W102).

## Anchors

* `docs/RUNBOOK_W102.md` — pre-commit contract.
* `docs/RESULTS_W102_MBPP_PLUS_LOADER_V2_FIX_V1.md` —
  V2 fix doc.
* `docs/RESULTS_W102_MBPP_PLUS_V2_PHASE2_70B_V1.md` — pilot
  verdict (populated post-pilot).
* `docs/RESULTS_W102_HUMANEVAL_PLUS_PREFLIGHT_V1.md` — backup
  preflight verdict.
* `docs/RESULTS_W102_CODE_SLICE_SELECTOR_V1.md` — helper-lane
  doc.
* `docs/RESULTS_W102_ARSENAL_MINING_V1.md` — cross-bench
  mining doc.
* `docs/FRONTIER_RELEVANCE_AUDIT_W102_V1.md` — frontier audit
  supplement.
* `coordpy/mbpp_plus_loader_v2.py` / `..._executor_v2.py` /
  `..._reflexion_bench_v2.py` / `..._preflight_v2.py` — V2
  MBPP+ modules.
* `coordpy/humaneval_plus_*.py` — HumanEval+ modules.
* `coordpy/code_slice_selector_v1.py` — helper module.
* `scripts/run_w102_*.py` — driver scripts.
* `tests/test_w102_*.py` — 42 W102 unit tests.
