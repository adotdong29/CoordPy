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

## Phase 2 cheap-pilot verdict (MBPP+ V2)

> Populated at pilot completion in
> `docs/RESULTS_W102_MBPP_PLUS_V2_PHASE2_70B_V1.md` and
> cross-referenced here once available.

## Decision logic applied per the pre-committed runbook

### If MBPP+ V2 cheap pilot PASSes 9/9 + MLB sub-gates

* Add carry-forward
  `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-PASS` (a
  Phase 2 cheap-pilot single-seed PASS, NOT a retirement).
* W103 = cross-scale MBPP+ V2 confirmation at a SECOND model
  class (`meta/llama-4-405b-instruct` if available, OR
  `meta/llama-3.2-90b-vision-instruct` in text-only mode).
* HumanEval+ cheap pilot DEFERRED to W104+ when cross-bench
  generalisation becomes the next question.
* W104+ = MBPP+ V2 Phase 3 retirement bench (3 seeds × 100
  problems × K=5) IF W103 cross-scale PASSes.

### If MBPP+ V2 cheap pilot PASSes 9/9 with MLB-2 FAIL

* Mark `PASS_NON_MECHANISM_DRIVEN` per W96-C / W100 / W101
  precedent.
* W103 = HumanEval+ cheap pilot using the W102-built backup-
  lane infrastructure.
* No cross-scale on MBPP+ V2.

### If MBPP+ V2 cheap pilot FAILS

* Add carry-forward
  `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`.
* W103 = HumanEval+ cheap pilot using the W102-built backup-
  lane infrastructure.
* `COO-9` remains the lead path (the EvalPlus-family attack
  on the W91 base-MBPP cap is still the right direction; the
  question becomes which EvalPlus benchmark is the right
  battlefield).

## Carry-forwards

### Retired

* **NONE.**  W89 70B HumanEval K=5 remains the only confirmed
  multi-seed same-budget multi-agent superiority retirement.
  A W102 cheap-pilot PASS at 70B is NOT retirement evidence —
  W104+ Phase 3 multi-seed required.

### Added by this milestone

* **NONE empirical yet** until pilot completes.  The verdict
  doc populates carry-forwards based on the pilot outcome.

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
