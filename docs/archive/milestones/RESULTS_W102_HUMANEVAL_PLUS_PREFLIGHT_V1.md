# W102 — HumanEval+ NIM-free preflight verdict V1 (backup lane)

> **2026-05-25.  7 of 7 PASS.  Backup-lane infrastructure is
> built + cheap-probe-validated.  No cheap pilot launched in
> W102 itself; if the W102 MBPP+ V2 cheap pilot FAILs or
> downgrades to `PASS_NON_MECHANISM_DRIVEN`, W103 attacks
> HumanEval+ using this infrastructure.  Otherwise HumanEval+
> stays the backup until cross-bench generalisation of a
> W102-confirmed positive becomes the next question.**

## Inputs

| Field | Value |
|---|---|
| Candidate mechanism | B (W89 sequential reflexion on HumanEval+ at K=5) |
| Target model | `meta/llama-3.3-70b-instruct` |
| HumanEval+ corpus | `~/.cache/coordpy/humaneval-plus.jsonl` (164 rows; SHA-256 `908377f1...` matches HF LFS oid) |
| Preflight script | `scripts/run_w102_humaneval_plus_preflight.py` |
| Preflight module | `coordpy.humaneval_plus_preflight_v1` |
| Verdict cid | `4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4` |

## Per-probe verdicts

| Probe | Verdict | Summary |
|---|---|---|
| **P1** corpus integrity | **PASS** | 164 problems loaded; 164 with `test`; 164 with `entry_point`; HF LFS SHA matches. |
| **P2** executor self-test on canonical solutions | **PASS** | 30 / 30 = 100.00 % canonical pass under the V1 executor. |
| **P3** A1@K=5 failure-residual estimate | **PASS** | Predicted HumanEval+ A1@K=5 = 72.86 % (W88 70B HumanEval A1 mean 85.56 % − published EvalPlus Hoeffding lower-bound drop 12.7 pp); saturation margin 17.14 pp. |
| **P4** decomposition argument | **PASS** | 1528 chars; W89 retirement on base HumanEval-70B is the empirical precedent. |
| **P5** extra-test-surface integrity | **PASS** | 164 / 164 = 100.00 % rows carry `def check(` (floor 95 %). |
| **AddrW102-Hplus-AntiPattern** | **PASS** | No `bounded_window` / `compaction` / `summary` tokens in `coordpy/humaneval_plus_reflexion_bench_v1.py`. |
| **AddrW102-Hplus-W89-Rescue** | **PASS** | W89 base-HumanEval rescue fraction 9.76 % ≥ 5 % threshold. |

**7 of 7 PASS.**

## Why this is the BACKUP and not the LEAD

Per the W101 battlefield-selection matrix
(`docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md`):

| Criterion | MBPP+ V2 (LEAD) | HumanEval+ (BACKUP) |
|---|---|---|
| C1 ceiling pressure | A — A1 drops ~13 pp on the GPT-3.5 family; ~15-18 pp on Llama-class (empirical extrapolation). | **B** — A1 drops similarly but base HumanEval is already 85.56 % on 70B; post-EvalPlus residual is comfortable but smaller than MBPP+. |
| C2 executor cleanness | A | A |
| C3 decomposition fit (W89 shape) | A | A |
| C4 per-problem failure surface | A (~9-10 expected unique-A1-failures on 30 problems at A1≈70%) | **B** (~5-7 expected unique-A1-failures on 30 problems at A1≈73%) |
| C5 reproducibility | A | A |
| C6 W93 preflight compatibility | A | A |
| C7 MBPP-ceiling-trap avoidance | A | A |
| C8 cheap-pilot budget | A (~330 calls) | A (~330 calls) |

HumanEval+ ranks BACKUP because its per-problem failure surface
is smaller (base HumanEval is harder for the model than base
MBPP at 70B; EvalPlus extras compress the residual further).
For W102 the LEAD attack is on the larger residual surface
(MBPP+ V2); HumanEval+ stays available as the W103 immediate
pivot if MBPP+ V2 FAILs.

## When to launch the HumanEval+ cheap pilot

Per the W102 RUNBOOK § "Lead lane — MBPP+ V2 (CONDITIONAL on
preflight)" decision logic:

1. **MBPP+ V2 cheap pilot PASSes** with MLB sub-gates clearing
   → W103 = MBPP+ V2 cross-scale confirmation (different model
   class).  HumanEval+ cheap pilot DEFERRED to a later
   milestone (W104+) when cross-bench generalisation becomes
   the next question.
2. **MBPP+ V2 cheap pilot PASSes** with MLB-2 FAILing →
   `PASS_NON_MECHANISM_DRIVEN`; W103 = HumanEval+ cheap pilot
   using this preflight-earned infrastructure.
3. **MBPP+ V2 cheap pilot FAILs** → carry-forward
   `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`; W103 =
   HumanEval+ cheap pilot using this preflight-earned
   infrastructure.

In all three cases, the HumanEval+ preflight verdict (this doc)
is the cheap-evidence basis the W103 runbook will cite.

## What this preflight does NOT do

* It does NOT launch the HumanEval+ cheap pilot in W102.
* It does NOT predict the HumanEval+ cheap-pilot outcome
  with high confidence — only the cheap pilot can do that.
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.
* It does NOT modify `coordpy/__init__.py`.

## Anchors

* `scripts/run_w102_humaneval_plus_preflight.py` — preflight
  runner.
* `coordpy/humaneval_plus_loader_v1.py` — HF JSONL loader.
* `coordpy/humaneval_plus_executor_v1.py` — V1 executor.
* `coordpy/humaneval_plus_reflexion_bench_v1.py` — W89
  mechanism wired with HumanEval+ loader+executor.
* `coordpy/humaneval_plus_preflight_v1.py` — 7-probe preflight.
* `docs/RUNBOOK_W102.md` — pre-commit contract.
* `docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md` — battlefield
  ranking that justifies BACKUP classification.
