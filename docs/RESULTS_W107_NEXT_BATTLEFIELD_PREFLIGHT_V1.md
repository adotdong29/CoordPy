# W107-β — next-code-battlefield NIM-free preflight (V1)

> **2026-05-28.  Lane β of W107 — the main empirical lane, because
> the Lane α 405B gate returned HTTP 404 for the fourth consecutive
> time (`docs/RESULTS_W107_405B_GATE_V1.md`).  This is the direct
> execution of the `COO-9` charter DoD on the post-EvalPlus code
> battlefield: pick a family + justify, build a loader/evaluator with
> the same fairness discipline, specify A0/A1/B before running,
> produce a runbook for the first attack.  ZERO NIM calls.**
>
> **Verdict: LiveCodeBench is the structurally-sound PRIMARY; APPS is
> the structural-pivot backup; NO pivot triggered.**  Offline probes
> all PASS.  The cheap pilot is W108 work, gated on the operator
> corpus-fetch + the W93 5-gate discipline.  Preflight verdict CID
> `55910d11e210c323fb1a393bbf8be1c3ffa2d19dd22f9e9f40e52ffc9746c6b6`.

## Why a next battlefield at all (the COO-9 frontier after W106)

The EvalPlus pair from the W101 battlefield-selection matrix is now
exhausted: **MBPP+ capped** (`W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`)
and **HumanEval+ retired single-class** (W105 RETIRED on Llama-3.3-70B
+2.33 pp FAIL_MARGIN on Llama-3.1-70B; W106 closed that branch).  The
two honest ways to a STRONGER claim than bounded-single-class are
(a) cross-scale-UP (405B — blocked, four 404s) or (b) a NEW code
benchmark family attacked under preflight-first discipline.  With (a)
blocked, (b) is the live path, and it is exactly `COO-9`.

## Selection rule applied (W101 C1–C8 + W107 S1–S3)

The W106 § 7 pre-commit named **LiveCodeBench primary / APPS backup**.
W107-β does not rubber-stamp that — it applies the W101 rubric to the
two W101-RESERVED candidates and a pre-committed structural-soundness
test (`docs/RUNBOOK_W107.md` § 4) that CAN pivot primary→backup inside
this milestone.

### The W101-reserved pair, re-graded for the post-EvalPlus frontier

| Criterion | LiveCodeBench | APPS |
|---|---|---|
| C1 ceiling pressure | **A** — published Llama-3.x-70B pass@1 ≈ 30–50 % on post-cutoff windows; residual ≥ +40 pp | **A** — interview/competition tiers 30–50 % |
| C2 executor cleanness | **B→A** — functional (`starter_code`) subset has a clean deterministic subprocess executor (proven offline below) | **A→B** — stdin/stdout executor is well-trodden but a different shape; `fn_name` subset is call-based |
| C3 decomposition fit | **A** — functional form == W89 "produce a complete function" shape | **B** — mostly stdin/stdout "produce a complete program"; fits but a different output shape |
| C7 contamination resistance | **A (decisive)** — time-anchored release_vN; the only candidate whose design answers "is the retirement real or contamination?" | **C** — 2021 vintage; almost certainly in Llama-3.x training data ⇒ a high pass rate may be contamination, weakening any claim |
| C5 reproducibility | **B** — single HF dataset, release-pinned + SHA-pinnable JSONL | **B** — single HF dataset; larger; difficulty-tiered |
| C8 cheap-pilot budget | **B** — ~330–500 calls on the functional subset | **C** — longer competitive problems push budget up |

**Decisive axis**: for a *publication-grade* multi-agent-superiority
claim (the Lane γ context), C7 contamination resistance dominates.
APPS grades cleaner on executor/decomposition (C2/C3) but its
contamination exposure would make any APPS-based retirement claim
contestable.  LiveCodeBench's time-anchoring is precisely the property
that makes a claim defensible.  Hence LiveCodeBench leads — provided
it survives the structural-soundness gates.

### Structural-soundness gates (S1∧S2∧S3 — the pivot test)

| Gate | Meaning | LiveCodeBench verdict |
|---|---|---|
| **S1** | a clean, no-LLM-judge, binary PASS/FAIL executor exists for a coherent subset of adequate size | **PASS** — the functional (`starter_code`) subset; proven by the offline executor self-test below |
| **S2** | a NIM-free A1@K=5 failure-residual estimate exists | **PASS (published-baseline-grade)** — leaderboard pass@1 ≈ 30–50 % ⇒ residual ≥ +40 pp; **weaker** than the EvalPlus pair's re-executed-sidecar residual; operator must verify the exact pinned-release number |
| **S3** | the W89 read→solve→execute→reflect→repair mechanism ports without becoming a new research project | **PASS** — functional form is exactly the W89 complete-function shape |

S1 ∧ S2 ∧ S3 = **TRUE** ⇒ **no pivot**; LiveCodeBench stays primary;
APPS held as the structural-pivot backup.

## Offline preflight probes (REAL, executed; NIM-free)

Run: `python scripts/run_w107_livecodebench_preflight.py` (verdict
`results/w107/livecodebench_preflight/preflight_verdict.json`).

### P2 — executor self-test (the genuine PASS evidence; gate G9 in miniature)

| Check | Result |
|---|---|
| gold top-level function PASSes all cases | **PASS** |
| gold `Solution().method` PASSes all cases | **PASS** |
| wrong solution FAILs (returncode 1; `CASE_FAIL` in stderr) | **PASS** |
| infinite-loop solution TIMES OUT (kill-after enforced) | **PASS** |

This proves the functional executor MACHINERY is clean: fresh `-I`
CPython subprocess, soft+kill wall timeout, binary PASS/FAIL on exit 0,
failure tail returned for the reflexion signal, NO LLM judge — the same
invariants that earned G9 across W86 → W105.

### Loader schema self-test (the W102 silent-degeneration guard)

| Check | Result |
|---|---|
| valid functional row accepted | **PASS** |
| valid stdin row accepted then filtered out (functional-only) | **PASS** |
| functional row correctly detected (`starter_code` non-empty) | **PASS** |
| row missing a required field is REFUSED (not silently degraded) | **PASS** |

## The β deliverables (RUNBOOK § 4.3), shipped

1. **Battlefield-selection rule** — the C1–C8 + S1–S3 verdict above.
2. **Loader scaffolding** — `coordpy/livecodebench_loader_v1.py`:
   SHA-pinnable release_vN JSONL, functional-subset filter, refuses
   unpinned/mismatched corpus (mirrors `humaneval_plus_loader_v1`).
3. **Evaluator scaffolding** — `coordpy/livecodebench_executor_v1.py`:
   clean functional-form subprocess executor (mirrors
   `humaneval_plus_executor_v1` cleanness invariants).
4. **A0 / A1 / B definitions** — byte-identical to W103/W104/W107-α:
   A0 single-shot zero-temp; A1 best-of-K=5 single agent; B W89
   sequential reflexion at the same K=5 budget.
5. **Cheap integrity probes** — executor self-test (above) + loader
   schema self-test (above) + operator-deferred corpus probes.
6. **Executor cleanness check** — S1 PASS (above).
7. **Failure-residual estimate** — S2: published-baseline-grade
   A1@K=5 ≈ 30–50 % ⇒ residual ≥ +40 pp; **operator must verify** the
   exact pinned-release number (the residual is leaderboard-grade, not
   a re-executed local sidecar — explicitly weaker than the EvalPlus
   pair's W101 residual, recorded honestly).
8. **Decomposition argument** — S3 PASS: functional form is the W89
   complete-function shape.
9. **Helper/slice-selection integration** —
   `coordpy.code_slice_selector_v1` (COO-14) is the slice proposer to
   wire onto the functional subset at fetch time, exactly as it
   proposed the W103 HumanEval+ slice (its 4th real downstream
   consumption).  Wiring is deferred to the corpus-fetch step because
   the selector needs the real per-problem surface.

## Honest caps (carry-forwards to register)

* `W107-L-LIVECODEBENCH-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP` — the
  exact upstream field names + test-case encoding MUST be confirmed
  against the live release_vN corpus before any pilot (the W102
  lesson: a wrong schema assumption nearly degraded a pilot once; this
  loader REFUSES on mismatch instead).
* `W107-L-LIVECODEBENCH-RESIDUAL-PUBLISHED-BASELINE-GRADE-CAP` — the
  S2 residual is published-baseline-grade, not re-executed-sidecar-
  grade; the operator must re-confirm pass@1 for the exact pinned
  release window before the cheap pilot is earned.
* `W107-L-LIVECODEBENCH-FUNCTIONAL-SUBSET-ONLY-CAP` — only the
  functional subset is in scope for the W89 mechanism; stdin/stdout
  problems are deferred.

## Operator-deferred probes (the W108 / fetch step)

Per `python scripts/run_w107_livecodebench_preflight.py
--print-fetch-playbook`: pin ONE post-cutoff release_vN + SHA; confirm
the live schema (W102 discipline); then re-run the preflight to compute
the real P1 corpus-integrity + functional-subset-size (≥ 30) + live
A1@K=5 residual probes.  Only then is a W108 cheap pilot earnable.

## What W108 becomes

Per `docs/RUNBOOK_W107.md` § 7: with α CLOSED and the β preflight
PASSing for LiveCodeBench, **W108 = the LiveCodeBench functional-subset
Phase 2 cheap pilot** (1 seed × 30 problems × K=5; ~330 NIM calls;
W93 5-gate + the W107-α-shape 9-gate + MLB sub-gates), AFTER the
operator corpus-fetch confirms the schema + the live residual.  If the
live fetch reveals LiveCodeBench is structurally wrong after all, the
pre-committed pivot to APPS applies in W108 without new paperwork.

## Anchors

* `docs/RUNBOOK_W107.md` § 4 — the selection rule + S1–S3 pivot test.
* `coordpy/livecodebench_loader_v1.py` + `coordpy/livecodebench_executor_v1.py`.
* `scripts/run_w107_livecodebench_preflight.py` + verdict JSON.
* `tests/test_w107_livecodebench_preflight_v1.py`.
* `docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md` — the C1–C8 rubric.
* `docs/RESULTS_W107_405B_GATE_V1.md` — why β is the main lane.
