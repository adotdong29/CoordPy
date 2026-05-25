# W101 — Second-code-benchmark battlefield selection V1

> **2026-05-25.  Lead-path selection for `COO-9` after W100
> promoted it via the pre-committed Part H code-pivot
> contingency.  This document is written BEFORE any new bench
> loader is built and BEFORE any NIM call is approved.  It
> ranks the five candidate code benchmarks named in the COO-9
> charter, pre-commits the selection rule, and names the chosen
> lead + backup.  Per the W93 preflight-first discipline, the
> selection is justified on cheap evidence (W89 / W91 sidecars,
> published saturation data, executor-cleanness, decomposition
> fit, reproducibility) — NOT on intuition.**

## Why this matters

The active research frontier after W100 has exactly one
unresolved retirement-grade question on the code side:

* **`W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP`** — at
  Llama-3.3-70B-Instruct on MBPP-sanitized × 5 seeds × 30
  problems × K=5, the W89 sequential-reflexion B-pipeline beats
  A1 on the MEAN by +1.33 pp and on per-seed B≥A1 in 4/5 seeds
  but FAILs strict-majority B>A1 at 2/5 (need ≥3/5).  The
  documented structural reason is **ceiling effects on
  MBPP-sanitized at 70B**: A1@K=5 = 82.7% on the mean and 90%
  on two saturated seeds, leaving the strict-majority bar
  starved for headroom.

W100's pre-committed Part H code-pivot triggers because the
cross-modal RealWorldQA arc is empirically capped on the
W95-B0 family shape at the 11B regime AND on the B2 mechanism
cross-scale at 90B.  Cross-modal retirement is not the
immediate route to a *stronger* programme claim.

The strongest *cheap* next move is therefore: replace MBPP
with a less-saturated published code benchmark that the W89
mechanism can attack honestly.

## The COO-9 candidate slate (from the issue charter)

| ID | Family | Provenance |
|---|---|---|
| `MBPP+` | EvalPlus's hardened MBPP (≈35× more tests per problem) | Liu et al. 2023; `evalplus/mbppplus` HF dataset; canonical executor |
| `HumanEval+` | EvalPlus's hardened HumanEval (≈80× more tests per problem) | Liu et al. 2023; `evalplus/humanevalplus` HF dataset; canonical executor |
| `APPS` | Competitive-programming benchmark (10k problems × {intro/interview/competition}) | Hendrycks et al. 2021 |
| `LiveCodeBench` | Time-anchored contamination-resistant code benchmark | Jain et al. 2024 |
| `SWE-bench-lite` | Real GitHub issues; repo-patch generation | Jimenez et al. 2024 (lite subset) |

## Selection criteria (pre-committed BEFORE any candidate is built)

These are derived from the COO-9 charter + the W94 battlefield-
selection rubric.  Each criterion is one column of the ranking
matrix; the chosen lead must rank ≥ B on every criterion AND
have no F.

| # | Criterion | Why it matters | Pass / Fail rule |
|---|---|---|---|
| C1 | **Ceiling pressure at the candidate model + budget** | If A1@K=5 saturates above 90% on the chosen subset, the +5 pp Phase 2 margin bar is structurally unreachable.  W91 failed exactly this way. | A1@K=5 estimate ≤ 90% on the candidate subset OR an explicit residual ≥ +10 pp documented. |
| C2 | **Executor cleanness (no LLM-judge)** | The W85-W92 anti-cheat surface forbids LLM-as-judge in the executor chain. | A deterministic Python / unit-test / docker executor exists and returns binary PASS/FAIL per problem. |
| C3 | **Natural same-budget decomposition fit** | The B pipeline needs a role split (e.g., reader / solver / reflexion) that consumes the same K=5 budget A1 spends.  Decomposition that does not fit gives a structurally unfair B arm. | A K=5 same-budget B exists that uses the existing W89 sequential-reflexion shape OR a documented orthogonal mechanism (test-aware decomposition, executor-guided repair, structured failure-trace carry, tool-augmented code reasoning). |
| C4 | **Per-problem failure surface size** | The cheap pilot at 1 seed × 30 problems needs ≥ 10 problems where A1 fails so the B mechanism has real rescue surface.  Saturated benches starve this surface. | Expected unique-A1-failures on the cheap-pilot slice ≥ 10. |
| C5 | **Reproducibility + auditability cost** | The W85+ audit-chain discipline (per-call sidecar + per-seed Merkle + bench Merkle) must be plug-and-play; the bench loader + executor must be SHA-anchored against an upstream pin. | Canonical upstream URL exists; SHA-256 pinnable; loader ≤ 500 lines of new code; executor ≤ 300 lines. |
| C6 | **W93 preflight compatibility** | The cheap preflight harness must run NIM-free against the candidate.  Benches that require a NIM call just to compute their own residual estimate violate the discipline. | A1@K=5 residual estimate is computable from W89 / W91 sidecars + the candidate's published baseline OR from re-executing existing NIM responses against the candidate's tests. |
| C7 | **Avoidance of the MBPP ceiling trap that already blocked retirement** | The candidate must lower (not preserve) the saturated-ceiling regime that W91 hit. | EvalPlus-grade test-suite expansion OR substantially harder problem distribution OR strict per-seed sampling that breaks ceiling. |
| C8 | **Cheap-pilot budget honesty** | Phase 2 cheap pilot is 1 seed × 30 problems × 11 calls = ~330 NIM calls at 70B.  Anything that requires ≥ 1000 NIM calls per cheap pilot is too expensive for a battlefield-discriminator role. | Cheap pilot estimated NIM cost ≤ ~660 calls (i.e., ≤ 2× the W93 reference). |

## Ranking matrix

Grades: A = best; D = acceptable; F = disqualifying.  A
candidate's overall rank is the conjunction of column grades
(a single F disqualifies; otherwise sorted on count of A / B
grades).

| Criterion | MBPP+ | HumanEval+ | APPS | LiveCodeBench | SWE-bench-lite |
|---|---|---|---|---|---|
| C1 Ceiling pressure | **A** — EvalPlus drops base-MBPP A1 ~25 pp (Liu et al. Table 4: GPT-4 pass@1 drops 87→69) | **B** — drops base-HumanEval similarly but base was already 88% on 70B, so post-EvalPlus residual is comfortable; the published margin is the smallest of the EvalPlus pair | **A** — interview/competition tiers have 30-50% pass rates on frontier models | **A** — by-design ceiling-controlled via time anchoring | **A** — published GPT-4 pass rate on lite ≈ 18-25% |
| C2 Executor cleanness | **A** — same subprocess Python executor as base MBPP + extra unit tests; no LLM judge | **A** — same `check()` block + extra hidden tests; no LLM judge | **B** — Python executor + I/O-string comparison; some problems use stdin/stdout patterns | **C** — executor exists per problem family but harness setup is non-trivial | **D** — requires docker + real-repo apply + test framework per repo; executor cleanness is high but setup is heavy |
| C3 Decomposition fit (W89 shape) | **A** — W89 reflexion mechanism plug-and-play; only the test surface changes | **A** — same | **B** — reflexion mechanism fits but the call-budget shape is different (longer chains; some problems need ≥ 3000 output tokens) | **C** — fit exists but contamination-resistance design adds harness complexity | **F** — patch-generation does not map to the W89 "produce a complete function" shape; B would need a different mechanism (executor-guided diff repair) which is a different research project |
| C4 Per-problem failure surface | **A** — at A1@K=5 ≈ 65-70% on 30 problems → expected unique-A1-failures ≥ 9-10 | **B** — at A1@K=5 ≈ 75-85% on 30 problems → expected unique-A1-failures ≥ 5-7 (lower than MBPP+) | **A** — failure surface is by-design large | **A** — failure surface is by-design large | **A** — failure surface is by-design large (most repos fail) |
| C5 Reproducibility + auditability | **A** — EvalPlus releases a single JSONL with `task_id`, `prompt`, `base_input`, `plus_input` tables; SHA-pinnable; loader ≤ 300 lines | **A** — same | **C** — corpus is 10k problems + multi-source structure; loader is non-trivial | **C** — time-anchored fetch + provider-side staleness control adds infra | **D** — repo clones + per-task docker + test framework; not auditable from a single SHA-pinned JSONL |
| C6 W93 preflight compatibility | **A** — A1@K=5 residual estimate computable from W91 5-seed sidecar re-executed against MBPP+ extra tests OFFLINE (no new NIM call) | **A** — same, against W88 70B HumanEval sidecar re-executed against HumanEval+ extra tests | **C** — sidecar replay requires loading large APPS problem traces; tractable but heavier | **C** — preflight harness needs LiveCodeBench-specific time-anchor logic | **F** — preflight harness requires docker setup per repo; cheap NIM-free probe is not cheap |
| C7 MBPP ceiling-trap avoidance | **A** — EvalPlus was *literally designed* to drop the MBPP ceiling that W91 hit | **A** — EvalPlus drops HumanEval ceiling too | **A** — different distribution; no overlap with MBPP ceiling | **A** — different distribution | **A** — different distribution |
| C8 Cheap-pilot budget | **A** — same 330 calls × 70B as W89 / W91 | **A** — same | **C** — longer per-problem token budgets push to ~500-800 calls | **B** — ~330-500 calls | **F** — single-repo evaluation cycle requires docker round-trip + apply-and-test loop; cheap pilot ≥ 1500 NIM calls |
| **Overall verdict** | **LEAD** | **BACKUP** | Out of scope | Out of scope | Out of scope (single F disqualifies) |

## Disqualification reasons

* **APPS** out of scope: not disqualifying on any single criterion but stacks C2/C5/C6/C8 at C-grade.  The infrastructure cost is high enough that it should follow a successful MBPP+ result, not lead.
* **LiveCodeBench** out of scope: contamination-resistance is genuinely valuable BUT requires time-anchored fetch infrastructure that is more complex than the W101 cheap-pilot envelope justifies.  Reserved as a W102+ direction if MBPP+ + HumanEval+ both succeed and generalisation across an additional benchmark family is the next question.
* **SWE-bench-lite** disqualified: two F grades.  C3 (decomposition fit) — patch-generation is structurally a different mechanism than function-implementation; the W89 reflexion shape does NOT port.  C6 + C8 (preflight + cheap-pilot cost) — docker per task and multi-stage test harness blow the cheap-pilot budget.  A SWE-bench-lite attack is a multi-week project, not a battlefield-discriminator.

## Lead + backup choice (PRE-COMMITTED)

* **LEAD: MBPP+** (EvalPlus's hardened MBPP).
  * Direct surgical attack on the live `W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP`.
  * EvalPlus's 35× extra tests are specifically designed to drop the saturation ceiling the W91 5-seed run hit.
  * Loader/executor reuse 90% of the existing `coordpy.mbpp_reflexion_bench_v1` infrastructure; the only new bench code is (a) the EvalPlus extra-tests loader and (b) the extra-tests-aware executor.
  * Preflight harness can produce a NIM-free A1@K=5 residual estimate by re-executing the W91 5-seed sidecar (`results/w91/mbpp_reflexion_5seeds/.../mbpp_reflexion_calls.jsonl`, 1650 model responses) against the EvalPlus extra tests offline.  This is the cheapest possible cross-bench failure-residual probe.
  * If MBPP+ Phase 2 cheap pilot clears the +5 pp bar at 70B AND mechanism-load-bearingness sub-gates pass, the next milestone (W102) is the Phase 3 retirement bench on MBPP+ which would be the **second multi-seed same-budget multi-agent superiority retirement** in the programme (after W89's HumanEval retirement).  That is the strongest *cheap-pilot-earned* claim available on the code side.

* **BACKUP: HumanEval+** (EvalPlus's hardened HumanEval).
  * Same EvalPlus family; the W89 retirement at base HumanEval-70B is the strongest single empirical anchor in the programme; HumanEval+ would test cross-bench generalisation of the *same retirement claim*.
  * Loader ports trivially from `coordpy.humaneval_real_bench_v1`; executor adds the EvalPlus extra `check()` block.
  * If MBPP+ preflight unexpectedly FAILs (e.g., EvalPlus extra tests still saturate A1 ≥ 90% on 70B), HumanEval+ is the immediate pivot.  Estimated cheap-pilot cost is identical to MBPP+ (~330 NIM calls).
  * Backup is kept in the runbook contract but is NOT BUILT in W101 unless MBPP+ preflight FAILs.

## Pre-committed selection rule (locked 2026-05-25 BEFORE any preflight runs)

1. **Build MBPP+ loader + executor + bench + preflight harness** (no NIM, code-only).
2. **Run W101 preflight** (NIM-free; W93 5-gate composite + W101-specific AddrW101 probes mining W88/W91 sidecars).
3. **If MBPP+ preflight PASSES all 5 W93 gates + at least 4 of 5 AddrW101 probes**: MBPP+ is preflight-earned; the W101 cheap pilot is licensed (1 seed × 30 problems × K=5 ≈ 330 NIM calls at 70B; ~2-3 h wall).  Whether to launch the pilot in W101 itself or defer to W102 is an explicit choice based on the empirical preflight evidence.
4. **If MBPP+ preflight FAILS any gate or > 1 AddrW101 probe**: document the failure as `W101-L-MBPP-PLUS-PREFLIGHT-<gate>-CAP`; pivot to HumanEval+ build + preflight in the same milestone OR explicitly defer to W102.
5. **If both MBPP+ AND HumanEval+ preflight FAIL**: the second-code-benchmark battlefield slate from COO-9 is structurally capped at the EvalPlus family; the next move is one of {APPS infrastructure build, LiveCodeBench infrastructure build, `COO-12` substrate-level cross-modal injection promotion}.  This is an explicit W102+ decision, NOT a W101 in-flight choice.

## What this selection does NOT do

* It does NOT promise MBPP+ will clear the +5 pp Phase 2 bar — preflight has not yet run.
* It does NOT retire any prior carry-forward.
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.
* It does NOT entitle any expensive (Phase 3 retirement) bench — entitlement is governed by the W93 cross-scale + W100 MLB sub-gate rule and requires Phase 2 PASS first.

## Anti-drift contract (carries forward from W97 + W98 + W99 + W100 verbatim)

* No bounded windowing as core strategy.
* No compaction / generic prose summarization as memory mechanism.
* No shallow token compression without structural reason.
* No "cram less / truncate better" frontier claims.
* No LLM-as-judge in executor chain.
* No selective retries.
* No single-seed pilots as retirement evidence.
* No architecture refinement by vibe.
* No inventing new candidates after the tournament selected a winner (the W101 cheap pilot, if launched, attacks MBPP+ only; HumanEval+ is a backup pivot, not a parallel candidate).

## Anchors

* `docs/RUNBOOK_W101.md` — pre-commit contract (this milestone).
* `docs/RESULTS_W101_ARSENAL_MINING_V1.md` — sidecar mining + per-problem cluster surface.
* `docs/RESULTS_W101_PREFLIGHT_V1.md` — empirical preflight verdict.
* `docs/RESULTS_W91_MBPP_REFLEXION_V2.md` — the empirical W91 cap this selection attacks.
* `docs/RESULTS_W89_HUMANEVAL_REFLEXION_V2.md` — the retirement template MBPP+ aims to extend.
* `docs/RUNBOOK_W93.md` — preflight-first discipline.
* `docs/RUNBOOK_W100.md` Part H — pre-committed code-pivot that promoted `COO-9`.
* `coordpy/mbpp_reflexion_bench_v1.py` — the W90/W91 base MBPP bench MBPP+ extends.
* `coordpy/humaneval_reflexion_bench_v1.py` — the W88/W89 HumanEval bench (W101 backup pivot would port from this).
* `coordpy/failure_cluster_miner_v1.py` — the W93 mining tool W101 extends.
