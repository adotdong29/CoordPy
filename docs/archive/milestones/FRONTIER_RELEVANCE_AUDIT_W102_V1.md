# W102 — Frontier relevance audit V1 (supplement to W97 / W98 / W99 / W100 / W101)

> **2026-05-25.  Sixth supplement, extending the W97 / W98 /
> W99 / W100 / W101 frontier-relevance audits with the W102
> MBPP+ V2 schema-fix + HumanEval+ backup-lane build + COO-14
> helper-lane delivery + cheap MBPP+ V2 pilot.  The W97 / W98 /
> W99 / W100 / W101 audit classifications all remain in force
> VERBATIM; this supplement only:**
>
> 1. Demotes the W101 `mbpp_plus_loader_v1` /
>    `mbpp_plus_executor_v1` to **historical artifact + anti-
>    pattern (silent-degeneration failure mode)** based on
>    the W102 fetch-step finding.
> 2. Adds the W102 V2 MBPP+ infrastructure + HumanEval+
>    backup-lane infrastructure + COO-14 code-side helper
>    to the **active frontier arsenal** column.
> 3. Re-asserts the cross-modal RealWorldQA arc as **frozen
>    at 11B** in the dead-direction-with-cross-scale-cap
>    column.
> 4. Re-asserts the W97 / W98 / W99 / W100 / W101 anti-pattern
>    column *verbatim*, with the W102 V1-silent-degeneration
>    failure mode added as a NEW anti-pattern explicitly named.
>
> No code is removed by this audit.  No version bump.  No PyPI
> publish.

## Why a supplement, not a rewrite

W101 V1 documented the classification as of the W101 cheap-
NIM-free preflight closing moment.  W102 added three new
empirical facts:

1. The W101 V1 loader schema is wrong against the actual
   EvalPlus release (parallel `plus_input` / `plus_output`
   arrays do NOT exist; the extra-test surface is inside a
   single `test` Python program).  V2 corrects this; V1 is
   demoted.
2. The W102 V2 MBPP+ preflight is **10 of 10 PASS**.  The
   cheap MBPP+ V2 pilot is genuinely earned per the pre-
   committed W102 decision logic.
3. The HumanEval+ backup-lane infrastructure is **built +
   preflight 7 of 7 PASS**.  The W103 fallback path is ready
   if the MBPP+ pilot fails or is downgraded.

W102's job is to ship the lead-lane fix + backup-lane build
+ helper-lane delivery, AND launch the cheap MBPP+ V2 pilot
if (and only if) the V2 preflight clears.  The pilot result
is recorded in `docs/RESULTS_W102_MBPP_PLUS_V2_PHASE2_70B_V1.md`
(populated at pilot completion).

## Active frontier arsenal — W102 additions

| Mechanism | Module(s) | Why frontier (with W102 evidence) |
|---|---|---|
| **MBPP+ V2 loader** (NEW W102) | `coordpy.mbpp_plus_loader_v2` | Active frontier infrastructure.  Reads the actual EvalPlus HF parquet schema; refuses rows with empty `extra_test_program` (closes the V1 silent-degeneration failure mode at the loader boundary). |
| **MBPP+ V2 executor** (NEW W102) | `coordpy.mbpp_plus_executor_v2` | Active frontier infrastructure.  Runs candidate against the canonical EvalPlus `test` program in a fresh CPython subprocess with `-I` (site enabled for numpy).  Three modes (base_and_plus / base_only / plus_only). |
| **MBPP+ V2 reflexion bench** (NEW W102) | `coordpy.mbpp_plus_reflexion_bench_v2` | Active frontier infrastructure.  Wires the W89 sequential-reflexion mechanism with V2 loader + V2 executor; adds `per_problem_b_first_pass_idx` for MLB-1 / MLB-2 sub-gate computation. |
| **MBPP+ V2 preflight** (NEW W102) | `coordpy.mbpp_plus_preflight_v2` | Active frontier operating-system piece.  Adds P5 (extra-test-surface integrity guard against V1 anti-pattern) + P6 (V1-vs-V2 canonical agreement); 10/10 PASS on real EvalPlus parquet. |
| **HumanEval+ loader** (NEW W102) | `coordpy.humaneval_plus_loader_v1` | Active frontier infrastructure.  SHA-pinned HF JSONL loader (164 problems; LFS oid pinned).  Backup-lane battlefield. |
| **HumanEval+ executor** (NEW W102) | `coordpy.humaneval_plus_executor_v1` | Active frontier infrastructure.  Subprocess CPython runner of the EvalPlus `check(candidate)` block; `-I` flag (numpy enabled); 15 s soft / 20 s kill timeouts (vs base HumanEval's 8 s / 12 s because EvalPlus iterations are ~80× longer). |
| **HumanEval+ reflexion bench** (NEW W102) | `coordpy.humaneval_plus_reflexion_bench_v1` | Active frontier infrastructure.  W89 mechanism byte-identical; dedicated seed namespace 102_001..; `per_problem_b_first_pass_idx` for MLB sub-gates. |
| **HumanEval+ preflight** (NEW W102) | `coordpy.humaneval_plus_preflight_v1` | Active frontier operating-system piece.  7-probe NIM-free preflight: P1 corpus, P2 executor self-test, P3 A1 residual estimate, P4 decomposition, P5 extra-test surface, AddrW102-Hplus-AntiPattern, AddrW102-Hplus-W89-Rescue.  7/7 PASS empirically. |
| **Code slice selector / candidate ranker (COO-14)** (NEW W102) | `coordpy.code_slice_selector_v1` | Active frontier operating-system piece.  Consumes arsenal-mining reports; ranks candidate benches by composite score; proposes cheap-pilot slices with cluster-aware justifications; refuses anti-pattern bench modules. |
| **W102 arsenal-mining extension** (NEW W102) | `scripts/run_w102_arsenal_mining.py` | Active frontier operating-system piece.  Re-executes W88 / W91 sidecars against HumanEval+ + MBPP+ V2 surfaces; produces cross-bench cluster surfaces the helper consumes. |

## Useful baselines (W102 changes from W101)

| Mechanism | Module(s) | Classification | W102 status |
|---|---|---|---|
| `bounded_window_baseline_v{1,2,3}` | `coordpy/bounded_window_baseline_v*.py` | UNCHANGED — useful falsifier targets. | Same. |
| `coordpy.mbpp_reflexion_bench_v1` (base MBPP) | `coordpy/mbpp_reflexion_bench_v1.py` | UNCHANGED — baseline-only since W101. | Same. |
| `coordpy.humaneval_reflexion_bench_v1` (HumanEval base) | `coordpy/humaneval_reflexion_bench_v1.py` | UNCHANGED — retirement-anchor (W89 70B retirement). | Same. |
| `coordpy.mbpp_plus_loader_v1` (W101 broken loader) | `coordpy/mbpp_plus_loader_v1.py` | **DEMOTED to historical artifact + anti-pattern (silent-degeneration failure mode)** by W102.  V1 is preserved for the W101 audit trail and as a regression-test exhibit; it is NOT the active frontier loader.  V2 (`coordpy.mbpp_plus_loader_v2`) is the canonical loader for any new cheap pilot. | Stays in-repo; explicitly classified anti-pattern. |
| `coordpy.mbpp_plus_executor_v1` (W101 broken executor) | `coordpy/mbpp_plus_executor_v1.py` | **DEMOTED — same reason**. | Stays in-repo; explicitly classified anti-pattern. |
| `coordpy.mbpp_plus_reflexion_bench_v1` (W101 broken bench) | `coordpy/mbpp_plus_reflexion_bench_v1.py` | **DEMOTED — same reason**.  Note: the bench-module-level imports are correct; only the underlying loader+executor are broken.  V2 bench is the canonical W102 path. | Stays in-repo. |
| `coordpy.mbpp_plus_preflight_v1` | `coordpy/mbpp_plus_preflight_v1.py` | **PARTIALLY RETAINED** — the cross-bench P3 / P4 / AddrW101 probes are reused by V2 preflight verbatim (they are sidecar-derived, not corpus-derived); P1 / P2 are overridden by V2 (V2-loader-aware). | Stays in-repo; reused by V2. |

## Historical artifacts (unchanged from W97 / W98 / W99 / W100 / W101 V1)

W90 / W92 / W88 (cross-modal code) / W81 / W83 / W84 unchanged.
Kept for regression / audit; not active path.

## Dead directions (W102 changes)

| Mechanism | Evidence against | NEW W102 status |
|---|---|---|
| **VLM-Verifier-Final-Turn as load-bearing rescue** | W96-C | UNCHANGED — refuted. |
| **W95-B0 free-text bullet extraction as sufficient on vision-bound benches** | W97 D2-B0 11B | UNCHANGED — refuted. |
| **B1 typed schema *with* `direct_answer_hint`** | W98 B1 11B | UNCHANGED — refuted. |
| **B4 typed schema *without* `direct_answer_hint`** | W99 B4 11B | UNCHANGED — refuted. |
| **Typed-extract-then-text-reason sub-family of W95-B0** | W97 + W98 + W99 11B | UNCHANGED — dead. |
| **W95-B0 family REPAIR via B2 mechanism (image-at-decision-boundary)** | W99 11B PASS + W100 90B FAIL | UNCHANGED — restricted to 11B regime only. |
| **Cross-modal RealWorldQA at the +5 pp Phase 2 bar at any scale** | W97 + W98 + W99 11B + W100 90B all FAIL | **UNCHANGED W102**: arc remains frozen at 11B; W102 does NOT re-open it. |
| **Base MBPP at K=5 same-budget at 70B for retirement** | W91 + W101 V1 loader scratch run | UNCHANGED — capped via ceiling saturation. |
| **MBPP+ V2 at K=5 same-budget at 70B for Phase 2 (cheap-pilot scale)** | W102 70B Phase 2 cheap pilot: B − A1 = −6.67 pp; MLB-2 = 22.22 % (FAIL); 3 a1_only regressions + 1 b_only rescue | **NEW W102 dead direction**: MBPP+ V2 was the W102 lead attack and FAILed the +5 pp Phase 2 bar by 11.67 pp on fresh K=5 sampling at seed 101_001.  The W102 arsenal-mining +5.28 pp prior (re-graded W91 historical responses) did NOT transfer to fresh sampling.  Carry-forward `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`.  Reflexion-mechanism load-bearingness on MBPP-family at 70B is empirically weaker than on HumanEval (22 % vs 47 % rescue rate).  W103 pivot = HumanEval+ via the W102 backup-lane infrastructure. |
| **Cross-bench arsenal-mining priors as substitute for fresh-sampling cheap pilots** | W102 arsenal-mining +5.28 pp prior vs empirical −6.67 pp (11.95 pp swing) | **NEW W102 anti-pattern**: re-grading HISTORICAL responses against a NEW test surface is at most an UPPER BOUND on the mechanism's value on the new surface; the fresh-sampling pilot is the ground truth.  Future runbooks must NOT treat arsenal-mining cross-bench numbers as cheap-pilot earning evidence; they earn the preflight P3 / AddrW101-P3 saturation-margin check but NOT the Phase 2 +5 pp margin bar. |
| **W101 V1 silent-degeneration loader (`plus_input` / `plus_output` parallel-array schema)** | W102 fetch step | **NEW W102 anti-pattern**: against the real EvalPlus release schema, V1 loader silently emits `plus_input = ()` / `plus_output = ()`, V1 executor silently produces 0 extra-test assertions, V1 "MBPP+" cheap pilot silently degenerates to base-MBPP behavior.  V2 is the corrective fix; V1 is marked anti-pattern + historical artifact. |

## Anti-patterns (NEVER promote as core strategy; baseline-only allowed)

**The W97 / W98 / W99 / W100 / W101 anti-pattern list remains
in force VERBATIM in W102**, with one explicit addition:

| Anti-pattern | W102 status |
|---|---|
| Bounded context window as product thesis | UNCHANGED — anti-pattern; baseline-only. |
| Compaction / generic prose summarization as memory mechanism | UNCHANGED — anti-pattern. |
| Shallow token compression without structural reason | UNCHANGED. |
| Context-pruning theater | UNCHANGED. |
| "Cram less / truncate better" as frontier memory system | UNCHANGED. |
| LLM-as-judge in executor chain | UNCHANGED — the W102 V2 + HumanEval+ executors are subprocess CPython; no LLM judges any candidate's correctness. |
| Selective retries | UNCHANGED — V2 + HumanEval+ benches follow the W88 / W90 contract: no early-stop on PASS; every K=5 budget element runs to completion. |
| Single-seed pilots as retirement evidence | UNCHANGED — W102 cheap pilot is a 1-seed × 30-problem cheap pilot AT THE Phase 2 SIZE.  Phase 3 retirement is W104+ if W102 + W103 both PASS. |
| Architecture refinement by vibe | UNCHANGED — W102 V2 is locked by the W102 RUNBOOK's pre-committed schema-fix decision logic. |
| Inventing new candidates after the tournament selected a winner | UNCHANGED — W101 tournament selected MBPP+ LEAD + HumanEval+ BACKUP; W102 ships both; the cheap pilot still attacks MBPP+ only. |
| Re-opening dead directions | UNCHANGED — cross-modal RealWorldQA arc + W95-B0 family + typed-extract sub-family remain dead. |
| **Silent-degeneration via schema assumption** (NEW W102) | The W101 V1 loader is the canonical example: assumed parallel arrays that did not exist in the actual release; would have silently degenerated the cheap pilot to a base-MBPP run.  The W102 V2 P5 probe is the structural defense.  Loaders + executors MUST verify their schema assumption against the real data BEFORE any NIM spend; preflight P-probes must explicitly catch silent-degeneration failure modes. |
| **Cross-bench arsenal-mining priors as cheap-pilot earning evidence** (NEW W102) | The W102 arsenal-mining run showed B − A1 = +5.28 pp on MBPP+ V2 when re-grading W91 historical responses; the empirical fresh-sampling pilot landed at −6.67 pp (11.95 pp swing).  Re-grading historical responses against a new surface measures the *upper bound* of what the mechanism could produce on that surface IF the sampling distribution stays the same; on a fresh seed with fresh K=5 sampling, the mechanism may behave very differently.  Arsenal-mining cross-bench numbers DO earn the preflight P3 / AddrW101-P3 saturation-margin check (the bench is unsaturated; reflexion has surface to attack) but they do NOT earn the +5 pp Phase 2 margin bar.  Future runbooks must keep the cheap pilot as the ground truth and arsenal mining as the cheap pre-flight prior. |

## What W102 cross-promotion is NOT

To pre-empt drift back toward commodity-LLM tricks under a new
name:

### W102 is NOT a cross-modal milestone

* Cross-modal RealWorldQA arc remains frozen at 11B per the
  W100 frontier audit.  W102 does NOT re-open it.
* If the W102 V2 MBPP+ cheap pilot succeeds, the next
  milestone (W103) is the cross-SCALE confirmation of the
  *code-line* result.

### W102 is NOT a Phase 3 retirement attempt

* The cheap pilot is a 1-seed × 30-problem cheap-pilot at the
  Phase 2 SIZE.  Phase 3 retirement is the 3-seed × 100-
  problem × K=5 retirement bench analogous to W89 HumanEval;
  that lives in W104+ if W102 + W103 both PASS.

### A W102 cheap-pilot PASS at 70B is NOT a multi-benchmark same-budget retirement

* A PASS at 70B Phase 2 means the W89 mechanism extends to a
  SECOND benchmark family at the cheap-pilot scale.  Full
  retirement of `W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP`
  requires multi-seed Phase 3 evidence.

## Honest classification of a W102 PASS / FAIL pattern

| W102 V2 preflight + cheap-pilot outcome | What we earn | What we do NOT claim |
|---|---|---|
| Preflight 10/10 PASS + cheap pilot B − A1 ≥ +5 pp + MLB sub-gates clearing at 70B | The W89 sequential-reflexion mechanism extends to a SECOND published code benchmark family (MBPP+ V2 EvalPlus-hardened) at the cheap-pilot scale; the W102 carry-forward is the W103 cross-scale confirmation entitlement. | Multi-seed same-budget retirement on MBPP+ V2 (Phase 3 needed). |
| Preflight 10/10 PASS + cheap pilot B − A1 < +5 pp | Mechanism is *partially* load-bearing on MBPP+ V2 but does not clear the Phase 2 bar; carry-forward `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`; W103 attacks HumanEval+ via backup lane. | Anything Phase 3.  Anything about W89 generalisation if mechanism fails on MBPP+ V2 specifically. |
| Preflight 10/10 PASS + cheap pilot B − A1 ≥ +5 pp + MLB-2 FAILS | `PASS_NON_MECHANISM_DRIVEN`; cross-scale W103 NOT entitled per W96-C / W100 precedent. | Mechanism load-bearingness on MBPP+ V2. |
| Preflight P5 FAILS at re-run | EvalPlus re-released with a non-iteration-shaped test program; loader/executor needs adapting; do NOT launch pilot. | Anything empirical. |
| Preflight P1 / P2 FAIL at re-run | Infrastructure bug in V2 loader or executor; fix; re-run preflight; do NOT launch pilot. | Anything empirical. |

## What this supplement DOES NOT do

* It does NOT claim multi-agent context is solved.
* It does NOT claim the W102 MBPP+ V2 cheap pilot will
  succeed — the pilot is launched after preflight 10/10 PASS;
  the verdict is recorded in
  `docs/RESULTS_W102_MBPP_PLUS_V2_PHASE2_70B_V1.md` after the
  pilot completes.
* It does NOT retire any prior carry-forward.
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.
* It does NOT touch `coordpy/__init__.py`.
* It does NOT re-open any dead direction.
* It does NOT delete `coordpy.mbpp_plus_loader_v1` /
  `coordpy.mbpp_plus_executor_v1` /
  `coordpy.mbpp_plus_reflexion_bench_v1` — they stay in-repo
  as W101 historical artifacts + anti-pattern exhibits.

## Honest scope

This is a *classification supplement* + *anti-drift contract*
+ *critical-finding documentation* recording the W102 V2 fix,
the HumanEval+ backup-lane infrastructure, and the COO-14
helper-lane delivery as active frontier operating-system code.
The empirical pilot verdict lives in
`docs/RESULTS_W102_MBPP_PLUS_V2_PHASE2_70B_V1.md` (populated
at pilot completion).

## Anchors

* `docs/RUNBOOK_W102.md` — W102 pre-commit contract.
* `docs/RESULTS_W102_MBPP_PLUS_LOADER_V2_FIX_V1.md` — V2
  schema-fix doc.
* `docs/RESULTS_W102_MBPP_PLUS_V2_PHASE2_70B_V1.md` — cheap
  pilot verdict (populated post-pilot).
* `docs/RESULTS_W102_HUMANEVAL_PLUS_PREFLIGHT_V1.md` — backup
  lane preflight verdict.
* `docs/RESULTS_W102_CODE_SLICE_SELECTOR_V1.md` — helper-lane
  deliverable doc.
* `docs/RESULTS_W102_MILESTONE_SUMMARY_V1.md` — W102 closing
  summary.
* `coordpy/mbpp_plus_loader_v2.py` etc. — V2 modules.
* `coordpy/humaneval_plus_*.py` — backup-lane modules.
* `coordpy/code_slice_selector_v1.py` — helper-lane module.
* `scripts/run_w102_*.py` — driver scripts.
