# Contamination-control framing (W111 V1) — the resistant ceiling is NOT reflexion-specific

> **2026-05-29 (W111 Lanes α + β + γ).** Extends
> `docs/CONTAMINATION_CONTROL_FRAMING_W110_V1.md`. W111 does NOT add a benchmark
> to the contamination 2×2 and does NOT touch the contamination confound — it
> tests a DIFFERENT MECHANISM, to answer the obvious objection to the W110
> bounded claim. Where this doc and any other disagree on the STATUS of a claim,
> `docs/THEOREM_REGISTRY.md` is authoritative; for the current position,
> `docs/RESEARCH_STATUS.md`.

## Why this doc exists

After W110, the honest bounded claim was: the two retirements are
contamination-EXPOSED-HumanEval-family-specific at 70B; the W89 *reflexion*
mechanism fails on contamination-resistant code (0/2: LiveCodeBench −3.33 pp,
BigCodeBench +0.00 pp). The obvious objection: **"maybe reflexion is just the
wrong mechanism — a better one would win on resistant code."** W111 confronts
that objection head-on instead of accepting the bounded claim by default.

## What W111 tested (a mechanism, not the confound)

The question is NOT "can reflexion look better elsewhere?" but: **is there a
genuinely different, fair, same-budget mechanism that beats A1
(self-consistency) on contamination-resistant code at 70B?**

A NIM-free re-execution census of all 300 W110 BigCodeBench candidates
(`scripts/mine_w111_resistant_failure_modes_v1.py`) localised the resistant
failure:

| failure class | % of 114 resistant failures | mechanism that attacks it |
|---|---|---|
| **SEMANTIC_LOGIC** (assertion / wrong output) | **81.6 %** | M3 (executor-grounded patcher) |
| API_GROUNDING (import/attr/name/sig) | **1.8 %** (both on `/51`, already rescued) | M2 (introspection) |
| TIMEOUT / ENV_HARNESS / OTHER | 16.6 % | — |

Hard-core (8 both-A1+B-fail problems): **6/8 mock-coupling** (the fix needs the
hidden test's mock setup — NOT in the executor `stderr_tail`; a *fair* mechanism
never sees the test source) + **2/8 output-value** (`/15`, `/20`).

## The slate (hypotheses before results)

* **M2 (tool-augmented local symbol/doc introspection)** — attacks
  API-grounding (1.8 %); cannot reveal hidden-test conventions. **KILLED at
  $0 NIM.**
* **M1 (library/spec-grounded planner→coder)** — attacks spec-comprehension; the
  failures are hidden-test-convention, not comprehension, and M1 sacrifices a
  self-consistency sample with no executor grounding. **KILLED at $0 NIM**
  (dominated by M3).
* **M3 (executor-grounded structured-failure patcher)** — typed expected/actual
  contract + minimal-patch on the latest candidate; never the test source; the
  one mechanism aligned with the dominant 81.6 % SEMANTIC class + the executor
  signal. **ADMITTED** to a smallest-decisive live probe.

## The M3 result (smallest-decisive, rescue-concentrated UPPER BOUND)

143 NIM calls, 13-problem hard-core slice, single seed:

> **A0 = 30.77 % / A1 = 30.77 % / M3 = 46.15 %; M3 − A1 = +15.38 pp (UPPER
> BOUND); MLB-1 = 61.5 %, MLB-2 = 12.5 %.** M3-only wins: `/13` (PATCH-LOOP
> rescue — reflexion B failed this) + `/20` (attempt-0 SAMPLING win, NOT the
> mechanism). Did NOT hold `/51` (reflexion B's rescue).

**M3's patch mechanism is sub-reflexion** (12.5 % rescue < reflexion's 25 % <
the 33 % floor) and its margin is non-mechanism-driven (1 patch + 1 sampling
win). It did NOT earn a fair pilot (the pre-committed EARN bar + W104→W105
erosion + W106 margin-cap discipline). Full verdict:
`docs/RESULTS_W111_M3_PATCHER_PROBE_70B_V1.md`.

## What W111 establishes (and does not)

* **The resistant ceiling is NOT reflexion-specific.** Two mechanisms now fail
  to beat A1 on contamination-resistant code at 70B (cheap-pilot scale):
  reflexion (0/2 benchmarks) and a genuinely-different executor-grounded patcher
  (M3, fair pilot not earned, sub-reflexion). It is a property of same-budget
  multi-call mechanisms at 70B against hidden-test-coupling difficulty — which
  makes the bounded contamination-EXPOSED-HumanEval-family claim MORE defensible,
  not less.
* **It does NOT prove the contamination confound** (W111 does not touch the 2×2;
  still two single-seed resistant points). It does NOT retire anything, does NOT
  weaken W89/W105, and does NOT prove "no mechanism can ever win" (a stronger
  model scale is untested; 405B CLOSED).
* **One honest positive:** M3's patch loop rescued `/13`, a hard-core problem
  reflexion could not — the different-mechanism idea is not vacuous, just
  sub-reflexion at this scale.

## The honesty rules (do / do-not)

* **DO say:** "W111 built and probed the one mechanism aligned with the resistant
  failure; it underperformed even reflexion, so the resistant ceiling is not
  reflexion-specific — the bounded contamination-exposed claim is the honest
  ceiling."
* **DO NOT say:** a different mechanism beat A1 on resistant code; M3's +15.38 pp
  is a real margin (it's a rescue-concentrated upper bound + a sampling win);
  M3 is worthless (it rescued `/13`); W111 proves no mechanism can ever win; W111
  retires anything or proves the confound; multi-agent context is solved.

## Anchors

* `docs/RUNBOOK_W111.md` — the pre-commit contract (locked + committed before NIM).
* `docs/RESULTS_W111_M3_PATCHER_PROBE_70B_V1.md` — the full M3 probe verdict.
* `results/w111/mechanism_mining/w110_bcb_failure_census.json` — the $0 census.
* `docs/CONTAMINATION_CONTROL_FRAMING_W110_V1.md` — the resistant 2×2 (unchanged by W111).
* `docs/THEOREM_REGISTRY.md` — authoritative claim status.
