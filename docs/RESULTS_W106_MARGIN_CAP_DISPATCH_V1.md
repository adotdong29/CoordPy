# W106 — Llama-3.1 margin-cap dispatch decision (V1)

> **2026-05-28.  The honest decision on the W105
> `meta/llama-3.1-70b-instruct` `FAIL_MARGIN` branch, computed by
> a pre-committed two-gate rule (`coordpy.margin_cap_dispatch_v1`)
> from the W105 consolidated Phase 3 retirement verdict.  No NIM
> call.**
>
> **Decision: `NO_GO`.**  A cheap multi-seed confirmation at
> Llama-3.1-70B is NOT honestly earned.  Accept the bounded
> single-class claim on `meta/llama-3.3-70b-instruct`.  **$0 NIM
> spent on the Llama-3.1 branch.**

## Why this decision had to be made

W105 closed SPLIT: `meta/llama-3.3-70b-instruct` RETIRED HumanEval+
at Phase 3 (+7.00 pp; 6/6 bars), `meta/llama-3.1-70b-instruct`
FAIL_MARGIN (+2.33 pp; 5/6 bars; only the margin bar fails).  The
pre-committed `docs/RESULTS_W105_W106_PLANNING_V1.md` Verdict C /
sub-case C1 routes the Llama-3.1 failure mode to the W104 RUNBOOK
§ Branch C dispatch table, which offers two honest moves: run a
HumanEval+ multi-seed cheap confirmation at Llama-3.1-70B on a
rescue-concentrated slice, **OR** explicitly accept the bounded
single-class claim and stop.  W106's job (per
`docs/RUNBOOK_W106.md` § 2) is to decide this with a pre-committed
rule, not on vibes.

## The two-gate decision rule (locked BEFORE evaluation)

### GATE 1 — Entitlement (pre-committed W104/W105 Branch C table)

| Failure signature | Entitled next step | NIM ceiling |
|---|---|---|
| margin < 0 AND MLB-2 < 33 % | LiveCodeBench preflight (NIM-free) | $0 |
| A1 ≥ 90 % (G2 saturation) | APPS preflight (NIM-free) | $0 |
| **0 ≤ margin < +5 pp AND MLB-2 ≥ 33 % AND A1 < 90 %** | **multi-seed cheap confirmation at affected class** | **~990 NIM calls** |
| margin < 0 AND MLB-2 ≥ 33 % AND A1 < 90 % | cross-scale-collapse audit + retired-class confirmation | ~3 300 NIM calls |

**Llama-3.1 W105 signature**: margin = +2.33 pp ∈ [0, +5);
MLB-2 = 50.54 % ≥ 33 %; A1 = 86.33 % < 90 %.  → matched row 3
→ **GATE 1 = ENTITLED** (cheap confirmation; ~990 NIM calls).

### GATE 2 — Verdict-changing power (the W106 margin-cap discipline)

A cheap confirmation runs only if it can HONESTLY convert the
fair-slice `FAIL_MARGIN` into `RETIRED`.  Requires ALL THREE:

| sub-gate | meaning | Llama-3.1 result |
|---|---|---|
| 2a fair battlefield | proposed slice is representative, NOT rescue-concentrated | **FAIL** — the only form Verdict C/C1 offers is a rescue-concentrated slice = an UPPER BOUND (W102 anti-pattern) |
| 2b no authoritative fair result yet | no fair broad-slice multi-seed Phase 3 verdict already exists | **FAIL** — W105 already ran the fair 100-problem × 3-seed Phase 3 bench (+2.33 pp); a cheaper re-run cannot overturn it |
| 2c fixable confound | the FAIL is a fixable confound, NOT a clean magnitude miss | **FAIL** — executor clean 100 %, budget byte-exact, slice CID matched, per-seed majority 3/3, MLB-2 50.54 % healthy = a clean true magnitude miss, nothing to fix |

**GATE 2 = FAIL** on all three sub-conditions.

### Decision

GO iff GATE 1 ENTITLED **AND** (2a ∧ 2b ∧ 2c).  GATE 2 fails ⇒
**`NO_GO`**.  Decision CID `de3dfb02…` (machine-readable at
`results/w106/margin_cap_dispatch/margin_cap_dispatch_decision.json`).

## The structural argument, in one paragraph

Retirement is defined on the **fair broad slice**.  W105 already
ran that fair broad-slice multi-seed Phase 3 bench on Llama-3.1-70B
and it returned +2.33 pp `FAIL_MARGIN` — that is the authoritative
verdict.  A rescue-concentrated cheap confirmation deliberately
concentrates the A1-failure problems where reflexion can rescue,
which inflates B − A1 by construction; it is the W102
"cheap-pilot-margin-is-an-upper-bound" anti-pattern (already
demonstrated by the W104 +10 pp on the 30-problem rescue-
concentrated inner kernel, which did NOT survive scale-up to the
broad slice — A1 rose to 86.33 %, compressing reflexion headroom).
Such a confirmation could only re-confirm an upper bound we
already know; it cannot convert a fair-slice FAIL_MARGIN into a
retirement.  And there is no confound to fix: the run was clean on
every anti-cheat axis.  Therefore the only honest move is to
ACCEPT the bounded single-class claim and STOP.  Spending ~990 NIM
calls here would buy nothing but emotional comfort — exactly the
spend the margin-cap discipline forbids.

## What this is NOT

* It is NOT mechanism collapse.  MLB-2 = 50.54 % is well above the
  33 % load-bearing floor; the W89 mechanism is real on
  Llama-3.1-70B.  This is a margin-MAGNITUDE miss on the broad
  slice, not a mechanism failure.
* It is NOT a refutation of the W104 cheap-pilot PASS — that PASS
  was on a rescue-concentrated slice and stands as a cheap-pilot
  (upper-bound) result.
* It is NOT a deferral.  The Llama-3.1 margin-cap branch is
  CLOSED with `W106-L-HUMANEVAL-PLUS-LLAMA31-70B-MARGIN-CAP-CHEAP-
  CONFIRMATION-NOT-EARNED-CAP`.  It can only re-open via a
  genuinely different battlefield (e.g. cross-scale-UP to 405B if
  it becomes reachable), never via a rescue-concentrated re-run.

## Carry-forward registered

`W106-L-HUMANEVAL-PLUS-LLAMA31-70B-MARGIN-CAP-CHEAP-CONFIRMATION-NOT-EARNED-CAP`
— see `docs/THEOREM_REGISTRY.md` § W106.

## Anchors

* `docs/RUNBOOK_W106.md` § 2 — the pre-committed two-gate rule.
* `coordpy/margin_cap_dispatch_v1.py` — the rule implementation
  (20 PASSing unit tests in `tests/test_w106_margin_cap_dispatch_v1.py`).
* `scripts/run_w106_margin_cap_dispatch.py` — the driver.
* `results/w106/margin_cap_dispatch/margin_cap_dispatch_decision.json`
  — machine-readable decision (CID `de3dfb02…`).
* `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA31_V1.md` — the
  W105 Llama-3.1 FAIL_MARGIN verdict this decision dispatches.
