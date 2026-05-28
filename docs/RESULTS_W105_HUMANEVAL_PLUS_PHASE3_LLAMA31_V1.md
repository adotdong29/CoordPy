# W105 — HumanEval+ Phase 3 retirement bench, class `meta/llama-3.1-70b-instruct` (V1)

> **2026-05-28.  Per-class Phase 3 retirement verdict for the
> SECOND earned model class, evaluated INDEPENDENTLY of the
> first class per the W105 RUNBOOK § "Phase 3 retirement-grade
> evaluation discipline" (evaluate each class separately first;
> never average a class-specific result into the other).**
>
> **Verdict: `FAIL_MARGIN` (5 / 6 retirement bars PASS; only the
> margin bar FAILs at mean B − A1 = +2.33 pp; mechanism STILL
> load-bearing at MLB-2 = 50.54 %).**

## Headline

* **Model class**: `meta/llama-3.1-70b-instruct` (the W104
  cross-generation cheap-pilot model class).
* **Verdict label**: `FAIL_MARGIN` — 5 of 6 retirement bars
  PASS; the margin bar (mean B − A1 ≥ +5 pp) FAILs.
* **Mean B − A1 across 3 seeds**: **+2.33 pp** (floor +5 pp;
  MISS by 2.67 pp).
* **Per-seed majority**: 3 / 3 seeds have B > A1 (the mechanism
  is directionally positive on every seed).
* **Per-problem majority**: 294 / 300 problem-seed cells have
  B ≥ A1 (floor 159).
* **A1 not saturated**: 85.00 % / 87.00 % / 87.00 % (all
  < 90 %).
* **Mean MLB-2 rescue rate**: 50.54 % (load-bearing — the
  mechanism is doing real work where invoked; the FAIL is a
  margin-size failure, NOT a mechanism-collapse failure).

## Per-cell results (3 seeds × 100 problems × K = 5)

| seed | A0 % | A1 @ K=5 % | B (seq-reflexion K=5) % | B − A1 pp | MLB-1 invocation % | MLB-2 rescue % | bench Merkle |
|---|---:|---:|---:|---:|---:|---:|---|
| 105001 | 79.00 | 85.00 | 90.00 | +5.00 | 24.00 | 58.33 | `acf956e8d56affd2...` |
| 105002 | 79.00 | 87.00 | 88.00 | +1.00 | 22.00 | 45.45 | `3e3c0301dbb37787...` |
| 105003 | 79.00 | 87.00 | 88.00 | +1.00 | 23.00 | 47.83 | `1011a848aca8cdb8...` |
| **mean** | **79.00** | **86.33** | **88.67** | **+2.33** | **23.00** | **50.54** | — |

## Retirement bars (W88 / W89 / W95 6-bar shape)

| # | Bar | Value | Threshold | PASS |
|---|---|---|---|---|
| 1 | Margin (mean B − A1 across seeds) | **+2.33 pp** | ≥ +5 pp | **NO** |
| 2 | Per-seed majority (B > A1) | 3 / 3 | ≥ 2 / 3 | YES |
| 3 | Per-problem majority (B ≥ A1) | 294 / 300 | ≥ 159 (53 %) | YES |
| 4 | A1 not saturated (per cell) | 85 / 87 / 87 % | all < 90 % | YES |
| 5 | Audit chain re-derives | 3 / 3 | ≥ 2 | YES |
| 6 | Executor stays clean | 100.00 % | = 100 % | YES |

## Why the margin failed at Phase 3 scale (honest diagnosis)

The W104 cross-generation cheap pilot on this SAME model class
produced B − A1 = **+10.00 pp** — but on the 30-problem
helper-anchored slice (the W103 inner kernel, which is 63 %
rescue-surface-concentrated `b_only_wins` + `shared_fails` +
`a1_only_wins`).  The W105 Phase 3 slice expands to 100
problems with 45 % mid-shell `shared_wins` + 25 % broad
corpus-fill — a much EASIER distribution for the strong A1
baseline.

On the broad Phase 3 slice, Llama-3.1-70B's A1 @ K = 5 rises to
86.33 % mean (vs 53.33 % on the helper-anchored W104 slice).
With A1 already catching ~86 % of problems, the sequential-
reflexion mechanism has little headroom: it is INVOKED on only
~23 % of problems (MLB-1) and RESCUES ~50 % of those (MLB-2),
which nets out to only +2.33 pp on the full slice.

This is the W102 anti-pattern lesson confirmed once more: the
cheap-pilot margin (+10 pp on a rescue-concentrated slice) is
an UPPER BOUND, not a Phase 3 prediction.  It is ALSO consistent
with the cross-scale / cross-class collapse pattern
(W96-A / W96-C / W100): the margin can erode meaningfully when
the evaluation distribution broadens, even when the mechanism
itself stays load-bearing.

Critically, the failure is NOT a mechanism collapse:
* MLB-2 = 50.54 % (well above the 33 % floor) — reflexion still
  rescues half the problems it is invoked on.
* Per-seed majority 3 / 3 — B beats A1 on every seed.
* The mechanism is directionally positive everywhere; it is the
  MAGNITUDE that misses the +5 pp retirement bar on the broad
  slice.

## Failure-mode classification (for W106 dispatch)

Per the W104 RUNBOOK § Branch C dispatch table, the Llama-3.1
failure signature is:

* Margin < +5 pp but ≥ 0 pp (+2.33 pp): YES.
* MLB-2 ≥ 33 % (50.54 %): YES.
* A1 not saturated (< 90 %): YES.

This is the **"per-seed sampling / distribution-broadening"
failure mode** → W106 lead step for this class =
HumanEval+ multi-seed cheap confirmation at Llama-3.1-70B on a
rescue-concentrated slice, OR a bounded class-specific claim.
See `docs/RESULTS_W105_W106_PLANNING_V1.md` § Branch C / Verdict
C sub-case C1.

## Carry-forward registered

`W105-L-HUMANEVAL-PLUS-RETIREMENT-LLAMA31-70B-MARGIN-CAP` —
the W89 sequential-reflexion mechanism does NOT clear the +5 pp
Phase 3 retirement margin bar on `meta/llama-3.1-70b-instruct`
at the 100-problem multi-seed slice (mean B − A1 = +2.33 pp),
even though it stays load-bearing (MLB-2 = 50.54 %) and
directionally positive (per-seed majority 3/3).  The W104
cheap-pilot +10 pp on the 30-problem rescue-concentrated slice
did NOT survive scale-up to the broad Phase 3 slice.

## Honest scope (what this per-class verdict does NOT claim)

* It does NOT retire HumanEval+ on Llama-3.1-70B at Phase 3.
* It does NOT refute the W104 cheap-pilot PASS — that PASS was
  on a different (rescue-concentrated) slice and stands as a
  cheap-pilot result.  W105 shows the margin does not GENERALISE
  to the broad Phase 3 slice at retirement scale.
* It does NOT cap the mechanism's load-bearingness — MLB-2 stays
  above the floor.
* It does NOT bear on the Llama-3.3-70B verdict, which RETIRED
  independently (see the Llama-3.3 per-class doc).

## Provenance (anti-cheat carry-forward verbatim from W88 – W104)

* Slice pack CID `8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314`
  (verified at run start).
* Corpus SHA-256 `908377f1daf28dcb...` (verified at run start).
* Preflight verdict CID `4f57a2cf...`.
* Same model on every arm within a cell; same K = 5 byte-exact
  budget; executor = `coordpy.humaneval_plus_executor_v1`; no
  LLM judge; per-call sidecars + per-seed Merkle + bench Merkle
  re-derive offline (audit chain 3/3).

## Anchors

* `docs/RUNBOOK_W105.md` — pre-commit contract.
* `coordpy/phase3_retirement_evaluator_v1.py` — the 6-bar
  evaluator that produced this verdict.
* `results/w105/humaneval_plus_phase3_retirement_bench/w105_phase3_20260527T125556Z_full_meta__slash__llama-3.1-70b-instruct/` —
  the 3 cell run dirs + per-cell verdicts + bench reports.
* `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA33_V1.md` —
  the FIRST class's independent verdict (RETIRED).
* `docs/RESULTS_W105_CROSS_CLASS_COMPARATOR_V1.md` — the
  cross-class comparator + entitlement verdict.
