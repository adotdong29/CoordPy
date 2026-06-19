# W105 — HumanEval+ Phase 3 retirement bench, class `meta/llama-3.3-70b-instruct` (V1)

> **2026-05-28.  Per-class Phase 3 retirement verdict for the
> FIRST earned model class, evaluated INDEPENDENTLY of the
> second class per the W105 RUNBOOK § "Phase 3 retirement-grade
> evaluation discipline" (evaluate each class separately first;
> never average a class-specific result into the other).**
>
> **Verdict: `RETIRED` (6 / 6 retirement bars PASS; MLB-2 load-
> bearing at 55.62 %).**

## Headline

* **Model class**: `meta/llama-3.3-70b-instruct` (the W89 +
> W103 retirement model class).
* **Verdict label**: `RETIRED` — all 6 W88 / W89 / W95 Phase 3
  retirement bars PASS, and MLB-2 rescue rate is above the
  33 % load-bearing floor.
* **Mean B − A1 across 3 seeds**: **+7.00 pp** (floor +5 pp).
* **Per-seed majority**: 3 / 3 seeds have B > A1.
* **Per-problem majority**: 295 / 300 problem-seed cells have
  B ≥ A1 (floor 159 = 53 % of 300).
* **A1 not saturated**: 84.00 % / 82.00 % / 82.00 % on the 3
  cells (all < 90 %).
* **Mean MLB-2 rescue rate**: 55.62 % (load-bearing).

## Per-cell results (3 seeds × 100 problems × K = 5)

| seed | A0 % | A1 @ K=5 % | B (seq-reflexion K=5) % | B − A1 pp | MLB-1 invocation % | MLB-2 rescue % | wall (s) | bench Merkle |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 105001 | 78.00 | 84.00 | 89.00 | +5.00 | 22.00 | 50.00 | 22394.2 | `82ed16f0d3c131c9...` |
| 105002 | 78.00 | 82.00 | 91.00 | +9.00 | 23.00 | 60.87 | 21541.4 | `ea4c791c25d43b2a...` |
| 105003 | 78.00 | 82.00 | 89.00 | +7.00 | 25.00 | 56.00 | (see report) | `d00737d0cb80ea3d...` |
| **mean** | **78.00** | **82.67** | **89.67** | **+7.00** | **23.33** | **55.62** | — | — |

## Retirement bars (W88 / W89 / W95 6-bar shape)

| # | Bar | Value | Threshold | PASS |
|---|---|---|---|---|
| 1 | Margin (mean B − A1 across seeds) | +7.00 pp | ≥ +5 pp | YES |
| 2 | Per-seed majority (B > A1) | 3 / 3 | ≥ 2 / 3 | YES |
| 3 | Per-problem majority (B ≥ A1) | 295 / 300 | ≥ 159 (53 %) | YES |
| 4 | A1 not saturated (per cell) | 84 / 82 / 82 % | all < 90 % | YES |
| 5 | Audit chain re-derives | 3 / 3 | ≥ 2 | YES |
| 6 | Executor stays clean | 100.00 % | = 100 % | YES |

## Mechanism-load-bearingness (MLB) at Phase 3 scale

* **MLB-1 (reflexion-cycle invocation rate)**: mean 23.33 %
  across the 3 cells.  This is BELOW the 33 % Phase-2 sub-gate
  floor — but MLB-1 is NOT a Phase 3 retirement bar.  At
  Phase 3 scale the slice's broad corpus-fill keeps A1 strong
  (mean 82.67 %), so attempt-0 fails on only ~ 23 % of problems
  (i.e., the reflexion cycle is only INVOKED on the harder
  quarter of the slice).
* **MLB-2 (reflexion rescue rate)**: mean 55.62 % across the 3
  cells (50.00 / 60.87 / 56.00 %).  This is the load-bearing
  signal: of the problems where attempt 0 fails and reflexion
  is invoked, the sequential-reflexion mechanism rescues
  ~ 56 %.  Above the 33 % floor on every cell — the W89
  mechanism is doing real work, NOT riding sampling variance.
* The MLB-2 rescue rate (55.62 %) is consistent with the W89
  base-HumanEval retirement template (47 %) and the W103
  HumanEval+ cheap pilot (47.06 %) at the SAME model class.

## Honest scope (what this per-class verdict DOES claim)

* The W89 sequential-reflexion mechanism RETIRES on HumanEval+
  at Phase 3 multi-seed scale on `meta/llama-3.3-70b-instruct`:
  3 seeds × 100 problems × K = 5; same-budget byte-exact;
  margin +7.00 pp; mechanism load-bearing (MLB-2 = 55.62 %).
* This is a SECOND confirmed multi-seed same-budget multi-agent
  superiority retirement on this model class — the FIRST was
  W89 on base HumanEval at +5.56 pp; W105 extends it to the
  EvalPlus-hardened HumanEval+ at +7.00 pp.

## Honest scope (what this per-class verdict does NOT claim)

* It does NOT claim cross-class retirement — that requires the
  second class (`meta/llama-3.1-70b-instruct`) to clear all 6
  bars independently AND the cross-class B − A1 difference to
  stay within ± 5 pp.  See the cross-class verdict doc.
* It does NOT claim cross-scale-UP generalisation — 405B was
  unreachable on NIM at the W105 run window (re-probed; HTTP
  404).
* It does NOT claim MBPP-family generalisation — the W102 cap
  stands.
* It does NOT claim cross-modal generalisation — RealWorldQA
  frozen at 11B per W100.
* It does NOT claim "multi-agent context is solved".

## Provenance (anti-cheat carry-forward verbatim from W88 – W104)

* Slice pack CID `8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314`
  (verified at run start; pilot refuses to run on mismatch).
* Inner-kernel CID `c35155956ece605c...` (W103 helper-priority
  30-problem slice preserved at the head).
* Corpus SHA-256 `908377f1daf28dcb...` (verified at run start).
* Preflight verdict CID `4f57a2cf...` (W102/W103 reused).
* Same model on every arm within a cell.
* Same K = 5 byte-exact budget on A1 / B; sequential reflexion
  runs the full K = 5 (no early-stop).
* Executor = `coordpy.humaneval_plus_executor_v1.run_humaneval_plus_executor_v1`;
  no LLM judge; subprocess CPython.
* Per-call sidecars + per-seed Merkle + bench Merkle re-derive
  offline (audit chain 3/3).

## Anchors

* `docs/RUNBOOK_W105.md` — pre-commit contract.
* `coordpy/phase3_retirement_evaluator_v1.py` — the 6-bar
  evaluator that produced this verdict.
* `results/w105/humaneval_plus_phase3_retirement_bench/w105_phase3_20260527T125554Z_full_meta__slash__llama-3.3-70b-instruct/` —
  the 3 cell run dirs + per-cell verdicts + bench reports.
* `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA31_V1.md` —
  the SECOND class's independent verdict.
* `docs/RESULTS_W105_CROSS_CLASS_COMPARATOR_V1.md` — the
  cross-class comparator + entitlement verdict.
