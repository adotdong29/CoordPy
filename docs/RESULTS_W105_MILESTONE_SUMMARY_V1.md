# W105 — Milestone summary V1

> **2026-05-28.  Three-lane milestone: HumanEval+ Phase 3
> retirement bench (the earned 6 600-call run) + run-hardening +
> W106 contingency.  Verdict: SPLIT — `meta/llama-3.3-70b-instruct`
> RETIRED (6/6 bars; mean B − A1 = +7.00 pp); `meta/llama-3.1-70b-instruct`
> FAIL_MARGIN (5/6 bars; mean B − A1 = +2.33 pp, mechanism still
> load-bearing at MLB-2 = 50.54 %).  Cross-class retirement NOT
> entitled (only one class cleared).  The bounded claim is a
> SECOND confirmed multi-seed same-budget multi-agent superiority
> retirement, single-class on Llama-3.3-70B HumanEval+.**

## Headline

* **Lead-lane verdict**: SPLIT.
  * **Llama-3.3-70B-Instruct → RETIRED** (the W89 + W103
    retirement model class; mean B − A1 = +7.00 pp across 3
    seeds; per-cell +5/+9/+7; 6/6 retirement bars; MLB-2 =
    55.62 % load-bearing).
  * **Llama-3.1-70B-Instruct → FAIL_MARGIN** (mean B − A1 =
    +2.33 pp; per-cell +5/+1/+1; 5/6 bars — only the margin bar
    fails; MLB-2 = 50.54 % still load-bearing; per-seed majority
    3/3).
* **Cross-class retirement**: NOT ENTITLED — the W105 RUNBOOK
  rule requires BOTH classes to clear all 6 bars; only one did.
* **Programme entitlement after W105**: a SECOND confirmed
  multi-seed same-budget multi-agent superiority retirement
  (after W89), bounded to HumanEval+ on `meta/llama-3.3-70b-instruct`.
* **Pack CID reused unchanged**: YES — `8be55f3bf1650df3...`
  byte-for-byte; inner kernel CID `c35155956ece605c...` and
  corpus SHA `908377f1daf28dcb...` verified at every cell's run
  start.
* **Core two-class matrix**: stayed FIXED (Llama-3.3-70B +
  Llama-3.1-70B); 405B never entered the core run.
* **405B reachability smoke**: attempted (HTTP 404 — still
  unreachable on NIM, same as W104); did NOT change the matrix.

## What W105 delivered

### Lane 1 — Lead lane (Phase 3 retirement bench)

* Executed the pre-built W105 Phase 3 slice pack BYTE-FOR-BYTE
  unchanged: 3 seeds (105 001 / 105 002 / 105 003) × 100
  problems × K = 5 × 2 model classes = **6 600 NIM calls**.
* Per-(model, seed) cell isolation; each cell's pack CID + inner
  kernel CID + corpus SHA verified at run start.
* Two model classes run as parallel processes (one per class);
  3 seeds serial within each class.  Total wall ≈ 26 h
  (Llama-3.3 finished ~hour 16; Llama-3.1 ~hour 26; Llama-3.1
  was slower per-call under heavier NIM 429 throttling).
* Per-class evaluation FIRST (the load-bearing surface), then
  the cross-class entitlement layer.

#### Per-class results

| Class | A0 (mean) | A1@K=5 (mean) | B (mean) | mean B − A1 | per-cell B − A1 | per-seed maj | MLB-2 | verdict |
|---|---:|---:|---:|---:|---|---:|---:|---|
| Llama-3.3-70B | 78.00 % | 82.67 % | 89.67 % | **+7.00 pp** | +5/+9/+7 | 3/3 | 55.62 % | **RETIRED** |
| Llama-3.1-70B | 79.00 % | 86.33 % | 88.67 % | **+2.33 pp** | +5/+1/+1 | 3/3 | 50.54 % | **FAIL_MARGIN** |

#### Why the split

The W104 cross-generation cheap pilot delivered +10.00 pp on
Llama-3.1-70B — but on the 30-problem rescue-concentrated W103
inner kernel.  The Phase 3 slice broadens to 100 problems (45 %
mid-shell `shared_wins` + 25 % corpus-fill).  On that broad
distribution Llama-3.1's A1 @ K = 5 rises to 86.33 %, leaving
reflexion little headroom (invoked on ~23 % of problems,
rescuing ~50 % of those → +2.33 pp net).  Llama-3.3 holds a
lower A1 (82.67 %), so reflexion retains +7.00 pp of headroom.
This confirms the W102 anti-pattern lesson (cheap-pilot margins
are upper bounds) and the cross-scale/cross-class collapse
pattern (W96-A / W96-C / W100) — the margin can erode on a
broader slice even when the mechanism stays load-bearing.

### Lane 2 — Ops/hardening lane

Durable guardrails that materially mattered on the 6 600-call
run:

* `coordpy/phase3_retirement_evaluator_v1.py` — per-class +
  cross-class 6-bar retirement evaluator (NEW explicit-import-
  only module; refuses to run on slice pack / corpus / duplicate
  seed / schema mismatch).
* `coordpy/cross_class_comparator_v1.py` — per-seed-aligned
  cross-class comparator (NEW explicit-import-only module; fixes
  the W104 V1 row-misalignment by matching seeds; refuses to run
  on iteration-task-id / slice / corpus / seed-set mismatch).
* `scripts/run_w105_phase3_retirement_bench.py` — Phase 3 driver
  with per-(model, seed) cell isolation, canary, resume-safe
  per-cell skipping (`phase3_cell_verdict.json` marks a cell
  complete), mid-run global + per-cell `progress.json`, automatic
  per-cell partial audit + partial per-class verdict emission,
  explicit 429 / 502 / socket-hang handling with per-cell
  `retry_log.jsonl`.
* `scripts/run_w105_canary_smoke.py` — 66-call canary BEFORE the
  full launch.
* `scripts/run_w105_405b_reachability_probe.py` — cheap 405B
  smoke (independent of the main run).
* `scripts/run_w105_consolidate.py` — post-hoc consolidator for
  the two parallel per-class run roots.
* `tests/test_w105_phase3_discipline_v1.py` — 18 PASSing unit
  tests.
* The hardening proved its value in-run: the Llama-3.3 class
  finished ~10 h before Llama-3.1; per-cell isolation meant each
  completed cell's evidence was durable; the auto-emitted partial
  per-class verdicts let the run be inspected mid-flight without
  disturbing it.

### Lane 3 — W106 planning lane

* `docs/RESULTS_W105_W106_PLANNING_V1.md` pre-committed W106
  under all five verdict shapes BEFORE the W105 verdict.  The
  empirical SPLIT outcome maps to **Verdict C, sub-case C1**
  (Llama-3.3 RETIRED, Llama-3.1 FAIL).
* W106 (pre-committed) = bounded-claim milestone on Llama-3.3-70B
  + the W104 RUNBOOK § Branch C dispatch keyed to the Llama-3.1
  failure mode (margin < +5 pp but ≥ 0, MLB-2 ≥ 33 %, A1 < 90 %
  → HumanEval+ multi-seed cheap confirmation at Llama-3.1-70B on
  a rescue-concentrated slice, OR accept the bounded single-class
  claim).
* The W106 dispatch JSON is machine-readable in the planning doc.

## Branch decision applied per pre-locked logic

* Per the W105 RUNBOOK § Lead lane decision logic + the W106
  planning artifact, the empirical SPLIT triggers **Verdict C
  sub-case C1**.
* Carry-forward added:
  `W105-L-HUMANEVAL-PLUS-RETIREMENT-LLAMA31-70B-MARGIN-CAP`.
* Carry-forward added (positive):
  `W105-L-HUMANEVAL-PLUS-RETIREMENT-LLAMA33-70B-PASS` — the
  SECOND confirmed multi-seed same-budget multi-agent
  superiority retirement (after W89), single-class on
  Llama-3.3-70B HumanEval+.
* Carry-forward retired: NONE.  W89 remains a confirmed
  retirement; W105 ADDS a second confirmed retirement on a
  different benchmark family at the same model class — it does
  not retire any prior cap.

`COO-9` REMAINS the lead path.

## 405B reachability smoke

* `scripts/run_w105_405b_reachability_probe.py` re-probed
  `meta/llama-3.1-405b-instruct` on NIM: **HTTP 404** (still
  not hosted), 222 ms.  Recorded at
  `results/w105/405b_reachability_probe/`.  Did NOT change the
  W105 core matrix.  The
  `W104-L-HUMANEVAL-PLUS-CROSS-SCALE-UP-PRIMARY-TARGET-405B-UNREACHABLE-ON-NIM-CAP`
  carry-forward stands.

## Empirical event log

| Step | Outcome |
|---|---|
| W105 RUNBOOK locked BEFORE any NIM call | YES |
| Hardening lane code + 18 unit tests PASS BEFORE any NIM call | YES |
| W106 planning lane shipped BEFORE the verdict | YES |
| 405B reachability smoke | HTTP 404 (unchanged); core matrix unaffected |
| Canary smoke (66 NIM calls) | PASS both classes (B − A1 ≥ −5 pp floor) |
| Pack CID reused unchanged | YES (`8be55f3bf1650df3...`) |
| Core two-class matrix changed | NO (stayed Llama-3.3-70B + Llama-3.1-70B) |
| Phase 3 NIM calls | 6 600 (3 seeds × 100 × K=5 × 2 classes) |
| Llama-3.3-70B per-class verdict | RETIRED (6/6 bars; +7.00 pp; MLB-2 55.62 %) |
| Llama-3.1-70B per-class verdict | FAIL_MARGIN (5/6 bars; +2.33 pp; MLB-2 50.54 %) |
| Cross-class retirement entitled | NO (only one class cleared) |
| MLB reporting load-bearing at Phase 3 | YES (both classes MLB-2 > 33 %) |
| Cross-class comparator (per-seed-aligned) | 242 stayed / 27 improved / 28 regressed / 3 flipped (clean) |
| Carry-forwards added | 2 (Llama-3.3 retirement PASS + Llama-3.1 margin cap) |
| Carry-forwards retired | 0 |
| Discipline validation # | 15th (W93 … W105) |
| Stable boundary preserved | YES (`coordpy.__version__=0.5.20`; `SDK_VERSION=coordpy.sdk.v3.43`; no PyPI; `coordpy/__init__.py` untouched) |
| New coordpy.* modules | 2 (`phase3_retirement_evaluator_v1` + `cross_class_comparator_v1`; explicit-import only) |
| Tests added | 18 (`tests/test_w105_phase3_discipline_v1.py`) |

## What W105 IS entitled to claim

* A SECOND confirmed multi-seed same-budget multi-agent
  superiority retirement (after W89): the W89 sequential-
  reflexion mechanism retires on **HumanEval+** at Phase 3
  multi-seed scale on **`meta/llama-3.3-70b-instruct`** (3 seeds
  × 100 problems × K = 5; +7.00 pp; MLB-2 load-bearing).
* The mechanism is load-bearing on BOTH model classes at Phase 3
  scale (MLB-2 55.62 % / 50.54 %) — it is doing real work, not
  riding sampling variance.
* The W93 – W105 preflight-first + cross-scale + multi-candidate-
  tournament-then-confirm + mechanism-load-bearingness +
  silent-degradation-anti-pattern-guard + arsenal-mining-prior-
  anti-pattern-guard + cross-class-row-alignment discipline now
  has FIFTEEN consecutive validations.

## What W105 does NOT claim

* It does NOT claim cross-class retirement (Llama-3.1-70B
  FAILed the margin bar).
* It does NOT claim cross-scale-UP retirement (405B unreachable).
* It does NOT claim MBPP-family retirement (W102 cap stands).
* It does NOT claim cross-modal retirement (RealWorldQA frozen
  at 11B per W100).
* It does NOT claim "multi-agent context is solved".
* It does NOT refute the W104 Llama-3.1 cheap-pilot PASS — that
  was a different (rescue-concentrated) slice; W105 shows the
  margin does not GENERALISE to the broad Phase 3 slice.

## W106 (obvious by the end of W105)

Per the SPLIT → Verdict C sub-case C1 mapping, W106 is a
bounded-claim + targeted-confirmation milestone:

1. **Register the Llama-3.3-70B HumanEval+ Phase 3 retirement**
   as the second confirmed retirement (theorem registry).
2. **Bounded claim** — do NOT claim cross-class; the Llama-3.1
   margin cap stands.
3. **Optional targeted confirmation** — HumanEval+ multi-seed
   cheap pilot at Llama-3.1-70B on a rescue-concentrated slice
   to test whether the +5 pp bar is reachable when the slice is
   not diluted by corpus-fill (the W104 cheap-pilot signal
   suggests it might be on a concentrated slice).  This is the
   Branch C dispatch entry for "margin < +5 pp but ≥ 0, MLB-2 ≥
   33 %".
4. **405B reachability remains a standing extension** — if 405B
   becomes hosted, the cross-scale-UP attempt re-opens.

`COO-9` remains the lead path.

## Anchors

* `docs/RUNBOOK_W105.md` — pre-commit contract.
* `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA33_V1.md` —
  Llama-3.3 RETIRED verdict.
* `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA31_V1.md` —
  Llama-3.1 FAIL_MARGIN verdict.
* `docs/RESULTS_W105_CROSS_CLASS_COMPARATOR_V1.md` — cross-class
  comparator + entitlement.
* `docs/RESULTS_W105_W106_PLANNING_V1.md` — W106 dispatch.
* `docs/FRONTIER_RELEVANCE_AUDIT_W105_V1.md` — 15th preflight-
  discipline validation.
* `coordpy/phase3_retirement_evaluator_v1.py` +
  `coordpy/cross_class_comparator_v1.py` — the two new modules.
* `results/w105/humaneval_plus_phase3_retirement_bench/w105_phase3_FINAL_consolidated/` —
  unified verdict + cross-class comparator.
