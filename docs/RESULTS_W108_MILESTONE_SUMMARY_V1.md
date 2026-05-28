# W108 — Milestone summary (LiveCodeBench real-data bug-fix + cheap-pilot FAIL + APPS readiness)

**One line:** W108 fixed a real-data binding bug in the partial LiveCodeBench
scaffold, EARNED the cheap pilot on the real contamination-resistant corpus,
ran it — and it is a clean Phase-2 FAIL (B − A1 = −3.33 pp; MLB-2 = 25 %).
The two confirmed retirements (W89, W105) STAND; W108 adds NO retirement and
introduces an explicit contamination-resistant boundary. `COO-9` stays lead.
$0 expensive 405B; the one expensive run was the earned 330-call cheap pilot.

W108 is NOT a new broad tournament — three gated lanes, executed per the
pre-committed `docs/RUNBOOK_W108.md` (locked BEFORE any expensive NIM call).

---

## Lane α — 405B reachability gate: CLOSED (5th consecutive 404)

Re-probed `meta/llama-3.1-405b-instruct` → **HTTP 404 (163 ms)** — the FIFTH
consecutive 404 (W104/W105/W106/W107/W108). GATE = CLOSED; β is the main lane.
`W104-L-…-405B-UNREACHABLE-ON-NIM-CAP` refreshed. No 405B pilot earned or
launched. Gate decision CID `e1af4451e4d15e7038016f7acd50d72560a77fc874470157c88d1426ee370e8d`.

---

## Lane β — LiveCodeBench main lane: real-data bug-fixed, pilot earned, Phase-2 FAIL

**1. Partial-scaffold audit + bug diagnosis.** The W108 work already in the
tree (`livecodebench_executor_v2` + `livecodebench_reflexion_bench_v1`) had a
gold-path smoke that FAILed even on a gold zigzag solution (A0=A1=B=0.0). Root
cause, reproduced end-to-end: the real `release_v6` corpus stores `metadata`
as a JSON **string** (`'{"func_name":"zigzagTraversal"}'`), but the W107 loader
read `func_name` only when `metadata` was a dict — so on real rows `func_name`
was silently `""`, the executor returned `ENTRY_NOT_FOUND` (rc 3) on every arm,
and all arms scored 0. The executor-V2 newline-per-arg decoder was itself
correct on real data (proven: gold zigzag passes when `func_name` is supplied).

**2. Fix + tests.** `livecodebench_loader_v1._resolve_func_name` now parses
metadata as dict OR JSON-string (+ a `starter_code` fallback), discharging
`W107-L-LIVECODEBENCH-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP`. Locked by
`tests/test_w108_livecodebench_realdata_v1.py` (19 tests incl. the exact bug,
its failure mode as a regression guard, the executor-V2 machinery, the full
bench gold-path A0=A1=B=1.0, and the slice selector).

**3. Real-data preflight — EARNED.** `scripts/run_w108_livecodebench_preflight.py`
(NIM-free): P1 SHA pin verified; P2 executor-V2 self-test incl. REAL gold/wrong
zigzag; P3 loader real-data self-test (63 functional, ALL `func_name` resolved,
ALL 63 plain-arg, difficulty 17e/26m/20h, dates 2025-01-11…2025-04-05 — ALL
post the Llama-3.x 2024-01-01 cutoff → C7 decisive); P4 deterministic
outcome-blind 30-slice. OVERALL PASS → pilot EARNED. Verdict CID `61b9961c…`.

**4. Canary + cheap pilot.** A 2-problem canary validated the live path proves
the fix (A1=50 %, valid `class Solution` extraction, healthy reflexion
pipeline). The full pilot (1 seed × 30 × K=5 = 330 calls; ~77 min; 444 HTTP-429
retries survived) returned:

> **A0 = 43.33 % / A1@K=5 = 63.33 % / B = 60.00 % / B − A1 = −3.33 pp;
> 7/9 gates; MLB-1 = 53.33 % PASS, MLB-2 = 25 % FAIL → `FAIL`
> (NON-mechanism-driven).**

Full verdict + interpretation: `docs/RESULTS_W108_LIVECODEBENCH_PHASE2_70B_V1.md`.
Headline: the W89 mechanism does NOT beat same-budget self-consistency on the
contamination-resistant 2025 functional subset — the FIRST contamination-
resistant test of the mechanism, and it FAILed. This RAISES (does not
establish) a contamination-confound hypothesis for the W89/W105 retirements,
which are both on contamination-exposed HumanEval-family problems.

---

## Lane γ — APPS backup: real scaffolding, pivot-ready, NOT triggered

Built to *real* (explicit-import only): `coordpy.apps_loader_v1` (call-based
`fn_name` functional subset; SHA-pinnable; refuse-on-mismatch),
`coordpy.apps_executor_v1` (subprocess, no-LLM-judge, native-arg-list decode +
output-wrapper tolerance), `scripts/run_w108_apps_preflight.py` (NIM-free
offline self-test PASS), `tests/test_w108_apps_backup_v1.py` (7 tests). The
exact LiveCodeBench-failure pivot conditions are codified (RUNBOOK § 6). Pivot
NOT triggered — LiveCodeBench passed its real-data structural-soundness test
(the FAIL is an empirical mechanism result, not a structural failure). APPS
stays backup; its 2021 vintage is contamination-exposed (C7 = C). The W108 FAIL
makes a W109 APPS contaminated-control contrast newly valuable (see below).

---

## Truth surface (unchanged where it matters)

* **Confirmed retirements: EXACTLY TWO**, both `meta/llama-3.3-70b-instruct` @
  70B — W89 (base HumanEval +5.56 pp) + W105 (HumanEval+ +7.00 pp; 6/6 bars;
  MLB-2 55.62 %). W108 adds NONE and retires NONE.
* **New boundary:** the retirements are NOT demonstrated on contamination-
  resistant data; the one attempt (LiveCodeBench 2025) cleanly FAILed.
* Not entitled (unchanged + W108): cross-class (Llama-3.1 FAIL_MARGIN),
  cross-scale-UP (405B 404 ×5), MBPP-family (W102), cross-modal (frozen @ 11B),
  contamination-resistant code superiority (W108 FAIL), "context solved".
* **Discipline: 18th consecutive preflight-discipline validation** (W93–W108) —
  W108's distinguishing contribution is catching a real-data binding bug BEFORE
  any NIM spend and earning the pilot honestly, then reporting a FAIL without
  spin.

---

## Stable boundary preserved

`coordpy.__version__ == 0.5.20`; `coordpy.SDK_VERSION == coordpy.sdk.v3.43`;
no PyPI publish; `coordpy/__init__.py` untouched. New work explicit-import only:
2 new APPS modules + the slice selector added to the W108 bench; the W107
loader edited in place (the documented confirm-at-fetch discharge). 49 tests
pass (19 LCB + 7 APPS + 16 W107 regression + the 7 pre-existing).

---

## W109 (made obvious)

**W109 = APPS contaminated-control contrast** (lead) OR multi-seed LiveCodeBench
de-noise (alternative), per RUNBOOK_W108 § 8 FAIL branch. The APPS scaffolding
is already real and pivot-ready. The scientific question W108 surfaced: does the
W89 reflexion advantage hold on a contamination-EXPOSED benchmark (APPS 2021)
while failing on a contamination-RESISTANT one (LiveCodeBench 2025)? Either
outcome sharpens the honest boundary. `COO-9` stays lead unless a W109 control
forces a different code-line move.
