# Frontier Relevance Audit — W108 (anti-drift contract)

18th consecutive preflight-discipline validation (W93–W108). W108 executed the
pre-committed `docs/RUNBOOK_W108.md` three-lane branch logic; the LiveCodeBench
cheap pilot was EARNED on real data and returned a clean Phase-2 FAIL.

## Discipline status (#18 consecutive)

W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 / W100 / W101 / W102
/ W103 / W104 / W105 / W106 / W107 / **W108**. W108's distinguishing additions:

1. **Real-data binding bug caught + fixed BEFORE any NIM spend** — the partial
   W108 scaffold's gold-path smoke (A0=A1=B=0.0) was diagnosed to a
   metadata-as-JSON-string loader bug, fixed, and locked in 19 tests; the
   real-data preflight then PASSed. The W102 silent-degeneration guard worked
   as designed.
2. **First contamination-resistant attack on the W89 mechanism** — and it
   FAILed, reported without spin.

## Active-frontier reclassifications (post-W108)

* **Active frontier (newly added):** `coordpy.livecodebench_loader_v1` (fixed),
  `livecodebench_executor_v2`, `livecodebench_reflexion_bench_v1` (+ slice
  selector), `apps_loader_v1`, `apps_executor_v1`, the W108 preflight + pilot +
  APPS preflight drivers, and `tests/test_w108_*`.
* **Baseline-only / capped (newly classified):** LiveCodeBench functional
  subset at 70B with K=5 same-budget sequential reflexion — empirically capped
  (`W108-L-LIVECODEBENCH-REFLEXION-PHASE2-70B-CAP`). Stays in-repo as a
  re-runnable battlefield (multi-seed de-noise or cross-model) and as the
  contamination-resistant control point.
* **Dead direction (unchanged):** base-MBPP K=5 retirement; cross-modal
  RealWorldQA at the +5 pp bar at any scale; 405B cross-scale-UP while 404.
* **Backup (real, pivot-ready):** APPS call-based scaffolding — contamination-
  exposed (C7=C); BACKUP evidence only.

## Anti-pattern column (carried forward verbatim + W108 additions)

UNCHANGED from W107, plus:

* **W108 anti-pattern (new):** treating a contamination-EXPOSED benchmark PASS
  as evidence of contamination-RESISTANT generalisation. The W89/W105
  retirements are on HumanEval-family (2021 problems, in-training); the first
  contamination-resistant test (LiveCodeBench 2025) FAILed. Do NOT present the
  retirements as if they were shown to be contamination-robust.
* **W108 anti-pattern (new):** over-reading a single-seed cheap-pilot margin.
  The −3.33 pp is a 1-problem net effect and MLB-2's 25 % is 4/16 — the FAIL is
  clean and load-bearing-floor-missing, but the contamination-confound is a
  HYPOTHESIS, not a finding, until a controlled follow-up runs.

## Carry-forwards

* **Added (T):** `W108-T-LIVECODEBENCH-REAL-DATA-BUGFIX-METADATA-JSON-STRING`,
  `W108-T-LIVECODEBENCH-REAL-DATA-PREFLIGHT-EARNED`,
  `W108-T-405B-GATE-FIFTH-404-CLOSED`,
  `W108-T-APPS-BACKUP-SCAFFOLDING-REAL-PIVOT-READY`.
* **Added (L):** `W108-L-LIVECODEBENCH-REFLEXION-PHASE2-70B-CAP`,
  `W108-L-REFLEXION-NOT-DEMONSTRATED-ON-CONTAMINATION-RESISTANT-BENCH-CAP`.
* **Discharged (infrastructure caps, confirmed on real data — NOT research
  retirements):** `W107-L-LIVECODEBENCH-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP`,
  `W107-L-LIVECODEBENCH-RESIDUAL-PUBLISHED-BASELINE-GRADE-CAP` (live A1 measured).
* **Retired (research retirements):** NONE. W89 + W105 remain the only two
  confirmed multi-seed same-budget multi-agent superiority retirements.

## Do not claim (W108 additions)

* That the W89 reflexion mechanism beats same-budget self-consistency on
  contamination-resistant code (W108: B − A1 = −3.33 pp; MLB-2 = 25 % FAIL).
* That LiveCodeBench produced a retirement or strengthened the claim (it
  produced a clean FAIL; the claim is now MORE bounded, not stronger).
* That the contamination-confound is established (it is a hypothesis from one
  single-seed cheap pilot).
* That APPS (if later run) would be publication-grade evidence (2021 vintage =
  contamination-exposed; backup/control only).
* That multi-agent context is "solved".

## Anchors

`docs/RUNBOOK_W108.md`; `docs/RESULTS_W108_LIVECODEBENCH_PHASE2_70B_V1.md`;
`docs/RESULTS_W108_MILESTONE_SUMMARY_V1.md`;
`results/w108/livecodebench_preflight/preflight_verdict.json`;
`results/w108/livecodebench_pilot/…/livecodebench_reflexion_bench_report.json`;
`results/w108/405b_reachability_probe/gate_decision.json`;
`results/w108/apps_preflight/preflight_verdict.json`.
