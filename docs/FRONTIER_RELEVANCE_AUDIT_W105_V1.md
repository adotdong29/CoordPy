# Frontier-relevance audit — W105 supplement V1

> **2026-05-27.  Supplement to the W104 V1 audit
> (`docs/FRONTIER_RELEVANCE_AUDIT_W104_V1.md`).  All W104
> classifications remain in force.  This audit adds W105's
> additions and re-confirms the W93 → W105 frontier discipline
> as the 15th consecutive preflight-discipline validation.**

## Summary

This is the W105 frontier-relevance audit, the 15th consecutive
W6X / W7X / W8X / W9X / W10X preflight-first + cross-scale +
multi-candidate-tournament-then-confirm + mechanism-load-
bearingness + silent-degradation-anti-pattern-guard +
arsenal-mining-prior-anti-pattern-guard + cross-class-row-
alignment-discipline-guard validation.  The active frontier
arsenal is extended; the dead-direction column is unchanged;
the anti-pattern column is unchanged; the baseline-only
falsifier column is unchanged.

## Active frontier arsenal (NEW W105 additions)

* **`coordpy.phase3_retirement_evaluator_v1`** — pure-Python,
  no-NIM, explicit-import-only Phase 3 retirement evaluator
  that applies the W88 / W89 / W95 6-bar retirement shape to a
  set of per-(model class, seed) Phase 3 cells.  Emits per-class
  verdict labels (`RETIRED`, `RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING`,
  `FAIL_<reason>`) FIRST, then layers the cross-class
  entitlement on top.  Refuses to run on slice pack CID
  mismatch, corpus SHA mismatch, duplicate seed, or
  unrecognised schema.

* **`coordpy.cross_class_comparator_v1`** — per-seed-aligned
  cross-class comparator that fixes the W104 V1 cross-scale
  comparator row-alignment failure mode.  W104 V1 iterated
  per-(problem-position) assuming both scales used the same
  bench-internal shuffle, but W103 (seed 103 001) and W104
  (seed 104 001) shuffles differed and the per-problem cluster-
  shift labels were arithmetically mis-aligned (aggregate stats
  stayed correct).  V1 (W105) iterates per-(matched seed) so
  each comparison pair has identical bench-internal shuffle on
  both sides.  Refuses to run on slice pack CID / corpus SHA /
  schema / iteration-task-id mismatch.

* **`scripts/run_w105_phase3_retirement_bench.py`** — Phase 3
  driver with per-(model, seed) cell isolation, canary smoke,
  resume-safe per-cell skipping, mid-run global+per-cell
  progress JSON, automatic per-cell partial audit emission,
  and explicit 429 / socket-hang / relaunch handling (retry
  log per cell).  Refuses to run on pack CID / corpus SHA
  mismatch.

* **`scripts/run_w105_canary_smoke.py`** — thin wrapper that
  runs the 66-call canary BEFORE the 6 600-call Phase 3
  envelope opens.  Canary acceptance is reachability +
  budget-envelope sanity, NOT a Phase 3 gate.

* **`scripts/run_w105_405b_reachability_probe.py`** — cheap
  sub-second NIM probe on `meta/llama-3.1-405b-instruct`.
  Records the result in an audit-trail JSON.  Independent of
  the main W105 run; either outcome does NOT change the W105
  core matrix unless the RUNBOOK is explicitly re-locked.

* **`docs/RUNBOOK_W105.md`** — pre-commit contract locked
  BEFORE any W105 NIM call.  7 explicit lock fields per the
  RUNBOOK § "Pre-locked runbook contract" (pack CID; model
  matrix; seed list; retirement bars; cross-class claim rule;
  canary rule; W106 branch logic).

* **`docs/RESULTS_W105_W106_PLANNING_V1.md`** — pre-commits
  W106 under all five verdict shapes the W105 Phase 3
  retirement verdict can take.  Branch dispatch JSON encoded
  for machine consumption.

* **`tests/test_w105_phase3_discipline_v1.py`** — 18 PASSing
  unit tests codifying (a) pack-CID-mismatch refuse-to-run,
  (b) corpus-SHA-mismatch refuse-to-run, (c) per-seed shuffle
  reproducibility, (d) resume-safe per-cell skipping, (e)
  evaluator PASS path on synthetic 6 cells, (f) evaluator
  per-bar FAIL paths (margin / A1 saturation / duplicate seed
  / slice pack mismatch), (g) cross-class entitlement only on
  BOTH-PASS, (h) cross-class comparator per-seed alignment,
  (i) cross-class comparator refuse-to-run on iteration
  mismatch / slice pack mismatch / seed-set mismatch, (j)
  Phase 3 verdict end-to-end on synthetic split data.

## Useful baseline-only

* `coordpy.bounded_window_baseline_v{1,2,3}` — unchanged.  Still
  falsifier-only targets.
* `coordpy.local_openai_compatible_facade_v1` — unchanged.
* `coordpy.controlled_runtime_substrate_v1` — unchanged.
* `coordpy.learned_consolidation_v1` — unchanged.
* `coordpy.long_horizon_reconstruction_substrate_v1+v2` —
  unchanged.

## Historical artifacts

W92 / W90 / W88 / W81 / W83 / W84 / W89 verdicts unchanged.

W101 / W102 (V1 + V2 lead-lane verdicts) unchanged.

W103 (HumanEval+ cheap pilot at 70B Llama-3.3 PASS) unchanged.

W104 (HumanEval+ cross-generation cheap pilot at 70B Llama-3.1
PASS_MECHANISM_DRIVEN; pre-locked 405B unreachable on NIM)
unchanged.

W104's V1 cross-scale comparator's row-alignment limitation is
recorded as a CLOSED issue in the W105 hardening lane —
W105 cross-class comparator V1 fixes it via per-seed iteration
alignment.

## Dead directions

* Cross-modal RealWorldQA arc — frozen at 11B per W100; W105
  carries it verbatim; does NOT reopen.
* MBPP+ V2 at 70B — W102 cap unchanged; W105 does NOT reopen.
* SWE-bench-lite — unconditionally out of scope; the W89
  reflexion mechanism does not have the structural shape to
  attack repo-level failure surfaces.

## Anti-patterns (REMAIN explicit anti-patterns)

* Bounded windowing.
* Context compaction.
* Generic prose summarisation.
* Shallow token compression.
* Context-pruning theater.
* "Cram less / truncate better" tricks.
* Re-grading historical responses against a new test surface
  and treating the result as cheap-pilot earning evidence
  (the W102 anti-pattern carry-forward; W105 carries it
  verbatim).
* Silent-degeneration via schema assumption (W102 anti-pattern
  carry-forward; the W101 V1 MBPP+ loader is permanently
  demoted to historical artifact + anti-pattern).
* Cross-scale / cross-class comparator iterating per-(problem-
  position) without checking bench-internal shuffle alignment
  (NEW W105 anti-pattern carry-forward; the W104 V1 comparator
  is the example; W105 V1 cross-class comparator codifies the
  defence by refusing to run on per-seed iteration mismatch).

## What the audit DOES claim

* W105 ships infrastructure ONLY (RUNBOOK + driver + evaluator
  + comparator + tests + canary + 405B probe + W106 planning)
  BEFORE any W105 Phase 3 NIM call.
* All 18 W105 unit tests PASS BEFORE the canary launches.
* All 79 tests across the W101 + W102 + W103 + W104 + W105 code
  line PASS.
* The W105 driver refuses to run on pack CID mismatch + corpus
  SHA mismatch + slice / iteration-task-id alignment violations.
* The W105 evaluator refuses to run on duplicate seed + slice
  pack mismatch + corpus SHA mismatch + unrecognised schema.
* The W105 cross-class comparator refuses to run on iteration
  alignment violation (the W104 V1 row-alignment lesson made
  load-bearing).

## What the audit does NOT claim

* That W105 retires the Phase 3 retirement bars — that's
  empirical evidence the Phase 3 run produces.
* That W89 generalises beyond TWO Llama-3.x-70B-Instruct model
  classes — even retirement at Phase 3 would be bounded to
  these two classes at 70B parameter scale.
* That the W104 405B unreachability has changed — the W105
  reachability probe is recorded independent of the main run.
* That `multi-agent context is solved` — none of W89 / W103 /
  W104 / W105 imply this.

## Discipline status

* W93 + W94 + W95 + W96-A + W96-C + W96-D + W97 + W98 + W99 +
  W100 + W101 + W102 + W103 + W104 + **W105** = 15 consecutive
  preflight-discipline validations.

## Anchors

* `docs/FRONTIER_RELEVANCE_AUDIT_W104_V1.md` — W104 frontier
  audit (carried forward verbatim except for the W104 V1
  comparator row-alignment limitation, closed by W105).
* `docs/RUNBOOK_W105.md` — W105 pre-commit contract.
* `coordpy/phase3_retirement_evaluator_v1.py` — Phase 3
  retirement evaluator (NEW W105).
* `coordpy/cross_class_comparator_v1.py` — per-seed-aligned
  cross-class comparator (NEW W105).
* `scripts/run_w105_phase3_retirement_bench.py` — Phase 3 driver.
* `scripts/run_w105_canary_smoke.py` — canary entrypoint.
* `scripts/run_w105_405b_reachability_probe.py` — 405B probe.
* `tests/test_w105_phase3_discipline_v1.py` — 18 hardening
  tests.
* `docs/RESULTS_W105_W106_PLANNING_V1.md` — W106 dispatch +
  claim scaffolding.
