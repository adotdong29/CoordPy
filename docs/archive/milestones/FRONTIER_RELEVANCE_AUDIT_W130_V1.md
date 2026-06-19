# Frontier-relevance audit — W130 (generation-ceiling attack on the hard ICPC clusters)

Supplement to the W127/W128/W129 audits; all prior classifications remain in force. W130
classifies the generator-failure atlas + the stronger same-budget generator slate against the
bounded-context anti-pattern.

## Active frontier arsenal (NEW in W130)

* `coordpy.generator_failure_atlas_v1` — the $0 generator-failure DIAGNOSIS: reconstructs the full
  old W128/W129 pool, grades every candidate with a mechanical failure signature (the official
  execution path), and classifies each problem's dominant GENERATOR-failure mode under a LOCKED
  taxonomy (`SOLVED` / `SELECTION_FIXABLE` / `HIDDEN_EDGE_STATE_MISS` / `COMPLEXITY_BLIND` /
  `WRONG_ALGORITHM_ADMISSIBLE` / `WRONG_ALGORITHM_NO_SKETCH` / `PARSE_IO_FAILURE`). The first tool
  to separate GENERATOR-fixable from CAPABILITY pool-dead failures + SELECTOR-fixable (W129) ones.
* `coordpy.stronger_generator_slate_v1` — the GG1 complexity-gated handoff / GG2
  counterexample-to-rewrite / GG3 family anti-pattern coach / GG4 budget router / GGLEAD slate,
  each at MATCHED K=5 budget with the W129 selector held FIXED downstream. The first module to
  attack the GENERATION layer directly (W128 attacked role-diverse generation+selection; W129
  attacked selection). GG2's in-loop digest-driven REWRITE is the load-bearing lever (it cracked
  the `doubleup` HIDDEN_EDGE pool-dead problem the old pool missed).
* The realness surface: `gg1_gate_control_v1` + `gg2_rewrite_control_v1` +
  `examine_hosted_controller_applicability_v1` (the W128/W129 W79 honest-mining sibling — kills the
  literal hosted-planner/handoff/substrate bridge as fake-different; the hosted cache planner is
  efficiency-only KV-prefix savings, not a capability lever).

## Useful baseline / diagnostic-only

* The atlas idiom-overlap admissibility heuristic — a TRANSPARENT, theme-biased signal (the W127
  47%-concordant lesson). It is an UPPER BOUND on generator-fixability and was empirically shown so
  in W130 (the 3 `WRONG_ALGORITHM_ADMISSIBLE` problems were cracked by no arm). Reported as a
  heuristic, never ground truth.
* GG1 complexity gate, GG3 anti-pattern coach, GG4 budget router — REAL mechanisms (controls pass,
  budget honest) but NOT load-bearing on this dev bench (0 new solves); GG3 additionally carries a
  family-boilerplate leakage RISK (caught by the guard). Kept as honest negatives.

## Dead directions / confirmed caps

* A stronger SAME-BUDGET generator does NOT lift the hard-cluster generation ceiling to the +2
  earn bar with the W129 selector fixed (`W130-L-GENERATION-CEILING-DEV-BENCH-CAP`): GG2 cracked
  exactly 1 HIDDEN_EDGE problem; the dominant `WRONG_ALGORITHM_ADMISSIBLE` mode is capability-bound
  (`W130-T-ADMISSIBLE-SKETCH-IS-CAPABILITY-NOT-GENERATION-FIXABLE`).
* The honest POSITIVE: the cap is PARTIALLY liftable — GG2's counterexample-rewrite is a real
  generation lever on HIDDEN_EDGE failures (`W130-T-COUNTEREXAMPLE-REWRITE-LIFTS-ONE-HIDDEN-EDGE`)
  — but one crack is below the bar, and the dominant failures need real model capability, not more
  same-budget generator engineering.

## Anti-patterns (unchanged, reinforced)

* Bounded-context / compaction / prose-summary / "cram less / truncate better" REMAIN explicit
  anti-patterns. W130 evidence reinforces this: the win, such as it is, came from an audited
  generate→verify→REWRITE→select mechanism on real code with a hidden-test-free public-signal
  digest — the OPPOSITE of a truncation trick.
* The atlas admissibility heuristic was NOT hand-tuned to the result (it over-counted
  fixability, and W130 honestly reported the over-count rather than trimming the heuristic to
  match).

## Do not claim

* That W130 earned a third retirement, cleared the +2 bar, or validated the generator slate (it
  did not: 1 new solve < 2, does not span ≥2 families/modes).
* That the generator "lifts the generation ceiling" generally (it cracked exactly ONE HIDDEN_EDGE
  problem; the dominant `WRONG_ALGORITHM_ADMISSIBLE` mode is capability-bound).
* That the atlas "admissible" label means a problem is generator-fixable (it is an idiom-overlap
  UPPER BOUND; the 3 admissible problems were cracked by no arm).
* That GG3's coach or GG4's router contributed (0 new solves; GG3 carries a leakage risk and is
  killed).
* That W130 weakens W89/W105, proves the contamination confound, or solves multi-agent context.

Anchors: `docs/RESULTS_W130_GENERATION_CEILING_ATTACK_V1.md`; `docs/RUNBOOK_W130.md`;
`docs/HOW_NOT_TO_OVERSTATE.md`; `results/w130/`.
