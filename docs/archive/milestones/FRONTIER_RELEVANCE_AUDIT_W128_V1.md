# Frontier-relevance audit — W128 (role-diverse algorithm search on the non-scaffoldable resistant clusters)

Classifies the W128 additions and re-affirms the standing arsenal/baseline/dead/anti-pattern
columns. Supplements the W127/W126/W125/W123/W121 audits; all prior classifications remain in
force.

## Active-frontier arsenal (NEW in W128)

* **`coordpy.role_diverse_algorithm_search_v1`** — the role-diverse algorithm SEARCH mechanism:
  a matched-K=5 pipeline (1 ANALYZE producing materially-different role artifacts — spec /
  invariants / complexity / N distinct algorithm sketches / derived counterexamples — + 4
  IMPLEMENT calls, one per sketch) followed by a NIM-free generate→verify→select→abstain over
  the implementations. **Active frontier**: the first CoordPy code mechanism that bridges the
  role-diverse SYNTHESIS stack (`integrated_synthesis` / `role_invariant_synthesis`, community
  35) to the ICPC resistant-code path (communities 174/329) — START 5–8 hops apart with no
  semantic edge. It is REAL (all 11 dev runs classify genuine REAL-diversity; the
  fake-diversity detector + the W41/W42 bridges + the abstain layer all work) and it **lifts the
  generation ceiling** on hard clusters (pool 3/11 > plain baseline 2/11). Its honest cap is at
  the SELECTION layer, not generation (see Dead directions).
* **The fake-diversity detector + positive control** (`DiversityReportV1.classify` /
  `fake_diversity_control_v1`) — a reusable NIM-free realness surface (the W125
  `MechanismFingerprintV1` analogue): a role-diverse run is REAL only if the sketches are
  materially distinct, ≥2 implementations differ after AST normalization, the counterexamples
  are new, and invariants exist; identical sketches MUST classify `FAKE_DIVERSE`. Active
  frontier (the correct discipline for any future role-diverse mechanism — it caught the W128
  sketch-parser bug by flagging 9/11 degenerate runs).
* **`examine_substrate_controller_applicability_v1`** — an honest-mining record: a
  machine-checkable applicability scan proving the W79 substrate-trust controllers are NOT a
  clean code-candidate consensus bridge (a literal call would be fake-different), so the
  consensus/abstain role is genuinely filled by the W41/W42 synthesis decisions. Active
  frontier (the discipline for mining an arsenal HONESTLY — "which candidate died and why").
* **The T1∧T2 earn gate + dev-bench/probe drivers** — an executable spend gate: fresh resistant
  NIM is earned ONLY by a real EXPOSED hard-cluster dev margin (T1) AND a named-hard-cluster
  match (T2). Active frontier (the selection-mechanism sibling of the W125/W126/W127 $0-or-cheap
  spend gates).

## Promoted (now WIRED to the resistant-code path)

The W41/W42 synthesis decision functions (`select_role_invariance_decision`,
`select_integrated_synthesis_decision`) are promoted from latent capsule-synthesis primitives
to the **load-bearing consensus/abstain layer of a code-candidate selector** — the first time
the W41/W42 abstain-on-divergence logic is applied to code generation.

## Baseline-only (unchanged)

The W120/W121 reflexion A0/A1/B arms remain the same-budget baselines (the dev bench's `plain`
arm == A1). The W127 scaffold line (G2→G3) is the reference arm. The W127 atlas + the W120
resistant 30-slice + the W121 EXPOSED corpus remain the substrate. `bounded_window_baseline_v{1,2,3}`
remain falsifier targets.

## Dead directions / capped (W128)

* **`W128-L-ROLE-DIVERSE-HARD-CLUSTER-DEV-BENCH-CAP`** — the role-diverse search line, though
  REAL, does NOT beat plain hosted generation on EXPOSED hard-cluster problems (RDA4 committed
  2/11 = baseline 2/11 = scaffold 2/11; net +0 < the +2 bar) ⇒ not validated ⇒ $0 resistant
  spend. The SELECTION-lever sibling of the W123→W127 cap taxonomy.
* **`W128-T-ROLE-DIVERSE-SEARCH-LIFTS-GENERATION-CEILING-BUT-SELECTION-CAPPED`** — the precise
  localization: enforced diversity lifts the generation ceiling (pool 3 > baseline 2; reaches a
  simulation_grid program i.i.d. sampling missed) but verification-based selection WITHOUT a
  hidden-test oracle cannot convert it (commits 2/3 pool wins, mis-commits the third) ⇒ the
  full mechanism nets +0. Selection/verification (NOT generation) is the bottleneck — the lever
  for any future role-diverse attempt is a better selection oracle (and W125 already showed the
  public-sample/derived signal is non-discriminating on this family) or a stronger model.
* **`W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP`** — the EXPOSED corpus has 0 graph_flow problems ⇒
  graph_flow is resistant-probe-only (cannot be exposed-dev-validated here).
* **The markdown-blind sketch parser** (`#{0,3}` section regex + `^\s*SKETCH` fallback) —
  superseded by the markdown-tolerant parser (`#{0,6}` + prefix-tolerant fallback); it
  false-negatives on `#### SKETCH A:` headers (it degenerated the first dev run). Do not
  reinstate.
* **The literal W79-substrate-controller consensus bridge** — killed as fake-inapplicable
  (substrate-trust-specific decision logic; see honest-mining record).

## Anti-patterns (REMAIN explicit anti-patterns; W128 reinforces)

Bounded-context / compaction / generic summarization / "cram less, truncate better" remain
anti-patterns, NOT the frontier path. W128 built a genuinely different mechanism (role-diverse
generate-verify-select-abstain), validated it on a disjoint same-family bench, localized its cap
to the selection layer, and bought ZERO resistant NIM because it did not earn — the OPPOSITE of
a truncation trick.

## Do-not-claim (see `docs/HOW_NOT_TO_OVERSTATE.md` W128 section)

The role-diverse mechanism is REAL but NOT validated — it does not beat plain generation on the
EXPOSED hard clusters (net +0), so it earns no resistant spend. The pool-ceiling lift (3 > 2) is
a genuine GENERATION signal but it is NOT a committed win and NOT a mechanism validation — the
selection layer caps it. The all-REAL-diversity result is post-parser-fix; the first run's 9/11
FAKE_DIVERSE was a measurement bug (markdown headers), corrected transparently, requiring a
clean re-run. No targeted resistant probe was run (T1 failed) ⇒ $0 resistant NIM; the resistant
field's role-diverse-search behaviour is UNTESTED (the cap is on the EXPOSED dev bench, not a
measured resistant 0/N). W89 (+5.56) + W105 (+7.00) stand as the only two retirements.
Multi-agent context is not "solved".
