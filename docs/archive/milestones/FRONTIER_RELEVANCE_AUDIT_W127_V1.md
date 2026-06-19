# Frontier-relevance audit — W127 (capability atlas + family-specific scaffold generation)

Classifies the W127 additions and re-affirms the standing arsenal/baseline/dead/anti-pattern
columns. Supplements the W126/W125/W123/W121 audits; all prior classifications remain in force.

## Active-frontier arsenal (NEW in W127)

* **`coordpy.resistant_capability_atlas_v1`** — the machine-checkable capability atlas: a
  hard re-executable failure-decomposition layer (typed per-generation failure categories,
  visibility, diversity) + a transparent lexicon family classifier over PUBLIC inputs + an
  analyst-only reference cross-check that quantifies the classifier's theme-bias
  (`atlas_label_agreement`). **Active frontier**: the first artifact that says WHICH
  capabilities are missing on the resistant field (not just "capability failure"), and the
  diagnosis that drives the R2 cluster-match gate. Its honest headline — the 22 are
  wrong-algorithm-dominated (95%) and **algorithmically DIVERSE** (top-2 ≈ 45–50%, no single
  dominant cluster; surface labels only 47% concordant with the actual algorithm) — is itself
  a durable finding about why a single family scaffold faces a diverse resistant target set.
* **`coordpy.family_scaffold_generation_v1`** — the family-specific algorithm-scaffold
  generation line: G1 scaffold library (AST-de-identified structural skeletons from EXPOSED
  accepted solutions, keyed by family), G2 family-level retriever (scaffold-compatible group
  hedging the classifier's theme-bias AT THE RETRIEVAL LAYER), G3 scaffolded fresh-generation
  controller, G4 deterministic scaffold policy. The FIRST module to wire the EXPOSED teacher
  corpus + the family taxonomy onto a FRESH-generation prompt (graphify: community 174,
  bridging the W126 leakage guard + the ICPC bench + the atlas). **Active frontier**: this is
  the fresh-trajectory mechanism the milestone validated on the EXPOSED dev bench (EARNED,
  net +2 across 2 families, leakage-clean after recalibration) and probed on the resistant
  field.
* **`reproduces_accepted_block_v1` (boilerplate-robust accepted-leak tripwire)** — a reusable
  no-leakage refinement: the "accepted solution shown to the model" leak is a CONTIGUOUS
  reproduced block, not a single shared universal idiom. Corrects the per-line false positive
  (the W127 dev bench measured it directly: winning candidates were different correct
  derivations sharing only `n, k = map(int, input().split())`). Positive control preserved (a
  planted accepted solution is still caught). Active frontier (the correct discipline for any
  future bench that grades model code against a known reference on memorizable problems).
* **The R1∧R2 earn gate + targeted-probe driver** (`run_w127_targeted_resistant_probe_v1`) —
  an executable spend gate: fresh resistant NIM is earned ONLY by a real EXPOSED-dev margin
  (R1) AND an atlas-identified scaffoldable resistant cluster the dev line targets (R2). Active
  frontier (the cheapest honest "is a resistant probe worth buying?" gate; the
  fresh-generation sibling of the W125/W126 $0 precursors).

## Promoted (now WIRED to the scaffold-generation path)

The EXPOSED-side accepted solutions (W121 family) are promoted from a motif/idiom prior (W126
S3) to a full **algorithm-scaffold library** (de-identified structural skeletons). The LOCKED
family taxonomy is shared across the atlas, the retriever, and the earn metric.

## Baseline-only (unchanged)

The W120/W121 reflexion A0/A1/B arms remain the same-budget baselines (the dev bench's
baseline arm == A1). The W126 grade cache + the 330 already-paid generations are the atlas
substrate. `bounded_window_baseline_v{1,2,3}` remain falsifier targets.

## Dead directions / capped (W127)

* **The boilerplate per-line accepted-leak check** — superseded by the contiguous-block
  tripwire (it false-positives on universal idioms; do not reinstate the per-line form).
* **`W127-L-RESISTANT-SCAFFOLD-FRESH-GEN-CAP`** — the dev-validated family-scaffold line, run
  FRESH (R1∧R2-earned) on the 6 string_processing scaffoldable resistant problems, created
  **0/6** new secret-passing solves over the old pool ⇒ family scaffolds do not close the
  resistant capability gap at Maverick's scale; the exposed-bench +2 signal did NOT transfer to
  the non-memorizable resistant field. Fresh scaffolded GENERATION joins the dead-at-cheap
  resistant levers (re-routing W125, deterministic synthesis W126).

## Anti-patterns (REMAIN explicit anti-patterns; W127 reinforces)

Bounded-context / compaction / generic summarization / "cram less, truncate better" remain
anti-patterns, NOT the frontier path. W127 explicitly built a real diagnosis + a real
fresh-generation mechanism, validated it on a disjoint same-family bench, and bought the
smallest honest resistant probe — the OPPOSITE of a truncation trick.

## Do-not-claim (see `docs/HOW_NOT_TO_OVERSTATE.md` W127 section)

The atlas's family labels are a transparent, theme-biased heuristic (47% reference-concordant),
NOT ground truth; the robust atlas findings are the hard wrong-algorithm decomposition + the
DIVERSITY of the failures. The EXPOSED dev-bench earn is real but WEAK (+2 net minimum, K=5,
n=8) and confounded (exposed-problem memorization, longer-prompt framing, sampling variance) —
it earns a clean resistant probe, NOT a validated-mechanism claim. The leakage recalibration is
a documented false-positive correction (boilerplate, not a real leak; positive control
preserved), applied transparently as outcome-relevant. W89 (+5.56) + W105 (+7.00) stand as the
only two retirements unless the probe + a broader pilot clear the bar. Multi-agent context is
not "solved".
