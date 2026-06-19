# Frontier-relevance audit — W129 (public-signal selection oracle)

Supplement to the W127/W128 audits; all prior classifications remain in force. W129 classifies
the public-signal selection-oracle line against the bounded-context anti-pattern.

## Active frontier arsenal (NEW in W129)

* `coordpy.public_signal_selection_oracle_v1` — the SO1 public-derived falsifier stack / SO2
  differential-disagreement (REAL bridge to `integrated_synthesis`) / SO3 verifier-final (mines
  the `mathvista_bench_v2` verifier-final pattern) / SO4 trust-weighted-abstain (realizes the
  `integrity_trust_coupled_consensus_v1` integrity-penalty + abstain CONCEPT natively over honest
  code-correctness trust). The FIRST module bridging the verifier-final + trust-coupled-consensus
  machinery onto the ICPC resistant-code SELECTION path.
* The $0 stored-pool RECON harness (`run_w129_stored_pool_recon_v1`) — reconstructs prior pools by
  replaying stored generations (keyed by `prompt_sha256`) and re-grades every candidate on
  PUBLIC+SECRET. A reusable, $0, falsifiable way to localize a SELECTION cap to the candidate
  level (it proved the pawnshop miss is a COMPLEXITY signal, not an information limit).
* The fake-selection positive control + `examine_trust_machinery_applicability_v1` (the W128 W79
  honest-mining sibling: kills the latent-substrate `TrustWeightedConsensusController` literal
  bridge as fake-different, machine-checkable).

## Useful baseline / diagnostic-only

* The generic STRESS scale-up (`derive_stress_cases_v1`) — a real differential TLE/crash
  falsifier, but NOT load-bearing on this family: a format-preserving scale-up does not construct
  the adversarial worst-case that exposes an O(N²) candidate (`W129-L-NIMFREE-GENERIC-STRESS-...`).
  Kept as a safe component (differential protection), disabled via `W129_STRESS_OFF` for eval speed.
* `bounded_window_baseline_v{1,2,3}` — unchanged falsifier targets.

## Dead directions / confirmed caps

* NIM-free GENERIC public-signal selection cannot convert the W128 lifted pool ceiling into a net
  committed gain on the hard clusters — and even a PERFECT selector (model-verifier included) is
  bounded by committed ≤ the GENERATION pool ceiling (3/11 = baseline+1) < the +2 earn bar
  (`W129-L-HARD-CLUSTER-GENERATION-CEILING-CAPS-SELECTION-EARN`). The binding cap is GENERATION.
* The SO3 verifier-final BREAKS the specific pawnshop complexity tie but is INCONSISTENT
  (over-abstains both-correct ties) and net-committed-unchanged vs W128 — a real-but-not-validated
  selector. The honest next lever is a stronger GENERATOR, not a better selector.

## Anti-patterns (unchanged, reinforced)

* Bounded-context / compaction / prose-summary / "cram less / truncate better" REMAIN explicit
  anti-patterns. W129 evidence reinforces this: the selection oracle is an audited
  generate→verify→select→abstain mechanism on real code with a hidden-test-free public-signal
  oracle and a machine-checked realness surface — the OPPOSITE of a truncation trick.
* Hand-tuning a bespoke adversarial worst-case to crack the single `pawnshop` fixture would be
  OVERFITTING — explicitly NOT done (we report the generic stress lever's honest limitation).

## Do not claim

* That W129 earned a third retirement, beat W128 on net, or validated the selector (it did not:
  committed ≤ +1 generation ceiling; the locked β0 gate failed on over-abstention).
* That a NIM-free oracle cashes out pawnshop at $0 (only the model verifier breaks it).
* That "selection is solved" (the specific complexity tie is verifier-breakable; the residual cap
  is generation, and the verifier over-abstains both-correct ties).
* That W129 weakens W89/W105, proves the contamination confound, or solves multi-agent context.

Anchors: `docs/RESULTS_W129_PUBLIC_SIGNAL_SELECTION_ORACLE_V1.md`; `docs/RUNBOOK_W129.md`;
`docs/HOW_NOT_TO_OVERSTATE.md`; `results/w129/`.
