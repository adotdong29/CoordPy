# Frontier-relevance audit — W114 (supplement)

Extends the W112 V1 audit (`docs/FRONTIER_RELEVANCE_AUDIT_W112_V1.md`); all prior
classifications remain in force. This supplement classifies the W114 artifacts so
the active frontier stays clean and the bounded ceiling is easy to defend.

## Active frontier arsenal (W114 additions)

* **`coordpy.stronger_model_cutoff_certification_v1`** — the per-model
  post-cutoff certification layer. The reusable, primary-source-grounded
  instrument that decides whether a reachable model is certifiably
  contamination-resistant on a given instrument (C1 KNOWN cutoff ∧ C2 ≥30
  functional resistant ∧ C3 reachable-stronger-comparable ∧ C4 not-already-
  settled). Carries the verified instrument-frontier record + official-source
  provenance + a W113↔W114 confidence-consistency guard. This is the durable
  asset W114 leaves behind: future milestones re-run it against a newer
  instrument / a newly-disclosed cutoff to decide spend.
* **`scripts/run_w114_stronger_model_certification_v1.py`** — the NIM-free
  driver; re-verifies the instrument histogram against the SHA-pinned corpus and
  emits `results/w114/certification/certification_verdict.json` (decision CID
  `258b6ed7…`).
* **The certification-supply lens** — "which models can even be cleanly tested,
  against which instrument" — is now a first-class frontier tool: it converts a
  vague "needs more data" into a precise, dated, per-model spend gate.

## Useful baseline-only / context

* The W113 instruments (`livecodebench_resistant_slice_v1`,
  `cross_scale_resistant_interpretation_v1`, `tier2_readiness_v1`) remain the
  resistant-slice + cross-scale-interp + tier-2-ranking machinery; W114 imports
  them unchanged (no duplication).
* The W112 reachability facts (Maverick reachable; 405B 404×6; tier-2 reachable)
  are carried as a FIXED PRIOR — reachability is not the binding gate in W114, so
  it was not re-probed.

## Dead / closed directions (reaffirmed, NOT reopened)

* No reopening MBPP+ V2 (W102), frozen cross-modal lines (RealWorldQA @ 11B),
  the closed Llama-3.1 rescue branch (W106), or APPS main-lane NIM (exposed
  control only).
* No 70B resistant reflexion de-noise (W109 rule).
* No second Maverick resistant reflexion rerun on the same instrument (redundant;
  no verdict-changing power).
* No 405B expensive run (404×6; gate closed).

## Anti-patterns (reaffirmed)

* bounded-context / compaction / prose-summary / "truncate better" REMAIN
  explicit anti-patterns, NOT the frontier path. W114's genuinely-different axis
  is a certification-supply analysis on real dated instruments — the opposite of
  a token-compression trick.
* "Count a close or confounded edge as a win" — explicitly rejected: W114 earns
  NO pilot because none is genuinely earned, rather than buying a redundant or
  uncertifiable run.

## Net

W114 adds one durable frontier asset (the certification layer) and one
load-bearing blocker (the resistant-instrument frontier lags the model frontier).
The active code-line frontier is now: **register the bounded ceiling (done) +
wait for / fetch a NEW certified post-cutoff instrument (W115)**. `COO-9` stays
lead. W89 + W105 remain the only two retirements.

Anchors: `docs/RESULTS_W114_STRONGER_MODEL_CERTIFICATION_V1.md`,
`docs/RESULTS_W114_MILESTONE_SUMMARY_V1.md`, `docs/RUNBOOK_W114.md`.
