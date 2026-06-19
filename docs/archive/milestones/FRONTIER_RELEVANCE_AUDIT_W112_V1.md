# Frontier-relevance audit — W112 (22nd consecutive preflight/earn-discipline validation)

## Discipline validation (W93–W112)

W112 is the **22nd consecutive** milestone to honour preflight/earn-first
discipline. Specifics this milestone:

* **Runbook locked BEFORE any NIM call** (`docs/RUNBOOK_W112.md`), including the
  sub-second stronger-model reachability sweep. The § 1α target-selection rule
  was locked BEFORE probing — so the catalogue could not be target-shopped
  post-hoc (the rule selected Maverick tier-1 even though a stronger-absolute
  code model, Qwen3-Coder-480B, was also reachable).
* **The one expensive run was earned** under § 1α-earn (eligible C-S1∧C-S2∧C-S3∧
  C-S4 target + a 2-problem canary confirming the plain code path). $0 on 405B
  (404×6), the other tier-2 targets, a second pilot (W113 pre-committed), APPS,
  Llama-3.1, and 70B reflexion.
* **Lane β earn-rule locked before any β NIM** — and Lane β spent $0 (the fair
  strengthening space is structurally sub-floor).

## New this milestone — a methodology finding that tightens the whole programme

`W112-T-CONTAMINATION-RESISTANCE-IS-MODEL-CUTOFF-RELATIVE`: a benchmark is
"resistant" only relative to a model whose cutoff pre-dates its release. This
RECLASSIFIES how cross-model resistant comparisons must be read and is now a
required check before any future stronger-model resistant probe — it caught a
confound that a naive reading ("Maverick reopened resistant superiority") would
have over-claimed.

## Reclassifications

* **Active frontier (newly added):** the stronger-model resistant-code lane
  (now LIVE — a stronger model is reachable); `W112-T-STRONGER-CODE-MODEL-
  REACHABLE-LLAMA4-MAVERICK`; the W112 reachability sweep + fair-reachability
  mining scripts; the model-cutoff-relativity check.
* **Confounded-positive (newly classified):** the W112 Maverick +10 pp — real,
  audited, 9/9 gates, but EXPOSED-column (for-Llama-4) + PASS_NON_MECHANISM_DRIVEN
  + single-seed ⇒ control-grade evidence for the confound, NOT a resistant win.
* **Dead-at-$0 (newly classified):** the fair M3-strengthening design space at
  the current scale/regime (S-A..S-D structurally sub-floor;
  `W112-T-FAIR-M3-STRENGTHENING-CEILING-SUB-FLOOR`). No further fair-M3 NIM.
* **Standing-extension (unchanged):** 405B (404×6); re-opens only if reachable.
* **Anti-pattern column (unchanged + W112 addition):** do NOT read a stronger
  model's margin on an older "resistant" benchmark as a resistant win without
  re-checking the benchmark's release date vs the model's cutoff.

## Carry-forwards

* **Retired:** NONE. W89 + W105 remain the only two confirmed retirements.
* **Added:** `W112-T-405B-GATE-SIXTH-404-CLOSED`,
  `W112-T-STRONGER-CODE-MODEL-REACHABLE-LLAMA4-MAVERICK`,
  `W112-T-STRONGER-MODEL-BIGCODEBENCH-MARGIN-REOPENS-BUT-MODEL-EXPOSED`,
  `W112-T-CONTAMINATION-RESISTANCE-IS-MODEL-CUTOFF-RELATIVE`,
  `W112-T-FAIR-M3-STRENGTHENING-CEILING-SUB-FLOOR`,
  `W112-L-STRONGER-MODEL-RESISTANT-SUPERIORITY-NOT-CLEANLY-DEMONSTRATED-CAP`.

## Do not claim (W112)

We solved multi-agent context; a stronger model reopened contamination-RESISTANT
superiority (it did not — exposed-for-Llama-4, non-mechanism-driven, single-seed);
the +10 pp is a clean mechanism win; the contamination-confound is proven; a fair
M3 strengthening is worth spend; W112 added a retirement (it did not).

## Stable boundary

`coordpy.__version__ == 0.5.20`; `coordpy.SDK_VERSION == coordpy.sdk.v3.43`; no
PyPI; `coordpy/__init__.py` untouched; ZERO new `coordpy.*` modules (2 new
scripts only; the pilot reused the W110 driver verbatim).
