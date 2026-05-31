# Frontier-relevance audit — W119 (supplement to W118 V1)

Extends the W118 V1 audit; all prior classifications remain in force. Classifies the new
W119 assets as **active frontier arsenal / baseline-only / historical / dead / anti-pattern**.

## Active frontier arsenal (W119 additions)

* **`coordpy.coordpy_icpc_public_functional_v1`** — the official-ICPC-package post-cutoff
  functional constructor + admission + certification pipeline. The current frontier
  instrument for the resistant-code lead path: it dissolved the W118 grader blocker by
  pivoting to a source family that ships the executable grader, and it is push-button for
  W120 (re-run on the next official ICPC regional drop).
  * `IcpcPublicInstrumentRuleV1` (P1..P8) — the locked official-package admissibility rule.
  * `OFFICIAL_ICPC_PACKAGE_FAMILY` + `icpc_family_grader_summary_v1` — the official source-
    family grader registry (the W119 supply surface).
  * `build_icpc_manifest_v1` + `ICPC_PACKAGE_LISTING_SNAPSHOT_V1` + `fetch_icpc_package_
    listing_v1` — deterministic manifest constructor + thin live GitHub-API fetch.
  * `run_icpc_stdin_executor_v1` — a real fresh-subprocess stdin/stdout code executor (the
    grader-execution path; reusable for the W120 pilot).
  * `W119_GRADER_SELFTEST_V1` + `grader_selftest_summary_v1` — the P8 self-test evidence.
  * `certify_models_on_icpc_manifest_v1` + `W120FireConditionV1` +
    `run_icpc_public_construction_v1` — reused C1..C4 + grader + slice certification + the
    pre-committed W120 trigger.
* **`scripts/run_w119_icpc_public_construction_v1.py`** — the push-button driver
  (writes `results/w119/icpc_public/`).
* **The grader self-test methodology** — running OFFICIAL accepted reference solutions
  against OFFICIAL secret cases in a fresh subprocess is the cheap, NIM-free way to certify
  a grader is real + executable before any model spend. Reusable for any future package
  family.

## Reused-not-duplicated (the durable chain)

`run_icpc_public_construction_v1` reuses, via explicit import and with NO duplication: the
W113 model-cutoff registry + `normalize_contest_date_v1` + `MIN_RESISTANT_SLICE`; the W114
`certify_model_v1` / `LatestResistantInstrumentV1` / `STRONGER_MODEL_CANDIDATES`; the W117
`run_upstream_construction_v1` (so the LCB-inherited decision CID re-derives byte-identical
`258b6ed7`); the W116 disclosure types; the W118 disclosure matrix. graphify confirms the
reuse edges.

## Baseline-only / falsifier targets (unchanged)

`bounded_window_baseline_v{1,2,3}` remain falsifier targets, not frontier. The W119
falsifiability tests (synthetic ≥30 grader-clean slice DOES certify + fire W120; ≥30
WITHOUT a self-test-passing grader does NOT) are the analogous falsifier guards for the
ICPC pipeline.

## Dead directions (W119 confirmations)

* **Extracting a grader from the LCB source family** (Codeforces API / LeetCode / AtCoder)
  — dead per W118; W119 confirms by SUCCEEDING elsewhere (the official ICPC family ships
  what the LCB family cannot).
* **The NWERC-2024 official static-package zip as a second surface** — 404 as of
  2026-05-30; not a usable aggregation path (recorded, not an anti-pattern — it may return).

## Anti-patterns (re-affirmed)

* **Sample-only / operator-synthesised graders** — refused (P5/P8). Only OFFICIAL shipped
  `data/secret` + shipped output validators + official accepted solutions count.
* **Sub-threshold pilots** — a 24-task pilot below the 30-slice bar is refused; running it
  would produce an under-powered, uninterpretable B−A1.
* **Bounded-context / compaction / token-compression / "truncate better"** — remain
  explicit anti-patterns, NOT the frontier path. W119 is a construction + certification +
  real-executor milestone, the opposite of a truncation trick.
* **Selling a grader-clean-but-count-short battlefield as a win** — W119 does not; the
  slice-count cap is the registered truth floor.

## Frontier status after W119

The resistant-code lead path is now **grader-UNblocked** (a genuinely-new official-ICPC
battlefield with a real executable grader) and blocked ONLY on a count: +6 post-cutoff
resistant pass-fail tasks to reach 30. The next official ICPC regional package drop (each
RMRC adds ~12/yr) or one clean official second-surface aggregation makes the Maverick pilot
push-button. `COO-9` stays the lead path.
