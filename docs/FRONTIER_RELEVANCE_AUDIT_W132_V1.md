# Frontier-relevance audit — W132 (supplement to the W131 V1 audit)

Extends the W131 V1 audit; all prior classifications remain in force. W132 classifies the
minted-battlefield artifacts.

## Active frontier arsenal (load-bearing for the code line)

* **`coordpy.resistant_by_construction_battlefield_v1`** — the minted-problem framework:
  the three-program oracle harness (`ref_source` answer key + independent `brute_source`
  cross-check + `naive_source` trap), the six per-problem quality gates, the
  `novelty_filter_v1` near-duplicate / official-identity guard, the content-addressed
  manifest, `certify_resistance_v1` (date + construction + reused W114 gate), and
  `select_core_slice_v1`. **This is the W132 advance** — it removes the W123 official-
  supply cap and the W131 cutoff-disclosure dependency as blockers on the EXISTENCE of a
  resistant instrument. Reusable + push-button.
* **`coordpy.resistant_by_construction_slate_v1`** — the 33-template slate (9/8/8/8 across
  the four targeted failure families). A reusable corpus; re-mintable at fresh seeds for
  multi-seed confirmation.
* **`scripts/run_w132_calibration_and_pilot_v1.py`** — the earned frontier pilot
  (slice-CID-guarded; `--model` swappable). EXECUTED this session on
  `meta/llama-3.3-70b-instruct` (the W105 retirement model; Maverick infra-down) ⇒ B−A1 =
  +3.33 pp FAIL. Maverick is the push-button CROSS-SCALE re-run on the same slice when its
  deployment recovers.
* The validated W120 bench (`icpc_reflexion_bench_v1`) + the audited grader
  (`coordpy_icpc_battlefield_v1.grade_icpc_candidate_case_v1`) + the verbatim W108 evaluator
  — consumed UNCHANGED; the minted battlefield bridges onto them (no mechanism drift).

## Useful baseline-only

* **`scripts/run_w132_dev_only_local_characterization_v1.py`** — a DEV_ONLY local-Ollama
  instrument-characterization driver. Useful to prove the β pipeline executes and to
  characterize difficulty, but `qwen2.5-coder:7b` is far weaker than the frontier target
  and the run is throughput-limited; it is **NOT** a frontier claim and cannot retire
  anything (every output field tagged `frontier_claim=false`).
* **`naive_source` programs** — the admissible-wrong traps. Diagnostic only (they define
  what the field resists); never a solution.

## Historical artifacts (unchanged)

W120–W131 ICPC battlefield + atlas + generator/selector arsenal unchanged. The official-
ICPC inherited battlefields remain supply-capped (W123) — W132 supersedes them as the
preferred resistant instrument ONLY in that it is mint-able at scale and resistant for ANY
cutoff; the OFFICIAL battlefields retain their value as the contamination-EXPOSED/RESISTANT
matched controls (W120/W121).

## Dead directions (do NOT revive)

* Treating the official-package supply cap (W123) or the cutoff-disclosure cap (W131) as
  the END of the resistant line — W132 shows a resistant instrument can be MINTED.
* Selling a synthetic toy benchmark, an official-task paraphrase, a near-duplicate set, or
  a leakage-tainted template as a resistant-by-construction win — explicitly guarded against
  (the discriminating gate + the novelty guard + the no-leakage rule + the planted-control
  tests).

## Anti-patterns (REMAIN explicit anti-patterns)

Bounded-context / compaction / prose-summary / "cram less / truncate better" remain
anti-patterns, explicitly NOT pursued. W132 is an instrument-construction + conditional-
pilot milestone, not a context-compression milestone.

## Do not claim (W132)

* That W132 retired anything (it did not; W89 + W105 remain the only two).
* That **Maverick** was tested or that this is a Maverick/cross-scale result — Maverick's
  deployment was infra-down (0-bytes hang while `llama-3.1-8b`/`llama-3.3-70b` responded);
  the pilot ran on `llama-3.3-70b` (the W105 retirement model), substituted PRE-SPEND.
* That the executed pilot's +3.33 pp is a margin / a win / mechanism-driven — it is a clean
  FAIL (< +5 pp; MLB-1 26.7%, MLB-2 25%, both FAIL): a single isolated complexity rescue, 0
  regressions, the mechanism is NOT load-bearing on the minted resistant field.
* That the DEV_ONLY local-7B characterization is a frontier or retirement result (it is
  neither; it characterizes the instrument on a weak local model).
* That the minted problems are "novel algorithms" (they are textbook algorithm FAMILIES;
  what is novel + resistant is the freshly-minted INSTANCE set + statements + case data,
  guarded against official reuse).
* That resistance-by-construction proves the mechanism works on resistant code — it removes
  the *instrument-supply* and *cutoff-disclosure* blockers on the instrument's EXISTENCE; the
  executed pilot then shows the mechanism does NOT beat A1 on the minted field at the W105
  retirement model (+3.33 pp FAIL), STRENGTHENING the bounded ceiling (single seed).
