# Frontier-relevance audit — W109 (supplement to W108 V1)

> Extends `docs/FRONTIER_RELEVANCE_AUDIT_W108_V1.md`; all prior classifications
> remain in force. Classifies the W109 artifacts as active frontier arsenal /
> useful control-baseline / dead direction / anti-pattern, so future milestones
> reuse what is load-bearing and avoid re-running what is capped.

## Active frontier arsenal (reusable, load-bearing)

* **`coordpy.apps_reflexion_bench_v1`** — the APPS call-based contamination-
  control bench (A0/A1/B byte-identical in shape to W89/W105/W108; difficulty-
  stratified outcome-blind slice; `max_tests` cap). Reusable for any
  contamination-EXPOSED call-based control.
* **`coordpy.livecodebench_denoise_decision_v1`** — the falsifiable two-gate
  de-noise decision rule (marginal POSITIVE miss ∧ MLB-2 load-bearing). Reusable
  whenever a single-seed FAIL tempts a "spend more to feel better" re-run; it is
  the verdict-changing-power discipline (the W106 margin-cap analogue for FAILs).
* **`scripts/fetch_w109_apps_corpus.py`** — the reproducible parquet→pinned-JSONL
  materializer (HF `refs/convert/parquet` path for loading-script datasets the
  HF API refuses). Reusable for any HF dataset behind a loading script.
* **The contamination-control 2×2** (`docs/CONTAMINATION_CONTROL_FRAMING_W109_V1.md`)
  — the exposed-vs-resistant × pass-vs-fail grid. The active instrument for the
  contamination-confound question; W110 extends its empty top-right cell.

## Useful control / baseline-only (NOT frontier superiority)

* **APPS (2021, contamination-EXPOSED, C7 = C)** — a CONTROL battlefield only.
  A PASS here is evidence ABOUT the confound, never a retirement and never
  publication-grade. The W109 +16.67 pp is a control signal, not a win to cite.

## Dead directions (capped — do NOT re-run)

* **A multi-seed LiveCodeBench de-noise** — NOT WARRANTED
  (`W109-T-LIVECODEBENCH-DENOISE-NOT-WARRANTED`): a negative-margin + weak-MLB-2
  FAIL cannot be de-noised into a PASS (multi-seed cuts variance, not the mean).
* **Contamination-RESISTANT same-budget code superiority via the W89 mechanism
  at 70B** — the only attempt (W108 LiveCodeBench) FAILed; unproven and not to
  be presented as shown.
* **The closed Llama-3.1 rescue-concentrated branch and the 405B re-probe** —
  stay closed (the W109 de-noise rule explicitly does not re-open them).

## Anti-patterns (reinforced by W109)

* **Bounded-context / compaction / prose-summary / token-compression** REMAIN
  explicit anti-patterns, NOT the frontier path. W109 did none of these.
* **NEW W109 lesson**: an exposed-benchmark PASS is a CONTROL signal, never a
  retirement. The honest move is to LABEL it as control evidence (and note the
  PASS_NON_MECHANISM_DRIVEN invocation caveat), not to bank a +16.67 pp headline.
* **NEW W109 lesson**: a high A0 on exposed data (73 % vs 43 % resistant) is a
  memorization-consistent signal, AND it mechanically suppresses MLB-1 — read
  the two together, do not cite the margin as clean mechanism-driven.

## Do not claim (W109 additions)

* That W109 PROVES contamination (SUPPORTED ≠ established; one single-seed pair).
* That APPS is a third retirement or publication-grade (it is contamination-
  EXPOSED control evidence).
* That W109 overwrites or weakens the W108 LiveCodeBench FAIL or the W89/W105
  retirements (it sharpens their boundary; it does not move them).
