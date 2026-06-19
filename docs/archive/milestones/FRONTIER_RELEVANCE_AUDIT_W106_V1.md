# Frontier-relevance audit — W106 supplement V1

> **2026-05-28.  Supplement to the W105 V1 audit
> (`docs/FRONTIER_RELEVANCE_AUDIT_W105_V1.md`).  All W105
> classifications remain in force.  This audit adds W106's
> additions and re-confirms the W93 → W106 frontier discipline as
> the 16th consecutive preflight-discipline validation.**

## Summary

This is the W106 frontier-relevance audit, the 16th consecutive
W6X / W7X / W8X / W9X / W10X preflight-first + cross-scale +
multi-candidate-tournament-then-confirm + mechanism-load-
bearingness + silent-degradation-anti-pattern-guard + arsenal-
mining-prior-anti-pattern-guard + cross-class-row-alignment +
**margin-cap-verdict-changing-power** discipline validation.  The
active frontier arsenal is extended by ONE module; the
dead-direction column is unchanged; the anti-pattern column is
unchanged and was actively exercised (the W106 NO-GO IS the
anti-pattern guard firing); the baseline-only falsifier column is
unchanged.

W106 is the FIRST milestone where the discipline question was not
"is this expensive run allowed?" but "**can this allowed run
change the verdict?**" — and answered NO, declining an ENTITLED
~990-call cheap confirmation.

## Active frontier arsenal (NEW W106 addition)

* **`coordpy.margin_cap_dispatch_v1`** — pure-Python, no-NIM,
  explicit-import-only TWO-GATE margin-cap dispatch rule.  GATE 1
  = the pre-committed W104/W105 Branch-C entitlement table (maps a
  `FAIL_<reason>` class's signature to an entitled next step +
  NIM ceiling).  GATE 2 = verdict-changing power (fair-battlefield
  ∧ no-authoritative-fair-result ∧ fixable-confound).  Consumes
  the `coordpy.phase3_retirement_evaluator_v1` verdict; refuses to
  run on schema / not-exactly-one-FAIL-class / no-RETIRED-class.
  It is a SIBLING of `phase3_retirement_evaluator_v1` +
  `cross_class_comparator_v1` (confirmed via `graphify explain` /
  `graphify affected`: same refuse-to-run error shape; consumes
  their verdict output).  20 PASSing unit tests.

* **graphify as part of the operating system** — the repo graph
  (`graphify update .`; `query` / `affected` / `explain` / `path`)
  is now part of the evidence + navigation loop, refreshed at the
  start AND close of the milestone with dated
  `graphify-out/YYYY-MM-DD/` backups.  This is infrastructure,
  not a claim.

## Useful baseline-only (unchanged)

`bounded_window_baseline_v{1,2,3}` remain falsifier targets, NOT
frontier methods.  The W106 NO-GO reinforces this: declining
spend that cannot change a verdict is the same discipline that
keeps bounded/compaction methods classified as anti-patterns
rather than the frontier path.

## Historical artifacts (unchanged)

W90 / W92 / W88 / W81 / W83 / W84 and all prior-wave arsenal
unchanged.

## Dead directions (unchanged + ONE sharpened)

* MBPP+ V2 at 70B (W102 cap) — dead; not re-opened.
* Cross-modal team organisation on HumanEval-Visual (W92) — dead.
* RealWorldQA cross-modal beyond 11B (W100) — frozen.
* **NEW W106 sharpening**: a *rescue-concentrated cheap
  confirmation as a path to retiring a fair-slice FAIL_MARGIN
  class* is now an explicitly DEAD move — it can only produce an
  upper bound (the W102 anti-pattern), never overturn the
  authoritative fair broad-slice Phase 3 verdict.  The only live
  way to strengthen the Llama-3.1 result is a genuinely different
  battlefield (cross-scale-UP to 405B if reachable), never a
  rescue-concentrated re-run.

## Anti-patterns (unchanged; actively exercised in W106)

bounded windowing / compaction / generic prose summarization /
shallow token compression / context-pruning theater / "cram less
/ truncate better" REMAIN explicit anti-patterns.  W106 adds the
**margin-gaming anti-pattern** to the actively-guarded set: buying
NIM spend on a slice constructed to inflate B − A1 is forbidden
even when the dispatch table entitles the spend, because it cannot
change the retirement verdict.  The W106 NO-GO is this guard
firing for the first time on an ENTITLED spend.

## Discipline validation count

W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 / W100 /
W101 / W102 / W103 / W104 / W105 / **W106** = **16**.

## Anchors

* `docs/FRONTIER_RELEVANCE_AUDIT_W105_V1.md` — the W105 audit this
  supplements.
* `docs/RESULTS_W106_MARGIN_CAP_DISPATCH_V1.md` — the NO-GO
  decision.
* `docs/RESULTS_W106_BOUNDED_RETIREMENT_REGISTRATION_V1.md` — the
  bounded registration.
* `coordpy/margin_cap_dispatch_v1.py` + `tests/test_w106_margin_cap_dispatch_v1.py`.
* `docs/RUNBOOK_W106.md` — pre-commit contract.
