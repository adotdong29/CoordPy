# Frontier-relevance audit — W107 supplement V1

> **2026-05-28.  Supplement to the W106 V1 audit
> (`docs/FRONTIER_RELEVANCE_AUDIT_W106_V1.md`).  All W106
> classifications remain in force.  This audit adds W107's additions
> and re-confirms the W93 → W107 frontier discipline as the **17th
> consecutive preflight-discipline validation**.**

## Summary

W107 is a gated branch milestone.  The active frontier arsenal is
extended by ONE battlefield-scaffolding pair (LiveCodeBench
loader+executor) plus the structural-soundness pivot test; the
dead-direction column gains ONE sharpening (the 405B cross-scale-UP
gate is now four-times-closed); the anti-pattern column is unchanged
and was actively respected (the closed Llama-3.1 rescue-slice branch
was NOT re-introduced); the baseline-only falsifier column is
unchanged.

W107 is the FIRST milestone where the discipline question was not "is
this run allowed?" (W93+) nor "can this allowed run change the
verdict?" (W106) but "**is this pre-committed primary battlefield
STRUCTURALLY SOUND, or must we pivot to the backup before spending a
cent?**" — answered by the S1∧S2∧S3 test, which kept LiveCodeBench as
primary on real offline evidence rather than on the W106 pre-commit
alone.

## Active frontier arsenal (NEW W107 additions)

* **`coordpy.livecodebench_loader_v1`** — pure-Python, no-NIM,
  explicit-import-only SHA-pinnable release_vN functional-subset
  loader.  Refuses unpinned releases, cross-version mixing, and
  schema mismatch (the W102 P5 silent-degeneration guard applied to a
  brand-new corpus).  A SIBLING of `humaneval_plus_loader_v1`.
* **`coordpy.livecodebench_executor_v1`** — pure-Python, no-NIM,
  explicit-import-only clean functional-form subprocess executor.
  Fresh `-I` CPython subprocess; soft+kill wall timeout; binary
  PASS/FAIL; failure tail returned for the reflexion signal; resolves
  top-level OR `Solution`-method entry points; NO LLM judge.  A
  SIBLING of `humaneval_plus_executor_v1` (confirmed via `graphify
  path`: the bench imports_from the executor; the new pair mirrors
  that wiring).  Offline self-test: gold PASS / wrong FAIL / loop
  TIMEOUT.
* **The S1∧S2∧S3 structural-soundness pivot test** — a pre-committed
  rule that can swap a pre-committed primary battlefield for its
  backup INSIDE a single milestone, on real offline evidence, before
  any NIM spend.  Infrastructure, not a claim.

* **graphify as part of the operating system** — refreshed from HEAD
  at start AND close; `query` / `path` / `explain` / `affected` used
  concretely to choose the β scaffolding's sibling template and the
  COO-14 slice helper.

## Useful baseline-only (unchanged)

`bounded_window_baseline_v{1,2,3}` remain falsifier targets, NOT
frontier methods.  W107-β selected a HARDER battlefield
(contamination-resistant LiveCodeBench), explicitly NOT a smaller
window — the anti-pattern boundary holds.

## Dead directions (unchanged + ONE sharpened)

* MBPP+ V2 at 70B (W102 cap) — dead; not re-opened.
* Cross-modal beyond 11B (W100) — frozen.
* The Llama-3.1 rescue-concentrated cheap confirmation (W106 NO-GO) —
  dead; NOT re-introduced under any label in W107.
* **NEW W107 sharpening**: the 405B cross-scale-UP path is now
  **four-times-closed** (W104/W105/W106/W107 all HTTP 404).  It is
  not declared permanently dead (it re-opens iff 405B becomes
  reachable), but it is no longer a live W107/W108 lane — the live
  strengthening path is a third code benchmark family.

## Anti-patterns (unchanged; actively respected in W107)

bounded windowing / compaction / generic prose summarization / shallow
token compression / context-pruning theater / "cram less / truncate
better" / margin-gaming REMAIN explicit anti-patterns.  W107 adds no
new anti-pattern; it actively respected the W106 margin-gaming guard
(no rescue-slice re-run) and the W102 silent-degeneration guard (the
LiveCodeBench loader refuses on schema mismatch instead of degrading).

W107 also records an **honesty refinement**, not an anti-pattern: the
distinction between **published-baseline-grade** and
**re-executed-sidecar-grade** failure-residual estimates.  The
EvalPlus pair (W101) had the stronger sidecar-grade residual; the
LiveCodeBench residual is only published-baseline-grade and is capped
as such (`W107-L-LIVECODEBENCH-RESIDUAL-PUBLISHED-BASELINE-GRADE-CAP`)
rather than presented as equivalent.

## Discipline validation count

W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 / W100 /
W101 / W102 / W103 / W104 / W105 / W106 / **W107** = **17**.

## Anchors

* `docs/FRONTIER_RELEVANCE_AUDIT_W106_V1.md` — the W106 audit this
  supplements.
* `docs/RESULTS_W107_405B_GATE_V1.md` — α gate verdict.
* `docs/RESULTS_W107_NEXT_BATTLEFIELD_PREFLIGHT_V1.md` — β preflight.
* `docs/CONSOLIDATED_CODE_RETIREMENT_NARRATIVE_V1.md` — γ narrative.
* `coordpy/livecodebench_loader_v1.py` + `coordpy/livecodebench_executor_v1.py` + `tests/test_w107_livecodebench_preflight_v1.py`.
* `docs/RUNBOOK_W107.md` — pre-commit contract.
