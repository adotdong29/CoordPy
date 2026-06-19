# W104 — Frontier-relevance audit V1 (supplement to W97 V1)

> **2026-05-26.  Audit supplement for the W104 HumanEval+ cross-
> scale Phase 2 cheap pilot.  Inherits the entire W97 V1 audit
> verbatim (`docs/FRONTIER_RELEVANCE_AUDIT_W97_V1.md`) PLUS the
> W98 / W99 / W100 / W101 / W102 / W103 extensions verbatim.
> This file adds ONLY the W104-specific surface to that base.**

## Audit purpose

The W93 – W103 discipline thread requires every milestone to
classify each piece of named infrastructure as one of:

* **Active-frontier arsenal** — the candidate / mechanism /
  helper actively being tested or consumed in this milestone's
  attack surface.
* **Useful baseline-only** — falsifier targets, anti-pattern
  comparison surfaces, and infrastructure that exists only to
  enforce the anti-pattern column.
* **Historical artifact** — frozen output from a past
  milestone that survives only as audit-chain evidence; not
  re-opened.
* **Dead direction** — empirically capped; explicitly NOT
  re-opened in this milestone.
* **Anti-pattern** — explicitly named as a forbidden pattern;
  the W93 – W103 anti-pattern list stays in force.

W104 is the FOURTEENTH consecutive validation of the preflight-
first + cross-scale + multi-candidate-tournament-then-confirm
+ mechanism-load-bearingness + silent-degradation-anti-pattern-
guard + arsenal-mining-prior-anti-pattern-guard discipline.

## Active-frontier arsenal (new W104 additions)

* **W104 cross-scale-pilot driver**
  (`scripts/run_w104_humaneval_plus_cross_scale_pilot.py`):
  consumes the W103 helper-anchored slice byte-for-byte;
  reachability smoke probe BEFORE NIM spend; resume-from-
  sidecar; cross-scale comparator emitted automatically.  This
  is the load-bearing cross-scale infrastructure for the
  code-benchmark line.
* **W104 cross-scale comparator V1**
  (`coordpy/cross_scale_comparator_v1.py`): single new
  explicit-import-only module; schema/provenance-guarded;
  refuses to run on slice / corpus / schema mismatch; emits
  per-problem cluster-shift classification (stayed / improved
  / regressed / flipped); aggregate arm deltas; cross-scale
  margin shift; cross-scale MLB-2 shift.
* **W104 W105 Phase 3 slice-pack pre-builder**
  (`scripts/run_w104_w105_phase3_slice_pack.py`): second real
  load-bearing downstream consumption of
  `coordpy.code_slice_selector_v1`; deterministic 100-problem
  slice with W103 inner kernel + helper mid-shell + corpus-
  fill outer top-up; SHA-anchored slice pack CID.
* **W104 target-selection rule V1**: deterministic 6-criterion
  rule (reachability + cross-scale legitimacy + same-budget
  comparability + anti-saturation legitimacy + anti-pattern
  token absence + practical runtime); pre-locked in
  `docs/RUNBOOK_W104.md` BEFORE any NIM call; primary 405B +
  pre-committed cross-generation backup at same scale.
* **W104 cross-scale-discipline test suite**
  (`tests/test_w104_cross_scale_discipline_v1.py`): 14 tests
  codifying the W102 + W103 + W104 lessons at unit-test
  granularity (sidecar resume-from-disk; comparator schema /
  slice / corpus refuse-to-run; cluster-shift correctness;
  target-selection determinism; W103 slice CID equality).

## Useful baseline-only (carry-forward verbatim from W101 / W102 / W103)

* `coordpy/bounded_window_baseline_v1.py`,
  `..._v2.py`, `..._v3.py` — falsifier targets; remain
  baseline-only.

## Historical artifacts (carry-forward verbatim from W102 / W103)

* `coordpy/mbpp_plus_loader_v1.py` /
  `..._executor_v1.py` /
  `..._reflexion_bench_v1.py` /
  `..._preflight_v1.py` — W101 V1 backup-lane infrastructure;
  DEMOTED to historical artifact + anti-pattern per the W102
  silent-degeneration audit; stays in repo for the W101
  audit-chain trail only.

## Dead directions (carry-forward verbatim from W100 + W102 +
W103)

* Cross-modal RealWorldQA (W100): frozen at 11B.
* W95-B0 family / typed-extract sub-family on RealWorldQA: dead.
* ChartQA: preflight-killed (W96-D); not re-opened.
* MBPP+ V2 cross-scale: dead (W102 cap + no entitlement).
* APPS / LiveCodeBench / SWE-bench-lite: out of scope per the
  W101 matrix; only Branch C (W104 FAIL) re-opens
  LiveCodeBench / APPS as NIM-FREE preflight candidates
  (NOT as expensive bench).
* W101 V1 MBPP+ infrastructure: anti-pattern per W102 (silent-
  degeneration via schema assumption).

## Anti-patterns (carry-forward verbatim from W97 V1 +
W98 / W99 / W100 / W101 / W102 / W103)

* Bounded windowing.
* Compaction.
* Context compaction.
* Generic prose summarization.
* Context pruning theater.
* Shallow token compression.
* "Cram less / truncate better" framing.
* Silent-degradation-via-schema-assumption (W102 carry).
* Cross-bench arsenal-mining priors as cheap-pilot earning
  evidence (W102 carry).
* **NEW W104 implication**: cross-scale Phase 2 comparison
  WITHOUT byte-equal slice is anti-pattern.  The W104
  comparator REFUSES to run on slice CID mismatch; any
  "approximate" cross-scale comparison that lets the slice
  drift between scales is named here as the structural
  failure mode the comparator guards against.

## Discipline thread

W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 /
W100 / W101 / W102 / W103 / **W104** = **FOURTEENTH consecutive
preflight-first + cross-scale + multi-candidate-tournament-
then-confirm + mechanism-load-bearingness + silent-
degradation-anti-pattern-guard + arsenal-mining-prior-anti-
pattern-guard validation**.

W104 EXTENDS the discipline thread with:

* **Cross-scale-comparator-refuse-to-run-on-slice-corpus-
  schema-mismatch** (new W104 hardening; codified in
  `coordpy.cross_scale_comparator_v1` + the 4 refuse-to-run
  tests in the W104 test file).
* **Sidecar-resume-from-disk** (new W104 hardening; codified
  in `scripts/run_w104_humaneval_plus_cross_scale_pilot.py`
  + the 2 resume tests in the W104 test file).
* **Pre-committed-cross-generation-fallback-on-reachability-
  smoke-FAIL** (new W104 hardening; the W104 RUNBOOK pre-
  locked Llama-3.1-70B-Instruct as the backup BEFORE the
  smoke probe; the pilot driver applied this deterministically
  when the 405B primary returned HTTP 404).

## Empirical event log (W104; populated 2026-05-26)

| Step | Outcome |
|---|---|
| W104 RUNBOOK locked BEFORE any NIM call | YES (this file's sibling `docs/RUNBOOK_W104.md`) |
| W104 hardening lane code shipped + tested BEFORE any NIM call | YES (`coordpy/cross_scale_comparator_v1.py` + 14 PASSing tests) |
| W104 W105 planning lane shipped BEFORE any NIM call | YES (`docs/RESULTS_W104_HELPER_W105_PLANNING_V1.md` + `data/w105/phase3_slice_pack/<RUN>/slice_pack.json`) |
| Reachability smoke probe on primary `meta/llama-3.1-405b-instruct` | FAIL (HTTP 404; NIM endpoint not hosted) |
| Reachability smoke probe on backup `meta/llama-3.1-70b-instruct` | PASS |
| Backup-target selection per pre-locked RUNBOOK | APPLIED |
| W103 slice CID equality at pilot start | VERIFIED (`c35155956ece605c...` byte-equal to W103) |
| HumanEval+ corpus SHA pin equality vs W103 | VERIFIED (`908377f1daf28dcb...` byte-equal to W103) |
| Cheap pilot empirical outcome | RECORDED in `docs/RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_V1.md` |
| Cross-scale comparator emitted | RECORDED in `docs/RESULTS_W104_CROSS_SCALE_COMPARATOR_V1.md` |
| Decision branch applied per pre-locked logic | RECORDED in `docs/RESULTS_W104_MILESTONE_SUMMARY_V1.md` |

## Honest framing

W104 honestly records that the pre-committed primary target
(`meta/llama-3.1-405b-instruct`) returned HTTP 404 on the
reachability smoke probe, and that the pre-committed backup
(`meta/llama-3.1-70b-instruct`) was applied per the pre-locked
RUNBOOK § Target-model selection rule.  The actual cross-scale
shape achieved is **cross-generation at the same parameter
scale** (Llama 3.1 vs Llama 3.3 at 70B), not cross-scale-UP
(70B → 405B).  This is a WEAKER form of cross-scale than the
primary target would have produced; the verdict doc records
this verbatim.

The pre-committed planning lane (`docs/RESULTS_W104_HELPER_
W105_PLANNING_V1.md`) was BUILT IN W104 BEFORE the pilot
launched; if the cross-generation result PASSes Branch A, the
W105 Phase 3 slice pack at the locked CID
`8be55f3bf1650df3...` is launched immediately.  If the cross-
generation result fails Branch C, the Branch C dispatch table
applies.  Either outcome leaves W105 as execution, not
paperwork.

## Anchors

* `docs/RUNBOOK_W104.md` — pre-commit contract.
* `docs/RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_V1.md` —
  pilot verdict.
* `docs/RESULTS_W104_CROSS_SCALE_COMPARATOR_V1.md` —
  cross-scale comparator narrative.
* `docs/RESULTS_W104_HELPER_W105_PLANNING_V1.md` —
  W105 Phase 3 slice pack + Branch C dispatch table.
* `docs/RESULTS_W104_MILESTONE_SUMMARY_V1.md` —
  milestone summary.
* `coordpy/cross_scale_comparator_v1.py` — comparator module.
* `scripts/run_w104_humaneval_plus_cross_scale_pilot.py` —
  pilot driver.
* `scripts/run_w104_w105_phase3_slice_pack.py` — W105 pack
  pre-builder.
* `tests/test_w104_cross_scale_discipline_v1.py` —
  hardening-lane tests.
