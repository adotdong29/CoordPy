# W103 — Frontier relevance audit V1 (supplement to W97 / W98 / W99 / W100 / W101 / W102)

> **2026-05-25.  Seventh supplement, extending the W97 / W98 /
> W99 / W100 / W101 / W102 frontier-relevance audits with the
> W103 HumanEval+ lead-pilot path + helper-consumption + code-
> line hardening contract.  All W97 – W102 audit classifications
> remain in force VERBATIM; this supplement only:**
>
> 1. Promotes `scripts/run_w103_humaneval_plus_pilot.py` +
>    `tests/test_w103_code_line_discipline_v1.py` to the
>    **active frontier arsenal** column.
> 2. Re-asserts `coordpy.mbpp_plus_loader_v1` /
>    `..._executor_v1` / `..._reflexion_bench_v1` as
>    **historical artifact + anti-pattern (silent-degeneration
>    failure mode)** per the W102 audit.
> 3. Re-asserts the cross-modal RealWorldQA arc as **frozen at
>    11B**.
> 4. Re-asserts the W97 / W98 / W99 / W100 / W101 / W102 anti-
>    pattern columns *verbatim*, with the W103 hardening
>    contract codifying the W102 silent-degeneration anti-
>    pattern + arsenal-mining-prior anti-pattern at unit-test
>    + verdict-write granularity.
> 5. Adds the W103 helper-consumption attestation as the first
>    real downstream use of `coordpy.code_slice_selector_v1`
>    in a NIM-spending pilot (COO-14 DoD item 4 realised).
>
> No code is removed by this audit.  No version bump.  No PyPI
> publish.  W103 ships zero new `coordpy.*` modules.

## Why a supplement, not a rewrite

W102 documented the classification as of the W102 cheap MBPP+ V2
pilot's FAIL verdict.  W103 adds three new empirical / structural
facts:

1. The W102 HumanEval+ preflight verdict
   (`4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4`)
   re-confirms deterministically on a fresh 2026-05-25 run.  The
   cheap pilot is structurally earned by preflight discipline,
   not by hope.
2. The W103 helper-anchored slice (slice CID
   `c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2`;
   28 unique HumanEval+ task_ids + 2 base-HumanEval top-ups)
   is the first NIM-spending pilot whose slice is
   `coordpy.code_slice_selector_v1`-driven rather than parallel-
   deterministic-seed-shuffled.  COO-14 transitions from "shipped
   helper with worked example" to "shipped helper as load-bearing
   pilot input".
3. The W103 hardening lane codifies the W102 silent-degeneration
   anti-pattern + arsenal-mining-prior anti-pattern into
   `tests/test_w103_code_line_discipline_v1.py` (8 PASSing tests)
   so neither failure mode can quietly recur on the code line.
   The W102 audit anti-pattern column transitions from "named
   anti-pattern" to "test-collection-time + verdict-write-time
   guardrail".

W103's job is to ship the lead-lane pilot at 70B + the hardening-
lane regressions + the W104 pre-commit by outcome.  The empirical
pilot verdict lives in
`docs/RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md` (populated
at pilot completion).

## Active frontier arsenal — W103 additions

| Mechanism | Module(s) | Why frontier (with W103 evidence) |
|---|---|---|
| **W103 HumanEval+ cheap-pilot driver** (NEW) | `scripts/run_w103_humaneval_plus_pilot.py` | Active frontier operating-system piece.  Consumes `coordpy.code_slice_selector_v1` as a load-bearing input (W102 COO-14 downstream-consumption deliverable); helper-anchored slice de-dups task_ids across historical seeds + tops up from base-HumanEval proposal + refuses if cluster mix collapses to all `shared_wins`.  Records provenance fields (corpus_sha + helper_proposal_cid + mining_report_cid + preflight_verdict_cid + slice_cid_helper_priority + slice_cid_bench_order + arsenal_mining_prior with explicit "NOT a Phase 2 gate input" earning_status). |
| **W103 code-line discipline tests** (NEW) | `tests/test_w103_code_line_discipline_v1.py` | Active frontier operating-system piece.  8 unit tests codifying the W102 lessons: (i) executor-level synthetic silent-degeneration row guard; (ii) preflight P5 probe fires on missing `def check(` blocks via mocked loader; (iii) `_evaluate_phase2_gates` function body MUST NOT reference any arsenal-mining-prior symbol (anti-pattern carry-forward); (iv) pilot driver MUST record provenance keys for corpus_sha / helper_proposal / mining_report / preflight_verdict / slice CIDs; (v) helper-anchored slice MUST de-dup task_ids across historical seeds; (vi) helper-anchored slice MUST refuse all-shared_wins collapse; (vii) production slice CID `c35155956ece605c…` is deterministically pinned; (viii) pilot driver MUST NOT contain forbidden anti-pattern tokens. |
| **W103 helper-consumption attestation** (NEW doc) | `docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md` | Active frontier operating-system piece.  The first published attestation that a NIM-spending pilot's slice is helper-anchored (not parallel-deterministic-seed-shuffled).  Realises COO-14 DoD item 4 ("Feed output into runbooks before expensive runs are approved") as a NIM-pilot earning constraint. |
| **W103 RUNBOOK pre-commit contract** (NEW doc) | `docs/RUNBOOK_W103.md` | Active frontier operating-system piece.  Pre-locks the lead-lane decision logic (slice rule + 9 Phase 2 gates + MLB-1 + MLB-2 sub-gates + branch decision logic) BEFORE the pilot driver was written + the W104 next-step pre-commit by outcome (Branch A = cross-scale; Branch B = mechanism variations; Branch C = code-line ranking refresh with explicit triage). |
| **W103 preflight reconfirmation** (NEW doc) | `docs/RESULTS_W103_HUMANEVAL_PLUS_PREFLIGHT_RECONFIRM_V1.md` | Active frontier operating-system piece.  Records the fresh 2026-05-25 7/7 PASS verdict cid byte-identical to W102's, demonstrating the SHA-pinned + deterministic preflight discipline is repeatable across milestones. |

## Useful baselines (W103 changes from W102)

| Mechanism | Module(s) | Classification | W103 status |
|---|---|---|---|
| `bounded_window_baseline_v{1,2,3}` | `coordpy/bounded_window_baseline_v*.py` | UNCHANGED — useful falsifier targets. | Same. |
| `coordpy.mbpp_reflexion_bench_v1` (base MBPP) | `coordpy/mbpp_reflexion_bench_v1.py` | UNCHANGED — baseline-only since W101. | Same. |
| `coordpy.humaneval_reflexion_bench_v1` (HumanEval base) | `coordpy/humaneval_reflexion_bench_v1.py` | UNCHANGED — retirement-anchor (W89 70B retirement). | Same. |
| `coordpy.mbpp_plus_loader_v1` (W101 broken loader) | `coordpy/mbpp_plus_loader_v1.py` | **UNCHANGED W103** — historical artifact + anti-pattern (silent-degeneration failure mode) per W102 audit. | Stays in-repo. |
| `coordpy.mbpp_plus_executor_v1` (W101 broken executor) | `coordpy/mbpp_plus_executor_v1.py` | **UNCHANGED W103** — same. | Stays in-repo. |
| `coordpy.mbpp_plus_reflexion_bench_v1` (W101 broken bench) | `coordpy/mbpp_plus_reflexion_bench_v1.py` | **UNCHANGED W103** — same. | Stays in-repo. |
| `coordpy.mbpp_plus_preflight_v1` | `coordpy/mbpp_plus_preflight_v1.py` | **UNCHANGED W103** — partially retained per W102. | Stays in-repo; reused by V2. |
| `coordpy.mbpp_plus_loader_v2` / `_executor_v2` / `_reflexion_bench_v2` / `_preflight_v2` | `coordpy/mbpp_plus_*_v2.py` | **DEMOTED W103 — dead direction at 70B Phase 2 (W102 verdict)**.  These modules are not deleted; they remain in-repo as the canonical EvalPlus MBPP+ infrastructure and are usable if a future milestone targets MBPP+ V2 at a different scale or with a different mechanism.  W103 does NOT attack MBPP+ V2 again. | Stays in-repo; not the W103 / W104 lead path. |

## Historical artifacts (unchanged from W97 / W98 / W99 / W100 / W101 / W102)

W90 / W92 / W88 (cross-modal code) / W81 / W83 / W84 unchanged.
Kept for regression / audit; not active path.

## Dead directions (W103 changes — pre-pilot)

Note: the dead-direction column is updated post-pilot in
`docs/RESULTS_W103_MILESTONE_SUMMARY_V1.md` § "Dead directions
(W103 verdict-driven additions)".  This pre-pilot list inherits
W102 verbatim:

| Mechanism | Evidence against | W103 status |
|---|---|---|
| **VLM-Verifier-Final-Turn as load-bearing rescue** | W96-C | UNCHANGED — refuted. |
| **W95-B0 free-text bullet extraction as sufficient on vision-bound benches** | W97 D2-B0 11B | UNCHANGED — refuted. |
| **B1 typed schema *with* `direct_answer_hint`** | W98 B1 11B | UNCHANGED — refuted. |
| **B4 typed schema *without* `direct_answer_hint`** | W99 B4 11B | UNCHANGED — refuted. |
| **Typed-extract-then-text-reason sub-family of W95-B0** | W97 + W98 + W99 11B | UNCHANGED — dead. |
| **W95-B0 family REPAIR via B2 mechanism (image-at-decision-boundary)** | W99 11B PASS + W100 90B FAIL | UNCHANGED — restricted to 11B regime only. |
| **Cross-modal RealWorldQA at the +5 pp Phase 2 bar at any scale** | W97 + W98 + W99 11B + W100 90B all FAIL | **UNCHANGED W103**: arc remains frozen at 11B; W103 does NOT re-open it. |
| **Base MBPP at K=5 same-budget at 70B for retirement** | W91 + W101 V1 loader scratch run | UNCHANGED — capped via ceiling saturation. |
| **MBPP+ V2 at K=5 same-budget at 70B for Phase 2 (cheap-pilot scale)** | W102 70B Phase 2 cheap pilot: B − A1 = −6.67 pp; MLB-2 = 22.22 % (FAIL) | UNCHANGED — W102 carry-forward `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP` stays in force.  W103 does NOT re-attempt MBPP+ V2 at 70B; re-running on a fresh seed would be hope-driven, not evidence-earned. |
| **Cross-bench arsenal-mining priors as substitute for fresh-sampling cheap pilots** | W102 arsenal-mining +5.28 pp prior vs empirical −6.67 pp (11.95 pp swing) | UNCHANGED — W102 anti-pattern in force.  W103 codifies this at unit-test granularity (`test_w103_arsenal_mining_prior_is_not_earning_evidence`) so any future refactor that lets an arsenal-mining symbol leak into `_evaluate_phase2_gates` fires CI immediately. |
| **W101 V1 silent-degeneration loader (`plus_input` / `plus_output` parallel-array schema)** | W102 fetch step | UNCHANGED — W102 anti-pattern in force.  W103 codifies the structural defence at unit-test granularity (`test_w103_humaneval_plus_executor_refuses_synthetic_silent_degeneration` + `test_w103_humaneval_plus_preflight_p5_catches_missing_check_block`). |

## Anti-patterns (NEVER promote as core strategy; baseline-only allowed)

**The W97 / W98 / W99 / W100 / W101 / W102 anti-pattern list
remains in force VERBATIM in W103**, with two W103 codifications:

| Anti-pattern | W103 status |
|---|---|
| Bounded context window as product thesis | UNCHANGED — anti-pattern; baseline-only.  W103 pilot driver passes the `test_w103_pilot_driver_does_not_import_anti_patterns` test (no `bounded_window` / `compaction` / etc. tokens in driver source). |
| Compaction / generic prose summarization as memory mechanism | UNCHANGED — anti-pattern. |
| Shallow token compression without structural reason | UNCHANGED. |
| Context-pruning theater | UNCHANGED. |
| "Cram less / truncate better" as frontier memory system | UNCHANGED. |
| LLM-as-judge in executor chain | UNCHANGED — the W103 HumanEval+ pilot's executor is subprocess CPython; no LLM judges any candidate's correctness. |
| Selective retries | UNCHANGED — W103 bench follows the W88 / W90 / W102 contract: no early-stop on PASS; every K=5 budget element runs to completion. |
| Single-seed pilots as retirement evidence | UNCHANGED — W103 cheap pilot is a 1-seed × 30-problem cheap pilot AT THE Phase 2 SIZE.  Phase 3 retirement is W105+ if W103 + W104 both PASS. |
| Architecture refinement by vibe | UNCHANGED — W103 pilot is locked by the W103 RUNBOOK's pre-committed decision logic. |
| Re-opening dead directions | UNCHANGED — cross-modal RealWorldQA arc + W95-B0 family + typed-extract sub-family + MBPP+ V2 at 70B remain dead. |
| **Silent-degeneration via schema assumption** (W102 anti-pattern; W103 codification) | The W101 V1 loader is the canonical example.  W103 codifies the structural defence at unit-test granularity: `test_w103_humaneval_plus_executor_refuses_synthetic_silent_degeneration` constructs a synthetic HumanEval+ row WITHOUT a `def check(` block and asserts the executor produces an observable FAIL; `test_w103_humaneval_plus_preflight_p5_catches_missing_check_block` mocks the loader to return a corpus where 1 of 2 rows lacks `def check(` and asserts P5 fires at the 95 % floor.  Loaders + executors MUST verify their schema assumption against the real data BEFORE any NIM spend; preflight P-probes MUST catch silent-degeneration failure modes; unit tests MUST fire on synthetic silent-degeneration shape. |
| **Cross-bench arsenal-mining priors as cheap-pilot earning evidence** (W102 anti-pattern; W103 codification) | The W102 arsenal-mining run showed +5.28 pp on MBPP+ V2 historically re-graded; the fresh-sampling pilot was −6.67 pp.  W103 codifies the structural defence at unit-test granularity: `test_w103_arsenal_mining_prior_is_not_earning_evidence` greps the `_evaluate_phase2_gates` function body and asserts no `ARSENAL_MINING_PRIOR` / `mining_report` symbol leaks into the gate evaluation logic; `test_w103_pilot_driver_records_provenance_fields` asserts the arsenal-mining prior is RECORDED in the provenance dict with explicit `earning_status: "NOT a Phase 2 gate input"` text. |
| **Slice selection by parallel deterministic seed instead of helper-anchored cluster mix** (NEW W103 anti-pattern candidate) | W101 / W102 cheap pilots used `seed=101_001` deterministic shuffles over the full corpus.  W103 demonstrates that helper-anchored slice selection (consuming `coordpy.code_slice_selector_v1` as a load-bearing input) produces a 63 % historically-hard cluster concentration vs ~33 % for the parallel deterministic shuffle — i.e., the rescue / stress surface is structurally larger per problem under helper anchoring.  This is not yet a hard anti-pattern (small N; W103 is the first such pilot), but the W103 audit RECORDS this as a "slice rule to track across W104+"; if a future pilot demonstrates that parallel-seed shuffles systematically under-test the rescue surface, the audit promotes parallel-seed-only-slice to anti-pattern. |

## What W103 cross-promotion is NOT

To pre-empt drift back toward commodity-LLM tricks under a new
name:

### W103 is NOT a cross-modal milestone

* Cross-modal RealWorldQA arc remains frozen at 11B per the
  W100 frontier audit.  W103 does NOT re-open it.
* If the W103 HumanEval+ cheap pilot succeeds, the next
  milestone (W104) is the cross-SCALE confirmation of the
  *code-line* result.

### W103 is NOT a Phase 3 retirement attempt

* The cheap pilot is a 1-seed × 30-problem cheap-pilot at the
  Phase 2 SIZE.  Phase 3 retirement is the 3-seed × 100-
  problem × K=5 retirement bench analogous to W89 HumanEval;
  that lives in W105+ if W103 + W104 both PASS.

### A W103 cheap-pilot PASS at 70B is NOT a multi-benchmark same-budget retirement

* A PASS at 70B Phase 2 means the W89 mechanism extends to a
  SECOND EvalPlus benchmark family at the cheap-pilot scale
  (the first was the bench's own retirement on base HumanEval).
  Full retirement of `W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP`
  requires multi-seed Phase 3 evidence on an EvalPlus surface
  that targets the MBPP-family cap directly; HumanEval+ does
  NOT relieve that specific cap.

### W103 is NOT a new-module milestone

* W103 ships ZERO new `coordpy.*` modules.  The lead lane
  reuses W102 backup-lane modules verbatim.  The hardening
  lane lives in tests + driver + result docs.  This is the
  honest scope: a milestone can be substantial without
  expanding the SDK surface.

## Honest classification of a W103 PASS / FAIL pattern

| W103 cheap-pilot outcome | What we earn | What we do NOT claim |
|---|---|---|
| Preflight 7/7 PASS + cheap pilot B − A1 ≥ +5 pp + MLB sub-gates clearing at 70B | The W89 sequential-reflexion mechanism extends to the SECOND EvalPlus benchmark in the family (HumanEval+) at the cheap-pilot scale; the W103 carry-forward is the W104 cross-scale confirmation entitlement. | Multi-seed same-budget retirement on HumanEval+ (Phase 3 needed).  Retirement of `W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP` (HumanEval+ does not relieve the MBPP-family cap). |
| Preflight 7/7 PASS + cheap pilot B − A1 < +5 pp | Mechanism is partially load-bearing on HumanEval+ but does not clear the Phase 2 bar; carry-forward `W103-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-70B-CAP`; W104 refreshes the code-line ranking per the runbook's Branch C triage. | Anything Phase 3.  Anything about W89 generalisation if mechanism fails on the same benchmark family it was retired on. |
| Preflight 7/7 PASS + cheap pilot B − A1 ≥ +5 pp + MLB-2 FAILS | `PASS_NON_MECHANISM_DRIVEN`; cross-scale W104 NOT entitled per W96-C / W100 precedent; W104 = mechanism-variation slate on HumanEval+ per the runbook's Branch B triage. | Mechanism load-bearingness on HumanEval+. |
| Preflight P-probe FAILS at re-run | Infrastructure or corpus bug; fix; re-run preflight; do NOT launch pilot. | Anything empirical. |

## What this supplement DOES NOT do

* It does NOT claim multi-agent context is solved.
* It does NOT claim the W103 HumanEval+ cheap pilot will
  succeed — the pilot is launched after preflight 7/7 PASS
  re-confirms; the verdict is recorded in
  `docs/RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md` after the
  pilot completes.
* It does NOT retire any prior carry-forward.
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.
* It does NOT touch `coordpy/__init__.py`.
* It does NOT re-open any dead direction.
* It does NOT delete the W101 V1 anti-pattern modules — they
  stay in-repo as historical artifacts + anti-pattern
  exhibits.

## Honest scope

This is a *classification supplement* + *anti-drift contract*
+ *hardening-codification recording* that the W103 milestone
ships the lead-lane pilot driver + helper-consumption +
hardening tests, AND launches the cheap HumanEval+ pilot at 70B
on the helper-anchored slice.  The empirical pilot verdict lives
in `docs/RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md`
(populated at pilot completion); the milestone summary is
finalised in `docs/RESULTS_W103_MILESTONE_SUMMARY_V1.md`.

## Anchors

* `docs/RUNBOOK_W103.md` — W103 pre-commit contract.
* `docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md` — slice
  attestation.
* `docs/RESULTS_W103_HUMANEVAL_PLUS_PREFLIGHT_RECONFIRM_V1.md`
  — preflight reconfirmation.
* `docs/RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md` — pilot
  verdict (populated post-pilot).
* `docs/RESULTS_W103_MILESTONE_SUMMARY_V1.md` — milestone
  summary.
* `scripts/run_w103_humaneval_plus_pilot.py` — pilot driver.
* `tests/test_w103_code_line_discipline_v1.py` — hardening
  tests.
* `coordpy/humaneval_plus_loader_v1.py` etc. — W102 backup-
  lane modules re-used verbatim by W103.
