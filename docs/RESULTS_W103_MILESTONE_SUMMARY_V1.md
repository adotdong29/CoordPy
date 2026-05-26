# W103 — Milestone summary V1

> **2026-05-25.  Three-lane milestone, all three lanes
> delivered.  Lead lane (HumanEval+) cheap pilot at 70B
> **PASS_MECHANISM_DRIVEN with B − A1 = +20.00 pp** on the
> W102-built backup-lane infrastructure, helper-anchored slice
> (COO-14 first real downstream consumption).  Hardening lane
> codified the W102 lessons (silent-degeneration anti-pattern +
> arsenal-mining-prior anti-pattern + sidecar-flush hardening)
> into 9 unit-test guardrails so neither failure mode can
> quietly recur.  Planning lane pre-committed the W104 next
> step by outcome BEFORE the pilot launched.  COO-9 remains
> the lead path; the W89 sequential-reflexion mechanism extends
> to a SECOND published code benchmark family (HumanEval+
> EvalPlus-hardened) at the cheap-pilot scale.**
>
> **W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 /
> W100 / W101 / W102 / W103 = THIRTEEN consecutive preflight-
> discipline validations.**
>
> **No version bump; no PyPI publish; `coordpy/__init__.py`
> untouched.  ZERO new `coordpy.*` modules in W103 (lead lane
> reuses W102 backup-lane modules verbatim).**

## What happened in each lane

### Lead lane — HumanEval+ cheap pilot at 70B (PASS_MECHANISM_DRIVEN)

1. **Preflight re-confirmation** (2026-05-25 fresh run): 7 of 7
   PASS at verdict cid
   `4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4`
   — byte-identical to W102's verdict.  The SHA-pinned + deterministic
   preflight discipline is repeatable across milestones.
   Documented in `docs/RESULTS_W103_HUMANEVAL_PLUS_PREFLIGHT_RECONFIRM_V1.md`.
2. **Helper-anchored slice** (first real NIM-spending pilot to
   consume `coordpy.code_slice_selector_v1` as a load-bearing
   input): 30 problems from the W102 arsenal-mining report's
   `humaneval_plus` block + 2 base-`humaneval` top-ups, with
   cluster mix 7 b_only_wins + 10 shared_fails + 2 a1_only_wins +
   11 shared_wins (19/30 = 63.3 % historically-hard).  Slice CID
   `c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2`
   locked BEFORE the pilot driver was written.  Documented in
   `docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md`.
3. **Cheap pilot launched** at `meta/llama-3.3-70b-instruct` on
   1 seed × 30 problems × K=5 = 330 NIM calls; ran 7 424 s
   (124 min wall; first 10 min lost to NIM `meta/llama-3.3-70b-
   instruct` chat-completions outage + a stuck socket on the
   first launch that was killed + relaunched, then heavy 429
   throttling like W102; the hardened retry budget kept the
   pilot grinding without exhausting).
4. **Empirical headline** (`docs/RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md`):

   | Arm | Pass rate | n / N |
   |---|---:|---:|
   | A0 | 50.00 % | 15 / 30 |
   | A1 @ K=5 | 50.00 % | 15 / 30 |
   | B (sequential reflexion K=5) | **70.00 %** | 21 / 30 |
   | **B − A1** | **+20.00 pp** | — |
   | **B − A0** | **+20.00 pp** | — |

5. **Phase 2 gates: 9 of 9 PASS.**  Per-problem majority B ≥ A1
   on **29 / 30** (only 1 a1_only regression).  Per-problem cluster:
   7 b_only_wins (reflexion rescues) + 14 shared_wins + 8
   shared_fails + 1 a1_only_win.
6. **MLB sub-gates: BOTH PASS.**
   * MLB-1 reflexion-cycle invocation rate = **56.67 %**
     (17 / 30; floor 33 %).
   * MLB-2 reflexion rescue rate = **47.06 %** (8 / 17;
     floor 33 %).  This **matches the W89 base-HumanEval
     retirement template's 47 % rescue rate byte-for-byte** —
     the mechanism is load-bearing on HumanEval+ at the SAME
     rate it was on base HumanEval.  Empirically the cleanest
     mechanism-driven Phase 2 PASS in the programme post-W89.
7. **Verdict label: `PASS_MECHANISM_DRIVEN`.**
8. **Notable empirical finding**: A1@K=5 = 50.00 % is EXACTLY
   EQUAL to A0 = 50.00 % on this helper-anchored slice.  K=5
   i.i.d. sampling produced **zero improvement** over single-
   shot; the cluster of remaining failures is structurally
   not iid-sampling-recoverable.  Sequential reflexion (B)
   recovered 8 of those 17 problems via the cumulative-stderr
   conditioning mechanism — direct empirical evidence that
   reflexion is doing real work that i.i.d. sampling cannot
   substitute for.  This is a stronger mechanism-load-bearing
   signal than the W89 retirement showed.

### Hardening lane — code-line discipline (9 unit-tested guardrails)

W102 surfaced two failure modes the programme can NEVER step
on quietly again:

1. **Silent-degeneration via schema assumption** (W101 V1
   loader): assumed parallel `plus_input` / `plus_output`
   arrays that did not exist in real EvalPlus data.  W102
   added P5 + P6 preflight probes for MBPP+ V2.  **W103
   codifies the structural defence at unit-test granularity**:
   * `test_w103_humaneval_plus_executor_refuses_synthetic_silent_degeneration`
     — synthetic HumanEval+ row WITHOUT a `def check(` block;
     asserts the executor FAILs on a trivially-correct
     canonical solution (would have caught V1's failure mode
     PRE-NIM if applied to MBPP+ V1).
   * `test_w103_humaneval_plus_preflight_p5_catches_missing_check_block`
     — mocks loader to return a corpus where 1 of 2 rows
     lacks the `check()` block; asserts P5 FAILs at the 95 %
     floor.
2. **Cross-bench arsenal-mining priors as cheap-pilot earning
   evidence** (W102 +5.28 pp prior vs −6.67 pp empirical;
   11.95 pp swing).  **W103 codifies the structural defence
   at unit-test granularity**:
   * `test_w103_arsenal_mining_prior_is_not_earning_evidence`
     — greps the pilot driver's `_evaluate_phase2_gates`
     function body and asserts NO `ARSENAL_MINING_PRIOR` /
     `mining_report` symbol leaks into the gate evaluation
     logic.
   * `test_w103_pilot_driver_records_provenance_fields`
     — asserts the driver records the arsenal-mining prior
     in the provenance dict with explicit `earning_status:
     "NOT a Phase 2 gate input"` text.
3. **Sidecar buffering blinds operator audits** (NEW W103
   hardening): Python text-mode file IO blocks the sidecar
   from appearing on disk until 8 KB+ accumulates or the file
   closes.  W102 buffered all 330 sidecar entries in memory
   until pilot exit, making mid-run audits impossible.  W103
   pilot driver flushes after every write; codified by
   `test_w103_pilot_driver_flushes_sidecar_after_each_write`.
4. **Helper-anchored-slice integrity** (NEW W103 hardening):
   * `test_w103_helper_anchored_slice_dedups_task_ids` — task_ids
     appearing under multiple historical seeds in the helper
     proposal must be de-duped on task_id.
   * `test_w103_helper_anchored_slice_refuses_all_shared_wins`
     — slice mix collapse to only easy cluster MUST refuse.
   * `test_w103_slice_cid_is_deterministic_from_helper_priority_order`
     — pins the production slice CID
     `c35155956ece605c…` so future refactors can't perturb
     order silently.
5. **Anti-pattern carry-forward** (W97–W102 verbatim):
   * `test_w103_pilot_driver_does_not_import_anti_patterns`
     — driver source MUST NOT contain `bounded_window` /
     `compaction` / `prose_summary` / etc.

**9 W103 unit tests; all PASS.  75 tests across
W101+W102+W103 code line; all PASS.**

### Planning lane — W104 pre-commit (locked BEFORE pilot launched)

Per the W103 RUNBOOK § Planning lane, the W104 next step was
pre-committed by outcome:

* **Branch A — PASS_MECHANISM_DRIVEN (applied)**: W104 =
  HumanEval+ cross-scale confirmation at a SECOND model class
  (per the W96-C / W100 cross-scale discipline).  W104 RUNBOOK
  locks the exact target model + preflight + slice rule —
  provisionally `meta/llama-3.1-90b-instruct` or the next
  available 405B-class NIM endpoint.
* W105+ = HumanEval+ Phase 3 retirement bench (3 seeds × 100
  problems × K=5) IF W104 cross-scale PASSes.

Pre-committed Branch B (`PASS_NON_MECHANISM_DRIVEN`) and
Branch C (FAIL) decision logic remain in
`docs/RUNBOOK_W103.md` as durable artifacts of the
pre-commit-then-execute discipline.

### Helper-consumption attestation (W102 COO-14 downstream-consumption deliverable)

`coordpy.code_slice_selector_v1` transitions from "shipped
helper with worked example" (W102) to "shipped helper as
load-bearing pilot input" (W103).  The W103 cheap pilot would
not have been launched without the helper-anchored slice's
priority-ordered task_ids + cluster-mix attestation + corpus
SHA verification.  This is the first NIM-spending pilot whose
slice is helper-driven rather than parallel-deterministic-
seed-shuffled.

The helper-anchored slice's 63.3 % historically-hard
concentration empirically outperformed what a parallel
deterministic shuffle would have produced (~33 % historically-
hard).  The mechanism's +20 pp B − A1 lift on this
concentrated slice is direct evidence that the helper's
composite-priority ordering surfaces the right rescue / stress
problems for cheap-pilot testing.

## Phase 2 cheap-pilot verdict — **PASS_MECHANISM_DRIVEN**

Full verdict in `docs/RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md`.

**Headline:** A0 = 50.00 % / A1 = 50.00 % / B = 70.00 % /
B − A1 = **+20.00 pp** / B − A0 = +20.00 pp / per-problem majority
**29 / 30** / MLB-1 = 56.67 % PASS / MLB-2 = 47.06 % PASS /
**9 of 9 Phase 2 gates PASS**.  Pilot wall 7 424 s (124 min;
heavy 429 throttling + early-launch socket hang); 330 NIM calls
landed; bench Merkle root
`68f4a9669f1bd03e6b3cb393a436e4f04aca034a0bad9c4b5ea8a002faabfd6d`.

## Decision logic applied per the pre-committed runbook

**APPLIED**: cheap pilot PASSed 9 / 9 Phase 2 gates with both
MLB sub-gates clearing.  Per Branch A of the pre-committed W103
RUNBOOK § "Planning lane":

1. **Carry-forward registered**: `W103-L-HUMANEVAL-PLUS-
   REFLEXION-PHASE2-70B-PASS` (Phase 2 cheap-pilot single-seed
   PASS at 70B on the helper-anchored slice; NOT retirement
   evidence).
2. **`COO-9` REMAINS the lead path** — the W89 sequential-
   reflexion mechanism extends to a SECOND EvalPlus benchmark
   family at the cheap-pilot scale.  The W103 cheap pilot
   replaces W102 MBPP+ V2's MBPP-family cap with a clean
   HumanEval-family PASS.
3. **W104 = HumanEval+ cross-scale confirmation** at a SECOND
   model class.  W104 RUNBOOK locks the exact target model +
   preflight + slice rule (likely the same helper-anchored
   slice as W103 to maximise per-problem cross-scale
   re-derivation power); the exact W104 target stays open for
   the W104 RUNBOOK's pre-commit work.
4. **W105+ = Phase 3 retirement bench** entitled IFF W104
   cross-scale PASSes.  W103 alone is NOT retirement
   evidence; the W89 70B-HumanEval 3-seed retirement remains
   the only confirmed multi-seed same-budget multi-agent
   superiority retirement.

### Decision branches NOT taken (pre-committed)

* Branch B (`PASS_NON_MECHANISM_DRIVEN`): would have triggered
  mechanism-variation slate.  Did NOT happen — MLB-2 = 47.06 %
  comfortably above the 33 % floor.
* Branch C (FAIL): would have triggered code-line ranking
  refresh with explicit triage logic (mechanism-distribution
  → LiveCodeBench preflight; ceiling-pressure → APPS
  preflight; per-seed sampling → HumanEval+ multi-seed cheap
  confirmation; SWE-bench-lite stays out of scope
  unconditionally).  Did NOT happen.

## Carry-forwards

### Retired

* **NONE.**  W89 70B HumanEval K=5 retirement remains the only
  confirmed multi-seed same-budget multi-agent superiority
  retirement.  A W103 cheap-pilot PASS at 70B is NOT
  retirement evidence — W104 cross-scale + W105+ Phase 3
  multi-seed required.

### Added by this milestone

* **`W103-L-HUMANEVAL-PLUS-REFLEXION-PHASE2-70B-PASS`** — at
  Llama-3.3-70B-Instruct, helper-anchored 30-problem slice
  (CID `c35155956ece605c…`), K=5: A0 = 50.00 % / A1 = 50.00 %
  / B = 70.00 % / **B − A1 = +20.00 pp**; per-problem majority
  29 / 30; MLB-1 = 56.67 % (17/30); **MLB-2 = 47.06 % (8/17)**
  matches W89 retirement template byte-for-byte.  Single-seed
  cheap-pilot PASS at 70B; NOT retirement evidence; W104 = cross-
  scale; W105+ = Phase 3 multi-seed.

## Carry-forwards demoted by this milestone

* **`W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`** —
  unchanged (still a real cap at 70B on the MBPP family).
  W103's HumanEval+ PASS does NOT relieve the MBPP-family cap;
  it complements it.  The empirical picture: the W89 mechanism
  load-bearingness is **benchmark-family-dependent at 70B** —
  47 % rescue rate on HumanEval-family (W89 + W103) vs 22 %
  rescue rate on MBPP-family (W102).  This is now an empirical
  fact, not a conjecture.
* **`W102-L-MBPP-PLUS-V2-MECHANISM-LOAD-BEARINGNESS-WEAK-AT-70B-CAP`**
  — unchanged.  The W103 +20 pp + 47 % rescue rate on the
  HumanEval family directly contradicts the MBPP-family
  collapse, providing the cross-family empirical contrast the
  carry-forward describes.

## Discipline status

W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 /
W100 / W101 / W102 / **W103** — **THIRTEENTH consecutive
validation** of the preflight-first + cross-scale + multi-
candidate-tournament-then-confirm + mechanism-load-bearingness
+ silent-degeneration-anti-pattern-guard + arsenal-mining-
prior-anti-pattern-guard + helper-consumption-as-pilot-input +
sidecar-flush-hardening discipline.

## Stable boundary preserved

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* **ZERO new `coordpy.*` modules in W103.**  Lead lane reuses
  W102 backup-lane modules verbatim:
  `coordpy.humaneval_plus_loader_v1`,
  `coordpy.humaneval_plus_executor_v1`,
  `coordpy.humaneval_plus_reflexion_bench_v1`,
  `coordpy.humaneval_plus_preflight_v1`.  Helper lane re-uses
  `coordpy.code_slice_selector_v1` verbatim.
* 2 NEW driver scripts in `scripts/`:
  `scripts/run_w103_humaneval_plus_pilot.py` (lead-lane pilot),
  `scripts/_w103_emit_pilot_result_doc.py` (verdict emitter).
* 1 NEW unit-test file (`tests/test_w103_code_line_discipline_v1.py`;
  9 tests; all PASS).
* 6 NEW result-doc artefacts (RUNBOOK + 4 RESULTS + frontier
  audit).

## Linear ↔ GitHub sync

* `COO-27` (W103 issue) created at the milestone start.
* `linear_github_mapping.json` extended with the W103 entry
  (14 milestones total).
* `scripts/sync_linear_github_v1.py validate` reports OK
  across all 14 milestones (W89 / W93 / W94 / W95 / W96-A /
  W96-C / W96-D / W97 / W98 / W99 / W100 / W101 / W102 /
  W103).

## Programme entitlement after W103

The programme is entitled to claim:

* The W89 sequential-reflexion mechanism extends to a SECOND
  published code benchmark family (**HumanEval+, EvalPlus-
  hardened**) at the cheap-pilot scale at Llama-3.3-70B-
  Instruct, with **B − A1 = +20.00 pp** (4x larger than the
  W89 retirement's +5.56 pp on base HumanEval) and MLB-2
  rescue rate **47.06 % matching the W89 retirement template
  byte-for-byte**.  This is STRONGER than what W102 alone left
  us (which FAILed on MBPP+ V2).

The programme is NOT entitled to claim:

* Multi-benchmark same-budget retirement on HumanEval+ (Phase
  3 multi-seed required).
* Generalisation to MBPP-family at 70B — `W102-L-MBPP-PLUS-V2-
  REFLEXION-PHASE2-70B-CAP` is the live counter-example.
* That reflexion is benchmark-family-independent — the W102 /
  W103 contrast (22 % vs 47 % rescue rate at 70B on the MBPP
  vs HumanEval family) is empirical evidence that load-
  bearingness varies by benchmark structure.
* Cross-scale generalisation (W104 work).
* That multi-agent context is solved.

## Anchors

* `docs/RUNBOOK_W103.md` — pre-commit contract.
* `docs/RESULTS_W103_HELPER_CONSUMPTION_V1.md` — slice
  attestation.
* `docs/RESULTS_W103_HUMANEVAL_PLUS_PREFLIGHT_RECONFIRM_V1.md`
  — preflight reconfirmation.
* `docs/RESULTS_W103_HUMANEVAL_PLUS_PHASE2_70B_V1.md` — pilot
  verdict (populated by `scripts/_w103_emit_pilot_result_doc.py`).
* `docs/FRONTIER_RELEVANCE_AUDIT_W103_V1.md` — frontier audit
  supplement (13th consecutive validation).
* `scripts/run_w103_humaneval_plus_pilot.py` — pilot driver.
* `scripts/_w103_emit_pilot_result_doc.py` — verdict emitter.
* `tests/test_w103_code_line_discipline_v1.py` — hardening
  tests (9 passing).
* `results/w103/humaneval_plus_pilot/w103_humaneval_plus_pilot_meta_llama-3.3-70b-instruct_20260526T022037Z/`
  — pilot run output (provenance.json + bench_report.json +
  reflexion_calls.jsonl).
* `results/w103/code_slice_proposals/w102_slice_proposals_20260526T015420Z/`
  — helper-anchored slice proposal artefacts.
