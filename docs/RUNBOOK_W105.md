# W105 — HumanEval+ Phase 3 retirement bench + run-hardening + W106 contingency (runbook)

> **Pre-commit contract for W105, locked 2026-05-27 BEFORE any
> W105 NIM call and BEFORE the W105 Phase 3 driver runs.**
>
> W104 closed with the HumanEval+ cross-generation Phase 2 cheap
> pilot `PASS_MECHANISM_DRIVEN` on the byte-equal W103 helper-
> anchored 30-problem slice at the pre-locked backup target
> `meta/llama-3.1-70b-instruct` (B − A1 = +10.00 pp; MLB-1
> 56.67 % + MLB-2 35.29 % both PASS).  The pre-locked primary
> `meta/llama-3.1-405b-instruct` was unreachable on NIM at the
> W104 run window (HTTP 404).  Per **Branch A** of the
> pre-committed W104 RUNBOOK § Planning lane, W105 is entitled
> to launch the HumanEval+ Phase 3 retirement bench against the
> two earned model classes.  `COO-9` REMAINS the lead path.
> `COO-29` is the W105 Linear issue.
>
> W105 is NOT a one-step milestone.  It advances THREE lanes in
> the same milestone:
>
> 1. **Lead lane (Phase 3 execution)** — execute the pre-built
>    W105 Phase 3 slice pack BYTE-FOR-BYTE unchanged at pack
>    CID `8be55f3bf1650df3...`; 3 seeds × 100 problems × K = 5
>    × 2 model classes = 6 600 NIM calls; evaluate per-class
>    Phase 3 retirement bars FIRST, then the cross-class claim.
> 2. **Ops/hardening lane** — durable guardrails that materially
>    matter on a 6 600-call run: canary smoke before the full
>    launch; resume-safe per-(model, seed) cell dispatch;
>    mid-run visibility; automatic partial audit emission;
>    explicit handling for 429 / socket-hang / relaunch events;
>    cross-class comparator that runs per-seed to avoid the
>    W104 V1 row-misalignment.
> 3. **W106 planning lane** — pre-commit W106 under BOTH the
>    clean-retirement branch and the split/FAIL branches so the
>    next milestone is execution, not paperwork.
>
> The full Phase 3 launch is CONDITIONAL on the canary smoke
> passing on both model classes.  The hardening lane and
> planning lane ship UNCONDITIONALLY.
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.  Advanced work remains explicit-import
> only.

## Linear

* New issue **`COO-29`** (W105): HumanEval+ Phase 3 retirement
  bench + run-hardening + W106 contingency.  Parent: `COO-6`.
  High priority.
* Related: `COO-9` (lead path) — remains at High; W105 is the
  retirement evaluation of the W103+W104 cheap-pilot evidence.
* Related: `COO-28` (W104; Done) — cross-generation Phase 2
  evidence W105 retires against.
* Related: `COO-27` (W103; Done) — single-scale Phase 2
  evidence W105 retires against.
* Related: `COO-14` (Done) — `coordpy.code_slice_selector_v1`
  consumed in the W105 slice pack at pack CID
  `8be55f3bf1650df3...` (third real downstream consumption;
  first W103 / second W104 / third W105 retirement evaluation).

## What is NOT in scope (anti-drift contract)

W105 explicitly does NOT:

1. Re-open the cross-modal RealWorldQA arc.  RealWorldQA stays
   frozen at 11B per the W100 frontier audit.  W101 / W102 /
   W103 / W104 carry this verbatim; W105 carries it forward.
2. Re-open the W95-B0 family, the typed-extract sub-family, or
   any RealWorldQA candidate.
3. Re-attempt MBPP+ V2 at 70B or 405B.  W102 FAIL is a cap;
   re-running on a fresh seed or scale would be hope-driven,
   not evidence-earned.
4. Promote `COO-12` (substrate-level cross-modal injection)
   absent fresh evidence; `COO-12` stays Low.
5. Build APPS / LiveCodeBench / SWE-bench-lite infrastructure.
   The W101 battlefield-selection matrix locked these out of
   scope; W105 inherits that decision verbatim.  Branch C
   (W105 FAIL) re-opens the LiveCodeBench preflight option,
   but only as the next preflight step, never as a running
   expensive bench in W105 itself.
6. Bump `coordpy.__version__` or `SDK_VERSION`.
7. Publish to PyPI.
8. Edit `coordpy/__init__.py`.  Any new W105 modules are
   explicit-import only.
9. Re-introduce any anti-pattern under a prettier name
   (bounded windowing; compaction; generic prose
   summarization; shallow token compression; context-pruning
   theater; "cram less / truncate better").  The W97 – W104
   frontier-relevance audits stay in force verbatim.
10. Run a new candidate tournament.  The W101 battlefield-
    selection matrix, the W103 helper-anchored slice (inner
    kernel), and the W105 pre-built Phase 3 pack at pack CID
    `8be55f3bf1650df3...` are ALL carried forward unchanged.
    W105 only changes the *scale* of evaluation (Phase 2 cheap
    pilot → Phase 3 retirement bench).
11. Re-grade historical W88 / W91 / W103 / W104 responses
    against a new test surface and treat the result as Phase 3
    earning evidence.  Per the W102 anti-pattern carry-forward,
    that is an UPPER BOUND only.  Fresh-K = 5 sampling at
    Phase 3 size + multi-seed is the W105 ground truth.
12. Widen the W105 core matrix to include `meta/llama-3.1-405b-instruct`
    UNLESS a deliberate re-lock of this RUNBOOK lands BEFORE any
    405B NIM spend.  A cheap sub-second reachability smoke
    probe on 405B IS allowed (and encouraged for the public
    record) but does NOT change the matrix.
13. Average a class-specific FAIL into a cross-class mean to
    hide it.  The cross-class claim is entitled IFF BOTH
    classes clear all 6 retirement bars on their own.

## Operational state (cheap evidence in hand BEFORE W105 starts)

| Field | Value |
|---|---|
| `coordpy.__version__` | `0.5.20` |
| `coordpy.SDK_VERSION` | `coordpy.sdk.v3.43` |
| W103 70B Phase 2 verdict | **`PASS_MECHANISM_DRIVEN`**; B − A1 = +20.00 pp; MLB-2 = 47.06 % |
| W104 70B Phase 2 (Llama-3.1) verdict | **`PASS_MECHANISM_DRIVEN`**; B − A1 = +10.00 pp; MLB-2 = 35.29 % |
| W105 pre-built Phase 3 slice pack CID | `8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314` |
| W105 pre-built Phase 3 n_problems | 100 |
| W105 pre-built Phase 3 seeds | (105 001, 105 002, 105 003) |
| W105 pre-built Phase 3 W103 inner kernel | 30 problems at the head of the helper-priority order |
| HumanEval+ corpus SHA-256 (LFS oid) | `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492` |
| W103 / W104 preflight verdict cid | `4f57a2cf60ae6a1bbecf15a3ae6e0a9d68a1f9f52d07abb1eb7c2de72e25f7a4` (re-confirmed twice; W105 reuses verbatim) |

## Critical W105 anti-pattern carry-forwards from W101 / W102 / W103 / W104

1. **`coordpy.mbpp_plus_loader_v1` is an anti-pattern**, not a
   loader.  W105 never touches it.
2. **Cross-bench arsenal-mining priors are an UPPER BOUND**, not
   a Phase 3 earning signal.
3. **MLB-2 rescue rate varies by benchmark family AND by model
   class.**  W102 MBPP+ V2 produced MLB-2 = 22.22 % at 70B; W103
   HumanEval+ at Llama-3.3-70B produced 47.06 %; W104 HumanEval+
   at Llama-3.1-70B produced 35.29 %.  W105 records ALL THREE
   as priors; the Phase 3 verdict reads against the per-class
   retirement bars on fresh K = 5 multi-seed sampling.
4. **Cross-scale collapse pattern from W96-A / W96-C / W100.**
   Three out of three cross-modal cross-scale 11B → 90B
   confirmations showed margin shifts of −5 to −10 pp from the
   smaller-scale Phase 2 result.  The cross-class shift seen at
   W104 (+20 → +10 pp; −10 pp) on the SAME parameter scale is
   evidence that cross-scale / cross-class shifts are real on
   the code line too.  W105 records this as a structural prior
   to take seriously.
5. **Sidecar mid-run visibility + resume-from-sidecar.**  W102
   buffered all sidecar entries until exit; W103 added per-write
   flush; W104 added resume-from-disk.  W105 inherits ALL of
   these AND extends them with **per-(model, seed) cell
   isolation** + **per-cell partial audit emission** + **global
   progress.json** so a kill-and-restart on hour 14 of a 21-hour
   run does NOT lose any prior cell's evidence.
6. **Cross-scale comparator V1 row-alignment lesson (W104).**
   The W104 V1 comparator iterates per-(problem-position)
   assuming the same iteration order at both scales.  In W104
   the two scales used different seeds (103001 vs 104001) so
   the per-seed shuffle differed and the per-problem cluster-
   shift labels were arithmetically mis-aligned (aggregate
   stats stayed correct).  W105 cross-class comparator V1
   runs **per-(matched seed)** so the bench's internal shuffle
   is identical on both sides of each comparison pair.

## Lead lane — HumanEval+ Phase 3 retirement bench (CONDITIONAL on canary)

### Decision logic (pre-locked BEFORE driver runs)

1. **Phase 3 driver builds + unit-tested + Linear-synced**
   regardless of canary outcome.  The driver mirrors the W104
   driver shape, with three W105 additions:
   (a) accepts a `--slice-pack` argument with the W105 pre-
   committed value defaulting to the slice_pack.json at pack
   CID `8be55f3bf1650df3...`;
   (b) iterates per-(model_class, seed) cell with per-cell
   isolation (own out_dir, own sidecar, own provenance);
   (c) emits the cross-class comparator block per-(matched seed)
   on completion of each `(class_a_seed_N, class_b_seed_N)`
   pair (avoids the W104 row-alignment failure mode).
2. **Canary smoke** (66 NIM calls): 1 seed (canary-only seed
   105 999) × 3 problems × K = 5 × 2 model classes.  3
   problems are picked deterministically from the slice pack's
   first 3 helper-priority entries (rescue-surface concentrated
   `b_only_wins`).  Canary serves THREE purposes:
   (a) reachability re-confirm on both model classes (W104's
   primary 405B was unreachable; reachability is not assumed);
   (b) per-call budget envelope sanity (no API breakage / no
   prompt-format regression);
   (c) cheap detection of a structural regression in the W102
   HumanEval+ bench module since W104's run.
3. **Canary acceptance bar**: B − A1 ≥ −5 pp per class.  This
   is INTENTIONALLY LOOSE — the canary is reachability +
   budget-envelope sanity, NOT a Phase 3 gate.  3 problems
   is too small to constrain Phase 3 statistics; the canary
   only fails on STRUCTURAL breakage (sampling all-fails, all
   identical outputs, executor degeneration).  A FAIL canary
   PAUSES the full launch and triggers a structural diagnosis.
4. **Pack CID verification at run start** (NEW W105 contract):
   the driver recomputes the SHA-256 of the slice pack's
   ordered task_ids list and refuses to run on mismatch.  The
   driver also verifies the inner-kernel CID matches the W103
   helper-priority CID `c35155956ece605c...`.
5. **Corpus SHA verification at run start** (NEW W105 contract):
   matches W103 + W104 verbatim; refuses to run on mismatch.
6. **Per-cell partial audit emission**: after each cell
   completes, the driver writes a `phase3_cell_verdict.json`
   into the cell's out_dir with per-cell Phase 2-shape gates
   (the cheap-pilot 9 gates + MLB sub-gates) computed at
   100-problem cell size.  This is a partial audit, NOT the
   Phase 3 verdict.
7. **Phase 3 retirement evaluator runs at end of all 6 cells**
   (or against partial data if at least one cell per class
   completed) via `coordpy.phase3_retirement_evaluator_v1`.
   Emits a per-class verdict + cross-class entitlement verdict.

### Phase 3 retirement bars (W88 / W89 / W95 6-bar shape; locked)

The 6 retirement bars are evaluated PER CLASS on the 3 (seed)
cells for that class.  Cross-class claim is layered on top
ONLY after both per-class verdicts.

| # | Bar | Threshold |
|---|---|---|
| 1 | Margin (mean across seeds within class) | mean B − A1 ≥ +5 pp |
| 2 | Per-seed majority (within class) | B > A1 on ≥ 2 / 3 seeds |
| 3 | Per-problem majority (averaged across seeds within class) | B ≥ A1 on ≥ 53 % of 100 problems × 3 seeds = ≥ 159 / 300 problem-seed cells |
| 4 | A1 not saturated (per (seed, class) cell) | A1 @ K = 5 < 90 % on each (seed, class) cell |
| 5 | Audit chain re-derives | ≥ 13 / 14 per-cell Merkle + provenance + sidecar re-derive PASS (per the W95 Phase 3 audit-coverage bar) |
| 6 | Executor stays clean | canonical-solution self-test 100 % on each of 100 problems (post-run) |

**Per-class verdict labels** (locked):

* `RETIRED` — all 6 bars PASS for that class.
* `RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING` — bars 1-6 all PASS
  BUT MLB-2 rescue rate < 33 % on that class (the W104 lesson:
  margin can clear while mechanism load-bearingness erodes).
* `FAIL_<reason>` — at least one bar FAILs.

**Cross-class claim entitlement** (locked):

* Cross-class retirement claim entitled IFF:
  (i) both per-class verdicts are `RETIRED`;
  AND
  (ii) cross-class mean `B − A1` difference within ± 5 pp
       (no cross-class collapse beyond the W104 cross-class
       shift envelope).
* If one class is `RETIRED` and the other is `FAIL_<reason>`,
  the claim is bounded to the retired class with the FAIL
  carry-forward registered for the other.
* If both classes FAIL, the claim is NOT entitled; W106
  applies the W104 RUNBOOK § Branch C dispatch table.

### Anti-cheat (verbatim from W88 – W104)

* Slice = pack CID `8be55f3bf1650df3...` unchanged; slice CID
  verified at run start.
* Same model on every arm within a cell.
* Same K = 5 byte-exact budget on A1 / B (sequential reflexion
  runs the FULL K = 5 budget; no early-stop).
* Executor = `coordpy.humaneval_plus_executor_v1.run_humaneval_plus_executor_v1`.
  No LLM judge; subprocess CPython.
* Corpus SHA-256-anchored at run start; mismatches refuse to
  run.
* No selective retries; each (seed, problem, arm) is exactly
  one set of calls.
* Per-call sidecars + per-seed Merkle + bench Merkle re-derive
  offline.

## Hardening lane — run-hardening (UNCONDITIONAL)

W102 + W103 + W104 surfaced run-discipline guardrails; W105
codifies the W105-specific ones the 6 600-call envelope
demands:

1. **Per-(model, seed) cell isolation** — each cell has its
   own out_dir + sidecar + provenance.  A failure in cell
   (class A, seed 105 001) cannot lose the evidence already
   captured in cell (class A, seed 105 002).

2. **Canary before full launch** — 1 seed × 3 problems × K = 5
   × 2 classes = 66 NIM calls.  Validates reachability + budget
   envelope + executor cleanness on both classes BEFORE the
   6 600-call envelope opens.  Canary uses seed 105 999 (not
   in the main matrix; results NOT mixed into Phase 3 evidence).

3. **Mid-run progress visibility** — global `progress.json` at
   the run root, updated after each cell's completion.  Tail-
   friendly stdout log.  Per-cell `progress.json` updated after
   each problem.

4. **Automatic per-cell partial audit** — after each cell
   completes, emit `phase3_cell_verdict.json` with Phase-2-shape
   gates at 100-problem cell size.  This is NOT the Phase 3
   verdict but lets a human eyeball cell health while the run
   is still in flight.

5. **Resume-safe per-(model, seed) dispatch** — if a cell is
   killed mid-run, restarting the driver against the same
   out-root detects which cells are complete (have
   `phase3_cell_verdict.json`) and skips them.  Within a
   partial cell, the W104 sidecar-resume continues to apply.

6. **429 / socket-hang / relaunch handling** — inherits the
   W104 retry-with-exponential-backoff + sidecar flush + corrupt
   trailing-line conservative-treatment behaviour; adds a
   `retry_log.jsonl` per cell that records every 429 / 5xx /
   timeout / socket-hang event for post-mortem.

7. **Cross-class comparator V1 — per-seed alignment** —
   `coordpy.cross_class_comparator_v1` (NEW W105 module; explicit-
   import only; not in `coordpy/__init__.py`).  Iterates
   per-(matched seed) so each comparison pair has IDENTICAL
   bench-internal shuffle order on both sides.  Refuses to run
   on slice / corpus / schema / cell-count mismatch.  Emits the
   per-seed cross-class shift on B − A1 + cluster shift counts.

### Hardening-lane deliverable (locked)

* `coordpy/phase3_retirement_evaluator_v1.py` — NEW Phase 3
  retirement evaluator (single new module; explicit-import only;
  not in `coordpy/__init__.py`).  ≤ 350 lines; pure-Python; no
  NIM.
* `coordpy/cross_class_comparator_v1.py` — NEW per-seed cross-
  class comparator (single new module; explicit-import only).
  ≤ 300 lines; pure-Python; no NIM.
* `scripts/run_w105_phase3_retirement_bench.py` — Phase 3 driver
  (per-cell isolation; canary; resume; mid-run visibility).
* `scripts/run_w105_canary_smoke.py` — canary entrypoint (66
  NIM calls; thin wrapper on the driver with `--canary`).
* `scripts/run_w105_405b_reachability_probe.py` — cheap 405B
  smoke (sub-second; NO main-run impact; records reachability
  for the public record).
* `tests/test_w105_phase3_discipline_v1.py` — new test file;
  ≥ 14 tests covering (a) pack-CID-mismatch refuse-to-run,
  (b) corpus-SHA-mismatch refuse-to-run, (c) per-seed shuffle
  reproducibility, (d) resume-safe per-cell skipping, (e)
  evaluator PASS path on synthetic 6 cells, (f) evaluator
  per-bar FAIL paths (one per bar), (g) cross-class entitlement
  ONLY on BOTH-PASS, (h) cross-class comparator per-seed
  alignment, (i) cross-class comparator refuse-to-run paths
  (slice / corpus / schema / cell-count).

## Planning lane — W106 pre-commit (UNCONDITIONAL)

The W106 next step is pre-committed in this milestone so the
boundary doesn't drift on outcome.  W106 is *execution* (NOT
paperwork) under all branches.

### Branch A — W105 cross-class RETIRED on both classes

W106 = theorem-registration milestone + optional 405B
extension + publication-strength consolidation.  No new
expensive bench at retirement scale; W106's job is to
register the retirement honestly and (if 405B is reachable by
W106's start) attempt the cross-scale-UP extension.

* W106 lead lane = theorem-registration in `docs/THEOREM_REGISTRY.md`
  (W89 retirement + W105 HumanEval+ retirement; bench-family
  generalisation claim bounded to two model classes at
  parameter scale 70B).
* W106 hardening lane = publication-strength consolidation
  (README badge update if any; CHANGELOG entry; consolidated
  cross-W89-W103-W104-W105 narrative doc).
* W106 planning lane = 405B reachability probe + (if reachable)
  Phase 2 cheap pilot pre-commit at 405B on the W105 slice pack
  inner kernel (30-problem head).

### Branch B — W105 one class RETIRED, one class FAIL

W106 = bounded-claim milestone + class-specific carry-forward
+ next code-line move per the W104 RUNBOOK § Branch C dispatch
keyed to the FAIL class's failure mode.

* If retired class = Llama-3.3-70B, failed class = Llama-3.1-70B:
  the W89 mechanism is bounded to the Llama-3.3 family at 70B.
  Carry-forward `W105-L-HUMANEVAL-PLUS-RETIREMENT-LLAMA31-70B-CAP`.
* If retired class = Llama-3.1-70B, failed class = Llama-3.3-70B:
  STRUCTURALLY SURPRISING (the W89 retirement was on Llama-3.3
  base-HumanEval; a Phase 3 FAIL there would refute the W103
  +20 pp cheap-pilot result on per-seed sampling).  Carry-forward
  `W105-L-HUMANEVAL-PLUS-PER-SEED-SAMPLING-VARIANCE-AT-LLAMA33-70B-CAP`.

### Branch C — W105 BOTH classes FAIL

W106 = Branch C dispatch table from the W104 RUNBOOK § Planning
lane, keyed to the worst-class failure mode (margin / per-seed
majority / A1 saturation / MLB collapse).  W106 builds NIM-free
preflight first on whichever of LiveCodeBench / APPS / 70B-only
Phase 3 the table selects.

### Pre-committed W106 dispatch JSON

```json
{
  "schema": "coordpy.w105_w106_dispatch.v1",
  "rules": [
    {
      "condition": {
        "per_class_verdict_llama33_70b": "RETIRED",
        "per_class_verdict_llama31_70b": "RETIRED",
        "cross_class_b_minus_a1_diff_pp_abs_le": 5.0
      },
      "next_step": "w106_theorem_registration_plus_optional_405b_extension",
      "estimated_nim_calls": 0
    },
    {
      "condition": {
        "per_class_verdict_llama33_70b": "RETIRED",
        "per_class_verdict_llama31_70b_starts_with": "FAIL"
      },
      "next_step": "w106_bounded_claim_llama33_70b_plus_w104_branch_c_dispatch_for_llama31_failmode",
      "estimated_nim_calls": "branch_c_table_entry"
    },
    {
      "condition": {
        "per_class_verdict_llama33_70b_starts_with": "FAIL",
        "per_class_verdict_llama31_70b": "RETIRED"
      },
      "next_step": "w106_per_seed_sampling_variance_audit_at_llama33_70b_plus_bounded_claim_llama31_70b",
      "estimated_nim_calls": "990_to_3300"
    },
    {
      "condition": {
        "per_class_verdict_llama33_70b_starts_with": "FAIL",
        "per_class_verdict_llama31_70b_starts_with": "FAIL"
      },
      "next_step": "w106_w104_branch_c_dispatch_keyed_to_worst_class_failmode",
      "estimated_nim_calls": "branch_c_table_entry"
    }
  ],
  "fallback": "code_line_ranking_refresh_by_w101_matrix"
}
```

The W106 driver that consumes this dispatch is built in W106
IF the W105 verdict triggers a branch that requires execution;
pre-committing the dispatch JSON here ensures the W106 driver
is execution-ready under any verdict.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New W105 explicit-import-only `coordpy.*` modules:
  * `coordpy.phase3_retirement_evaluator_v1` — per-class +
    cross-class retirement evaluator.
  * `coordpy.cross_class_comparator_v1` — per-seed-aligned
    cross-class comparator.
* New W105 artefacts:
  * `docs/RUNBOOK_W105.md` (this file).
  * `scripts/run_w105_phase3_retirement_bench.py` (Phase 3
    driver).
  * `scripts/run_w105_canary_smoke.py` (canary entrypoint).
  * `scripts/run_w105_405b_reachability_probe.py` (cheap 405B
    smoke).
  * `tests/test_w105_phase3_discipline_v1.py` (hardening
    tests).
  * `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA33_V1.md`
    (per-class Phase 3 verdict; populated post-run).
  * `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA31_V1.md`
    (per-class Phase 3 verdict; populated post-run).
  * `docs/RESULTS_W105_CROSS_CLASS_COMPARATOR_V1.md` (cross-
    class comparator narrative; populated post-run).
  * `docs/RESULTS_W105_W106_PLANNING_V1.md` (W106 dispatch +
    claim scaffolding).
  * `docs/RESULTS_W105_MILESTONE_SUMMARY_V1.md` (milestone
    summary; populated at close).
  * `docs/FRONTIER_RELEVANCE_AUDIT_W105_V1.md` (frontier audit
    supplement; 15th preflight-discipline validation).

## Phase 3 retirement-grade evaluation discipline (locked)

Per the W93 – W104 retirement-evaluation discipline:

1. **Evaluate each model class separately first** — the W105
   per-class verdict is the load-bearing surface.  Cross-class
   claims layer on top.
2. **Do NOT hide failure at one class by averaging it into the
   other.** — if one class clears and one does not, the claim
   is per-class only.
3. **Only make the stronger "across two model classes" claim
   if both classes clear ALL bars** AND the cross-class B − A1
   gap is within the W104-observed envelope (± 5 pp on the
   matched-slice cross-class shift).
4. **MLB reporting stays load-bearing at Phase 3 scale** — the
   evaluator emits per-class MLB-1 + MLB-2 alongside the 6
   retirement bars.  A class can `RETIRE` on the 6 bars but
   the verdict label downgrades to
   `RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING` if MLB-2 < 33 %
   (mirrors the W96-C / W100 precedent for Phase 3).

## Operational plan

### Phase 1 — done in W105 (NO NIM)

1. **(W105 hardening lane)** —
   `coordpy/phase3_retirement_evaluator_v1.py` (≤ 350 lines;
   pure-Python; explicit-import only) +
   `coordpy/cross_class_comparator_v1.py` (≤ 300 lines;
   pure-Python; explicit-import only).
2. **(W105 hardening tests)** —
   `tests/test_w105_phase3_discipline_v1.py`; ≥ 14 tests; all
   PASS.
3. **(W105 lead-lane driver)** —
   `scripts/run_w105_phase3_retirement_bench.py`; per-cell
   isolation; canary; resume; mid-run visibility.
4. **(W105 canary entrypoint)** —
   `scripts/run_w105_canary_smoke.py`.
5. **(W105 cheap 405B reachability probe)** —
   `scripts/run_w105_405b_reachability_probe.py`.
6. **(W105 W106 planning artifact)** —
   `docs/RESULTS_W105_W106_PLANNING_V1.md`; documents the
   pre-committed Branch A / B / C dispatch under every
   per-class verdict shape.
7. **(W105 frontier-relevance audit supplement)** —
   `docs/FRONTIER_RELEVANCE_AUDIT_W105_V1.md`; 15th consecutive
   preflight-discipline validation.
8. **(Linear ↔ GitHub sync)** — create `COO-29` (done; this
   RUNBOOK lock is the W105 charter); append a `W105` entry to
   `linear_github_mapping.json` post-launch.

### Phase 2 — conditional on canary PASS (66 NIM calls)

1. **Launch canary smoke** on both model classes:
   ```bash
   NVIDIA_API_KEY=... python scripts/run_w105_canary_smoke.py
   ```
2. **Evaluate canary acceptance bar** (B − A1 ≥ −5 pp per
   class).  PASS ⇒ Phase 3.  FAIL ⇒ structural diagnosis;
   Phase 3 paused; no carry-forward retired.

### Phase 3 — conditional on canary PASS (6 600 NIM calls)

1. **Launch HumanEval+ Phase 3 retirement bench**:
   ```bash
   NVIDIA_API_KEY=... python scripts/run_w105_phase3_retirement_bench.py \
       --slice-pack data/w105/phase3_slice_pack/w105_phase3_slice_pack_20260526T215647Z/slice_pack.json
   ```
2. **Per-cell partial audit emitted automatically** after each
   cell.
3. **End-of-run Phase 3 retirement evaluator** runs
   automatically (or against partial data if at least one cell
   per class completed).  Verdict goes in
   `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA33_V1.md` +
   `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA31_V1.md`.
4. **Cross-class comparator** runs automatically after all 6
   cells complete (or partial after at least one matched seed
   pair).  Doc:
   `docs/RESULTS_W105_CROSS_CLASS_COMPARATOR_V1.md`.
5. **Apply pre-locked decision branch** (A / B / C above) and
   record in `docs/RESULTS_W105_MILESTONE_SUMMARY_V1.md`.

### Phase 4 — optional cheap 405B reachability probe (sub-second NIM)

1. Independent of the main run, attempt a cheap reachability
   probe on `meta/llama-3.1-405b-instruct`:
   ```bash
   NVIDIA_API_KEY=... python scripts/run_w105_405b_reachability_probe.py
   ```
2. Result recorded in
   `docs/RESULTS_W105_MILESTONE_SUMMARY_V1.md`.  Either outcome
   does NOT change the W105 core run.

## Pre-launch prediction (recorded 2026-05-27 BEFORE W105 Phase 3 launches)

> "Subjective priors over the W105 HumanEval+ Phase 3 retirement
> bench on the pre-built 100-problem slice pack with K = 5 at 3
> seeds across two earned Llama-3.x 70B model classes,
> conditional on canary PASS:
>
> * Probability A1 @ K = 5 clears the saturation gate (< 90 %)
>   on each (seed, class) cell: **~ 92 %** (the slice has 25 %
>   broad corpus-fill which should keep mean A1 well below the
>   helper-anchored slice's A1; predicted slice A1 ≈ 60-70 % at
>   the helper-priority head + ≈ 75-85 % on the corpus-fill tail;
>   mean ≈ 65-80 % per cell).
> * Probability per-class Llama-3.3-70B retires (all 6 bars +
>   MLB-2 ≥ 33 %): **~ 50-60 %** (W103 +20 pp cheap pilot at the
>   same model class on the inner kernel is the strongest prior;
>   the 70-problem mid-shell + corpus-fill extension dilutes the
>   margin somewhat by adding shared_wins; cross-seed sampling
>   variance can move the mean ± 5 pp at 100 problems).
> * Probability per-class Llama-3.1-70B retires: **~ 35-45 %**
>   (W104 +10 pp cheap pilot at this class on the inner kernel
>   is the prior; lower than Llama-3.3 because the cheap-pilot
>   margin is half; the +5 pp Phase 3 bar is tight; cross-seed
>   variance more likely to push at least one cell across the
>   per-seed-majority bar).
> * Probability cross-class retirement entitled (both classes
>   RETIRED + cross-class shift within ± 5 pp): **~ 25-35 %**
>   (compound probability; the cross-class gap at cheap pilot
>   was 10 pp — pushing the margin envelope; Phase 3 mean
>   could easily land outside the ± 5 pp band).
> * Probability at least one class retires: **~ 65-75 %**
>   (independent of cross-class entitlement; the Llama-3.3-70B
>   class is the more likely retirement winner).
> * Probability NO class retires: **~ 25-35 %** (per-seed
>   sampling variance + 100-problem mean shift can land both
>   classes at margin < +5 pp even when the underlying
>   mechanism is real).
>
> If W105 cross-class retires cleanly, the programme is
> entitled to the **second confirmed multi-seed same-budget
> multi-agent superiority retirement after W89** — bounded to
> HumanEval+ on Llama-3.x-70B model classes.  THAT IS NOT
> "multi-agent context solved"; THAT IS NOT cross-scale-UP
> generalisation (405B still unreachable); THAT IS NOT MBPP-
> family generalisation (W102 cap stands); THAT IS NOT cross-
> modal generalisation (RealWorldQA frozen at 11B).  It IS a
> structurally honest second confirmed retirement on a
> different benchmark family from the same mechanism, at
> Phase 3 multi-seed quality at the same parameter scale
> across two model classes.
>
> If W105 retires one class only, the bounded claim is
> 'W89 retirement extends to HumanEval+ on Llama-3.x-70B-Instruct
> AT THE RETIRED MODEL CLASS' — a real but narrower advance.
>
> If W105 FAILs both classes, the W103 + W104 cheap-pilot
> PASSes stay as anchors but no Phase 3 retirement is earned;
> W106 applies Branch C dispatch."

## Honest framing

W105's job is to:

1. **Honestly execute the pre-built slice pack** byte-for-byte.
   No re-derivation of the slice mid-run.
2. **Honestly evaluate each model class separately first**.
   Cross-class claims layer on top of per-class verdicts.
3. **Honestly ship hardening** that addresses the 6 600-call
   envelope's real failure modes (mid-run kill / 429 storm /
   model-specific reachability shift).
4. **Honestly pre-commit W106** under every per-class verdict
   shape so milestone boundaries don't drift on result.
5. **Launch Phase 3 ONLY IF the canary PASSes**.  No buying long
   runs from hope.

If W105 retires cleanly across both classes, the programme is
entitled to the *stronger* claim that the W89 reflexion
mechanism retires on HumanEval+ at Phase 3 multi-seed scale
across TWO Llama-3.x-70B model classes.  Retirement-grade
generalisation across MORE classes / scales / families
requires further milestones.  W105 is NOT a multi-benchmark
retirement (HumanEval is W89, HumanEval+ is W105; MBPP+ V2
remains capped at W102; cross-modal remains frozen at W100;
SWE-bench-lite stays unconditionally out of scope).

If W105 retires one class only, the bounded claim is `W89
retirement extends to HumanEval+ on Llama-3.x-70B-Instruct AT
THE RETIRED MODEL CLASS`; the failed class registers a carry-
forward that bounds the cross-class claim.

If W105 FAILs both classes, the W103 + W104 cheap-pilot
PASSes stay as anchors but no Phase 3 retirement is earned;
W106 applies the pre-committed Branch C dispatch.  Either
outcome preserves the W93 – W104 preflight-first + cross-scale
+ multi-candidate-tournament-then-confirm + mechanism-load-
bearingness + silent-degradation-anti-pattern-guard + arsenal-
mining-prior-anti-pattern-guard + cross-class-row-alignment-
discipline as the 15th consecutive validation.
