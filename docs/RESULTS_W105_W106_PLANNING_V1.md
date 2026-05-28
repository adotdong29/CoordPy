# W105 — W106 planning artifact V1 (claim scaffolding + fallback)

> **2026-05-27.  Pre-builds the W106 next step under all four
> shapes the W105 Phase 3 retirement verdict can take.  Built
> BEFORE the W105 Phase 3 verdict so EITHER outcome leaves W106
> as execution, not paperwork.**

## Why pre-build W106 now

The W93 – W104 discipline thread says: pre-commit milestone
boundaries BEFORE empirical evidence arrives.  W104 retired with
the cross-generation `PASS_MECHANISM_DRIVEN` outcome on the
byte-equal W103 slice; the W104 RUNBOOK pre-committed W105 as the
HumanEval+ Phase 3 retirement bench AND pre-committed the
Branch C fallback dispatch.  Both lanes shipped in the same
milestone.

W105 in turn pre-commits W106 under all four verdict shapes the
W105 Phase 3 retirement verdict can take:

* **Verdict A** — both per-class verdicts `RETIRED` AND cross-
  class B − A1 within ± 5 pp envelope → cross-class retirement
  ENTITLED.  W106 = theorem registration + optional 405B
  reachability extension + publication-grade consolidation.
* **Verdict B** — both per-class verdicts `RETIRED` BUT cross-
  class B − A1 outside ± 5 pp envelope → bounded cross-class
  retirement (each class retired separately but the cross-class
  shift is too wide to claim "across two model classes").  W106
  = per-class theorem registration + cross-class-shift audit.
* **Verdict C** — exactly one per-class verdict `RETIRED`, the
  other `FAIL_<reason>` → bounded single-class claim.  W106 =
  per-class theorem registration on the retired class + the
  W104 RUNBOOK § Branch C dispatch keyed to the failed class's
  failure mode.
* **Verdict D** — both per-class verdicts `FAIL_<reason>` →
  no Phase 3 retirement.  W106 = W104 RUNBOOK § Branch C
  dispatch keyed to the WORST failure mode across the two
  classes.  The W103 + W104 cheap-pilot PASSes remain anchors.

Plus a fifth latent shape:

* **Verdict E** — `RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING` on at
  least one class (margin clears all 6 bars but MLB-2 < 33 %
  on that class) → downgraded claim layer.  W106 = MLB-2 audit
  + mechanism-variation pilot on the affected class.

## Verdict A — cross-class RETIRED

### W106 lead lane: theorem registration

* Register a new W105 theorem in `docs/THEOREM_REGISTRY.md`:
  > "W89 sequential-reflexion retires on HumanEval+ at Phase 3
  > multi-seed scale across TWO Llama-3.x-70B-Instruct model
  > classes (Llama-3.3-70B + Llama-3.1-70B), 3 seeds × 100
  > problems × K = 5; same-budget byte-exact; mechanism load-
  > bearing (mean MLB-2 ≥ 33 %); cross-class B − A1 shift within
  > the ± 5 pp envelope."
* Add to `docs/RESEARCH_STATUS.md` headline.
* Add to `docs/HOW_NOT_TO_OVERSTATE.md`: explicit non-claims
  ("does NOT solve multi-agent context; does NOT imply 405B
  generalisation; does NOT imply MBPP+ V2 retirement; does NOT
  imply cross-modal retirement").

### W106 hardening lane: publication-grade consolidation

* Consolidated cross-W89 + W103 + W104 + W105 narrative doc
  (`docs/CONSOLIDATED_CODE_RETIREMENT_NARRATIVE_V1.md`).
* CHANGELOG entry summarising the W89 → W105 retirement arc.

### W106 planning lane: 405B reachability extension

* Run `scripts/run_w106_405b_reachability_probe.py` (W106 will
  build this; reuses W105 405B probe shape).  Three possible
  outcomes:
  * **405B reachable on NIM** → W106 Phase 2 cheap pilot at 405B
    on the W105 slice pack 30-problem inner kernel (matches the
    W104 cheap-pilot shape; cheap-pilot earning rule applies).
  * **405B still unreachable** → W106 cross-scale-UP attempt
    deferred to next NIM-budget allocation; `W105-L-RETIREMENT-
    BOUND-TO-70B-CLASS-CAP` carry-forward registers the
    structural gap.
  * **405B reachable but with a STRUCTURALLY DIFFERENT API
    surface** (different prompt format / different output
    constraints) → record the new constraints + decide whether
    the W105 mechanism survives at 405B at all.

## Verdict B — both RETIRED but cross-class shift outside envelope

* Register per-class theorems separately:
  > "W89 retires on HumanEval+ at Phase 3 multi-seed on Llama-
  > 3.3-70B-Instruct (3 seeds × 100 problems × K = 5; same-
  > budget byte-exact)."
  >
  > "W89 retires on HumanEval+ at Phase 3 multi-seed on Llama-
  > 3.1-70B-Instruct (3 seeds × 100 problems × K = 5; same-
  > budget byte-exact)."
* DO NOT claim cross-class retirement; the envelope rule blocks
  the stronger claim.
* W106 = cross-class-shift audit (per-seed cluster-shift mining
  + per-(model class, problem) failure-pattern analysis) to
  understand WHY the cross-class shift exceeded the envelope.

## Verdict C — one RETIRED, one FAIL

### Sub-case C1: Llama-3.3-70B RETIRED + Llama-3.1-70B FAIL

* Register single-class theorem on Llama-3.3-70B (matches the
  W89 base-HumanEval retirement model class).
* Add carry-forward `W105-L-HUMANEVAL-PLUS-RETIREMENT-LLAMA31-70B-CAP`.
* W106 = W104 RUNBOOK § Branch C dispatch keyed to the Llama-
  3.1 failure mode:
  * Margin < 0 + MLB-2 < 33 % → LiveCodeBench preflight (NIM-
    free).
  * G2 saturation → APPS preflight (NIM-free).
  * Margin < +5 pp but ≥ 0 + MLB-2 ≥ 33 % → multi-seed cheap
    confirmation on Llama-3.1-70B (~ 990 NIM calls at 70B
    rate).
  * Margin < 0 + MLB-2 ≥ 33 % + G2 < 90 % → cross-class-
    collapse audit + 3-seed Phase 3-shape confirmation at
    Llama-3.3-70B ONLY (~ 3 300 NIM calls).

### Sub-case C2: Llama-3.1-70B RETIRED + Llama-3.3-70B FAIL

* STRUCTURALLY SURPRISING — would refute the W103 +20 pp cheap
  pilot at Llama-3.3-70B on per-seed sampling.
* Register single-class theorem on Llama-3.1-70B.
* Add carry-forward `W105-L-HUMANEVAL-PLUS-PER-SEED-SAMPLING-
  VARIANCE-AT-LLAMA33-70B-CAP`.
* W106 = per-seed sampling variance audit at Llama-3.3-70B (3
  seeds × 30-problem cheap pilot on the W103 slice inner kernel;
  ~ 990 NIM calls).  If the per-seed variance audit fails to
  recover the W103 result, the W103 PASS is downgraded.

## Verdict D — both FAIL

* No Phase 3 retirement.  The W103 + W104 cheap-pilot PASSes
  stay as 1-scale-cheap-pilot anchors.
* W106 = W104 RUNBOOK § Branch C dispatch keyed to the WORST
  failure mode across the two classes.
* Anti-drift check: do NOT reopen MBPP+ V2.  Do NOT reopen
  cross-modal RealWorldQA arc.  Do NOT bounded-window /
  compaction / prose-summarize.

## Verdict E — RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING

* At least one class clears all 6 bars BUT MLB-2 < 33 % on
  that class.  The W96-C / W100 precedent applies.
* DO NOT claim retirement on that class.
* Add carry-forward
  `W105-L-HUMANEVAL-PLUS-RETIREMENT-MLB2-WEAK-ON-{CLASS}-CAP`.
* W106 = mechanism-variation pilot on the affected class
  (parallel B-style variants matching the W104 RUNBOOK §
  Branch B template; cheap-pilot budget 330 NIM calls per
  variant).

## Branch dispatch JSON (machine-readable)

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
      "estimated_nim_calls_for_405b_extension": 330
    },
    {
      "condition": {
        "per_class_verdict_llama33_70b": "RETIRED",
        "per_class_verdict_llama31_70b": "RETIRED",
        "cross_class_b_minus_a1_diff_pp_abs_gt": 5.0
      },
      "next_step": "w106_per_class_theorems_plus_cross_class_shift_audit",
      "estimated_nim_calls": 0
    },
    {
      "condition": {
        "per_class_verdict_llama33_70b": "RETIRED",
        "per_class_verdict_llama31_70b_starts_with": "FAIL"
      },
      "next_step": "w106_bounded_claim_llama33_70b_plus_w104_branch_c_dispatch_for_llama31_failmode",
      "estimated_nim_calls_branch_c_table_entry": true
    },
    {
      "condition": {
        "per_class_verdict_llama33_70b_starts_with": "FAIL",
        "per_class_verdict_llama31_70b": "RETIRED"
      },
      "next_step": "w106_per_seed_sampling_variance_audit_at_llama33_70b_plus_bounded_claim_llama31_70b",
      "estimated_nim_calls": 990
    },
    {
      "condition": {
        "per_class_verdict_llama33_70b_starts_with": "FAIL",
        "per_class_verdict_llama31_70b_starts_with": "FAIL"
      },
      "next_step": "w106_w104_branch_c_dispatch_keyed_to_worst_class_failmode",
      "estimated_nim_calls_branch_c_table_entry": true
    },
    {
      "condition": {
        "any_class_verdict": "RETIRED_MARGIN_DRIVEN_NON_LOAD_BEARING"
      },
      "next_step": "w106_mechanism_variation_pilot_on_affected_class",
      "estimated_nim_calls_per_variant": 330
    }
  ],
  "fallback": "code_line_ranking_refresh_by_w101_matrix"
}
```

## W104 § Branch C dispatch (carried forward verbatim for Verdict C / D fallbacks)

| FAIL mode (worst-class) | W106 lead step | NIM-spend ceiling |
|---|---|---|
| Margin < 0 AND MLB-2 < 33 % | LiveCodeBench preflight (NIM-free) | $0 |
| G2 saturation (A1 ≥ 90 %) | APPS preflight (NIM-free) | $0 |
| Margin < +5 pp but ≥ 0 AND MLB-2 ≥ 33 % | multi-seed cheap confirmation at affected class | ~ 990 NIM calls |
| Margin < 0 AND MLB-2 ≥ 33 % AND G2 < 90 % | cross-class-collapse audit + 3-seed Phase 3-shape confirmation at the retired class ONLY | ~ 3 300 NIM calls |
| SWE-bench-lite | OUT OF SCOPE | — |

## Honest framing

* This artifact is **planning**, not evidence.  No claim earned
  here.  All five verdict shapes have execution-ready W106
  scaffolding so the W105 verdict triggers execution, not
  paperwork.
* The cross-class envelope (± 5 pp) is the same envelope that
  governed the W104 cross-scale collapse risk model.  W104
  cross-generation shift was -10 pp on the 30-problem cheap
  pilot; if Phase 3's larger sample shrinks the variance, the
  cross-class envelope could narrow under the rule.  Either way
  the bar is locked here.
* SWE-bench-lite stays unconditionally out of scope under every
  W106 branch.  RealWorldQA stays frozen at 11B.  MBPP+ V2 stays
  capped.  No bounded / compaction / summary anti-patterns.

## Anchors

* `docs/RUNBOOK_W105.md` — pre-commit contract.
* `docs/RUNBOOK_W104.md` — Branch C dispatch table (W106 fallback
  surface).
* `coordpy/phase3_retirement_evaluator_v1.py` — per-class +
  cross-class evaluator that produces the per-class verdict labels
  consumed by this dispatch.
* `coordpy/cross_class_comparator_v1.py` — per-seed-aligned cross-
  class comparator (consumed by the Verdict B audit).
* `data/w105/phase3_slice_pack/w105_phase3_slice_pack_20260526T215647Z/slice_pack.json` —
  locked W105 Phase 3 slice pack.
