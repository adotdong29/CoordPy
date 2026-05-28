# W106 — bounded second-retirement registration (V1)

> **2026-05-28.  The strongest claim the W105 evidence licenses,
> registered cleanly and aggressively — but honestly.  No new run;
> this is registration of already-earned truth.**

## The registered claim (strongest the evidence licenses)

> **The W89 sequential-reflexion mechanism RETIRES on HumanEval+
> (EvalPlus-hardened) at Phase 3 multi-seed scale on
> `meta/llama-3.3-70b-instruct`** — 3 seeds (105 001 / 105 002 /
> 105 003) × 100 problems × K = 5; same-budget byte-exact;
> mean B − A1 = **+7.00 pp** (per-cell +5.00 / +9.00 / +7.00);
> per-seed majority 3/3; per-problem 295/300; A1 84/82/82 % (all
> < 90 %); audit chain 3/3; executor clean 100 %; **MLB-2 =
> 55.62 % load-bearing**.  This is the **SECOND confirmed
> multi-seed same-budget multi-agent superiority retirement after
> W89**, on a DIFFERENT benchmark family (W89 = base HumanEval at
> +5.56 pp; W105 = HumanEval+ at +7.00 pp).

This is real, and it is aggressive: the programme now has TWO
independent confirmed retirements of the same mechanism on two
different benchmark families, both at Phase 3 multi-seed
retirement quality, both with the mechanism demonstrably
load-bearing (MLB-2 47 % on W89 / W103 → 55.62 % at W105 Phase 3).

## The boundedness (impossible to miss, by design)

The claim is bounded on **three axes simultaneously**:

| Axis | Bound | Cap |
|---|---|---|
| Model class | ONE: `meta/llama-3.3-70b-instruct` | `meta/llama-3.1-70b-instruct` FAILed Phase 3 (+2.33 pp) → `W105-L-...-LLAMA31-70B-MARGIN-CAP` + `W106-L-...-CHEAP-CONFIRMATION-NOT-EARNED-CAP` |
| Benchmark family | ONE: HumanEval+ (plus base HumanEval at W89) | MBPP+ V2 FAILed at 70B → `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP` |
| Parameter scale | ONE: 70B | 405B unreachable on NIM (HTTP 404) → `W104-L-...-405B-UNREACHABLE-ON-NIM-CAP` |

And it does NOT touch:

* **Cross-class generalisation** — NOT entitled; the W105 RUNBOOK
  rule requires BOTH classes to clear all 6 bars; only one did.
* **Cross-scale-UP generalisation** — 405B blocked.
* **Cross-modal generalisation** — RealWorldQA frozen at 11B
  (W100); MathVista margin-capped (W95/W96-A); ChartQA
  preflight-saturated (W96-D).
* **"Multi-agent context solved"** — NOT established; the
  canonical surface is `docs/HONEST_FRAMING_POST_W87.md`.

## What W106 changed vs W105

W106 changed the **registration** and the **dispatch decision**,
NOT the **evidence**.  The empirical evidence is the W105 Phase 3
bench (6 600 NIM calls, already spent and audited).  W106:

1. Encodes the bounded second retirement as a registered claim
   (`W106-T-BOUNDED-SECOND-RETIREMENT-REGISTERED`) in the theorem
   registry, the research-status banner, and the
   do-not-overstate rules — with the boundedness front-and-centre.
2. Closes the Llama-3.1 margin-cap branch NO-GO (see
   `docs/RESULTS_W106_MARGIN_CAP_DISPATCH_V1.md`).

The programme is entitled to a STRONGER claim than before W105 (a
SECOND confirmed retirement now exists) but NOT a stronger claim
than W105 itself produced (still single-class; still no
cross-class / cross-scale-UP / multi-benchmark same-budget
retirement).

## The two confirmed retirements (canonical list)

1. **W89** — base HumanEval × `meta/llama-3.3-70b-instruct` × K=5;
   B − A1 = +5.56 pp; 2/3 seeds; all retirement bars met; audit
   7/7.  (`W89-T-HE-REFLEXION-70B-BEATS-A1`.)
2. **W105** — HumanEval+ (EvalPlus-hardened) ×
   `meta/llama-3.3-70b-instruct` × K=5; mean B − A1 = +7.00 pp;
   3/3 seeds; 6/6 bars; MLB-2 55.62 %.
   (`W105-T-HUMANEVAL-PLUS-RETIREMENT-LLAMA33-70B`; registered by
   `W106-T-BOUNDED-SECOND-RETIREMENT-REGISTERED`.)

Both on the SAME model class at the SAME parameter scale.  That is
the honest extent of the same-budget multi-agent superiority
retirement evidence as of W106.

## Anchors

* `docs/THEOREM_REGISTRY.md` § W106 + canonical claims table.
* `docs/RESEARCH_STATUS.md` — banner.
* `docs/HOW_NOT_TO_OVERSTATE.md` § W106 + meta-rule bullet.
* `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA33_V1.md` — the
  RETIRED evidence.
* `docs/RESULTS_W106_MARGIN_CAP_DISPATCH_V1.md` — the Llama-3.1
  NO-GO decision.
* `docs/RUNBOOK_W106.md` § 1 — the locked registration target.
