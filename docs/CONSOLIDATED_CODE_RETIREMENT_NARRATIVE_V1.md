# Consolidated code-retirement narrative (V1) — W89 → W106

> **2026-05-28 (W107 Lane γ).  Publication-grade consolidation of the
> same-budget multi-agent-superiority code-retirement arc.  This
> document is the single defensible narrative tying together the TWO
> confirmed retirements and the bounded claim structure around them.
> It spends $0 NIM — it registers exactly what W89 → W106 already
> earned, with the boundedness made impossible to miss.  Where this
> doc and any other disagree on the STATUS of a claim,
> `docs/THEOREM_REGISTRY.md` is authoritative.**

## The one-paragraph claim

The CoordPy programme has **two confirmed multi-seed same-budget
multi-agent-superiority retirements**, both of the W89
sequential-reflexion mechanism, both on `meta/llama-3.3-70b-instruct`
at the 70B parameter scale: **W89** on base HumanEval (B − A1 =
+5.56 pp) and **W105** on EvalPlus-hardened HumanEval+ (mean B − A1 =
+7.00 pp at 3 seeds × 100 problems × K=5; 6/6 retirement bars; MLB-2 =
55.62 % load-bearing).  This is genuine cross-benchmark-family
generalisation of one mechanism at retirement quality.  It is bounded
on three axes simultaneously — **one model class, one-plus benchmark
family, one parameter scale** — and it does NOT establish cross-class,
cross-scale-UP, MBPP-family, cross-modal, or "multi-agent context
solved".

## What "retirement" means here (the bar, stated once)

A carry-forward is *retired* only when the W89 sequential-reflexion
B-pipeline beats the strong same-budget single-agent baseline A1 on a
**multi-seed Phase 3 bench** clearing ALL of:

1. **G2** A1 not saturated (A1@K=5 < 90 %) — there is real headroom.
2. **G3** B > A1 (strict).
3. **G4** margin (B − A1) ≥ +5 pp.
4. **G5** (B − A0) ≥ +5 pp — the team also beats the single-shot floor.
5. **G6** per-problem majority (B ≥ A1 on the majority of problems).
6. Per-seed majority (B > A1 on the majority of seeds).
7. Budget byte-exact (B and A1 spend the identical K-budget).
8. Audit chain re-derives offline; executor clean (no LLM judge).
9. **Mechanism load-bearing**: MLB-2 rescue rate ≥ 33 % — the
   reflexion step is *causally* responsible for the wins, not variance.

A single-seed or rescue-concentrated +X pp is an **upper bound**, never
a retirement (the W102 anti-pattern, confirmed repeatedly below).

## The arc, milestone by milestone

| W## | Battlefield × model | Scale | Result | B − A1 | Mechanism (MLB-2) | Status |
|---|---|---|---|---|---|---|
| **W89** | base HumanEval × Llama-3.3-70B | Phase 3 (multi-seed K=5) | **RETIRED** | **+5.56 pp** | load-bearing (~47 %) | the FIRST confirmed retirement |
| W102 | MBPP+ V2 × Llama-3.3-70B | Phase 2 cheap pilot | **FAIL** | −6.67 pp | weak (22.22 %) | `W102-L-MBPP-PLUS-V2-...-CAP` |
| W103 | HumanEval+ × Llama-3.3-70B | Phase 2 cheap pilot | PASS_MECH | +20.00 pp | load-bearing (47.06 %) | earns cross-scale step |
| W104 | HumanEval+ × Llama-3.1-70B | Phase 2 cheap pilot (cross-GEN) | PASS_MECH | +10.00 pp | load-bearing (35.29 %) | earns Phase 3 bench |
| **W105** | HumanEval+ × Llama-3.3-70B | **Phase 3 (3×100×K5)** | **RETIRED** | **+7.00 pp** | **load-bearing (55.62 %)** | the SECOND confirmed retirement |
| W105 | HumanEval+ × Llama-3.1-70B | Phase 3 (3×100×K5) | **FAIL_MARGIN** | +2.33 pp | load-bearing (50.54 %) | `W105-L-...-LLAMA31-70B-MARGIN-CAP` |
| W106 | (registration + dispatch) | — | NO-GO | — | — | bounded claim REGISTERED; Llama-3.1 branch CLOSED |

### W89 — the first retirement (the template)

The W89 wave V2 retired the same-budget multi-agent-superiority
carry-forwards at 70B on base HumanEval: B 91.1 % > A1 85.6 % by
**+5.56 pp**, B beats A1 on 2/3 seeds, all retirement bars met, audit
7/7 PASS.  This is the empirical template every later attack is graded
against (`W89-T-HE-REFLEXION-70B-BEATS-A1`).

### W102 — the negative that disciplined the rest

MBPP+ V2 FAILed at 70B (B − A1 = −6.67 pp; MLB-1 30 %, MLB-2 22.22 %).
The decisive structural lesson: **reflexion load-bearingness is
benchmark-family-dependent** (HumanEval-family ~35–47 % rescue vs
MBPP-family 22 %), and **re-grading historical responses against a new
test surface is an upper bound, not earning evidence**.  This is the
anti-pattern that later vetoes the W106 cheap confirmation.

### W103 → W104 — the cheap-pilot chain (two model classes)

HumanEval+ cleared Phase 2 at +20.00 pp on Llama-3.3-70B (W103;
MLB-2 47.06 %, byte-for-byte the W89 rescue rate) and at +10.00 pp on
Llama-3.1-70B (W104; cross-GENERATION, not cross-scale-UP, because the
405B primary was already HTTP 404).  Two model classes cleared the
cheap-pilot bar — but a cheap-pilot margin is an upper bound, so a
multi-seed Phase 3 bench was required to retire.

### W105 — the second retirement, and the split

The earned Phase 3 bench (6 600 NIM calls; pre-built pack CID
`8be55f3bf1650df3…`; 3 seeds × 100 problems × K=5 × 2 classes) closed
**SPLIT**:

* **`meta/llama-3.3-70b-instruct` RETIRED** — mean B − A1 = +7.00 pp
  (per-cell +5/+9/+7); 6/6 bars; per-seed 3/3; per-problem 295/300;
  A1 84/82/82 % (all < 90 %); MLB-2 = 55.62 % load-bearing.  The
  SECOND confirmed retirement, on a DIFFERENT benchmark family from
  W89 (`W105-T-HUMANEVAL-PLUS-RETIREMENT-LLAMA33-70B`).
* **`meta/llama-3.1-70b-instruct` FAIL_MARGIN** — mean B − A1 =
  +2.33 pp (per-cell +5/+1/+1); 5/6 bars; only the margin bar fails;
  MLB-2 = 50.54 % still load-bearing.  The W104 cheap-pilot +10 pp
  (on a 30-problem rescue-concentrated slice) did **not survive** the
  broad 100-problem slice — A1 rose to 86.33 %, compressing reflexion
  headroom.  The W102 upper-bound anti-pattern + the W96-A/W96-C/W100
  cross-scale-collapse pattern, confirmed on the code line.

Cross-class retirement is **NOT entitled**: the entitlement rule
requires BOTH classes to clear all 6 bars; only one did
(`W105-L-...-CROSS-CLASS-RETIREMENT-NOT-ENTITLED-CAP`).

### W106 — registration + the disciplined NO-GO

W106 spent NO new NIM.  It (a) REGISTERED the W105 Llama-3.3-70B
retirement at full strength as the bounded second retirement
(`W106-T-BOUNDED-SECOND-RETIREMENT-REGISTERED`), and (b) decided the
Llama-3.1 `FAIL_MARGIN` branch with a pre-committed two-gate rule
(`coordpy.margin_cap_dispatch_v1`): the Branch-C table ENTITLED a
~990-call cheap confirmation, but the verdict-changing-power gate
VETOED it — a rescue-concentrated slice is an upper bound (W102), the
authoritative fair broad-slice Phase 3 verdict already ran (+2.33 pp),
and the miss is a clean true magnitude miss with no confound.
**NO-GO; $0 NIM** (`W106-L-...-CHEAP-CONFIRMATION-NOT-EARNED-CAP`).
The branch is CLOSED, not deferred — it can only re-open via a
genuinely different battlefield, never a rescue-concentrated re-run.

## The boundedness (the part that must never be dropped)

| Axis | What IS established | What is NOT, and why |
|---|---|---|
| **Model class** | Llama-3.3-70B (W89 + W105) | NOT cross-class — Llama-3.1-70B FAIL_MARGIN at +2.33 pp; W106 closed that branch NO-GO |
| **Benchmark family** | base HumanEval (W89) + HumanEval+ (W105) | NOT MBPP-family — W102 FAIL caps it; reflexion is family-dependent at 70B |
| **Parameter scale** | 70B | NOT cross-scale-UP — `meta/llama-3.1-405b-instruct` HTTP 404 on NIM at W104/W105/W106/**W107** (four consecutive) |
| **Modality** | text-only code | NOT cross-modal — RealWorldQA frozen at 11B (W100); MathVista margin-capped; ChartQA preflight-saturated |
| **Meta-claim** | two real retirements | NOT "multi-agent context solved" — see `docs/HONEST_FRAMING_POST_W87.md` |

## Why the programme is entitled to a stronger claim than before W105 — but only just

Before W105 there was ONE confirmed retirement (W89).  After W105/W106
there are TWO (W89 + W105), on two benchmark families, both with the
mechanism demonstrably load-bearing (MLB-2 47 % → 55.62 %).  That is a
real strengthening: cross-benchmark-family generalisation of one
mechanism at retirement quality.  It is NOT a cross-class /
cross-scale-UP / multi-benchmark-same-budget retirement, and the
programme has repeatedly, deliberately declined to manufacture one
(W105 cross-class not entitled; W106 margin-cap NO-GO; W107-α 405B
gate closed at the fourth 404).

## The two honest paths to a genuinely stronger claim (and their status)

1. **Cross-scale-UP** (single-class 70B → 405B): the genuine
   strengthening axis.  BLOCKED — `meta/llama-3.1-405b-instruct` is
   HTTP 404 on NIM at four consecutive probes (W104–W107).  Re-opens
   only if 405B becomes reachable.
2. **A third code-benchmark family** under preflight-first discipline:
   the live path.  W107-β preflighted **LiveCodeBench** (primary;
   time-anchored contamination resistance is the decisive
   publication-grade property) with **APPS** as the structural-pivot
   backup; the functional-subset executor is clean and the W89
   decomposition fits, so W108 is a cheap pilot (after operator
   corpus-fetch) or an honest no-go — not paperwork.

## Post-W106 update — the third-code-family path was executed (W108 + W109)

Path #2 above (a third code-benchmark family) has now been RUN, and it
delivered a contamination boundary rather than a third retirement:

* **W108 — LiveCodeBench (contamination-RESISTANT 2025)**: the first test of
  the W89 mechanism on contamination-resistant data **FAILed** (B − A1 =
  −3.33 pp; MLB-2 = 25 %).
* **W109 — APPS (contamination-EXPOSED 2021), as a control**: the SAME
  mechanism **RECOVERED** a large same-budget win (B − A1 = +16.67 pp; 9/9
  gates; MLB-2 = 57 %; 0 regressions; `PASS_NON_MECHANISM_DRIVEN`), reinforced
  by an A0 single-shot gap (73.33 % exposed vs 43.33 % resistant).

This **double dissociation by vintage** is evidence CONSISTENT with a
**contamination-confound**: the same-budget reflexion advantage replicates on
contamination-EXPOSED code (APPS, like the W89/W105 HumanEval-family) but not
on contamination-RESISTANT code. The confound is now **SUPPORTED but NOT
established** (one single-seed control pair; the APPS PASS is
non-mechanism-driven on invocation; APPS is contamination-EXPOSED ⇒ control
evidence only, never a third retirement). Net effect on this narrative: the
two confirmed retirements (W89, W105) **STAND**, but they are now explicitly
bounded as **contamination-EXPOSED HumanEval-family**, and a contamination-
RESISTANT same-budget code superiority remains **unproven** (the only attempt
FAILed). See `docs/CONTAMINATION_CONTROL_FRAMING_W109_V1.md`.

## How to cite this honestly (for any external write-up)

* DO say: "two confirmed same-budget multi-agent-superiority
  retirements of the W89 reflexion mechanism, both at 70B on
  Llama-3.3, on base HumanEval (+5.56 pp) and HumanEval+ (+7.00 pp,
  multi-seed, mechanism load-bearing)."
* DO say: "bounded to one model class / one-plus benchmark family /
  one parameter scale; not cross-class, not cross-scale-UP, not
  MBPP-family, not cross-modal."
* DO NOT say: "multi-agent teams beat single agents" without the
  bounds; "the mechanism generalises across model classes" (Llama-3.1
  FAILed); "across scales" (405B unreachable); "across code
  benchmarks broadly" (MBPP+ FAILed); or "multi-agent context is
  solved".

## Anchors

* `docs/THEOREM_REGISTRY.md` — authoritative claim status (W89, W105,
  W106, W107 sections).
* `docs/RESEARCH_STATUS.md` — canonical current truth + banner.
* `docs/HOW_NOT_TO_OVERSTATE.md` — the do-not-overstate rules.
* `docs/HONEST_FRAMING_POST_W87.md` — what is shipped vs not.
* `docs/RESULTS_W106_BOUNDED_RETIREMENT_REGISTRATION_V1.md` — the W105
  registration.
* `docs/RESULTS_W107_405B_GATE_V1.md` + `docs/RESULTS_W107_NEXT_BATTLEFIELD_PREFLIGHT_V1.md`
  — the W107 gate + β preflight.
