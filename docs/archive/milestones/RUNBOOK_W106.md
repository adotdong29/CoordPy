# W106 — bounded-retirement registration + Llama-3.1 margin-cap dispatch + graphify truth-sync (runbook)

> **Pre-commit contract for W106, locked 2026-05-28 BEFORE any
> W106 NIM call (including the optional cheap 405B reachability
> side-probe) and BEFORE the W106 margin-cap dispatch driver
> runs.**
>
> W105 (`COO-29`; Done) closed **SPLIT**: `meta/llama-3.3-70b-instruct`
> **RETIRED** the W89 sequential-reflexion mechanism on HumanEval+
> at Phase 3 multi-seed scale (3 seeds × 100 problems × K = 5;
> mean B − A1 = **+7.00 pp**; 6/6 bars; MLB-2 = 55.62 %
> load-bearing) — the SECOND confirmed multi-seed same-budget
> multi-agent superiority retirement after W89, on a different
> benchmark family.  `meta/llama-3.1-70b-instruct` **FAIL_MARGIN**
> (mean B − A1 = **+2.33 pp**; 5/6 bars; only the margin bar
> fails; MLB-2 = 50.54 % still load-bearing).  Cross-class
> retirement **NOT entitled**.
>
> Per the pre-committed `docs/RESULTS_W105_W106_PLANNING_V1.md`
> **Verdict C / sub-case C1** (= `docs/RUNBOOK_W105.md` § Planning
> lane **Branch B**, retired class = Llama-3.3-70B, failed class =
> Llama-3.1-70B), W106 is a **coupled claim-registration +
> margin-cap-dispatch milestone**.  It is NOT a new benchmark
> tournament.  `COO-9` REMAINS the lead path.  `COO-30` is the
> W106 Linear issue.
>
> W106 is NOT a one-step milestone.  It advances THREE lanes in
> the same milestone:
>
> 1. **Claim / theorem lane** — register the W105 result cleanly
>    and aggressively, but honestly: the strongest claim the
>    evidence licenses, with the boundedness impossible to miss.
> 2. **Llama-3.1 margin-cap dispatch lane** — decide the
>    Llama-3.1 branch honestly via a pre-committed two-gate
>    decision rule (entitlement gate + verdict-changing-power
>    gate).  Run a cheap confirmation ONLY if it is honestly
>    earned; otherwise accept the bounded single-class claim and
>    stop — no NIM spend bought from emotional temptation.
> 3. **Graphify / truth-sync lane** — graphify is part of the
>    operating system for this repo now: refresh at start, use
>    `query` / `affected` / `explain` / `path` to find affected
>    truth surfaces, refresh again before close.
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`; `coordpy/__init__.py` untouched.  Any new
> W106 module is explicit-import only.

## Linear

* New issue **`COO-30`** (W106): bounded-retirement registration
  + Llama-3.1 margin-cap dispatch + graphify truth-sync.  Parent:
  `COO-6`.  High priority.
* Related: `COO-9` (lead path) — remains at High; W106 registers
  the retirement evaluation it earned at W103 → W104 → W105.
* Related: `COO-29` (W105; Done) — the empirical Phase 3 verdict
  W106 registers.
* Parent `COO-6` backlog refreshed with the W104 → W105 → W106
  arc (its description + comment thread were stale at W98 / W103).

## What is NOT in scope (anti-drift contract, carried verbatim from W105 + sharpened)

W106 explicitly does NOT:

1. Run a new benchmark-family tournament.  The W101 battlefield-
   selection matrix + the W103 helper-anchored slice + the W105
   pre-built Phase 3 pack are all carried forward unchanged.
2. Re-open MBPP+ V2.  `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`
   stands; re-running on a fresh seed or scale would be hope-
   driven, not evidence-earned.
3. Re-open the frozen cross-modal lines.  RealWorldQA stays
   frozen at 11B per the W100 frontier audit; MathVista stays
   margin-capped; ChartQA stays preflight-saturated.
4. Run a 405B EXPENSIVE bench.  A cheap sub-second 405B
   reachability SIDE-PROBE is allowed (and refreshes the public
   record) but does NOT change the W106 matrix and must not
   distract from the main objective.
5. Bump `coordpy.__version__` or `SDK_VERSION`.
6. Publish to PyPI.
7. Edit `coordpy/__init__.py`.  Any new W106 module is
   explicit-import only.
8. Re-introduce any anti-pattern under a prettier name (bounded
   windowing; compaction; generic prose summarization; shallow
   token compression; context-pruning theater; "cram less /
   truncate better").  The W97 – W105 frontier-relevance audits
   stay in force verbatim.
9. **Over-average the Llama-3.3 retirement across classes to
   manufacture a cross-class claim, OR strengthen the bounded
   single-class claim into a cross-class claim absent new
   earned evidence.**  The Llama-3.3-70B retirement is
   already-earned truth and is registered at full strength; the
   Llama-3.1-70B result stays bounded and is repeated as bounded.
10. **Buy a cheap Llama-3.1 confirmation just because the result
    is emotionally tempting.**  The only honest extra spend is a
    sharply targeted cheap confirmation that can change the
    retirement verdict; if the structural argument is absent, no
    run.

## Operational state (cheap evidence in hand BEFORE W106 starts)

| Field | Value |
|---|---|
| `coordpy.__version__` | `0.5.20` |
| `coordpy.SDK_VERSION` | `coordpy.sdk.v3.43` |
| W105 Llama-3.3-70B Phase 3 verdict | **`RETIRED`** (6/6 bars; mean B − A1 = +7.00 pp; per-cell +5/+9/+7; A1 84/82/82 %; MLB-2 = 55.62 %) |
| W105 Llama-3.1-70B Phase 3 verdict | **`FAIL_MARGIN`** (5/6 bars; mean B − A1 = +2.33 pp; per-cell +5/+1/+1; A1 85/87/87 %; MLB-2 = 50.54 %) |
| W105 cross-class entitlement | **NOT ENTITLED** (`CLASS_B_RETIRED_CLASS_A_FAIL_BOUNDED_CLAIM`) |
| W105 Phase 3 slice pack CID | `8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314` |
| W105 consolidated verdict json | `results/w105/humaneval_plus_phase3_retirement_bench/w105_phase3_FINAL_consolidated/phase3_retirement_verdict.json` |
| HumanEval+ corpus SHA-256 (LFS oid) | `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492` |
| 405B reachability (W105 Phase 4 probe) | HTTP 404 — unreachable on NIM (re-probed 2026-05-27) |
| graphify graph build commit (W106 start) | `ae54d00` (73 026 nodes / 237 304 edges / 2 336 communities) |

## 1. Strongest-claim registration target (LOCKED)

The single registration target W106 must encode — verbatim — into
the canonical truth surfaces:

> **The W89 sequential-reflexion mechanism RETIRES on HumanEval+
> (EvalPlus-hardened) at Phase 3 multi-seed scale on
> `meta/llama-3.3-70b-instruct` (3 seeds × 100 problems × K = 5;
> same-budget byte-exact; mean B − A1 = +7.00 pp; 6/6 retirement
> bars; per-seed majority 3/3; per-problem 295/300; MLB-2 =
> 55.62 % load-bearing).  This is the SECOND confirmed multi-seed
> same-budget multi-agent superiority retirement after W89, on a
> DIFFERENT benchmark family (W89 = base HumanEval at +5.56 pp;
> W105 = HumanEval+ at +7.00 pp).  It is bounded to ONE model
> class (Llama-3.3-70B) on ONE benchmark family (HumanEval+) at
> ONE parameter scale (70B).**

The boundedness that MUST travel with the claim, everywhere it
appears:

* **NOT cross-class.**  `meta/llama-3.1-70b-instruct` FAILed the
  +5 pp Phase 3 margin bar at +2.33 pp (`W105-L-HUMANEVAL-PLUS-
  RETIREMENT-LLAMA31-70B-MARGIN-CAP` + `W105-L-HUMANEVAL-PLUS-
  CROSS-CLASS-RETIREMENT-NOT-ENTITLED-CAP`).  W106 closes this
  branch (see § 2) — it does NOT promote it.
* **NOT cross-scale-UP.**  405B remains unreachable on NIM (HTTP
  404).  `W104-L-HUMANEVAL-PLUS-CROSS-SCALE-UP-PRIMARY-TARGET-
  405B-UNREACHABLE-ON-NIM-CAP` stands.
* **NOT MBPP-family.**  `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`
  stands.
* **NOT cross-modal.**  RealWorldQA frozen at 11B (W100).
* **NOT "multi-agent context solved".**

W106 updates: `docs/THEOREM_REGISTRY.md` (new W106 registration
section + banner), `docs/RESEARCH_STATUS.md` (banner), and
`docs/HOW_NOT_TO_OVERSTATE.md` (new W106 do-not-overstate section
+ meta-rule bullet).  The two W105-T / W105-L entries already in
the registry are the empirical anchors; W106 adds the
registration milestone + the margin-cap dispatch verdict on top.

## 2. Llama-3.1 margin-cap dispatch decision rule (LOCKED, two-gate)

The pre-committed `docs/RESULTS_W105_W106_PLANNING_V1.md` Verdict
C / sub-case C1 routes the Llama-3.1 failure mode through the
W104 RUNBOOK § Branch C dispatch table.  That table ENTITLES a
next step but does NOT mandate it.  W106 adds the margin-cap
verdict-changing-power gate on top.  A cheap confirmation is RUN
**iff BOTH gates pass**:

### GATE 1 — Entitlement (pre-committed Branch C table lookup)

Classify the affected class's W105 failure signature and look up
the entitled next step:

| Failure signature | Entitled next step | NIM ceiling |
|---|---|---|
| margin < 0 AND MLB-2 < 33 % | LiveCodeBench preflight (NIM-free) | $0 |
| A1 ≥ 90 % (G2 saturation) | APPS preflight (NIM-free) | $0 |
| **0 ≤ margin < +5 pp AND MLB-2 ≥ 33 % AND A1 < 90 %** | **multi-seed cheap confirmation at affected class** | **~990 NIM calls** |
| margin < 0 AND MLB-2 ≥ 33 % AND A1 < 90 % | cross-class-collapse audit + 3-seed Phase 3-shape confirmation at the RETIRED class only | ~3 300 NIM calls |

**Llama-3.1 W105 signature**: margin = +2.33 pp ∈ [0, +5);
MLB-2 = 50.54 % ≥ 33 %; A1 = 86.33 % < 90 % (no G2 saturation).
→ **matched row 3** → GATE 1 = **ENTITLED** (cheap confirmation,
~990 NIM calls).

### GATE 2 — Verdict-changing power (the W106 margin-cap discipline)

Entitlement is necessary, not sufficient.  The cheap confirmation
is run only if it can HONESTLY convert the fair-slice
`FAIL_MARGIN` into a retirement-grade `RETIRED`.  Requires **ALL
THREE** sub-conditions:

* **2a — FAIR BATTLEFIELD.**  The proposed confirmation slice
  must be a fair retirement battlefield (representative of the
  broad corpus distribution), NOT selected to concentrate
  A1-failure problems where B can rescue.  A rescue-concentrated
  slice inflates B − A1 by construction and is an UPPER BOUND
  (the W102 anti-pattern), not a retirement battlefield.
* **2b — NO AUTHORITATIVE FAIR RESULT ALREADY EXISTS.**  There
  must NOT already be an authoritative fair broad-slice
  multi-seed Phase 3 result for this class.  If the fair
  retirement bench already ran, a cheaper re-run cannot overturn
  it; retirement is defined on the fair broad slice.
* **2c — FIXABLE CONFOUND.**  The FAIL must be plausibly
  attributable to a fixable confound (parser/executor bug,
  budget mismatch, slice-pack mismatch, reachability/sampling
  artifact) rather than a true magnitude miss.  Clean executor +
  byte-exact budget + matched slice CID + per-seed-positive +
  healthy MLB-2 = a true magnitude miss = nothing to fix.

**Decision**: GO iff GATE 1 = ENTITLED **AND** (2a ∧ 2b ∧ 2c).
Otherwise **NO-GO** → accept the bounded single-class claim,
register the margin-cap close carry-forward, spend $0.

The decision is computed mechanically by
`coordpy.margin_cap_dispatch_v1` from the W105 consolidated
`phase3_retirement_verdict.json` (no NIM); the module refuses to
run on schema / class-count / verdict-label mismatch.

### Pre-applied verdict for the Llama-3.1 branch (recorded BEFORE any NIM call)

* GATE 1: **ENTITLED** (matched row 3).
* GATE 2a: **FAIL** — the only confirmation form Verdict C / C1
  offers is a *rescue-concentrated* slice; that is an upper
  bound, not a fair retirement battlefield.  A non-rescue-
  concentrated cheap slice would merely re-measure ~+2.33 pp and
  add no information.
* GATE 2b: **FAIL** — the fair broad 100-problem × 3-seed Phase 3
  bench ALREADY RAN at W105 and returned +2.33 pp `FAIL_MARGIN`.
  That is the authoritative retirement verdict for this class.
* GATE 2c: **FAIL** — no confound: executor clean 100 %, budget
  byte-exact, slice-pack CID matched at run start, per-seed
  majority 3/3 (directionally positive everywhere), MLB-2 =
  50.54 % healthy.  It is a TRUE magnitude miss.

GATE 2 fails on all three sub-conditions ⇒ **W106 VERDICT =
NO-GO**.  Accept the bounded single-class claim on Llama-3.3-70B.
No Llama-3.1 cheap confirmation is launched.  $0 NIM spend on the
Llama-3.1 branch.

## 3. Rescue-concentrated-slice construction rule (LOCKED; only used if § 2 returned GO)

> **This rule is recorded for completeness and falsifiability.
> Because § 2 returned NO-GO, it is NOT exercised in W106.**

IF (counterfactually) a cheap confirmation were earned, the
rescue-concentrated slice would be built deterministically from
the W105 per-problem evidence (NOT hand-picked):

1. From the W105 Llama-3.1 `per_problem` arrays across the 3
   seeds, select the problems where A1 @ K = 5 fails on the
   majority of seeds (the rescue surface).
2. Take the top-N (N = 30) by A1-failure frequency, tie-broken by
   the W103 helper-priority order, to match the W104 cheap-pilot
   inner-kernel shape.
3. Freeze the slice CID + the construction rule + the seed list
   (3 seeds; ~990 NIM calls = 3 × 30 × K = 5 × 2 arms +
   overhead) in this runbook BEFORE the run.
4. The result would be explicitly labelled an **UPPER-BOUND
   probe**, NOT a retirement bench — it could only sharpen the
   bounded claim's mechanism story, never convert FAIL_MARGIN to
   RETIRED (which is why § 2 vetoes it).

## 4. Cheap-confirmation gates (LOCKED; only used if § 2 returned GO)

> **NOT exercised in W106 (NO-GO).  Recorded for falsifiability.**

The cheap confirmation, if run, would be evaluated against the
W103/W104 Phase 2 9-gate shape PLUS the explicit
upper-bound-honesty gate:

* The 9 Phase 2 gates (byte-identical to W103/W104).
* MLB-1 ≥ 33 % invocation; MLB-2 ≥ 33 % rescue.
* **Upper-bound-honesty gate**: the result is recorded as an
  upper bound and CANNOT by itself retire the class; only a fair
  broad-slice multi-seed Phase 3 bench can, and W105 already ran
  that and returned FAIL_MARGIN.  Therefore even a +X pp cheap
  confirmation PASS would leave the per-class verdict at
  `FAIL_MARGIN` / bounded.

This gate is why the cheap confirmation is structurally incapable
of changing the verdict, and therefore why § 2 returns NO-GO.

## 5. No-go rule (LOCKED — the realized W106 path)

When § 2 returns NO-GO (the realized W106 outcome):

1. Accept the bounded single-class claim: HumanEval+ Phase 3
   retirement on `meta/llama-3.3-70b-instruct` ONLY.
2. Register a NEW carry-forward closing the Llama-3.1 branch:
   `W106-L-HUMANEVAL-PLUS-LLAMA31-70B-MARGIN-CAP-CHEAP-CONFIRMATION-NOT-EARNED-CAP`
   — the Llama-3.1 margin-cap miss is a TRUE fair-slice magnitude
   miss (not a confound, not mechanism collapse); a rescue-
   concentrated cheap confirmation is an upper bound and cannot
   convert it to a retirement, so no NIM spend is honestly
   earned.  This carry-forward CLOSES (does not merely defer) the
   W105 Llama-3.1 dispatch branch.
3. Register the dispatch mechanism as mechanically-checked:
   `W106-T-MARGIN-CAP-DISPATCH-V1-SHIPS`.
4. $0 NIM spend on the Llama-3.1 branch.
5. `COO-9` stays the lead path; the bounded retirement is its
   strongest registered claim.

## 6. Graphify deliverables (LOCKED)

* **Initial refresh** (DONE at W106 start): `graphify update .`
  rebuilt the graph from current HEAD `ae54d00` (73 026 nodes /
  237 304 edges / 2 336 communities); dated backup written to
  `graphify-out/2026-05-28/`.
* **Concrete usage during the milestone**: `graphify query`
  (HumanEval+ retirement-claim provenance + doc truth surfaces),
  `graphify affected` (reverse deps of
  `phase3_retirement_evaluator_v1.py` → sibling error-class
  pattern with `cross_scale_comparator_v1`), `graphify explain`
  (`cross_class_comparator_v1` contained functions + rationale
  edge — the sibling template the new dispatch module mirrors).
  Finding: the new module sits as a SIBLING of the W104/W105
  comparators/evaluators (explicit-import-only; refuse-to-run
  error class), and the doc-layer truth surfaces are navigated
  directly (semantic extraction needs an LLM key not set here).
* **End-of-milestone refresh** (REQUIRED before close): re-run
  `graphify update .` after the material code/doc edits so
  `graphify-out/` matches W106 repo truth; record the new build
  commit; keep the dated `graphify-out/YYYY-MM-DD/` backup
  pattern consistent.

## 7. W107 branch logic (LOCKED — make the next milestone execution, not paperwork)

W106 closes with: the bounded second retirement REGISTERED;
the Llama-3.1 margin-cap branch CLOSED (NO-GO); `COO-9` lead.
The only honest ways to make the HumanEval+ retirement claim
STRONGER than bounded-single-class are (a) cross-scale-UP (405B)
or (b) a new code-benchmark family under preflight-first
discipline.  W107 is pre-committed under the 405B reachability
gate:

* **W107-α (405B reachable on NIM)** — W107 = HumanEval+ Phase 2
  cheap pilot at `meta/llama-3.1-405b-instruct` on the W105 slice
  pack 30-problem inner kernel (cross-scale-UP; matches the W104
  cheap-pilot shape; the cheap-pilot earning rule applies).  PASS
  ⇒ W108 = cross-scale-UP Phase 3 retirement bench.  This is the
  genuine strengthening path (single-class → cross-scale).
* **W107-β (405B still unreachable)** — W107 = NIM-FREE preflight
  for the NEXT code-benchmark battlefield under `COO-9` (per the
  W104 Branch C table + the W101 matrix): LiveCodeBench primary /
  APPS backup.  NO new NIM spend in W107-β until a preflight +
  cheap pilot earns it.  HumanEval+ is NOT re-run (already retired
  single-class); MBPP+ V2 is NOT re-opened (capped); SWE-bench-
  lite stays unconditionally out of scope.
* **W107-γ (consolidation lane; can run in parallel)** —
  publication-grade consolidated narrative of the W89 → W103 →
  W104 → W105 → W106 code-retirement arc (the W105 Verdict-A
  hardening-lane content, deferred because the realized verdict
  was C/bounded).

The 405B reachability gate is decided by a cheap sub-second side-
probe (§ 4 of the anti-drift contract permits it).  If
`NVIDIA_API_KEY` is present at W106 close, the probe is re-run
and the result recorded; otherwise the standing W105 HTTP 404
carries forward and W107 enters Branch β by default until the
probe is re-run.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New W106 explicit-import-only `coordpy.*` module:
  * `coordpy.margin_cap_dispatch_v1` — pure-Python two-gate
    margin-cap dispatch decision rule; no NIM; refuses to run on
    schema / class-count / verdict-label mismatch.
* New W106 artefacts:
  * `docs/RUNBOOK_W106.md` (this file).
  * `coordpy/margin_cap_dispatch_v1.py`.
  * `scripts/run_w106_margin_cap_dispatch.py` (consumes the W105
    consolidated verdict json; emits the GO/NO-GO decision).
  * `tests/test_w106_margin_cap_dispatch_v1.py` (≥ 14 tests; all
    PASS).
  * `docs/RESULTS_W106_MARGIN_CAP_DISPATCH_V1.md` (the dispatch
    decision + NO-GO rationale).
  * `docs/RESULTS_W106_BOUNDED_RETIREMENT_REGISTRATION_V1.md`
    (the strongest-claim registration narrative).
  * `docs/RESULTS_W106_MILESTONE_SUMMARY_V1.md` (milestone
    summary).
  * `docs/FRONTIER_RELEVANCE_AUDIT_W106_V1.md` (frontier audit
    supplement; 16th preflight-discipline validation).

## Phase plan

### Phase 1 — done in W106 (NO NIM)

1. Lock this runbook (the W106 charter) BEFORE any NIM call.
2. Build `coordpy.margin_cap_dispatch_v1` + tests; all PASS.
3. Run the dispatch driver against the W105 consolidated verdict
   json → records the **NO-GO** decision deterministically.
4. Register the bounded retirement across the truth surfaces
   (Claim lane).
5. graphify refresh at start + concrete usage + refresh at close
   (Truth-sync lane).
6. Linear: create `COO-30`; comment on `COO-9` + refresh `COO-6`;
   append a `W106` entry to `linear_github_mapping.json`.

### Phase 2 — conditional cheap confirmation (NOT TRIGGERED)

§ 2 returned NO-GO.  No Phase 2 NIM spend on the Llama-3.1 branch.

### Phase 3 — optional cheap 405B reachability side-probe (sub-second NIM; only if key present)

Independent of the main objective; refreshes the public record;
informs the W107-α / W107-β branch.  If `NVIDIA_API_KEY` is
absent, the standing W105 HTTP 404 carries forward.

## Honest framing

W106's job is to:

1. **Register the already-earned Llama-3.3-70B HumanEval+
   retirement at full strength** — second confirmed multi-seed
   same-budget retirement after W89, on a different benchmark
   family, at ONE model class.
2. **Make the boundedness impossible to miss** — single-class,
   single-family, single-scale; cross-class NOT entitled;
   cross-scale-UP blocked; MBPP-family capped; cross-modal frozen;
   "context solved" NOT established.
3. **Decide the Llama-3.1 branch honestly** — a pre-committed
   two-gate rule returns NO-GO because a rescue-concentrated
   cheap confirmation is an upper bound that cannot overturn the
   authoritative fair-slice FAIL_MARGIN.  No NIM bought from
   temptation.
4. **Use graphify as part of the evidence + navigation loop** —
   refresh, query, refresh.
5. **Leave W107 obvious** — 405B reachability gate decides
   α (cross-scale-UP cheap pilot) vs β (next-battlefield
   NIM-free preflight); γ consolidation can run in parallel.

The Llama-3.3-70B retirement is real and is registered.  The
Llama-3.1-70B result stays bounded and is repeated as bounded.
The programme is entitled to a STRONGER claim than before W105
(it now has a SECOND confirmed retirement) but NOT a stronger
claim than W105 already licensed (still single-class, still no
cross-class / cross-scale-UP / multi-benchmark same-budget
retirement).
