# W106 — milestone summary (V1)

> **2026-05-28.  W106 = bounded-retirement registration +
> Llama-3.1 margin-cap dispatch + graphify truth-sync.  A coupled
> claim-registration + margin-cap-dispatch milestone, NOT a new
> benchmark tournament.  No expensive run.  $0 NIM on the
> Llama-3.1 branch.  `COO-9` stays the lead path.**

## Three lanes

### Lane 1 — Claim / theorem lane: bounded second retirement REGISTERED

The W105 `meta/llama-3.3-70b-instruct` HumanEval+ Phase 3
retirement (+7.00 pp; 6/6 bars; MLB-2 55.62 %) is registered as
the **SECOND confirmed multi-seed same-budget multi-agent
superiority retirement after W89**, bounded to ONE model class /
ONE benchmark family / ONE parameter scale.  Registered across:
the theorem registry (`W106-T-BOUNDED-SECOND-RETIREMENT-REGISTERED`
+ banner + canonical claims row), the research-status banner, and
the do-not-overstate rules (new W106 section + meta-rule bullet).
Boundedness is front-and-centre everywhere.  See
`docs/RESULTS_W106_BOUNDED_RETIREMENT_REGISTRATION_V1.md`.

### Lane 2 — Llama-3.1 margin-cap dispatch: NO-GO

A pre-committed two-gate rule (`coordpy.margin_cap_dispatch_v1`)
decided the Llama-3.1-70B `FAIL_MARGIN` branch:

* **GATE 1 (entitlement)** = ENTITLED — the failure signature
  (margin +2.33 ∈ [0,+5); MLB-2 50.54 % ≥ 33 %; A1 86.33 % < 90 %)
  matches the Branch-C "cheap confirmation (~990 NIM calls)" row.
* **GATE 2 (verdict-changing power)** = FAIL on all three
  sub-conditions: a rescue-concentrated slice is an upper bound
  (W102 anti-pattern); the authoritative fair broad-slice Phase 3
  verdict already ran at W105 (+2.33 pp); the miss is a clean true
  magnitude miss (no confound).

**Decision = `NO_GO`.**  Accept the bounded single-class claim.
$0 NIM.  Carry-forward
`W106-L-HUMANEVAL-PLUS-LLAMA31-70B-MARGIN-CAP-CHEAP-CONFIRMATION-NOT-EARNED-CAP`.
Decision CID `de3dfb02…`.  See
`docs/RESULTS_W106_MARGIN_CAP_DISPATCH_V1.md`.

### Lane 3 — Graphify / truth-sync

* `graphify update .` at START → graph rebuilt from HEAD
  `ae54d00` (73 026 nodes / 237 304 edges / 2 336 communities);
  dated backup `graphify-out/2026-05-28/`.
* Concrete usage: `graphify query` (HumanEval+ retirement-claim
  provenance + doc truth surfaces); `graphify affected`
  (`phase3_retirement_evaluator_v1.py` → sibling error-class with
  `cross_scale_comparator_v1`); `graphify explain`
  (`cross_class_comparator_v1` — the sibling template the new
  dispatch module mirrors).  Finding: the new
  `margin_cap_dispatch_v1` module sits as a SIBLING of the
  W104/W105 evaluators/comparators; doc truth surfaces navigated
  directly.
* `graphify update .` at CLOSE → graph re-built from the W106
  commit so `graphify-out/` matches W106 repo truth.

## Verdict shape applied

Per the pre-committed `docs/RESULTS_W105_W106_PLANNING_V1.md`
Verdict C / sub-case C1 (= `docs/RUNBOOK_W105.md` § Planning lane
Branch B): retired class = Llama-3.3-70B, failed class =
Llama-3.1-70B.  W106 executed the pre-committed dispatch:
bounded-claim registration on the retired class + the W104 Branch
C dispatch keyed to the Llama-3.1 failure mode, gated by the
margin-cap verdict-changing-power rule → NO-GO.

## Carry-forwards

* **Added**: `W106-T-BOUNDED-SECOND-RETIREMENT-REGISTERED`,
  `W106-T-MARGIN-CAP-DISPATCH-V1-SHIPS`,
  `W106-L-HUMANEVAL-PLUS-LLAMA31-70B-MARGIN-CAP-CHEAP-CONFIRMATION-NOT-EARNED-CAP`.
* **Retired**: NONE.  W106 adds no empirical retirement and
  retires no prior cap.  W89 + W105 remain the two confirmed
  retirements (both Llama-3.3-70B).
* **Standing**: `W105-L-...-LLAMA31-70B-MARGIN-CAP`,
  `W105-L-...-CROSS-CLASS-RETIREMENT-NOT-ENTITLED-CAP`,
  `W102-L-MBPP-PLUS-V2-...-CAP`,
  `W104-L-...-405B-UNREACHABLE-ON-NIM-CAP`, RealWorldQA-frozen-at-11B.

## Is the programme entitled to a stronger claim than before?

**Yes — but bounded.**  Before W105 there was ONE confirmed
retirement (W89).  After W105/W106 there are TWO (W89 + W105),
both on `meta/llama-3.3-70b-instruct`.  That IS stronger.  It is
NOT a cross-class / cross-scale-UP / multi-benchmark same-budget
retirement, and W106 deliberately declined to manufacture one.

## W107 (left obvious)

**405B reachability gate** (decided by a cheap sub-second
side-probe; standing W105 result = HTTP 404):

* **W107-α (405B reachable)** — HumanEval+ Phase 2 cheap pilot at
  `meta/llama-3.1-405b-instruct` on the W105 inner-kernel slice
  (cross-scale-UP; cheap-pilot earning rule).  PASS ⇒ W108 Phase 3
  cross-scale-UP retirement bench.  The genuine strengthening
  path (single-class → cross-scale).
* **W107-β (405B unreachable)** — NIM-FREE preflight for the next
  code-benchmark battlefield under `COO-9` (LiveCodeBench primary
  / APPS backup); no NIM until a preflight + cheap pilot earns it.
* **W107-γ (parallel)** — publication-grade W89→W103→W104→W105→W106
  consolidated narrative.

## Stable boundary

* `coordpy.__version__` = `0.5.20` (unchanged).
* `coordpy.SDK_VERSION` = `coordpy.sdk.v3.43` (unchanged).
* No PyPI publish.  `coordpy/__init__.py` untouched.
* ONE new explicit-import-only module: `coordpy.margin_cap_dispatch_v1`.
* 20 new PASSing unit tests.

## Discipline

W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 / W100 /
W101 / W102 / W103 / W104 / W105 / **W106** = **16th consecutive
preflight-first + margin-cap-discipline validation**.  W106's
distinguishing addition: the margin-cap verdict-changing-power
gate, which converted an ENTITLED ~990-call cheap confirmation
into a disciplined NO-GO by asking not "are we allowed to spend?"
but "can this spend change the verdict?".

## Anchors

* `docs/RUNBOOK_W106.md` — pre-commit contract.
* `docs/RESULTS_W106_BOUNDED_RETIREMENT_REGISTRATION_V1.md`
* `docs/RESULTS_W106_MARGIN_CAP_DISPATCH_V1.md`
* `docs/FRONTIER_RELEVANCE_AUDIT_W106_V1.md`
* `coordpy/margin_cap_dispatch_v1.py` + `tests/test_w106_margin_cap_dispatch_v1.py`
* `results/w106/margin_cap_dispatch/margin_cap_dispatch_decision.json`
* `linear_github_mapping.json` (W106 entry) + `COO-30`.
