# W111 — Milestone summary (different-MECHANISM tournament on contamination-resistant code → bounded-claim fallback EARNED)

**One line:** W111 ran the honest aggressive move — a genuinely DIFFERENT
mechanism on contamination-resistant code, not defeat by default. A NIM-free
mechanism-mining census (re-executing all 300 W110 BigCodeBench candidates)
localised the resistant failure to **81.6 % SEMANTIC hidden-test coupling /
1.8 % API-grounding**, which **killed M1 (planner) and M2 (introspection) at
$0** and admitted **M3 (executor-grounded structured-failure patcher)** — a
materially-different mechanism. M3's smallest-decisive 143-call probe shows it
rescued ONE hard-core problem reflexion could not (`/13`, via the patch loop)
but its mechanism is **sub-reflexion and non-load-bearing** (patch rescue rate
12.5 % < reflexion's 25 % < the 33 % floor; its +15.38 pp on the
rescue-concentrated slice is an upper bound inflated by one attempt-0 sampling
win). **M3 did NOT earn the fair 30-slice pilot ⇒ the bounded contamination-
EXPOSED-HumanEval-family claim is the honest code ceiling** (per the
pre-committed § 4.3 EARN bar + the W104→W105 erosion + W106 margin-cap
discipline). The two confirmed retirements (W89, W105) STAND. `COO-9` stays
lead. The one expensive run was the earned 143-call M3 probe; $0 on M1/M2, the
fair pilot, APPS, reflexion, and 405B.

W111 is NOT a new benchmark tournament, NOT a reflexion rerun — three gated
lanes, executed per the pre-committed `docs/RUNBOOK_W111.md` (locked + committed
BEFORE the only NIM call).

---

## Lane α — different-mechanism MAIN lane

**1. Full NIM-free mechanism-mining pass ($0).**
`scripts/mine_w111_resistant_failure_modes_v1.py` re-executed all 300 W110
BigCodeBench A1+B candidates through the real `unittest` executor. Resistant
failure distribution (114 failures): **SEMANTIC_LOGIC 81.6 %**, TIMEOUT 7.9 %,
ENV_HARNESS 6.1 %, OTHER 2.6 %, **API_GROUNDING 1.8 %** (the only 2 are on
`/51`, already rescued). Hard-core ablation (8 both-fail problems): **6/8
mock-coupling** (fix needs the hidden test source = unreachable in a fair
regime), **2/8 output-value** (`/15`, `/20`).

**2. Candidate slate (hypotheses before results; RUNBOOK § 2–3).**
* **M2 (tool-augmented local symbol/doc introspection)** — attacks API_grounding
  (1.8 %); introspection can't reveal hidden-test conventions. **KILLED $0.**
* **M1 (library/spec-grounded planner→coder)** — attacks comprehension (failures
  are hidden-convention, not comprehension); sacrifices a self-consistency
  sample, no executor grounding. **KILLED $0**, dominated by M3.
* **M3 (executor-grounded structured-failure patcher)** — typed expected/actual
  contract + minimal-patch on the latest candidate (NEVER the test source);
  targets the dominant 81.6 % SEMANTIC class with executor grounding; genuinely
  different from prose reflexion. **ADMITTED** to a live probe.

**3. M3 smallest-decisive live probe (the only NIM spend; 143 calls; ~69 min).**
On the pinned rescue-concentrated 13-problem hard-core slice (CID `b611fae0…`):

> **A0 = 30.77 % / A1 = 30.77 % / M3 = 46.15 % / M3 − A1 = +15.38 pp (UPPER
> BOUND); MLB-1 = 61.5 %, MLB-2 = 12.5 %.** M3-only wins: `/13` (PATCH-LOOP
> rescue — reflexion B failed this) + `/20` (attempt-0 SAMPLING win, NOT the
> mechanism). 0 regressions vs A1. Did NOT hold `/51` (reflexion B's rescue).

**4. Decision (RUNBOOK § 4.3 + W104/W106 discipline).** Literal verdict =
`AMBIGUOUS` (rescued 1 OUTPUT_VALUE but didn't hold `/51`). Resolved to **M3 did
NOT earn the fair pilot**: (a) EARN bar (hold `/51`) not met; (b) the margin is
NON-mechanism-driven (MLB-2 12.5 % < reflexion's 25 % < 33 % floor; 1 patch + 1
sampling win); (c) a rescue-concentrated upper bound with a sub-floor mechanism
cannot clear +5 pp mechanism-driven on the fair slice (W104→W105: +10 pp →
+2.33 pp) ⇒ fair-pilot verdict-changing power LOW (W106) ⇒ **NOT WARRANTED**.
Full verdict: `docs/RESULTS_W111_M3_PATCHER_PROBE_70B_V1.md`.

---

## Lane β — benchmark / claim-discipline lane

The bounded-claim fallback rule was **pre-committed** (RUNBOOK § 6, locked before
NIM) as a LAST resort, with the M3 cheap-pilot gates (§ 5) and the
BigCodeBench-primary / LiveCodeBench-secondary rule (§ 2.4). The realized branch:
M1 + M2 killed NIM-free; M3 probed and did not earn the fair pilot (sub-reflexion
mechanism) ⇒ **bounded-claim fallback EARNED**. No NIM on APPS, the old reflexion
mechanism, LiveCodeBench (the secondary cross-check is gated on an M3 fair-slice
PASS that did not happen), or 405B. The interpretation evaluator
(`evaluate_phase2_gates_v1`) was reused mechanism-agnostically; the W110-specific
`interpret_second_resistant_result_v1` was correctly NOT used (W111 tests a
mechanism, not the confound).

---

## Lane γ — graphify / truth-tightening lane

* **graphify**: refreshed from HEAD at start (`d41265d5`, "No code-graph
  topology changes detected" ⇒ already current) + re-ingested the new M3 module
  mid-milestone (M3 bench node degree 23) + refreshed at end. `explain` confirms
  the bench + interpretation modules are graph-wired; `path` M3↔W110-reflexion =
  3-hop sibling (M3 imports the shared code-extractor); `affected` traces M3's
  reuse of the BigCodeBench loader/executor; `query` located the resistant/
  retirement claim surfaces.
* **Claim tightening**: RESEARCH_STATUS / THEOREM_REGISTRY / HOW_NOT_TO_OVERSTATE
  / CONSOLIDATED narrative / CONTAMINATION_CONTROL_FRAMING_W111 / FRONTIER audit /
  CHANGELOG updated so the boundary is now defensible against the "maybe a better
  mechanism wins" objection — the one mechanism aligned with the resistant
  failure was built and underperformed even reflexion.

---

## Truth surface (sharpened, not weakened)

* **Confirmed retirements: EXACTLY TWO**, both `meta/llama-3.3-70b-instruct` @
  70B — W89 (base HumanEval +5.56 pp) + W105 (HumanEval+ +7.00 pp). W111 adds
  NONE and retires NONE.
* **Sharper boundary:** the resistant-code ceiling is now shown **NOT specific
  to the reflexion mechanism** — a genuinely-different executor-grounded patcher
  (M3) also fails to beat same-budget self-consistency on contamination-resistant
  code at 70B (cheap-pilot scale). The bounded two-retirement
  contamination-EXPOSED-HumanEval-family claim is the honest code ceiling.
* **Honest positive (registered, not overstated):** M3's patch loop rescued ONE
  hard-core problem reflexion could not (`/13`), so the different-mechanism idea
  is not vacuous — but at a sub-reflexion rescue rate, it is a lateral trade.
* Not entitled (unchanged + W111): cross-class, cross-scale-UP (405B 404 ×5),
  MBPP-family, cross-modal, **contamination-RESISTANT code superiority via ANY
  tested mechanism (reflexion 0/2; M3 sub-reflexion, fair pilot not earned)**, a
  THIRD retirement, contamination PROVEN, "context solved".
* **Discipline: 21st consecutive preflight-discipline validation** (W93–W111) —
  W111's distinguishing contribution: a NIM-free failure-mode census that drove
  a $0 kill of 2 of 3 candidates, a materially-different mechanism (M3) built +
  unit-tested with a test-source-non-leak fairness guard, a smallest-decisive
  probe that measured M3 sub-reflexion, and a pre-committed-rule + established-
  discipline refusal to spend on a fair pilot that cannot change the verdict.

---

## Stable boundary preserved

`coordpy.__version__ == 0.5.20`; `coordpy.SDK_VERSION == coordpy.sdk.v3.43`;
no PyPI publish; `coordpy/__init__.py` untouched. New work explicit-import only:
1 new module (`executor_grounded_patcher_v1`) + 2 scripts (mining + probe) + 9
tests. W110 benchmark-line tests still green.

---

## W112 (made obvious)

**W112 = the honest post-fallback move** (RUNBOOK § 8): (1) a cross-scale-UP
probe of reflexion/M3 on a stronger code model IF one becomes reachable on NIM
(405B stays CLOSED; standing extension), OR (2) a NIM-free-earned residual M3
strengthening that plausibly clears the 33 % rescue floor (never another
sub-floor rerun), OR (3) acceptance of the bounded two-retirement
contamination-EXPOSED-HumanEval-family claim as the programme's honest code
ceiling. NOT a reflexion rerun, APPS, 405B/Llama-3.1, MBPP+ V2, or
bounded-context drift. `COO-9` stays lead.
