# W110 — Milestone summary (SECOND contamination-resistant benchmark: BigCodeBench Phase-2 FAIL + claim tightening)

**One line:** W110 selected **BigCodeBench 2024** as the SECOND
contamination-RESISTANT benchmark (rejecting SWE-bench-lite + LiveBench-coding
at $0), built + fetched + preflighted it (gold-green 968/1140; pilot EARNED),
and the W89 mechanism **FAILed again on contamination-resistant code**
(B − A1 = +0.00 pp; A0 63.33/A1 70.00/B 70.00%; MLB-1 40% PASS, MLB-2 25% FAIL)
— the SAME weak 25% rescue rate as W108 LiveCodeBench. **So the W108 FAIL was
NOT LCB-specific: reflexion fails on contamination-resistant code GENERALLY at
70B.** The contamination-confound moves **SUPPORTED → STRENGTHENED toward a
finding** (still not proven). The two confirmed retirements (W89, W105) STAND;
the boundary tightens to **contamination-EXPOSED-specific at 70B**. `COO-9`
stays lead. The one expensive run was the earned 330-call pilot; $0 on
SWE-bench-lite, LiveBench, APPS, LCB de-noise, and 405B.

W110 is NOT a new broad tournament — three gated lanes, executed per the
pre-committed `docs/RUNBOOK_W110.md` (locked BEFORE any expensive NIM call).

---

## Lane α — second contamination-resistant benchmark MAIN lane

**1. Narrow selection (real-data-probed, $0 NIM).** Locked the S1∧S2∧S3∧S4 +
feasibility selection rule (`RUNBOOK_W110` § 2) BEFORE building. Evaluated the
smallest honest resistant slate:
* **SWE-bench-lite (the COO-9 charter's named candidate) — REJECTED**: the
  in-repo `coordpy/_internal/tasks/swe_*` scaffolding is explicitly *"Not
  SWE-bench end-to-end"* — a synthetic 4-instance MiniSWEBank on the old
  strategy ablation, not the W89 A0/A1/B mechanism; real instances need a
  Docker/per-repo-env harness (fails S2) and produce multi-file patches that
  break the K=5 single-artifact byte-exact budget (fails S3); public PR gold +
  test patches (S1 weak).
* **LiveBench-coding — REJECTED on a real-data probe**: it IS LiveCodeBench
  repackaged (`task: LCB_generation`; `citation: …via livecodebench`; identical
  test-case schema) ⇒ "rerunning LCB" (fails S4). Killed cheaply at $0.
* **BigCodeBench v0.1.4 — SELECTED**: 2024-06 post-cutoff release (S1); clean
  `unittest` oracle (S2); single `task_func` K=5 (S3); novel library-composition
  (S4). All gates hold on real-data probe.

**2. Real fetch + build.** Fetched + SHA-pinned `bigcode/bigcodebench` v0.1.4
(shard SHA `d9a4965821c9…`; JSONL SHA `ca4f352e…`; 1140 problems) via
`scripts/fetch_w110_bigcodebench_corpus.py`. Built 4 explicit-import-only
modules (loader + `unittest` executor + reflexion bench + the Lane β
interpretation rule), byte-identical A0/A1/B in shape to W89/W105/W108/W109.

**3. Headless-Agg correctness fix (caught in-milestone).** The first preflight/
pilot used the macOS *interactive* matplotlib backend (360/1140 BigCodeBench
tasks plot) — it popped GUI windows AND risked a blocking `plt.show()` falsely
TIMING-OUT correct chart solutions. Forced headless `Agg`; re-preflight
recovered **+32 gold-green (936→968)** — confirming the interactive backend had
been falsely failing chart solutions. A wall-stability guard (drop golds ≥ 20 s)
made the slice CID **reproducible** (`b69bf3a0…`, reproduced across two runs).

**4. Preflight EARNED + cheap pilot.** Real-data preflight P1∧P2∧P3∧P4 PASS
(executor self-test 5/5 — no false-pass on real gold; verdict CID `6be9fc8e…`).
The pilot (1 seed × 30 × K=5 = 330 calls; ~106 min) returned:

> **A0 = 63.33% / A1@K=5 = 70.00% / B = 70.00% / B − A1 = +0.00 pp;
> 7/9 gates; MLB-1 = 40.00% PASS, MLB-2 = 25.00% FAIL → `FAIL`.**

B rescued 1 problem past A1 (BigCodeBench/51) and regressed on 1 (/26) — net
zero. Reflexion invoked on 12/30 (MLB-1 PASS) but rescued only 3/12 (MLB-2 25%,
= W108's rate). Full verdict: `docs/RESULTS_W110_BIGCODEBENCH_PHASE2_70B_V1.md`.

---

## Lane β — APPS / LiveCodeBench interpretation lane (pre-committed claim-change)

`coordpy.contamination_resistant_interpretation_v1` — the canonical 9-gate+MLB
evaluator + a falsifiable rule mapping the verdict to the confound implication,
**locked before the verdict** (`RUNBOOK_W110` § 6; framing doc with all three
branches). The realized branch (FAIL): **confound STRENGTHENS toward a finding;
earns_phase3 = False.** The 2×2 resistant column is now **2 FAIL** (LCB −3.33,
BigCodeBench +0.00; both MLB-2 25%) vs **3 exposed PASS** — the W108 FAIL is
GENERAL, not LCB-specific. APPS stays exposed-control (no further APPS NIM —
$0). 6 PASSing tests.

---

## Lane γ — claim / graphify / truth-tightening lane

* **graphify**: refreshed from HEAD at start (`1e8f131`) + re-ingested the new
  W110 modules mid-milestone (`73a0212`, 74832 nodes) + refreshed at end.
  `explain run_bigcodebench_reflexion_bench_v1` confirms it is graph-wired
  (degree 19, imported by the pilot + tests); `path` = 4 hops to BOTH the
  `run_apps_reflexion_bench_v1` and `run_livecodebench_reflexion_bench_v1`
  benches (the right structural sibling of the W108/W109 line); `affected
  load_bigcodebench_v1` traces the loader→bench impact; `query "contamination…"`
  located the claim surface in the W110 preflight verdict + docs.
* **Framing**: `docs/CONTAMINATION_CONTROL_FRAMING_W110_V1.md` — the 2×2 with a
  SECOND resistant point + the pre-committed branches + the filled FAIL outcome.
* **Claim tightening**: RESEARCH_STATUS / THEOREM_REGISTRY / HOW_NOT_TO_OVERSTATE
  / CONSOLIDATED narrative / CHANGELOG updated so the boundary is impossible to
  miss — the confound is STRENGTHENED (not proven); the retirements are
  contamination-EXPOSED-specific at 70B; resistant superiority is 0/2.

---

## Truth surface (sharpened, not weakened)

* **Confirmed retirements: EXACTLY TWO**, both `meta/llama-3.3-70b-instruct` @
  70B — W89 (base HumanEval +5.56 pp) + W105 (HumanEval+ +7.00 pp). W110 adds
  NONE and retires NONE.
* **New, sharper boundary:** the same-budget reflexion advantage is now
  contamination-EXPOSED-specific at 70B — it PASSes on 3 exposed benchmarks and
  FAILs on BOTH contamination-resistant benchmarks tested (LiveCodeBench 2025
  −3.33 pp; BigCodeBench 2024 +0.00 pp; both MLB-2 25%). The
  contamination-confound is **STRENGTHENED toward a finding, NOT proven** (two
  single-seed resistant points; orthogonal-difficulty not excluded).
* Not entitled (unchanged + W110): cross-class, cross-scale-UP (405B 404 ×5),
  MBPP-family, cross-modal, **contamination-RESISTANT code superiority (now 0/2
  — LCB + BigCodeBench both FAIL)**, a THIRD retirement, contamination PROVEN,
  "context solved".
* **Discipline: 20th consecutive preflight-discipline validation** (W93–W110) —
  W110's distinguishing contribution: a real-data selection that rejected the
  charter's named candidate (SWE-bench-lite) + a look-alike (LiveBench=LCB) at
  $0, an in-milestone executor correctness fix (headless Agg / +32 gold-green),
  and a clean second resistant FAIL that answers the LCB-specific-vs-general
  question without overclaiming.

---

## Stable boundary preserved

`coordpy.__version__ == 0.5.20`; `coordpy.SDK_VERSION == coordpy.sdk.v3.43`;
no PyPI publish; `coordpy/__init__.py` untouched. New work explicit-import only:
4 new modules (`bigcodebench_loader_v1` / `bigcodebench_executor_v1` /
`bigcodebench_reflexion_bench_v1` / `contamination_resistant_interpretation_v1`)
+ 3 new scripts (fetch / preflight / pilot). 17 new W110 tests pass + the W108
(19+7) and W109 (6+5) benchmark-line tests still green (54 together).

---

## W111 (made obvious)

**W111 = register the tightened contamination-EXPOSED-specific-at-70B boundary**
and decide the honest next move: (a) a DIFFERENT mechanism that might beat
same-budget self-consistency on contamination-resistant code, or (b) acceptance
of the bounded two-retirement contamination-EXPOSED-HumanEval-family claim as
the honest code ceiling. A multi-seed de-noise of either resistant FAIL is NOT
WARRANTED (a +0.00 / −3.33 pp weak-MLB-2 point cannot be de-noised into a PASS).
405B + Llama-3.1 stay closed; APPS stays exposed-control. `COO-9` stays lead.
