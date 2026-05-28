# W109 — Milestone summary (APPS contaminated-control contrast + LCB de-noise decision + claim tightening)

**One line:** W109 fetched the REAL APPS corpus, ran the contamination-control
cheap pilot, and the W89 mechanism **RECOVERED a large same-budget win on
contamination-EXPOSED APPS (B − A1 = +16.67 pp; 9/9 gates; MLB-2 = 57 %; 0
regressions) — the exact OPPOSITE of its W108 FAIL on contamination-RESISTANT
LiveCodeBench (−3.33 pp).** This double dissociation by vintage is **evidence
CONSISTENT with a contamination-confound — NOT proof, NOT a retirement** (APPS
is exposed → control only). The two confirmed retirements (W89, W105) STAND;
the boundary around them is now SHARPER. `COO-9` stays lead. The one expensive
run was the earned 330-call APPS pilot; $0 on LCB de-noise (NOT WARRANTED) and
$0 on 405B (gate stays closed; not re-probed).

W109 is NOT a new broad tournament — three gated lanes, executed per the
pre-committed `docs/RUNBOOK_W109.md` (locked BEFORE any expensive NIM call).

---

## Lane α — APPS contaminated-control MAIN lane: real fetch, pilot earned, PASS-non-mechanism-driven

**1. Real fetch + pin.** Fetched `codeparrot/apps` via the `refs/convert/parquet`
data-only branch @ commit `0f10e424` (config `all`, split `test`, 743 MB, 5 000
problems; the `main` branch uses a loading script that the HF parquet API
refuses). Materialized the SHA-pinned call-based subset (38 problems: 28
interview / 10 introductory; `apps-test.jsonl` SHA `f6c44d76…`) via
`scripts/fetch_w109_apps_corpus.py` (reproducible: shard SHAs + convert commit
pinned).

**2. Schema confirmed on real data.** `input_output` is a JSON **string** →
`{fn_name, inputs, outputs}`; `inputs[i]` a positional-arg list; output-wrapper
convention HETEROGENEOUS (16 bare / 18 genuine-list / 4 one-element-wrapper),
faithfully matched by the executor's `output==expected OR output==expected[0]`
(official APPS semantics); tests/problem ∈ [2,5]. DISCHARGES the two W108 APPS
confirm-at-fetch caps.

**3. Bench built + preflight earned.** `coordpy.apps_reflexion_bench_v1` (A0/A1/B
byte-identical in shape to W89/W103/W105/W108; difficulty-stratified
outcome-blind slice; `max_tests` cap). Real-data preflight P1∧P2∧P3∧P4 OVERALL
PASS (executor clean on a REAL gold solution + wrong-fails; slice CID
`783687d6…`; verdict CID `0cf1a8e2…`) → pilot EARNED.

**4. Canary + cheap pilot.** A 2-problem canary validated the live path (A0=A1=B
=100 % on 2 easy probes; parseable `class Solution` output; executor scores
correctly). The full pilot (1 seed × 30 × K=5 = 330 calls; ~75 min) returned:

> **A0 = 73.33 % / A1@K=5 = 73.33 % / B = 90.00 % / B − A1 = +16.67 pp;
> 9/9 gates; MLB-2 = 57.14 % PASS, MLB-1 = 23.33 % FAIL → `PASS_NON_MECHANISM_DRIVEN`.**

B regressed on 0 problems and beat A1 on 5 (4 genuine reflexion rescues + 1
attempt-0 sampling win). MLB-1 fails only because A0 = 73 % is high (few
problems need repair) — itself a memorization-consistent signal. Full verdict:
`docs/RESULTS_W109_APPS_CONTROL_PHASE2_70B_V1.md`.

---

## Lane β — LiveCodeBench de-noise DECISION lane: NOT WARRANTED ($0 NIM)

`coordpy.livecodebench_denoise_decision_v1` — a falsifiable two-gate rule
(Gate 1 = marginal POSITIVE miss 0 < B−A1 < +5 pp; Gate 2 = MLB-2 ≥ 33 %).
Applied to W108 (B − A1 = −3.33 pp; MLB-2 = 25 %): **BOTH gates FAIL ⇒ NOT
WARRANTED** (decision CID `290afa46…`). A multi-seed de-noise reduces VARIANCE,
not the MEAN; carrying −3.33 pp to the +5 pp bar needs a +8.33 pp mean shift
that more seeds cannot supply, and the weak MLB-2 is structural, not noise. The
single-seed FAIL bounds the claim; the confound question is verdict-changing
only via the APPS control (Lane α), not more LCB seeds. **Does NOT re-open the
closed Llama-3.1 rescue-concentrated branch** (keys on the fair broad-slice
margin, same model class). 5 PASSing tests.

---

## Lane γ — claim / graphify / truth-tightening lane

* **graphify**: refreshed from HEAD `e7d8bc7` at start ("No code-graph topology
  changes detected" ⇒ already current for W108), re-ingested the new W109
  modules, refreshed again at end. `explain apps_reflexion_bench_v1` confirms
  it is graph-wired (degree 24, `imports_from apps_loader_v1` +
  `imports run_apps_executor_v1`) as the structural SIBLING of the LiveCodeBench
  bench (`path` = 6 hops through the driver layer); `query` located the
  contamination-claim surface in `results/w108/` + the docs.
* **Contamination-control framing doc** (`docs/CONTAMINATION_CONTROL_FRAMING_W109_V1.md`):
  the exposed-vs-resistant 2×2 + the honesty rules, now with the empirical cell
  filled (exposed APPS PASS-on-margin vs resistant LCB FAIL).
* **Claim tightening**: `docs/RESEARCH_STATUS.md`, `docs/THEOREM_REGISTRY.md`,
  `docs/HOW_NOT_TO_OVERSTATE.md`, `CHANGELOG.md` updated so the boundedness is
  impossible to miss — the confound is SUPPORTED (not proven), the two
  retirements may be contamination-linked, APPS is control-only.

---

## Truth surface (sharpened, not weakened)

* **Confirmed retirements: EXACTLY TWO**, both `meta/llama-3.3-70b-instruct` @
  70B — W89 (base HumanEval +5.56 pp) + W105 (HumanEval+ +7.00 pp; MLB-2 56 %).
  W109 adds NONE and retires NONE.
* **New, sharper boundary:** the same-budget advantage RECOVERS on
  contamination-EXPOSED APPS (+16.67 pp) and FAILs on contamination-RESISTANT
  LiveCodeBench (−3.33 pp). The contamination-confound is now **SUPPORTED by a
  double dissociation, NOT established** (one single-seed control pair; APPS
  PASS non-mechanism-driven on invocation).
* Not entitled (unchanged + W109): cross-class, cross-scale-UP (405B 404 ×5),
  MBPP-family, cross-modal, contamination-RESISTANT code superiority (W108
  FAIL), a THIRD retirement (APPS exposed), contamination PROVEN, "context
  solved".
* **Discipline: 19th consecutive preflight-discipline validation** (W93–W109) —
  W109's distinguishing contribution: fetching + pinning a real contaminated-
  control corpus and running a clean control contrast WITHOUT overclaiming the
  PASS.

---

## Stable boundary preserved

`coordpy.__version__ == 0.5.20`; `coordpy.SDK_VERSION == coordpy.sdk.v3.43`;
no PyPI publish; `coordpy/__init__.py` untouched. New work explicit-import only:
2 new modules (`apps_reflexion_bench_v1` + `livecodebench_denoise_decision_v1`);
the W108 `apps_loader_v1`/`apps_executor_v1` reused unchanged; 3 new scripts.
11 new W109 tests pass (6 bench + 5 de-noise) + 19 W108 LCB + 7 W108 APPS green.

---

## W110 (made obvious)

**W110 = a SECOND contamination-RESISTANT benchmark** (the verdict-changing move
for the now-live confound question: is the LiveCodeBench FAIL LCB-specific, or
general to contamination-resistant data?). A multi-seed APPS de-noise is lower
value (APPS can't retire). 405B + Llama-3.1 stay closed. `COO-9` stays lead.
