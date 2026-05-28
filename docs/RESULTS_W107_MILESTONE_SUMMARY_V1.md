# W107 — milestone summary (V1)

> **2026-05-28.  W107 = 405B reachability gate (α) + next-code-
> battlefield NIM-free preflight (β) + publication-grade consolidation
> (γ).  A gated branch milestone with THREE lanes, NOT a new broad
> benchmark tournament.  $0 expensive NIM (only the sub-second free
> 405B side-probe).  `COO-9` stays the lead path.**

## Three lanes

### Lane α — 405B reachability gate: CLOSED (4th consecutive 404)

`docs/RUNBOOK_W107.md` was locked FIRST (before any NIM call).  Then
`meta/llama-3.1-405b-instruct` was re-probed on NIM: **HTTP 404,
183 ms** — the FOURTH consecutive 404 (W104/W105/W106/W107).  GATE =
CLOSED (decision CID `332d4ef9…`).  No 405B cheap pilot earned or
launched; the § 3 cheap-pilot rule was NOT exercised (it requires GATE
= OPEN).  `W104-L-...-405B-UNREACHABLE-ON-NIM-CAP` refreshed.  Lane β
becomes the main empirical lane.  See
`docs/RESULTS_W107_405B_GATE_V1.md`.

### Lane β — next-code-battlefield preflight: LiveCodeBench PRIMARY (no pivot)

The main empirical lane, executing the `COO-9` charter DoD on the
post-EvalPlus battlefield, NIM-free.  Applying the W101 C1–C8 rubric +
the W107 S1∧S2∧S3 structural-soundness test:

* **LiveCodeBench is the structurally-sound PRIMARY** — time-anchored
  contamination resistance (C7) is the decisive publication-grade
  property; the functional (`starter_code`) subset has a clean
  deterministic executor (proven offline: gold-top-level PASS,
  gold-Solution-method PASS, wrong FAIL, infinite-loop TIMEOUT); the
  W89 decomposition fits.  S1 ∧ S2 ∧ S3 all hold.
* **APPS is the structural-pivot backup** — cleaner stdin/stdout
  executor fit but 2021-vintage contamination exposure would weaken
  any claim built on it.  **No pivot triggered.**

Shipped (explicit-import-only, NIM-free):
`coordpy.livecodebench_loader_v1` (SHA-pinnable functional-subset
loader; refuses unpinned/mismatched — the W102 silent-degeneration
guard) + `coordpy.livecodebench_executor_v1` (clean functional-form
subprocess executor) + `scripts/run_w107_livecodebench_preflight.py`
(selection verdict + offline probes) + 16 PASSing tests.  Verdict CID
`55910d11…`.  The A1 residual is honestly recorded as
published-baseline-grade pending operator corpus-fetch.  See
`docs/RESULTS_W107_NEXT_BATTLEFIELD_PREFLIGHT_V1.md`.

### Lane γ — publication-grade consolidation

`docs/CONSOLIDATED_CODE_RETIREMENT_NARRATIVE_V1.md` ships the W89 →
W103 → W104 → W105 → W106 same-budget code-retirement arc as one
defensible narrative: TWO confirmed retirements (W89 +5.56 pp; W105
+7.00 pp), both Llama-3.3-70B @ 70B, bounded on three axes, with the
five non-claims harmonized.  The claim surface was harmonized across
`docs/THEOREM_REGISTRY.md` (banner + W107 section), `docs/RESEARCH_STATUS.md`
(banner), and `docs/HOW_NOT_TO_OVERSTATE.md` (banner + W107 section).

## Verdict shape applied

Per `docs/RUNBOOK_W106.md` § 7 (re-locked in `docs/RUNBOOK_W107.md`):
405B gate CLOSED (404) ⇒ W107-β is the main lane ⇒ LiveCodeBench
preflighted as the structurally-sound primary ⇒ W108 = cheap pilot
after operator corpus-fetch (or the APPS pivot, or an honest no-go).

## Carry-forwards

* **Added (W107-T)**: `W107-T-405B-GATE-FOURTH-404-CLOSED`,
  `W107-T-LIVECODEBENCH-PREFLIGHT-V1-SHIPS`,
  `W107-T-CODE-RETIREMENT-NARRATIVE-CONSOLIDATED`.
* **Added (W107-L)**:
  `W107-L-LIVECODEBENCH-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP`,
  `W107-L-LIVECODEBENCH-RESIDUAL-PUBLISHED-BASELINE-GRADE-CAP`,
  `W107-L-LIVECODEBENCH-FUNCTIONAL-SUBSET-ONLY-CAP`.
* **Refreshed**: `W104-L-...-405B-UNREACHABLE-ON-NIM-CAP` (4th 404).
* **Retired**: NONE.  W107 adds no empirical retirement and retires no
  cap.  W89 + W105 remain the two confirmed retirements (both
  Llama-3.3-70B).
* **Standing**: `W105-L-...-LLAMA31-70B-MARGIN-CAP`,
  `W105-L-...-CROSS-CLASS-RETIREMENT-NOT-ENTITLED-CAP`,
  `W106-L-...-CHEAP-CONFIRMATION-NOT-EARNED-CAP`,
  `W102-L-MBPP-PLUS-V2-...-CAP`, RealWorldQA-frozen-at-11B.

## graphify (truth-sync lane)

* `graphify update .` at START → graph from HEAD `a560202`
  (73 199 nodes / 237 569 edges / 2 364 communities); the watcher
  reported no topology change (working tree == HEAD).
* Concrete usage: `graphify query` (bounded-retirement claim
  provenance); `graphify path` (`humaneval_plus_reflexion_bench_v1.py
  --imports_from--> humaneval_plus_executor_v1.py` — the loader→
  executor→bench wiring the β scaffolding mirrors); `graphify explain
  code_slice_selector_v1` (the COO-14 slice helper β integrates);
  `graphify affected` (loader/executor reverse-dep probes).
* `graphify update .` at CLOSE → graph re-built from the W107 commit
  so `graphify-out/` matches W107 repo truth.

## Is the programme entitled to a stronger claim than before?

**No.**  W107 adds no empirical retirement.  The strongest claim is
unchanged from W105/W106: TWO confirmed retirements, both
Llama-3.3-70B @ 70B, bounded on three axes (class / family / scale).
W107 made the next honest path to a stronger claim concrete (a third
code benchmark under preflight discipline) and recorded that the other
path (405B cross-scale-UP) stays blocked.

## W108 (left obvious)

* **W108 (default)** — LiveCodeBench functional-subset Phase 2 cheap
  pilot (1 seed × 30 problems × K=5; ~330 NIM calls; W93 5-gate +
  the W107-α-shape 9-gate + MLB sub-gates), AFTER the operator
  corpus-fetch confirms the schema + the live A1 residual.
* **W108 (pivot)** — if the live fetch shows LiveCodeBench is
  structurally wrong, the pre-committed APPS pivot applies in the
  same milestone.
* **W108 (no-go)** — if both LiveCodeBench AND APPS fail S1∧S2∧S3 on
  real data, the post-EvalPlus code battlefield is structurally
  capped; the next live move is decided explicitly (e.g. `COO-12`),
  never a re-run of a capped/frozen line.
* **W108-α (latent)** — re-opens only if 405B becomes reachable.

## Stable boundary

* `coordpy.__version__` = `0.5.20` (unchanged).
* `coordpy.SDK_VERSION` = `coordpy.sdk.v3.43` (unchanged).
* No PyPI publish.  `coordpy/__init__.py` untouched.
* TWO new explicit-import-only modules:
  `coordpy.livecodebench_loader_v1` + `coordpy.livecodebench_executor_v1`.
* 16 new PASSing unit tests.

## Discipline

W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 / W98 / W99 / W100 /
W101 / W102 / W103 / W104 / W105 / W106 / **W107** = **17th consecutive
preflight-first + margin-cap-discipline validation**.  W107's
distinguishing addition: a **structural-soundness pivot test**
(S1∧S2∧S3) that can swap the pre-committed primary→backup inside a
single milestone, and an **honest residual-grade distinction**
(published-baseline-grade vs re-executed-sidecar-grade) recorded as a
cap rather than glossed over.

## Anchors

* `docs/RUNBOOK_W107.md` — pre-commit contract.
* `docs/RESULTS_W107_405B_GATE_V1.md` — α gate verdict.
* `docs/RESULTS_W107_NEXT_BATTLEFIELD_PREFLIGHT_V1.md` — β preflight.
* `docs/CONSOLIDATED_CODE_RETIREMENT_NARRATIVE_V1.md` — γ narrative.
* `docs/FRONTIER_RELEVANCE_AUDIT_W107_V1.md` — frontier audit (17th).
* `coordpy/livecodebench_loader_v1.py` + `coordpy/livecodebench_executor_v1.py` + `tests/test_w107_livecodebench_preflight_v1.py`.
* `scripts/run_w107_405b_reachability_gate.py` + `scripts/run_w107_livecodebench_preflight.py`.
* `results/w107/` (probe + gate + preflight artifacts).
* `linear_github_mapping.json` (W107 entry) + `COO-31`.
