# RUNBOOK — W109 (APPS contaminated-control contrast + LiveCodeBench de-noise decision + claim tightening)

**Pre-commit contract. Locked BEFORE any expensive NIM call** (the APPS
contamination-control cheap pilot). `COO-9` REMAINS the lead path. No version
bump; no PyPI; `coordpy/__init__.py` untouched; advanced work explicit-import
only.

This milestone executes the W108 pre-commit (`docs/RUNBOOK_W108.md` § 8 FAIL
branch; `COO-32` close comment): **W108 LiveCodeBench Phase-2 = `FAIL` ⇒ W109
= APPS contaminated-control contrast (lead) OR multi-seed LiveCodeBench
de-noise.** W109 resolves BOTH: it runs the APPS control AND decides the LCB
de-noise question with a falsifiable rule.

W109 is NOT a new broad benchmark tournament. It is exactly THREE lanes: the
APPS contaminated-control main lane (α); a LiveCodeBench de-noise / follow-up
DECISION lane (β); a graphify-backed claim-tightening lane (γ). W109 does NOT
drift into bounded-context / compaction / summarization / token-compression
work — those remain anti-patterns, not the frontier path.

**The scientific question (stated once, sharply):** W108's first
contamination-RESISTANT test of the W89 mechanism (LiveCodeBench 2025) FAILed
(B − A1 = −3.33 pp; MLB-2 = 25 %). Does the SAME mechanism, under the SAME
same-budget K=5 contract, RECOVER on a contamination-EXPOSED 2021 benchmark
(APPS)? **A PASS is evidence CONSISTENT with a contamination-confound — NOT
proof, and NOT a retirement (APPS is contamination-exposed). A FAIL materially
WEAKENS the confound hypothesis and tightens the mechanism's boundary.**

---

## Linear

* `COO-9` (second/next code benchmark battlefield) — High, **lead path**.
  W109 executes its DoD on the APPS contamination-control contrast.
* `COO-33` (to be created) — the W109 milestone sub-issue under `COO-6`.
* Sync discipline: GitHub is canonical truth; Linear synced at end-of-
  milestone via `scripts/sync_linear_github_v1.py` + `linear_github_mapping.json`.

---

## What is NOT in scope (anti-drift)

* No reopening MBPP+ V2 (`W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`).
* No reopening the frozen cross-modal lines (RealWorldQA @ 11B / MathVista /
  ChartQA).
* No reopening the closed Llama-3.1 rescue-concentrated branch
  (`W106-L-…-MARGIN-CAP-…-NOT-EARNED-CAP`). The Lane β rule explicitly forbids
  rescue-concentrated slices and cross-class re-runs (§ 4).
* No expensive 405B run (the gate is CLOSED at the 5th 404; W109 does not even
  re-probe it — that lane is exhausted for now and re-opens only on
  reachability + a re-lock).
* No LLM-as-judge anywhere; functional/call-based subset only.
* **No claim that an APPS PASS proves contamination, retires anything, or is
  publication-grade** — APPS is 2021 contamination-EXPOSED; it is CONTROL
  evidence about the confound hypothesis, never a third retirement.
* **No claim that an APPS PASS overwrites the W108 LiveCodeBench FAIL** — both
  stand; they are a contrast, not a replacement.

---

## Operational state (pre-W109 facts)

| Fact | Value |
|---|---|
| Confirmed retirements | EXACTLY TWO, both `meta/llama-3.3-70b-instruct` @ 70B: **W89** base HumanEval +5.56 pp; **W105** HumanEval+ +7.00 pp (6/6 bars; MLB-2 55.62 %). Both contamination-EXPOSED HumanEval-family (2021). |
| W108 LiveCodeBench (contamination-RESISTANT 2025) | Phase-2 **FAIL**: A0=43.33 / A1@K=5=63.33 / B=60.00 %; B − A1 = −3.33 pp; 7/9 gates; MLB-1 53.33 % PASS, MLB-2 25 % FAIL. First contamination-resistant test of the mechanism; it FAILed. |
| Contamination-confound | **OPEN hypothesis, not a finding** (`W108-L-REFLEXION-NOT-DEMONSTRATED-ON-CONTAMINATION-RESISTANT-BENCH-CAP`). |
| 405B gate | CLOSED — HTTP 404 ×5 (W104–W108). Not re-probed in W109. |
| APPS scaffolding | loader + executor + offline preflight + 7 tests built at W108 (real, pivot-ready). |
| Stable boundary | `coordpy.__version__ == 0.5.20`; `SDK_VERSION == coordpy.sdk.v3.43` |

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α — APPS contaminated-control MAIN lane.** Fetch the REAL
  `codeparrot/apps` corpus → materialize + SHA-pin the call-based JSONL
  (§ 2) → confirm schema + output-wrapper convention on real data → build the
  APPS reflexion bench byte-identical in shape to the W89/W103/W105/W108 line
  → run the REAL-DATA preflight (§ 3) → if it PASSes, the cheap pilot is
  EARNED; lock this RUNBOOK, then run the pilot (§ 4) and evaluate the same 9
  gates + MLB-1/MLB-2. If the real-data preflight FAILs structurally, STOP
  honestly (do NOT fake a control result from a broken battlefield).
* **Lane β — LiveCodeBench de-noise DECISION lane.** Build the falsifiable
  two-gate decision rule (§ 5) BEFORE any further LCB NIM. Apply it to the
  W108 result. Spend more LCB NIM ONLY if WARRANTED (verdict-changing power),
  NEVER out of discomfort with the negative. Produce the W110 implication.
* **Lane γ — claim / graphify / truth-tightening lane.** Refresh graphify from
  HEAD at start + end; use `query`/`affected`/`explain`/`path` to find the
  claim surfaces; tighten the boundedness around what W89/W105 prove, what
  W108 bounds, and what APPS can/cannot establish; add a contamination-control
  framing doc (§ 6).

The lanes run in the same milestone. The only expensive run is the APPS cheap
pilot (Lane α), and ONLY if earned.

---

## § 2 — APPS real-data fetch / schema-confirm / residual rule (LOCKED)

The fetch is executed in-milestone (pyarrow + network egress available) via
`scripts/fetch_w109_apps_corpus.py`. **Pinned provenance:**

* dataset `codeparrot/apps` (Hendrycks et al., 2021 — contamination-EXPOSED,
  2021 vintage; C7 = C).
* revision `refs/convert/parquet` @ commit
  `0f10e424e13e1c2a69f851e153097b71b6734a1f`; config `all`, split `test`.
* shards `0000.parquet` (424 202 850 B, SHA-256 `f1c36415…`) + `0001.parquet`
  (319 566 256 B, SHA-256 `3b02746c…`); 5 000 test problems.
* **call-based (functional) subset = 38 problems** (28 interview + 10
  introductory; 0 competition) — rows whose `input_output` (a JSON **string**)
  decodes to a dict with a non-empty `fn_name`.
* materialized `~/.cache/coordpy/apps-test.jsonl` (40 075 B);
  **SHA-256 `f6c44d76be0eea7669f0ccbd90b6b45fb03a4327d06682073b5cd8f905310918`**
  (the loader REFUSES on mismatch / missing cache / schema mismatch — the W102
  silent-degeneration guard).

**Confirmed real schema** (discharges `W108-L-APPS-LOADER-V1-SCHEMA-CONFIRM-AT-FETCH-CAP`):

* `input_output` is a JSON **string** decoding to `{fn_name, inputs, outputs}`;
* `inputs[i]` is a JSON list of positional arguments (single `json.loads` +
  splat — NOT the LiveCodeBench newline-per-arg form);
* **output-wrapper convention is HETEROGENEOUS** (16 bare / 18 genuine-list /
  4 one-element-wrapper across the 38). The executor's `_matches`
  (equality OR len-1 unwrap) FAITHFULLY mirrors the official APPS
  `output == expected OR output == expected[0]` semantics — confirmed on real
  data (discharges `W108-L-APPS-EXECUTOR-V1-OUTPUT-WRAPPER-TOLERANCE-CAP`);
* tests/problem ∈ [2, 5] (max 5) — the `max_tests_per_problem=25` bench cap
  truncates NOTHING on this corpus.

**Residual rule:** the LIVE A1@K=5 residual is NOT pre-measured (no APPS
sidecar exists in-repo). It is measured BY the cheap pilot (gate G2), exactly
as for LiveCodeBench at W108. APPS evidence is contamination-EXPOSED CONTROL
only and is never the publication-grade time-anchored claim surface
(`W108-L-APPS-CONTAMINATION-EXPOSED-2021-VINTAGE-CAP`).

---

## § 3 — Gold-path correctness bar before any pilot (LOCKED — and PASSED)

NO NIM is spent until ALL hold (they DO, as of this lock — verdict CID
`0cf1a8e2b02acc1511c5db1f2fe3ce79771987a2b9b9759c2dfd978bf5498e7b`,
`results/w109/apps_preflight/preflight_verdict.json`):

1. `tests/test_w109_apps_reflexion_bench_v1.py` green (6 tests, incl. the
   full-bench gold-path A0=A1=B=1.0, reflexion-rescue MLB exercise, the
   `max_tests` cap, and the REAL-corpus → locked-slice-CID binding) +
   `tests/test_w109_livecodebench_denoise_decision_v1.py` green (5 tests) +
   `tests/test_w108_apps_backup_v1.py` still green (7 tests).
2. P1 corpus integrity: the SHA-pinned loader reads 38 call-based problems.
3. P2 executor self-test: synthetic gold (top-level + Solution) PASS, wrong
   FAIL, infinite-loop TIMEOUT, AND a REAL gold `reverseWords` solution PASSes
   / a wrong one FAILs on the live corpus (no false-pass on real data).
4. P3 loader real-data: 38 ≥ 30, ALL `fn_name` resolved, every problem has
   tests; contamination framing recorded (C7 = C, 2021 EXPOSED — the control).
5. P4 deterministic outcome-blind 30-slice reproduces slice CID
   `783687d6109d2e452aba8a32952b5569ed7c03d8aa1d040f1a22ef18688c6dcc`
   (22 interview / 8 introductory); the pilot `--dry-run` reproduces it.

If any regress, the pilot is NOT launched.

---

## § 4 — APPS cheap-pilot gates (LOCKED — Lane α, earned)

**Slice (G1, pre-committed):** the deterministic outcome-blind
difficulty-stratified 30-problem slice from
`select_apps_functional_slice_v1`; slice CID
`783687d6109d2e452aba8a32952b5569ed7c03d8aa1d040f1a22ef18688c6dcc`
(22 interview / 8 introductory).

**A0 / A1 / B (byte-identical mechanism to W89/W103/W105/W108):**

* **A0** — stock single-shot at T=0.0 (1 call/problem).
* **A1** — first-pass-among-K=5 self-consistency at T=0.7 (5 calls).
* **B** — sequential-reflexion-K=5 at T=0.7, each turn conditioned on the
  cumulative (candidate, executor-stderr) history (5 calls).
* Budget byte-exact (A1 and B both K=5; no early-stop); same model on all arms;
  executor truth = subprocess exit 0 iff every call-based test matches (first
  `max_tests=25`, which is all of them here); NO LLM-as-judge.

**Run:** 1 seed (109001) × 30 problems × K=5 = **330 NIM calls** at
`meta/llama-3.3-70b-instruct` (the SAME W89/W105/W108 class — so the
contamination contrast is clean, single-class). A ≤2-problem canary (~22 calls)
runs first to validate the live path on real APPS output.

**The 9 Phase-2 gates + MLB sub-gates (verbatim from W103/W104/W105/W108):**

| Gate | Pass condition |
|---|---|
| G1 slice pre-committed | slice CID pinned (above) |
| G2 A1 < 90 % | A1@K=5 non-saturated (real headroom) |
| G3 B > A1 | strict |
| G4 (B − A1) ≥ +5 pp | margin bar |
| G5 (B − A0) ≥ +5 pp | vs single-shot |
| G6 per-problem majority | B did not regress vs A1 on ≥ 16 of 30 |
| G7 budget byte-exact | A1/B both K=5 |
| G8 audit chain re-derives | per-call CIDs + per-seed/bench Merkle |
| G9 executor clean | no-LLM-judge subprocess |
| MLB-1 | reflexion invoked on ≥ 33 % of problems |
| MLB-2 | of invocations, ≥ 33 % rescued by reflexion |

**Verdict labels (exact):** `PASS_MECHANISM_DRIVEN` iff 9/9 + both MLB;
`PASS_NON_MECHANISM_DRIVEN` iff 9/9 but an MLB fails; `FAIL` otherwise.

**What a W109 APPS PASS entitles (and does NOT):** a `PASS_MECHANISM_DRIVEN`
entitles exactly: "the W89 mechanism recovers same-budget superiority on a
contamination-EXPOSED 2021 code benchmark (APPS) at cheap-pilot scale at 70B,
having FAILed on contamination-RESISTANT LiveCodeBench 2025 — a contrast
CONSISTENT with a contamination-confound." It does **NOT** prove the confound
(one control pair is not proof), does **NOT** add a third retirement (APPS is
exposed; retirement needs contamination-resistant + Phase-3 multi-seed), and
does **NOT** overwrite the W108 LiveCodeBench FAIL.

---

## § 5 — LiveCodeBench de-noise decision rule (LOCKED — Lane β)

`coordpy.livecodebench_denoise_decision_v1.decide_livecodebench_denoise_v1`.
A multi-seed LCB de-noise is **WARRANTED** iff BOTH gates hold:

* **Gate 1 — marginal POSITIVE miss**: 0 < (B − A1) < +5 pp. A negative point
  cannot be de-noised into a ≥ +5 pp PASS (more seeds reduce VARIANCE, not the
  MEAN). 
* **Gate 2 — mechanism load-bearing**: MLB-2 ≥ 33 %.

Forbidden (NOT the closed Llama-3.1 branch under a new name): the rule keys on
the FAIR broad-slice margin (NOT a rescue-concentrated slice) and the SAME
model class (cross-class is the closed W106 branch).

**Applied to W108 (B − A1 = −3.33 pp; MLB-2 = 25 %): Gate 1 FAILS (negative) ∧
Gate 2 FAILS (weak) ⇒ NOT WARRANTED** (decision CID `290afa46…`). The
single-seed FAIL already bounds the claim (required mean shift to PASS =
+8.33 pp, which multi-seed cannot supply). **$0 further LCB NIM.** The
contamination-confound question is verdict-changing only via the W109 APPS
CONTROL contrast, not via more LCB seeds. (W110 could run the de-noise ONLY in
the counterfactual where W108 had been a marginal positive miss — it was not.)

---

## § 6 — graphify deliverables (LOCKED)

* Refresh at start (`graphify update .`; confirm built from current HEAD —
  done: e7d8bc7, "No code-graph topology changes detected" ⇒ already current).
* Re-ingest after adding the APPS bench + de-noise modules (done).
* Use concretely: `graphify query` for the retirement-claim + contamination
  boundary surface; `graphify explain` on `apps_loader_v1`,
  `apps_executor_v1`, `apps_reflexion_bench_v1`,
  `livecodebench_reflexion_bench_v1`; `graphify path` between the APPS bench
  and the LiveCodeBench bench line (sibling check); `graphify affected` on the
  loader/truth surfaces.
* Refresh at end and record exactly what graphify changed in file selection /
  understanding (`docs/RESULTS_W109_*` + the milestone summary).

---

## § 7 — W110 branch logic (LOCKED)

Driven by the W109 APPS pilot outcome (Lane α), NOT by an LCB de-noise (§ 5
NOT WARRANTED):

* **APPS Phase-2 = `PASS_MECHANISM_DRIVEN`** ⇒ contamination-confound gains
  support (mechanism PASSes on exposed APPS, FAILed on resistant LCB). **W110 =
  a SECOND contamination-RESISTANT benchmark** (e.g., a later LiveCodeBench
  `release` window, or a 2024+ contest-dated functional set) to test whether
  the LCB FAIL is LCB-specific or general to contamination-resistant data —
  the only path that could move the confound from hypothesis toward finding.
  NOT more APPS (exposed → cannot retire).
* **APPS Phase-2 = `PASS_NON_MECHANISM_DRIVEN`** ⇒ margin-without-mechanism;
  register the APPS MLB cap; W110 weighs whether a different slice/seed is
  verdict-changing (it usually is not — W100/W104 discipline).
* **APPS Phase-2 = `FAIL`** ⇒ contamination-confound materially WEAKENED (the
  mechanism fails on BOTH a contamination-resistant (LCB 2025) AND a
  contamination-EXPOSED-non-HumanEval (APPS 2021) functional benchmark). The
  superiority is then bounded as **HumanEval-family-specific** at 70B. **W110 =
  register that tightened boundary**; the honest next move is a DIFFERENT
  mechanism (not a re-run of a capped/frozen line) or acceptance of the
  tightly-bounded two-retirement claim. `COO-9`'s "second code battlefield"
  charter would be substantially answered (negatively) for code-superiority
  GENERALISATION.
* **Pilot could not launch** (NIM unavailable) ⇒ W109 closes
  preflight-earned-but-operator-gated; W110 = launch the earned APPS pilot
  (no new earning work needed).

---

## § 8 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION ==
  "coordpy.sdk.v3.43"`; no PyPI publish; `coordpy/__init__.py` untouched.
* New modules are explicit-import only: `coordpy.apps_reflexion_bench_v1`
  (the contamination-control bench) + `coordpy.livecodebench_denoise_decision_v1`
  (the Lane β rule). The W108 `apps_loader_v1` / `apps_executor_v1` are reused
  unchanged. New scripts: `fetch_w109_apps_corpus.py`,
  `run_w109_apps_preflight.py`, `run_w109_apps_pilot.py`.

---

## Honest framing

W109, on an APPS Phase-2 PASS, entitles exactly: "the W89 mechanism recovers
same-budget superiority on contamination-EXPOSED APPS (2021) after FAILing on
contamination-RESISTANT LiveCodeBench (2025) — a contrast CONSISTENT with a
contamination-confound." It is NOT proof of the confound (one control pair),
NOT a third retirement (APPS is exposed), and does NOT change the
cross-class / cross-scale / cross-modal boundaries or overwrite the W108 FAIL.
On an APPS FAIL, the confound hypothesis WEAKENS and the mechanism's boundary
tightens to HumanEval-family-specific. The two confirmed retirements remain
W89 + W105. The de-noise lane is NOT WARRANTED ($0 further LCB NIM).
