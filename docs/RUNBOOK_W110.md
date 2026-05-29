# RUNBOOK — W110 (second contamination-RESISTANT benchmark selection + real preflight + cheap-pilot earning + claim tightening)

**Pre-commit contract. Locked BEFORE any expensive NIM call** (the
BigCodeBench contamination-resistant cheap pilot). `COO-9` REMAINS the lead
path. No version bump; no PyPI; `coordpy/__init__.py` untouched; advanced work
explicit-import only.

This milestone executes the W109 pre-commit (`docs/RUNBOOK_W109.md` § 7 PASS
branch + `docs/RESULTS_W109_APPS_CONTROL_PHASE2_70B_V1.md` § 6; `COO-33` close
comment): **W109 APPS contaminated-control = `PASS_NON_MECHANISM_DRIVEN`
(B − A1 = +16.67 pp; double dissociation by vintage) ⇒ W110 = a SECOND
contamination-RESISTANT benchmark** to test whether the W108 LiveCodeBench FAIL
is LCB-specific or GENERAL to contamination-resistant code.

W110 is NOT a new broad benchmark tournament. It is exactly THREE lanes: a
narrow contamination-resistant battlefield selection + real preflight +
cheap-pilot earning attempt (α, the main lane); an APPS/LiveCodeBench
interpretation lane that pre-commits how W110's result changes the claim (β);
a graphify-backed claim-tightening lane (γ). W110 does NOT drift into
bounded-context / compaction / summarization / token-compression work — those
remain anti-patterns, not the frontier path.

**The scientific question (stated once, sharply):** the only
publication-grade-strong cell of the contamination 2×2 (a PASS on
contamination-RESISTANT data) is EMPTY; W108's one attempt there (LiveCodeBench
2025) FAILed (B − A1 = −3.33 pp; MLB-2 = 25 %). W109's APPS exposed-control
PASS made a contamination-confound SUPPORTED-not-proven. **Is the W108 FAIL
LiveCodeBench-SPECIFIC, or does the W89 mechanism fail GENERALLY on
contamination-resistant code?** A SECOND, genuinely-different
contamination-resistant benchmark is the verdict-changing move; the answer
sharpens the boundary whichever way it lands.

---

## Linear

* `COO-9` (second/next code benchmark battlefield) — High, **lead path**.
  W110 executes its DoD on a second contamination-resistant battlefield. The
  COO-9 charter explicitly pre-registers "SWE-bench-lite if the evaluation
  story stays honest" as a candidate — W110 evaluates it under the locked
  selection rule (§ 2).
* `COO-34` (to be created) — the W110 milestone sub-issue under `COO-6`.
* Sync discipline: GitHub is canonical truth; Linear synced at end-of-
  milestone via `scripts/sync_linear_github_v1.py` + `linear_github_mapping.json`.

---

## What is NOT in scope (anti-drift)

* No reopening MBPP+ V2 (`W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`).
* No reopening the frozen cross-modal lines (RealWorldQA @ 11B / MathVista /
  ChartQA).
* No reopening the closed Llama-3.1 rescue-concentrated branch
  (`W106-L-…-MARGIN-CAP-…-NOT-EARNED-CAP`).
* No expensive 405B run (the gate is CLOSED at the 5th 404; W110 does not even
  re-probe it).
* No LLM-as-judge anywhere; functional/test-based subset only.
* **No reopening APPS as the MAIN lane** — APPS is 2021 contamination-EXPOSED;
  it is CONTROL evidence only (Lane β), never a third retirement, never
  publication-grade. No more APPS NIM unless there is clear verdict-changing
  power (there is not — § 6).
* **No "just rerunning LiveCodeBench and calling it a new benchmark"** — a
  disjoint later LCB release window only tests "is the FAIL specific to the
  early-2025 LCB slice?", a strictly weaker question than "LCB-specific vs
  general?". It is explicitly EXCLUDED as the W110 lead (§ 2).
* No claim that a contamination-RESISTANT PASS, if obtained, is a retirement
  until a Phase-3 multi-seed retirement bench clears (this milestone is a
  Phase-2 cheap pilot at most).

---

## Operational state (pre-W110 facts)

| Fact | Value |
|---|---|
| Confirmed retirements | EXACTLY TWO, both `meta/llama-3.3-70b-instruct` @ 70B: **W89** base HumanEval +5.56 pp; **W105** HumanEval+ +7.00 pp (6/6 bars; MLB-2 55.62 %). Both contamination-EXPOSED HumanEval-family (2021). |
| W108 LiveCodeBench (contamination-RESISTANT 2025) | Phase-2 **FAIL**: A0=43.33 / A1@K=5=63.33 / B=60.00 %; B − A1 = −3.33 pp; 7/9 gates; MLB-2 = 25 % FAIL. First contamination-resistant test; FAILed. |
| W109 APPS (contamination-EXPOSED 2021, control) | Phase-2 **`PASS_NON_MECHANISM_DRIVEN`**: A0=73.33 / A1=73.33 / B=90.00 %; B − A1 = +16.67 pp; 9/9 gates; MLB-2 = 57.14 % PASS, MLB-1 = 23.33 % FAIL; 0 regressions. Double dissociation by vintage. |
| Contamination-confound | **SUPPORTED, NOT established** (`W109-T-CONTAMINATION-CONFOUND-SUPPORTED-NOT-PROVEN`); one single-seed control pair; APPS PASS non-mechanism-driven on invocation. |
| 405B gate | CLOSED — HTTP 404 ×5 (W104–W108). Not re-probed in W110. |
| Operational reachability (W110 probe) | `NVIDIA_API_KEY` present; `huggingface.co` HTTP 200 (a new corpus fetch is possible). The LCB `release_v6` corpus is still cached locally (SHA `bb4c364f…`). |
| Stable boundary | `coordpy.__version__ == 0.5.20`; `SDK_VERSION == coordpy.sdk.v3.43` |

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α — second contamination-resistant benchmark MAIN lane.** Apply the
  locked selection rule (§ 2) to the smallest honest resistant slate → choose
  the lead resistant candidate → fetch its REAL corpus + SHA-pin → confirm the
  schema + executor cleanness on real data → build the reflexion bench
  byte-identical in shape to the W89/W105/W108/W109 line → run the REAL-DATA
  preflight (§ 4). If the preflight PASSes, the cheap pilot is EARNED; fill the
  § 3 pins, then run the pilot (§ 5) and evaluate the same 9 gates + MLB-1/MLB-2.
  If the real-data preflight FAILs structurally, STOP honestly on that
  candidate → pivot to the single fallback (§ 2.4) IN THIS milestone OR record
  an honest no-go (do NOT fake a resistant result from a broken battlefield).
* **Lane β — APPS / LiveCodeBench interpretation lane.** Build the falsifiable
  interpretation rule (§ 6) BEFORE the W110 pilot lands, so the claim-change is
  pre-committed, not back-fitted. Apply it to the W110 outcome. Keep APPS in
  its lane (control only); spend NO further APPS NIM (no verdict-changing
  power).
* **Lane γ — claim / graphify / truth-tightening lane.** Refresh graphify from
  HEAD at start + end; use `query`/`affected`/`explain`/`path` for file
  selection + the claim surfaces; tighten the boundedness so the contamination
  boundary is defensible after W110 no matter the result (§ 7).

The lanes run in the same milestone. The only expensive run is the
BigCodeBench cheap pilot (Lane α), and ONLY if earned.

---

## § 2 — resistant-benchmark selection rule (LOCKED — applied BEFORE any build)

**The rule.** A candidate is the W110 lead contamination-resistant battlefield
iff it passes the structural-soundness test **S1 ∧ S2 ∧ S3 ∧ S4** (extending
the W107 S1∧S2∧S3 with a genuine-difference axis) AND the feasibility test
**F1 ∧ F2 ∧ F3**:

* **S1 — contamination-RESISTANCE.** Time-anchored / released AFTER the
  Llama-3.x training cutoff (≈ 2024-01-01), OR otherwise credibly not in the
  training corpus. (HumanEval / HumanEval+ / MBPP+ / APPS are EXPOSED 2021 →
  excluded as a resistant lead.)
* **S2 — executor cleanness.** A deterministic, no-LLM-judge subprocess
  executor over real functional/unit tests, reusable from the W108/W109 line —
  NO Docker / per-repo environment / network dependency.
* **S3 — same-budget comparability.** A SINGLE self-contained code artifact
  gradable at K=5 byte-exact, byte-identical A0/A1/B mechanism to
  W89/W105/W108/W109. (Multi-file repo patches break the K=5 single-artifact
  byte-exact budget.)
* **S4 — genuinely-different battlefield.** NOT the W108 LiveCodeBench slice or
  a disjoint LCB release window (that is "rerunning LCB"); NOT APPS as the main
  lane; NOT a capped/frozen line.
* **F1 cheap-pilot feasibility** (≈ 330 NIM calls; ~1 seed × 30 × K=5).
* **F2 decomposition fit** for sequential reflexion (read spec → solve →
  execute → reflect on executor stderr → repair).
* **F3 verdict-changing power** relative to the W108/W109 pair (a second
  resistant FAIL strengthens the confound; a resistant PASS shows the W108 FAIL
  was LCB-specific).

### § 2.1 — the narrow slate evaluated (NOT a broad tournament)

The smallest honest resistant slate implied by repo history + the COO-9
charter (`{MBPP+, HumanEval+, APPS, LiveCodeBench, SWE-bench-lite}`), pruned to
genuinely-resistant + genuinely-different candidates and ranked on real-data
probes (HF datasets-server `first-rows`/`splits`; $0 NIM):

| Candidate | S1 resistant | S2 clean exec | S3 same-budget | S4 different | Verdict |
|---|---|---|---|---|---|
| **SWE-bench-lite** (charter lead) | ✗ weak (public GitHub PR gold + test patches → solution-leakage; not time-anchored post-cutoff) | ✗ (real instances need Docker / per-repo env; the in-repo `coordpy/_internal/tasks/swe_*` scaffolding is an explicitly synthetic 4-instance MiniSWEBank on the OLD `STRATEGY_NAIVE/ROUTING/SUBSTRATE` ablation — NOT the W89 A0/A1/B mechanism; `swe_real_shape_mini.jsonl` = 6 synthetic `external/calc`-style rows, not real SWE-bench data) | ✗ (repo-level multi-file patch breaks K=5 single-artifact byte-exact budget) | ✓ | **REJECTED** — fails S2 ∧ S3, S1 weak; not cheap |
| **LiveBench-coding** | ~ (mixed dates) | ✓ | ✓ | ✗ | **REJECTED on real-data probe** — it IS LiveCodeBench repackaged (`task: LCB_generation`; `citation: …via livecodebench`; identical `public_test_cases`/`private_test_cases` schema). Using it = "rerunning LCB" (S4 ✗). Killed cheaply at $0. |
| **BigCodeBench (v0.1.4)** | ✓ (HF `createdAt 2024-06-05`, v0.1.4 `lastModified 2025-04-30` — released AFTER the ≈2024-01 cutoff; novel compositional task construction) | ✓ (deterministic `unittest.TestCase` `test`; subprocess executor; no LLM judge) | ✓ (single `def task_func(...)` completion; K=5 byte-exact) | ✓ (different authorship + construction from LCB; library-composition, not contest scraping) | **SELECTED** — S1∧S2∧S3∧S4 ∧ F1∧F2∧F3 all hold on real-data probe |

### § 2.2 — the selected lead: BigCodeBench

`bigcode/bigcodebench`, split **`v0.1.4`** (latest; carries the upstream
test-correctness patches → fewer flaky golds; the problems are unchanged from
v0.1.0 / 2024-06, so the contamination-resistance anchor is the 2024-06 release
date). Real-data schema (confirmed via HF probe; binds in § 3):
`{task_id, complete_prompt, instruct_prompt, canonical_solution, code_prompt,
test, entry_point, doc_struct, libs}`.

* The model completes `task_func` from `complete_prompt` (signature + docstring
  spec). The executor writes `<candidate solution>` + the `test` source (which
  defines `class TestCases(unittest.TestCase)` referencing `task_func`) and
  runs `unittest` in a fresh `-I` subprocess; PASS iff every test passes;
  stderr tail is the reflexion signal. Byte-identical A0/A1/B / K=5 / Merkle
  discipline to W109.

### § 2.3 — honest S1 caveat (registered, not hidden)

BigCodeBench composes well-known library APIs (the primitives are obviously in
training data); its RESISTANCE is at the **task+hidden-test level** — the
specific composed problems and their tests were authored/released 2024-06,
post-cutoff, exactly analogous to LiveCodeBench's contest-date anchoring of
familiar LeetCode primitives. This is a defensible time-anchored resistance,
but it is **construction+release-date resistance**, not the strict
contest-date anchoring of LCB. Registered as
`W110-L-BIGCODEBENCH-RELEASE-DATE-RESISTANCE-NOT-CONTEST-DATE-CAP`.

### § 2.4 — the single fallback (if BigCodeBench fails real-data soundness)

If the BigCodeBench real-data preflight FAILs S2 on real data (gold golds do
not pass in this environment even after the dependency install + gold-green
filter, i.e. < 30 gold-green problems), pivot IN-milestone to the **disjoint,
strictly-later LiveCodeBench `release_v6` window** (the 33 functional problems
W108 did NOT pilot; dates 2025-02-16…2025-04-05; corpus already cached + SHA-
pinned `bb4c364f…`; executor proven at W108). This is the cleanest available
real fallback (proven executor, $0 fetch) — explicitly framed as a WEAKER
"is the FAIL early-2025-LCB-window-specific?" probe (NOT a genuinely-different
benchmark), used ONLY if no genuinely-different resistant battlefield can be
made structurally clean enough in-milestone. If even that cannot be made
clean, record an honest no-go.

---

## § 3 — BigCodeBench real-data fetch / schema-confirm / gold-green rule (LOCKED; pins filled at fetch — NIM-free)

The fetch is executed in-milestone (pyarrow + HF egress available) via
`scripts/fetch_w110_bigcodebench_corpus.py`. **Pinned provenance (filled after
the NIM-free fetch, before the pilot):**

* dataset `bigcode/bigcodebench` (BigCode, 2024-06 — contamination-RESISTANT,
  post-cutoff release; C7 = A-grade by release-date anchoring per § 2.3).
* split `v0.1.4`; HF parquet `refs/convert/parquet` resolve path
  `default/v0.1.4/0000.parquet`.
* parquet shard `0000.parquet` (2 362 110 B, SHA-256
  `d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267`);
  **1140 problems** (FILLED at fetch).
* materialized `~/.cache/coordpy/bigcodebench-v0_1_4.jsonl` (4 982 979 B);
  **SHA-256 `ca4f352e68ec06111ba807f55802914339f4d23a90eb71989126359cefb3b018`**
  (FILLED at fetch; the loader REFUSES on mismatch / missing cache / schema
  mismatch — the W102 silent-degeneration guard).
* **Executor environment (pinned):** the `unittest` oracle runs under
  `~/.cache/coordpy/bcb_venv/bin/python` — a `--system-site-packages` venv
  carrying the BigCodeBench library stack (numpy/pandas/scipy/sklearn/
  matplotlib + seaborn/pyparsing/regex/django/click/nltk/bs4/openpyxl/
  statsmodels/lxml/Pillow/textblob/folium/holidays/geopy), installed into the
  venv's OWN site so the `-I` subprocess (which drops only the USER site)
  imports them. Cap `W110-L-BIGCODEBENCH-EXECUTOR-V1-EXEC-NAMESPACE-NOT-FILE-
  MODULE-CAP`: the executor exec's the candidate into a namespace (not a file
  module), so the ~4 tasks whose tests re-import the solution by module name
  fail gold-green and are dropped (never false-passes).

**Gold-green rule (the BigCodeBench-specific executor-cleanness guard, the
W108/W109 analogue):** the preflight runs each problem's `canonical_solution`
through the real `unittest` executor IN THIS ENVIRONMENT and keeps ONLY
problems whose gold passes (a `gold_green` subset). Gold-green filtering is an
EXECUTOR-ENVIRONMENT property (missing libs / flaky env tests), NOT a model
outcome — A0/A1/B have not run — so it is anti-cheat-safe and is the analogue
of the W108 `func_name`-resolved filter and the W109 wrapper-tolerance check.
The slice (§ 4) is selected ONLY from `gold_green`. Required-libs install is a
NIM-free operator step recorded in the fetch playbook.

**Residual rule:** the LIVE A1@K=5 residual is NOT pre-measured (no
BigCodeBench sidecar in-repo). It is measured BY the cheap pilot (gate G2),
exactly as for LiveCodeBench at W108 and APPS at W109.

---

## § 4 — Gold-path correctness bar before any pilot (LOCKED — NIM-free)

NO NIM is spent until ALL hold (verdict at
`results/w110/bigcodebench_preflight/preflight_verdict.json`):

1. `tests/test_w110_bigcodebench_reflexion_bench_v1.py` green (full-bench
   gold-path A0=A1=B=1.0 with a stub gen returning the gold; reflexion-rescue
   MLB exercise; slice-selector determinism + stratification; REAL-corpus →
   locked-slice-CID binding) + `tests/test_w110_bigcodebench_loader_v1.py` green
   (SHA-pin refuse; schema-shape refuse; call-shape parse).
2. **P1 corpus integrity:** the SHA-pinned loader reads 1140 problems; schema
   matches the § 3 fields.
3. **P2 executor self-test:** synthetic gold (top-level `task_func`) PASS,
   wrong FAIL, infinite-loop TIMEOUT, AND ≥ 1 REAL `canonical_solution` PASSes
   / a corrupted gold FAILs on the live corpus (no false-pass on real data).
4. **P3 loader real-data + gold-green:** `gold_green` ≥ 30 problems (the gold
   passes its own `test` in this environment); contamination framing recorded
   (C7 = A-grade release-date resistance, 2024-06; the second resistant
   benchmark).
5. **P4** deterministic outcome-blind 30-slice from `gold_green` reproduces the
   pinned slice CID `b69bf3a0999f0cdc2ccb097d2a67e3100095fda07bce47d4da8a7e840bbfd66a`;
   the pilot `--dry-run` reproduces it.

**RESULT (FILLED — NIM-free; `results/w110/bigcodebench_preflight/preflight_verdict.json`,
verdict CID `6be9fc8e4b674f955471a6e6d3b2337d0e5faf1aa3dbb1f15a6b7af84db1d8dd`):
OVERALL PASS — pilot EARNED.** P1 1140 problems; P2 all five executor
self-tests pass (synthetic gold PASS / wrong FAIL / infinite-loop TIMEOUT /
REAL gold PASS / corrupted gold FAIL — no false-pass on real data);
P3 **gold_green = 968/1140** (971 gold-pass, 3 excluded ≥20 s by the
wall-stability guard; dropped 99 missing-dep + 70 non-dep; ≫ the 30 needed);
P4 30-slice (buckets libs2:13 / libs3plus:17) deterministic.

**Executor headlessness (corrected after a window-popping iteration):** an
initial run used the macOS *interactive* matplotlib backend (360 of 1140
BigCodeBench tasks plot), which popped GUI windows AND risked a blocking
`plt.show()` falsely TIMING-OUT correct chart solutions. The executor now
forces the headless **`Agg`** backend (`MPLBACKEND=Agg` in-process + in the
subprocess env). Re-preflighting under Agg recovered **+32 gold-green** (936 →
968) — confirming the interactive backend had been falsely failing chart
solutions — and the wall-stability guard (drop golds ≥ 20 s) made the slice CID
**reproducible** across runs (`b69bf3a0…`; the flaky-slow BigCodeBench/0 is
correctly excluded). Cap `W110-L-BIGCODEBENCH-EXECUTOR-V1-HEADLESS-AGG-FIX`.

If any regress, the pilot is NOT launched; pivot per § 2.4 or honest no-go.

---

## § 5 — BigCodeBench cheap-pilot gates (LOCKED — Lane α, if earned)

**Slice (G1, pre-committed):** the deterministic outcome-blind `n_libs`-
stratified 30-problem slice from `select_bigcodebench_slice_v1` over
`gold_green`; slice CID
`b69bf3a0999f0cdc2ccb097d2a67e3100095fda07bce47d4da8a7e840bbfd66a` (FILLED;
buckets libs2:13 / libs3plus:17). The pilot consumes the EXACT
`slice_task_ids` from the preflight verdict.

**A0 / A1 / B (byte-identical mechanism to W89/W103/W105/W108/W109):**

* **A0** — stock single-shot at T=0.0 (1 call/problem).
* **A1** — first-pass-among-K=5 self-consistency at T=0.7 (5 calls).
* **B** — sequential-reflexion-K=5 at T=0.7, each turn conditioned on the
  cumulative (candidate, executor-stderr) history (5 calls).
* Budget byte-exact (A1 and B both K=5; no early-stop); same model on all arms;
  executor truth = subprocess `unittest` exit 0 iff every test passes; NO
  LLM-as-judge.

**Run:** 1 seed (110001) × 30 problems × K=5 = **330 NIM calls** at
`meta/llama-3.3-70b-instruct` (the SAME W89/W105/W108/W109 class — so the
contamination contrast is clean, single-class). A ≤2-problem canary (~22 calls)
runs first to validate the live path on real BigCodeBench output.

**The 9 Phase-2 gates + MLB sub-gates (verbatim from W103/W104/W105/W108/W109):**

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

**What a W110 BigCodeBench PASS entitles (and does NOT):** a
`PASS_MECHANISM_DRIVEN` entitles exactly: "the W89 mechanism beats same-budget
self-consistency on a SECOND contamination-RESISTANT code benchmark
(BigCodeBench, 2024-06 release-date-anchored) at cheap-pilot scale at 70B —
showing the W108 LiveCodeBench FAIL is NOT general to contamination-resistant
code." It does **NOT** add a retirement (that needs a Phase-3 multi-seed bench,
W111+), does **NOT** prove the absence of a contamination effect (one
resistant PASS + one resistant FAIL is a split, not a clean result), and does
**NOT** overwrite the W108 FAIL or the W109 APPS control.

---

## § 6 — Lane β: second-resistant interpretation rule (LOCKED — pre-committed claim-change)

`coordpy.contamination_resistant_interpretation_v1.interpret_second_resistant_result_v1`
takes the W110 BigCodeBench outcome + the fixed W108 LCB FAIL + W109 APPS PASS
and returns the pre-committed claim implication. The branches:

* **W110 resistant FAIL** (B − A1 < +5 pp, OR MLB-2 < 33 % with margin < +5 pp)
  ⇒ **the contamination-confound STRENGTHENS toward a finding**: the W89
  mechanism now FAILs on TWO genuinely-different contamination-RESISTANT code
  benchmarks (LiveCodeBench 2025 + BigCodeBench 2024) while PASSing on THREE
  contamination-EXPOSED HumanEval-family/APPS benchmarks. The boundary tightens
  to **"contamination-EXPOSED-specific at 70B"**; the W108 FAIL is shown
  GENERAL, not LCB-specific. Still NOT proof (single-seed each; 2 resistant
  points), and APPS stays exposed-control. `COO-9`'s code-superiority
  GENERALISATION charter is substantially answered (negatively) for
  contamination-resistant code.
* **W110 resistant PASS_MECHANISM_DRIVEN** (9/9 + both MLB) ⇒ **the
  contamination-confound WEAKENS materially**: a clean mechanism-driven
  same-budget win exists on contamination-RESISTANT code, so the W108 FAIL was
  **LCB-SPECIFIC** (benchmark-idiosyncratic), not a contamination effect. This
  EARNS a Phase-3 multi-seed BigCodeBench retirement bench (W111) — the only
  path to a contamination-resistant THIRD retirement. NOT itself a retirement.
* **W110 resistant PASS_NON_MECHANISM_DRIVEN** (9/9 gates but an MLB fails) ⇒
  **weak/ambiguous**: the margin exists but the mechanism is not cleanly
  load-bearing on invocation; register the cap; the confound stays
  SUPPORTED-not-proven; W111 weighs a multi-seed de-noise (usually low value —
  W100/W104 discipline) vs a different resistant benchmark. Does NOT earn a
  Phase-3 bench by itself.

APPS stays CONTROL-only in all branches (never a retirement, never
publication-grade). No further APPS NIM (no verdict-changing power; the
+16.67 pp margin is already large and clean).

---

## § 7 — graphify deliverables (LOCKED)

* Refresh at start (`graphify update .`; confirm built from current HEAD —
  done: `1e8f131f`, "No code-graph topology changes detected" ⇒ already
  current).
* Re-ingest after adding the BigCodeBench bench + interpretation modules.
* Use concretely: `graphify query` for the retirement-claim + contamination-
  boundary surface; `graphify explain` on `bigcodebench_loader_v1`,
  `bigcodebench_executor_v1`, `bigcodebench_reflexion_bench_v1`,
  `contamination_resistant_interpretation_v1`; `graphify path` between the
  BigCodeBench bench and the W89/W105/W108/W109 benchmark line (sibling check);
  `graphify affected` on the loader / truth surfaces.
* Refresh at end and record exactly what graphify changed in file selection /
  understanding (`docs/RESULTS_W110_*` + the milestone summary).

---

## § 8 — W111 branch logic (LOCKED)

Driven by the W110 BigCodeBench pilot outcome (Lane α) via the § 6 rule:

* **W110 resistant PASS_MECHANISM_DRIVEN** ⇒ **W111 = BigCodeBench Phase-3
  multi-seed retirement bench** (3 seeds × 100 × K=5; mirrors W103→W105) — the
  earned path to a contamination-RESISTANT THIRD retirement. 405B + Llama-3.1
  stay closed.
* **W110 resistant FAIL** ⇒ **W111 = register the tightened
  "contamination-EXPOSED-specific at 70B" boundary** (two resistant FAILs); the
  honest next move is a DIFFERENT mechanism (not a re-run of a capped/frozen
  line), OR acceptance of the tightly-bounded two-retirement
  contamination-exposed claim. `COO-9` generalisation charter answered
  negatively for resistant code.
* **W110 resistant PASS_NON_MECHANISM_DRIVEN** ⇒ **W111 weighs** a multi-seed
  BigCodeBench de-noise (low value) vs a THIRD genuinely-different resistant
  benchmark; no Phase-3 entitlement.
* **Pilot could not launch** (NIM unavailable / impractical in-session) ⇒ W110
  closes preflight-earned-but-operator-gated; **W111 = launch the earned
  BigCodeBench pilot** (no new earning work needed).
* **BigCodeBench failed real-data soundness ⇒ § 2.4 fallback fired** ⇒ W111
  branch logic keys on the fallback's outcome under the same § 6 rule.

---

## § 9 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION ==
  "coordpy.sdk.v3.43"`; no PyPI publish; `coordpy/__init__.py` untouched.
* New modules are explicit-import only: `coordpy.bigcodebench_loader_v1`,
  `coordpy.bigcodebench_executor_v1`, `coordpy.bigcodebench_reflexion_bench_v1`
  (the second resistant bench), `coordpy.contamination_resistant_interpretation_v1`
  (the Lane β rule). New scripts: `fetch_w110_bigcodebench_corpus.py`,
  `run_w110_bigcodebench_preflight.py`, `run_w110_bigcodebench_pilot.py`.

---

## Honest framing

W110, on a BigCodeBench Phase-2 `PASS_MECHANISM_DRIVEN`, entitles exactly: "the
W89 mechanism beats same-budget self-consistency on a SECOND
contamination-RESISTANT code benchmark at cheap-pilot scale at 70B, showing the
W108 LiveCodeBench FAIL was LCB-specific, not general." On a FAIL, the
contamination-confound STRENGTHENS (two resistant FAILs) and the boundary
tightens to contamination-EXPOSED-specific at 70B. Either way the boundary gets
SHARPER. It is NOT a retirement (Phase-3 is W111+), NOT proof either way (single
-seed; one resistant PASS + one resistant FAIL is a split), and does NOT change
the cross-class / cross-scale-UP / cross-modal boundaries. The two confirmed
retirements remain W89 + W105. APPS stays exposed-control. The 405B + Llama-3.1
branches stay closed.
