# RESULTS — W110 BigCodeBench second-contamination-RESISTANT Phase-2 cheap pilot (70B)

**Verdict: `FAIL` (clean, NON-mechanism-driven). 7/9 Phase-2 gates; B − A1 = +0.00 pp; MLB-1 = 40.00% PASS, MLB-2 = 25.00% < 33% FAIL.**

This is the **SECOND** contamination-RESISTANT test of the W89
sequential-reflexion mechanism (the first was W108 LiveCodeBench 2025, also a
FAIL). On the contamination-RESISTANT BigCodeBench 2024 functional set at 70B,
sequential reflexion (B) does NOT beat same-budget self-consistency (A1) — they
TIE at 70.00%. Reflexion was genuinely invoked (40% of problems) but rescued
weakly (25%, = the W108 LiveCodeBench rescue rate, ~half the HumanEval-family
47–57%). **The W108 LiveCodeBench FAIL is therefore NOT LiveCodeBench-specific:
the mechanism fails on TWO genuinely-different contamination-RESISTANT code
benchmarks while PASSing on THREE contamination-EXPOSED ones.** The two
confirmed retirements (W89, W105) are UNCHANGED. W110 adds NO retirement.

---

## 1. Run identity (audit chain)

| Field | Value |
|---|---|
| Model | `meta/llama-3.3-70b-instruct` (the W89/W105/W108/W109 retirement class — clean single-class contrast) |
| Corpus | `bigcode/bigcodebench`, split `v0.1.4`, `refs/convert/parquet` shard `0000.parquet` (SHA `d9a4965821c9…`) |
| Materialized JSONL | `~/.cache/coordpy/bigcodebench-v0_1_4.jsonl` (4 982 979 B); **SHA-256 `ca4f352e68ec06111ba807f55802914339f4d23a90eb71989126359cefb3b018`** |
| Contamination | **RESISTANT** — BigCodeBench released 2024-06 (HF `createdAt` 2024-06-05), AFTER the ≈2024-01 Llama-3.x cutoff; novel library-composition tasks + post-cutoff `unittest` oracles. C7 = A-grade (release-date anchoring; `W110-L-BIGCODEBENCH-RELEASE-DATE-RESISTANCE-NOT-CONTEST-DATE-CAP`) |
| Gold-green pool | 968 / 1140 (971 gold-pass; 3 excluded ≥ 20 s by the wall-stability guard; 99 missing-dep, 70 non-dep dropped) |
| Pilot slice | 30 problems; n_libs buckets {libs2: 13, libs3plus: 17}; deterministic outcome-blind, gold-green only |
| Slice CID | `b69bf3a0999f0cdc2ccb097d2a67e3100095fda07bce47d4da8a7e840bbfd66a` (reproduced across two runs) |
| Preflight verdict CID | `6be9fc8e4b674f955471a6e6d3b2337d0e5faf1aa3dbb1f15a6b7af84db1d8dd` |
| Executor | `bigcodebench_executor_v1` — fresh `-I` subprocess running the row's `unittest` oracle under **headless `Agg`** (no GUI; no `plt.show()` blocking); deps in a `--system-site-packages` venv; NO LLM judge |
| Seed | 110001 (single-seed cheap pilot) |
| K (A1 and B) | 5, byte-exact; no early-stop |
| NIM calls | 330 (A0 1 + A1 5 + B 5 per problem × 30) |
| Wall | 6 381.5 s (~106 min; heavy HTTP-429 throttling survived) |
| Bench Merkle root | `128dfb191d048842bf5015cc5ce9b24290cb505a1e7b3ebf85b98a8bcd96194c` |

A0/A1/B are byte-identical in mechanism to W89/W103/W105/W108/W109
(`coordpy.bigcodebench_reflexion_bench_v1`, shape-identical to the APPS +
LiveCodeBench benches); only the corpus, executor (`unittest`-oracle vs
call-based), and the prompt (implement `task_func` from the `complete_prompt`
spec) differ. Executor truth = the subprocess `unittest` exit code.

---

## 2. Empirical result

| Arm | pass@1 |
|---|---|
| A0 (single-shot T=0) | **63.33 %** (19/30) |
| A1 (first-pass-among-K=5, T=0.7) | **70.00 %** (21/30) |
| B (sequential-reflexion-K=5, T=0.7) | **70.00 %** (21/30) |
| **B − A1** | **+0.00 pp** |
| B − A0 | +6.67 pp |

Per-problem A1-vs-B transition surface (30 problems):

| both pass | A1-only (B regressed) | B-only (B rescued past A1) | neither |
|---|---|---|---|
| 20 | **1** (BigCodeBench/26) | **1** (BigCodeBench/51) | 8 |

B rescued exactly ONE problem past A1 (BigCodeBench/51) and regressed on
exactly ONE (BigCodeBench/26) — a clean **net zero**. Of the 12 problems where
reflexion was invoked (B's attempt-0 failed), 3 were rescued by a later attempt
(BigCodeBench/16, /21, /51) — but 2 of those (/16, /21) were ALSO solved by A1,
so reflexion's UNIQUE contribution over same-budget self-consistency is the
single BigCodeBench/51, offset by the single regression.

---

## 3. The 9 Phase-2 gates + MLB sub-gates

| Gate | Pass | Value |
|---|---|---|
| G1 slice pre-committed | ✅ | slice CID pinned + reproduced by `--dry-run` |
| G2 A1 < 90 % | ✅ | 70.00 % (non-saturated — real headroom existed) |
| G3 B > A1 | ❌ | 70.00 % = 70.00 % (not strict) |
| G4 (B − A1) ≥ +5 pp | ❌ | +0.00 pp |
| G5 (B − A0) ≥ +5 pp | ✅ | +6.67 pp |
| G6 per-problem majority (≥ 16/30) | ✅ | 29/30 (B did not regress vs A1 on 29) |
| G7 budget byte-exact | ✅ | A1, B both K=5 |
| G8 audit chain re-derives | ✅ | per-call CIDs + per-seed/bench Merkle |
| G9 executor clean | ✅ | no-LLM-judge `unittest` subprocess (headless Agg) |
| MLB-1 invocation ≥ 33 % | ✅ | **40.00 %** (12/30 problems triggered reflexion) |
| MLB-2 rescue ≥ 33 % | ❌ | **25.00 %** (3/12 invocations rescued) |

**7/9 core gates pass; the two failures are G3 (B > A1) and G4 (margin), plus
the MLB-2 load-bearing sub-gate. Verdict = `FAIL` (NON-mechanism-driven).**

---

## 4. Honest interpretation (what this is, and is NOT)

**What it IS — a clean, well-formed, mechanism-EXERCISED resistant FAIL.** This
is a STRONGER resistant FAIL than W108. On W109 APPS, MLB-1 failed (23%
invocation — the model solved most exposed problems first-shot, so reflexion
was barely tested). On W110 BigCodeBench, **MLB-1 PASSES at 40%** — reflexion
WAS genuinely invoked on 12/30 problems — and it STILL failed to produce a
net advantage: its rescue rate was **25%** (3/12), identical to W108
LiveCodeBench (25%) and ~half the HumanEval-family rate (W103 47% / W105 56%).
So the FAIL is not a "didn't get exercised" artifact: on contamination-
resistant code the reflexion repair loop fires at a healthy rate but its
rescues are roughly cancelled by its regressions (net B − A1 = +0.00 pp).

**The single most important fact:** this is the SECOND contamination-resistant
test of the W89 mechanism, and it FAILed. Combined with W108:

| benchmark | vintage | B − A1 | MLB-2 rescue | verdict |
|---|---|---|---|---|
| HumanEval (W89) | EXPOSED 2021 | +5.56 pp | ~47% | RETIRED |
| HumanEval+ (W105) | EXPOSED 2021 | +7.00 pp | 55.62% | RETIRED |
| APPS (W109) | EXPOSED 2021 | +16.67 pp | 57.14% | PASS (control) |
| **LiveCodeBench (W108)** | **RESISTANT 2025** | **−3.33 pp** | **25%** | **FAIL** |
| **BigCodeBench (W110)** | **RESISTANT 2024** | **+0.00 pp** | **25%** | **FAIL** |

The exposed/resistant dissociation is now **3 PASS vs 2 FAIL**, on two
genuinely-different resistant benchmarks (a contest-problem set and a
library-composition set). **The W108 LiveCodeBench FAIL is NOT
LiveCodeBench-specific** — the W89 mechanism fails on contamination-resistant
code GENERALLY at 70B.

**What it is NOT — proof of a contamination confound.** Each resistant point is
single-seed (n=30). Two resistant FAILs + three exposed PASSes is a STRONG
dissociation, but the confound is still not *proven*: (a) the resistant
benchmarks could share a difficulty/structure property orthogonal to
contamination (both are "harder" than HumanEval-family in ways beyond vintage);
(b) single-seed margins carry variance. The contamination-confound moves from
**SUPPORTED (W109) → STRENGTHENED toward a finding (W110)**, NOT proven.

**What it does NOT change.** The two confirmed retirements (W89 +5.56 pp; W105
+7.00 pp) STAND — they are real same-budget results on their benchmarks. W110
adds NO retirement and retires NO research carry-forward. What it changes is the
**boundary: it now tightens to contamination-EXPOSED-specific at 70B** — the
W89/W105 same-budget reflexion superiority is demonstrated only on
contamination-EXPOSED HumanEval-family code, and FAILS on every
contamination-resistant benchmark tested (2/2). This makes the programme's
claim MORE carefully bounded and MORE defensible, not weaker.

---

## 5. Carry-forwards

**Added (theorem / infrastructure anchors):**
* `W110-T-BIGCODEBENCH-REAL-DATA-FETCH-PINNED` — fetched + SHA-pinned the real
  `bigcode/bigcodebench` v0.1.4 corpus (shard SHA `d9a4965821c9…`; JSONL SHA
  `ca4f352e…`; 1140 problems) via `scripts/fetch_w110_bigcodebench_corpus.py`.
* `W110-T-BIGCODEBENCH-SECOND-RESISTANT-PREFLIGHT-EARNED` — real-data preflight
  P1∧P2∧P3∧P4 PASS under the headless Agg executor (gold-green 968/1140; slice
  CID `b69bf3a0…` reproduced across runs; verdict CID `6be9fc8e…`).
* `W110-T-CONTAMINATION-CONFOUND-STRENGTHENED-NOT-PROVEN` — the W89 mechanism
  now FAILs on TWO genuinely-different contamination-RESISTANT code benchmarks
  (LiveCodeBench 2025 B−A1=−3.33 pp; BigCodeBench 2024 B−A1=+0.00 pp; both
  MLB-2=25%) while PASSing on THREE contamination-EXPOSED ones. The W108 FAIL is
  shown GENERAL, not LCB-specific. Confound SUPPORTED→STRENGTHENED, NOT proven
  (single-seed each; 2 resistant points; orthogonal-difficulty not excluded).
* `W110-T-BIGCODEBENCH-EXECUTOR-V1-HEADLESS-AGG-FIX` — forced `MPLBACKEND=Agg`
  after an initial interactive-backend run popped GUI windows AND falsely
  TIMED-OUT chart solutions; re-preflight recovered +32 gold-green (936→968).
  Reusable lesson for any matplotlib-heavy benchmark executor.

**Added (caps):**
* `W110-L-BIGCODEBENCH-REFLEXION-PHASE2-70B-CAP` — BigCodeBench v0.1.4
  gold-green subset, `meta/llama-3.3-70b-instruct`, 1 seed × 30 × K=5:
  B − A1 = +0.00 pp; 7/9 gates; MLB-1 40% PASS, MLB-2 25% < 33% FAIL. Clean
  FAIL, NON-mechanism-driven.
* `W110-L-REFLEXION-FAILS-ON-CONTAMINATION-RESISTANT-CODE-GENERALLY-CAP` — the
  W89 same-budget reflexion superiority is NOT demonstrated on
  contamination-resistant code on ANY of the two genuinely-different resistant
  benchmarks tested (LiveCodeBench 2025 + BigCodeBench 2024). The boundary is
  contamination-EXPOSED-specific at 70B. NOT a re-runnable margin-cap (a third
  resistant benchmark could test it, but the de-noise rule does NOT apply — a
  +0.00 pp / weak-MLB-2 point cannot be de-noised into a PASS).
* `W110-L-BIGCODEBENCH-GOLD-GREEN-WALL-STABILITY-GUARD-CAP` — gold-green is
  defined as gold-passes AND wall < 20 s, so the pool + slice CID are
  reproducible (a few pathologically-slow golds, e.g. BigCodeBench/0, flicker
  at the timeout boundary).
* `W110-L-BIGCODEBENCH-RELEASE-DATE-RESISTANCE-NOT-CONTEST-DATE-CAP` — the
  resistance is novel-composition + 2024-06 release-date anchoring, not the
  strict contest-date anchoring of LiveCodeBench.
* `W110-L-BIGCODEBENCH-EXECUTOR-V1-EXEC-NAMESPACE-NOT-FILE-MODULE-CAP` — ~4
  tasks whose tests re-import the solution as a file module fail gold-green and
  are dropped (never false-PASS).

**Rejected as the lead at $0 NIM (selection, not empirical caps):**
* SWE-bench-lite — structurally unfit (synthetic in-repo scaffolding; real
  instances need Docker/per-repo env; multi-file patches break K=5 byte-exact).
* LiveBench-coding — IS LiveCodeBench repackaged (`task: LCB_generation`); a
  re-run of LCB, not a genuinely-different benchmark.

**NOT retired:** the two confirmed retirements (W89, W105) — unchanged.

---

## 6. What W111 becomes

Per `docs/RUNBOOK_W110.md` § 8 (FAIL branch) + the Lane β rule
(`contamination_resistant_interpretation_v1`, `confound_direction=STRENGTHENS`,
`earns_phase3=False`): **W111 = register the tightened
contamination-EXPOSED-specific-at-70B boundary** and decide the honest next
move. The verdict-changing question W110 was built to answer ("is the W108 FAIL
LCB-specific or general?") is now ANSWERED — **general**. So the live options
are: (a) a DIFFERENT mechanism (not a re-run of a capped/frozen line) that
might beat same-budget self-consistency on contamination-resistant code; or
(b) acceptance of the tightly-bounded two-retirement
**contamination-EXPOSED-HumanEval-family** claim as the programme's honest
ceiling on code. A multi-seed de-noise of either resistant FAIL is NOT
WARRANTED (a +0.00 pp / −3.33 pp point with weak MLB-2 cannot be de-noised into
a PASS — variance, not mean — the W109 LCB de-noise rule generalises). The
closed Llama-3.1 + 405B branches stay closed; APPS stays exposed-control.
`COO-9`'s "second code battlefield" generalisation charter is now
substantially answered (negatively) for contamination-resistant code; `COO-9`
stays the lead path for the (a)/(b) decision unless a different code-line move
is forced.
