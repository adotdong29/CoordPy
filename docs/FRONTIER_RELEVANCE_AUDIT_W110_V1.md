# Frontier-relevance audit — W110 (supplement to W108/W109 V1)

> Extends `docs/FRONTIER_RELEVANCE_AUDIT_W109_V1.md`; all prior classifications
> remain in force. Classifies the W110 artifacts as active frontier arsenal /
> useful control-baseline / dead direction / anti-pattern, so future milestones
> reuse what is load-bearing and avoid re-running what is capped.

## Discipline status (#20 consecutive — W93–W110)

W110 executed the pre-committed `docs/RUNBOOK_W110.md` three-lane branch logic;
the BigCodeBench cheap pilot was EARNED on real data and returned a clean
Phase-2 FAIL. W110's distinguishing additions:

1. **Real-data candidate REJECTION at $0** — the COO-9 charter's named candidate
   (SWE-bench-lite) was rejected on structural grounds (synthetic in-repo
   scaffolding; Docker/per-repo-env; multi-file patches break K=5), and a
   look-alike (LiveBench-coding) was killed on a real-data probe (it IS
   LiveCodeBench repackaged). No NIM spent to learn this.
2. **In-milestone executor correctness fix** — caught a matplotlib
   interactive-backend bug (popped GUI windows + falsely TIMED-OUT chart
   solutions); forced headless Agg; recovered +32 gold-green.
3. **Second contamination-resistant FAIL** — answers the W108
   LCB-specific-vs-general question (general), reported without spin.

## Active frontier arsenal (reusable, load-bearing)

* **`coordpy.bigcodebench_loader_v1` / `bigcodebench_executor_v1` /
  `bigcodebench_reflexion_bench_v1`** — the BigCodeBench second-resistant line
  (A0/A1/B byte-identical in shape to W89/W105/W108/W109; deterministic
  `unittest`-oracle subprocess executor under headless Agg; gold-green +
  wall-stability slice). Reusable for any `unittest`-shaped code benchmark.
* **`coordpy.contamination_resistant_interpretation_v1`** — the canonical
  9-gate+MLB evaluator + the falsifiable second-resistant interpretation rule
  (FAIL→strengthens / PASS_MECH→LCB-specific+earns-Phase3 / PASS_NON→unchanged).
  Reusable single source of truth for Phase-2 gate evaluation.
* **`scripts/fetch_w110_bigcodebench_corpus.py`** — the HF parquet→pinned-JSONL
  materializer with libs-string normalization (the W109 pattern, extended).
* **The headless-Agg + wall-stability executor pattern** — `MPLBACKEND=Agg`
  in-process + subprocess env, and a gold-runtime guard for reproducible
  pools. Reusable for ANY matplotlib/plotting-heavy benchmark.
* **The contamination 2×2 with a second resistant point**
  (`docs/CONTAMINATION_CONTROL_FRAMING_W110_V1.md`) — now 3 exposed PASS vs 2
  resistant FAIL; the active instrument for the confound question.

## Useful control / baseline-only (NOT frontier superiority)

* **BigCodeBench v0.1.4 gold-green subset at 70B with K=5 same-budget
  reflexion** — empirically capped (`W110-L-BIGCODEBENCH-REFLEXION-PHASE2-70B-CAP`).
  Stays in-repo as a re-runnable contamination-resistant battlefield (a third
  resistant benchmark, a different mechanism, or a cross-scale probe could reuse
  it) — but NOT a margin to cite.

## Dead directions (capped — do NOT re-run)

* **SWE-bench-lite as a clean same-budget cheap pilot** — structurally unfit at
  70B without a Docker/per-repo-environment harness that breaks the K=5
  single-artifact byte-exact budget; the in-repo `swe_*` scaffolding is
  synthetic. Do NOT present it as a near-term clean battlefield.
* **LiveBench-coding as a "different" benchmark** — it is LiveCodeBench
  repackaged; using it would be rerunning W108.
* **Contamination-RESISTANT same-budget code superiority via the W89 mechanism
  at 70B** — now **0/2** (LiveCodeBench 2025 FAIL + BigCodeBench 2024 FAIL);
  `W110-L-REFLEXION-FAILS-ON-CONTAMINATION-RESISTANT-CODE-GENERALLY-CAP`.
  Unproven and not to be presented as shown.
* **A multi-seed de-noise of either resistant FAIL** — NOT WARRANTED (a +0.00 /
  −3.33 pp point with weak MLB-2 cannot be de-noised into a +5 pp PASS; the W109
  de-noise rule generalises). Closed Llama-3.1 + 405B re-probe stay closed.

## Anti-patterns (reinforced + new at W110)

* **Bounded-context / compaction / prose-summary / token-compression** REMAIN
  explicit anti-patterns, NOT the frontier path. W110 did none of these.
* **NEW W110 anti-pattern**: treating an in-repo benchmark-SHAPE scaffold (the
  synthetic SWE MiniSWEBank) or a benchmark-shape AUDIT tool (`coordpy-import`)
  as a real same-budget executor. A real battlefield needs a real corpus + a
  real deterministic executor on the W89 A0/A1/B mechanism.
* **NEW W110 anti-pattern**: running a matplotlib-heavy benchmark executor under
  an interactive GUI backend — it pops windows AND can FALSELY time-out correct
  chart solutions via a blocking `plt.show()` (W110 lost 32 gold-green to this
  before the Agg fix). Always force `MPLBACKEND=Agg` in benchmark executors.
* **NEW W110 lesson**: a second resistant FAIL with MLB-1 PASS (reflexion
  genuinely invoked) + weak MLB-2 is a STRONGER negative than W108/W109 — the
  mechanism is exercised and still doesn't help. Read MLB-1 and MLB-2 together.

## Do not claim (W110 additions)

* That the W89 reflexion mechanism beats same-budget self-consistency on
  contamination-resistant code (now 0/2: LCB −3.33 pp; BigCodeBench +0.00 pp).
* That the W108 LiveCodeBench FAIL was LCB-specific (W110 shows it is general).
* That the contamination-confound is PROVEN (STRENGTHENED ≠ established; two
  single-seed resistant points; orthogonal difficulty not excluded).
* That BigCodeBench is perfectly contamination-resistant (release-date, not
  contest-date, anchoring).
* That W110 retires anything or weakens the W89/W105 retirements (it sharpens
  their boundary to contamination-EXPOSED-specific at 70B; it does not move
  them).
* That multi-agent context is "solved".

## Carry-forwards

* **Added (T):** `W110-T-BIGCODEBENCH-REAL-DATA-FETCH-PINNED`,
  `W110-T-BIGCODEBENCH-SECOND-RESISTANT-PREFLIGHT-EARNED`,
  `W110-T-CONTAMINATION-CONFOUND-STRENGTHENED-NOT-PROVEN`,
  `W110-T-BIGCODEBENCH-EXECUTOR-V1-HEADLESS-AGG-FIX`.
* **Added (L):** `W110-L-BIGCODEBENCH-REFLEXION-PHASE2-70B-CAP`,
  `W110-L-REFLEXION-FAILS-ON-CONTAMINATION-RESISTANT-CODE-GENERALLY-CAP`,
  `W110-L-BIGCODEBENCH-GOLD-GREEN-WALL-STABILITY-GUARD-CAP`,
  `W110-L-BIGCODEBENCH-RELEASE-DATE-RESISTANCE-NOT-CONTEST-DATE-CAP`,
  `W110-L-BIGCODEBENCH-EXECUTOR-V1-EXEC-NAMESPACE-NOT-FILE-MODULE-CAP`.
* **Retired (research retirements):** NONE. W89 + W105 remain the only two
  confirmed multi-seed same-budget multi-agent superiority retirements.

## Anchors

`docs/RUNBOOK_W110.md`; `docs/RESULTS_W110_BIGCODEBENCH_PHASE2_70B_V1.md`;
`docs/RESULTS_W110_MILESTONE_SUMMARY_V1.md`;
`docs/CONTAMINATION_CONTROL_FRAMING_W110_V1.md`;
`results/w110/bigcodebench_preflight/preflight_verdict.json`;
`results/w110/bigcodebench_pilot/.../bigcodebench_reflexion_bench_report.json`.
