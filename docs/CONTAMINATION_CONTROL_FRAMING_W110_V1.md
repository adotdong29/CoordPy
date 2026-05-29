# Contamination-control framing (W110 V1) — the SECOND contamination-resistant test

> **2026-05-29 (W110 Lanes β + γ).** Extends
> `docs/CONTAMINATION_CONTROL_FRAMING_W109_V1.md`. The single honest account of
> what the W110 second contamination-RESISTANT benchmark (BigCodeBench) can and
> cannot establish, with the interpretation branches **pre-committed before the
> pilot verdict** (the integrity point). Where this doc and any other disagree
> on the STATUS of a claim, `docs/THEOREM_REGISTRY.md` is authoritative; for the
> current position, `docs/RESEARCH_STATUS.md`.

## Why this doc exists

After W108 + W109 the contamination 2×2 had its decisive cell — a PASS on
contamination-RESISTANT data — **empty**, with W108's one resistant attempt
(LiveCodeBench 2025) a FAIL (B − A1 = −3.33 pp; MLB-2 = 25 %), and W109's
exposed control (APPS 2021) a PASS-on-margin (+16.67 pp, non-mechanism-driven).
That made the contamination-confound **SUPPORTED but NOT proven**, on a single
exposed/resistant pair. A single resistant FAIL cannot tell apart:

* **(a) a contamination confound** — the W89/W105 advantage is partly
  benchmark-familiarity, so it vanishes on clean post-cutoff data; from
* **(b) LiveCodeBench-specific idiosyncrasy** — the FAIL is about LCB's
  particular contest-problem distribution, not contamination in general.

**W110 runs the test that discriminates (a) from (b): a SECOND,
genuinely-different contamination-RESISTANT code benchmark (BigCodeBench 2024),
same mechanism, same K=5 budget, same model class.**

## The 2×2 (now with a second resistant point)

| | contamination-EXPOSED (≤ 2024 cutoff) | contamination-RESISTANT (≥ 2024 release) |
|---|---|---|
| **mechanism PASSes** | HumanEval (W89 ✅ +5.56 pp), HumanEval+ (W105 ✅ +7.00 pp), APPS (W109 ✅ +16.67 pp, non-mech) | *(the publication-grade-strong cell — W108 LCB FAILed; W110 BigCodeBench is the SECOND attempt)* |
| **mechanism FAILs** | MBPP+ V2 (W102 ❌) | LiveCodeBench 2025 (W108 ❌ −3.33 pp); **BigCodeBench 2024 (W110 — see § outcome)** |

W110's job is to put a SECOND token in the resistant column. With two resistant
points instead of one, "LCB-specific" (b) becomes testable: if the mechanism
also fails on BigCodeBench, (b) is implausible and (a) strengthens; if it
passes cleanly, the W108 FAIL was (b) and (a) weakens.

## Why BigCodeBench is the right second resistant benchmark

Selected over SWE-bench-lite and LiveBench-coding under the locked W110
selection rule (`docs/RUNBOOK_W110.md` § 2, S1∧S2∧S3∧S4 + feasibility), on
real-data probes ($0 NIM):

* **SWE-bench-lite (the COO-9 charter's named candidate) — REJECTED.** Its
  in-repo scaffolding (`coordpy/_internal/tasks/swe_*`) is explicitly *"Not
  SWE-bench end-to-end"* — a synthetic 4-instance MiniSWEBank on the old
  `STRATEGY_NAIVE/ROUTING/SUBSTRATE` ablation, not the W89 A0/A1/B mechanism;
  real instances need a Docker / per-repo-environment harness (fails S2
  executor-cleanness) and produce multi-file patches that break the K=5
  single-artifact byte-exact budget (fails S3 same-budget). Its gold + test
  patches are public GitHub PRs (S1 contamination weak, not time-anchored).
* **LiveBench-coding — REJECTED on a real-data probe.** It IS LiveCodeBench
  repackaged (`task: LCB_generation`; `citation: …via livecodebench`; identical
  `public_test_cases`/`private_test_cases` schema). Using it would be "rerunning
  LiveCodeBench," which the milestone explicitly forbids (fails S4
  genuine-difference). Killed cheaply at $0 NIM.
* **BigCodeBench v0.1.4 — SELECTED.** Released 2024-06 (HF `createdAt`
  2024-06-05), AFTER the ≈2024-01 Llama-3.x cutoff ⇒ time-anchored
  contamination-resistant (S1); a deterministic `unittest.TestCase` oracle ⇒
  clean no-LLM-judge subprocess executor (S2); a single `def task_func(...)`
  completion graded at K=5 byte-exact ⇒ same-budget A0/A1/B (S3); novel
  library-composition tasks authored by the BigCode project, a genuinely
  different construction from LCB's contest scraping (S4).

**Honest S1 caveat** (`W110-L-BIGCODEBENCH-RELEASE-DATE-RESISTANCE-NOT-CONTEST-DATE-CAP`):
the library primitives BigCodeBench composes are obviously in-training; its
resistance is the **novel composition + 2024-06 release date**, not the strict
contest-date anchoring of LiveCodeBench. It is a defensible second resistant
point of a slightly different *kind* — which, for a discrimination test, is a
feature (two different flavours of resistance, not a near-duplicate of LCB).

## Pre-committed interpretation (locked BEFORE the verdict)

Per `coordpy.contamination_resistant_interpretation_v1` + `RUNBOOK_W110` § 6.
The branch is selected purely by the Phase-2 verdict label:

* **W110 BigCodeBench FAIL** ⇒ the W89 mechanism FAILs on TWO
  genuinely-different contamination-RESISTANT code benchmarks (LCB 2025 +
  BigCodeBench 2024) while PASSing on THREE contamination-EXPOSED
  HumanEval-family/APPS benchmarks. The confound **STRENGTHENS toward a
  finding** (still not proof — single-seed each, two resistant points); the
  boundary tightens to **contamination-EXPOSED-specific at 70B**; the W108 FAIL
  is shown GENERAL, not LCB-specific. NOT a retirement.
* **W110 BigCodeBench PASS_MECHANISM_DRIVEN** ⇒ a clean mechanism-driven
  same-budget win exists on contamination-RESISTANT code, so the W108 LCB FAIL
  was **LCB-SPECIFIC**; the confound **WEAKENS materially**. This EARNS a
  Phase-3 multi-seed BigCodeBench retirement bench (W111) — the only path to a
  contamination-resistant THIRD retirement. NOT itself a retirement.
* **W110 BigCodeBench PASS_NON_MECHANISM_DRIVEN** ⇒ margin without clean
  load-bearing mechanism; register the cap; the confound stays
  **SUPPORTED-not-proven**; no Phase-3 entitlement.

In ALL branches: the two confirmed retirements (W89, W105) STAND; APPS stays
contamination-EXPOSED control-only; the boundary gets SHARPER, not fuzzier.

## The honesty rules (do / do-not)

* **DO say:** "W110 ran a SECOND contamination-resistant benchmark
  (BigCodeBench 2024) to test whether the W108 LiveCodeBench FAIL was
  LCB-specific or general."
* **DO say (FAIL):** "the mechanism fails on two genuinely-different resistant
  benchmarks; the confound strengthens; the advantage is
  contamination-exposed-specific at 70B — not proven, two single-seed points."
* **DO say (PASS_MECHANISM_DRIVEN):** "a clean resistant same-budget win exists;
  the W108 FAIL was LCB-specific; this earns a Phase-3 retirement bench (W111)."
* **DO NOT say:** a single W110 PASS proves the absence of a contamination
  effect (one resistant PASS + one resistant FAIL is a split, not a clean
  result); that W110 retires anything (Phase-3 is W111+); that BigCodeBench is
  perfectly contamination-resistant (release-date, not contest-date, anchoring).
* **DO NOT say:** multi-agent context is solved, or that the mechanism
  generalises across code benchmarks broadly (MBPP+ ❌, and at least LCB ❌).

## § W110 empirical result (FILLED AT PILOT VERDICT)

<!-- PILOT_RESULT_PLACEHOLDER: filled from
results/w110/bigcodebench_pilot/.../bigcodebench_reflexion_bench_report.json
after the 1 seed × 30 × K=5 = 330-call run completes. -->

## Anchors

* `docs/RUNBOOK_W110.md` — the pre-commit contract (locked before NIM).
* `docs/RESULTS_W110_BIGCODEBENCH_PHASE2_70B_V1.md` — the full W110 verdict.
* `docs/CONTAMINATION_CONTROL_FRAMING_W109_V1.md` — the W108/W109 2×2 origin.
* `docs/RESULTS_W108_LIVECODEBENCH_PHASE2_70B_V1.md` — the first resistant FAIL.
* `docs/THEOREM_REGISTRY.md` — authoritative claim status.
* `docs/RESEARCH_STATUS.md` — canonical current truth.
