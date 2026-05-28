# Contamination-control framing (W109 V1) — APPS vs LiveCodeBench

> **2026-05-28 (W109 Lane γ).** The single honest explanation of the
> contamination-control logic the programme is now running, and exactly what
> each outcome can and cannot establish. Where this doc and any other disagree
> on the STATUS of a claim, `docs/THEOREM_REGISTRY.md` is authoritative; for
> the current position, `docs/RESEARCH_STATUS.md`.

## Why this doc exists

The two confirmed retirements of the W89 sequential-reflexion mechanism — W89
(base HumanEval, +5.56 pp) and W105 (HumanEval+, +7.00 pp) — are BOTH on
**contamination-EXPOSED** benchmarks: base HumanEval (2021) and HumanEval+ (the
same 2021 problems with extra tests). Both predate the Llama-3.x training
cutoff (≈2024-01-01) and are almost certainly inside the training corpus.

W108 ran the **first contamination-RESISTANT** test of the mechanism on
**LiveCodeBench** (2025, time-anchored, post-cutoff) and it **FAILed**
(B − A1 = −3.33 pp; MLB-2 = 25 %). That raised — but did not establish — a
**contamination-confound hypothesis**: that the W89/W105 reflexion-superiority
may be partly linked to benchmark familiarity rather than pure
reasoning-repair.

A single resistant FAIL cannot settle the question, because the LiveCodeBench
FAIL is equally consistent with (a) a contamination confound, (b)
benchmark-family difficulty unrelated to contamination, or (c) single-seed
cheap-pilot noise on a 1-problem margin. **W109 runs the control that
discriminates (a) from (b)/(c): the same mechanism, same budget, on a
contamination-EXPOSED but ALSO-not-HumanEval benchmark (APPS, 2021).**

## The 2×2 (the whole logic, on one grid)

The mechanism + same-budget K=5 contract are held byte-identical across all
cells. The only axes are **benchmark vintage** (exposed vs resistant) and the
**Phase-2 outcome** (PASS vs FAIL).

| | contamination-EXPOSED (≤ 2024 cutoff) | contamination-RESISTANT (≥ 2025) |
|---|---|---|
| **mechanism PASSes** | HumanEval (W89 ✅), HumanEval+ (W105 ✅), **APPS (W109 — this milestone)** | *(none yet — the genuinely strong cell)* |
| **mechanism FAILs** | MBPP+ V2 (W102 ❌ — same-family-adjacent) | **LiveCodeBench 2025 (W108 ❌)** |

Reading the grid:

* **The only publication-grade-strong cell is top-right** (PASS on
  contamination-resistant data). It is currently **empty**, and W108's one
  attempt there FAILed.
* **APPS sits in the top-left/bottom-left column** (exposed). It is a CONTROL,
  never a third retirement: a PASS there cannot be publication-grade because
  the model may have memorised 2021 APPS.

## What each W109 APPS outcome establishes (pre-committed reading)

* **APPS PASS (mechanism-driven)** ⇒ the mechanism recovers on
  contamination-EXPOSED APPS while FAILing on contamination-RESISTANT
  LiveCodeBench. This is a **double dissociation by vintage** and is **evidence
  CONSISTENT with the contamination-confound hypothesis** — but NOT proof. One
  exposed/resistant pair, single-seed, cannot rule out that APPS and HumanEval
  share a difficulty/structure property orthogonal to contamination. It does
  NOT add a retirement and does NOT overwrite the W108 FAIL.
* **APPS FAIL** ⇒ the mechanism fails on a contamination-EXPOSED benchmark too,
  so exposure is NOT sufficient for the reflexion advantage. The confound
  hypothesis **WEAKENS materially**, and the boundary tightens: the
  superiority is **HumanEval-family-specific** at 70B (works on base
  HumanEval + HumanEval+, fails on MBPP+ V2, LiveCodeBench, and APPS).

Either way, **the boundary gets sharper, not fuzzier** — which is the point.

## The honesty rules (do / do-not)

* **DO say:** "two confirmed retirements (W89 + W105), both on
  contamination-EXPOSED HumanEval-family at 70B; the first contamination-
  RESISTANT test (LiveCodeBench 2025) FAILed; W109 runs a contamination-
  EXPOSED control (APPS 2021)."
* **DO say (APPS PASS):** "consistent with a contamination-confound; not proof."
* **DO say (APPS FAIL):** "confound weakened; the advantage is
  HumanEval-family-specific at 70B."
* **DO NOT say:** an APPS PASS proves contamination, retires anything, is
  publication-grade, or overwrites the LiveCodeBench FAIL.
* **DO NOT say:** the contamination-confound is established (it is an OPEN
  hypothesis until a contamination-RESISTANT PASS exists or a larger control
  battery converges).
* **DO NOT say:** multi-agent context is solved, or that the mechanism
  generalises across code benchmarks broadly (MBPP+ ❌, LiveCodeBench ❌).

## Why APPS, specifically, is the right control

APPS (Hendrycks et al., 2021) shares the **call-based functional shape** with
LiveCodeBench (LeetCode-style `fn_name` entry, decoded-argument calls,
deterministic subprocess executor, no LLM judge) — so the executor, the
A0/A1/B mechanism, the K=5 budget, and the 9 gates + MLB sub-gates are
byte-identical to the W108 LiveCodeBench bench. The ONLY material difference
from W108 is **vintage** (2021 exposed vs 2025 resistant). That makes APPS the
cleanest available contrast: it isolates the contamination axis while holding
the task shape, mechanism, and budget fixed.

Its limitation is exactly its role: 2021 vintage ⇒ contamination-EXPOSED
(C7 = C) ⇒ control evidence only, never the publication-grade time-anchored
claim surface that only a contamination-RESISTANT PASS could provide.

## W109 empirical result (APPS contaminated-control cheap pilot)

*(This section is finalized after the earned cheap pilot completes; see
`docs/RESULTS_W109_APPS_CONTROL_PHASE2_70B_V1.md` for the full verdict + the
locked pre-commit `docs/RUNBOOK_W109.md`. Corpus: `codeparrot/apps`
refs/convert/parquet @ `0f10e424…`, call-based subset 38 problems, JSONL SHA
`f6c44d76…`; slice CID `783687d6…`; 1 seed × 30 × K=5 = 330 NIM calls at
`meta/llama-3.3-70b-instruct`.)*

## Anchors

* `docs/RUNBOOK_W109.md` — the pre-commit contract (locked before NIM).
* `docs/RESULTS_W108_LIVECODEBENCH_PHASE2_70B_V1.md` — the resistant FAIL.
* `docs/CONSOLIDATED_CODE_RETIREMENT_NARRATIVE_V1.md` — the W89→W106 arc.
* `docs/THEOREM_REGISTRY.md` — authoritative claim status.
* `docs/RESEARCH_STATUS.md` — canonical current truth.
