# W101 — Frontier relevance audit V1 (supplement to W97 / W98 / W99 / W100)

> **2026-05-25.  Fifth supplement, extending the W97 / W98 /
> W99 / W100 frontier-relevance audits with the post-W100
> code-pivot.  The W97 / W98 / W99 / W100 audit classifications
> all remain in force VERBATIM; this supplement only:**
>
> 1. Records the promotion of `COO-9` to the lead path (was
>    already a frontier-relevant candidate; now the lead).
> 2. Adds the W101 second-code-benchmark-tournament + MBPP+
>    lead infrastructure to the *active frontier arsenal* column.
> 3. Re-asserts the cross-modal RealWorldQA arc as **frozen at
>    11B** in the dead-direction-with-cross-scale-cap column.
> 4. Re-asserts the W97 / W98 / W99 / W100 anti-pattern column
>    *verbatim*.
>
> No code is removed by this audit.  No version bump.  No PyPI
> publish.

## Why a supplement, not a rewrite

W100 V1 documented the classification as of the W100 90B Phase 2
closing moment.  W100's empirical evidence established two new
structural facts:

1. The W95-B0-family REPAIR via the B2 image-at-decision-boundary
   mechanism is restricted to the 11B regime; cross-scale 90B
   generalisation is NOT earned.
2. The cross-modal RealWorldQA arc is structurally restricted
   to the 11B regime AND the pre-committed Part H code-pivot
   triggers `COO-9` promotion.

W101's job is to **execute the code-pivot infrastructure** —
select the second code benchmark family, mine the W89 / W91
sidecars, build the loader + executor + bench + preflight, and
land the empirical preflight verdict that licenses (or denies)
the cheap NIM pilot.

This audit records that:

* The active frontier *lead* is now `COO-9` → MBPP+ (EvalPlus's
  hardened MBPP) per the W101 battlefield-selection ranking.
* The active backup is HumanEval+ (EvalPlus's hardened HumanEval),
  built only if MBPP+ preflight FAILs.
* The cross-modal arc is dead at the +5 pp Phase 2 bar at any
  scale on RealWorldQA.
* The W101 preflight + arsenal-mining infrastructure is itself
  frontier-relevant operating-system code (joins the W93 /
  W96-D / W99 / W100 preflight-discipline + cross-scale +
  multi-candidate-tournament-then-confirm stack).

## Active frontier arsenal — W101 additions

| Mechanism | Module(s) | Why frontier (with W101 evidence) |
|---|---|---|
| **W89 sequential reflexion B-pipeline** (unchanged from W89) | `coordpy.humaneval_reflexion_bench_v1` | Active frontier mechanism.  Retirement on HumanEval-70B established at +5.56 pp.  W101 extends the same mechanism to MBPP+ with the runbook's K=5 same-budget byte-exact contract preserved. |
| **MBPP+ loader** (NEW W101) | `coordpy.mbpp_plus_loader_v1` | Active frontier infrastructure.  Lifts the W91 ceiling-saturation cap (`W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP`) by re-targeting the bench at EvalPlus's hardened test surface.  SHA-pinned canonical EvalPlus release artifact with explicit operator-fetch step; the loader refuses to operate without an authoritative integrity check (W93 preflight-first discipline). |
| **MBPP+ executor** (NEW W101) | `coordpy.mbpp_plus_executor_v1` | Active frontier infrastructure.  Subprocess CPython executor that runs candidate against base + plus assertions in a single pass; three modes (base_and_plus / base_only / plus_only) for the cross-bench-stability probe + the canonical Phase 2 bench. |
| **MBPP+ reflexion bench** (NEW W101) | `coordpy.mbpp_plus_reflexion_bench_v1` | Active frontier infrastructure.  Wires the W89 sequential-reflexion mechanism with the MBPP+ loader + executor; A0 / A1 / B byte-identical mechanism shape; per-call sidecars + per-seed Merkle + bench Merkle re-derivation preserved. |
| **MBPP+ preflight harness** (NEW W101) | `coordpy.mbpp_plus_preflight_v1` | Active frontier operating-system piece.  Extends W93 5-gate harness with MBPP+-specific probes (P1 corpus integrity, P2 executor self-test on canonical solutions, P3 A1@K=5 failure-residual estimate via W91 sidecar re-execution + published EvalPlus drop Hoeffding lower bound, P4 decomposition argument) + W101 AddrW101 probes (AddrW101-P1 mechanism-load-bearing prior, AddrW101-P2 per-problem cluster structure, AddrW101-P3 cross-bench failure-residual stability, AddrW101-P4 anti-pattern guard). |
| **W101 arsenal-mining script** (NEW W101) | `scripts/run_w101_arsenal_mining.py` | Active frontier operating-system piece.  Re-executes W88 / W91 calls offline (no NIM) to produce per-(seed, task_id, arm) cluster surface; output is the structured failure-cluster JSON the preflight reads. |
| **W101 multi-candidate-tournament discipline** | `docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md` (this milestone) | Active frontier operating-system piece.  Codifies the 5-candidate × 8-criterion matrix the W94 / W96-D battlefield-selection rubric implied but did not explicitly write down; sets the LEAD + BACKUP convention W102+ inherits. |

## Useful baselines (W101 changes from W97 / W98 / W99 / W100)

| Mechanism | Module(s) | Classification | W101 status |
|---|---|---|---|
| `bounded_window_baseline_v{1,2,3}` | `coordpy/bounded_window_baseline_v*.py` | UNCHANGED — useful falsifier targets the substrate-coupled methods must beat. | Same. |
| `coordpy.mbpp_reflexion_bench_v1` (base MBPP) | `coordpy/mbpp_reflexion_bench_v1.py` | **Demoted to baseline-only** by W101.  W91 5-seed 70B run showed the per-seed strict majority cap (`W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP`); MBPP+ is the structural fix.  The base-MBPP module remains in-repo for regression / audit / cross-bench comparison but is NOT the active frontier code-side bench. | Stays in-repo; explicitly classified baseline-only. |
| `coordpy.humaneval_reflexion_bench_v1` (HumanEval base) | `coordpy/humaneval_reflexion_bench_v1.py` | **Retirement-anchor** — the W89 70B retirement on this bench remains the only confirmed multi-seed same-budget multi-agent superiority retirement.  Stays active frontier. | Same. |
| A0 / A1 baselines | unchanged | unchanged | unchanged |

## Historical artifacts (unchanged from W97 / W98 / W99 / W100 V1)

W90 / W92 / W88 (cross-modal code) / W81 / W83 / W84 unchanged.
Kept for regression / audit; not active path.

## Dead directions (W101 changes)

| Mechanism | Evidence against | NEW W101 status |
|---|---|---|
| **VLM-Verifier-Final-Turn as load-bearing rescue** | W96-C: 0/11 at 11B; 1/7 at 90B | UNCHANGED — refuted. |
| **W95-B0 free-text bullet extraction as sufficient on vision-bound benches** | W97 D2-B0 11B: B − A1 = −6.67 pp | UNCHANGED — refuted. |
| **B1 typed schema *with* `direct_answer_hint`** | W98 B1 11B: B − A1 = −6.67 pp via 5 multi-choice regressions | UNCHANGED — refuted. |
| **B4 typed schema *without* `direct_answer_hint`** | W99 B4 11B: B − A1 = −16.67 pp | UNCHANGED — refuted. |
| **Typed-extract-then-text-reason sub-family of W95-B0** | THREE empirical refutations at 11B | UNCHANGED — dead. |
| **W95-B0 family REPAIR via B2 mechanism (image-at-decision-boundary)** | W99 11B: PASS structurally; W100 90B: FAIL (cross-scale collapse; MLB-2 FAIL) | UNCHANGED — restricted to 11B regime only. |
| **Cross-modal RealWorldQA at the +5 pp Phase 2 bar at any scale** | W97 + W98 + W99 11B + W100 90B all FAIL the +5 pp Phase 2 bar | **EXTENDED W101**: the arc is frozen at 11B; W101 does NOT re-open it. |
| **Base MBPP at K=5 same-budget at 70B for retirement** | W91: per-seed strict majority FAILS 2/5 due to ceiling saturation | **NEW W101 classification**: capped via ceiling saturation; the surgical fix is MBPP+, which W101 builds. |

## Anti-patterns (NEVER promote as core strategy; baseline-only allowed) — UNCHANGED VERBATIM FROM W100

**The W97 / W98 / W99 / W100 anti-pattern list remains in force
VERBATIM in W101.**

| Anti-pattern | W101 status |
|---|---|
| Bounded context window as product thesis | UNCHANGED — anti-pattern; baseline-only. |
| Compaction / generic prose summarization as memory mechanism | UNCHANGED — anti-pattern; the W101 bench module's AddrW101-P4 probe explicitly scans for these tokens and refuses to license the cheap pilot if found. |
| Shallow token compression without structural reason | UNCHANGED. |
| Context-pruning theater | UNCHANGED. |
| "Cram less / truncate better" as frontier memory system | UNCHANGED. |
| LLM-as-judge in executor chain | UNCHANGED — the W101 MBPP+ executor is subprocess CPython; no LLM judges any candidate's correctness. |
| Selective retries | UNCHANGED — the W101 bench follows the W88 / W90 contract: no early-stop on PASS; every K=5 budget element runs to completion. |
| Single-seed pilots as retirement evidence | UNCHANGED — W101 cheap pilot is a 1-seed × 30-problem cheap pilot AT THE Phase 2 SIZE, not retirement evidence.  Phase 3 retirement requires multi-seed × multi-problem and is DEFERRED to W103+ if W101 + W102 cross-scale both PASS. |
| Architecture refinement by vibe | UNCHANGED — W101's MBPP+ choice is locked by the 5-candidate × 8-criterion battlefield-selection matrix BEFORE any code is built. |
| Inventing new candidates after the tournament selected a winner | UNCHANGED — the W101 tournament selected MBPP+ as LEAD + HumanEval+ as BACKUP; the cheap pilot attacks MBPP+ only. |
| Re-opening dead directions | UNCHANGED — the W95-B0 family + typed-extract sub-family + cross-modal RealWorldQA arc remain dead.  W101 does NOT re-open any of them. |

## What W101 cross-promotion is NOT

To pre-empt drift back toward commodity-LLM tricks under a new
name:

### W101 is NOT a cross-modal milestone

* Cross-modal RealWorldQA arc is frozen at 11B per the W100
  frontier audit.  W101 does NOT re-open it.
* If the W101 MBPP+ cheap pilot succeeds, the next milestone
  (W102) is the cross-SCALE confirmation of the *code-line*
  result — not a return to cross-MODAL.

### W101 is NOT a Phase 3 retirement attempt

* The cheap pilot is a 1-seed × 30-problem cheap-pilot AT THE
  Phase 2 SIZE.  Phase 3 retirement is the 3-seed × 100-problem
  × K=5 retirement bench analogous to W89 HumanEval; that lives
  in W103 if W102 cross-scale PASSes.

### W101 is NOT a NIM-spending milestone (by default)

* W101's deliverable is the discipline + infrastructure +
  ranked battlefield slate + empirically-grounded preflight
  verdict.  The cheap NIM pilot is conditional on the operator
  fetching MBPP+ data + the preflight re-running clean.

### A W101 cheap-pilot PASS at 70B is NOT a multi-benchmark same-budget retirement

* A PASS at 70B Phase 2 means the W89 mechanism extends to a
  SECOND benchmark family at the cheap-pilot scale.  Full
  retirement of `W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP` (or
  any new W101 carry-forward) requires Phase 3 multi-seed
  evidence.

## Honest classification of a W101 PASS / FAIL pattern

| W101 preflight + cheap-pilot outcome | What we earn | What we do NOT claim |
|---|---|---|
| Preflight 8/8 PASS + cheap pilot B − A1 ≥ +5 pp + MLB sub-gates clearing at 70B | The W89 sequential-reflexion mechanism extends to a SECOND published code benchmark family (MBPP+) at the cheap-pilot scale; the W101 carry-forward is the W102 cross-scale confirmation entitlement. | Multi-seed same-budget retirement on MBPP+ (Phase 3 needed). |
| Preflight 8/8 PASS + cheap pilot B − A1 < +5 pp | Mechanism is *partially* load-bearing on MBPP+ but does not clear the Phase 2 bar; carry-forward `W101-L-MBPP-PLUS-REFLEXION-PHASE2-70B-CAP`; W102 attacks HumanEval+ instead. | Anything Phase 3.  Anything about W89 generalisation if the mechanism fails on MBPP+ specifically. |
| Preflight 8/8 PASS + cheap pilot B − A1 ≥ +5 pp + MLB-2 FAILS | `PASS_NON_MECHANISM_DRIVEN`; cross-scale W102 NOT entitled per W96-C / W100 precedent. | Mechanism load-bearingness on MBPP+. |
| Preflight P3 FAILS at re-run (MBPP+ A1 ≥ 90 %) | MBPP+ saturates Llama-3.3-70B at K=5; the EvalPlus extra tests aren't hard enough at 70B; pivot to HumanEval+ in W102.  Adds carry-forward `W101-L-MBPP-PLUS-PREFLIGHT-P3-SATURATION-CAP`. | Anything about HumanEval+. |
| Preflight P1 / P2 FAIL at re-run (corpus + executor self-test) | Infrastructure bug in MBPP+ loader or executor; fix; re-run preflight; do NOT launch pilot. | Anything empirical. |

## What this supplement DOES NOT do

* It does NOT claim multi-agent context is solved.
* It does NOT claim the W101 MBPP+ cheap pilot will succeed —
  no empirical NIM evidence exists yet (this milestone is the
  infrastructure + preflight; the cheap pilot is the conditional
  next step).
* It does NOT retire any prior carry-forward.
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.
* It does NOT touch `coordpy/__init__.py`.
* It does NOT re-open any dead direction.

## Honest scope

This is a *classification supplement* + *anti-drift contract*
recording the W101 code-pivot infrastructure as active frontier
operating-system code.  The empirical verdict (preflight 6/8
PASS; 2 DEFERRED) is documented in `docs/RESULTS_W101_PREFLIGHT_V1.md`.
The cheap pilot is the conditional next step the operator
authorises after fetching MBPP+ data.

## Anchors

* `docs/RUNBOOK_W101.md` — W101 pre-commit contract.
* `docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md` — the
  5-candidate × 8-criterion battlefield ranking.
* `docs/RESULTS_W101_ARSENAL_MINING_V1.md` — W88/W91 sidecar
  cluster surface.
* `docs/RESULTS_W101_PREFLIGHT_V1.md` — empirical preflight
  verdict.
* `coordpy/mbpp_plus_loader_v1.py` — MBPP+ corpus loader.
* `coordpy/mbpp_plus_executor_v1.py` — MBPP+ extra-tests-aware
  executor.
* `coordpy/mbpp_plus_reflexion_bench_v1.py` — A0/A1/B reflexion
  bench.
* `coordpy/mbpp_plus_preflight_v1.py` — NIM-free preflight
  harness.
* `scripts/run_w101_arsenal_mining.py` — offline sidecar
  re-executor.
* `scripts/run_w101_mbpp_plus_preflight.py` — preflight runner.
* `scripts/run_w101_mbpp_plus_pilot.py` — conditional cheap-pilot
  driver.
