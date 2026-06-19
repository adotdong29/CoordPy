# W100 — Frontier relevance audit V1 (supplement to W97 + W98 + W99 V1)

> 2026-05-25.  Fourth supplement, extending the W97 + W98 + W99
> frontier-relevance audits with the W100 cross-scale 90B
> confirmation of B2 (frontier lead) and B5 (baseline-only
> ceiling reference).  The W97 + W98 + W99 audit classifications
> all remain in force; this supplement only:
>
> 1. records the *cross-scale-confirmation* status of B2 + B5
>    (the empirical verdict is delivered by the W100 90B Phase 2
>    pilots, which are separate deliverables under
>    `docs/RUNBOOK_W100.md`),
> 2. re-asserts the W97 + W98 + W99 anti-pattern column
>    *verbatim*, and
> 3. re-asserts that the typed-extract-then-text-reason sub-
>    family of W95-B0 (D2-B0 + W98 B1 + W99 B4) remains *dead*
>    after THREE distinct empirical refutations at 11B.
>
> No code is removed by this audit.  No version bump.  No PyPI
> publish.

## Why a supplement, not a rewrite

W99 V1 documented the classification as of the W99 multi-
candidate-tournament closing moment.  W99's empirical evidence
established two new structural facts:

1. The W95-B0 family is *repairable* if the image stays alive
   at the decision boundary (B2 mechanism PASSed Phase 2 at 11B).
2. The typed-extract-then-text-reason sub-family is *capped*
   through THREE distinct mechanisms (D2-B0 free-text bullet
   extraction; W98 B1 typed schema *with* `direct_answer_hint`;
   W99 B4 typed schema *without* `direct_answer_hint`).

W100's job is the **cross-scale confirmation** of fact #1.  This
audit supplement does NOT alter the W99 classifications; it
records that:

* the active frontier *lead* is B2, with cross-scale 90B
  confirmation pending in this milestone;
* the active baseline-only ceiling reference is B5, with
  cross-scale 90B confirmation pending in this milestone;
* the dead-direction column is unchanged from W99 and the
  W100 verdict will not change it absent extraordinary
  evidence (specifically, only a 90B B2 PASS with mechanism-
  load-bearingness sub-gates clearing would re-open a *narrower*
  active-frontier classification, and only a 90B B2 FAIL would
  expand the dead-direction column with a "cross-scale-bound"
  qualifier).

## Active frontier arsenal — W100 status (cross-scale pending)

| Mechanism | Module(s) | W100 cross-scale status |
|---|---|---|
| **B2 — Direct-vision final-turn answerer** | `coordpy/realworldqa_bench_v3.py` (built W98) | **LEAD; pending 90B Phase 2 confirmation in W100.**  W99 11B PASS gave +6.67 pp margin and 8 / 9 gates with structural verdict `STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP`; final-VLM rescued 3 / 3 invocations.  Cross-scale 90B Phase 2 entitled by W99 Option-A.  If 90B PASSes with MLB sub-gates clearing, the image-at-decision-boundary mechanism is *cross-scale-load-bearing*. |
| **B5 — Question-type router (switch baseline)** | `coordpy/realworldqa_bench_v5.py` (built W99) | **Baseline-only ceiling reference; pending 90B Phase 2 confirmation in W100.**  W99 11B oracle 30 / 30 = empirical 30 / 30; per-route 18 / 18 + 12 / 12.  Cross-scale 90B Phase 2 entitled by W99 Option-A.  A B5 90B PASS confirms the cross-scale routing-ceiling, *not* structural team superiority. |
| **W99 multi-candidate cheap-discriminator rule** | `docs/RUNBOOK_W99.md`; `docs/RUNBOOK_W100.md` (anti-tournament constraint) | UNCHANGED.  W100 is explicitly *not* a new tournament; the W99 multi-candidate rule remains the canonical operating principle for tournaments — W100 carries forward its winner. |
| **W100 cross-scale-confirmation rule** | NEW (this milestone): `docs/RUNBOOK_W100.md` | "After a multi-candidate tournament selects a winner at the cheap scale, the next milestone is a cross-scale CONFIRMATION at the expensive scale on the same winner — NOT a new tournament."  This codifies the anti-drift rule the user's Part C brief imposed. |
| **W100 mechanism-load-bearingness sub-gates** | NEW (this milestone): `docs/RUNBOOK_W100.md` § MLB-1 + MLB-2 | "A PASS at the cross-scale level is only frontier-relevant if the mechanism's rescue rate stays in the load-bearing regime (≥ 33 % rescue rate when invoked; invocation rate ≤ 50 % of problems)."  This codifies the W96-C C1 lesson (the cross-scale-collapse pattern). |
| **W99 + W100 failure-cluster + cross-scale-mining stack** | `coordpy.failure_cluster_miner_v1` + W97 / W98 / W99 sidecars + W100 pre-pilot AddrW100 probes | UNCHANGED.  W100 adds two NIM-free cross-scale probes (AddrW100-B2-P5 + AddrW100-B5-P4) that ride on the W99 stack; no new mining surface is opened in W100. |

## Useful baselines (W100 changes from W97 / W98 / W99)

| Mechanism | Module(s) | Classification | W100 status |
|---|---|---|---|
| `bounded_window_baseline_v{1,2,3}` | `coordpy/bounded_window_baseline_v*.py` | UNCHANGED — useful falsifier targets the substrate-coupled methods must beat. | Same. |
| B5 question-type router | `coordpy/realworldqa_bench_v5.py` | **Baseline-only ceiling / floor reference.** | **UNCHANGED — even a B5 90B PASS does NOT promote B5 to frontier.**  B5 is a switch; it does not introduce a new mechanism for cross-modal team coordination.  Its PASS only proves the cross-scale routing-ceiling. |
| A0 / A1 baselines | unchanged | unchanged | unchanged |

## Historical artifacts (unchanged from W97 / W98 / W99 V1)

W90 / W92 / W88 / W81 / W83 / W84 unchanged.  Kept for
regression / audit; not active path.

## Dead directions (unchanged from W99 V1 — three empirical refutations on the typed sub-family)

| Mechanism | Evidence against | NEW W100 status |
|---|---|---|
| **VLM-Verifier-Final-Turn as load-bearing rescue** | W96-C: 0 / 11 at 11B; 1 / 7 at 90B | UNCHANGED — refuted.  B2's final-VLM is a *committed answerer*, not a binary verifier; structurally distinct. |
| **W95-B0 free-text bullet extraction as sufficient on vision-bound benches** | W97 D2-B0 11B: B − A1 = −6.67 pp | UNCHANGED — refuted on RealWorldQA yes/no perception. |
| **B1 typed schema *with* `direct_answer_hint`** | W98 B1 11B: B − A1 = −6.67 pp via 5 multi-choice regressions | UNCHANGED — refuted. |
| **B4 typed schema *without* `direct_answer_hint`** | W99 B4 11B: B − A1 = −16.67 pp (WORSE than W98 B1) | UNCHANGED — refuted; the hint-removal hypothesis is empirically wrong. |
| **Typed-extract-then-text-reason sub-family of W95-B0 as a whole** | THREE empirical refutations at 11B (D2-B0 + W98 B1 + W99 B4) | UNCHANGED — dead.  Do not re-open in W100 or W101 absent a structurally new fix (e.g., image-at-decision-boundary, which is the B2 mechanism — and B2 is NOT in this sub-family). |
| **W95-B0-derived family as a whole** | RESOLVED at W99: family is *repairable* via the B2 mechanism; the cap was at the *no-image-at-decision-boundary* level, not the family level. | UNCHANGED — the resolution stands; W100 confirms it cross-scale. |

## Anti-patterns (NEVER promote as core strategy; baseline-only allowed) — UNCHANGED VERBATIM FROM W99

**The W97 + W98 + W99 anti-pattern list remains in force VERBATIM in W100.**

| Anti-pattern | W100 status |
|---|---|
| Bounded context window as product thesis | UNCHANGED — anti-pattern; baseline-only. |
| Compaction / generic prose summarization as memory mechanism | UNCHANGED — anti-pattern; W97 + W98 + W99 evidence reinforces. |
| Shallow token compression without structural reason | UNCHANGED. |
| Context-pruning theater | UNCHANGED. |
| "Cram less / truncate better" as frontier memory system | UNCHANGED. |
| LLM-as-judge in executor chain | UNCHANGED. |
| Selective retries | UNCHANGED. |
| Single-seed pilots as retirement evidence | UNCHANGED — W100 cheap pilots are CROSS-SCALE CONFIRMATIONS at the Phase 2 size, not retirement evidence. |
| Architecture refinement by vibe | UNCHANGED — W100 confirms the W99 winner B2 via cross-scale 90B Phase 2; no new architecture is introduced. |
| Inventing new candidates after the tournament selected a winner | **NEW W100 anti-pattern.**  W99 selected the winner; W100 confirms it.  Inventing a B6 / B7 / etc. in W100 would be tournament-restart-by-vibe; explicitly forbidden by Part C of the user's W100 brief. |

## What W100 cross-scale confirmation is NOT

To pre-empt drift back toward commodity-LLM tricks under a new
name:

### W100 is NOT a new tournament

* W99 ran the multi-candidate tournament (B2 + B4 + B5).
* W99 selected the winner (B2) and the ceiling reference (B5).
* W100 is the **single-winner cross-scale confirmation**.  It
  does not re-litigate the winner.  It does not re-open B4 or
  the typed-extract sub-family.

### A B5 PASS at 90B is NOT a B2 PASS at 90B

* B5 is a switch baseline.  Its PASS at any scale records the
  routing-ceiling, NOT a structural team mechanism.
* The frontier-relevant question in W100 is whether B2
  generalises cross-scale.  Only the B2 verdict answers that.

### A B2 PASS at 90B with low mechanism-load-bearingness sub-gate values is NOT a frontier-relevant PASS

* Per the W96-C C1 lesson, a PASS at 90B with low rescue rate
  (< 1 / 3 rescues when invoked) is variance-driven and NOT
  load-bearing.
* W100 explicitly codifies mechanism-load-bearingness sub-gates
  MLB-1 (invocation rate ≤ 50 %) + MLB-2 (rescue rate ≥ 33 %)
  in the runbook.

### A B2 FAIL at 90B is NOT a refutation of the 11B 100 % PASS

* The 11B PASS stands as carry-forward
  `W99-L-REALWORLDQA-B2-DIRECT-VISION-FINAL-TURN-PHASE2-11B-STRUCTURAL-PASS-SLICE-SATURATION-CAP`
  regardless of W100's 90B verdict.
* A 90B FAIL adds a *cross-scale-bound* qualifier to the
  carry-forward; it does not erase the 11B truth.
* `COO-9` is promoted to lead path in that case per the W100
  code-pivot contingency (Part H of the W100 brief).

## Honest classification of a W100 PASS / FAIL pattern

| W100 90B outcome | What we earn | What we do NOT claim |
|---|---|---|
| B2 PASS with MLB sub-gates clearing | The image-at-decision-boundary mechanism is **cross-scale load-bearing** on RealWorldQA at 11B AND 90B.  Phase 3 entitlement granted.  W95-B0 family REPAIR confirmed cross-scale. | Multi-agent context superiority on RealWorldQA — still requires Phase 3 retirement evidence. |
| B2 PASS WITHOUT MLB sub-gates clearing (`PASS_NON_MECHANISM_DRIVEN`) | The 90B PASS is variance-driven; B2 is NOT a frontier mechanism at 90B even though the gates clear.  Phase 3 NOT entitled. | That the 90B PASS reflects the same structural mechanism as 11B. |
| B2 FAIL with B5 PASS at 90B | Cross-scale routing-ceiling result reproducible; structural frontier mechanism does NOT generalize. | That the structural claim survives cross-scale; nothing about B2 generalising. |
| B2 FAIL with B5 FAIL at 90B | Both the routing-ceiling AND the structural-mechanism caps reproduce in the harsher 90B regime; the slice may be the limiting factor, not the architecture. | That RealWorldQA is solvable by the existing W95-B0 arsenal at the +5 pp Phase 2 bar at any scale. |
| B2 PASS (with MLB clearing) AND B5 PASS at 90B | The strongest possible W100 outcome: structural mechanism load-bearing cross-scale AND routing-ceiling reproducible cross-scale.  Phase 3 entitlement granted. | Anything Phase 3 (retirement-grade) without running the retirement bench. |

## What this supplement DOES NOT do

* It does NOT claim multi-agent context is solved.
* It does NOT claim any W100 candidate will beat A1 at 90B —
  no empirical evidence exists yet (this audit is a
  classification supplement, not a benchmark result).
* It does NOT retire any prior carry-forward.
* It does NOT propose any new bench code.  B2 + B5 are reused
  unchanged from W98 / W99.
* It does NOT bump `coordpy.__version__` or
  `SDK_VERSION`.
* It does NOT publish to PyPI.

## Honest scope

This is a *classification supplement* + *anti-drift contract*.
The W100 cross-scale confirmation is delivered by the 90B
Phase 2 pilots, which are separate deliverables under
`docs/RUNBOOK_W100.md`.  This supplement records the rules of
engagement; the verdicts land in the W100 results docs.
