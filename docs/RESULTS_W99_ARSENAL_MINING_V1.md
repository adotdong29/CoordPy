# W99 — Extended arsenal-mining for RealWorldQA (B2 + B4 + B5)

> 2026-05-25.  Third extension of the W96-D / W97 / W98
> arsenal-mining passes
> (``docs/RESULTS_W96D_ARSENAL_MINING_V1.md``,
> ``docs/RESULTS_W97_ARSENAL_MINING_V1.md``,
> ``docs/RESULTS_W98_ARSENAL_MINING_V1.md``) for the **W99
> multi-candidate RealWorldQA tournament** that follows the
> W98 B1 Phase 2 11B FAIL (B − A1 = −6.67 pp; gates 2 / 3 / 4
> FAIL; ``W98-L-REALWORLDQA-B1-TYPED-SCHEMA-PHASE2-11B-CAP``).
> Goal: select the strongest arsenal- and structure-driven W99
> candidates that **attack the empirically-mined W97 + W98
> failure surface** — the 5 W97 unique-A1-rescues on vision-
> bound yes/no perception (4 / 5 recovered by W98 B1; 1
> remaining viewer-pov gap) AND the 5 W98 multi-choice
> regressions where B1's reader-hint anchoring stopped
> reflexion-cycling.  No NIM calls in this pass.  No version
> bump.

## Honest grounding — what W97 + W98 revealed

### W97 D2-B0 per-problem confusion table at 11B-Vision

```
              A1 PASS   A1 FAIL   Total
B PASS           22         3        25
B FAIL            5         0         5
              ─────────────────────────
Total            27         3        30
```

D2-B0 (W97) PASS = 25 / 30 = 83.33 %; A1 (W97) PASS = 27 / 30
= 90.00 %; B − A1 = −6.67 pp.

### W98 B1 per-problem confusion table at 11B-Vision (same slice)

```
              A1 PASS   A1 FAIL   Total
B1 PASS          23         1        24
B1 FAIL           3         3         6
              ─────────────────────────
Total            26         4        30
```

B1 (W98) PASS = 24 / 30 = 80.00 %; A1 (W98) PASS = 26 / 30 =
86.67 %; B1 − A1 = −6.67 pp.

### Per-problem diff vs W97 D2-B0 on the same slice

W98 B1 recovered 4 of 5 W97 unique-A1-rescues:

* ✅ ``000135`` (Yes; "stop signs") — question-typed solver
  fixes yes/no → number bias.
* ✅ ``000403`` (No; "Is the light green?") — typed schema +
  typed solver recover state.
* ✅ ``000555`` (No; "are the cars facing left?") — typed
  orientation primitive.
* ✅ ``000718`` (Yes; depth ordering) — typed depth primitive.
* ❌ ``000615`` (No; "Is the traffic light green for us?") —
  neither candidate recovers; needs ``viewer_pov`` primitive
  the schema does not yet have.

W98 B1 regressed 5 W97 D2-B0 wins:

* ❌ ``000013`` (B; vehicle direction multi-choice)
* ❌ ``000155`` (B; gun direction multi-choice)
* ❌ ``000204`` (C; how-many-lanes multi-choice)
* ❌ ``000225`` (A; speed-limit number multi-choice)
* ❌ ``000713`` (2; how-many-cars numeric)

**Root cause (from W98 RESULTS doc)**: the reader's
``direct_answer_hint`` field is often wrong on multi-choice;
the typed-format solver becomes more confident in the hint
and stops the K=4 reflexion-cycling that drove D2-B0's wins.

## Per-question-type cluster analysis (W97 slice; from W98 sidecars)

| Type | Count | D2-B0 W97 PASS | A1 W97 PASS | Notes |
|---|---:|---:|---:|---|
| ``multi_choice_letter`` | 18 | 18 / 18 = 100 % | 15 / 18 = 83 % | D2-B0 OWNS multi-choice; A1's 3 fails are 000013 / 000155 / 000441. |
| ``yes_no`` | 6 | 1 / 6 = 17 % | 6 / 6 = 100 % | A1 OWNS yes/no.  D2-B0's 5 fails are 000135 / 000250 / 000403 / 000555 / 000615 / 000718 (wait — D2-B0 PASSed 000250).  Actually D2-B0 PASSed 1 / 6 = 17 %. |
| ``numeric`` | 4 | 4 / 4 = 100 % | 4 / 4 = 100 % | Both arms tie; no discriminating signal. |
| ``short_text`` | 2 | 2 / 2 = 100 % | 2 / 2 = 100 % | Both arms tie. |
| **Total** | **30** | **25 / 30** | **27 / 30** | |

**Sharp empirical fact**: per-question-type, D2-B0 owns
multi-choice (18 / 18 = 100 %) and A1 owns yes/no (6 / 6 =
100 %).  On numeric + short_text both arms tie at 100 %.
This is the structural basis for the W99 B5 oracle
prediction: route by type and you get 30 / 30.

## W99 candidate-slate design constraints (carry-forward + new)

Carry-forward from W98:

1. **Preserve D2-B0's both-pass and unique-B-rescues** via
   short-circuit on text-solver PASS.
2. **Provide explicit state primitives + depth ordering** —
   B4 retains B1's schema (the 4 / 5 yes/no recovery surface).
3. **Keep image alive at decision boundary** — B2 (image at
   final turn on the failure cluster).
4. **K=5 byte-exact**.
5. **No bounded / compaction / summary** in the lead path.

New from W98 sidecar mining:

6. **Strip ``direct_answer_hint``** as a minimal repair of
   B1's regression mechanism (B4).
7. **Allow a routing baseline** to bound how much team
   superiority is achievable by routing alone (B5).

## Mechanism re-ranking for W99

| Mechanism | Module / origin | W99 leverage | Selected for W99? |
|---|---|---|---|
| **C1 — Direct-vision final-turn answerer (B2)** | ``coordpy.realworldqa_bench_v3`` (W98) | NIM-free upper bound +6.67 pp realistic.  Structural frontier mechanism: image at decision boundary; distinct from W96-C C1 verifier (committed answerer, not binary).  Already preflight-earned in W98. | **YES — B2 (frontier lead)** |
| **C2 — Typed schema sans ``direct_answer_hint`` (B4)** | NEW ``coordpy.realworldqa_bench_v4`` (W99) | Minimal repair of W98 B1.  Preserves yes/no schema fix (4 / 5 recovery) AND restores reflexion-cycling on multi-choice (removes hint anchoring).  No NIM-free oracle; pure reasoning prediction. | **YES — B4 (close-cousin repair)** |
| **C3 — Question-type router (B5)** | NEW ``coordpy.realworldqa_bench_v5`` (W99) | NIM-free ORACLE on W97 slice predicts +10.00 pp (30 / 30).  Switch baseline; bounds the *routing ceiling*.  Honest classification: baseline-only ceiling, NOT frontier mechanism. | **YES — B5 (switch baseline / ceiling reference)** |
| **C4 — Add ``viewer_pov`` to B1 schema** | hypothetical W100+ refinement | Would address the 1 residual yes/no failure (000615) but at the cost of further schema rigidity. | NOT this milestone (deferred to W100+ if W99 yields no path). |
| **C5 — VLM-Verifier-Final-Turn (W96-C C1 re-port)** | ``coordpy.mathvista_bench_v2`` | EMPIRICALLY REFUTED on MathVista (0/11 at 11B; 1/7 at 90B = not load-bearing).  W98 audit re-affirmed refuted on RealWorldQA. | NO — refuted. |
| **C6 — VLM-in-loop every turn (W90 re-port)** | ``coordpy.cross_modal_vlm_loop_bench_v1`` | EMPIRICALLY REFUTED on HumanEval-Visual K=5. | NO — refuted. |
| **C7 — Three-role split (W92 re-port)** | ``coordpy.cross_modal_role_specialized_bench_v1`` | EMPIRICALLY REFUTED on HumanEval-Visual K=5. | NO — refuted. |
| **C8 — Tool-augmented solver (W96-C C2)** | ``coordpy.tool_call_substrate_v1`` | RealWorldQA failure mode is perception, not arithmetic. | NO — leverage mismatch. |
| **C9 — Substrate-level multi-modal payload** | ``coordpy.multi_modal_payload_v1``, ``coordpy.vision_substrate_v1`` | Requires VLM hidden-state access; NIM HTTPS does not expose patch embeddings.  Belongs to ``COO-12`` substrate path. | NO — infeasible via NIM HTTPS. |
| **C10 — Adversarial consensus repair** | ``coordpy.adversarial_consensus_repair_v1`` | Requires parallel solver branches; D2-Bx are sequential. | NO — wrong shape. |
| **C11 — Failure-cluster miner** | ``coordpy.failure_cluster_miner_v1`` | Already used in W98 + W99 mining passes (analysis tool). | CARRIED — analysis tool. |
| **C12 — W93 cheap-preflight composite** | ``coordpy.realworldqa_preflight_v1`` + W99 addressability probes | The gating discipline for any NIM call. | CARRIED — discipline. |
| **C13 — Bounded-window baseline V3** | ``coordpy.bounded_window_baseline_v3`` | Wrong axis entirely; explicit anti-pattern. | NO — anti-pattern. |
| **C14 — Generic prose-summary "memory" mechanism** | (would-be new module; **rejected**) | Anti-pattern per the frontier audit; W97 D2-B0's bullet extraction (which is itself a form of prose summary) FAILED on yes/no.  W98 B1's typed schema (structured) succeeded on yes/no — the contrast empirically refutes the prose-summary anti-pattern. | NO — explicit anti-pattern. |

## W99 candidate slate (final)

* **B2 — ``coordpy.realworldqa_bench_v3``** (built W98).
  Direct-vision final-turn answerer.  Structural frontier
  lead.  K=5 byte-exact: 1 reader + 3 text solver + 1
  (final-VLM | text-solver retry pad).
* **B4 — ``coordpy.realworldqa_bench_v4``** (new W99).  Typed
  schema sans ``direct_answer_hint``.  Minimal repair of B1.
  K=5 byte-exact: 1 typed reader + 4 typed solver.
* **B5 — ``coordpy.realworldqa_bench_v5``** (new W99).
  Question-type router.  Switch baseline.  K=5 byte-exact
  on either route.

All three:

* preserve K=5 byte-exact budget;
* preserve same VLM model on every arm (anti-cheat);
* preserve first-PASS short-circuit semantics where applicable;
* are arsenal-driven from W97 + W98 failure-cluster diagnoses;
* are NOT bounded / compaction / summary tricks.

## W99 NIM-free addressability probes (per candidate)

Layered on top of the W96-D D2 composite (P1..P4):

### B2 probes

* **AddrW99-B2-P1 — NIM-free upper bound from W97 confusion
  table.**  Counts both-pass / unique-B / unique-A1 / neither;
  computes best / realistic (80 % final-VLM rescue) /
  conservative (50 %) PASS rates.  Threshold: realistic ≥
  A1 + 5 pp.
* **AddrW99-B2-P2 — short-circuit static.**  Static code
  audit of ``_run_b_direct_vision_final``.
* **AddrW99-B2-P3 — final-VLM rescue prior.**  A1 wins 5 / 5
  of the unique-A1-rescue cluster by definition.
* **AddrW99-B2-P4 — budget exact.**  K=5.

### B4 probes

* **AddrW99-B4-P1 — schema primitives retained.**  Static
  schema audit (state / orientation / depth / text_in_object).
* **AddrW99-B4-P2 — hint field removed.**  Static audit
  confirming ``direct_answer_hint`` is gone from the solver
  template and mentioned ≤ 1× in the reader prompt (only the
  explicit removal admonition).
* **AddrW99-B4-P3 — budget exact.**  K=5.

### B5 probes

* **AddrW99-B5-P1 — ORACLE simulation on W97 sidecars.**
  Computes the exact predicted B5 PASS rate by routing each
  W97 slice problem and reading the W97 D2-B0 / A1 per-
  problem outcome.  Threshold: predicted ≥ A1 + 5 pp.
* **AddrW99-B5-P2 — parser correctness.**  ≥ 90 % on the W97
  slice (re-uses W98 AddrP6).
* **AddrW99-B5-P3 — budget exact.**  K=5 on either route.

## NIM-free predictions (locked BEFORE any NIM call)

### B2 (direct-vision final-turn answerer)

| Scenario | Predicted PASS rate | B2 − A1 (W97 A1=90 %) | Cleared +5 pp? |
|---|---:|---:|---|
| Conservative (50 % final-VLM rescue) | 90.00 % | +0.00 pp | NO |
| Realistic (80 % final-VLM rescue) | 96.67 % | +6.67 pp | **YES** |
| Best (100 % final-VLM rescue) | 100.00 % | +10.00 pp | **YES** |

### B4 (typed schema sans hint)

No NIM-free oracle.  Reasoning-only prediction: the
``direct_answer_hint`` was the proximate cause of W98 B1's 5
multi-choice regressions; removing it restores reflexion-
cycling discipline.  Predicted PASS rate ~ 26-29 / 30 = 87-
97 %.  Subjective probability of clearing +5 pp: ~ 25-40 %.

### B5 (switch baseline)

**ORACLE on W97 slice: B5 = 30 / 30 = 100.00 %; B5 − A1 (W97)
= +10.00 pp.**  Subject to live NIM sampling variance and the
single short_text parser-edge case (29 / 30 = 96.7 % parser
accuracy → 1 misrouted problem at worst; expected live B5 ~
93-100 %).

## D2-B4 (B4) frozen schema

Byte-identical to W98 B1's typed schema EXCEPT
``direct_answer_hint`` is REMOVED:

```json
{
  "scene_summary": "<free-form 1-sentence overview>",
  "objects": [
    {
      "label": "car",
      "color": "red",
      "x_region": "left",
      "y_region": "middle",
      "depth": "near",
      "orientation": "facing_left",
      "state": "moving",
      "text_in_object": ""
    }
  ],
  "counts_by_label": {"car": 2, "stop_sign": 1},
  "spatial_relations": [
    {"a": "obj_001", "b": "obj_002",
     "relation": "left_of", "near": true}
  ],
  "text_in_scene": [
    {"location": "background", "text": "STOP"}
  ],
  "uncertain": ["<list of features the reader could not
                  extract reliably>"]
}
```

The reader prompt explicitly admonishes against including a
``direct_answer_hint`` field or guessing the answer.  The
typed-solver template no longer references any hint.

## D2-B5 (B5) routing rule

```python
def b5_route_for_question(question: str) -> str:
    qt = detect_question_type_v2(question)
    if qt == QUESTION_TYPE_MULTI_CHOICE_LETTER:
        return "vlm_team_b0"   # W97 D2-B0
    return "a1_vlm_k5"          # A1 K=5
```

Each route runs at K=5 byte-exact.  Same VLM model on every
arm.  Executor invariants unchanged.

## What W99 is NOT trying to prove

* W99 is **not** claiming multi-agent context superiority on
  RealWorldQA.  Even a B5 PASS only proves the per-question
  routing ceiling.
* W99 is **not** introducing any bounded / compaction /
  summary mechanism.  All three candidates are structured.
* W99 is **not** a Phase 3 retirement bench.  At most one
  cross-scale 90B Phase 2 is in scope per surviving
  candidate.

## Cross-references

* ``docs/RUNBOOK_W99.md`` — pre-commit contract.
* ``docs/FRONTIER_RELEVANCE_AUDIT_W99_V1.md`` — frontier vs
  baseline-only vs dead vs anti-pattern (supplement to W97 +
  W98).
* ``docs/RESULTS_W98_REALWORLDQA_B1_PHASE2_V1.md`` — W98
  failure-cluster mining (this doc extends).
* ``coordpy/realworldqa_bench_v3.py`` — B2 (built W98).
* ``coordpy/realworldqa_bench_v4.py`` — B4 (new W99).
* ``coordpy/realworldqa_bench_v5.py`` — B5 (new W99).
* ``tests/test_w99_realworldqa_bench_v4_v5.py`` — unit tests.

## Honest scope of this doc

This is an *arsenal-mining classification + cheap-probe-
feasible diagnosis*, not a benchmark result.  No NIM was
spent in producing this document; per-failure diagnoses are
NIM-free re-reads of W97 + W98 sidecars already on disk.  The
only honest claim from this doc is "the W99 candidate slate
is B2 + B4 + B5 because the W97 + W98 failure surface
structure makes each candidate plausibly addressable, while
every other arsenal candidate is either refuted, infeasible,
wrong-shape, or anti-pattern."  No carry-forward is retired
by this doc.  No version bump.  No PyPI publish.
