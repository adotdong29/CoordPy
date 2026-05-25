# W97 — Extended arsenal-mining for RealWorldQA (D2)

> 2026-05-25.  Extension of the W96-D arsenal-mining pass
> (`docs/RESULTS_W96D_ARSENAL_MINING_V1.md`) specifically for the
> RealWorldQA (D2) lead path.  Goal: select the cheapest
> meaningful W97 D2 candidate that is **arsenal- and
> structure-driven**, not W95-B0-inertia-driven; explicitly
> record alternatives considered + rejected.  No NIM calls in
> this pass.  No version bump.

## Why this pass differs from W96-D arsenal mining

The W96-D arsenal mining inventoried mechanisms in the abstract
("ChartQA primary, RealWorldQA backup"); D1 (ChartQA) was the
lead at the time so the discussion was chart-oriented.  D1 was
killed cheaply at preflight at both scales by P3 saturation.
D2 is now the lead.  This pass:

1. Re-reads the inventory through the **RealWorldQA-specific
   structural lens** — what the benchmark actually rewards.
2. Adds a new candidate **D2-B1** (structured scene-graph
   extraction) that the W96-D pass marked "low ChartQA leverage;
   low RealWorldQA leverage" — and **revises that classification
   upward** given a tighter reading of RealWorldQA's question
   distribution.
3. Reaffirms that the preflight-earned shape is **D2-B0**
   (W95-B0 scene-port) and that any D2-B1 launch is a *separate*
   future move, not part of this W97 cheap pilot.

## RealWorldQA structural profile (revised)

RealWorldQA is 765 problems pairing a real-world driving / scene
image with a question.  Question distribution (from the
preflight 5-sample inspection and public dataset descriptions):

| Question class | Example | Structural feature |
|---|---|---|
| **Counting** | "How many cars are visible?" | Object enumeration; sensitive to occluded / partial objects.  Counts are an explicit structural extraction. |
| **Spatial relations** | "Is the truck to the left of or right of the bus?" | Bounding-box-style reasoning; reward goes to representations that preserve geometric ordering. |
| **Identification** | "What color is the car closest to the camera?" | Object attribute extraction with conditional ("closest"). |
| **Sign / text reading** | "What does the sign in the foreground say?" | OCR-like.  Hard to recover from a free-text extraction without the original image. |
| **Action / activity** | "Is the pedestrian crossing the street?" | Mostly visual; minimal symbolic recoverability. |
| **Multi-choice (A/B/C/D)** | "Which lane is the car in? A) left B) center C) right" | Explicit answer set; reduces extraction-vs-reasoning entanglement. |

Per the preflight P4 sample: gold answers include single-letter
multi-choice ("A", "B", "C", "D"), free-form numbers ("2"), and
free-form short text ("yes", "no", "right", colour words, etc.).

**Key structural feature distinguishing RealWorldQA from
MathVista:**

* MathVista's problems are *math-over-vision* — once the image's
  numeric / geometric facts are extracted, the math step is
  text-only and well-typed.
* RealWorldQA's problems are *vision-with-natural-language-
  question* — the vision step is the discriminator; the language
  step is shallow.

This inverts the W95-B0 advantage profile.  On MathVista the
extraction step lets the text-LM do real arithmetic work the
unified VLM doesn't do as cleanly.  On RealWorldQA the extraction
step may discard signal the unified VLM otherwise retains
implicitly.

## Mechanism re-ranking for D2

| Mechanism | Module / origin | D2 leverage (revised) | Selected for W97? |
|---|---|---|---|
| **D2-B0 — W95-B0 scene-port (1 VLM reader + 4 text solver/reflexion)** | `coordpy.realworldqa_bench_v1` (new in W97; mirrors `coordpy.mathvista_bench_v1`) | Conservative; preflight-earned; structurally vulnerable for spatial-only / counting questions because text-LM cannot re-see the image.  Expected B − A1 distribution: centred around 0 pp ± a few pp; lower expected upside than on MathVista. | **YES — D2-B0 LEAD (preflight-earned cheap pilot candidate)** |
| **D2-B1 — Structured scene-graph extraction (objects + bounding-box-region tags + spatial relations + counts as JSON)** | NEW; design doc only in W97; implementation deferred | The structurally-strongest *NIM-only* candidate: forces lossless extraction of the spatial primitives RealWorldQA actually tests.  The text solver reasons over the explicit JSON graph instead of free-text bullets.  Risk: schema brittleness; chart/scene mismatch; extraction errors propagate.  Cheap to define; expensive to verify (own Phase 2 pilot). | **DOCUMENTED — D2-B1 ARCHITECTURAL REFINEMENT, NOT IMPLEMENTED THIS MILESTONE.**  Earns its own preflight only if D2-B0 Phase 2 leaves residual on spatial-question slices. |
| **VLM-Verifier-Final-Turn (C1 port)** | `coordpy.mathvista_bench_v2` | **EMPIRICALLY REFUTED** on MathVista at K=5 byte-exact (W96-C verifier rescue 0/11 at 11B; 1/7 at 90B = not load-bearing).  Same K-budget shape on RealWorldQA carries the same refutation prior. | **NO — W96-C carry-forward applies.** |
| **VLM-in-loop (W90 port; image every turn)** | `coordpy.cross_modal_vlm_loop_bench_v1` | **EMPIRICALLY REFUTED** on HumanEval-Visual K=5 (W90).  W93 failure-cluster diagnosis showed the pattern is closer to A1 i.i.d. K=5 than a structurally distinct team.  RealWorldQA is more vision-dependent than HumanEval-Visual, but the structural risk of "no decomposition advantage over IID sampling" carries over. | **NO — W90 carry-forward applies.** |
| **Three-role split (W92 VLM-Planner + Code-Implementer + VLM-Verifier)** | `coordpy.cross_modal_role_specialized_bench_v1` | **EMPIRICALLY REFUTED** on HumanEval-Visual K=5 (−10.71 pp; 0/7 seeds).  K=5 budget burns on two VLM calls. | **NO — W92 carry-forward applies.** |
| **Substrate-level multi-modal patch payload** | `coordpy.multi_modal_payload_v1`, `coordpy.vision_substrate_v1` | Requires VLM hidden-state access.  NIM HTTPS does NOT expose patch embeddings.  Load-bearing only for hardware-local substrate work (COO-12). | **NO — INFEASIBLE via NIM HTTPS.** |
| **Adversarial consensus repair** | `coordpy.adversarial_consensus_repair_v1` | Requires parallel branches; D2-B0 / B1 solver chain is sequential.  Wrong shape. | **NO — wrong shape.** |
| **Tool-augmented solver (W96-C C2; arithmetic substrate)** | `coordpy.tool_call_substrate_v1`, `coordpy.code_substrate_v1` | RealWorldQA is *not arithmetic-heavy*.  Counting questions are the only arithmetic-adjacent subset; even there the bottleneck is extraction, not arithmetic.  Adds budget complexity without obvious leverage. | **NO — leverage mismatch.** |
| **Failure-cluster mining** | `coordpy.failure_cluster_miner_v1` | Useful AFTER a Phase 2 pilot lands and produces sidecars.  Not load-bearing for the preflight-to-pilot step. | **DEFER — post-pilot tool.** |
| **W93 cheap-preflight composite** | `coordpy.realworldqa_preflight_v1` (already exists from W96-D) | Already PASSED for D2-B0 at both 11B (residual 26.56 pp) and 90B (residual 20.51 pp). | **CARRIED — preflight already earned.** |
| **Bounded-window baseline V3** | `coordpy.bounded_window_baseline_v3` | Wrong shape entirely (long-horizon synthetic substrate; nothing to do with VLM K=5 sampling).  Listed only to make the audit's separation explicit. | **NO — baseline only; wrong axis.** |

## D2-B0 vs D2-B1 trade-off summary

|  | D2-B0 (lead) | D2-B1 (refinement, not in this milestone) |
|---|---|---|
| Extraction shape | Free-text bullet list | Structured JSON scene-graph |
| Solver shape | Text-LM (no image) | Text-LM (no image) reasoning over JSON |
| Budget | K=5 byte-exact (1 reader + 4 solver/reflexion) | K=5 byte-exact (1 reader + 4 solver/reflexion) |
| Extraction risk | Lossy on spatial / counting subsets | Tightly schema-constrained; lossy in the opposite way (rigid schema may miss free-form attributes) |
| Implementation cost | None new; prompts adapted from W95-B0 | New prompt + parsed schema validation in B-solver |
| Preflight status | EARNED (W96-D D2 preflight PASS at both scales) | NOT earned; would need its own preflight before any NIM call |
| Phase 2 cost | ~330 NIM calls per scale | ~330 NIM calls per scale |
| Honest expectation | Centred around 0 pp ± few pp; positive signal not guaranteed | Higher expected mean IF the schema lands on the right primitives; higher variance |
| Falsifier role | Establishes whether the W95-B0 shape generalises to RealWorldQA | Establishes whether structured-extraction recovers any signal D2-B0 leaves on the table |

The **honest design decision** is: D2-B0 is the W97 lead because
it is preflight-earned; D2-B1 is the architectural backup whose
preflight is itself a separate cheap probe.  Running both
simultaneously would scatter into two half-serious variants
violating the W97 discipline rule.

## Pre-pilot prediction (locked 2026-05-25 BEFORE NIM)

> "On the 1 × 30 cheap pilot at 11B-Vision, D2-B0's likely
> outcomes (in decreasing order of subjective probability):
>
> 1. **B narrowly loses or ties A1** (B − A1 in [−5, +3] pp) —
>    consistent with the structural argument that RealWorldQA
>    rewards preserved visual signal over text-LM reasoning.
>    Would carry-forward `W97-L-REALWORLDQA-D2-B0-PHASE2-CAP`
>    and licence a D2-B1 preflight as the W97 architectural
>    refinement.
> 2. **B wins by a narrow margin (B − A1 in [+3, +7] pp)** —
>    consistent with the W95 MathVista pattern; would license
>    a cross-scale 90B Phase 2 and (if cross-scale PASS) Phase 3.
> 3. **B wins by a strong margin (B − A1 ≥ +10 pp)** —
>    structurally surprising; would warrant arsenal-mining
>    re-read of why the W95-B0 shape lands so cleanly on
>    spatial scenes.
> 4. **B saturates with A1 at ceiling** — preflight already
>    estimated A1@K=5 ≈ 73 % at 11B; would require an
>    unexpectedly hard slice to push past 90 %.  Low prior.
>
> The most likely outcome is (1) narrow tie/lose, given the
> structural argument."

This prediction is anchored in the run record (the smoke
sidecar; written BEFORE the pilot launches).

## D2-B1 design (deferred to W97-B if D2-B0 leaves residual)

For honesty + future re-use, the D2-B1 schema is sketched here:

```json
{
  "scene_summary": "<free-form 1-sentence overview>",
  "objects": [
    {
      "id": "obj_001",
      "label": "car",
      "color": "red",
      "size": "medium",
      "x_region": "center",         // left / center / right
      "y_region": "middle",         // top / middle / bottom
      "depth": "near",              // near / mid / far
      "confidence": "high"          // high / medium / low
    }
  ],
  "counts": {"car": 2, "person": 3, "traffic_light": 1},
  "spatial_relations": [
    {"a": "obj_001", "b": "obj_002",
     "relation": "left_of", "near": true}
  ],
  "text_in_scene": [
    {"location": "background", "text": "STOP"}
  ],
  "uncertain": ["<list of features the reader could not extract>"]
}
```

The B-solver receives this JSON as ground truth and answers the
question using *only* the structured fields, never the original
image.  The schema is rigid but recoverable; counts and spatial
relations are explicit; text-in-scene captures the sign-reading
subset.

D2-B1 is documented here so:

1. The W97-B (if invoked) inherits a frozen schema.
2. The W97 cheap pilot's failure analysis (`failure_cluster_miner_v1`
   on the W97 D2-B0 sidecars) can answer "which RealWorldQA
   subset would D2-B1 plausibly rescue?" without re-inventing
   the schema.
3. Reviewers can audit whether D2-B1 actually attacks the right
   failure mode if the W97 D2-B0 evidence supports it.

## Decision recorded

* **W97 D2-B0 is the LEAD candidate for the cheap pilot.**
* **W97 D2-B1 (structured scene-graph) is the DOCUMENTED
  REFINEMENT** — its preflight is not yet earned; deferred.
* **No other candidate is selected** for this W97 milestone.

The W97 runbook (`docs/RUNBOOK_W97.md`) locks the Phase 2 pilot
gates BEFORE any NIM call.

## Honest framing

This is an *arsenal-mining classification*, not a benchmark
result.  D2-B0 has not yet produced any Phase 2 evidence on
RealWorldQA; the only honest claim from this doc is "D2-B0 is
the preflight-earned candidate selected as the W97 lead and
D2-B1 is the architectural refinement deferred pending D2-B0
evidence."  No NIM was spent in producing this document.  No
carry-forward retirement is asserted.  No version bump.
