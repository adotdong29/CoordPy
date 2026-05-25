# W98 — Extended arsenal-mining for RealWorldQA (W97-B / W98 slate)

> 2026-05-25.  Second extension of the W96-D / W97 arsenal-mining
> passes (`docs/RESULTS_W96D_ARSENAL_MINING_V1.md`,
> `docs/RESULTS_W97_ARSENAL_MINING_V1.md`) for the **W98 multi-
> candidate RealWorldQA assault** that follows the W97 D2-B0
> Phase 2 11B FAIL (B − A1 = −6.67 pp; gates 2/3/4 FAIL;
> `W97-L-REALWORLDQA-D2-B0-PHASE2-11B-CAP`).  Goal: select the
> 2–3 strongest arsenal- and structure-driven W98 candidates
> that **attack the empirically-mined W97 failure cluster** —
> the 5 unique-A1-rescues on vision-bound yes/no perception and
> the 3 unique-B-rescues on multi-choice spatial.  No NIM calls
> in this pass.  No version bump.

## Honest grounding — what W97 actually revealed

W97's per-problem confusion table at 11B-Vision:

```
              A1 PASS   A1 FAIL   Total
B PASS           22         3        25
B FAIL            5         0         5
              ─────────────────────────
Total            27         3        30
```

The W97 result doc summarised the 5 unique-A1-rescues by
question class and the 3 unique-B-rescues by question class.
This W98 mining pass goes one step deeper: **it reads the
actual per-problem reader extractions and B-solver responses
from `results/w97/realworldqa_pilot/.../{vlm_calls,text_calls}.jsonl`
to diagnose WHY each unique-A1-rescue failed**, so the W98
candidate slate is mined against the real failure surface, not
against the W97 doc's executive summary alone.

### Per-failure diagnosis (W97 D2-B0 unique-A1-rescue cluster)

| pid | Q (head) | Gold | What extraction said | What B-solver returned | Root-cause class |
|---|---|---|---|---|---|
| `rwqa_test_000135` | "Are there any stop signs?" | `Yes` | "1 stop sign" | `1` then `2` then `1` then `1` | **OUTPUT-FORMAT MISMATCH** — extraction had the answer; solver treated it as a counting question. |
| `rwqa_test_000403` | "Is the light green?" | `No` | Degenerate repeating list of "streetlight, street sign"; never records traffic-light state | `1` `0` `2` `0` | **LOSSY EXTRACTION** — reader collapsed into degenerate loop; no state primitive recorded. |
| `rwqa_test_000555` | "are the cars facing left?" | `No` | "A silver car is in the center… facing left.  A red truck on the right… facing right.  …  The cars are not facing left." | `2` `1` `0` `1` | **OUTPUT-FORMAT MISMATCH** — extraction explicitly contains the correct answer in prose ("are not facing left"); solver returned counts. |
| `rwqa_test_000615` | "Is the traffic light green for us?" | `No` | "The traffic light is currently red." | `2` `1` `2` `2` | **OUTPUT-FORMAT MISMATCH** — extraction contains the correct state primitive ("currently red"); solver returned counts. |
| `rwqa_test_000718` | "Is the large truck closer further from camera than the pickup truck?" | `Yes` | Object list with positions ("center", "left") but NO depth field | `1` `0` `1` `0` | **LOSSY EXTRACTION (depth ordering missing)** — reader recorded x-region but not depth/near-far primitive. |

**Cluster taxonomy (5 unique-A1-rescues):**

* **3 × output-format mismatch** (000135, 000555, 000615) — the
  extraction contains the right answer in prose; the B-solver
  system prompt biases yes/no questions toward numbers
  ("output ONLY the letter (for multi-choice) or the number /
  short answer (color, yes/no, single word)").
* **1 × lossy extraction (state primitive missing)** (000403) —
  reader degenerated into a repeating object list and never
  recorded the traffic-light state.
* **1 × lossy extraction (depth ordering missing)** (000718) —
  reader recorded 2D position but not relative depth (near/far).

This is decisive: **a typed solver prompt that knows the question
expects yes/no (or numeric / short-text / multi-choice-letter)
would plausibly recover ≥ 3 of the 5 failures** even without
any change to the extraction shape.  A schema-constrained
extraction with explicit state and depth primitives would
plausibly recover the remaining 2.

### Per-rescue diagnosis (W97 D2-B0 unique-B-rescue cluster)

| pid | Q (head) | Gold | What extraction said | What B-solver returned | Why D2-B0 won |
|---|---|---|---|---|---|
| `rwqa_test_000013` | "Which direction is the vehicle directly in front of us travelling?  A. Straight  B. Left  C. Right" | `B` | Degenerate "1 white car, 1 black car…" loop | `C` `B` `A` `B` | Extraction was noise; the 4-turn reflexion cycled `C/B/A/B`; **K=4 across 3 options has high hit probability** — this is partly luck not mechanism. |
| `rwqa_test_000155` | "Which direction is the gun facing?  A. left B. down C. right" | `B` | "The gun patch is oriented with the barrel pointing towards the bottom left corner."  + "Answer: C" hint from reader | `C` `B` `A` `D` | Reader added a wrong answer hint (`C`); the text-solver flipped to `B` on turn 2 and PASS short-circuited.  **Cycling helped, not mechanism.** |
| `rwqa_test_000441` | "Where is the letter c relative in the entire word in this image?  A. no c B. left C. middle" | `C` | "The letter 'C' is located in the middle of the word 'WELCOME'."  + "Answer: C" hint | `C` `B` `B` `C` | Reader's hint was correct; first turn picked `C` and short-circuited.  **Clean win.** |

**Honest read of unique-B-rescues:** only 000441 is a *clean*
mechanism win.  The other two are K=4-multi-choice luck —
cycling A/B/C across 4 turns on a 3-option multi-choice has
≈ 75 % hit probability under uniform sampling regardless of
mechanism.  This sharpens the W98 design constraint: **a W98
candidate that breaks D2-B0's 4-turn reflexion cycling on
multi-choice would regress on 000013 and 000155**.  The W98
candidate must preserve the multi-turn cycling discipline.

## W98 candidate-slate design constraints (derived from the diagnosis)

1. **Preserve D2-B0's 22 / 30 both-pass and ≥ 2 / 3 unique-B-
   rescues** (the K=4 reflexion cycling on multi-choice is the
   load-bearing part of those wins).  Any candidate that
   short-circuits on text-solver PASS preserves this.
2. **Fix the output-format mismatch on yes/no** by adding a
   typed solver prompt that does NOT bias yes/no toward
   numeric.  This is the highest-leverage cheap fix — recovers
   plausibly 3 of 5 unique-A1-rescues.
3. **Provide explicit state primitives** (traffic-light colour,
   object orientation, sign text) **and depth ordering** in
   the extraction schema.  This addresses the remaining 2
   failures (000403 traffic-light state + 000718 depth-ordering).
4. **Keep the image alive at the decision boundary** as a
   distinct architectural arm: a structurally different
   mechanism from "fix the extraction" — direct visual grounding
   on the final turn, image present.  This is what the user
   explicitly listed as a "what we DO want" mechanism.
5. **All candidates must be K=5 byte-exact** — same anti-cheat
   budget as A1 and D2-B0.  No mechanism may use a 6th call.
6. **No bounded-window / compaction / summary mechanism may
   appear in the lead path.**  These remain baselines-only per
   `docs/FRONTIER_RELEVANCE_AUDIT_W97_V1.md`.

## Mechanism re-ranking for W98

| Mechanism | Module / origin | W98 leverage | Selected for W98? |
|---|---|---|---|
| **C1 — Typed scene-graph extraction + question-typed solver (D2-B1 lead)** | NEW `coordpy.realworldqa_bench_v2` (this milestone) | **Directly attacks 5/5 unique-A1-rescues**: typed schema recovers state primitives + depth ordering (2 failures); question-typed solver prompt fixes yes/no output-format bias (3 failures).  Preserves D2-B0 short-circuit on multi-choice (3 unique-B-rescues unaffected). | **YES — C1 LEAD (highest expected leverage; addresses every diagnosed failure)** |
| **C2 — Direct-vision final-turn answerer (D2-B2 lead)** | NEW `coordpy.realworldqa_bench_v3` (this milestone) | Structurally different from C1: the FINAL turn sees the image + scene-graph + prior text-solver candidates and produces the canonical answer.  Direct visual grounding at the decision boundary.  Short-circuits on text-solver PASS so multi-choice wins are preserved.  Distinct from W96-C C1 verifier (which was a binary agree/disagree; this is an answerer with full visual access). | **YES — C2 LEAD (distinct mechanism: image at decision boundary; no verifier-override risk)** |
| **C3 — Question-type oracle router (D2-B3 cheap-baseline)** | Sketched only; not implemented | Cheap-baseline best-of-two: route multi-choice questions to D2-B0 (where it wins 3/3 unique-B-rescues + most both-pass) and route yes/no / state questions to A1 K=5 (where unified VLM wins 5/5 unique-A1-rescues).  Structurally a *switch* rather than a substrate mechanism.  Valuable as a *floor reference* for C1 / C2 but not arsenal-driven. | **NO — DOCUMENTED ONLY** as the architectural baseline both serious candidates must beat.  Implementing it would be evidence-collecting, not frontier mechanism. |
| **C4 — VLM-Verifier-Final-Turn (W96-C C1 re-port to RealWorldQA)** | `coordpy.mathvista_bench_v2` | EMPIRICALLY REFUTED on MathVista (W96-C: verifier rescue rate 0/11 at 11B; 1/7 at 90B = not load-bearing).  Failure mode of C4 (binary agree/disagree, rarely overrides) is structurally different from C2 (committed answerer with image+context).  Re-introducing C4 without new positive cheap-probe evidence is barred by the frontier audit. | **NO — refuted; the W96-C carry-forward applies.** |
| **C5 — VLM-in-loop (every turn) (W90 re-port)** | `coordpy.cross_modal_vlm_loop_bench_v1` | EMPIRICALLY REFUTED on HumanEval-Visual K=5.  At K=5 the architecture closely resembles A1 i.i.d. K=5 sampling and offers no structural advantage. | **NO — refuted; W90 carry-forward applies.** |
| **C6 — Three-role split (W92 re-port)** | `coordpy.cross_modal_role_specialized_bench_v1` | EMPIRICALLY REFUTED on HumanEval-Visual K=5 (−10.71 pp; 0/7 seeds).  K=5 budget is too tight for two-VLM-call architectures. | **NO — refuted; W92 carry-forward applies.** |
| **C7 — Tool-augmented solver (W96-C C2)** | `coordpy.tool_call_substrate_v1`, `coordpy.code_substrate_v1` | RealWorldQA is not arithmetic-heavy.  Counting questions are the only arithmetic-adjacent subset, and there extraction is the bottleneck not arithmetic. | **NO — leverage mismatch with RealWorldQA failure mode.** |
| **C8 — Substrate-level multi-modal patch payload** | `coordpy.multi_modal_payload_v1`, `coordpy.vision_substrate_v1` | Requires VLM hidden-state access; NIM HTTPS does not expose patch embeddings.  Belongs to `COO-12` substrate path; out of NIM-only frontier. | **NO — infeasible via NIM HTTPS.** |
| **C9 — Adversarial consensus repair** | `coordpy.adversarial_consensus_repair_v1` | Requires parallel solver branches; D2-Bx chains are sequential. | **NO — wrong shape.** |
| **C10 — Failure-cluster miner** | `coordpy.failure_cluster_miner_v1` | Already used in this W98 mining pass to diagnose the 5 / 3 cluster.  Will be used again to mine W98 sidecars if either candidate runs. | **CARRIED — analysis tool, not a competing candidate.** |
| **C11 — W93 cheap-preflight composite** | `coordpy.realworldqa_preflight_v1` (already exists from W96-D) | The 4-probe composite for RealWorldQA.  D2 preflight already PASSED at both scales in W96-D; W98 candidates' preflight gates layer on top with W98-specific addressability probes (next section). | **CARRIED — gating discipline for any NIM call.** |
| **C12 — Bounded-window baseline V3** | `coordpy.bounded_window_baseline_v3` | Wrong axis entirely (long-horizon synthetic substrate; nothing to do with VLM K=5 sampling on a single image).  Listed for separation. | **NO — baseline only; wrong axis.** |
| **C13 — Generic prose-summary "memory" mechanism** | (would-be new module; **rejected**) | Anti-pattern per the frontier audit; refuted by W78–W83.  Bullet extraction in D2-B0 already IS a form of prose summary and it FAILED on yes/no.  More of the same in a "memory" wrapper does not help. | **NO — explicit anti-pattern.** |

## W98 candidate slate (final)

* **B1 (C1) — `coordpy.realworldqa_bench_v2`**: structured
  scene-graph extraction + question-typed solver prompt.  Direct
  attack on the W97 root-cause cluster (output-format mismatch
  + lossy extraction).  Highest expected leverage on the
  failure cluster.
* **B2 (C2) — `coordpy.realworldqa_bench_v3`**: direct-vision
  final-turn answerer.  Distinct mechanism (image at decision
  boundary, no extraction-reasoning round trip).  Different
  failure-mode profile than B1.

Both candidates:

* preserve K=5 byte-exact budget;
* preserve same VLM model on every arm (anti-cheat);
* preserve the W95-B0 first-PASS short-circuit so D2-B0's
  multi-choice wins are not regressed;
* are arsenal-driven from this milestone's failure-cluster
  diagnosis, not from W95-B0 inertia;
* are NOT bounded-window / compaction / summary tricks.

A third candidate (B3 = question-type oracle router) is
sketched in this doc but explicitly **not implemented** —
implementing it would scatter the milestone into a
non-substrate-coupled switch baseline.  If both B1 and B2 die
in preflight or pilot, B3 may be considered in a follow-up
milestone as a floor measurement.

## W98 candidate-slate honest predictions (locked BEFORE any NIM call)

### B1 (typed scene-graph + question-typed solver)

* **Highest-probability outcome (subjective):** B1 narrowly
  beats A1 on the 96_504_002 / 30-problem slice
  (B − A1 ∈ [+3, +10] pp), driven primarily by recovering 3 of
  5 unique-A1-rescues via the question-typed solver fix and
  preserving the existing 22 / 30 + 3 / 30.  Probability of
  ≥ +5 pp margin: subjectively ~ 30–40 %.
* **Second-most-likely:** B1 ties or narrowly beats A1
  (B − A1 ∈ [0, +3] pp); the typed solver fix recovers some
  failures but the schema-constrained extraction regresses
  some both-pass problems.  Subjective: 30–40 %.
* **Third:** B1 loses to A1 (B − A1 < 0); the schema rigidity
  breaks more than it fixes.  Subjective: 15–25 %.

### B2 (direct-vision final-turn answerer)

* **Highest-probability outcome:** B2 narrowly beats A1
  (B − A1 ∈ [+3, +10] pp), driven by recovering 4 of 5
  unique-A1-rescues via the final-turn image access and
  preserving the existing 22 / 30 + 3 / 30 via the text-solver
  short-circuit.  Probability of ≥ +5 pp margin: subjectively
  ~ 35–45 %.
* **Second-most-likely:** B2 ties or narrowly beats A1; the
  final-turn re-answerer behaves like A1 sampling on the
  failure cluster (no team advantage at the decision boundary).
  Subjective: 30–40 %.
* **Third:** B2 loses to A1; the final-turn re-answerer
  overrides correct text-solver answers on some both-pass
  problems.  Subjective: 15–25 %.

**No expensive run is purchased on these priors alone.**  Each
candidate must clear its pre-committed preflight + a NIM-free
addressability probe BEFORE any NIM call.

## W98-specific NIM-free addressability probes (in addition to the W93/W96-D composite)

| Probe | Candidate | Threshold | Implementation |
|---|---|---|---|
| **AddrP1 — typed-prompt yes/no recovery rate** | B1 | ≥ 60 % of the 5 unique-A1-rescues plausibly recoverable by typed solver prompt alone | NIM-free re-read of the W97 reader extractions: count how many failure-cluster extractions contain the answer in prose. |
| **AddrP2 — schema-coverage of failure cluster** | B1 | Schema must include `objects[].state`, `objects[].orientation`, `objects[].depth`, `objects[].text_in_object` | Static schema audit against the 5 W97 failures' required primitives. |
| **AddrP3 — direct-vision rescue prior** | B2 | A1 K=5 rescues 5 / 5 of the failure cluster; B2's final turn has equivalent visual access | A1 already proven to PASS on all 5 in W97 (definitionally — they are unique-A1-rescues). |
| **AddrP4 — short-circuit preserves both-pass + unique-B** | B1 + B2 | First-PASS short-circuit logic in the bench preserves D2-B0's 22 / 30 + 3 / 30 wins | Static code audit: both new benches mirror D2-B0's `ship_first_pass` semantics on text-solver turns BEFORE invoking the W98-specific arm. |
| **AddrP5 — budget exact** | B1 + B2 | Total calls = 1 + K + K = 11 per problem at K=5 | Static code audit: same arm counts as D2-B0. |

## Cross-candidate decision logic (PRE-LOCKED before any NIM call)

The W98 milestone uses a **two-phase cheap-discriminator**
pattern to avoid running TWO expensive pilots when one is
sufficient:

**Phase 1 — NIM-free preflight.**  Run W96-D composite + W98
addressability probes for both candidates.  Any candidate
failing any of the AddrP1–AddrP5 probes is **KILLED** and
documented.

**Phase 2 — Cheap NIM pilot (only if entitled).**  Promote the
**single best surviving candidate** to a 1-seed × 30-problem
× K=5 pilot at 11B (~330 NIM calls).  Ranking criterion:
expected leverage on the W97 failure cluster from the AddrP1
and AddrP3 probes.

If exactly one candidate survives preflight: run it.

If both candidates survive preflight: rank by `addressability
score` = (count_of_diagnosed_failures_addressed) and run only
the higher-ranked.  Document the lower-ranked candidate as
deferred to W99 only if the chosen pilot PASSes Phase 2 and
the loser's distinct mechanism remains plausibly load-bearing.

If neither candidate survives preflight: document both kills,
pivot to `COO-9` (second code benchmark) per Part H of the
W98 brief.

## D2-B1 (B1) frozen schema

```json
{
  "scene_summary": "<free-form 1-sentence overview>",
  "question_type": "yes_no | multi_choice_letter | numeric | short_text",
  "expected_answer_form_hint": "<one-line guide for the solver>",
  "objects": [
    {
      "id": "obj_001",
      "label": "car",
      "color": "red",
      "size": "medium",
      "x_region": "center",
      "y_region": "middle",
      "depth": "near",
      "orientation": "facing_left",
      "state": "moving",
      "text_in_object": "",
      "confidence": "high"
    }
  ],
  "counts_by_label": {"car": 2, "person": 3, "traffic_light": 1},
  "spatial_relations": [
    {"a": "obj_001", "b": "obj_002",
     "relation": "left_of", "near": true,
     "depth_relation": "in_front_of"}
  ],
  "text_in_scene": [
    {"location": "background", "text": "STOP"}
  ],
  "direct_answer_hint": "<reader's most confident answer to the question, if it can extract one from the image; the solver may override>",
  "uncertain": ["<list of features the reader could not extract>"]
}
```

Key W98-additions over the W97 D2-B1 sketch:

* `question_type` + `expected_answer_form_hint` — addresses the
  3 output-format-mismatch failures.
* `objects[].state` — addresses 000403 (traffic-light state).
* `objects[].orientation` — addresses 000555 (cars facing
  direction).
* `objects[].depth` and `spatial_relations[].depth_relation`
  — addresses 000718 (depth ordering).
* `direct_answer_hint` — captures any in-extraction answer
  prose so the typed-solver does not have to re-derive it.
* `text_in_object` — addresses sign-reading subset (future).

The solver receives the typed JSON + the question + the
`question_type` hint and is constrained to output the answer
form matching `question_type`.

## D2-B2 (B2) architecture sketch

```
Turn 0: vlm_scene_reader (T=0.0; sees image)
        → free-form extraction (same as D2-B0)
Turn 1: text_solver_initial (T=0.7; no image)
        → candidate-1
Turn 2: text_solver_reflexion (T=0.7; no image)
        → candidate-2
Turn 3: text_solver_reflexion (T=0.7; no image)
        → candidate-3
Turn 4: VLM_FINAL_ANSWERER (T=0.0; sees image + extraction
        + question + question_type_hint + prior candidates)
        → final canonical answer
```

Total: 5 calls = 1 + 3 + 1.  K=5 byte-exact.

**Short-circuit**: if any of turns 1/2/3 PASSes the executor,
the bench ships that answer and skips turn 4 (preserves
D2-B0's wins; no regression risk on multi-choice).  If all
text-solver turns FAIL, turn 4 runs with full visual access
and the canonical answer.

**Critical distinction from W96-C C1 verifier** (which was
empirically refuted):

* W96-C C1 verifier was a **binary agree/disagree** model
  reviewing one prior candidate.  Empirically the verifier
  rescue rate was 0/11 = 0 % at 11B and 1/7 = 14 % at 90B (not
  load-bearing).
* W98 B2 final turn is a **committed answerer** that produces
  the canonical answer with full visual + scene-graph +
  prior-candidates access.  Decision surface is the question
  itself, not a meta-decision about a prior answer.
* W98 B2 is invoked only when all text-solver turns FAIL —
  exactly the failure cluster where A1 wins by re-seeing the
  image (5/5 unique-A1-rescues had A1 PASS).  Mechanistically
  it offers the failure cluster the same image access A1 has
  but with the structured-extraction context the team
  produced.

## Anti-pattern guardrails (do not regress)

* **No bounded context window as a "memory" mechanism.**
* **No compaction / prose-summary as a frontier mechanism**
  (D2-B0's free-text extraction is itself a form of prose
  summary and is being de-risked, not extended).
* **No LLM-as-judge in the executor chain** (every arm routes
  through `evaluate_realworldqa_answer_v1`).
* **No selective retries** (K=5 byte-exact on every arm).
* **No single-seed pilot used as retirement-grade evidence**
  (W98 cheap pilot is a discriminator, not retirement).

## Cross-references

* `docs/RUNBOOK_W98.md` — pre-commit contract + pre-committed
  gates (this milestone).
* `docs/FRONTIER_RELEVANCE_AUDIT_W97_V1.md` + `docs/FRONTIER_RELEVANCE_AUDIT_W98_V1.md`
  — frontier vs baseline-only vs dead vs anti-pattern.
* `docs/RESULTS_W97_REALWORLDQA_D2_B0_PHASE2_V1.md` — W97
  failure-cluster mining (this doc extends the per-failure
  diagnosis).
* `coordpy/realworldqa_bench_v2.py` — B1 implementation
  (next deliverable in this milestone).
* `coordpy/realworldqa_bench_v3.py` — B2 implementation
  (next deliverable in this milestone).
* `tests/test_w98_realworldqa_bench_v2_v3.py` — unit tests
  for both new benches.

## Honest scope of this doc

This is an *arsenal-mining classification + cheap-probe-feasible
diagnosis*, not a benchmark result.  No NIM was spent in
producing this document; the per-failure diagnoses are NIM-free
re-reads of W97 sidecars already on disk.  The only honest claim
from this doc is "the W98 candidate slate is B1 + B2 because
the W97 failure cluster's empirical structure makes both
candidates plausibly addressable, while every other arsenal
candidate is either refuted, infeasible, or wrong-shape."  No
carry-forward is retired by this doc.  No version bump.  No
PyPI publish.
