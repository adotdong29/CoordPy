# Archive — historical documents, not the active position

> Everything under `docs/archive/` is **historical record**, kept for
> auditability and research-history continuity. None of it is the
> *current* scientific or product position of the programme. Where a
> claim under `docs/archive/` disagrees with a doc in `docs/` or the
> repo top level, **the canonical doc wins**.
>
> If you are new to the repo, do not start here. Start with
> [`docs/START_HERE.md`](../START_HERE.md), then
> [`docs/RESEARCH_STATUS.md`](../RESEARCH_STATUS.md), then
> [`docs/THEOREM_REGISTRY.md`](../THEOREM_REGISTRY.md).
>
> Last touched: SDK v3.7 documentation consolidation, 2026-04-26.

## Why this archive exists

The Context Zero / CoordPy programme has produced a long sequence of
milestone notes, theory volumes, sprint prompts, and pre-CoordPy design
documents. Aggregated at the repo root they made the *active* position
hard to see quickly. The consolidation moves the historical record
here so that:

  * the small canonical doc set in `docs/` and at the repo root is
    obviously the live entry point;
  * none of the per-milestone phrasing reads like a current claim; and
  * no research history is deleted — every document below is intact
    and reachable through the index.

## Where the active position lives now

| Topic                                | Canonical doc                                                                |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| One-pass orientation                 | [`docs/START_HERE.md`](../START_HERE.md)                                     |
| What is true *now*                   | [`docs/RESEARCH_STATUS.md`](../RESEARCH_STATUS.md)                           |
| Theorem-by-theorem status            | [`docs/THEOREM_REGISTRY.md`](../THEOREM_REGISTRY.md)                         |
| What may be claimed (do-not-overstate) | [`docs/HOW_NOT_TO_OVERSTATE.md`](../HOW_NOT_TO_OVERSTATE.md)                 |
| Run-boundary capsule formalism (W3)  | [`docs/CAPSULE_FORMALISM.md`](../CAPSULE_FORMALISM.md)                       |
| Team-boundary capsule formalism (W4) | [`docs/CAPSULE_TEAM_FORMALISM.md`](../CAPSULE_TEAM_FORMALISM.md)             |
| Long-running master plan             | [`docs/context_zero_master_plan.md`](../context_zero_master_plan.md)         |
| Two-Mac MLX distributed runbook      | [`docs/MLX_DISTRIBUTED_RUNBOOK.md`](../MLX_DISTRIBUTED_RUNBOOK.md)           |
| Latest milestone (SDK v3.7)          | [`docs/RESULTS_COORDPY_SCALE_VS_STRUCTURE.md`](../RESULTS_COORDPY_SCALE_VS_STRUCTURE.md) |
| Repo entrypoint / quick start        | [`README.md`](../../README.md)                                               |
| SDK release history                  | [`CHANGELOG.md`](../../CHANGELOG.md)                                         |
| Substrate architecture               | [`ARCHITECTURE.md`](../../ARCHITECTURE.md)                                   |

## Archive layout

### `capsule-research/` — early Capsule research-center milestones

These notes record the moves through which the **Context Capsule**
abstraction graduated from a product label into a research center
(formalism + ML problem + unification audit). They predate the
SDK-v3.x runtime contracts; the canonical statement now lives in
[`docs/CAPSULE_FORMALISM.md`](../CAPSULE_FORMALISM.md) and
[`docs/THEOREM_REGISTRY.md`](../THEOREM_REGISTRY.md).

  * `RESULTS_CAPSULE_LEARNING.md`
  * `RESULTS_CAPSULE_RESEARCH_MILESTONE.md`
  * `RESULTS_CAPSULE_RESEARCH_MILESTONE2.md`
  * `RESULTS_CAPSULE_RESEARCH_MILESTONE3.md`
  * `RESULTS_CAPSULE_RESEARCH_MILESTONE4.md`
  * `RESULTS_CAPSULE_RESEARCH_MILESTONE5.md`
  * `RESULTS_CAPSULE_RESEARCH_MILESTONE6.md`

### `coordpy-milestones/` — older CoordPy SDK milestone notes

Per-milestone narrative for SDK v3.0 → v3.6. Useful for tracing
*why* a contract was sharpened the way it was. The current state of
each contract lives in
[`docs/RESEARCH_STATUS.md`](../RESEARCH_STATUS.md) and
[`docs/THEOREM_REGISTRY.md`](../THEOREM_REGISTRY.md); the latest
milestone (SDK v3.7) is kept live at
[`docs/RESULTS_COORDPY_SCALE_VS_STRUCTURE.md`](../RESULTS_COORDPY_SCALE_VS_STRUCTURE.md).

  * `RESULTS_COORDPY_CAPSULE.md` (SDK v3 — Capsule Contract C1..C6)
  * `RESULTS_COORDPY_CAPSULE_NATIVE.md` (SDK v3.1 — capsule-native runtime)
  * `RESULTS_COORDPY_INTRA_CELL.md` (SDK v3.2 — intra-cell + detached witness)
  * `RESULTS_COORDPY_DEEP_INTRA_CELL.md` (SDK v3.3 — sub-intra-cell + audit + determinism)
  * `RESULTS_COORDPY_INNER_LOOP.md` (SDK v3.4 — LLM byte boundary)
  * `RESULTS_COORDPY_TEAM_COORD.md` (SDK v3.5 — multi-agent team coordination)
  * `RESULTS_COORDPY_DISTRIBUTED.md` (SDK v3.6 — two-Mac distributed-inference + cross-LLM)

### `pre-coordpy-theory/` — pre-CoordPy Context Zero research volumes

The original Context Zero research programme: 12 formal theorems, the
72-framework theoretical survey, the Phase-1 MVP spec, the four-phase
roadmap, the seven open questions, and the million-agent vision. The
canonical formal model has since been re-centered on the **Context
Capsule** — see [`docs/CAPSULE_FORMALISM.md`](../CAPSULE_FORMALISM.md)
and the W3/W4/W5/W6 theorem families in
[`docs/THEOREM_REGISTRY.md`](../THEOREM_REGISTRY.md).

  * `PROOFS.md` — original 12 formal theorems (now superseded by the W3..W6 families).
  * `EXTENDED_MATH.md`, `EXTENDED_MATH_[2-7].md` — 7 volumes of mathematical grounding (the 72-framework survey).
  * `OPEN_QUESTIONS.md` — seven foundational open questions.
  * `FRAMEWORK.md` — original problem formulation; routing-as-causal-inference.
  * `EVALUATION.md` — pre-CoordPy metrics + benchmarks + falsifiability.
  * `MVP.md` — Phase-1 spec.
  * `ROADMAP.md` — original four-phase research plan.
  * `VISION_MILLIONS.md` — million-agent forward-looking vision.
  * `MATH_AUDIT.md` — `USED / STRUCTURAL / BUILT` accounting of the 72 frameworks.
  * `HIERARCHICAL_DECOMPOSITION.md` — Phase-7 hierarchical-decomposition design note.
  * `WAVES.md` — Waves 1–5 build-out trail.

### `legacy-progress-notes/` — sprint prompts, agent prompts, old summaries

Prompts written for past coding-sprint agents, paradigm-shift summary
notes, the pre-CoordPy benchmark-reproduction guide, and historical
delivery summaries. These were never canonical — they were artefacts
of one-shot sprints. Kept for traceability only.

  * `ADVANCEMENT_TO_10_10.md`
  * `AGENT_IMPLEMENTATION_PROMPT.md`
  * `AGENT_NETWORK_DESIGN.md`
  * `AGENT_NEXT_SPRINT_PROMPT.md`
  * `AGENT_SPRINT_3WEEK.md`
  * `BENCHMARK.md` — pre-CoordPy reproduction guide; superseded by the *Fastest path from zero to a real report* section in [`docs/START_HERE.md`](../START_HERE.md).
  * `CONTEXT_SOLUTION_REVIEW.md`
  * `FINAL_VALIDATION.md`
  * `PARADIGM_SHIFT_10_10.md`
  * `PARADIGM_SHIFT_DELIVERY_SUMMARY.md`
  * `THEOREMS_AUTO.md` — auto-generated theorem index; superseded by the human-curated [`docs/THEOREM_REGISTRY.md`](../THEOREM_REGISTRY.md).

## How to read an archived doc safely

  1. Note the dated header (every old milestone note has a "Last
     touched" / SDK-version line). Do not generalise its claims past
     that date.
  2. Cross-check any theorem reference against
     [`docs/THEOREM_REGISTRY.md`](../THEOREM_REGISTRY.md) — theorem
     numbers may have been sharpened or retracted since the archived
     note was written.
  3. Cross-check anything labelled *"current"* against
     [`docs/RESEARCH_STATUS.md`](../RESEARCH_STATUS.md). The status
     vocabulary in
     [`docs/HOW_NOT_TO_OVERSTATE.md`](../HOW_NOT_TO_OVERSTATE.md)
     names what is *proved / proved-conditional / mechanically-checked
     / empirical / conjectural / retracted* now.
  4. If an archived doc cites a path like ``docs/RESULTS_COORDPY_X.md``
     or a top-level ``PROOFS.md``, the current location is
     ``docs/archive/<theme>/<filename>.md`` — see the layout above.

## Do not edit archived docs

The archive is a historical record. New writing goes into the
canonical docs; corrections to past readings are recorded as
**retractions** in
[`docs/THEOREM_REGISTRY.md`](../THEOREM_REGISTRY.md) and
[`docs/RESEARCH_STATUS.md`](../RESEARCH_STATUS.md), not by editing the
old milestone note in place.
