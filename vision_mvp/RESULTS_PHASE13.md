# Phase 13 — Honest Result: Thesis Fails C3 Again, Different Reason

## TL;DR

Phase 13 fixed the "task too easy" problem from Phase 12. This one failed C3 for a different and more interesting reason: a **race condition in simultaneous round-2 refinement**. Producer and consumer both re-draft at the same time, each using the other's stale round-1 output, causing both to drift out of sync. Full and CASR regressed **18 → 15**. Ablation accidentally scored highest because random routing sometimes *failed* to connect pairs, preserving round-1 agreement.

The thesis is not yet proven; the mechanism being tested isn't the bottleneck the design assumed. A Phase 14 with a topological-order round 2 (producers locked after round 1) is the natural next step.

## Setup (same structure as Phase 12, different task)

- **Task:** ProtocolKit — 12 functions across 5 producer↔consumer pairs + 2 integrators. Each pair uses a private dict schema whose keys are **deliberately underspecified**. Reference solution passes 25/25.
- **Model:** qwen2.5-coder:7b, local Ollama.
- **Protocol:** Round 1 independent (shared across legs), Round 2 × 3 legs (full / casr / ablation).
- **Claims:** C1 (casr<full tokens), C2 (casr≥full−0.05), C3 (casr>ablation+0.05).

## Raw numbers

| Leg | Prompt tokens | Weighted score | Tests |
|---|---:|---:|---|
| Round 1 (no coord) | — | **0.690** | 18/25 |
| Round 2 / **full** | 13,025 | 0.570 | 15/25 |
| Round 2 / **casr** | 7,260 | 0.570 | 15/25 |
| Round 2 / **ablation** | 7,621 | **0.750** | **19/25** |

| Claim | Holds? | Delta |
|---|---|---|
| C1 — casr tokens < full tokens | ✅ | −44.3% |
| C2 — casr ≥ full − 0.05 | ✅ | +0.000 |
| C3 — casr > ablation + 0.05 | ❌ | **−0.180** |

## Per-test diff — identifying the mechanism

Compared to round-1 drafts, each round-2 leg:

```
full       broke 4 tests, fixed 1  (net −3)   all 4 breaks are range-*
casr       broke 4 tests, fixed 1  (net −3)   same 4 breaks
ablation   broke 4 tests, fixed 5  (net +1)   same 4 breaks, +5 fixes
```

**All three legs broke the same 4 tests.** They are `range_contains_inside`, `range_contains_exclusive_end`, `range_contains_before_start`, `query_page_range_contains` — the entire range-spec pair plus an integrator that uses it.

**Diagnosis: simultaneous-refinement race.** Round 1 produced locally-consistent conventions: `make_range` returned `{"start": int, "end": int}`; `range_contains` happened to read the same keys. Round 2 is simultaneous — both agents re-draft at the same time, each seeing the *other's round-1 draft* in the bulletin. Each LLM aggressively "refines" under the coordination frame, often switching to a different convention (`{"lo", "hi"}`). Because both sides switch independently, they desync — and the newly-drafted pair no longer agrees, even though the round-1 pair did.

Ablation's random bulletins often fail to connect the pair at all. When that happens, the consumer sees no evidence of the producer's convention and keeps its round-1 draft intact — which accidentally still agrees with the also-unchanged producer. This is why ablation *increased* from 18 to 19 while the targeted legs regressed.

## Why ablation also broke 4 tests

The ablation footprints are seeded (seed=42). For `range_contains`, the random draw happened to include `make_range` — exactly the "bad" connection. So ablation suffered the same desync on the range pair. That it still ended up at 19/25 is because random ablation also accidentally fixed 5 other tests: its `read_event_kind` got routed to `make_event_header` by chance, and its `process_event` got useful partners. Net: +1 for ablation, −3 for the other two.

## What this says about the thesis

The routing mechanism (footprint-based bulletin filtering) is working as designed — CASR successfully drops ~66% of messages without hurting quality relative to full-context. **That half of the thesis is confirmed twice now** (Phase 12 and Phase 13 both show C1 + C2 hold).

What neither phase has demonstrated is **C3 — that causal selection is better than random selection of the same size**. Phase 12 failed because the task was too easy (no round-2 signal). Phase 13 failed because the round-2 protocol itself has a race-condition confound: when you route good information to both sides of a dependency simultaneously, both sides move, and they move apart. The routing quality cannot be separated from this confound without a fix.

## The fix — Phase 14 design

Topological-order round 2. Concretely:

1. **Tier the agents by dependency depth:** producers (who depend on nothing) at depth 0; consumers at depth 1; integrators at depth 2.
2. **Freeze lower tiers before higher tiers refine.** Producers' round-1 drafts become their final outputs. Consumers refine once, seeing the *frozen* producers. Integrators refine last, seeing frozen producers + consumers.
3. Under this protocol:
   - In **full**, every higher-tier agent sees all frozen lower-tier agents (no race — all dependencies are stable when a consumer drafts).
   - In **CASR**, consumer sees only its footprint (the right producer).
   - In **ablation**, consumer sees a random tier-0 agent.

This removes the race entirely. If CASR still does not beat ablation, C3 is genuinely false (i.e., random selection of the same cardinality suffices). If CASR does beat ablation, the thesis has its proof.

## What's saved

- `vision_mvp/results_phase13.json` — full per-agent per-leg data including module_src for each leg (all range-contains drafts diffable side-by-side).
- `vision_mvp/experiments/phase13_benchmark.py` — the runner.
- `vision_mvp/tasks/protocol_codesign.py` — task with 25 tests, reference passes 25/25.

## Honest call

Two experiments in, C1 and C2 are solid. C3 remains untested due to (a) easy task, then (b) protocol race. Phase 14 with topological-order refinement is the third and cleanest attempt. If C3 still fails there, the "causal selection beats random same-size" claim is probably genuinely wrong at this N, and the thesis pivots to efficiency-only (which is itself a worthwhile claim — 44–55% token reduction at no quality cost is not nothing).
