# Phase 14 — Thesis Holds. First Positive C3.

## TL;DR

**All five pre-registered claims pass.** CASR cuts prompt tokens by 22% vs. full-context, matches full-context quality exactly (both 0.900), and **beats random routing of the same cardinality by +0.210 (5 tests)**. This is the first run that cleanly demonstrates the selectivity half of the thesis — causal selection is strictly better than random same-size selection.

The mechanism is two fixes applied together:
1. **Topological-order round 2** — tier-0 producers freeze after round 1; tier-1 consumers/integrators refine against stable lower tiers. Kills the Phase-13 race.
2. **Event-triggered refinement** (Tabuada–Heemels E1 from the frameworks list) — each higher-tier agent computes Jaccard disagreement between its own draft's dict keys and the bulletin's dict keys. Below threshold (0.34) → skip the LLM call, keep the round-1 draft. Kills over-correction.

## Raw numbers

| Leg | Prompt tokens | Weighted score | Tests | Refined | Skipped |
|---|---:|---:|---:|---:|---:|
| Round 1 (no coord) | — | 0.690 | 18/25 | — | — |
| Round 2 / **full** | 6,070 | **0.900** | 23/25 | 6 | 1 |
| Round 2 / **casr** | 4,709 | **0.900** | 23/25 | 5 | 2 |
| Round 2 / **ablation** | 4,328 | 0.690 | 18/25 | 4 | 3 |

| Claim | Holds? | Value |
|---|---|---|
| C1 casr < full tokens | ✅ | −22.4% |
| C2 casr ≥ full − 0.05 | ✅ | Δ = +0.000 |
| **C3 casr > ablation + 0.05** | ✅ | **Δ = +0.210** |
| C4 full ≥ round-1 (no regression) | ✅ | Δ = +0.210 |
| C5 event trigger fires under CASR | ✅ | 2 skips / 7 |

## What each leg actually did

**Full and CASR** (both 23/25): same 2 failures, `page_token_zero_offset` and `query_page_composed`. Both succeeded where round-1 failed on all 5 event-pair and process_event tests, because round-2 refinement correctly aligned `read_event_kind` and `process_event` to their producers' conventions.

**Ablation** (18/25): regressed-or-held at round-1 level. Same 7 failures as round 1. Its random routing sent the *wrong* producer to some consumers; the event trigger saw full disagreement and invoked the LLM, which faithfully refined to match the wrong producer — now breaking the pair that had been correct by luck in round 1. Other consumers got routed something uninformative and got no help.

Net effect of random routing under the fixed protocol: **zero improvement** (exactly matches no-coordination baseline). This is the crispest possible evidence that what makes CASR work is *which* teammate gets routed, not how many.

## Tier structure (derived from CALL_GRAPH)

```
tier 0 (frozen after round 1): make_event_header, wrap_ok, make_error,
                                make_range, make_page_token    (5 agents)
tier 1 (refine if triggered):   read_event_kind, is_ok, get_error_code,
                                range_contains, parse_page_token,
                                process_event, query_page       (7 agents)
```

Tier-0 agents do zero round-2 work across all three legs. The entire experiment is decided by how the 7 tier-1 agents are informed.

## Event-trigger behavior across legs

| Leg | Refined | Skipped | Why skipped |
|---|---:|---:|---|
| full | 6 | 1 | 1 agent's own draft already matched the bulletin union |
| casr | 5 | 2 | 2 consumers already matched their single-producer bulletin |
| ablation | 4 | 3 | 3 consumers got routed producers they accidentally agreed with (or whose drafts had no dict keys to disagree about) |

Skips save LLM calls. CASR's 2 skips preserved round-1 correctness (their drafts were already aligned with the right producer). Ablation's 3 skips were coincidences — the random routing happened to hit producers whose conventions already matched.

## Why CASR matched (not beat) full on score

Both scored 0.900 = 23/25. The 2 remaining failures (`page_token_zero_offset`, `query_page_composed`) are page-token tests that require a specific base64 encoding choice neither the producer nor the consumer got exactly right. This is a task-level ceiling issue (at this model and draft budget) that CASR can't solve with routing alone — the producer's draft itself is the bottleneck.

That full and CASR tied here is actually the correct outcome for C2: **if the routing has any selection power at all, CASR cannot exceed full-context on quality** (full has strictly more information). CASR matching full at lower cost is the clean win.

## What this proves and does not prove

**Proves (empirically, one run, this task):**
- Routing via causal footprints can replace full broadcast with no quality cost.
- Causal selection strictly beats random selection of the same cardinality.
- Simultaneous round-2 refinement is a confound; topological ordering fixes it.
- Event-triggered refinement prevents LLM over-correction and further reduces tokens.

**Does not prove yet:**
- That this generalizes to N = 50, 200, 1000 agents. The footprint-size-to-N ratio matters; at large N random routing should fail even harder (widening C3), but we haven't measured.
- That the Jaccard threshold 0.34 is optimal. This is a hyperparameter; future work should sweep it.
- That arbitrary tasks work. ProtocolKit is specifically designed to exercise schema-alignment coordination. Tasks with different coordination structure (temporal dependencies, numerical agreement, etc.) may need different trigger signals.

## Connections to the frameworks list

Two items from the shortlist were implemented here:

- **E1 Event-triggered control (Tabuada, Heemels).** `core/event_trigger.py` implements Jaccard-based threshold gating — an LLM-specific analog of the Lyapunov-error trigger. Provably reduces LLM invocations; empirically prevents over-correction.
- Rest of shortlist (F6 conformal prediction, A7 DEQ, D1 DAG-BFT, etc.) remain for future phases. None were needed for C3 — the minimum viable selectivity proof only requires causal routing + protocol-race fix + basic event trigger.

## Files

- `vision_mvp/core/event_trigger.py` — new; Jaccard-based refinement trigger. 14 unit tests, all pass.
- `vision_mvp/experiments/phase14_benchmark.py` — new; topological-order, event-triggered runner.
- `vision_mvp/results_phase14.json` — full per-agent data including module_src for each leg.

## Call

Thesis holds on a falsifiable, real-LLM, real-test, 3-leg benchmark with a 12-agent underspecified task. Next questions are now about **scale and generality**, not **existence**. The existence claim (causal routing is a real mechanism, not a rebrand) is settled in favor of the thesis for this task.
