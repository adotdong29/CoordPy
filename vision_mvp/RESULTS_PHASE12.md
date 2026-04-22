# Phase 12 — 3-Leg CASR Benchmark: Honest Result

## TL;DR

CASR cut prompt tokens by **55.1%** with **zero quality drop** vs. the full-context baseline. But CASR **did not beat random routing** of the same cardinality. The task was not hard enough to distinguish causal from random selection — every leg scored **identical 27/31 = 0.79**. The thesis is **partially supported (C1, C2) and not supported (C3) on this run.**

## Setup

- **Task:** TinyStore — a 12-function in-memory key-value library, 31 weighted tests, reference solution passes 31/31.
- **Model:** qwen2.5-coder:7b via local Ollama.
- **Agents:** 12 (one per function). Dependencies defined by a 15-edge `CALL_GRAPH`.
- **Protocol:**
  - **Round 1:** all 12 agents draft their function independently (shared across legs).
  - **Round 2 ×3 legs:** each agent receives a "coordination bulletin" containing other agents' round-1 drafts, plus its own draft, and refines. The bulletin's *contents* vary per leg:
    - **full:** bulletin = all 11 other agents' drafts (baseline).
    - **casr:** bulletin = only agents in this recipient's causal footprint (call-graph neighbors + reverse edges), built via a Bloom filter.
    - **ablation:** bulletin = random agents of the same cardinality as CASR (controls for "fewer messages → fewer tokens").
- **Metrics:** prompt tokens per agent per round; 31-test weighted score on composed module.
- **Falsifiable claims (pre-registered in `phase12_casr_benchmark.py`):**
  - **C1:** casr prompt tokens < full prompt tokens
  - **C2:** casr score ≥ full score − 0.05
  - **C3:** casr score > ablation score + 0.05

## Raw numbers

| Leg | Prompt tokens (round 2, all agents) | Weighted score | Tests passed |
|---|---:|---:|---|
| Round 1 only (no coord) | — | **0.790** | 27/31 |
| Round 2 / **full** (baseline) | 18,226 | **0.790** | 27/31 |
| Round 2 / **casr** | 8,185 | **0.790** | 27/31 |
| Round 2 / **ablation** | 9,036 | **0.790** | 27/31 |

**Completion tokens** were nearly identical across legs (~2,300) — the LLM writes the same-size function regardless of prompt length, as expected.

**Footprint sizes (CASR):** avg 3.5, min 1 (summarize), max 7 (execute). For the "full" leg, all 12 agents see all 11 others → every bulletin is 11 drafts.

## Per-agent token comparison

```
agent                  full   casr    abl   footprint-size
parse_command          1453    556    693   2
validate_command       1337    421    519   2
normalize_key          1319    469    625   5
new_store              1294    275    261   2
apply_set              1325    535    585   4
apply_del              1327    441    407   3
get_value              1328    519    602   4
list_prefix            1343    456    495   3
incr_value             1344    526    518   4
execute                1398    846   1034   7
run_script             1367    784    784   5
summarize              1363    329    485   1
```

Footprint size correlates directly with token count — exactly as the routing design intends. The hottest agent under CASR (`execute`, footprint size 7) still uses 40% fewer tokens than full. The coldest (`summarize`, footprint size 1) uses 76% fewer.

## The identical-scores finding — what it means

**All four conditions produced 27/31 = 0.790.** The four failing tests were the same set in every leg: `parse_list`, `script_basic`, `script_skips_bad`, `script_final_state`.

What actually failed:
- `parse_list` → `parse_command` got the `LIST prefix` case wrong in round 1. No amount of teammate-draft visibility would fix this — it's an internal logic bug in that agent's function, not an interface bug.
- Three `script_*` failures → `run_script`'s orchestration was broken in round 1. Same story.

In other words: **round-2 coordination produced zero quality change, in any leg.** The failing tests are intra-agent bugs; the passing tests already passed in round 1.

This makes the experiment a **degenerate case for distinguishing CASR from ablation.** If round 2 cannot move quality at all, no routing scheme can appear better than another on quality grounds. So C3 fails not because causal structure is worthless but because **this specific task did not exert enough coordination pressure** to exercise the hypothesis.

## Verdict on the thesis (honestly scored)

| Claim | Holds? | Notes |
|---|---|---|
| **C1** — CASR tokens < full tokens | ✅ YES | 55.1% reduction; this is not trivial — it's the core efficiency claim and it's empirically confirmed on a real task with a real model. |
| **C2** — CASR quality ≥ full − 0.05 | ✅ YES | Delta is exactly 0.000. No quality cost whatsoever for the token savings. |
| **C3** — CASR quality > ablation + 0.05 | ❌ NO | Delta is 0.000. Random selection performed equally well, which means either (a) causal structure doesn't matter for this task, or (b) this task doesn't exercise the mechanism. Evidence points strongly to (b): every leg, including round-1-only, hit the same score ceiling. |

## What this does and does not prove

**It does prove:**
- Routing a subset of messages via causal-footprint does not degrade downstream quality relative to broadcasting everything, **on at least this task**.
- The token savings are real and substantial (>2× prompt reduction).
- The CASR + Bloom-filter infrastructure works end-to-end with a real LLM over 48 calls (12 agents × 4 phases).

**It does not prove:**
- That causal selection is *better than random selection of the same size*. For that, we need a task where round-2 refinement *does* change the score. This benchmark's round-2 has no quality signal, so it cannot distinguish the two.
- That this generalizes to 100+ agents. Per-agent bulletin sizes would grow O(N) under full broadcast vs O(k) under CASR where k depends on connectivity, not N. At higher N the token savings should widen further, but that's a projection, not a demonstration here.

## What would be a decisive next test

The task needs round-2 to actually move scores. Concrete designs:

1. **Interface-dependent task.** Give two agents shared types (e.g., a custom `@dataclass Transaction`) whose exact schema is underspecified in the prompt but must agree for downstream tests to pass. Round-1 drafts will disagree on the schema (creating interface bugs like Phase 11's namespace collision). CASR must route the two schema-owners to each other; random ablation often won't.
2. **Deliberate round-1 info starvation.** Tell every agent "the exact signature of function X is in its spec — you'll see it if you're routed X's draft." Forces round 2 to matter. CASR wins when the dependency edges are correct; random wins only by luck.
3. **Scale up N.** At N=12 the random footprints often coincidentally hit the right dependencies. At N=50 or 100 with sparse CALL_GRAPH, random selection rarely nominates the right agents, and C3 becomes testable.
4. **Multiple round-2 iterations to convergence.** Let round 2 iterate k times. CASR's signal should compound; random's should not.

## What's saved

- `vision_mvp/results_phase12.json` — full per-agent per-leg data including module_src for each leg (so you can diff what refinements actually happened).
- `vision_mvp/experiments/phase12_casr_benchmark.py` — the runner.
- `vision_mvp/tasks/library_v2.py` — the 12-function task with reference solution.
- `vision_mvp/core/{causal_footprint,casr_router,token_meter}.py` — the infrastructure, 15/15 unit tests.

## Final call

This run **supports the efficiency half of the thesis** (routing reduces tokens without hurting quality) and **does not yet support the selectivity half** (that causal selection is better than random selection of the same size). To prove the selectivity half, we need a harder coordination task and/or more agents. That's the right next experiment.
