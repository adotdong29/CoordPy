# Phase 7 Research: Hierarchical Task Decomposition

**Status:** design / proposal (not yet implemented).
**Purpose:** extend CASR from flat-team consensus to real multi-step work.

---

## The gap this closes

Phases 1–6 gave us a flat team of N agents that converge on a single answer.
That's useful for classification / consensus / summarization-by-voting —
but real tasks (code review, software engineering, planning, research) are
*multi-step*:

    Goal → decompose → sub-goals → sub-results → synthesize → final answer

The current CASRRouter collapses the whole team into one workspace. For a
50-agent code review, that's fine. For a 5000-agent research program with
orthogonal workstreams, you want hierarchy: each workstream gets its own
CASR team, plus a coordinator CASR team that fuses workstream results.

MERA (Volume 1) already predicts this is the right structure. MERA is a
*tree* tensor network with disentanglers at each level. Hierarchical CASR
is exactly MERA applied to an agent team.

---

## The proposed architecture

```
                  ┌──────────────────────┐
                  │ Orchestrator (1 CASR)│
                  │   5–10 manager agents│
                  └──────────┬───────────┘
          ┌─────────┬────────┼────────┬─────────┐
          ▼         ▼        ▼        ▼         ▼
       ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
       │Team1│  │Team2│  │Team3│  │Team4│  │Team5│    Worker CASR teams
       │ 200 │  │ 200 │  │ 200 │  │ 200 │  │ 200 │    (each = full CASR)
       │agnt │  │agnt │  │agnt │  │agnt │  │agnt │
       └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
```

**Per-level communication:**
- Worker team i → orchestrator: a single "team summary" embedding + text
  (= one agent equivalent of output).
- Orchestrator → worker team i: a "sub-goal" instruction + current global
  consensus (= one agent equivalent of input).

The team's total context budget is:
```
  C_total = O(L · k_workspace · log(N_level))
```
where L is the number of levels and N_level is the per-level team size.
For two levels with 5 teams of 200 each (N = 1 000 total):
```
  C_total = O(2 · ⌈log₂ 200⌉) = O(16) tokens per round per agent.
```
Same order as flat CASR at N=1000 (which would be ⌈log₂ 1000⌉ = 10),
with slight overhead for the level boundary.

**The scaling win is that each sub-team solves an easier problem.** A flat
1000-agent team converging on "review this 500-line codebase" has to pool
noisy per-file observations. A hierarchical team gives each sub-team one
file or one concern (security / perf / correctness / API / docs) and the
orchestrator fuses.

---

## API sketch

```python
from vision_mvp import HierarchicalRouter

router = HierarchicalRouter(
    worker_teams=[
        CASRRouter(n_agents=200, state_dim=64, task_rank=8),  # security
        CASRRouter(n_agents=200, state_dim=64, task_rank=8),  # perf
        CASRRouter(n_agents=200, state_dim=64, task_rank=8),  # correctness
        CASRRouter(n_agents=200, state_dim=64, task_rank=8),  # API design
        CASRRouter(n_agents=200, state_dim=64, task_rank=8),  # docs
    ],
    orchestrator=CASRRouter(n_agents=10, state_dim=64, task_rank=3),
    sub_goal_decomposer=decompose_fn,          # LLM / rule-based task split
    synthesizer=synthesize_fn,                 # LLM / rule-based aggregation
)

result = router.run(question="Review this code:\n```python\n...\n```",
                    rounds=3)
```

**`decompose_fn(global_goal) -> [sub_goal_i]`:** one LLM call at the
orchestrator. Outputs a list of sub-goals, one per worker team.

**`synthesize_fn(sub_results) -> final_result`:** one LLM call that takes
each worker team's consensus and produces a single structured final answer.

Worker teams run independently in parallel; orchestrator runs once at top
and once at bottom.

---

## Formal properties (to be proven in Phase-7 PROOFS.md section)

**Theorem 13 (Hierarchical peak context).** With L levels and branching
factor b, peak per-agent context:

    peak_i  ≤  L · ⌈log₂ b⌉

For a balanced tree, this is O(log N) in the total N via log_b(N) = L · log_b(b).

**Theorem 14 (Hierarchical bandwidth).** Total messages per round in an
L-level tree with total N agents:

    msg_per_round  =  O(N)   with O(log N / L) per-level cost.

**Theorem 15 (Composition).** CASR at each level + CASR at orchestrator
level is itself CASR — any MinContext bound from Phases 1–6 applies
recursively. No new assumptions needed.

---

## Where this plugs into the existing code

Mostly new scaffolding — the existing `CASRRouter` becomes a *sub-component*
and `HierarchicalRouter` is the new public object. All existing tests and
single-level experiments still run unchanged.

Rough file layout:
```
vision_mvp/
  core/
    hierarchical_router.py    # NEW — orchestrates sub-routers
    task_decomposer.py        # NEW — LLM-driven task splitter
    task_synthesizer.py       # NEW — LLM-driven aggregator
  api.py
    HierarchicalRouter class  # NEW public API
  experiments/
    phase8_hierarchical.py    # NEW — end-to-end demo
  tests/
    test_hierarchical_router.py
```

---

## The research bet

Flat CASR works beautifully on the consensus task (empirically validated
through Phase 6). But real production workflows are *nested*:

- Legal: "Review this 200-page contract" decomposes into sections, then
  clauses, then terms.
- Science: "Find novel drug candidates" decomposes into targets,
  scaffolds, compounds, assays.
- Engineering: "Ship this feature" decomposes into design, implementation,
  tests, deploy, monitor.

Hierarchical CASR is the generalization that makes the framework match
real organizational structure. The math already works (72 frameworks in
`EXTENDED_MATH_[1-7].md` are all naturally hierarchical). The engineering
task is to wire it up.

---

## Estimated effort to ship

| Piece | Est |
|---|---|
| HierarchicalRouter class + tests | 1 day |
| Task decomposer (rule + LLM) | ½ day |
| Synthesizer (LLM) | ½ day |
| Phase 8 demo experiment | ½ day |
| Theorems 13–15 + proofs | 1 day |
| Results writeup | ½ day |

Total: ~4 engineer-days to have a production-ready Phase 7 Hierarchical
addition to context-zero.

This doc is the spec. Build follows.
