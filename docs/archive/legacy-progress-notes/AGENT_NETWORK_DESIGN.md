# Phase 10 — The Agent Network: Hundreds Collaborating on One Task

**The gap Phase 9 exposed:** for role-based workflows at small-N, our
system reduces to classical map-reduce — no novel math applied. To
actually demonstrate something new, we need a **team of agents** where:
- Each agent owns a DIFFERENT piece of one overall task
- Agents send TARGETED messages to each other (not just forward pipeline)
- The per-agent context DOES NOT grow linearly with team size
- Disagreements can be DETECTED and reconciled, not just averaged

This is the "team that talks" you asked for. Not a pipeline, not a
consensus crowd — a real networked organization.

---

## Architecture: three novel mechanisms, one system

### Mechanism 1 — Sparse MoE routing (scales the messaging layer)

**Problem solved:** how does an agent's message reach the 5 relevant
peers without broadcasting to all N?

**Solution:** each agent carries a learned **key vector** `k_i ∈ ℝ^d`
("what I do") that lives in an embedding space. Each outgoing message
carries a **query vector** `q_m = φ(message)` ("what this is about").
The router selects top-k recipients by dot-product similarity.

**Complexity:** O(√N · d) per message via k-means clustered key index
(Routing Transformer trick). For N=500 agents, that's ~22 cluster
centroids to compare against, then ~22 agents in the winning cluster.

**Self-learning:** when agent j's reply is re-used or @-mentioned by
others, nudge j's key toward that query space (InfoNCE-style). Over
time keys organize around actual expertise, no labels needed.

### Mechanism 2 — Hyperbolic address space (scales the namespace)

**Problem solved:** for tree-structured task decompositions, Euclidean
embeddings suffer sibling-subtree crosstalk — messages leak from one
branch to its cousins.

**Solution:** embed agents in the **Lorentz model** of hyperbolic space,
where the volume of a ball grows as e^r. Each subtree of the task DAG
gets its own exponentially-sized "room" at constant radius, so
neighbors-in-the-tree are close but cousins are far. Subscriptions are
**horoballs** (balls tangent to the boundary at infinity), so an agent
can subscribe to "all descendants of subtopic T" with one inner-product
test.

**When it helps:** task DAGs with >4 levels and >4-way branching. Below
that, Euclidean is fine. At scale, the exponential-volume advantage is
decisive.

### Mechanism 3 — Sheaf H¹ consistency monitor (observability)

**Problem solved:** in a team of 500 agents holding 500 partial beliefs,
how do you tell what the team disagrees about?

**Solution:** put a cellular sheaf on the agent graph. Each edge
enforces a local consistency constraint (agents sharing an interface
must match on that interface). Compute the sheaf Laplacian; the
near-zero eigenvectors and the H¹ cohomology generators **localize
disagreement to specific edges and specific coordinates**.

The dashboard literally lights up where the team disagrees. You then
route reconciliation messages only between those agents, not across the
whole team.

**Complexity:** O((N·d)³) worst case (solving a linear system) but
sparse — O(E · d²) for E edges, which is small.

---

## System components

```
vision_mvp/
  core/
    agent_keys.py          Learned keys + clustered index
    sparse_router.py       Top-k MoE routing
    hyperbolic.py          Lorentz-model embeddings, horoball tests
    sheaf_monitor.py       Sheaf Laplacian + H¹ diagnostic
    task_board.py          DAG of subtasks with claim/complete/deps
    network_agent.py       NetworkAgent: inbox + key + task-claim logic
    agent_network.py       Orchestrator: registry + bus + routing + monitor

  tasks/
    collaborative_build.py A real multi-role, interconnected task

  experiments/
    phase10_network.py     Live run with 30–100 LLM agents
```

---

## What a round looks like

1. **Each agent polls** its inbox (only messages the router delivered
   to it) and the task board (claimable subtasks matching its key).
2. **Each agent acts** — processes one or two priority items (LLM call).
3. **Each agent posts** any output: new claims, replies, broadcasts to
   dependent subtasks. Each post carries a topic embedding.
4. **Router delivers** posts to top-k relevant agents via clustered
   key index (O(√N · d)).
5. **Keys adapt** — agents whose replies get re-used move closer to
   those query embeddings.
6. **Every K rounds**, the consistency monitor computes H¹ on the
   agent belief graph. If any edge residual > threshold, a
   reconciliation sub-task is auto-spawned and routed to the involved
   agents.

Per-agent context per round:
- Own state: O(1)
- Inbox: O(k) — router delivers only k most-relevant messages
- Task board read: O(log N) for own subtree
- Key update: O(d)
- **Total: O(k · message_size + log N)**

Compare to AutoGen/CrewAI where per-agent context grows with number of
peers in the conversation — O(N·H) — and blows up past ~100 agents.

---

## The real task — "Collaborative Build"

200 agents jointly produce a deliverable where:
- The overall task decomposes into ~40 subtasks with explicit
  dependencies (DAG)
- Each agent has a specialty (e.g. `data_cleaning`, `auth`, `api`,
  `tests`, `docs`, `deployment`, …) — 10-20 roles
- Agents self-claim subtasks matching their specialty
- Completed subtasks post outputs to the bus; downstream agents pick
  them up via MoE routing
- Final deliverable = joint artifact (say, a specification document
  with all subtask outputs integrated)

Scoring:
- Task completion rate (of 40 subtasks, how many finished)
- Inter-agent messages/round (should be O(k), not O(N²))
- Per-agent peak context (should be bounded)
- Final artifact quality (held-out grading)

---

## Honest scope for Phase 10

**Must** — sparse MoE routing, task board, at least 30 agents, real LLM.
**Should** — hyperbolic embedding as an option (toggle).
**Should** — sheaf H¹ diagnostic printed (doesn't need to auto-reconcile
— just detect and show).
**Would be nice** — 200-agent scale. 30 agents is a realistic evening.

Even at 30 agents, this is the first demo where:
- Agents DO NOT read each other's outputs directly
- Routing is LEARNED via key updates, not hand-wired
- Disagreements are LOCALIZED, not averaged away
- The architecture is the first thing that is qualitatively different
  from a CrewAI DAG

This is what turns the framework from "we do pipelines + consensus"
into "we have an org-scale networked team."
