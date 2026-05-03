# Context-Zero: Research Roadmap

## The Bet (One Sentence)

Context bloat in multi-agent systems is a routing problem — solved by computing per-agent minimum sufficient statistics via causal abstraction and scale projection — yielding O(H·log(N)) effective context per agent instead of O(N·H²).

---

## Phase 1: MVP (~2 months)

**Goal:** Validate that causal-selection + scale-projection reduces context without breaking tasks.

**What we build:**
- 3-agent team (1 orchestrator + 2 workers)
- Hand-specified causal footprints (no learning yet)
- LLM-based scale projections (structured prompt summaries)
- Simple event-sourced message bus
- SWE-bench evaluation harness

**What Stage 3 status:** Disabled (τᵢ = 0, world model not implemented)

**Success gate:** CER ≥ 3, task completion drop < 5% on SWE-bench Lite evaluation set

**Key risk:** Hand-specified footprints are wrong. Mitigation: audit every failed task to determine which filtered event was needed, then adjust footprint.

**Theoretical work in parallel:**
- Investigate fixed-point convergence (Open Question 1)
- Verify composability condition for the five-level scale taxonomy

**Deliverables:**
- `mvp/` codebase (event bus, agent harness, SWE-bench integration)
- Phase 1 evaluation report (CER histogram, completion rates, ablation, failure analysis)
- Updated FRAMEWORK.md with any theoretical corrections from MVP experience

---

## Phase 2: Learned Routing (~4 months)

**Goal:** Replace hand-specified footprints with learned Bloom filters. Add Stage 3 (predictive coding).

**What we build:**
- Automated causal footprint estimation via empirical perturbation analysis
- Bloom filter construction from estimated footprints
- Simple world model M_i (next-event embedding prediction, small transformer)
- Stage 3 surprise filter with tunable τᵢ
- Scale from 3 to 10 agents

**Bloom filter learning process:**
1. Run 500 tasks with τᵢ = 0 (all events delivered, as Phase 1)
2. For each agent × event type pair, randomly mask the event and measure Δ in task outcome
3. Events with significant Δ → in footprint. Events with Δ ≈ 0 → not in footprint.
4. Build Bloom filter with empirically estimated footprint
5. Compare BFFPR (false positive rate) vs. hand-specified baseline

**World model training:**
1. Collect Phase 1 event logs as training data
2. Train M_i: input = last 20 events in aᵢ's context, output = predicted next-event embedding
3. Measure SFNR (false negative rate) on held-out tasks
4. Set τᵢ = threshold at which SFNR < 5%
5. Begin bootstrapping curriculum (gradually increase τᵢ across iterations)

**Success gate:** CER ≥ 10 on 10-agent teams, task completion within 8% of Phase 1 baseline

**Theoretical work:**
- Formalize the relationship between empirically estimated footprints and the theoretical do-calculus definition
- Begin world model bootstrapping convergence analysis (Open Question 3)
- First pass at optimal β problem (Open Question 6)

**Deliverables:**
- `routing/` module: automated footprint estimation + Bloom filter construction
- `world_model/` module: M_i training, prediction, surprise computation
- Phase 2 evaluation report: compare learned vs. hand-specified footprints, Stage 3 contribution
- Draft of fixed-point convergence proof (partial)

---

## Phase 3: Generalization (~6 months)

**Goal:** Remove the remaining manual constraints. Scale to 50+ agents. Evaluate beyond software development.

**What we build:**
- Scale inference model (given task_description + role_description → scale level)
- DAG topology support (peer-to-peer agent communication)
- HumanEval Multi-Agent benchmark
- GAIA evaluation

**Scale inference:**
- Train a classifier on (task_description, role_description, measured_scale_outcome) triples
- "Measured scale outcome" = the scale assignment that, in Phase 1-2 data, minimized context while maintaining task completion
- Evaluate: does inferred scale match human-assigned scale? Does task completion improve?

**DAG topology:**
- Implement relative-scale projection (scale direction is from sender to receiver, not absolute)
- Validate composability of relative-scale projections
- Test with a 5-agent team that has peer-to-peer communication between workers

**HumanEval Multi-Agent construction:**
- Decompose HumanEval 164 problems into 3-5 subtasks each
- Manually annotate causal footprints per subtask-role pair (ground truth)
- Evaluate CASR routing precision against ground truth annotations

**Success gate:**
- 50-agent team within 10% completion of 5-agent team, CER ≥ 15
- Scale inference accuracy > 80% on held-out roles and tasks
- DAG topology works without composability violations

**Theoretical work:**
- Complete DAG composability analysis (Open Question 4)
- Write up scale inference as a formal meta-learning problem
- Submit CASR framework as a workshop paper

**Deliverables:**
- `scale_inference/` module
- `dag_routing/` extension
- HumanEval Multi-Agent benchmark dataset
- Phase 3 evaluation report
- Workshop paper draft

---

## Phase 4: Theory Completion (Ongoing)

**Goal:** Make the theoretical claims rigorous. Submit as a conference paper.

**What we prove (or fail to prove):**
- Fixed-point convergence theorem (Open Question 1)
- O(H·log(N)) complexity bound, rigorously (with explicit conditions)
- Tightness of the Rate-Distortion bound for CASR
- Convergence of the world model bootstrapping curriculum

**What we evaluate:**
- Full "solved" criterion: 50-agent/1000-round team within 5% of 5-agent/50-round team
- Cross-task transfer experiments (Open Question 7)
- Adversarial robustness analysis (Open Question 5)

**What we write:**
- Full conference paper: problem formulation, CASR framework, empirical validation, theoretical analysis
- Target venues: NeurIPS, ICML, or ICLR (multi-agent track or systems track)

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Hand-specified footprints are systematically wrong for SWE-bench | Medium | High (kills Phase 1) | Run 20 exploratory tasks before full eval; adjust footprints based on failure analysis |
| Scale projection LLM calls add too much latency | High | Medium (limits practical use) | Replace with fine-tuned smaller model in Phase 2; cache projections for repeated event types |
| Fixed-point convergence fails theoretically | Low | High (undermines framework) | Fall back to empirically-defined T_i* (measured, not derived); weaker but sufficient claim |
| World model never calibrates | Medium | High (kills Stage 3) | If SFNR stays > 20% after 1000 training tasks, drop Stage 3 from the framework |
| SWE-bench isn't the right benchmark | Low | Medium | Add GAIA early (Phase 2) to ensure results aren't artifact of SWE-bench structure |
| The dominant waste is redundancy, not irrelevance | Medium | High (kills thesis) | Run falsifiability analysis in Week 3 of Phase 1 before committing further |

---

## The Falsifiability Check (Do First)

Before committing to the full roadmap, run this check in Week 3 of Phase 1:

1. On 20 SWE-bench tasks with naive full-context sharing, log every event received by each agent
2. For each event, measure: did the agent's output change when this event was masked?
3. Calculate: what fraction of events were causally relevant (masking changed output)?

**If > 80% of events are causally relevant:** The thesis is wrong. Most context is necessary, not irrelevant. Pivot to redundancy compression (LLMLingua-style) rather than routing.

**If < 50% of events are causally relevant:** Strong evidence for the thesis. Proceed with confidence.

**If 50-80%:** Mixed. Routing helps but isn't the whole story. Combine routing with compression.

This measurement costs ~2 days of compute and saves potentially wasted months.

---

## What Success Looks Like

A multi-agent AI system where:

- Teams of 50 agents work on 1,000-round tasks without context collapse
- Each agent receives only what it needs — no "lost in the middle" degradation
- Adding agents improves task throughput, not just context load
- The routing decisions are auditable and interpretable (you can see why each event was or wasn't delivered)
- The framework can be dropped into existing agent systems (AutoGen, CrewAI, LangGraph) as a layer

At that point, the context window is no longer the bottleneck for agent team scaling. The bottleneck becomes agent quality and task decomposition — which are tractable engineering problems.
