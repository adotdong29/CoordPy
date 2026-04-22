# Context-Zero: Evaluation Framework

This document defines the metrics, benchmarks, baselines, and success criteria for validating the CASR framework. The thesis is falsifiable — this document specifies exactly what would falsify it.

---

## Core Metrics

### 1. Task Completion Rate at Scale (Primary)

**Definition:** Percentage of benchmark tasks completed successfully as a function of team size and history depth.

**Measurement:** Sweep over:
- N_agents ∈ {3, 5, 10, 20, 50, 100}
- H_depth ∈ {10, 50, 200, 500, 1000} (task rounds to completion)

For each (N, H) pair: run 100 task instances, measure fraction completed correctly.

**What this tests:** Whether CASR maintains task quality as the team scales — the primary claim of the framework.

**Expected result (if thesis is correct):** CASR completion rate stays roughly constant as N and H increase. Naive baseline degrades sharply above ~N=10, H=100.

### 2. Context Efficiency Ratio (CER)

**Definition:**
```
CER = (total tokens processed by naive baseline) / (total tokens processed by CASR)
```

Measured per agent, per task, then averaged.

**What this tests:** The compression factor CASR achieves in practice. The theoretical prediction is CER ≈ N·H / log(N), growing with team size.

**Per-stage attribution:** Also measure:
- CER_Stage1: reduction from Bloom filter (causal selection)
- CER_Stage2: additional reduction from scale projection
- CER_Stage3: additional reduction from surprise filter

This decomposition tells us which stage contributes most — critical for understanding where to invest further.

### 3. Distortion-Accuracy Curve (DAC)

**Definition:** Plot task completion rate vs. distortion budget D for a fixed (N, H).

**How to measure:** Vary τᵢ (the surprise threshold) across agents from τ=0 (deliver everything) to τ=∞ (deliver nothing except fixed-point events). For each τ, measure completion rate.

**What this tests:** Whether the Rate-Distortion bound is approximately tight — i.e., whether CASR is operating near the theoretical minimum for a given accuracy level.

**Expected shape:** A smooth curve from ~100% completion (τ=0, full context) to ~0% completion (τ=∞, no context), with a "knee" in the curve where large compression corresponds to small accuracy loss. The knee should appear at the theoretically predicted R(D) value.

**If the curve has no knee:** The compression is not selective — removing context always hurts proportionally. This would suggest that the causal footprint and scale projection are not effective at separating relevant from irrelevant information.

### 4. Surprise Filter False Negative Rate (SFNR)

**Definition:**
```
SFNR = (events suppressed by Stage 3 that would have changed agent action) /
       (total events suppressed by Stage 3)
```

**How to measure:** Run tasks with τ=0 (full delivery). For each event e suppressed by Stage 3 (δ < τ) in a shadow run, simulate the agent's action with and without e. If the action differs, count it as a false negative.

**Target:** SFNR < 5%. If higher, τᵢ is too aggressive or the world model is miscalibrated.

**What this tests:** The predictive coding stage's precision — whether "low surprise" reliably equals "not needed."

### 5. Bloom Filter False Positive Rate (BFFPR)

**Definition:**
```
BFFPR = (events delivered by Stage 1 that turn out to be causally irrelevant) /
        (total events delivered by Stage 1)
```

**How to measure:** For delivered events, estimate post-hoc whether the event was in the agent's true causal footprint by perturbation analysis.

**Target:** BFFPR < 20% (acceptable over-delivery). The Bloom filter is designed to have zero false negatives; false positives are acceptable overhead.

---

## Benchmarks

### Primary: SWE-bench

SWE-bench (Jimenez et al.) tests software engineering agents on real GitHub issues. Consists of 2,294 tasks requiring code changes to solve issues in popular Python repositories.

**Why SWE-bench:** It has a clear ground truth (does the patch pass the tests?), it involves multiple subtasks per issue (plan → implement → test → debug), and it has existing baselines from major frameworks.

**Modification for CASR evaluation:**
- Instrument the agent framework to log all inter-agent messages and context sizes
- Run SWE-bench with increasing team sizes: single agent, 3-agent team, 5-agent team, 10-agent team
- Measure: (a) resolve rate, (b) tokens per resolution, (c) CER vs. naive baseline

**Key new metric:** tokens per correct resolution — not just resolve rate. Current SOTA systems are not evaluated on this dimension.

### Secondary: GAIA

GAIA (Mialon et al.) tests general-purpose multi-step question answering requiring tool use, web browsing, and multi-modal reasoning.

**Why GAIA:** Tasks involve longer reasoning chains than SWE-bench, stressing the H_depth dimension more than the N_agents dimension.

### Tertiary: HumanEval Multi-Agent (to be constructed)

HumanEval Multi-Agent is a new benchmark derived from HumanEval (function-level coding), modified to:
1. Decompose each function into subtasks (specification, implementation, testing, documentation)
2. Assign subtasks to specific agent roles
3. Define ground-truth subtask assignments (which agent should receive which context)

**Why construct this benchmark:** It provides the only existing structured way to measure whether context was routed to the *right* agent, not just whether the final task succeeded.

**Construction methodology:**
1. Take the 164 HumanEval problems
2. For each problem, manually decompose into 3-5 subtasks
3. Annotate which information is needed at which subtask (creates ground-truth causal footprints)
4. Use the annotated footprints to score routing precision

---

## Baselines

| Baseline | Description | Expected Failure Mode |
|----------|-------------|----------------------|
| Naive full-context | Every agent receives everything (current AutoGen default) | Context size explodes; "lost in the middle" degrades accuracy above N=10, H=100 |
| Recency window | Each agent receives last K tokens (sliding window, K=8K) | Loses constraint information set early in task; fails on long dependencies |
| RAG retrieval | Each agent retrieves top-k similar messages from history | Confuses semantic similarity with causal relevance; slow (embedding lookup per event) |
| LLMLingua compression | Hard compression before delivery (20x ratio) | Destroys rare but critical facts; heuristic not causal |
| MemGPT hierarchical memory | Pull-based memory with episodic/semantic tiers | Fails to deliver unsolicited critical events (push problem, not pull problem) |
| CASR Stage 1 only | Bloom filter but no scale projection or surprise filter | Shows Stage 1 contribution in isolation |
| CASR Stages 1+2 | Bloom filter + scale projection, no surprise filter | Shows MVP contribution; what we build first |
| CASR Full | All three stages | The full claim |

---

## The Falsifiability Condition

The thesis is falsifiable. Specifically, the thesis fails if **any** of:

1. **Irrelevance is not the dominant waste:** If >80% of context tokens in naive systems turn out to be causally relevant (agents actually use them), then routing is not the right attack surface — compression is. Test by measuring the true causal footprint size per agent empirically.

2. **Scale projection destroys necessary information:** If CER_Stage2 is positive (context is reduced) but task completion drops more than the distortion budget allows, the RG projection is too lossy. Test by measuring DAC with Stage 2 disabled.

3. **The world model never calibrates:** If SFNR stays above 20% even after Phase 2 world model training, the predictive coding filter is not viable. Test by measuring SFNR over training iterations.

4. **O(H·log(N)) claim doesn't hold empirically:** If CER grows slower than O(N·H / log(N)) as N and H increase, the hierarchical compression argument is wrong. Test directly from scaling experiments.

If any of these fail, the framework needs fundamental revision, not tuning.

---

## Definition of "Solved"

**The problem is solved when:**

> A 50-agent team operating on a 1,000-round task achieves task completion rate within 5% of a 5-agent team on a 50-round task, while processing fewer total tokens than the naive baseline on the *5-agent, 50-round* task.

This is deliberately demanding: not only must the 50-agent system *not degrade* compared to the 5-agent system, it must be *more token-efficient* than the 5-agent naive baseline in absolute terms — despite coordinating 10x more agents.

This criterion can only be met if:
1. CASR's context routing is genuinely sub-linear in N
2. Scale projection effectively prevents orchestrator flooding
3. The world model surprise filter eliminates redundant transmission

**A weaker threshold for declaring "MVP successful":**

> A 10-agent team on a 200-round task achieves task completion within 10% of a 3-agent team on a 50-round task, with CER ≥ 5 (CASR uses at most 1/5 the tokens of naive baseline).

---

## Measurement Protocol

**Per experiment:**
1. Select benchmark task set (100 tasks minimum per condition)
2. Configure team (N agents, scale assignments, Bloom filters, τᵢ values)
3. Run tasks with full instrumentation (log all events, context sizes, delivery decisions)
4. Compute all five metrics
5. Compare against all baselines
6. Report 95% confidence intervals (not just means)

**Ablation required:** For any positive result, run the ablation table:

| Configuration | CER | Task Completion | Interpretation |
|---|---|---|---|
| Naive | 1.0 | baseline | |
| Stage 1 only | ? | ? | Causal selection contribution |
| Stage 2 only | ? | ? | Scale projection contribution |
| Stage 3 only | ? | ? | Surprise filter contribution |
| Stage 1+2 | ? | ? | MVP |
| Stage 1+2+3 | ? | ? | Full CASR |
| Stage 2+3 only | ? | ? | What happens without causal selection? |

The ablation isolates each component's contribution, identifying where the gains actually come from.
