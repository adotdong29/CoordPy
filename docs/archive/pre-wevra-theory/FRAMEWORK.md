# Context-Zero: Theoretical Framework

## Abstract

Multi-agent AI teams face an exponential context crisis. When N agents communicate over H rounds of history, naive context sharing grows as O(N·H²) — a 10-agent team with 100 rounds of history generates ~100,000 context-units of total information load. No existing framework addresses this at the level of first principles.

This document argues that context bloat is a **routing problem, not a compression problem**. Current solutions (LLMLingua, MemGPT, KV cache tricks) optimize for redundancy — removing tokens that say the same thing twice. The dominant waste is *irrelevance* — transmitting to agent A information whose interventional effect on A's actions is zero. We propose **CASR** (Causal-Abstraction Scale-Renormalized Routing), a three-stage pipeline grounded in information theory, causal inference, and renormalization group theory that reduces per-agent effective context from O(N·H²) to O(H·log(N)) for hierarchically structured teams.

**The falsifiable claim:** A well-designed routing scheme reduces effective context per agent by >80% without measurable task-completion degradation on structured multi-agent benchmarks.

---

## Section 1: The Problem, Formally

### 1.1 Definitions

Let an **agent team** be a directed graph G = (A, E) where A = {a₁, a₂, ..., aₙ} is the set of agents and E ⊆ A × A is the communication topology.

Each agent aᵢ maintains a **context window** Cᵢ(t) at discrete timestep t. A **message** m_{j→i}(t) is transmitted from aⱼ to aᵢ at time t. The context update rule in naive full-sharing systems is:

```
Cᵢ(t+1) = Cᵢ(t) ∪ { m_{j→i}(t) : (j,i) ∈ E }
```

### 1.2 Context Growth Rates

**Orchestrator pattern** (one orchestrator, N workers):
- Each worker sends results upward: |m_{worker→orch}(t)| ≈ k tokens
- Orchestrator accumulates: |C_orch(H)| = N·H·k = O(N·H)

**Worker pattern** (worker receives orchestrator state + sibling updates):
- If orchestrator passes full context down: |C_worker(t)| = |C_orch(t)| = O(N·t)
- Worker context after H rounds: |C_worker(H)| = Σᵢ O(N·i) = O(N·H²)

**The O(N·H²) collapse:** In a recursive multi-agent system where each agent forwards its full context to sub-agents, context at depth d grows as O(N^d · H^(2d)). This is the exponential context explosion that makes deep agent hierarchies currently infeasible.

### 1.3 The Effective Context Window Problem

Models advertise large context windows (200K tokens) but effective utilization is dramatically smaller. The **"lost in the middle" phenomenon** (Liu et al., TACL 2024) shows:

- >30% accuracy drop when the relevant answer document moves from position 1 or 20 to position 10 (out of 20)
- Performance follows a U-shaped curve — models over-attend to beginning and end tokens
- This is not a bug but a mathematical consequence of how positional encodings interact with softmax attention

**Why this happens mathematically:** Rotary Position Embeddings (RoPE) introduce a long-term decay effect in attention weights. For tokens at relative distance r, the attention contribution is modulated by a factor proportional to cos(r·θ) for some base angle θ. At large r, this factor decays toward zero, systematically de-emphasizing middle-sequence tokens.

Formally, for attention score A_{ij} between query at position i and key at position j:
```
A_{ij} = softmax( q_i · R_{i-j} k_j / √d )
```
where R_{i-j} is the rotation matrix for relative distance (i-j). The rotational decay means that even if k_j contains maximally relevant information, if |i-j| is large, A_{ij} approaches the minimum of the softmax distribution.

**Consequence:** Stuffing context naively doesn't help past ~50K effective tokens for most tasks. The architectural solution (bigger context) is necessary but not sufficient — you also need to ensure relevant information is *routable* to the positions where the model attends.

### 1.4 The Routing Problem

Current frameworks have no theory of relevance. They use one of:
- **Recency**: keep the last K tokens (sliding window)
- **Similarity**: retrieve the top-k most similar chunks (RAG)
- **Everything**: pass full context (naive, what most frameworks do)

None answers the correct question:

> *What information does agent aᵢ need to take its next action, conditional on aᵢ's current task zᵢ and the global goal G?*

This is a **conditional independence** question. Let X be the full context history, Y_i be agent aᵢ's next action, and z_i be aᵢ's current task description. The information aᵢ needs is the **Markov blanket** of Y_i given z_i in the causal graph over (X, z_i, Y_i).

Define: x_j is **causally relevant** to aᵢ if:
```
P(Y_i | do(x_j = v), z_i) ≠ P(Y_i | z_i) for some value v
```
Otherwise, x_j is **causally irrelevant** — intervening on it changes nothing, and it can be safely dropped from Cᵢ.

### 1.5 Why Existing Solutions Fail

Each existing approach makes an implicit assumption about what constitutes "relevant" context. Here is where each breaks:

| Approach | Implicit Relevance Assumption | Where It Breaks |
|----------|------------------------------|-----------------|
| Hard compression (LLMLingua) | Tokens with low perplexity are less relevant | Proper nouns, rare constraints, and critical edge cases have high perplexity and are pruned |
| Soft compression (AutoCompressor) | Continuous embeddings capture all semantics | Learned embeddings may discard structurally rare but task-critical facts |
| KV cache reuse (KVzip) | The same compressed cache serves all agents | Different agent roles need different context projections |
| Hierarchical memory (MemGPT) | Episodic/semantic/procedural ontology covers all facts | Emergent task structures don't fit predefined memory types |
| Sliding window attention | Temporal locality — recent tokens are more relevant | Constraints set 50 rounds ago remain binding |
| State space models (Mamba) | Fixed-size recurrent state is sufficient | Multi-resolution tasks need different granularities simultaneously |
| RAG | Semantic similarity to current query ≈ relevance | A fact can be causally necessary without being semantically similar |

The common failure: all optimize for *proxy measures* of relevance (perplexity, similarity, recency) rather than the principled measure — causal effect on agent action.

---

## Section 2: Prior Work

### 2.1 Context Compression

**LLMLingua** (EMNLP 2023, ACL 2024): Up to 20x compression with minimal performance loss using a small "budget controller" model to identify low-perplexity (less surprising) tokens for removal. **Gap:** task-agnostic — compresses globally rather than per-agent-role.

**AutoCompressor** (EMNLP 2023): Fine-tunes an LLM to compress long contexts into summary vectors (continuous embeddings as soft prompts). **Gap:** loses discrete structure; summary vectors are opaque to causal analysis.

**LLMLingua-2** (2024): BERT-level encoder for token classification; 3-6x faster than LLMLingua. Still fundamentally a redundancy-removal approach.

**Gist Tokens**: Condenses prompts into learnable special tokens. High compression ratio but requires fine-tuning per task distribution.

**The gap in all compression work:** These operate on a single agent's context after it has already been assembled. They do not address the *routing decision* — what to include in the context in the first place.

### 2.2 Memory Architectures

**MemGPT / Letta** (arXiv:2310.08560): Treats LLM as a processor with virtual context management inspired by OS memory hierarchies (working memory → main memory → disk). Agents manage their own memory through explicit function calls. **Gap:** pull-based (agents fetch when they need) — doesn't solve push-routing (who sends what without being asked).

**G-Memory** (arXiv:2506.07398): Hierarchical aggregate trees for multi-agent memory, compressing at each level. **Gap:** still requires a predefined hierarchy; does not derive the hierarchy from causal structure.

**Mem0**: Hybrid retrieval — structural lookups first, then vector search. Practical engineering, but heuristic relevance determination.

**The gap:** All memory architectures solve *retrieval* (given a query, find relevant facts). They don't solve *routing* (given an event, determine which agents need it without waiting to be asked).

### 2.3 Multi-Agent Communication Protocols

**AutoGen** (2024): Actor model with asynchronous message passing. Good observability via OpenTelemetry. **Failure:** agents in conversation mode pass full conversation histories; no mechanism to scope context to recipient's role.

**CrewAI**: Role-based memory with RAG support. Schema validation via Pydantic. **Failure:** implicit shared context becomes a liability at scale; no explicit causal footprint per role.

**LangGraph**: Stateful directed graphs with checkpointing and explicit state annotations. **Failure:** explicit state doesn't prevent loading enormous state objects into every node.

**MetaGPT**: Encodes Standard Operating Procedures into prompt sequences. **Failure:** SOPs structure the bloat but don't reduce it; context still scales with team size and history.

**Common failure across all frameworks:** inter-agent communication is treated as a software engineering problem (message passing, serialization, delivery guarantees) rather than an information-theoretic problem (minimum sufficient statistics per recipient).

### 2.4 Theoretical Frameworks Applied in Adjacent Domains

**Information Bottleneck** (Tishby et al. 1999; arXiv:2501.00999): Applied to explain DNN training as lossy compression. Shows that models converge toward the IB bound: min I(T; X) - β·I(T; Y). **Not yet applied** to agent-to-agent communication.

**Renormalization Group** in ML (Mehta & Schwab 2014): RBMs perform RG-like coarse-graining. **Not yet applied** to multi-agent context hierarchies.

**Predictive Coding** (Friston's Free Energy Principle, *Nature Reviews Neuroscience* 2010): The brain transmits only prediction errors. **Not yet applied** to agent communication protocols.

**Causal Abstraction** (Geiger et al., arXiv:2301.04709): Framework for identifying when a high-level causal model is implemented by a low-level one. **Not yet applied** to context selection.

**The gap:** Each of these frameworks has been applied *within* a single model or *within* a single domain. None has been applied *across* agent boundaries to solve the routing problem.

---

## Section 3: The CASR Framework

### 3.1 Overview

CASR (Causal-Abstraction Scale-Renormalized Routing) is a three-stage pipeline applied to each event before it is delivered to a recipient agent.

```
Event e (from sender aⱼ)
    │
    ▼
[Stage 1: SELECT]
Causal footprint filter:
Is e in the interventional Markov blanket of aᵢ given zᵢ?
→ If NO: drop. Bloom filter O(1) pre-filter.
→ If YES: continue.
    │
    ▼
[Stage 2: PROJECT]
Scale projection:
Map e to recipient aᵢ's operating scale sᵢ.
P_{sᵢ}(e) = compressed representation at granularity sᵢ
    │
    ▼
[Stage 3: TRANSMIT]
Surprise filter:
δ = KL(aᵢ's prediction of state || actual state after e)
→ If δ < τᵢ: skip (aᵢ's world model already predicted this)
→ If δ ≥ τᵢ: deliver P_{sᵢ}(e) to aᵢ
```

### 3.2 Stage 1: Causal Abstraction (SELECT)

**Definition:** The **causal footprint** of agent aᵢ is the set of event types whose interventional effect on aᵢ's action distribution is nonzero:

```
FP(aᵢ, zᵢ) = { e : ∃v such that P(Yᵢ | do(e=v), zᵢ) ≠ P(Yᵢ | zᵢ) }
```

**Algorithm (offline, per agent role):**
1. Enumerate event types in the system (tool calls, messages, state changes)
2. For each event type e and agent role rᵢ: estimate whether e is in FP(aᵢ, zᵢ) by do-calculus over the team's causal graph, or by empirical testing (perturb e, measure Δ in aᵢ's action distribution)
3. Store FP(aᵢ) as a Bloom filter B_i for O(1) membership queries

**At runtime:** Before delivering event e to aᵢ, query B_i(e.type). If B_i returns "definitely not in footprint," drop without reading e's content. If B_i returns "possibly in footprint," proceed to Stage 2.

**Key property:** Bloom filters have no false negatives — if B_i says "not in footprint," the event is definitively irrelevant. False positives (events that pass the filter but are actually irrelevant) result in over-delivery, not missed information. This asymmetry is safe: we can accept unnecessary transmission but not missed necessary information.

**Bloom filter staleness:** Footprints are recomputed at task-phase transitions. Within a phase, the causal structure is assumed stable.

### 3.3 Stage 2: Renormalization Group Projection (PROJECT)

**Scale assignment:** Each agent aᵢ is assigned an operating scale sᵢ at instantiation. For software development teams:

| Level | Scale | Example Agent Role | Relevant Granularity |
|-------|-------|--------------------|---------------------|
| 0 | Token | Linter, formatter | Individual tokens, syntax |
| 1 | Statement | Code writer, test generator | Single statements, tool calls |
| 2 | Function | Subagent, debugger | Function-level changes, subtask results |
| 3 | Module | Orchestrator | Cross-function changes, subsystem state |
| 4 | System | Meta-orchestrator, planner | Architecture, goals, constraints |

**Scale projection operators:** For each scale s, define a projection operator P_s that maps full event e to its representation at scale s:

```
P_s : Events → Compressed_Events_at_scale_s
```

**Composability constraint (the key mathematical requirement):**
```
P_{s1} ∘ P_{s2} = P_{max(s1, s2)}
```
Projecting first to scale s2, then to scale s1, gives the same result as projecting directly to the coarser scale. This is the analog of the RG group law. It means the projection hierarchy is consistent — there are no "artifacts" from intermediate projections.

**Fixed points:** Some information must be preserved at all scales — these are the scale-invariant fixed points:
- Global task goal
- Hard constraints and invariants
- Final outputs and commitments

An event containing fixed-point information is never compressed by scale projection; it passes through P_s unchanged for all s.

**The log(N) factor:** In a balanced tree hierarchy with branching factor b and N agents, there are log_b(N) levels. As information moves up one level, its scale increases by 1 and its representation size decreases by a factor of r (the compression ratio per scale step). The total context received by a level-k orchestrator from its subtree is:

```
|C_orch(k)| = b^(log_b(N) - k) · H · k_msg · r^k
```

For appropriate r (e.g., r = 1/b), this stabilizes at O(H) regardless of N, giving the O(H·log(N)) behavior across the full hierarchy.

### 3.4 Stage 3: Predictive Coding Transmission Filter (TRANSMIT)

**World model:** Each agent aᵢ maintains a lightweight generative model M_i of the shared workspace state. At each step, M_i predicts the next event's embedding.

**Surprise signal:**
```
δᵢ(e) = KL( M_i's prediction distribution || actual event e's embedding )
```

**Transmission rule:**
```
Deliver P_{sᵢ}(e) to aᵢ  iff  δᵢ(e) ≥ τᵢ
```

If δ is small, aᵢ's model was correct — no new information. If δ is large, aᵢ's model was wrong — deliver the correction.

**Connection to ant pheromone decay:** Events that are consistently predictable (high-frequency routine operations) accumulate low δ over time and stop being transmitted. Events that are consistently surprising (errors, exceptions, novel decisions) always transmit. This is information-theoretically equivalent to pheromone decay — salience evaporates with redundancy.

**Connection to Information Bottleneck:** The surprise signal δᵢ(e) is exactly the information in e that is *not already* in aᵢ's current context T_i. Transmitting only high-δ events is equivalent to transmitting only the component of e that increases I(T_i; X) — i.e., only what genuinely expands the agent's knowledge.

**World model bootstrapping:** The world model M_i cannot be trained before deployment (chicken-and-egg). Proposed curriculum:
1. Phase 0: Set τᵢ = 0 (everything transmits). Collect agent interaction data.
2. Phase 1: Train M_i on collected data. Set τᵢ = 10th percentile of δ distribution.
3. Phase 2: Iteratively increase τᵢ as M_i improves.

### 3.5 Unified Formalization

Let the **minimum sufficient context** for agent aᵢ at time t be:
```
T_i*(t) = argmin_{T ⊆ X(t)} |T|  subject to  I(T; Yᵢ | zᵢ) = I(X(t); Yᵢ | zᵢ)
```
This is the minimum context that preserves all information relevant to aᵢ's action.

CASR approximates T_i* through:
1. **Causal abstraction** (Stage 1): removes events outside FP(aᵢ, zᵢ)
2. **Scale projection** (Stage 2): coarse-grains remaining events to aᵢ's resolution
3. **Surprise filter** (Stage 3): removes events already captured in M_i's current state

The combined approximation is:
```
T_i^CASR(t) = { P_{sᵢ}(e) : e ∈ X(t), e ∈ FP(aᵢ, zᵢ), δᵢ(e) ≥ τᵢ }
```

**The distortion:** D(aᵢ, T_i^CASR) = P(aᵢ takes suboptimal action | T_i^CASR) - P(aᵢ takes suboptimal action | X(t))

Rate-Distortion theory gives a lower bound on |T_i^CASR| for any given distortion tolerance D_max:
```
|T_i^CASR| ≥ R(D_max)  (the rate-distortion function)
```

This provides a theoretical floor — CASR cannot compress below R(D_max) without exceeding acceptable task error. The framework is designed to approach but not violate this bound.

### 3.6 Complexity Analysis

**Claim:** For a balanced tree of depth L with branching factor b (N = b^L agents total), CASR yields per-agent effective context of O(H · L) = O(H · log_b(N)).

**Argument (informal):**
- At depth d from leaves, agent scale is s_d = d
- Information arriving at depth d has been projected through d scale steps
- Each scale step reduces event representation size by factor r
- Total context at depth d: Σ_t Σ_{descendants} P_{s_d}(e) filtered through FP + surprise
- With appropriate r, the geometric series of compressed messages from all descendants converges to O(H) per level
- Summed across L levels of ancestry: O(H · L) = O(H · log(N))

**What must hold for this to be true:**
1. Scale projections are genuinely lossy (r < 1 per level)
2. The fixed-point information (task goals, constraints) does not grow with N
3. The Bloom filter false positive rate is bounded (otherwise FP(aᵢ) grows to include everything)
4. The world model M_i remains calibrated as N grows (surprise filter doesn't degrade)

Conditions 1-3 are by construction. Condition 4 is empirically testable and represents the main risk to the complexity claim.

---

## Section 4: What CASR Is Not

To be precise about scope:

- **Not a compression algorithm**: CASR does not compress individual messages. It routes them selectively. Compression is orthogonal and can be applied on top.
- **Not a memory system**: CASR does not store or retrieve historical context. It determines what to transmit in real time. MemGPT-style memory systems are complementary.
- **Not an attention mechanism**: CASR operates at the system level (between agents), not at the model level (inside an agent's transformer). Sparse attention is orthogonal.
- **Not applicable to single-agent systems**: CASR only provides gains when multiple agents with different roles and scales communicate.

---

## Appendix A: Mathematical Background

### A.1 Information Bottleneck

The IB method (Tishby, Pereira, Bialek 1999) finds a compressed representation T of input X that preserves information about output Y:

```
min_{p(t|x)} I(T; X) - β · I(T; Y)
```

- I(T; X) is the mutual information between T and X (measures compression)
- I(T; Y) is the mutual information between T and Y (measures task relevance)
- β is a Lagrange multiplier controlling the trade-off

The solution defines an optimal trade-off curve: for each compression level I(T; X), the maximum achievable I(T; Y). This is the **information plane** — the theoretical bound on how much task-relevant information can be preserved at a given compression ratio.

**Application to CASR:** Each stage of CASR is an operation in the information plane. Stage 1 (causal selection) moves T toward the boundary of the plane by removing variables with I(T; Y_i) = 0. Stage 2 (scale projection) moves along the boundary by trading irrelevant detail for smaller representation. Stage 3 (surprise filter) removes variables already captured in M_i.

### A.2 Rate-Distortion Theory

For a source X and distortion measure d(x, x̂), the Rate-Distortion function R(D) is the minimum average number of bits needed to describe X such that the expected distortion E[d(X, X̂)] ≤ D:

```
R(D) = min_{p(x̂|x) : E[d(X,X̂)] ≤ D} I(X; X̂)
```

**Application to CASR:** Define distortion as d(X, T_i^CASR) = (probability of suboptimal agent action given T_i^CASR) - (probability given full context X). R(D) then gives the minimum context size for acceptable task performance. CASR cannot achieve compression beyond R(D) without degrading task quality.

### A.3 Do-Calculus for Causal Relevance

For a causal DAG over variables (X, Z, Y), variable x_j is causally relevant to Y given Z if and only if x_j is not d-separated from Y given Z in the graph with all arrows into x_j removed (the "do" operation).

**Practical approximation for CASR:** Full do-calculus over large causal graphs is computationally expensive. For practical deployment, the causal footprint is estimated by:
1. Empirical perturbation: randomly mask event type e in a simulation environment, measure Δ in agent action distribution
2. Graph-based d-separation: build the team communication graph, compute d-separation queries offline
3. Conservative approximation: include event types whose d-separation is uncertain (better false positive than false negative)

### A.4 O(H · log(N)) Derivation Sketch

Consider a complete binary tree (b=2) of depth L, so N = 2^L agents. Workers are at depth 0 (scale s=0). Each level adds 1 to scale (s increases by 1 going up).

Define r = 0.5 (each scale projection halves the token count of an event). At depth d, an orchestrator receives events from b^d = 2^d descendants, each compressed by factor r^d = 2^(-d):

```
|C_{orch at depth d}(H)| = 2^d descendants × H events × k tokens × 2^(-d) compression
                         = H · k (constant, regardless of d or N)
```

Summing across all of aᵢ's ancestors (at most L of them in the chain from root to leaf):
```
Total context for any agent = O(H · L) = O(H · log₂(N))
```

This argument holds under the assumption that the compression factor r = 1/b exactly. In practice, r depends on the scale projection implementation and must be empirically validated. The key insight is that the tree structure causes the two exponentials (descendants growing as b^d, compression shrinking as r^d) to cancel — but only if r ≤ 1/b.

---

## References

- Liu et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts." *TACL*. arXiv:2307.03172
- Tishby, Pereira, Bialek (1999). "The Information Bottleneck Method." arXiv:physics/0004057
- Geiger et al. (2023). "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations." arXiv:2301.04709
- Mehta & Schwab (2014). "An Exact Mapping between the Variational Renormalization Group and Deep Learning." arXiv:1410.3831
- Friston (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*
- Jiang et al. (2023). "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models." arXiv:2310.05736
- Peng et al. (2023). "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752
- Wu et al. (2023). "AutoCompressors: Adapted LLMs for Efficient Context Compression." EMNLP 2023
- Shannon (1959). "Coding Theorems for a Discrete Source with a Fidelity Criterion." IRE Convention Record
