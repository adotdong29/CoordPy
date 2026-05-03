# Context-Zero: Open Research Questions

These are the questions the CASR framework does not yet answer. They define the boundary between what this project claims and what it defers. Each is stated as a precise, researchable question — not a vague direction.

---

## Resolution status (post-build)

| OQ | Resolution | Module / evidence |
|---|---|---|
| OQ1 (fixed point) | **addressed** via contraction analysis + Anderson DEQ | `core/contraction.py`, `core/deq_numpy.py` |
| OQ2 (scale inference) | **operationalised** via first-order Reptile | `core/meta_learn.py` |
| OQ3 (world-model bootstrap) | **dissolved** — CTW gives universal prediction without training | `core/ctw_predictor.py` |
| OQ4 (DAG topology) | still open; partial via consistent hashing + ITC | `core/routing_hash.py`, `core/itc.py` |
| OQ5 (adversarial robustness) | **stack shipped** — cuckoo filter + PeerReview + VRF + CBF + DP | `core/cuckoo_filter.py`, `core/peer_review.py`, `core/vrf_committee.py`, `core/cbf.py`, `core/dp.py` |
| OQ6 (optimal β) | **resolved** as PAC-Bayes Lagrange multiplier | `core/pac_bayes.py` |
| OQ7 (cross-task transfer) | still open; Bayesian coresets + BNP are the building blocks | `core/coreset.py`, `core/bnp.py` |

See `WAVES.md` for the mechanism-level implementation trail, and the Wave-1..5 test files in `vision_mvp/tests/` for numerical verification.

---

## 1. The Fixed-Point Problem

**Question:** Does the minimum sufficient context iteration converge, and if so, under what conditions?

**The problem:**

The minimum sufficient context T_i* for agent aᵢ is defined as:
```
T_i* = argmin |T| such that I(T; Y_i | z_i) = I(X; Y_i | z_i)
```

But Y_i (the agent's action distribution) *depends on T_i*. This creates a fixed-point equation: the "right" context for an agent depends on what the agent would do with that context, which depends on the context. Formally:

```
T_i* = f(T_i*)  where f is the compression operator
```

**Why it matters:** If this fixed-point does not exist or does not converge to a unique solution, the CASR framework's theoretical foundation is unstable. Two different starting points could converge to different T_i*, meaning the "minimum sufficient context" is not well-defined.

**Possible resolution approaches:**
- Show that f is a contraction mapping (Banach fixed-point theorem guarantees unique convergence)
- Show that the IB-defined T_i* is always a fixed point of f, even if other fixed points exist
- Prove convergence holds for the specific class of agent action distributions that arise in practice (e.g., distributions over discrete action spaces)

**Open mathematical status:** Unknown. This is a genuine research question requiring formal proof.

---

## 2. The Scale Inference Problem

**Question:** Can agent operating scales be inferred from task descriptions, or must they always be assigned manually?

**The problem:**

CASR assigns scales at instantiation. This requires a human to decide that a "code writer" operates at scale=1 (Statement) while an "orchestrator" operates at scale=3 (Module). But:

1. Task-appropriate scales may vary within a role. A code writer implementing a small hotfix may need scale=1; one implementing a large refactor may need scale=3 to see the module-level structure.
2. In general-purpose agent teams (not just software development), there is no predefined scale taxonomy.
3. Misassigned scales are catastrophic: an orchestrator at scale=1 floods with irrelevant detail; a worker at scale=4 receives only abstractions and can't take concrete actions.

**Research direction:**

Can we train a model that, given (task_description, role_description), outputs an appropriate scale assignment? This is essentially a meta-learning problem: learn the *right level of abstraction* for a given task-role combination.

Potential training signal: in hindsight, a scale assignment is "correct" if it minimizes context size while keeping task completion rate above threshold. This can be measured from experiment logs.

**Connection to cognitive science:** This mirrors the human ability to know what level of detail to discuss in a given context — a skill that appears to be learned from experience rather than hard-coded.

---

## 3. The World Model Bootstrapping Problem

**Question:** How do you train the Stage 3 world model when you need the world model to generate the training data?

**The problem:**

The surprise filter (Stage 3) requires a per-agent generative model M_i trained on event sequences from the agent team. But:

1. To generate training data, you need to run the agent team.
2. Before M_i is trained, you can't use Stage 3 (τᵢ = 0, everything delivers).
3. Running with τᵢ = 0 generates different (noisier) event sequences than running with trained M_i.
4. The world model trained on τᵢ = 0 data may not generalize to the τᵢ > 0 regime.

**Proposed curriculum:**
- Phase 0: τᵢ = 0. Collect N_0 tasks. Train M_i^(0).
- Phase 1: τᵢ = τ_low. Run N_1 tasks. Augment training set. Retrain M_i^(1).
- Iterate: gradually increase τᵢ as M_i improves.

**Open question:** Does this curriculum converge? Or does each increase in τᵢ cause distribution shift that destabilizes M_i?

**Analogous problem:** This is structurally similar to self-play training in reinforcement learning (RLHF, AlphaZero), where the training distribution is generated by the model being trained. Lessons from that literature likely apply.

---

## 4. DAG Topology Extension

**Question:** CASR is defined for tree-structured agent hierarchies. How does it extend to DAG topologies (peer-to-peer communication, shared resources, multiple orchestrators)?

**The problem:**

The O(H·log(N)) complexity argument relies on the tree structure: at each level, the branching factor b and compression factor r cancel. In a DAG:

- A worker agent may report to two orchestrators simultaneously. Which scale does its output get projected to?
- Two workers may communicate peer-to-peer. There is no parent-child relationship to define scale direction.
- Shared resources (databases, code repositories) are accessed by multiple agents at different scales. What is the "scale" of a shared resource?

**The composability constraint problem:**

The RG projection's composability requirement is:
```
P_{s1}(P_{s2}(e)) = P_{max(s1,s2)}(e)
```

This is well-defined when s1 and s2 are in a total order (the tree hierarchy). In a DAG, an event may need to be delivered to multiple agents at different scales via different paths. The composability of these multi-path projections is not guaranteed.

**Potential approach:** Define scale not as an absolute level but as a *relative* measure between sender and receiver. The projection applied to event e from agent aⱼ (scale sⱼ) to agent aᵢ (scale sᵢ) is P_{|sᵢ - sⱼ|} in the direction of increasing abstraction. This decouples scale from tree depth.

**Mathematical status:** The composability condition for this relative-scale formulation needs to be verified. It is an open algebraic question.

---

## 5. Adversarial Robustness

**Question:** Is CASR robust to malicious agents that craft messages designed to exploit the routing system?

**The attack surface:**

A malicious agent in the team could:
1. Emit events with types that pass the Bloom filter for high-value target agents (type spoofing)
2. Craft event bodies that appear to have low surprise (δ < τᵢ) to the world model, suppressing legitimate surprise detection
3. Inflate the event log to trigger the periodic full-state synchronization (forcing context flooding)
4. Emit fixed-point events (with `is_fixed_point=True`) to guarantee delivery to all agents

**Why it matters for real-world deployment:** In enterprise multi-agent systems, individual agents may be instantiated from different providers, fine-tuned on different data, or even intentionally adversarial (testing, red-teaming). The routing system must be robust to a single compromised agent.

**Known from CS security:** Bloom filters are not adversarially robust — a sufficiently powerful adversary can enumerate elements that hash to any target bucket. This is a known weakness of Bloom filters in adversarial settings (used in Bitcoin protocol attacks).

**Potential mitigations:**
- Authenticate event type declarations (cryptographic signing by a trusted authority)
- Use adversarially-robust approximate membership structures instead of classical Bloom filters
- Rate-limit fixed-point event emission per agent
- Anomaly detection on per-agent event emission patterns

**Status:** This is an important engineering concern but a secondary research question. The framework should acknowledge the attack surface; mitigation design is deferred to a security-focused phase.

---

## 6. The Optimal β Problem

**Question:** What is the right trade-off parameter β between compression and task performance for different agent roles and task phases?

**Context:**

The Information Bottleneck formulation is:
```
min I(T; X) - β · I(T; Y)
```

β controls the compression-accuracy trade-off. In CASR, β manifests through the distortion_budget parameter and the surprise threshold τ. Different agents and task phases need different β values:

- During task planning: low β (high information, need rich context)
- During execution of well-understood subtasks: high β (compressed context is sufficient)
- During debugging and error recovery: low β (need detailed context to diagnose)

**The question:** Can β be set automatically, or does it require manual tuning per phase?

**Connection to Phase 3:** The scale inference problem (Open Question 2) is related — if you can infer the right scale, you may also be able to infer the right β. Both are about learning the right level of abstraction for a given context.

---

## 7. Cross-Task Context Transfer

**Question:** Can CASR-compressed context from one task be transferred as initialization for a related task, reducing cold-start overhead?

**The motivation:**

In long-running agent teams that execute many related tasks (e.g., maintaining a large codebase), most context from task T-1 is irrelevant to task T, but some is highly relevant (architectural decisions, established conventions, known constraints). Currently, each task starts with an empty context — cold start.

**The CASR angle:** The causal footprint F_{task T} may overlap significantly with F_{task T-1} for the same agent role. If we can identify this overlap, we can carry over the relevant T-1 context without transmitting all of it.

**Connection to continual learning:** This is related to the continual learning problem — how to transfer knowledge from old tasks without interference. The IB framework provides a natural formulation: find the component of T-1's compressed context that has nonzero mutual information with T's outcomes.

---

## Prioritization

Research priority order for the project:

1. **Fixed-point convergence** (Question 1): Foundational. If it doesn't converge, the framework needs redesign. Investigate theoretically in parallel with MVP.

2. **World model bootstrapping** (Question 3): Required for Phase 2 (Stage 3 implementation). Start curriculum experiments as soon as Phase 1 MVP generates data.

3. **Optimal β** (Question 6): Practical for Phase 2 tuning. Can be addressed empirically before theoretical resolution.

4. **Scale inference** (Question 2): Required for Phase 3 generalization. Begin theoretical framing while MVP runs.

5. **DAG topology** (Question 4): Required for real-world deployment beyond simple hierarchies. Address in Phase 3.

6. **Cross-task transfer** (Question 7): Valuable but not blocking. Deferred to Phase 4.

7. **Adversarial robustness** (Question 5): Important for production deployment but not core to the thesis. Deferred to Phase 4 or a separate security-focused track.
