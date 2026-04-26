# Context-Zero: MVP Specification

## Goal

Validate the **core thesis** with the minimal possible implementation: that causal-abstraction-based routing plus scale projection reduces context size without measurable task accuracy degradation.

The MVP does **not** need to prove the full CASR framework. It proves one thing: selective routing based on agent role + scale projection works better than naive full-context sharing.

**Success criterion:** <5% task completion drop on SWE-bench with measurable token reduction (CER ≥ 3) on a 3-agent team compared to naive full-context baseline.

---

## What the MVP Does

Implements **Stage 1 (SELECT) + Stage 2 (PROJECT)** of CASR. Stage 3 (TRANSMIT/surprise filter) is disabled — every event that passes the Bloom filter is delivered.

**Specifically:**
1. Three-agent team: 1 orchestrator + 2 workers
2. Hand-specified causal footprints per role (no learned Bloom filter)
3. LLM-based scale projections (orchestrator at Module scale=3, workers at Statement scale=1)
4. Event sourcing: all actions logged as immutable events
5. SWE-bench evaluation

---

## Team Configuration

```
Orchestrator (scale=3, Module level)
├── Worker A: Implementation (scale=1, Statement level)
└── Worker B: Testing (scale=1, Statement level)
```

**Orchestrator role:** Plans task decomposition, assigns subtasks, monitors progress, integrates results. Needs to know: task goals, subtask completions, test results, errors. Does NOT need: individual code edits, intermediate variable states, linter output.

**Worker A (Implementation) role:** Writes code changes. Needs to know: task specification, file contents, function signatures, existing tests. Does NOT need: orchestrator's planning reasoning, other workers' test output details.

**Worker B (Testing) role:** Writes and runs tests. Needs to know: implementation results, test framework, error messages. Does NOT need: orchestrator's planning reasoning, Worker A's intermediate edits.

---

## Hand-Specified Causal Footprints

Instead of computing footprints via do-calculus, we hand-specify them for the MVP. This is a conscious simplification — the goal is to validate that *selective routing with the right footprint* works, before investing in automated footprint computation.

```
Orchestrator causal footprint (receives):
  TASK_GOAL_UPDATE       ✓ (fixed point, always)
  HARD_CONSTRAINT        ✓ (fixed point, always)
  ERROR_UNHANDLED        ✓ (fixed point, always)
  TASK_COMPLETE          ✓ (fixed point, always)
  FUNCTION_COMPLETE      ✓ (subtask completions)
  MODULE_COMPLETE        ✓
  TEST_RESULT            ✓ (pass/fail summary only, at scale=3)
  AGENT_SPAWN            ✓
  AGENT_TERMINATE        ✓
  FILE_EDIT              ✗ (too granular for orchestrator)
  TOOL_CALL              ✗ (internal worker operations)
  TOOL_RESULT            ✗ (except errors, which become ERROR_UNHANDLED)
  MESSAGE_AGENT          ✓ (direct messages always delivered)

Worker A (Implementation) causal footprint:
  TASK_GOAL_UPDATE       ✓
  HARD_CONSTRAINT        ✓
  ERROR_UNHANDLED        ✓
  TASK_COMPLETE          ✓
  FILE_EDIT              ✓ (needs to know file state)
  TOOL_CALL              ✓ (own calls and relevant peer calls)
  TOOL_RESULT            ✓ (own results only)
  TEST_RESULT            ✓ (summary: pass/fail + error message)
  FUNCTION_COMPLETE      ✓
  MODULE_COMPLETE        ✓
  MESSAGE_AGENT          ✓

Worker B (Testing) causal footprint:
  TASK_GOAL_UPDATE       ✓
  HARD_CONSTRAINT        ✓
  ERROR_UNHANDLED        ✓
  TASK_COMPLETE          ✓
  FILE_EDIT              ✓ (needs to see code to write tests)
  FUNCTION_COMPLETE      ✓ (triggers test writing)
  TOOL_CALL              ✗ (doesn't need implementation tool calls)
  TOOL_RESULT            ✓ (test execution results)
  TEST_RESULT            ✓
  MESSAGE_AGENT          ✓
```

---

## Scale Projections for MVP

For the MVP, scale projections are implemented as structured LLM calls:

**scale=1 event → scale=3 projection (for orchestrator):**

```
Input: Full event body at scale=1 (e.g., a file edit with 200 lines of diff)
Prompt: "Summarize the following code change at the module level.
         Include: what function/module changed, what the change accomplishes,
         whether it introduces any errors or breaks any interfaces.
         Maximum 3 sentences."
Output: Module-level summary (~50 tokens vs. ~400 tokens for full diff)
```

**scale=1 TEST_RESULT → scale=3 projection:**

```
Input: Full test output (may include 500 lines of stack traces)
Prompt: "Summarize the test result.
         Include: pass/fail, test name, if failed: one-line root cause.
         Maximum 2 sentences."
Output: ~30 tokens vs. ~500 tokens
```

**Fixed-point events** are never projected — delivered verbatim to all agents.

---

## Event Bus (Simplified)

For the MVP, the event bus is a simple Python class:

```python
class CASRBus:
    def __init__(self):
        self.event_log = []  # append-only
        self.subscribers = {}  # agent_id → AgentConfig
    
    def publish(self, event: Event):
        self.event_log.append(event)
        for agent_id, config in self.subscribers.items():
            if agent_id == event.sender_id:
                continue
            # Stage 1: Bloom filter (hand-specified for MVP)
            if not config.causal_footprint.might_contain(event.event_type):
                continue  # definitely not relevant
            # Stage 2: Scale projection
            projected = project_to_scale(event, config.scale)
            # Stage 3: Disabled (τ=0)
            # Deliver
            config.context_queue.append(projected)
    
    def get_context(self, agent_id: str) -> List[Event]:
        return self.subscribers[agent_id].context_queue
```

**No Stage 3 in MVP.** The world model is unimplemented; τᵢ = 0 means all post-Stage-2 events are delivered.

---

## SWE-bench Setup

**Task selection:** Start with SWE-bench Lite (300 instances). Use the first 100 instances for development, hold out the second 100 for evaluation.

**Baseline to beat:** Single-agent baseline using Claude Sonnet on SWE-bench (current ~50-55% resolve rate for top systems). The goal is NOT to improve resolve rate — it's to maintain it while reducing tokens.

**Team assignment:**
- Orchestrator decomposes the issue into implementation + testing subtasks
- Worker A implements the code change
- Worker B writes and runs tests
- Orchestrator integrates results and submits the patch

**Instrumentation:**
- Log total tokens in each agent's context at each step
- Log which events were filtered at Stage 1 and Stage 2
- Log final patch and test outcome
- Compute CER per task

---

## Instrumentation and Logging

Every run must capture:

```python
@dataclass
class RunLog:
    task_id: str
    team_config: dict
    
    # Per agent, per step
    context_sizes: Dict[str, List[int]]  # agent_id → [tokens at each step]
    events_published: int
    events_delivered_per_agent: Dict[str, int]
    events_filtered_stage1: int  # Bloom filter drops
    events_compressed_stage2: Dict[str, int]  # tokens saved by projection
    
    # Outcome
    task_completed: bool
    patch_correct: bool  # did it pass the tests?
    total_tokens_naive: int  # simulated: what would full-context cost?
    total_tokens_casr: int   # actual
    cer: float  # total_tokens_naive / total_tokens_casr
```

The `total_tokens_naive` is computed by simulating what the baseline would have consumed — run the same events through the bus without any filtering.

---

## What the MVP Does NOT Cover

Explicitly deferred to later phases:

- Stage 3 (world model + surprise filter)
- Learned Bloom filters (automated causal footprint estimation)
- Dynamic scale assignment
- DAG topologies (peer-to-peer between workers)
- Teams larger than 3 agents
- World model training curriculum
- Adversarial robustness
- Non-software-development domains

The MVP is intentionally narrow. If it succeeds, Phase 2 expands scope. If it fails, the failure tells us whether the thesis is wrong or just the MVP design.

---

## MVP Failure Modes and Interpretations

| Outcome | Interpretation | Next Step |
|---------|---------------|-----------|
| CER ≥ 3, completion drop < 5% | MVP success. Thesis validated at MVP level. | Proceed to Phase 2 (learned footprints, Stage 3) |
| CER ≥ 3, completion drop > 5% | Scale projection is too lossy OR hand-specified footprint misses critical events | Audit which events were filtered when the task failed. Adjust footprint or projection. |
| CER < 3, completion as expected | Filtering is too conservative (footprint includes too much) | Tighten footprint definitions. Measure which event types are most commonly delivered. |
| CER < 3, completion drop > 5% | Fundamental problem with the approach | Run the falsifiability analysis (Section in EVALUATION.md). Measure true causal footprint size empirically. |
| CER as expected, but orchestrator makes worse decisions | Scale projection loses critical information despite correct footprint | Review projection prompts. Add verbatim preservation for specific event patterns. |

---

## Timeline

**Week 1-2:** Framework and event bus implementation. Test with a single synthetic task (not SWE-bench).

**Week 3-4:** SWE-bench integration. Run development set (first 100 tasks) with both CASR and naive baseline.

**Week 5-6:** Instrumentation, metric computation, ablation. Identify failure cases.

**Week 7-8:** Tuning and evaluation set run (second 100 tasks). Write up results.

**Deliverable:** A report with:
1. CER histogram across 100 evaluation tasks
2. Task completion rate: CASR vs. naive baseline
3. Per-stage attribution (CER_Stage1, CER_Stage2)
4. Failure case analysis (which event types were incorrectly filtered when tasks failed)
5. Recommendation for Phase 2 priorities
