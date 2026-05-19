# W84 / P1 #37 — Hard Cost / Latency Budget Enforcement V1

## Summary

Ships an end-to-end hard-budget enforcement subsystem that takes
a `RunBudgetSpecV1` with `max_cost_usd` / `max_per_step_latency_ms`
/ `max_total_tokens` / `max_tool_calls` / `max_recompute_flops`
and *provably* refuses to violate them. Every refused action
emits a content-addressed `BudgetBreachAuditV1` capsule carrying
pre/post budgets, the would-be action, and the breach axes.

## Definition-of-Done bars

| Bar | Status |
| --- | ------ |
| `RunBudgetSpecV1` exists and is content-addressed | ✅ |
| `BudgetEnforcerV1` integration point exists | ✅ (`BudgetEnforcerV1.propose` is the integration hook the W83 composed pipeline calls before every commit) |
| On a deliberately-over-budget regime, the pipeline commits 0 times and emits N abstains, each with a `BudgetBreachAuditV1` capsule | ✅ (`n_commits = 0`, `n_abstain_on_breach = 10`, `all_proposals_audit_logged = True`) |
| On a deliberately-under-budget regime, the pipeline commits exactly as it would without the enforcer | ✅ (`under_budget_commits_exactly_match = True`) |
| The cost model is content-addressed; identical configs produce identical cost CIDs | ✅ |
| `RESULTS__BUDGET_ENFORCEMENT.md` captures the contract + measured behaviour | ✅ (this file) |

## Measured numbers

Over-budget stress regime (`max_total_cost_usd = 0.0005`,
10 proposals of `runtime_recompute` with 200 + 100 tokens
each):

| Claim | Value |
| ----- | ----- |
| `n_proposals` | 10 |
| `n_commits` | **0** |
| `n_abstain_on_breach` | **10** |
| `all_proposals_audit_logged` | True |
| `breach_axes_seen` | `["cost_usd"]` |

Under-budget regime (`max_total_cost_usd = 10.0`):

| Claim | Value |
| ----- | ----- |
| `under_budget_commits_exactly_match` | True |

## Cost model V1

Static table indexed by `(action, model_cid)`:

```python
prompt_cost_per_kt = 0.003   # $/1K prompt tokens
output_cost_per_kt = 0.006   # $/1K output tokens
tool_cost_per_call = 0.001   # $/tool call
flops_cost_per_megaflop = 1e-6
action_multipliers = (
    ("replay", 0.10),
    ("runtime_recompute", 1.00),
    ("transcript_recompute", 0.40),
    ("promote_to_richer_substrate", 4.00),
    ("abstain", 0.00),
)
```

Cost estimate is monotone in tokens + tool calls
(verified by `test_w84_cost_model_content_addressed_and_monotone`).
`promote_to_richer_substrate > replay` (different actions have
different costs); `abstain == 0` (verified by
`test_w84_cost_model_action_multipliers_distinct`).

## Anti-cheat compliance

* **Refusal is not silent.** Every breach proposal that fires
  the enforcer emits a `BudgetBreachAuditV1` capsule
  (`test_w84_breach_audit_not_silently_dropped`).
* **Cost model is not "so loose nothing is ever over-budget".**
  The cost model produces monotone-in-tokens estimates that
  match the deployment's stated tolerance; over-budget regimes
  fire as expected (test bench above).
* **Enforcer is not silently-disabled in production.** The
  `enforcer_disabled = True` flag IS visible in
  `RunBudgetSpecV1.cid()` — so any audit of the run carries
  the disabled flag along with everything else
  (`test_w84_enforcer_disabled_flag_recorded_in_audit`).
* **Abstain is not failure.** The contract treats abstain-on-
  breach as the *correct* behaviour under hard budgets.
* **Latency budgets are not skipped.** The per-step latency
  budget is enforced and tested
  (`test_w84_enforcer_blocks_per_step_latency`).
* **Tools count toward the budget.** Tool call count is
  enforced (`test_w84_enforcer_blocks_tool_call_overshoot`)
  AND tool calls contribute to the cost estimate (monotone
  test).

## Honest scope (V1)

* `W84-L-BUDGET-V1-RUN-LEVEL-CAP` — V1 enforces per-run budgets;
  per-tenant + per-agent budgets are V2.
* `W84-L-BUDGET-V1-STATIC-USD-CAP` — V1 cost model is a static
  table; dynamic pricing is V2.
* `W84-L-BUDGET-V1-HARD-ABSTAIN-CAP` — V1 enforces by hard
  abstain; graceful degradation is V2.
* `W84-L-BUDGET-V1-NO-MEMORY-BUDGET-CAP` — memory budget is V2.

## Reproduction

```python
from coordpy.budget_enforcement_v1 import (
    run_budget_stress_bench_v1,
)
rep = run_budget_stress_bench_v1()
print(rep.to_dict())
```

Tests: `tests/test_w84_budget_enforcement.py` (12 tests, all
passing).
