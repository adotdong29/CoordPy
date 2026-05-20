"""W86 #37 — BudgetEnforcerV1 inserted into the W83 composed pipeline.

Tests the wrapper module that gates W83 composed_recovery
scenarios with BudgetEnforcerV1. All three load-bearing bools
of the #37 DoD bullet 2 ("BudgetEnforcerV1 is inserted into the
W83 composed pipeline") are CI-asserted.
"""
from __future__ import annotations

import pytest


def test_w86_budget_enforced_composed_recovery_imports():
    from coordpy.budget_enforced_composed_recovery_v1 import (
        BudgetEnforcedRecoveryBenchReportV1,
        ScenarioBudgetOutcomeV1,
        W86_BUDGET_ENFORCED_RECOVERY_V1_SCHEMA_VERSION,
        run_budget_enforced_composed_recovery_v1,
        run_budget_integration_head_to_head_v1,
    )
    assert (
        W86_BUDGET_ENFORCED_RECOVERY_V1_SCHEMA_VERSION
        == "coordpy.budget_enforced_composed_recovery_v1.v1")


def test_w86_budget_integration_zero_commits_under_tiny_budget():
    """Load-bearing #37 bullet 3: on an over-budget regime
    (tiny budget), the W83 composed pipeline is GATED — zero
    commits, every scenario refused, every refusal carries a
    BudgetBreachAuditV1."""
    from coordpy.budget_enforced_composed_recovery_v1 import (
        run_budget_integration_head_to_head_v1,
    )
    r = run_budget_integration_head_to_head_v1(n_regimes=4)
    t = r["tiny_budget"]["report"]
    assert int(t["n_scenarios"]) == 4
    assert int(t["n_committed"]) == 0
    assert int(t["n_refused"]) == 4
    assert t["zero_commits_when_over_budget"] is True
    assert (
        t["every_refusal_audit_carries_breach_audit"]
        is True)


def test_w86_budget_integration_huge_budget_matches_no_enforcer():
    """Load-bearing #37 bullet 4: on an under-budget regime
    (huge budget), the pipeline commits the SAME number of
    times as it would without the enforcer — no behavior
    change inside the budget envelope."""
    from coordpy.budget_enforced_composed_recovery_v1 import (
        run_budget_integration_head_to_head_v1,
    )
    r = run_budget_integration_head_to_head_v1(n_regimes=4)
    h = r["huge_budget"]["report"]
    assert int(h["n_scenarios"]) == 4
    assert int(h["n_committed"]) == int(
        h["n_committed_no_enforcer"])
    assert h["under_budget_matches_no_enforcer"] is True


def test_w86_budget_integration_breach_audit_chain_re_derives():
    """Anti-cheat: the breach-audit Merkle root must re-derive
    from the per-scenario breach audits when run with the same
    config."""
    from coordpy.budget_enforced_composed_recovery_v1 import (
        run_budget_integration_head_to_head_v1,
    )
    r1 = run_budget_integration_head_to_head_v1(n_regimes=3)
    r2 = run_budget_integration_head_to_head_v1(n_regimes=3)
    # Same seed_root → same scenarios → same predicted action
    # CIDs → same breach audits → same breach Merkle root.
    assert (
        r1["tiny_budget"]["report"]["breach_audit_merkle_root"]
        == r2["tiny_budget"]["report"][
            "breach_audit_merkle_root"])
    assert (
        r1["tiny_budget"]["report_cid"]
        == r2["tiny_budget"]["report_cid"])


def test_w86_budget_integration_spec_and_cost_model_cids():
    """Both the budget spec and the cost model must be content-
    addressed and recorded in the bench report."""
    from coordpy.budget_enforced_composed_recovery_v1 import (
        run_budget_integration_head_to_head_v1,
    )
    r = run_budget_integration_head_to_head_v1(n_regimes=3)
    t = r["tiny_budget"]["report"]
    h = r["huge_budget"]["report"]
    assert len(str(t["budget_spec_cid"])) == 64
    assert len(str(t["cost_model_cid"])) == 64
    # Cost model CID is the same across both budgets (only
    # the spec differs).
    assert t["cost_model_cid"] == h["cost_model_cid"]
    assert t["budget_spec_cid"] != h["budget_spec_cid"]


def test_w86_budget_integration_tool_calls_count_toward_budget():
    """Anti-cheat: tool calls MUST be counted toward
    max_tool_calls; you cannot bypass the budget by labeling
    the action as a tool call."""
    from coordpy.budget_enforcement_v1 import (
        BudgetEnforcerV1, CandidateActionV1,
        RunBudgetSpecV1, default_cost_model_v1,
        W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
    )
    spec = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=10.0,
        max_per_step_latency_ms=10_000.0,
        max_total_tokens=10_000,
        max_tool_calls=1,  # only ONE tool call permitted
        max_recompute_flops=1e10)
    enf = BudgetEnforcerV1(
        spec=spec, cost_model=default_cost_model_v1())
    tool = CandidateActionV1(
        action_name="tool_call",
        predicted_usd=0.001,
        predicted_latency_ms=10.0,
        predicted_tokens=10,
        is_tool_call=True,
        predicted_flops=1.0)
    v1 = enf.check(tool)
    assert v1.permitted is True
    enf.commit(tool)
    v2 = enf.check(tool)
    assert v2.permitted is False
    assert v2.breach_audit is not None
    assert v2.breach_audit.breached_axis == "max_tool_calls"
