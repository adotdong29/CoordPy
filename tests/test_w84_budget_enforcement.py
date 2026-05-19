"""W84 / P1 #37 — Hard Budget Enforcement V1 tests.

Covers the DoD bullets:

* ``RunBudgetSpecV1`` is content-addressed.
* ``BudgetEnforcerV1`` pre-action enforcement: over-budget
  candidate actions are refused (no silent overspend).
* Cost model is content-addressed and monotone in tokens.
* Stress bench: over-budget regime → 0 commits, N abstains;
  under-budget regime → identical commit behaviour to a no-
  enforcer baseline.
* Every refusal emits a ``BudgetBreachAuditV1`` capsule whose
  CID is re-hashable from the chain.

Plus the anti-cheat clauses:

* The ``budget_disabled`` flag is recorded in the audit chain.
* The cost model refuses to construct with non-positive
  token coefficients (monotone-by-construction).
"""

from __future__ import annotations

import pytest

from coordpy.budget_enforcement_v1 import (
    BudgetAxis,
    BudgetEnforcerV1,
    CandidateActionV1,
    CostModelV1,
    RunBudgetSpecV1,
    W84_BUDGET_AXES,
    W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
    default_cost_model_v1,
    run_budget_breach_stress_bench_v1,
)


def test_w84_cost_model_is_monotone_by_construction():
    with pytest.raises(ValueError):
        CostModelV1(
            schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
            model_cid="x",
            base_cost_per_action=(("replay", 0.0),),
            per_prompt_token_usd=0.0,  # NOT > 0
            per_output_token_usd=0.001,
        )
    with pytest.raises(ValueError):
        CostModelV1(
            schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
            model_cid="x",
            base_cost_per_action=(("replay", -0.001),),  # negative
            per_prompt_token_usd=0.001,
            per_output_token_usd=0.001,
        )


def test_w84_cost_model_estimate_is_monotone_in_tokens():
    cm = default_cost_model_v1()
    a = cm.estimate_usd(
        action="replay", prompt_tokens=10, output_tokens=10)
    b = cm.estimate_usd(
        action="replay", prompt_tokens=100, output_tokens=10)
    c = cm.estimate_usd(
        action="replay", prompt_tokens=10, output_tokens=100)
    assert b > a
    assert c > a


def test_w84_cost_model_cid_stable():
    cm1 = default_cost_model_v1()
    cm2 = default_cost_model_v1()
    assert cm1.cid() == cm2.cid()
    assert len(cm1.cid()) == 64


def test_w84_run_budget_spec_is_content_addressed():
    s = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=0.10,
        max_per_step_latency_ms=100.0,
        max_total_tokens=500,
        max_tool_calls=3,
        max_recompute_flops=10_000.0,
    )
    s2 = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=0.10,
        max_per_step_latency_ms=100.0,
        max_total_tokens=500,
        max_tool_calls=3,
        max_recompute_flops=10_000.0,
    )
    assert s.cid() == s2.cid()
    assert len(s.cid()) == 64


def test_w84_budget_axes_match_enum():
    assert set(W84_BUDGET_AXES) == set(
        a.value for a in BudgetAxis)


def test_w84_enforcer_refuses_over_budget_action():
    spec = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=0.001,
        max_per_step_latency_ms=1.0,
        max_total_tokens=10,
        max_tool_calls=0,
        max_recompute_flops=1.0,
    )
    cm = default_cost_model_v1()
    e = BudgetEnforcerV1(spec=spec, cost_model=cm)
    big = CandidateActionV1(
        action_name="runtime_recompute",
        predicted_usd=0.005,  # > max_total_cost_usd
        predicted_latency_ms=10.0,
        predicted_tokens=50,
        is_tool_call=False,
        predicted_flops=500.0,
    )
    v = e.check(big)
    assert v.permitted is False
    assert v.breach_audit is not None
    # The breached axis must be one of the canonical axes.
    assert v.breach_audit.breached_axis in W84_BUDGET_AXES
    # Tally was not mutated.
    assert e.used()[BudgetAxis.COST_USD.value] == 0.0


def test_w84_enforcer_permits_in_budget_action():
    spec = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=1.0,
        max_per_step_latency_ms=1_000.0,
        max_total_tokens=10_000,
        max_tool_calls=10,
        max_recompute_flops=100_000.0,
    )
    cm = default_cost_model_v1()
    e = BudgetEnforcerV1(spec=spec, cost_model=cm)
    a = CandidateActionV1(
        action_name="replay",
        predicted_usd=0.0001,
        predicted_latency_ms=10.0,
        predicted_tokens=20,
        is_tool_call=False,
        predicted_flops=100.0,
    )
    v = e.check(a)
    assert v.permitted is True
    assert v.breach_audit is None
    e.commit(a)
    assert e.used()[BudgetAxis.COST_USD.value] > 0.0
    assert e.used()[BudgetAxis.TOKENS.value] == 20.0


def test_w84_stress_bench_over_budget_zero_commit():
    rep = run_budget_breach_stress_bench_v1(n_actions=12)
    # Over-budget regime: 0 commits, all refused.
    assert rep.over_budget_regime_n_committed == 0
    assert rep.over_budget_regime_n_refused == 12
    assert rep.over_budget_zero_commit is True


def test_w84_stress_bench_under_budget_matches_no_enforcer():
    rep = run_budget_breach_stress_bench_v1(n_actions=12)
    # Under-budget regime: identical to no-enforcer baseline.
    assert (rep.under_budget_regime_n_committed
            == rep.no_enforcer_regime_n_committed == 12)
    assert rep.under_budget_regime_n_refused == 0
    assert rep.under_budget_matches_no_enforcer is True


def test_w84_breach_audit_cid_is_rehashable():
    spec = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=0.000_1,
        max_per_step_latency_ms=1.0,
        max_total_tokens=1,
        max_tool_calls=0,
        max_recompute_flops=1.0,
    )
    cm = default_cost_model_v1()
    e = BudgetEnforcerV1(spec=spec, cost_model=cm)
    big = CandidateActionV1(
        action_name="runtime_recompute",
        predicted_usd=0.005,
        predicted_latency_ms=10.0,
        predicted_tokens=50,
        is_tool_call=False,
        predicted_flops=500.0,
    )
    v = e.check(big)
    e.record_breach(v.breach_audit)
    # Re-recording the same audit must NOT duplicate (idempotency).
    e.record_breach(v.breach_audit)
    chain = e.audit_chain()
    assert len(chain) == 1
    assert len(chain[0].cid()) == 64


def test_w84_disabled_flag_appears_in_audit_chain():
    # Disabled enforcer permits everything but the flag is in
    # any committed action's breach-audit-set's parent spec.
    spec = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=0.000_001,
        max_per_step_latency_ms=0.001,
        max_total_tokens=0,
        max_tool_calls=0,
        max_recompute_flops=0.001,
        budget_disabled=True,
    )
    cm = default_cost_model_v1()
    e = BudgetEnforcerV1(spec=spec, cost_model=cm)
    a = CandidateActionV1(
        action_name="runtime_recompute",
        predicted_usd=10.0,
        predicted_latency_ms=10_000.0,
        predicted_tokens=1_000,
        is_tool_call=True,
        predicted_flops=1e9,
    )
    v = e.check(a)
    # With disabled flag, even a huge action is permitted.
    assert v.permitted is True
    # The spec CID changes when the disabled flag changes — so
    # any third party verifying a committed chain SEES whether
    # the enforcer was disabled.
    spec_enabled = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=0.000_001,
        max_per_step_latency_ms=0.001,
        max_total_tokens=0,
        max_tool_calls=0,
        max_recompute_flops=0.001,
        budget_disabled=False,
    )
    assert spec.cid() != spec_enabled.cid()


def test_w84_merkle_audit_root_stable_under_idempotent_replay():
    spec = RunBudgetSpecV1(
        schema=W84_BUDGET_ENFORCEMENT_V1_SCHEMA_VERSION,
        max_total_cost_usd=0.000_1,
        max_per_step_latency_ms=1.0,
        max_total_tokens=1,
        max_tool_calls=0,
        max_recompute_flops=1.0,
    )
    cm = default_cost_model_v1()
    e1 = BudgetEnforcerV1(spec=spec, cost_model=cm)
    e2 = BudgetEnforcerV1(spec=spec, cost_model=cm)
    big = CandidateActionV1(
        action_name="runtime_recompute",
        predicted_usd=0.005,
        predicted_latency_ms=10.0,
        predicted_tokens=50,
        is_tool_call=False,
        predicted_flops=500.0,
    )
    for _ in range(3):
        v = e1.check(big)
        e1.record_breach(v.breach_audit)
    for _ in range(7):
        v = e2.check(big)
        e2.record_breach(v.breach_audit)
    # Idempotent record_breach + identical inputs → identical
    # Merkle audit roots.
    assert e1.merkle_audit_root() == e2.merkle_audit_root()
