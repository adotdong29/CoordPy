"""W84 / P1 #37 — Hard Cost/Latency Budget Enforcement tests."""

from __future__ import annotations


def test_w84_run_budget_spec_content_addressed():
    """DoD bar: RunBudgetSpecV1 is content-addressed."""
    from coordpy.budget_enforcement_v1 import (
        build_run_budget_spec_v1,
    )
    a = build_run_budget_spec_v1(
        run_cid="r", max_total_cost_usd=1.0,
        max_total_tokens=100)
    b = build_run_budget_spec_v1(
        run_cid="r", max_total_cost_usd=1.0,
        max_total_tokens=100)
    assert a.cid() == b.cid()
    c = build_run_budget_spec_v1(
        run_cid="r", max_total_cost_usd=1.0,
        max_total_tokens=200)  # different
    assert a.cid() != c.cid()


def test_w84_cost_model_content_addressed_and_monotone():
    """DoD bar: identical configs produce identical cost CIDs.
    Anti-cheat: cost model produces monotone-in-tokens
    estimates."""
    from coordpy.budget_enforcement_v1 import (
        build_default_cost_model_v1,
    )
    a = build_default_cost_model_v1()
    b = build_default_cost_model_v1()
    assert a.cid() == b.cid()
    # Monotone in prompt_tokens.
    c1 = a.estimate_cost_usd(
        action="runtime_recompute",
        prompt_tokens=100, output_tokens=50)
    c2 = a.estimate_cost_usd(
        action="runtime_recompute",
        prompt_tokens=200, output_tokens=50)
    assert c2 > c1
    # Monotone in tool_calls.
    c3 = a.estimate_cost_usd(
        action="runtime_recompute",
        prompt_tokens=100, output_tokens=50,
        tool_calls=0)
    c4 = a.estimate_cost_usd(
        action="runtime_recompute",
        prompt_tokens=100, output_tokens=50,
        tool_calls=5)
    assert c4 > c3


def test_w84_enforcer_blocks_over_budget_commit():
    """DoD bar: on an over-budget proposal, the pipeline
    commits 0 times and emits an abstain decision with a
    breach audit capsule."""
    from coordpy.budget_enforcement_v1 import (
        ActionProposalV1, BudgetCommitVerdict,
        BudgetEnforcerV1, build_default_cost_model_v1,
        build_run_budget_spec_v1,
    )
    cm = build_default_cost_model_v1()
    spec = build_run_budget_spec_v1(
        max_total_cost_usd=0.0001)  # near zero
    enf = BudgetEnforcerV1(spec=spec, cost_model=cm)
    p = ActionProposalV1(
        action="runtime_recompute",
        prompt_tokens=1000, output_tokens=500,
        tool_calls=0, recompute_flops=0,
        measured_step_latency_ms=10.0)
    verdict, audit = enf.propose(proposal=p)
    assert verdict == (
        BudgetCommitVerdict.ABSTAIN_ON_BREACH.value)
    assert audit is not None
    assert "cost_usd" in audit.breach_axes
    # Audit is content-addressed.
    assert len(audit.cid()) == 64


def test_w84_enforcer_allows_under_budget_commit():
    """DoD bar: on a deliberately-under-budget regime, the
    pipeline commits as it would without the enforcer (no
    behaviour change inside budget)."""
    from coordpy.budget_enforcement_v1 import (
        ActionProposalV1, BudgetCommitVerdict,
        BudgetEnforcerV1, build_default_cost_model_v1,
        build_run_budget_spec_v1,
    )
    cm = build_default_cost_model_v1()
    spec = build_run_budget_spec_v1(
        max_total_cost_usd=10.0,
        max_total_tokens=1_000_000)
    enf = BudgetEnforcerV1(spec=spec, cost_model=cm)
    p = ActionProposalV1(
        action="replay",
        prompt_tokens=50, output_tokens=10,
        tool_calls=0, recompute_flops=0,
        measured_step_latency_ms=5.0)
    verdict, audit = enf.propose(proposal=p)
    assert verdict == (
        BudgetCommitVerdict.COMMIT_ALLOWED.value)
    assert audit is None


def test_w84_enforcer_blocks_token_overshoot():
    """Token budget breach must be detected."""
    from coordpy.budget_enforcement_v1 import (
        ActionProposalV1, BudgetCommitVerdict,
        BudgetEnforcerV1, build_default_cost_model_v1,
        build_run_budget_spec_v1,
    )
    cm = build_default_cost_model_v1()
    spec = build_run_budget_spec_v1(
        max_total_cost_usd=10.0,
        max_total_tokens=100)
    enf = BudgetEnforcerV1(spec=spec, cost_model=cm)
    p = ActionProposalV1(
        action="runtime_recompute",
        prompt_tokens=200, output_tokens=0,
        tool_calls=0, recompute_flops=0,
        measured_step_latency_ms=5.0)
    verdict, audit = enf.propose(proposal=p)
    assert verdict == (
        BudgetCommitVerdict.ABSTAIN_ON_BREACH.value)
    assert "total_tokens" in audit.breach_axes


def test_w84_enforcer_blocks_per_step_latency():
    """DoD bar: per-step latency budget is enforced (not just
    cost). The issue's anti-cheat: do not skip latency budgets."""
    from coordpy.budget_enforcement_v1 import (
        ActionProposalV1, BudgetBreachAxis,
        BudgetCommitVerdict,
        BudgetEnforcerV1, build_default_cost_model_v1,
        build_run_budget_spec_v1,
    )
    cm = build_default_cost_model_v1()
    spec = build_run_budget_spec_v1(
        max_total_cost_usd=10.0,
        max_per_step_latency_ms=50.0)
    enf = BudgetEnforcerV1(spec=spec, cost_model=cm)
    p = ActionProposalV1(
        action="runtime_recompute",
        prompt_tokens=10, output_tokens=10,
        tool_calls=0, recompute_flops=0,
        measured_step_latency_ms=500.0)  # over
    verdict, audit = enf.propose(proposal=p)
    assert verdict == (
        BudgetCommitVerdict.ABSTAIN_ON_BREACH.value)
    assert (
        BudgetBreachAxis.PER_STEP_LATENCY_MS.value
        in audit.breach_axes)


def test_w84_enforcer_blocks_tool_call_overshoot():
    """DoD bar: tools count toward max_tool_calls."""
    from coordpy.budget_enforcement_v1 import (
        ActionProposalV1, BudgetCommitVerdict,
        BudgetEnforcerV1, build_default_cost_model_v1,
        build_run_budget_spec_v1,
    )
    cm = build_default_cost_model_v1()
    spec = build_run_budget_spec_v1(
        max_total_cost_usd=10.0, max_tool_calls=2)
    enf = BudgetEnforcerV1(spec=spec, cost_model=cm)
    # First two proposals commit; third overshoots.
    p = ActionProposalV1(
        action="runtime_recompute",
        prompt_tokens=10, output_tokens=10,
        tool_calls=1, recompute_flops=0,
        measured_step_latency_ms=5.0)
    enf.propose(proposal=p)
    enf.commit(proposal=p)
    enf.propose(proposal=p)
    enf.commit(proposal=p)
    verdict, audit = enf.propose(proposal=p)
    assert verdict == (
        BudgetCommitVerdict.ABSTAIN_ON_BREACH.value)
    assert "tool_calls" in audit.breach_axes


def test_w84_breach_audit_records_pre_and_would_be_state():
    """DoD bar: breach audit carries the pre-budget, the
    post-budget, the would-be action, and the reason."""
    from coordpy.budget_enforcement_v1 import (
        ActionProposalV1, BudgetEnforcerV1,
        build_default_cost_model_v1,
        build_run_budget_spec_v1,
    )
    cm = build_default_cost_model_v1()
    spec = build_run_budget_spec_v1(
        max_total_cost_usd=0.001)
    enf = BudgetEnforcerV1(spec=spec, cost_model=cm)
    p = ActionProposalV1(
        action="runtime_recompute",
        prompt_tokens=500, output_tokens=500,
        tool_calls=0, recompute_flops=0,
        measured_step_latency_ms=5.0)
    _, audit = enf.propose(proposal=p)
    assert audit is not None
    d = audit.to_dict()
    assert "pre_budget_remaining_cost_usd" in d
    assert "would_cost_usd" in d
    assert "breach_axes" in d
    assert "would_be_action" in d
    assert d["would_be_action"] == "runtime_recompute"


def test_w84_enforcer_disabled_flag_recorded_in_audit():
    """DoD bar: enforcer-disabled flag must appear in the
    audit chain when used. V1 returns ENFORCER_DISABLED
    verdict (no audit) — but the spec CID + the disabled
    flag are still recorded on the spec's audit trail."""
    from coordpy.budget_enforcement_v1 import (
        ActionProposalV1, BudgetCommitVerdict,
        BudgetEnforcerV1, build_default_cost_model_v1,
        build_run_budget_spec_v1,
    )
    cm = build_default_cost_model_v1()
    spec = build_run_budget_spec_v1(
        max_total_cost_usd=0.0001,
        enforcer_disabled=True)
    enf = BudgetEnforcerV1(spec=spec, cost_model=cm)
    p = ActionProposalV1(
        action="runtime_recompute",
        prompt_tokens=10000, output_tokens=10000,
        tool_calls=0, recompute_flops=0,
        measured_step_latency_ms=5.0)
    verdict, audit = enf.propose(proposal=p)
    assert verdict == (
        BudgetCommitVerdict.ENFORCER_DISABLED.value)
    # The disabled-flag IS in the spec's CID — so any audit
    # of the run can pull this out.
    assert "enforcer_disabled" in spec.to_dict()
    assert bool(spec.to_dict()["enforcer_disabled"])


def test_w84_budget_stress_bench_passes():
    """DoD bar: deliberately over-budget regime -> 0 commits;
    deliberately under-budget regime -> exact commit count
    matches proposal count."""
    from coordpy.budget_enforcement_v1 import (
        run_budget_stress_bench_v1,
    )
    rep = run_budget_stress_bench_v1()
    d = rep.to_dict()
    assert int(rep.n_commits) == 0, d
    assert int(rep.n_abstain_on_breach) == 10, d
    assert bool(rep.all_proposals_audit_logged), d
    assert bool(rep.under_budget_commits_exactly_match), d
    assert len(rep.breach_axes_seen) >= 1, d


def test_w84_breach_audit_not_silently_dropped():
    """Anti-cheat: every refusal must emit a content-addressed
    breach audit — silently dropping is the opposite of
    enforcement."""
    from coordpy.budget_enforcement_v1 import (
        ActionProposalV1, BudgetEnforcerV1,
        build_default_cost_model_v1,
        build_run_budget_spec_v1,
    )
    cm = build_default_cost_model_v1()
    spec = build_run_budget_spec_v1(
        max_total_cost_usd=0.0001,
        record_breach_audit=True)
    enf = BudgetEnforcerV1(spec=spec, cost_model=cm)
    p = ActionProposalV1(
        action="runtime_recompute",
        prompt_tokens=100, output_tokens=100,
        tool_calls=0, recompute_flops=0,
        measured_step_latency_ms=5.0)
    # 5 over-budget proposals -> 5 audits.
    for _ in range(5):
        enf.propose(proposal=p)
    assert len(enf.breach_audits) == 5


def test_w84_cost_model_action_multipliers_distinct():
    """Anti-cheat: cost model produces distinct costs for
    distinct actions (so the action axis is meaningful)."""
    from coordpy.budget_enforcement_v1 import (
        build_default_cost_model_v1,
    )
    cm = build_default_cost_model_v1()
    replay = cm.estimate_cost_usd(
        action="replay",
        prompt_tokens=1000, output_tokens=500)
    promote = cm.estimate_cost_usd(
        action="promote_to_richer_substrate",
        prompt_tokens=1000, output_tokens=500)
    assert promote > replay  # promote is more expensive
    abstain = cm.estimate_cost_usd(
        action="abstain",
        prompt_tokens=1000, output_tokens=500)
    assert abstain == 0.0  # abstain costs nothing
