"""W83 — composed long-horizon multi-agent recovery tests."""

from __future__ import annotations


def test_w83_recovery_regime_set_carries_forward_w79_regimes():
    from coordpy.composed_long_horizon_multi_agent_recovery_v1 import (
        W83_ALL_REGIMES,
        W83_CARRY_FORWARD_REGIMES,
        W83_NEW_REGIME,
    )
    # All 19 W79 regimes are carried forward.
    assert len(W83_CARRY_FORWARD_REGIMES) == 19
    # Plus exactly 1 new regime.
    assert W83_NEW_REGIME == (
        "composed_long_horizon_under_compound_failure")
    assert len(W83_ALL_REGIMES) == 20
    assert W83_NEW_REGIME in W83_ALL_REGIMES


def test_w83_recovery_regime_scenario_content_addressed():
    from coordpy.composed_long_horizon_multi_agent_recovery_v1 import (
        build_regime_scenario_v1,
    )
    a = build_regime_scenario_v1(regime="baseline", seed=1)
    b = build_regime_scenario_v1(regime="baseline", seed=1)
    assert str(a.cid()) == str(b.cid())
    c = build_regime_scenario_v1(regime="baseline", seed=2)
    assert str(a.cid()) != str(c.cid())


def test_w83_recovery_bench_overall_success_high():
    from coordpy.composed_long_horizon_multi_agent_recovery_v1 import (
        run_composed_recovery_bench_v1,
        W83_ALL_REGIMES,
    )
    rep = run_composed_recovery_bench_v1(
        regimes=W83_ALL_REGIMES,
        n_scenarios_per_regime=2,
        n_team_members=7)
    assert float(rep.overall_task_success_rate) >= 0.75
    assert float(rep.overall_audit_verifiable_rate) >= 1.0


def test_w83_recovery_bench_new_regime_succeeds():
    from coordpy.composed_long_horizon_multi_agent_recovery_v1 import (
        run_composed_recovery_bench_v1,
        W83_NEW_REGIME,
    )
    rep = run_composed_recovery_bench_v1(
        regimes=(W83_NEW_REGIME,),
        n_scenarios_per_regime=4)
    assert int(rep.n_regimes) == 1
    new_regime_report = rep.per_regime[0]
    assert new_regime_report.regime == W83_NEW_REGIME
    assert float(new_regime_report.task_success_rate) >= 0.50


def test_w83_recovery_bench_emits_audit_chain_for_committed():
    from coordpy.composed_long_horizon_multi_agent_recovery_v1 import (
        run_composed_recovery_bench_v1,
        W83_ALL_REGIMES,
    )
    rep = run_composed_recovery_bench_v1(
        regimes=W83_ALL_REGIMES[:5],
        n_scenarios_per_regime=2)
    # Every commit emits a Merkle root.
    assert float(
        rep.overall_audit_verifiable_rate) >= 1.0


def test_w83_recovery_witness_emitted():
    from coordpy.composed_long_horizon_multi_agent_recovery_v1 import (
        emit_composed_recovery_witness_v1,
        run_composed_recovery_bench_v1,
    )
    rep = run_composed_recovery_bench_v1(
        n_scenarios_per_regime=1)
    w = emit_composed_recovery_witness_v1(bench=rep)
    assert len(w.cid()) == 64
