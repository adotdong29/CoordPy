"""W82 / P2 #15 — Simultaneous compound-failure bench tests.

Covers:
- factor mask is content-addressed
- all 32 masks (2^5) enumerate
- scenario is deterministic on seed + factor mask
- all five required metrics are recorded per outcome
- failure mode attribution is well-defined per strategy
- the four canonical strategies are present
- W82 compound-repair strictly beats every baseline on the
  load-bearing all-5-active mask
- bench end-to-end is deterministic
"""

from __future__ import annotations


def test_w82_cfb_factor_mask_content_addressed():
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
    )
    a = CompoundFailureFactorMaskV1(
        contradiction=True, corruption=True)
    b = CompoundFailureFactorMaskV1(
        contradiction=True, corruption=True)
    c = CompoundFailureFactorMaskV1(
        contradiction=True, corruption=False)
    assert a.cid() == b.cid()
    assert a.cid() != c.cid()
    assert a.n_active() == 2
    assert c.n_active() == 1
    assert "contradiction" in a.active_factors()
    assert "corruption" in a.active_factors()


def test_w82_cfb_enumerates_all_32_masks():
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        all_factor_masks_v1,
    )
    all32 = all_factor_masks_v1()
    assert len(all32) == 32
    # Distinct CIDs
    assert len({m.cid() for m in all32}) == 32
    # At least one mask has all 5 active
    assert any(m.n_active() == 5 for m in all32)
    # At least one mask has 0 active
    assert any(m.n_active() == 0 for m in all32)


def test_w82_cfb_scenario_deterministic_on_seed_and_mask():
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        build_compound_failure_scenario_v1,
    )
    mask = CompoundFailureFactorMaskV1(
        contradiction=True, corruption=True, blackout=True)
    a = build_compound_failure_scenario_v1(
        seed=42, factor_mask=mask)
    b = build_compound_failure_scenario_v1(
        seed=42, factor_mask=mask)
    c = build_compound_failure_scenario_v1(
        seed=43, factor_mask=mask)
    assert a.cid() == b.cid()
    assert a.cid() != c.cid()
    assert a.n_witnesses == b.n_witnesses


def test_w82_cfb_inactive_factor_does_not_corrupt():
    """If corruption factor is off, no witness has the
    corruption noise pattern (large L2 deviation from gt)."""
    import numpy as np
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        build_compound_failure_scenario_v1,
    )
    # Mask with only blackout — no corruption, no contradiction
    mask = CompoundFailureFactorMaskV1(blackout=True)
    sc = build_compound_failure_scenario_v1(
        seed=7, factor_mask=mask)
    # All witnesses must be near ground truth.
    for w in sc.witnesses:
        err = float(np.linalg.norm(
            w.value - sc.ground_truth))
        assert err < 0.5, (
            f"witness {w.witness_id} has error {err} but "
            f"no corruption/contradiction was active")


def test_w82_cfb_strategy_set_has_four_canonical():
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        build_compound_failure_strategy_set_v1,
        W82_CFB_STRATEGY_NAIVE_MAJORITY,
        W82_CFB_STRATEGY_BOUNDED_WINDOW_K128,
        W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY,
        W82_CFB_STRATEGY_W82_COMPOUND_REPAIR,
    )
    s = build_compound_failure_strategy_set_v1()
    assert set(s.keys()) == {
        W82_CFB_STRATEGY_NAIVE_MAJORITY,
        W82_CFB_STRATEGY_BOUNDED_WINDOW_K128,
        W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY,
        W82_CFB_STRATEGY_W82_COMPOUND_REPAIR,
    }


def test_w82_cfb_outcome_records_all_required_metrics():
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        build_compound_failure_scenario_v1,
        build_compound_failure_strategy_set_v1,
        W82_CFB_STRATEGY_W82_COMPOUND_REPAIR,
    )
    mask = CompoundFailureFactorMaskV1(blackout=True)
    sc = build_compound_failure_scenario_v1(
        seed=2, factor_mask=mask)
    s = build_compound_failure_strategy_set_v1()
    o = s[W82_CFB_STRATEGY_W82_COMPOUND_REPAIR](sc)
    d = o.to_dict()
    for k in (
            "task_success",
            "reconstruction_fidelity",
            "visible_tokens_used",
            "replay_flops",
            "primary_failure_factor",
    ):
        assert k in d


def test_w82_cfb_blackout_restart_drive_substrate_path_costs():
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        build_compound_failure_scenario_v1,
        build_compound_failure_strategy_set_v1,
        W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY,
        W82_CFB_STRATEGY_W82_COMPOUND_REPAIR,
    )
    none_mask = CompoundFailureFactorMaskV1()
    blackout_restart_mask = CompoundFailureFactorMaskV1(
        blackout=True, restart=True)
    sc_none = build_compound_failure_scenario_v1(
        seed=17, factor_mask=none_mask)
    sc_br = build_compound_failure_scenario_v1(
        seed=17, factor_mask=blackout_restart_mask)
    strategies = build_compound_failure_strategy_set_v1()
    sub_none = strategies[W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY](sc_none)
    sub_br = strategies[W82_CFB_STRATEGY_SUBSTRATE_V2_ONLY](sc_br)
    w82_none = strategies[W82_CFB_STRATEGY_W82_COMPOUND_REPAIR](sc_none)
    w82_br = strategies[W82_CFB_STRATEGY_W82_COMPOUND_REPAIR](sc_br)
    assert sub_none.task_success and sub_br.task_success
    assert w82_none.task_success and w82_br.task_success
    assert sub_none.replay_flops < sub_br.replay_flops
    assert w82_none.replay_flops < w82_br.replay_flops


def test_w82_cfb_primary_failure_attribution_per_strategy():
    """Bounded-window's primary failure under blackout must
    be `blackout`, not `corruption`."""
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        build_compound_failure_scenario_v1,
        build_compound_failure_strategy_set_v1,
        W82_CFB_STRATEGY_BOUNDED_WINDOW_K128,
        W82_CFB_STRATEGY_NAIVE_MAJORITY,
    )
    mask = CompoundFailureFactorMaskV1(
        contradiction=True, corruption=True, blackout=True)
    sc = build_compound_failure_scenario_v1(
        seed=11, factor_mask=mask)
    s = build_compound_failure_strategy_set_v1()
    bw_outcome = s[W82_CFB_STRATEGY_BOUNDED_WINDOW_K128](sc)
    assert (
        not bw_outcome.task_success and
        bw_outcome.primary_failure_factor == "blackout")
    naive_outcome = s[W82_CFB_STRATEGY_NAIVE_MAJORITY](sc)
    # Naive majority fails under contradiction (top
    # attribution per strategy).
    assert (
        not naive_outcome.task_success and
        naive_outcome.primary_failure_factor in
        ("contradiction", "corruption"))


def test_w82_cfb_w82_dominates_at_all5_active():
    """Load-bearing W82 P2 #15 claim: at all-5-active mask,
    W82 compound-repair strictly beats every baseline."""
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        run_compound_failure_bench_end_to_end_v1,
        W82_CFB_STRATEGY_W82_COMPOUND_REPAIR,
    )
    all5 = CompoundFailureFactorMaskV1(
        contradiction=True, corruption=True,
        replacement=True, restart=True, blackout=True)
    report, witness = run_compound_failure_bench_end_to_end_v1(
        factor_masks=[all5], n_scenarios_per_mask=8)
    assert witness.w82_dominates_at_all5_active is True
    assert witness.w82_success_rate_at_all5 >= 0.7, (
        f"W82 success rate {witness.w82_success_rate_at_all5} "
        f"at all5 active — must be >= 0.7 for the W82 bar")
    assert (
        witness.max_baseline_success_rate_at_all5 <= 0.2), (
        f"Some baseline succeeded at all5 active with rate "
        f"{witness.max_baseline_success_rate_at_all5} — "
        f"the bench is too easy")


def test_w82_cfb_per_cell_metrics_present():
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        run_compound_failure_bench_v1,
    )
    masks = [
        CompoundFailureFactorMaskV1(),  # nothing active
        CompoundFailureFactorMaskV1(blackout=True),
    ]
    report = run_compound_failure_bench_v1(
        factor_masks=masks, n_scenarios_per_mask=2)
    n_cells = (
        int(len(masks)) * int(len(report.strategy_labels)))
    assert len(report.per_cell_success_rate) == n_cells
    assert len(report.per_cell_mean_fidelity) == n_cells
    assert len(
        report.per_cell_mean_visible_tokens) == n_cells
    assert len(report.per_cell_mean_replay_flops) == n_cells


def test_w82_cfb_bench_deterministic():
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        run_compound_failure_bench_end_to_end_v1,
    )
    masks = [
        CompoundFailureFactorMaskV1(),
        CompoundFailureFactorMaskV1(blackout=True),
    ]
    a, _ = run_compound_failure_bench_end_to_end_v1(
        factor_masks=masks, n_scenarios_per_mask=3)
    b, _ = run_compound_failure_bench_end_to_end_v1(
        factor_masks=masks, n_scenarios_per_mask=3)
    assert a.cid() == b.cid()


def test_w82_cfb_all_strategies_pass_under_no_failure():
    """All strategies must trivially win when no factor is
    active."""
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        run_compound_failure_bench_end_to_end_v1,
    )
    none_mask = CompoundFailureFactorMaskV1()
    report, _ = run_compound_failure_bench_end_to_end_v1(
        factor_masks=[none_mask], n_scenarios_per_mask=4)
    # Naive majority succeeds (no contradiction or corruption)
    assert float(report.success_rate(
        none_mask, "naive_majority")) >= 0.95
    # W82 compound-repair always succeeds with no factor
    assert float(report.success_rate(
        none_mask, "w82_compound_repair")) >= 0.95


def test_w82_cfb_witness_cid_responds_to_config():
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        run_compound_failure_bench_end_to_end_v1,
    )
    all5 = CompoundFailureFactorMaskV1(
        contradiction=True, corruption=True,
        replacement=True, restart=True, blackout=True)
    _, w_a = run_compound_failure_bench_end_to_end_v1(
        factor_masks=[all5], n_scenarios_per_mask=2)
    _, w_b = run_compound_failure_bench_end_to_end_v1(
        factor_masks=[all5], n_scenarios_per_mask=4)
    assert w_a.cid() != w_b.cid()


def test_w82_cfb_full_32_mask_sweep():
    """The default 2^5 = 32 mask sweep runs and reports."""
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        run_compound_failure_bench_end_to_end_v1,
    )
    report, witness = run_compound_failure_bench_end_to_end_v1(
        n_scenarios_per_mask=1)
    assert len(report.factor_masks) == 32
    # 32 masks * 4 strategies = 128 cells
    assert len(report.per_cell_success_rate) == 128
    assert witness.w82_dominates_at_all5_active is True


def test_w82_cfb_w82_fails_under_extreme_factor_budget():
    """Honest-scope test: under extreme factor budgets
    (contradicting + corrupted > honest majority), W82
    compound-repair MUST also fail. This validates the
    W82-L-COMPOUND-FAILURE-V1-FACTOR-BUDGET-CAP limitation
    is real and honest, not just a paper claim."""
    from coordpy.simultaneous_compound_failure_benchmark_v1 import (
        CompoundFailureFactorMaskV1,
        build_compound_failure_scenario_v1,
        build_compound_failure_strategy_set_v1,
        W82_CFB_STRATEGY_W82_COMPOUND_REPAIR,
    )
    mask = CompoundFailureFactorMaskV1(
        contradiction=True, corruption=True)
    strategies = build_compound_failure_strategy_set_v1()
    # Force contradiction_fraction + corruption_fraction up
    # to 0.85 of witnesses (massively over-budget).
    n_total = 12
    fail_count = 0
    for seed in range(n_total):
        sc = build_compound_failure_scenario_v1(
            seed=seed * 17 + 3,
            factor_mask=mask,
            contradiction_fraction=0.55,
            corruption_fraction=0.30,
            replacement_fraction=0.0)
        o = strategies[W82_CFB_STRATEGY_W82_COMPOUND_REPAIR](sc)
        if not o.task_success:
            fail_count += 1
    # Under such extreme corruption + contradiction (85% of
    # witnesses are adversarial), the W82 strategy must fail
    # on a substantial fraction of seeds — this is the honest-
    # scope point of W82-L-FACTOR-BUDGET-CAP.
    assert fail_count >= 1, (
        f"W82 should fail under extreme factor budget but "
        f"won on all {n_total} seeds — limitation may be "
        f"overstated")
