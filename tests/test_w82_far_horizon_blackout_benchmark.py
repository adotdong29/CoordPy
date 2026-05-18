"""W82 / P2 #10 — Far-horizon blackout benchmark tests.

Covers:
- horizon ladder is explicit (not one default)
- determinism on seed
- bench reports include all five required metrics per cell
- failure curves are produced for every strategy
- substrate strictly dominates all bounded baselines on the
  load-bearing ladder
- substrate's *own* failure mode (carrier truncation) is
  reported honestly
- witness CID is content-addressed
- end-to-end runner returns deterministic report
"""

from __future__ import annotations


def test_w82_default_horizon_ladder_is_explicit():
    """The default ladder must contain at least 4 horizons,
    monotonically increasing, with at least one >= 1000 and at
    least one >= 16000 (far past today's default comfort zone).
    """
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        W82_DEFAULT_HORIZON_LADDER,
    )
    h = W82_DEFAULT_HORIZON_LADDER
    assert len(h) >= 4
    assert sorted(h) == list(h)
    assert max(h) >= 16_000
    assert min(h) >= 1000


def test_w82_scenario_is_content_addressed():
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        build_far_horizon_blackout_scenario_v1,
    )
    a = build_far_horizon_blackout_scenario_v1(
        seed=7, horizon_turns=1000)
    b = build_far_horizon_blackout_scenario_v1(
        seed=7, horizon_turns=1000)
    c = build_far_horizon_blackout_scenario_v1(
        seed=8, horizon_turns=1000)
    assert a.cid() == b.cid()
    assert a.cid() != c.cid()


def test_w82_scenarios_deterministic_on_base_seed():
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        build_far_horizon_blackout_scenarios_v1,
    )
    a = build_far_horizon_blackout_scenarios_v1(
        horizons=(100, 200), n_per_horizon=3, base_seed=42)
    b = build_far_horizon_blackout_scenarios_v1(
        horizons=(100, 200), n_per_horizon=3, base_seed=42)
    assert len(a) == len(b) == 6
    for ai, bi in zip(a, b):
        assert ai.cid() == bi.cid()


def test_w82_strategy_set_covers_four_categories():
    """Strategy set must include transcript-only, bounded-
    window (multiple ks), rolling-summary, and substrate."""
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        build_far_horizon_strategy_set_v1,
        W82_STRATEGY_TRANSCRIPT_ONLY,
        W82_STRATEGY_ROLLING_SUMMARY,
        W82_STRATEGY_LHR_SUBSTRATE_V2,
    )
    s = build_far_horizon_strategy_set_v1()
    assert W82_STRATEGY_TRANSCRIPT_ONLY in s
    assert W82_STRATEGY_ROLLING_SUMMARY in s
    assert W82_STRATEGY_LHR_SUBSTRATE_V2 in s
    bounded_labels = [
        k for k in s if k.startswith("bounded_window_k")]
    assert len(bounded_labels) >= 3


def test_w82_outcome_records_all_required_metrics():
    """Each outcome must record: task_success, fidelity,
    visible tokens, replay cost, recompute flops, carrier
    walks."""
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        build_far_horizon_blackout_scenario_v1,
        build_far_horizon_strategy_set_v1,
        W82_STRATEGY_LHR_SUBSTRATE_V2,
    )
    sc = build_far_horizon_blackout_scenario_v1(
        seed=11, horizon_turns=500)
    strategies = build_far_horizon_strategy_set_v1()
    o = strategies[W82_STRATEGY_LHR_SUBSTRATE_V2](sc)
    d = o.to_dict()
    for k in (
            "task_success",
            "reconstruction_fidelity",
            "visible_tokens_used",
            "replay_flops",
            "recompute_flops",
            "carrier_walks",
    ):
        assert k in d


def test_w82_substrate_strictly_dominates_bounded_baselines_at_h1000():
    """The load-bearing W82 P2 #10 bar: on horizon 1000+ the
    substrate has success rate strictly greater than every
    bounded baseline."""
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        run_far_horizon_blackout_bench_end_to_end_v1,
        W82_STRATEGY_LHR_SUBSTRATE_V2,
    )
    report, _w = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(1000,), n_per_horizon=3)
    sub_curve = report.curve_for(W82_STRATEGY_LHR_SUBSTRATE_V2)
    sub_rate = float(sub_curve.success_rate_at(1000))
    assert sub_rate >= 0.95, (
        f"substrate failed at h=1000 with rate {sub_rate}")
    for c in report.failure_curves:
        if str(c.strategy_label) == str(
                W82_STRATEGY_LHR_SUBSTRATE_V2):
            continue
        other_rate = float(c.success_rate_at(1000))
        assert other_rate <= 0.0 + 1e-12, (
            f"baseline {c.strategy_label} unexpectedly "
            f"succeeded at h=1000 with rate {other_rate}")


def test_w82_substrate_dominates_across_full_ladder():
    """Substrate dominates every baseline across the whole
    default horizon ladder."""
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        run_far_horizon_blackout_bench_end_to_end_v1,
        emit_far_horizon_witness_v1,
        W82_DEFAULT_HORIZON_LADDER,
    )
    report, witness = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=W82_DEFAULT_HORIZON_LADDER,
        n_per_horizon=2)
    assert set(witness.substrate_dominates_at_horizons) == (
        set(int(h) for h in W82_DEFAULT_HORIZON_LADDER))


def test_w82_substrate_failure_curve_when_carrier_truncated():
    """Substrate honestly fails when the carrier is truncated
    below the horizon."""
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        run_far_horizon_blackout_bench_end_to_end_v1,
        W82_STRATEGY_LHR_SUBSTRATE_V2,
    )
    # Carrier depth = 50 means the substrate keeps only the
    # 50 most recent entries. Source-turns at horizon 1000
    # fall outside, so substrate fails.
    report, _w = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(1000, 4000),
        n_per_horizon=2,
        substrate_carrier_depth=50)
    sub_curve = report.curve_for(W82_STRATEGY_LHR_SUBSTRATE_V2)
    for h in (1000, 4000):
        rate = float(sub_curve.success_rate_at(h))
        assert rate <= 0.0 + 1e-12, (
            f"truncated substrate must fail at h={h} but "
            f"got rate {rate}")


def test_w82_bounded_window_k_succeeds_only_below_k():
    """The bounded-window-k baseline must succeed exactly when
    horizon < k."""
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        build_far_horizon_blackout_scenario_v1,
        build_far_horizon_strategy_set_v1,
    )
    strategies = build_far_horizon_strategy_set_v1()
    for k in (4, 32, 64, 128):
        label = f"bounded_window_k{k}"
        assert label in strategies, (
            f"missing bounded_window_k strategy: {label}")
        sc_in = build_far_horizon_blackout_scenario_v1(
            seed=7, horizon_turns=int(k - 1),
            n_restart_cycles=0,
            branch_rejoin_offset=int(k + 1000))
        sc_out = build_far_horizon_blackout_scenario_v1(
            seed=7, horizon_turns=int(k),
            n_restart_cycles=0,
            branch_rejoin_offset=int(k + 1000))
        assert strategies[label](sc_in).task_success
        assert not strategies[label](sc_out).task_success


def test_w82_visible_baselines_respond_to_restart_and_branch():
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        build_far_horizon_blackout_scenario_v1,
        build_far_horizon_strategy_set_v1,
        W82_STRATEGY_TRANSCRIPT_ONLY,
        W82_STRATEGY_LHR_SUBSTRATE_V2,
    )
    strategies = build_far_horizon_strategy_set_v1()
    clear = build_far_horizon_blackout_scenario_v1(
        seed=7, horizon_turns=16,
        n_restart_cycles=0,
        branch_rejoin_offset=10_000)
    stressed = build_far_horizon_blackout_scenario_v1(
        seed=7, horizon_turns=16,
        n_restart_cycles=2,
        branch_rejoin_offset=4)
    assert strategies[W82_STRATEGY_TRANSCRIPT_ONLY](clear).task_success
    assert not strategies[W82_STRATEGY_TRANSCRIPT_ONLY](stressed).task_success
    assert strategies[W82_STRATEGY_LHR_SUBSTRATE_V2](stressed).task_success


def test_w82_failure_curves_present_for_every_strategy():
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        run_far_horizon_blackout_bench_end_to_end_v1,
    )
    report, _w = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(500,), n_per_horizon=2)
    assert len(report.failure_curves) == len(
        report.strategy_labels)
    seen = set(c.strategy_label
               for c in report.failure_curves)
    assert seen == set(report.strategy_labels)


def test_w82_bench_report_cid_is_deterministic():
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        run_far_horizon_blackout_bench_end_to_end_v1,
    )
    a, _ = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(500, 1000), n_per_horizon=2)
    b, _ = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(500, 1000), n_per_horizon=2)
    assert a.cid() == b.cid()
    assert a.config_cid == b.config_cid


def test_w82_witness_cid_changes_when_horizons_differ():
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        run_far_horizon_blackout_bench_end_to_end_v1,
    )
    _, w_a = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(500,), n_per_horizon=2)
    _, w_b = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(1000,), n_per_horizon=2)
    assert w_a.cid() != w_b.cid()


def test_w82_failure_curve_monotone_non_increasing_for_baselines():
    """Bounded baselines' success rate should be non-increasing
    as horizon grows (they have a hard cutoff at k)."""
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        run_far_horizon_blackout_bench_end_to_end_v1,
    )
    report, _w = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(50, 100, 1000, 4000), n_per_horizon=2)
    for c in report.failure_curves:
        if c.strategy_label.startswith("bounded_window_k"):
            rates = list(c.success_rate_at_horizon)
            assert all(
                rates[i] >= rates[i + 1]
                for i in range(len(rates) - 1)), (
                f"{c.strategy_label} not monotone: {rates}")


def test_w82_substrate_carrier_walks_logarithmic_in_horizon():
    """The substrate must spend O(log n) walks, not O(n)."""
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        run_far_horizon_blackout_bench_end_to_end_v1,
        W82_STRATEGY_LHR_SUBSTRATE_V2,
    )
    report, _w = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(100, 1000, 16_000), n_per_horizon=1)
    sub_curve = report.curve_for(W82_STRATEGY_LHR_SUBSTRATE_V2)
    walks = list(sub_curve.mean_carrier_walks_at_horizon)
    # For carrier sizes 100 / 1000 / 16000 with fanout 4
    # the hop counts should be ~4, ~5, ~7. Sanity-check
    # they are far less than the horizons.
    for h, w in zip(sub_curve.horizons, walks):
        assert float(w) <= 32.0, (
            f"substrate walks at h={h} too high: {w}")
        assert float(w) < float(h) / 2.0, (
            f"substrate walks at h={h} not logarithmic: {w}")


def test_w82_witness_substrate_dominates_signal_strong():
    """Witness must report substrate dominates at every
    horizon on the default ladder."""
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        run_far_horizon_blackout_bench_end_to_end_v1,
    )
    _r, w = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(1000, 4000, 16_000), n_per_horizon=2)
    assert set(w.substrate_dominates_at_horizons) == {
        1000, 4000, 16_000}


def test_w82_at_horizon_100k_substrate_still_wins():
    """Far-horizon stress: at horizon 100,000 the substrate
    still beats every baseline."""
    from coordpy.far_horizon_blackout_benchmark_v1 import (
        run_far_horizon_blackout_bench_end_to_end_v1,
        W82_STRATEGY_LHR_SUBSTRATE_V2,
    )
    report, _w = run_far_horizon_blackout_bench_end_to_end_v1(
        horizons=(100_000,), n_per_horizon=1)
    sub_curve = report.curve_for(W82_STRATEGY_LHR_SUBSTRATE_V2)
    sub_rate = float(sub_curve.success_rate_at(100_000))
    assert sub_rate == 1.0, (
        f"substrate failed at h=100k with rate {sub_rate}")
    for c in report.failure_curves:
        if str(c.strategy_label) == str(
                W82_STRATEGY_LHR_SUBSTRATE_V2):
            continue
        rate = float(c.success_rate_at(100_000))
        assert rate <= 0.0 + 1e-12, (
            f"baseline {c.strategy_label} unexpectedly "
            f"succeeded at h=100k with rate {rate}")
