"""W84 / P1 #36 — Capacity Scaling Experiments tests."""

from __future__ import annotations


def test_w84_capacity_bench_harness_runs_three_axes():
    """DoD bar: CapacityBenchHarnessV1 exists and runs the
    three target axes."""
    from coordpy.capacity_scaling_bench_v1 import (
        run_capacity_bench_v1,
    )
    rep = run_capacity_bench_v1(
        agent_scales=(10, 50, 200),
        event_scales=(1_000, 10_000),
        token_scales=(1_000, 10_000),
    )
    assert len(rep.agent_count_curve) == 3
    assert len(rep.event_graph_curve_baseline) == 2
    assert len(rep.event_graph_curve_remediation) == 2
    assert len(rep.token_throughput_curve) == 2


def test_w84_agent_count_axis_at_200_agents():
    """V1 stretches to 200 agents."""
    from coordpy.capacity_scaling_bench_v1 import (
        measure_agent_count_axis,
    )
    m = measure_agent_count_axis(n_agents=200)
    assert int(m.scale) == 200
    assert m.wall_clock_seconds > 0.0
    assert int(m.peak_memory_bytes) > 0


def test_w84_event_graph_axis_at_100k_events():
    """V1 stretches to 100k events (in-memory bench)."""
    from coordpy.capacity_scaling_bench_v1 import (
        measure_event_graph_axis,
    )
    m = measure_event_graph_axis(
        n_events=100_000, use_remediation=False,
        n_queries=8)
    assert int(m.scale) == 100_000


def test_w84_token_throughput_axis_at_100k_tokens():
    """V1 stretches to 100k tokens."""
    from coordpy.capacity_scaling_bench_v1 import (
        measure_token_throughput_axis,
    )
    m = measure_token_throughput_axis(n_tokens=100_000)
    assert int(m.scale) == 100_000


def test_w84_capacity_curves_are_multi_scale():
    """DoD bar: scaling curves are reported (not a single
    point). Anti-cheat: do not run at scale 100 once and call
    it scaling."""
    from coordpy.capacity_scaling_bench_v1 import (
        run_capacity_bench_v1,
    )
    rep = run_capacity_bench_v1(
        agent_scales=(10, 50, 200),
        event_scales=(1_000, 10_000),
        token_scales=(1_000, 10_000),
    )
    # Each curve has >= 2 distinct scales.
    scales = {int(m.scale) for m in rep.agent_count_curve}
    assert len(scales) >= 3
    scales = {int(m.scale) for m in rep.event_graph_curve_baseline}
    assert len(scales) >= 2


def test_w84_cliff_identified_honestly():
    """DoD bar: at least one cliff is identified honestly."""
    from coordpy.capacity_scaling_bench_v1 import (
        run_capacity_bench_v1,
    )
    rep = run_capacity_bench_v1(
        agent_scales=(10, 50, 200),
        event_scales=(1_000, 10_000, 100_000),
        token_scales=(1_000, 10_000),
    )
    # Event graph baseline must show a cliff (query latency
    # blows up as N grows).
    assert (
        rep.event_graph_cliff_baseline.cliff_scale > 0)
    # Honest reporting: cliff_factor > 1.0 means ops/sec
    # dropped.
    assert (
        rep.event_graph_cliff_baseline.cliff_factor > 1.0)


def test_w84_remediation_pushes_cliff_one_order_of_magnitude():
    """DoD bar: at least one remediation patch ships and the
    cliff moves at least one order of magnitude."""
    from coordpy.capacity_scaling_bench_v1 import (
        run_capacity_bench_v1,
    )
    rep = run_capacity_bench_v1(
        agent_scales=(10,),
        event_scales=(1_000, 10_000, 100_000),
        token_scales=(1_000,),
    )
    assert bool(
        rep.remediation_pushes_cliff_one_om), rep.to_dict()
    # The remediation's qps at the largest scale must be
    # >= 10x the baseline's qps at the same scale.
    base_largest = float(
        rep.event_graph_curve_baseline[-1].operations_per_second)
    rem_largest = float(
        rep.event_graph_curve_remediation[-1].operations_per_second)
    assert rem_largest >= float(base_largest) * 10.0, (
        f"base={base_largest} rem={rem_largest}")


def test_w84_memory_and_wall_clock_reported_honestly():
    """DoD bar: memory + wall-clock reported honestly (not
    just pass/fail)."""
    from coordpy.capacity_scaling_bench_v1 import (
        measure_event_graph_axis,
    )
    m = measure_event_graph_axis(
        n_events=10_000, use_remediation=False)
    assert float(m.wall_clock_seconds) > 0.0
    assert int(m.peak_memory_bytes) > 0


def test_w84_remediation_does_not_remove_correctness():
    """Anti-cheat: don't "remediate" by removing a load-
    bearing safety check. The remediation must still answer
    the same queries correctly."""
    from coordpy.capacity_scaling_bench_v1 import (
        _AppendOnlyEventLogV1, _InMemoryEventGraphV1,
    )
    base = _InMemoryEventGraphV1()
    rem = _AppendOnlyEventLogV1()
    for i in range(100):
        kind = f"k{i % 4}"
        base.insert(eid=f"e{i}", kind=kind, payload=f"p{i}")
        rem.insert(eid=f"e{i}", kind=kind, payload=f"p{i}")
    # Both must return the same set of IDs for any kind query.
    for kind in (f"k{i}" for i in range(4)):
        bs = set(base.query_by_kind(kind=kind))
        rs = set(rem.query_by_kind(kind=kind))
        assert bs == rs, (kind, bs, rs)


def test_w84_capacity_bench_report_cid_stable():
    """The capacity bench report is content-addressed."""
    from coordpy.capacity_scaling_bench_v1 import (
        run_capacity_bench_v1,
    )
    # Two runs may have slightly different wall-clock numbers;
    # we only assert the report has a stable CID-shape (64
    # hex). The cliff identification is what's load-bearing.
    rep = run_capacity_bench_v1(
        agent_scales=(10,),
        event_scales=(1_000, 10_000),
        token_scales=(1_000,),
    )
    assert len(rep.cid()) == 64
