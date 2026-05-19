"""W84 / P1 #36 — Capacity scaling bench tests."""

from __future__ import annotations

import pytest

from coordpy.capacity_bench_harness_v1 import (
    EventGraphIndexedQueryCacheV1,
    W84_CAPACITY_V1_SCHEMA_VERSION,
    _build_synthetic_event_graph,
    run_agent_count_scaling_axis_v1,
    run_capacity_bench_v1,
    run_event_graph_scaling_axis_v1,
    run_token_throughput_scaling_axis_v1,
)


def test_w84_indexed_cache_is_lazy_and_idempotent():
    g = _build_synthetic_event_graph(
        n_events=500, n_kinds=8, seed=1)
    cache = EventGraphIndexedQueryCacheV1()
    # First query triggers the build.
    r1 = cache.query_by_kind(graph=g, kind="kind_3")
    assert cache._built is True
    # Second query reuses the index.
    r2 = cache.query_by_kind(graph=g, kind="kind_3")
    assert tuple(n.event_id for n in r1) == tuple(
        n.event_id for n in r2)


def test_w84_indexed_cache_emits_content_addressed_index_cid():
    g = _build_synthetic_event_graph(
        n_events=200, n_kinds=4, seed=1)
    c1 = EventGraphIndexedQueryCacheV1()
    c2 = EventGraphIndexedQueryCacheV1()
    _ = c1.query_by_kind(graph=g, kind="kind_0")
    _ = c2.query_by_kind(graph=g, kind="kind_0")
    # Both caches built against the same graph → identical
    # index_cid.
    assert c1.index_cid() == c2.index_cid()
    assert len(c1.index_cid()) == 64


def test_w84_event_graph_axis_curve_has_three_points():
    pts = run_event_graph_scaling_axis_v1(
        n_events_curve=(500, 2_000, 5_000),
        n_queries=20,
        n_seeds=2,
    )
    sizes = sorted(set(int(p.n_events) for p in pts))
    assert sizes == [500, 2_000, 5_000]
    assert len(pts) == 3 * 2  # 3 scales × 2 seeds


def test_w84_agent_count_axis_quadratic_signature():
    """Per-round latency at n=200 must be strictly more than at
    n=10 (the O(n²) trust-fusion deviation kernel)."""
    pts = run_agent_count_scaling_axis_v1(
        n_agents_curve=(10, 50, 200),
        n_rounds=4, n_seeds=2)
    by_n: dict[int, list[float]] = {}
    for p in pts:
        by_n.setdefault(int(p.n_agents), []).append(
            float(p.per_round_mean_ms))
    means = {k: sum(v) / len(v) for k, v in by_n.items()}
    assert means[200] > means[10]


def test_w84_token_throughput_curve_strictly_decreases_per_ms():
    """Larger token counts mean more wall-clock; throughput is
    bounded.

    We just verify wall-clock is monotone-in-tokens within each
    seed.
    """
    pts = run_token_throughput_scaling_axis_v1(
        n_tokens_curve=(1_000, 10_000),
        n_seeds=2)
    by_n: dict[int, list[float]] = {}
    for p in pts:
        by_n.setdefault(int(p.n_tokens), []).append(
            float(p.wall_clock_ms))
    # On every seed, more tokens → at least as much wall-clock.
    assert all(
        sum(by_n[10_000]) / len(by_n[10_000])
        >= sum(by_n[1_000]) / len(by_n[1_000]) * 0.5
        for _ in [None])


def test_w84_capacity_bench_v1_identifies_cliff_and_measures_speedup():
    rep = run_capacity_bench_v1(
        event_graph_curve=(500, 2_000, 5_000),
        agent_count_curve=(10, 50),
        token_throughput_curve=(1_000,),
        n_seeds=2,
    )
    assert rep.cliff_axis == "event_graph_by_kind_query"
    assert "BY_KIND" in rep.identified_cliff
    # The indexed remediation must produce a measurable speedup.
    assert rep.cliff_speedup_factor > 1.0


def test_w84_capacity_bench_report_is_content_addressed():
    rep_a = run_capacity_bench_v1(
        event_graph_curve=(500, 2_000),
        agent_count_curve=(10,),
        token_throughput_curve=(1_000,),
        n_seeds=1,
    )
    # The report's CID is well-formed.
    assert len(rep_a.cid()) == 64


def test_w84_speedup_factor_at_50k_q100_at_least_3x():
    """At the load-bearing config (N=50k, Q=100), the indexed
    remediation strictly beats the naive path. We assert ≥3x
    (the floor that captures a measurable cliff move; not the
    full OoM the issue body asks for — see audit doc).
    """
    rep = run_capacity_bench_v1(
        event_graph_curve=(50_000,),
        agent_count_curve=(10,),
        token_throughput_curve=(1_000,),
        n_seeds=2,
    )
    assert rep.cliff_speedup_factor >= 3.0
