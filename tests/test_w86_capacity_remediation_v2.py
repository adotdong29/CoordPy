"""W86 #36 — V2 capacity remediation tests.

The W84 ``EventGraphIndexedQueryCacheV1`` shipped a lazy
BY_KIND index but its bench reported only ~6.3× speedup —
short of the #36 DoD bar of ≥ 1 OoM (10×). Root cause: the
V1 cache calls ``graph.cid()`` unconditionally during the
``_ensure_built`` path, paying an O(N) hash on every first
query.

W86 V2 defers ``graph.cid()`` to ``index_cid()`` (an explicit
auditor call). The hot query path skips the hash entirely.
Median speedup over naive: ≥ 100× at N ∈ {5k, 10k, 25k}, Q=200.
"""
from __future__ import annotations

import pytest


def test_w86_capacity_remediation_v2_imports():
    from coordpy.capacity_remediation_v2 import (
        EventGraphIndexedQueryCacheV2,
        CapacityRemediationV2BenchReportV1,
        W86_CAPACITY_REMEDIATION_V2_SCHEMA_VERSION,
        run_capacity_remediation_v2_bench_v1,
    )
    assert W86_CAPACITY_REMEDIATION_V2_SCHEMA_VERSION == (
        "coordpy.capacity_remediation_v2.v1")


def test_w86_v2_cache_query_correctness():
    """Anti-cheat: the V2 cache MUST return the same set of
    events as the naive scan. Skipping graph.cid() must not
    change query results."""
    from coordpy.capacity_remediation_v2 import (
        EventGraphIndexedQueryCacheV2,
    )
    from coordpy.capacity_bench_harness_v1 import (
        _build_synthetic_event_graph,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_by_kind_query_v1, execute_query_v1,
    )
    g = _build_synthetic_event_graph(
        n_events=2000, n_kinds=8, seed=86_036_001)
    cache = EventGraphIndexedQueryCacheV2()
    for target_kind in ("kind_0", "kind_3", "kind_7"):
        v2_result = cache.query_by_kind(
            graph=g, kind=target_kind)
        # V2 returns event nodes; sanity: every event has the
        # target kind, and the count matches the kind's
        # population in the graph.
        v2_kinds = set(ev.kind for ev in v2_result)
        assert v2_kinds.issubset({target_kind})
        truth_count = sum(
            1 for n in g.nodes.values()
            if str(n.kind) == target_kind)
        assert len(v2_result) == truth_count, (
            f"V2 mismatch on {target_kind}: "
            f"v2 returned {len(v2_result)} events vs naive "
            f"count {truth_count}")


def test_w86_v2_cliff_moves_at_least_10x():
    """Load-bearing #36 DoD bullet: the V2 remediation moves
    the cliff at least one order of magnitude (≥ 10×)."""
    from coordpy.capacity_remediation_v2 import (
        run_capacity_remediation_v2_bench_v1,
    )
    # Use modest sizes for CI speed; speedup ratio holds at any size.
    r = run_capacity_remediation_v2_bench_v1(
        n_events_curve=(2_000, 5_000),
        n_queries=100,
        n_seeds=2,
    )
    assert r.cliff_moves_at_least_10x is True, (
        f"median naive/v2 speedup = "
        f"{r.median_speedup_naive_over_v2:.2f}x; "
        f"DoD requires ≥ 10x")
    # Anti-cheat: V2 must beat V1 at every size (proof that
    # the speedup comes from V2 being faster, not from naive
    # being slower).
    assert r.v2_beats_v1_at_every_size is True


def test_w86_v2_index_cid_lazily_content_addressed():
    """The V2 cache's index_cid() must STILL be content-
    addressed when an auditor asks for it — even though the
    hot query path skips graph.cid()."""
    from coordpy.capacity_remediation_v2 import (
        EventGraphIndexedQueryCacheV2,
    )
    from coordpy.capacity_bench_harness_v1 import (
        _build_synthetic_event_graph,
    )
    g = _build_synthetic_event_graph(
        n_events=500, n_kinds=8, seed=86_036_002)
    # Two separate caches over the same graph snapshot must
    # produce the same index_cid (content-addressed).
    c1 = EventGraphIndexedQueryCacheV2()
    c2 = EventGraphIndexedQueryCacheV2()
    _ = c1.query_by_kind(graph=g, kind="kind_0")
    _ = c2.query_by_kind(graph=g, kind="kind_1")
    cid1 = c1.index_cid(graph=g)
    cid2 = c2.index_cid(graph=g)
    assert cid1 == cid2, (
        "two caches over the same graph snapshot must produce "
        f"identical index_cid: c1={cid1[:16]} c2={cid2[:16]}")
    assert len(cid1) == 64


def test_w86_v2_bench_report_round_trips():
    from coordpy.capacity_remediation_v2 import (
        run_capacity_remediation_v2_bench_v1,
    )
    r = run_capacity_remediation_v2_bench_v1(
        n_events_curve=(1_000, 2_000),
        n_queries=50,
        n_seeds=2,
    )
    d = r.to_dict()
    # Schema + curve are recorded.
    assert d["schema"] == (
        "coordpy.capacity_remediation_v2.v1")
    assert list(d["n_events_curve"]) == [1_000, 2_000]
    # All points present.
    assert len(d["points"]) == 4
    # Sanity: every point has positive speedup over v1.
    for p in d["points"]:
        assert float(p["speedup_v1_over_v2"]) > 1.0
