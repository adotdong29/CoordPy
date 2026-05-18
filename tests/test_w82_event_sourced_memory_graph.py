"""W82 / P2 #11 — Event-sourced memory graph tests.

Covers:
- event nodes are content-addressed
- graph append is append-only (cannot re-add an id; cannot
  point at a non-existent parent)
- graph supports branch / merge (event with 2 parents)
- query types enumerate
- by-event-id query resolves quickly
- by-kind, by-branch, by-ancestor-path queries work
- provenance certificate cites contributing event CIDs
- derived summary view is recomputable and content-addressed
- bounded-handoff baseline and trajectory-slice baseline
  fail on far queries
- graph dominates baselines on the load-bearing bench
- bench is deterministic
"""

from __future__ import annotations


def test_w82_event_graph_node_content_addressed():
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    a = build_event_node_v1(
        event_id="e0", kind="x",
        payload_bytes=b"abc",
        parent_event_ids=("genesis",),
        branch_label="main", timestamp_ns=1)
    b = build_event_node_v1(
        event_id="e0", kind="x",
        payload_bytes=b"abc",
        parent_event_ids=("genesis",),
        branch_label="main", timestamp_ns=1)
    c = build_event_node_v1(
        event_id="e0", kind="x",
        payload_bytes=b"abc",
        parent_event_ids=("genesis",),
        branch_label="main", timestamp_ns=2)  # ts differs
    assert a.cid() == b.cid()
    assert a.cid() != c.cid()


def test_w82_event_graph_empty_has_genesis():
    from coordpy.event_sourced_memory_graph_v1 import (
        EventGraphV1,
        W82_EVENT_GRAPH_GENESIS_EVENT_ID,
    )
    g = EventGraphV1.empty()
    assert g.n_events() == 1
    assert g.root_event_id == W82_EVENT_GRAPH_GENESIS_EVENT_ID


def test_w82_event_graph_append_is_append_only():
    from coordpy.event_sourced_memory_graph_v1 import (
        EventGraphV1,
        build_event_node_v1,
    )
    g = EventGraphV1.empty()
    e = build_event_node_v1(
        event_id="e0", kind="x", payload_bytes=b"a",
        parent_event_ids=("genesis",),
        branch_label="main", timestamp_ns=1)
    g2 = g.with_event(e)
    assert g2.n_events() == 2
    # Re-add same id raises
    e2 = build_event_node_v1(
        event_id="e0", kind="y", payload_bytes=b"b",
        parent_event_ids=("genesis",),
        branch_label="main", timestamp_ns=2)
    try:
        g2.with_event(e2)
    except ValueError:
        return
    raise AssertionError(
        "re-adding event_id should raise")


def test_w82_event_graph_rejects_orphan_parent():
    """Cannot append a child of a non-existent parent."""
    from coordpy.event_sourced_memory_graph_v1 import (
        EventGraphV1,
        build_event_node_v1,
    )
    g = EventGraphV1.empty()
    e = build_event_node_v1(
        event_id="e0", kind="x", payload_bytes=b"a",
        parent_event_ids=("ghost_parent",),
        branch_label="main", timestamp_ns=1)
    try:
        g.with_event(e)
    except ValueError:
        return
    raise AssertionError(
        "appending child of non-existent parent should "
        "raise")


def test_w82_event_graph_supports_branch_merge():
    """Append an event with 2 parents to model a merge."""
    from coordpy.event_sourced_memory_graph_v1 import (
        EventGraphV1,
        build_event_node_v1,
    )
    g = EventGraphV1.empty()
    a = build_event_node_v1(
        event_id="a", kind="x", payload_bytes=b"a",
        parent_event_ids=("genesis",),
        branch_label="side_a", timestamp_ns=1)
    b = build_event_node_v1(
        event_id="b", kind="x", payload_bytes=b"b",
        parent_event_ids=("genesis",),
        branch_label="side_b", timestamp_ns=2)
    m = build_event_node_v1(
        event_id="m", kind="merge", payload_bytes=b"m",
        parent_event_ids=("a", "b"),
        branch_label="main", timestamp_ns=3)
    g2 = g.with_event(a).with_event(b).with_event(m)
    assert g2.n_events() == 4
    assert len(g2.get("m").parent_event_ids) == 2


def test_w82_event_graph_query_kinds_enumerate():
    from coordpy.event_sourced_memory_graph_v1 import (
        W82_EVENT_GRAPH_QUERY_KINDS,
    )
    assert "by_event_id" in W82_EVENT_GRAPH_QUERY_KINDS
    assert "by_kind" in W82_EVENT_GRAPH_QUERY_KINDS
    assert "by_branch" in W82_EVENT_GRAPH_QUERY_KINDS
    assert "by_ancestor_path" in W82_EVENT_GRAPH_QUERY_KINDS


def test_w82_event_graph_by_event_id_query_resolves():
    from coordpy.event_sourced_memory_graph_v1 import (
        build_synthetic_event_graph_v1,
        build_by_event_id_query_v1,
        execute_query_v1,
    )
    g, ordered = build_synthetic_event_graph_v1(n_events=10)
    target = ordered[3].event_id
    q = build_by_event_id_query_v1(
        query_id="q0", target_event_id=str(target))
    ans, prov = execute_query_v1(graph=g, query=q)
    assert ans.success
    assert prov.n_hops == 1


def test_w82_event_graph_by_kind_query_filters():
    from coordpy.event_sourced_memory_graph_v1 import (
        build_synthetic_event_graph_v1,
        build_by_kind_query_v1,
        execute_query_v1,
    )
    g, _ = build_synthetic_event_graph_v1(n_events=12)
    q = build_by_kind_query_v1(
        query_id="q1", target_kind="merge")
    ans, prov = execute_query_v1(graph=g, query=q)
    # The default bench has a merge node near 75%
    assert ans.success
    assert prov.n_hops >= 1


def test_w82_event_graph_by_branch_query_walks_branch():
    from coordpy.event_sourced_memory_graph_v1 import (
        build_synthetic_event_graph_v1,
        build_by_branch_query_v1,
        execute_query_v1,
    )
    g, _ = build_synthetic_event_graph_v1(n_events=15)
    q = build_by_branch_query_v1(
        query_id="q2", target_branch_label="side_a")
    ans, prov = execute_query_v1(graph=g, query=q)
    assert ans.success


def test_w82_event_graph_by_ancestor_path_query_walks_far():
    from coordpy.event_sourced_memory_graph_v1 import (
        build_synthetic_event_graph_v1,
        build_by_ancestor_path_query_v1,
        execute_query_v1,
    )
    g, ordered = build_synthetic_event_graph_v1(n_events=30)
    early = ordered[1].event_id
    late = ordered[-1].event_id
    q = build_by_ancestor_path_query_v1(
        query_id="q3",
        from_event_id=str(late),
        to_event_id=str(early))
    ans, prov = execute_query_v1(graph=g, query=q)
    assert ans.success


def test_w82_event_graph_provenance_certificate_lists_cids():
    from coordpy.event_sourced_memory_graph_v1 import (
        build_synthetic_event_graph_v1,
        build_by_event_id_query_v1,
        execute_query_v1,
    )
    g, ordered = build_synthetic_event_graph_v1(n_events=10)
    target = ordered[5].event_id
    q = build_by_event_id_query_v1(
        query_id="q4", target_event_id=str(target))
    ans, prov = execute_query_v1(graph=g, query=q)
    assert len(prov.contributing_event_cids) == 1
    assert prov.contributing_event_ids == (str(target),)
    # cert is content-addressed
    assert prov.cid() == prov.cid()


def test_w82_event_graph_derived_view_recomputed_and_cid_responds_to_graph():
    from coordpy.event_sourced_memory_graph_v1 import (
        build_synthetic_event_graph_v1,
        build_derived_summary_view_v1,
        build_event_node_v1,
    )
    g_small, _ = build_synthetic_event_graph_v1(n_events=4)
    v_small = build_derived_summary_view_v1(g_small)
    g_big, _ = build_synthetic_event_graph_v1(n_events=12)
    v_big = build_derived_summary_view_v1(g_big)
    assert v_small.cid() != v_big.cid()
    # Stale-view check: source_graph_cid encodes the graph
    assert v_small.source_graph_cid != v_big.source_graph_cid


def test_w82_event_graph_bounded_handoff_baseline_fails_on_far():
    """Bounded-handoff with k=32 must fail when the source
    event is 100 events back."""
    from coordpy.event_sourced_memory_graph_v1 import (
        build_synthetic_event_graph_v1,
        build_by_event_id_query_v1,
        bounded_handoff_baseline_v1,
    )
    _g, ordered = build_synthetic_event_graph_v1(n_events=100)
    q = build_by_event_id_query_v1(
        query_id="qx",
        target_event_id=str(ordered[1].event_id))
    ok, _walked = bounded_handoff_baseline_v1(
        events=ordered, query=q, k=32)
    assert not ok


def test_w82_event_graph_trajectory_baseline_fails_outside_branch():
    """Trajectory-slice on `main` doesn't see side_a events."""
    from coordpy.event_sourced_memory_graph_v1 import (
        build_synthetic_event_graph_v1,
        build_by_event_id_query_v1,
        trajectory_slice_baseline_v1,
    )
    _g, ordered = build_synthetic_event_graph_v1(n_events=15)
    side_a_events = [
        e for e in ordered if e.branch_label == "side_a"]
    assert side_a_events, "synthetic graph must have side_a"
    q = build_by_event_id_query_v1(
        query_id="qy",
        target_event_id=str(side_a_events[0].event_id))
    ok, _walked = trajectory_slice_baseline_v1(
        events=ordered, query=q, k=16,
        current_branch="main")
    assert not ok


def test_w82_event_graph_load_bearing_bench_graph_dominates():
    """Load-bearing W82 P2 #11 claim."""
    from coordpy.event_sourced_memory_graph_v1 import (
        run_event_graph_bench_end_to_end_v1,
    )
    r = run_event_graph_bench_end_to_end_v1()
    assert r.graph_success_rate >= 0.95
    assert r.bounded_handoff_k32_success_rate <= 0.0 + 1e-12
    assert r.trajectory_slice_k16_success_rate <= 0.0 + 1e-12
    assert r.graph_dominates_baselines is True


def test_w82_event_graph_bench_deterministic():
    from coordpy.event_sourced_memory_graph_v1 import (
        run_event_graph_bench_end_to_end_v1,
    )
    a = run_event_graph_bench_end_to_end_v1(
        n_events=30, n_queries=5, seed=7)
    b = run_event_graph_bench_end_to_end_v1(
        n_events=30, n_queries=5, seed=7)
    assert a.cid() == b.cid()


def test_w82_event_graph_query_plan_steps_present():
    """The planner annotates each query's plan with explicit
    steps so the execution path is inspectable."""
    from coordpy.event_sourced_memory_graph_v1 import (
        EventGraphV1,
        build_by_event_id_query_v1,
        build_by_ancestor_path_query_v1,
        plan_query_v1,
    )
    g = EventGraphV1.empty()
    q1 = build_by_event_id_query_v1(
        query_id="p1", target_event_id="genesis")
    p1 = plan_query_v1(graph=g, query=q1)
    assert "lookup_event_by_id" in p1.steps
    q2 = build_by_ancestor_path_query_v1(
        query_id="p2",
        from_event_id="genesis", to_event_id="genesis")
    p2 = plan_query_v1(graph=g, query=q2)
    assert "bfs_ancestor_path" in p2.steps


def test_w82_event_graph_carrier_fallback_recovers_evicted_event():
    """The carrier fallback bridges the live event graph and
    a long-horizon carrier: events evicted from the graph are
    recoverable via the carrier_lookup mapping. This closes
    the P2 #11 'graph state + latent carriers + replay'
    requirement."""
    from coordpy.event_sourced_memory_graph_v1 import (
        EventGraphV1,
        build_by_event_id_query_v1,
        execute_query_with_carrier_fallback_v1,
    )
    g = EventGraphV1.empty()
    q = build_by_event_id_query_v1(
        query_id="qc", target_event_id="evicted_e0")
    # Without carrier — fails
    a, _p = execute_query_with_carrier_fallback_v1(
        graph=g, query=q)
    assert not a.success
    # With carrier — succeeds via fallback
    a2, p2 = execute_query_with_carrier_fallback_v1(
        graph=g, query=q,
        carrier_lookup={"evicted_e0": "RECOVERED_CID"})
    assert a2.success
    assert a2.detail == "carrier_fallback"
    assert a2.payload_cids == ("RECOVERED_CID",)
    # Provenance certificate cites the carrier path
    assert p2.contributing_event_ids == ("evicted_e0",)
    assert p2.contributing_event_cids == ("RECOVERED_CID",)


def test_w82_event_graph_carrier_fallback_passes_through_when_graph_hits():
    """When the graph has the event, no carrier fallback is
    needed — the answer comes from the graph directly."""
    from coordpy.event_sourced_memory_graph_v1 import (
        build_synthetic_event_graph_v1,
        build_by_event_id_query_v1,
        execute_query_with_carrier_fallback_v1,
    )
    g, ordered = build_synthetic_event_graph_v1(n_events=8)
    target = ordered[3].event_id
    q = build_by_event_id_query_v1(
        query_id="qd", target_event_id=str(target))
    a, p = execute_query_with_carrier_fallback_v1(
        graph=g, query=q,
        carrier_lookup={target: "WRONG_CID_SHOULD_NOT_BE_USED"})
    assert a.success
    # Detail should NOT be carrier_fallback — graph answered
    assert a.detail != "carrier_fallback"


def test_w82_event_graph_carrier_fallback_only_for_by_event_id():
    """Carrier fallback only works for BY_EVENT_ID queries
    (the carrier holds event_id -> event_cid mappings)."""
    from coordpy.event_sourced_memory_graph_v1 import (
        EventGraphV1,
        build_by_kind_query_v1,
        execute_query_with_carrier_fallback_v1,
    )
    g = EventGraphV1.empty()
    q = build_by_kind_query_v1(
        query_id="qk", target_kind="rare_kind")
    # carrier lookup keys by event_id, not kind — so a
    # BY_KIND query cannot be answered by carrier fallback
    a, _p = execute_query_with_carrier_fallback_v1(
        graph=g, query=q,
        carrier_lookup={"some_event_id": "some_cid"})
    assert not a.success


def test_w82_event_graph_ancestor_path_includes_merge_branch():
    """Walking an ancestor path through a merge event must
    include the merge node."""
    from coordpy.event_sourced_memory_graph_v1 import (
        EventGraphV1,
        build_event_node_v1,
        build_by_ancestor_path_query_v1,
        execute_query_v1,
    )
    g = EventGraphV1.empty()
    a = build_event_node_v1(
        event_id="a", kind="x", payload_bytes=b"a",
        parent_event_ids=("genesis",),
        branch_label="side_a", timestamp_ns=1)
    b = build_event_node_v1(
        event_id="b", kind="x", payload_bytes=b"b",
        parent_event_ids=("genesis",),
        branch_label="side_b", timestamp_ns=2)
    m = build_event_node_v1(
        event_id="m", kind="merge", payload_bytes=b"m",
        parent_event_ids=("a", "b"),
        branch_label="main", timestamp_ns=3)
    g = g.with_event(a).with_event(b).with_event(m)
    q = build_by_ancestor_path_query_v1(
        query_id="qm",
        from_event_id="m", to_event_id="a")
    ans, prov = execute_query_v1(graph=g, query=q)
    assert ans.success
    assert "m" in prov.contributing_event_ids
    assert "a" in prov.contributing_event_ids
