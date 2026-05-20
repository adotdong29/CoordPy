"""Tests for ``coordpy.event_graph_garbage_collection_v1``."""

from __future__ import annotations

import pytest

from coordpy.event_graph_garbage_collection_v1 import (
    GCBenchReportV1,
    GCEventV1,
    GCPolicyV1,
    JSONLPersistentStoreV1,
    RetentionClass,
    mark_reachable_v1,
    restore_event_from_grace_v1,
    run_100k_event_gc_bench_v1,
    run_gc_pass_v1,
    verify_chain_across_gc_v1,
)
from coordpy.event_sourced_memory_graph_v1 import (
    EventGraphV1,
    build_event_node_v1,
)


def _build_linear_chain(n: int = 10, kind: str = "chain_link"):
    g = EventGraphV1.empty()
    chain_ids = []
    parent = g.root_event_id
    for i in range(n):
        eid = f"chain_{i:03d}"
        ev = build_event_node_v1(
            event_id=eid, kind=kind,
            payload_bytes=f"p_{i}".encode(),
            parent_event_ids=(parent,),
            branch_label="main",
            timestamp_ns=1_000 + i)
        g = g.with_event(ev)
        chain_ids.append(eid)
        parent = eid
    return g, chain_ids


def test_gc_policy_is_content_addressed():
    p1 = GCPolicyV1()
    p2 = GCPolicyV1()
    assert p1.cid() == p2.cid()
    p3 = GCPolicyV1(min_age_ns=100)
    assert p1.cid() != p3.cid()


def test_mark_reachable_walks_full_ancestry():
    g, chain_ids = _build_linear_chain(n=5)
    policy = GCPolicyV1(min_age_ns=0)
    reachable = mark_reachable_v1(
        g, root_event_ids=[chain_ids[-1]], policy=policy)
    # Walking from the tip should include genesis + every link.
    assert g.root_event_id in reachable
    for eid in chain_ids:
        assert eid in reachable


def test_critical_events_always_marked_reachable():
    g = EventGraphV1.empty()
    parent = g.root_event_id
    crit = build_event_node_v1(
        event_id="anchor_0", kind="commit_anchor",
        payload_bytes=b"anchor",
        parent_event_ids=(parent,),
        branch_label="anchors", timestamp_ns=1)
    g = g.with_event(crit)
    policy = GCPolicyV1(min_age_ns=0)
    # Declare NO roots — the anchor should still survive due to
    # its critical kind.
    reachable = mark_reachable_v1(
        g, root_event_ids=[], policy=policy)
    assert "anchor_0" in reachable


def test_mark_and_sweep_purges_ephemeral_off_chain():
    g, chain_ids = _build_linear_chain(n=5)
    # Add an off-chain ephemeral event.
    eph = build_event_node_v1(
        event_id="eph_0", kind="retry_attempt",
        payload_bytes=b"eph",
        parent_event_ids=(chain_ids[2],),
        branch_label="retries", timestamp_ns=2_000)
    g = g.with_event(eph)

    policy = GCPolicyV1(
        min_age_ns=0, grace_window_ns=1_000_000_000)
    result = run_gc_pass_v1(
        g, policy=policy,
        root_event_ids=[chain_ids[-1]],
        now_ns=10_000)
    assert "eph_0" not in result.graph_after.nodes
    # Chain links must still be there.
    for cid in chain_ids:
        assert cid in result.graph_after.nodes


def test_durable_event_enters_grace_buffer_first():
    g = EventGraphV1.empty()
    parent = g.root_event_id
    durable = build_event_node_v1(
        event_id="dur_0", kind="random_kind",
        payload_bytes=b"dur",
        parent_event_ids=(parent,),
        branch_label="other", timestamp_ns=1_000)
    g = g.with_event(durable)
    policy = GCPolicyV1(
        min_age_ns=0, grace_window_ns=10_000)
    result = run_gc_pass_v1(
        g, policy=policy, root_event_ids=[],
        now_ns=5_000)
    # Durable event was unreachable but goes into grace buffer
    # first; it's removed from the live graph.
    assert "dur_0" not in result.graph_after.nodes
    assert "dur_0" in result.grace_buffer_after


def test_restore_from_grace_works():
    g = EventGraphV1.empty()
    parent = g.root_event_id
    durable = build_event_node_v1(
        event_id="dur_0", kind="random_kind",
        payload_bytes=b"dur",
        parent_event_ids=(parent,),
        branch_label="other", timestamp_ns=1_000)
    g = g.with_event(durable)
    policy = GCPolicyV1(
        min_age_ns=0, grace_window_ns=10_000)
    result = run_gc_pass_v1(
        g, policy=policy, root_event_ids=[],
        now_ns=5_000)
    restored, new_grace = restore_event_from_grace_v1(
        result.graph_after, result.grace_buffer_after,
        event_id="dur_0")
    assert "dur_0" in restored.nodes
    assert "dur_0" not in new_grace


def test_grace_window_expiry_hard_purges():
    g = EventGraphV1.empty()
    parent = g.root_event_id
    durable = build_event_node_v1(
        event_id="dur_0", kind="random_kind",
        payload_bytes=b"dur",
        parent_event_ids=(parent,),
        branch_label="other", timestamp_ns=1_000)
    g = g.with_event(durable)
    policy = GCPolicyV1(
        min_age_ns=0, grace_window_ns=10_000)
    # First pass: dur_0 enters grace.
    pass1 = run_gc_pass_v1(
        g, policy=policy, root_event_ids=[],
        now_ns=2_000)
    assert "dur_0" in pass1.grace_buffer_after

    # Second pass at now_ns past grace_window: dur_0 is
    # hard-purged from the grace buffer.
    pass2 = run_gc_pass_v1(
        pass1.graph_after, policy=policy,
        root_event_ids=[],
        now_ns=pass1.gc_event.gc_timestamp_ns
              + policy.grace_window_ns + 1,
        grace_buffer=pass1.grace_buffer_after)
    assert "dur_0" not in pass2.grace_buffer_after


def test_chain_verifies_across_gc():
    g, chain_ids = _build_linear_chain(n=10)
    # Add off-chain ephemerals.
    for i in range(20):
        eph = build_event_node_v1(
            event_id=f"eph_{i:03d}", kind="retry_attempt",
            payload_bytes=b"e",
            parent_event_ids=(chain_ids[i % 10],),
            branch_label="retries", timestamp_ns=10_000 + i)
        g = g.with_event(eph)

    policy = GCPolicyV1(min_age_ns=0)
    result = run_gc_pass_v1(
        g, policy=policy,
        root_event_ids=[chain_ids[-1]],
        now_ns=20_000)

    verdict = verify_chain_across_gc_v1(
        graph=result.graph_after,
        gc_events=(result.gc_event,),
        declared_root_event_ids=[chain_ids[-1]])
    assert verdict["chain_verifies"] is True


def test_dangling_parent_makes_chain_not_verify():
    """Inject a fake event whose parent is not in the graph
    AND not GC'd → chain_verifies must be False.
    """
    g, _ = _build_linear_chain(n=3)
    fake_ev = build_event_node_v1(
        event_id="orphan", kind="random",
        payload_bytes=b"orphan",
        parent_event_ids=("nonexistent_event",),
        branch_label="orphans", timestamp_ns=999_999)
    # Construct a fake graph that includes the orphan.
    new_nodes = dict(g.nodes)
    new_nodes["orphan"] = fake_ev
    bad_graph = EventGraphV1(
        schema=g.schema, nodes=new_nodes,
        branch_tips=g.branch_tips, root_event_id=g.root_event_id)
    verdict = verify_chain_across_gc_v1(
        graph=bad_graph, gc_events=(),
        declared_root_event_ids=[])
    assert verdict["chain_verifies"] is False
    assert verdict["dangling_parent_count"] >= 1


def test_gc_event_is_content_addressed():
    g, chain_ids = _build_linear_chain(n=3)
    policy = GCPolicyV1(min_age_ns=0)
    r1 = run_gc_pass_v1(
        g, policy=policy, root_event_ids=[chain_ids[-1]],
        now_ns=100)
    r2 = run_gc_pass_v1(
        g, policy=policy, root_event_ids=[chain_ids[-1]],
        now_ns=100)
    assert r1.gc_event.cid() == r2.gc_event.cid()


def test_jsonl_persistent_store_round_trip(tmp_path):
    path = str(tmp_path / "store.jsonl")
    g, chain_ids = _build_linear_chain(n=3)
    store = JSONLPersistentStoreV1(path=path)
    for cid in chain_ids:
        store.append_event(g.nodes[cid])
    # Run a GC pass + write the GC event.
    policy = GCPolicyV1(min_age_ns=0)
    result = run_gc_pass_v1(
        g, policy=policy, root_event_ids=[chain_ids[-1]],
        now_ns=100)
    store.append_gc_event(result.gc_event)
    events_back, gcs_back = store.read_back()
    assert len(events_back) == 3
    assert len(gcs_back) == 1


def test_100k_event_bench_meets_dod():
    rep = run_100k_event_gc_bench_v1(n_events=100_000)
    # DoD: GC reduces memory by ≥ 80% while preserving the
    # load-bearing chain end-to-end verifiable.
    assert rep.memory_reduction_fraction >= 0.80
    assert rep.chain_verifies_after_gc is True
    assert rep.grace_restore_works is True
    assert rep.persistent_store_round_trip is True
    # Sanity: at least some events were purged.
    assert rep.n_events_purged > 0
    # The genesis + critical anchors + chain links survived.
    assert rep.n_events_after_gc >= rep.n_events_critical


def test_bench_report_cid_is_deterministic():
    r1 = run_100k_event_gc_bench_v1(n_events=10_000, seed=86_045)
    r2 = run_100k_event_gc_bench_v1(n_events=10_000, seed=86_045)
    assert r1.report_cid == r2.report_cid


def test_critical_kind_never_purged():
    g = EventGraphV1.empty()
    parent = g.root_event_id
    crit = build_event_node_v1(
        event_id="commit_anchor_0", kind="commit_anchor",
        payload_bytes=b"a",
        parent_event_ids=(parent,),
        branch_label="anchors", timestamp_ns=1)
    g = g.with_event(crit)
    policy = GCPolicyV1(min_age_ns=0)
    # Declare NOT critical's id as root; critical kind alone
    # must save it.
    result = run_gc_pass_v1(
        g, policy=policy, root_event_ids=[],
        now_ns=100_000)
    assert "commit_anchor_0" in result.graph_after.nodes


def test_genesis_is_always_retained():
    g, _ = _build_linear_chain(n=3)
    policy = GCPolicyV1(min_age_ns=0, retain_all_genesis=True)
    # No roots declared.
    result = run_gc_pass_v1(
        g, policy=policy, root_event_ids=[],
        now_ns=100_000)
    assert g.root_event_id in result.graph_after.nodes


def test_min_age_blocks_premature_gc():
    g, chain_ids = _build_linear_chain(n=3)
    # Add an unreachable event with a recent timestamp.
    eph = build_event_node_v1(
        event_id="eph_0", kind="retry_attempt",
        payload_bytes=b"x",
        parent_event_ids=(chain_ids[0],),
        branch_label="retries", timestamp_ns=99_999_999_999)
    g = g.with_event(eph)
    # Policy: min_age_ns is huge; eph_0 is too young to GC.
    policy = GCPolicyV1(
        min_age_ns=10_000_000_000_000)
    result = run_gc_pass_v1(
        g, policy=policy, root_event_ids=[chain_ids[-1]],
        now_ns=100_000_000_000)  # < min_age_ns past eph's ts
    # eph_0 was NOT eligible for sweep; still in graph.
    assert "eph_0" in result.graph_after.nodes
