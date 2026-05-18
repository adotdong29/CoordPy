"""W82 / P2 #16 — Distributed substrate coordination tests.

Covers:
- simulated host has runtime signature + event graph
- migration envelope is content-addressed + merkle-rooted
- envelope verify catches corruption
- envelope apply is idempotent (replaying gives same state)
- partition event is content-addressed
- heal+sync produces eventual consistency
- post-heal merkle roots match on all hosts (load-bearing)
- cross-runtime envelope works
- consistency verdicts enumerate
- bench is deterministic
"""

from __future__ import annotations


def test_w82_distributed_consistency_verdicts_enumerate():
    from coordpy.distributed_substrate_coordination_v1 import (
        W82_DISTRIBUTED_CONSISTENCY_VERDICTS,
        ConsistencyVerdict,
    )
    assert "exact" in W82_DISTRIBUTED_CONSISTENCY_VERDICTS
    assert "eventual" in W82_DISTRIBUTED_CONSISTENCY_VERDICTS
    assert "approximate" in W82_DISTRIBUTED_CONSISTENCY_VERDICTS
    assert "best_effort" in W82_DISTRIBUTED_CONSISTENCY_VERDICTS
    assert ConsistencyVerdict.EVENTUAL.value == "eventual"


def test_w82_distributed_host_content_addressed():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
    )
    a = build_simulated_host_v1(host_id="h1")
    b = build_simulated_host_v1(host_id="h1")
    c = build_simulated_host_v1(host_id="h2")
    assert a.cid() == b.cid()
    assert a.cid() != c.cid()
    assert a.merkle_root_cid() == b.merkle_root_cid()


def test_w82_distributed_host_append_event_is_pure():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    h = build_simulated_host_v1(host_id="h")
    e = build_event_node_v1(
        event_id="e0", kind="x", payload_bytes=b"a",
        parent_event_ids=(h.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    h2 = h.append_event(e)
    assert h.event_graph.n_events() == 1  # unchanged
    assert h2.event_graph.n_events() == 2
    assert h.cid() != h2.cid()


def test_w82_distributed_envelope_content_addressed():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    e1 = build_event_node_v1(
        event_id="e1", kind="x", payload_bytes=b"a",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    src = src.append_event(e1)
    env_a = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("e1",))
    env_b = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("e1",))
    assert env_a.cid() == env_b.cid()
    assert env_a.merkle_root_cid == env_b.merkle_root_cid


def test_w82_distributed_envelope_verify_clean_envelope_ok():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
        verify_migration_envelope_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    e1 = build_event_node_v1(
        event_id="e1", kind="x", payload_bytes=b"a",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    src = src.append_event(e1)
    env = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("e1",))
    assert verify_migration_envelope_v1(env) == (
        IntegrityVerdict.OK.value)


def test_w82_distributed_apply_envelope_grows_target_graph():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
        apply_migration_envelope_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    e1 = build_event_node_v1(
        event_id="e1", kind="x", payload_bytes=b"a",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    src = src.append_event(e1)
    env = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("e1",))
    tgt2, verdict = apply_migration_envelope_v1(
        envelope=env, target=tgt)
    assert verdict == IntegrityVerdict.OK.value
    assert tgt2.event_graph.n_events() == 2
    assert "e1" in tgt2.event_graph.nodes


def test_w82_distributed_apply_envelope_idempotent():
    """Replaying an envelope must not double-append events."""
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
        apply_migration_envelope_v1,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    e1 = build_event_node_v1(
        event_id="e1", kind="x", payload_bytes=b"a",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    src = src.append_event(e1)
    env = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("e1",))
    tgt2, _v1 = apply_migration_envelope_v1(
        envelope=env, target=tgt)
    tgt3, _v2 = apply_migration_envelope_v1(
        envelope=env, target=tgt2)
    assert tgt2.event_graph.n_events() == 2
    assert tgt3.event_graph.n_events() == 2  # idempotent


def test_w82_distributed_corrupt_envelope_detected():
    """If the envelope's events have been tampered with, the
    merkle re-hash catches it."""
    import dataclasses
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
        verify_migration_envelope_v1,
        MigrationEnvelopeV1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    e1 = build_event_node_v1(
        event_id="e1", kind="x", payload_bytes=b"a",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    src = src.append_event(e1)
    env = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("e1",))
    # Tamper: replace the events tuple with a different event
    # but keep the declared event_cids the same.
    e2 = build_event_node_v1(
        event_id="e1", kind="x", payload_bytes=b"DIFFERENT",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    bad_env = dataclasses.replace(env, events=(e2,))
    v = verify_migration_envelope_v1(bad_env)
    assert v == IntegrityVerdict.CORRUPT.value


def test_w82_distributed_envelope_rejects_extra_unauthenticated_events():
    import dataclasses
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
        verify_migration_envelope_v1,
        apply_migration_envelope_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    e1 = build_event_node_v1(
        event_id="e1", kind="x", payload_bytes=b"a",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    src = src.append_event(e1)
    extra = build_event_node_v1(
        event_id="e2", kind="x", payload_bytes=b"b",
        parent_event_ids=("e1",),
        branch_label="main", timestamp_ns=2)
    env = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("e1",))
    bad_env = dataclasses.replace(
        env, events=(e1, extra), n_events=2)
    assert verify_migration_envelope_v1(bad_env) == (
        IntegrityVerdict.CORRUPT.value)
    tgt2, verdict = apply_migration_envelope_v1(
        envelope=bad_env, target=tgt)
    assert verdict == IntegrityVerdict.CORRUPT.value
    assert "e1" not in tgt2.event_graph.nodes
    assert "e2" not in tgt2.event_graph.nodes


def test_w82_distributed_extra_events_rejected():
    import dataclasses
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
        apply_migration_envelope_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    e1 = build_event_node_v1(
        event_id="e1", kind="x", payload_bytes=b"a",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    src = src.append_event(e1)
    env = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("e1",))
    e_extra = build_event_node_v1(
        event_id="e_extra", kind="x",
        payload_bytes=b"EXTRA",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=2)
    bad_env = dataclasses.replace(
        env, events=(e1, e_extra), n_events=2)
    tgt2, verdict = apply_migration_envelope_v1(
        envelope=bad_env, target=tgt)
    assert verdict == (
        IntegrityVerdict.CORRUPT.value)
    assert "e1" not in tgt2.event_graph.nodes
    assert "e_extra" not in tgt2.event_graph.nodes


def test_w82_distributed_partition_event_content_addressed():
    from coordpy.distributed_substrate_coordination_v1 import (
        PartitionEventV1,
        W82_DISTRIBUTED_V1_SCHEMA_VERSION,
    )
    a = PartitionEventV1(
        schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
        partition_id="p0",
        side_a_host_ids=("h_a",),
        side_b_host_ids=("h_b", "h_c"),
        started_at_step=10, healed_at_step=15)
    b = PartitionEventV1(
        schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
        partition_id="p0",
        side_a_host_ids=("h_a",),
        side_b_host_ids=("h_b", "h_c"),
        started_at_step=10, healed_at_step=15)
    c = PartitionEventV1(
        schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
        partition_id="p0",
        side_a_host_ids=("h_a",),
        side_b_host_ids=("h_b", "h_c"),
        started_at_step=10, healed_at_step=20)
    assert a.cid() == b.cid()
    assert a.cid() != c.cid()
    assert a.duration_steps() == 5


def test_w82_distributed_heal_sync_converges_after_partition():
    """The load-bearing claim: after heal, all hosts have the
    same merkle root."""
    from coordpy.distributed_substrate_coordination_v1 import (
        run_distributed_coordination_bench_v1,
        ConsistencyVerdict,
    )
    r = run_distributed_coordination_bench_v1()
    assert r.partition_detected is True
    assert r.all_hosts_merkle_match_post_heal is True
    assert r.sync_consistency_verdict == (
        ConsistencyVerdict.EVENTUAL.value)


def test_w82_distributed_cross_runtime_envelope_works():
    from coordpy.distributed_substrate_coordination_v1 import (
        run_distributed_coordination_bench_v1,
    )
    from coordpy.cryptographic_state_integrity_v1 import (
        IntegrityVerdict,
    )
    r = run_distributed_coordination_bench_v1()
    assert r.cross_runtime_envelope_verdict == (
        IntegrityVerdict.OK.value)
    # Genesis + 3 migrated events = 4
    assert r.cross_runtime_target_event_count == 4


def test_w82_distributed_replicate_event_to_hosts_skip_exclude():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        replicate_event_to_hosts_v1,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    hosts = [
        build_simulated_host_v1(host_id=f"h_{i}")
        for i in range(3)]
    e = build_event_node_v1(
        event_id="e1", kind="x", payload_bytes=b"a",
        parent_event_ids=(
            hosts[0].event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    new_hosts = replicate_event_to_hosts_v1(
        event=e, hosts=tuple(hosts),
        exclude_host_ids=("h_1",))
    assert new_hosts[0].event_graph.n_events() == 2
    assert new_hosts[1].event_graph.n_events() == 1
    assert new_hosts[2].event_graph.n_events() == 2


def test_w82_distributed_bench_deterministic():
    from coordpy.distributed_substrate_coordination_v1 import (
        run_distributed_coordination_bench_v1,
    )
    a = run_distributed_coordination_bench_v1(seed=7)
    b = run_distributed_coordination_bench_v1(seed=7)
    assert a.cid() == b.cid()


def test_w82_distributed_bench_cid_changes_on_seed():
    from coordpy.distributed_substrate_coordination_v1 import (
        run_distributed_coordination_bench_v1,
    )
    a = run_distributed_coordination_bench_v1(seed=7)
    b = run_distributed_coordination_bench_v1(seed=8)
    assert a.cid() != b.cid()


def test_w82_distributed_sync_decision_records_pre_post_roots():
    """The sync decision must record both pre-heal and post-
    heal merkle roots so divergence is auditable."""
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        heal_partition_and_sync_v1,
        PartitionEventV1,
        W82_DISTRIBUTED_V1_SCHEMA_VERSION,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    h_a = build_simulated_host_v1(host_id="h_a")
    h_b = build_simulated_host_v1(host_id="h_b")
    # Diverge: each host appends a different event.
    e_a = build_event_node_v1(
        event_id="e_a", kind="x", payload_bytes=b"a",
        parent_event_ids=(
            h_a.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=1)
    e_b = build_event_node_v1(
        event_id="e_b", kind="x", payload_bytes=b"b",
        parent_event_ids=(
            h_b.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=2)
    h_a = h_a.append_event(e_a)
    h_b = h_b.append_event(e_b)
    partition = PartitionEventV1(
        schema=W82_DISTRIBUTED_V1_SCHEMA_VERSION,
        partition_id="p",
        side_a_host_ids=("h_a",),
        side_b_host_ids=("h_b",),
        started_at_step=0, healed_at_step=1)
    new_hosts, decision = heal_partition_and_sync_v1(
        hosts_pre_heal=(h_a, h_b),
        partition=partition)
    assert decision.converged is True
    assert (
        decision.merkle_root_pre_heal_per_host["h_a"] !=
        decision.merkle_root_post_heal_per_host["h_a"])
    assert (
        decision.merkle_root_post_heal_per_host["h_a"] ==
        decision.merkle_root_post_heal_per_host["h_b"])


def test_w82_distributed_budget_policy_content_addressed():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_migration_budget_policy_v1,
    )
    a = build_migration_budget_policy_v1(
        max_bytes_per_migration=1000,
        max_events_per_migration=10)
    b = build_migration_budget_policy_v1(
        max_bytes_per_migration=1000,
        max_events_per_migration=10)
    c = build_migration_budget_policy_v1(
        max_bytes_per_migration=2000,
        max_events_per_migration=10)
    assert a.cid() == b.cid()
    assert a.cid() != c.cid()


def test_w82_distributed_budget_policy_accepts_small_envelope():
    """A small envelope under both bytes and events caps
    should be accepted."""
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
        build_migration_budget_policy_v1,
        decide_migration_budget_v1,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    e = build_event_node_v1(
        event_id="e1", kind="x", payload_bytes=b"small",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=100)
    src = src.append_event(e)
    env = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("e1",))
    policy = build_migration_budget_policy_v1(
        max_bytes_per_migration=10_000,
        max_events_per_migration=10)
    decision = decide_migration_budget_v1(
        policy=policy, envelope=env,
        current_time_ns=200)
    assert decision.accepted is True
    assert decision.rejection_reason == ""


def test_w82_distributed_budget_policy_rejects_oversize_bytes():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
        build_migration_budget_policy_v1,
        decide_migration_budget_v1,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    e = build_event_node_v1(
        event_id="big", kind="x",
        payload_bytes=b"x" * 5000,
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=10)
    src = src.append_event(e)
    env = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("big",))
    policy = build_migration_budget_policy_v1(
        max_bytes_per_migration=1000,  # too tight
        max_events_per_migration=10)
    decision = decide_migration_budget_v1(
        policy=policy, envelope=env, current_time_ns=100)
    assert decision.accepted is False
    assert "max_bytes_per_migration" in (
        decision.rejection_reason)


def test_w82_distributed_budget_policy_rejects_too_many_events():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
        build_migration_budget_policy_v1,
        decide_migration_budget_v1,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    eids: list[str] = []
    prev = src.event_graph.root_event_id
    for i in range(5):
        ev = build_event_node_v1(
            event_id=f"e{i}", kind="x",
            payload_bytes=b"a",
            parent_event_ids=(str(prev),),
            branch_label="main", timestamp_ns=i + 1)
        src = src.append_event(ev)
        eids.append(ev.event_id)
        prev = ev.event_id
    env = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=tuple(eids))
    policy = build_migration_budget_policy_v1(
        max_bytes_per_migration=10_000,
        max_events_per_migration=2)
    decision = decide_migration_budget_v1(
        policy=policy, envelope=env, current_time_ns=100)
    assert decision.accepted is False
    assert "max_events_per_migration" in (
        decision.rejection_reason)


def test_w82_distributed_budget_policy_rejects_stale_events():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
        build_migration_budget_policy_v1,
        decide_migration_budget_v1,
    )
    from coordpy.event_sourced_memory_graph_v1 import (
        build_event_node_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    ev = build_event_node_v1(
        event_id="ancient", kind="x", payload_bytes=b"a",
        parent_event_ids=(
            src.event_graph.root_event_id,),
        branch_label="main", timestamp_ns=10)
    src = src.append_event(ev)
    env = build_migration_envelope_v1(
        source=src, target=tgt,
        event_ids_to_migrate=("ancient",))
    policy = build_migration_budget_policy_v1(
        max_bytes_per_migration=10_000,
        max_events_per_migration=10,
        min_freshness_age_ns=100)  # 100ns freshness cap
    decision = decide_migration_budget_v1(
        policy=policy, envelope=env,
        current_time_ns=10_000)  # event is 9990ns old
    assert decision.accepted is False
    assert "min_freshness_age_ns" in (
        decision.rejection_reason)


def test_w82_distributed_envelope_with_missing_event_raises():
    from coordpy.distributed_substrate_coordination_v1 import (
        build_simulated_host_v1,
        build_migration_envelope_v1,
    )
    src = build_simulated_host_v1(host_id="src")
    tgt = build_simulated_host_v1(host_id="tgt")
    try:
        build_migration_envelope_v1(
            source=src, target=tgt,
            event_ids_to_migrate=("ghost_event",))
    except ValueError:
        return
    raise AssertionError(
        "migrating a non-existent event_id should raise")
