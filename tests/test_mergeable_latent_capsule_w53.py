"""W53 M3 mergeable_latent_capsule tests."""

from __future__ import annotations

from coordpy.mergeable_latent_capsule import (
    MergeAuditTrail,
    MergeOperator,
    W53_MLSC_KIND_BRANCH,
    W53_MLSC_KIND_MERGE,
    W53_MLSC_KIND_ROOT,
    W53_MLSC_VERIFIER_FAILURE_MODES,
    compute_consensus_quorum,
    emit_mlsc_witness,
    make_root_capsule,
    merge_capsules,
    step_branch_capsule,
    verify_mlsc_witness,
)


def test_root_capsule_has_no_parents_and_kind_root() -> None:
    c = make_root_capsule(
        branch_id="b1", payload=[0.1, 0.2, 0.3],
        confidence=0.7)
    assert c.kind == W53_MLSC_KIND_ROOT
    assert c.parent_cids == ()
    assert not c.is_merge
    assert 0.0 <= c.confidence <= 1.0


def test_step_branch_capsule_inherits_branch_id() -> None:
    a = make_root_capsule(
        branch_id="b1", payload=[0.1, 0.2, 0.3])
    b = step_branch_capsule(
        parent=a, payload=[0.4, 0.5, 0.6])
    assert b.kind == W53_MLSC_KIND_BRANCH
    assert b.branch_id == "b1"
    assert b.parent_cids == (a.cid(),)
    assert b.turn_index == 1


def test_merge_capsule_records_audit_entry() -> None:
    op = MergeOperator(factor_dim=3)
    audit = MergeAuditTrail.empty()
    a = make_root_capsule(
        branch_id="b1", payload=[0.1, 0.2, 0.3],
        confidence=0.7)
    b = make_root_capsule(
        branch_id="b2", payload=[0.4, 0.5, 0.6],
        confidence=0.5)
    m = merge_capsules(op, [a, b], audit_trail=audit)
    assert m.kind == W53_MLSC_KIND_MERGE
    assert len(audit.entries) == 1
    assert m.cid() in audit.entries
    assert m.branch_id == "merge:b1+b2"


def test_merge_capsule_replay_deterministic() -> None:
    op = MergeOperator(factor_dim=3)
    audit1 = MergeAuditTrail.empty()
    audit2 = MergeAuditTrail.empty()
    a = make_root_capsule(
        branch_id="b1", payload=[0.1, 0.2, 0.3],
        confidence=0.7)
    b = make_root_capsule(
        branch_id="b2", payload=[0.4, 0.5, 0.6],
        confidence=0.5)
    m1 = merge_capsules(op, [a, b], audit_trail=audit1)
    m2 = merge_capsules(op, [a, b], audit_trail=audit2)
    assert m1.cid() == m2.cid()


def test_merge_audit_walk_to_roots() -> None:
    op = MergeOperator(factor_dim=3)
    audit = MergeAuditTrail.empty()
    store: dict = {}
    a = make_root_capsule(
        branch_id="a", payload=[0.1, 0.2, 0.3])
    b = make_root_capsule(
        branch_id="b", payload=[0.4, 0.5, 0.6])
    c = make_root_capsule(
        branch_id="c", payload=[0.7, 0.8, 0.9])
    for cap in (a, b, c):
        store[cap.cid()] = cap
    m_ab = merge_capsules(
        op, [a, b], audit_trail=audit)
    store[m_ab.cid()] = m_ab
    m_all = merge_capsules(
        op, [m_ab, c], audit_trail=audit)
    store[m_all.cid()] = m_all
    roots = audit.walk_to_roots(
        m_all.cid(), capsule_store=store)
    assert sorted(roots) == sorted(
        [a.cid(), b.cid(), c.cid()])


def test_consensus_quorum_reaches_with_consistent_branches() -> None:
    op = MergeOperator(factor_dim=3)
    audit = MergeAuditTrail.empty()
    consistent = [
        make_root_capsule(
            branch_id=f"b{i}",
            payload=[0.5 + 0.05 * i, 0.4, 0.3],
            confidence=0.5)
        for i in range(3)
    ]
    outlier = make_root_capsule(
        branch_id="out",
        payload=[-1.0, -1.0, -1.0],
        confidence=0.5)
    res = compute_consensus_quorum(
        consistent + [outlier],
        operator=op, audit_trail=audit,
        k_required=2, cosine_floor=0.9)
    assert res.quorum_reached
    assert not res.abstain
    assert "out" not in res.selected_branch_ids


def test_consensus_quorum_abstains_when_no_clique() -> None:
    op = MergeOperator(factor_dim=3)
    audit = MergeAuditTrail.empty()
    branches = [
        make_root_capsule(
            branch_id=f"r_{i}",
            payload=[float(j) - 0.5 + 0.1 * i
                     for j in range(3)],
            confidence=0.5)
        for i in range(3)
    ]
    res = compute_consensus_quorum(
        branches, operator=op, audit_trail=audit,
        k_required=3, cosine_floor=0.99)
    assert not res.quorum_reached
    assert res.abstain
    assert res.consensus_capsule_cid == ""


def test_mlsc_witness_emit_and_verify() -> None:
    op = MergeOperator(factor_dim=3)
    audit = MergeAuditTrail.empty()
    store: dict = {}
    a = make_root_capsule(
        branch_id="b1", payload=[0.1, 0.2, 0.3],
        confidence=0.7)
    b = make_root_capsule(
        branch_id="b2", payload=[0.4, 0.5, 0.6],
        confidence=0.5)
    store[a.cid()] = a
    store[b.cid()] = b
    m = merge_capsules(op, [a, b], audit_trail=audit)
    store[m.cid()] = m
    w = emit_mlsc_witness(
        leaf=m, operator=op,
        audit_trail=audit,
        capsule_store=store)
    v = verify_mlsc_witness(
        w,
        expected_leaf_cid=m.cid(),
        expected_operator_cid=op.cid(),
        expected_audit_trail_cid=audit.cid(),
        min_audit_entry_count=1,
        min_n_unique_roots=2)
    assert v["ok"] is True


def test_mlsc_verifier_has_failure_modes() -> None:
    assert len(W53_MLSC_VERIFIER_FAILURE_MODES) >= 9
