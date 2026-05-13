"""W54 M3 — Mergeable Latent State Capsule V2 tests."""

from __future__ import annotations

from coordpy.mergeable_latent_capsule_v2 import (
    MergeAuditTrailV2,
    MergeOperatorV2,
    W54_MLSC_V2_VERIFIER_FAILURE_MODES,
    compute_consensus_quorum_v2,
    emit_mlsc_v2_witness,
    make_root_capsule_v2,
    merge_capsules_v2,
    step_branch_capsule_v2,
    verify_mlsc_v2_witness,
)


def test_root_capsule_has_provenance_for_each_tag() -> None:
    cap = make_root_capsule_v2(
        branch_id="r0",
        payload=[1.0, 0.0, 0.0, 0.0],
        confidence=0.8, trust=0.9,
        fact_tags=("t1", "t2"))
    pmap = dict(cap.fact_tag_provenance)
    assert "t1" in pmap and "t2" in pmap


def test_merge_v2_records_disagreement_and_weights() -> None:
    op = MergeOperatorV2(factor_dim=4)
    audit = MergeAuditTrailV2.empty()
    p_a = make_root_capsule_v2(
        branch_id="a",
        payload=[1.0, 0.0, 0.0, 0.0],
        confidence=0.7, trust=0.9)
    p_b = make_root_capsule_v2(
        branch_id="b",
        payload=[0.0, 1.0, 0.0, 0.0],
        confidence=0.7, trust=0.9)
    merged = merge_capsules_v2(
        op, [p_a, p_b], audit_trail=audit)
    assert len(merged.disagreement_per_dim) == 4
    assert sum(merged.disagreement_per_dim) > 0.0
    assert abs(sum(merged.merge_weights) - 1.0) < 1e-6


def test_merge_v2_trust_weights_higher_trust_more() -> None:
    op = MergeOperatorV2(factor_dim=4)
    audit = MergeAuditTrailV2.empty()
    high = make_root_capsule_v2(
        branch_id="h",
        payload=[1.0, 0.0, 0.0, 0.0],
        confidence=0.5, trust=0.9)
    low = make_root_capsule_v2(
        branch_id="l",
        payload=[0.0, 1.0, 0.0, 0.0],
        confidence=0.5, trust=0.1)
    merged = merge_capsules_v2(
        op, [high, low], audit_trail=audit)
    # weights[0] is for high; should be > weights[1].
    assert merged.merge_weights[0] > merged.merge_weights[1]


def test_consensus_quorum_v2_reaches_on_consistent() -> None:
    op = MergeOperatorV2(factor_dim=4)
    audit = MergeAuditTrailV2.empty()
    target = [0.8, 0.3, -0.4, 0.2]
    branches = [
        make_root_capsule_v2(
            branch_id=f"b{i}",
            payload=[t + 0.02 * i for t in target],
            confidence=0.8, trust=0.9)
        for i in range(4)
    ]
    res = compute_consensus_quorum_v2(
        branches, operator=op, audit_trail=audit,
        k_required=2, cosine_floor=0.5,
        allow_fallback=False,
        fallback_cosine_floor=0.0)
    assert res.quorum_reached
    assert not res.abstain
    assert not res.fallback_used


def test_consensus_quorum_v2_fallback_when_kmin_unreachable() -> None:
    op = MergeOperatorV2(factor_dim=4)
    audit = MergeAuditTrailV2.empty()
    branches = [
        make_root_capsule_v2(
            branch_id=f"b{i}",
            payload=[1.0 if i == 0 else -1.0,
                     0.0, 0.0, 0.0],
            confidence=0.5 + 0.1 * i, trust=0.8)
        for i in range(2)
    ]
    res = compute_consensus_quorum_v2(
        branches, operator=op, audit_trail=audit,
        k_required=2, cosine_floor=0.99,
        allow_fallback=True,
        fallback_cosine_floor=-1.0)
    assert not res.quorum_reached
    assert res.fallback_used


def test_w54_mlsc_v2_verifier_failure_modes_count() -> None:
    assert len(W54_MLSC_V2_VERIFIER_FAILURE_MODES) == 12


def test_w54_mlsc_v2_witness_round_trips() -> None:
    op = MergeOperatorV2(factor_dim=4)
    audit = MergeAuditTrailV2.empty()
    store: dict = {}
    a = make_root_capsule_v2(
        branch_id="a",
        payload=[1.0, 0.0, 0.0, 0.0],
        confidence=0.8, trust=0.9,
        fact_tags=("t1",))
    b = make_root_capsule_v2(
        branch_id="b",
        payload=[0.0, 1.0, 0.0, 0.0],
        confidence=0.8, trust=0.9,
        fact_tags=("t2",))
    store[a.cid()] = a
    store[b.cid()] = b
    merged = merge_capsules_v2(
        op, [a, b], audit_trail=audit)
    store[merged.cid()] = merged
    w = emit_mlsc_v2_witness(
        leaf=merged, operator=op,
        audit_trail=audit, capsule_store=store)
    v = verify_mlsc_v2_witness(
        w,
        expected_leaf_cid=merged.cid(),
        expected_operator_cid=op.cid(),
        expected_audit_trail_cid=audit.cid(),
        capsule_store=store, audit_trail=audit)
    assert v["ok"] is True
