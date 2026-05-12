"""W53 M7 branch_merge_memory_v3 tests."""

from __future__ import annotations

from coordpy.branch_merge_memory_v3 import (
    BranchMergeMemoryV3Head,
    emit_bmm_v3_witness,
    evaluate_consensus_recall,
    verify_bmm_v3_witness,
)


def test_bmm_v3_consensus_reaches_with_consistent_branches() -> None:
    h = BranchMergeMemoryV3Head.init(
        factor_dim=4, n_branch_pages=4,
        n_cycle_pages=4, n_joint_pages=4,
        n_consensus_pages=4, k_required=2,
        cosine_floor=0.5, seed=11)
    rec = evaluate_consensus_recall(
        h, n_branches=4, n_consistent=3,
        factor_dim=4, seed=7)
    assert rec >= 0.5


def test_bmm_v3_abstains_when_quorum_too_high() -> None:
    h = BranchMergeMemoryV3Head.init(
        factor_dim=4, n_branch_pages=4,
        n_cycle_pages=4, n_joint_pages=4,
        n_consensus_pages=4, k_required=4,
        cosine_floor=0.99, seed=13)
    rec = evaluate_consensus_recall(
        h, n_branches=4, n_consistent=2,
        factor_dim=4, seed=7)
    # No K=4 quorum at cosine 0.99 → abstain.
    assert rec == 0.0


def test_bmm_v3_witness_emit_and_verify() -> None:
    h = BranchMergeMemoryV3Head.init(
        factor_dim=4, n_branch_pages=4,
        n_cycle_pages=4, n_joint_pages=4,
        n_consensus_pages=4, k_required=2,
        cosine_floor=0.5, seed=17)
    rec = evaluate_consensus_recall(
        h, n_branches=4, n_consistent=3,
        factor_dim=4, seed=7)
    w = emit_bmm_v3_witness(
        head=h, consensus_recall=rec)
    v = verify_bmm_v3_witness(
        w,
        expected_head_cid=h.cid(),
        expected_inner_v2_cid=h.inner_v2.cid(),
        expected_n_consensus_pages=4,
        expected_k_required=2,
        min_consensus_recall=0.5)
    assert v["ok"] is True


def test_bmm_v3_consensus_audit_count_grows() -> None:
    h = BranchMergeMemoryV3Head.init(
        factor_dim=4, n_branch_pages=4,
        n_cycle_pages=4, n_joint_pages=4,
        n_consensus_pages=4, k_required=2,
        cosine_floor=0.5, seed=19)
    rec = evaluate_consensus_recall(
        h, n_branches=4, n_consistent=3,
        factor_dim=4, seed=7)
    assert len(h.consensus_audit) >= 1
