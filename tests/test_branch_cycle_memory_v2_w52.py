"""Per-module tests for the W52 branch/cycle memory V2."""

from __future__ import annotations

import pytest

from coordpy.branch_cycle_memory_v2 import (
    BranchCycleMemoryV2Head,
    MergeAuditEntry,
    W52_DEFAULT_BCM_V2_N_JOINT_PAGES,
    emit_branch_cycle_memory_v2_witness,
    evaluate_joint_recall_v2,
    fit_branch_cycle_memory_v2,
    synthesize_branch_cycle_memory_v2_training_set,
    verify_branch_cycle_memory_v2_witness,
)


def test_v2_init_has_joint_pages() -> None:
    head = BranchCycleMemoryV2Head.init(
        factor_dim=4, n_branch_pages=4, n_cycle_pages=4,
        n_joint_pages=16, seed=1)
    assert head.n_joint_pages == 16
    assert len(head.joint_pages) == 16


def test_v2_write_to_joint_appends_slot() -> None:
    head = BranchCycleMemoryV2Head.init(
        factor_dim=4, n_branch_pages=2, n_cycle_pages=2,
        n_joint_pages=4, seed=2)
    head.write_to_joint(
        branch_index=0, cycle_index=0,
        key=[0.1, 0.2, 0.3, 0.4],
        value=[1.0, 2.0, 3.0, 4.0],
        fact_tag="t0")
    assert len(head.joint_pages[0].slots) == 1


def test_v2_fit_joint_recall_beats_v1() -> None:
    """The V2 head with joint pages should beat V1 (no joints)
    on a per-pair regime."""
    from coordpy.branch_cycle_memory import BranchCycleMemoryHead
    from coordpy.branch_cycle_memory_v2 import (
        evaluate_v1_joint_recall_baseline,
    )
    ts = synthesize_branch_cycle_memory_v2_training_set(
        n_examples=12, factor_dim=4,
        n_branch_pages=3, n_cycle_pages=3,
        seed=1)
    head, _ = fit_branch_cycle_memory_v2(
        ts, n_steps=24, seed=1)
    v2_recall = evaluate_joint_recall_v2(head, ts.examples)
    v1 = BranchCycleMemoryHead.init(
        factor_dim=4, n_branch_pages=3, n_cycle_pages=3,
        seed=1)
    v1_recall = evaluate_v1_joint_recall_baseline(
        v1, ts.examples)
    assert v2_recall > v1_recall


def test_v2_merge_audit_log_starts_empty() -> None:
    head = BranchCycleMemoryV2Head.init(
        factor_dim=4, seed=3)
    assert head.merge_audit == []


def test_v2_witness_round_trips() -> None:
    ts = synthesize_branch_cycle_memory_v2_training_set(
        n_examples=8, factor_dim=4,
        n_branch_pages=2, n_cycle_pages=2,
        seed=4)
    head, trace = fit_branch_cycle_memory_v2(
        ts, n_steps=12, seed=4)
    w = emit_branch_cycle_memory_v2_witness(
        head=head, training_trace=trace,
        examples=ts.examples[:4])
    v = verify_branch_cycle_memory_v2_witness(
        w, expected_head_cid=head.cid(),
        expected_n_joint_pages=head.n_joint_pages)
    assert v["ok"] is True
