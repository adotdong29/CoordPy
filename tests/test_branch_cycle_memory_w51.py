"""Unit tests for W51 M6 — branch/cycle memory head."""

from __future__ import annotations

import pytest

from coordpy.branch_cycle_memory import (
    BranchCycleMemoryHead,
    W51_BRANCH_CYCLE_MEMORY_VERIFIER_FAILURE_MODES,
    W51_DEFAULT_BCM_N_BRANCH_PAGES,
    W51_DEFAULT_BCM_N_CYCLE_PAGES,
    apply_writes_to_head,
    emit_branch_cycle_memory_witness,
    evaluate_branch_cycle_recall,
    evaluate_branch_cycle_recall_specialised,
    evaluate_generic_memory_recall,
    fit_branch_cycle_memory,
    synthesize_branch_cycle_memory_training_set,
    verify_branch_cycle_memory_witness,
)


def test_head_default_branch_pages_and_cycle_pages() -> None:
    head = BranchCycleMemoryHead.init(seed=11)
    assert head.n_branch_pages == W51_DEFAULT_BCM_N_BRANCH_PAGES
    assert head.n_cycle_pages == W51_DEFAULT_BCM_N_CYCLE_PAGES
    assert len(head.branch_pages) == head.n_branch_pages
    assert len(head.cycle_pages) == head.n_cycle_pages


def test_head_init_stable_seed() -> None:
    a = BranchCycleMemoryHead.init(seed=11)
    b = BranchCycleMemoryHead.init(seed=11)
    assert a.cid() == b.cid()


def test_write_to_branch_appends_slot() -> None:
    head = BranchCycleMemoryHead.init(seed=11, factor_dim=4)
    head.write_to_branch(
        branch_index=0, key=[1.0, 0.0, 0.0, 0.0],
        value=[0.0, 1.0, 0.0, 0.0])
    assert len(head.branch_pages[0].slots) == 1


def test_branch_isolation_via_pages() -> None:
    """Per-branch pages are isolated: write to branch 0,
    read from branch 1 returns zero."""
    head = BranchCycleMemoryHead.init(seed=11, factor_dim=4)
    head.write_to_branch(
        branch_index=0,
        key=[1.0, 0.0, 0.0, 0.0],
        value=[5.0, 5.0, 5.0, 5.0])
    # Read from branch 1 (a different page)
    out = head.branch_pages[1].read_value(
        [1.0, 0.0, 0.0, 0.0])
    # Branch 1 page is empty → zero output
    assert all(v == 0.0 for v in out)
    # Branch 0 page returns the value
    out_b0 = head.branch_pages[0].read_value(
        [1.0, 0.0, 0.0, 0.0])
    assert all(abs(v) > 0.1 for v in out_b0)


def test_branch_cycle_specialisation_strict_gain() -> None:
    """BCM strict gain over generic single-page baseline."""
    ts = synthesize_branch_cycle_memory_training_set(
        n_examples=8, factor_dim=4, n_branch_pages=4,
        n_cycle_pages=4, seed=11)
    generic_recall = evaluate_generic_memory_recall(
        ts.examples, factor_dim=4)
    head, _ = fit_branch_cycle_memory(
        ts, n_steps=24, seed=11)
    bcm_recall = evaluate_branch_cycle_recall_specialised(
        head, ts.examples)
    # H6 bar of +0.15
    assert bcm_recall - generic_recall > 0.0


def test_witness_passes_clean_verifier() -> None:
    ts = synthesize_branch_cycle_memory_training_set(
        n_examples=8, factor_dim=4, n_branch_pages=4,
        n_cycle_pages=4, seed=11)
    head, trace = fit_branch_cycle_memory(
        ts, n_steps=24, seed=11)
    w = emit_branch_cycle_memory_witness(
        head=head, training_trace=trace,
        examples=ts.examples)
    v = verify_branch_cycle_memory_witness(
        w, expected_head_cid=head.cid(),
        expected_trace_cid=trace.cid(),
        recall_floor=0.0,
        expected_n_branch_pages=head.n_branch_pages,
        expected_n_cycle_pages=head.n_cycle_pages)
    assert v["ok"] is True


def test_verifier_has_8_failure_modes() -> None:
    assert len(
        W51_BRANCH_CYCLE_MEMORY_VERIFIER_FAILURE_MODES) == 8
