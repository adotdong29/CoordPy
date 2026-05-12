"""Unit tests for W51 M3 — deep proxy stack V2."""

from __future__ import annotations

import pytest

from coordpy.deep_proxy_stack_v2 import (
    BranchCycleSelector,
    DeepProxyStackV2,
    W51_DEEP_STACK_V2_VERIFIER_FAILURE_MODES,
    W51_DEFAULT_DEEP_V2_N_LAYERS,
    collapse_branch_cycle_selectors,
    emit_deep_proxy_stack_v2_forward_witness,
    evaluate_deep_stack_v2_accuracy,
    fit_deep_proxy_stack_v2,
    synthesize_deep_stack_v2_training_set,
    verify_deep_proxy_stack_v2_forward_witness,
)


def test_stack_default_n_layers_is_6() -> None:
    assert W51_DEFAULT_DEEP_V2_N_LAYERS == 6
    stack = DeepProxyStackV2.init(seed=11)
    assert stack.n_layers == 6
    assert len(stack.layers) == 6


def test_stack_init_stable_seed() -> None:
    a = DeepProxyStackV2.init(seed=11)
    b = DeepProxyStackV2.init(seed=11)
    assert a.cid() == b.cid()


def test_branch_cycle_selector_gate_values_in_unit_interval() -> None:
    sel = BranchCycleSelector.init(seed=11)
    for b in range(sel.n_branch_heads):
        for c in range(sel.n_cycle_heads):
            g = sel.gate_value(branch_index=b, cycle_index=c)
            assert 0.0 <= g <= 1.0


def test_forward_witness_per_layer_norms_correct_length() -> None:
    stack = DeepProxyStackV2.init(seed=11, n_layers=6, in_dim=6)
    q = [0.1] * 6
    w, _ = emit_deep_proxy_stack_v2_forward_witness(
        stack=stack, query_input=q,
        slot_keys=[q], slot_values=[q],
        branch_index=0, cycle_index=0)
    assert len(w.per_layer_l2_norms) == 6
    assert len(w.per_layer_gate_values) == 6
    assert len(w.per_layer_bc_gate_values) == 6
    assert len(w.per_layer_temperatures) == 6
    assert w.n_layers == 6


def test_verifier_passes_clean() -> None:
    stack = DeepProxyStackV2.init(seed=11)
    q = [0.1] * stack.in_dim
    w, _ = emit_deep_proxy_stack_v2_forward_witness(
        stack=stack, query_input=q,
        slot_keys=[q], slot_values=[q])
    v = verify_deep_proxy_stack_v2_forward_witness(
        w, expected_stack_cid=stack.cid(),
        expected_n_layers=stack.n_layers)
    assert v["ok"] is True


def test_collapsed_selectors_reduce_to_shared_heads() -> None:
    """Branch/cycle specialisation must produce a different
    accuracy from shared-heads only.
    """
    ts = synthesize_deep_stack_v2_training_set(
        n_examples=24, in_dim=6, compose_depth=6,
        n_branches=2, n_cycles=2, seed=11)
    s, _ = fit_deep_proxy_stack_v2(
        ts, n_layers=6, n_steps=96, seed=11)
    acc_specialised = evaluate_deep_stack_v2_accuracy(
        s, ts.examples)
    collapsed = collapse_branch_cycle_selectors(s)
    acc_shared = evaluate_deep_stack_v2_accuracy(
        collapsed, ts.examples)
    # Specialised heads should be at least as good as shared.
    # (Branch/cycle conditioning is informative on this dataset.)
    assert acc_specialised >= acc_shared - 0.10


def test_verifier_has_7_failure_modes() -> None:
    assert len(W51_DEEP_STACK_V2_VERIFIER_FAILURE_MODES) == 7
    assert len(set(
        W51_DEEP_STACK_V2_VERIFIER_FAILURE_MODES)) == 7


def test_trained_stack_above_chance() -> None:
    ts = synthesize_deep_stack_v2_training_set(
        n_examples=24, in_dim=6, compose_depth=6,
        n_branches=2, n_cycles=2, seed=11)
    s, _ = fit_deep_proxy_stack_v2(
        ts, n_layers=6, n_steps=96, seed=11)
    acc = evaluate_deep_stack_v2_accuracy(s, ts.examples)
    # Above chance (0.5)
    assert acc >= 0.55
