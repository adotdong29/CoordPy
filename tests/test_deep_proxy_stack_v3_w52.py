"""Per-module tests for the W52 deep proxy stack V3."""

from __future__ import annotations

import pytest

from coordpy.deep_proxy_stack_v3 import (
    DeepProxyStackV3,
    collapse_role_banks,
    emit_deep_proxy_stack_v3_forward_witness,
    evaluate_deep_stack_v3_accuracy,
    fit_deep_proxy_stack_v3,
    synthesize_deep_stack_v3_training_set,
    verify_deep_proxy_stack_v3_forward_witness,
)


def test_v3_init_has_l8_layers() -> None:
    s = DeepProxyStackV3.init(n_layers=8, in_dim=6, n_roles=2,
                              seed=1)
    assert s.n_layers == 8
    assert len(s.layers) == 8


def test_v3_per_layer_role_bank_zero_when_collapsed() -> None:
    s = DeepProxyStackV3.init(n_layers=4, in_dim=4, n_roles=2,
                              seed=1)
    collapse_role_banks(s)
    for layer in s.layers:
        for v in layer.role_bank.w_proj.values:
            assert v == 0.0


def test_v3_forward_witness_round_trips() -> None:
    s = DeepProxyStackV3.init(n_layers=4, in_dim=4, n_roles=2,
                              seed=2)
    q = [0.1, 0.2, 0.3, 0.4]
    w, h = emit_deep_proxy_stack_v3_forward_witness(
        stack=s, query_input=q,
        slot_keys=[q], slot_values=[q],
        role_index=0, branch_index=0, cycle_index=0)
    assert w.n_layers == 4
    assert w.n_roles == 2
    v = verify_deep_proxy_stack_v3_forward_witness(
        w, expected_stack_cid=s.cid(),
        expected_n_layers=4, expected_n_roles=2)
    assert v["ok"] is True


def test_v3_fit_classifier_converges() -> None:
    ts = synthesize_deep_stack_v3_training_set(
        n_examples=24, in_dim=6, compose_depth=8,
        n_branches=2, n_cycles=2, n_roles=2, seed=1)
    s, trace = fit_deep_proxy_stack_v3(
        ts, n_layers=4, n_steps=24, seed=1)
    assert not trace.diverged
    acc = evaluate_deep_stack_v3_accuracy(s, ts.examples)
    assert acc >= 0.5
