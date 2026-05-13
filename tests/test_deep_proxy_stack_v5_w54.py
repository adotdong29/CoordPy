"""W54 M6 — Deep Proxy Stack V5 tests."""

from __future__ import annotations

import random

from coordpy.deep_proxy_stack_v5 import (
    DeepProxyStackV5,
    W54_DEEP_V5_VERIFIER_FAILURE_MODES,
    emit_deep_proxy_stack_v5_forward_witness,
    verify_deep_proxy_stack_v5_forward_witness,
)


def test_deep_v5_has_12_layers() -> None:
    s = DeepProxyStackV5.init(
        n_layers=12, in_dim=4, factor_dim=4,
        n_heads=2, n_branch_heads=2, n_cycle_heads=2,
        n_roles=2, seed=1)
    assert s.n_layers == 12


def test_deep_v5_abstain_short_circuit_on_pathology() -> None:
    s = DeepProxyStackV5.init(
        n_layers=12, in_dim=4, factor_dim=4,
        n_heads=2, n_branch_heads=2, n_cycle_heads=2,
        n_roles=2,
        abstain_threshold=0.15, seed=1)
    huge_q = [1e6] * 4
    w, _ = emit_deep_proxy_stack_v5_forward_witness(
        stack=s, query_input=huge_q,
        slot_keys=[huge_q], slot_values=[huge_q])
    assert w.abstain_short_circuit or w.corruption_flag


def test_deep_v5_disagreement_head_returns_per_dim_l1() -> None:
    s = DeepProxyStackV5.init(
        n_layers=12, in_dim=4, factor_dim=4,
        n_heads=2, n_branch_heads=2, n_cycle_heads=2,
        n_roles=2, seed=1)
    a = [0.5, 0.0, 0.0, 0.0]
    b = [0.0, 0.5, 0.0, 0.0]
    merged, dis = s.disagreement_head(a, b)
    assert len(dis) == 4
    for v in dis:
        assert v >= 0.0


def test_deep_v5_witness_round_trips() -> None:
    s = DeepProxyStackV5.init(
        n_layers=12, in_dim=4, factor_dim=4,
        n_heads=2, n_branch_heads=2, n_cycle_heads=2,
        n_roles=2, seed=1)
    q = [0.1, 0.2, 0.3, 0.4]
    w, _ = emit_deep_proxy_stack_v5_forward_witness(
        stack=s, query_input=q,
        slot_keys=[q], slot_values=[q],
        paired_input=[0.2, 0.1, 0.4, 0.3])
    v = verify_deep_proxy_stack_v5_forward_witness(
        w, expected_stack_cid=s.cid(),
        expected_n_layers=12)
    assert v["ok"] is True


def test_w54_deep_v5_verifier_failure_modes_count() -> None:
    assert len(W54_DEEP_V5_VERIFIER_FAILURE_MODES) == 7
