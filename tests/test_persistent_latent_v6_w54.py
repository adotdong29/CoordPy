"""W54 M1 — Persistent Latent State V6 tests."""

from __future__ import annotations

import random

from coordpy.persistent_latent_v6 import (
    PersistentLatentStateV6Chain,
    V6StackedCell,
    W54_DEFAULT_V6_MAX_CHAIN_WALK_DEPTH,
    W54_V6_VERIFIER_FAILURE_MODES,
    emit_persistent_v6_witness,
    merge_persistent_states_v6,
    step_persistent_state_v6,
    verify_persistent_v6_witness,
)


def test_v6_cell_init_has_4_layers() -> None:
    cell = V6StackedCell.init(
        state_dim=4, input_dim=4, n_layers=4, seed=1)
    assert cell.n_layers == 4


def test_v6_step_value_is_deterministic() -> None:
    cell = V6StackedCell.init(
        state_dim=4, input_dim=4, n_layers=4, seed=1)
    prev = [[0.0] * 4 for _ in range(4)]
    out1, _ = cell.step_value(
        prev_layer_states=prev,
        input_x=[0.1, 0.2, 0.3, 0.4],
        anchor_skip=[0.5] * 4, ema_skip=[0.05] * 4)
    out2, _ = cell.step_value(
        prev_layer_states=prev,
        input_x=[0.1, 0.2, 0.3, 0.4],
        anchor_skip=[0.5] * 4, ema_skip=[0.05] * 4)
    for la, lb in zip(out1, out2):
        for a, b in zip(la, lb):
            assert abs(a - b) < 1e-12


def test_v6_state_chain_walk_depth() -> None:
    cell = V6StackedCell.init(
        state_dim=4, input_dim=4, n_layers=4, seed=1)
    chain = PersistentLatentStateV6Chain.empty()
    prev = None
    anchor = [0.5] * 4
    for t in range(20):
        s = step_persistent_state_v6(
            cell=cell, prev_state=prev,
            carrier_values=[0.1, 0.2, 0.3, 0.4],
            turn_index=t, role="r0",
            anchor_skip=anchor)
        chain.add(s)
        prev = s
    w = emit_persistent_v6_witness(
        state=prev, cell=cell, chain=chain,
        max_walk_depth=64)
    assert w.chain_walk_depth >= 16


def test_v6_merge_records_disagreement() -> None:
    cell = V6StackedCell.init(
        state_dim=4, input_dim=4, n_layers=4, seed=1)
    sa = step_persistent_state_v6(
        cell=cell, prev_state=None,
        carrier_values=[1.0, 0.0, 0.0, 0.0],
        turn_index=0, role="r0", branch_id="a",
        anchor_skip=[1.0, 0.0, 0.0, 0.0])
    sb = step_persistent_state_v6(
        cell=cell, prev_state=None,
        carrier_values=[0.0, 1.0, 0.0, 0.0],
        turn_index=0, role="r0", branch_id="b",
        anchor_skip=[0.0, 1.0, 0.0, 0.0])
    merged, dis = merge_persistent_states_v6(
        cell=cell, state_a=sa, state_b=sb,
        merged_branch_id="m")
    assert merged.is_merge
    assert merged.second_parent_state_cid == sb.cid()
    assert merged.disagreement_l1_sum >= 0.0
    # Disagreement per layer should be non-negative.
    for layer_dis in dis:
        for v in layer_dis:
            assert v >= 0.0


def test_v6_verifier_failure_modes_count() -> None:
    assert len(W54_V6_VERIFIER_FAILURE_MODES) == 10


def test_v6_verifier_rejects_wrong_cell_cid() -> None:
    cell = V6StackedCell.init(
        state_dim=4, input_dim=4, n_layers=4, seed=1)
    chain = PersistentLatentStateV6Chain.empty()
    s = step_persistent_state_v6(
        cell=cell, prev_state=None,
        carrier_values=[0.1, 0.2, 0.3, 0.4],
        turn_index=0, role="r0",
        anchor_skip=[0.5] * 4)
    chain.add(s)
    w = emit_persistent_v6_witness(
        state=s, cell=cell, chain=chain)
    v_bad = verify_persistent_v6_witness(
        w, expected_cell_cid="ff" * 32)
    assert v_bad["ok"] is False
    assert "w54_v6_cell_cid_mismatch" in v_bad["failures"]
