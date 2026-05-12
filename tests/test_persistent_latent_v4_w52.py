"""Per-module tests for the W52 persistent latent V4 module."""

from __future__ import annotations

import pytest

from coordpy.persistent_latent_v4 import (
    PersistentLatentStateV4,
    PersistentLatentStateV4Chain,
    V4StackedCell,
    W52_DEFAULT_V4_MAX_CHAIN_WALK_DEPTH,
    W52_V4_NO_PARENT_STATE,
    emit_persistent_v4_witness,
    evaluate_v4_long_horizon_recall,
    fit_persistent_v4,
    step_persistent_state_v4,
    synthesize_v4_training_set,
    verify_persistent_v4_witness,
)


def test_v4_cell_init_n_layers_two() -> None:
    cell = V4StackedCell.init(
        state_dim=8, input_dim=8, n_layers=2, seed=1)
    assert cell.n_layers == 2
    assert cell.state_dim == 8


def test_v4_cell_step_value_at_trivial_params_persists() -> None:
    """At trivial params (gate near 0, identity skip), the state
    should persist through noisy inputs."""
    cell = V4StackedCell.init(
        state_dim=4, input_dim=4, n_layers=2, seed=42)
    prev = [[0.5, 0.5, 0.5, 0.5], [0.3, 0.3, 0.3, 0.3]]
    noise = [0.01, 0.01, 0.01, 0.01]
    nxt, gates = cell.step_value(
        prev_layer_states=prev, input_x=noise, skip_input=None)
    assert len(nxt) == 2
    assert len(nxt[0]) == 4
    assert len(gates) == 2


def test_v4_state_chain_walk_depth() -> None:
    cell = V4StackedCell.init(
        state_dim=4, input_dim=4, n_layers=2, seed=2)
    chain = PersistentLatentStateV4Chain.empty()
    state = None
    for t in range(5):
        x = [float((t + 1) % 5) * 0.1] * 4
        new = step_persistent_state_v4(
            cell=cell, prev_state=state, carrier_values=x,
            turn_index=t, role="r0", cycle_index=0,
            skip_input=x)
        chain.add(new)
        state = new
    assert state is not None
    walk = chain.walk_from(state.cid())
    assert len(walk) == 5


def test_v4_witness_round_trips() -> None:
    cell = V4StackedCell.init(
        state_dim=4, input_dim=4, n_layers=2, seed=3)
    chain = PersistentLatentStateV4Chain.empty()
    state = step_persistent_state_v4(
        cell=cell, prev_state=None,
        carrier_values=[0.1, 0.2, 0.3, 0.4],
        turn_index=0, role="r0")
    chain.add(state)
    w = emit_persistent_v4_witness(
        state=state, cell=cell, chain=chain)
    assert w.state_cid == state.cid()
    res = verify_persistent_v4_witness(
        w,
        expected_state_cid=state.cid(),
        expected_cell_cid=cell.cid(),
        expected_n_layers=2)
    assert res["ok"] is True


def test_v4_fit_converges_on_short_sequence() -> None:
    ts = synthesize_v4_training_set(
        n_sequences=4, sequence_length=8, state_dim=4,
        input_dim=4, seed=1)
    cell, trace = fit_persistent_v4(
        ts, n_steps=32, seed=1, truncate_bptt=3)
    recall = evaluate_v4_long_horizon_recall(cell, ts.examples)
    assert recall > 0.3
    assert not trace.diverged


def test_v4_chain_depth_within_max_walk() -> None:
    cell = V4StackedCell.init(
        state_dim=4, input_dim=4, n_layers=2, seed=2)
    chain = PersistentLatentStateV4Chain.empty()
    state = None
    for t in range(W52_DEFAULT_V4_MAX_CHAIN_WALK_DEPTH + 3):
        x = [float(t % 5) * 0.1] * 4
        state = step_persistent_state_v4(
            cell=cell, prev_state=state, carrier_values=x,
            turn_index=t, role="r0", cycle_index=0)
        chain.add(state)
    walk = chain.walk_from(state.cid())
    assert len(walk) == W52_DEFAULT_V4_MAX_CHAIN_WALK_DEPTH
