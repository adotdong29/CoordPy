"""W55 M1 — persistent latent V7 unit tests."""

from __future__ import annotations

from coordpy.persistent_latent_v7 import (
    PersistentLatentStateV7Chain,
    V7StackedCell,
    W55_DEFAULT_V7_MAX_CHAIN_WALK_DEPTH,
    W55_V7_VERIFIER_FAILURE_MODES,
    emit_persistent_v7_witness,
    evaluate_v7_long_horizon_recall,
    fit_persistent_v7,
    merge_persistent_states_v7,
    step_persistent_state_v7,
    verify_persistent_v7_witness,
)


def test_v7_cell_has_5_layers_by_default() -> None:
    cell = V7StackedCell.init(seed=1)
    assert cell.n_layers == 5


def test_v7_chain_walk_depth_default_is_128() -> None:
    assert W55_DEFAULT_V7_MAX_CHAIN_WALK_DEPTH == 128


def test_v7_step_advances_turn_index() -> None:
    cell = V7StackedCell.init(
        state_dim=4, input_dim=4, n_layers=5, seed=1)
    s = step_persistent_state_v7(
        cell=cell, prev_state=None,
        carrier_values=[0.1, 0.2, 0.3, 0.4],
        turn_index=0, role="r0",
        anchor_skip=[0.1, 0.2, 0.3, 0.4])
    s2 = step_persistent_state_v7(
        cell=cell, prev_state=s,
        carrier_values=[0.5, 0.6, 0.7, 0.8],
        turn_index=1, role="r0",
        anchor_skip=[0.1, 0.2, 0.3, 0.4])
    assert s2.turn_index == 1
    assert s2.parent_state_cid == s.cid()


def test_v7_chain_walk_returns_history() -> None:
    cell = V7StackedCell.init(
        state_dim=4, input_dim=4, n_layers=5, seed=1)
    chain = PersistentLatentStateV7Chain.empty()
    s = None
    for t in range(10):
        s = step_persistent_state_v7(
            cell=cell, prev_state=s,
            carrier_values=[0.1, 0.2, 0.3, 0.4],
            turn_index=t, role="r0",
            anchor_skip=[0.1, 0.2, 0.3, 0.4])
        chain.add(s)
    history = chain.walk_from(s.cid())
    assert len(history) == 10


def test_v7_witness_records_chain_walk() -> None:
    cell = V7StackedCell.init(seed=1, state_dim=4, n_layers=5)
    chain = PersistentLatentStateV7Chain.empty()
    s = step_persistent_state_v7(
        cell=cell, prev_state=None,
        carrier_values=[0.1] * 4, turn_index=0,
        role="r0", anchor_skip=[0.1] * 4)
    chain.add(s)
    w = emit_persistent_v7_witness(
        state=s, cell=cell, chain=chain)
    assert w.chain_walk_depth == 1
    assert w.cell_cid == cell.cid()


def test_v7_verifier_failure_mode_count() -> None:
    assert len(W55_V7_VERIFIER_FAILURE_MODES) == 11


def test_v7_merge_emits_low_high_bounds() -> None:
    cell = V7StackedCell.init(
        state_dim=4, input_dim=4, n_layers=5, seed=1)
    s_a = step_persistent_state_v7(
        cell=cell, prev_state=None,
        carrier_values=[1.0, 0.0, 0.5, 0.0],
        turn_index=0, role="r0", branch_id="a",
        anchor_skip=[1.0, 0.0, 0.5, 0.0])
    s_b = step_persistent_state_v7(
        cell=cell, prev_state=None,
        carrier_values=[0.0, 1.0, 0.5, 0.0],
        turn_index=0, role="r0", branch_id="b",
        anchor_skip=[0.0, 1.0, 0.5, 0.0])
    merged, disagreement, low, high = (
        merge_persistent_states_v7(
            cell=cell, state_a=s_a, state_b=s_b,
            merged_branch_id="m"))
    assert merged.is_merge
    # Low ≤ merged ≤ high per dim, per layer.
    for layer in range(len(low)):
        for i in range(len(low[layer])):
            assert (
                low[layer][i]
                <= max(s_a.layer_states[layer][i],
                        s_b.layer_states[layer][i]) + 1e-9)
            assert (
                high[layer][i]
                >= min(s_a.layer_states[layer][i],
                        s_b.layer_states[layer][i]) - 1e-9)


def test_v7_fit_recovers_inner_v6() -> None:
    cell, trace = fit_persistent_v7(
        state_dim=4, input_dim=4, n_layers=5,
        n_sequences=2, sequence_length=8, n_steps=4,
        seed=11)
    assert cell.n_layers == 5
    assert cell.inner_v6.n_layers == 4
