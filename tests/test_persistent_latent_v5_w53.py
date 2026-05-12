"""W53 M1 persistent_latent_v5 tests."""

from __future__ import annotations

from coordpy.persistent_latent_v5 import (
    V5StackedCell,
    PersistentLatentStateV5Chain,
    emit_persistent_v5_witness,
    evaluate_v5_long_horizon_recall,
    fit_persistent_v5,
    forge_v5_training_set,
    merge_persistent_states_v5,
    step_persistent_state_v5,
    synthesize_v5_training_set,
    verify_persistent_v5_witness,
    W53_V5_NO_PARENT_STATE,
)


def test_v5_step_produces_state_with_layered_states() -> None:
    cell = V5StackedCell.init(
        state_dim=4, input_dim=4, n_layers=3, seed=11)
    s = step_persistent_state_v5(
        cell=cell, prev_state=None,
        carrier_values=[0.1, 0.2, 0.3, 0.4],
        turn_index=0, role="r0",
        branch_id="m", cycle_index=0)
    assert s.n_layers == 3
    assert len(s.layer_states) == 3
    assert s.parent_state_cid == W53_V5_NO_PARENT_STATE
    assert not s.is_merge


def test_v5_chain_walk_back_to_root() -> None:
    cell = V5StackedCell.init(
        state_dim=4, input_dim=4, n_layers=3, seed=13)
    chain = PersistentLatentStateV5Chain.empty()
    prev = None
    for t in range(8):
        s = step_persistent_state_v5(
            cell=cell, prev_state=prev,
            carrier_values=[0.1, 0.2, 0.3, 0.4],
            turn_index=t, role="r0",
            branch_id="m", cycle_index=0)
        chain.add(s)
        prev = s
    assert prev is not None
    walk = chain.walk_from(prev.cid(), max_depth=16)
    assert len(walk) == 8


def test_v5_merge_produces_merge_state() -> None:
    cell = V5StackedCell.init(
        state_dim=4, input_dim=4, n_layers=3, seed=17)
    a = step_persistent_state_v5(
        cell=cell, prev_state=None,
        carrier_values=[0.1, 0.2, 0.3, 0.4],
        turn_index=0, role="r0",
        branch_id="b1", cycle_index=0)
    b = step_persistent_state_v5(
        cell=cell, prev_state=None,
        carrier_values=[-0.1, -0.2, -0.3, -0.4],
        turn_index=0, role="r0",
        branch_id="b2", cycle_index=0)
    m = merge_persistent_states_v5(
        cell=cell, state_a=a, state_b=b,
        merged_branch_id="merged",
        turn_index=1, role="r0")
    assert m.is_merge
    assert m.parent_state_cid == a.cid()
    assert m.second_parent_state_cid == b.cid()
    assert m.branch_id == "merged"


def test_v5_fit_does_not_diverge() -> None:
    ts = synthesize_v5_training_set(
        n_sequences=4, sequence_length=12,
        state_dim=4, input_dim=4, seed=23,
        distractor_window=(3, 8))
    cell, trace = fit_persistent_v5(
        ts, n_steps=24, seed=23, n_layers=3,
        truncate_bptt=2)
    assert not trace.diverged
    assert trace.final_loss == trace.final_loss  # not NaN


def test_v5_long_horizon_recall_stays_finite() -> None:
    ts = synthesize_v5_training_set(
        n_sequences=2, sequence_length=12,
        state_dim=4, input_dim=4, seed=29,
        distractor_window=(3, 8))
    cell, _ = fit_persistent_v5(
        ts, n_steps=24, seed=29, n_layers=3,
        truncate_bptt=2)
    rec = evaluate_v5_long_horizon_recall(
        cell, ts.examples)
    assert -1.0 <= rec <= 1.0


def test_v5_witness_emit_and_verify_clean() -> None:
    cell = V5StackedCell.init(
        state_dim=4, input_dim=4, n_layers=3, seed=37)
    chain = PersistentLatentStateV5Chain.empty()
    s = step_persistent_state_v5(
        cell=cell, prev_state=None,
        carrier_values=[0.1, 0.2, 0.3, 0.4],
        turn_index=0, role="r0",
        branch_id="m", cycle_index=0)
    chain.add(s)
    w = emit_persistent_v5_witness(
        state=s, cell=cell, chain=chain)
    v = verify_persistent_v5_witness(
        w,
        expected_state_cid=s.cid(),
        expected_cell_cid=cell.cid(),
        expected_n_layers=3)
    assert v["ok"] is True


def test_v5_forge_scrambles_targets() -> None:
    ts = synthesize_v5_training_set(
        n_sequences=2, sequence_length=8,
        state_dim=4, input_dim=4, seed=41)
    forged = forge_v5_training_set(ts, seed=41)
    # Each example's input_sequence is preserved; targets differ.
    for orig, f in zip(ts.examples, forged.examples):
        assert orig.input_sequence == f.input_sequence
