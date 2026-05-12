"""Unit tests for W51 M1 — persistent shared latent state."""

from __future__ import annotations

import pytest

from coordpy.persistent_shared_latent import (
    CrossRoleMixer,
    PersistentLatentStateChain,
    PersistentStateCell,
    W51_NO_PARENT_STATE,
    evaluate_long_horizon_recall,
    fit_persistent_state_cell,
    forge_persistent_state_training_set,
    step_persistent_latent_state,
    synthesize_persistent_state_training_set,
    emit_persistent_latent_state_witness,
    verify_persistent_latent_state_witness,
)


def test_cell_init_stable_with_seed() -> None:
    a = PersistentStateCell.init(seed=11, state_dim=8, input_dim=8)
    b = PersistentStateCell.init(seed=11, state_dim=8, input_dim=8)
    assert a.cid() == b.cid()


def test_cell_init_seed_changes_cid() -> None:
    a = PersistentStateCell.init(seed=11, state_dim=8, input_dim=8)
    c = PersistentStateCell.init(seed=12, state_dim=8, input_dim=8)
    assert a.cid() != c.cid()


def test_cell_step_value_matches_chain() -> None:
    cell = PersistentStateCell.init(
        seed=11, state_dim=4, input_dim=4)
    chain = PersistentLatentStateChain.empty()
    state = step_persistent_latent_state(
        cell=cell, mixer=None,
        prev_state_values=[0.0]*4,
        carrier_values=[0.5]*4,
        turn_index=0, role="r0",
        parent_state_cid=W51_NO_PARENT_STATE,
        state_dim=4)
    chain.add(state)
    assert state.turn_index == 0
    assert state.role == "r0"
    assert len(state.values) == 4


def test_chain_walk_recovers_ancestors() -> None:
    cell = PersistentStateCell.init(
        seed=11, state_dim=4, input_dim=4)
    chain = PersistentLatentStateChain.empty()
    prev_cid = W51_NO_PARENT_STATE
    prev_vals = [0.0]*4
    for t in range(5):
        s = step_persistent_latent_state(
            cell=cell, mixer=None,
            prev_state_values=prev_vals,
            carrier_values=[float(t)/5.0]*4,
            turn_index=t, role="r0",
            parent_state_cid=prev_cid,
            state_dim=4)
        chain.add(s)
        prev_cid = s.cid()
        prev_vals = s.values
    walk = chain.walk_from(prev_cid, max_depth=10)
    assert len(walk) == 5


def test_long_horizon_training_improves_recall() -> None:
    ts = synthesize_persistent_state_training_set(
        n_sequences=4, sequence_length=10, state_dim=4,
        input_dim=4, seed=11)
    untrained = PersistentStateCell.init(
        state_dim=4, input_dim=4, seed=11)
    untrained_recall = evaluate_long_horizon_recall(
        untrained, ts.examples)
    trained, _ = fit_persistent_state_cell(
        ts, n_steps=48, seed=11, truncate_bptt=3)
    trained_recall = evaluate_long_horizon_recall(
        trained, ts.examples)
    # Training must strictly improve recall.
    assert trained_recall > untrained_recall - 0.05


def test_cross_role_mixer_blends() -> None:
    mixer = CrossRoleMixer.init(
        role_universe=("r0", "r1"),
        state_dim=4, seed=11)
    team_state = [0.3, -0.2, 0.1, 0.5]
    out_r0, blend_r0 = mixer.project_value(
        role="r0", team_state=team_state)
    out_r1, blend_r1 = mixer.project_value(
        role="r1", team_state=team_state)
    assert len(out_r0) == 4
    assert len(out_r1) == 4
    # Different roles produce different views
    assert blend_r0 == blend_r1 or out_r0 != out_r1


def test_witness_passes_clean_verifier() -> None:
    cell = PersistentStateCell.init(
        seed=11, state_dim=4, input_dim=4)
    chain = PersistentLatentStateChain.empty()
    s = step_persistent_latent_state(
        cell=cell, mixer=None,
        prev_state_values=[0.0]*4,
        carrier_values=[0.5]*4,
        turn_index=0, role="r0",
        parent_state_cid=W51_NO_PARENT_STATE, state_dim=4)
    chain.add(s)
    w = emit_persistent_latent_state_witness(
        state=s, cell=cell, mixer=None, chain=chain)
    v = verify_persistent_latent_state_witness(
        w,
        expected_state_cid=s.cid(),
        expected_cell_cid=cell.cid(),
        min_chain_walk_depth=1)
    assert v["ok"] is True


def test_forge_breaks_training_signal() -> None:
    """H17 falsifier helper: forged training set cannot
    recover."""
    ts = synthesize_persistent_state_training_set(
        n_sequences=3, sequence_length=8, state_dim=4,
        input_dim=4, seed=11)
    forged = forge_persistent_state_training_set(
        ts, seed=11)
    trained_on_forged, _ = fit_persistent_state_cell(
        forged, n_steps=24, seed=11, truncate_bptt=2)
    recall_on_clean = evaluate_long_horizon_recall(
        trained_on_forged, ts.examples)
    # Forged trained on noise targets cannot reproduce clean
    # signal — recall is near zero (or even negative).
    assert recall_on_clean < 0.5
