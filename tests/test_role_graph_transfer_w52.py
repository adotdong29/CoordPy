"""Per-module tests for the W52 role-graph transfer."""

from __future__ import annotations

import pytest

from coordpy.role_graph_transfer import (
    RoleGraphMixer,
    build_unfitted_role_graph_mixer,
    emit_role_graph_witness,
    evaluate_equal_weight_accuracy,
    evaluate_role_graph_accuracy,
    fit_role_graph_mixer,
    forge_role_graph_training_set,
    synthesize_role_graph_training_set,
    verify_role_graph_witness,
)


def test_role_graph_mixer_init_includes_all_pairs() -> None:
    mixer = build_unfitted_role_graph_mixer(
        role_universe=("r0", "r1", "r2"),
        state_dim=4, seed=1)
    # 3*3 = 9 with self-loops
    assert len(mixer.edges) == 9


def test_role_graph_project_value_returns_state_dim() -> None:
    mixer = build_unfitted_role_graph_mixer(
        role_universe=("r0", "r1", "r2"),
        state_dim=4, seed=2)
    x = [0.1, 0.2, 0.3, 0.4]
    out = mixer.project_value(
        src_role="r0", dst_role="r1", x=x)
    assert len(out) == 4


def test_role_graph_fit_decreases_loss() -> None:
    ts = synthesize_role_graph_training_set(
        role_universe=("r0", "r1", "r2"),
        state_dim=4, n_examples_per_edge=4, seed=3)
    mixer, trace = fit_role_graph_mixer(
        ts, n_steps=48, seed=3)
    assert not trace.diverged
    assert trace.loss_tail[-1] < trace.loss_head[0]


def test_role_graph_beats_equal_weight_after_training() -> None:
    ts = synthesize_role_graph_training_set(
        role_universe=("r0", "r1", "r2", "r3"),
        state_dim=4, n_examples_per_edge=4, seed=4)
    mixer, _ = fit_role_graph_mixer(
        ts, n_steps=64, seed=4)
    rg_acc = evaluate_role_graph_accuracy(mixer, ts.examples)
    ew_acc = evaluate_equal_weight_accuracy(mixer, ts.examples)
    assert rg_acc > ew_acc


def test_role_graph_witness_round_trips() -> None:
    ts = synthesize_role_graph_training_set(
        role_universe=("r0", "r1"),
        state_dim=4, n_examples_per_edge=3, seed=5)
    mixer, trace = fit_role_graph_mixer(
        ts, n_steps=24, seed=5)
    w = emit_role_graph_witness(
        mixer=mixer, training_trace=trace,
        examples=ts.examples)
    v = verify_role_graph_witness(
        w, expected_mixer_cid=mixer.cid(),
        expected_n_edges=len(mixer.edges))
    assert v["ok"] is True


def test_forge_role_graph_changes_cid() -> None:
    ts = synthesize_role_graph_training_set(
        role_universe=("r0", "r1"),
        state_dim=4, n_examples_per_edge=2, seed=6)
    forged = forge_role_graph_training_set(ts, seed=6)
    assert ts.cid() != forged.cid()
