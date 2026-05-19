"""W83 — composed learned memory tests."""

from __future__ import annotations


def test_w83_composed_module_cid_deterministic():
    from coordpy.composed_learned_memory_v1 import (
        build_composed_learned_memory_module_v1,
    )
    a = build_composed_learned_memory_module_v1(seed=42)
    b = build_composed_learned_memory_module_v1(seed=42)
    assert str(a.cid()) == str(b.cid())
    c = build_composed_learned_memory_module_v1(seed=43)
    assert str(a.cid()) != str(c.cid())


def test_w83_composed_module_forward_shapes():
    from coordpy.composed_learned_memory_v1 import (
        build_composed_learned_memory_module_v1,
        build_composed_long_horizon_dataset_v1,
    )
    m = build_composed_learned_memory_module_v1()
    X, Y = build_composed_long_horizon_dataset_v1(
        n_sequences=2, seq_len=14)
    H, S_seq, R, Yhat = m.forward_sequence(X[0])
    assert H.shape == (14, m.hidden_dim)
    assert S_seq.shape == (
        14 + 1, m.K_slots, m.memory_dim)
    assert R.shape == (14, m.memory_dim)
    assert Yhat.shape == (14, m.output_dim)


def test_w83_composed_training_reduces_loss():
    from coordpy.composed_learned_memory_v1 import (
        build_composed_learned_memory_module_v1,
        build_composed_long_horizon_dataset_v1,
        train_composed_learned_memory_module,
    )
    m = build_composed_learned_memory_module_v1(seed=99)
    X, Y = build_composed_long_horizon_dataset_v1(
        n_sequences=10, seq_len=14, seed=100)
    m_post, rep = train_composed_learned_memory_module(
        module=m,
        train_sequences=X.tolist(),
        train_targets=Y.tolist(),
        n_iters=40)
    assert float(rep.post_loss) < float(rep.pre_loss)
    assert bool(rep.converged)


def test_w83_composed_training_deterministic_on_seed():
    from coordpy.composed_learned_memory_v1 import (
        build_composed_learned_memory_module_v1,
        build_composed_long_horizon_dataset_v1,
        train_composed_learned_memory_module,
    )
    m_a = build_composed_learned_memory_module_v1(seed=7)
    m_b = build_composed_learned_memory_module_v1(seed=7)
    X, Y = build_composed_long_horizon_dataset_v1(
        n_sequences=6, seq_len=14, seed=8)
    a, _ = train_composed_learned_memory_module(
        module=m_a,
        train_sequences=X.tolist(),
        train_targets=Y.tolist(),
        n_iters=30)
    b, _ = train_composed_learned_memory_module(
        module=m_b,
        train_sequences=X.tolist(),
        train_targets=Y.tolist(),
        n_iters=30)
    assert str(a.cid()) == str(b.cid())


def test_w83_composed_beats_v2_and_ridge_on_default_dataset():
    from coordpy.composed_learned_memory_v1 import (
        build_composed_learned_memory_module_v1,
        build_composed_long_horizon_dataset_v1,
        compare_composed_vs_baselines_v1,
        train_composed_learned_memory_module,
    )
    m = build_composed_learned_memory_module_v1(
        seed=83_001_001)
    X, Y = build_composed_long_horizon_dataset_v1(
        n_sequences=14, seq_len=18,
        seed=83_001_001 + 1)
    m, _ = train_composed_learned_memory_module(
        module=m,
        train_sequences=X.tolist(),
        train_targets=Y.tolist(),
        n_iters=80)
    rep = compare_composed_vs_baselines_v1(
        composed=m,
        eval_sequences=X.tolist(),
        eval_targets=Y.tolist(),
        baseline_train_iters=70)
    assert bool(rep.composed_beats_v2), rep.to_dict()
    assert bool(rep.composed_beats_ridge), rep.to_dict()


def test_w83_composed_memory_compressed_snapshot_cid_changes():
    from coordpy.composed_learned_memory_v1 import (
        build_composed_learned_memory_module_v1,
        build_composed_long_horizon_dataset_v1,
    )
    m = build_composed_learned_memory_module_v1()
    X, _ = build_composed_long_horizon_dataset_v1(
        n_sequences=2, seq_len=14, seed=1)
    a = m.compressed_snapshot_cid(X=X[0])
    b = m.compressed_snapshot_cid(X=X[1])
    c = m.compressed_snapshot_cid(X=X[0])
    assert a == c
    assert a != b


def test_w83_composed_witness_emitted():
    from coordpy.composed_learned_memory_v1 import (
        build_composed_learned_memory_module_v1,
        emit_composed_learned_memory_witness_v1,
    )
    m = build_composed_learned_memory_module_v1()
    w = emit_composed_learned_memory_witness_v1(module=m)
    assert len(w.cid()) == 64
