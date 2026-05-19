"""W83 — cross-runtime learned hidden-state projector tests."""

from __future__ import annotations


def test_w83_projector_fit_is_deterministic():
    import numpy as np
    from coordpy.cross_runtime_hidden_state_projector_v1 import (
        fit_learned_hidden_state_projector_v1,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8)).astype(np.float64)
    Y = (X @ rng.standard_normal((8, 12)).astype(np.float64)
         + rng.standard_normal((50, 12)) * 0.05)
    a = fit_learned_hidden_state_projector_v1(
        source_states=X, target_states=Y)
    b = fit_learned_hidden_state_projector_v1(
        source_states=X, target_states=Y)
    assert str(a.cid()) == str(b.cid())
    assert int(a.source_hidden_dim) == 8
    assert int(a.target_hidden_dim) == 12


def test_w83_projector_post_loss_lower_than_pre():
    import numpy as np
    from coordpy.cross_runtime_hidden_state_projector_v1 import (
        fit_learned_hidden_state_projector_v1,
    )
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 6)).astype(np.float64)
    Y = (X @ rng.standard_normal((6, 10)).astype(np.float64)
         + rng.standard_normal((100, 10)) * 0.05)
    a = fit_learned_hidden_state_projector_v1(
        source_states=X, target_states=Y)
    assert float(a.train_loss_post) < float(a.train_loss_pre)


def test_w83_projector_beats_w82_on_anchor_cosine_and_classifier():
    from coordpy.cross_runtime_hidden_state_projector_v1 import (
        run_cross_runtime_projector_bench_v1,
    )
    rep = run_cross_runtime_projector_bench_v1()
    assert bool(rep.learned_beats_w82_cosine), rep.to_dict()
    assert bool(
        rep.learned_beats_w82_classifier), rep.to_dict()


def test_w83_projector_witness_emitted():
    import numpy as np
    from coordpy.cross_runtime_hidden_state_projector_v1 import (
        emit_learned_hidden_state_projector_witness_v1,
        fit_learned_hidden_state_projector_v1,
        run_cross_runtime_projector_bench_v1,
    )
    rng = np.random.default_rng(13)
    X = rng.standard_normal((40, 5))
    Y = rng.standard_normal((40, 7))
    proj = fit_learned_hidden_state_projector_v1(
        source_states=X, target_states=Y)
    rep = run_cross_runtime_projector_bench_v1()
    w = emit_learned_hidden_state_projector_witness_v1(
        projector=proj, bench=rep)
    assert len(w.cid()) == 64
