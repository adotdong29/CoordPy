"""W83 — recurrent slot reconstruction tests."""

from __future__ import annotations


def test_w83_slot_recon_module_cid_deterministic():
    from coordpy.recurrent_slot_reconstruction_v1 import (
        build_recurrent_slot_reconstruction_head_v1,
    )
    a = build_recurrent_slot_reconstruction_head_v1(seed=42)
    b = build_recurrent_slot_reconstruction_head_v1(seed=42)
    assert str(a.cid()) == str(b.cid())
    c = build_recurrent_slot_reconstruction_head_v1(seed=43)
    assert str(a.cid()) != str(c.cid())


def test_w83_slot_recon_forward_shapes():
    from coordpy.recurrent_slot_reconstruction_v1 import (
        build_recurrent_slot_reconstruction_head_v1,
        build_cross_offset_reconstruction_dataset_v1,
    )
    head = build_recurrent_slot_reconstruction_head_v1()
    Ss, Qs, _ = build_cross_offset_reconstruction_dataset_v1(
        n_sequences=2, seq_len=10)
    H, Attn, R, Y = head.forward_query_sequence(
        S=Ss[0], Q=Qs[0])
    assert H.shape == (10, head.hidden_dim)
    assert Attn.shape == (10, head.K_slots)
    assert R.shape == (10, head.memory_dim)
    assert Y.shape == (10, head.output_dim)


def test_w83_slot_recon_training_reduces_loss():
    from coordpy.recurrent_slot_reconstruction_v1 import (
        build_recurrent_slot_reconstruction_head_v1,
        build_cross_offset_reconstruction_dataset_v1,
        train_recurrent_slot_reconstruction_head,
    )
    head = build_recurrent_slot_reconstruction_head_v1()
    Ss, Qs, Ys = build_cross_offset_reconstruction_dataset_v1(
        n_sequences=8, seq_len=14)
    head_post, rep = train_recurrent_slot_reconstruction_head(
        module=head,
        train_slots=[s.tolist() for s in Ss],
        train_queries=[q.tolist() for q in Qs],
        train_targets=[y.tolist() for y in Ys],
        n_iters=40)
    assert float(rep.post_loss) < float(rep.pre_loss)
    assert bool(rep.converged)


def test_w83_slot_recon_beats_query_only_ridge_and_nearest():
    from coordpy.recurrent_slot_reconstruction_v1 import (
        build_recurrent_slot_reconstruction_head_v1,
        build_cross_offset_reconstruction_dataset_v1,
        compare_recurrent_slot_reconstruction_vs_baselines_v1,
        train_recurrent_slot_reconstruction_head,
    )
    head = build_recurrent_slot_reconstruction_head_v1(
        seed=83_002_001)
    Ss, Qs, Ys = build_cross_offset_reconstruction_dataset_v1(
        n_sequences=14, seq_len=14, seed=83_002_001 + 1)
    head, _ = train_recurrent_slot_reconstruction_head(
        module=head,
        train_slots=[s.tolist() for s in Ss],
        train_queries=[q.tolist() for q in Qs],
        train_targets=[y.tolist() for y in Ys],
        n_iters=120)
    rep = (
        compare_recurrent_slot_reconstruction_vs_baselines_v1(
            head=head,
            eval_slots=[s.tolist() for s in Ss],
            eval_queries=[q.tolist() for q in Qs],
            eval_targets=[y.tolist() for y in Ys]))
    assert bool(rep.head_beats_ridge_query_only), rep.to_dict()
    assert bool(rep.head_beats_nearest_slot), rep.to_dict()


def test_w83_slot_recon_competitive_with_full_information_ridge():
    from coordpy.recurrent_slot_reconstruction_v1 import (
        build_recurrent_slot_reconstruction_head_v1,
        build_cross_offset_reconstruction_dataset_v1,
        compare_recurrent_slot_reconstruction_vs_baselines_v1,
        train_recurrent_slot_reconstruction_head,
    )
    head = build_recurrent_slot_reconstruction_head_v1(
        seed=83_002_002)
    Ss, Qs, Ys = build_cross_offset_reconstruction_dataset_v1(
        n_sequences=14, seq_len=14, seed=83_002_002 + 1)
    head, _ = train_recurrent_slot_reconstruction_head(
        module=head,
        train_slots=[s.tolist() for s in Ss],
        train_queries=[q.tolist() for q in Qs],
        train_targets=[y.tolist() for y in Ys],
        n_iters=120)
    rep = (
        compare_recurrent_slot_reconstruction_vs_baselines_v1(
            head=head,
            eval_slots=[s.tolist() for s in Ss],
            eval_queries=[q.tolist() for q in Qs],
            eval_targets=[y.tolist() for y in Ys]))
    assert bool(
        rep.head_competitive_with_ridge_full_features), (
            rep.to_dict())


def test_w83_slot_recon_witness_emitted():
    from coordpy.recurrent_slot_reconstruction_v1 import (
        build_recurrent_slot_reconstruction_head_v1,
        emit_recurrent_slot_reconstruction_witness_v1,
    )
    head = build_recurrent_slot_reconstruction_head_v1()
    w = emit_recurrent_slot_reconstruction_witness_v1(head=head)
    assert len(w.cid()) == 64
