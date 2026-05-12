"""W53 M2 multi_hop_translator_v3 tests."""

from __future__ import annotations

from coordpy.multi_hop_translator_v3 import (
    W53_DEFAULT_MH_V3_BACKENDS,
    build_unfitted_quint_translator,
    emit_multi_hop_v3_witness,
    fit_quint_translator,
    score_quint_fidelity,
    synthesize_quint_training_set,
    uncertainty_aware_arbitration,
    verify_multi_hop_v3_witness,
)


def test_quint_translator_has_5_backends() -> None:
    tr = build_unfitted_quint_translator(seed=11)
    assert len(tr.backends) == 5
    assert tr.backends == W53_DEFAULT_MH_V3_BACKENDS


def test_quint_translator_has_20_edges() -> None:
    tr = build_unfitted_quint_translator(seed=11)
    # 5 backends × 4 dst each (excluding self) = 20.
    assert len(tr.edges) == 20


def test_quint_chain_len4_fidelity_finite() -> None:
    ts = synthesize_quint_training_set(
        n_examples=4, code_dim=6,
        feature_dim=6, seed=11)
    tr = build_unfitted_quint_translator(
        code_dim=6, feature_dim=6, seed=11)
    fid = score_quint_fidelity(tr, ts.examples)
    # Cosine bounded.
    assert -1.0 <= fid.chain_len4_fid_mean <= 1.0


def test_quint_witness_emit_and_verify() -> None:
    ts = synthesize_quint_training_set(
        n_examples=4, code_dim=6,
        feature_dim=6, seed=13)
    tr, _ = fit_quint_translator(
        ts, n_steps=64, seed=13)
    w = emit_multi_hop_v3_witness(
        translator=tr, examples=ts.examples)
    v = verify_multi_hop_v3_witness(
        w,
        expected_translator_cid=tr.cid(),
        expected_n_backends=5,
        expected_chain_length=4)
    assert v["ok"] is True


def test_uncertainty_arbitration_reports_per_dim_std() -> None:
    ts = synthesize_quint_training_set(
        n_examples=4, code_dim=4,
        feature_dim=4, seed=17)
    tr = build_unfitted_quint_translator(
        code_dim=4, feature_dim=4, seed=17)
    paths = (
        ("A", "E"), ("A", "B", "E"),
        ("A", "B", "C", "E"))
    arb = uncertainty_aware_arbitration(
        tr, paths=paths,
        input_vec=ts.examples[0].feature_by_backend["A"],
        feature_dim=4)
    assert len(arb.prediction) == 4
    assert len(arb.per_dim_std) == 4
    assert all(s >= 0.0 for s in arb.per_dim_std)
    assert 0.0 <= arb.aggregate_confidence <= 1.0
