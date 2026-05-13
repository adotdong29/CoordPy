"""W54 M2 — Multi-Hop Translator V4 tests."""

from __future__ import annotations

from coordpy.multi_hop_translator_v4 import (
    W54_DEFAULT_MH_V4_BACKENDS,
    W54_MH_V4_VERIFIER_FAILURE_MODES,
    build_unfitted_hex_translator,
    disagreement_compromise_arbitration,
    emit_multi_hop_v4_witness,
    fit_hex_translator,
    score_hex_fidelity,
    synthesize_hex_training_set,
    verify_multi_hop_v4_witness,
)


def test_hex_translator_has_6_backends() -> None:
    tr = build_unfitted_hex_translator(
        code_dim=4, feature_dim=4, seed=1)
    assert len(tr.backends) == 6
    assert tr.backends == W54_DEFAULT_MH_V4_BACKENDS


def test_hex_chain_len5_fidelity() -> None:
    ts = synthesize_hex_training_set(
        n_examples=12, code_dim=6, feature_dim=6, seed=1)
    tr, _ = fit_hex_translator(ts, n_steps=64, seed=1)
    fid = score_hex_fidelity(tr, ts.examples[:8])
    assert 0.0 <= fid.chain_len5_fid_mean <= 1.0
    assert fid.chain_len5_fid_mean >= 0.5


def test_compromise_arbitration_picks_subset() -> None:
    tr = build_unfitted_hex_translator(
        code_dim=4, feature_dim=4, seed=1)
    paths = (
        ("A", "F"),
        ("A", "B", "F"),
        ("A", "B", "C", "F"),
    )
    res = disagreement_compromise_arbitration(
        tr, paths=paths,
        input_vec=[0.1, 0.2, 0.3, 0.4],
        feature_dim=4,
        compromise_floor=0.5)
    assert res.n_paths_total == 3
    # pick rate + abstain rate sums to 1
    assert (
        (res.n_paths_selected > 0 and not res.abstain)
        or res.abstain)


def test_w54_mh_v4_verifier_failure_modes_count() -> None:
    assert len(W54_MH_V4_VERIFIER_FAILURE_MODES) == 6


def test_w54_mh_v4_witness_cid_deterministic() -> None:
    ts = synthesize_hex_training_set(
        n_examples=8, code_dim=4, feature_dim=4, seed=2)
    tr, _ = fit_hex_translator(ts, n_steps=24, seed=2)
    w1 = emit_multi_hop_v4_witness(
        translator=tr, examples=ts.examples[:4])
    w2 = emit_multi_hop_v4_witness(
        translator=tr, examples=ts.examples[:4])
    assert w1.cid() == w2.cid()


def test_w54_mh_v4_witness_rejects_wrong_translator() -> None:
    ts = synthesize_hex_training_set(
        n_examples=8, code_dim=4, feature_dim=4, seed=3)
    tr, _ = fit_hex_translator(ts, n_steps=24, seed=3)
    w = emit_multi_hop_v4_witness(
        translator=tr, examples=ts.examples[:4])
    v_bad = verify_multi_hop_v4_witness(
        w, expected_translator_cid="ff" * 32,
        expected_n_backends=6,
        expected_chain_length=5)
    assert v_bad["ok"] is False
    assert (
        "w54_mh_v4_translator_cid_mismatch"
        in v_bad["failures"])
