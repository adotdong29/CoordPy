"""Unit tests for W51 M2 — triple-backend translator."""

from __future__ import annotations

import pytest

from coordpy.cross_backend_translator import (
    TripleBackendTranslator,
    W51_TRIPLE_BACKEND_VERIFIER_FAILURE_MODES,
    build_unfitted_triple_backend_translator,
    emit_triple_backend_translator_witness,
    fit_triple_backend_translator,
    forge_triple_backend_training_set,
    run_triple_realism_anchor_probe,
    score_triple_backend_fidelity,
    synthesize_triple_backend_training_set,
    verify_triple_backend_translator_witness,
)


def test_unfitted_translator_stable_seed() -> None:
    a = build_unfitted_triple_backend_translator(seed=11)
    b = build_unfitted_triple_backend_translator(seed=11)
    assert a.cid() == b.cid()


def test_unfitted_translator_low_fidelity() -> None:
    t = build_unfitted_triple_backend_translator(
        feature_dim=8, code_dim=8, seed=11)
    ts = synthesize_triple_backend_training_set(
        n_examples=8, feature_dim=8, code_dim=8, seed=11)
    fid = score_triple_backend_fidelity(t, ts.examples)
    # Untrained = near chance
    assert abs(fid.direct_ab) < 0.5
    assert abs(fid.direct_ac) < 0.5


def test_trained_translator_high_direct_fidelity() -> None:
    ts = synthesize_triple_backend_training_set(
        n_examples=24, feature_dim=8, code_dim=8, seed=11)
    trained, _ = fit_triple_backend_translator(
        ts, n_steps=192, seed=11)
    fid = score_triple_backend_fidelity(trained, ts.examples)
    assert fid.direct_ab > 0.6
    assert fid.direct_ac > 0.6


def test_trained_translator_transitivity_gap_bounded() -> None:
    ts = synthesize_triple_backend_training_set(
        n_examples=24, feature_dim=8, code_dim=8, seed=11)
    trained, _ = fit_triple_backend_translator(
        ts, n_steps=192, seed=11)
    fid = score_triple_backend_fidelity(trained, ts.examples)
    # Transitivity gap stays bounded with the joint loss
    assert fid.transitivity_gap < 0.20


def test_forged_translator_cannot_recover() -> None:
    ts = synthesize_triple_backend_training_set(
        n_examples=16, feature_dim=8, code_dim=8, seed=11)
    forged = forge_triple_backend_training_set(ts, seed=11)
    translator, _ = fit_triple_backend_translator(
        forged, n_steps=96, seed=11)
    fid = score_triple_backend_fidelity(
        translator, ts.examples)
    # Forged training cannot reproduce clean shift pattern
    assert abs(fid.direct_ab) < 0.5


def test_witness_passes_clean_verifier() -> None:
    ts = synthesize_triple_backend_training_set(
        n_examples=16, feature_dim=8, code_dim=8, seed=11)
    trained, trace = fit_triple_backend_translator(
        ts, n_steps=96, seed=11)
    w = emit_triple_backend_translator_witness(
        translator=trained, training_trace=trace,
        probes=ts.examples[:8])
    v = verify_triple_backend_translator_witness(
        w, expected_translator_cid=trained.cid(),
        expected_trace_cid=trace.cid(),
        direct_fidelity_floor=0.3,
        transitivity_gap_ceiling=0.50,
        expected_tags=(
            ts.tag_a, ts.tag_b, ts.tag_c))
    assert v["ok"] is True


def test_realism_anchor_skips_when_no_backends() -> None:
    payload = run_triple_realism_anchor_probe()
    assert payload["anchor_status"] == "synthetic_only"
    assert payload["skipped_ok"] == 1.0


def test_translator_verifier_has_9_failure_modes() -> None:
    assert len(W51_TRIPLE_BACKEND_VERIFIER_FAILURE_MODES) == 9
    assert len(set(
        W51_TRIPLE_BACKEND_VERIFIER_FAILURE_MODES)) == 9
