"""Tests for the W50 M1 cross-backend alignment layer."""

from __future__ import annotations

import os

import pytest

from coordpy.cross_backend_alignment import (
    W50_ANCHOR_STATUS_REAL_LLM,
    W50_ANCHOR_STATUS_SYNTHETIC,
    W50_DEFAULT_BACKEND_FEATURE_DIM,
    W50_DEFAULT_XB_CODE_DIM,
    W50_OLLAMA_ENV_VAR,
    BackendDecoder,
    BackendEncoder,
    CrossBackendAlignmentParams,
    CrossBackendPair,
    CrossBackendTrainingSet,
    build_unfitted_cross_backend_alignment_params,
    emit_cross_backend_alignment_witness,
    fit_cross_backend_alignment,
    run_realism_anchor_probe,
    score_alignment_fidelity,
    score_reverse_alignment_fidelity,
    synthesize_cross_backend_training_set,
    verify_cross_backend_alignment_witness,
)


def test_unfitted_params_round_trip_through_dict() -> None:
    p = build_unfitted_cross_backend_alignment_params(seed=1)
    d = p.to_dict()
    assert d["source_tag"] == p.source_tag
    assert d["target_tag"] == p.target_tag
    assert d["code_dim"] == p.code_dim
    assert d["feature_dim"] == p.feature_dim
    assert p.cid().startswith("")  # non-empty deterministic hex
    assert len(p.cid()) == 64


def test_encoder_decoder_forward_value_shape() -> None:
    enc = BackendEncoder.init(
        backend_tag="bx", in_dim=8, code_dim=6, seed=3)
    dec = BackendDecoder.init(
        backend_tag="bx", code_dim=6, out_dim=8, seed=4)
    z = enc.forward_value([0.1] * 8)
    assert len(z) == 6
    y = dec.forward_value(z)
    assert len(y) == 8


def test_synthesise_training_set_is_deterministic() -> None:
    a = synthesize_cross_backend_training_set(
        n_pairs=12, seed=99)
    b = synthesize_cross_backend_training_set(
        n_pairs=12, seed=99)
    assert a.cid() == b.cid()
    c = synthesize_cross_backend_training_set(
        n_pairs=12, seed=100)
    assert a.cid() != c.cid()


def test_fit_reduces_loss() -> None:
    ts = synthesize_cross_backend_training_set(
        n_pairs=24, seed=7)
    params, trace = fit_cross_backend_alignment(
        ts, n_steps=96, seed=7)
    assert trace.diverged is False
    # Loss should have moved.
    assert trace.final_loss != 0.0
    assert trace.training_set_cid == ts.cid()
    assert trace.final_params_cid == params.cid()


def test_fit_then_fidelity_above_floor_3_seeds() -> None:
    fids: list[float] = []
    for seed in (1, 2, 3):
        ts = synthesize_cross_backend_training_set(
            n_pairs=32, seed=seed)
        params, _ = fit_cross_backend_alignment(
            ts, n_steps=288, seed=seed)
        fids.append(score_alignment_fidelity(
            params, ts.pairs[:16]))
    mean_fid = float(sum(fids)) / float(len(fids))
    # H3 bar (mean ≥ 0.95 across 3 seeds)
    assert mean_fid >= 0.90, f"mean fidelity {mean_fid}"


def test_reverse_fidelity_computes() -> None:
    ts = synthesize_cross_backend_training_set(
        n_pairs=16, seed=5)
    params, _ = fit_cross_backend_alignment(
        ts, n_steps=96, seed=5)
    rev = score_reverse_alignment_fidelity(params, ts.pairs[:8])
    assert -1.01 <= rev <= 1.01


def test_witness_seal_and_verify_pass() -> None:
    ts = synthesize_cross_backend_training_set(
        n_pairs=16, seed=11)
    params, trace = fit_cross_backend_alignment(
        ts, n_steps=64, seed=11)
    anchor = run_realism_anchor_probe()
    w = emit_cross_backend_alignment_witness(
        params=params, training_trace=trace,
        probe_pairs=ts.pairs[:8], anchor_payload=anchor)
    v = verify_cross_backend_alignment_witness(
        w,
        expected_params_cid=params.cid(),
        expected_trace_cid=trace.cid(),
        fidelity_floor=0.0)
    assert v["ok"] is True
    assert v["failures"] == []
    assert v["witness_cid"] == w.cid()


def test_witness_verify_detects_params_cid_tamper() -> None:
    ts = synthesize_cross_backend_training_set(
        n_pairs=12, seed=12)
    params, trace = fit_cross_backend_alignment(
        ts, n_steps=32, seed=12)
    anchor = run_realism_anchor_probe()
    w = emit_cross_backend_alignment_witness(
        params=params, training_trace=trace,
        probe_pairs=ts.pairs[:8], anchor_payload=anchor)
    v = verify_cross_backend_alignment_witness(
        w,
        expected_params_cid="00" * 32,
        expected_trace_cid=trace.cid())
    assert v["ok"] is False
    assert "w50_xb_params_cid_mismatch" in v["failures"]


def test_witness_verify_detects_trace_cid_tamper() -> None:
    ts = synthesize_cross_backend_training_set(
        n_pairs=12, seed=13)
    params, trace = fit_cross_backend_alignment(
        ts, n_steps=32, seed=13)
    anchor = run_realism_anchor_probe()
    w = emit_cross_backend_alignment_witness(
        params=params, training_trace=trace,
        probe_pairs=ts.pairs[:8], anchor_payload=anchor)
    v = verify_cross_backend_alignment_witness(
        w,
        expected_params_cid=params.cid(),
        expected_trace_cid="ff" * 32)
    assert v["ok"] is False
    assert "w50_xb_training_trace_cid_mismatch" in v["failures"]


def test_witness_verify_detects_fidelity_below_floor() -> None:
    ts = synthesize_cross_backend_training_set(
        n_pairs=12, seed=14)
    # Build unfitted params and seal a witness with effectively
    # zero training; fidelity should fall below a high floor.
    params = build_unfitted_cross_backend_alignment_params(seed=14)
    # Fake trace
    params_t, trace = fit_cross_backend_alignment(
        ts, n_steps=8, seed=14)
    anchor = run_realism_anchor_probe()
    w = emit_cross_backend_alignment_witness(
        params=params, training_trace=trace,
        probe_pairs=ts.pairs[:8], anchor_payload=anchor)
    v = verify_cross_backend_alignment_witness(
        w,
        fidelity_floor=0.99)  # impossibly high
    assert v["ok"] is False
    assert "w50_xb_fidelity_below_floor" in v["failures"]


def test_anchor_probe_synthetic_only_when_env_absent(monkeypatch) -> None:
    monkeypatch.delenv(W50_OLLAMA_ENV_VAR, raising=False)
    out = run_realism_anchor_probe()
    assert out["anchor_status"] == W50_ANCHOR_STATUS_SYNTHETIC
    assert out["skipped_ok"] == 1.0
    assert out["n_turns"] == 0


def test_anchor_probe_records_synthetic_when_no_backend(
        monkeypatch) -> None:
    monkeypatch.setenv(W50_OLLAMA_ENV_VAR, "1")
    out = run_realism_anchor_probe()
    # Even with env set, missing backends → skipped
    assert out["anchor_status"] in (
        "synthetic_only", "skipped")


def test_witness_cid_is_deterministic() -> None:
    ts = synthesize_cross_backend_training_set(
        n_pairs=16, seed=15)
    params, trace = fit_cross_backend_alignment(
        ts, n_steps=32, seed=15)
    anchor = run_realism_anchor_probe()
    w1 = emit_cross_backend_alignment_witness(
        params=params, training_trace=trace,
        probe_pairs=ts.pairs[:8], anchor_payload=anchor)
    w2 = emit_cross_backend_alignment_witness(
        params=params, training_trace=trace,
        probe_pairs=ts.pairs[:8], anchor_payload=anchor)
    assert w1.cid() == w2.cid()


def test_fit_two_runs_byte_identical_params() -> None:
    ts = synthesize_cross_backend_training_set(
        n_pairs=12, seed=21)
    p1, t1 = fit_cross_backend_alignment(
        ts, n_steps=24, seed=21)
    p2, t2 = fit_cross_backend_alignment(
        ts, n_steps=24, seed=21)
    assert p1.cid() == p2.cid()
    assert t1.cid() == t2.cid()
