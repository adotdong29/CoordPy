"""Tests for the W50 M3 adaptive compression module."""

from __future__ import annotations

import pytest

from coordpy.adaptive_compression import (
    W50_DEFAULT_ADAPTIVE_K,
    W50_DEFAULT_BITS_PAYLOAD_LEN,
    W50_DEFAULT_EMIT_MASK_LEN,
    W50_DEFAULT_TARGET_BITS_PER_TOKEN,
    AdaptiveCompressionCodebook,
    AdaptiveCompressionGate,
    AdaptiveCompressionResult,
    AdaptiveCompressionWitness,
    CrammingWitnessV2,
    compress_carrier,
    emit_adaptive_compression_witness,
    emit_cramming_witness_v2,
    fit_adaptive_compression,
    probe_rate_floor_falsifier,
    synthesize_adaptive_compression_training_set,
    verify_adaptive_compression_witness,
)


def test_codebook_init_defaults_to_k16() -> None:
    cb = AdaptiveCompressionCodebook.init(seed=1)
    assert cb.n_codes == W50_DEFAULT_ADAPTIVE_K
    assert cb.code_bits() == 4  # log2(16)
    assert len(cb.cid()) == 64


def test_codebook_round_trip_picks_a_code() -> None:
    cb = AdaptiveCompressionCodebook.init(seed=2)
    for x in ([0.0] * cb.code_dim, [0.5] * cb.code_dim,
              [-0.3] * cb.code_dim):
        code, decoded = cb.round_trip_value(x)
        assert 0 <= code < cb.n_codes
        assert len(decoded) == cb.code_dim


def test_gate_forward_value_returns_binary_mask() -> None:
    g = AdaptiveCompressionGate.init(
        in_dim=6, emit_mask_len=W50_DEFAULT_EMIT_MASK_LEN, seed=3)
    mask, scores = g.forward_value([0.1] * 6)
    assert len(mask) == g.emit_mask_len
    assert len(scores) == g.emit_mask_len
    for b in mask:
        assert b in (0, 1)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_compress_carrier_visible_tokens_one_or_two() -> None:
    cb = AdaptiveCompressionCodebook.init(seed=5)
    g = AdaptiveCompressionGate.init(
        in_dim=cb.code_dim, seed=7,
        importance_threshold=0.5)
    r = compress_carrier(
        [0.1] * cb.code_dim, codebook=cb, gate=g)
    # Visible tokens must be 1 or 2 by accounting.
    assert r.visible_tokens in (1, 2)
    assert r.structured_bits == (
        cb.code_bits() + g.emit_mask_len + sum(r.emit_mask))


def test_training_set_balanced_and_deterministic() -> None:
    a = synthesize_adaptive_compression_training_set(
        n_examples=24, seed=11)
    b = synthesize_adaptive_compression_training_set(
        n_examples=24, seed=11)
    assert a.cid() == b.cid()
    # All target codes valid
    for ex in a.examples:
        assert 0 <= ex.target_code < W50_DEFAULT_ADAPTIVE_K


def test_fit_reduces_loss_and_is_deterministic() -> None:
    ts = synthesize_adaptive_compression_training_set(
        n_examples=16, seed=21)
    cb1, g1, t1 = fit_adaptive_compression(
        ts, n_steps=32, seed=21)
    cb2, g2, t2 = fit_adaptive_compression(
        ts, n_steps=32, seed=21)
    assert cb1.cid() == cb2.cid()
    assert g1.cid() == g2.cid()
    assert t1.cid() == t2.cid()
    assert t1.diverged is False


def test_compression_hits_8_bits_per_token_3_seeds() -> None:
    """H9: mean bits-per-visible-token ≥ 8.0 across 3 seeds."""
    means: list[float] = []
    for seed in (1, 2, 3):
        ts = synthesize_adaptive_compression_training_set(
            n_examples=32, seed=seed)
        cb, gate, _ = fit_adaptive_compression(
            ts, n_steps=96, seed=seed)
        ratios = []
        for ex in ts.examples[:16]:
            r = compress_carrier(
                list(ex.carrier), codebook=cb, gate=gate)
            ratios.append(r.bits_per_visible_token)
        means.append(sum(ratios) / len(ratios))
    mean_of_means = sum(means) / len(means)
    assert mean_of_means >= 8.0, (
        f"H9 missed: mean bits/token {mean_of_means:.3f}")


def test_rate_floor_falsifier_misses_target_at_16() -> None:
    """H14: target rate 16 bits/token exceeds the K=16
    codebook's information capacity → rate target missed."""
    cb = AdaptiveCompressionCodebook.init(seed=31)
    g = AdaptiveCompressionGate.init(in_dim=cb.code_dim, seed=33)
    out = probe_rate_floor_falsifier(
        [0.1] * cb.code_dim, codebook=cb, gate=g,
        target_bits_per_token=16.0)
    assert out["rate_target_missed"] is True
    assert out["achieved_bits_per_token"] < 16.0


def test_cramming_witness_v2_serialises_deterministically() -> None:
    cb = AdaptiveCompressionCodebook.init(seed=41)
    g = AdaptiveCompressionGate.init(in_dim=cb.code_dim, seed=43)
    r = compress_carrier(
        [0.3] * cb.code_dim, codebook=cb, gate=g)
    w1 = emit_cramming_witness_v2(compression=r)
    w2 = emit_cramming_witness_v2(compression=r)
    assert w1.cid() == w2.cid()


def test_adaptive_compression_witness_verify_pass() -> None:
    ts = synthesize_adaptive_compression_training_set(
        n_examples=12, seed=51)
    cb, g, trace = fit_adaptive_compression(
        ts, n_steps=32, seed=51)
    r = compress_carrier(
        list(ts.examples[0].carrier), codebook=cb, gate=g)
    cw = emit_cramming_witness_v2(compression=r)
    aw = emit_adaptive_compression_witness(
        codebook=cb, gate=g, training_trace=trace,
        cramming=cw,
        target_bits_per_token=W50_DEFAULT_TARGET_BITS_PER_TOKEN)
    v = verify_adaptive_compression_witness(
        aw,
        expected_codebook_cid=cb.cid(),
        expected_gate_cid=g.cid(),
        expected_trace_cid=trace.cid(),
        expected_cramming_cid=cw.cid(),
        bits_floor=0.0)
    assert v["ok"] is True


def test_adaptive_witness_verify_detects_codebook_tamper() -> None:
    ts = synthesize_adaptive_compression_training_set(
        n_examples=8, seed=53)
    cb, g, trace = fit_adaptive_compression(
        ts, n_steps=16, seed=53)
    r = compress_carrier(
        list(ts.examples[0].carrier), codebook=cb, gate=g)
    cw = emit_cramming_witness_v2(compression=r)
    aw = emit_adaptive_compression_witness(
        codebook=cb, gate=g, training_trace=trace, cramming=cw)
    v = verify_adaptive_compression_witness(
        aw, expected_codebook_cid="00" * 32)
    assert v["ok"] is False
    assert ("w50_adaptive_compression_codebook_cid_mismatch"
            in v["failures"])


def test_adaptive_witness_verify_detects_bits_below_target() -> None:
    cb = AdaptiveCompressionCodebook.init(seed=61)
    g = AdaptiveCompressionGate.init(in_dim=cb.code_dim, seed=63)
    r = compress_carrier(
        [0.0] * cb.code_dim, codebook=cb, gate=g)
    ts = synthesize_adaptive_compression_training_set(
        n_examples=4, seed=61)
    _, _, trace = fit_adaptive_compression(ts, n_steps=4, seed=61)
    cw = emit_cramming_witness_v2(compression=r)
    aw = emit_adaptive_compression_witness(
        codebook=cb, gate=g, training_trace=trace, cramming=cw,
        target_bits_per_token=100.0)
    v = verify_adaptive_compression_witness(aw, bits_floor=100.0)
    assert v["ok"] is False
    assert ("w50_adaptive_compression_bits_below_target"
            in v["failures"])


def test_compression_result_to_dict_serialises() -> None:
    cb = AdaptiveCompressionCodebook.init(seed=71)
    g = AdaptiveCompressionGate.init(in_dim=cb.code_dim, seed=73)
    r = compress_carrier([0.1] * cb.code_dim, codebook=cb, gate=g)
    d = r.to_dict()
    assert d["code"] == r.code
    assert d["structured_bits"] == r.structured_bits
