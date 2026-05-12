"""Unit tests for W51 M4 — hierarchical compression."""

from __future__ import annotations

import pytest

from coordpy.hierarchical_compression import (
    HierarchicalCodebook,
    HierarchicalEmitGate,
    W51_DEFAULT_HIER_DEGRADATION_BUDGETS,
    W51_DEFAULT_HIER_EMIT_MASK_LEN,
    W51_DEFAULT_HIER_K1,
    W51_DEFAULT_HIER_K2,
    W51_HIERARCHICAL_COMPRESSION_VERIFIER_FAILURE_MODES,
    compress_carrier_hierarchical,
    emit_cramming_witness_v3,
    emit_hierarchical_compression_witness,
    fit_hierarchical_compression,
    probe_degradation_curve,
    probe_rate_floor_v2_falsifier,
    synthesize_hierarchical_compression_training_set,
    verify_hierarchical_compression_witness,
)


def test_codebook_defaults_k1_32_k2_16() -> None:
    cb = HierarchicalCodebook.init(seed=11)
    assert cb.n_coarse == 32
    assert cb.n_fine == 16
    assert cb.coarse_bits() == 5
    assert cb.fine_bits() == 4


def test_codebook_init_stable_seed() -> None:
    a = HierarchicalCodebook.init(seed=11)
    b = HierarchicalCodebook.init(seed=11)
    assert a.cid() == b.cid()


def test_encode_decode_round_trip() -> None:
    cb = HierarchicalCodebook.init(seed=11, n_coarse=8, n_fine=4)
    x = [0.5, -0.2, 0.3, 0.1, 0.0, 0.0]
    c, f = cb.encode_value(x)
    assert 0 <= c < 8
    assert 0 <= f < 4
    decoded = cb.decode(coarse=c, fine=f)
    assert len(decoded) == cb.code_dim


def test_compression_bits_per_token_at_full_emit() -> None:
    cb = HierarchicalCodebook.init(seed=11)
    gate = HierarchicalEmitGate.init(
        in_dim=cb.code_dim,
        emit_mask_len=W51_DEFAULT_HIER_EMIT_MASK_LEN,
        seed=23, importance_threshold=0.0)
    # Force all emit bits open
    gate.w_emit.values = [10.0] * len(gate.w_emit.values)
    gate.w_level.values = [10.0] * len(gate.w_level.values)
    x = [0.5] * cb.code_dim
    res = compress_carrier_hierarchical(
        x, codebook=cb, gate=gate)
    # At full emit: coarse_bits + 2 + fine_bits + emit_mask_len + mask_sum
    # = 5 + 2 + 4 + 14 + 14 = 39 / 3 = 13.0
    assert res.bits_per_visible_token >= 12.0


def test_degradation_curve_monotone_or_bounded() -> None:
    cb = HierarchicalCodebook.init(seed=11)
    gate = HierarchicalEmitGate.init(
        in_dim=cb.code_dim,
        emit_mask_len=W51_DEFAULT_HIER_EMIT_MASK_LEN,
        seed=23)
    x = [0.3, -0.2, 0.1, 0.5, 0.0, -0.4]
    dc = probe_degradation_curve(
        x, codebook=cb, gate=gate,
        budgets=W51_DEFAULT_HIER_DEGRADATION_BUDGETS)
    assert len(dc) == 4
    # All budgets must produce positive bits/token
    for p in dc:
        assert p.achieved_bits_per_token > 0.0


def test_rate_floor_falsifier_at_20_bits_misses() -> None:
    cb = HierarchicalCodebook.init(seed=11)
    gate = HierarchicalEmitGate.init(
        in_dim=cb.code_dim,
        emit_mask_len=W51_DEFAULT_HIER_EMIT_MASK_LEN,
        seed=23)
    x = [0.5] * cb.code_dim
    res = probe_rate_floor_v2_falsifier(
        x, codebook=cb, gate=gate,
        target_bits_per_token=20.0)
    assert res["rate_target_missed"] is True


def test_witness_passes_clean_verifier() -> None:
    ts = synthesize_hierarchical_compression_training_set(
        n_examples=16, code_dim=6, n_coarse=16, n_fine=8,
        emit_mask_len=10, seed=11)
    cb, gate, trace = fit_hierarchical_compression(
        ts, n_steps=24, seed=11)
    x = list(ts.examples[0].carrier)
    res = compress_carrier_hierarchical(
        x, codebook=cb, gate=gate)
    cw = emit_cramming_witness_v3(compression=res)
    from coordpy.hierarchical_compression import _cosine
    decoded = cb.decode(
        coarse=res.coarse_code, fine=res.fine_code)
    retention = _cosine(x, decoded)
    dc = probe_degradation_curve(x, codebook=cb, gate=gate)
    w = emit_hierarchical_compression_witness(
        codebook=cb, gate=gate, training_trace=trace,
        cramming=cw, retention_cosine=retention,
        degradation_curve=dc)
    v = verify_hierarchical_compression_witness(
        w, expected_codebook_cid=cb.cid(),
        expected_gate_cid=gate.cid(),
        expected_trace_cid=trace.cid(),
        expected_cramming_cid=cw.cid())
    assert v["ok"] is True


def test_verifier_has_9_failure_modes() -> None:
    assert len(
        W51_HIERARCHICAL_COMPRESSION_VERIFIER_FAILURE_MODES) == 9
