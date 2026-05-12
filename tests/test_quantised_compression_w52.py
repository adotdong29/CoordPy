"""Per-module tests for the W52 quantised compression."""

from __future__ import annotations

import pytest

from coordpy.quantised_compression import (
    QuantisedBudgetGate,
    QuantisedCodebookV4,
    compress_carrier_quantised,
    emit_cramming_witness_v4,
    emit_quantised_compression_witness,
    fit_quantised_compression,
    probe_quantised_degradation_curve,
    probe_quantised_rate_floor_falsifier,
    synthesize_quantised_compression_training_set,
    verify_quantised_compression_witness,
)


def test_codebook_v4_has_k1_k2_k3_levels() -> None:
    cb = QuantisedCodebookV4.init(
        n_coarse=32, n_fine=16, n_ultra=8,
        code_dim=4, seed=1)
    assert cb.n_coarse == 32
    assert cb.n_fine == 16
    assert cb.n_ultra == 8
    # Bits for each level
    assert cb.coarse_bits() == 5
    assert cb.fine_bits() == 4
    assert cb.ultra_bits() == 3


def test_codebook_v4_encode_decode_round_trip() -> None:
    cb = QuantisedCodebookV4.init(
        n_coarse=8, n_fine=4, n_ultra=4,
        code_dim=4, seed=2)
    x = [0.5, -0.3, 0.7, 0.1]
    c, f, u = cb.encode_value(x)
    decoded = cb.decode(coarse=c, fine=f, ultra=u)
    assert len(decoded) == 4


def test_compress_carrier_quantised_visible_tokens_3_at_full() -> None:
    cb = QuantisedCodebookV4.init(
        n_coarse=32, n_fine=16, n_ultra=8,
        code_dim=4, seed=3)
    gate = QuantisedBudgetGate.init(
        in_dim=4, emit_mask_len=12, seed=3)
    # Force importance threshold low so the gate emits.
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    x = [0.4, -0.2, 0.6, 0.0]
    res = compress_carrier_quantised(
        x, codebook=cb, gate=gate)
    assert res.visible_tokens == 3
    assert res.structured_bits >= 14  # tighter than 12


def test_quantised_target_bits_witness_meets_14() -> None:
    cb = QuantisedCodebookV4.init(
        n_coarse=32, n_fine=16, n_ultra=8,
        code_dim=6, seed=4)
    gate = QuantisedBudgetGate.init(
        in_dim=6, emit_mask_len=16, seed=4)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    x = [0.3, 0.4, -0.1, 0.2, 0.0, -0.5]
    res = compress_carrier_quantised(
        x, codebook=cb, gate=gate)
    cw = emit_cramming_witness_v4(compression=res)
    assert cw.bits_per_visible_token >= 14.0


def test_quantised_degradation_curve_monotonic() -> None:
    cb = QuantisedCodebookV4.init(
        n_coarse=8, n_fine=4, n_ultra=4,
        code_dim=4, seed=5)
    gate = QuantisedBudgetGate.init(
        in_dim=4, emit_mask_len=8, seed=5)
    x = [0.1, -0.2, 0.3, 0.0]
    curve = probe_quantised_degradation_curve(
        x, codebook=cb, gate=gate,
        budgets=(8, 4, 2, 1))
    assert len(curve) == 4
    # Min bits/token should stay positive.
    assert min(p.bits_per_visible_token for p in curve) > 0.0


def test_quantised_rate_floor_falsifier_at_32_bits() -> None:
    cb = QuantisedCodebookV4.init(
        n_coarse=32, n_fine=16, n_ultra=8,
        code_dim=4, seed=6)
    gate = QuantisedBudgetGate.init(
        in_dim=4, emit_mask_len=16, seed=6)
    x = [0.1, 0.2, 0.3, 0.4]
    res = probe_quantised_rate_floor_falsifier(
        x, codebook=cb, gate=gate,
        target_bits_per_token=32.0)
    assert res.rate_target_missed is True


def test_fit_quantised_compression_runs() -> None:
    ts = synthesize_quantised_compression_training_set(
        n_examples=8, code_dim=4, n_coarse=8, n_fine=4,
        n_ultra=4, emit_mask_len=8, seed=7)
    cb, gate, trace = fit_quantised_compression(
        ts, n_steps=4, seed=7)
    assert not trace.diverged
    assert cb.cid() != ""


def test_witness_round_trips() -> None:
    cb = QuantisedCodebookV4.init(
        n_coarse=8, n_fine=4, n_ultra=4,
        code_dim=4, seed=8)
    gate = QuantisedBudgetGate.init(
        in_dim=4, emit_mask_len=8, seed=8)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [1.0] * len(gate.w_emit.values)
    x = [0.1, 0.2, 0.3, 0.4]
    res = compress_carrier_quantised(
        x, codebook=cb, gate=gate)
    cw = emit_cramming_witness_v4(compression=res)
    from coordpy.quantised_compression import (
        QuantisedCompressionTrainingTrace,
    )
    empty_trace = QuantisedCompressionTrainingTrace(
        seed=0, n_steps=0, final_loss=0.0,
        final_grad_norm=0.0, loss_head=(), loss_tail=(),
        training_set_cid="", final_codebook_cid=cb.cid(),
        final_gate_cid=gate.cid(), diverged=False)
    w = emit_quantised_compression_witness(
        codebook=cb, gate=gate, training_trace=empty_trace,
        cramming=cw, retention_cosine=0.9)
    v = verify_quantised_compression_witness(
        w, expected_codebook_cid=cb.cid(),
        expected_gate_cid=gate.cid())
    assert v["ok"] is True
