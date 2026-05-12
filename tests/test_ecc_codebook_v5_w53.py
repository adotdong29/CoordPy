"""W53 M5 ecc_codebook_v5 tests."""

from __future__ import annotations

import random

from coordpy.ecc_codebook_v5 import (
    ECCCodebookV5,
    W53_DEFAULT_ECC_TARGET_BITS_PER_TOKEN,
    W53_ECC_PARITY_BITS_PER_TOKEN,
    compress_carrier_ecc,
    decode_with_parity_check,
    emit_ecc_compression_witness,
    emit_ecc_robustness_witness,
    flip_random_bit,
    probe_ecc_rate_floor_falsifier,
    verify_ecc_compression_witness,
)
from coordpy.quantised_compression import QuantisedBudgetGate


def _build_ecc(seed: int = 11):
    cb = ECCCodebookV5.init(seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=cb.code_dim, emit_mask_len=16,
        seed=int(seed) + 5)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [
        1.0] * len(gate.w_emit.values)
    return cb, gate


def test_ecc_meets_target_bits_per_token() -> None:
    cb, gate = _build_ecc()
    sample = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    res = compress_carrier_ecc(
        sample, codebook=cb, gate=gate)
    assert (
        res.bits_per_visible_token
        >= W53_DEFAULT_ECC_TARGET_BITS_PER_TOKEN)


def test_ecc_parity_bits_present() -> None:
    cb, gate = _build_ecc()
    sample = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    res = compress_carrier_ecc(
        sample, codebook=cb, gate=gate)
    assert len(res.parity_bits) == 4


def test_ecc_decode_clean_no_corruption() -> None:
    cb, gate = _build_ecc()
    sample = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    res = compress_carrier_ecc(
        sample, codebook=cb, gate=gate)
    attempt = decode_with_parity_check(
        codebook=cb,
        coarse=res.coarse_code,
        fine=res.fine_code,
        ultra=res.ultra_code,
        ultra2=res.ultra2_code,
        observed_parity=res.parity_bits)
    assert attempt.n_corrupted_segments == 0
    assert not attempt.abstain
    assert not attempt.corrected_partial


def test_ecc_decode_single_bit_flip_detects() -> None:
    cb, gate = _build_ecc()
    sample = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    res = compress_carrier_ecc(
        sample, codebook=cb, gate=gate)
    co, fi, ul, u2, _ = flip_random_bit(
        coarse=res.coarse_code,
        fine=res.fine_code,
        ultra=res.ultra_code,
        ultra2=res.ultra2_code,
        codebook=cb, seed=1)
    attempt = decode_with_parity_check(
        codebook=cb, coarse=co, fine=fi,
        ultra=ul, ultra2=u2,
        observed_parity=res.parity_bits)
    assert attempt.n_corrupted_segments >= 1


def test_ecc_robustness_witness_high_detect_rate() -> None:
    cb, gate = _build_ecc()
    rng = random.Random(0)
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(20)
    ]
    w = emit_ecc_robustness_witness(
        carriers=carriers, codebook=cb, gate=gate, seed=5)
    assert w.detect_rate >= 0.8
    assert w.correction_rate >= 0.3


def test_ecc_compression_witness_target_met() -> None:
    cb, gate = _build_ecc()
    sample = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    res = compress_carrier_ecc(
        sample, codebook=cb, gate=gate)
    w = emit_ecc_compression_witness(
        codebook=cb, compression=res)
    assert w.target_met
    assert w.parity_bits_count == int(
        W53_ECC_PARITY_BITS_PER_TOKEN)


def test_ecc_rate_floor_falsifier_misses_at_40_bits() -> None:
    cb, gate = _build_ecc()
    sample = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    res = probe_ecc_rate_floor_falsifier(
        sample, codebook=cb, gate=gate,
        target_bits_per_token=40.0)
    assert res.rate_target_missed


def test_ecc_witness_verifier() -> None:
    cb, gate = _build_ecc()
    sample = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    res = compress_carrier_ecc(
        sample, codebook=cb, gate=gate)
    w = emit_ecc_compression_witness(
        codebook=cb, compression=res)
    v = verify_ecc_compression_witness(
        w, expected_codebook_cid=cb.cid(),
        min_bits_per_token=14.0,
        expected_n_ultra2=cb.n_ultra2)
    assert v["ok"] is True
