"""W54 M8 — ECC Codebook V6 tests."""

from __future__ import annotations

import random

from coordpy.ecc_codebook_v6 import (
    ECCCodebookV6,
    W54_DEFAULT_ECC_V6_K5,
    W54_DEFAULT_ECC_V6_TARGET_BITS_PER_TOKEN,
    W54_ECC_V6_VERIFIER_FAILURE_MODES,
    compress_carrier_ecc_v6,
    emit_ecc_v6_compression_witness,
    probe_ecc_v6_rate_floor_falsifier,
    verify_ecc_v6_compression_witness,
)
from coordpy.quantised_compression import QuantisedBudgetGate


def test_ecc_v6_codebook_has_5_levels() -> None:
    cb = ECCCodebookV6.init(
        n_coarse=32, n_fine=16, n_ultra=8,
        n_ultra2=4, n_ultra3=W54_DEFAULT_ECC_V6_K5,
        code_dim=6, seed=1)
    assert cb.n_ultra3 == W54_DEFAULT_ECC_V6_K5


def test_ecc_v6_compression_meets_target_bits() -> None:
    cb = ECCCodebookV6.init(
        n_coarse=32, n_fine=16, n_ultra=8,
        n_ultra2=4, n_ultra3=2, code_dim=6, seed=1)
    gate = QuantisedBudgetGate.init(
        in_dim=6, emit_mask_len=16, seed=2)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [
        1.0] * len(gate.w_emit.values)
    rng = random.Random(1)
    carrier = [rng.uniform(-1, 1) for _ in range(6)]
    res = compress_carrier_ecc_v6(
        carrier, codebook=cb, gate=gate)
    assert (
        res.bits_per_visible_token_v6
        >= W54_DEFAULT_ECC_V6_TARGET_BITS_PER_TOKEN)


def test_ecc_v6_rate_floor_falsifier_reproduces() -> None:
    cb = ECCCodebookV6.init(
        n_coarse=32, n_fine=16, n_ultra=8,
        n_ultra2=4, n_ultra3=2, code_dim=6, seed=1)
    gate = QuantisedBudgetGate.init(
        in_dim=6, emit_mask_len=16, seed=2)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [
        1.0] * len(gate.w_emit.values)
    rng = random.Random(2)
    carrier = [rng.uniform(-1, 1) for _ in range(6)]
    res = probe_ecc_v6_rate_floor_falsifier(
        carrier, codebook=cb, gate=gate,
        target_bits_per_token=64.0)
    assert res.rate_target_missed


def test_ecc_v6_hamming_encode_returns_7_bit_tuples() -> None:
    cb = ECCCodebookV6.init(
        n_coarse=32, n_fine=16, n_ultra=8,
        n_ultra2=4, n_ultra3=2, code_dim=6, seed=1)
    h_c, h_f, h_u, h_u2, h_u3 = (
        cb.hamming_encode_segments(
            coarse=5, fine=3, ultra=2,
            ultra2=1, ultra3=1))
    for h in (h_c, h_f, h_u, h_u2, h_u3):
        assert len(h) == 7


def test_w54_ecc_v6_witness_round_trips() -> None:
    cb = ECCCodebookV6.init(
        n_coarse=32, n_fine=16, n_ultra=8,
        n_ultra2=4, n_ultra3=2, code_dim=6, seed=1)
    gate = QuantisedBudgetGate.init(
        in_dim=6, emit_mask_len=16, seed=2)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [
        1.0] * len(gate.w_emit.values)
    rng = random.Random(7)
    carrier = [rng.uniform(-1, 1) for _ in range(6)]
    res = compress_carrier_ecc_v6(
        carrier, codebook=cb, gate=gate)
    w = emit_ecc_v6_compression_witness(
        codebook=cb, compression=res)
    v = verify_ecc_v6_compression_witness(
        w, expected_codebook_cid=cb.cid(),
        min_n_ultra3=2)
    assert v["ok"] is True


def test_w54_ecc_v6_verifier_failure_modes_count() -> None:
    assert len(W54_ECC_V6_VERIFIER_FAILURE_MODES) == 5
