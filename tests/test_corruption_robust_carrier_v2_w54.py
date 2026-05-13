"""W54 M5 — Corruption-Robust Carrier V2 tests."""

from __future__ import annotations

import random

from coordpy.corruption_robust_carrier_v2 import (
    CorruptionRobustCarrierV2,
    W54_CRC_V2_VERIFIER_FAILURE_MODES,
    emit_corruption_robustness_v2_witness,
    hamming_7_4_decode,
    hamming_7_4_encode,
    probe_hostile_channel_v2,
    verify_corruption_robustness_v2_witness,
)
from coordpy.ecc_codebook_v5 import ECCCodebookV5
from coordpy.quantised_compression import QuantisedBudgetGate


def _build_crc_v2(seed: int = 1) -> CorruptionRobustCarrierV2:
    cb = ECCCodebookV5.init(code_dim=6, seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=6, emit_mask_len=16, seed=int(seed) + 1)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [
        1.0] * len(gate.w_emit.values)
    return CorruptionRobustCarrierV2.init(
        codebook=cb, gate=gate, repetition=5)


def test_hamming_7_4_round_trips() -> None:
    """Encode + decode without errors recovers the data."""
    for d1 in range(2):
        for d2 in range(2):
            for d3 in range(2):
                for d4 in range(2):
                    cw = hamming_7_4_encode(
                        [d1, d2, d3, d4])
                    out, syndrome, _ = hamming_7_4_decode(cw)
                    assert out == (d1, d2, d3, d4)
                    assert syndrome == 0


def test_hamming_7_4_corrects_single_bit_flip() -> None:
    """A single-bit flip yields syndrome > 0 and recovers data."""
    data = [1, 0, 1, 0]
    cw = list(hamming_7_4_encode(data))
    # Flip bit 2 (one of the data bits).
    cw[2] = 1 - cw[2]
    out, syndrome, _ = hamming_7_4_decode(cw)
    assert syndrome != 0
    assert out == tuple(data)


def test_crc_v2_probe_single_bit_correct_rate() -> None:
    crc = _build_crc_v2(seed=1)
    rng = random.Random(1)
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(40)
    ]
    res = probe_hostile_channel_v2(
        carriers, crc_v2=crc, flip_intensity=1.0, seed=2)
    assert res.single_correct_rate >= 0.95
    assert res.silent_failure_rate <= 0.05


def test_crc_v2_witness_round_trips() -> None:
    crc = _build_crc_v2(seed=3)
    rng = random.Random(3)
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(10)
    ]
    w = emit_corruption_robustness_v2_witness(
        crc_v2=crc, carriers=carriers,
        flip_intensity=1.0, seed=5)
    v = verify_corruption_robustness_v2_witness(
        w, expected_crc_v2_cid=crc.cid(),
        min_single_correct_rate=0.5,
        max_silent_failure_rate=0.1,
        expected_repetition=5)
    assert v["ok"] is True


def test_w54_crc_v2_verifier_failure_modes_count() -> None:
    assert len(W54_CRC_V2_VERIFIER_FAILURE_MODES) == 5
