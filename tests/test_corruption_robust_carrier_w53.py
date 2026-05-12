"""W53 M8 corruption_robust_carrier tests."""

from __future__ import annotations

import random

from coordpy.corruption_robust_carrier import (
    CorruptionRobustCarrier,
    emit_corruption_robustness_witness,
    probe_hostile_channel,
    verify_corruption_robustness_witness,
)
from coordpy.ecc_codebook_v5 import ECCCodebookV5
from coordpy.quantised_compression import QuantisedBudgetGate


def _build_crc(seed: int = 11) -> CorruptionRobustCarrier:
    cb = ECCCodebookV5.init(seed=int(seed))
    gate = QuantisedBudgetGate.init(
        in_dim=cb.code_dim, emit_mask_len=16,
        seed=int(seed) + 5)
    gate.importance_threshold = 0.0
    gate.w_emit.values = [
        1.0] * len(gate.w_emit.values)
    return CorruptionRobustCarrier.init(
        codebook=cb, gate=gate, repetition=3)


def test_crc_single_bit_detects_and_corrects() -> None:
    crc = _build_crc()
    rng = random.Random(0)
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(10)
    ]
    res = probe_hostile_channel(
        carriers, crc=crc, flip_intensity=1.0, seed=7)
    assert res.detect_rate >= 0.8
    assert res.correction_rate >= 0.3


def test_crc_two_bit_graceful_degrade() -> None:
    crc = _build_crc()
    rng = random.Random(0)
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(20)
    ]
    res = probe_hostile_channel(
        carriers, crc=crc, flip_intensity=2.0, seed=11)
    assert res.silent_failure_rate <= 0.30


def test_crc_witness_emit_and_verify() -> None:
    crc = _build_crc()
    rng = random.Random(0)
    carriers = [
        [rng.uniform(-1, 1) for _ in range(6)]
        for _ in range(8)
    ]
    w = emit_corruption_robustness_witness(
        crc=crc, carriers=carriers,
        flip_intensity=1.0, seed=3)
    v = verify_corruption_robustness_witness(
        w, expected_crc_cid=crc.cid(),
        min_detect_rate=0.5,
        max_silent_failure_rate=0.5,
        expected_repetition=3)
    assert v["ok"] is True


def test_crc_repetition_payload_majority_vote() -> None:
    crc = _build_crc()
    sample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    co, fi, ul, u2, rep, par = (
        crc.encode_with_repetition_payload(sample))
    # Triple length.
    assert len(rep) % 3 == 0
    attempt, recovered = crc.decode_with_majority(
        coarse=co, fine=fi, ultra=ul, ultra2=u2,
        observed_parity=par,
        repetition_payload=rep)
    assert len(recovered) * 3 == len(rep)


def test_crc_cid_stable() -> None:
    crc1 = _build_crc(seed=11)
    crc2 = _build_crc(seed=11)
    assert crc1.cid() == crc2.cid()
