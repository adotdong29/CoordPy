"""W53 M10 uncertainty_layer tests."""

from __future__ import annotations

import random

from coordpy.uncertainty_layer import (
    W53_UNCERT_LEVEL_ABSTAIN,
    W53_UNCERT_LEVEL_HIGH,
    W53_UNCERT_LEVEL_LOW,
    W53_UNCERT_LEVEL_MEDIUM,
    calibration_check,
    compose_uncertainty_report,
    emit_uncertainty_layer_witness,
    verify_uncertainty_layer_witness,
)


def test_compose_high_confidence_report() -> None:
    r = compose_uncertainty_report(
        persistent_v5_confidence=0.9,
        multi_hop_v3_confidence=0.85,
        mlsc_capsule_confidence=0.8,
        deep_v4_corruption_confidence=0.92,
        crc_silent_failure_rate=0.05)
    assert r.composite_confidence > 0.75
    assert r.level == W53_UNCERT_LEVEL_HIGH


def test_compose_abstain_report() -> None:
    r = compose_uncertainty_report(
        persistent_v5_confidence=0.05,
        multi_hop_v3_confidence=0.05,
        mlsc_capsule_confidence=0.05,
        deep_v4_corruption_confidence=0.05,
        crc_silent_failure_rate=0.95)
    assert r.composite_confidence < 0.20
    assert r.level == W53_UNCERT_LEVEL_ABSTAIN


def test_calibration_check_passes_when_separable() -> None:
    rng = random.Random(7)
    confs: list[float] = []
    accs: list[float] = []
    for i in range(30):
        is_high = (i % 2 == 0)
        c = (
            rng.uniform(0.8, 1.0)
            if is_high
            else rng.uniform(0.05, 0.15))
        a = (
            rng.uniform(0.7, 1.0)
            if is_high
            else rng.uniform(0.0, 0.3))
        confs.append(c)
        accs.append(a)
    res = calibration_check(
        confs, accs, min_calibration_gap=0.10)
    assert res.calibrated
    assert res.calibration_gap >= 0.10


def test_calibration_check_fails_when_random() -> None:
    rng = random.Random(11)
    confs = [rng.uniform(0.0, 1.0) for _ in range(30)]
    accs = [rng.uniform(0.0, 1.0) for _ in range(30)]
    res = calibration_check(
        confs, accs, min_calibration_gap=0.5)
    assert not res.calibrated


def test_uncertainty_witness_emit_and_verify() -> None:
    r = compose_uncertainty_report(
        persistent_v5_confidence=0.7,
        multi_hop_v3_confidence=0.7,
        mlsc_capsule_confidence=0.7,
        deep_v4_corruption_confidence=0.7,
        crc_silent_failure_rate=0.05)
    cal = calibration_check(
        [0.9, 0.9, 0.05, 0.05],
        [0.9, 0.9, 0.05, 0.05],
        min_calibration_gap=0.10)
    w = emit_uncertainty_layer_witness(
        report=r, calibration=cal)
    v = verify_uncertainty_layer_witness(
        w, require_calibrated=True,
        min_calibration_gap=0.10)
    assert v["ok"] is True
