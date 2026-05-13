"""W54 M10 — Uncertainty Layer V2 tests."""

from __future__ import annotations

import random

from coordpy.uncertainty_layer import calibration_check
from coordpy.uncertainty_layer_v2 import (
    W54_UNCERT_V2_VERIFIER_FAILURE_MODES,
    calibration_check_under_noise,
    compose_uncertainty_report_v2,
    emit_uncertainty_layer_v2_witness,
    verify_uncertainty_layer_v2_witness,
)


def test_uncertainty_v2_high_components_yield_high_composite() -> None:
    r = compose_uncertainty_report_v2(
        persistent_v6_confidence=0.9,
        multi_hop_v4_confidence=0.9,
        mlsc_v2_capsule_confidence=0.9,
        deep_v5_corruption_confidence=0.9,
        crc_v2_silent_failure_rate=0.0,
        component_disagreements={})
    assert r.composite_confidence >= 0.75
    assert r.level == "high"


def test_uncertainty_v2_low_components_yield_abstain() -> None:
    r = compose_uncertainty_report_v2(
        persistent_v6_confidence=0.05,
        multi_hop_v4_confidence=0.05,
        mlsc_v2_capsule_confidence=0.05,
        deep_v5_corruption_confidence=0.05,
        crc_v2_silent_failure_rate=0.9,
        component_disagreements={})
    assert r.level == "abstain"
    assert r.composite_confidence < 0.2


def test_uncertainty_v2_disagreement_downweights_composite() -> None:
    base = compose_uncertainty_report_v2(
        persistent_v6_confidence=0.8,
        multi_hop_v4_confidence=0.8,
        mlsc_v2_capsule_confidence=0.8,
        deep_v5_corruption_confidence=0.8,
        crc_v2_silent_failure_rate=0.1,
        component_disagreements={})
    disagreed = compose_uncertainty_report_v2(
        persistent_v6_confidence=0.8,
        multi_hop_v4_confidence=0.8,
        mlsc_v2_capsule_confidence=0.8,
        deep_v5_corruption_confidence=0.8,
        crc_v2_silent_failure_rate=0.1,
        component_disagreements={
            "persistent_v6": 1.5, "mlsc_v2": 1.5})
    assert (
        disagreed.composite_confidence
        < base.composite_confidence)


def test_uncertainty_v2_rationale_is_non_empty() -> None:
    r = compose_uncertainty_report_v2(
        persistent_v6_confidence=0.05,
        multi_hop_v4_confidence=0.05,
        mlsc_v2_capsule_confidence=0.05,
        deep_v5_corruption_confidence=0.05,
        crc_v2_silent_failure_rate=0.9,
        component_disagreements={})
    assert len(r.rationale) > 0


def test_calibration_check_under_noise_returns_finite() -> None:
    rng = random.Random(7)
    confs = [
        rng.uniform(0.0, 1.0) for _ in range(20)]
    accs = [
        rng.uniform(0.0, 1.0) for _ in range(20)]
    res = calibration_check_under_noise(
        confs, accs, noise_magnitude=0.1, seed=1)
    assert -1.0 <= res.calibration_gap_noisy <= 1.0


def test_uncertainty_v2_witness_round_trips() -> None:
    rng = random.Random(11)
    confs = []
    accs = []
    for i in range(30):
        is_high = (i % 3 != 0)
        report = compose_uncertainty_report_v2(
            persistent_v6_confidence=(
                rng.uniform(0.7, 0.9)
                if is_high else rng.uniform(0.0, 0.2)),
            multi_hop_v4_confidence=(
                rng.uniform(0.6, 0.9)
                if is_high else rng.uniform(0.0, 0.3)),
            mlsc_v2_capsule_confidence=(
                rng.uniform(0.6, 0.9)
                if is_high else rng.uniform(0.0, 0.3)),
            deep_v5_corruption_confidence=(
                rng.uniform(0.6, 0.9)
                if is_high else rng.uniform(0.0, 0.3)),
            crc_v2_silent_failure_rate=(
                rng.uniform(0.0, 0.1)
                if is_high else rng.uniform(0.4, 0.8)),
            component_disagreements={})
        confs.append(report.composite_confidence)
        accs.append(
            rng.uniform(0.7, 1.0)
            if is_high
            else rng.uniform(0.0, 0.4))
    clean = calibration_check(
        confs, accs, min_calibration_gap=0.10)
    noisy = calibration_check_under_noise(
        confs, accs, noise_magnitude=0.1, seed=1)
    # Use the last report as the witness report.
    w = emit_uncertainty_layer_v2_witness(
        report=report,
        calibration_clean=clean,
        calibration_noisy=noisy)
    v = verify_uncertainty_layer_v2_witness(w)
    assert v["ok"] is True


def test_w54_uncert_v2_verifier_failure_modes_count() -> None:
    assert len(W54_UNCERT_V2_VERIFIER_FAILURE_MODES) == 6
