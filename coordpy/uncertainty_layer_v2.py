"""W54 M10 — Uncertainty Layer V2.

Extends W53 V1 with three additions:

* **per-component noise injection** + **calibration-under-noise**
  check — verifies that the calibration gap survives when noise
  is injected into each per-component confidence
* **per-decision rationale tag** — every uncertainty report
  records *which component* triggered the (low-confidence)
  decision, plus a one-line ``rationale`` string suitable for
  audit
* **disagreement-weighted composite** — when component
  disagreement scalars are provided, the composite confidence
  down-weights components reporting high disagreement so the
  uncertainty layer respects the V6/V2-MLSC disagreement signal

The V2 report extends ``UncertaintyReport`` with:
``component_disagreements``, ``component_with_lowest_confidence``,
``rationale``, ``disagreement_weight_factor``.

Honest scope: pure-Python only, capsule-layer only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

from .uncertainty_layer import (
    CalibrationCheckResult,
    UncertaintyReport,
    W53_UNCERT_LEVEL_ABSTAIN,
    W53_UNCERT_LEVEL_HIGH,
    W53_UNCERT_LEVEL_LOW,
    W53_UNCERT_LEVEL_MEDIUM,
    W53_DEFAULT_UNCERT_HIGH_THRESHOLD,
    W53_DEFAULT_UNCERT_LOW_THRESHOLD,
    W53_DEFAULT_UNCERT_MEDIUM_THRESHOLD,
    calibration_check,
    compose_uncertainty_report,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W54_UNCERT_V2_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v2.v1")

W54_UNCERT_V2_RATIONALE_HIGH: str = (
    "all_components_high_confidence")
W54_UNCERT_V2_RATIONALE_MEDIUM: str = (
    "moderate_aggregate_confidence")
W54_UNCERT_V2_RATIONALE_LOW: str = (
    "low_aggregate_confidence_triggered_by_component")
W54_UNCERT_V2_RATIONALE_ABSTAIN: str = (
    "abstain_threshold_breached_by_component")
W54_UNCERT_V2_RATIONALE_DISAGREEMENT: str = (
    "high_disagreement_triggered_downweight")


# =============================================================================
# Helpers
# =============================================================================


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _round_floats(
        values: Sequence[float], precision: int = 12,
) -> list[float]:
    return [float(round(float(v), precision)) for v in values]


# =============================================================================
# UncertaintyReportV2
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UncertaintyReportV2:
    """V2 composite report with disagreement-weighting + rationale."""

    persistent_v6_confidence: float
    multi_hop_v4_confidence: float
    mlsc_v2_capsule_confidence: float
    deep_v5_corruption_confidence: float
    crc_v2_silent_failure_rate: float
    composite_confidence: float
    disagreement_weight_factor: float
    level: str
    component_with_lowest_confidence: str
    component_disagreements: tuple[
        tuple[str, float], ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_v6_confidence": float(round(
                self.persistent_v6_confidence, 12)),
            "multi_hop_v4_confidence": float(round(
                self.multi_hop_v4_confidence, 12)),
            "mlsc_v2_capsule_confidence": float(round(
                self.mlsc_v2_capsule_confidence, 12)),
            "deep_v5_corruption_confidence": float(round(
                self.deep_v5_corruption_confidence, 12)),
            "crc_v2_silent_failure_rate": float(round(
                self.crc_v2_silent_failure_rate, 12)),
            "composite_confidence": float(round(
                self.composite_confidence, 12)),
            "disagreement_weight_factor": float(round(
                self.disagreement_weight_factor, 12)),
            "level": str(self.level),
            "component_with_lowest_confidence": str(
                self.component_with_lowest_confidence),
            "component_disagreements": [
                [str(n), float(round(float(v), 12))]
                for (n, v) in self.component_disagreements],
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_uncertainty_report_v2",
            "report": self.to_dict()})


def _level_for(
        x: float, *,
        high_threshold: float,
        medium_threshold: float,
        low_threshold: float,
) -> str:
    if float(x) >= float(high_threshold):
        return W53_UNCERT_LEVEL_HIGH
    if float(x) >= float(medium_threshold):
        return W53_UNCERT_LEVEL_MEDIUM
    if float(x) >= float(low_threshold):
        return W53_UNCERT_LEVEL_LOW
    return W53_UNCERT_LEVEL_ABSTAIN


def compose_uncertainty_report_v2(
        *,
        persistent_v6_confidence: float = 1.0,
        multi_hop_v4_confidence: float = 1.0,
        mlsc_v2_capsule_confidence: float = 1.0,
        deep_v5_corruption_confidence: float = 1.0,
        crc_v2_silent_failure_rate: float = 0.0,
        component_disagreements: Mapping[str, float] | None = None,
        weights: tuple[float, float, float, float, float] = (
            0.25, 0.20, 0.20, 0.20, 0.15),
        disagreement_temperature: float = 1.0,
        high_threshold: float = (
            W53_DEFAULT_UNCERT_HIGH_THRESHOLD),
        medium_threshold: float = (
            W53_DEFAULT_UNCERT_MEDIUM_THRESHOLD),
        low_threshold: float = (
            W53_DEFAULT_UNCERT_LOW_THRESHOLD),
) -> UncertaintyReportV2:
    """V2 composite — same form as V1 but multiplied by a
    disagreement-weight factor in ``(0, 1]``:

        factor = exp(-T * mean_disagreement)

    so high disagreement → small factor → smaller composite.
    """
    # Use V1 composer as the base.
    base = compose_uncertainty_report(
        persistent_v5_confidence=float(
            persistent_v6_confidence),
        multi_hop_v3_confidence=float(
            multi_hop_v4_confidence),
        mlsc_capsule_confidence=float(
            mlsc_v2_capsule_confidence),
        deep_v4_corruption_confidence=float(
            deep_v5_corruption_confidence),
        crc_silent_failure_rate=float(
            crc_v2_silent_failure_rate),
        weights=weights,
        high_threshold=float(high_threshold),
        medium_threshold=float(medium_threshold),
        low_threshold=float(low_threshold))
    cd = (
        dict(component_disagreements)
        if component_disagreements is not None
        else {})
    cd_pairs = tuple(sorted(
        (str(k), float(v)) for k, v in cd.items()))
    mean_d = (
        float(sum(float(v) for _, v in cd_pairs))
        / float(max(1, len(cd_pairs)))
        if cd_pairs else 0.0)
    factor = float(math.exp(
        -float(disagreement_temperature)
        * float(max(0.0, mean_d))))
    factor = float(max(0.0, min(1.0, factor)))
    composite = float(base.composite_confidence) * float(factor)
    composite = float(max(0.0, min(1.0, composite)))
    level = _level_for(
        composite,
        high_threshold=float(high_threshold),
        medium_threshold=float(medium_threshold),
        low_threshold=float(low_threshold))
    # Identify lowest-confidence component.
    components = (
        ("persistent_v6", float(persistent_v6_confidence)),
        ("multi_hop_v4", float(multi_hop_v4_confidence)),
        ("mlsc_v2", float(mlsc_v2_capsule_confidence)),
        ("deep_v5", float(deep_v5_corruption_confidence)),
        ("crc_v2_safety",
         float(1.0 - crc_v2_silent_failure_rate)),
    )
    lowest = min(components, key=lambda x: float(x[1]))
    # Rationale.
    if level == W53_UNCERT_LEVEL_ABSTAIN:
        rationale = (
            f"{W54_UNCERT_V2_RATIONALE_ABSTAIN}:"
            f"{lowest[0]}={lowest[1]:.4g}")
    elif level == W53_UNCERT_LEVEL_LOW:
        rationale = (
            f"{W54_UNCERT_V2_RATIONALE_LOW}:"
            f"{lowest[0]}={lowest[1]:.4g}")
    elif level == W53_UNCERT_LEVEL_HIGH:
        rationale = str(W54_UNCERT_V2_RATIONALE_HIGH)
    else:
        rationale = str(W54_UNCERT_V2_RATIONALE_MEDIUM)
    if mean_d > 0.5:
        rationale = (
            rationale + ";"
            + W54_UNCERT_V2_RATIONALE_DISAGREEMENT
            + f":mean_disagreement={mean_d:.4g}")
    return UncertaintyReportV2(
        persistent_v6_confidence=float(
            persistent_v6_confidence),
        multi_hop_v4_confidence=float(
            multi_hop_v4_confidence),
        mlsc_v2_capsule_confidence=float(
            mlsc_v2_capsule_confidence),
        deep_v5_corruption_confidence=float(
            deep_v5_corruption_confidence),
        crc_v2_silent_failure_rate=float(
            crc_v2_silent_failure_rate),
        composite_confidence=float(composite),
        disagreement_weight_factor=float(factor),
        level=str(level),
        component_with_lowest_confidence=str(lowest[0]),
        component_disagreements=cd_pairs,
        rationale=str(rationale),
    )


# =============================================================================
# Noise calibration check
# =============================================================================


@dataclasses.dataclass(frozen=True)
class NoiseCalibrationResult:
    """Calibration check under per-component noise injection."""

    n_high_conf: int
    n_low_conf: int
    noise_magnitude: float
    high_conf_accuracy: float
    low_conf_accuracy: float
    calibration_gap_noisy: float
    calibrated_under_noise: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_high_conf": int(self.n_high_conf),
            "n_low_conf": int(self.n_low_conf),
            "noise_magnitude": float(round(
                self.noise_magnitude, 12)),
            "high_conf_accuracy": float(round(
                self.high_conf_accuracy, 12)),
            "low_conf_accuracy": float(round(
                self.low_conf_accuracy, 12)),
            "calibration_gap_noisy": float(round(
                self.calibration_gap_noisy, 12)),
            "calibrated_under_noise": bool(
                self.calibrated_under_noise),
        }


def calibration_check_under_noise(
        confidences: Sequence[float],
        accuracies: Sequence[float],
        *,
        noise_magnitude: float = 0.1,
        seed: int = 12345,
        high_threshold: float = (
            W53_DEFAULT_UNCERT_HIGH_THRESHOLD),
        low_threshold: float = (
            W53_DEFAULT_UNCERT_LOW_THRESHOLD),
        min_calibration_gap: float = 0.10,
) -> NoiseCalibrationResult:
    """Inject zero-mean uniform noise into each confidence; run
    the V1 calibration_check on the noised series."""
    rng = random.Random(int(seed))
    noisy: list[float] = []
    for c in confidences:
        n = (
            float(rng.uniform(-1.0, 1.0))
            * float(noise_magnitude))
        v = float(c) + n
        noisy.append(float(max(0.0, min(1.0, v))))
    base = calibration_check(
        noisy, accuracies,
        high_threshold=float(high_threshold),
        low_threshold=float(low_threshold),
        min_calibration_gap=float(min_calibration_gap))
    return NoiseCalibrationResult(
        n_high_conf=int(base.n_high_conf),
        n_low_conf=int(base.n_low_conf),
        noise_magnitude=float(noise_magnitude),
        high_conf_accuracy=float(base.high_conf_accuracy),
        low_conf_accuracy=float(base.low_conf_accuracy),
        calibration_gap_noisy=float(
            base.calibration_gap),
        calibrated_under_noise=bool(base.calibrated),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UncertaintyLayerV2Witness:
    composite_confidence: float
    disagreement_weight_factor: float
    level: str
    component_with_lowest_confidence: str
    rationale: str
    calibrated_clean: bool
    calibration_gap_clean: float
    calibrated_under_noise: bool
    calibration_gap_noisy: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_confidence": float(round(
                self.composite_confidence, 12)),
            "disagreement_weight_factor": float(round(
                self.disagreement_weight_factor, 12)),
            "level": str(self.level),
            "component_with_lowest_confidence": str(
                self.component_with_lowest_confidence),
            "rationale": str(self.rationale),
            "calibrated_clean": bool(self.calibrated_clean),
            "calibration_gap_clean": float(round(
                self.calibration_gap_clean, 12)),
            "calibrated_under_noise": bool(
                self.calibrated_under_noise),
            "calibration_gap_noisy": float(round(
                self.calibration_gap_noisy, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w54_uncertainty_layer_v2_witness",
            "witness": self.to_dict()})


def emit_uncertainty_layer_v2_witness(
        *,
        report: UncertaintyReportV2,
        calibration_clean: CalibrationCheckResult,
        calibration_noisy: NoiseCalibrationResult,
) -> UncertaintyLayerV2Witness:
    return UncertaintyLayerV2Witness(
        composite_confidence=float(
            report.composite_confidence),
        disagreement_weight_factor=float(
            report.disagreement_weight_factor),
        level=str(report.level),
        component_with_lowest_confidence=str(
            report.component_with_lowest_confidence),
        rationale=str(report.rationale),
        calibrated_clean=bool(calibration_clean.calibrated),
        calibration_gap_clean=float(
            calibration_clean.calibration_gap),
        calibrated_under_noise=bool(
            calibration_noisy.calibrated_under_noise),
        calibration_gap_noisy=float(
            calibration_noisy.calibration_gap_noisy),
        n_probes=int(
            calibration_clean.n_high_conf
            + calibration_clean.n_low_conf),
    )


# =============================================================================
# Verifier
# =============================================================================

W54_UNCERT_V2_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w54_uncert_v2_composite_out_of_bounds",
    "w54_uncert_v2_level_invalid",
    "w54_uncert_v2_calibration_gap_below_floor",
    "w54_uncert_v2_calibrated_under_noise_required_but_not",
    "w54_uncert_v2_disagreement_weight_out_of_bounds",
    "w54_uncert_v2_rationale_missing",
)


def verify_uncertainty_layer_v2_witness(
        witness: UncertaintyLayerV2Witness,
        *,
        require_calibrated_clean: bool = False,
        require_calibrated_under_noise: bool = False,
        min_calibration_gap_clean: float | None = None,
        min_calibration_gap_noisy: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if not (0.0 <= float(witness.composite_confidence)
            <= 1.0):
        failures.append(
            "w54_uncert_v2_composite_out_of_bounds")
    if witness.level not in (
            W53_UNCERT_LEVEL_HIGH,
            W53_UNCERT_LEVEL_MEDIUM,
            W53_UNCERT_LEVEL_LOW,
            W53_UNCERT_LEVEL_ABSTAIN):
        failures.append("w54_uncert_v2_level_invalid")
    if (require_calibrated_clean
            and not witness.calibrated_clean):
        failures.append(
            "w54_uncert_v2_calibration_gap_below_floor")
    if (require_calibrated_under_noise
            and not witness.calibrated_under_noise):
        failures.append(
            "w54_uncert_v2_calibrated_under_noise_required_but_not")
    if (min_calibration_gap_clean is not None
            and witness.calibration_gap_clean
            < float(min_calibration_gap_clean)):
        failures.append(
            "w54_uncert_v2_calibration_gap_below_floor")
    if (min_calibration_gap_noisy is not None
            and witness.calibration_gap_noisy
            < float(min_calibration_gap_noisy)):
        failures.append(
            "w54_uncert_v2_calibration_gap_below_floor")
    if not (0.0 <= float(witness.disagreement_weight_factor)
            <= 1.0):
        failures.append(
            "w54_uncert_v2_disagreement_weight_out_of_bounds")
    if not witness.rationale:
        failures.append("w54_uncert_v2_rationale_missing")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W54_UNCERT_V2_SCHEMA_VERSION",
    "W54_UNCERT_V2_RATIONALE_HIGH",
    "W54_UNCERT_V2_RATIONALE_MEDIUM",
    "W54_UNCERT_V2_RATIONALE_LOW",
    "W54_UNCERT_V2_RATIONALE_ABSTAIN",
    "W54_UNCERT_V2_RATIONALE_DISAGREEMENT",
    "W54_UNCERT_V2_VERIFIER_FAILURE_MODES",
    "UncertaintyReportV2",
    "NoiseCalibrationResult",
    "UncertaintyLayerV2Witness",
    "compose_uncertainty_report_v2",
    "calibration_check_under_noise",
    "emit_uncertainty_layer_v2_witness",
    "verify_uncertainty_layer_v2_witness",
]
