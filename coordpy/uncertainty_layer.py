"""W53 M10 — Uncertainty / Confidence Layer.

Composes the per-component confidence scalars produced by:

* persistent V5 (chain_walk_depth + update_gate_l1_sum)
* multi-hop V3 (per-edge confidence + arbitration uncertainty)
* mergeable capsule (parents' confidence + audit count)
* deep proxy V4 (corruption flag + corruption confidence)
* corruption-robust carrier (silent_failure_rate)

into a single per-turn ``UncertaintyReport`` that callers can
use to gate downstream behaviour (transcript fallback,
abstention, K-of-N quorum thresholds).

The composition is a learned/heuristic weighted sum + a
*calibration check*: high-confidence reports must be strictly
more accurate than low-confidence reports across a probe set.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Sequence


# =============================================================================
# Schema, defaults
# =============================================================================

W53_UNCERT_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer.v1")

W53_UNCERT_LEVEL_HIGH: str = "high"
W53_UNCERT_LEVEL_MEDIUM: str = "medium"
W53_UNCERT_LEVEL_LOW: str = "low"
W53_UNCERT_LEVEL_ABSTAIN: str = "abstain"

W53_DEFAULT_UNCERT_HIGH_THRESHOLD: float = 0.75
W53_DEFAULT_UNCERT_MEDIUM_THRESHOLD: float = 0.45
W53_DEFAULT_UNCERT_LOW_THRESHOLD: float = 0.20


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


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# =============================================================================
# UncertaintyReport
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UncertaintyReport:
    """Composite per-turn confidence."""

    persistent_v5_confidence: float
    multi_hop_v3_confidence: float
    mlsc_capsule_confidence: float
    deep_v4_corruption_confidence: float
    crc_silent_failure_rate: float
    composite_confidence: float
    level: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_v5_confidence": float(round(
                self.persistent_v5_confidence, 12)),
            "multi_hop_v3_confidence": float(round(
                self.multi_hop_v3_confidence, 12)),
            "mlsc_capsule_confidence": float(round(
                self.mlsc_capsule_confidence, 12)),
            "deep_v4_corruption_confidence": float(round(
                self.deep_v4_corruption_confidence, 12)),
            "crc_silent_failure_rate": float(round(
                self.crc_silent_failure_rate, 12)),
            "composite_confidence": float(round(
                self.composite_confidence, 12)),
            "level": str(self.level),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_uncertainty_report",
            "report": self.to_dict()})


def compose_uncertainty_report(
        *,
        persistent_v5_confidence: float = 1.0,
        multi_hop_v3_confidence: float = 1.0,
        mlsc_capsule_confidence: float = 1.0,
        deep_v4_corruption_confidence: float = 1.0,
        crc_silent_failure_rate: float = 0.0,
        weights: tuple[float, float, float, float, float] = (
            0.25, 0.20, 0.20, 0.20, 0.15),
        high_threshold: float = (
            W53_DEFAULT_UNCERT_HIGH_THRESHOLD),
        medium_threshold: float = (
            W53_DEFAULT_UNCERT_MEDIUM_THRESHOLD),
        low_threshold: float = (
            W53_DEFAULT_UNCERT_LOW_THRESHOLD),
) -> UncertaintyReport:
    """Composite = weighted sum, with crc_silent_failure_rate
    interpreted as ``1 - silent_failure_rate`` (penalty)."""
    crc_safety = 1.0 - float(crc_silent_failure_rate)
    crc_safety = float(max(0.0, min(1.0, crc_safety)))
    composite = (
        float(weights[0])
        * float(max(0.0, min(1.0, persistent_v5_confidence)))
        + float(weights[1])
        * float(max(0.0, min(1.0, multi_hop_v3_confidence)))
        + float(weights[2])
        * float(max(0.0, min(1.0, mlsc_capsule_confidence)))
        + float(weights[3])
        * float(max(0.0, min(1.0,
                              deep_v4_corruption_confidence)))
        + float(weights[4]) * float(crc_safety))
    composite = float(max(0.0, min(1.0, composite)))
    if composite >= float(high_threshold):
        level = W53_UNCERT_LEVEL_HIGH
    elif composite >= float(medium_threshold):
        level = W53_UNCERT_LEVEL_MEDIUM
    elif composite >= float(low_threshold):
        level = W53_UNCERT_LEVEL_LOW
    else:
        level = W53_UNCERT_LEVEL_ABSTAIN
    return UncertaintyReport(
        persistent_v5_confidence=float(
            persistent_v5_confidence),
        multi_hop_v3_confidence=float(
            multi_hop_v3_confidence),
        mlsc_capsule_confidence=float(
            mlsc_capsule_confidence),
        deep_v4_corruption_confidence=float(
            deep_v4_corruption_confidence),
        crc_silent_failure_rate=float(
            crc_silent_failure_rate),
        composite_confidence=float(composite),
        level=str(level),
    )


# =============================================================================
# Calibration check
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CalibrationCheckResult:
    """Result of comparing high vs low confidence accuracy."""

    n_high_conf: int
    n_low_conf: int
    high_conf_accuracy: float
    low_conf_accuracy: float
    calibration_gap: float
    calibrated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_high_conf": int(self.n_high_conf),
            "n_low_conf": int(self.n_low_conf),
            "high_conf_accuracy": float(round(
                self.high_conf_accuracy, 12)),
            "low_conf_accuracy": float(round(
                self.low_conf_accuracy, 12)),
            "calibration_gap": float(round(
                self.calibration_gap, 12)),
            "calibrated": bool(self.calibrated),
        }


def calibration_check(
        confidences: Sequence[float],
        accuracies: Sequence[float],
        *,
        high_threshold: float = (
            W53_DEFAULT_UNCERT_HIGH_THRESHOLD),
        low_threshold: float = (
            W53_DEFAULT_UNCERT_LOW_THRESHOLD),
        min_calibration_gap: float = 0.10,
) -> CalibrationCheckResult:
    """Test that high-confidence > low-confidence accuracy."""
    n = min(len(confidences), len(accuracies))
    high_acc: list[float] = []
    low_acc: list[float] = []
    for i in range(n):
        c = float(confidences[i])
        a = float(accuracies[i])
        if c >= float(high_threshold):
            high_acc.append(a)
        elif c <= float(low_threshold):
            low_acc.append(a)
    h_mean = (
        float(sum(high_acc) / max(1, len(high_acc)))
        if high_acc else 0.0)
    l_mean = (
        float(sum(low_acc) / max(1, len(low_acc)))
        if low_acc else 0.0)
    gap = float(h_mean - l_mean)
    calibrated = (
        len(high_acc) > 0 and len(low_acc) > 0
        and gap >= float(min_calibration_gap))
    return CalibrationCheckResult(
        n_high_conf=int(len(high_acc)),
        n_low_conf=int(len(low_acc)),
        high_conf_accuracy=float(h_mean),
        low_conf_accuracy=float(l_mean),
        calibration_gap=float(gap),
        calibrated=bool(calibrated),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UncertaintyLayerWitness:
    composite_confidence: float
    level: str
    calibrated: bool
    calibration_gap: float
    high_conf_accuracy: float
    low_conf_accuracy: float
    n_probes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_confidence": float(round(
                self.composite_confidence, 12)),
            "level": str(self.level),
            "calibrated": bool(self.calibrated),
            "calibration_gap": float(round(
                self.calibration_gap, 12)),
            "high_conf_accuracy": float(round(
                self.high_conf_accuracy, 12)),
            "low_conf_accuracy": float(round(
                self.low_conf_accuracy, 12)),
            "n_probes": int(self.n_probes),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w53_uncertainty_layer_witness",
            "witness": self.to_dict()})


def emit_uncertainty_layer_witness(
        *,
        report: UncertaintyReport,
        calibration: CalibrationCheckResult,
) -> UncertaintyLayerWitness:
    return UncertaintyLayerWitness(
        composite_confidence=float(
            report.composite_confidence),
        level=str(report.level),
        calibrated=bool(calibration.calibrated),
        calibration_gap=float(calibration.calibration_gap),
        high_conf_accuracy=float(
            calibration.high_conf_accuracy),
        low_conf_accuracy=float(
            calibration.low_conf_accuracy),
        n_probes=int(
            calibration.n_high_conf
            + calibration.n_low_conf),
    )


# =============================================================================
# Verifier
# =============================================================================


W53_UNCERT_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w53_uncert_composite_out_of_bounds",
    "w53_uncert_level_invalid",
    "w53_uncert_calibration_gap_below_floor",
    "w53_uncert_calibrated_required_but_not",
)


def verify_uncertainty_layer_witness(
        witness: UncertaintyLayerWitness,
        *,
        require_calibrated: bool = False,
        min_calibration_gap: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if not (0.0 <= float(witness.composite_confidence)
            <= 1.0):
        failures.append(
            "w53_uncert_composite_out_of_bounds")
    if witness.level not in (
            W53_UNCERT_LEVEL_HIGH,
            W53_UNCERT_LEVEL_MEDIUM,
            W53_UNCERT_LEVEL_LOW,
            W53_UNCERT_LEVEL_ABSTAIN):
        failures.append("w53_uncert_level_invalid")
    if (require_calibrated and not witness.calibrated):
        failures.append(
            "w53_uncert_calibrated_required_but_not")
    if (min_calibration_gap is not None
            and witness.calibration_gap
            < float(min_calibration_gap)):
        failures.append(
            "w53_uncert_calibration_gap_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W53_UNCERT_SCHEMA_VERSION",
    "W53_UNCERT_LEVEL_HIGH",
    "W53_UNCERT_LEVEL_MEDIUM",
    "W53_UNCERT_LEVEL_LOW",
    "W53_UNCERT_LEVEL_ABSTAIN",
    "W53_DEFAULT_UNCERT_HIGH_THRESHOLD",
    "W53_DEFAULT_UNCERT_MEDIUM_THRESHOLD",
    "W53_DEFAULT_UNCERT_LOW_THRESHOLD",
    "W53_UNCERT_VERIFIER_FAILURE_MODES",
    "UncertaintyReport",
    "CalibrationCheckResult",
    "UncertaintyLayerWitness",
    "compose_uncertainty_report",
    "calibration_check",
    "emit_uncertainty_layer_witness",
    "verify_uncertainty_layer_witness",
]
