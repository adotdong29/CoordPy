"""W55 M10 — Uncertainty Layer V3.

Extends W54 V2 with three additions:

* **per-fact-tag uncertainty propagation** — instead of a single
  composite confidence, V3 carries a per-fact map
  ``{tag → (confidence, n_contributors)}`` that propagates
  through capsule merges. The fact-level uncertainty composes
  as the weighted geometric mean of contributors.
* **adversarial calibration check** — verifies calibration under
  bounded *adversarial* perturbation (worst-case bit-flip on the
  composite confidence direction), not just random noise.
* **trust-weighted composite confidence** — each component's
  confidence is multiplied by its trust scalar before the
  composite is computed, so low-trust components contribute
  proportionally less.

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
    W53_UNCERT_LEVEL_ABSTAIN,
    W53_UNCERT_LEVEL_HIGH,
    W53_UNCERT_LEVEL_LOW,
    W53_UNCERT_LEVEL_MEDIUM,
    W53_DEFAULT_UNCERT_HIGH_THRESHOLD,
    W53_DEFAULT_UNCERT_LOW_THRESHOLD,
    W53_DEFAULT_UNCERT_MEDIUM_THRESHOLD,
    calibration_check,
)
from .uncertainty_layer_v2 import (
    NoiseCalibrationResult,
    W54_UNCERT_V2_RATIONALE_ABSTAIN,
    W54_UNCERT_V2_RATIONALE_DISAGREEMENT,
    W54_UNCERT_V2_RATIONALE_HIGH,
    W54_UNCERT_V2_RATIONALE_LOW,
    W54_UNCERT_V2_RATIONALE_MEDIUM,
    calibration_check_under_noise,
    compose_uncertainty_report_v2,
)


# =============================================================================
# Schema, defaults
# =============================================================================

W55_UNCERT_V3_SCHEMA_VERSION: str = (
    "coordpy.uncertainty_layer_v3.v1")

W55_UNCERT_V3_RATIONALE_TRUST_DOWNWEIGHT: str = (
    "trust_weighted_downweight_triggered")


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
# Per-fact uncertainty propagation
# =============================================================================


@dataclasses.dataclass(frozen=True)
class FactUncertainty:
    """One fact_tag's uncertainty propagation record."""

    tag: str
    confidence: float
    n_contributors: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": str(self.tag),
            "confidence": float(round(self.confidence, 12)),
            "n_contributors": int(self.n_contributors),
        }


def propagate_fact_uncertainty(
        per_parent_facts: Sequence[
            Sequence[FactUncertainty]],
) -> list[FactUncertainty]:
    """Propagate per-fact uncertainty across multiple parents.

    For each tag, the composite confidence is the geometric
    mean of all contributor confidences, weighted by 1/N.
    The contributor count sums across all parents.
    """
    tag_confs: dict[str, list[float]] = {}
    tag_counts: dict[str, int] = {}
    for parent_facts in per_parent_facts:
        for fu in parent_facts:
            tag_confs.setdefault(fu.tag, []).append(
                float(fu.confidence))
            tag_counts[fu.tag] = (
                tag_counts.get(fu.tag, 0)
                + int(fu.n_contributors))
    out: list[FactUncertainty] = []
    for tag, confs in sorted(tag_confs.items()):
        if not confs:
            geomean = 0.0
        else:
            # Geometric mean (clip below 1e-12 to avoid log(0)).
            log_sum = 0.0
            for c in confs:
                log_sum += math.log(max(1e-12, float(c)))
            geomean = float(math.exp(
                log_sum / float(len(confs))))
        out.append(FactUncertainty(
            tag=tag,
            confidence=float(max(0.0, min(1.0, geomean))),
            n_contributors=int(tag_counts.get(tag, 0)),
        ))
    return out


# =============================================================================
# Adversarial calibration check
# =============================================================================


def calibration_check_under_adversarial(
        confidences: Sequence[float],
        accuracies: Sequence[float],
        *,
        perturbation_magnitude: float = 0.1,
        seed: int = 0,
        min_calibration_gap: float = 0.10,
        n_trials: int = 8,
) -> NoiseCalibrationResult:
    """Calibration check under *worst-case* adversarial
    perturbation.

    For each trial, sweep ``n_trials`` directions of perturbation
    (each a sign vector). Adversarial sign biases each confidence
    toward the wrong side. Returns the *worst-case* gap across
    trials as a ``NoiseCalibrationResult``.
    """
    if not confidences or not accuracies:
        return NoiseCalibrationResult(
            n_high_conf=0, n_low_conf=0,
            noise_magnitude=float(perturbation_magnitude),
            high_conf_accuracy=0.0,
            low_conf_accuracy=0.0,
            calibration_gap_noisy=0.0,
            calibrated_under_noise=False,
        )
    rng = random.Random(int(seed))
    n = len(confidences)
    worst_gap = float("inf")
    worst_calibrated = True
    worst_high_acc = 0.0
    worst_low_acc = 0.0
    worst_n_high = 0
    worst_n_low = 0
    for trial in range(int(n_trials)):
        signs = [
            (1.0 if rng.random() > 0.5 else -1.0)
            for _ in range(n)
        ]
        perturbed: list[float] = []
        for i in range(n):
            adv_dir = (
                1.0 if accuracies[i] < 0.5 else -1.0)
            p = float(confidences[i]) + (
                float(perturbation_magnitude)
                * float(signs[i]) * float(adv_dir))
            perturbed.append(
                float(max(0.0, min(1.0, p))))
        res = calibration_check(
            perturbed, accuracies,
            min_calibration_gap=float(min_calibration_gap))
        if res.calibration_gap < worst_gap:
            worst_gap = float(res.calibration_gap)
            worst_calibrated = bool(res.calibrated)
            worst_high_acc = float(res.high_conf_accuracy)
            worst_low_acc = float(res.low_conf_accuracy)
            worst_n_high = int(res.n_high_conf)
            worst_n_low = int(res.n_low_conf)
    return NoiseCalibrationResult(
        n_high_conf=int(worst_n_high),
        n_low_conf=int(worst_n_low),
        noise_magnitude=float(perturbation_magnitude),
        high_conf_accuracy=float(worst_high_acc),
        low_conf_accuracy=float(worst_low_acc),
        calibration_gap_noisy=float(worst_gap),
        calibrated_under_noise=bool(worst_calibrated),
    )


# =============================================================================
# UncertaintyReportV3
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UncertaintyReportV3:
    """V3 composite report with per-fact + trust weighting +
    adversarial."""

    persistent_v7_confidence: float
    multi_hop_v5_confidence: float
    mlsc_v3_capsule_confidence: float
    deep_v6_corruption_confidence: float
    crc_v3_silent_failure_rate: float
    trust_weights: tuple[tuple[str, float], ...]
    composite_confidence: float
    trust_weighted_composite: float
    disagreement_weight_factor: float
    fact_uncertainties: tuple[FactUncertainty, ...]
    level: str
    component_with_lowest_confidence: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "persistent_v7_confidence": float(round(
                self.persistent_v7_confidence, 12)),
            "multi_hop_v5_confidence": float(round(
                self.multi_hop_v5_confidence, 12)),
            "mlsc_v3_capsule_confidence": float(round(
                self.mlsc_v3_capsule_confidence, 12)),
            "deep_v6_corruption_confidence": float(round(
                self.deep_v6_corruption_confidence, 12)),
            "crc_v3_silent_failure_rate": float(round(
                self.crc_v3_silent_failure_rate, 12)),
            "trust_weights": [
                [str(n), float(round(float(v), 12))]
                for (n, v) in self.trust_weights],
            "composite_confidence": float(round(
                self.composite_confidence, 12)),
            "trust_weighted_composite": float(round(
                self.trust_weighted_composite, 12)),
            "disagreement_weight_factor": float(round(
                self.disagreement_weight_factor, 12)),
            "fact_uncertainties": [
                fu.to_dict() for fu in self.fact_uncertainties],
            "level": str(self.level),
            "component_with_lowest_confidence": str(
                self.component_with_lowest_confidence),
            "rationale": str(self.rationale),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_uncertainty_report_v3",
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


def compose_uncertainty_report_v3(
        *,
        persistent_v7_confidence: float = 1.0,
        multi_hop_v5_confidence: float = 1.0,
        mlsc_v3_capsule_confidence: float = 1.0,
        deep_v6_corruption_confidence: float = 1.0,
        crc_v3_silent_failure_rate: float = 0.0,
        trust_weights: Mapping[str, float] | None = None,
        component_disagreements: (
            Mapping[str, float] | None) = None,
        fact_uncertainties: (
            Sequence[FactUncertainty] | None) = None,
        high_threshold: float = W53_DEFAULT_UNCERT_HIGH_THRESHOLD,
        medium_threshold: float = (
            W53_DEFAULT_UNCERT_MEDIUM_THRESHOLD),
        low_threshold: float = W53_DEFAULT_UNCERT_LOW_THRESHOLD,
) -> UncertaintyReportV3:
    """Compose a V3 uncertainty report.

    Trust-weighted composite: each component's confidence is
    multiplied by its trust scalar (defaulting to 1.0 if not
    given), then averaged (with the CRC inverted: 1 - silent
    failure rate).
    """
    v2_report = compose_uncertainty_report_v2(
        persistent_v6_confidence=float(
            persistent_v7_confidence),
        multi_hop_v4_confidence=float(
            multi_hop_v5_confidence),
        mlsc_v2_capsule_confidence=float(
            mlsc_v3_capsule_confidence),
        deep_v5_corruption_confidence=float(
            deep_v6_corruption_confidence),
        crc_v2_silent_failure_rate=float(
            crc_v3_silent_failure_rate),
        component_disagreements=dict(
            component_disagreements or {}),
        high_threshold=float(high_threshold),
        medium_threshold=float(medium_threshold),
        low_threshold=float(low_threshold))
    twm = dict(trust_weights or {})
    components = [
        ("persistent_v7", float(persistent_v7_confidence)),
        ("multi_hop_v5", float(multi_hop_v5_confidence)),
        ("mlsc_v3", float(mlsc_v3_capsule_confidence)),
        ("deep_v6", float(deep_v6_corruption_confidence)),
        ("crc_v3",
         float(1.0 - float(crc_v3_silent_failure_rate))),
    ]
    sum_w = 0.0
    sum_v = 0.0
    for name, c in components:
        t = float(twm.get(name, 1.0))
        t = float(max(0.0, min(1.0, t)))
        sum_w += t
        sum_v += t * float(c)
    if sum_w > 1e-30:
        tw_composite = float(sum_v / sum_w)
    else:
        tw_composite = float(v2_report.composite_confidence)
    tw_composite = float(max(0.0, min(1.0, tw_composite)))
    rationale = v2_report.rationale
    # If trust-weighted composite differs significantly from
    # untrusted composite, mention it.
    if (abs(tw_composite - v2_report.composite_confidence)
            >= 0.05):
        rationale = (
            W55_UNCERT_V3_RATIONALE_TRUST_DOWNWEIGHT)
    level = _level_for(
        tw_composite,
        high_threshold=float(high_threshold),
        medium_threshold=float(medium_threshold),
        low_threshold=float(low_threshold))
    fus = tuple(fact_uncertainties or ())
    return UncertaintyReportV3(
        persistent_v7_confidence=float(
            persistent_v7_confidence),
        multi_hop_v5_confidence=float(
            multi_hop_v5_confidence),
        mlsc_v3_capsule_confidence=float(
            mlsc_v3_capsule_confidence),
        deep_v6_corruption_confidence=float(
            deep_v6_corruption_confidence),
        crc_v3_silent_failure_rate=float(
            crc_v3_silent_failure_rate),
        trust_weights=tuple(sorted(twm.items())),
        composite_confidence=float(
            v2_report.composite_confidence),
        trust_weighted_composite=float(tw_composite),
        disagreement_weight_factor=float(
            v2_report.disagreement_weight_factor),
        fact_uncertainties=fus,
        level=str(level),
        component_with_lowest_confidence=str(
            v2_report.component_with_lowest_confidence),
        rationale=str(rationale),
    )


# =============================================================================
# Witness
# =============================================================================


@dataclasses.dataclass(frozen=True)
class UncertaintyLayerV3Witness:
    report_cid: str
    composite_confidence: float
    trust_weighted_composite: float
    disagreement_weight_factor: float
    calibration_clean_gap: float
    calibration_noisy_gap: float
    calibration_adversarial_gap: float
    calibrated_clean: bool
    calibrated_under_noise: bool
    calibrated_under_adversarial: bool
    level: str
    rationale: str
    n_fact_uncertainties: int
    component_with_lowest_confidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_cid": str(self.report_cid),
            "composite_confidence": float(round(
                self.composite_confidence, 12)),
            "trust_weighted_composite": float(round(
                self.trust_weighted_composite, 12)),
            "disagreement_weight_factor": float(round(
                self.disagreement_weight_factor, 12)),
            "calibration_clean_gap": float(round(
                self.calibration_clean_gap, 12)),
            "calibration_noisy_gap": float(round(
                self.calibration_noisy_gap, 12)),
            "calibration_adversarial_gap": float(round(
                self.calibration_adversarial_gap, 12)),
            "calibrated_clean": bool(self.calibrated_clean),
            "calibrated_under_noise": bool(
                self.calibrated_under_noise),
            "calibrated_under_adversarial": bool(
                self.calibrated_under_adversarial),
            "level": str(self.level),
            "rationale": str(self.rationale),
            "n_fact_uncertainties": int(
                self.n_fact_uncertainties),
            "component_with_lowest_confidence": str(
                self.component_with_lowest_confidence),
        }

    def cid(self) -> str:
        return _sha256_hex({
            "kind": "w55_uncertainty_layer_v3_witness",
            "witness": self.to_dict()})


def emit_uncertainty_layer_v3_witness(
        *,
        report: UncertaintyReportV3,
        calibration_clean: CalibrationCheckResult,
        calibration_noisy: NoiseCalibrationResult,
        calibration_adversarial: NoiseCalibrationResult,
) -> UncertaintyLayerV3Witness:
    return UncertaintyLayerV3Witness(
        report_cid=str(report.cid()),
        composite_confidence=float(report.composite_confidence),
        trust_weighted_composite=float(
            report.trust_weighted_composite),
        disagreement_weight_factor=float(
            report.disagreement_weight_factor),
        calibration_clean_gap=float(
            calibration_clean.calibration_gap),
        calibration_noisy_gap=float(
            calibration_noisy.calibration_gap_noisy),
        calibration_adversarial_gap=float(
            calibration_adversarial.calibration_gap_noisy),
        calibrated_clean=bool(calibration_clean.calibrated),
        calibrated_under_noise=bool(
            calibration_noisy.calibrated_under_noise),
        calibrated_under_adversarial=bool(
            calibration_adversarial
            .calibrated_under_noise),
        level=str(report.level),
        rationale=str(report.rationale),
        n_fact_uncertainties=int(
            len(report.fact_uncertainties)),
        component_with_lowest_confidence=str(
            report.component_with_lowest_confidence),
    )


# =============================================================================
# Verifier
# =============================================================================

W55_UNCERT_V3_VERIFIER_FAILURE_MODES: tuple[str, ...] = (
    "w55_uncert_v3_report_cid_mismatch",
    "w55_uncert_v3_composite_out_of_bounds",
    "w55_uncert_v3_trust_weighted_out_of_bounds",
    "w55_uncert_v3_calibration_gap_below_floor",
    "w55_uncert_v3_adversarial_gap_below_floor",
)


def verify_uncertainty_layer_v3_witness(
        witness: UncertaintyLayerV3Witness,
        *,
        expected_report_cid: str | None = None,
        min_calibration_gap: float | None = None,
        min_adversarial_gap: float | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if (expected_report_cid is not None
            and witness.report_cid
            != str(expected_report_cid)):
        failures.append("w55_uncert_v3_report_cid_mismatch")
    if not (
            0.0 <= float(witness.composite_confidence)
            <= 1.0):
        failures.append(
            "w55_uncert_v3_composite_out_of_bounds")
    if not (
            0.0
            <= float(witness.trust_weighted_composite)
            <= 1.0):
        failures.append(
            "w55_uncert_v3_trust_weighted_out_of_bounds")
    if (min_calibration_gap is not None
            and witness.calibration_clean_gap
            < float(min_calibration_gap)):
        failures.append(
            "w55_uncert_v3_calibration_gap_below_floor")
    if (min_adversarial_gap is not None
            and witness.calibration_adversarial_gap
            < float(min_adversarial_gap)):
        failures.append(
            "w55_uncert_v3_adversarial_gap_below_floor")
    return {
        "ok": (len(failures) == 0),
        "failures": failures,
        "witness_cid": witness.cid(),
    }


__all__ = [
    "W55_UNCERT_V3_SCHEMA_VERSION",
    "W55_UNCERT_V3_RATIONALE_TRUST_DOWNWEIGHT",
    "W55_UNCERT_V3_VERIFIER_FAILURE_MODES",
    "FactUncertainty",
    "UncertaintyReportV3",
    "UncertaintyLayerV3Witness",
    "propagate_fact_uncertainty",
    "calibration_check_under_adversarial",
    "compose_uncertainty_report_v3",
    "emit_uncertainty_layer_v3_witness",
    "verify_uncertainty_layer_v3_witness",
]
