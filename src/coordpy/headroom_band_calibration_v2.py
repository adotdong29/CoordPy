"""W138 / COO-9 — headroom-band calibration v2 (A1-as-RATE + Wilson + HB3/HB4 band admission).

W137's calibration measured A1 with ``n_a1=1`` (a single any-of-K draw on ONE instance), so the
strong-anchor a1_rate could only read {0,1} — the field looked BIMODAL by construction and
``count_pairs_absdiff_le_d`` was culled HC4_DEAD on a coin-flip the W137 mechanism bench then rescued
at +33pp.  This module fixes that: it measures A1 as a **population pass-rate over n_cal >= 8
instances** and admits a (family, knob) cell to the headroom band only when that rate is genuinely
INTERMEDIATE (Wilson-95% interval excludes 0 and 1) and the cell DISCRIMINATES across the ladder.

Band-admission (RUNBOOK_W138 §6): a cell is admitted iff
  * **HC3** strong-anchor A0 not one-shot-saturated  (a0_rate < ``hc3_ceiling`` = 0.80), AND
  * **HB3** strong-anchor a1_rate is intermediate: ``BAND_LO <= a1_rate <= BAND_HI`` AND the Wilson-95%
    interval of the pass count strictly excludes 0 and 1 (so it is not a small-sample {0,1}), AND
  * **HB4** cross-scale discrimination: ``strong_best_rate > small_best_rate`` (same-direction ladder
    signal; the IRT discrimination criterion — metabench arXiv:2407.12844, tinyBenchmarks
    arXiv:2402.14992).
Cells are RANKED by closeness to a1_rate ≈ 0.5 (IRT peak Fisher information; ATLAS arXiv:2511.04689),
and per FAMILY the best-ranked admitted knob is selected for the corpus.

Reuses ``model_ladder_calibration_v1.calibrate_template_v1`` VERBATIM for the A0/A1 measurement (only
with n_a0 / n_a1 raised); adds the Wilson interval + band verdict + per-family knob selection.  Pure /
deterministic / explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Optional, Sequence

from .headroom_band_slate_v3 import BandCandidateV1
from .model_ladder_calibration_v1 import (
    CalibrationModelV1, GenFn, LADDER_V1, TemplateCalibrationV1, calibrate_template_v1)
from .resistant_by_construction_battlefield_v1 import _sha256_hex

BAND_CALIBRATION_V2_SCHEMA_VERSION: str = "coordpy.headroom_band_calibration_v2.v1"

# RUNBOOK_W138 §6 band window (strong-anchor a1_rate point estimate).
BAND_LO: float = 0.15
BAND_HI: float = 0.85
HC3_CEILING: float = 0.80


def wilson_interval_v1(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials (n==0 -> (0,1))."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclasses.dataclass(frozen=True)
class BandCellVerdictV1:
    cell_id: str
    family: str
    mode: str
    knob_value: int
    strong_a0_rate: float
    strong_a1_rate: float
    strong_a1_passes: int
    n_a1: int
    small_a0_rate: float
    small_best_rate: float
    strong_best_rate: float
    wilson_lo: float
    wilson_hi: float
    hc3_headroom: bool        # strong a0 not saturated
    hb3_intermediate: bool    # a1_rate in band AND Wilson excludes 0,1
    hb4_discriminates: bool   # strong_best > small_best
    admitted: bool
    reason: str

    @property
    def rank_key(self) -> float:
        """Distance from the IRT peak-information point (a1_rate ≈ 0.5); lower is better."""
        return abs(self.strong_a1_rate - 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {"cell_id": self.cell_id, "family": self.family, "mode": self.mode,
                "knob_value": int(self.knob_value),
                "strong_a0_rate": round(self.strong_a0_rate, 4),
                "strong_a1_rate": round(self.strong_a1_rate, 4),
                "strong_a1_passes": int(self.strong_a1_passes), "n_a1": int(self.n_a1),
                "small_a0_rate": round(self.small_a0_rate, 4),
                "small_best_rate": round(self.small_best_rate, 4),
                "strong_best_rate": round(self.strong_best_rate, 4),
                "wilson_lo": round(self.wilson_lo, 4), "wilson_hi": round(self.wilson_hi, 4),
                "hc3_headroom": bool(self.hc3_headroom),
                "hb3_intermediate": bool(self.hb3_intermediate),
                "hb4_discriminates": bool(self.hb4_discriminates),
                "admitted": bool(self.admitted), "reason": self.reason,
                "rank_key": round(self.rank_key, 4)}


def band_verdict_v1(cal: TemplateCalibrationV1, *, cell_id: str, knob_value: int,
                    band_lo: float = BAND_LO, band_hi: float = BAND_HI,
                    hc3_ceiling: float = HC3_CEILING) -> BandCellVerdictV1:
    """Apply HC3 ∧ HB3 ∧ HB4 to a model-ladder calibration record."""
    strong = cal.anchor()
    small = next((m for m in cal.per_model if m.tier == "small"), None)
    s_a0 = strong.a0_rate if strong else 0.0
    s_a1 = strong.a1_rate if strong else 0.0
    s_passes = int(sum(strong.a1_passed)) if strong else 0
    n_a1 = len(strong.a1_passed) if strong else 0
    sm_a0 = small.a0_rate if small else 0.0
    s_best = strong.best_rate if strong else 0.0
    sm_best = small.best_rate if small else 0.0
    lo, hi = wilson_interval_v1(s_passes, n_a1)

    hc3 = bool(s_a0 < hc3_ceiling)
    wilson_excludes_extremes = bool(lo > 0.0 and hi < 1.0)
    hb3 = bool(band_lo <= s_a1 <= band_hi and wilson_excludes_extremes)
    hb4 = bool(s_best > sm_best)
    admitted = bool(hc3 and hb3 and hb4)
    if admitted:
        reason = "ADMITTED_BAND"
    elif not hc3:
        reason = f"HC3_SATURATED(a0={s_a0:.2f}>={hc3_ceiling})"
    elif not hb3 and s_a1 == 0.0:
        reason = "HB3_DEAD(a1_rate==0)"
    elif not hb3 and s_a1 == 1.0:
        reason = "HB3_SATURATED(a1_rate==1)"
    elif not hb3 and not wilson_excludes_extremes:
        reason = f"HB3_WILSON_TOUCHES_EXTREME(lo={lo:.2f},hi={hi:.2f})"
    elif not hb3:
        reason = f"HB3_OUT_OF_BAND(a1={s_a1:.2f} not in [{band_lo},{band_hi}])"
    else:
        reason = f"HB4_NO_DISCRIMINATION(strong={s_best:.2f}<=small={sm_best:.2f})"
    return BandCellVerdictV1(
        cell_id=cell_id, family=cal.family, mode=cal.mode, knob_value=int(knob_value),
        strong_a0_rate=s_a0, strong_a1_rate=s_a1, strong_a1_passes=s_passes, n_a1=n_a1,
        small_a0_rate=sm_a0, small_best_rate=sm_best, strong_best_rate=s_best,
        wilson_lo=lo, wilson_hi=hi, hc3_headroom=hc3, hb3_intermediate=hb3,
        hb4_discriminates=hb4, admitted=admitted, reason=reason)


@dataclasses.dataclass(frozen=True)
class BandCalibrationReportV1:
    schema: str
    ladder: tuple[str, ...]
    n_a0: int
    n_a1: int
    K_a1: int
    band_lo: float
    band_hi: float
    per_cell: tuple[BandCellVerdictV1, ...]

    def admitted_cells(self) -> tuple[BandCellVerdictV1, ...]:
        return tuple(c for c in self.per_cell if c.admitted)

    def best_knob_per_family(self) -> dict[str, BandCellVerdictV1]:
        """Per family, the admitted cell whose a1_rate is closest to 0.5 (peak information)."""
        best: dict[str, BandCellVerdictV1] = {}
        for c in sorted(self.admitted_cells(), key=lambda c: c.rank_key):
            best.setdefault(c.family, c)
        return best

    def surviving_families(self) -> tuple[str, ...]:
        return tuple(sorted(self.best_knob_per_family().keys()))

    def surviving_modes(self) -> tuple[str, ...]:
        return tuple(sorted({c.mode for c in self.best_knob_per_family().values()}))

    def calibration_cid(self) -> str:
        return _sha256_hex({"k": "w138_band_calibration_v2", "ladder": list(self.ladder),
                            "n_a0": self.n_a0, "n_a1": self.n_a1, "K_a1": self.K_a1,
                            "band": [self.band_lo, self.band_hi],
                            "verdicts": {c.cell_id: c.admitted for c in self.per_cell}})

    def to_dict(self) -> dict[str, Any]:
        bk = self.best_knob_per_family()
        return {"schema": self.schema, "ladder": list(self.ladder),
                "n_a0": self.n_a0, "n_a1": self.n_a1, "K_a1": self.K_a1,
                "band_lo": self.band_lo, "band_hi": self.band_hi,
                "n_admitted_cells": len(self.admitted_cells()),
                "surviving_families": list(self.surviving_families()),
                "surviving_modes": list(self.surviving_modes()),
                "n_surviving_families": len(self.surviving_families()),
                "n_surviving_modes": len(self.surviving_modes()),
                "best_knob_per_family": {f: c.to_dict() for f, c in bk.items()},
                "calibration_cid": self.calibration_cid(),
                "per_cell": [c.to_dict() for c in self.per_cell]}


def calibrate_band_cell_v1(cand: BandCandidateV1, *, gen_for_model: Callable[[str], GenFn],
                           ladder: Sequence[CalibrationModelV1] = LADDER_V1,
                           n_a0: int = 5, n_a1: int = 8, K_a1: int = 5,
                           max_tokens: int = 1536, timeout_s: float = 8.0,
                           mint_timeout_s: float = 1.0, minted_date: str = "2026-06-05",
                           hc3_ceiling: float = HC3_CEILING, band_lo: float = BAND_LO,
                           band_hi: float = BAND_HI,
                           on_call: Optional[Callable[[str, str, int], None]] = None,
                           ) -> BandCellVerdictV1:
    """Calibrate ONE (family, knob) candidate: A1-as-rate over n_a1 instances + band admission."""
    cal = calibrate_template_v1(
        cand.template, gen_for_model=gen_for_model, ladder=ladder, n_a0=int(n_a0),
        n_a1=int(n_a1), K_a1=int(K_a1), hc3_ceiling=hc3_ceiling, max_tokens=int(max_tokens),
        timeout_s=float(timeout_s), mint_timeout_s=float(mint_timeout_s),
        minted_date=minted_date, on_call=on_call)
    return band_verdict_v1(cal, cell_id=cand.cell_id, knob_value=cand.knob_value,
                           band_lo=band_lo, band_hi=band_hi, hc3_ceiling=hc3_ceiling)


def build_band_calibration_report_v1(verdicts: Sequence[BandCellVerdictV1], *,
                                     ladder: Sequence[CalibrationModelV1] = LADDER_V1,
                                     n_a0: int, n_a1: int, K_a1: int,
                                     band_lo: float = BAND_LO, band_hi: float = BAND_HI,
                                     ) -> BandCalibrationReportV1:
    return BandCalibrationReportV1(
        schema=BAND_CALIBRATION_V2_SCHEMA_VERSION,
        ladder=tuple(cm.model_id for cm in ladder), n_a0=int(n_a0), n_a1=int(n_a1),
        K_a1=int(K_a1), band_lo=float(band_lo), band_hi=float(band_hi),
        per_cell=tuple(verdicts))


__all__ = [
    "BAND_CALIBRATION_V2_SCHEMA_VERSION", "BAND_LO", "BAND_HI", "HC3_CEILING",
    "wilson_interval_v1", "BandCellVerdictV1", "band_verdict_v1",
    "BandCalibrationReportV1", "calibrate_band_cell_v1", "build_band_calibration_report_v1",
]
