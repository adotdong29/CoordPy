"""W139 / COO-9 — per-MODEL (per-tier) headroom-band calibration (A1-as-RATE at EVERY tier).

W138 measured A1 as a population rate but ONLY at the strong anchor (``model_ladder_calibration_v1.
calibrate_template_v1`` runs A1 at ``cm.is_anchor`` only), so the headroom band was the ANCHOR's band.
The W138 cross-tier FAIL was a direct consequence: ``meta/llama-3.1-70b-instruct`` was tested on the
anchor's band where it is near-SATURATED (A1≈70%), so blind reflexion already maxed it and the witness
could only TIE.  The fix (Fluid Benchmarking arXiv:2509.11106: the value of a benchmark item depends on
the model's capability, so items should be selected PER MODEL at peak Fisher information) is to measure
A1 as a rate at EVERY tier and admit a SEPARATE per-tier band — each tier tested at its own p≈0.5 point.

This module mints ``n_cal`` calibration instances ONCE per (family, knob) cell (so every tier sees the
SAME instances — a matched comparison) and runs A0 (temp 0) + A1 (K, temp 0.7, any-of-K) at each tier,
then applies the per-tier band gate HT3 (intermediate a1_rate with a Wilson-95% interval excluding 0
and 1, and a0 not one-shot-saturated).  It reuses ``_run_a0`` / ``_run_a1`` / ``wilson_interval_v1`` /
``mint_problem_v1`` VERBATIM.  The witness-usability capability prior (the signal the W139 capability-
matched controller routes on) is measured separately in ``capability_matched_witness_compiler_v1``.

Pure / deterministic except the (already-audited) program-execution subprocess; NO model inference
lives here (that is the W139 driver script).  Explicit-import only; ``coordpy/__init__.py`` untouched.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Sequence

from .headroom_band_slate_v3 import BandCandidateV1
from .headroom_band_calibration_v2 import BAND_HI, BAND_LO, HC3_CEILING, wilson_interval_v1
from .model_ladder_calibration_v1 import CalibrationModelV1, GenFn, _run_a0, _run_a1
from .resistant_by_construction_battlefield_v1 import _sha256_hex, mint_problem_v1

PER_TIER_BAND_CALIBRATION_V1_SCHEMA_VERSION: str = "coordpy.per_tier_band_calibration_v1.v1"

# calibration seeds: DISJOINT from the corpus split seed ranges (139_2xx..139_4xx) and the W138 base.
PER_TIER_CALIBRATION_SEED_BASE: int = 139_900_000

# Ladder V2: the W138 small + anchor PLUS the mid tier W138 found anchor-band-saturated.  The mid tier
# is exactly the model W138 could only TIE on; per-tier recalibration tests it at ITS OWN band.
LADDER_V2: tuple[CalibrationModelV1, ...] = (
    CalibrationModelV1(model_id="meta/llama-3.1-8b-instruct", tier="small", is_anchor=False),
    CalibrationModelV1(model_id="meta/llama-3.1-70b-instruct", tier="mid", is_anchor=False),
    CalibrationModelV1(model_id="meta/llama-3.3-70b-instruct", tier="strong", is_anchor=True),
)

# RUNBOOK_W139 §4 — the continuous-N knob grids (easy->hard so EACH tier has a p≈0.5 knob) and the
# locked shared family set (>=1 complexity TIMEOUT family + >=1 HIDDEN_EDGE WRONG_ANSWER family).
CX_KNOB_GRID_V139: tuple[int, ...] = (2_000, 6_000, 20_000, 50_000)
FUNC_KNOB_GRID_V139: tuple[int, ...] = (1_500, 4_000, 12_000, 30_000)
W139_FAMILIES: tuple[str, ...] = (
    "count_pairs_sum_le_t", "sum_nearest_smaller_left", "count_subarrays_sum_le_s",
    "count_pairs_absdiff_le_d", "max_j_minus_i_le",        # COMPLEXITY (TIMEOUT)
    "subarrays_sum_and_range",                              # HIDDEN_EDGE (WRONG_ANSWER)
)


@dataclasses.dataclass(frozen=True)
class PerTierStatV1:
    """One tier's A0/A1-as-rate measurement on one (family, knob) cell."""
    model_id: str
    tier: str
    is_anchor: bool
    a0_passed: tuple[bool, ...]
    a1_passed: tuple[bool, ...]
    n_calls: int

    @property
    def a0_rate(self) -> float:
        return sum(self.a0_passed) / len(self.a0_passed) if self.a0_passed else 0.0

    @property
    def a1_rate(self) -> float:
        return sum(self.a1_passed) / len(self.a1_passed) if self.a1_passed else 0.0

    @property
    def best_rate(self) -> float:
        return max(self.a0_rate, self.a1_rate)

    @property
    def wilson(self) -> tuple[float, float]:
        return wilson_interval_v1(int(sum(self.a1_passed)), len(self.a1_passed))

    def in_band(self, *, band_lo: float = BAND_LO, band_hi: float = BAND_HI,
                a0_ceil: float = HC3_CEILING) -> bool:
        """HT3 for this tier: intermediate a1_rate (Wilson excludes 0,1) AND a0 not saturated."""
        lo, hi = self.wilson
        return bool(band_lo <= self.a1_rate <= band_hi and lo > 0.0 and hi < 1.0
                    and self.a0_rate < a0_ceil)

    @property
    def rank_key(self) -> float:
        """Distance from the IRT peak-information point (a1_rate ≈ 0.5)."""
        return abs(self.a1_rate - 0.5)

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.wilson
        return {"model_id": self.model_id, "tier": self.tier, "is_anchor": bool(self.is_anchor),
                "a0_rate": round(self.a0_rate, 4), "a1_rate": round(self.a1_rate, 4),
                "a1_passes": int(sum(self.a1_passed)), "n_a1": len(self.a1_passed),
                "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
                "in_band": self.in_band(), "n_calls": int(self.n_calls)}


@dataclasses.dataclass(frozen=True)
class PerTierCellVerdictV1:
    cell_id: str
    family: str
    mode: str
    knob_value: int
    per_tier: tuple[PerTierStatV1, ...]

    def tier(self, name: str) -> Optional[PerTierStatV1]:
        return next((t for t in self.per_tier if t.tier == name), None)

    def anchor(self) -> Optional[PerTierStatV1]:
        return next((t for t in self.per_tier if t.is_anchor), None)

    def in_band_tiers(self, **kw: Any) -> tuple[str, ...]:
        return tuple(t.tier for t in self.per_tier if t.in_band(**kw))

    def to_dict(self) -> dict[str, Any]:
        return {"cell_id": self.cell_id, "family": self.family, "mode": self.mode,
                "knob_value": int(self.knob_value),
                "in_band_tiers": list(self.in_band_tiers()),
                "per_tier": [t.to_dict() for t in self.per_tier]}


def calibrate_cell_per_tier_v1(cand: BandCandidateV1, *, gen_for_model: Callable[[str], GenFn],
                               ladder: Sequence[CalibrationModelV1] = LADDER_V2,
                               n_cal: int = 5, K: int = 5, max_tokens: int = 1536,
                               timeout_s: float = 8.0, mint_timeout_s: float = 1.0,
                               minted_date: str = "2026-06-07",
                               on_call: Optional[Callable[[str, str, int], None]] = None,
                               ) -> PerTierCellVerdictV1:
    """Calibrate ONE (family, knob) candidate at EVERY tier (A1-as-rate per tier).  Mints the
    calibration instances once (shared across tiers) so the per-tier comparison is matched."""
    pilots = []
    for r in range(int(n_cal)):
        mp = mint_problem_v1(cand.template.minted, global_seed=PER_TIER_CALIBRATION_SEED_BASE + r,
                             timeout_s=float(mint_timeout_s))
        pilots.append(mp.to_pilot_problem(minted_date=minted_date))

    per_tier: list[PerTierStatV1] = []
    for cm in ladder:
        gen = gen_for_model(cm.model_id)
        a0: list[bool] = []
        for i, p in enumerate(pilots):
            if on_call:
                on_call(cand.cell_id, cm.model_id, i)
            a0.append(_run_a0(p, gen, max_tokens=int(max_tokens), timeout_s=float(timeout_s)))
        a1: list[bool] = []
        for p in pilots:
            a1.append(_run_a1(p, gen, K=int(K), max_tokens=int(max_tokens),
                              timeout_s=float(timeout_s)))
        per_tier.append(PerTierStatV1(
            model_id=cm.model_id, tier=cm.tier, is_anchor=bool(cm.is_anchor),
            a0_passed=tuple(a0), a1_passed=tuple(a1), n_calls=len(pilots) + len(pilots) * int(K)))
    return PerTierCellVerdictV1(
        cell_id=cand.cell_id, family=cand.family, mode=cand.mode,
        knob_value=int(cand.knob_value), per_tier=tuple(per_tier))


@dataclasses.dataclass(frozen=True)
class PerTierBandReportV1:
    schema: str
    ladder: tuple[str, ...]
    n_cal: int
    K: int
    band_lo: float
    band_hi: float
    a0_ceil: float
    per_cell: tuple[PerTierCellVerdictV1, ...]

    def band_for_tier(self, tier: str) -> dict[str, PerTierCellVerdictV1]:
        """Per family, the cell whose ``tier``-stat is in-band with a1_rate closest to 0.5."""
        best: dict[str, PerTierCellVerdictV1] = {}
        best_rk: dict[str, float] = {}
        for c in self.per_cell:
            st = c.tier(tier)
            if st is None or not st.in_band(band_lo=self.band_lo, band_hi=self.band_hi,
                                            a0_ceil=self.a0_ceil):
                continue
            if c.family not in best or st.rank_key < best_rk[c.family]:
                best[c.family], best_rk[c.family] = c, st.rank_key
        return best

    def anchor_tier_name(self) -> str:
        for c in self.per_cell:
            a = c.anchor()
            if a is not None:
                return a.tier
        return "strong"

    def shared_families(self) -> tuple[str, ...]:
        """Families with an in-band cell at the ANCHOR AND on >=1 other tier (matched comparison)."""
        anchor = self.anchor_tier_name()
        anchor_fams = set(self.band_for_tier(anchor).keys())
        other_fams: set[str] = set()
        for t in {c.tier for cell in self.per_cell for c in cell.per_tier}:
            if t == anchor:
                continue
            other_fams |= set(self.band_for_tier(t).keys())
        return tuple(sorted(anchor_fams & other_fams))

    def per_tier_calibration_cid(self) -> str:
        return _sha256_hex({"k": "w139_per_tier_band_calibration_v1", "ladder": list(self.ladder),
                            "n_cal": self.n_cal, "K": self.K,
                            "band": [self.band_lo, self.band_hi], "a0_ceil": self.a0_ceil,
                            "cells": {c.cell_id: {t.tier: [round(t.a0_rate, 4), round(t.a1_rate, 4)]
                                                  for t in c.per_tier} for c in self.per_cell}})

    def all_tier_names(self) -> tuple[str, ...]:
        seen: list[str] = []
        for cell in self.per_cell:
            for t in cell.per_tier:
                if t.tier not in seen:
                    seen.append(t.tier)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        band_for_tier: dict[str, Any] = {}
        for t in self.all_tier_names():
            band_for_tier[t] = {
                f: {"cell_id": c.cell_id, "knob_value": c.knob_value,
                    "a1_rate": round(c.tier(t).a1_rate, 4) if c.tier(t) else None,
                    "a0_rate": round(c.tier(t).a0_rate, 4) if c.tier(t) else None,
                    "mode": c.mode}
                for f, c in self.band_for_tier(t).items()}
        return {
            "schema": self.schema, "ladder": list(self.ladder), "n_cal": self.n_cal, "K": self.K,
            "band_lo": self.band_lo, "band_hi": self.band_hi, "a0_ceil": self.a0_ceil,
            "anchor_tier": self.anchor_tier_name(),
            "shared_families": list(self.shared_families()),
            "band_for_tier": band_for_tier,
            "per_tier_calibration_cid": self.per_tier_calibration_cid(),
            "per_cell": [c.to_dict() for c in self.per_cell]}


def build_per_tier_band_report_v1(verdicts: Sequence[PerTierCellVerdictV1], *,
                                  ladder: Sequence[CalibrationModelV1] = LADDER_V2,
                                  n_cal: int, K: int, band_lo: float = BAND_LO,
                                  band_hi: float = BAND_HI, a0_ceil: float = HC3_CEILING,
                                  ) -> PerTierBandReportV1:
    return PerTierBandReportV1(
        schema=PER_TIER_BAND_CALIBRATION_V1_SCHEMA_VERSION,
        ladder=tuple(cm.model_id for cm in ladder), n_cal=int(n_cal), K=int(K),
        band_lo=float(band_lo), band_hi=float(band_hi), a0_ceil=float(a0_ceil),
        per_cell=tuple(verdicts))


__all__ = [
    "PER_TIER_BAND_CALIBRATION_V1_SCHEMA_VERSION", "PER_TIER_CALIBRATION_SEED_BASE", "LADDER_V2",
    "CX_KNOB_GRID_V139", "FUNC_KNOB_GRID_V139", "W139_FAMILIES",
    "PerTierStatV1", "PerTierCellVerdictV1", "calibrate_cell_per_tier_v1",
    "PerTierBandReportV1", "build_per_tier_band_report_v1",
]
