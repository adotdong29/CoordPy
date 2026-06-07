"""W139 unit tests — per-tier band gate + cross-scale report (no NIM)."""
from __future__ import annotations

from coordpy.per_tier_band_calibration_v1 import (
    LADDER_V2, CX_KNOB_GRID_V139, FUNC_KNOB_GRID_V139, W139_FAMILIES,
    PerTierStatV1, PerTierCellVerdictV1, build_per_tier_band_report_v1)


def _stat(tier, is_anchor, a0, a1):
    return PerTierStatV1(model_id=f"m_{tier}", tier=tier, is_anchor=is_anchor,
                         a0_passed=tuple(a0), a1_passed=tuple(a1), n_calls=len(a0) + len(a1) * 5)


def test_ladder_v2_has_three_tiers_incl_anchor():
    tiers = [m.tier for m in LADDER_V2]
    assert tiers == ["small", "mid", "strong"]
    assert sum(1 for m in LADDER_V2 if m.is_anchor) == 1
    assert next(m for m in LADDER_V2 if m.is_anchor).model_id == "meta/llama-3.3-70b-instruct"


def test_in_band_culls_saturated_and_dead_admits_intermediate():
    T, F = True, False
    sat = _stat("strong", True, [T] * 5, [T] * 8)        # a1=1.0 -> Wilson hi==1 -> cull
    dead = _stat("strong", True, [F] * 5, [F] * 8)        # a1=0.0 -> Wilson lo==0 -> cull
    inter = _stat("strong", True, [F] * 5, [T, T, T, T, F, F, F, F])  # a1=0.5 -> admit
    assert not sat.in_band()
    assert not dead.in_band()
    assert inter.in_band()


def test_in_band_rejects_one_shot_saturated_a0():
    # a1 intermediate but a0 already saturated (>=0.80) -> not informative
    s = _stat("strong", True, [True] * 5, [True, True, True, True, False, False, False, False])
    assert s.a0_rate >= 0.80
    assert not s.in_band()


def test_per_tier_report_band_for_tier_and_shared_families():
    # count_pairs in-band on strong+mid; subarrays in-band on small only -> shared = {count_pairs}
    T, F = True, False
    inter = [T, T, T, T, F, F, F, F]
    dead = [F] * 8
    c1 = PerTierCellVerdictV1("count_pairs_sum_le_t@20000", "count_pairs_sum_le_t", "COMPLEXITY_BLIND",
                             20000, (_stat("small", False, [F] * 5, dead),
                                     _stat("mid", False, [F] * 5, inter),
                                     _stat("strong", True, [F] * 5, inter)))
    c2 = PerTierCellVerdictV1("subarrays_sum_and_range@1500", "subarrays_sum_and_range",
                             "HIDDEN_EDGE_STATE_MISS", 1500,
                             (_stat("small", False, [F] * 5, inter),
                              _stat("mid", False, [T] * 5, [T] * 8),
                              _stat("strong", True, [T] * 5, [T] * 8)))
    rep = build_per_tier_band_report_v1([c1, c2], ladder=LADDER_V2, n_cal=5, K=5)
    assert "count_pairs_sum_le_t" in rep.band_for_tier("strong")
    assert "count_pairs_sum_le_t" in rep.band_for_tier("mid")
    assert "subarrays_sum_and_range" in rep.band_for_tier("small")
    # shared = in anchor band AND >=1 other tier band
    assert rep.shared_families() == ("count_pairs_sum_le_t",)
    cid = rep.per_tier_calibration_cid()
    assert isinstance(cid, str) and len(cid) == 64


def test_locked_grids_and_families():
    assert CX_KNOB_GRID_V139 == (2000, 6000, 20000, 50000)
    assert FUNC_KNOB_GRID_V139 == (1500, 4000, 12000, 30000)
    assert "subarrays_sum_and_range" in W139_FAMILIES          # >=1 HIDDEN_EDGE
    assert "count_pairs_sum_le_t" in W139_FAMILIES             # >=1 COMPLEXITY
