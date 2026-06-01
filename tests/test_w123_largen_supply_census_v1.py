"""W123 Lane-alpha tests — official ICPC large-n supply census.

Locks the live-verified finding: the RESISTANT side is hard-capped at 4
post-cutoff surfaces (51 raw / ~45 tier-1, all already mined by W120, below 100
even at 100% yield), while the EXPOSED side scales past 100 (135 raw / ~113
tier-1) — so the >=100 MATCHED battlefield is blocked SOLELY by post-cutoff
supply.  Also checks the gate is falsifiable (a synthetic both-sides-rich census
flips to CONSTRUCTIBLE).
"""

from coordpy.icpc_largen_supply_census_v1 import (
    MAVERICK_CUTOFF_BOUNDARY,
    OFFICIAL_ICPC_SURFACE_CENSUS_V1,
    SurfaceCensusV1,
    assess_largen_supply_v1,
)


# ---- the real official-org supply ----------------------------------------

def test_matched_battlefield_unreachable_resistant_is_the_cap():
    v = assess_largen_supply_v1()
    assert v["verdict"] == "LARGEN_MATCHED_BATTLEFIELD_UNREACHABLE_OFFICIAL_FAMILY"
    assert v["largen_matched_battlefield_constructible"] is False
    assert v["largen_spend_gate_open"] is False
    # resistant is the binding cap; exposed actually scales past 100
    assert v["resistant"]["reaches_target_even_at_upper_bound"] is False
    assert v["exposed"]["reaches_target_even_at_upper_bound"] is True


def test_only_four_post_cutoff_surfaces_all_mined_by_w120():
    res = [s for s in OFFICIAL_ICPC_SURFACE_CENSUS_V1
           if s.is_post_cutoff(MAVERICK_CUTOFF_BOUNDARY)]
    assert len(res) == 4
    assert {s.key for s in res} == {
        "RMRC:2024-2025", "RMRC:2025-2026", "ECNA:2024-2025", "ECNA:2025-2026"}
    assert all(s.used_by == "W120" for s in res)


def test_resistant_raw_supply_equals_w120_n_seen():
    v = assess_largen_supply_v1()
    # 13 + 13 + 13 + 12 = 51 raw == W120 n_seen (consistency anchor)
    assert v["resistant"]["raw_problem_packages"] == 51
    assert v["resistant"]["n_surfaces"] == 4
    # below 100 even at a 100% tier-1 yield -> structural cap
    assert v["resistant"]["raw_upper_bound"] < 100
    assert v["resistant"]["deficit_vs_target_estimated"] >= 50


def test_exposed_scales_past_100():
    v = assess_largen_supply_v1()
    # RMRC 2017/18/19/20/21/22-23 (69) + ECNA 2019-20..2023-24 (66) = 135 raw
    assert v["exposed"]["raw_problem_packages"] == 135
    assert v["exposed"]["est_tier1"] >= 100
    assert v["exposed"]["reaches_target_even_at_upper_bound"] is True


def test_only_zero_package_repos_are_excluded():
    nonpkg = {s.key for s in OFFICIAL_ICPC_SURFACE_CENSUS_V1 if not s.package_bearing}
    # RMRC 2023-2024 ships 0 problem.yaml; mid-atlantic is a README stub
    assert nonpkg == {"RMRC:2023-2024", "MIDATL:stub"}
    # the older RMRC repos DO ship packages and count as exposed
    pkg = {s.key for s in OFFICIAL_ICPC_SURFACE_CENSUS_V1 if s.package_bearing}
    assert {"RMRC:2017", "RMRC:2018", "RMRC:2019", "RMRC:2020"} <= pkg
    v = assess_largen_supply_v1()
    assert "icpc/na-mid-atlantic-public" in v["non_package_org_repos"]


# ---- falsifiability: enough supply on BOTH sides opens the gate ----------

def _big_census():
    surfaces = []
    for i in range(9):  # 9 post-cutoff seasons x 15 = 135 raw resistant
        surfaces.append(SurfaceCensusV1(
            f"RMRC:future-{i}", "RMRC", f"icpc/future-{i}", "2025-09-01", 15, True, ""))
    for i in range(9):  # 9 pre-cutoff seasons x 15 = 135 raw exposed
        surfaces.append(SurfaceCensusV1(
            f"ECNA:old-{i}", "ECNA", f"icpc/old-{i}", "2020-11-01", 15, True, ""))
    return tuple(surfaces)


def test_synthetic_large_supply_opens_gate():
    v = assess_largen_supply_v1(_big_census())
    assert v["resistant"]["reaches_target_estimated"] is True
    assert v["exposed"]["reaches_target_estimated"] is True
    assert v["verdict"] == "LARGEN_MATCHED_BATTLEFIELD_CONSTRUCTIBLE"
    assert v["largen_spend_gate_open"] is True


def test_census_cid_deterministic():
    assert assess_largen_supply_v1()["census_cid"] == assess_largen_supply_v1()["census_cid"]
