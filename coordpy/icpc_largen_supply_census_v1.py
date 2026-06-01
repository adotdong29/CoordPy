"""CoordPy W123 — official ICPC large-n supply census (Lane alpha).

W122 closed the single-seed caveat but left the matched resistant-vs-exposed ICPC
contrast **unresolvable at n=30** (3-seed means resistant +4.44 / exposed +8.89,
both out-of-band, both non-mechanism-driven; branch B4).  The pre-committed W123
escalation is "larger n PER FIELD (>=100/field), NOT more n=30 seeds".

This module answers, deterministically and NIM-free, the binding Lane-alpha
question: **can the SAME official ``github.com/icpc`` package family be scaled to
>=100 pass-fail tasks on BOTH the post-cutoff (resistant) and pre-cutoff
(exposed) sides?**  It is a SUPPLY census over the official org, not a grader and
not a pilot.  It does not fetch models and does not spend.

The census is a pinned snapshot of every official-org problem-package surface,
verified LIVE via the GitHub API on 2026-05-31 (RMRC repos counted by
``problem.yaml``; the ECNA archive counted by per-year ``*.zip`` Kattis packages).
The companion script re-derives the counts live and asserts they still match.

Headline (see verdict): only FOUR post-cutoff package surfaces exist in the
official org (RMRC 2024-25, RMRC 2025-26, ECNA 2024-25, ECNA 2025-26) totalling
**51 raw problems** (= W120's n_seen), and W120 already mined ALL FOUR.  There is
no fifth post-cutoff official ICPC package surface, so the resistant side is
hard-capped at ~45 tier-1 (51 even at a 100% yield) — far below 100.  The
**exposed** side, by contrast, scales fine (135 raw across 11 pre-cutoff
surfaces, ~113 tier-1 >= 100).  So the >=100/field MATCHED battlefield cannot be
built — blocked SOLELY by post-cutoff (resistant) contest-package supply, not by
the rule, the grader, or curation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Sequence

# Same boundary the whole ICPC arc is anchored to (Maverick KNOWN Aug-2024).
MAVERICK_CUTOFF_BOUNDARY: str = "2024-08-31"

# The W123 escalation bar from RUNBOOK_W122 §8 / the B4 fire-condition.
TARGET_PER_FIELD: int = 100

MIN_RESISTANT_SLICE: int = 30  # referenced for context, not the W123 gate

# Observed tier-1 yields (core pure pass-fail / raw problems seen) from the
# already-graded W120 resistant and W121 exposed constructions.  Used ONLY to
# ESTIMATE tier-1 supply; the raw upper bound is also reported and is what the
# verdict's gate uses, so the estimate is never load-bearing on its own.
RESISTANT_TIER1_YIELD: float = 45.0 / 51.0   # W120: 45 tier-1 of 51 seen
EXPOSED_TIER1_YIELD: float = 42.0 / 50.0     # W121: 42 tier-1 of 50 seen

SCHEMA_VERSION: str = "coordpy.icpc_largen_supply_census_v1.v1"


@dataclasses.dataclass(frozen=True)
class SurfaceCensusV1:
    """One official ICPC org surface and its problem-package supply.

    ``n_problem_packages`` is the raw count of problem packages on the surface
    (RMRC repos: number of ``problem.yaml``; ECNA archive: number of per-year
    ``*.zip`` packages).  ``package_bearing`` is False for repos that ship no
    Kattis problem packages (websites / tooling / empty stubs / 0-package repos).
    """

    key: str                 # stable id, e.g. "RMRC:2024-2025"
    surface_family: str      # "RMRC" | "ECNA" | "MIDATL"
    source: str              # repo or archive folder (``repo@year`` for ECNA)
    contest_date: str        # YYYY-MM-DD (season anchor; ECNA regionals are Nov)
    n_problem_packages: int
    package_bearing: bool
    used_by: str             # "W120" | "W121" | "" (not yet mined)
    note: str = ""

    def is_post_cutoff(self, boundary: str = MAVERICK_CUTOFF_BOUNDARY) -> bool:
        return self.package_bearing and self.contest_date > boundary

    def is_pre_cutoff(self, boundary: str = MAVERICK_CUTOFF_BOUNDARY) -> bool:
        return self.package_bearing and self.contest_date <= boundary


# The pinned official-org census (problem-package counts verified LIVE via the
# GitHub API 2026-05-31; RMRC = problem.yaml count, ECNA = per-year *.zip count).
OFFICIAL_ICPC_SURFACE_CENSUS_V1: tuple[SurfaceCensusV1, ...] = (
    # ---- post-cutoff (resistant); ALL FOUR already mined by W120 (51 raw) ----
    SurfaceCensusV1("RMRC:2024-2025", "RMRC", "icpc/na-rocky-mountain-2024-2025-public",
                    "2024-12-03", 13, True, "W120"),
    SurfaceCensusV1("RMRC:2025-2026", "RMRC", "icpc/na-rocky-mountain-2025-2026-public",
                    "2025-11-13", 13, True, "W120"),
    SurfaceCensusV1("ECNA:2024-2025", "ECNA", "icpc/na-ecna-archive@2024-2025",
                    "2024-11-11", 13, True, "W120"),
    SurfaceCensusV1("ECNA:2025-2026", "ECNA", "icpc/na-ecna-archive@2025-2026",
                    "2025-11-10", 12, True, "W120"),
    # ---- pre-cutoff (exposed) RMRC; W121 mined 2021 + 2022-2023 ----
    SurfaceCensusV1("RMRC:2017", "RMRC", "icpc/rmc17-public",
                    "2017-11-04", 11, True, "", "exposed headroom"),
    SurfaceCensusV1("RMRC:2018", "RMRC", "icpc/na-rocky-mountain-2018-public",
                    "2018-11-03", 10, True, "", "exposed headroom"),
    SurfaceCensusV1("RMRC:2019", "RMRC", "icpc/na-rocky-mountain-2019-public",
                    "2019-11-02", 11, True, "", "exposed headroom"),
    SurfaceCensusV1("RMRC:2020", "RMRC", "icpc/na-rocky-mountain-2020-public",
                    "2021-02-27", 11, True, "", "exposed headroom (2020-21 season)"),
    SurfaceCensusV1("RMRC:2021", "RMRC", "icpc/na-rocky-mountain-2021-public",
                    "2022-03-14", 14, True, "W121"),
    SurfaceCensusV1("RMRC:2022-2023", "RMRC", "icpc/na-rocky-mountain-2022-2023-public",
                    "2023-02-25", 12, True, "W121"),
    # ---- pre-cutoff (exposed) ECNA; W121 mined 2022-2023 + 2023-2024 ----
    SurfaceCensusV1("ECNA:2019-2020", "ECNA", "icpc/na-ecna-archive@2019-2020",
                    "2019-11-09", 17, True, "", "exposed headroom"),
    SurfaceCensusV1("ECNA:2020-2021", "ECNA", "icpc/na-ecna-archive@2020-2021",
                    "2020-11-14", 12, True, "", "exposed headroom"),
    SurfaceCensusV1("ECNA:2021-2022", "ECNA", "icpc/na-ecna-archive@2021-2022",
                    "2021-11-13", 13, True, "", "exposed headroom"),
    SurfaceCensusV1("ECNA:2022-2023", "ECNA", "icpc/na-ecna-archive@2022-2023",
                    "2022-11-12", 12, True, "W121"),
    SurfaceCensusV1("ECNA:2023-2024", "ECNA", "icpc/na-ecna-archive@2023-2024",
                    "2023-11-11", 12, True, "W121"),
    # ---- org repos that ship NO problem packages (can never be graded) ----
    SurfaceCensusV1("RMRC:2023-2024", "RMRC", "icpc/na-rocky-mountain-2023-2024-public",
                    "2024-01-16", 0, False, "", "0 problem.yaml (no packages shipped)"),
    SurfaceCensusV1("MIDATL:stub", "MIDATL", "icpc/na-mid-atlantic-public",
                    "2018-08-16", 0, False, "", "README-only stub; no packages"),
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def _cid(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()[:12]


def _side_summary(
    surfaces: Sequence[SurfaceCensusV1],
    yield_ratio: float,
    target: int,
) -> dict:
    raw = sum(s.n_problem_packages for s in surfaces)
    est_tier1 = int(round(raw * yield_ratio))
    return {
        "n_surfaces": len(surfaces),
        "surface_keys": [s.key for s in surfaces],
        "raw_problem_packages": raw,
        "est_tier1": est_tier1,
        "raw_upper_bound": raw,
        "reaches_target_estimated": est_tier1 >= target,
        "reaches_target_even_at_upper_bound": raw >= target,
        "deficit_vs_target_estimated": max(0, target - est_tier1),
    }


def assess_largen_supply_v1(
    census: Sequence[SurfaceCensusV1] = OFFICIAL_ICPC_SURFACE_CENSUS_V1,
    *,
    boundary: str = MAVERICK_CUTOFF_BOUNDARY,
    target_per_field: int = TARGET_PER_FIELD,
) -> dict:
    """Census the official org and decide whether a >=100/field matched
    battlefield is constructible.  Returns the machine-checkable verdict the
    W123 runbook consumes."""
    resistant = [s for s in census if s.is_post_cutoff(boundary)]
    exposed = [s for s in census if s.is_pre_cutoff(boundary)]
    non_package = [s for s in census if not s.package_bearing]

    res = _side_summary(resistant, RESISTANT_TIER1_YIELD, target_per_field)
    exp = _side_summary(exposed, EXPOSED_TIER1_YIELD, target_per_field)

    both_reach = bool(res["reaches_target_estimated"] and exp["reaches_target_estimated"])
    both_reach_ub = bool(res["reaches_target_even_at_upper_bound"]
                         and exp["reaches_target_even_at_upper_bound"])

    # The resistant side is structurally hard-capped: only post-cutoff seasons
    # that have actually been published as graded packages can ever count, and
    # its RAW total is below the target even at a 100% tier-1 yield.
    if not res["reaches_target_even_at_upper_bound"]:
        blocker = (
            f"RESISTANT supply hard-capped at {res['raw_upper_bound']} raw "
            f"(~{res['est_tier1']} tier-1) across {res['n_surfaces']} post-cutoff "
            f"official package surfaces, ALL already mined by W120; below {target_per_field} "
            f"even at a 100% tier-1 yield; no fifth post-cutoff github.com/icpc package "
            f"surface exists (deficit >= {target_per_field - res['raw_upper_bound']} raw). "
            f"EXPOSED scales fine ({exp['raw_upper_bound']} raw / ~{exp['est_tier1']} tier-1 "
            f">= {target_per_field}) — the cap is SOLELY post-cutoff. Only a FUTURE official "
            f"regional drop (RMRC/ECNA 2026-2027+) or a newly published official region lifts it."
        )
        verdict = "LARGEN_MATCHED_BATTLEFIELD_UNREACHABLE_OFFICIAL_FAMILY"
    elif not exp["reaches_target_even_at_upper_bound"]:
        blocker = (
            f"EXPOSED supply caps at {exp['raw_upper_bound']} raw "
            f"(deficit >= {target_per_field - exp['raw_upper_bound']})."
        )
        verdict = "LARGEN_MATCHED_BATTLEFIELD_UNREACHABLE_OFFICIAL_FAMILY"
    elif both_reach:
        blocker = "NONE"
        verdict = "LARGEN_MATCHED_BATTLEFIELD_CONSTRUCTIBLE"
    else:
        blocker = "borderline: upper bound reaches target but estimated tier-1 does not"
        verdict = "LARGEN_MATCHED_BATTLEFIELD_BORDERLINE"

    largen_constructible = verdict == "LARGEN_MATCHED_BATTLEFIELD_CONSTRUCTIBLE"

    payload = {
        "schema": SCHEMA_VERSION,
        "milestone": "W123",
        "lane": "alpha",
        "boundary": boundary,
        "target_per_field": target_per_field,
        "resistant": res,
        "exposed": exp,
        "n_non_package_org_repos": len(non_package),
        "non_package_org_repos": [s.source for s in non_package],
        "both_fields_reach_target_estimated": both_reach,
        "both_fields_reach_target_upper_bound": both_reach_ub,
        "largen_matched_battlefield_constructible": largen_constructible,
        # Lane-beta spend gate: large-n NIM is earned ONLY if BOTH sides can be
        # built to >=100 from the official family.  Resistant cannot, so shut.
        "largen_spend_gate_open": largen_constructible,
        "binding_blocker": blocker,
        "verdict": verdict,
    }
    payload["census_cid"] = _cid({
        "surfaces": [dataclasses.asdict(s) for s in census],
        "boundary": boundary, "target": target_per_field,
    })
    return payload
