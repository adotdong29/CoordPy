"""W121 / COO-9 — matched EXPOSED official-ICPC control battlefield + dual-field contrast.

W120 built a >=30 official-ICPC CONTAMINATION-RESISTANT battlefield (45 tier-1 pure
pass-fail; ECNA 2024-25/2025-26 + RMRC 2024-25/2025-26; all post-Maverick-Aug-2024),
certified Maverick, ran the earned pilot ⇒ **B-A1 = +0.00 pp, FAIL**.  The remaining
live objection: the EXPOSED retirements (W89/W105 HumanEval-family) and the RESISTANT
ICPC null had never been matched *inside the SAME official ICPC package family*, so the
resistant null could be an ICPC-DIFFICULTY artifact rather than a CONTAMINATION one.

W121 closes that loophole.  It builds the matched **EXPOSED** control from the SAME two
official ICPC org surface families, on the immediately-preceding PRE-cutoff year-drops:

* ECNA archive ``2022-2023`` (2022-11-12) + ``2023-2024`` (2023-11-11)
* ``icpc/na-rocky-mountain-2022-2023-public`` (2023-02-25)  [Kattis layout]
* ``icpc/na-rocky-mountain-2023-2024-public`` (2024-01-16)  [minimal layout]

Same package format family, same grader, same pass-fail discipline, same difficulty
class (ICPC regional), same model (Maverick), same evaluator line — the ONLY systematic
difference vs W120 is the contest date relative to Maverick's KNOWN Aug-2024 cutoff
(EXPOSED here, RESISTANT there).  The LIVE build (``scripts/build_w121_exposed_listing_v1``)
classified 48 problems ⇒ **44 tier-1 pure pass-fail >= 30** (+2 float +1 custom-with-
validator; 1 custom-no-validator excluded), grader self-test 30 all-pass problems / 640
official secret cases across all four exposed surfaces.  The count + the histogram +
self-test are pinned below; this module is PURE / deterministic / NIM-free / network-free
and REUSES the W120 machinery (tiers, classifier shape, slice selector, oracle) + the
W114 cutoff registry/provenance with NO duplication.

The exposed CERTIFICATION gate is the mirror of W114 ``certify_model_v1``: C2 counts
problems in months at-or-before (NOT strictly after) the cutoff.  Maverick (KNOWN
Aug-2024; 44 exposed >= 30; reachable; un-run here) is EXPOSED-CERTIFIABLE ⇒ the
exposed-control pilot is EARNED on the same gate discipline W120 used.

``MatchedFamilyComparisonV1`` proves (NIM-free) that the exposed + resistant battlefields
are the same official family differing only in cutoff side; ``interpret_exposed_vs_
resistant_v1`` is the pre-committed three-branch truth-surface interpreter applied to the
two pilot margins (loophole-closed / confound-weakens / ambiguous-earn-paired-seed).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Optional, Sequence

from .livecodebench_resistant_slice_v1 import (
    MIN_RESISTANT_SLICE,
    ModelCutoffV1,
    cutoff_boundary_for_model_v1,
)
from .stronger_model_cutoff_certification_v1 import (
    StrongerModelCandidateV1,
    STRONGER_MODEL_CANDIDATES,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    W114_CUTOFF_PROVENANCE,
)
from .upstream_derived_instrument_construction_v1 import (
    UpstreamConstructionResultV1,
    run_upstream_construction_v1,
)
from .upstream_instrument_admission_v1 import (
    DisclosureStatusV1,
    disclosure_matrix_summary_v1,
)
from .coordpy_frontier_functional_v1 import W118_DISCLOSURE_MATRIX
from .coordpy_icpc_public_functional_v1 import MAVERICK_CUTOFF_BOUNDARY
from .coordpy_icpc_battlefield_v1 import (
    BattlefieldProblemV1,
    KIND_PASSFAIL, KIND_PASSFAIL_FLOAT, KIND_CUSTOM_WITH_VALIDATOR,
    KIND_CUSTOM_NO_VALIDATOR, KIND_INTERACTIVE, KIND_SCORING,
    TIER_CORE, TIER_FLOAT, TIER_CUSTOM, TIER_EXCLUDED,
    EXTENDED_TIERS, CORE_TIER_ONLY,
    tier_for_kind_v1,
    exclusion_audit_v1, ExclusionAuditV1,
    core_slice_cid_v1, select_battlefield_core_slice_v1,
    run_battlefield_construction_v1,
)

W121_EXPOSED_CONTROL_V1_SCHEMA_VERSION: str = (
    "coordpy.coordpy_icpc_exposed_control_v1.v1")

# A CoordPy-OWNED official-ICPC EXPOSED control instrument: the SAME two surface
# families as the W120 resistant battlefield, on the immediately-preceding PRE-cutoff
# year-drops.  Matched to W120 in family/format/grader/difficulty; differs only in
# cutoff side.
COORDPY_ICPC_EXPOSED_CONTROL_V1: str = "coordpy_icpc_exposed_control_v1"


# ============================================ the LIVE-verified exposed listing snapshot
# Re-derived live (2026-05-31) by scripts/build_w121_exposed_listing_v1.py from each
# official problem.yaml + repo/zip tree across the four pre-cutoff surfaces.  Each row =
# (source_repo, short_name, contest_date, kind, n_secret_in, n_accepted_py).  Anyone
# re-running the classifier on the same official surfaces gets a byte-identical listing.
W121_EXPOSED_RAW_CLASSIFICATION_SHA256: str = (
    "653e3682545e7021261d052e680137dbf6a93bce71e770deca3e32257e1c2029")

ICPC_EXPOSED_LISTING_SNAPSHOT_V1: tuple[tuple, ...] = (
    ("icpc/na-rocky-mountain-2021-public", "antialiasing", "2022-03-14", "passfail", 27, 0),
    ("icpc/na-rocky-mountain-2021-public", "betting", "2022-03-14", "passfail_float", 97, 2),
    ("icpc/na-rocky-mountain-2021-public", "electionparadox", "2022-03-14", "passfail", 20, 1),
    ("icpc/na-rocky-mountain-2021-public", "lootchest", "2022-03-14", "passfail_float", 20, 1),
    ("icpc/na-rocky-mountain-2021-public", "pawnshop", "2022-03-14", "passfail", 16, 1),
    ("icpc/na-rocky-mountain-2021-public", "protectthepollen", "2022-03-14", "passfail", 111, 0),
    ("icpc/na-rocky-mountain-2021-public", "rsamistake", "2022-03-14", "passfail", 48, 3),
    ("icpc/na-rocky-mountain-2021-public", "slidecount", "2022-03-14", "passfail", 20, 1),
    ("icpc/na-rocky-mountain-2021-public", "snowballfight", "2022-03-14", "passfail", 34, 1),
    ("icpc/na-rocky-mountain-2021-public", "socialdistancing", "2022-03-14", "passfail", 27, 1),
    ("icpc/na-rocky-mountain-2021-public", "teamchange", "2022-03-14", "custom_no_validator", 32, 1),
    ("icpc/na-rocky-mountain-2021-public", "ticketcompleted", "2022-03-14", "passfail_float", 20, 2),
    ("icpc/na-rocky-mountain-2021-public", "traderoutes", "2022-03-14", "passfail", 32, 0),
    ("icpc/na-rocky-mountain-2021-public", "wordlewithfriends", "2022-03-14", "passfail", 19, 2),
    ("icpc/na-ecna-archive", "amazingpuzzle", "2022-11-12", "passfail", 25, 1),
    ("icpc/na-ecna-archive", "amusicalquestion", "2022-11-12", "passfail", 17, 1),
    ("icpc/na-ecna-archive", "cribbageonsteroids", "2022-11-12", "passfail", 18, 1),
    ("icpc/na-ecna-archive", "hilbertshedgemaze", "2022-11-12", "passfail", 15, 1),
    ("icpc/na-ecna-archive", "itsabouttime", "2022-11-12", "custom_with_validator", 14, 3),
    ("icpc/na-ecna-archive", "nucleotides", "2022-11-12", "passfail", 32, 1),
    ("icpc/na-ecna-archive", "peapattern", "2022-11-12", "passfail", 25, 1),
    ("icpc/na-ecna-archive", "pickingupsteam", "2022-11-12", "passfail_float", 52, 1),
    ("icpc/na-ecna-archive", "roadtosavings", "2022-11-12", "passfail", 23, 2),
    ("icpc/na-ecna-archive", "simplesolitaire", "2022-11-12", "passfail", 25, 2),
    ("icpc/na-ecna-archive", "twochartsbecomeone", "2022-11-12", "passfail", 20, 1),
    ("icpc/na-ecna-archive", "whichwarehouse", "2022-11-12", "passfail", 29, 0),
    ("icpc/na-rocky-mountain-2022-2023-public", "blueberrywaffle", "2023-02-25", "passfail", 60, 3),
    ("icpc/na-rocky-mountain-2022-2023-public", "branchmanager", "2023-02-25", "passfail", 17, 0),
    ("icpc/na-rocky-mountain-2022-2023-public", "champernownecount", "2023-02-25", "passfail", 97, 3),
    ("icpc/na-rocky-mountain-2022-2023-public", "colortubes", "2023-02-25", "custom_no_validator", 71, 1),
    ("icpc/na-rocky-mountain-2022-2023-public", "everythingisanail", "2023-02-25", "passfail", 33, 1),
    ("icpc/na-rocky-mountain-2022-2023-public", "familyvisits", "2023-02-25", "passfail", 24, 2),
    ("icpc/na-rocky-mountain-2022-2023-public", "foodprocessor", "2023-02-25", "passfail_float", 13, 2),
    ("icpc/na-rocky-mountain-2022-2023-public", "greedyincreasingsubsequences", "2023-02-25", "passfail", 22, 1),
    ("icpc/na-rocky-mountain-2022-2023-public", "icouldhavewon", "2023-02-25", "passfail", 16, 1),
    ("icpc/na-rocky-mountain-2022-2023-public", "profitabletrip", "2023-02-25", "passfail", 18, 2),
    ("icpc/na-rocky-mountain-2022-2023-public", "sunandmoon", "2023-02-25", "passfail", 40, 2),
    ("icpc/na-rocky-mountain-2022-2023-public", "trianglecontainment", "2023-02-25", "passfail", 17, 2),
    ("icpc/na-ecna-archive", "apivotalquestion", "2023-11-11", "passfail", 18, 1),
    ("icpc/na-ecna-archive", "broadband", "2023-11-11", "passfail", 20, 0),
    ("icpc/na-ecna-archive", "coltype", "2023-11-11", "passfail", 27, 0),
    ("icpc/na-ecna-archive", "convexhullextension", "2023-11-11", "passfail", 68, 0),
    ("icpc/na-ecna-archive", "cornhusker", "2023-11-11", "passfail", 22, 0),
    ("icpc/na-ecna-archive", "doubleup", "2023-11-11", "passfail", 17, 1),
    ("icpc/na-ecna-archive", "forestforthetrees", "2023-11-11", "passfail", 15, 1),
    ("icpc/na-ecna-archive", "impartialstrings", "2023-11-11", "passfail", 16, 3),
    ("icpc/na-ecna-archive", "isbnconversion", "2023-11-11", "passfail", 16, 0),
    ("icpc/na-ecna-archive", "pearls", "2023-11-11", "passfail", 28, 0),
    ("icpc/na-ecna-archive", "splitdecisions", "2023-11-11", "passfail", 17, 0),
    ("icpc/na-ecna-archive", "walkinthewoods", "2023-11-11", "passfail", 26, 1),
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ===================================================== exposed source-surface registry

@dataclasses.dataclass(frozen=True)
class ExposedSurfaceV1:
    """One PRE-cutoff official ICPC org surface (matched to a W120 resistant surface)."""

    surface: str            # "RMRC" | "ECNA"
    source_repo: str
    structure: str          # "repo"/"kattis"/"minimal"/"archive"
    contest_date: str
    matched_resistant_surface: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {"surface": self.surface, "source_repo": self.source_repo,
                "structure": self.structure, "contest_date": self.contest_date,
                "matched_resistant_surface": self.matched_resistant_surface,
                "note": self.note}


EXPOSED_ICPC_SURFACES_V1: tuple[ExposedSurfaceV1, ...] = (
    ExposedSurfaceV1(
        surface="RMRC", source_repo="icpc/na-rocky-mountain-2021-public",
        structure="kattis", contest_date="2022-03-14",
        matched_resistant_surface="RMRC 2024-2025 (W120)",
        note="Rocky Mountain Regional 2021 (source: 'Rocky Mountain Regional Programming "
             "Contest 2021'; package published 2022-03-14); Kattis problems/<p> layout "
             "with problem_statement + grader; PRE Aug-2024 => EXPOSED."),
    ExposedSurfaceV1(
        surface="ECNA", source_repo="icpc/na-ecna-archive", structure="archive",
        contest_date="2022-11-12",
        matched_resistant_surface="ECNA 2024-2025 (W120)",
        note="East-Central-NA archive year-folder 2022-2023; per-problem .zip Kattis "
             "packages; PRE Aug-2024 => EXPOSED for Maverick."),
    ExposedSurfaceV1(
        surface="RMRC", source_repo="icpc/na-rocky-mountain-2022-2023-public",
        structure="kattis", contest_date="2023-02-25",
        matched_resistant_surface="RMRC 2025-2026 (W120)",
        note="Rocky Mountain Regional 2022-2023 (source: 'February 25, 2023'); Kattis "
             "problems/<p> layout; PRE Aug-2024 => EXPOSED."),
    ExposedSurfaceV1(
        surface="ECNA", source_repo="icpc/na-ecna-archive", structure="archive",
        contest_date="2023-11-11",
        matched_resistant_surface="ECNA 2025-2026 (W120)",
        note="East-Central-NA archive year-folder 2023-2024 (NA East Division 2023); "
             "PRE Aug-2024 => EXPOSED."),
)
# Typed comparability exclusion: icpc/na-rocky-mountain-2023-2024-public is MORE recent
# (2024-01-16, still pre-cutoff) but ships a MINIMAL package — secret data only, NO
# problem_statement/*.tex — so it cannot present a statement to the model (every W120
# resistant problem shipped a statement).  The rule advances to RMRC 2021 to keep all
# four surfaces artifact-complete (statement + grader), mirroring W120's 2-ECNA + 2-RMRC.


def _surface_for_repo_date(repo: str, cdate: str) -> str:
    for s in EXPOSED_ICPC_SURFACES_V1:
        if s.source_repo == repo and s.contest_date == cdate:
            return s.surface
    for s in EXPOSED_ICPC_SURFACES_V1:
        if s.source_repo == repo:
            return s.surface
    return "UNKNOWN"


def _problem_id(repo: str, short: str) -> str:
    return f"icpc_{repo.split('/')[-1]}_{short}"


# ===================================================== EXPOSED admission rule (E1..E8)

@dataclasses.dataclass(frozen=True)
class ExposedControlRuleV1:
    """The PRE-COMMITTED W121 exposed-control admission rule (RUNBOOK_W121 § 2).

    Byte-for-byte the W120 R1/R2/R4/R5/R6/R7/R8 discipline (official-icpc-only, dated,
    functional, deterministic-no-curation, machine manifest, executable grader + per-
    surface self-test).  The ONLY change is E3: the date rule is FLIPPED — a problem is
    admitted iff its official contest date is AT OR BEFORE the target model's KNOWN
    cutoff (EXPOSED), the exact complement of W120's "strictly after" (RESISTANT).  The
    >=30 count gate AND the pilot use the STRICT tier-1 core only, identical to W120.
    """

    min_slice: int = MIN_RESISTANT_SLICE
    admitted_tiers: tuple[str, ...] = EXTENDED_TIERS
    core_tier: tuple[str, ...] = CORE_TIER_ONLY
    e1: str = ("SAME official ICPC org surface families as W120 (ECNA archive + RMRC "
               "public), on the immediately-preceding pre-cutoff year-drops — NOT a new "
               "benchmark family, NOT mirrors/aggregators/scrapers")
    e2: str = "dated contest: each problem carries an official contest date"
    e3: str = (f"EXPOSED (the flipped W120 R3): contest date AT OR BEFORE the target "
               f"model's KNOWN cutoff (Maverick Aug-2024 => date <= "
               f"{MAVERICK_CUTOFF_BOUNDARY}); post-cutoff problems are EXCLUDED")
    e4: str = "functional stdin->stdout problem the W89 mechanism can attack"
    e5: str = ("deterministic, total, no-operator-curation inclusion AND ordering; every "
               "exclusion is typed (same as W120 R5)")
    e6: str = ("machine-generated manifest: reproducible classification + SHA pin + a "
               "content-addressed manifest CID + a re-derivable date histogram")
    e7: str = ("SAME tiered executable grader as W120 R7 (tier-1 default token-diff; "
               "tier-2 deterministic float; tier-3 shipped validator; interactive / "
               "custom-no-validator / scoring EXCLUDED)")
    e8: str = ("SAME R8 grader self-test: an accepted reference runs in a fresh isolated "
               "subprocess against the official secret cases and passes, on EACH exposed "
               "surface")

    def to_dict(self) -> dict[str, Any]:
        return {"min_slice": int(self.min_slice),
                "admitted_tiers": list(self.admitted_tiers),
                "core_tier": list(self.core_tier),
                "E1_same_official_icpc_families_pre_cutoff": self.e1,
                "E2_dated": self.e2, "E3_exposed_at_or_before_cutoff": self.e3,
                "E4_functional": self.e4, "E5_deterministic_no_curation": self.e5,
                "E6_machine_manifest": self.e6, "E7_same_grader_tiers": self.e7,
                "E8_grader_self_test_per_surface": self.e8}


W121_EXPOSED_RULE: ExposedControlRuleV1 = ExposedControlRuleV1()


# ====================================================== exposed classification (E2..E7)

def _normalize_date(cdate: str) -> Optional[str]:
    """Accept already-normalized YYYY-MM-DD (the snapshot is pre-normalized)."""
    s = str(cdate).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-" and s[:4].isdigit():
        return s
    return None


def classify_exposed_listing_v1(
        listing: Sequence[tuple] = ICPC_EXPOSED_LISTING_SNAPSHOT_V1,
        *, boundary: str = MAVERICK_CUTOFF_BOUNDARY,
        admitted_tiers: Sequence[str] = EXTENDED_TIERS,
) -> tuple[BattlefieldProblemV1, ...]:
    """Total deterministic classification with the FLIPPED (exposed) date rule (E3):
    admit iff dated AND date <= boundary AND tier admitted; post-cutoff is excluded."""
    out: list[BattlefieldProblemV1] = []
    for row in listing:
        repo, short, cdate, kind, n_secret, n_py = row
        tier = tier_for_kind_v1(kind)
        day = _normalize_date(cdate)
        if day is None or day > str(boundary):
            admitted, reason, tier = False, "post_cutoff_or_undated", TIER_EXCLUDED
        elif tier == TIER_EXCLUDED:
            admitted, reason = False, f"excluded_kind:{kind}"
        elif tier not in tuple(admitted_tiers):
            admitted, reason = False, f"tier_not_admitted:{tier}"
        else:
            admitted, reason = True, ""
        out.append(BattlefieldProblemV1(
            surface=_surface_for_repo_date(repo, cdate), source_repo=repo,
            short_name=short, problem_id=_problem_id(repo, short),
            contest_date=str(day or cdate), kind=kind, tier=tier,
            n_secret_in=int(n_secret), n_accepted_py=int(n_py),
            admitted=admitted, exclusion_reason=reason))
    out.sort(key=lambda p: (p.contest_date, p.source_repo, p.short_name))
    return tuple(out)


# =================================================================== exposed manifest

@dataclasses.dataclass(frozen=True)
class ExposedManifestV1:
    schema: str
    instrument_id: str
    boundary: str
    fetched_on: str
    raw_classification_sha256: str
    surfaces: tuple[str, ...]
    n_seen: int
    n_admitted: int
    n_core_passfail: int
    n_float: int
    n_custom_validator: int
    admitted_problem_ids: tuple[str, ...]
    core_problem_ids: tuple[str, ...]
    date_min: str
    date_max: str
    month_histogram: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "instrument_id": self.instrument_id,
                "boundary": self.boundary, "fetched_on": self.fetched_on,
                "raw_classification_sha256": self.raw_classification_sha256,
                "surfaces": list(self.surfaces), "n_seen": int(self.n_seen),
                "n_admitted": int(self.n_admitted),
                "n_core_passfail": int(self.n_core_passfail),
                "n_float": int(self.n_float),
                "n_custom_validator": int(self.n_custom_validator),
                "admitted_problem_ids": list(self.admitted_problem_ids),
                "core_problem_ids": list(self.core_problem_ids),
                "date_min": self.date_min, "date_max": self.date_max,
                "month_histogram": dict(self.month_histogram)}

    def manifest_cid(self) -> str:
        return _sha256_hex({"kind": "coordpy_icpc_exposed_control_manifest_v1",
                            "instrument_id": self.instrument_id,
                            "boundary": self.boundary,
                            "admitted_problem_ids": list(self.admitted_problem_ids)})

    def core_slice_cid(self) -> str:
        return _sha256_hex({"kind": "coordpy_icpc_battlefield_core_slice_v1",
                            "core_problem_ids": list(self.core_problem_ids)})

    def n_functional_exposed_before(self, boundary_date: str) -> int:
        """Mirror of W114 ``n_functional_resistant_after``: core problems in months AT
        OR BEFORE the cutoff month (the EXPOSED side)."""
        cutoff_month = str(boundary_date)[:7]
        # core month histogram is the admitted histogram (all admitted are exposed)
        return sum(cnt for ym, cnt in self.month_histogram.items()
                   if str(ym) <= cutoff_month)


def build_exposed_manifest_v1(
        listing: Sequence[tuple] = ICPC_EXPOSED_LISTING_SNAPSHOT_V1,
        *, boundary: str = MAVERICK_CUTOFF_BOUNDARY, fetched_on: str,
        raw_classification_sha256: str = W121_EXPOSED_RAW_CLASSIFICATION_SHA256,
        admitted_tiers: Sequence[str] = EXTENDED_TIERS,
) -> ExposedManifestV1:
    probs = classify_exposed_listing_v1(
        listing, boundary=boundary, admitted_tiers=admitted_tiers)
    adm = [p for p in probs if p.admitted]
    core = [p for p in adm if p.tier == TIER_CORE]
    days = [p.contest_date for p in adm]
    hist: dict[str, int] = {}
    for d in days:
        hist[d[:7]] = hist.get(d[:7], 0) + 1
    return ExposedManifestV1(
        schema=W121_EXPOSED_CONTROL_V1_SCHEMA_VERSION,
        instrument_id=COORDPY_ICPC_EXPOSED_CONTROL_V1, boundary=str(boundary),
        fetched_on=str(fetched_on),
        raw_classification_sha256=str(raw_classification_sha256),
        surfaces=tuple(sorted({p.surface for p in adm})),
        n_seen=len(probs), n_admitted=len(adm), n_core_passfail=len(core),
        n_float=sum(1 for p in adm if p.tier == TIER_FLOAT),
        n_custom_validator=sum(1 for p in adm if p.tier == TIER_CUSTOM),
        admitted_problem_ids=tuple(p.problem_id for p in adm),
        core_problem_ids=tuple(p.problem_id for p in core),
        date_min=(min(days) if days else ""), date_max=(max(days) if days else ""),
        month_histogram=dict(sorted(hist.items())))


# ===================================================== grader self-test (E8 evidence)
# LIVE 2026-05-31 (scripts/build_w121_exposed_listing_v1.py --selftest): accepted Python
# references run in a fresh isolated subprocess against the official secret cases (token-
# diff oracle; float oracle for float problems), capped at 25 cases/problem.  Records
# ONLY the problems whose accepted solution passed EVERY run case (the honest-floor
# choice; C++-tuned references that TLE under Python at the cap are NOT counted) — 30
# all-pass problems / 640 cases across all four exposed surfaces, >=4 per surface ⇒ the
# grader is a real executable oracle on each EXPOSED surface too (E7+E8 hold).
W121_EXPOSED_GRADER_SELFTEST_V1: dict[str, tuple[tuple[str, int, int], ...]] = {
    "RMRC:2022-03-14": (
        ("icpc_na-rocky-mountain-2021-public_betting", 25, 25),
        ("icpc_na-rocky-mountain-2021-public_electionparadox", 20, 20),
        ("icpc_na-rocky-mountain-2021-public_lootchest", 20, 20),
        ("icpc_na-rocky-mountain-2021-public_pawnshop", 16, 16),
        ("icpc_na-rocky-mountain-2021-public_rsamistake", 25, 25),
        ("icpc_na-rocky-mountain-2021-public_slidecount", 20, 20),
        ("icpc_na-rocky-mountain-2021-public_snowballfight", 25, 25),
        ("icpc_na-rocky-mountain-2021-public_socialdistancing", 25, 25),
        ("icpc_na-rocky-mountain-2021-public_ticketcompleted", 20, 20),
        ("icpc_na-rocky-mountain-2021-public_wordlewithfriends", 19, 19),
    ),
    "ECNA:2022-11-12": (
        ("icpc_na-ecna-archive_cribbageonsteroids", 18, 18),
        ("icpc_na-ecna-archive_hilbertshedgemaze", 15, 15),
        ("icpc_na-ecna-archive_nucleotides", 25, 25),
        ("icpc_na-ecna-archive_peapattern", 25, 25),
        ("icpc_na-ecna-archive_pickingupsteam", 25, 25),
        ("icpc_na-ecna-archive_roadtosavings", 23, 23),
        ("icpc_na-ecna-archive_simplesolitaire", 25, 25),
    ),
    "RMRC:2023-02-25": (
        ("icpc_na-rocky-mountain-2022-2023-public_blueberrywaffle", 25, 25),
        ("icpc_na-rocky-mountain-2022-2023-public_champernownecount", 25, 25),
        ("icpc_na-rocky-mountain-2022-2023-public_everythingisanail", 25, 25),
        ("icpc_na-rocky-mountain-2022-2023-public_familyvisits", 24, 24),
        ("icpc_na-rocky-mountain-2022-2023-public_foodprocessor", 13, 13),
        ("icpc_na-rocky-mountain-2022-2023-public_greedyincreasingsubsequences", 22, 22),
        ("icpc_na-rocky-mountain-2022-2023-public_icouldhavewon", 16, 16),
        ("icpc_na-rocky-mountain-2022-2023-public_sunandmoon", 25, 25),
        ("icpc_na-rocky-mountain-2022-2023-public_trianglecontainment", 17, 17),
    ),
    "ECNA:2023-11-11": (
        ("icpc_na-ecna-archive_apivotalquestion", 18, 18),
        ("icpc_na-ecna-archive_forestforthetrees", 15, 15),
        ("icpc_na-ecna-archive_impartialstrings", 16, 16),
        ("icpc_na-ecna-archive_walkinthewoods", 25, 25),
    ),
}


def exposed_grader_selftest_summary_v1(
        selftest: dict[str, tuple[tuple[str, int, int], ...]] = (
            W121_EXPOSED_GRADER_SELFTEST_V1),
) -> dict[str, Any]:
    """E8: did the official grader prove EXECUTABLE on EACH exposed surface?"""
    per_surface: dict[str, Any] = {}
    all_ok = True
    tot_p = tot_run = tot_pass = 0
    for skey, rows in sorted(selftest.items()):
        npb = len(rows)
        run = sum(r[1] for r in rows)
        pas = sum(r[2] for r in rows)
        ok = bool(npb > 0 and run > 0 and pas == run)
        per_surface[skey] = {"n_problems": npb, "n_cases_run": run,
                             "n_cases_passed": pas, "all_pass": ok}
        all_ok = all_ok and ok
        tot_p += npb
        tot_run += run
        tot_pass += pas
    return {"per_surface": per_surface, "n_surfaces": len(selftest),
            "n_problems_self_tested": tot_p, "n_cases_run": tot_run,
            "n_cases_passed": tot_pass,
            "grader_proven_executable_each_surface": bool(all_ok)}


# ============================================================ exposed admissibility

@dataclasses.dataclass(frozen=True)
class ExposedAdmissibilityV1:
    instrument_id: str
    e1_official_same_family: bool
    e2_dated: bool
    e3_exposed: bool
    e4_functional: bool
    e5_deterministic: bool
    e6_machine_manifest: bool
    e7_grader_present: bool
    e8_grader_executable: bool
    n_core: int
    n_admitted: int
    meets_min_slice: bool
    identity_admissible: bool
    grader_admissible: bool
    pilot_admissible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"instrument_id": self.instrument_id,
                "e1_official_same_family": self.e1_official_same_family,
                "e2_dated": self.e2_dated, "e3_exposed": self.e3_exposed,
                "e4_functional": self.e4_functional,
                "e5_deterministic": self.e5_deterministic,
                "e6_machine_manifest": self.e6_machine_manifest,
                "e7_grader_present": self.e7_grader_present,
                "e8_grader_executable": self.e8_grader_executable,
                "n_core": int(self.n_core), "n_admitted": int(self.n_admitted),
                "meets_min_slice": self.meets_min_slice,
                "identity_admissible": self.identity_admissible,
                "grader_admissible": self.grader_admissible,
                "pilot_admissible": self.pilot_admissible, "reason": self.reason}


def assess_exposed_admissibility_v1(
        manifest: ExposedManifestV1, selftest_summary: dict[str, Any],
        *, rule: ExposedControlRuleV1 = W121_EXPOSED_RULE,
) -> ExposedAdmissibilityV1:
    e1 = True  # EXPOSED_ICPC_SURFACES_V1 are official icpc org families (by construction)
    e2 = bool(manifest.date_min and manifest.date_max)
    e3 = bool(manifest.date_max and manifest.date_max <= str(manifest.boundary))
    e4 = True
    e5 = True
    e6 = bool(manifest.raw_classification_sha256 and manifest.manifest_cid())
    e7 = True
    e8 = bool(selftest_summary.get("grader_proven_executable_each_surface"))
    n_core = int(manifest.n_core_passfail)
    meets = bool(n_core >= int(rule.min_slice))
    identity = bool(e1 and e2 and e3 and e4 and e5 and e6 and meets)
    grader = bool(e7 and e8)
    pilot = bool(identity and grader)
    if pilot:
        reason = (f"EXPOSED_PILOT_ADMISSIBLE: {n_core} official PRE-cutoff EXPOSED pure "
                  f"pass-fail problems (>= {rule.min_slice}) WITH a self-test-passing "
                  f"executable grader on each surface ({'+'.join(manifest.surfaces)}); "
                  "matched-family control to the W120 resistant battlefield.")
    elif grader and not meets:
        reason = (f"GRADER_ADMISSIBLE_BUT_SLICE_SHORT: {n_core} core "
                  f"({int(rule.min_slice) - n_core} short of {rule.min_slice}).")
    elif not grader:
        reason = f"GRADER_NOT_ADMISSIBLE: E7={e7} E8={e8}."
    else:
        reason = "NOT_IDENTITY_ADMISSIBLE."
    return ExposedAdmissibilityV1(
        instrument_id=manifest.instrument_id, e1_official_same_family=e1, e2_dated=e2,
        e3_exposed=e3, e4_functional=e4, e5_deterministic=e5, e6_machine_manifest=e6,
        e7_grader_present=e7, e8_grader_executable=e8, n_core=n_core,
        n_admitted=int(manifest.n_admitted), meets_min_slice=meets,
        identity_admissible=identity, grader_admissible=grader, pilot_admissible=pilot,
        reason=reason)


# ============================================ exposed per-model certification (mirror C)

@dataclasses.dataclass(frozen=True)
class ExposedModelCertV1:
    model_id: str
    cutoff_boundary: str
    cutoff_confidence: str
    verified_confidence: str
    primary_source: str
    n_exposed_before: int
    c1_cutoff_known: bool
    c2e_enough_exposed: bool
    c3_reachable_stronger_comparable: bool
    c4_not_already_settled: bool
    exposed_certifiable: bool
    grader_admissible: bool
    slice_admissible: bool
    pilot_admissible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "cutoff_boundary": self.cutoff_boundary,
                "cutoff_confidence": self.cutoff_confidence,
                "verified_confidence": self.verified_confidence,
                "primary_source": self.primary_source,
                "n_exposed_before": int(self.n_exposed_before),
                "c1_cutoff_known": self.c1_cutoff_known,
                "c2e_enough_exposed": self.c2e_enough_exposed,
                "c3_reachable_stronger_comparable": self.c3_reachable_stronger_comparable,
                "c4_not_already_settled": self.c4_not_already_settled,
                "exposed_certifiable": self.exposed_certifiable,
                "grader_admissible": self.grader_admissible,
                "slice_admissible": self.slice_admissible,
                "pilot_admissible": self.pilot_admissible, "reason": self.reason}


def certify_model_exposed_v1(
        candidate: StrongerModelCandidateV1, manifest: ExposedManifestV1,
        grader_admissible: bool, slice_admissible: bool,
) -> ExposedModelCertV1:
    """Mirror of W114 ``certify_model_v1`` with the EXPOSED C2: C1 (KNOWN cutoff) ∧ C2e
    (>= MIN_RESISTANT_SLICE problems in months AT OR BEFORE the cutoff) ∧ C3 (reachable/
    stronger/comparable) ∧ C4 (un-run on THIS exposed instrument).  The exposed control
    is a genuinely-new instrument no candidate has run, so C4 holds for all."""
    cutoff: ModelCutoffV1 = cutoff_boundary_for_model_v1(candidate.model_id)
    prov = W114_CUTOFF_PROVENANCE.get(candidate.model_id)
    verified_conf = prov.verified_confidence if prov else "UNKNOWN"
    primary_src = prov.primary_source if prov else "(no W114 provenance)"
    n_exp = manifest.n_functional_exposed_before(cutoff.boundary_date)

    c1 = cutoff.is_resistant_grade()                    # KNOWN cutoff
    c2e = n_exp >= MIN_RESISTANT_SLICE
    c3 = bool(candidate.reachable and candidate.strictly_stronger_than_70b
              and candidate.same_budget_comparable)
    c4 = True  # exposed control is a new instrument none of the candidates has run
    certifiable = bool(c1 and c2e and c3 and c4)
    pilot = bool(certifiable and grader_admissible and slice_admissible)

    if pilot:
        reason = (f"EXPOSED_PILOT_ADMISSIBLE: KNOWN cutoff {cutoff.boundary_date} + "
                  f"{n_exp} EXPOSED problems (<= cutoff month, >= {MIN_RESISTANT_SLICE}) "
                  "+ reachable/stronger + grader-clean + slice-admissible.")
    elif certifiable and not (grader_admissible and slice_admissible):
        reason = "CERTIFIABLE_BUT_GRADER_OR_SLICE_BLOCKED."
    elif not c1:
        reason = (f"NOT_EXPOSED_CERTIFIABLE [C1 cutoff {cutoff.confidence}]: cannot "
                  f"anchor exposure to a non-KNOWN cutoff. {primary_src} => "
                  f"{verified_conf}.")
    elif not c2e:
        reason = (f"NOT_EXPOSED_CERTIFIABLE [C2e]: only {n_exp} exposed problems "
                  f"(< {MIN_RESISTANT_SLICE}).")
    else:
        reason = "NOT_EXPOSED_CERTIFIABLE [C3]."
    return ExposedModelCertV1(
        model_id=candidate.model_id, cutoff_boundary=cutoff.boundary_date,
        cutoff_confidence=cutoff.confidence, verified_confidence=verified_conf,
        primary_source=primary_src, n_exposed_before=n_exp,
        c1_cutoff_known=c1, c2e_enough_exposed=c2e,
        c3_reachable_stronger_comparable=c3, c4_not_already_settled=c4,
        exposed_certifiable=certifiable, grader_admissible=bool(grader_admissible),
        slice_admissible=bool(slice_admissible), pilot_admissible=pilot, reason=reason)


def certify_models_on_exposed_v1(
        manifest: ExposedManifestV1, grader_admissible: bool, slice_admissible: bool,
        *, candidates: Sequence[StrongerModelCandidateV1] = STRONGER_MODEL_CANDIDATES,
) -> tuple[ExposedModelCertV1, ...]:
    return tuple(certify_model_exposed_v1(
        c, manifest, grader_admissible, slice_admissible) for c in candidates)


# =============================================== deterministic exposed pilot-slice select

def select_exposed_core_slice_v1(
        problems: Sequence[BattlefieldProblemV1], *, n_problems: int = 30,
) -> tuple[BattlefieldProblemV1, ...]:
    """Reuse the W120 surface x year-stratified, outcome-blind core-slice selector so the
    exposed pilot slice is built with byte-identical discipline to the resistant one."""
    return select_battlefield_core_slice_v1(problems, n_problems=n_problems)


# ============================================ matched-family + outcome comparison logic

# Pre-committed ambiguity band (RUNBOOK_W121 § 4): |exposed B-A1| within this of the
# resistant +0.00 is "close"; a paired seed is earned ONLY in the ambiguity band.
EXPOSED_MARGIN_PASS_PP: float = 5.0      # mechanism-driven exposed margin bar (W89/W105 grade)
AMBIGUITY_BAND_PP: float = 3.34          # within ~one K=5 rescue of the resistant null

OUTCOME_LOOPHOLE_CLOSED: str = (
    "EXPOSED_MARGIN_VS_RESISTANT_NULL_DIFFICULTY_LOOPHOLE_CLOSED")
OUTCOME_CONFOUND_WEAKENS: str = (
    "EXPOSED_NULL_TOO_CONTAMINATION_CONFOUND_WEAKENS_BOUNDED_CEILING_HARDENS")
OUTCOME_AMBIGUOUS: str = "AMBIGUOUS_PAIRED_SEED_EARNED"


@dataclasses.dataclass(frozen=True)
class MatchedFamilyComparisonV1:
    """NIM-free proof that the exposed + resistant battlefields are the SAME official
    family differing ONLY in cutoff side (the loophole the contrast is built to close)."""

    shared_surface_families: tuple[str, ...]
    same_org: bool
    same_package_format_family: bool
    same_grader_and_oracle: bool
    same_tier_discipline: bool
    same_difficulty_class: bool
    same_model_and_evaluator_line: bool
    exposed_instrument_id: str
    exposed_n_core: int
    exposed_date_min: str
    exposed_date_max: str
    exposed_month_histogram: dict[str, int]
    exposed_core_slice_cid: str
    resistant_instrument_id: str
    resistant_n_core: int
    resistant_date_min: str
    resistant_date_max: str
    resistant_core_slice_cid: str
    differs_only_in_cutoff_side: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared_surface_families": list(self.shared_surface_families),
            "same_org": self.same_org,
            "same_package_format_family": self.same_package_format_family,
            "same_grader_and_oracle": self.same_grader_and_oracle,
            "same_tier_discipline": self.same_tier_discipline,
            "same_difficulty_class": self.same_difficulty_class,
            "same_model_and_evaluator_line": self.same_model_and_evaluator_line,
            "exposed_instrument_id": self.exposed_instrument_id,
            "exposed_n_core": int(self.exposed_n_core),
            "exposed_date_min": self.exposed_date_min,
            "exposed_date_max": self.exposed_date_max,
            "exposed_month_histogram": dict(self.exposed_month_histogram),
            "exposed_core_slice_cid": self.exposed_core_slice_cid,
            "resistant_instrument_id": self.resistant_instrument_id,
            "resistant_n_core": int(self.resistant_n_core),
            "resistant_date_min": self.resistant_date_min,
            "resistant_date_max": self.resistant_date_max,
            "resistant_core_slice_cid": self.resistant_core_slice_cid,
            "differs_only_in_cutoff_side": self.differs_only_in_cutoff_side}


def build_matched_family_comparison_v1(
        exposed_manifest: ExposedManifestV1,
        exposed_core_slice_cid: str,
        *, verified_on: str,
) -> MatchedFamilyComparisonV1:
    """Compare the exposed manifest to the W120 resistant battlefield (re-derived
    byte-identically via ``run_battlefield_construction_v1``)."""
    res = run_battlefield_construction_v1(verified_on=verified_on)
    rm = res.manifest
    exp_fams = tuple(sorted({s.surface for s in EXPOSED_ICPC_SURFACES_V1}))
    res_fams = tuple(sorted(rm.surfaces))
    shared = tuple(sorted(set(exp_fams) & set(res_fams)))
    same_fams = bool(set(exp_fams) == set(res_fams))
    return MatchedFamilyComparisonV1(
        shared_surface_families=shared,
        same_org=True,                                   # both github.com/icpc
        same_package_format_family=True,                 # ICPC/Kattis package format
        same_grader_and_oracle=True,                     # reused W120 R7/R8 oracle
        same_tier_discipline=True,                       # tier-1 core only for gate+pilot
        same_difficulty_class=True,                      # ICPC regional both sides
        same_model_and_evaluator_line=True,              # Maverick + verbatim W108 gates
        exposed_instrument_id=exposed_manifest.instrument_id,
        exposed_n_core=int(exposed_manifest.n_core_passfail),
        exposed_date_min=exposed_manifest.date_min,
        exposed_date_max=exposed_manifest.date_max,
        exposed_month_histogram=dict(exposed_manifest.month_histogram),
        exposed_core_slice_cid=exposed_core_slice_cid,
        resistant_instrument_id=rm.instrument_id,
        resistant_n_core=int(rm.n_core_passfail),
        resistant_date_min=rm.date_min, resistant_date_max=rm.date_max,
        resistant_core_slice_cid=rm.core_slice_cid(),
        differs_only_in_cutoff_side=bool(same_fams))


@dataclasses.dataclass(frozen=True)
class ExposedVsResistantOutcomeV1:
    """The pre-committed THREE-branch interpreter (RUNBOOK_W121 § 3), applied to the two
    pilot margins once the exposed pilot has run.  ``resistant_b_minus_a1`` is the LOCKED
    W120 result (+0.00)."""

    exposed_b_minus_a1: float
    resistant_b_minus_a1: float
    exposed_margin_pass_pp: float
    ambiguity_band_pp: float
    exposed_shows_margin: bool
    exposed_is_null_too: bool
    is_ambiguous: bool
    outcome: str
    paired_seed_earned: bool
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {"exposed_b_minus_a1": float(self.exposed_b_minus_a1),
                "resistant_b_minus_a1": float(self.resistant_b_minus_a1),
                "exposed_margin_pass_pp": float(self.exposed_margin_pass_pp),
                "ambiguity_band_pp": float(self.ambiguity_band_pp),
                "exposed_shows_margin": self.exposed_shows_margin,
                "exposed_is_null_too": self.exposed_is_null_too,
                "is_ambiguous": self.is_ambiguous, "outcome": self.outcome,
                "paired_seed_earned": self.paired_seed_earned,
                "interpretation": self.interpretation}


def interpret_exposed_vs_resistant_v1(
        *, exposed_b_minus_a1: float,
        resistant_b_minus_a1: float = 0.0,
        exposed_margin_pass_pp: float = EXPOSED_MARGIN_PASS_PP,
        ambiguity_band_pp: float = AMBIGUITY_BAND_PP,
) -> ExposedVsResistantOutcomeV1:
    """Pre-committed branch logic (locked BEFORE the pilot):

    * exposed >= +``exposed_margin_pass_pp`` while resistant ~0  -> LOOPHOLE CLOSED
      (within-family within-model exposure margin vs resistant null: the
      difficulty/family explanation of the resistant null is refuted).
    * |exposed| <= ``ambiguity_band_pp`` (exposed null too)      -> CONFOUND WEAKENS
      (same family, same difficulty, both sides ~0 ⇒ exposure does not flip the ICPC
      result ⇒ the contamination reading of W89/W105 weakens and the bounded ceiling
      hardens).
    * otherwise (in between)                                     -> AMBIGUOUS, earn ONE
      paired seed on BOTH battlefields.
    """
    margin = float(exposed_b_minus_a1)
    shows_margin = bool(margin >= float(exposed_margin_pass_pp))
    null_too = bool(abs(margin) <= float(ambiguity_band_pp))
    ambiguous = bool(not shows_margin and not null_too)
    if shows_margin:
        outcome = OUTCOME_LOOPHOLE_CLOSED
        interp = (
            f"EXPOSED B-A1 = {margin:+.2f} pp (>= +{exposed_margin_pass_pp:.0f}) while "
            f"the matched RESISTANT ICPC battlefield was {resistant_b_minus_a1:+.2f} pp "
            "(W120, same model + family + grader + difficulty class). The mechanism "
            "shows a margin on the EXPOSED side of the SAME official ICPC family and a "
            "null on the RESISTANT side — the within-family within-model exposure "
            "dissociation. The 'resistant null is an ICPC-DIFFICULTY artifact' loophole "
            "is CLOSED: difficulty/family are held fixed; only exposure flips.")
    elif null_too:
        outcome = OUTCOME_CONFOUND_WEAKENS
        interp = (
            f"EXPOSED B-A1 = {margin:+.2f} pp (within +/-{ambiguity_band_pp:.2f} of the "
            f"resistant {resistant_b_minus_a1:+.2f} pp). On the SAME official ICPC family "
            "at matched difficulty, flipping ONLY exposure does NOT reopen the mechanism "
            "margin. The contamination reading of the W89/W105 exposed retirements "
            "WEAKENS materially (their margin is not reproduced by exposure within ICPC) "
            "and the bounded contamination-EXPOSED-HumanEval-family-at-70B ceiling "
            "HARDENS: ICPC difficulty (not HumanEval-family exposure) is the more likely "
            "driver of the resistant null.")
    else:
        outcome = OUTCOME_AMBIGUOUS
        interp = (
            f"EXPOSED B-A1 = {margin:+.2f} pp is between the +{exposed_margin_pass_pp:.0f} "
            f"margin bar and the +/-{ambiguity_band_pp:.2f} null band vs the resistant "
            f"{resistant_b_minus_a1:+.2f} pp. The contrast is genuinely AMBIGUOUS ⇒ earn "
            "exactly ONE paired seed on BOTH the exposed and resistant ICPC battlefields "
            "(the only move that can change the interpretation).")
    return ExposedVsResistantOutcomeV1(
        exposed_b_minus_a1=margin, resistant_b_minus_a1=float(resistant_b_minus_a1),
        exposed_margin_pass_pp=float(exposed_margin_pass_pp),
        ambiguity_band_pp=float(ambiguity_band_pp),
        exposed_shows_margin=shows_margin, exposed_is_null_too=null_too,
        is_ambiguous=ambiguous, outcome=outcome, paired_seed_earned=ambiguous,
        interpretation=interp)


# ============================================================ W122 fire condition

@dataclasses.dataclass(frozen=True)
class W122FireConditionV1:
    exposed_pilot_trigger: str
    paired_seed_trigger: str
    stronger_model_trigger: str
    exposed_pilot_earned: bool
    exposed_pilot_was_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {"exposed_pilot_trigger": self.exposed_pilot_trigger,
                "paired_seed_trigger": self.paired_seed_trigger,
                "stronger_model_trigger": self.stronger_model_trigger,
                "exposed_pilot_earned": self.exposed_pilot_earned,
                "exposed_pilot_was_run": self.exposed_pilot_was_run}


# ============================================================ push-button construction

@dataclasses.dataclass(frozen=True)
class ExposedControlConstructionResultV1:
    schema: str
    verified_on: str
    rule: dict[str, Any]
    surfaces: tuple[ExposedSurfaceV1, ...]
    manifest: ExposedManifestV1
    exclusion_audit: ExclusionAuditV1
    grader_selftest: dict[str, Any]
    admissibility: ExposedAdmissibilityV1
    per_model: tuple[ExposedModelCertV1, ...]
    matched_family: MatchedFamilyComparisonV1
    disclosure_summary: dict[str, Any]
    lcb_inherited: UpstreamConstructionResultV1
    verdict: str
    exposed_pilot_earned: bool
    n_exposed_certifiable_models: int
    w122_fire_condition: W122FireConditionV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "verified_on": self.verified_on,
            "rule": self.rule, "surfaces": [s.to_dict() for s in self.surfaces],
            "manifest": self.manifest.to_dict(),
            "manifest_cid": self.manifest.manifest_cid(),
            "core_slice_cid": self.manifest.core_slice_cid(),
            "exclusion_audit": self.exclusion_audit.to_dict(),
            "grader_selftest": self.grader_selftest,
            "admissibility": self.admissibility.to_dict(),
            "per_model": [m.to_dict() for m in self.per_model],
            "matched_family": self.matched_family.to_dict(),
            "disclosure_summary": self.disclosure_summary,
            "lcb_inherited_verdict": str(self.lcb_inherited.verdict),
            "lcb_inherited_decision_cid": (
                self.lcb_inherited.upstream_admission.frontier_certification
                .decision.cid()),
            "verdict": self.verdict,
            "exposed_pilot_earned": bool(self.exposed_pilot_earned),
            "n_exposed_certifiable_models": int(self.n_exposed_certifiable_models),
            "w122_fire_condition": self.w122_fire_condition.to_dict()}

    def cid(self) -> str:
        return _sha256_hex({"kind": "w121_exposed_control_construction_result_v1",
                            "result": self.to_dict()})


def run_exposed_control_construction_v1(
        listing: Sequence[tuple] = ICPC_EXPOSED_LISTING_SNAPSHOT_V1,
        *, verified_on: str,
        raw_classification_sha256: str = W121_EXPOSED_RAW_CLASSIFICATION_SHA256,
        boundary: str = MAVERICK_CUTOFF_BOUNDARY,
        rule: ExposedControlRuleV1 = W121_EXPOSED_RULE,
        selftest_summary: Optional[dict[str, Any]] = None,
        candidates: Sequence[StrongerModelCandidateV1] = STRONGER_MODEL_CANDIDATES,
        disclosure_matrix: Sequence[DisclosureStatusV1] = W118_DISCLOSURE_MATRIX,
) -> ExposedControlConstructionResultV1:
    """The push-button W121 exposed-control construction + admission + exposed
    certification + matched-family comparison (RUNBOOK_W121 § 1).  NIM-free.

    On the W121 live pass the exposed tier-1 core is 44 >= 30 and the grader self-tests
    pass on all four exposed surfaces, so Maverick (KNOWN Aug-2024; 44 exposed; reachable;
    un-run here) is EXPOSED-certifiable ⇒ the exposed-control pilot is EARNED."""
    problems = classify_exposed_listing_v1(
        listing, boundary=boundary, admitted_tiers=rule.admitted_tiers)
    manifest = build_exposed_manifest_v1(
        listing, boundary=boundary, fetched_on=verified_on,
        raw_classification_sha256=raw_classification_sha256,
        admitted_tiers=rule.admitted_tiers)
    audit = exclusion_audit_v1(problems)
    st = selftest_summary or exposed_grader_selftest_summary_v1()
    adm = assess_exposed_admissibility_v1(manifest, st, rule=rule)
    per_model = certify_models_on_exposed_v1(
        manifest, adm.grader_admissible, adm.meets_min_slice, candidates=candidates)

    core_slice = select_exposed_core_slice_v1(problems, n_problems=int(rule.min_slice))
    matched = build_matched_family_comparison_v1(
        manifest, core_slice_cid_v1(core_slice), verified_on=verified_on)

    base = disclosure_matrix_summary_v1(disclosure_matrix)
    lcb = run_upstream_construction_v1()

    n_cert = sum(1 for m in per_model if m.exposed_certifiable)
    pilot_targets = [m for m in per_model if m.pilot_admissible]
    pilot_earned = bool(pilot_targets)
    verdict = VERDICT_CERTIFIABLE if pilot_earned else VERDICT_NONE

    fire = W122FireConditionV1(
        exposed_pilot_trigger=(
            "W121 earned the exposed-control pilot (44 exposed core >= 30 + Maverick "
            "exposed-certifiable); W122 carries the executed exposed-vs-resistant "
            "contrast forward."),
        paired_seed_trigger=(
            "ONLY if the exposed-vs-resistant contrast lands in the +/-"
            f"{AMBIGUITY_BAND_PP:.2f} pp ambiguity band: earn ONE paired seed on BOTH "
            "ICPC battlefields."),
        stronger_model_trigger=(
            "A reachable stronger-than-Maverick model disclosing a PRIMARY-KNOWN cutoff "
            "(prefer the strongest honest target on BOTH matched ICPC battlefields)."),
        exposed_pilot_earned=pilot_earned, exposed_pilot_was_run=False)

    return ExposedControlConstructionResultV1(
        schema=W121_EXPOSED_CONTROL_V1_SCHEMA_VERSION, verified_on=str(verified_on),
        rule=rule.to_dict(), surfaces=EXPOSED_ICPC_SURFACES_V1, manifest=manifest,
        exclusion_audit=audit, grader_selftest=st, admissibility=adm,
        per_model=per_model, matched_family=matched, disclosure_summary=base,
        lcb_inherited=lcb, verdict=verdict, exposed_pilot_earned=pilot_earned,
        n_exposed_certifiable_models=int(n_cert), w122_fire_condition=fire)


__all__ = [
    "W121_EXPOSED_CONTROL_V1_SCHEMA_VERSION", "COORDPY_ICPC_EXPOSED_CONTROL_V1",
    "W121_EXPOSED_RAW_CLASSIFICATION_SHA256", "ICPC_EXPOSED_LISTING_SNAPSHOT_V1",
    "ExposedSurfaceV1", "EXPOSED_ICPC_SURFACES_V1",
    "ExposedControlRuleV1", "W121_EXPOSED_RULE",
    "classify_exposed_listing_v1",
    "ExposedManifestV1", "build_exposed_manifest_v1",
    "W121_EXPOSED_GRADER_SELFTEST_V1", "exposed_grader_selftest_summary_v1",
    "ExposedAdmissibilityV1", "assess_exposed_admissibility_v1",
    "ExposedModelCertV1", "certify_model_exposed_v1", "certify_models_on_exposed_v1",
    "select_exposed_core_slice_v1",
    "EXPOSED_MARGIN_PASS_PP", "AMBIGUITY_BAND_PP",
    "OUTCOME_LOOPHOLE_CLOSED", "OUTCOME_CONFOUND_WEAKENS", "OUTCOME_AMBIGUOUS",
    "MatchedFamilyComparisonV1", "build_matched_family_comparison_v1",
    "ExposedVsResistantOutcomeV1", "interpret_exposed_vs_resistant_v1",
    "W122FireConditionV1",
    "ExposedControlConstructionResultV1", "run_exposed_control_construction_v1",
    "VERDICT_CERTIFIABLE", "VERDICT_NONE",
]
