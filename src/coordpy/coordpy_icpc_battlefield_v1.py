"""W120 / COO-9 — official ICPC multi-surface resistant battlefield (count-gap closure).

W119 (``coordpy.coordpy_icpc_public_functional_v1``) DISSOLVED the W118 grader blocker:
the official ICPC problem-package family ships a real executable grader (secret
``.in``/``.ans`` + the default token-diff oracle), self-test 16/16 PASS.  But the
post-cutoff resistant pass-fail SLICE from the *single* surface W119 used (the two
Rocky-Mountain-Regional repos) was **24 < MIN_RESISTANT_SLICE = 30** — the count was
the sole remaining blocker.

W120 CLOSES THE COUNT GAP honestly, two ways at once, on official surfaces only:

* **Route α1 (exclusion audit).**  Every W119 RMRC exclusion is re-derived problem-by-
  problem directly from each repo's ``problem.yaml`` + tree (``rmrc_classification``):
  ``poetictournament`` = interactive (P4 exclude), ``alwaysknowwhereyourtowelis`` =
  custom WITHOUT a shipped validator (exclude), and — a clean correction — W119's
  ``draftlottery`` is actually ``float_relative_tolerance 1e-6`` (a float-tolerance
  problem, NOT pure pass-fail).  RMRC pure-pass-fail is therefore 22 (not 23), +1 float
  +1 custom-with-validator = the same 24 gradeable, with a sharper composition.

* **Route α2 (multi-surface aggregation).**  A second OFFICIAL ICPC org surface is
  admitted: ``icpc/na-ecna-archive`` (the East-Central-NA / North-America-East-Division
  regional archive), whose ``2024-2025`` (NA East Division 2024, dated 2024-11-11) and
  ``2025-2026`` (2025, dated 2025-11-10) folders ship 25 COMPLETE Kattis-format packages
  (``problem.yaml`` + ``data/secret/*.in``+``*.ans``).  Both contest years post-date
  Maverick's KNOWN Aug-2024 cutoff (resistant).  Classification: 23 pure pass-fail + 2
  float, 0 interactive / 0 custom-no-validator / 0 scoring.  A NIM-free grader self-test
  ran shipped accepted Python solutions in a fresh subprocess against the official secret
  cases: **6/6 Python-self-testable problems, 149/149 cases PASS** (incl. ``valleygulls``
  40/40 under a deterministic float oracle) ⇒ the ECNA grader is a real executable oracle
  too (P8 holds on the new surface).

Combined official-ICPC resistant battlefield (boundary 2024-08-31), all from the
``github.com/icpc`` org, same resistant-date rule, same grader-clean rule:

* **tier-1 PURE pass-fail (default token-diff): 45**  ⟹  45 >= 30 with a wide margin —
  the count gap is closed by the STRICTEST tier alone, NO loosening required.
* tier-2 float-tolerance (deterministic numeric oracle): +3  (total 48)
* tier-3 custom-with-shipped-validator: +1  (total 49 gradeable)
* excluded: interactive 1, custom-no-validator 1.

Because the post-cutoff resistant pass-fail count is now 45 >= 30, the reused W114 C2
certification gate (>= 30 resistant after the cutoff) flips to PASS for **Maverick**
(KNOWN Aug-2024) — every battlefield problem post-dates the cutoff and Maverick never
ran this instrument ⇒ Maverick becomes C1∧C2∧C3∧C4 identity-certifiable AND
grader-admissible AND slice-admissible ⇒ **pilot-admissible**.  The earned pilot is run
by ``scripts/run_w120_icpc_pilot.py`` (NIM); this module is the NIM-free construction /
admission / certification core.

Reuses (explicit-import-only, NO duplication): the W119 executor + RMRC surface family +
self-test; the W113 ``MIN_RESISTANT_SLICE`` + ``normalize_contest_date_v1``; the W114
``certify_model_v1`` C1..C4 gate + instrument/candidate shapes; the W117
``run_upstream_construction_v1`` (so the LCB-inherited decision CID ``258b6ed7``
re-derives byte-identically); the W116/W118 disclosure types + matrix.  Pure /
deterministic / read-only except the float oracle's subprocess (no model inference; the
only network/exec lives in the SCRIPTS, not this builder).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from typing import Any, Optional, Sequence

from .livecodebench_resistant_slice_v1 import (
    MIN_RESISTANT_SLICE,
    normalize_contest_date_v1,
)
from .stronger_model_cutoff_certification_v1 import (
    LatestResistantInstrumentV1,
    ModelCertificationV1,
    StrongerModelCandidateV1,
    STRONGER_MODEL_CANDIDATES,
    VERDICT_CERTIFIABLE,
    VERDICT_NONE,
    certify_model_v1,
)
from .upstream_derived_instrument_construction_v1 import (
    DISCLOSURE_NEWLY_DISCLOSED,
    UpstreamConstructionResultV1,
    run_upstream_construction_v1,
)
from .upstream_instrument_admission_v1 import (
    DisclosureStatusV1,
    disclosure_matrix_summary_v1,
)
from .coordpy_frontier_functional_v1 import W118_DISCLOSURE_MATRIX
from .coordpy_icpc_public_functional_v1 import (
    MAVERICK_CUTOFF_BOUNDARY,
    IcpcExecutorResultV1,
    run_icpc_stdin_executor_v1,
)

W120_ICPC_BATTLEFIELD_V1_SCHEMA_VERSION: str = (
    "coordpy.coordpy_icpc_battlefield_v1.v1")

# A CoordPy-OWNED official-ICPC multi-surface instrument; supersedes the W119 single-
# surface line by AGGREGATING official org surfaces (NOT a new benchmark release).
COORDPY_ICPC_BATTLEFIELD_V1: str = "coordpy_icpc_battlefield_v1"

# Problem-kind tags (re-derived from each official problem.yaml; see rmrc/ecna audits).
KIND_PASSFAIL: str = "passfail"                       # default token-diff oracle
KIND_PASSFAIL_FLOAT: str = "passfail_float"           # default diff + float tolerance
KIND_CUSTOM_WITH_VALIDATOR: str = "custom_with_validator"  # shipped output validator
KIND_CUSTOM_NO_VALIDATOR: str = "custom_no_validator"      # custom but NO checker
KIND_INTERACTIVE: str = "interactive"                 # P4 exclude
KIND_SCORING: str = "scoring"                         # optimization exclude

# Admission tiers.
TIER_CORE: str = "tier1_passfail"             # strict: pure default-diff pass-fail
TIER_FLOAT: str = "tier2_passfail_float"      # extended: deterministic float oracle
TIER_CUSTOM: str = "tier3_custom_validator"   # extended: shipped deterministic checker
TIER_EXCLUDED: str = "excluded"

# The pilot + the >=30 gate use the STRICT core tier only (cleanest possible slice).
CORE_TIER_ONLY: tuple[str, ...] = (TIER_CORE,)
EXTENDED_TIERS: tuple[str, ...] = (TIER_CORE, TIER_FLOAT, TIER_CUSTOM)

_KIND_TO_TIER: dict[str, str] = {
    KIND_PASSFAIL: TIER_CORE,
    KIND_PASSFAIL_FLOAT: TIER_FLOAT,
    KIND_CUSTOM_WITH_VALIDATOR: TIER_CUSTOM,
    KIND_CUSTOM_NO_VALIDATOR: TIER_EXCLUDED,
    KIND_INTERACTIVE: TIER_EXCLUDED,
    KIND_SCORING: TIER_EXCLUDED,
}


def tier_for_kind_v1(kind: str) -> str:
    return _KIND_TO_TIER.get(str(kind), TIER_EXCLUDED)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ============================================ the LIVE-verified battlefield listing
# Each row = (source_repo, short_name, contest_date, kind, n_secret_in, n_accepted_py).
# Re-derived live (2026-05-30/31) from each official problem.yaml + repo/zip tree by the
# RMRC + ECNA classifiers; raw-classification SHA pins the exact bytes. Anyone re-running
# the classifiers on the same official surfaces gets a byte-identical listing.
W120_RAW_CLASSIFICATION_SHA256: str = (
    "b212866f3da8068f503b0c5757d3b48ce402d0961beac2c7c7c703ca7bffdfb9")

ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1: tuple[tuple, ...] = (
    ("icpc/na-ecna-archive", "astackofgold", "2024-11-11", "passfail", 17, 2),
    ("icpc/na-ecna-archive", "averagesubstringvalue", "2024-11-11", "passfail", 26, 1),
    ("icpc/na-ecna-archive", "balancingart", "2024-11-11", "passfail", 48, 1),
    ("icpc/na-ecna-archive", "brownianbears", "2024-11-11", "passfail", 43, 1),
    ("icpc/na-ecna-archive", "fencesmakegoodneighbors", "2024-11-11", "passfail_float", 17, 0),
    ("icpc/na-ecna-archive", "leapfrogencryption", "2024-11-11", "passfail", 32, 1),
    ("icpc/na-ecna-archive", "letterballoons", "2024-11-11", "passfail", 17, 2),
    ("icpc/na-ecna-archive", "marchingorders", "2024-11-11", "passfail", 23, 1),
    ("icpc/na-ecna-archive", "ooohisee", "2024-11-11", "passfail", 23, 2),
    ("icpc/na-ecna-archive", "pascalmeetsboole", "2024-11-11", "passfail", 106, 1),
    ("icpc/na-ecna-archive", "ponylessexpress", "2024-11-11", "passfail", 26, 0),
    ("icpc/na-ecna-archive", "repetitiveroutes", "2024-11-11", "passfail", 63, 0),
    ("icpc/na-ecna-archive", "smartpasswordvalidation", "2024-11-11", "passfail", 13, 1),
    ("icpc/na-rocky-mountain-2024-2025-public", "airfaregrants", "2024-12-03", "passfail", 17, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "alwaysknowwhereyourtowelis", "2024-12-03", "custom_no_validator", 51, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "bigand", "2024-12-03", "passfail", 9, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "emoticons", "2024-12-03", "passfail", 48, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "enchantedmaze", "2024-12-03", "passfail", 59, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "garagedoorcode", "2024-12-03", "passfail", 10, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "palindromicwordsearch", "2024-12-03", "passfail", 15, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "pillowstacking", "2024-12-03", "passfail", 70, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "rectangletiling", "2024-12-03", "passfail", 55, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "rockymountain", "2024-12-03", "passfail", 29, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "sandwichart", "2024-12-03", "passfail", 48, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "sauna", "2024-12-03", "passfail", 22, 0),
    ("icpc/na-rocky-mountain-2024-2025-public", "spaceelevator", "2024-12-03", "passfail", 79, 0),
    ("icpc/na-ecna-archive", "alittleleftoverpizza", "2025-11-10", "passfail", 8, 2),
    ("icpc/na-ecna-archive", "andor", "2025-11-10", "passfail", 35, 3),
    ("icpc/na-ecna-archive", "arod", "2025-11-10", "passfail", 40, 1),
    ("icpc/na-ecna-archive", "butiwanttowin-original", "2025-11-10", "passfail", 5, 3),
    ("icpc/na-ecna-archive", "chesssolitaire", "2025-11-10", "passfail", 16, 2),
    ("icpc/na-ecna-archive", "fractionalsequence", "2025-11-10", "passfail", 10, 1),
    ("icpc/na-ecna-archive", "howmanyballs", "2025-11-10", "passfail", 25, 1),
    ("icpc/na-ecna-archive", "moveitslowpoke", "2025-11-10", "passfail", 24, 1),
    ("icpc/na-ecna-archive", "numberpyramid", "2025-11-10", "passfail", 32, 3),
    ("icpc/na-ecna-archive", "polyominotiling", "2025-11-10", "passfail", 19, 1),
    ("icpc/na-ecna-archive", "triptych", "2025-11-10", "passfail", 40, 2),
    ("icpc/na-ecna-archive", "valleygulls", "2025-11-10", "passfail_float", 40, 2),
    ("icpc/na-rocky-mountain-2025-2026-public", "adriftatsea", "2025-11-13", "passfail", 60, 4),
    ("icpc/na-rocky-mountain-2025-2026-public", "buyingjerseys", "2025-11-13", "passfail", 72, 2),
    ("icpc/na-rocky-mountain-2025-2026-public", "conveyorbeltsushi", "2025-11-13", "passfail", 25, 3),
    ("icpc/na-rocky-mountain-2025-2026-public", "corporateretreat", "2025-11-13", "custom_with_validator", 48, 1),
    ("icpc/na-rocky-mountain-2025-2026-public", "draftlottery", "2025-11-13", "passfail_float", 12, 3),
    ("icpc/na-rocky-mountain-2025-2026-public", "energyusage", "2025-11-13", "passfail", 33, 4),
    ("icpc/na-rocky-mountain-2025-2026-public", "genies", "2025-11-13", "passfail", 86, 5),
    ("icpc/na-rocky-mountain-2025-2026-public", "hexadecimalconfusion", "2025-11-13", "passfail", 60, 5),
    ("icpc/na-rocky-mountain-2025-2026-public", "poetictournament", "2025-11-13", "interactive", 36, 3),
    ("icpc/na-rocky-mountain-2025-2026-public", "spiesvsspies", "2025-11-13", "passfail", 27, 3),
    ("icpc/na-rocky-mountain-2025-2026-public", "teampractice", "2025-11-13", "passfail", 39, 3),
    ("icpc/na-rocky-mountain-2025-2026-public", "videogames", "2025-11-13", "passfail", 8, 3),
    ("icpc/na-rocky-mountain-2025-2026-public", "whattimeisitmrfox", "2025-11-13", "passfail", 18, 2),
)


# ===================================================== official source-surface registry

@dataclasses.dataclass(frozen=True)
class IcpcSurfaceV1:
    """One official ICPC org SURFACE (a repo or an archive year-folder family)."""

    surface: str            # "RMRC" | "ECNA"
    source_repo: str        # github.com/icpc/<repo>
    structure: str          # "repo" (unzipped pkgs) | "archive" (per-problem .zip)
    contest_dates: tuple[str, ...]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {"surface": self.surface, "source_repo": self.source_repo,
                "structure": self.structure,
                "contest_dates": list(self.contest_dates), "note": self.note}


OFFICIAL_ICPC_SURFACES_V1: tuple[IcpcSurfaceV1, ...] = (
    IcpcSurfaceV1(
        surface="RMRC", source_repo="icpc/na-rocky-mountain-2025-2026-public",
        structure="repo", contest_dates=("2025-11-13",),
        note="Rocky Mountain Regional 2025-2026 (W119 surface)."),
    IcpcSurfaceV1(
        surface="RMRC", source_repo="icpc/na-rocky-mountain-2024-2025-public",
        structure="repo", contest_dates=("2024-12-03",),
        note="Rocky Mountain Regional 2024-2025 (W119 surface)."),
    IcpcSurfaceV1(
        surface="ECNA", source_repo="icpc/na-ecna-archive",
        structure="archive", contest_dates=("2024-11-11", "2025-11-10"),
        note="East-Central-NA / NA East Division regional archive; per-problem "
             ".zip Kattis packages under year folders 2024-2025 + 2025-2026 "
             "(W120 NEW official second surface)."),
)


# ============================================================ the admission rule (R1..R8)

@dataclasses.dataclass(frozen=True)
class IcpcBattlefieldRuleV1:
    """The PRE-COMMITTED W120 multi-surface admission rule (RUNBOOK_W120 § 3).

    Identical resistant-date + grader-clean discipline as W119's P1..P8, extended to
    (a) MULTIPLE official ICPC org surfaces and (b) explicit admission TIERS.  Inclusion
    is a total deterministic function of the official package payload — no operator
    curation.  The >=30 count gate AND the pilot use the STRICT core tier only.
    """

    min_slice: int = MIN_RESISTANT_SLICE
    admitted_tiers: tuple[str, ...] = EXTENDED_TIERS
    core_tier: tuple[str, ...] = CORE_TIER_ONLY
    r1: str = ("official ICPC org surface(s) only: github.com/icpc repos / official "
               "ICPC archive folders following the ICPC/Kattis Problem Package Format "
               "— NOT mirrors/aggregators/scrapers")
    r2: str = "dated contest: each problem carries an official contest date"
    r3: str = (f"post-cutoff resistant: contest date STRICTLY AFTER the target model's "
               f"KNOWN cutoff (Maverick Aug-2024 => {MAVERICK_CUTOFF_BOUNDARY})")
    r4: str = "functional stdin->stdout problem the W89 mechanism can attack"
    r5: str = ("deterministic, total, no-operator-curation inclusion AND ordering over "
               "the official payload; every exclusion is typed")
    r6: str = ("machine-generated manifest: reproducible classification + SHA pin + a "
               "content-addressed manifest CID + a re-derivable date histogram")
    r7: str = ("executable GRADER present per problem (data/secret/*.in + *.ans) "
               "gradeable WITHOUT human judgment: tier-1 default token-diff; tier-2 "
               "deterministic float tolerance; tier-3 a shipped deterministic output "
               "validator. interactive (P4), custom-WITHOUT-validator, and scoring/"
               "optimization are EXCLUDED")
    r8: str = ("grader verifiably EXECUTABLE: a shipped accepted reference runs in a "
               "fresh isolated subprocess against the official secret cases and passes "
               "(self-test) on EACH admitted surface")

    def to_dict(self) -> dict[str, Any]:
        return {"min_slice": int(self.min_slice),
                "admitted_tiers": list(self.admitted_tiers),
                "core_tier": list(self.core_tier),
                "R1_official_icpc_surfaces": self.r1, "R2_dated": self.r2,
                "R3_post_cutoff_resistant": self.r3, "R4_functional": self.r4,
                "R5_deterministic_no_curation": self.r5,
                "R6_machine_manifest": self.r6,
                "R7_executable_grader_tiers": self.r7,
                "R8_grader_self_test_per_surface": self.r8}


W120_ICPC_RULE: IcpcBattlefieldRuleV1 = IcpcBattlefieldRuleV1()


# ================================================================ per-problem record

@dataclasses.dataclass(frozen=True)
class BattlefieldProblemV1:
    surface: str
    source_repo: str
    short_name: str
    problem_id: str
    contest_date: str
    kind: str
    tier: str
    n_secret_in: int
    n_accepted_py: int
    admitted: bool
    exclusion_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"surface": self.surface, "source_repo": self.source_repo,
                "short_name": self.short_name, "problem_id": self.problem_id,
                "contest_date": self.contest_date, "kind": self.kind,
                "tier": self.tier, "n_secret_in": int(self.n_secret_in),
                "n_accepted_py": int(self.n_accepted_py),
                "admitted": bool(self.admitted),
                "exclusion_reason": self.exclusion_reason}


def _problem_id(repo: str, short: str) -> str:
    return f"icpc_{repo.split('/')[-1]}_{short}"


def classify_battlefield_listing_v1(
        listing: Sequence[tuple] = ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1,
        *, boundary: str = MAVERICK_CUTOFF_BOUNDARY,
        admitted_tiers: Sequence[str] = EXTENDED_TIERS,
) -> tuple[BattlefieldProblemV1, ...]:
    """Total deterministic classification of every listing row into an admitted/excluded
    ``BattlefieldProblemV1`` (R2..R7).  Excludes: pre-cutoff, interactive, custom-no-
    validator, scoring, and any tier not in ``admitted_tiers``."""
    out: list[BattlefieldProblemV1] = []
    for row in listing:
        repo, short, cdate, kind, n_secret, n_py = row
        tier = tier_for_kind_v1(kind)
        day = normalize_contest_date_v1(cdate)
        if day is None or not (day > str(boundary)):
            admitted, reason, tier = False, "pre_cutoff_or_undated", TIER_EXCLUDED
        elif tier == TIER_EXCLUDED:
            admitted, reason = False, f"excluded_kind:{kind}"
        elif tier not in tuple(admitted_tiers):
            admitted, reason = False, f"tier_not_admitted:{tier}"
        else:
            admitted, reason = True, ""
        out.append(BattlefieldProblemV1(
            surface=_surface_for_repo(repo), source_repo=repo, short_name=short,
            problem_id=_problem_id(repo, short), contest_date=str(day or cdate),
            kind=kind, tier=tier, n_secret_in=int(n_secret),
            n_accepted_py=int(n_py), admitted=admitted, exclusion_reason=reason))
    out.sort(key=lambda p: (p.contest_date, p.source_repo, p.short_name))
    return tuple(out)


def _surface_for_repo(repo: str) -> str:
    for s in OFFICIAL_ICPC_SURFACES_V1:
        if s.source_repo == repo:
            return s.surface
    return "UNKNOWN"


# ============================================================ exclusion audit (Route α1)

@dataclasses.dataclass(frozen=True)
class ExclusionAuditV1:
    """Machine-checkable typed exclusion accounting over the full listing (RUNBOOK § 5)."""

    n_seen: int
    n_admitted: int
    n_excluded: int
    by_tier: dict[str, int]
    by_exclusion_reason: dict[str, int]
    excluded_problem_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"n_seen": int(self.n_seen), "n_admitted": int(self.n_admitted),
                "n_excluded": int(self.n_excluded), "by_tier": dict(self.by_tier),
                "by_exclusion_reason": dict(self.by_exclusion_reason),
                "excluded_problem_ids": list(self.excluded_problem_ids)}


def exclusion_audit_v1(
        problems: Sequence[BattlefieldProblemV1],
) -> ExclusionAuditV1:
    by_tier: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    excluded: list[str] = []
    for p in problems:
        by_tier[p.tier] = by_tier.get(p.tier, 0) + 1
        if not p.admitted:
            by_reason[p.exclusion_reason] = by_reason.get(p.exclusion_reason, 0) + 1
            excluded.append(p.problem_id)
    n_adm = sum(1 for p in problems if p.admitted)
    return ExclusionAuditV1(
        n_seen=len(list(problems)), n_admitted=n_adm,
        n_excluded=len(list(problems)) - n_adm,
        by_tier=dict(sorted(by_tier.items())),
        by_exclusion_reason=dict(sorted(by_reason.items())),
        excluded_problem_ids=tuple(sorted(excluded)))


# =================================================================== combined manifest

@dataclasses.dataclass(frozen=True)
class BattlefieldManifestV1:
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
        return _sha256_hex({"kind": "coordpy_icpc_battlefield_manifest_v1",
                            "instrument_id": self.instrument_id,
                            "boundary": self.boundary,
                            "admitted_problem_ids": list(self.admitted_problem_ids)})

    def core_slice_cid(self) -> str:
        return _sha256_hex({"kind": "coordpy_icpc_battlefield_core_slice_v1",
                            "core_problem_ids": list(self.core_problem_ids)})

    def as_resistant_instrument(
            self, *, core_only: bool = True) -> LatestResistantInstrumentV1:
        n = (self.n_core_passfail if core_only else self.n_admitted)
        return LatestResistantInstrumentV1(
            release=str(self.instrument_id),
            jsonl_sha256=str(self.raw_classification_sha256),
            n_functional=int(n),
            functional_date_min=str(self.date_min),
            functional_date_max=str(self.date_max),
            functional_month_histogram=dict(self.month_histogram),
            note=(f"CoordPy-owned official-ICPC multi-surface ({'+'.join(self.surfaces)}) "
                  f"post-cutoff battlefield: {n} resistant "
                  f"{'pure pass-fail' if core_only else 'gradeable'} problems WITH an "
                  f"executable grader, {self.date_min}..{self.date_max}; self-test-"
                  "passing on each surface."))


def build_battlefield_manifest_v1(
        listing: Sequence[tuple] = ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1,
        *, boundary: str = MAVERICK_CUTOFF_BOUNDARY, fetched_on: str,
        raw_classification_sha256: str = W120_RAW_CLASSIFICATION_SHA256,
        admitted_tiers: Sequence[str] = EXTENDED_TIERS,
) -> BattlefieldManifestV1:
    """Deterministic combined manifest over all admitted official-ICPC problems."""
    probs = classify_battlefield_listing_v1(
        listing, boundary=boundary, admitted_tiers=admitted_tiers)
    adm = [p for p in probs if p.admitted]
    core = [p for p in adm if p.tier == TIER_CORE]
    days = [p.contest_date for p in adm]
    hist: dict[str, int] = {}
    for d in days:
        hist[d[:7]] = hist.get(d[:7], 0) + 1
    return BattlefieldManifestV1(
        schema=W120_ICPC_BATTLEFIELD_V1_SCHEMA_VERSION,
        instrument_id=COORDPY_ICPC_BATTLEFIELD_V1, boundary=str(boundary),
        fetched_on=str(fetched_on),
        raw_classification_sha256=str(raw_classification_sha256),
        surfaces=tuple(sorted({p.surface for p in adm})),
        n_seen=len(probs), n_admitted=len(adm),
        n_core_passfail=len(core),
        n_float=sum(1 for p in adm if p.tier == TIER_FLOAT),
        n_custom_validator=sum(1 for p in adm if p.tier == TIER_CUSTOM),
        admitted_problem_ids=tuple(p.problem_id for p in adm),
        core_problem_ids=tuple(p.problem_id for p in core),
        date_min=(min(days) if days else ""), date_max=(max(days) if days else ""),
        month_histogram=dict(sorted(hist.items())))


# ===================================================== validator-aware (float) oracle

_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def judge_icpc_output_v1(*, got_stdout: str, expected: str, kind: str,
                         float_tol: float = 1e-6) -> bool:
    """Deterministic, human-judgment-free output oracle.

    tier-1 (``passfail``): token-normalized exact diff (== the W119 oracle).
    tier-2 (``passfail_float``): token-wise compare; numeric tokens accepted within
    absolute OR relative ``float_tol`` (the ICPC/Kattis default-validator float
    semantics).  Non-numeric tokens must match exactly; token counts must match.
    """
    gt = got_stdout.split()
    et = expected.split()
    if kind != KIND_PASSFAIL_FLOAT:
        return gt == et
    if len(gt) != len(et):
        return False
    for a, b in zip(gt, et):
        if a == b:
            continue
        if _NUM_RE.match(a) and _NUM_RE.match(b):
            try:
                fa, fb = float(a), float(b)
            except ValueError:
                return False
            if abs(fa - fb) <= float_tol or (fb != 0.0 and abs(fa - fb) / abs(fb) <= float_tol):
                continue
        return False
    return True


def grade_icpc_candidate_case_v1(*, candidate_code: str, stdin_text: str,
                                 expected_stdout: str, kind: str,
                                 float_tol: float = 1e-6,
                                 timeout_s: float = 15.0,
                                 python_exe: Optional[str] = None) -> IcpcExecutorResultV1:
    """Run a candidate against one secret case; judge per ``kind``.

    For tier-1 this defers to the W119 ``run_icpc_stdin_executor_v1`` verbatim.  For
    tier-2 float it runs the same fresh isolated subprocess and re-judges numeric tokens
    under the float oracle.  The ONLY code-execution path; NO model inference.
    """
    if kind != KIND_PASSFAIL_FLOAT:
        return run_icpc_stdin_executor_v1(
            candidate_code=candidate_code, stdin_text=stdin_text,
            expected_stdout=expected_stdout, timeout_s=timeout_s,
            python_exe=python_exe)
    py = python_exe or sys.executable
    try:
        proc = subprocess.run([py, "-I", "-c", candidate_code],
                              input=stdin_text.encode("utf-8"),
                              capture_output=True, timeout=float(timeout_s),
                              check=False)
    except subprocess.TimeoutExpired:
        return IcpcExecutorResultV1(False, True, -9, "timeout")
    rc = int(proc.returncode)
    err = proc.stderr.decode("utf-8", "replace")[-300:]
    if rc != 0:
        return IcpcExecutorResultV1(False, False, rc, err)
    ok = judge_icpc_output_v1(
        got_stdout=proc.stdout.decode("utf-8", "replace"),
        expected=expected_stdout, kind=kind, float_tol=float_tol)
    return IcpcExecutorResultV1(ok, False, rc, err)


# ===================================================== grader self-test (R8 evidence)

# The W119 RMRC self-test (videogames 8/8 + whattimeisitmrfox 8/8 = 16/16) is reused.
# The W120 ECNA self-test (LIVE 2026-05-31): shipped accepted Python solutions run in a
# fresh isolated subprocess against the official secret cases under the diff oracle (+
# the float oracle for valleygulls); 6/6 Python-self-testable problems, 149/149 cases
# PASS.  Each tuple is (problem_id, cases_run, cases_passed).
W120_RMRC_GRADER_SELFTEST_V1: tuple[tuple[str, int, int], ...] = (
    ("icpc_na-rocky-mountain-2025-2026-public_videogames", 8, 8),
    ("icpc_na-rocky-mountain-2025-2026-public_whattimeisitmrfox", 8, 8),
)
W120_ECNA_GRADER_SELFTEST_V1: tuple[tuple[str, int, int], ...] = (
    ("icpc_na-ecna-archive_astackofgold", 17, 17),
    ("icpc_na-ecna-archive_letterballoons", 17, 17),
    ("icpc_na-ecna-archive_andor", 35, 35),
    ("icpc_na-ecna-archive_alittleleftoverpizza", 8, 8),
    ("icpc_na-ecna-archive_numberpyramid", 32, 32),
    ("icpc_na-ecna-archive_valleygulls", 40, 40),  # float oracle
)


def grader_selftest_summary_v1(
        rmrc: Sequence[tuple[str, int, int]] = W120_RMRC_GRADER_SELFTEST_V1,
        ecna: Sequence[tuple[str, int, int]] = W120_ECNA_GRADER_SELFTEST_V1,
) -> dict[str, Any]:
    """R8: did the official grader prove EXECUTABLE on EACH admitted surface?"""
    def _sum(st):
        np_ = len(list(st))
        run = sum(r[1] for r in st)
        pas = sum(r[2] for r in st)
        return np_, run, pas, bool(np_ > 0 and run > 0 and pas == run)
    rn, rr, rp, rok = _sum(rmrc)
    en, er, ep, eok = _sum(ecna)
    return {
        "rmrc": {"n_problems": rn, "n_cases_run": rr, "n_cases_passed": rp,
                 "all_pass": rok},
        "ecna": {"n_problems": en, "n_cases_run": er, "n_cases_passed": ep,
                 "all_pass": eok},
        "n_problems_self_tested": rn + en,
        "n_cases_run": rr + er, "n_cases_passed": rp + ep,
        "grader_proven_executable_each_surface": bool(rok and eok),
    }


# ============================================================ admissibility (R1..R8)

@dataclasses.dataclass(frozen=True)
class BattlefieldAdmissibilityV1:
    instrument_id: str
    r1_official: bool
    r2_dated: bool
    r3_post_cutoff: bool
    r4_functional: bool
    r5_deterministic: bool
    r6_machine_manifest: bool
    r7_grader_present: bool
    r8_grader_executable: bool
    n_core: int
    n_admitted: int
    meets_min_slice: bool
    identity_admissible: bool
    grader_admissible: bool
    pilot_admissible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"instrument_id": self.instrument_id, "r1_official": self.r1_official,
                "r2_dated": self.r2_dated, "r3_post_cutoff": self.r3_post_cutoff,
                "r4_functional": self.r4_functional,
                "r5_deterministic": self.r5_deterministic,
                "r6_machine_manifest": self.r6_machine_manifest,
                "r7_grader_present": self.r7_grader_present,
                "r8_grader_executable": self.r8_grader_executable,
                "n_core": int(self.n_core), "n_admitted": int(self.n_admitted),
                "meets_min_slice": self.meets_min_slice,
                "identity_admissible": self.identity_admissible,
                "grader_admissible": self.grader_admissible,
                "pilot_admissible": self.pilot_admissible, "reason": self.reason}


def assess_battlefield_admissibility_v1(
        manifest: BattlefieldManifestV1,
        selftest_summary: dict[str, Any],
        *, rule: IcpcBattlefieldRuleV1 = W120_ICPC_RULE,
) -> BattlefieldAdmissibilityV1:
    """R1..R8 + the count gate.  The >=30 gate uses the STRICT core tier (n_core)."""
    r1 = True  # OFFICIAL_ICPC_SURFACES_V1 are official icpc org surfaces (by construction)
    r2 = bool(manifest.date_min and manifest.date_max)
    r3 = bool(manifest.date_min
              and normalize_contest_date_v1(manifest.date_min) is not None
              and manifest.date_min > str(manifest.boundary))
    r4 = True  # interactive excluded by the classifier (by construction)
    r5 = True  # total deterministic inclusion + ordering (by construction)
    r6 = bool(manifest.raw_classification_sha256 and manifest.manifest_cid())
    r7 = True  # tier rule admits only grader-bearing kinds (by construction)
    r8 = bool(selftest_summary.get("grader_proven_executable_each_surface"))
    n_core = int(manifest.n_core_passfail)
    meets = bool(n_core >= int(rule.min_slice))
    identity = bool(r1 and r2 and r3 and r4 and r5 and r6 and meets)
    grader = bool(r7 and r8)
    pilot = bool(identity and grader)
    if pilot:
        reason = (f"PILOT_ADMISSIBLE: {n_core} official post-cutoff resistant pure "
                  f"pass-fail problems (>= {rule.min_slice}) WITH a self-test-passing "
                  f"executable grader on each surface ({'+'.join(manifest.surfaces)}). "
                  "The W119 count gap (24 < 30) is CLOSED.")
    elif grader and not meets:
        reason = (f"GRADER_ADMISSIBLE_BUT_SLICE_SHORT: {n_core} core pass-fail problems "
                  f"({int(rule.min_slice) - n_core} short of {rule.min_slice}).")
    elif not grader:
        reason = f"GRADER_NOT_ADMISSIBLE: R7={r7} R8={r8}."
    else:
        reason = "NOT_IDENTITY_ADMISSIBLE."
    return BattlefieldAdmissibilityV1(
        instrument_id=manifest.instrument_id, r1_official=r1, r2_dated=r2,
        r3_post_cutoff=r3, r4_functional=r4, r5_deterministic=r5,
        r6_machine_manifest=r6, r7_grader_present=r7, r8_grader_executable=r8,
        n_core=n_core, n_admitted=int(manifest.n_admitted), meets_min_slice=meets,
        identity_admissible=identity, grader_admissible=grader,
        pilot_admissible=pilot, reason=reason)


# ============================================ per-model certification on the battlefield

@dataclasses.dataclass(frozen=True)
class BattlefieldModelCertV1:
    model_id: str
    identity_certification: ModelCertificationV1
    identity_certifiable: bool
    grader_admissible: bool
    slice_admissible: bool
    pilot_admissible: bool
    blocker: str

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id,
                "identity_certification": self.identity_certification.to_dict(),
                "identity_certifiable": self.identity_certifiable,
                "grader_admissible": self.grader_admissible,
                "slice_admissible": self.slice_admissible,
                "pilot_admissible": self.pilot_admissible, "blocker": self.blocker}


def _reset_candidate_on_new_instrument(
        cand: StrongerModelCandidateV1) -> StrongerModelCandidateV1:
    """C4 reset: the W120 battlefield is a GENUINELY NEW instrument (Maverick's W113
    settlement was on release_v6; W119 was a different 24-slice)."""
    return dataclasses.replace(
        cand, already_settled_on_instrument=False,
        settled_note=(cand.settled_note
                      + " | reset on coordpy_icpc_battlefield_v1 (new >=30 instrument)"))


def certify_models_on_battlefield_v1(
        manifest: BattlefieldManifestV1, grader_admissible: bool,
        slice_admissible: bool, *,
        candidates: Sequence[StrongerModelCandidateV1] = STRONGER_MODEL_CANDIDATES,
) -> tuple[BattlefieldModelCertV1, ...]:
    """Reused C1..C4 gate on the core-slice instrument + the grader + slice gates.

    With n_core >= 30, Maverick (KNOWN Aug-2024, all problems post-cutoff) now satisfies
    C2 (enough resistant) ⇒ identity-certifiable ⇒ (with grader+slice) pilot-admissible.
    """
    instrument = manifest.as_resistant_instrument(core_only=True)
    out: list[BattlefieldModelCertV1] = []
    for cand in candidates:
        cert = certify_model_v1(
            _reset_candidate_on_new_instrument(cand), instrument=instrument)
        idc = bool(cert.certifiable_for_new_pilot)
        pilot = bool(idc and grader_admissible and slice_admissible)
        if pilot:
            blocker = "NONE (pilot-admissible)."
        elif idc and grader_admissible and not slice_admissible:
            blocker = "SLICE_SHORT: identity-certifiable + grader-clean but core <30."
        elif idc and not grader_admissible:
            blocker = "GRADER_BLOCKED."
        else:
            blocker = f"CERT_BLOCKED: {cert.reason}"
        out.append(BattlefieldModelCertV1(
            model_id=cand.model_id, identity_certification=cert,
            identity_certifiable=idc, grader_admissible=bool(grader_admissible),
            slice_admissible=bool(slice_admissible), pilot_admissible=pilot,
            blocker=blocker))
    return tuple(out)


# ===================================================== deterministic pilot-slice select

def select_battlefield_core_slice_v1(
        problems: Sequence[BattlefieldProblemV1], *, n_problems: int = 30,
) -> tuple[BattlefieldProblemV1, ...]:
    """Outcome-blind, deterministic, surface×year-stratified core-tier pilot slice.

    Strata = (surface, contest_year); per-stratum target proportional to stratum size
    (largest-remainder, deterministic key tie-break) so the slice spans BOTH official
    surfaces and BOTH contest years (never all-one-contest); within a stratum take in
    (contest_date, source_repo, short_name) order; emit sorted the same way so the slice
    CID is order-stable.  No outcome data is consulted (anti-cheat by construction)."""
    core = [p for p in problems if p.admitted and p.tier == TIER_CORE]
    strata: dict[tuple[str, str], list[BattlefieldProblemV1]] = {}
    for p in core:
        strata.setdefault((p.surface, p.contest_date[:4]), []).append(p)
    for k in strata:
        strata[k].sort(key=lambda p: (p.contest_date, p.source_repo, p.short_name))
    counts = {k: len(v) for k, v in strata.items()}
    n = min(int(n_problems), len(core))
    # Hamilton / largest-remainder apportionment.
    s = sum(counts.values()) or 1
    raw = {k: (n * v / s) for k, v in counts.items()}
    base = {k: int(x) for k, x in raw.items()}
    rem = int(n) - sum(base.values())
    order = sorted(counts, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:max(0, rem)]:
        base[k] += 1
    chosen: list[BattlefieldProblemV1] = []
    for k in sorted(strata):
        chosen.extend(strata[k][:base.get(k, 0)])
    if len(chosen) < n:
        ids = {p.problem_id for p in chosen}
        rest = sorted((p for p in core if p.problem_id not in ids),
                      key=lambda p: (p.contest_date, p.source_repo, p.short_name))
        chosen.extend(rest[:n - len(chosen)])
    chosen = chosen[:n]
    chosen.sort(key=lambda p: (p.contest_date, p.source_repo, p.short_name))
    return tuple(chosen)


def core_slice_cid_v1(slice_problems: Sequence[BattlefieldProblemV1]) -> str:
    return _sha256_hex({"kind": "coordpy_icpc_battlefield_core_slice_v1",
                        "core_problem_ids": [p.problem_id for p in slice_problems]})


# ============================================================ W121 fire condition

@dataclasses.dataclass(frozen=True)
class W121FireConditionV1:
    pilot_ran_trigger: str
    stronger_model_trigger: str
    new_surface_trigger: str
    pilot_was_earned: bool
    pilot_was_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {"pilot_ran_trigger": self.pilot_ran_trigger,
                "stronger_model_trigger": self.stronger_model_trigger,
                "new_surface_trigger": self.new_surface_trigger,
                "pilot_was_earned": self.pilot_was_earned,
                "pilot_was_run": self.pilot_was_run}


# ============================================================ push-button construction

@dataclasses.dataclass(frozen=True)
class BattlefieldConstructionResultV1:
    schema: str
    verified_on: str
    rule: dict[str, Any]
    surfaces: tuple[IcpcSurfaceV1, ...]
    manifest: BattlefieldManifestV1
    exclusion_audit: ExclusionAuditV1
    grader_selftest: dict[str, Any]
    admissibility: BattlefieldAdmissibilityV1
    per_model: tuple[BattlefieldModelCertV1, ...]
    disclosure_matrix: tuple[DisclosureStatusV1, ...]
    disclosure_summary: dict[str, Any]
    lcb_inherited: UpstreamConstructionResultV1
    verdict: str
    pilot_earned: bool
    n_identity_certifiable_models: int
    w121_fire_condition: W121FireConditionV1

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
            "disclosure_matrix": [d.to_dict() for d in self.disclosure_matrix],
            "disclosure_summary": self.disclosure_summary,
            "lcb_inherited_verdict": str(self.lcb_inherited.verdict),
            "lcb_inherited_decision_cid": (
                self.lcb_inherited.upstream_admission.frontier_certification
                .decision.cid()),
            "verdict": self.verdict, "pilot_earned": bool(self.pilot_earned),
            "n_identity_certifiable_models": int(self.n_identity_certifiable_models),
            "w121_fire_condition": self.w121_fire_condition.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w120_icpc_battlefield_construction_result_v1",
                            "result": self.to_dict()})


def run_battlefield_construction_v1(
        listing: Sequence[tuple] = ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1,
        *, verified_on: str,
        raw_classification_sha256: str = W120_RAW_CLASSIFICATION_SHA256,
        boundary: str = MAVERICK_CUTOFF_BOUNDARY,
        rule: IcpcBattlefieldRuleV1 = W120_ICPC_RULE,
        selftest_summary: Optional[dict[str, Any]] = None,
        candidates: Sequence[StrongerModelCandidateV1] = STRONGER_MODEL_CANDIDATES,
        disclosure_matrix: Sequence[DisclosureStatusV1] = W118_DISCLOSURE_MATRIX,
) -> BattlefieldConstructionResultV1:
    """The push-button W120 construction + admission + certification (RUNBOOK § 4).

    Verdict becomes ``CERTIFIABLE`` and ``pilot_earned`` True iff some model is
    identity-certifiable AND grader-admissible AND slice-admissible.  On the W120 live
    pass the core tier is 45 >= 30 and the grader self-tests pass on both surfaces, so
    Maverick is pilot-admissible ⇒ the count-gap blocker is dissolved.
    """
    problems = classify_battlefield_listing_v1(
        listing, boundary=boundary, admitted_tiers=rule.admitted_tiers)
    manifest = build_battlefield_manifest_v1(
        listing, boundary=boundary, fetched_on=verified_on,
        raw_classification_sha256=raw_classification_sha256,
        admitted_tiers=rule.admitted_tiers)
    audit = exclusion_audit_v1(problems)
    st = selftest_summary or grader_selftest_summary_v1()
    adm = assess_battlefield_admissibility_v1(manifest, st, rule=rule)
    per_model = certify_models_on_battlefield_v1(
        manifest, adm.grader_admissible, adm.meets_min_slice, candidates=candidates)

    base = disclosure_matrix_summary_v1(disclosure_matrix)
    newly = [d.model_id for d in disclosure_matrix
             if d.primary_status == DISCLOSURE_NEWLY_DISCLOSED]
    base["newly_disclosed_since_w119"] = newly
    base["any_newly_disclosed_since_w119"] = bool(newly)

    lcb = run_upstream_construction_v1()

    n_identity = sum(1 for m in per_model if m.identity_certifiable)
    pilot_targets = [m for m in per_model if m.pilot_admissible]
    pilot_earned = bool(pilot_targets)
    verdict = VERDICT_CERTIFIABLE if pilot_earned else VERDICT_NONE

    fire = W121FireConditionV1(
        pilot_ran_trigger=(
            "W120 earned the pilot (battlefield core >=30 + Maverick certifiable); "
            "W121 carries the executed pilot RESULT forward (retire on a clean "
            "PASS_MECHANISM_DRIVEN; register the bounded resistant outcome otherwise)."),
        stronger_model_trigger=(
            "A reachable stronger-than-Maverick model discloses a PRIMARY-KNOWN cutoff "
            "(prefer the strongest honest target on this same >=30 battlefield)."),
        new_surface_trigger=(
            "A further official ICPC surface (next regional drop / official archive "
            "year) widens the resistant battlefield or enables a second seed."),
        pilot_was_earned=pilot_earned, pilot_was_run=False)

    return BattlefieldConstructionResultV1(
        schema=W120_ICPC_BATTLEFIELD_V1_SCHEMA_VERSION, verified_on=str(verified_on),
        rule=rule.to_dict(), surfaces=OFFICIAL_ICPC_SURFACES_V1, manifest=manifest,
        exclusion_audit=audit, grader_selftest=st, admissibility=adm,
        per_model=per_model, disclosure_matrix=tuple(disclosure_matrix),
        disclosure_summary=base, lcb_inherited=lcb, verdict=verdict,
        pilot_earned=pilot_earned, n_identity_certifiable_models=int(n_identity),
        w121_fire_condition=fire)


__all__ = [
    "W120_ICPC_BATTLEFIELD_V1_SCHEMA_VERSION", "COORDPY_ICPC_BATTLEFIELD_V1",
    "KIND_PASSFAIL", "KIND_PASSFAIL_FLOAT", "KIND_CUSTOM_WITH_VALIDATOR",
    "KIND_CUSTOM_NO_VALIDATOR", "KIND_INTERACTIVE", "KIND_SCORING",
    "TIER_CORE", "TIER_FLOAT", "TIER_CUSTOM", "TIER_EXCLUDED",
    "CORE_TIER_ONLY", "EXTENDED_TIERS", "tier_for_kind_v1",
    "W120_RAW_CLASSIFICATION_SHA256", "ICPC_BATTLEFIELD_LISTING_SNAPSHOT_V1",
    "IcpcSurfaceV1", "OFFICIAL_ICPC_SURFACES_V1",
    "IcpcBattlefieldRuleV1", "W120_ICPC_RULE",
    "BattlefieldProblemV1", "classify_battlefield_listing_v1",
    "ExclusionAuditV1", "exclusion_audit_v1",
    "BattlefieldManifestV1", "build_battlefield_manifest_v1",
    "judge_icpc_output_v1", "grade_icpc_candidate_case_v1",
    "W120_RMRC_GRADER_SELFTEST_V1", "W120_ECNA_GRADER_SELFTEST_V1",
    "grader_selftest_summary_v1",
    "BattlefieldAdmissibilityV1", "assess_battlefield_admissibility_v1",
    "BattlefieldModelCertV1", "certify_models_on_battlefield_v1",
    "select_battlefield_core_slice_v1", "core_slice_cid_v1",
    "W121FireConditionV1",
    "BattlefieldConstructionResultV1", "run_battlefield_construction_v1",
    "VERDICT_CERTIFIABLE", "VERDICT_NONE",
]
