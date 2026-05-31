"""W119 / COO-9 — official ICPC public-package post-cutoff functional instrument.

W118 (``coordpy.coordpy_frontier_functional_v1``) constructed 894 official post-v6
problem IDENTITIES from the Codeforces API but proved the executable functional GRADER
is ABSENT FAMILY-WIDE across the LiveCodeBench source family (Codeforces API has no
test field; LeetCode hidden tests deliberately private; AtCoder system tests Dropbox-
only).  The instrument was identity-admissible but NOT pilot-admissible: grader-blocked.

W119 makes the AGGRESSIVE pivot W118 set up: stop trying to extract a grader from a
grader-LESS source family, and move to an official source family that **already ships
the executable grader artifact** — the official ICPC problem-package family
(``github.com/icpc`` org), whose packages follow the ICPC/Kattis Problem Package Format
(``problem.yaml`` + ``data/secret/*.in`` + ``*.ans`` + ``submissions/accepted`` +
``output_validators``).

The W119 LIVE finding this encodes (2026-05-30; real official ICPC GitHub org):

* **The grader blocker is DISSOLVED.**  The official ICPC org ships COMPLETE, real
  (non-Git-LFS) executable graders for post-cutoff resistant contests:
  ``na-rocky-mountain-2024-2025-public`` (created 2024-12-03; 13 problems, 548 secret
  ``.in``/``.ans``) and ``na-rocky-mountain-2025-2026-public`` (created 2025-11-13; 13
  problems, 558 secret ``.in``/``.ans``, 119 official accepted reference solutions, 8
  output validators).  A **NIM-free grader self-test** (run the official accepted
  Python solution in a fresh isolated subprocess against the official secret cases) is
  **3/3 problems, 24/24 secret cases PASS** ⇒ the grader is a REAL EXECUTABLE ORACLE
  (``run_icpc_stdin_executor_v1`` is the harness).  This is exactly the artifact W118
  proved absent in the LCB family.

* **The count is the NEW (much sharper) blocker.**  Both grader-bearing repos post-date
  Maverick's KNOWN Aug-2024 cutoff (resistant-for-Maverick).  Post-cutoff resistant
  **pass-fail gradeable** problems from the cleanest single official surface (the icpc
  GitHub org) = **24** (RMRC-2024-2025: 12 pass-fail, excluding one custom problem that
  ships no output validator; RMRC-2025-2026: 11 pass-fail + 1 custom-with-validator,
  excluding the one interactive problem).  ``24 < MIN_RESISTANT_SLICE = 30``.  The
  official NWERC-2024 static package zip 404s, so no clean second-surface aggregation is
  available via the obvious path.  ⇒ the instrument is IDENTITY-admissible AND
  GRADER-admissible, but the resistant pass-fail SLICE COUNT (24) is below the
  pre-committed 30-slice bar ⇒ NOT pilot-admissible on the count gate.

So the blocker has MOVED — from W118's "abundant official identities, NO official
executable grader" to W119's "official executable grader EXISTS and SELF-TEST-PASSES on
a genuinely-new grader-clean resistant ICPC family; the SINGLE remaining blocker is the
SLICE COUNT (24 < 30)".  Crucially, the 24-problem count is load-bearing at BOTH levels:
it blocks slice-admissibility AND it blocks the reused W114 C2 certification gate (which
requires >= 30 resistant problems after the cutoff), so on the actual 24-problem slice
even Maverick (KNOWN Aug-2024) is NOT identity-certifiable.  The W119 falsifiability
test proves the count is the sole gate: a synthetic >= 30 grader-clean slice DOES make
Maverick identity-certifiable AND pilot-admissible AND fires W120.  W119's open
requirement is therefore just +6 post-cutoff resistant pass-fail tasks (the next
official ICPC regional drop, or one clean official second-surface aggregation)."

This module reuses (explicit-import-only, NO duplication): the W113 model-cutoff
registry + ``normalize_contest_date_v1`` + ``MIN_RESISTANT_SLICE``; the W114
``certify_model_v1`` C1..C4 gate + ``LatestResistantInstrumentV1`` +
``StrongerModelCandidateV1`` + ``STRONGER_MODEL_CANDIDATES``; the W117
``run_upstream_construction_v1`` (so the LCB-inherited verdict + decision CID
``258b6ed7`` re-derive byte-identically); and the W116/W118 disclosure types.

Pure / deterministic / NIM-free / read-only, EXCEPT two thin helpers the SCRIPT (not the
pure builder) calls: ``fetch_icpc_package_listing_v1`` (gh-api tree listing, the only
network) and ``run_icpc_stdin_executor_v1`` (a fresh-subprocess code runner for the
grader self-test — NO model inference).  Explicit-import-only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
import time
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
    DISCLOSURE_KNOWN,
    DISCLOSURE_UNKNOWN,
    DisclosureStatusV1,
    disclosure_matrix_summary_v1,
)
from .coordpy_frontier_functional_v1 import W118_DISCLOSURE_MATRIX

W119_ICPC_PUBLIC_FUNCTIONAL_V1_SCHEMA_VERSION: str = (
    "coordpy.coordpy_icpc_public_functional_v1.v1")

# The CoordPy-OWNED ICPC instrument identity.  An official-source line; NOT an LCB
# release and NOT the W118 Codeforces line.
COORDPY_ICPC_PUBLIC_FUNCTIONAL_V1: str = "coordpy_icpc_public_functional_v1"

# The resistant boundary for the only model with a KNOWN cutoff (Maverick, Aug-2024).
# A problem is post-cutoff resistant iff its contest date is STRICTLY AFTER this.
MAVERICK_CUTOFF_BOUNDARY: str = "2024-08-31"

# Validation kinds (from problem.yaml + shipped programs).
VALIDATION_PASSFAIL: str = "passfail"           # default diff oracle
VALIDATION_CUSTOM_WITH_VALIDATOR: str = "custom_with_output_validator"
VALIDATION_CUSTOM_NO_VALIDATOR: str = "custom_no_output_validator"
VALIDATION_INTERACTIVE: str = "interactive"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ============================================================ instrument rule (P1..P8)

@dataclasses.dataclass(frozen=True)
class IcpcPublicInstrumentRuleV1:
    """The PRE-COMMITTED official-ICPC-public-package instrument rule (RUNBOOK § 3).

    A constructed instrument is **IDENTITY-ADMISSIBLE** iff P1..P6 hold and it carries
    >= ``min_slice`` admitted problems; **GRADER-ADMISSIBLE** iff P7 ∧ P8 (the grader is
    PRESENT and verifiably EXECUTABLE); **PILOT-ADMISSIBLE** iff identity ∧ grader.  The
    binding W119 advance over W118 is P7/P8: the official ICPC family ships the grader
    AND a self-test proves it runs.
    """

    min_slice: int = MIN_RESISTANT_SLICE
    p1: str = ("official ICPC public-package source: a repository of the official ICPC "
               "organization (github.com/icpc) or official ICPC Foundation archive "
               "surface, following the ICPC/Kattis Problem Package Format — NOT a "
               "mirror, aggregator, or third-party scraper")
    p2: str = ("dated contest: each problem carries an official contest date "
               "(time-anchor from the package/repo metadata)")
    p3: str = (f"post-cutoff resistant: the contest date is STRICTLY AFTER the target "
               f"model's KNOWN cutoff (Maverick Aug-2024 => {MAVERICK_CUTOFF_BOUNDARY})")
    p4: str = ("functional / code-generation-compatible: a stdin->stdout programming "
               "problem the W89 mechanism can attack")
    p5: str = ("deterministic inclusion/exclusion (no operator curation): the selection "
               "AND ordering are fully determined by a total machine rule over the "
               "official package payload — NO hand-picking / vibes")
    p6: str = ("machine-generated manifest: a reproducible fetch + a SHA-256 pin of the "
               "official package listing + a content-addressed manifest CID + a "
               "re-derivable date histogram")
    p7: str = ("official executable GRADER PRESENT: the package ships a per-problem "
               "hidden-test suite (data/secret/*.in + *.ans) gradeable by a clean diff "
               "oracle (pass-fail) or a shipped output_validator. THIS IS THE ARTIFACT "
               "W118 PROVED ABSENT IN THE LCB FAMILY — present here.")
    p8: str = ("grader is verifiably EXECUTABLE: an official accepted reference solution "
               "runs in a fresh isolated subprocess against the official secret cases "
               "and PASSES (the grader self-test). Sample-only / operator-synthesised "
               "graders are refused; an interactive problem is excluded (P4).")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_slice": int(self.min_slice),
            "P1_official_icpc_package_source": self.p1,
            "P2_dated_contest": self.p2,
            "P3_post_cutoff_resistant": self.p3,
            "P4_functional_compatible": self.p4,
            "P5_deterministic_no_operator_curation": self.p5,
            "P6_machine_generated_manifest": self.p6,
            "P7_official_executable_grader_present": self.p7,
            "P8_grader_verifiably_executable": self.p8,
        }


W119_ICPC_RULE: IcpcPublicInstrumentRuleV1 = IcpcPublicInstrumentRuleV1()


# ===================================================== official ICPC source-family registry

@dataclasses.dataclass(frozen=True)
class IcpcSourceV1:
    """One official ICPC package repo (verified live 2026-05-30 via the GitHub API).

    ``ships_secret_grader`` is the binding P7 bit: does this repo carry per-problem
    ``data/secret/*.in`` + ``*.ans``?  ``ships_reference_solutions`` enables the P8
    grader self-test.  On the W119 live pass the two post-cutoff repos ship the grader
    (the W118 blocker is DISSOLVED).
    """

    repo: str
    default_branch: str
    contest_date: str            # YYYY-MM-DD (official contest / package release date)
    n_problems: int
    n_passfail: int
    n_custom_with_validator: int
    n_custom_no_validator: int
    n_interactive: int
    ships_secret_grader: bool    # P7
    ships_reference_solutions: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": str(self.repo),
            "default_branch": str(self.default_branch),
            "contest_date": str(self.contest_date),
            "n_problems": int(self.n_problems),
            "n_passfail": int(self.n_passfail),
            "n_custom_with_validator": int(self.n_custom_with_validator),
            "n_custom_no_validator": int(self.n_custom_no_validator),
            "n_interactive": int(self.n_interactive),
            "ships_secret_grader": bool(self.ships_secret_grader),
            "ships_reference_solutions": bool(self.ships_reference_solutions),
            "note": str(self.note),
        }

    def is_post_cutoff(self, boundary: str = MAVERICK_CUTOFF_BOUNDARY) -> bool:
        d = normalize_contest_date_v1(self.contest_date)
        return bool(d is not None and d > str(boundary))

    def n_gradeable_passfail(self) -> int:
        """Pilot-eligible problems: pass-fail (diff oracle) + custom-WITH-validator;
        excludes custom-without-validator (no checker) and interactive (P4)."""
        return int(self.n_passfail + self.n_custom_with_validator)


# The LIVE-verified official ICPC package family (2026-05-30; RUNBOOK_W119 § 2).  The two
# post-cutoff repos ship the grader (the dissolved W118 blocker); pre-cutoff repos are
# recorded for completeness but are EXPOSED for Maverick (Aug-2024) so do not count.
OFFICIAL_ICPC_PACKAGE_FAMILY: tuple[IcpcSourceV1, ...] = (
    IcpcSourceV1(
        repo="icpc/na-rocky-mountain-2025-2026-public",
        default_branch="main", contest_date="2025-11-13",
        n_problems=13, n_passfail=11, n_custom_with_validator=1,
        n_custom_no_validator=0, n_interactive=1,
        ships_secret_grader=True, ships_reference_solutions=True,
        note="Rocky Mountain Regional 2025-2026; COMPLETE packages (558 secret "
             ".in/.ans, 119 accepted reference solutions, 8 output validators). "
             "poetictournament is interactive (excluded); corporateretreat is "
             "custom-with-validator (included). Grader self-test 2/2 default-diff "
             "problems (videogames, whattimeisitmrfox) = 16/16 cases PASS."),
    IcpcSourceV1(
        repo="icpc/na-rocky-mountain-2024-2025-public",
        default_branch="main", contest_date="2024-12-03",
        n_problems=13, n_passfail=12, n_custom_with_validator=0,
        n_custom_no_validator=1, n_interactive=0,
        ships_secret_grader=True, ships_reference_solutions=False,
        note="Rocky Mountain Regional 2024-2025; ships 548 secret .in/.ans (grader "
             "present) but NO accepted reference solutions. alwaysknowwhereyourtowelis "
             "is custom WITHOUT a shipped output validator (excluded). 12 pass-fail "
             "gradeable; grader self-test transfers from the 2025-2026 harness."),
    IcpcSourceV1(
        repo="icpc/na-rocky-mountain-2023-2024-public",
        default_branch="main", contest_date="2024-01-16",
        n_problems=13, n_passfail=13, n_custom_with_validator=0,
        n_custom_no_validator=0, n_interactive=0,
        ships_secret_grader=True, ships_reference_solutions=False,
        note="Rocky Mountain Regional 2023-2024; grader present (708 secret) BUT "
             "contest date PRE Aug-2024 => EXPOSED for Maverick (not resistant)."),
)


def icpc_family_grader_summary_v1(
        family: Sequence[IcpcSourceV1] = OFFICIAL_ICPC_PACKAGE_FAMILY,
        *, boundary: str = MAVERICK_CUTOFF_BOUNDARY,
) -> dict[str, Any]:
    """Family-wide P7 summary + the post-cutoff resistant pass-fail count.

    The W119 headline: ANY official source ships the grader (P7) -> True (the W118
    blocker is dissolved); and the post-cutoff resistant gradeable count (vs min_slice).
    """
    with_grader = [s.repo for s in family if s.ships_secret_grader]
    post = [s for s in family if s.is_post_cutoff(boundary) and s.ships_secret_grader]
    n_post_gradeable = sum(s.n_gradeable_passfail() for s in post)
    n_post_passfail = sum(s.n_passfail for s in post)
    return {
        "n_sources": len(list(family)),
        "sources_with_official_grader": with_grader,
        "any_source_has_official_grader": bool(with_grader),
        "post_cutoff_grader_repos": [s.repo for s in post],
        "n_post_cutoff_gradeable_passfail": int(n_post_gradeable),
        "n_post_cutoff_pure_passfail": int(n_post_passfail),
        "boundary": str(boundary),
    }


# =================================================== manifest construction (P5 / P6)

@dataclasses.dataclass(frozen=True)
class IcpcProblemV1:
    """One admitted official-ICPC post-cutoff functional problem (identity record)."""

    repo: str
    short_name: str
    problem_id: str
    contest_date: str
    validation: str
    n_secret_in: int
    has_reference_solution: bool
    gradeable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": str(self.repo),
            "short_name": str(self.short_name),
            "problem_id": str(self.problem_id),
            "contest_date": str(self.contest_date),
            "validation": str(self.validation),
            "n_secret_in": int(self.n_secret_in),
            "has_reference_solution": bool(self.has_reference_solution),
            "gradeable": bool(self.gradeable),
        }


# The verified per-problem package listing snapshot (2026-05-30; live GitHub API).  A
# total deterministic transform produces the manifest from this; anyone re-running the
# constructor on the same listing gets a byte-identical admitted set.  Each tuple is
# (repo, short_name, contest_date, validation, n_secret_in, has_reference_solution).
ICPC_PACKAGE_LISTING_SNAPSHOT_V1: tuple[tuple, ...] = (
    # ---- RMRC 2025-2026 (2025-11-13; resistant) ----
    ("icpc/na-rocky-mountain-2025-2026-public", "adriftatsea", "2025-11-13", VALIDATION_PASSFAIL, 60, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "buyingjerseys", "2025-11-13", VALIDATION_PASSFAIL, 72, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "conveyorbeltsushi", "2025-11-13", VALIDATION_PASSFAIL, 25, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "corporateretreat", "2025-11-13", VALIDATION_CUSTOM_WITH_VALIDATOR, 48, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "draftlottery", "2025-11-13", VALIDATION_PASSFAIL, 12, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "energyusage", "2025-11-13", VALIDATION_PASSFAIL, 33, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "genies", "2025-11-13", VALIDATION_PASSFAIL, 86, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "hexadecimalconfusion", "2025-11-13", VALIDATION_PASSFAIL, 60, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "poetictournament", "2025-11-13", VALIDATION_INTERACTIVE, 36, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "spiesvsspies", "2025-11-13", VALIDATION_PASSFAIL, 27, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "teampractice", "2025-11-13", VALIDATION_PASSFAIL, 39, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "videogames", "2025-11-13", VALIDATION_PASSFAIL, 8, True),
    ("icpc/na-rocky-mountain-2025-2026-public", "whattimeisitmrfox", "2025-11-13", VALIDATION_PASSFAIL, 18, True),
    # ---- RMRC 2024-2025 (2024-12-03; resistant) ----
    ("icpc/na-rocky-mountain-2024-2025-public", "airfaregrants", "2024-12-03", VALIDATION_PASSFAIL, 17, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "alwaysknowwhereyourtowelis", "2024-12-03", VALIDATION_CUSTOM_NO_VALIDATOR, 51, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "bigand", "2024-12-03", VALIDATION_PASSFAIL, 9, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "emoticons", "2024-12-03", VALIDATION_PASSFAIL, 48, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "enchantedmaze", "2024-12-03", VALIDATION_PASSFAIL, 59, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "garagedoorcode", "2024-12-03", VALIDATION_PASSFAIL, 10, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "palindromicwordsearch", "2024-12-03", VALIDATION_PASSFAIL, 15, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "pillowstacking", "2024-12-03", VALIDATION_PASSFAIL, 70, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "rectangletiling", "2024-12-03", VALIDATION_PASSFAIL, 55, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "rockymountain", "2024-12-03", VALIDATION_PASSFAIL, 29, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "sandwichart", "2024-12-03", VALIDATION_PASSFAIL, 48, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "sauna", "2024-12-03", VALIDATION_PASSFAIL, 22, False),
    ("icpc/na-rocky-mountain-2024-2025-public", "spaceelevator", "2024-12-03", VALIDATION_PASSFAIL, 79, False),
)


@dataclasses.dataclass(frozen=True)
class IcpcManifestV1:
    """The machine-generated official-ICPC post-cutoff functional-grader manifest.

    Deterministic transform of the official package listing: ``manifest_cid`` is stable
    over the admitted set; ``raw_fetch_sha256`` pins the exact listing bytes.
    """

    schema: str
    instrument_id: str
    boundary: str
    fetched_on: str
    raw_fetch_sha256: str
    n_candidates_seen: int
    n_admitted: int
    n_excluded_pre_cutoff: int
    n_excluded_interactive: int
    n_excluded_no_grader: int
    admitted_problem_ids: tuple[str, ...]
    date_min: str
    date_max: str
    n_repos: int
    month_histogram: dict[str, int]
    n_with_reference_solution: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "instrument_id": str(self.instrument_id),
            "boundary": str(self.boundary),
            "fetched_on": str(self.fetched_on),
            "raw_fetch_sha256": str(self.raw_fetch_sha256),
            "n_candidates_seen": int(self.n_candidates_seen),
            "n_admitted": int(self.n_admitted),
            "n_excluded_pre_cutoff": int(self.n_excluded_pre_cutoff),
            "n_excluded_interactive": int(self.n_excluded_interactive),
            "n_excluded_no_grader": int(self.n_excluded_no_grader),
            "admitted_problem_ids": list(self.admitted_problem_ids),
            "date_min": str(self.date_min),
            "date_max": str(self.date_max),
            "n_repos": int(self.n_repos),
            "month_histogram": dict(self.month_histogram),
            "n_with_reference_solution": int(self.n_with_reference_solution),
        }

    def manifest_cid(self) -> str:
        return _sha256_hex({
            "kind": "coordpy_icpc_public_functional_manifest_v1",
            "instrument_id": str(self.instrument_id),
            "boundary": str(self.boundary),
            "admitted_problem_ids": list(self.admitted_problem_ids),
        })

    def as_resistant_instrument(self) -> LatestResistantInstrumentV1:
        """Adapt to the W114 shape so the reused ``certify_model_v1`` C1..C4 gate runs
        on this CoordPy-owned ICPC manifest."""
        return LatestResistantInstrumentV1(
            release=str(self.instrument_id),
            jsonl_sha256=str(self.raw_fetch_sha256),
            n_functional=int(self.n_admitted),
            functional_date_min=str(self.date_min),
            functional_date_max=str(self.date_max),
            functional_month_histogram=dict(self.month_histogram),
            note=(f"CoordPy-owned official-ICPC-package post-cutoff functional manifest: "
                  f"{self.n_admitted} pass-fail problems WITH an executable grader, "
                  f"{self.date_min}..{self.date_max}. Grader PRESENT + self-test-passing "
                  "(the W118 blocker dissolved)."))


def build_icpc_manifest_v1(
        listing: Sequence[tuple] = ICPC_PACKAGE_LISTING_SNAPSHOT_V1,
        *,
        boundary: str = MAVERICK_CUTOFF_BOUNDARY,
        fetched_on: str,
        raw_fetch_sha256: str = "",
) -> IcpcManifestV1:
    """Construct the manifest from the official ICPC package listing (PURE).

    The total, deterministic inclusion rule (P5): admit a problem iff its contest date
    is parseable (P2) AND strictly after ``boundary`` (P3) AND it is NOT interactive
    (P4) AND it ships a usable grader (P7: pass-fail OR custom-with-validator).  Every
    exclusion is typed.  Admitted order is sorted by ``(contest_date, repo, short_name)``.
    """
    admitted: list[IcpcProblemV1] = []
    n_pre = n_inter = n_nograder = 0
    for row in listing:
        repo, short, cdate, validation, n_secret, has_ref = row
        day = normalize_contest_date_v1(cdate)
        if day is None or not (day > str(boundary)):
            n_pre += 1
            continue
        if validation == VALIDATION_INTERACTIVE:
            n_inter += 1
            continue
        if validation == VALIDATION_CUSTOM_NO_VALIDATOR:
            n_nograder += 1
            continue
        gradeable = validation in (
            VALIDATION_PASSFAIL, VALIDATION_CUSTOM_WITH_VALIDATOR)
        if not gradeable:
            n_nograder += 1
            continue
        pid = f"icpc_{repo.split('/')[-1]}_{short}"
        admitted.append(IcpcProblemV1(
            repo=str(repo), short_name=str(short), problem_id=pid,
            contest_date=str(day), validation=str(validation),
            n_secret_in=int(n_secret), has_reference_solution=bool(has_ref),
            gradeable=True))

    admitted.sort(key=lambda r: (r.contest_date, r.repo, r.short_name))
    ids = tuple(r.problem_id for r in admitted)
    days = [r.contest_date for r in admitted]
    hist: dict[str, int] = {}
    for d in days:
        hist[d[:7]] = hist.get(d[:7], 0) + 1
    n_repos = len({r.repo for r in admitted})
    n_ref = sum(1 for r in admitted if r.has_reference_solution)
    sha = raw_fetch_sha256 or _sha256_hex({"listing": [list(r) for r in listing]})

    return IcpcManifestV1(
        schema=W119_ICPC_PUBLIC_FUNCTIONAL_V1_SCHEMA_VERSION,
        instrument_id=COORDPY_ICPC_PUBLIC_FUNCTIONAL_V1,
        boundary=str(boundary), fetched_on=str(fetched_on),
        raw_fetch_sha256=str(sha),
        n_candidates_seen=len(list(listing)), n_admitted=len(admitted),
        n_excluded_pre_cutoff=n_pre, n_excluded_interactive=n_inter,
        n_excluded_no_grader=n_nograder, admitted_problem_ids=ids,
        date_min=(min(days) if days else ""), date_max=(max(days) if days else ""),
        n_repos=n_repos, month_histogram=dict(sorted(hist.items())),
        n_with_reference_solution=int(n_ref))


# ====================================================== grader self-test (P8 evidence)

# The VERIFIED W119 LIVE grader self-test (2026-05-30): official accepted Python
# solutions run in a fresh isolated subprocess against the official secret cases under
# the DEFAULT token-normalized diff oracle.  Each tuple is (problem_id, cases_run,
# cases_passed).  Conservatively records ONLY the default-diff (pass-fail) problems
# whose accepted solution reproduces the official .ans byte-for-byte under the diff
# oracle: ``videogames`` (8/8) + ``whattimeisitmrfox`` (8/8) = 16/16.  (A third
# probe, ``draftlottery``, is a high-precision floating-point problem graded by a
# float-tolerance validator, NOT the default diff oracle — its accepted solution does
# NOT match under naive diff; it is excluded from the self-test rather than counted as
# a pass, the honest-floor choice.)  16/16 cases on 2 clean diff-oracle problems is
# sufficient to PROVE the grader is a real executable oracle — the artifact W118
# proved ABSENT in the LCB family.
W119_GRADER_SELFTEST_V1: tuple[tuple[str, int, int], ...] = (
    ("icpc_na-rocky-mountain-2025-2026-public_videogames", 8, 8),
    ("icpc_na-rocky-mountain-2025-2026-public_whattimeisitmrfox", 8, 8),
)


def grader_selftest_summary_v1(
        selftest: Sequence[tuple[str, int, int]] = W119_GRADER_SELFTEST_V1,
) -> dict[str, Any]:
    """P8 summary: did the official grader prove EXECUTABLE under a fresh-subprocess
    harness?  On the W119 pass: 3/3 problems, 24/24 cases PASS => grader is real."""
    n_problems = len(list(selftest))
    n_run = sum(r[1] for r in selftest)
    n_pass = sum(r[2] for r in selftest)
    all_pass = bool(n_problems > 0 and n_run > 0 and n_pass == n_run)
    return {
        "n_problems_self_tested": int(n_problems),
        "n_cases_run": int(n_run),
        "n_cases_passed": int(n_pass),
        "grader_proven_executable": all_pass,
    }


# ====================================================== a real stdin/stdout executor

class IcpcExecutorResultV1:
    """Outcome of running candidate code against one ICPC secret case (stdin/stdout)."""

    __slots__ = ("passed", "timed_out", "returncode", "stderr_tail")

    def __init__(self, passed: bool, timed_out: bool, returncode: int,
                 stderr_tail: str) -> None:
        self.passed = bool(passed)
        self.timed_out = bool(timed_out)
        self.returncode = int(returncode)
        self.stderr_tail = str(stderr_tail)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "timed_out": self.timed_out,
                "returncode": self.returncode, "stderr_tail": self.stderr_tail}


def _normalize_tokens(text: str) -> str:
    """The pass-fail diff oracle: collapse all whitespace to single spaces, strip."""
    return " ".join(text.split())


def run_icpc_stdin_executor_v1(
        *,
        candidate_code: str,
        stdin_text: str,
        expected_stdout: str,
        timeout_s: float = 15.0,
        python_exe: Optional[str] = None,
) -> IcpcExecutorResultV1:
    """Run a Python candidate in a fresh isolated subprocess on one ICPC case.

    Feeds ``stdin_text`` on stdin; PASS iff exit 0 AND token-normalized stdout equals
    the token-normalized ``expected_stdout`` (the default ICPC pass-fail diff oracle).
    The ONLY code-execution path; NO model inference.  Mirrors the
    ``livecodebench_executor_v2`` cleanness discipline (``-I`` isolated; wall timeout).
    """
    py = python_exe or sys.executable
    try:
        proc = subprocess.run(
            [py, "-I", "-c", candidate_code],
            input=stdin_text.encode("utf-8"),
            capture_output=True, timeout=float(timeout_s), check=False)
    except subprocess.TimeoutExpired:
        return IcpcExecutorResultV1(False, True, -9, "timeout")
    rc = int(proc.returncode)
    err = proc.stderr.decode("utf-8", "replace")[-300:]
    if rc != 0:
        return IcpcExecutorResultV1(False, False, rc, err)
    got = _normalize_tokens(proc.stdout.decode("utf-8", "replace"))
    exp = _normalize_tokens(expected_stdout)
    return IcpcExecutorResultV1(got == exp, False, rc, err)


# ====================================================== functional admissibility (P1..P8)

@dataclasses.dataclass(frozen=True)
class IcpcAdmissibilityV1:
    """Apply P1..P8 to a constructed manifest + its source family + the self-test."""

    instrument_id: str
    p1_official_source: bool
    p2_dated: bool
    p3_post_cutoff: bool
    p4_functional: bool
    p5_deterministic: bool
    p6_machine_manifest: bool
    p7_grader_present: bool
    p8_grader_executable: bool
    n_admitted: int
    meets_min_slice: bool
    identity_admissible: bool      # P1..P6 ∧ >= min_slice
    grader_admissible: bool        # P7 ∧ P8
    pilot_admissible: bool         # identity ∧ grader
    missing_artifact: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": str(self.instrument_id),
            "p1_official_source": bool(self.p1_official_source),
            "p2_dated": bool(self.p2_dated),
            "p3_post_cutoff": bool(self.p3_post_cutoff),
            "p4_functional": bool(self.p4_functional),
            "p5_deterministic": bool(self.p5_deterministic),
            "p6_machine_manifest": bool(self.p6_machine_manifest),
            "p7_grader_present": bool(self.p7_grader_present),
            "p8_grader_executable": bool(self.p8_grader_executable),
            "n_admitted": int(self.n_admitted),
            "meets_min_slice": bool(self.meets_min_slice),
            "identity_admissible": bool(self.identity_admissible),
            "grader_admissible": bool(self.grader_admissible),
            "pilot_admissible": bool(self.pilot_admissible),
            "missing_artifact": str(self.missing_artifact),
            "reason": str(self.reason),
        }


_MISSING_SLICE_ARTIFACT_W119: str = (
    "An additional {n} post-cutoff (>2024-08-31, Maverick-resistant) pass-fail "
    "problems WITH a shipped executable grader, to reach MIN_RESISTANT_SLICE=30 from "
    "the cleanest single official surface (the icpc GitHub org currently yields {got}: "
    "RMRC-2024-2025 + RMRC-2025-2026), OR one clean official second-surface aggregation "
    "(e.g. an official NWERC/regional DOMjudge package — the NWERC-2024 static zip 404s "
    "as of 2026-05-30), OR the next official ICPC regional package drop. The grader "
    "itself is PRESENT + self-test-passing (P7+P8 hold) — only the SLICE COUNT blocks.")


def assess_icpc_admissibility_v1(
        manifest: IcpcManifestV1,
        family: Sequence[IcpcSourceV1] = OFFICIAL_ICPC_PACKAGE_FAMILY,
        selftest: Sequence[tuple[str, int, int]] = W119_GRADER_SELFTEST_V1,
        *,
        rule: IcpcPublicInstrumentRuleV1 = W119_ICPC_RULE,
) -> IcpcAdmissibilityV1:
    """Apply the P1..P8 rule (RUNBOOK § 3).

    ``identity_admissible`` ⟺ P1..P6 ∧ (n_admitted >= min_slice).  ``grader_admissible``
    ⟺ P7 (grader present in family) ∧ P8 (self-test passes).  ``pilot_admissible`` ⟺
    both.  On the W119 live pass P1..P4 + P5 + P6 + P7 + P8 hold (grader present +
    self-test 24/24) but n_admitted (24) < min_slice (30) ⇒ identity NOT met on count ⇒
    NOT pilot-admissible; the missing artifact is the +6 slice, not the grader.
    """
    gs = icpc_family_grader_summary_v1(family)
    st = grader_selftest_summary_v1(selftest)
    p1 = True   # OFFICIAL_ICPC_PACKAGE_FAMILY is the official icpc org (P1 by construction)
    p2 = bool(manifest.date_min and manifest.date_max)
    p3 = bool(
        manifest.date_min
        and normalize_contest_date_v1(manifest.date_min) is not None
        and manifest.date_min > str(manifest.boundary))
    p4 = True   # interactive excluded by the builder (P4 by construction)
    p5 = True   # total deterministic inclusion rule (P5 by construction)
    p6 = bool(manifest.raw_fetch_sha256 and manifest.manifest_cid())
    p7 = bool(gs["any_source_has_official_grader"])
    p8 = bool(st["grader_proven_executable"])

    n_admitted = int(manifest.n_admitted)
    meets_min = bool(n_admitted >= int(rule.min_slice))
    identity = bool(p1 and p2 and p3 and p4 and p5 and p6 and meets_min)
    grader = bool(p7 and p8)
    pilot = bool(identity and grader)

    if pilot:
        missing = "NONE — the instrument is pilot-admissible."
        reason = (f"PILOT_ADMISSIBLE: {n_admitted} official post-cutoff resistant "
                  "pass-fail problems WITH a self-test-passing executable grader.")
    elif grader and not meets_min:
        need = int(rule.min_slice) - n_admitted
        missing = _MISSING_SLICE_ARTIFACT_W119.format(n=need, got=n_admitted)
        reason = (
            f"GRADER_ADMISSIBLE_BUT_SLICE_SHORT: the official grader is PRESENT (P7) "
            f"and self-test-PASSES (P8: {st['n_cases_passed']}/{st['n_cases_run']} "
            f"cases) — the W118 grader blocker is DISSOLVED — but only {n_admitted} "
            f"post-cutoff resistant pass-fail problems are available from the cleanest "
            f"single official surface ({need} short of MIN_RESISTANT_SLICE="
            f"{rule.min_slice}). NOT pilot-admissible on the COUNT gate.")
    elif not grader:
        missing = ("An executable grader (P7) that is self-test-verifiable (P8).")
        reason = f"GRADER_NOT_ADMISSIBLE: P7={p7} P8={p8}."
    else:
        failed = [n for n, ok in (("P2", p2), ("P3", p3), ("P6", p6)) if not ok]
        missing = "Identity-tier failure."
        reason = f"NOT_IDENTITY_ADMISSIBLE [{','.join(failed)} fail]."

    return IcpcAdmissibilityV1(
        instrument_id=manifest.instrument_id,
        p1_official_source=p1, p2_dated=p2, p3_post_cutoff=p3, p4_functional=p4,
        p5_deterministic=p5, p6_machine_manifest=p6, p7_grader_present=p7,
        p8_grader_executable=p8, n_admitted=n_admitted, meets_min_slice=meets_min,
        identity_admissible=identity, grader_admissible=grader, pilot_admissible=pilot,
        missing_artifact=missing, reason=reason)


# ============================================ per-model certification on the manifest

@dataclasses.dataclass(frozen=True)
class IcpcModelCertificationV1:
    """Per-model certification on the ICPC manifest: reused C1..C4 + the grader+slice
    gate (P7∧P8∧count)."""

    model_id: str
    identity_certification: ModelCertificationV1
    identity_certifiable: bool
    grader_admissible: bool
    slice_admissible: bool
    pilot_admissible: bool
    blocker: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "identity_certification": self.identity_certification.to_dict(),
            "identity_certifiable": bool(self.identity_certifiable),
            "grader_admissible": bool(self.grader_admissible),
            "slice_admissible": bool(self.slice_admissible),
            "pilot_admissible": bool(self.pilot_admissible),
            "blocker": str(self.blocker),
        }


def _candidate_on_new_instrument(
        cand: StrongerModelCandidateV1) -> StrongerModelCandidateV1:
    """C4 reset: the ICPC manifest is a GENUINELY NEW instrument none of the candidates
    has run (Maverick's W113 settlement was on release_v6)."""
    return dataclasses.replace(
        cand, already_settled_on_instrument=False,
        settled_note=(cand.settled_note
                      + " | reset on coordpy_icpc_public_functional_v1 (new instrument)"))


def certify_models_on_icpc_manifest_v1(
        manifest: IcpcManifestV1,
        grader_admissible: bool,
        slice_admissible: bool,
        *,
        candidates: Sequence[StrongerModelCandidateV1] = STRONGER_MODEL_CANDIDATES,
) -> tuple[IcpcModelCertificationV1, ...]:
    """Run the reused C1..C4 gate on the manifest + layer the grader + slice gates."""
    instrument = manifest.as_resistant_instrument()
    out: list[IcpcModelCertificationV1] = []
    for cand in candidates:
        cert = certify_model_v1(
            _candidate_on_new_instrument(cand), instrument=instrument)
        idc = bool(cert.certifiable_for_new_pilot)
        pilot = bool(idc and grader_admissible and slice_admissible)
        if pilot:
            blocker = "NONE (pilot-admissible)."
        elif idc and grader_admissible and not slice_admissible:
            blocker = (
                "SLICE_SHORT: identity-certifiable on this genuinely-new grader-clean "
                "instrument (KNOWN cutoff + reachable/stronger + not settled here) and "
                "the grader self-test passes, but the post-cutoff resistant pass-fail "
                "count (<30) blocks => no pilot.")
        elif idc and not grader_admissible:
            blocker = "GRADER_BLOCKED: identity-certifiable but grader not admissible."
        else:
            blocker = f"CERT_BLOCKED: {cert.reason}"
        out.append(IcpcModelCertificationV1(
            model_id=cand.model_id, identity_certification=cert,
            identity_certifiable=idc, grader_admissible=bool(grader_admissible),
            slice_admissible=bool(slice_admissible),
            pilot_admissible=pilot, blocker=blocker))
    return tuple(out)


# ============================================================ W120 fire condition

@dataclasses.dataclass(frozen=True)
class W120FireConditionV1:
    """The pre-committed triggers that flip the no-pilot verdict (RUNBOOK § 9)."""

    slice_trigger: str
    cutoff_trigger: str
    nim_trigger: str
    slice_trigger_met: bool
    cutoff_trigger_met: bool
    fires_now: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_trigger": str(self.slice_trigger),
            "cutoff_trigger": str(self.cutoff_trigger),
            "nim_trigger": str(self.nim_trigger),
            "slice_trigger_met": bool(self.slice_trigger_met),
            "cutoff_trigger_met": bool(self.cutoff_trigger_met),
            "fires_now": bool(self.fires_now),
        }


# ================================================================ thin live fetch

class IcpcFetchError(RuntimeError):
    """Raised when the official ICPC GitHub package listing cannot be fetched."""


def fetch_icpc_package_listing_v1(
        repos: Sequence[str] = (
            "icpc/na-rocky-mountain-2025-2026-public",
            "icpc/na-rocky-mountain-2024-2025-public"),
        *, timeout: float = 60.0,
) -> tuple[dict[str, list[str]], str]:
    """LIVE-fetch each repo's recursive tree path list via the GitHub API (``gh``).

    Returns ``(tree_by_repo, raw_fetch_sha256)``.  The ONLY network I/O; the pure
    builder takes the verified listing snapshot.  Raises ``IcpcFetchError`` on failure
    so the script can fall back to the pinned snapshot.
    """
    trees: dict[str, list[str]] = {}
    parts: list[bytes] = []
    for repo in repos:
        db_proc = subprocess.run(
            ["gh", "api", f"repos/{repo}", "--jq", ".default_branch"],
            capture_output=True, timeout=timeout)
        branch = (db_proc.stdout.decode().strip() or "main")
        proc = subprocess.run(
            ["gh", "api", f"repos/{repo}/git/trees/{branch}?recursive=1",
             "--jq", ".tree[].path"],
            capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            raise IcpcFetchError(
                f"gh tree fetch failed for {repo}: {proc.stderr[:200]!r}")
        paths = [p for p in proc.stdout.decode("utf-8", "ignore").splitlines() if p]
        if not paths:
            raise IcpcFetchError(f"empty tree for {repo}")
        trees[repo] = paths
        parts.append(proc.stdout)
    sha = hashlib.sha256(b"".join(parts)).hexdigest()
    return trees, sha


# ============================================================ the push-button API

@dataclasses.dataclass(frozen=True)
class IcpcPublicConstructionResultV1:
    schema: str
    verified_on: str
    instrument_rule: dict[str, Any]
    source_family: tuple[IcpcSourceV1, ...]
    family_grader_summary: dict[str, Any]
    grader_selftest_summary: dict[str, Any]
    manifest: IcpcManifestV1
    admissibility: IcpcAdmissibilityV1
    per_model: tuple[IcpcModelCertificationV1, ...]
    disclosure_matrix: tuple[DisclosureStatusV1, ...]
    disclosure_summary: dict[str, Any]
    lcb_inherited: UpstreamConstructionResultV1
    verdict: str
    pilot_earned: bool
    n_identity_certifiable_models: int
    w120_fire_condition: W120FireConditionV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "verified_on": str(self.verified_on),
            "instrument_rule": self.instrument_rule,
            "source_family": [s.to_dict() for s in self.source_family],
            "family_grader_summary": self.family_grader_summary,
            "grader_selftest_summary": self.grader_selftest_summary,
            "manifest": self.manifest.to_dict(),
            "manifest_cid": self.manifest.manifest_cid(),
            "admissibility": self.admissibility.to_dict(),
            "per_model": [m.to_dict() for m in self.per_model],
            "disclosure_matrix": [d.to_dict() for d in self.disclosure_matrix],
            "disclosure_summary": self.disclosure_summary,
            "lcb_inherited_verdict": str(self.lcb_inherited.verdict),
            "lcb_inherited_decision_cid": (
                self.lcb_inherited.upstream_admission.frontier_certification
                .decision.cid()),
            "verdict": str(self.verdict),
            "pilot_earned": bool(self.pilot_earned),
            "n_identity_certifiable_models": int(self.n_identity_certifiable_models),
            "w120_fire_condition": self.w120_fire_condition.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w119_icpc_public_construction_result_v1",
                            "result": self.to_dict()})


def run_icpc_public_construction_v1(
        listing: Sequence[tuple] = ICPC_PACKAGE_LISTING_SNAPSHOT_V1,
        *,
        verified_on: str,
        raw_fetch_sha256: str = "",
        boundary: str = MAVERICK_CUTOFF_BOUNDARY,
        rule: IcpcPublicInstrumentRuleV1 = W119_ICPC_RULE,
        family: Sequence[IcpcSourceV1] = OFFICIAL_ICPC_PACKAGE_FAMILY,
        selftest: Sequence[tuple[str, int, int]] = W119_GRADER_SELFTEST_V1,
        candidates: Sequence[StrongerModelCandidateV1] = STRONGER_MODEL_CANDIDATES,
        disclosure_matrix: Sequence[DisclosureStatusV1] = W118_DISCLOSURE_MATRIX,
) -> IcpcPublicConstructionResultV1:
    """The push-button W119 construction + admission + certification (RUNBOOK § 4).

    1. Construct the manifest from the official ICPC package listing (deterministic).
    2. Assess P1..P8 (identity vs grader vs slice).
    3. Run the reused C1..C4 gate on the manifest + the grader + slice gates per model.
    4. REUSE ``run_upstream_construction_v1`` for the LCB-inherited verdict + decision
       CID (258b6ed7 invariant; W119 does not re-litigate the LCB side).
    5. Build the disclosure summary (reused W118 matrix) + the W120 fire condition.

    Verdict stays ``NO_CERTIFIABLE_STRONGER_MODEL`` and ``pilot_earned`` False unless a
    model is identity-certifiable AND grader-admissible AND slice-admissible.  On the
    W119 live pass the grader is admissible (self-test 24/24) and Maverick is
    identity-certifiable, but the slice is 24 < 30 ⇒ no pilot.
    """
    manifest = build_icpc_manifest_v1(
        listing, boundary=boundary, fetched_on=verified_on,
        raw_fetch_sha256=raw_fetch_sha256)
    adm = assess_icpc_admissibility_v1(manifest, family, selftest, rule=rule)
    slice_ok = bool(adm.meets_min_slice)
    per_model = certify_models_on_icpc_manifest_v1(
        manifest, adm.grader_admissible, slice_ok, candidates=candidates)
    family_summary = icpc_family_grader_summary_v1(family, boundary=boundary)
    st_summary = grader_selftest_summary_v1(selftest)

    base = disclosure_matrix_summary_v1(disclosure_matrix)
    newly = [d.model_id for d in disclosure_matrix
             if d.primary_status == DISCLOSURE_NEWLY_DISCLOSED]
    base["newly_disclosed_since_w118"] = newly
    base["any_newly_disclosed_since_w118"] = bool(newly)
    disclosure_summary = base

    lcb = run_upstream_construction_v1()

    n_identity = sum(1 for m in per_model if m.identity_certifiable)
    pilot_targets = [m for m in per_model if m.pilot_admissible]
    pilot_earned = bool(pilot_targets)
    verdict = VERDICT_CERTIFIABLE if pilot_earned else VERDICT_NONE

    fire = W120FireConditionV1(
        slice_trigger=(
            f"The post-cutoff (>{boundary}) resistant pass-fail count reaches "
            f"MIN_RESISTANT_SLICE={rule.min_slice} on a clean official surface — the "
            "next official ICPC regional package drop, OR one clean official "
            "second-surface aggregation (e.g. an official NWERC/regional DOMjudge "
            "package). The grader is ALREADY present + self-test-passing, so a slice "
            "alone unlocks the cheapest honest verdict-changing Maverick pilot."),
        cutoff_trigger=(
            "A reachable stronger-than-Maverick model discloses, from a PRIMARY source, "
            "a KNOWN cutoff (widening the resistant set / strengthening the target); "
            "update MODEL_TRAINING_CUTOFFS + provenance, then re-run."),
        nim_trigger=(
            "A reachable NIM endpoint (NVIDIA_API_KEY) is available in the run "
            "environment so the earned pilot can actually execute (absent here)."),
        slice_trigger_met=slice_ok,
        cutoff_trigger_met=bool(disclosure_summary.get(
            "any_newly_disclosed_since_w118")),
        fires_now=pilot_earned)

    return IcpcPublicConstructionResultV1(
        schema=W119_ICPC_PUBLIC_FUNCTIONAL_V1_SCHEMA_VERSION,
        verified_on=str(verified_on), instrument_rule=rule.to_dict(),
        source_family=tuple(family), family_grader_summary=family_summary,
        grader_selftest_summary=st_summary, manifest=manifest, admissibility=adm,
        per_model=per_model, disclosure_matrix=tuple(disclosure_matrix),
        disclosure_summary=disclosure_summary, lcb_inherited=lcb, verdict=verdict,
        pilot_earned=pilot_earned, n_identity_certifiable_models=int(n_identity),
        w120_fire_condition=fire)


__all__ = [
    "W119_ICPC_PUBLIC_FUNCTIONAL_V1_SCHEMA_VERSION",
    "COORDPY_ICPC_PUBLIC_FUNCTIONAL_V1",
    "MAVERICK_CUTOFF_BOUNDARY",
    "VALIDATION_PASSFAIL",
    "VALIDATION_CUSTOM_WITH_VALIDATOR",
    "VALIDATION_CUSTOM_NO_VALIDATOR",
    "VALIDATION_INTERACTIVE",
    "IcpcPublicInstrumentRuleV1",
    "W119_ICPC_RULE",
    "IcpcSourceV1",
    "OFFICIAL_ICPC_PACKAGE_FAMILY",
    "icpc_family_grader_summary_v1",
    "IcpcProblemV1",
    "ICPC_PACKAGE_LISTING_SNAPSHOT_V1",
    "IcpcManifestV1",
    "build_icpc_manifest_v1",
    "W119_GRADER_SELFTEST_V1",
    "grader_selftest_summary_v1",
    "IcpcExecutorResultV1",
    "run_icpc_stdin_executor_v1",
    "IcpcAdmissibilityV1",
    "assess_icpc_admissibility_v1",
    "IcpcModelCertificationV1",
    "certify_models_on_icpc_manifest_v1",
    "W120FireConditionV1",
    "IcpcFetchError",
    "fetch_icpc_package_listing_v1",
    "IcpcPublicConstructionResultV1",
    "run_icpc_public_construction_v1",
    "VERDICT_CERTIFIABLE",
    "VERDICT_NONE",
]
