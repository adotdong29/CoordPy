"""W118 / COO-9 — CoordPy-OWNED post-v6 functional-instrument construction.

W117 attacked the upstream CONSTRUCTION provenance at eight authoritative surfaces
and proved that NO post-v6 instrument can be *inherited* from LiveCodeBench's
published provenance (LCB publishes only packaged releases — no collection pipeline,
no forward problem-id manifest), so the W117 ``B1`` criterion (authoritative
LCB-published provenance) refused every post-v6 path.

W118 asks the strictly harder, more aggressive question and does NOT just re-run the
W117 checker:

> Stop waiting for a packaged ``release_v7``.  Can CoordPy itself CONSTRUCT a
> reproducible, machine-checkable, OFFICIAL-SOURCE post-v6 functional instrument —
> built directly from the official contest source family LCB already names
> (Codeforces / AtCoder / LeetCode), with provenance STRICTER than LCB's currently
> published post-v6 story — without pretending it is an LCB release?

This module is that constructor.  It is a NEW, explicitly CoordPy-OWNED instrument
line (``coordpy_frontier_functional_v1``); it is **not** "LCB v7" and it does **not**
smuggle hand-curated problems in under "upstream-derived".  It locks the instrument
rule BEFORE building (``FrontierFunctionalInstrumentRuleV1`` — O1..O7), runs a REAL
constructor against the official Codeforces API (``build_frontier_manifest_from_
codeforces_v1`` over ``fetch_codeforces_official_v1``), and applies a deterministic,
no-operator-curation inclusion rule to produce a machine-generated manifest with a
pinned fetch SHA + a content-addressed manifest CID + a date histogram.

The W118 LIVE finding this encodes (2026-05-30; real Codeforces API):

* **Identity tier (O1..O6) is SOLVED.**  The official Codeforces API
  (``contest.list`` + ``problemset.problems``) yields, deterministically and
  reproducibly, **903 FINISHED PROGRAMMING-type problems dated strictly after the
  ``release_v6`` functional frontier 2025-04-05** (2025-04-05 .. 2026-05-30, 131
  contests) — far above ``MIN_RESISTANT_SLICE`` = 30.  A dated, post-v6,
  functional problem-IDENTITY manifest IS officially constructible.
* **Grader tier (O7) is the BLOCKER, and it is a SOURCE-FAMILY property.**  A
  *runnable* functional instrument needs an executable hidden-test suite per problem
  to GRADE generated code.  No source in the official family publishes one through a
  clean, official, machine-checkable surface: the Codeforces API exposes problem
  metadata only (no test field / no test endpoint); LeetCode's tests are "not public
  — even premium users cannot access them"; AtCoder distributes system tests only via
  a Dropbox shared folder (no official API; automation discouraged).  Sample-only
  grading is non-credible (high false-pass on hidden tests) and operator-synthesised
  tests are exactly the B2/O5 operator curation the discipline refuses.  ⇒ the
  instrument is **identity-admissible but NOT pilot-admissible**; the EXACT missing
  artifact is a *reproducible official executable per-problem test suite*.

So the blocker has MOVED — from W117's "no post-v6 problem identities can be
constructed" to W118's "post-v6 problem identities are abundantly constructible (903,
official, dated, deterministic), but no official source publishes the executable
functional GRADER".  Sharper still on certification: on the identity tier, Maverick
(KNOWN cutoff August-2024) has ALL 903 problems resistant (every contest post-dates
2024-08) on a GENUINELY NEW instrument it never ran — so it is C1∧C2∧C3∧C4
identity-certifiable; the ONLY thing blocking a verdict-changing Maverick pilot is the
missing official grader (O7).

This module reuses (explicit-import-only, NO duplication): the W113 model-cutoff
registry + ``normalize_contest_date_v1`` + ``MIN_RESISTANT_SLICE``; the W114
``certify_model_v1`` C1..C4 gate + ``LatestResistantInstrumentV1`` +
``STRONGER_MODEL_CANDIDATES``; and the W117 ``run_upstream_construction_v1`` (so the
LCB-inherited certification verdict + its decision CID ``258b6ed7`` re-derive
byte-identically — W118 ADDS the CoordPy-owned lane on top, it does not re-litigate
the LCB side).

Pure / deterministic / NIM-free / read-only, EXCEPT the single thin
``fetch_codeforces_official_v1`` network helper (urllib; the only I/O; the script,
not the pure builder, calls it — mirroring the W117 script's corpus-reverify pattern).
Explicit-import-only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Optional, Sequence

from .livecodebench_resistant_slice_v1 import (
    CONFIDENCE_KNOWN,
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

W118_FRONTIER_FUNCTIONAL_V1_SCHEMA_VERSION: str = (
    "coordpy.coordpy_frontier_functional_v1.v1")

# The CoordPy-OWNED instrument identity.  Deliberately NOT "release_v7" / "LCB v7":
# this is a CoordPy line built from official sources, with its own provenance.
COORDPY_FRONTIER_FUNCTIONAL_V1: str = "coordpy_frontier_functional_v1"

# The post-v6 frontier: the admitted ``release_v6`` functional frontier date.  A
# CoordPy-owned problem is post-v6 iff its contest date is STRICTLY AFTER this.
FRONTIER_DATE: str = "2025-04-05"

# Official Codeforces API endpoints (the clean, official, machine-checkable surface).
CODEFORCES_CONTEST_LIST_URL: str = "https://codeforces.com/api/contest.list?gym=false"
CODEFORCES_PROBLEMSET_URL: str = "https://codeforces.com/api/problemset.problems"

# Codeforces problem ``type`` that is functional / code-generation-compatible.
_CF_PROGRAMMING_TYPE: str = "PROGRAMMING"
# Codeforces contest ``phase`` for a real, finished, dated past contest.
_CF_FINISHED_PHASE: str = "FINISHED"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


# ============================================================ instrument rule (§3)

@dataclasses.dataclass(frozen=True)
class FrontierFunctionalInstrumentRuleV1:
    """The PRE-COMMITTED CoordPy-OWNED post-v6 functional-instrument rule (RUNBOOK § 3).

    A constructed instrument is **IDENTITY-ADMISSIBLE** iff O1..O6 hold and it carries
    >= ``min_slice`` admitted problems; it is **PILOT-ADMISSIBLE** iff it is also
    GRADER-ADMISSIBLE (O7).  The split is the load-bearing W118 distinction: the
    official source family yields the identities (O1..O6) but NOT the executable
    grader (O7).
    """

    min_slice: int = MIN_RESISTANT_SLICE
    o1: str = ("official source family: a source LCB itself names "
               "(Codeforces / AtCoder / LeetCode), accessed through its OFFICIAL "
               "surface (e.g. the official Codeforces API) — NOT a random mirror, "
               "aggregator, or third-party scraper, and NOT presented as an LCB "
               "release")
    o2: str = ("dated problems: each problem carries an official contest start date "
               "(time-anchor)")
    o3: str = (f"post-v6: the contest date is STRICTLY AFTER the release_v6 "
               f"functional frontier {FRONTIER_DATE}")
    o4: str = ("functional / code-generation-compatible: a programming problem the "
               "W89 mechanism can attack")
    o5: str = ("deterministic inclusion/exclusion (no operator curation): the "
               "selection AND ordering are fully determined by a total machine rule "
               "over the official payload, so anyone re-running it on the same bytes "
               "obtains the byte-identical set — NO hand-picking / vibes")
    o6: str = ("machine-generated manifest: a reproducible fetch + a SHA-256 pin of "
               "the official source bytes + a content-addressed manifest CID + a "
               "re-derivable date histogram")
    o7: str = ("official executable functional GRADER artifact: a reproducible, "
               "official, machine-checkable per-problem hidden-test suite to GRADE "
               "generated code. Sample-only tests are insufficient (high hidden-test "
               "false-pass => an uninterpretable B-A1); operator-synthesised tests "
               "are operator curation (refused by O5). This is the binding gate.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_slice": int(self.min_slice),
            "O1_official_source_family": self.o1,
            "O2_dated_problems": self.o2,
            "O3_post_v6": self.o3,
            "O4_functional_compatible": self.o4,
            "O5_deterministic_no_operator_curation": self.o5,
            "O6_machine_generated_manifest": self.o6,
            "O7_official_executable_grader": self.o7,
        }


W118_FRONTIER_RULE: FrontierFunctionalInstrumentRuleV1 = (
    FrontierFunctionalInstrumentRuleV1())


# ===================================================== official-source-family registry

# Per-source test-artifact status (the O7 axis), classed at the source level.
TEST_ARTIFACT_CLEAN_API: str = "CLEAN_OFFICIAL_API"       # reproducible official API
TEST_ARTIFACT_DROPBOX_NON_API: str = "DROPBOX_NON_API"    # official but Dropbox, no API
TEST_ARTIFACT_NONE: str = "NONE"                          # not published at all

SOURCE_CODEFORCES_API: str = "codeforces_official_api"
SOURCE_ATCODER: str = "atcoder_official"
SOURCE_LEETCODE: str = "leetcode_official"


@dataclasses.dataclass(frozen=True)
class OfficialSourceV1:
    """One member of the official source family LCB names + its W118 LIVE surfaces.

    ``has_official_executable_test_suite`` is the binding O7 bit: does this source
    publish a reproducible per-problem hidden-test suite through a clean, official,
    machine-checkable surface?  On the W118 live pass every member is False — the
    grader artifact is absent FAMILY-WIDE (verified 2026-05-30, primary sources).
    """

    source_kind: str
    official_surface: str
    has_problem_metadata_api: bool
    has_official_dates: bool
    has_statements: bool
    has_sample_tests: bool
    has_official_executable_test_suite: bool   # O7
    test_artifact_status: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": str(self.source_kind),
            "official_surface": str(self.official_surface),
            "has_problem_metadata_api": bool(self.has_problem_metadata_api),
            "has_official_dates": bool(self.has_official_dates),
            "has_statements": bool(self.has_statements),
            "has_sample_tests": bool(self.has_sample_tests),
            "has_official_executable_test_suite": bool(
                self.has_official_executable_test_suite),
            "test_artifact_status": str(self.test_artifact_status),
            "note": str(self.note),
        }


# The LIVE-verified official source family (2026-05-30; RUNBOOK_W118 § 2).  All three
# sources LCB names; each verified at its official surface.  Codeforces is the clean
# machine-checkable IDENTITY source (drives the manifest); none of the three publishes
# a clean official executable GRADER (O7) — the family-wide blocker.
OFFICIAL_SOURCE_FAMILY: tuple[OfficialSourceV1, ...] = (
    OfficialSourceV1(
        source_kind=SOURCE_CODEFORCES_API,
        official_surface="https://codeforces.com/api/ (contest.list + "
                         "problemset.problems)",
        has_problem_metadata_api=True,
        has_official_dates=True,           # contest.list startTimeSeconds
        has_statements=True,               # problem HTML page (scrape)
        has_sample_tests=True,             # sample I/O in the statement HTML
        has_official_executable_test_suite=False,  # NO test field / endpoint in API
        test_artifact_status=TEST_ARTIFACT_NONE,
        note="Clean official JSON API yields problem IDENTITY metadata + contest "
             "dates (the W118 manifest source). The API record carries NO test field "
             "and there is NO official test-case endpoint; hidden judge tests are "
             "never published. => identity-admissible source, grader-absent."),
    OfficialSourceV1(
        source_kind=SOURCE_ATCODER,
        official_surface="https://atcoder.jp/ (problem pages) + Dropbox system-test "
                         "shared folder",
        has_problem_metadata_api=False,    # no official problems JSON API (kenkoooo = 3rd-party)
        has_official_dates=True,           # contest pages carry dates
        has_statements=True,
        has_sample_tests=True,
        has_official_executable_test_suite=False,  # Dropbox != clean official API
        test_artifact_status=TEST_ARTIFACT_DROPBOX_NON_API,
        note="The ONLY family member that publishes full system test cases — but via "
             "a Dropbox shared folder, NOT a clean official machine-checkable API; "
             "AtCoder discourages automated downloading. Fails O1 (clean official "
             "surface) + O6 (reproducible machine fetch). The closest-to-viable path, "
             "still not admissible under the no-mirror / reproducible-fetch rule."),
    OfficialSourceV1(
        source_kind=SOURCE_LEETCODE,
        official_surface="https://leetcode.com/graphql (semi-official, undocumented)",
        has_problem_metadata_api=True,     # GraphQL problem content
        has_official_dates=True,           # contest schedule
        has_statements=True,
        has_sample_tests=True,             # example cases in statement
        has_official_executable_test_suite=False,  # hidden tests deliberately private
        test_artifact_status=TEST_ARTIFACT_NONE,
        note="LeetCode's hidden test cases are 'not public — even premium users "
             "cannot access them' (deliberate, to prevent copying). No official "
             "test-case API; third-party extraction violates ToS. Grader-absent."),
)


def source_family_grader_summary_v1(
        family: Sequence[OfficialSourceV1] = OFFICIAL_SOURCE_FAMILY,
) -> dict[str, Any]:
    """Family-wide O7 summary: does ANY official source publish a clean executable
    grader?  On the W118 pass: none (the load-bearing grader blocker)."""
    with_grader = [
        s.source_kind for s in family
        if s.has_official_executable_test_suite]
    with_identity = [
        s.source_kind for s in family
        if s.has_problem_metadata_api and s.has_official_dates]
    return {
        "n_sources": len(list(family)),
        "sources_with_clean_identity_api": with_identity,
        "sources_with_official_executable_grader": with_grader,
        "any_source_has_official_grader": bool(with_grader),
        "any_source_has_clean_identity_api": bool(with_identity),
    }


# =================================================== manifest construction (§4 / §5)

@dataclasses.dataclass(frozen=True)
class FrontierProblemV1:
    """One admitted CoordPy-owned post-v6 functional problem (identity record)."""

    source: str
    contest_id: int
    index: str
    problem_id: str
    name: str
    contest_date: str          # YYYY-MM-DD
    problem_type: str
    functional_compatible: bool
    has_official_test_suite: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "contest_id": int(self.contest_id),
            "index": str(self.index),
            "problem_id": str(self.problem_id),
            "name": str(self.name),
            "contest_date": str(self.contest_date),
            "problem_type": str(self.problem_type),
            "functional_compatible": bool(self.functional_compatible),
            "has_official_test_suite": bool(self.has_official_test_suite),
        }


@dataclasses.dataclass(frozen=True)
class FrontierManifestV1:
    """The machine-generated CoordPy-owned post-v6 functional-identity manifest.

    Deterministic transform of the official Codeforces payload: anyone re-running
    ``build_frontier_manifest_from_codeforces_v1`` on the same input bytes obtains a
    byte-identical manifest (``manifest_cid`` stable).  ``raw_fetch_sha256`` pins the
    exact official source bytes the manifest was built from (provenance).
    """

    schema: str
    instrument_id: str
    source: str
    frontier_date: str
    fetched_on: str
    raw_fetch_sha256: str
    n_candidates_seen: int
    n_admitted: int
    n_excluded_not_programming: int
    n_excluded_not_finished: int
    n_excluded_missing_date: int
    n_excluded_not_after_frontier: int
    admitted_problem_ids: tuple[str, ...]
    date_min: str
    date_max: str
    n_contests: int
    month_histogram: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "instrument_id": str(self.instrument_id),
            "source": str(self.source),
            "frontier_date": str(self.frontier_date),
            "fetched_on": str(self.fetched_on),
            "raw_fetch_sha256": str(self.raw_fetch_sha256),
            "n_candidates_seen": int(self.n_candidates_seen),
            "n_admitted": int(self.n_admitted),
            "n_excluded_not_programming": int(self.n_excluded_not_programming),
            "n_excluded_not_finished": int(self.n_excluded_not_finished),
            "n_excluded_missing_date": int(self.n_excluded_missing_date),
            "n_excluded_not_after_frontier": int(
                self.n_excluded_not_after_frontier),
            "admitted_problem_ids": list(self.admitted_problem_ids),
            "date_min": str(self.date_min),
            "date_max": str(self.date_max),
            "n_contests": int(self.n_contests),
            "month_histogram": dict(self.month_histogram),
        }

    def manifest_cid(self) -> str:
        """Content-address the ADMITTED SET (ids + boundary facts), so the manifest
        identity is reproducible from the deterministic transform alone."""
        return _sha256_hex({
            "kind": "coordpy_frontier_functional_manifest_v1",
            "instrument_id": str(self.instrument_id),
            "frontier_date": str(self.frontier_date),
            "admitted_problem_ids": list(self.admitted_problem_ids),
        })

    def as_resistant_instrument(self) -> LatestResistantInstrumentV1:
        """Adapt to the W114 ``LatestResistantInstrumentV1`` shape so the reused
        ``certify_model_v1`` C1..C4 gate runs on this CoordPy-owned manifest."""
        return LatestResistantInstrumentV1(
            release=str(self.instrument_id),
            jsonl_sha256=str(self.raw_fetch_sha256),
            n_functional=int(self.n_admitted),
            functional_date_min=str(self.date_min),
            functional_date_max=str(self.date_max),
            functional_month_histogram=dict(self.month_histogram),
            note=(f"CoordPy-owned official-source ({self.source}) post-v6 functional "
                  f"IDENTITY manifest: {self.n_admitted} problems "
                  f"{self.date_min}..{self.date_max}. IDENTITY tier only — no "
                  "official executable grader (O7), so NOT pilot-runnable."))


def build_frontier_manifest_from_codeforces_v1(
        contest_list_payload: dict[str, Any],
        problemset_payload: dict[str, Any],
        *,
        frontier_date: str = FRONTIER_DATE,
        fetched_on: str,
        raw_fetch_sha256: str = "",
) -> FrontierManifestV1:
    """Construct the manifest from official Codeforces API payloads (PURE).

    The total, deterministic inclusion rule (O5): admit a problem iff its ``type`` is
    PROGRAMMING (O4) AND its contest ``phase`` is FINISHED (a real dated past contest)
    AND its contest's start date is parseable (O2) AND strictly after ``frontier_date``
    (O3).  Every exclusion is typed.  ``problem_id`` is ``"cf_{contestId}_{index}"``;
    the admitted order is sorted by ``(contest_date, contest_id, index)`` so the set +
    ordering are reproducible (no operator discretion).

    No network here — the caller passes parsed payloads (the script fetches them).
    ``raw_fetch_sha256`` (the SHA of the concatenated official source bytes) is recorded
    for provenance; if "" it is derived from the canonical payloads.
    """
    contests = (contest_list_payload or {}).get("result", []) or []
    cdate: dict[int, str] = {}
    cphase: dict[int, str] = {}
    for c in contests:
        cid = c.get("id")
        if cid is None:
            continue
        cphase[int(cid)] = str(c.get("phase", ""))
        st = c.get("startTimeSeconds")
        if st is not None:
            # Normalize the epoch seconds to a UTC YYYY-MM-DD day WITHOUT importing a
            # tz-aware clock (determinism): integer civil-date from the epoch.
            cdate[int(cid)] = _epoch_to_utc_day(int(st))

    problems = (
        ((problemset_payload or {}).get("result", {}) or {}).get("problems", [])
        or [])

    admitted: list[FrontierProblemV1] = []
    n_not_prog = n_not_finished = n_missing_date = n_not_after = 0
    for p in problems:
        cid = p.get("contestId")
        index = str(p.get("index", ""))
        ptype = str(p.get("type", ""))
        if ptype != _CF_PROGRAMMING_TYPE:
            n_not_prog += 1
            continue
        if cid is None or cphase.get(int(cid)) != _CF_FINISHED_PHASE:
            n_not_finished += 1
            continue
        day = normalize_contest_date_v1(cdate.get(int(cid)))
        if day is None:
            n_missing_date += 1
            continue
        if not (day > str(frontier_date)):
            n_not_after += 1
            continue
        admitted.append(FrontierProblemV1(
            source=SOURCE_CODEFORCES_API,
            contest_id=int(cid), index=index,
            problem_id=f"cf_{int(cid)}_{index}",
            name=str(p.get("name", "")),
            contest_date=day, problem_type=ptype,
            functional_compatible=True,
            has_official_test_suite=False))   # O7: never available from CF API

    admitted.sort(key=lambda r: (r.contest_date, r.contest_id, r.index))
    ids = tuple(r.problem_id for r in admitted)
    days = [r.contest_date for r in admitted]
    hist: dict[str, int] = {}
    for d in days:
        hist[d[:7]] = hist.get(d[:7], 0) + 1
    n_contests = len({r.contest_id for r in admitted})

    sha = raw_fetch_sha256 or _sha256_hex(
        {"contest_list": contest_list_payload, "problemset": problemset_payload})

    return FrontierManifestV1(
        schema=W118_FRONTIER_FUNCTIONAL_V1_SCHEMA_VERSION,
        instrument_id=COORDPY_FRONTIER_FUNCTIONAL_V1,
        source=SOURCE_CODEFORCES_API,
        frontier_date=str(frontier_date),
        fetched_on=str(fetched_on),
        raw_fetch_sha256=str(sha),
        n_candidates_seen=len(problems),
        n_admitted=len(admitted),
        n_excluded_not_programming=n_not_prog,
        n_excluded_not_finished=n_not_finished,
        n_excluded_missing_date=n_missing_date,
        n_excluded_not_after_frontier=n_not_after,
        admitted_problem_ids=ids,
        date_min=(min(days) if days else ""),
        date_max=(max(days) if days else ""),
        n_contests=n_contests,
        month_histogram=dict(sorted(hist.items())),
    )


# Days per month for civil-date conversion (Jan..Dec); Feb handled via leap rule.
_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _epoch_to_utc_day(epoch_seconds: int) -> str:
    """Convert epoch seconds to a UTC ``YYYY-MM-DD`` day deterministically.

    Pure integer civil-date arithmetic (no wall-clock import; ``datetime.now`` is
    banned in this codebase for determinism, and even ``utcfromtimestamp`` is avoided
    here to keep the transform self-contained + offline-reproducible).
    """
    days = int(epoch_seconds) // 86400          # whole days since 1970-01-01 (UTC)
    y = 1970
    while True:
        leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
        ydays = 366 if leap else 365
        if days < ydays:
            break
        days -= ydays
        y += 1
    mo = 0
    while True:
        dim = _DAYS_IN_MONTH[mo]
        if mo == 1 and (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)):
            dim = 29
        if days < dim:
            break
        days -= dim
        mo += 1
    return f"{y:04d}-{mo + 1:02d}-{days + 1:02d}"


# ====================================================== functional admissibility (§3)

@dataclasses.dataclass(frozen=True)
class FrontierFunctionalAdmissibilityV1:
    """Apply O1..O7 to a constructed manifest + its source."""

    instrument_id: str
    o1_official_source: bool
    o2_dated: bool
    o3_post_v6: bool
    o4_functional: bool
    o5_deterministic_no_curation: bool
    o6_machine_manifest: bool
    o7_official_grader: bool
    n_admitted: int
    meets_min_slice: bool
    identity_admissible: bool      # O1..O6 ∧ >= min_slice
    grader_admissible: bool        # O7
    pilot_admissible: bool         # identity ∧ grader
    missing_artifact: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": str(self.instrument_id),
            "o1_official_source": bool(self.o1_official_source),
            "o2_dated": bool(self.o2_dated),
            "o3_post_v6": bool(self.o3_post_v6),
            "o4_functional": bool(self.o4_functional),
            "o5_deterministic_no_curation": bool(
                self.o5_deterministic_no_curation),
            "o6_machine_manifest": bool(self.o6_machine_manifest),
            "o7_official_grader": bool(self.o7_official_grader),
            "n_admitted": int(self.n_admitted),
            "meets_min_slice": bool(self.meets_min_slice),
            "identity_admissible": bool(self.identity_admissible),
            "grader_admissible": bool(self.grader_admissible),
            "pilot_admissible": bool(self.pilot_admissible),
            "missing_artifact": str(self.missing_artifact),
            "reason": str(self.reason),
        }


_MISSING_GRADER_ARTIFACT_W118: str = (
    "A reproducible, OFFICIAL, machine-checkable per-problem EXECUTABLE TEST SUITE "
    "(hidden judge tests) for >= 30 of the post-v6 functional problems, published "
    "through a clean official surface of a source LCB names. Verified ABSENT "
    "family-wide (2026-05-30): the Codeforces API exposes problem metadata only (no "
    "test field/endpoint); LeetCode hidden tests are deliberately private (not even "
    "premium-accessible); AtCoder publishes system tests only via a Dropbox shared "
    "folder (no official API, automation discouraged). Sample-only grading is "
    "non-credible (hidden-test false-pass) and operator-synthesised tests are "
    "operator curation (refused by O5/B2). This is the exact, named W118 blocker.")


def assess_frontier_functional_admissibility_v1(
        manifest: FrontierManifestV1,
        source: OfficialSourceV1,
        *,
        rule: FrontierFunctionalInstrumentRuleV1 = W118_FRONTIER_RULE,
) -> FrontierFunctionalAdmissibilityV1:
    """Apply the O1..O7 rule (RUNBOOK § 3) to one constructed manifest + source.

    ``identity_admissible`` ⟺ O1..O6 ∧ (n_admitted >= min_slice).  ``pilot_admissible``
    (the thing that can earn a pilot) ⟺ identity_admissible ∧ O7 (an official
    executable grader exists).  On the W118 live pass O1..O6 hold with 903 admitted but
    O7 fails ⇒ identity-admissible, NOT pilot-admissible; the missing artifact is named.
    """
    o1 = bool(
        source.source_kind in {SOURCE_CODEFORCES_API, SOURCE_ATCODER, SOURCE_LEETCODE}
        and source.has_problem_metadata_api and source.has_official_dates)
    o2 = bool(manifest.date_min and manifest.date_max)
    o3 = bool(
        manifest.date_min
        and normalize_contest_date_v1(manifest.date_min) is not None
        and manifest.date_min > str(manifest.frontier_date))
    o4 = True   # the manifest admits only PROGRAMMING-type problems (O4 by construction)
    o5 = True   # the inclusion rule is total + deterministic (O5 by construction)
    o6 = bool(manifest.raw_fetch_sha256 and manifest.manifest_cid())
    o7 = bool(source.has_official_executable_test_suite)

    n_admitted = int(manifest.n_admitted)
    meets_min = bool(n_admitted >= int(rule.min_slice))
    identity = bool(o1 and o2 and o3 and o4 and o5 and o6 and meets_min)
    pilot = bool(identity and o7)
    missing = ("NONE — the instrument is pilot-admissible." if pilot
               else _MISSING_GRADER_ARTIFACT_W118 if (identity and not o7)
               else "Identity tier not met (see O1..O6 / min_slice).")

    if pilot:
        reason = (
            f"PILOT_ADMISSIBLE: {n_admitted} official post-v6 functional problems "
            "WITH an official executable grader => build slice + run pilot.")
    elif identity and not o7:
        reason = (
            f"IDENTITY_ADMISSIBLE_BUT_GRADER_ABSENT: {n_admitted} official post-v6 "
            f"functional problems ({manifest.date_min}..{manifest.date_max}) satisfy "
            "O1..O6 (>= min_slice) — the IDENTITY tier is solved — but O7 fails: no "
            "official executable grader exists for this source family. NOT "
            "pilot-runnable. Missing artifact named (the load-bearing W118 blocker).")
    elif not meets_min:
        reason = (
            f"NOT_IDENTITY_ADMISSIBLE [< min_slice]: only {n_admitted} admitted "
            f"(< {rule.min_slice}).")
    else:
        failed = [
            name for name, ok in (
                ("O1_official", o1), ("O2_dated", o2), ("O3_post_v6", o3),
                ("O4_functional", o4), ("O6_manifest", o6))
            if not ok]
        reason = f"NOT_IDENTITY_ADMISSIBLE [{','.join(failed)} fail]."

    return FrontierFunctionalAdmissibilityV1(
        instrument_id=manifest.instrument_id,
        o1_official_source=o1, o2_dated=o2, o3_post_v6=o3, o4_functional=o4,
        o5_deterministic_no_curation=o5, o6_machine_manifest=o6,
        o7_official_grader=o7, n_admitted=n_admitted, meets_min_slice=meets_min,
        identity_admissible=identity, grader_admissible=o7, pilot_admissible=pilot,
        missing_artifact=missing, reason=reason)


# ============================================ per-model certification on the manifest

@dataclasses.dataclass(frozen=True)
class FrontierModelCertificationV1:
    """Per-model certification on the CoordPy-owned manifest: the IDENTITY-tier C1..C4
    gate (reused ``certify_model_v1``) PLUS the O7 grader gate.

    ``identity_certifiable`` is C1∧C2∧C3∧C4 on the manifest's date histogram (Maverick
    flips to certifiable here: KNOWN cutoff + 903 resistant + a genuinely-new instrument
    it never ran).  ``pilot_admissible`` additionally requires the grader (O7) — always
    False on the W118 family, so the grader is the binding blocker.
    """

    model_id: str
    identity_certification: ModelCertificationV1
    identity_certifiable: bool
    grader_admissible: bool
    pilot_admissible: bool
    blocker: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "identity_certification": self.identity_certification.to_dict(),
            "identity_certifiable": bool(self.identity_certifiable),
            "grader_admissible": bool(self.grader_admissible),
            "pilot_admissible": bool(self.pilot_admissible),
            "blocker": str(self.blocker),
        }


def _candidate_on_new_instrument(
        cand: StrongerModelCandidateV1) -> StrongerModelCandidateV1:
    """C4 is instrument-specific: the CoordPy-owned manifest is a GENUINELY NEW
    instrument none of the candidates has run, so ``already_settled_on_instrument``
    resets to False (Maverick's W113 settlement was on release_v6, a different slice)."""
    return dataclasses.replace(
        cand, already_settled_on_instrument=False,
        settled_note=(cand.settled_note
                      + " | reset on coordpy_frontier_functional_v1 (new instrument)"))


def certify_models_on_manifest_v1(
        manifest: FrontierManifestV1,
        grader_admissible: bool,
        *,
        candidates: Sequence[StrongerModelCandidateV1] = STRONGER_MODEL_CANDIDATES,
) -> tuple[FrontierModelCertificationV1, ...]:
    """Run the reused C1..C4 gate on the manifest + layer the O7 grader gate.

    For each candidate: identity-certify against the manifest's histogram (C4 reset —
    new instrument); then ``pilot_admissible`` = identity_certifiable ∧ grader_admissible.
    """
    instrument = manifest.as_resistant_instrument()
    out: list[FrontierModelCertificationV1] = []
    for cand in candidates:
        cert = certify_model_v1(
            _candidate_on_new_instrument(cand), instrument=instrument)
        idc = bool(cert.certifiable_for_new_pilot)
        pilot = bool(idc and grader_admissible)
        if pilot:
            blocker = "NONE (pilot-admissible)."
        elif idc and not grader_admissible:
            blocker = (
                "GRADER_BLOCKED (O7): identity-certifiable on this new instrument "
                "(KNOWN cutoff + >= 30 resistant problems + reachable/stronger + not "
                "settled here) but no official executable grader exists => no pilot.")
        else:
            blocker = f"CERT_BLOCKED: {cert.reason}"
        out.append(FrontierModelCertificationV1(
            model_id=cand.model_id, identity_certification=cert,
            identity_certifiable=idc, grader_admissible=bool(grader_admissible),
            pilot_admissible=pilot, blocker=blocker))
    return tuple(out)


# ===================================================== W118 disclosure matrix (Lane β)

# The W118 LIVE primary-source disclosure matrix (2026-05-30), re-verified DEEPER than
# W117 and EXTENDED with the newly-noted GLM-5.  Documentation/audit state; does NOT
# feed the certification CID (which re-derives byte-identically to W114..W117 via the
# reused W117 chain = 258b6ed7).
W118_DISCLOSURE_MATRIX: tuple[DisclosureStatusV1, ...] = (
    DisclosureStatusV1(
        model_id="meta/llama-4-maverick-17b-128e-instruct",
        primary_status=DISCLOSURE_KNOWN,
        primary_source="Official Meta llama-models llama4/MODEL_CARD.md "
                       "(re-fetched 2026-05-30): 'Knowledge cutoff: August 2024' + "
                       "'The pretraining data has a cutoff of August 2024' VERBATIM",
        aggregator_signal="Aug-2024 (corroborated)",
        stronger_than_70b=True,
        certifiable_blocker="C4 on release_v6 (settled, W113); on "
                            "coordpy_frontier_functional_v1 it is identity-"
                            "CERTIFIABLE but O7-grader-BLOCKED",
        note="KNOWN Aug-2024, re-confirmed verbatim. On the CoordPy-owned manifest "
             "ALL 903 problems are resistant for Maverick and it never ran this "
             "slice => identity-certifiable; the ONLY blocker is the missing official "
             "grader (O7)."),
    DisclosureStatusV1(
        model_id="qwen/qwen3-coder-480b-a35b-instruct",
        primary_status=DISCLOSURE_UNKNOWN,
        primary_source="Official HF card raw README "
                       "(Qwen/Qwen3-Coder-480B-A35B-Instruct), re-fetched "
                       "2026-05-30: NO CUTOFF STATED",
        aggregator_signal="(none usable; tech report 2025)",
        stronger_than_70b=True,
        certifiable_blocker="C1 (UNKNOWN cutoff) — even on the post-v6 manifest, no "
                            "KNOWN boundary to certify resistance against",
        note="Reconfirmed UNKNOWN from the official card raw markdown."),
    DisclosureStatusV1(
        model_id="deepseek-ai/deepseek-v4-pro",
        primary_status=DISCLOSURE_UNKNOWN,
        primary_source="Official DeepSeek V4 model-card PDF "
                       "(fe-static.deepseek.com/.../deepseek-V4-model-card-EN.pdf), "
                       "re-fetched DIRECTLY 2026-05-30: NO CUTOFF STATED",
        aggregator_signal="non-primary blogs 'Apr 2026' (C2-exposed; post-frontier)",
        stronger_than_70b=True,
        certifiable_blocker="C1 (UNKNOWN from primary); aggregator figure post-dates "
                            "the frontier => C2-exposed",
        note="DEEPER pass: the official V4 PDF re-fetched directly still states no "
             "cutoff; only a non-primary aggregator figure exists (C2-exposed)."),
    DisclosureStatusV1(
        model_id="mistralai/mistral-small-4-119b-2603",
        primary_status=DISCLOSURE_UNKNOWN,
        primary_source="Official Mistral docs models_overview (re-fetched "
                       "2026-05-30): lists 'Mistral Small 4' v26.03, NO CUTOFF STATED",
        aggregator_signal="OpenRouter '2025-06' (non-primary; post-frontier => "
                          "C2-exposed)",
        stronger_than_70b=True,
        certifiable_blocker="C1 (UNKNOWN from primary); aggregator estimate "
                            "post-dates the frontier => C2-exposed",
        note="Re-confirmed REAL (v26.03) with primary docs disclosing NO cutoff."),
    DisclosureStatusV1(
        model_id="zai-org/glm-5",
        primary_status=DISCLOSURE_UNKNOWN,
        primary_source="Z.ai GLM-5 docs / GitHub (zai-org/GLM-5), surfaced "
                       "2026-05-30: NO PRIMARY CUTOFF STATED",
        aggregator_signal="listicle 'training ~Feb 2026' (non-primary; ~10 months "
                          "past the Apr-2025 frontier => C2-exposed)",
        stronger_than_70b=True,
        certifiable_blocker="C1 (UNKNOWN from primary) + C2 (aggregator date "
                            "post-frontier) + reachability UNVERIFIED (not in the "
                            "W112 reachable NIM catalogue)",
        note="NEWLY NOTED since W117 (a strong 2026 open coder, SWE-bench 77.8%). "
             "But its only date is a non-primary post-frontier figure and "
             "reachability is unverified => NOT a primary-KNOWN <=2025-01 cutoff; "
             "does not move the verdict."),
)


def disclosure_delta_since_w117_v1(
        matrix: Sequence[DisclosureStatusV1] = W118_DISCLOSURE_MATRIX,
) -> dict[str, Any]:
    """Lane β summary: counts + whether anything is NEWLY DISCLOSED with a usable
    primary-KNOWN cutoff since W117 (none — GLM-5 is newly NOTED but C2-exposed +
    reachability-unverified, status UNKNOWN, so it does not move the verdict)."""
    base = disclosure_matrix_summary_v1(matrix)
    newly = [
        d.model_id for d in matrix
        if d.primary_status == DISCLOSURE_NEWLY_DISCLOSED]
    base["newly_disclosed_since_w117"] = newly
    base["any_newly_disclosed_since_w117"] = bool(newly)
    base["newly_noted_uncertifiable"] = [
        d.model_id for d in matrix if "NEWLY NOTED" in d.note]
    return base


# ============================================================ W119 fire condition

@dataclasses.dataclass(frozen=True)
class W119FireConditionV1:
    """The pre-committed triggers that flip the no-go verdict (RUNBOOK § 9)."""

    grader_artifact_trigger: str
    packaged_or_construction_trigger: str
    cutoff_trigger: str
    grader_artifact_trigger_met: bool
    packaged_or_construction_trigger_met: bool
    cutoff_trigger_met: bool
    fires_now: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "grader_artifact_trigger": str(self.grader_artifact_trigger),
            "packaged_or_construction_trigger": str(
                self.packaged_or_construction_trigger),
            "cutoff_trigger": str(self.cutoff_trigger),
            "grader_artifact_trigger_met": bool(self.grader_artifact_trigger_met),
            "packaged_or_construction_trigger_met": bool(
                self.packaged_or_construction_trigger_met),
            "cutoff_trigger_met": bool(self.cutoff_trigger_met),
            "fires_now": bool(self.fires_now),
        }


# ================================================================ thin live fetch

class CodeforcesFetchError(RuntimeError):
    """Raised when the official Codeforces API cannot be reached / returns non-OK."""


def fetch_codeforces_official_v1(
        *,
        timeout: float = 45.0,
        contest_list_url: str = CODEFORCES_CONTEST_LIST_URL,
        problemset_url: str = CODEFORCES_PROBLEMSET_URL,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """LIVE-fetch the official Codeforces ``contest.list`` + ``problemset.problems``.

    Returns ``(contest_list_payload, problemset_payload, raw_fetch_sha256)`` where the
    SHA pins the concatenated raw response bytes (provenance).  The ONLY network I/O in
    this module; the pure builder takes its output.  Raises ``CodeforcesFetchError`` on
    failure so the caller can fall back to a pinned snapshot.
    """
    import urllib.request    # local import: keep the module import-time pure/offline

    raw_parts: list[bytes] = []
    payloads: list[dict[str, Any]] = []
    for url in (contest_list_url, problemset_url):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                raw = resp.read()
        except Exception as e:  # noqa: BLE001
            raise CodeforcesFetchError(f"fetch failed for {url}: {e}") from e
        raw_parts.append(raw)
        try:
            doc = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001
            raise CodeforcesFetchError(f"non-JSON from {url}: {e}") from e
        if str(doc.get("status")) != "OK":
            raise CodeforcesFetchError(
                f"API status != OK from {url}: {doc.get('comment')}")
        payloads.append(doc)
    sha = hashlib.sha256(b"".join(raw_parts)).hexdigest()
    return payloads[0], payloads[1], sha


# ============================================================ the push-button API

@dataclasses.dataclass(frozen=True)
class FrontierFunctionalConstructionResultV1:
    schema: str
    verified_on: str
    instrument_rule: dict[str, Any]
    source_family: tuple[OfficialSourceV1, ...]
    source_family_grader_summary: dict[str, Any]
    manifest: FrontierManifestV1
    admissibility: FrontierFunctionalAdmissibilityV1
    per_model: tuple[FrontierModelCertificationV1, ...]
    disclosure_matrix: tuple[DisclosureStatusV1, ...]
    disclosure_summary: dict[str, Any]
    lcb_inherited: UpstreamConstructionResultV1
    verdict: str
    pilot_earned: bool
    n_identity_certifiable_models: int
    w119_fire_condition: W119FireConditionV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": str(self.schema),
            "verified_on": str(self.verified_on),
            "instrument_rule": self.instrument_rule,
            "source_family": [s.to_dict() for s in self.source_family],
            "source_family_grader_summary": self.source_family_grader_summary,
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
            "n_identity_certifiable_models": int(
                self.n_identity_certifiable_models),
            "w119_fire_condition": self.w119_fire_condition.to_dict(),
        }

    def cid(self) -> str:
        return _sha256_hex({"kind": "w118_frontier_functional_construction_result_v1",
                            "result": self.to_dict()})


def run_frontier_functional_construction_v1(
        contest_list_payload: dict[str, Any],
        problemset_payload: dict[str, Any],
        *,
        verified_on: str,
        raw_fetch_sha256: str = "",
        frontier_date: str = FRONTIER_DATE,
        rule: FrontierFunctionalInstrumentRuleV1 = W118_FRONTIER_RULE,
        source: OfficialSourceV1 = OFFICIAL_SOURCE_FAMILY[0],
        family: Sequence[OfficialSourceV1] = OFFICIAL_SOURCE_FAMILY,
        candidates: Sequence[StrongerModelCandidateV1] = STRONGER_MODEL_CANDIDATES,
) -> FrontierFunctionalConstructionResultV1:
    """The push-button W118 construction + admission + certification (RUNBOOK § 4).

    1. Construct the manifest from the official Codeforces payloads (deterministic).
    2. Assess O1..O7 (identity tier vs grader tier).
    3. Run the reused C1..C4 gate on the manifest + the O7 grader gate per model.
    4. REUSE ``run_upstream_construction_v1`` for the LCB-inherited verdict + decision
       CID (258b6ed7 invariant; W118 does not re-litigate the LCB side).
    5. Build the Lane β disclosure summary + the structured W119 fire condition.

    Verdict stays ``NO_CERTIFIABLE_STRONGER_MODEL`` and ``pilot_earned`` False unless a
    model is BOTH identity-certifiable AND grader-admissible (none is on the W118 live
    pass: the grader is absent family-wide).  Re-running this on updated inputs (a new
    official grader artifact / a newly primary-KNOWN cutoff) is the W119 operation.
    """
    manifest = build_frontier_manifest_from_codeforces_v1(
        contest_list_payload, problemset_payload,
        frontier_date=frontier_date, fetched_on=verified_on,
        raw_fetch_sha256=raw_fetch_sha256)
    adm = assess_frontier_functional_admissibility_v1(manifest, source, rule=rule)
    per_model = certify_models_on_manifest_v1(
        manifest, adm.grader_admissible, candidates=candidates)
    grader_summary = source_family_grader_summary_v1(family)
    disclosure_summary = disclosure_delta_since_w117_v1(W118_DISCLOSURE_MATRIX)

    # The LCB-inherited verdict + decision CID (reused unchanged => 258b6ed7).
    lcb = run_upstream_construction_v1()

    n_identity = sum(1 for m in per_model if m.identity_certifiable)
    pilot_targets = [m for m in per_model if m.pilot_admissible]
    pilot_earned = bool(pilot_targets)
    # Any pilot-admissible target flips the verdict: a stronger-than-Maverick target
    # would be a clean reopening, and Maverick alone (identity-certifiable on a
    # genuinely-new instrument it never ran) would be a verdict-changing run. On the
    # W118 live pass none is pilot-admissible (grader absent family-wide).
    verdict = VERDICT_CERTIFIABLE if pilot_earned else VERDICT_NONE

    grader_met = bool(adm.grader_admissible)
    packaged_or_constr_met = bool(
        lcb.w118_fire_condition.packaged_release_trigger_met
        or lcb.w118_fire_condition.construction_provenance_trigger_met)
    cutoff_met = bool(disclosure_summary.get("any_newly_disclosed_since_w117"))
    fires_now = bool(pilot_earned)

    fire = W119FireConditionV1(
        grader_artifact_trigger=(
            "An OFFICIAL, reproducible, machine-checkable executable per-problem test "
            f"suite for >= {rule.min_slice} of the post-v6 functional problems appears "
            "on a clean official surface of a source LCB names (Codeforces / AtCoder / "
            "LeetCode) — making coordpy_frontier_functional_v1 pilot-admissible. "
            "Maverick is ALREADY identity-certifiable on it, so a grader alone unlocks "
            "the cheapest honest verdict-changing pilot. Missing artifact: "
            + _MISSING_GRADER_ARTIFACT_W118),
        packaged_or_construction_trigger=(
            "A packaged LCB release_v7+ is admitted to the loader, OR an LCB-PUBLISHED "
            "post-v6 construction provenance appears (the inherited W118 triggers; "
            "re-run run_upstream_construction_v1)."),
        cutoff_trigger=(
            "A reachable stronger-than-Maverick model discloses, from a PRIMARY "
            f"source, a KNOWN cutoff <= the manifest frontier {manifest.date_min} (so "
            "it gets a >= 30 resistant slice on coordpy_frontier_functional_v1); "
            "update MODEL_TRAINING_CUTOFFS + W114_CUTOFF_PROVENANCE, then re-run. "
            "Combined with a grader, run it."),
        grader_artifact_trigger_met=grader_met,
        packaged_or_construction_trigger_met=packaged_or_constr_met,
        cutoff_trigger_met=cutoff_met,
        fires_now=fires_now)

    return FrontierFunctionalConstructionResultV1(
        schema=W118_FRONTIER_FUNCTIONAL_V1_SCHEMA_VERSION,
        verified_on=str(verified_on),
        instrument_rule=rule.to_dict(),
        source_family=tuple(family),
        source_family_grader_summary=grader_summary,
        manifest=manifest,
        admissibility=adm,
        per_model=per_model,
        disclosure_matrix=W118_DISCLOSURE_MATRIX,
        disclosure_summary=disclosure_summary,
        lcb_inherited=lcb,
        verdict=verdict,
        pilot_earned=pilot_earned,
        n_identity_certifiable_models=int(n_identity),
        w119_fire_condition=fire)


__all__ = [
    "W118_FRONTIER_FUNCTIONAL_V1_SCHEMA_VERSION",
    "COORDPY_FRONTIER_FUNCTIONAL_V1",
    "FRONTIER_DATE",
    "CODEFORCES_CONTEST_LIST_URL",
    "CODEFORCES_PROBLEMSET_URL",
    "FrontierFunctionalInstrumentRuleV1",
    "W118_FRONTIER_RULE",
    "TEST_ARTIFACT_CLEAN_API",
    "TEST_ARTIFACT_DROPBOX_NON_API",
    "TEST_ARTIFACT_NONE",
    "SOURCE_CODEFORCES_API",
    "SOURCE_ATCODER",
    "SOURCE_LEETCODE",
    "OfficialSourceV1",
    "OFFICIAL_SOURCE_FAMILY",
    "source_family_grader_summary_v1",
    "FrontierProblemV1",
    "FrontierManifestV1",
    "build_frontier_manifest_from_codeforces_v1",
    "FrontierFunctionalAdmissibilityV1",
    "assess_frontier_functional_admissibility_v1",
    "FrontierModelCertificationV1",
    "certify_models_on_manifest_v1",
    "W118_DISCLOSURE_MATRIX",
    "disclosure_delta_since_w117_v1",
    "W119FireConditionV1",
    "CodeforcesFetchError",
    "fetch_codeforces_official_v1",
    "FrontierFunctionalConstructionResultV1",
    "run_frontier_functional_construction_v1",
    "VERDICT_CERTIFIABLE",
    "VERDICT_NONE",
]
