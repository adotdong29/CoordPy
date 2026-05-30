#!/usr/bin/env python3
"""W118-α/β/γ — CoordPy-OWNED post-v6 functional-instrument construction.

NIM-free (no model inference; the ONLY network is the official Codeforces API, a
read-only public metadata fetch).  Runs the LOCKED W118 constructor
(``coordpy.coordpy_frontier_functional_v1``):

* LIVE-fetches the official Codeforces API (``contest.list`` + ``problemset.problems``)
  — or loads a pinned snapshot when ``COORDPY_CF_SNAPSHOT`` is set / the network is
  unavailable — and builds the CoordPy-owned post-v6 functional-IDENTITY manifest
  deterministically (PROGRAMMING-type ∧ FINISHED ∧ contest date strictly after the
  release_v6 frontier 2025-04-05), pinning the raw-fetch SHA + a manifest CID + a date
  histogram;
* applies the pre-committed O1..O7 instrument rule (identity tier O1..O6 vs grader tier
  O7) + the official-source-family grader registry (Codeforces / AtCoder / LeetCode);
* runs the reused W114 C1..C4 certification gate on the manifest + the O7 grader gate
  per model (Maverick is identity-CERTIFIABLE on this genuinely-new instrument, but
  GRADER-BLOCKED);
* reuses the W117 ``run_upstream_construction_v1`` for the LCB-inherited verdict +
  decision CID (258b6ed7, byte-identical to W114/W115/W116/W117);
* emits the sharpened W118 disclosure matrix (Lane β; incl. the newly-noted GLM-5) +
  the structured W119 fire condition.

It asserts the decision-CID byte-identical invariant and the "no pilot earned" /
"identity-admissible-but-grader-absent" outcome.  $0 NIM.

Usage::

    python scripts/run_w118_frontier_functional_construction_v1.py
    COORDPY_CF_SNAPSHOT=/path/to/snapshot.json python scripts/run_w118_...py  # offline
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.coordpy_frontier_functional_v1 import (  # noqa: E402
    VERDICT_NONE,
    CodeforcesFetchError,
    build_frontier_manifest_from_codeforces_v1,
    fetch_codeforces_official_v1,
    run_frontier_functional_construction_v1,
)

OUT = ROOT / "results" / "w118" / "frontier_functional"
_CACHE = Path(os.path.expanduser("~/.cache/coordpy"))


def _load_payloads() -> tuple[dict, dict, str, str]:
    """Return ``(contest_list, problemset, raw_sha256, provenance_note)``.

    Prefers a pinned ``COORDPY_CF_SNAPSHOT`` JSON (``{"contest_list":..,
    "problemset":..}``) for offline reproducibility; otherwise LIVE-fetches the official
    Codeforces API and best-effort caches the raw snapshot under ``~/.cache/coordpy``.
    """
    snap = os.environ.get("COORDPY_CF_SNAPSHOT", "")
    if snap and os.path.exists(snap):
        doc = json.loads(Path(snap).read_text())
        import hashlib
        sha = hashlib.sha256(
            json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return (doc["contest_list"], doc["problemset"], sha,
                f"pinned snapshot {snap}")
    cl, ps, sha = fetch_codeforces_official_v1()
    try:
        _CACHE.mkdir(parents=True, exist_ok=True)
        (_CACHE / "cf_snapshot_w118.json").write_text(
            json.dumps({"contest_list": cl, "problemset": ps}))
    except Exception:  # noqa: BLE001
        pass
    return cl, ps, sha, "live official Codeforces API (contest.list + problemset.problems)"


def main() -> int:
    verified_on = "2026-05-30"
    try:
        contest_list, problemset, raw_sha, provenance = _load_payloads()
    except CodeforcesFetchError as e:
        print(f"!! official-source fetch failed and no pinned snapshot: {e}")
        print("   (set COORDPY_CF_SNAPSHOT to a pinned snapshot to run offline)")
        return 3

    result = run_frontier_functional_construction_v1(
        contest_list, problemset, verified_on=verified_on,
        raw_fetch_sha256=raw_sha)
    manifest = result.manifest
    adm = result.admissibility
    lcb = result.lcb_inherited
    decision_cid = (
        lcb.upstream_admission.frontier_certification.decision.cid())

    artifact = {
        "schema": "coordpy.w118_frontier_functional_construction.v1",
        "milestone": "W118-alpha-beta-gamma",
        "official_source_provenance": provenance,
        "result": result.to_dict(),
        "result_cid": result.cid(),
        "lcb_inherited_decision_cid": decision_cid,
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frontier_functional_verdict.json").write_text(
        json.dumps(artifact, indent=2, default=str))
    # The full machine-generated manifest (problem ids + histogram + provenance SHAs).
    (OUT / "coordpy_frontier_functional_v1_manifest.json").write_text(
        json.dumps({"manifest": manifest.to_dict(),
                    "manifest_cid": manifest.manifest_cid()},
                   indent=2, default=str))

    print("=== W118 CoordPy-OWNED post-v6 functional-instrument construction ===")
    print(f"  verified_on: {verified_on}")
    print(f"  official source provenance: {provenance}")
    print(f"  raw_fetch_sha256: {manifest.raw_fetch_sha256[:16]}...")
    print("  [Lane α — manifest construction (official Codeforces API)]")
    print(f"    instrument_id          = {manifest.instrument_id}  (NOT 'LCB v7')")
    print(f"    candidates_seen        = {manifest.n_candidates_seen}")
    print(f"    ADMITTED (post-v6 fn)  = {manifest.n_admitted}  "
          f"(>= MIN_SLICE 30: {manifest.n_admitted >= 30})")
    print(f"    excluded not_programming = {manifest.n_excluded_not_programming}; "
          f"not_finished = {manifest.n_excluded_not_finished}; "
          f"missing_date = {manifest.n_excluded_missing_date}; "
          f"not_after_frontier = {manifest.n_excluded_not_after_frontier}")
    print(f"    date range             = {manifest.date_min} .. {manifest.date_max} "
          f"({manifest.n_contests} contests)")
    print(f"    month histogram        = {manifest.month_histogram}")
    print(f"    manifest_cid           = {manifest.manifest_cid()[:16]}...")
    print("  [O1..O7 admissibility]")
    print(f"    O1 official={int(adm.o1_official_source)} O2 dated={int(adm.o2_dated)} "
          f"O3 post_v6={int(adm.o3_post_v6)} O4 functional={int(adm.o4_functional)} "
          f"O5 deterministic={int(adm.o5_deterministic_no_curation)} "
          f"O6 manifest={int(adm.o6_machine_manifest)} "
          f"O7 grader={int(adm.o7_official_grader)}")
    print(f"    identity_admissible = {adm.identity_admissible}; "
          f"grader_admissible = {adm.grader_admissible}; "
          f"pilot_admissible = {adm.pilot_admissible}")
    print(f"    :: {adm.reason}")
    print("  [official-source-family grader registry (O7 family-wide)]")
    for s in result.source_family:
        print(f"    {s.source_kind:24s} identity_api={int(s.has_problem_metadata_api)} "
              f"official_grader={int(s.has_official_executable_test_suite)} "
              f"[{s.test_artifact_status}]")
    gs = result.source_family_grader_summary
    print(f"    any_source_has_official_grader = "
          f"{gs['any_source_has_official_grader']}; "
          f"any_source_has_clean_identity_api = "
          f"{gs['any_source_has_clean_identity_api']}")
    print("  [per-model certification on the manifest "
          "(C1..C4 identity + O7 grader gate)]")
    for m in result.per_model:
        c = m.identity_certification
        print(f"    {m.model_id:46s} n_res={c.n_functional_resistant:4d} "
              f"id_certifiable={int(m.identity_certifiable)} "
              f"grader={int(m.grader_admissible)} "
              f"pilot_admissible={int(m.pilot_admissible)}")
        print(f"        :: {m.blocker}")
    print("  [Lane β — disclosure matrix (primary-source, DEEPER pass)]")
    for d in result.disclosure_matrix:
        print(f"    {d.model_id:46s} {d.primary_status:9s} "
              f">70B={int(d.stronger_than_70b)} blocker={d.certifiable_blocker[:60]}")
    ds = result.disclosure_summary
    print(f"    disclosure counts = {ds['counts']}; "
          f"newly_disclosed_since_w117 = {ds['any_newly_disclosed_since_w117']}; "
          f"newly_noted_uncertifiable = {ds.get('newly_noted_uncertifiable')}")
    print(f"  VERDICT: {result.verdict}; pilot_earned = {result.pilot_earned}; "
          f"identity_certifiable_models = {result.n_identity_certifiable_models}")
    print(f"  LCB-inherited verdict: {lcb.verdict}; "
          f"decision_cid = {decision_cid}")
    print(f"  result CID: {result.cid()}")
    w119 = result.w119_fire_condition
    print("  [W119 fire condition]")
    print(f"    fires_now = {w119.fires_now} "
          f"(grader_met={w119.grader_artifact_trigger_met}; "
          f"packaged/construction_met={w119.packaged_or_construction_trigger_met}; "
          f"cutoff_met={w119.cutoff_trigger_met})")
    if result.verdict == VERDICT_NONE:
        print("    grader trigger:       " + w119.grader_artifact_trigger)
        print("    packaged/constr trig: " + w119.packaged_or_construction_trigger)
        print("    cutoff trigger:       " + w119.cutoff_trigger)
        print("  W118 MISSING OFFICIAL ARTIFACT (load-bearing):")
        print(f"    {adm.missing_artifact}")
    print(f"  artifact: {OUT / 'frontier_functional_verdict.json'}")
    print(f"  manifest: {OUT / 'coordpy_frontier_functional_v1_manifest.json'}")

    # Exit non-zero ONLY on an integrity failure — the LCB-inherited decision CID must
    # stay byte-identical to the W114/W115/W116/W117 invariant 258b6ed7, the identity
    # tier must actually be solved (>= 30 admitted), and the grader must be absent (the
    # expected family-wide blocker). A no-go verdict is a valid, expected outcome.
    expected_decision_cid_prefix = "258b6ed7"
    cid_ok = decision_cid.startswith(expected_decision_cid_prefix)
    identity_ok = manifest.n_admitted >= 30 and adm.identity_admissible
    expected_no_grader = (not adm.grader_admissible)  # the W118 live truth
    integrity_ok = bool(cid_ok and identity_ok and expected_no_grader
                        and result.verdict == VERDICT_NONE)
    if not cid_ok:
        print(f"  !! decision CID drift: expected prefix "
              f"{expected_decision_cid_prefix}, got {decision_cid}")
    return 0 if integrity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
