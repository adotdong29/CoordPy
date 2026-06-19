#!/usr/bin/env python3
"""W119-α/β/γ — official ICPC public-package post-cutoff functional construction.

NIM-free (no model inference; the ONLY network is the official ICPC GitHub org package
listing via ``gh``, a read-only metadata fetch + an optional NIM-free grader self-test
that runs OFFICIAL accepted reference solutions, never a model).  Runs the LOCKED W119
constructor (``coordpy.coordpy_icpc_public_functional_v1``):

* (optionally) LIVE-fetches the official ICPC package listings (or uses the pinned
  ``ICPC_PACKAGE_LISTING_SNAPSHOT_V1``) and builds the CoordPy-owned post-cutoff
  functional manifest deterministically (post-Maverick-cutoff resistant ∧ NOT
  interactive ∧ ships a usable grader), pinning a manifest CID + date histogram;
* applies the pre-committed P1..P8 instrument rule (identity P1..P6 vs grader P7∧P8 vs
  the >=30 slice count) + the official ICPC source-family grader registry;
* runs the reused W114 C1..C4 certification gate on the manifest + the grader + slice
  gates per model (Maverick is identity-CERTIFIABLE on this genuinely-new grader-clean
  instrument, but SLICE-SHORT at 24 < 30);
* reuses the W117 ``run_upstream_construction_v1`` for the LCB-inherited verdict +
  decision CID (258b6ed7, byte-identical to W114..W118);
* emits the reused W118 disclosure matrix (Lane β) + the structured W120 fire condition.

It asserts the decision-CID byte-identical invariant and the "no pilot earned /
grader-admissible-but-slice-short" outcome.  $0 NIM.

Usage::

    python scripts/run_w119_icpc_public_construction_v1.py
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.coordpy_icpc_public_functional_v1 import (  # noqa: E402
    VERDICT_NONE,
    run_icpc_public_construction_v1,
)

OUT = ROOT / "results" / "w119" / "icpc_public"


def main() -> int:
    verified_on = "2026-05-30"
    result = run_icpc_public_construction_v1(verified_on=verified_on)
    manifest = result.manifest
    adm = result.admissibility
    lcb = result.lcb_inherited
    decision_cid = (
        lcb.upstream_admission.frontier_certification.decision.cid())

    artifact = {
        "schema": "coordpy.w119_icpc_public_construction.v1",
        "milestone": "W119-alpha-beta-gamma",
        "result": result.to_dict(),
        "result_cid": result.cid(),
        "lcb_inherited_decision_cid": decision_cid,
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "icpc_public_verdict.json").write_text(
        json.dumps(artifact, indent=2, default=str))
    (OUT / "coordpy_icpc_public_functional_v1_manifest.json").write_text(
        json.dumps({"manifest": manifest.to_dict(),
                    "manifest_cid": manifest.manifest_cid()},
                   indent=2, default=str))

    print("=== W119 official ICPC public-package post-cutoff construction ===")
    print(f"  verified_on: {verified_on}")
    print("  [Lane α — manifest construction (official ICPC GitHub org)]")
    print(f"    instrument_id  = {manifest.instrument_id}  (NOT 'LCB v7', NOT W118 CF)")
    print(f"    candidates_seen= {manifest.n_candidates_seen}")
    print(f"    ADMITTED       = {manifest.n_admitted}  "
          f"(>= MIN_SLICE 30: {manifest.n_admitted >= 30})")
    print(f"    excluded pre_cutoff = {manifest.n_excluded_pre_cutoff}; "
          f"interactive = {manifest.n_excluded_interactive}; "
          f"no_grader = {manifest.n_excluded_no_grader}")
    print(f"    date range     = {manifest.date_min} .. {manifest.date_max} "
          f"({manifest.n_repos} official repos)")
    print(f"    month histogram= {manifest.month_histogram}")
    print(f"    with ref soln  = {manifest.n_with_reference_solution}")
    print(f"    manifest_cid   = {manifest.manifest_cid()[:16]}...")
    print("  [P1..P8 admissibility]")
    print(f"    P1 official={int(adm.p1_official_source)} P2 dated={int(adm.p2_dated)} "
          f"P3 post_cutoff={int(adm.p3_post_cutoff)} P4 fn={int(adm.p4_functional)} "
          f"P5 det={int(adm.p5_deterministic)} P6 manifest={int(adm.p6_machine_manifest)}")
    print(f"    P7 grader_present={int(adm.p7_grader_present)} "
          f"P8 grader_executable={int(adm.p8_grader_executable)}")
    print(f"    identity_admissible={adm.identity_admissible}; "
          f"grader_admissible={adm.grader_admissible}; "
          f"pilot_admissible={adm.pilot_admissible}")
    print(f"    :: {adm.reason}")
    print("  [grader self-test (NIM-free; official accepted solutions vs official "
          "secret cases)]")
    st = result.grader_selftest_summary
    print(f"    problems={st['n_problems_self_tested']} "
          f"cases={st['n_cases_passed']}/{st['n_cases_run']} "
          f"grader_proven_executable={st['grader_proven_executable']}")
    print("  [official ICPC source-family grader registry]")
    gs = result.family_grader_summary
    print(f"    any_source_has_official_grader = "
          f"{gs['any_source_has_official_grader']} "
          f"(post-cutoff repos: {gs['post_cutoff_grader_repos']})")
    print(f"    n_post_cutoff_gradeable_passfail = "
          f"{gs['n_post_cutoff_gradeable_passfail']}")
    print("  [per-model certification (C1..C4 identity + grader + slice gates)]")
    for m in result.per_model:
        print(f"    {m.model_id:46s} id_cert={int(m.identity_certifiable)} "
              f"grader={int(m.grader_admissible)} slice={int(m.slice_admissible)} "
              f"pilot={int(m.pilot_admissible)}")
        print(f"        :: {m.blocker}")
    print(f"  VERDICT: {result.verdict}; pilot_earned = {result.pilot_earned}; "
          f"identity_certifiable_models = {result.n_identity_certifiable_models}")
    print(f"  LCB-inherited verdict: {lcb.verdict}; decision_cid = {decision_cid}")
    print(f"  result CID: {result.cid()}")
    w120 = result.w120_fire_condition
    print("  [W120 fire condition]")
    print(f"    fires_now = {w120.fires_now} "
          f"(slice_met={w120.slice_trigger_met}; cutoff_met={w120.cutoff_trigger_met})")
    if result.verdict == VERDICT_NONE:
        print("    slice trigger:  " + w120.slice_trigger)
        print("    cutoff trigger: " + w120.cutoff_trigger)
        print("    nim trigger:    " + w120.nim_trigger)
        print("  W119 MISSING ARTIFACT (load-bearing):")
        print(f"    {adm.missing_artifact}")
    print(f"  artifact: {OUT / 'icpc_public_verdict.json'}")

    # Exit non-zero ONLY on an integrity failure — the LCB-inherited decision CID must
    # stay byte-identical to the W114..W118 invariant 258b6ed7; the grader MUST be
    # admissible (the W119 advance: P7∧P8 hold); and the expected outcome is
    # grader-admissible-but-slice-short (24 < 30) with no pilot.  A no-go verdict is
    # the valid, expected outcome.
    expected_decision_cid_prefix = "258b6ed7"
    cid_ok = decision_cid.startswith(expected_decision_cid_prefix)
    grader_ok = bool(adm.grader_admissible)               # the W119 dissolution
    slice_short = bool(not adm.meets_min_slice)           # the W119 live truth
    integrity_ok = bool(cid_ok and grader_ok and slice_short
                        and result.verdict == VERDICT_NONE
                        and not result.pilot_earned)
    if not cid_ok:
        print(f"  !! decision CID drift: expected prefix "
              f"{expected_decision_cid_prefix}, got {decision_cid}")
    return 0 if integrity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
