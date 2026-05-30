#!/usr/bin/env python3
"""W117-α/β/γ — durable upstream-DERIVED instrument-CONSTRUCTION pipeline.

NIM-free.  Runs the LOCKED W117 construction pipeline
(``coordpy.upstream_derived_instrument_construction_v1``) against the live-verified
``W117_PROVENANCE_SNAPSHOT`` (RUNBOOK_W117 § 2, primary-source upstream-PROVENANCE /
construction attack 2026-05-30):

* the EIGHT-surface construction-provenance view (HF commit log / refs / discussions,
  GitHub commits / tags / repo pipeline structure, README provenance, runner loader);
* the pre-committed construction rule (A1..A5 ∧ B1 ∧ B2) + the construction attempt
  (raw-contest hand-assembly REFUSED by A1 ∧ B1 ∧ B2; the LCB-published-pipeline
  template construction-admissible-in-principle but artifact-absent) => 0 realizable
  construction-admissible NEW instruments + the EXACT missing upstream artifact;
* the reused W116 packaged-admission + per-model go/no-go matrix (decision CID
  258b6ed7, byte-identical to W114/W115/W116);
* the sharpened disclosure matrix (DeepSeek V4 primary-PDF re-confirmed no-cutoff;
  Maverick 'August 2024' verbatim; nothing newly-disclosed since W116);
* the structured W118 fire condition (packaged / construction-provenance / cutoff).

It ALSO re-verifies the pinned functional month histogram against the live
SHA-pinned ``release_v6`` corpus when present (a preflight-style discharge that the
instrument the decision rests on matches real bytes) and asserts the decision-CID
byte-identical invariant.  $0 NIM.

Usage::

    python scripts/run_w117_upstream_construction_v1.py
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.upstream_derived_instrument_construction_v1 import (  # noqa: E402
    VERDICT_NONE,
    run_upstream_construction_v1,
)
from coordpy.stronger_model_cutoff_certification_v1 import (  # noqa: E402
    LATEST_RESISTANT_INSTRUMENT,
)
from coordpy.livecodebench_resistant_slice_v1 import (  # noqa: E402
    normalize_contest_date_v1,
)

OUT = ROOT / "results" / "w117" / "upstream_construction"


def _live_corpus_path() -> str:
    return os.environ.get(
        "COORDPY_LIVECODEBENCH_CACHE",
        os.path.expanduser("~/.cache/coordpy/livecodebench-test6.jsonl"))


def _reverify_histogram_against_corpus() -> dict:
    """Re-derive the functional month histogram from the live SHA-pinned corpus.

    Returns ``available`` / ``sha_ok`` / ``histogram_match`` + details.  Does NOT
    raise on a missing corpus (the pinned histogram is the authority; this is a
    best-effort discharge that real bytes agree)."""
    path = _live_corpus_path()
    inst = LATEST_RESISTANT_INSTRUMENT
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return {"available": False, "path": path,
                "note": "corpus absent; pinned histogram used as authority"}
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    sha_ok = (sha.lower() == inst.jsonl_sha256.lower())
    hist: dict[str, int] = {}
    n_func = 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if not str(row.get("starter_code") or "").strip():
            continue  # stdin/stdout = non-functional
        n_func += 1
        day = normalize_contest_date_v1(row.get("contest_date"))
        if day:
            hist[day[:7]] = hist.get(day[:7], 0) + 1
    match = (hist == dict(inst.functional_month_histogram))
    return {"available": True, "path": path, "sha_ok": sha_ok,
            "n_functional": n_func, "histogram": dict(sorted(hist.items())),
            "histogram_match": match}


def main() -> int:
    result = run_upstream_construction_v1()
    corpus_check = _reverify_histogram_against_corpus()
    adm = result.upstream_admission
    fc = adm.frontier_certification

    artifact = {
        "schema": "coordpy.w117_upstream_construction.v1",
        "milestone": "W117-alpha-beta-gamma",
        "result": result.to_dict(),
        "result_cid": result.cid(),
        "decision_cid": fc.decision.cid(),
        "admission_result_cid": adm.cid(),
        "corpus_reverification": corpus_check,
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "upstream_construction_verdict.json").write_text(
        json.dumps(artifact, indent=2, default=str))

    print("=== W117 upstream-DERIVED instrument-construction pipeline ===")
    print(f"  verified_on: {result.verified_on}")
    print("  [construction provenance — EIGHT surfaces]")
    for s in result.provenance_surfaces:
        print(f"    {s.surface_kind:28s} post_frontier_lcb_artifact="
              f"{int(s.has_post_frontier_lcb_artifact)} :: {s.finding}")
    print(f"    surfaces_with_post_frontier_artifact = "
          f"{result.n_surfaces_with_post_frontier_artifact}")
    print("  [construction rule] A1..A5 (reused) ∧ B1 authoritative-LCB-provenance ∧ "
          "B2 no-operator-curation")
    att = result.construction_attempt
    for p in att.proposal_admissibility:
        print(f"    {p.label:46s} A={int(p.a_admissible)} B1="
              f"{int(p.b1_authoritative_provenance)} B2="
              f"{int(p.b2_no_operator_curation)} "
              f"cadm={int(p.construction_admissible)} "
              f"realizable={int(p.realizable)}")
        print(f"        :: {p.reason}")
    print(f"  constructed = {att.constructed}; realizable NEW = "
          f"{att.n_construction_admissible_new}; admissible-in-principle = "
          f"{att.n_construction_admissible_in_principle}")
    print("  [per-model disclosure-status matrix] (primary-source, DEEPER pass)")
    for d in result.disclosure_matrix:
        print(f"    {d.model_id:46s} {d.primary_status:24s} "
              f">70B={int(d.stronger_than_70b)} blocker={d.certifiable_blocker}")
    ds = result.disclosure_summary
    print(f"    disclosure counts = {ds['counts']}; usable NEW KNOWN-cutoff target = "
          f"{ds['any_usable_new_known_cutoff_target']}; newly-disclosed-since-W116 = "
          f"{ds['any_newly_disclosed_since_w116']}")
    print("  [per-model certification go/no-go] "
          "(C1 known / C2 >=30 / C3 reach-stronger / C4 not-settled)")
    for m in fc.decision.per_model:
        print(f"    [{m.rank_tier}.{m.rank_within_tier}] {m.model_id:42s} "
              f"cutoff={m.cutoff_boundary}[{m.cutoff_confidence}] "
              f"n_res={m.n_functional_resistant:2d} "
              f"C1={int(m.c1_cutoff_known)} C2={int(m.c2_enough_resistant)} "
              f"C3={int(m.c3_reachable_stronger_comparable)} "
              f"C4={int(m.c4_not_already_settled)} "
              f"=> certifiable={m.certifiable_for_new_pilot}")
    print(f"  VERDICT: {result.verdict}"
          + (f" (target={result.target_model})" if result.target_model else ""))
    print(f"  disclosure_consistency_ok: {fc.disclosure_consistency_ok}")
    print(f"  result CID: {result.cid()}")
    print(f"  decision CID: {fc.decision.cid()}")
    print(f"  corpus re-verification: available="
          f"{corpus_check.get('available')} sha_ok={corpus_check.get('sha_ok')} "
          f"histogram_match={corpus_check.get('histogram_match')}")
    w118 = result.w118_fire_condition
    print("  [W118 fire condition]")
    print(f"    fires_now = {w118.fires_now} "
          f"(packaged_met={w118.packaged_release_trigger_met}; "
          f"construction_met={w118.construction_provenance_trigger_met}; "
          f"cutoff_met={w118.cutoff_trigger_met})")
    if result.verdict == VERDICT_NONE:
        print("    packaged trigger:     " + w118.packaged_release_trigger)
        print("    construction trigger: " + w118.construction_provenance_trigger)
        print("    cutoff trigger:       " + w118.cutoff_trigger)
        print("  W117 MISSING UPSTREAM ARTIFACT (load-bearing):")
        print(f"    {att.missing_artifact}")
    print(f"  artifact: {OUT / 'upstream_construction_verdict.json'}")

    # Exit non-zero ONLY on an integrity failure (corpus present but SHA/hist
    # mismatch, a live-disclosure vs encoded-registry divergence, or the decision CID
    # drifting from the W114/W115/W116 byte-identical 258b6ed7 invariant) — NOT on a
    # no-go verdict (a no-go is a valid, expected outcome).
    expected_decision_cid_prefix = "258b6ed7"
    cid_ok = fc.decision.cid().startswith(expected_decision_cid_prefix)
    integrity_ok = (
        fc.disclosure_consistency_ok
        and cid_ok
        and ((not corpus_check.get("available"))
             or (corpus_check.get("sha_ok")
                 and corpus_check.get("histogram_match"))))
    if not cid_ok:
        print(f"  !! decision CID drift: expected prefix "
              f"{expected_decision_cid_prefix}, got {fc.decision.cid()}")
    return 0 if integrity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
