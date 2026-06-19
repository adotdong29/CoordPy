#!/usr/bin/env python3
"""W116-α/β/γ — durable upstream instrument-ADMISSION pipeline.

NIM-free.  Runs the LOCKED W116 upstream-admission pipeline
(``coordpy.upstream_instrument_admission_v1``) against the live-verified
``W116_UPSTREAM_SNAPSHOT`` (RUNBOOK_W116 § 2, primary-source upstream-supply attack
2026-05-30):

* the FOUR-surface upstream-supply view (lite tree / loader ALLOWED_FILES +
  release_latest / full dataset / GitHub) + the pre-committed admissibility rule;
* the per-instrument admissibility verdicts (release_v6 admissible-but-not-new; the
  'planned v7' rumor REFUSED by A1 + A4) => 0 admissible NEW instruments;
* the per-model go/no-go matrix (reuses the W115/W114 gate; decision CID 258b6ed7);
* the four-way disclosure-status matrix (Mistral-Small-4 CONFIRMED REAL + primary
  NO-cutoff);
* the upstream-change detector (no change vs the encoded baseline) + the structured
  W117 fire condition.

It ALSO re-verifies the pinned functional month histogram against the live
SHA-pinned ``release_v6`` corpus when present (a preflight-style discharge that the
instrument the decision rests on matches real bytes).  $0 NIM.

Usage::

    python scripts/run_w116_upstream_admission_v1.py
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

from coordpy.upstream_instrument_admission_v1 import (  # noqa: E402
    VERDICT_NONE,
    run_upstream_admission_v1,
)
from coordpy.stronger_model_cutoff_certification_v1 import (  # noqa: E402
    LATEST_RESISTANT_INSTRUMENT,
)
from coordpy.livecodebench_resistant_slice_v1 import (  # noqa: E402
    normalize_contest_date_v1,
)

OUT = ROOT / "results" / "w116" / "upstream_admission"


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
    result = run_upstream_admission_v1()
    corpus_check = _reverify_histogram_against_corpus()

    artifact = {
        "schema": "coordpy.w116_upstream_admission.v1",
        "milestone": "W116-alpha-beta-gamma",
        "result": result.to_dict(),
        "result_cid": result.cid(),
        "decision_cid": result.frontier_certification.decision.cid(),
        "frontier_result_cid": result.frontier_certification.cid(),
        "corpus_reverification": corpus_check,
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "upstream_admission_verdict.json").write_text(
        json.dumps(artifact, indent=2, default=str))

    fc = result.frontier_certification
    print("=== W116 upstream instrument-admission pipeline ===")
    print(f"  verified_on: {result.verified_on}")
    print("  [upstream supply — four surfaces]")
    snap_change = result.upstream_change
    print(f"    lite tree latest = {fc.release_detection.latest_observed} "
          f"(v{fc.release_detection.latest_observed_num}); "
          f"release_latest -> "
          f"{result.frontier_certification.frontier_summary.release} (admitted)")
    print(f"    upstream change vs baseline: any_change="
          f"{snap_change.any_change} :: {'; '.join(snap_change.notes)}")
    print("  [instrument admissibility] (A1 auth / A2 dated / A3 functional / "
          "A4 sha-pinnable / A5 histogram)")
    for a in result.instrument_admissibility:
        print(f"    {a.label:38s} admissible={int(a.admissible)} "
              f"new={int(a.admissible_new_instrument)} :: {a.reason}")
    print(f"  n_admissible_NEW_instruments = "
          f"{result.n_admissible_new_instruments}")
    print("  [per-model disclosure-status matrix] (primary-source, 2026-05-30)")
    for d in result.disclosure_matrix:
        print(f"    {d.model_id:46s} {d.primary_status:22s} "
              f">70B={int(d.stronger_than_70b)} blocker={d.certifiable_blocker}")
    ds = result.disclosure_summary
    print(f"    disclosure counts = {ds['counts']}; "
          f"any usable NEW KNOWN-cutoff target = "
          f"{ds['any_usable_new_known_cutoff_target']}")
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
    w117 = result.w117_fire_condition
    print("  [W117 fire condition]")
    print(f"    fires_now = {w117.fires_now} "
          f"(instrument_trigger_met={w117.instrument_trigger_met}; "
          f"cutoff_trigger_met={w117.cutoff_trigger_met})")
    if result.verdict == VERDICT_NONE:
        print("    instrument trigger: " + w117.instrument_trigger)
        print("    cutoff trigger:     " + w117.cutoff_trigger)
        print("  W116 BLOCKER (live-re-verified, sharpened both sides):")
        print(f"    {fc.decision.w115_blocker}")
    print(f"  artifact: {OUT / 'upstream_admission_verdict.json'}")

    # Exit non-zero ONLY on an integrity failure (corpus present but SHA/hist
    # mismatch, a live-disclosure vs encoded-registry divergence, or the decision
    # CID drifting from the W114/W115 byte-identical 258b6ed7 invariant) — NOT on a
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
