#!/usr/bin/env python3
"""W115-α/β/γ — durable future-fire certification / instrument-supply pipeline.

NIM-free.  Runs the LOCKED W115 frontier-certification pipeline
(``coordpy.frontier_certification_pipeline_v1``) against the live-verified
``W115_FRONTIER_SNAPSHOT`` (RUNBOOK_W115 § 2, primary-source re-check 2026-05-29):

* the latest-official-release detector (is there a release_v7+ to admit?);
* the frontier-date histogram / summary + threshold table;
* the per-model go/no-go matrix with exact blocker reasons (reuses the W114 gate);
* the disclosure-consistency guard (live disclosures vs the encoded registry);
* the structured W116 fire condition.

It ALSO re-verifies the pinned functional month histogram against the live
SHA-pinned ``release_v6`` corpus when present (a preflight-style discharge that the
instrument frontier the decision rests on matches real bytes).  $0 NIM.

Usage::

    python scripts/run_w115_frontier_certification_v1.py
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

from coordpy.frontier_certification_pipeline_v1 import (  # noqa: E402
    VERDICT_NONE,
    run_frontier_certification_v1,
)
from coordpy.stronger_model_cutoff_certification_v1 import (  # noqa: E402
    LATEST_RESISTANT_INSTRUMENT,
)
from coordpy.livecodebench_resistant_slice_v1 import (  # noqa: E402
    normalize_contest_date_v1,
)

OUT = ROOT / "results" / "w115" / "frontier_certification"


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
    result = run_frontier_certification_v1()
    corpus_check = _reverify_histogram_against_corpus()

    artifact = {
        "schema": "coordpy.w115_frontier_certification.v1",
        "milestone": "W115-alpha-beta-gamma",
        "result": result.to_dict(),
        "result_cid": result.cid(),
        "decision_cid": result.decision.cid(),
        "corpus_reverification": corpus_check,
        "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frontier_certification_verdict.json").write_text(
        json.dumps(artifact, indent=2, default=str))

    rd = result.release_detection
    fs = result.frontier_summary
    print("=== W115 frontier certification / instrument-supply pipeline ===")
    print(f"  verified_on: {result.verified_on}")
    print("  [latest-release detector]")
    print(f"    admitted latest = {rd.latest_admitted} "
          f"(v{rd.latest_admitted_num}); observed latest = {rd.latest_observed} "
          f"(v{rd.latest_observed_num})")
    print(f"    newer_release_available = {rd.newer_release_available}"
          + (f"; observed-not-admitted = {list(rd.observed_not_admitted)}"
             if rd.observed_not_admitted else ""))
    print("  [frontier-date summary]")
    print(f"    {fs.release}: n_functional={fs.n_functional}; "
          f"{fs.functional_date_min}..{fs.functional_date_max}")
    print(f"    histogram = {fs.month_histogram}")
    print(f"    max KNOWN cutoff month admitting >= {fs.min_slice} resistant = "
          f"{fs.max_cutoff_month_for_min_slice or '(none)'}")
    print("  [per-model go/no-go matrix] "
          "(C1 known / C2 >=30 / C3 reach-stronger / C4 not-settled)")
    for m in result.decision.per_model:
        print(f"    [{m.rank_tier}.{m.rank_within_tier}] {m.model_id:42s} "
              f"cutoff={m.cutoff_boundary}[{m.cutoff_confidence}] "
              f"n_res={m.n_functional_resistant:2d} "
              f"C1={int(m.c1_cutoff_known)} C2={int(m.c2_enough_resistant)} "
              f"C3={int(m.c3_reachable_stronger_comparable)} "
              f"C4={int(m.c4_not_already_settled)} "
              f"=> certifiable={m.certifiable_for_new_pilot}")
    print(f"  VERDICT: {result.verdict}"
          + (f" (target={result.target_model})" if result.target_model else ""))
    print(f"  disclosure_consistency_ok: {result.disclosure_consistency_ok}")
    print(f"  result CID: {result.cid()}")
    print(f"  decision CID: {result.decision.cid()}")
    print(f"  corpus re-verification: available="
          f"{corpus_check.get('available')} sha_ok={corpus_check.get('sha_ok')} "
          f"histogram_match={corpus_check.get('histogram_match')}")
    fc = result.w116_fire_condition
    print("  [W116 fire condition]")
    print(f"    fires_now = {fc.fires_now} "
          f"(instrument_trigger_met={fc.instrument_trigger_met}; "
          f"cutoff_trigger_met={fc.cutoff_trigger_met})")
    if result.verdict == VERDICT_NONE:
        print("    instrument trigger: " + fc.instrument_trigger)
        print("    cutoff trigger:     " + fc.cutoff_trigger)
        print("  W115 BLOCKER (W114 carry-forward, live-re-verified):")
        print(f"    {result.decision.w115_blocker}")
    print(f"  artifact: {OUT / 'frontier_certification_verdict.json'}")

    # Exit non-zero ONLY on an integrity failure (corpus present but SHA/hist
    # mismatch, or a live-disclosure vs encoded-registry divergence) — NOT on a
    # no-go verdict (a no-go is a valid, expected outcome).
    integrity_ok = result.disclosure_consistency_ok and (
        (not corpus_check.get("available"))
        or (corpus_check.get("sha_ok") and corpus_check.get("histogram_match")))
    return 0 if integrity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
