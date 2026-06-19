#!/usr/bin/env python3
"""W86 / P2 #42 Drift V1 — offline audit-chain verifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()


def _derive_report_cid(report_dict: dict) -> str:
    d = {**report_dict, "report_cid": ""}
    return _sha256({
        "kind": "w86_drift_bench_report_v1",
        "report": d})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to drift_v1_bench_report.json")
    args = p.parse_args(argv)

    overall = json.loads(
        Path(args.report).read_bytes().decode("utf-8"))
    rep = overall.get("report", {})
    if not rep:
        print("FAIL no report section")
        return 1
    notes: list[str] = []
    ok = True

    declared = rep.get("report_cid", "")
    derived = _derive_report_cid(rep)
    if declared != derived:
        notes.append(
            f"FAIL report_cid: recorded={declared} "
            f"derived={derived}")
        ok = False
    else:
        notes.append(f"PASS report_cid = {declared}")

    bars = [
        "detector_fires_when_changed",
        "detector_does_not_fire_when_unchanged",
        "stale_verdict_marks_old_capsule_stale",
        "stale_verdict_marks_fresh_capsule_fresh",
        "fallback_recommendation_is_recompute_for_stale",
        "new_memory_strictly_beats_stale_on_holdout",
    ]
    for b in bars:
        v = rep.get(b)
        if v is True:
            notes.append(f"PASS {b}")
        else:
            notes.append(f"FAIL {b}: got {v!r}")
            ok = False

    notes.append(
        f"INFO drift_score_unchanged = "
        f"{rep.get('drift_score_unchanged')}, "
        f"drift_score_changed = "
        f"{rep.get('drift_score_changed')}, "
        f"threshold = {rep.get('threshold')}")
    notes.append(
        f"INFO stale_holdout_mse = "
        f"{rep.get('stale_holdout_mse')}, "
        f"new_holdout_mse = {rep.get('new_holdout_mse')}")
    # Strict improvement check.
    if float(rep.get("new_holdout_mse", 1.0)) >= float(
            rep.get("stale_holdout_mse", 0.0)):
        notes.append(
            "FAIL new_holdout_mse >= stale_holdout_mse "
            "(not a strict beat)")
        ok = False
    else:
        notes.append(
            "PASS new_holdout_mse < stale_holdout_mse "
            "(strict beat)")

    for cid in ("old_weights_cid", "new_weights_cid"):
        v = rep.get(cid, "")
        if len(v) != 64:
            notes.append(f"FAIL {cid} not 64 chars: {v!r}")
            ok = False
        else:
            notes.append(f"PASS {cid} = {v}")
    # Old vs new must differ.
    if rep.get("old_weights_cid") == rep.get("new_weights_cid"):
        notes.append(
            "FAIL old_weights_cid == new_weights_cid")
        ok = False

    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
