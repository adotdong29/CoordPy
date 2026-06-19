#!/usr/bin/env python3
"""W86 / P2 #39 DP V1 — offline audit-chain verifier."""
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
        "kind": "w86_dp_bench_report_v1", "report": d})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to dp_v1_bench_report.json")
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

    pat_count = int(rep.get("pii_redaction_pattern_count", 0))
    if pat_count < 5:
        notes.append(
            f"FAIL pii_redaction_pattern_count {pat_count} < 5")
        ok = False
    else:
        notes.append(
            f"PASS pii_redaction_pattern_count = {pat_count}")

    bars = [
        "pii_not_in_output",
        "dp_committed_value_within_3_sigma",
        "budget_breach_refused",
        "utility_curve_is_monotonic",
        "raw_value_not_in_capsule_dict",
    ]
    for b in bars:
        v = rep.get(b)
        if v is True:
            notes.append(f"PASS {b}")
        else:
            notes.append(f"FAIL {b}: got {v!r}")
            ok = False

    redactions = int(rep.get("pii_redactions_made", 0))
    notes.append(
        f"INFO pii_redactions_made = {redactions}")
    if redactions < 5:
        notes.append(
            f"FAIL pii_redactions_made {redactions} < 5")
        ok = False

    curve = rep.get("utility_curve_points", [])
    if len(curve) < 5:
        notes.append(
            f"FAIL utility curve points {len(curve)} < 5")
        ok = False
    else:
        notes.append(
            f"PASS utility curve points = {len(curve)}")

    for cid in ("dp_capsule_cid", "integrity_anchor_cid"):
        v = rep.get(cid, "")
        if len(v) != 64:
            notes.append(
                f"FAIL {cid} not 64 chars: {v!r}")
            ok = False
        else:
            notes.append(f"PASS {cid} = {v}")

    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
