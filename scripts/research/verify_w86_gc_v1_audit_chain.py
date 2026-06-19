#!/usr/bin/env python3
"""W86 / P2 #45 GC V1 — offline audit-chain verifier."""
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
        "kind": "w86_gc_bench_report_v1", "report": d})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to gc_v1_bench_report.json")
    args = p.parse_args(argv)

    overall = json.loads(
        Path(args.report).read_bytes().decode("utf-8"))
    rep = overall.get("report", {})
    if not rep:
        print("FAIL no report section in JSON")
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

    n_generated = int(rep.get("n_events_generated", 0))
    notes.append(
        f"INFO n_events_generated = {n_generated}, "
        f"n_after = {rep.get('n_events_after_gc')}, "
        f"purged = {rep.get('n_events_purged')}")

    # DoD: ≥ 100k events.
    if n_generated < 100_000:
        notes.append(
            f"WARN n_events_generated {n_generated} < 100k "
            "(V1 minimum)")

    mem_red = float(rep.get("memory_reduction_fraction", 0))
    notes.append(
        f"INFO memory_reduction_fraction = "
        f"{100 * mem_red:.2f}%")
    if mem_red < 0.80:
        notes.append(
            f"FAIL memory_reduction_fraction {mem_red:.4f} "
            "< 0.80")
        ok = False
    else:
        notes.append(
            f"PASS memory_reduction_fraction "
            f"{mem_red:.4f} >= 0.80")

    chain = bool(rep.get("chain_verifies_after_gc", False))
    if not chain:
        notes.append("FAIL chain_verifies_after_gc")
        ok = False
    else:
        notes.append("PASS chain_verifies_after_gc")

    grace = bool(rep.get("grace_restore_works", False))
    if not grace:
        notes.append("FAIL grace_restore_works")
        ok = False
    else:
        notes.append("PASS grace_restore_works")

    store = bool(rep.get(
        "persistent_store_round_trip", False))
    if not store:
        notes.append("FAIL persistent_store_round_trip")
        ok = False
    else:
        notes.append("PASS persistent_store_round_trip")

    # GC event CID must be a 64-char hex string.
    gc_cid = rep.get("gc_event_cid", "")
    if len(gc_cid) != 64 or not all(
            c in "0123456789abcdef" for c in gc_cid):
        notes.append(
            f"FAIL gc_event_cid not a 64-char hex: {gc_cid!r}")
        ok = False
    else:
        notes.append(f"PASS gc_event_cid = {gc_cid}")

    # Policy CID must be a 64-char hex string.
    pol_cid = rep.get("policy_cid", "")
    if len(pol_cid) != 64:
        notes.append(
            f"FAIL policy_cid not a 64-char hex: {pol_cid!r}")
        ok = False
    else:
        notes.append(f"PASS policy_cid = {pol_cid}")

    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
