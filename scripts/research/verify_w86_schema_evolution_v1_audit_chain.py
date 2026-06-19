#!/usr/bin/env python3
"""W86 / P2 #41 — offline schema-evolution audit-chain verifier."""
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
        "kind": "w86_schema_evolution_bench_report_v1",
        "report": d})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to schema_evolution_v1_bench_report.json")
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

    bars = [
        ("chain_verifies_across_migration", True),
        ("deprecated_payload_readable", True),
        ("deprecation_warning_emitted", True),
        ("deterministic_migration", True),
        ("provenance_preserved", True),
    ]
    for name, expected in bars:
        actual = rep.get(name)
        if actual == expected:
            notes.append(f"PASS {name} = {actual}")
        else:
            notes.append(
                f"FAIL {name}: expected {expected}, "
                f"got {actual}")
            ok = False

    for cid_field in (
            "registry_cid_before", "registry_cid_after",
            "migration_plan_cid", "v1_payload_cid",
            "v2_payload_cid", "migration_event_cid"):
        v = rep.get(cid_field, "")
        if len(v) != 64:
            notes.append(
                f"FAIL {cid_field} not 64 chars: {v!r}")
            ok = False
        else:
            notes.append(f"PASS {cid_field} = {v}")

    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
