#!/usr/bin/env python3
"""W86 / P2 #43 Multi-Tenancy V1 — offline audit-chain verifier."""
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
        "kind": "w86_multi_tenancy_bench_report_v1",
        "report": d})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to multi_tenancy_v1_bench_report.json")
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
        "cross_tenant_read_refused",
        "cross_tenant_denial_event_emitted",
        "audit_anchors_distinct",
        "budget_isolation_holds",
        "token_swap_refused",
        "no_b_bytes_in_a_chain",
    ]
    for b in bars:
        v = rep.get(b)
        if v is True:
            notes.append(f"PASS {b}")
        else:
            notes.append(f"FAIL {b}: got {v!r}")
            ok = False

    for cid in (
            "tenant_a_cid", "tenant_b_cid",
            "tenant_a_anchor_root", "tenant_b_anchor_root"):
        v = rep.get(cid, "")
        if len(v) != 64:
            notes.append(
                f"FAIL {cid} not 64 chars: {v!r}")
            ok = False
        else:
            notes.append(f"PASS {cid} = {v}")
    if rep.get("tenant_a_anchor_root") == rep.get(
            "tenant_b_anchor_root"):
        notes.append(
            "FAIL tenant_a_anchor_root == tenant_b_anchor_root "
            "(anchors NOT distinct)")
        ok = False

    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
