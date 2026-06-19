#!/usr/bin/env python3
"""W86 / P2 #40 MPC V1 — offline audit-chain verifier."""
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
        "kind": "w86_cross_org_mpc_bench_report_v1",
        "report": d})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to mpc_v1_bench_report.json")
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
        "sum_matches",
        "no_cleartext_secrets_crossed_orgs",
        "drop_out_test_works",
        "all_proofs_valid",
        "forged_share_rejected",
        "insufficient_shares_recovers_nothing",
    ]
    for b in bars:
        v = rep.get(b)
        if v is True:
            notes.append(f"PASS {b}")
        else:
            notes.append(f"FAIL {b}: got {v!r}")
            ok = False

    # At least one summed-share CID + one pedersen proof CID
    # must be present.
    sscids = rep.get("summed_share_capsule_cids", [])
    ppcids = rep.get("pedersen_proof_cids", [])
    if not sscids:
        notes.append("FAIL no summed-share CIDs")
        ok = False
    else:
        notes.append(
            f"PASS summed-share CIDs (count={len(sscids)})")
    if not ppcids:
        notes.append("FAIL no Pedersen proof CIDs")
        ok = False
    else:
        notes.append(
            f"PASS pedersen-proof CIDs (count={len(ppcids)})")

    for cid_list, name in (
            (sscids, "summed_share"),
            (ppcids, "pedersen_proof")):
        for c in cid_list:
            if len(c) != 64:
                notes.append(
                    f"FAIL {name}_cid not 64 chars: {c!r}")
                ok = False
                break

    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
