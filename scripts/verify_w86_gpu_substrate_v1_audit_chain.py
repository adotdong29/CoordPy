#!/usr/bin/env python3
"""W86 / P2 #44 GPU Substrate V1 — offline audit-chain verifier."""
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
        "kind": "w86_gpu_substrate_bench_report_v1",
        "report": d})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to gpu_substrate_v1_bench_report.json")
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

    notes.append(
        f"INFO cuda_device_name = "
        f"{rep.get('cuda_device_name')!r}, "
        f"capability = {rep.get('cuda_capability')!r}, "
        f"precision_tier = {rep.get('precision_tier')!r}")
    notes.append(
        f"INFO pos_replay_max_abs_diff = "
        f"{rep.get('pos_replay_max_abs_diff')}, "
        f"tier_tolerance = {rep.get('tier_tolerance')}")
    notes.append(
        f"INFO neg_replay_max_abs_diff = "
        f"{rep.get('neg_replay_max_abs_diff')}")

    bars = [
        "pos_replay_within_tier_tolerance",
        "pos_intercept_moves_cid",
        "pos_forwards_byte_identical",
        "neg_replay_breaks_byte_identity",
        "tp_readback_passthrough_byte_identical",
    ]
    for b in bars:
        v = rep.get(b)
        if v is True:
            notes.append(f"PASS {b}")
        else:
            notes.append(f"FAIL {b}: got {v!r}")
            ok = False

    for cid in (
            "determinism_wrapper_result_cid",
            "tensor_parallel_config_cid",
            "pos_forward_trace_cid_first",
            "pos_forward_trace_cid_second"):
        v = rep.get(cid, "")
        if len(v) != 64:
            notes.append(f"FAIL {cid} not 64 chars: {v!r}")
            ok = False
        else:
            notes.append(f"PASS {cid} = {v}")
    # Pos forwards byte-identical → both CIDs match.
    if (rep.get("pos_forwards_byte_identical") is True
            and rep.get("pos_forward_trace_cid_first")
            != rep.get("pos_forward_trace_cid_second")):
        notes.append(
            "FAIL pos_forwards_byte_identical=True but trace "
            "CIDs differ")
        ok = False

    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
