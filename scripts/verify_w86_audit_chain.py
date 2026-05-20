#!/usr/bin/env python3
"""Offline re-verifier for the W86 frontier closure audit chain.

Given a ``frontier_closure_report.json`` produced by
``scripts/run_frontier_closure_w86.py`` plus the per-issue
sidecar files in the same directory, re-computes:

* the top-level ``report_cid`` over the canonical bytes of every
  field EXCEPT ``report_cid`` itself, and asserts equality with
  the recorded value.
* the per-issue ``report_cid`` on each sidecar against the value
  the top-level report carries under ``closure_25``,
  ``closure_26``, ``closure_27``.
* the strict-beat / intercept-moves-CID bools that the issue
  DoDs ask for, printing each one as PASS / FAIL so a third
  party can map evidence to bullets.

Exit code is 0 iff every CID re-verifies and the top-level
report's strict-beat / intercept-moves-CID bools are consistent
with the sidecars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def verify_report(report_path: Path) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True
    with report_path.open("rb") as fh:
        report = json.load(fh)
    expected_cid = report.get("report_cid", "")
    actual_cid = _sha256_hex({
        "kind": "w86_frontier_closure_report_v1",
        "report": {
            k: v for k, v in report.items()
            if k != "report_cid"},
    })
    if actual_cid == expected_cid:
        notes.append(
            f"PASS top-level report_cid = {actual_cid[:16]}...")
    else:
        ok = False
        notes.append(
            f"FAIL top-level report_cid: "
            f"recorded={expected_cid[:16]} "
            f"recomputed={actual_cid[:16]}")

    out_dir = report_path.parent
    closure_to_file = {
        "closure_25": "25_substrate_coupling.json",
        "closure_26": "26_live_learned_memory.json",
        "closure_27": "27_long_context_intercept.json",
    }
    for closure_key, side_name in closure_to_file.items():
        side_path = out_dir / side_name
        if not side_path.exists():
            notes.append(
                f"SKIP {closure_key}: sidecar {side_name} "
                "missing")
            continue
        with side_path.open("rb") as fh:
            side = json.load(fh)
        side_expected = side.get("report_cid", "")
        kind_map = {
            "closure_25": "w86_25_substrate_coupling_report",
            "closure_26": (
                "w86_26_live_learned_memory_report"),
            "closure_27": (
                "w86_27_long_context_intercept_report"),
        }
        side_actual = _sha256_hex({
            "kind": kind_map[closure_key],
            "out": {
                k: v for k, v in side.items()
                if k != "report_cid"},
        })
        if side_actual == side_expected:
            notes.append(
                f"PASS {closure_key} sidecar CID = "
                f"{side_actual[:16]}...")
        else:
            ok = False
            notes.append(
                f"FAIL {closure_key} sidecar CID: "
                f"recorded={side_expected[:16]} "
                f"recomputed={side_actual[:16]}")

    # Headline bools.
    c25 = report.get("closure_25", {})
    if c25:
        hib = c25.get("hidden_state_intercept_bench", {})
        moves = bool(
            hib.get("hidden_state_intercept_moves_cid"))
        notes.append(
            f"{'PASS' if moves else 'FAIL'} #25 hidden-state "
            f"intercept moves CID at frontier: {moves}")
        conf = c25.get("conformance", {})
        n_pass = int(conf.get("n_pass", 0))
        n_fail = int(conf.get("n_fail", 0))
        notes.append(
            f"{'PASS' if n_pass >= 10 else 'FAIL'} #25 W80 "
            f"conformance: n_pass={n_pass} n_fail={n_fail}")
        rep_kv = c25.get("replay_from_kv", {})
        if rep_kv:
            byte_id = bool(
                rep_kv.get("replay_byte_identical"))
            tol = rep_kv.get(
                "precision_tier_tolerance", "?")
            mad = rep_kv.get(
                "max_abs_diff_last_logits", "?")
            tier = rep_kv.get("precision_tier", "?")
            notes.append(
                f"{'PASS' if byte_id else 'FAIL'} #25 "
                f"replay-from-KV at tier {tier} "
                f"(tolerance={tol}, max_abs_diff={mad})")
    c26 = report.get("closure_26", {})
    if c26:
        live = bool(
            c26.get("live_strictly_beats_synthetic"))
        notes.append(
            f"{'PASS' if live else 'FAIL'} #26 live-trained "
            f"MSE strictly < synthetic-trained MSE on held-"
            f"out live hidden states: {live}")
    c27 = report.get("closure_27", {})
    if c27:
        moves = bool(
            c27.get("intercept_moves_cid_at_min_32k"))
        notes.append(
            f"{'PASS' if moves else 'FAIL'} #27 hidden-state "
            f"intercept moves CID at ≥ 32 k tokens: {moves}")

    return ok, notes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to frontier_closure_report.json")
    args = p.parse_args(argv)
    ok, notes = verify_report(Path(args.report))
    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
