#!/usr/bin/env python3
"""W86 / P2 #38 BFT V1 — offline audit-chain verifier.

Given a ``bft_v1_suite_report.json`` produced by
``scripts/run_w86_bft_v1_bench.py``, prints PASS/FAIL per #38
DoD bullet and exits 0 iff every load-bearing bool is True.

Anti-cheat: every report's ``report_cid`` MUST re-derive from
the report dict via the canonical hash.
"""
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
        "kind": "w86_bft_bench_report_v1", "report": d})


def _derive_suite_cid(suite_payload: dict) -> str:
    d = {
        "collusion_report_cid": str(
            suite_payload.get("collusion_report_cid", "")),
        "refuse_report_cid": str(
            suite_payload.get("refuse_report_cid", "")),
        "equivocation_report_cid": str(
            suite_payload.get("equivocation_report_cid", "")),
        "closed": bool(suite_payload.get("closed", False)),
        "report_cid": "",
    }
    return _sha256({
        "kind": "w86_bft_bench_suite_v1", "suite": d})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report", required=True,
        help="path to bft_v1_suite_report.json")
    args = p.parse_args(argv)

    overall = json.loads(
        Path(args.report).read_bytes().decode("utf-8"))

    notes: list[str] = []
    ok = True

    declared_suite_cid = overall.get("suite_cid", "")
    derived_suite_cid = _derive_suite_cid(overall)
    if declared_suite_cid != derived_suite_cid:
        notes.append(
            f"FAIL suite_cid: recorded={declared_suite_cid} "
            f"derived={derived_suite_cid}")
        ok = False
    else:
        notes.append(f"PASS suite_cid = {declared_suite_cid}")

    reports = overall.get("reports", [])
    if len(reports) != 3:
        notes.append(
            f"FAIL expected 3 reports, got {len(reports)}")
        ok = False
    else:
        notes.append(f"PASS report_count = 3")

    bench_kinds_present = {r.get("bench_kind") for r in reports}
    required_kinds = {
        "collusion_at_byzantine_bound_v1",
        "refuse_to_commit_above_byzantine_bound_v1",
        "equivocation_detection_v1",
    }
    missing = required_kinds - bench_kinds_present
    if missing:
        notes.append(f"FAIL missing benches: {sorted(missing)}")
        ok = False
    else:
        notes.append(
            f"PASS all 3 load-bearing bench kinds present")

    # DoD: ByzantineWitnessV1 schema exists with cryptographic
    # signatures over the value (Ed25519 or similar).
    # → Module-level import test by the CI; not checked here.

    # Re-derive each report_cid.
    for r in reports:
        declared = r.get("report_cid", "")
        derived = _derive_report_cid(r)
        kind = r.get("bench_kind", "?")
        if declared != derived:
            notes.append(
                f"FAIL {kind} report_cid: recorded={declared} "
                f"derived={derived}")
            ok = False
        else:
            notes.append(f"PASS {kind} report_cid")

    # Per-bench load-bearing bools.
    by_kind: dict[str, dict] = {
        r.get("bench_kind"): r for r in reports}

    # Collusion at f: must commit μ exactly with zero error.
    col = by_kind.get("collusion_at_byzantine_bound_v1", {})
    if col:
        committed = col.get("committed", False)
        err = col.get("committed_error", None)
        safety = col.get("safety_holds", False)
        notes.append(
            f"INFO collusion: n={col.get('n')}, "
            f"f_target={col.get('f_target')}, "
            f"committed_value={col.get('committed_value')}, "
            f"committed_error={err}")
        if not committed:
            notes.append("FAIL collusion: must commit μ")
            ok = False
        elif err is None or err > 1e-9:
            notes.append(
                f"FAIL collusion: committed_error {err} > 1e-9")
            ok = False
        else:
            notes.append("PASS collusion: committed μ exactly")
        if not safety:
            notes.append("FAIL collusion: safety_holds=False")
            ok = False

    # Refuse at f > bound: must NOT commit.
    ref = by_kind.get(
        "refuse_to_commit_above_byzantine_bound_v1", {})
    if ref:
        committed = ref.get("committed", False)
        verdict = ref.get("verdict")
        notes.append(
            f"INFO refuse: n={ref.get('n')}, "
            f"f_target={ref.get('f_target')}, "
            f"f_bound={ref.get('f_byzantine_bound')}, "
            f"verdict={verdict}")
        if committed:
            notes.append(
                "FAIL refuse: must NOT commit at f > bound")
            ok = False
        else:
            notes.append(
                f"PASS refuse: did not commit (verdict={verdict})")

    # Equivocation: must produce ≥1 independently-verifiable
    # evidence capsule AND refuse to commit.
    eq = by_kind.get("equivocation_detection_v1", {})
    if eq:
        ev_count = eq.get("equivocation_evidence_count", 0)
        ev_verifiable = eq.get(
            "equivocation_independently_verifiable", False)
        committed = eq.get("committed", False)
        notes.append(
            f"INFO equivocation: n={eq.get('n')}, "
            f"evidence_count={ev_count}, "
            f"independently_verifiable={ev_verifiable}, "
            f"committed={committed}")
        if ev_count < 1:
            notes.append(
                "FAIL equivocation: no evidence produced")
            ok = False
        elif not ev_verifiable:
            notes.append(
                "FAIL equivocation: evidence not "
                "independently verifiable")
            ok = False
        elif committed:
            notes.append(
                "FAIL equivocation: committed despite evidence")
            ok = False
        else:
            notes.append(
                "PASS equivocation: evidence produced and "
                "round refused to commit")

    closed = bool(overall.get("closed", False))
    if not closed:
        notes.append("FAIL suite.closed=False")
        ok = False
    else:
        notes.append("PASS suite.closed=True")

    for n in notes:
        print(n)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
