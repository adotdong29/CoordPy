#!/usr/bin/env python3
"""W101 — MBPP+ NIM-free preflight harness.

Runs `coordpy.mbpp_plus_preflight_v1.run_mbpp_plus_preflight_v1`
against the W101 arsenal-mining report + (optionally cached)
MBPP+ corpus and writes a structured verdict JSON the W101
RUNBOOK + cheap-pilot driver can read.

NIM-free.  Costs only local CPU + (if MBPP+ data is cached)
~30 subprocess executor calls for the canonical-solution
self-test.

Usage::

    python scripts/run_w101_mbpp_plus_preflight.py
    python scripts/run_w101_mbpp_plus_preflight.py \\
        --arsenal-mining-report results/w101/arsenal_mining/<RUN>/mining_report.json
    python scripts/run_w101_mbpp_plus_preflight.py \\
        --skip-executor-self-test
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.mbpp_plus_preflight_v1 import (  # noqa: E402
    run_mbpp_plus_preflight_v1,
)


def _latest_arsenal_report() -> Path:
    latest_ptr = (
        ROOT / "results" / "w101" / "arsenal_mining"
        / "latest_run.txt")
    if not latest_ptr.exists():
        return ROOT / "results" / "w101" / "arsenal_mining" / (
            "MISSING") / "mining_report.json"
    pointer = latest_ptr.read_text().strip()
    cand = Path(pointer)
    if not cand.is_absolute():
        cand = latest_ptr.parent / cand
    return cand / "mining_report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W101 MBPP+ NIM-free preflight harness"))
    ap.add_argument(
        "--arsenal-mining-report",
        default=str(_latest_arsenal_report()),
        help="Path to W101 arsenal-mining report JSON")
    ap.add_argument(
        "--mbpp-plus-cache",
        default=None,
        help=(
            "Optional MBPP+ JSONL cache path; if not provided, "
            "uses COORDPY_MBPP_PLUS_CACHE env var or default "
            "~/.cache/coordpy/mbpp-plus.jsonl.gz"))
    ap.add_argument(
        "--bench-module-path",
        default=str(
            ROOT / "coordpy"
            / "mbpp_plus_reflexion_bench_v1.py"),
        help=(
            "Path to W101 bench module (read by AddrW101-P4 "
            "anti-pattern guard)"))
    ap.add_argument(
        "--skip-executor-self-test", action="store_true",
        help=(
            "Skip the P2 executor self-test (faster smoke; "
            "still writes the verdict)"))
    ap.add_argument(
        "--executor-self-test-sample",
        type=int, default=30,
        help="Number of canonical solutions to executor-test")
    ap.add_argument(
        "--candidate-model",
        default="meta/llama-3.3-70b-instruct",
        help="Target model name (records in verdict only)")
    ap.add_argument(
        "--out-dir",
        default=str(
            ROOT / "results" / "w101" / "mbpp_plus_preflight"),
        help="Output root")
    args = ap.parse_args()

    run_id = _dt.datetime.utcnow().strftime(
        "%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / f"w101_mbpp_plus_preflight_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    arsenal_path = Path(args.arsenal_mining_report)
    bench_path = Path(args.bench_module_path)
    verdict = run_mbpp_plus_preflight_v1(
        cache_path=args.mbpp_plus_cache,
        arsenal_mining_report_path=arsenal_path,
        bench_module_path=bench_path,
        candidate="B (W89 sequential reflexion on MBPP+)",
        candidate_model=str(args.candidate_model),
        run_executor_self_test=(
            not args.skip_executor_self_test),
        executor_self_test_sample=int(
            args.executor_self_test_sample),
    )
    verdict_dict = verdict.to_dict()
    with open(out_dir / "verdict.json", "w") as f:
        json.dump(verdict_dict, f, indent=2, default=str)
    latest = out_dir.parent / "latest_run.txt"
    with open(latest, "w") as f:
        f.write(out_dir.name + "\n")
    print()
    print(
        f"  candidate: {verdict.candidate}")
    print(
        f"  candidate model: {verdict.candidate_model}")
    print(
        f"  arsenal-mining report: {arsenal_path}")
    print()
    for p in verdict.probes:
        emoji = "PASS" if p.passed else "FAIL"
        print(f"  [{emoji}] {p.probe_id}: {p.summary}")
    print()
    print(
        f"  n_passed: {verdict.n_passed} / "
        f"{len(verdict.probes)} (required >= "
        f"{verdict.n_required})")
    print(
        f"  overall_passes: {verdict.overall_passes}")
    print(
        f"  verdict_cid: {verdict.verdict_cid}")
    print()
    print(f"  out_dir: {out_dir}")
    return 0 if verdict.overall_passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
