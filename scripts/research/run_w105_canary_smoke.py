#!/usr/bin/env python3
"""W105 — Canary smoke entrypoint (thin wrapper).

Runs the W105 canary smoke: 1 seed × 3 problems × K=5 × 2
model classes = 66 NIM calls.  Validates reachability + budget
envelope + executor cleanness on both earned model classes
BEFORE the full Phase 3 envelope opens.

Canary acceptance bar: B − A1 ≥ −5 pp per class (very loose).
The canary is reachability + budget-envelope sanity, NOT a
Phase 3 gate.  A FAIL canary PAUSES the full launch.

Usage::

    export NVIDIA_API_KEY=...
    python scripts/run_w105_canary_smoke.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W105 canary smoke entrypoint"))
    ap.add_argument(
        "--slice-pack",
        default=str(
            ROOT / "data" / "w105" / "phase3_slice_pack"
            / "w105_phase3_slice_pack_20260526T215647Z"
            / "slice_pack.json"),
        help="Path to W105 pre-built slice pack JSON.")
    ap.add_argument(
        "--humaneval-plus-cache", default=None,
        help="HumanEval+ JSONL cache path override")
    ap.add_argument(
        "--out-root", default=str(
            ROOT / "results" / "w105"
            / "humaneval_plus_phase3_retirement_bench"),
        help="Output root")
    args = ap.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts"
            / "run_w105_phase3_retirement_bench.py"),
        "--canary",
        "--slice-pack", args.slice_pack,
        "--out-root", args.out_root,
    ]
    if args.humaneval_plus_cache:
        cmd += ["--humaneval-plus-cache",
                args.humaneval_plus_cache]
    print(f"  invoking: {' '.join(cmd)}", flush=True)
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
