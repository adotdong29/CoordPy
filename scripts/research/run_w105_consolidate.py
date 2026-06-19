#!/usr/bin/env python3
"""W105 — Consolidate per-class run roots into a unified verdict.

When the Phase 3 retirement bench is run with two parallel
``--only-class`` processes (one per model class), each process
produces its own run root with cells for only one class.  This
consolidator symlinks the per-class ``class_*`` sub-directories
into a single consolidated run root and re-runs the driver's
emit functions to produce the unified Phase 3 retirement
verdict + cross-class comparator.

Usage::

    python scripts/run_w105_consolidate.py \\
        --source results/w105/humaneval_plus_phase3_retirement_bench/<class_a_run> \\
        --source results/w105/humaneval_plus_phase3_retirement_bench/<class_b_run> \\
        --out-root results/w105/humaneval_plus_phase3_retirement_bench
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        "W105 consolidator for parallel per-class Phase 3 "
        "runs"))
    ap.add_argument(
        "--source", action="append", required=True,
        help="Source run root (may be repeated)")
    ap.add_argument(
        "--out-root", default=str(
            ROOT / "results" / "w105"
            / "humaneval_plus_phase3_retirement_bench"),
        help="Consolidated output root")
    ap.add_argument(
        "--label", default=None,
        help=(
            "Optional explicit consolidated run name; if "
            "omitted, uses w105_phase3_<ts>_consolidated"))
    args = ap.parse_args()
    sources = [Path(s).resolve() for s in args.source]
    for s in sources:
        if not s.exists() or not s.is_dir():
            raise SystemExit(
                f"source run root missing or not a dir: {s}")
    run_id = _dt.datetime.now(
        _dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = (
        args.label
        if args.label
        else f"w105_phase3_{run_id}_consolidated")
    consolidated = Path(args.out_root) / name
    consolidated.mkdir(parents=True, exist_ok=True)
    print(f"  consolidating into {consolidated}")
    # Symlink class_* dirs from each source.
    class_dirs_seen: set[str] = set()
    for src in sources:
        for child in sorted(src.iterdir()):
            if not child.is_dir():
                continue
            if not child.name.startswith("class_"):
                continue
            if child.name in class_dirs_seen:
                print(
                    f"  WARNING: class {child.name} appears in "
                    f"multiple sources; skipping later one ({src})",
                    flush=True)
                continue
            link = consolidated / child.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(child.resolve(), link)
            class_dirs_seen.add(child.name)
            print(f"  linked {child.name} <- {src.name}")
    # Carry forward slice_pack_reference from the first source.
    src_ref = sources[0] / "slice_pack_reference.json"
    if src_ref.exists():
        import shutil
        shutil.copy(src_ref, consolidated /
                    "slice_pack_reference.json")
    # Now run the driver's emit functions.
    from scripts.run_w105_phase3_retirement_bench import (
        _emit_cross_class_comparator,
        _emit_per_class_partial_verdict_doc,
    )
    _emit_per_class_partial_verdict_doc(run_root=consolidated)
    _emit_cross_class_comparator(run_root=consolidated)
    latest = consolidated.parent / "latest_run.txt"
    with open(latest, "w") as f:
        f.write(consolidated.name + "\n")
    print(f"  consolidated run root: {consolidated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
