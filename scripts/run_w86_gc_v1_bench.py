#!/usr/bin/env python3
"""W86 / P2 #45 GC V1 — bench driver.

Runs the 100k-event GC bench and writes a content-addressed
report. Offline-re-verifiable via
``scripts/verify_w86_gc_v1_audit_chain.py``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coordpy.event_graph_garbage_collection_v1 import (  # noqa: E402
    run_100k_event_gc_bench_v1,
)


def _utc_stamp() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-events", type=int, default=100_000,
        help="number of events to generate (default 100 000)")
    p.add_argument(
        "--n-critical-anchors", type=int, default=10)
    p.add_argument("--seed", type=int, default=86_045)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else _REPO_ROOT / "results" / "w86" / "gc" /
             f"w86_gc_{_utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = run_100k_event_gc_bench_v1(
        n_events=args.n_events,
        n_critical_anchors=args.n_critical_anchors,
        seed=args.seed)

    report_dict = {
        "kind": "w86_gc_v1_bench_report",
        "schema": (
            "coordpy.event_graph_garbage_collection_v1.w86_gc_v1"),
        "report": rep.to_dict(),
    }
    report_path = out_dir / "gc_v1_bench_report.json"
    report_path.write_text(json.dumps(
        report_dict, indent=2, sort_keys=True))

    print(f"wrote {report_path}")
    print(
        f"  n_events_generated:  {rep.n_events_generated}\n"
        f"  n_events_after_gc:   {rep.n_events_after_gc}\n"
        f"  n_events_purged:     {rep.n_events_purged}\n"
        f"  memory_reduction:    "
        f"{100 * rep.memory_reduction_fraction:.2f}%\n"
        f"  chain_verifies:      {rep.chain_verifies_after_gc}\n"
        f"  grace_restore_works: {rep.grace_restore_works}\n"
        f"  store_round_trip:    "
        f"{rep.persistent_store_round_trip}\n"
        f"  report_cid:          {rep.report_cid}")
    closed = (
        rep.memory_reduction_fraction >= 0.80
        and rep.chain_verifies_after_gc
        and rep.grace_restore_works
        and rep.persistent_store_round_trip)
    return 0 if closed else 1


if __name__ == "__main__":
    sys.exit(main())
