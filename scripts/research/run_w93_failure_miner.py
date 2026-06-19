"""W93 failure-cluster miner CLI runner.

Discovers all W88–W92 bench reports under results/w8X/ and
results/w9X/ trees and produces a structured failure-cluster
report.  No NIM calls; cheap; runs in seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coordpy.failure_cluster_miner_v1 import (
    discover_runs, mine_all_runs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="W93 failure-cluster miner")
    parser.add_argument(
        "--root", default=str(ROOT / "results"))
    parser.add_argument(
        "--out", default=str(
            ROOT / "results" / "w93" /
            "failure_clusters.json"))
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"no such root: {root}")

    print(f"scanning {root} ...")
    candidates = discover_runs(root)
    print(f"found {len(candidates)} bench reports")
    for c in candidates:
        print(f"  {c.relative_to(root)}")

    report = mine_all_runs(root)
    print()
    print(f"mined {len(report['per_run'])} runs successfully")
    print()
    print("Cross-run B − A1 (pp) per bench kind:")
    for kind, runs in report["cross_run_patterns"]["by_bench_kind"].items():
        deltas = [r["mean_b_minus_a1_pp"] for r in runs]
        if deltas:
            print(
                f"  {kind}: n={len(deltas)}, "
                f"min={min(deltas):+.2f}, max={max(deltas):+.2f}, "
                f"mean={sum(deltas)/len(deltas):+.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nfailure-cluster report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
