#!/usr/bin/env python3
"""W86 / P2 #38 BFT V1 — bench driver.

Runs the three load-bearing BFT V1 benches and writes a
content-addressed report:

  1. collusion at f = ⌊(n-1)/3⌋, n=7 → must commit μ exactly
  2. refuse-to-commit at f = f_bound+1, n=4 → must NOT commit
  3. equivocation detection, n=4 → must produce independently-
     verifiable equivocation evidence

The output JSON is offline-re-verifiable via
``scripts/verify_w86_bft_v1_audit_chain.py``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

# Add repo root to path so coordpy imports work without install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coordpy.byzantine_fault_tolerance_v1 import (  # noqa: E402
    run_bft_v1_full_suite,
)


def _utc_stamp() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-collusion", type=int, default=7,
        help="N for the collusion bench (must be >= 4)")
    p.add_argument(
        "--n-refuse", type=int, default=4,
        help="N for the refuse-to-commit bench (must be >= 4)")
    p.add_argument(
        "--n-equiv", type=int, default=4,
        help="N for the equivocation bench (must be >= 4)")
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=0.3)
    p.add_argument("--target-delta", type=float, default=7.7)
    p.add_argument("--seed", type=int, default=86_038)
    p.add_argument(
        "--out-dir", default=None,
        help="output directory (default: results/w86/bft/<ts>/)")
    args = p.parse_args(argv)

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else _REPO_ROOT / "results" / "w86" / "bft" /
             f"w86_bft_{_utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = run_bft_v1_full_suite(
        n_collusion=args.n_collusion,
        n_refuse=args.n_refuse,
        n_equiv=args.n_equiv,
        mu=args.mu, delta=args.delta,
        target_delta=args.target_delta,
        seed=args.seed)

    report_dict = {
        "kind": "w86_bft_v1_suite_report",
        "schema": (
            "coordpy.byzantine_fault_tolerance_v1.w86_bft_v1"),
        "suite_cid": suite.report_cid,
        "closed": suite.closed,
        "reports": [r.to_dict() for r in suite.reports],
        "collusion_report_cid": suite.collusion_report_cid,
        "refuse_report_cid": suite.refuse_report_cid,
        "equivocation_report_cid": suite.equivocation_report_cid,
    }

    report_path = out_dir / "bft_v1_suite_report.json"
    report_path.write_text(
        json.dumps(report_dict, indent=2, sort_keys=True))

    print(f"wrote {report_path}")
    print(f"closed: {suite.closed}")
    for r in suite.reports:
        print(
            f"  {r.bench_kind}: verdict={r.verdict}, "
            f"safety_holds={r.safety_holds}, "
            f"committed={r.committed}")
    return 0 if suite.closed else 1


if __name__ == "__main__":
    sys.exit(main())
