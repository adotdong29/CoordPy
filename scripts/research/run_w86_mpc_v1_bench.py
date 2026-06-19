#!/usr/bin/env python3
"""W86 / P2 #40 MPC V1 — bench driver."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coordpy.mpc_secret_sharing_v1 import (  # noqa: E402
    run_cross_org_mpc_bench_v1,
)


def _utc_stamp() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=int, default=4)
    p.add_argument("--seed", type=int, default=86_040)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else _REPO_ROOT / "results" / "w86" / "mpc"
        / f"w86_mpc_{_utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = run_cross_org_mpc_bench_v1(
        threshold=args.threshold, seed=args.seed)
    report_dict = {
        "kind": "w86_mpc_v1_bench_report",
        "schema": "coordpy.mpc_secret_sharing_v1.w86_v1",
        "report": rep.to_dict(),
    }
    out_path = out_dir / "mpc_v1_bench_report.json"
    out_path.write_text(json.dumps(
        report_dict, indent=2, sort_keys=True))

    print(f"wrote {out_path}")
    summary = {
        k: rep.to_dict()[k] for k in [
            "n_orgs", "total_parties", "threshold",
            "sum_matches", "no_cleartext_secrets_crossed_orgs",
            "drop_out_test_works", "all_proofs_valid",
            "forged_share_rejected",
            "insufficient_shares_recovers_nothing",
            "report_cid",
        ]
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")
    closed = (
        rep.sum_matches
        and rep.no_cleartext_secrets_crossed_orgs
        and rep.drop_out_test_works
        and rep.all_proofs_valid
        and rep.forged_share_rejected
        and rep.insufficient_shares_recovers_nothing)
    return 0 if closed else 1


if __name__ == "__main__":
    sys.exit(main())
