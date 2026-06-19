#!/usr/bin/env python3
"""W86 / P2 #39 DP V1 — bench driver."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coordpy.differential_privacy_v1 import (  # noqa: E402
    run_dp_v1_bench,
)


def _utc_stamp() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=86_039)
    p.add_argument(
        "--n-samples-per-eps", type=int, default=1000)
    p.add_argument(
        "--budget-total-epsilon", type=float, default=2.0)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else _REPO_ROOT / "results" / "w86" / "dp"
        / f"w86_dp_{_utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = run_dp_v1_bench(
        n_samples_per_eps=args.n_samples_per_eps,
        budget_total_epsilon=args.budget_total_epsilon,
        seed=args.seed)

    report_dict = {
        "kind": "w86_dp_v1_bench_report",
        "schema": "coordpy.differential_privacy_v1.w86_v1",
        "report": rep.to_dict(),
    }
    out_path = out_dir / "dp_v1_bench_report.json"
    out_path.write_text(
        json.dumps(report_dict, indent=2, sort_keys=True))

    print(f"wrote {out_path}")
    summary = {
        k: rep.to_dict()[k] for k in [
            "pii_redaction_pattern_count",
            "pii_redactions_made",
            "pii_not_in_output",
            "dp_committed_value_within_3_sigma",
            "budget_breach_refused",
            "utility_curve_is_monotonic",
            "raw_value_not_in_capsule_dict",
            "report_cid",
        ]
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")
    closed = (
        rep.pii_redaction_pattern_count >= 5
        and rep.pii_redactions_made >= 5
        and rep.pii_not_in_output
        and rep.dp_committed_value_within_3_sigma
        and rep.budget_breach_refused
        and rep.utility_curve_is_monotonic
        and rep.raw_value_not_in_capsule_dict)
    return 0 if closed else 1


if __name__ == "__main__":
    sys.exit(main())
