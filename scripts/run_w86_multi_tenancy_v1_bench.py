#!/usr/bin/env python3
"""W86 / P2 #43 Multi-Tenancy V1 — bench driver."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coordpy.multi_tenancy_isolation_v1 import (  # noqa: E402
    run_two_tenant_isolation_bench_v1,
)


def _utc_stamp() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=86_043)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else _REPO_ROOT / "results" / "w86" / "tenancy"
        / f"w86_tenancy_{_utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = run_two_tenant_isolation_bench_v1(seed=args.seed)
    report_dict = {
        "kind": "w86_multi_tenancy_v1_bench_report",
        "schema": "coordpy.multi_tenancy_isolation_v1.w86_v1",
        "report": rep.to_dict(),
    }
    out_path = out_dir / "multi_tenancy_v1_bench_report.json"
    out_path.write_text(
        json.dumps(report_dict, indent=2, sort_keys=True))

    print(f"wrote {out_path}")
    for k, v in rep.to_dict().items():
        print(f"  {k}: {v}")

    closed = (
        rep.cross_tenant_read_refused
        and rep.cross_tenant_denial_event_emitted
        and rep.audit_anchors_distinct
        and rep.budget_isolation_holds
        and rep.token_swap_refused
        and rep.no_b_bytes_in_a_chain)
    return 0 if closed else 1


if __name__ == "__main__":
    sys.exit(main())
