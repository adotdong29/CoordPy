#!/usr/bin/env python3
"""W86 / P2 #41 Schema Evolution V1 — bench driver."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coordpy.schema_evolution_v1 import (  # noqa: E402
    run_in_flight_schema_upgrade_bench_v1,
)


def _utc_stamp() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--timestamp-ns", type=int, default=86_041_000_000_000)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else _REPO_ROOT / "results" / "w86" / "schema_evolution"
        / f"w86_schema_evolution_{_utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = run_in_flight_schema_upgrade_bench_v1(
        timestamp_ns=args.timestamp_ns)

    report_dict = {
        "kind": "w86_schema_evolution_v1_bench_report",
        "schema": "coordpy.schema_evolution_v1.w86_schema_evolution_v1",
        "report": rep.to_dict(),
    }
    report_path = out_dir / "schema_evolution_v1_bench_report.json"
    report_path.write_text(json.dumps(
        report_dict, indent=2, sort_keys=True))

    print(f"wrote {report_path}")
    for k, v in rep.to_dict().items():
        print(f"  {k}: {v}")

    closed = (
        rep.chain_verifies_across_migration
        and rep.deprecated_payload_readable
        and rep.deprecation_warning_emitted
        and rep.deterministic_migration
        and rep.provenance_preserved)
    return 0 if closed else 1


if __name__ == "__main__":
    sys.exit(main())
