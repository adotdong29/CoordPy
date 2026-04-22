"""Phase-46 CI/deployment consumer.

Consumes a ``product_report.json`` (``phase45.product_report.v1``)
emitted by ``vision_mvp.product.runner`` and produces a pass/fail
verdict suitable for a CI pipeline (GitHub Actions, local make
target, deployment controller, etc.).

Checks performed, in order:

  1. **Schema check** — report declares ``phase45.product_report.v1``.
  2. **Profile compatibility** — report's ``profile`` is in an
     operator-supplied whitelist (``--require-profile`` or
     ``--allow-profile``). Default: any known profile.
  3. **Readiness threshold** — ``readiness.n_passed_all /
     readiness.n >= --min-ready-fraction`` (default 1.0) and
     ``readiness.ready`` is True unless ``--allow-not-ready``.
  4. **Sweep outcome** — if a sweep block is present and executed
     (``mode == 'mock'``), require min pass@1 per strategy ≥
     ``--min-pass-at-1`` (default 1.0) across all cells.
     If the sweep is a *recorded* real-LLM launch or is skipped
     because of readiness, that is surfaced but not failed by
     default (use ``--require-sweep-executed`` to force).
  5. **Artifact presence** — ``product_report.json`` +
     ``product_summary.txt`` + ``readiness_verdict.json`` listed
     in ``artifacts``. Missing artifacts are a hard fail.

Exit code 0 iff every check passes. A non-zero exit carries a
blocker list on stdout + in the output JSON.

Usage:

    python3 -m vision_mvp.product.ci_gate \\
        --report vision_mvp/artifacts/phase45_rc_bundled/product_report.json \\
        --min-ready-fraction 1.0 --min-pass-at-1 1.0

    # Multi-report mode — every report must pass.
    python3 -m vision_mvp.product.ci_gate \\
        --report A/product_report.json B/product_report.json \\
        --require-profile bundled_57 bundled_57_mock_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from vision_mvp.product import profiles as _profiles


CI_SCHEMA = "phase46.ci_verdict.v1"
EXPECTED_REPORT_SCHEMA = "phase45.product_report.v1"


def evaluate_report(report_path: str,
                      *,
                      allow_profiles: tuple[str, ...] | None = None,
                      require_profiles: tuple[str, ...] | None = None,
                      min_ready_fraction: float = 1.0,
                      min_pass_at_1: float = 1.0,
                      allow_not_ready: bool = False,
                      require_sweep_executed: bool = False,
                      ) -> dict[str, Any]:
    if not os.path.exists(report_path):
        return {
            "schema": CI_SCHEMA, "report_path": report_path,
            "ok": False, "blockers": ["report_not_found"],
            "checks": {},
        }
    with open(report_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)

    checks: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []

    # 1. Schema.
    got_schema = report.get("schema")
    schema_ok = got_schema == EXPECTED_REPORT_SCHEMA
    checks["schema"] = {"ok": schema_ok, "expected": EXPECTED_REPORT_SCHEMA,
                        "got": got_schema}
    if not schema_ok:
        blockers.append(f"schema_mismatch:{got_schema!r}")

    # 2. Profile compatibility.
    prof = report.get("profile")
    known = set(_profiles.list_profiles())
    allowed: set[str]
    if require_profiles:
        allowed = set(require_profiles)
    elif allow_profiles:
        allowed = set(allow_profiles)
    else:
        allowed = known
    prof_ok = prof in allowed
    checks["profile"] = {"ok": prof_ok, "profile": prof,
                          "allowed": sorted(allowed)}
    if not prof_ok:
        blockers.append(f"profile_not_allowed:{prof}")

    # 3. Readiness.
    rd = report.get("readiness") or {}
    n = rd.get("n", 0)
    n_passed = rd.get("n_passed_all", 0)
    ready_flag = bool(rd.get("ready", False))
    frac = (n_passed / n) if n else 0.0
    ready_ok = (ready_flag or allow_not_ready) and frac >= min_ready_fraction
    checks["readiness"] = {
        "ok": ready_ok,
        "ready": ready_flag,
        "n": n, "n_passed_all": n_passed,
        "pass_fraction": round(frac, 6),
        "min_pass_fraction": min_ready_fraction,
        "blockers": rd.get("blockers") or [],
    }
    if not ready_ok:
        if not ready_flag and not allow_not_ready:
            blockers.append("readiness_not_ready")
        if frac < min_ready_fraction:
            blockers.append(
                f"readiness_fraction_below_threshold:"
                f"{frac:.3f}<{min_ready_fraction:.3f}")

    # 4. Sweep.
    sw = report.get("sweep")
    sweep_info: dict[str, Any] = {"ok": True, "present": sw is not None}
    if sw is None:
        sweep_info["note"] = "no sweep configured in profile"
        if require_sweep_executed:
            sweep_info["ok"] = False
            blockers.append("sweep_required_but_absent")
    elif sw.get("skipped"):
        sweep_info["skipped"] = True
        sweep_info["reason"] = sw.get("reason")
        if require_sweep_executed:
            sweep_info["ok"] = False
            blockers.append(f"sweep_skipped:{sw.get('reason')}")
    elif sw.get("mode") == "real":
        sweep_info["mode"] = "real"
        sweep_info["executed_in_process"] = sw.get("executed_in_process",
                                                     False)
        if require_sweep_executed and not sw.get("executed_in_process"):
            sweep_info["ok"] = False
            blockers.append("sweep_recorded_not_executed")
    elif sw.get("mode") == "mock":
        cells = sw.get("cells", [])
        min_cell_pass_at_1 = 1.0
        bad_cells: list[str] = []
        for c in cells:
            pooled = c.get("pooled", {})
            for strat, p in pooled.items():
                v = p.get("pass_at_1", 0.0)
                if v < min_cell_pass_at_1:
                    min_cell_pass_at_1 = v
                if v < min_pass_at_1:
                    bad_cells.append(
                        f"parser={c.get('parser_mode')}/"
                        f"apply={c.get('apply_mode')}/"
                        f"nd={c.get('n_distractors')}/"
                        f"strategy={strat}:pass@1={v:.3f}")
        sweep_info.update({
            "mode": "mock", "n_cells": len(cells),
            "min_pass_at_1": min_cell_pass_at_1,
            "required_min_pass_at_1": min_pass_at_1,
            "bad_cells": bad_cells,
        })
        if bad_cells:
            sweep_info["ok"] = False
            blockers.append(
                f"sweep_pass_at_1_below_threshold:{len(bad_cells)}_cell(s)")
    checks["sweep"] = sweep_info

    # 5. Artifact presence.
    required = {"product_report.json",
                 "product_summary.txt",
                 "readiness_verdict.json"}
    got = set(report.get("artifacts") or [])
    missing = sorted(required - got)
    checks["artifacts"] = {
        "ok": not missing, "required": sorted(required),
        "missing": missing,
    }
    if missing:
        blockers.append(f"artifacts_missing:{missing}")

    return {
        "schema": CI_SCHEMA,
        "report_path": os.path.abspath(report_path),
        "product_report_profile": prof,
        "ok": not blockers,
        "blockers": blockers,
        "checks": checks,
    }


def aggregate(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Union verdict across multiple reports."""
    all_ok = all(v["ok"] for v in verdicts) if verdicts else False
    return {
        "schema": CI_SCHEMA + ".aggregate",
        "n_reports": len(verdicts),
        "ok": all_ok,
        "verdicts": verdicts,
        "aggregate_blockers": [
            f"{v['report_path']}: {b}"
            for v in verdicts for b in v["blockers"]],
    }


def _render_verdict(v: dict[str, Any]) -> str:
    lines = [f"=== phase46 CI gate ==="]
    lines.append(f"report : {v['report_path']}")
    lines.append(f"profile: {v.get('product_report_profile','-')}")
    lines.append(f"ok     : {v['ok']}")
    for (name, c) in v.get("checks", {}).items():
        lines.append(f"  - {name:<10s} ok={c.get('ok')}  "
                      f"{ {k: c[k] for k in c if k != 'ok'} }")
    if v["blockers"]:
        lines.append(f"blockers: {v['blockers']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase-46 CI gate over product_report.json.")
    ap.add_argument("--report", nargs="+", required=True,
                     help="One or more paths to product_report.json.")
    ap.add_argument("--require-profile", nargs="+", default=None,
                     help="Only accept these profiles (whitelist).")
    ap.add_argument("--allow-profile", nargs="+", default=None,
                     help="Alias for --require-profile.")
    ap.add_argument("--min-ready-fraction", type=float, default=1.0)
    ap.add_argument("--min-pass-at-1", type=float, default=1.0)
    ap.add_argument("--allow-not-ready", action="store_true")
    ap.add_argument("--require-sweep-executed", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    verdicts: list[dict[str, Any]] = []
    for p in args.report:
        v = evaluate_report(
            p,
            allow_profiles=tuple(args.allow_profile)
                if args.allow_profile else None,
            require_profiles=tuple(args.require_profile)
                if args.require_profile else None,
            min_ready_fraction=args.min_ready_fraction,
            min_pass_at_1=args.min_pass_at_1,
            allow_not_ready=args.allow_not_ready,
            require_sweep_executed=args.require_sweep_executed)
        verdicts.append(v)
        print(_render_verdict(v))

    agg = aggregate(verdicts)
    print(f"=== aggregate ===\nok={agg['ok']}  "
           f"n_reports={agg['n_reports']}  "
           f"aggregate_blockers={len(agg['aggregate_blockers'])}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(agg, fh, indent=2, default=str)
        print(f"Wrote {args.out}")
    return 0 if agg["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
