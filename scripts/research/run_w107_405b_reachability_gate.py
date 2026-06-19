#!/usr/bin/env python3
"""W107 — 405B reachability GATE decision recorder.

Thin, deterministic wrapper over the W105 reachability probe
(`scripts/run_w105_405b_reachability_probe.py`).  It reads the most
recent W107 probe.json and emits the machine-readable α/β gate
decision defined in ``docs/RUNBOOK_W107.md`` § 2:

* ``status == "reachable"`` (HTTP 200)  -> GATE = OPEN  -> Lane α LIVE
* ``http_status == 404``                -> GATE = CLOSED -> Lane β
* any other status                      -> GATE = CLOSED (indeterminate)

No NIM call is made here — run the probe first.  This recorder is
re-runnable and idempotent; it never widens the W107 matrix.

Usage::

    # 1) run the probe (one sub-second NIM call):
    python scripts/run_w105_405b_reachability_probe.py \
        --out-root results/w107/405b_reachability_probe
    # 2) record the gate decision (no NIM):
    python scripts/run_w107_405b_reachability_gate.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROBE_ROOT = (
    ROOT / "results" / "w107" / "405b_reachability_probe")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _cid(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _resolve_latest_probe(probe_root: Path) -> Path:
    pointer = probe_root / "latest_run.txt"
    if pointer.exists():
        sub = pointer.read_text().strip()
        cand = probe_root / sub / "probe.json"
        if cand.exists():
            return cand
    # fall back to the newest probe_* dir
    candidates = sorted(probe_root.glob("probe_*/probe.json"))
    if candidates:
        return candidates[-1]
    raise SystemExit(
        f"No W107 probe.json under {probe_root}. Run "
        "scripts/run_w105_405b_reachability_probe.py first.")


def _decide_gate(probe: dict) -> dict:
    status = str(probe.get("status") or "")
    http_status = probe.get("http_status")
    if status == "reachable" and int(http_status or 0) == 200:
        gate = "OPEN"
        lane = "alpha_live"
        rationale = (
            "405B reachable on NIM (HTTP 200); Lane alpha becomes "
            "LIVE; evaluate the RUNBOOK_W107 section 3 cheap-pilot "
            "earning rule next.")
    elif status == "http_error" and int(http_status or 0) == 404:
        gate = "CLOSED"
        lane = "beta_main"
        rationale = (
            "405B unreachable on NIM (HTTP 404); Lane alpha CLOSED "
            "for W107; Lane beta (next-code-battlefield NIM-free "
            "preflight) is the main empirical lane. Refresh "
            "W104-L-...-405B-UNREACHABLE-ON-NIM-CAP with this "
            "timestamp.")
    else:
        gate = "CLOSED"
        lane = "beta_main"
        rationale = (
            f"405B gate indeterminate (status={status!r}, "
            f"http_status={http_status!r}); treated as CLOSED for "
            "branch purposes; Lane beta is the main lane. Raw "
            "status recorded verbatim for the public record.")
    return {
        "schema": "coordpy.w107_405b_reachability_gate.v1",
        "gate": gate,
        "lane": lane,
        "rationale": rationale,
        "probe_status": status,
        "probe_http_status": http_status,
        "probe_model": probe.get("model"),
        "probe_ts_utc": probe.get("ts_utc"),
        "probe_wall_ms": probe.get("wall_ms"),
        "consecutive_404_history": [
            "W104", "W105", "W106", "W107"],
        "recorded_ts_utc": _dt.datetime.now(
            _dt.timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="W107 405B reachability gate recorder (no NIM)")
    ap.add_argument(
        "--probe-root", default=str(DEFAULT_PROBE_ROOT),
        help="W107 probe output root")
    ap.add_argument(
        "--out", default=None,
        help="Gate decision JSON path (default: "
             "<probe-root>/gate_decision.json)")
    args = ap.parse_args()
    probe_root = Path(args.probe_root)
    probe_path = _resolve_latest_probe(probe_root)
    probe = json.loads(probe_path.read_text())
    decision = _decide_gate(probe)
    decision["probe_artifact"] = str(
        probe_path.relative_to(ROOT))
    decision["decision_cid"] = _cid(
        {k: v for k, v in decision.items()
         if k != "recorded_ts_utc"})
    out = Path(args.out) if args.out else (
        probe_root / "gate_decision.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2, default=str))
    print(json.dumps(decision, indent=2, default=str))
    print(f"  gate decision written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
